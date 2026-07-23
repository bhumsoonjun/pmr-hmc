/* PMR-HMC native production kernel.
 *
 * Consumes a FROZEN atlas packed by native.py (charts, caches, weights, h,
 * t-defense) and runs the exact production chain: chart Gibbs draw ->
 * harmonic rotation + cached residual kicks -> true endpoint MH; global
 * independence moves from q (optionally t-defended). Targets are implemented
 * natively (formulas mirror targets.py exactly; verified vs JAX in the bench).
 *
 * Chart type codes: 0 gauss, 1 tcomp, 2 multishear, 3 scale, 4 polar, 5 hier.
 * Cache type codes: 0 zero, 1 scalar-RBF, 2 local-affine kNN.
 *
 * cc -O3 -ffast-math -shared -fPIC pmr_kernel.c -o pmr_kernel.dylib
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define LOG2PI 1.8378770664093453

/* ----------------------------- RNG (xoshiro256++) ----------------------- */
typedef struct { uint64_t s[4]; int have; double spare; } rng_t;
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t rng_next(rng_t *r) {
    uint64_t res = rotl(r->s[0] + r->s[3], 23) + r->s[0];
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1]; r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t; r->s[3] = rotl(r->s[3], 45);
    return res;
}
static void rng_seed(rng_t *r, uint64_t seed) {
    r->have = 0; r->spare = 0.0;
    for (int i = 0; i < 4; i++) {          /* splitmix64 */
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        r->s[i] = z ^ (z >> 31);
    }
}
static inline double runif(rng_t *r) { return (rng_next(r) >> 11) * 0x1.0p-53; }
static double rnorm(rng_t *r) {
    /* Box-Muller spare lives in rng_t: per-chain, so identical seeds give
       identical streams regardless of call history or threading */
    if (r->have) { r->have = 0; return r->spare; }
    double u, v, s;
    do { u = 2 * runif(r) - 1; v = 2 * runif(r) - 1; s = u * u + v * v; } while (s >= 1 || s == 0);
    double m = sqrt(-2 * log(s) / s);
    r->spare = v * m; r->have = 1;
    return u * m;
}
static double rgamma_shape(rng_t *r, double a) {      /* Marsaglia-Tsang, a >= 1 */
    double d = a - 1.0 / 3.0, c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do { x = rnorm(r); v = 1 + c * x; } while (v <= 0);
        v = v * v * v;
        double u = runif(r);
        if (u < 1 - 0.0331 * x * x * x * x) return d * v;
        if (log(u) < 0.5 * x * x + d * (1 - v + log(v))) return d * v;
    }
}

/* ------------------------------- targets --------------------------------- */
/* user-supplied density callback: returns U(x) = -log pi(x) (unnormalized) */
typedef double (*pmr_ufn)(const double *x, int d, void *ctx);

typedef struct {
    int id, d, n, p;
    const double *par, *X, *y;
    pmr_ufn fn;   /* non-NULL -> callback target, id ignored */
    void *ctx;
} target_t;

static inline double log_sigmoid(double t) {
    return t > 0 ? -log1p(exp(-t)) : t - log1p(exp(t));
}

static double target_U(const target_t *T, const double *x) {
    if (T->fn) return T->fn(x, T->d, T->ctx);
    int d = T->d;
    double u = 0;
    switch (T->id) {
    case 0: /* gauss_iid */
        for (int i = 0; i < d; i++) u += x[i] * x[i];
        return 0.5 * u;
    case 1: { /* gauss_prec: X is d*d precision */
        for (int i = 0; i < d; i++) {
            double s = 0;
            for (int j = 0; j < d; j++) s += T->X[i * d + j] * x[j];
            u += x[i] * s;
        }
        return 0.5 * u; }
    case 2: { /* mixture2: par=[sep, w1] */
        double sep = T->par[0], w1 = T->par[1];
        double a = 0, b = 0;
        for (int i = 0; i < d; i++) { double xi = x[i]; a += xi * xi; b += xi * xi; }
        a += -2 * sep * x[0] + sep * sep - (-2 * (-sep) * x[0] + sep * sep); /* placeholder */
        a = 0; b = 0;
        for (int i = 0; i < d; i++) {
            double d1 = x[i] - (i == 0 ? sep : 0.0), d2 = x[i] - (i == 0 ? -sep : 0.0);
            a += d1 * d1; b += d2 * d2;
        }
        double la = log(w1) - 0.5 * a, lb = log(1 - w1) - 0.5 * b;
        double m = la > lb ? la : lb;
        return -(m + log(exp(la - m) + exp(lb - m))); }
    case 3: { /* banana: par=[b] */
        double b = T->par[0];
        u = x[0] * x[0] / 200.0;
        double t = x[1] + b * x[0] * x[0] - 100.0 * b;
        u += 0.5 * t * t;
        for (int i = 2; i < d; i++) u += 0.5 * x[i] * x[i];
        return u; }
    case 4: { /* funnel */
        double v = x[0], s = 0;
        for (int i = 1; i < d; i++) s += x[i] * x[i];
        return v * v / 18.0 + 0.5 * (d - 1) * v + 0.5 * s * exp(-v); }
    case 5: { /* ring: par=[R, w] */
        double R = T->par[0], w = T->par[1];
        double rho = sqrt(x[0] * x[0] + x[1] * x[1] + 1e-12);
        u = (rho - R) * (rho - R) / (2 * w * w);
        for (int i = 2; i < d; i++) u += 0.5 * x[i] * x[i];
        return u; }
    case 6: { /* student_t: par=[nu] */
        double nu = T->par[0], s = 0;
        for (int i = 0; i < d; i++) s += x[i] * x[i];
        return 0.5 * (nu + d) * log1p(s / nu); }
    case 7: /* logcosh */
        for (int i = 0; i < d; i++) { double a = fabs(x[i]); u += a + log1p(exp(-2 * a)); }
        return u;
    case 8: { /* rosenbrock: par=[a, b] */
        double a = T->par[0], b = T->par[1];
        for (int p = 0; p < d / 2; p++) {
            double xe = x[2 * p], xo = x[2 * p + 1];
            u += (a - xe) * (a - xe) / 20.0 + b * (xo - xe * xe) * (xo - xe * xe) / 10.0;
        }
        return u; }
    case 9: { /* squiggle: par=[freq] */
        double t = x[1] + sin(T->par[0] * x[0]);
        u = x[0] * x[0] / 200.0 + 0.5 * t * t;
        for (int i = 2; i < d; i++) u += 0.5 * x[i] * x[i];
        return u; }
    case 10: { /* logreg: X n*p, y n; d = p+1; prior sd 2.5 */
        for (int r = 0; r < T->n; r++) {
            double t = x[0];
            for (int j = 0; j < T->p; j++) t += T->X[r * T->p + j] * x[1 + j];
            u -= T->y[r] * log_sigmoid(t) + (1 - T->y[r]) * log_sigmoid(-t);
        }
        double s = 0;
        for (int i = 0; i < d; i++) s += x[i] * x[i];
        return u + 0.5 * s / 6.25; }
    case 11: { /* poisson: X n*p, y n; d = p+1; prior sd 2.5 */
        for (int r = 0; r < T->n; r++) {
            double e = x[0];
            for (int j = 0; j < T->p; j++) e += T->X[r * T->p + j] * x[1 + j];
            if (e > 20) e = 20; if (e < -20) e = -20;
            u -= T->y[r] * e - exp(e);
        }
        double s = 0;
        for (int i = 0; i < d; i++) s += x[i] * x[i];
        return u + 0.5 * s / 6.25; }
    }
    return NAN;
}

/* ------------------------------ atlas ------------------------------------ */
typedef struct {
    int d, K;
    const int *ctype; const long *coff; const double *cb;
    const int *qtype; const long *qoff; const double *qb;
    const double *log_ws;
    const double *tdef; /* [nu, eps, mu(d), sdiag(d)] or nu<=0 off */
} atlas_t;

/* comp blob accessors ----------------------------------------------------- */
#define CB(A,k) ((A)->cb + (A)->coff[k])

static double gauss_logpdf_at(const double *mu, const double *Linv, double logdet_half,
                              int d, const double *x, double *zbuf) {
    double q = 0;
    for (int i = 0; i < d; i++) {
        double s = 0;
        for (int j = 0; j <= i; j++) s += Linv[i * d + j] * (x[j] - mu[j]);
        zbuf[i] = s; q += s * s;
    }
    return -0.5 * d * LOG2PI - logdet_half - 0.5 * q;
}

/* multishear unshear into buf */
static void ms_unshear(const double *b, int d, const double *x, double *u) {
    const double *mu = b;
    int np = (int)b[2 * d * d + d + 1];
    const double *pr = b + 2 * d * d + d + 2;
    memcpy(u, x, d * sizeof(double));
    for (int p = 0; p < np; p++) {
        int dr = (int)pr[4 * p], t = (int)pr[4 * p + 1];
        double g = pr[4 * p + 2], m2 = pr[4 * p + 3];
        double c = x[dr] - mu[dr];
        u[t] -= g * (c * c - m2);
    }
}

/* tri (sinh-arcsinh triangular) helpers ----------------------------------- */
static inline double sas_f(double z, double e, double dl) { return sinh((asinh(z) + e) / dl); }
static inline double sas_finv(double y, double e, double dl) { return sinh(dl * asinh(y) - e); }
static inline double sas_dz(double z, double e, double dl) {
    return cosh((asinh(z) + e) / dl) / (dl * sqrt(1.0 + z * z));
}
/* clamped driver exponent; matches the Python chart exactly (map + Jacobian
   share the clamp, so the bijection and its log-det stay mutually exact) */
static inline double tri_expo(const double *gam, const double *drv, const double *z, int i) {
    int dr = (int)drv[i];
    if (dr < 0) return 0.0;
    double t = gam[i] * z[dr];
    return t > 30.0 ? 30.0 : (t < -30.0 ? -30.0 : t);
}

static double comp_logpdf(const atlas_t *A, int k, const double *x, double *zb, double *xb) {
    int d = A->d;
    const double *b = CB(A, k);
    switch (A->ctype[k]) {
    case 0:
        return gauss_logpdf_at(b, b + d + d * d, b[d + 2 * d * d], d, x, zb);
    case 1: { /* tcomp: student-t density */
        double nu = b[d + 2 * d * d + 1];
        const double *mu = b, *Linv = b + d + d * d;
        double logdet_half = b[d + 2 * d * d];
        double q = 0;
        for (int i = 0; i < d; i++) {
            double s = 0;
            for (int j = 0; j <= i; j++) s += Linv[i * d + j] * (x[j] - mu[j]);
            q += s * s;
        }
        double ln = lgamma((nu + d) / 2) - lgamma(nu / 2) - 0.5 * d * log(nu * M_PI) - logdet_half;
        return ln - 0.5 * (nu + d) * log1p(q / nu); }
    case 2:
        ms_unshear(b, d, x, xb);
        return gauss_logpdf_at(b, b + d + d * d, b[d + 2 * d * d], d, xb, zb);
    case 3: { /* scale: [j, vbar, mu(d), sig(d), alpha(d), beta(d)] */
        int j = (int)b[0]; double vbar = b[1];
        const double *mu = b + 2, *sig = b + 2 + d, *al = b + 2 + 2 * d, *be = b + 2 + 3 * d;
        double v = x[j], q = 0, logdet = log(sig[j]);
        double zj = (v - mu[j]) / sig[j]; q += zj * zj;
        for (int i = 0; i < d; i++) {
            if (i == j) continue;
            double s = exp(0.5 * (al[i] + be[i] * (v - vbar)));
            double zi = (x[i] - mu[i]) / (sig[i] * s);
            q += zi * zi;
            logdet += log(sig[i]) + 0.5 * (al[i] + be[i] * (v - vbar));
        }
        return -0.5 * d * LOG2PI - logdet - 0.5 * q; }
    case 4: { /* polar: [i, j, R, sr, c, Kw, mu(d), sig(d)] */
        int i = (int)b[0], j = (int)b[1];
        double R = b[2], sr = b[3], c = b[4];
        int Kw = (int)b[5];
        const double *mu = b + 6, *sig = b + 6 + d;
        double u0 = x[i] - mu[i], u1 = x[j] - mu[j];
        double r = sqrt(u0 * u0 + u1 * u1) + 1e-300, th = atan2(u1, u0);
        double zr = (r - R) / sr, base = zr * zr, ld = log(sr) + log(c) + log(r);
        for (int t = 0; t < d; t++) {
            if (t == i || t == j) continue;
            double zt = (x[t] - mu[t]) / sig[t];
            base += zt * zt; ld += log(sig[t]);
        }
        double m = -1e300;
        double terms[64]; int nt = 0;
        for (int kk = -Kw; kk <= Kw; kk++) {
            double zth = (th + 2 * M_PI * kk) / c;
            double lp = -0.5 * (base + zth * zth);
            terms[nt++] = lp; if (lp > m) m = lp;
        }
        double s = 0;
        for (int t = 0; t < nt; t++) s += exp(terms[t] - m);
        return -0.5 * d * LOG2PI + m + log(s) - ld; }
    case 5: { /* hier: [v, loc, gam, vbar, nC, C(nC), mu(d), sig(d), cs(d)] */
        int v = (int)b[0], loc = (int)b[1];
        double gam = b[2], vbar = b[3];
        int nC = (int)b[4];
        const double *C = b + 5, *mu = b + 5 + nC, *sig = b + 5 + nC + d, *cs = b + 5 + nC + 2 * d;
        double t = exp(gam * (x[v] - vbar)), q = 0, ld = 0;
        char inC[512]; memset(inC, 0, d);
        for (int i = 0; i < nC; i++) inC[(int)C[i]] = 1;
        double zv = (x[v] - mu[v]) / sig[v]; q += zv * zv; ld += log(sig[v]);
        double zl = (x[loc] - mu[loc]) / sig[loc]; q += zl * zl; ld += log(sig[loc]);
        for (int i = 0; i < d; i++) {
            if (i == v || i == loc) continue;
            if (inC[i]) {
                double z = (x[i] - x[loc]) / (cs[i] * t);
                q += z * z; ld += log(cs[i]) + gam * (x[v] - vbar);
            } else {
                double z = (x[i] - mu[i]) / sig[i];
                q += z * z; ld += log(sig[i]);
            }
        }
        return -0.5 * d * LOG2PI - ld - 0.5 * q; }
    case 6: { /* tri: [ord(d), mu(d), sig(d), eps(d), delta(d), drv(d), gam(d)] */
        const double *ord = b, *mu = b + d, *sig = b + 2 * d, *eps = b + 3 * d;
        const double *del = b + 4 * d, *drv = b + 5 * d, *gam = b + 6 * d;
        double q = 0, ld = 0;
        for (int oi = 0; oi < d; oi++) {   /* forward substitution: drivers first */
            int i = (int)ord[oi];
            double ex = tri_expo(gam, drv, zb, i);
            zb[i] = sas_finv((x[i] - mu[i]) / (exp(ex) * sig[i]), eps[i], del[i]);
            ld += ex + log(sig[i]) + log(sas_dz(zb[i], eps[i], del[i]));
        }
        for (int i = 0; i < d; i++) q += zb[i] * zb[i];
        double v = -0.5 * d * LOG2PI - 0.5 * q - ld;
        return isnan(v) ? -INFINITY : v; }
    }
    return NAN;
}

/* to_z: cur_lam threads the tcomp auxiliary within one transition */
static void comp_to_z(const atlas_t *A, int k, const double *x, double *z, rng_t *r,
                      double *cur_lam, double *xb) {
    int d = A->d;
    const double *b = CB(A, k);
    switch (A->ctype[k]) {
    case 0: {
        const double *mu = b, *Linv = b + d + d * d;
        for (int i = 0; i < d; i++) {
            double s = 0;
            for (int j = 0; j <= i; j++) s += Linv[i * d + j] * (x[j] - mu[j]);
            z[i] = s;
        }
        *cur_lam = 1.0; break; }
    case 1: {
        const double *mu = b, *Linv = b + d + d * d;
        double nu = b[d + 2 * d * d + 1], q = 0;
        for (int i = 0; i < d; i++) {
            double s = 0;
            for (int j = 0; j <= i; j++) s += Linv[i * d + j] * (x[j] - mu[j]);
            z[i] = s; q += s * s;
        }
        double lam = rgamma_shape(r, 0.5 * (nu + d)) / (0.5 * (nu + q));
        if (lam < 1e-12) lam = 1e-12;
        *cur_lam = lam;
        double sq = sqrt(lam);
        for (int i = 0; i < d; i++) z[i] *= sq;
        break; }
    case 2: {
        ms_unshear(b, d, x, xb);
        const double *mu = b, *Linv = b + d + d * d;
        for (int i = 0; i < d; i++) {
            double s = 0;
            for (int j = 0; j <= i; j++) s += Linv[i * d + j] * (xb[j] - mu[j]);
            z[i] = s;
        }
        *cur_lam = 1.0; break; }
    case 3: {
        int j = (int)b[0]; double vbar = b[1];
        const double *mu = b + 2, *sig = b + 2 + d, *al = b + 2 + 2 * d, *be = b + 2 + 3 * d;
        double v = x[j];
        z[j] = (v - mu[j]) / sig[j];
        for (int i = 0; i < d; i++) {
            if (i == j) continue;
            double s = exp(0.5 * (al[i] + be[i] * (v - vbar)));
            z[i] = (x[i] - mu[i]) / (sig[i] * s);
        }
        *cur_lam = 1.0; break; }
    case 4: {
        int i = (int)b[0], j = (int)b[1];
        double R = b[2], sr = b[3], c = b[4];
        int Kw = (int)b[5];
        const double *mu = b + 6, *sig = b + 6 + d;
        double u0 = x[i] - mu[i], u1 = x[j] - mu[j];
        double rr = sqrt(u0 * u0 + u1 * u1) + 1e-300, th = atan2(u1, u0);
        double w[64], m = -1e300; int nb = 2 * Kw + 1;
        for (int kk = 0; kk < nb; kk++) {
            double zth = (th + 2 * M_PI * (kk - Kw)) / c;
            w[kk] = -0.5 * zth * zth; if (w[kk] > m) m = w[kk];
        }
        double s = 0;
        for (int kk = 0; kk < nb; kk++) { w[kk] = exp(w[kk] - m); s += w[kk]; }
        double u = runif(r) * s, acc = 0; int pick = nb - 1;
        for (int kk = 0; kk < nb; kk++) { acc += w[kk]; if (u <= acc) { pick = kk; break; } }
        z[i] = (rr - R) / sr;
        z[j] = (th + 2 * M_PI * (pick - Kw)) / c;
        for (int t = 0; t < d; t++)
            if (t != i && t != j) z[t] = (x[t] - mu[t]) / sig[t];
        *cur_lam = 1.0; break; }
    case 5: {
        int v = (int)b[0], loc = (int)b[1];
        double gam = b[2], vbar = b[3];
        int nC = (int)b[4];
        const double *C = b + 5, *mu = b + 5 + nC, *sig = b + 5 + nC + d, *cs = b + 5 + nC + 2 * d;
        char inC[512]; memset(inC, 0, d);
        for (int i = 0; i < nC; i++) inC[(int)C[i]] = 1;
        double t = exp(gam * (x[v] - vbar));
        z[v] = (x[v] - mu[v]) / sig[v];
        z[loc] = (x[loc] - mu[loc]) / sig[loc];
        for (int i = 0; i < d; i++) {
            if (i == v || i == loc) continue;
            z[i] = inC[i] ? (x[i] - x[loc]) / (cs[i] * t) : (x[i] - mu[i]) / sig[i];
        }
        *cur_lam = 1.0; break; }
    case 6: {
        const double *ord = b, *mu = b + d, *sig = b + 2 * d, *eps = b + 3 * d;
        const double *del = b + 4 * d, *drv = b + 5 * d, *gam = b + 6 * d;
        for (int oi = 0; oi < d; oi++) {
            int i = (int)ord[oi];
            double sc = exp(tri_expo(gam, drv, z, i));
            z[i] = sas_finv((x[i] - mu[i]) / (sc * sig[i]), eps[i], del[i]);
        }
        *cur_lam = 1.0; break; }
    }
}

static void comp_from_z(const atlas_t *A, int k, const double *z, double *x, double cur_lam) {
    int d = A->d;
    const double *b = CB(A, k);
    switch (A->ctype[k]) {
    case 0: case 1: case 2: {
        const double *mu = b, *L = b + d;
        double sq = A->ctype[k] == 1 ? sqrt(cur_lam) : 1.0;
        for (int i = 0; i < d; i++) {
            double s = 0;
            for (int j = 0; j <= i; j++) s += L[i * d + j] * z[j];
            x[i] = mu[i] + s / sq;
        }
        if (A->ctype[k] == 2) {
            int np = (int)b[2 * d * d + d + 1];
            const double *pr = b + 2 * d * d + d + 2;
            for (int p = 0; p < np; p++) {
                int dr = (int)pr[4 * p], t = (int)pr[4 * p + 1];
                double g = pr[4 * p + 2], m2 = pr[4 * p + 3];
                double c = x[dr] - mu[dr];
                x[t] += g * (c * c - m2);
            }
        }
        break; }
    case 3: {
        int j = (int)b[0]; double vbar = b[1];
        const double *mu = b + 2, *sig = b + 2 + d, *al = b + 2 + 2 * d, *be = b + 2 + 3 * d;
        double v = mu[j] + sig[j] * z[j];
        x[j] = v;
        for (int i = 0; i < d; i++) {
            if (i == j) continue;
            double s = exp(0.5 * (al[i] + be[i] * (v - vbar)));
            x[i] = mu[i] + sig[i] * s * z[i];
        }
        break; }
    case 4: {
        int i = (int)b[0], j = (int)b[1];
        double R = b[2], sr = b[3], c = b[4];
        const double *mu = b + 6, *sig = b + 6 + d;
        double r = R + sr * z[i], psi = c * z[j];
        x[i] = mu[i] + r * cos(psi);
        x[j] = mu[j] + r * sin(psi);
        for (int t = 0; t < d; t++)
            if (t != i && t != j) x[t] = mu[t] + sig[t] * z[t];
        break; }
    case 5: {
        int v = (int)b[0], loc = (int)b[1];
        double gam = b[2], vbar = b[3];
        int nC = (int)b[4];
        const double *C = b + 5, *mu = b + 5 + nC, *sig = b + 5 + nC + d, *cs = b + 5 + nC + 2 * d;
        char inC[512]; memset(inC, 0, d);
        for (int i = 0; i < nC; i++) inC[(int)C[i]] = 1;
        double xv = mu[v] + sig[v] * z[v];
        double xl = mu[loc] + sig[loc] * z[loc];
        double t = exp(gam * (xv - vbar));
        x[v] = xv; x[loc] = xl;
        for (int i = 0; i < d; i++) {
            if (i == v || i == loc) continue;
            x[i] = inC[i] ? xl + cs[i] * t * z[i] : mu[i] + sig[i] * z[i];
        }
        break; }
    case 6: {
        const double *ord = b, *mu = b + d, *sig = b + 2 * d, *eps = b + 3 * d;
        const double *del = b + 4 * d, *drv = b + 5 * d, *gam = b + 6 * d;
        for (int oi = 0; oi < d; oi++) {
            int i = (int)ord[oi];
            double sc = exp(tri_expo(gam, drv, z, i));
            x[i] = mu[i] + sc * sig[i] * sas_f(z[i], eps[i], del[i]);
        }
        break; }
    }
}

/* mixture ops */
static double mix_logq(const atlas_t *A, const double *x, double *comp_out,
                       double *zb, double *xb) {
    double m = -1e300;
    for (int k = 0; k < A->K; k++) {
        comp_out[k] = A->log_ws[k] + comp_logpdf(A, k, x, zb, xb);
        if (comp_out[k] > m) m = comp_out[k];
    }
    double s = 0;
    for (int k = 0; k < A->K; k++) s += exp(comp_out[k] - m);
    return m + log(s);
}

/* ------------------------------ caches ----------------------------------- */
static void cache_query(const atlas_t *A, int k, const double *z, double *f) {
    int d = A->d;
    const double *b = A->qb + A->qoff[k];
    memset(f, 0, d * sizeof(double));
    if (A->qtype[k] == 0) return;
    if (A->qtype[k] == 1) { /* rbf: [s, M, V(d*s), C(M*s), ell(M), a(M)] */
        int s = (int)b[0], M = (int)b[1];
        const double *V = b + 2, *C = b + 2 + d * s, *ell = b + 2 + d * s + M * s, *a = ell + M;
        double y[16];
        for (int j = 0; j < s; j++) {
            double acc = 0;
            for (int i = 0; i < d; i++) acc += z[i] * V[i * s + j];
            y[j] = acc;
        }
        double fu[16]; memset(fu, 0, s * sizeof(double));
        for (int m = 0; m < M; m++) {
            double r2 = 0;
            for (int j = 0; j < s; j++) {
                double dj = y[j] - C[m * s + j];
                r2 += dj * dj;
            }
            double r = sqrt(r2) / ell[m];
            if (r >= 1.0) continue;
            double om = 1.0 - r;
            double fac = -20.0 * om * om * om / (ell[m] * ell[m]) * a[m];
            for (int j = 0; j < s; j++) fu[j] += fac * (y[j] - C[m * s + j]);
        }
        for (int i = 0; i < d; i++) {
            double acc = 0;
            for (int j = 0; j < s; j++) acc += V[i * s + j] * fu[j];
            f[i] = acc;
        }
        return;
    }
    /* affine: [s, n, kq, rho, fmax, V(d*s), Y(n*s), Fs(n*s), B(n*s*s)] */
    int s = (int)b[0], n = (int)b[1], kq = (int)b[2];
    double rho = b[3], fmax = b[4];
    const double *V = b + 5, *Y = V + d * s, *Fs = Y + n * s, *B = Fs + n * s;
    double y[16];
    for (int j = 0; j < s; j++) {
        double acc = 0;
        for (int i = 0; i < d; i++) acc += z[i] * V[i * s + j];
        y[j] = acc;
    }
    /* kq nearest: max-heap over the k best (O(n log k), no O(n*k) rescan) */
    int idx[32]; double dist[32]; int nk = kq < n ? kq : n;
    int hs = 0;
    for (int m = 0; m < n; m++) {
        double r2 = 0;
        const double *Ym = Y + (long)m * s;
        for (int j = 0; j < s; j++) { double dj = y[j] - Ym[j]; r2 += dj * dj; }
        if (hs < nk) {                       /* push */
            int i = hs++; dist[i] = r2; idx[i] = m;
            while (i > 0) { int p2 = (i - 1) >> 1;
                if (dist[p2] < dist[i]) { double td = dist[p2]; dist[p2] = dist[i]; dist[i] = td;
                    int ti = idx[p2]; idx[p2] = idx[i]; idx[i] = ti; i = p2; } else break; }
        } else if (r2 < dist[0]) {           /* replace root, sift down */
            dist[0] = r2; idx[0] = m; int i = 0;
            for (;;) { int l = 2 * i + 1, r = l + 1, big = i;
                if (l < nk && dist[l] > dist[big]) big = l;
                if (r < nk && dist[r] > dist[big]) big = r;
                if (big == i) break;
                double td = dist[big]; dist[big] = dist[i]; dist[i] = td;
                int ti = idx[big]; idx[big] = idx[i]; idx[i] = ti; i = big; }
        }
    }
    double dmax = 0, dmin = 1e300;
    for (int t = 0; t < nk; t++) {
        dist[t] = sqrt(dist[t]);
        if (dist[t] > dmax) dmax = dist[t];
        if (dist[t] < dmin) dmin = dist[t];
    }
    double scale = dmax > 0 ? dmax : 1.0, wsum = 0;
    double fu[16]; memset(fu, 0, s * sizeof(double));
    for (int t = 0; t < nk; t++) {
        double w = exp(-0.5 * (dist[t] / scale) * (dist[t] / scale));
        wsum += w;
        const double *Bm = B + (long)idx[t] * s * s;
        for (int j = 0; j < s; j++) {
            double pred = Fs[idx[t] * s + j];
            for (int j2 = 0; j2 < s; j2++)
                pred += Bm[j * s + j2] * (y[j2] - Y[idx[t] * s + j2]);
            fu[j] += w * pred;
        }
    }
    for (int j = 0; j < s; j++) fu[j] /= wsum;
    double gate = 1.0;
    if (dmin > rho) { double e = (dmin - rho) / rho; gate = exp(-e * e); }
    double nf = 0;
    for (int j = 0; j < s; j++) { fu[j] *= gate; nf += fu[j] * fu[j]; }
    nf = sqrt(nf);
    if (fmax > 0 && nf > fmax)
        for (int j = 0; j < s; j++) fu[j] *= fmax / nf;
    for (int i = 0; i < d; i++) {
        double acc = 0;
        for (int j = 0; j < s; j++) acc += V[i * s + j] * fu[j];
        f[i] = acc;
    }
}

/* ------------------------------ kernel ----------------------------------- */
static long run_kernel(target_t T,
             int d, int K,
             const int *ctype, const long *coff, const double *cblob,
             const int *qtype, const long *qoff, const double *qblob,
             const double *log_ws, const double *tdef,
             double h, double p_global, double T0, double T1, int L_cap,
             const double *x0, uint64_t seed,
             long n_samples, double *chain_out, double *stats_out) {
    atlas_t A = { d, K, ctype, coff, cblob, qtype, qoff, qblob, log_ws, tdef };
    rng_t R; rng_seed(&R, seed);
    double *x = malloc(d * sizeof(double)), *z = malloc(d * sizeof(double));
    double *p = malloc(d * sizeof(double)), *y = malloc(d * sizeof(double));
    double *f = malloc(d * sizeof(double)), *zb = malloc(d * sizeof(double));
    double *xb = malloc(d * sizeof(double)), *lc = malloc(K * sizeof(double));
    memcpy(x, x0, d * sizeof(double));
    double U_x = target_U(&T, x);
    double lq_x = mix_logq(&A, x, lc, zb, xb);
    long nU = 1, nl = 0, nla = 0, ng = 0, nga = 0;
    double tnu = tdef[0], teps = tdef[1];
    const double *tmu = tdef + 2, *tsd = tdef + 2 + d;

    for (long it = 0; it < n_samples; it++) {
        if (runif(&R) < p_global) {
            /* global independence move (t-defended density g) */
            int use_t = tnu > 0 && runif(&R) < teps;
            if (use_t) {
                double lam = rgamma_shape(&R, 0.5 * tnu) / (0.5 * tnu);
                if (lam < 1e-12) lam = 1e-12;
                for (int i = 0; i < d; i++) y[i] = tmu[i] + tsd[i] * rnorm(&R) / sqrt(lam);
            } else {
                /* sample mixture comp by weight */
                double u = runif(&R), acc = 0; int kk = K - 1;
                for (int k = 0; k < K; k++) { acc += exp(log_ws[k]); if (u <= acc) { kk = k; break; } }
                for (int i = 0; i < d; i++) z[i] = rnorm(&R);
                comp_from_z(&A, kk, z, y, 1.0);
                if (ctype[kk] == 1) { /* tcomp sample needs prior lam */
                    double nu = CB(&A, kk)[d + 2 * d * d + 1];
                    double lam = rgamma_shape(&R, 0.5 * nu) / (0.5 * nu);
                    if (lam < 1e-12) lam = 1e-12;
                    comp_from_z(&A, kk, z, y, lam);
                }
            }
            double U_y = target_U(&T, y); nU++;
            ng++;
            if (isfinite(U_y)) {
                double lq_y = mix_logq(&A, y, lc, zb, xb);
                double lgx = lq_x, lgy = lq_y;
                if (tnu > 0) { /* g = (1-eps) q + eps t (diagonal student-t) */
                    double qx = 0, qy = 0, ldt = 0;
                    for (int i = 0; i < d; i++) {
                        double a1 = (x[i] - tmu[i]) / tsd[i], a2 = (y[i] - tmu[i]) / tsd[i];
                        qx += a1 * a1; qy += a2 * a2; ldt += log(tsd[i]);
                    }
                    double ln = lgamma((tnu + d) / 2) - lgamma(tnu / 2)
                        - 0.5 * d * log(tnu * M_PI) - ldt;
                    double ltx = ln - 0.5 * (tnu + d) * log1p(qx / tnu);
                    double lty = ln - 0.5 * (tnu + d) * log1p(qy / tnu);
                    double mx = lq_x + log1p(-teps) > ltx + log(teps) ? lq_x + log1p(-teps) : ltx + log(teps);
                    lgx = mx + log(exp(lq_x + log1p(-teps) - mx) + exp(ltx + log(teps) - mx));
                    double my = lq_y + log1p(-teps) > lty + log(teps) ? lq_y + log1p(-teps) : lty + log(teps);
                    lgy = my + log(exp(lq_y + log1p(-teps) - my) + exp(lty + log(teps) - my));
                }
                double la = (U_x + lgx) - (U_y + lgy);
                if (log(runif(&R)) < (la < 0 ? la : 0)) {
                    memcpy(x, y, d * sizeof(double));
                    U_x = U_y; lq_x = lq_y; nga++;
                }
            }
        } else {
            /* local chart move */
            double lq_chk = mix_logq(&A, x, lc, zb, xb); (void)lq_chk;
            double m = -1e300;
            for (int k = 0; k < K; k++) if (lc[k] > m) m = lc[k];
            double s = 0;
            for (int k = 0; k < K; k++) s += exp(lc[k] - m);
            double u = runif(&R) * s, acc = 0; int kk = K - 1;
            for (int k = 0; k < K; k++) { acc += exp(lc[k] - m); if (u <= acc) { kk = k; break; } }
            double cur_lam;
            comp_to_z(&A, kk, x, z, &R, &cur_lam, xb);
            for (int i = 0; i < d; i++) p[i] = rnorm(&R);
            double Tt = T0 + runif(&R) * (T1 - T0);
            int L = (int)(Tt / h + 0.5);
            if (L < 1) L = 1; if (L > L_cap) L = L_cap;
            double eps = Tt / L, ch = cos(eps), sh = sin(eps);
            double H0 = U_x + lq_x;
            for (int i = 0; i < d; i++) H0 += 0.5 * (z[i] * z[i] + p[i] * p[i]);
            for (int st = 0; st < L; st++) {
                cache_query(&A, kk, z, f);
                for (int i = 0; i < d; i++) p[i] -= 0.5 * eps * f[i];
                for (int i = 0; i < d; i++) {
                    double zi = ch * z[i] + sh * p[i];
                    p[i] = -sh * z[i] + ch * p[i];
                    z[i] = zi;
                }
                cache_query(&A, kk, z, f);
                for (int i = 0; i < d; i++) p[i] -= 0.5 * eps * f[i];
            }
            comp_from_z(&A, kk, z, y, cur_lam);
            nl++;
            int okf = 1;
            for (int i = 0; i < d; i++) if (!isfinite(y[i])) okf = 0;
            if (okf) {
                double U_y = target_U(&T, y); nU++;
                if (isfinite(U_y)) {
                    double lq_y = mix_logq(&A, y, lc, zb, xb);
                    double H1 = U_y + lq_y;
                    for (int i = 0; i < d; i++) H1 += 0.5 * (z[i] * z[i] + p[i] * p[i]);
                    double la = H0 - H1;
                    if (log(runif(&R)) < (la < 0 ? la : 0)) {
                        memcpy(x, y, d * sizeof(double));
                        U_x = U_y; lq_x = lq_y; nla++;
                    }
                }
            }
        }
        memcpy(chain_out + it * d, x, d * sizeof(double));
    }
    stats_out[0] = (double)nl; stats_out[1] = nl ? (double)nla / nl : 0;
    stats_out[2] = (double)ng; stats_out[3] = ng ? (double)nga / ng : 0;
    stats_out[4] = (double)nU;
    free(x); free(z); free(p); free(y); free(f); free(zb); free(xb); free(lc);
    return 0;
}

/* ------------------------- exported entry points ------------------------- */
/* d is capped so fixed scratch (hier membership bitmap) can never overflow */
#define PMR_MAX_D 4096

long pmr_run(int target_id, const double *tpar, int ntpar,
             const double *tX, const double *ty, int tn, int tp,
             int d, int K,
             const int *ctype, const long *coff, const double *cblob,
             const int *qtype, const long *qoff, const double *qblob,
             const double *log_ws, const double *tdef,
             double h, double p_global, double T0, double T1, int L_cap,
             const double *x0, uint64_t seed,
             long n_samples, double *chain_out, double *stats_out) {
    if (d > PMR_MAX_D) return -1;
    target_t T = { target_id, d, tn, tp, tpar, tX, ty, NULL, NULL };
    (void)ntpar;
    return run_kernel(T, d, K, ctype, coff, cblob, qtype, qoff, qblob, log_ws,
                      tdef, h, p_global, T0, T1, L_cap, x0, seed, n_samples,
                      chain_out, stats_out);
}

long pmr_run_cb(pmr_ufn fn, void *ctx,
                int d, int K,
                const int *ctype, const long *coff, const double *cblob,
                const int *qtype, const long *qoff, const double *qblob,
                const double *log_ws, const double *tdef,
                double h, double p_global, double T0, double T1, int L_cap,
                const double *x0, uint64_t seed,
                long n_samples, double *chain_out, double *stats_out) {
    if (d > PMR_MAX_D) return -1;
    target_t T = { -1, d, 0, 0, NULL, NULL, NULL, fn, ctx };
    return run_kernel(T, d, K, ctype, coff, cblob, qtype, qoff, qblob, log_ws,
                      tdef, h, p_global, T0, T1, L_cap, x0, seed, n_samples,
                      chain_out, stats_out);
}

/* -------- parallel multi-chain over a shared frozen atlas (pthreads) ----- */
#ifndef _WIN32
#include <pthread.h>

typedef struct {
    target_t T;
    int d, K;
    const int *ctype; const long *coff; const double *cblob;
    const int *qtype; const long *qoff; const double *qblob;
    const double *log_ws; const double *tdef;
    double h, p_global, T0k, T1k; int L_cap;
    const double *x0; uint64_t seed;
    long n_samples; double *chain_out; double *stats_out;
} job_t;

typedef struct { job_t *jobs; int n_jobs; int next; pthread_mutex_t mu; } queue_t;

static void *worker(void *arg) {
    queue_t *Q = (queue_t *)arg;
    for (;;) {
        pthread_mutex_lock(&Q->mu);
        int j = Q->next < Q->n_jobs ? Q->next++ : -1;
        pthread_mutex_unlock(&Q->mu);
        if (j < 0) return NULL;
        job_t *b = &Q->jobs[j];
        run_kernel(b->T, b->d, b->K, b->ctype, b->coff, b->cblob, b->qtype,
                   b->qoff, b->qblob, b->log_ws, b->tdef, b->h, b->p_global,
                   b->T0k, b->T1k, b->L_cap, b->x0, b->seed, b->n_samples,
                   b->chain_out, b->stats_out);
    }
}

long pmr_run_multi(int target_id, const double *tpar, int ntpar,
                   const double *tX, const double *ty, int tn, int tp,
                   int d, int K,
                   const int *ctype, const long *coff, const double *cblob,
                   const int *qtype, const long *qoff, const double *qblob,
                   const double *log_ws, const double *tdef,
                   double h, double p_global, double T0, double T1, int L_cap,
                   const double *x0s,       /* n_chains x d starting points */
                   const uint64_t *seeds,   /* n_chains seeds */
                   long n_samples, int n_chains, int n_threads,
                   double *chains_out,      /* n_chains x n_samples x d */
                   double *stats_out) {     /* n_chains x 5 */
    if (d > PMR_MAX_D || n_chains < 1) return -1;
    (void)ntpar;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > n_chains) n_threads = n_chains;
    job_t *jobs = malloc(n_chains * sizeof(job_t));
    for (int c = 0; c < n_chains; c++) {
        target_t T = { target_id, d, tn, tp, tpar, tX, ty, NULL, NULL };
        job_t b = { T, d, K, ctype, coff, cblob, qtype, qoff, qblob, log_ws,
                    tdef, h, p_global, T0, T1, L_cap,
                    x0s + (long)c * d, seeds[c], n_samples,
                    chains_out + (long)c * n_samples * d, stats_out + 5L * c };
        jobs[c] = b;
    }
    queue_t Q = { jobs, n_chains, 0, PTHREAD_MUTEX_INITIALIZER };
    pthread_t *ts = malloc(n_threads * sizeof(pthread_t));
    for (int t = 0; t < n_threads; t++) pthread_create(&ts[t], NULL, worker, &Q);
    for (int t = 0; t < n_threads; t++) pthread_join(ts[t], NULL);
    free(ts); free(jobs);
    return 0;
}
#else
long pmr_run_multi(int target_id, const double *tpar, int ntpar,
                   const double *tX, const double *ty, int tn, int tp,
                   int d, int K,
                   const int *ctype, const long *coff, const double *cblob,
                   const int *qtype, const long *qoff, const double *qblob,
                   const double *log_ws, const double *tdef,
                   double h, double p_global, double T0, double T1, int L_cap,
                   const double *x0s, const uint64_t *seeds,
                   long n_samples, int n_chains, int n_threads,
                   double *chains_out, double *stats_out) {
    (void)n_threads;
    for (int c = 0; c < n_chains; c++) {
        long rc = pmr_run(target_id, tpar, ntpar, tX, ty, tn, tp, d, K, ctype,
                          coff, cblob, qtype, qoff, qblob, log_ws, tdef, h,
                          p_global, T0, T1, L_cap, x0s + (long)c * d, seeds[c],
                          n_samples, chains_out + (long)c * n_samples * d,
                          stats_out + 5L * c);
        if (rc) return rc;
    }
    return 0;
}
#endif
