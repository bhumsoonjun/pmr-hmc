"""Residual Transport-Atlas HMC (PMR-HMC, Milestone B) — exact sampler with
zero true gradients and one true log-density eval per production transition.

Milestone A machinery (scalar RBF residual, per-chart kappa selection,
residual-guided atlas birth, forward-KL weights, error-knee h) plus:

  - TRANSPORT CHARTS (Milestone B): mixture components are invertible chart maps
    T_k with q_k(x) = phi(T_k^{-1} x) |det DT_k^{-1}|; conditional on chart k the
    latent Hamiltonian is  H = |z|^2/2 + |p|^2/2 + R(T_k(z))  with the Jacobian
    absorbed into the exactly evaluated q — the Gaussian part stays exactly
    harmonic even for nonlinear charts. Implemented charts:
      * GaussComp   x = mu + L z                 (affine Pathfinder chart)
      * ShearComp   x = mu + L z, x_t += g((x_dr-mu_dr)^2 - m2)   (unit det)
      * ScaleComp   v = mu_v + s_v z_v;  x_i = mu_i + s_i e^{(a_i+b_i(v-vbar))/2} z_i
    fitted from warm-up chain states (quadratic-ridge / log-conditional-scale
    regressions); the banana is EXACTLY a sheared Gaussian and Neal's funnel is
    EXACTLY a scale transport, so a correct fit drives R -> const, acceptance -> 1.
  - DEFENSIVE STUDENT-T GLOBAL BRANCH (Milestone C): global independence
    proposals come from g = (1-eps) q_atlas + eps t_nu(mu0, S0); the MH ratio
    uses g exactly, and R / the local charts never see the t (tail insurance
    without touching the efficient local geometry).

Exactness: all production densities (q, responsibilities, g) are exact
logsumexp; charts are deterministic invertible maps; caches frozen; the scalar
RBF force is the exact gradient of a compactly supported energy.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import gammaln, logsumexp

LOG2PI = np.log(2.0 * np.pi)
LOG_CHI2_1_MEAN = -1.2703628454614782  # E[log chi^2_1]


# ---------------------------------------------------------------------------
# Chart components
# ---------------------------------------------------------------------------
class GaussComp:
    kind = "gauss"

    def __init__(self, mu, cov):
        self.mu = np.asarray(mu, dtype=float)
        self.d = len(self.mu)
        self.cov = np.asarray(cov, dtype=float)
        self.L = np.linalg.cholesky(self.cov)
        self.Linv = solve_triangular(self.L, np.eye(self.d), lower=True)
        self.prec = self.Linv.T @ self.Linv
        self.logdet_half = float(np.sum(np.log(np.diag(self.L))))

    def to_z(self, x, rng=None):
        return self.Linv @ (x - self.mu)

    def from_z(self, z):
        return self.mu + self.L @ z

    def logpdf(self, x):
        z = self.to_z(x)
        return -0.5 * self.d * LOG2PI - self.logdet_half - 0.5 * z @ z

    def grad_logpdf(self, x):
        return -self.prec @ (x - self.mu)

    def Jt_grad(self, x, g):  # J = dx/dz at T^{-1}(x); returns J^T g
        return self.L.T @ g

    def maha(self, x):
        return float(np.linalg.norm(self.to_z(x)))

    def sample(self, rng):
        return self.mu + self.L @ rng.standard_normal(self.d)

    def principal_axis(self):
        lam, V = np.linalg.eigh(self.cov)
        return V[:, -1] * np.sqrt(lam[-1])


class ShearComp:
    """x = mu + L z, then x_t += gam ((x_dr - mu_dr)^2 - m2). Triangular with
    unit extra Jacobian (dr coordinate untouched by the shear)."""

    kind = "shear"

    def __init__(self, mu, cov, dr, t, gam, m2):
        assert dr != t
        self.base = GaussComp(mu, cov)
        self.mu = self.base.mu
        self.d = self.base.d
        self.dr, self.t, self.gam, self.m2 = dr, t, gam, m2

    def _unshear(self, x):
        u = np.array(x, dtype=float)
        u[self.t] -= self.gam * ((x[self.dr] - self.mu[self.dr]) ** 2 - self.m2)
        return u

    def to_z(self, x, rng=None):
        return self.base.to_z(self._unshear(x))

    def from_z(self, z):
        u = self.base.from_z(z)
        x = np.array(u)
        x[self.t] += self.gam * ((u[self.dr] - self.mu[self.dr]) ** 2 - self.m2)
        return x

    def logpdf(self, x):
        return self.base.logpdf(self._unshear(x))  # unit shear determinant

    def grad_logpdf(self, x):
        u = self._unshear(x)
        gy = -self.base.prec @ (u - self.mu)
        g = np.array(gy)
        g[self.dr] = gy[self.dr] - 2.0 * self.gam * (x[self.dr] - self.mu[self.dr]) * gy[self.t]
        return g

    def Jt_grad(self, x, g):
        gs = np.array(g, dtype=float)
        gs[self.dr] += 2.0 * self.gam * (x[self.dr] - self.mu[self.dr]) * g[self.t]
        return self.base.L.T @ gs

    def maha(self, x):
        return float(np.linalg.norm(self.to_z(x)))

    def sample(self, rng):
        return self.from_z(rng.standard_normal(self.d))

    def principal_axis(self):
        return self.base.principal_axis()


class ScaleComp:
    """Triangular non-centering chart: v = mu[j] + sig[j] z_j;
    x_i = mu[i] + sig[i] exp((a_i + b_i (v - vbar))/2) z_i  (i != j)."""

    kind = "scale"

    def __init__(self, j, mu, sig, alpha, beta, vbar):
        self.j = j
        self.mu = np.asarray(mu, dtype=float)
        self.sig = np.asarray(sig, dtype=float)
        self.d = len(self.mu)
        self.alpha = np.asarray(alpha, dtype=float)  # length d, alpha[j] unused = 0
        self.beta = np.asarray(beta, dtype=float)  # beta[j] = 0
        self.vbar = float(vbar)
        self.rest = np.array([i for i in range(self.d) if i != j])

    def _s_expo(self, v):
        # clamped: keeps the scale finite when the chain probes an extreme v;
        # to_z/from_z/logpdf all use this same exponent so the Jacobian is exact
        return np.clip(0.5 * (self.alpha + self.beta * (v - self.vbar)), -40.0, 40.0)

    def _s(self, v):
        return np.exp(self._s_expo(v))  # (d,), entry j = 1-ish

    def to_z(self, x, rng=None):
        j = self.j
        v = x[j]
        s = self._s(v)
        z = np.empty(self.d)
        z[j] = (v - self.mu[j]) / self.sig[j]
        r = self.rest
        z[r] = (x[r] - self.mu[r]) / (self.sig[r] * s[r])
        return z

    def from_z(self, z):
        j = self.j
        v = self.mu[j] + self.sig[j] * z[j]
        s = self._s(v)
        x = np.empty(self.d)
        x[j] = v
        r = self.rest
        x[r] = self.mu[r] + self.sig[r] * s[r] * z[r]
        return x

    def logpdf(self, x):
        j = self.j
        v = x[j]
        z = self.to_z(x)
        logdet = np.log(self.sig[j]) + np.sum(
            np.log(self.sig[self.rest]) + self._s_expo(v)[self.rest])
        out = -0.5 * self.d * LOG2PI - logdet - 0.5 * z @ z
        return out if not np.isnan(out) else -np.inf

    def grad_logpdf(self, x):
        j = self.j
        v = x[j]
        s = self._s(v)
        z = self.to_z(x)
        g = np.empty(self.d)
        r = self.rest
        g[r] = -z[r] / (self.sig[r] * s[r])
        g[j] = (-z[j] / self.sig[j]
                + 0.5 * float(np.sum(self.beta[r] * (z[r] ** 2 - 1.0))))
        return g

    def Jt_grad(self, x, g):
        j = self.j
        v = x[j]
        s = self._s(v)
        z = self.to_z(x)
        r = self.rest
        out = np.empty(self.d)
        out[r] = self.sig[r] * s[r] * g[r]
        out[j] = self.sig[j] * (g[j] + 0.5 * float(
            np.sum(self.beta[r] * self.sig[r] * s[r] * z[r] * g[r])))
        return out

    def maha(self, x):
        return float(np.linalg.norm(self.to_z(x)))

    def sample(self, rng):
        return self.from_z(rng.standard_normal(self.d))

    def principal_axis(self):
        return None  # no meaningful linear axis; isotropic probes only


class MultiShearComp:
    """Composed triangular shears on DISJOINT pairs (all applied in one chart):
    x_t += gam ((x_dr - mu_dr)^2 - m2) for each (dr, t, gam, m2). A target that
    curves in several pairs simultaneously (Rosenbrock) needs ONE chart that
    absorbs all of them — separate single-pair mixture components each leave
    the other curvatures in their residual. Unit extra determinant."""

    kind = "shear"

    def __init__(self, mu, cov, pairs):
        self.base = GaussComp(mu, cov)
        self.mu = self.base.mu
        self.d = self.base.d
        self.pairs = list(pairs)  # (dr, t, gam, m2); t's unique, t not any dr

    def _unshear(self, x):
        u = np.array(x, dtype=float)
        for dr, t, g, m2 in self.pairs:
            u[t] -= g * ((x[dr] - self.mu[dr]) ** 2 - m2)
        return u

    def to_z(self, x, rng=None):
        return self.base.to_z(self._unshear(x))

    def from_z(self, z):
        u = self.base.from_z(z)
        x = np.array(u)
        for dr, t, g, m2 in self.pairs:
            x[t] += g * ((u[dr] - self.mu[dr]) ** 2 - m2)
        return x

    def logpdf(self, x):
        return self.base.logpdf(self._unshear(x))

    def grad_logpdf(self, x):
        u = self._unshear(x)
        gy = -self.base.prec @ (u - self.mu)
        g = np.array(gy)
        for dr, t, gam, _ in self.pairs:
            g[dr] -= 2.0 * gam * (x[dr] - self.mu[dr]) * gy[t]
        return g

    def Jt_grad(self, x, g):
        gs = np.array(g, dtype=float)
        for dr, t, gam, _ in self.pairs:
            gs[dr] += 2.0 * gam * (x[dr] - self.mu[dr]) * g[t]
        return self.base.L.T @ gs

    def maha(self, x):
        return float(np.linalg.norm(self.to_z(x)))

    def sample(self, rng):
        return self.from_z(rng.standard_normal(self.d))

    def principal_axis(self):
        return self.base.principal_axis()


class TComp:
    """Student-t chart via the exact Gaussian scale-mixture auxiliary:

        lam ~ Gamma(nu/2, nu/2),   x = mu + L z / sqrt(lam).

    to_z SAMPLES lam from its exact conditional  lam | x ~ Gamma((nu+d)/2,
    (nu+delta)/2)  (a Gibbs step on the auxiliary); conditional on (k, lam) the
    chart is affine-Gaussian, the latent Hamiltonian keeps the exact harmonic
    form, and the component density q_k is the marginal multivariate-t (exact).
    The lam is transition-scoped state: from_z / Jt_grad reuse the lam set by
    the preceding to_z (single-threaded kernel). Deterministic contexts
    (rng=None: caches, trust, maha) use the conditional mean."""

    kind = "tcomp"

    def __init__(self, mu, cov, nu=4.0):
        self.mu = np.asarray(mu, dtype=float)
        self.d = len(self.mu)
        self.cov = np.asarray(cov, dtype=float)
        self.nu = float(nu)
        self.L = np.linalg.cholesky(self.cov)
        self.Linv = solve_triangular(self.L, np.eye(self.d), lower=True)
        self.prec = self.Linv.T @ self.Linv
        self.logdet_half = float(np.sum(np.log(np.diag(self.L))))
        self._lognorm = (gammaln((self.nu + self.d) / 2) - gammaln(self.nu / 2)
                         - 0.5 * self.d * np.log(self.nu * np.pi) - self.logdet_half)
        self._lam = 1.0

    def _delta(self, x):
        w = self.Linv @ (x - self.mu)
        return float(w @ w)

    def logpdf(self, x):
        return float(self._lognorm
                     - 0.5 * (self.nu + self.d) * np.log1p(self._delta(x) / self.nu))

    def grad_logpdf(self, x):
        return -(self.nu + self.d) / (self.nu + self._delta(x)) * (self.prec @ (x - self.mu))

    def to_z(self, x, rng=None):
        delta = self._delta(x)
        shape = 0.5 * (self.nu + self.d)
        rate = 0.5 * (self.nu + delta)
        lam = rng.gamma(shape, 1.0 / rate) if rng is not None else shape / rate
        self._lam = float(max(lam, 1e-12))
        return np.sqrt(self._lam) * (self.Linv @ (x - self.mu))

    def from_z(self, z):
        return self.mu + (self.L @ z) / np.sqrt(self._lam)

    def Jt_grad(self, x, g):
        return (self.L.T @ g) / np.sqrt(self._lam)

    def maha(self, x):
        return float(np.sqrt(self._delta(x)))

    def sample(self, rng):
        lam = rng.gamma(0.5 * self.nu, 2.0 / self.nu)
        return self.mu + (self.L @ rng.standard_normal(self.d)) / np.sqrt(max(lam, 1e-12))

    def principal_axis(self):
        lam, V = np.linalg.eigh(self.cov)
        return V[:, -1] * np.sqrt(lam[-1])


class HierComp:
    """Hierarchical location-scale chart (learned non-centering):

        x_v = a + b z_v            (log-scale driver)
        x_l = m + s_l z_l          (group location — a COORDINATE, not a constant)
        x_j = x_l + c_j e^{gam (x_v - vbar)} z_j   for children j in C
        x_k = mu_k + sig_k z_k     otherwise.

    Triangular with explicit Jacobian b s_l (prod c_j) e^{|C| gam (x_v - vbar)}
    (prod sig_k). Exactly the practitioner's non-centered reparameterization of
    eight-schools-style hierarchies (theta_j = mu + tau z_j at gam=1), which
    ScaleComp cannot express because its location is constant."""

    kind = "hier"

    def __init__(self, v, loc, children, mu, sig, cs, gam, vbar):
        self.v, self.loc = v, loc
        self.C = np.asarray(children, dtype=int)
        self.mu = np.asarray(mu, dtype=float)
        self.sig = np.asarray(sig, dtype=float)
        self.cs = np.asarray(cs, dtype=float)  # length d; entries only used at C
        self.gam, self.vbar = float(gam), float(vbar)
        self.d = len(self.mu)
        inC = set(self.C.tolist()) | {v, loc}
        self.other = np.asarray([k for k in range(self.d) if k not in inC], dtype=int)

    def _tau_expo(self, xv):
        # clamped like TriComp/ScaleComp: map + Jacobian share the exponent
        return float(np.clip(self.gam * (xv - self.vbar), -40.0, 40.0))

    def _tau(self, xv):
        return np.exp(self._tau_expo(xv))

    def to_z(self, x, rng=None):
        z = np.empty(self.d)
        z[self.v] = (x[self.v] - self.mu[self.v]) / self.sig[self.v]
        z[self.loc] = (x[self.loc] - self.mu[self.loc]) / self.sig[self.loc]
        t = self._tau(x[self.v])
        z[self.C] = (x[self.C] - x[self.loc]) / (self.cs[self.C] * t)
        z[self.other] = (x[self.other] - self.mu[self.other]) / self.sig[self.other]
        return z

    def from_z(self, z):
        x = np.empty(self.d)
        x[self.v] = self.mu[self.v] + self.sig[self.v] * z[self.v]
        x[self.loc] = self.mu[self.loc] + self.sig[self.loc] * z[self.loc]
        t = self._tau(x[self.v])
        x[self.C] = x[self.loc] + self.cs[self.C] * t * z[self.C]
        x[self.other] = self.mu[self.other] + self.sig[self.other] * z[self.other]
        return x

    def _logdet(self, xv):
        return (np.log(self.sig[self.v]) + np.log(self.sig[self.loc])
                + float(np.sum(np.log(self.cs[self.C])))
                + len(self.C) * self._tau_expo(xv)
                + float(np.sum(np.log(self.sig[self.other]))))

    def logpdf(self, x):
        z = self.to_z(x)
        out = float(-0.5 * self.d * LOG2PI - 0.5 * z @ z - self._logdet(x[self.v]))
        return out if not np.isnan(out) else -np.inf

    def grad_logpdf(self, x):
        z = self.to_z(x)
        t = self._tau(x[self.v])
        g = np.empty(self.d)
        g[self.other] = -z[self.other] / self.sig[self.other]
        g[self.C] = -z[self.C] / (self.cs[self.C] * t)
        g[self.loc] = -z[self.loc] / self.sig[self.loc] + float(
            np.sum(z[self.C] / (self.cs[self.C] * t)))
        g[self.v] = (-z[self.v] / self.sig[self.v]
                     + self.gam * float(np.sum(z[self.C] ** 2))
                     - len(self.C) * self.gam)
        return g

    def Jt_grad(self, x, g):
        z = self.to_z(x)
        t = self._tau(x[self.v])
        out = np.empty(self.d)
        out[self.other] = self.sig[self.other] * g[self.other]
        out[self.C] = self.cs[self.C] * t * g[self.C]
        out[self.loc] = self.sig[self.loc] * (g[self.loc] + float(np.sum(g[self.C])))
        out[self.v] = self.sig[self.v] * (
            g[self.v] + self.gam * float(np.sum(self.cs[self.C] * t * z[self.C] * g[self.C])))
        return out

    def maha(self, x):
        return float(np.linalg.norm(self.to_z(x)))

    def sample(self, rng):
        return self.from_z(rng.standard_normal(self.d))

    def principal_axis(self):
        return None


class PolarComp:
    """Polar/ring transport chart on coordinate pair (i, j):

        r = R + sig_r z_i,  psi = c z_j,
        (x_i, x_j) = mu_(i,j) + r (cos psi, sin psi),
        x_rest = mu_rest + sig_rest z_rest.

    With c ~ 4 the angular law is a wrapped Gaussian ~ uniform on the circle, and
    the exact component density is a winding-number logsumexp (branches k with
    z_j = (theta + 2 pi k)/c; |det DT| = sig_r c r is branch-independent at fixed
    x). The chart map is deliberately non-injective (wrapping), so to_z SAMPLES
    the winding branch ~ phi(z) — the exact conditional over preimages — which is
    required for detailed balance (a fixed principal branch would break the
    reverse path when a trajectory wraps). Harmonic z_j dynamics circulate the
    ring — the fix for arc-chart hopping. Truncation at |k| <= Kw is below double
    precision for c <= 5."""

    kind = "polar"

    def __init__(self, i, j, mu, R, sig_r, sig_rest, c=4.0):
        self.i, self.j = i, j
        self.mu = np.asarray(mu, dtype=float)
        self.d = len(self.mu)
        self.R, self.sr, self.c = float(R), float(sig_r), float(c)
        self.rest = np.array([k for k in range(self.d) if k not in (i, j)])
        self.sig = np.asarray(sig_rest, dtype=float)  # length d; entries i,j unused
        self.Kw = int(np.ceil(4.0 * c / (2 * np.pi))) + 1
        self._log_rest_det = float(np.sum(np.log(self.sig[self.rest]))) if len(self.rest) else 0.0

    def _uv(self, x):
        u0, u1 = x[self.i] - self.mu[self.i], x[self.j] - self.mu[self.j]
        r = np.sqrt(u0 * u0 + u1 * u1) + 1e-300
        return u0, u1, r, np.arctan2(u1, u0)

    def _branches(self, x):
        u0, u1, r, th = self._uv(x)
        zr = (r - self.R) / self.sr
        ks = np.arange(-self.Kw, self.Kw + 1)
        zth = (th + 2 * np.pi * ks) / self.c
        return zr, zth, r

    def to_z(self, x, rng=None):
        zr, zth, r = self._branches(x)
        lw = -0.5 * zth**2
        if rng is not None:
            p = np.exp(lw - logsumexp(lw))
            cp = np.cumsum(p)
            pick = min(int(np.searchsorted(cp, rng.random() * cp[-1])), len(zth) - 1)
        else:
            pick = int(np.argmax(lw))
        z = np.empty(self.d)
        z[self.i] = zr
        z[self.j] = zth[pick]
        z[self.rest] = (x[self.rest] - self.mu[self.rest]) / self.sig[self.rest]
        return z

    def from_z(self, z):
        r = self.R + self.sr * z[self.i]
        psi = self.c * z[self.j]
        x = np.empty(self.d)
        x[self.i] = self.mu[self.i] + r * np.cos(psi)
        x[self.j] = self.mu[self.j] + r * np.sin(psi)
        x[self.rest] = self.mu[self.rest] + self.sig[self.rest] * z[self.rest]
        return x

    def logpdf(self, x):
        zr, zth, r = self._branches(x)
        zrest = (x[self.rest] - self.mu[self.rest]) / self.sig[self.rest]
        base = -0.5 * self.d * LOG2PI - 0.5 * (zr**2 + float(zrest @ zrest))
        logdet = np.log(self.sr) + np.log(self.c) + np.log(r) + self._log_rest_det
        return float(logsumexp(base - 0.5 * zth**2) - logdet)

    def grad_logpdf(self, x):
        zr, zth, r = self._branches(x)
        u0, u1, _, _ = self._uv(x)
        w = np.exp(-0.5 * zth**2 - logsumexp(-0.5 * zth**2))
        zth_bar = float(w @ zth)  # softmax-averaged branch coordinate
        g = np.zeros(self.d)
        zrest = (x[self.rest] - self.mu[self.rest]) / self.sig[self.rest]
        g[self.rest] = -zrest / self.sig[self.rest]
        # d z_r / dx = u/(r sr); d theta / dx = (-u1, u0)/r^2; d log r / dx = u/r^2
        gr = -zr / (r * self.sr) - 1.0 / r**2  # coefficient on u (radial direction + logdet)
        gth = -zth_bar / (self.c * r**2)  # coefficient on (-u1, u0)
        g[self.i] = gr * u0 + gth * (-u1)
        g[self.j] = gr * u1 + gth * u0
        return g

    def Jt_grad(self, x, g):
        u0, u1, r, _ = self._uv(x)
        out = np.empty(self.d)
        out[self.rest] = self.sig[self.rest] * g[self.rest]
        cth, sth = u0 / r, u1 / r
        out[self.i] = self.sr * (cth * g[self.i] + sth * g[self.j])
        out[self.j] = self.c * (-u1 * g[self.i] + u0 * g[self.j])
        return out

    def maha(self, x):
        return float(np.linalg.norm(self.to_z(x)))

    def sample(self, rng):
        return self.from_z(rng.standard_normal(self.d))

    def principal_axis(self):
        return None


class TDefense:
    """Multivariate Student-t for the defensive global branch."""

    def __init__(self, mu, cov, nu=4.0):
        self.mu = np.asarray(mu, dtype=float)
        self.d = len(self.mu)
        self.nu = float(nu)
        self.L = np.linalg.cholesky(np.asarray(cov, dtype=float))
        self.logdet_half = float(np.sum(np.log(np.diag(self.L))))
        self._lognorm = (gammaln((self.nu + self.d) / 2) - gammaln(self.nu / 2)
                         - 0.5 * self.d * np.log(self.nu * np.pi) - self.logdet_half)

    def logpdf(self, x):
        z = solve_triangular(self.L, x - self.mu, lower=True)
        return float(self._lognorm - 0.5 * (self.nu + self.d) * np.log1p(z @ z / self.nu))

    def sample(self, rng):
        w = rng.chisquare(self.nu) / self.nu
        return self.mu + (self.L @ rng.standard_normal(self.d)) / np.sqrt(w)


# ---------------------------------------------------------------------------
# Mixture of charts
# ---------------------------------------------------------------------------
class MixtureRef:
    def __init__(self, comps, log_ws):
        self.comps = list(comps)
        self.K = len(comps)
        self.d = comps[0].d
        lw = np.array(log_ws, dtype=float)  # copy: sanitized in place below
        # a NaN weight (poisoned forward-KL estimate) must not propagate through
        # logsumexp into every weight; an unevaluable component gets weight 0
        lw[~np.isfinite(lw)] = -np.inf
        if not np.any(np.isfinite(lw)):
            lw = np.zeros(self.K)
        self.log_ws = lw - logsumexp(lw)
        # Batch the plain-Gaussian components: one einsum instead of K Python
        # calls per density evaluation (the production hot path).
        gidx = [k for k, c in enumerate(comps) if type(c) is GaussComp]
        self._gidx = np.asarray(gidx, dtype=int)
        self._tidx = [k for k in range(self.K) if type(comps[k]) is not GaussComp]
        if gidx:
            self._gmu = np.stack([comps[k].mu for k in gidx])
            self._gLinv = np.stack([comps[k].Linv for k in gidx])
            self._gprec = np.stack([comps[k].prec for k in gidx])
            self._glogdet = np.array([comps[k].logdet_half for k in gidx])

    def comp_logpdf(self, x):
        out = np.empty(self.K)
        if len(self._gidx):
            diff = x[None, :] - self._gmu
            z = np.einsum("gij,gj->gi", self._gLinv, diff)
            out[self._gidx] = (-0.5 * self.d * LOG2PI - self._glogdet
                               - 0.5 * np.einsum("gi,gi->g", z, z))
        for k in self._tidx:
            out[k] = self.comps[k].logpdf(x)
        return out

    def logq(self, x):
        return float(logsumexp(self.log_ws + self.comp_logpdf(x)))

    def logq_and_resp(self, x):
        lc = self.log_ws + self.comp_logpdf(x)
        lq = logsumexp(lc)
        return float(lq), np.exp(lc - lq)

    def grad_logq(self, x):
        lq, resp = self.logq_and_resp(x)
        g = np.zeros(self.d)
        if len(self._gidx):
            rg = resp[self._gidx]
            diff = x[None, :] - self._gmu
            g -= np.einsum("g,gij,gj->i", rg, self._gprec, diff)
        for k in self._tidx:
            if resp[k] > 1e-12:
                g += resp[k] * self.comps[k].grad_logpdf(x)
        return g

    def min_maha(self, x):
        return min(c.maha(x) for c in self.comps)

    def to_chart(self, k, x, rng=None):
        return self.comps[k].to_z(x, rng)

    def from_chart(self, k, z):
        return self.comps[k].from_z(z)

    def sample(self, rng, n=1):
        ks = rng.choice(self.K, size=n, p=np.exp(self.log_ws))
        xs = np.array([self.comps[k].sample(rng) for k in ks])
        return xs if n > 1 else xs[0]


# ---------------------------------------------------------------------------
# Pathfinder-lite warm-up (Gaussian components + gradient anchors)
# ---------------------------------------------------------------------------
def _regularized_cov(hess, floor_frac=1e-8, abs_floor=1e-10):
    hess = 0.5 * (hess + hess.T)
    lam, V = np.linalg.eigh(hess)
    lam = np.abs(lam)
    lam = np.maximum(lam, max(lam.max() * floor_frac, abs_floor))
    return V @ np.diag(1.0 / lam) @ V.T


def build_components(target, rng, n_starts=10, max_iter=1000, elbo_samples=16, max_K=12):
    d = target.d
    anchors = []

    def fg(x):
        u, g = target.value_and_grad(x)
        anchors.append((np.asarray(x, dtype=float).copy(), u, g))
        return u, g

    paths = []
    for _ in range(n_starts):
        x0 = rng.standard_normal(d) * 2.0 * target.init_scale
        iterates = [x0.copy()]
        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": max_iter, "gtol": 1e-8, "ftol": 1e-14},
                       callback=lambda xk: iterates.append(xk.copy()))
        iterates.append(np.asarray(res.x, dtype=float))
        paths.append(iterates)

    components = []
    for iterates in paths:
        n = len(iterates)
        idxs = sorted({max(0, int(n * f) - 1) for f in (0.2, 0.4, 0.6, 0.8, 1.0)} | {n - 1})
        for i in idxs:
            mu = iterates[i]
            cov = _regularized_cov(target.hessian(mu))
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                continue
            zs = rng.standard_normal((elbo_samples, d))
            us = [target.U(mu + L @ z) for z in zs]
            if not np.all(np.isfinite(us)):
                continue
            elbo = -float(np.mean(us)) + float(np.sum(np.log(np.diag(L))))
            components.append((elbo, mu, cov))

    components.sort(key=lambda c: -c[0])
    elbo_floor = components[0][0] - 30.0
    kept = []
    for elbo, mu, cov in components:
        if elbo < elbo_floor:
            continue
        dup = False
        for _, mu2, prec2 in kept:
            diff = mu - mu2
            if np.sqrt(diff @ prec2 @ diff) < 1.2:
                dup = True
                break
        if not dup:
            kept.append((cov, mu, np.linalg.inv(cov)))
        if len(kept) >= max_K:
            break
    comps = [GaussComp(mu, cov) for cov, mu, _ in kept]

    if target.defensive and comps:
        mus = np.array([c.mu for c in comps])
        spread = np.mean([np.diag(c.cov) for c in comps], axis=0)
        if len(comps) > 1:
            spread = spread + np.var(mus, axis=0)
        comps.append(GaussComp(np.mean(mus, axis=0), np.diag(9.0 * spread + 1e-6)))
    return comps, anchors


# ---------------------------------------------------------------------------
# Transport-chart fitting (Milestone B) from warm-up chain states
# ---------------------------------------------------------------------------
def fit_shear_comp(states, r2_min=0.45, reg=1e-3, exclude=()):
    """Detect x_t ~ ... + gam (x_dr)^2 structure via the PARTIAL R^2 of x_dr^2
    beyond the linear term (a skewed visited distribution otherwise leaks an
    unmodeled linear component into (x-mean)^2 and deflates plain R^2; the
    linear part belongs to the base Gaussian's correlation, which the unshear
    covariance absorbs). Exact when the target is a sheared Gaussian (banana)."""
    X = np.asarray(states)
    n, d = X.shape
    if n < 50:
        return None
    Xc = X - X.mean(axis=0)
    var = Xc.var(axis=0) + 1e-300
    best = None
    for dr in range(d):
        xd = Xc[:, dr]
        q = xd**2
        qc = q - q.mean()
        qc = qc - (qc @ xd / (xd @ xd + 1e-300)) * xd  # partial out the linear term
        vq = qc @ qc
        if vq <= 0:
            continue
        cov_qt = qc @ Xc / n  # (d,)
        r2 = (n * cov_qt**2) / (vq * var)
        r2[dr] = 0.0
        for tt in range(d):
            if (dr, tt) in exclude:
                r2[tt] = 0.0
        t = int(np.argmax(r2))
        if best is None or r2[t] > best[0]:
            best = (float(r2[t]), dr, t, float(n * cov_qt[t] / vq))
    if best is None or best[0] < r2_min:
        return None
    _, dr, t, gam = best
    mu_dr = float(X[:, dr].mean())
    m2 = float(((X[:, dr] - mu_dr) ** 2).mean())
    Y = np.array(X)
    Y[:, t] -= gam * ((X[:, dr] - mu_dr) ** 2 - m2)
    mu = Y.mean(axis=0)
    cov = np.cov(Y.T) + reg * np.diag(Y.var(axis=0) + 1e-9)
    # Chain-based moments lower-bound the true spread along the slow driver
    # direction while mixing is incomplete; over-dispersing the reference is
    # the safe side (keeps pi/q bounded, cache absorbs the smooth residual).
    D = np.ones(len(mu))
    D[dr] = 1.5
    cov = cov * np.outer(D, D)
    try:
        return ShearComp(mu, cov, dr, t, gam, m2)
    except np.linalg.LinAlgError:
        return None


def fit_scale_comp(states, r2_min=0.12, beta_min=0.1, exclude=()):
    """Detect a conditional-scale driver: log((x_i - mu_i)^2) ~ a_i + b_i v.
    Exact for Neal's funnel (b_i = 1). Bias-corrected via E[log chi^2_1]."""
    X = np.asarray(states)
    n, d = X.shape
    if n < 50 or d < 3:
        return None
    med = np.median(X, axis=0)
    best = None
    for j in range(d):
        if j in exclude:
            continue
        v = X[:, j]
        vc = v - v.mean()
        vv = vc @ vc
        if vv <= 0:
            continue
        r2s, betas, alphas = [], [], []
        for i in range(d):
            if i == j:
                continue
            ell = np.log((X[:, i] - med[i]) ** 2 + 1e-300)
            b = float(vc @ (ell - ell.mean()) / vv)
            resid = ell - ell.mean() - b * vc
            r2 = 1.0 - float(resid @ resid) / max(float((ell - ell.mean()) @ (ell - ell.mean())), 1e-300)
            r2s.append(r2)
            betas.append(b)
            alphas.append(float(ell.mean() - LOG_CHI2_1_MEAN))
        score = float(np.mean(r2s))
        if best is None or score > best[0]:
            best = (score, j, np.array(betas), np.array(alphas))
    if best is None or best[0] < r2_min:
        return None
    score, j, betas, alphas_raw = best
    if np.mean(np.abs(betas)) < beta_min:
        return None
    mu = med.copy()
    mu[j] = float(X[:, j].mean())
    sig = np.ones(d)
    sig[j] = 1.5 * float(X[:, j].std() + 1e-12)  # driver over-dispersion (see shear)
    vbar = mu[j]
    alpha = np.zeros(d)
    beta = np.zeros(d)
    rest = [i for i in range(d) if i != j]
    # alpha folds sigma_i: s_i^2 e^{alpha} = conditional var at v = vbar
    beta[rest] = betas
    alpha[rest] = alphas_raw + beta[rest] * (float(np.mean(np.asarray(states)[:, j])) - vbar)
    return ScaleComp(j, mu, sig, alpha, beta, vbar)


def fit_polar_comp(states, cv_max=0.25, rmin_sds=4.0, exclude=()):
    """Detect ring structure: a coordinate pair whose radius about a fitted
    center concentrates (CV(r) << 0.52, the Gaussian-pair value). Center refined
    by descent on Var(r). Radial width over-dispersed 1.5x (see shear)."""
    X = np.asarray(states)
    n, d = X.shape
    if n < 60 or d < 2:
        return None
    def _descend(P2, c0, iters=30):
        c = c0.copy()
        for _ in range(iters):
            u = P2 - c
            r = np.linalg.norm(u, axis=1) + 1e-12
            c = c + np.mean((r - r.mean())[:, None] * (u / r[:, None]), axis=0)
        return c

    best = None
    for i in range(d):
        for j in range(i + 1, d):
            if (i, j) in exclude:
                continue
            P2 = X[:, [i, j]]
            # Candidate centers: data mean, Kasa algebraic fit, and Var(r)
            # descent from each. The Kasa fit alone finds tangent-circle
            # degeneracies on arc data (small R hugging the arc); the mean
            # alone is biased inward. Best-CV candidate wins, then hard gates.
            cands = [P2.mean(axis=0)]
            A = np.column_stack([2 * P2[:, 0], 2 * P2[:, 1], np.ones(n)])
            sol, *_ = np.linalg.lstsq(A, P2[:, 0] ** 2 + P2[:, 1] ** 2, rcond=None)
            cands.append(sol[:2])
            cands.append(_descend(P2, cands[0]))
            cands.append(_descend(P2, cands[1]))
            for c2 in cands:
                u = P2 - c2
                r = np.linalg.norm(u, axis=1) + 1e-12
                # Angular SUPPORT gates (density-free: a handful of dispersed
                # refined-probe points must count against a clustered chain):
                # >=10/24 occupied sectors and no empty half-circle, else R is
                # unidentifiable (short arc / tangent-circle degeneracy).
                th = np.arctan2(u[:, 1], u[:, 0])
                occ = np.unique(((th + np.pi) / (2 * np.pi / 24)).astype(int) % 24)
                if len(occ) < 10:
                    continue
                gaps = np.diff(np.concatenate([occ, [occ[0] + 24]]))
                if gaps.max() > 12:
                    continue
                # Density-balanced radius: mean of per-sector mean radii.
                bins = ((th + np.pi) / (2 * np.pi / 24)).astype(int) % 24
                r_bin = [r[bins == b].mean() for b in occ]
                R0 = float(np.mean(r_bin))
                sr = float(np.sqrt(np.mean((r - R0) ** 2)))
                if R0 <= 1e-9 or sr / R0 >= cv_max or R0 < rmin_sds * sr:
                    continue
                if best is None or sr / R0 < best[0]:
                    best = (sr / R0, i, j, c2, R0, sr)
    if best is None:
        return None
    _, i, j, c2, R0, sr = best
    mu_full = X.mean(axis=0)
    mu_full[[i, j]] = c2
    sig = np.maximum(X.std(axis=0), 1e-6)
    cand = PolarComp(i, j, mu_full, R0, 1.5 * sr + 1e-9, sig)
    # Validation gate (false positives on near-Gaussian posteriors steal
    # mixture weight): the polar chart must beat a plain full-d Gaussian on
    # mean data log-density by a margin, else it is not ring structure.
    try:
        g = GaussComp(X.mean(axis=0), np.cov(X.T) + 1e-3 * np.diag(X.var(axis=0) + 1e-9))
    except np.linalg.LinAlgError:
        return cand
    # Sector-BALANCED comparison: judged on raw density, a Gaussian centered on
    # the dominant arc cluster beats a (correct) uniform-angle ring chart —
    # coverage imbalance must not decide the gate. <=8 points per sector.
    u = X[:, [i, j]] - c2
    th = np.arctan2(u[:, 1], u[:, 0])
    bins = ((th + np.pi) / (2 * np.pi / 24)).astype(int) % 24
    sub_idx = []
    for b in np.unique(bins):
        ids = np.where(bins == b)[0]
        sub_idx.extend(ids[:8])
    sub = X[np.asarray(sub_idx)]
    lp_p = float(np.mean([cand.logpdf(x) for x in sub]))
    lp_g = float(np.mean([g.logpdf(x) for x in sub]))
    return cand if lp_p > lp_g - 1.0 else None


def fit_hier_comp(states, r2_min=0.15, slope_lo=1.0, slope_hi=3.0, min_children=3):
    """Detect hierarchical location-scale structure: children j whose deviation
    from a LOCATION COORDINATE x_l has log-square linear in a driver x_v with
    slope ~2 (Var(x_j - x_l | x_v) = c_j^2 e^{2 gam x_v}). Search over (v, l)
    pairs; fit gam (shared, = mean slope / 2), c_j, and affine params."""
    X = np.asarray(states)
    n, d = X.shape
    if n < 60 or d < 5:
        return None
    if d > 60:  # cap the O(d^2) pair scan by variance rank
        keep = np.argsort(-X.var(axis=0))[:60]
    else:
        keep = np.arange(d)
    best = None
    for v in keep:
        xv = X[:, v]
        vc = xv - xv.mean()
        vv = float(vc @ vc)
        if vv <= 1e-12:
            continue
        for loc in keep:
            if loc == v:
                continue
            D = X - X[:, loc][:, None]  # deviations from the location coordinate
            ell = np.log(D**2 + 1e-300)
            ellc = ell - ell.mean(axis=0)
            slopes = (vc @ ellc) / vv  # per-coordinate regression slopes
            resid = ellc - np.outer(vc, slopes)
            tot = np.sum(ellc**2, axis=0) + 1e-300
            r2 = 1.0 - np.sum(resid**2, axis=0) / tot
            ok = np.array([j for j in range(d)
                           if j not in (v, loc) and r2[j] >= r2_min
                           and slope_lo <= slopes[j] <= slope_hi])
            if len(ok) < min_children:
                continue
            score = float(len(ok) * np.mean(r2[ok]))
            if best is None or score > best[0]:
                best = (score, v, loc, ok, slopes, ell)
    if best is None:
        return None
    _, v, loc, C, slopes, ell = best
    xv = X[:, v]
    vbar = float(xv.mean())
    gam = float(np.clip(np.mean(slopes[C]) / 2.0, 0.4, 1.6))
    mu = X.mean(axis=0)
    sig = np.maximum(X.std(axis=0), 1e-6)
    sig[v] *= 1.5  # driver over-dispersion (see shear/scale)
    cs = np.ones(d)
    # log c_j from the mean of log((x_j-x_l)^2) - 2 gam (x_v - vbar), chi^2_1-corrected
    cs[C] = np.exp(0.5 * (np.mean(ell[:, C] - 2.0 * gam * (xv - vbar)[:, None], axis=0)
                          - LOG_CHI2_1_MEAN))
    return HierComp(int(v), int(loc), C, mu, sig, np.maximum(cs, 1e-9), gam, vbar)


def transport_sig(c):
    """Instance signature: gating must be per-instance, not per-kind — a
    Rosenbrock has 5 independent shears, a horseshoe has p scale funnels."""
    if c.kind == "shear":
        prs = getattr(c, "pairs", None)
        if prs is not None:
            return ("shear", tuple(sorted((p[0], p[1]) for p in prs)))
        return ("shear", ((c.dr, c.t),))
    if c.kind == "scale":
        return ("scale", c.j)
    if c.kind == "polar":
        return ("polar", c.i, c.j)
    if c.kind == "hier":
        return ("hier", c.v, c.loc)
    if c.kind == "tri":
        return ("tri",)
    return (c.kind,)


def fit_transport_comps(states, exclude_sigs=(), max_shear=6, max_scale=4, max_polar=2):
    comps = []
    ex = set(exclude_sigs)
    used = {p for s in ex if s[0] == "shear" for p in s[1]}
    Xw = np.array(states, dtype=float)
    pairs = []
    for _ in range(max_shear):
        sh = fit_shear_comp(Xw, exclude={(d1, t1) for d1 in range(Xw.shape[1])
                                         for t1 in used} | {(t1, d1) for d1 in range(Xw.shape[1])
                                                            for t1 in used})
        if sh is None:
            break
        dr, t, gam, m2 = sh.dr, sh.t, sh.gam, sh.m2
        mu_dr = sh.mu[dr]
        pairs.append((dr, t, gam, m2, mu_dr))
        used.update({t, dr})
        Xw[:, t] -= gam * ((Xw[:, dr] - mu_dr) ** 2 - m2)  # peel and re-scan
    if pairs:
        mu = Xw.mean(axis=0)
        cov = np.cov(Xw.T) + 1e-3 * np.diag(Xw.var(axis=0) + 1e-9)
        D = np.ones(Xw.shape[1])
        # Mild per-driver widening: 1.5x compounds ruinously across many pairs
        # (5 drivers -> ~4 nats of R variance); path+probe data now has spread.
        wid = 1.5 if len(pairs) == 1 else 1.15
        for dr, _, _, _, _ in pairs:
            D[dr] = wid
        cov = cov * np.outer(D, D)
        try:
            comps.append(MultiShearComp(mu, cov, [(dr, t, g, m2) for dr, t, g, m2, _ in pairs]))
        except np.linalg.LinAlgError:
            pass
    used_j = {s[1] for s in ex if s[0] == "scale"}
    for _ in range(max_scale):
        sc = fit_scale_comp(states, exclude=used_j)
        if sc is None:
            break
        comps.append(sc)
        used_j.add(sc.j)
    used_p = {(s[1], s[2]) for s in ex if s[0] == "polar"}
    for _ in range(max_polar):
        po = fit_polar_comp(states, exclude=used_p)
        if po is None:
            break
        comps.append(po)
        used_p.add((po.i, po.j))
    hi = fit_hier_comp(states)
    if hi is not None and transport_sig(hi) not in ex:
        comps.append(hi)
    if ("tri",) not in ex:
        tr = fit_tri_comp(states)
        if tr is not None:
            comps.append(tr)
    return comps


# ---------------------------------------------------------------------------
# Forward-KL mixture weights (overlap-aware) + pruning
# ---------------------------------------------------------------------------
def sample_pool(target, comps, rng, n_per=32):
    xs, us = [], []
    for c in comps:
        for _ in range(n_per):
            x = c.sample(rng)
            u = target.U(x)
            if np.isfinite(u):
                xs.append(x)
                us.append(u)
    return xs, us


def _comp_logpdfs(xs, comps):
    return np.array([[c.logpdf(x) for x in xs] for c in comps])


def fit_weights_forward_kl(pool_xs, pool_us, comps, gen_comps=None, iters=200):
    if gen_comps is None:
        gen_comps = comps
    lpdf = _comp_logpdfs(pool_xs, comps)
    K, N = lpdf.shape
    lpdf_gen = _comp_logpdfs(pool_xs, gen_comps)
    log_m = logsumexp(lpdf_gen, axis=0) - np.log(len(gen_comps))
    log_wbar = -np.asarray(pool_us) - log_m
    wbar = np.exp(log_wbar - logsumexp(log_wbar))
    wbar = np.minimum(wbar, np.sqrt(N) * np.mean(wbar))  # Ionides-style truncation
    wbar = wbar / wbar.sum()
    log_w = np.full(K, -np.log(K))
    for _ in range(iters):
        lr = log_w[:, None] + lpdf
        lr = lr - logsumexp(lr, axis=0, keepdims=True)
        new_w = np.exp(lr) @ wbar
        log_w = np.log(np.maximum(new_w, 1e-300))
        log_w -= logsumexp(log_w)
    return log_w, lpdf, wbar


def prune_components(comps, log_ws, min_w=1e-4):
    keep = [k for k in range(len(comps)) if np.exp(log_ws[k]) >= min_w]
    if not keep:
        keep = [int(np.argmax(log_ws))]
    comps = [comps[k] for k in keep]
    lw = np.array([log_ws[k] for k in keep])
    return comps, lw - logsumexp(lw)


# ---------------------------------------------------------------------------
# Residual surrogates (unchanged from Milestone A; forces mapped via Jt_grad)
# ---------------------------------------------------------------------------
class ZeroCache:
    kind = "zero"

    def query(self, z):
        return 0.0 * z

    def trust(self, z):
        return 0.0


def _active_subspace(F, energy_frac=0.95, s_max=12):
    C = F.T @ F / max(1, len(F))
    if not np.all(np.isfinite(C)):
        C = np.eye(F.shape[1])
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    csum = np.cumsum(lam) / max(lam.sum(), 1e-300)
    s = int(np.searchsorted(csum, energy_frac) + 1)
    return V[:, : max(1, min(s, s_max, F.shape[1]))]


class ChartCache:
    kind = "affine"

    def __init__(self, Z, F, k_query=16, s_max=12):
        self.n = len(Z)
        self.Z = np.asarray(Z)
        F = np.asarray(F)
        self.V = _active_subspace(F, s_max=s_max)
        self.s = self.V.shape[1]
        self.Y = self.Z @ self.V
        self.Fs = F @ self.V
        self.k = min(k_query, self.n)
        self.tree = cKDTree(self.Y)
        if self.n >= 2:
            nn_d, _ = self.tree.query(self.Y, k=2)
            self.rho = 2.0 * float(np.percentile(nn_d[:, 1], 95))
        else:
            self.rho = 1.0
        norms = np.linalg.norm(self.Fs, axis=1)
        self.f_max = 5.0 * float(np.percentile(norms, 99)) if self.n else 0.0
        self._fit_affine()

    def _fit_affine(self):
        s = self.s
        k_nb = min(max(2 * s + 4, 8), self.n)
        self.B = np.zeros((self.n, s, s))
        if self.n < s + 2:
            return
        _, nb = self.tree.query(self.Y, k=k_nb)
        for i in range(self.n):
            idx = np.atleast_1d(nb[i])
            dY = self.Y[idx] - self.Y[i]
            dF = self.Fs[idx] - self.Fs[i]
            A = dY.T @ dY + 1e-8 * np.trace(dY.T @ dY) / s * np.eye(s) + 1e-12 * np.eye(s)
            self.B[i] = np.linalg.solve(A, dY.T @ dF).T

    def _predict_u(self, y):
        # Brute-force kNN: for n <= a few thousand anchors in s <= 12 dims, one
        # vectorized distance pass beats a cKDTree call's Python overhead.
        d2 = np.einsum("ns,ns->n", self.Y - y[None, :], self.Y - y[None, :])
        if self.n > self.k:
            idx = np.argpartition(d2, self.k - 1)[: self.k]
        else:
            idx = np.arange(self.n)
        dist = np.sqrt(d2[idx])
        scale = dist.max() if dist.max() > 0 else 1.0
        w = np.exp(-0.5 * (dist / scale) ** 2)
        w /= w.sum()
        dY = y[None, :] - self.Y[idx]
        f = w @ (self.Fs[idx] + np.einsum("kij,kj->ki", self.B[idx], dY))
        dmin = dist.min()
        if dmin > self.rho:
            f = f * np.exp(-(((dmin - self.rho) / self.rho) ** 2))
        nf = np.linalg.norm(f)
        if self.f_max > 0 and nf > self.f_max:
            f = f * (self.f_max / nf)
        return f

    def query(self, z):
        return self.V @ self._predict_u(z @ self.V)

    def trust(self, z):
        y = z @ self.V
        dmin = float(np.sqrt(np.min(np.einsum("ns,ns->n", self.Y - y, self.Y - y))))
        return 1.0 if dmin <= self.rho else float(np.exp(-(((dmin - self.rho) / self.rho) ** 2)))


class ScalarRBFCache:
    kind = "rbf"

    def __init__(self, Z, F, R_vals, M=96, gamma=1.8, s_max=12, ridge=1e-4):
        Z = np.asarray(Z)
        F = np.asarray(F)
        R_vals = np.asarray(R_vals, dtype=float)
        self.V = _active_subspace(F, s_max=s_max)
        self.s = self.V.shape[1]
        U = Z @ self.V
        Fs = F @ self.V
        n = len(U)
        M = int(min(M, max(8, n // 2)))
        centers = [int(np.argmax(np.linalg.norm(Fs, axis=1)))]
        dmin = np.linalg.norm(U - U[centers[0]], axis=1)
        for _ in range(M - 1):
            j = int(np.argmax(dmin))
            centers.append(j)
            dmin = np.minimum(dmin, np.linalg.norm(U - U[j], axis=1))
        self.C = U[centers]
        ctree = cKDTree(self.C)
        cd, _ = ctree.query(self.C, k=min(4, M))
        near = cd[:, -1] if M > 1 else np.ones(M)
        self.ell = gamma * np.maximum(near, 1e-8 * (near.max() if M > 1 else 1.0) + 1e-12)
        Phi, Grows = self._design(U)
        b_val = R_vals - np.mean(R_vals)
        sig_R = np.std(b_val) + 1e-12
        sig_g = np.std(Fs) + 1e-12
        A = np.vstack([Phi / sig_R, Grows.reshape(n * self.s, M) / sig_g])
        b = np.concatenate([b_val / sig_R, Fs.reshape(-1) / sig_g])
        AtA = A.T @ A
        lam = ridge * (np.trace(AtA) / M + 1e-12)
        self.a = np.linalg.solve(AtA + lam * np.eye(M), A.T @ b)

    def _design(self, U):
        diff = U[:, None, :] - self.C[None, :, :]
        r = np.linalg.norm(diff, axis=2) / self.ell[None, :]
        om = np.where(r < 1.0, 1.0 - r, 0.0)
        Phi = om**4 * (4.0 * r + 1.0)
        fac = -20.0 * om**3 / (self.ell[None, :] ** 2)
        Grows = diff * fac[:, :, None]
        return Phi, np.transpose(Grows, (0, 2, 1))

    def _grad_u(self, u):
        diff = u[None, :] - self.C
        r = np.linalg.norm(diff, axis=1) / self.ell
        om = np.maximum(1.0 - r, 0.0)
        fac = -20.0 * om**3 / (self.ell**2) * self.a
        return fac @ diff

    def query(self, z):
        return self.V @ self._grad_u(z @ self.V)

    def trust(self, z):
        r = np.linalg.norm((z @ self.V)[None, :] - self.C, axis=1) / self.ell
        return 1.0 if float(np.min(r)) < 1.0 else 0.0


def _kappa(cache, Z_val, F_val):
    num = den = 0.0
    for z, f in zip(Z_val, F_val):
        pred = cache.query(np.asarray(z))
        num += float(np.sum((f - pred) ** 2))
        den += float(np.sum(f**2))
    return num / max(den, 1e-300)


def select_cache(Z, F, R_vals, eps=0.05):
    n = len(Z)
    if n < 25:
        return ZeroCache(), 1.0
    F_arr = np.asarray(F)
    if np.mean(np.sum(F_arr**2, axis=1)) < 1e-4 * F_arr.shape[1]:
        return ZeroCache(), 1.0
    val = list(range(0, n, 5))
    tr = [i for i in range(n) if i % 5 != 0]
    Ztr, Ftr, Rtr = [Z[i] for i in tr], [F[i] for i in tr], [R_vals[i] for i in tr]
    Zv, Fv = [Z[i] for i in val], [F[i] for i in val]
    cands = []
    try:
        cands.append((_kappa(ChartCache(Ztr, Ftr), Zv, Fv), "affine"))
    except Exception:
        pass
    try:
        cands.append((_kappa(ScalarRBFCache(Ztr, Ftr, Rtr), Zv, Fv), "rbf"))
    except Exception:
        pass
    cands = [c for c in cands if np.isfinite(c[0])]
    if not cands:
        return ZeroCache(), 1.0
    kappa, kind = min(cands)
    if kappa >= 1.0 - eps:
        return ZeroCache(), kappa
    # Cost-aware tie-break: an RBF query is far cheaper than the affine kNN
    # scan — prefer it unless the affine cache is substantially more accurate.
    kd = {k: v for v, k in cands}
    if "rbf" in kd and kd["rbf"] < 1.0 - eps and kd["rbf"] <= kd.get("affine", np.inf) * 1.3:
        return ScalarRBFCache(Z, F, R_vals), kd["rbf"]
    if kind == "rbf":
        return ScalarRBFCache(Z, F, R_vals), kappa
    return ChartCache(Z, F), kappa


def build_caches(mixture, anchors, mode, resp_min=0.02, logq_floor_gap=12.0):
    K = mixture.K
    if mode == "zero" or not anchors:
        return [ZeroCache() for _ in range(K)], [1.0] * K
    anchors = [(x, u, g) for x, u, g in anchors
               if np.all(np.isfinite(x)) and np.isfinite(u)
               and np.all(np.isfinite(g)) and np.linalg.norm(g) < 1e12]
    if not anchors:
        return [ZeroCache() for _ in range(K)], [1.0] * K
    lqs = np.array([mixture.logq(x) for x, _, _ in anchors])
    floor = np.max(lqs) - max(logq_floor_gap, 3.0 * np.std(lqs))
    per_Z = [[] for _ in range(K)]
    per_F = [[] for _ in range(K)]
    per_R = [[] for _ in range(K)]
    for (x, u, g), lq in zip(anchors, lqs):
        if lq < floor:
            continue
        gR = g + mixture.grad_logq(x)
        if not np.all(np.isfinite(gR)):
            continue
        _, resp = mixture.logq_and_resp(x)
        for k in range(K):
            if resp[k] >= resp_min:
                per_Z[k].append(mixture.to_chart(k, x))
                per_F[k].append(mixture.comps[k].Jt_grad(x, gR))
                per_R[k].append(u + lq)
    caches, kappas = [], []
    for k in range(K):
        if len(per_Z[k]) >= 4:
            if mode == "affine":
                caches.append(ChartCache(per_Z[k], per_F[k]) if len(per_Z[k]) >= 8 else ZeroCache())
                kappas.append(float("nan"))
            else:
                c, kap = select_cache(per_Z[k], per_F[k], per_R[k])
                caches.append(c)
                kappas.append(kap)
        else:
            caches.append(ZeroCache())
            kappas.append(1.0)
    return caches, kappas


# ---------------------------------------------------------------------------
# Residual-guided atlas birth
# ---------------------------------------------------------------------------
def probe_frontier(target, mixture, sampler, rng, n=192, stretch=(1.5, 3.5)):
    out = []
    axes = [c.principal_axis() for c in mixture.comps]
    ws = np.exp(mixture.log_ws)
    xs = []
    for i in range(n):
        k = rng.choice(mixture.K, p=ws)
        c = rng.uniform(*stretch)
        if i % 2 == 0 or axes[k] is None:
            x = mixture.from_chart(k, c * rng.standard_normal(mixture.d))
        else:
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            x = (mixture.comps[k].mu + sgn * c * axes[k]
                 + mixture.from_chart(k, 0.3 * rng.standard_normal(mixture.d))
                 - mixture.comps[k].from_z(np.zeros(mixture.d)))
        if np.all(np.isfinite(x)):
            xs.append(x)
    us = target.U_batch(xs) if xs else []
    for x, u in zip(xs, us):
        if np.isfinite(u):
            out.append((x, u, mixture.logq(x), sampler._trust_at(x)))
    return out


def _refine_probes(target, mixture, probes, maha_min, n_refine=8, max_iter=4):
    cands = [(u + lq, x) for x, u, lq, t in probes if mixture.min_maha(x) >= maha_min]
    cands.sort(key=lambda t: t[0])
    out = []
    for _, x0 in cands[:n_refine]:
        res = minimize(lambda v: target.value_and_grad(v), x0, jac=True,
                       method="L-BFGS-B", options={"maxiter": max_iter})
        y = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(y)) or mixture.min_maha(y) < maha_min:
            continue
        u = target.U(y)
        if np.isfinite(u):
            out.append((y, u, mixture.logq(y), 0.0))
    return out


def birth_charts(target, mixture, frontier_chain, frontier_probe=(), max_new=4,
                 maha_min=1.5, score_min=1.0, refined=None):
    if refined is None:
        refined = _refine_probes(target, mixture, list(frontier_probe), maha_min)
    frontier = list(frontier_chain) + list(frontier_probe) + refined
    if not frontier:
        return []
    a = np.array([-(u + lq) for _, u, lq, _ in frontier])
    base_pool = frontier_chain if frontier_chain else frontier
    med = np.median([-(u + lq) for _, u, lq, _ in base_pool])
    scored = sorted(
        ((max(0.0, ai - med) * (1.0 - t), x) for (x, _, _, t), ai in zip(frontier, a)),
        key=lambda p: -p[0],
    )
    new = []
    for score, x in scored:
        if score < score_min or len(new) >= max_new:
            break
        if mixture.min_maha(x) < maha_min:
            continue
        if any(c.maha(x) < maha_min for c in new):
            continue
        cov = 1.44 * _regularized_cov(target.hessian(x))  # geometry-uncertainty widening
        try:
            new.append(GaussComp(x.copy(), cov))
        except np.linalg.LinAlgError:
            continue
    return new




class TriComp:
    """Generic monotone triangular chart (chart class 7):

        x_i = mu_i + exp(gam_i * z_{drv_i}) * sig_i * SAS(z_i; eps_i, delta_i),

    SAS(z) = sinh((asinh z + eps)/delta) (Jones-Pewsey sinh-arcsinh: eps =
    skew, delta = tail weight; eps=0, delta=1 = identity). Drivers reference
    LATENT coordinates computed earlier in `order`, so the inverse is exact
    forward substitution. Subsumes skewness, exponential/heavy tails, and
    per-coordinate (multi-driver) scale funnels; fitted by closed-form
    quantile matching with shrinkage toward identity."""

    kind = "tri"

    def __init__(self, order, mu, sig, eps, delta, drv, gam):
        self.order = np.asarray(order, dtype=int)
        self.mu = np.asarray(mu, dtype=float)
        self.sig = np.asarray(sig, dtype=float)
        self.eps = np.asarray(eps, dtype=float)
        self.delta = np.asarray(delta, dtype=float)
        self.drv = np.asarray(drv, dtype=int)   # -1 = none
        self.gam = np.asarray(gam, dtype=float)
        self.d = len(self.mu)

    @staticmethod
    def _sas(z, e, dl):
        return np.sinh((np.arcsinh(z) + e) / dl)

    @staticmethod
    def _sas_inv(y, e, dl):
        return np.sinh(dl * np.arcsinh(y) - e)

    @staticmethod
    def _sas_dz(z, e, dl):
        return np.cosh((np.arcsinh(z) + e) / dl) / (dl * np.sqrt(1 + z * z))

    # Driver exponents are clamped to +-30: from_z/to_z/logpdf all use the SAME
    # clamped scale, so the map stays an exact bijection with an exact Jacobian
    # (the clamp just freezes the scale beyond |gam*z_drv| = 30, where a
    # spurious fit would otherwise overflow to inf/0 and poison the mixture).
    def _sc_expo(self, i, z):
        return float(np.clip(self.gam[i] * z[self.drv[i]], -30.0, 30.0)) if self.drv[i] >= 0 else 0.0

    def from_z(self, z):
        x = np.empty(self.d)
        for i in self.order:
            sc = np.exp(self._sc_expo(i, z))
            x[i] = self.mu[i] + sc * self.sig[i] * self._sas(z[i], self.eps[i], self.delta[i])
        return x

    def to_z(self, x, rng=None):
        z = np.empty(self.d)
        for i in self.order:
            sc = np.exp(self._sc_expo(i, z))
            z[i] = self._sas_inv((x[i] - self.mu[i]) / (sc * self.sig[i]),
                                 self.eps[i], self.delta[i])
        return z

    def logpdf(self, x):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            z = self.to_z(x)
            ld = 0.0
            for i in range(self.d):
                ld += self._sc_expo(i, z) + np.log(self.sig[i]) \
                    + np.log(self._sas_dz(z[i], self.eps[i], self.delta[i]))
            v = float(-0.5 * self.d * LOG2PI - 0.5 * z @ z - ld)
        # a point this chart cannot evaluate has zero reference density here;
        # NaN (inf-inf in the far tail) must never leak into mixture weights
        return v if not np.isnan(v) else -np.inf

    def grad_logpdf(self, x, h=1e-5):
        g = np.empty(self.d)
        for i in range(self.d):
            e = np.zeros(self.d); e[i] = h
            g[i] = (self.logpdf(x + e) - self.logpdf(x - e)) / (2 * h)
        return g

    def Jt_grad(self, x, g, h=1e-5):
        z = self.to_z(x)
        out = np.empty(self.d)
        for k in range(self.d):
            e = np.zeros(self.d); e[k] = h
            out[k] = g @ (self.from_z(z + e) - self.from_z(z - e)) / (2 * h)
        return out

    def maha(self, x):
        return float(np.linalg.norm(self.to_z(x)))

    def sample(self, rng):
        return self.from_z(rng.standard_normal(self.d))

    def principal_axis(self):
        return None


def _wq(v, w, qs):
    i = np.argsort(v); cw = np.cumsum(w[i]); cw = cw / cw[-1]
    return np.interp(np.asarray(qs) / 100.0, cw, v[i])


def fit_tri_comp(states, r2_min=0.12, eff_min=0.06, shrink_n=200.0, weights=None):
    """Fit TriComp from warm-up states: per-coordinate driver scan (log-square
    regression, per-child best driver), then closed-form quantile matching of
    (eps, delta) on the scale-adjusted residual, shrunk toward identity.
    Returns None when everything is near-Gaussian."""
    X = np.asarray(states, dtype=float)
    n, d = X.shape
    if n < 80:
        return None
    W = (np.ones(n) / n) if weights is None else np.asarray(weights) / np.sum(weights)
    if weights is not None:
        # degenerate IS weights (a handful of pool points carry all the mass)
        # make the weighted quantiles garbage; unweighted is the safer fit
        wess = 1.0 / max(float(W @ W), 1e-300)
        if not np.isfinite(wess) or wess < 25.0:
            W = np.ones(n) / n
    mu = np.array([_wq(X[:, j], W, [50])[0] for j in range(d)])
    sig = np.maximum(1.4826 * np.array([_wq(np.abs(X[:, j] - mu[j]), W, [50])[0] for j in range(d)]), 1e-9)
    U = (X - mu) / sig
    ell = np.log(U**2 + 1e-300)
    ellc = ell - ell.mean(axis=0)
    drv = np.full(d, -1, dtype=int)
    gam = np.zeros(d)
    for j in range(d):
        best = (0.0, -1, 0.0)
        for v in range(d):
            if v == j: continue
            uv = U[:, v] - U[:, v].mean()
            vv = float(uv @ uv)
            if vv < 1e-12: continue
            b = float(uv @ ellc[:, j] / vv)
            r2 = (b * b * vv) / max(float(ellc[:, j] @ ellc[:, j]), 1e-300)
            if r2 > best[0]:
                best = (r2, v, b)
        if best[0] >= r2_min and 0.3 <= abs(best[2]) <= 4.0:
            drv[j], gam[j] = best[1], best[2] / 2.0
    # break cycles: a coordinate that serves as a driver keeps no driver itself
    is_driver = np.zeros(d, bool)
    for j in range(d):
        if drv[j] >= 0: is_driver[drv[j]] = True
    for j in range(d):
        if is_driver[j]: drv[j] = -1
    order = list(np.where(is_driver)[0]) + [j for j in range(d) if not is_driver[j]]
    # SAS fit per coordinate: outer coarse-grid over (eps, delta), inner
    # closed-form OLS of empirical quantiles on SAS(normal quantiles).
    lam = n / (n + shrink_n)
    eps = np.zeros(d); delta = np.ones(d)
    mu_f = mu.copy(); sig_f = sig.copy()
    zq = np.array([-1.6449, -0.9674, -0.4307, 0.0, 0.4307, 0.9674, 1.6449])
    pq = [5.0, 16.66, 33.33, 50.0, 66.66, 83.33, 95.0]
    e_grid = np.linspace(-1.2, 1.2, 13)
    d_grid = np.linspace(0.4, 2.6, 12)
    for j in range(d):
        u = U[:, j].copy()
        if drv[j] >= 0:
            u = u * np.exp(-gam[j] * U[:, drv[j]])
        xq = _wq(u, W, pq)
        best = (np.inf, 0.0, 1.0, 0.0, 1.0)
        for e in e_grid:
            for dl in d_grid:
                sv = np.sinh((np.arcsinh(zq) + e) / dl)
                A = np.column_stack([np.ones(7), sv])
                coef, res, *_ = np.linalg.lstsq(A, xq, rcond=None)
                r = float(np.sum((A @ coef - xq) ** 2))
                if r < best[0] and coef[1] > 1e-6:
                    best = (r, e, dl, coef[0], coef[1])
        _, e, dl, m_, s_ = best
        # snap to identity unless clearly better than the identity fit
        sv0 = zq
        A0 = np.column_stack([np.ones(7), sv0])
        c0, *_ = np.linalg.lstsq(A0, xq, rcond=None)
        r0 = float(np.sum((A0 @ c0 - xq) ** 2))
        if best[0] > 0.75 * r0:
            e, dl, m_, s_ = 0.0, 1.0, float(c0[0]), float(c0[1])
        eps[j] = lam * e
        delta[j] = 1.0 + lam * (dl - 1.0)
        if drv[j] < 0:
            mu_f[j] = mu[j] + sig[j] * m_
            sig_f[j] = sig[j] * s_
        else:
            sig_f[j] = sig[j] * s_
    # driver enters as LATENT z; rescale gam from U-units to z-units
    for j in range(d):
        if drv[j] >= 0:
            v = drv[j]
            gam[j] = gam[j] * max(sig_f[v] / sig[v], 1e-3)
    effect = max(np.max(np.abs(eps)), np.max(np.abs(delta - 1.0)))
    if effect < eff_min and not np.any(drv >= 0):
        return None
    return TriComp(order, mu_f, sig_f, eps, delta, drv, gam)



# ---------------------------------------------------------------------------
# Pareto-CMA atlas construction (PC-RTA): CMA emitters in chart-whitened
# coordinates + NSGA-II-style environmental selection of candidate charts.
# Warm-up only — production exactness untouched.
# ---------------------------------------------------------------------------
class CMAEmitter:
    """Standard (mu/mu_w, lambda) CMA-ES with rank-1 + rank-mu updates and CSA,
    run in the parent chart's whitened z-coordinates (L-BFGS/transport geometry
    removes the bulk conditioning; CMA learns the local correction + drift)."""

    def __init__(self, parent, d, lam=8, sigma0=0.7):
        self.parent = parent
        self.d, self.lam = d, lam
        self.m = np.zeros(d)
        self.sigma = sigma0
        self.C = np.eye(d)
        self.pc = np.zeros(d)
        self.ps = np.zeros(d)
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mu = mu
        self.mueff = 1.0 / float(np.sum(self.w**2))
        self.cc = (4 + self.mueff / d) / (d + 4 + 2 * self.mueff / d)
        self.c1 = 2.0 / ((d + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((d + 2) ** 2 + self.mueff))
        self.cs = (self.mueff + 2) / (d + self.mueff + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (d + 1)) - 1) + self.cs
        self.chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))
        self.g = 0
        self._decompose()

    def _decompose(self):
        self.C = 0.5 * (self.C + self.C.T)
        lam_, B = np.linalg.eigh(self.C)
        self.B = B
        self.Dg = np.sqrt(np.maximum(lam_, 1e-12))

    def ask(self, rng):
        self.zs = [rng.standard_normal(self.d) for _ in range(self.lam)]
        self.ys = [self.B @ (self.Dg * z) for z in self.zs]
        return [self.parent.from_z(self.m + self.sigma * y) for y in self.ys]

    def tell(self, order):
        sel = list(order)[: self.mu]
        yw = np.sum([wi * self.ys[i] for wi, i in zip(self.w, sel)], axis=0)
        zw = np.sum([wi * self.zs[i] for wi, i in zip(self.w, sel)], axis=0)
        self.m = self.m + self.sigma * yw
        self.ps = ((1 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.B @ zw))
        denom = np.sqrt(1 - (1 - self.cs) ** (2 * (self.g + 1)))
        hsig = 1.0 if (np.linalg.norm(self.ps) / max(denom, 1e-12)
                       < (1.4 + 2 / (self.d + 1)) * self.chiN) else 0.0
        self.pc = ((1 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw)
        rank_mu = np.sum([wi * np.outer(self.ys[i], self.ys[i])
                          for wi, i in zip(self.w, sel)], axis=0)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc)
                               + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= float(np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.chiN - 1)))
        self.sigma = float(np.clip(self.sigma, 1e-3, 5.0))
        self.g += 1
        self._decompose()


def _weighted_quantile(vals, ws, q):
    idx = np.argsort(vals)
    cw = np.cumsum(ws[idx])
    cw = cw / cw[-1]
    return float(vals[idx][np.searchsorted(cw, q, side="left").clip(0, len(idx) - 1)])


def _inflated_logpdf(comp, x, c=3.0):
    """Density of the chart with its latent z scaled by c (generic over chart
    types): logpdf + |z|^2/2 (1 - 1/c^2) - d log c."""
    z2 = float(np.sum(comp.to_z(x) ** 2))
    return comp.logpdf(x) + 0.5 * z2 * (1.0 - 1.0 / c**2) - comp.d * np.log(c)


def build_eval_pool(rng, pool_xs, pool_us, gen_comps, visited, mixture, cap=500,
                    frontier_pts=(), frontier_share=0.15):
    """Common weighted validation pool: generating-mixture IS samples (truncated
    self-normalized weights) blended with corrected chain states (~pi, uniform)
    AND this round's frontier candidates (probes/CMA offspring, IS-weighted
    against an inflated-atlas proposal) — without the frontier group, a
    candidate chart in an uncovered region has no pool mass near it and every
    coverage objective degenerates to the -log(1-eps) dilution penalty.
    Warm-up-only machinery — weights affect selection quality, never exactness."""
    nA = min(len(pool_xs), cap // 2 if visited else cap)
    idxA = rng.choice(len(pool_xs), size=nA, replace=False) if len(pool_xs) > nA else np.arange(len(pool_xs))
    XA = [np.asarray(pool_xs[i]) for i in idxA]
    UA = np.array([pool_us[i] for i in idxA])
    lpdf_gen = _comp_logpdfs(XA, gen_comps)
    log_m = logsumexp(lpdf_gen, axis=0) - np.log(len(gen_comps))
    lwA = -UA - log_m
    wA = np.exp(lwA - logsumexp(lwA))
    wA = np.minimum(wA, np.sqrt(len(wA)) * np.mean(wA))
    wA = wA / wA.sum()
    X, U_all, w_all = XA, list(UA), list(wA)
    if visited:
        nB = min(len(visited), cap - nA)
        idxB = rng.choice(len(visited), size=nB, replace=False) if len(visited) > nB else np.arange(len(visited))
        zeta = min(0.5, nB / (nA + nB))
        w_all = [wi * (1 - zeta) for wi in w_all]
        for i in idxB:
            X.append(np.asarray(visited[i][0]))
            U_all.append(visited[i][1])
            w_all.append(zeta / nB)
    if frontier_pts:
        lws = []
        pts = []
        for x, u in frontier_pts:
            x = np.asarray(x, dtype=float)
            if not (np.all(np.isfinite(x)) and np.isfinite(u)):
                continue
            lm = logsumexp([lw + _inflated_logpdf(c, x)
                            for lw, c in zip(mixture.log_ws, mixture.comps)])
            lws.append(-u - lm)
            pts.append((x, u))
        if pts:
            wf = np.exp(np.asarray(lws) - logsumexp(lws))
            wf = np.minimum(wf, np.sqrt(len(wf)) * np.mean(wf))
            wf = wf / wf.sum() * frontier_share
            w_all = [wi * (1 - frontier_share) for wi in w_all]
            for (x, u), wi in zip(pts, wf):
                X.append(x)
                U_all.append(u)
                w_all.append(float(wi))
    w_arr = np.asarray(w_all)
    return dict(
        X=X,
        logpi=-np.asarray(U_all),
        w=w_arr / w_arr.sum(),
        lq=np.array([mixture.logq(x) for x in X]),
    )


def _eval_objectives(comp, pool, mixture, target, rng, eps_grid=(0.05, 0.15, 0.3), S=8):
    """(J_bulk, J_front, J_stable) + standard errors for the two coverage gains."""
    lphi = np.array([comp.logpdf(x) for x in pool["X"]])
    lq, w = pool["lq"], pool["w"]
    d_n = pool["logpi"] - lq
    tau = _weighted_quantile(d_n, w, 0.8)
    sd = max(_weighted_quantile(d_n, w, 0.95) - tau, 1.0)
    v = w * np.logaddexp(0.0, (d_n - tau) / sd)  # softplus deficit emphasis
    v = v / max(v.sum(), 1e-300)
    Jb = Jf = -np.inf
    se_b = se_f = 0.0
    for eps in eps_grid:
        delta = np.logaddexp(np.log1p(-eps) + lq, np.log(eps) + lphi) - lq
        jb = float(w @ delta)
        jf = float(v @ delta)
        if jb > Jb:
            Jb, se_b = jb, float(np.sqrt(np.sum(w**2 * (delta - jb) ** 2)))
        if jf > Jf:
            Jf, se_f = jf, float(np.sqrt(np.sum(v**2 * (delta - jf) ** 2)))
    Rs = []
    for _ in range(S):
        y = comp.sample(rng)
        u = target.U(y)
        if np.isfinite(u):
            Rs.append(u + np.logaddexp(np.log(0.85) + mixture.logq(y),
                                       np.log(0.15) + comp.logpdf(y)))
    if len(Rs) >= 4:
        Rs = np.asarray(Rs)
        eta = float(np.quantile(np.abs(Rs - np.median(Rs)), 0.9))
    else:
        eta = np.inf
    return Jb, Jf, -eta, se_b, se_f


def _sym_kl(a, b):
    tr = np.trace(a.prec @ b.cov + b.prec @ a.cov) - 2 * a.d
    dm = a.mu - b.mu
    return 0.25 * (tr + dm @ (a.prec + b.prec) @ dm)


def _nondominated_fronts(J):
    """Peel Pareto fronts (maximize all columns). Returns front index per row."""
    n = len(J)
    front = np.full(n, -1)
    remaining = set(range(n))
    f = 0
    while remaining:
        cur = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if j != i and np.all(J[j] >= J[i]) and np.any(J[j] > J[i]):
                    dominated = True
                    break
            if not dominated:
                cur.append(i)
        for i in cur:
            front[i] = f
            remaining.discard(i)
        f += 1
    return front


def pareto_birth(target, mixture, cands, pool, rng, max_new=4, maha_min=1.5,
                 pre_gate=0.5, hess_cap=8, beta_lcb=1.0):
    """NSGA-II-style environmental selection of candidate charts: feasibility
    (LCB coverage gain > 0) -> nondominated sort on (J_bulk, J_front, J_stable)
    -> dual objective/geometric (symmetric-KL) crowding -> capacity."""
    d_n = pool["logpi"] - pool["lq"]
    med = _weighted_quantile(d_n, pool["w"], 0.5)
    seen = []
    scored = []
    for x, u in cands:
        x = np.asarray(x, dtype=float)
        if not (np.all(np.isfinite(x)) and np.isfinite(u)):
            continue
        if mixture.min_maha(x) < maha_min:
            continue
        a = -(u + mixture.logq(x))
        if a - med < pre_gate:
            continue
        if any(np.linalg.norm(x - s) < 1e-8 for s in seen):
            continue
        seen.append(x)
        scored.append((a, x))
    if not scored:
        return []
    scored.sort(key=lambda t: -t[0])
    comps = []
    for _, x in scored[:hess_cap]:
        cov = 1.44 * _regularized_cov(target.hessian(x))  # geometry-uncertainty widening
        try:
            comps.append(GaussComp(x, cov))
        except np.linalg.LinAlgError:
            continue
    if not comps:
        return []
    rows = [_eval_objectives(c, pool, mixture, target, rng) for c in comps]
    feas = [i for i, (jb, jf, js, sb, sf) in enumerate(rows)
            if np.isfinite(js) and (jb - beta_lcb * sb > 0 or jf - beta_lcb * sf > 0)]
    if not feas:
        return []
    J = np.array([[rows[i][0], rows[i][1], rows[i][2]] for i in feas])
    fronts = _nondominated_fronts(J)
    # dual crowding: normalized objective spread + geometric spread (sym-KL)
    rng_obj = J.max(axis=0) - J.min(axis=0) + 1e-12
    cd_obj = np.zeros(len(feas))
    for m in range(3):
        order = np.argsort(J[:, m])
        for r, i in enumerate(order):
            lo = J[order[max(0, r - 1)], m]
            hi = J[order[min(len(order) - 1, r + 1)], m]
            cd_obj[i] += (hi - lo) / rng_obj[m]
    cd_geo = np.zeros(len(feas))
    if len(feas) > 1:
        for ii, i in enumerate(feas):
            ds = [np.log1p(_sym_kl(comps[i], comps[j])) for j in feas if j != i]
            cd_geo[ii] = float(np.mean(ds))
    cd = 0.5 * cd_obj / (cd_obj.max() + 1e-12) + 0.5 * cd_geo / (cd_geo.max() + 1e-12)
    order = sorted(range(len(feas)), key=lambda ii: (fronts[ii], -cd[ii]))
    picked = []
    for ii in order:
        c = comps[feas[ii]]
        if any(p.maha(c.mu) < maha_min or c.maha(p.mu) < maha_min for p in picked):
            continue
        picked.append(c)
        if len(picked) >= max_new:
            break
    return picked


def prune_components_loo(comps, log_ws, lpdf, wbar, tau=0.02, w_floor=1e-3):
    """Leave-one-out pruning with unique-coverage protection: a low-weight chart
    survives if removing it costs > tau nats of pool log-coverage OR it is the
    dominant explainer of a meaningful pool point (unique tail/mode coverage)."""
    K, N = lpdf.shape
    lw = np.asarray(log_ws)
    lqa = logsumexp(lw[:, None] + lpdf, axis=0)
    L_full = float(wbar @ lqa)
    resp = np.exp(lw[:, None] + lpdf - lqa[None, :])
    wmed = np.median(wbar)
    keep = []
    for j in range(K):
        wj = float(np.exp(lw[j]))
        if wj < 1e-8:
            continue
        protected = bool(np.any((resp[j] > 0.9) & (wbar > wmed)))
        if wj >= w_floor or protected:
            keep.append(j)
            continue
        others = [k for k in range(K) if k != j]
        lq_minus = logsumexp(lw[others, None] - np.log1p(-wj) + lpdf[others], axis=0)
        if L_full - float(wbar @ lq_minus) >= tau:
            keep.append(j)
    if not keep:
        keep = [int(np.argmax(lw))]
    comps = [comps[k] for k in keep]
    lw = lw[keep]
    return comps, lw - logsumexp(lw)


def lshade_explore(target, mixture, rng, seeds, gens=10, np0=30, np_min=6, T0=8.0, H=6):
    """L-SHADE-style warm-up explorer (success-history DE, linear population
    reduction, tempered stochastic selection). Difference vectors between
    members on a curved valley point ALONG the valley — geometry acquisition
    without a covariance model. Warm-up only; returns all evaluated (x, U)."""
    d = mixture.d
    pop = [np.asarray(x, dtype=float) for x in seeds if np.all(np.isfinite(x))][:np0]
    while len(pop) < np0:
        pop.append(mixture.sample(rng) + rng.standard_normal(d))
    pop = np.array(pop)
    fit = target.U_batch(pop)
    ok = np.isfinite(fit)
    pop, fit = pop[ok], fit[ok]
    MF, MCR, k = np.full(H, 0.5), np.full(H, 0.7), 0
    out = [(pop[i].copy(), float(fit[i])) for i in range(len(pop))]
    for g in range(gens):
        NP = max(np_min, int(round(np0 - (np0 - np_min) * g / gens)))
        order = np.argsort(fit)
        pop, fit = pop[order[:NP]], fit[order[:NP]]
        if len(pop) < 4:
            break
        T = T0 * (1.0 - g / gens) + 1.0
        trials = []
        for i in range(len(pop)):
            r = rng.integers(0, H)
            F = float(np.clip(rng.standard_cauchy() * 0.1 + MF[r], 0.05, 1.0))
            CR = float(np.clip(rng.normal(MCR[r], 0.1), 0.0, 1.0))
            pb = pop[rng.integers(0, max(1, int(0.2 * len(pop))))]
            a, b = pop[rng.integers(0, len(pop))], pop[rng.integers(0, len(pop))]
            mut = pop[i] + F * (pb - pop[i]) + F * (a - b)
            mask = rng.random(d) < CR
            mask[rng.integers(0, d)] = True
            trials.append((i, np.where(mask, mut, pop[i]), F, CR))
        tU = target.U_batch([t[1] for t in trials])
        SF, SCR = [], []
        for (i, trial, F, CR), u in zip(trials, tU):
            if not np.isfinite(u):
                continue
            out.append((trial.copy(), float(u)))
            if u < fit[i] or rng.random() < np.exp(min(0.0, (fit[i] - u) / T)):
                pop[i], fit[i] = trial, u
                SF.append(F)
                SCR.append(CR)
        if SF:
            MF[k] = float(np.sum(np.square(SF)) / max(np.sum(SF), 1e-9))  # Lehmer
            MCR[k] = float(np.mean(SCR))
            k = (k + 1) % H
    return out


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------
class DualAveraging:
    def __init__(self, h0, target=0.7, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = np.log(h0)
        self.target, self.gamma, self.t0, self.kappa = target, gamma, t0, kappa
        self.log_h = np.log(h0)
        self.log_h_bar = np.log(h0)
        self.H_bar = 0.0
        self.t = 0

    def update(self, accept_prob):
        self.t += 1
        eta = 1.0 / (self.t + self.t0)
        self.H_bar = (1 - eta) * self.H_bar + eta * (self.target - accept_prob)
        self.log_h = self.mu - np.sqrt(self.t) / self.gamma * self.H_bar
        self.log_h = float(np.clip(self.log_h, np.log(0.05), np.log(2.0 * np.pi)))
        w = self.t ** (-self.kappa)
        self.log_h_bar = w * self.log_h + (1 - w) * self.log_h_bar

    @property
    def h(self):
        return float(np.exp(self.log_h))

    @property
    def h_final(self):
        return float(np.exp(self.log_h_bar))


class PMRHMC:
    def __init__(self, target, mixture, anchors, rng, surrogate="auto", p_global=0.2,
                 T_range=(0.35 * np.pi, 0.65 * np.pi), L_cap=48, t_defense=None, t_eps=0.1):
        self.target = target
        self.rng = rng
        self.surrogate = surrogate
        self.p_global = p_global
        self.T_range = T_range
        self.L_cap = L_cap
        self.anchors = list(anchors)
        self.h = 0.5
        self.t_def = t_defense  # TDefense or None
        self.t_eps = t_eps
        self.set_mixture(mixture)
        self.stats = dict(local=0, local_acc=0, global_=0, global_acc=0, mean_L=0.0)

    def set_mixture(self, mixture):
        self.mix = mixture
        self._rebuild_caches()

    def _rebuild_caches(self):
        self.caches, self.kappas = build_caches(self.mix, self.anchors, self.surrogate)

    def _log_g(self, x, lq):
        """Global proposal density: g = (1-eps) q + eps t (t never enters R)."""
        if self.t_def is None:
            return lq
        return float(np.logaddexp(np.log1p(-self.t_eps) + lq,
                                  np.log(self.t_eps) + self.t_def.logpdf(x)))

    def _local(self, x, U_x, lq_x, h, T_fixed=None, record_dH=None):
        mix, rng = self.mix, self.rng
        _, resp = mix.logq_and_resp(x)
        k = int(np.searchsorted(np.cumsum(resp), rng.random() * resp.sum()))
        k = min(k, mix.K - 1)
        z = mix.to_chart(k, x, rng)
        p = rng.standard_normal(mix.d)
        T = T_fixed if T_fixed is not None else rng.uniform(*self.T_range)
        L = int(max(1, min(self.L_cap, round(T / h))))
        eps = T / L
        cache = self.caches[k]
        H0 = 0.5 * (z @ z + p @ p) + U_x + lq_x
        c, s = np.cos(eps), np.sin(eps)
        for _ in range(L):
            p = p - 0.5 * eps * cache.query(z)
            z, p = c * z + s * p, -s * z + c * p
            p = p - 0.5 * eps * cache.query(z)
        y = mix.from_chart(k, z)
        if not np.all(np.isfinite(y)):
            self.stats["local"] += 1
            self.stats["mean_L"] += L
            if record_dH is not None:
                record_dH.append(np.inf)
            return x, U_x, lq_x, 0.0
        U_y = self.target.U(y)
        lq_y = mix.logq(y)
        H1 = 0.5 * (z @ z + p @ p) + U_y + lq_y
        dH = H1 - H0 if np.isfinite(H1) else np.inf
        if record_dH is not None:
            record_dH.append(dH)
        a = min(1.0, np.exp(min(0.0, -dH))) if np.isfinite(dH) else 0.0
        self.stats["local"] += 1
        self.stats["mean_L"] += L
        if rng.random() < a:
            self.stats["local_acc"] += 1
            return y, U_y, lq_y, a
        return x, U_x, lq_x, a

    def _global(self, x, U_x, lq_x):
        rng = self.rng
        if self.t_def is not None and rng.random() < self.t_eps:
            y = self.t_def.sample(rng)
        else:
            y = self.mix.sample(rng)
        U_y = self.target.U(y)
        if not np.isfinite(U_y):
            self.stats["global_"] += 1
            return x, U_x, lq_x
        lq_y = self.mix.logq(y)
        log_a = (U_x + self._log_g(x, lq_x)) - (U_y + self._log_g(y, lq_y))
        self.stats["global_"] += 1
        if np.log(rng.random()) < min(0.0, log_a):
            self.stats["global_acc"] += 1
            return y, U_y, lq_y
        return x, U_x, lq_x

    def _trust_at(self, x):
        _, resp = self.mix.logq_and_resp(x)
        k = int(np.argmax(resp))
        return self.caches[k].trust(self.mix.to_chart(k, x))

    def tune_round(self, n_iter, da, collect_grads=False, collect_cap=300, frontier=None,
                   visited=None):
        rng = self.rng
        x, U_x, lq_x = self._state
        collected = 0
        for _ in range(n_iter):
            x_prev = x
            x, U_x, lq_x, a = self._local(x, U_x, lq_x, da.h)
            da.update(a)
            moved = x is not x_prev
            if moved and visited is not None:
                visited.append((x.copy(), U_x))
            if moved and frontier is not None:
                frontier.append((x.copy(), U_x, lq_x, self._trust_at(x)))
            if (collect_grads and moved and collected < collect_cap and rng.random() < 0.5):
                u2, g = self.target.value_and_grad(x)
                self.anchors.append((x.copy(), u2, g))
                collected += 1
        self._state = (x, U_x, lq_x)
        return collected

    def knee_refine_h(self, h_da, n_traj=30, floor=0.05):
        if h_da > 2.0 * floor:
            return h_da
        T_mid = 0.5 * (self.T_range[0] + self.T_range[1])
        Vs = {}
        x, U_x, lq_x = self._state
        for h in (h_da, 0.5 * h_da):
            rec = []
            for _ in range(n_traj):
                x, U_x, lq_x, _ = self._local(x, U_x, lq_x, h, T_fixed=T_mid, record_dH=rec)
            dh = np.clip(np.array([v for v in rec if np.isfinite(v)]), -25, 25)
            Vs[h] = float(np.mean(dh**2)) if len(dh) else 0.0
        self._state = (x, U_x, lq_x)
        V1, V2 = Vs[h_da], Vs[0.5 * h_da]
        if V1 < 1e-4 or V1 <= V2:
            return h_da
        V_sur = max((16.0 * V2 - V1) / 15.0, 0.0)
        ch4 = max(16.0 / 15.0 * (V1 - V2), 1e-12)
        if V_sur <= 0.0:
            return h_da
        h_star = h_da * (V_sur / ch4) ** 0.25
        return float(np.clip(h_star, 0.5 * h_da, min(4.0 * h_da, 2.0 * np.pi)))

    def init_state(self):
        x = self.mix.sample(self.rng)
        U_x = self.target.U(x)
        lq_x = self.mix.logq(x)
        self._state = (x, U_x, lq_x)

    def run(self, n_samples):
        rng = self.rng
        x, U_x, lq_x = self._state
        chain = np.empty((n_samples, self.mix.d))
        t0 = time.perf_counter()
        for i in range(n_samples):
            if rng.random() < self.p_global:
                x, U_x, lq_x = self._global(x, U_x, lq_x)
            else:
                x, U_x, lq_x, _ = self._local(x, U_x, lq_x, self.h)
            chain[i] = x
        self.stats["prod_time"] = time.perf_counter() - t0
        if self.stats["local"]:
            self.stats["mean_L"] /= self.stats["local"]
        return chain


def enrich_anchors(target, mixture, anchors, rng, n=512):
    xs = mixture.sample(rng, n)
    for x in xs:
        u, g = target.value_and_grad(x)
        anchors.append((x, u, g))
    return anchors


def run_pmr(target, seed=42, surrogate="auto", p_global=0.2, n_samples=12000,
            n_starts=10, n_enrich=512, n_tune=400, n_tune2=300, pool_per_comp=32,
            birth_rounds=5, max_K_total=28, transport=False, t_defense=False,
            pareto_cma=False, n_emitters=4, cma_lam=8, lbfgs_max_iter=1000,
            lshade=False, return_sampler=False):
    """Full pipeline (Milestone A + optional B transport charts + C t-defense)."""
    rng = np.random.default_rng(seed)
    target.set_phase("warmup")
    t0 = time.perf_counter()

    comps, anchors = build_components(target, rng, n_starts=n_starts, max_iter=lbfgs_max_iter)
    gen_comps = list(comps)
    pool_xs, pool_us = sample_pool(target, comps, rng, n_per=pool_per_comp)

    last_wbar = [None]

    def _weight_and_build(protect=()):
        nonlocal comps, log_ws, mixture
        lw, lp, wb = fit_weights_forward_kl(pool_xs, pool_us, comps, gen_comps)
        last_wbar[0] = wb
        if pareto_cma:
            comps, log_ws = prune_components_loo(comps, lw, lp, wb)
        else:
            comps, log_ws = prune_components(comps, lw)
        # protection floor: a just-fitted transport gets one round at weight
        # 0.02 even if forward-KL zeroes it (path/pool mismatch on its first
        # fit); if it then earns no weight it is pruned next round.
        for c in protect:
            if c not in comps:
                comps.append(c)
                log_ws = np.concatenate([log_ws + np.log(0.98), [np.log(0.02)]])
        mixture = MixtureRef(comps, log_ws)

    def _admit(new_comps):
        for c in new_comps:
            comps.append(c)
            gen_comps.append(c)
            xs2, us2 = sample_pool(target, [c], rng, n_per=pool_per_comp)
            pool_xs.extend(xs2)
            pool_us.extend(us2)
        _weight_and_build(protect=[c for c in new_comps if c.kind != "gauss"])

    log_ws = None
    mixture = None
    _weight_and_build()
    n_transport = 0

    # Transport bootstrap from L-BFGS PATH anchors: the optimizer traces curved
    # valleys (all five Rosenbrock pairs, banana arms) before any chain exists —
    # chain-data-only fitting starves exactly on the targets that need charts.
    if transport and len(anchors) >= 60:
        pathA = np.asarray([a[0] for a in anchors])
        pathU = np.asarray([a[1] for a in anchors])
        good = pathA[pathU <= np.median(pathU) + 25.0]
        if len(good) >= 60:
            tc = fit_transport_comps(good)
            if tc:
                n_transport += len(tc)
                _admit(tc)
    if transport and len(pool_xs) >= 80 and last_wbar[0] is not None:
        # pi-weighted tri fit from the pool (path anchors are ascent-biased)
        tr = fit_tri_comp(np.asarray(pool_xs), weights=last_wbar[0])
        if tr is not None and not any(c.kind == "tri" for c in comps):
            n_transport += 1
            _admit([tr])

    # Heavy-tail detection by direct U-ray probes (chain-independent — a chain
    # confined to a Gaussian atlas never reaches the tails it would need to
    # measure): along chart rays, Gaussian growth gives (U8-U4)/(U4-U2) ~ 4,
    # Student-t ~ 1.2; the ratio inverts to nu. Convert Gaussian charts to
    # exact scale-mixture t-charts when tails are heavy.
    ratios = []
    c0 = mixture.comps[int(np.argmax(mixture.log_ws))]
    for _ in range(4):
        u = rng.standard_normal(mixture.d)
        u /= np.linalg.norm(u)
        us_ray = []
        for r_ in (3.0, 9.0, 27.0):
            ur = target.U(c0.from_z(r_ * u))
            if not np.isfinite(ur):
                break
            us_ray.append(ur)
        if len(us_ray) == 3 and us_ray[1] - us_ray[0] > 1e-3:
            ratios.append((us_ray[2] - us_ray[1]) / (us_ray[1] - us_ray[0]))
    # Gaussian ray ratio at radii (3,9,27) is 9; heavy tails give ~1-3.
    if ratios and np.median(ratios) < 4.0:
        tr = float(np.median(ratios))
        nu = 3.0 if tr < 1.5 else (5.0 if tr < 2.5 else 8.0)
        newts = [TComp(c.mu, c.cov, nu) for c in comps if type(c) is GaussComp]
        if newts:
            comps = [c for c in comps if type(c) is not GaussComp]
            _admit(newts)

    if surrogate != "zero" and n_enrich:
        enrich_anchors(target, mixture, anchors, rng, n=n_enrich)

    t_def = None
    sampler = PMRHMC(target, mixture, anchors, rng, surrogate=surrogate,
                     p_global=p_global)
    n_born = 0
    if p_global < 1.0:
        sampler.init_state()
        da = DualAveraging(sampler.h)
        frontier = []
        visited = []
        _expl_pts = []
        if lshade:
            seeds = ([c.mu for c in comps if hasattr(c, "mu")][:10]
                     + [mixture.sample(rng) for _ in range(10)])
            expl = lshade_explore(target, mixture, rng, seeds)
            for x, u in expl:
                frontier.append((x, u, mixture.logq(x), sampler._trust_at(x)))
                _expl_pts.append(x)
        sampler.tune_round(n_tune, da, collect_grads=(surrogate != "zero"),
                           frontier=frontier, visited=visited)

        def refit(new_comps):
            nonlocal comps, log_ws, mixture
            for c in new_comps:
                comps.append(c)
                gen_comps.append(c)
                xs2, us2 = sample_pool(target, [c], rng, n_per=pool_per_comp)
                pool_xs.extend(xs2)
                pool_us.extend(us2)
            lw, lpdf2, wbar2 = fit_weights_forward_kl(pool_xs, pool_us, comps, gen_comps)
            if pareto_cma:
                comps, log_ws = prune_components_loo(comps, lw, lpdf2, wbar2)
            else:
                comps, log_ws = prune_components(comps, lw)
            mixture = MixtureRef(comps, log_ws)
            if surrogate != "zero":
                enrich_anchors(target, mixture, sampler.anchors, rng, n=max(128, n_enrich // 4))
            sampler.set_mixture(mixture)
            sampler.init_state()

        geo_pts = list(_expl_pts)  # refined probes + L-SHADE explorer states —
        # the transport fits need their spread (visited-only data never wraps a
        # ring or reaches deep arms until the chart already exists)

        def visited_arr():
            return np.asarray([v[0] for v in visited] + geo_pts)

        def try_transport(min_states=40):
            """Fit transport charts from ALL visited states so far; add kinds not
            yet present (early-round chains are too narrow — e.g. the round-1
            banana chain spans |x0|<~4 and the shear R^2 is data-starved, so the
            fit must be retried as birth widens coverage)."""
            if not transport or len(visited) < min_states:
                return []
            # A transport INSTANCE blocks refits only while it carries real
            # weight — EXCEPT shears: excluding covered pairs is what produced
            # disjoint partial compositions (Rosenbrock). Shears always refit
            # the FULL composition from current data; the supersede rule then
            # replaces subsumed partials, and a new fit that is itself a subset
            # of an existing composition is dropped here.
            have = {transport_sig(c) for c, lw in zip(mixture.comps, mixture.log_ws)
                    if c.kind != "gauss" and c.kind != "shear" and np.exp(lw) >= 0.05}
            fits = fit_transport_comps(visited_arr(), exclude_sigs=have)
            def _sp(c):
                prs = getattr(c, "pairs", None)
                return ({(p[0], p[1]) for p in prs} if prs is not None
                        else {(c.dr, c.t)})
            existing = [_sp(c) for c in mixture.comps if c.kind == "shear"]
            return [c for c in fits
                    if not (c.kind == "shear" and any(_sp(c) <= ep for ep in existing))]

        emitters = []

        def cma_round_candidates():
            """One CMA generation per emitter: offspring ranked by trust-gated
            frontier deficit (cheap, 1 U eval each — already paid); survivors
            join the birth-candidate pool."""
            while len(emitters) < min(2, len(comps)):
                k = int(np.argsort(-mixture.log_ws)[len(emitters)])
                emitters.append(CMAEmitter(comps[k], mixture.d, lam=cma_lam))
            cands = []
            for e in emitters:
                xs_off = e.ask(rng)
                a_s, info = [], []
                for x in xs_off:
                    if np.all(np.isfinite(x)):
                        u = target.U(x)
                    else:
                        u = np.inf
                    if np.isfinite(u):
                        a = -(u + mixture.logq(x))
                        tr = sampler._trust_at(x)
                    else:
                        a, tr = -np.inf, 1.0
                    a_s.append(a)
                    info.append((x, u, tr))
                fin = [a for a in a_s if np.isfinite(a)]
                if not fin:
                    e.sigma *= 0.5
                    continue
                med_a = float(np.median(fin))
                score = [((1.0 - tr) * (a - med_a) if np.isfinite(a) else -np.inf)
                         for a, (_, _, tr) in zip(a_s, info)]
                order = list(np.argsort(score)[::-1])
                e.tell(order)
                for i in order[: cma_lam // 2]:
                    x, u, _ = info[i]
                    if np.isfinite(u):
                        cands.append((x, u))
            return cands

        dry = 0
        dry_cap = 3 if pareto_cma else 2
        bonus = 0
        elite = []  # elitist cross-round frontier archive (CMA-ME style)

        def _pairs_of(c):
            prs = getattr(c, "pairs", None)
            if prs is not None:
                return {(p[0], p[1]) for p in prs}
            return {(c.dr, c.t)}

        rnd = 0
        while rnd < birth_rounds + bonus:
            rnd += 1
            if len(comps) >= max_K_total or dry >= dry_cap:
                break
            probes = probe_frontier(target, mixture, sampler, rng)
            refined = _refine_probes(target, mixture, probes, 1.5)
            geo_pts.extend(x for x, _, _, _ in refined)
            elite = [(x, u, mixture.logq(x), t) for x, u, _, t in elite]
            merged = list(frontier) + elite
            if merged:
                med_a = float(np.median([-(u + lq) for _, u, lq, _ in merged]))
                merged.sort(key=lambda e: -max(0.0, -(e[1] + e[2]) - med_a) * (1.0 - e[3]))
                elite = merged[:60]
            frontier = merged
            if pareto_cma:
                off = cma_round_candidates()
                cands = off + [(x, u) for x, u, _, _ in refined]
                cands += [(x, u) for x, u, _, t in frontier if t < 0.5]
                pool = build_eval_pool(rng, pool_xs, pool_us, gen_comps, visited, mixture,
                                       frontier_pts=cands)
                new = try_transport() + pareto_birth(target, mixture, cands, pool, rng)
                if not new and dry >= 1:
                    # Robustness fallback: when LCB gating starves growth entirely
                    # (noisy objectives on a badly-covered target), admit charts by
                    # the score rule so coverage — and transport detection, which
                    # feeds on visited states — keep expanding.
                    new = birth_charts(target, mixture, frontier, probes, max_new=2,
                                       refined=refined)
            else:
                new = try_transport() + birth_charts(target, mixture, frontier, probes,
                                                     refined=refined)
            if not new:
                dry += 1
                # keep tuning so `visited` grows and transport detection can
                # still fire on a later dry round (the loop starves otherwise)
                da = DualAveraging(da.h_final)
                sampler.tune_round(n_tune // 2, da, collect_grads=(surrogate != "zero"),
                                   frontier=frontier, visited=visited)
                continue
            dry = 0
            if any(c.kind != "gauss" for c in new) and bonus < 2:
                bonus += 1  # adaptive budget: transports landing -> keep looking
            n_transport += sum(c.kind != "gauss" for c in new)
            n_born += sum(c.kind == "gauss" for c in new)
            for ns in [c for c in new if c.kind == "shear"]:
                # supersede: a fuller composed shear replaces subsumed partials
                comps[:] = [c for c in comps
                            if not (c.kind == "shear" and _pairs_of(c) <= _pairs_of(ns))]
            refit(new)
            if pareto_cma:
                for c in new:
                    if c.kind == "gauss":
                        emitters.append(CMAEmitter(c, mixture.d, lam=cma_lam))
                del emitters[:-n_emitters]  # FIFO cap
            frontier = []
            da = DualAveraging(da.h_final)
            sampler.tune_round(n_tune // 2, da, collect_grads=(surrogate != "zero"),
                               frontier=frontier, visited=visited)

        # Final transport refit from the fullest visited set: append the refit
        # chart (old ones stay in gen_comps for pool provenance; forward-KL
        # prunes whichever version is worse).
        if transport and len(visited) >= 40:
            have_refit = fit_transport_comps(visited_arr())
            if have_refit:
                n_transport += sum(1 for c in have_refit if c.kind not in
                                   {c2.kind for c2 in comps})
                refit(have_refit)

        # Milestone C: defensive heavy-tailed global proposal (fitted last, so it
        # wraps the final atlas; never enters R or the local charts).
        if t_defense:
            mus = np.array([c.mu if hasattr(c, "mu") else c.from_z(np.zeros(mixture.d))
                            for c in comps])
            base = np.mean([np.diag(c.cov) if isinstance(c, GaussComp)
                            else np.var(visited_arr(), axis=0) for c in comps], axis=0)
            if len(comps) > 1:
                base = base + np.var(mus, axis=0)
            t_def = TDefense(np.mean(mus, axis=0), np.diag(4.0 * base + 1e-6), nu=4.0)
            sampler.t_def = t_def

        sampler._rebuild_caches()
        da2 = DualAveraging(da.h_final)
        sampler.tune_round(n_tune2, da2)
        sampler.h = sampler.knee_refine_h(da2.h_final)
    else:
        sampler.init_state()

    warm_time = time.perf_counter() - t0
    warm_counts = dict(target.counts["warmup"])

    target.set_phase("production")
    chain = sampler.run(n_samples)
    cache_kinds = {}
    for c in sampler.caches:
        cache_kinds[c.kind] = cache_kinds.get(c.kind, 0) + 1
    chart_kinds = {}
    for c in sampler.mix.comps:
        chart_kinds[c.kind] = chart_kinds.get(c.kind, 0) + 1
    info = dict(
        K=sampler.mix.K,
        n_born=n_born,
        n_transport=n_transport,
        chart_kinds=chart_kinds,
        cache_kinds=cache_kinds,
        h=sampler.h,
        warm_time=warm_time,
        prod_time=sampler.stats["prod_time"],
        warm_U=warm_counts["U"],
        warm_grad=warm_counts["grad"],
        prod_U=target.counts["production"]["U"],
        prod_grad=target.counts["production"]["grad"],
        acc_local=(sampler.stats["local_acc"] / max(1, sampler.stats["local"])),
        acc_global=(sampler.stats["global_acc"] / max(1, sampler.stats["global_"])),
        mean_L=sampler.stats["mean_L"],
        n_anchors=len(sampler.anchors),
    )
    if return_sampler:
        return chain, info, sampler
    return chain, info
