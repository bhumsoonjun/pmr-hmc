"""Benchmark targets for PMR-HMC.

Each target is a CountedTarget: a JAX log-density with autodiff gradients and
per-phase evaluation counters, so warmup vs production true-model cost can be
reported exactly. Truth moments (where analytic) power the correctness checks.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


class CountedTarget:
    """Wraps a JAX logp with U/grad/hessian access and phase-tagged counters."""

    def __init__(self, name, d, logp_fn, truth_mean=None, truth_var=None, init_scale=3.0, defensive=False):
        self.name = name
        self.d = d
        self.init_scale = init_scale
        self.defensive = defensive
        self.truth_mean = None if truth_mean is None else np.asarray(truth_mean, dtype=float)
        self.truth_var = None if truth_var is None else np.asarray(truth_var, dtype=float)

        def U_fn(x):
            return -logp_fn(x)

        self.logp_jax = logp_fn
        self.U_jax = U_fn
        self._U = jax.jit(U_fn)
        self._Ub = jax.jit(jax.vmap(U_fn))
        self._vg = jax.jit(jax.value_and_grad(U_fn))
        self._hess = jax.jit(jax.hessian(U_fn))
        self.counts = {
            "warmup": {"U": 0, "grad": 0},
            "production": {"U": 0, "grad": 0},
        }
        self.phase = "warmup"

    def set_phase(self, phase):
        assert phase in self.counts
        self.phase = phase

    def U(self, x):
        self.counts[self.phase]["U"] += 1
        return float(self._U(jnp.asarray(x)))

    def U_batch(self, X):
        X = np.asarray(X)
        self.counts[self.phase]["U"] += len(X)
        return np.asarray(self._Ub(jnp.asarray(X)), dtype=float)

    def value_and_grad(self, x):
        c = self.counts[self.phase]
        c["U"] += 1
        c["grad"] += 1
        u, g = self._vg(jnp.asarray(x))
        return float(u), np.asarray(g, dtype=float)

    def hessian(self, x):
        # One reverse-over-forward hessian ~ d gradient passes; count it as such.
        self.counts[self.phase]["grad"] += self.d
        return np.asarray(self._hess(jnp.asarray(x)), dtype=float)

    def total(self, kind):
        return sum(self.counts[p][kind] for p in self.counts)

    # Cost model: reverse-mode value_and_grad ~ 2.5x a bare density eval.
    def eval_units(self, phase=None):
        phases = [phase] if phase else list(self.counts)
        return sum(self.counts[p]["U"] + 2.5 * self.counts[p]["grad"] for p in phases)


def gauss_iid(d=50):
    def logp(x):
        return -0.5 * jnp.sum(x**2)

    t = CountedTarget("gauss_iid", d, logp, truth_mean=np.zeros(d), truth_var=np.ones(d), init_scale=3.0)
    t.native_spec = (0, [], None, None)
    return t


def gauss_ill(d=50, cond=1e4, seed=0):
    rng = np.random.default_rng(seed)
    variances = np.logspace(-np.log10(cond) / 2, np.log10(cond) / 2, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    prec = Q @ np.diag(1.0 / variances) @ Q.T
    cov = Q @ np.diag(variances) @ Q.T
    prec_j = jnp.asarray(prec)

    def logp(x):
        return -0.5 * x @ (prec_j @ x)

    t = CountedTarget(
        "gauss_ill", d, logp, truth_mean=np.zeros(d), truth_var=np.diag(cov).copy(), init_scale=10.0
    )
    t.native_spec = (1, [], prec, None)
    return t


def mixture2(d=10, sep=4.0, w1=0.7):
    mu1 = np.zeros(d)
    mu1[0] = sep
    mu2 = -mu1
    mu1_j, mu2_j = jnp.asarray(mu1), jnp.asarray(mu2)
    lw1, lw2 = np.log(w1), np.log(1.0 - w1)

    def logp(x):
        a = lw1 - 0.5 * jnp.sum((x - mu1_j) ** 2)
        b = lw2 - 0.5 * jnp.sum((x - mu2_j) ** 2)
        return jax.scipy.special.logsumexp(jnp.stack([a, b]))

    m1 = w1 * sep + (1 - w1) * (-sep)
    var1 = w1 * (sep**2 + 1) + (1 - w1) * (sep**2 + 1) - m1**2
    tm = np.zeros(d)
    tm[0] = m1
    tv = np.ones(d)
    tv[0] = var1
    t = CountedTarget("mixture2", d, logp, truth_mean=tm, truth_var=tv, init_scale=6.0)
    t.native_spec = (2, [sep, w1], None, None)
    return t


def banana(d=20, b=0.1):
    def logp(x):
        u = x[0] ** 2 / 200.0 + 0.5 * (x[1] + b * x[0] ** 2 - 100.0 * b) ** 2
        u = u + 0.5 * jnp.sum(x[2:] ** 2)
        return -u

    tm = np.zeros(d)
    tv = np.ones(d)
    tv[0] = 100.0
    tm[1] = 100.0 * b - b * 100.0  # E[x1] = 100b - b E[x0^2] = 0
    tv[1] = 1.0 + b**2 * 2 * 100.0**2  # Var = 1 + b^2 Var(x0^2)
    t = CountedTarget("banana", d, logp, truth_mean=tm, truth_var=tv, init_scale=8.0, defensive=True)
    t.native_spec = (3, [b], None, None)
    return t


def funnel(d=10):
    # v = x[0] ~ N(0, 9); x_i | v ~ N(0, e^v)
    def logp(x):
        v = x[0]
        rest = x[1:]
        return -(v**2) / 18.0 - 0.5 * (d - 1) * v - 0.5 * jnp.sum(rest**2) * jnp.exp(-v)

    tm = np.zeros(d)
    tv = np.ones(d) * np.exp(4.5)
    tv[0] = 9.0
    t = CountedTarget("funnel", d, logp, truth_mean=tm, truth_var=tv, init_scale=3.0, defensive=True)
    t.native_spec = (4, [], None, None)
    return t


def ring(d=20, R=10.0, w=0.5):
    """2-d circular ridge embedded in d dims: curved structure that neither the
    shear nor the scale transport family can absorb — the atlas must tile it."""

    def logp(x):
        rho = jnp.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12)
        return -((rho - R) ** 2) / (2.0 * w**2) - 0.5 * jnp.sum(x[2:] ** 2)

    tm = np.zeros(d)
    tv = np.ones(d)
    tv[0] = tv[1] = (R**2 + w**2) / 2.0  # E[rho^2]/2, Jacobian tilt ~O(w^2/R^2) ignored
    t = CountedTarget("ring", d, logp, truth_mean=tm, truth_var=tv, init_scale=8.0, defensive=True)
    t.native_spec = (5, [R, w], None, None)
    return t


def logreg(n=400, p=25, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[:5] = [2.0, -2.0, 1.5, -1.0, 1.0]
    logits = 0.5 + X @ beta_true
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(float)
    Xj, yj = jnp.asarray(X), jnp.asarray(y)
    prior_scale = 2.5

    def logp(x):
        t = x[0] + Xj @ x[1:]
        ll = jnp.sum(yj * jax.nn.log_sigmoid(t) + (1.0 - yj) * jax.nn.log_sigmoid(-t))
        return ll - 0.5 * jnp.sum(x**2) / prior_scale**2

    # No analytic truth; benchmark uses a long NUTS run as the reference.
    t = CountedTarget("logreg", p + 1, logp, init_scale=1.0)
    t.native_spec = (10, [], X, y)
    return t


def eight_schools():
    """Centered eight-schools: the canonical hierarchical funnel in the wild.
    x = [mu, log_tau, theta_1..8], d=10."""
    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    s = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    def logp(x):
        mu, log_tau = x[0], x[1]
        theta = x[2:]
        tau = jnp.exp(log_tau)
        ll = -0.5 * jnp.sum(((y - theta) / s) ** 2)
        lp = -0.5 * jnp.sum(((theta - mu) / tau) ** 2) - 8.0 * log_tau
        return ll + lp - 0.5 * (mu / 5.0) ** 2 - 0.5 * (log_tau / 1.5) ** 2

    return CountedTarget("eight_schools", 10, logp, init_scale=2.0, defensive=True)


def hier_binom(J=50, seed=13):
    """Centered hierarchical binomial (synthetic): x = [mu, log_tau, alpha_1..J]."""
    rng = np.random.default_rng(seed)
    alpha_true = rng.normal(-1.0, 0.8, J)
    n = rng.integers(20, 100, J)
    yv = rng.binomial(n, 1.0 / (1.0 + np.exp(-alpha_true)))
    nj, yj = jnp.asarray(n, dtype=jnp.float64), jnp.asarray(yv, dtype=jnp.float64)

    def logp(x):
        mu, log_tau = x[0], x[1]
        a = x[2:]
        tau = jnp.exp(log_tau)
        ll = jnp.sum(yj * jax.nn.log_sigmoid(a) + (nj - yj) * jax.nn.log_sigmoid(-a))
        lp = -0.5 * jnp.sum(((a - mu) / tau) ** 2) - J * log_tau
        return ll + lp - 0.5 * (mu / 2.0) ** 2 - 0.5 * (log_tau / 1.5) ** 2

    return CountedTarget("hier_binom", J + 2, logp, init_scale=1.5, defensive=True)


def breast_logreg():
    """Bayesian logistic regression on the real breast-cancer dataset
    (569 x 30, standardized) + intercept; prior N(0, 2.5^2). d=31."""
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = (data.data - data.data.mean(0)) / data.data.std(0)
    Xj, yj = jnp.asarray(X), jnp.asarray(data.target.astype(float))

    def logp(x):
        t = x[0] + Xj @ x[1:]
        ll = jnp.sum(yj * jax.nn.log_sigmoid(t) + (1.0 - yj) * jax.nn.log_sigmoid(-t))
        return ll - 0.5 * jnp.sum(x**2) / 2.5**2

    return CountedTarget("breast_logreg", 31, logp, init_scale=1.0)


def vilya_bcf(C=8, n_first=600, n_pos=400, seed=5):
    """Faithful reproduction of the Vilya joint BCF hurdle-NB objective
    (vilya/scripts/ads_budget/joint_bcf_hurdle_nb.py): Gaussian first-stage
    log-spend regression whose tanh-standardized residual (control function)
    enters a truncated negative-binomial second stage with per-campaign
    intercepts + slopes, ridge priors, and soft slope-clip penalties.
    Synthetic data drawn from the model at realistic sizes. d = 37."""
    rng = np.random.default_rng(seed)

    def fs_design(camp, dd, trend):
        n = len(camp)
        Xf = np.zeros((n, C + 3))
        Xf[:, 0] = 1.0
        Xf[np.arange(n), 1 + camp] = 1.0
        Xf[:, C + 1] = dd
        Xf[:, C + 2] = trend
        return Xf

    camp_f = rng.integers(0, C, n_first)
    dd_f = (rng.random(n_first) < 0.15).astype(float)
    Xf = fs_design(camp_f, dd_f, rng.normal(0, 1, n_first))
    fs_beta_true = np.concatenate([[3.0], rng.normal(0, 0.5, C), [0.4], [0.2]])
    sig_true = 0.6
    ls_f = Xf @ fs_beta_true + rng.normal(0, sig_true, n_first)

    camp_p = rng.integers(0, C, n_pos)
    dd_p = (rng.random(n_pos) < 0.15).astype(float)
    Xpf = fs_design(camp_p, dd_p, rng.normal(0, 1, n_pos))
    ls_p = Xpf @ fs_beta_true + rng.normal(0, sig_true, n_pos)
    cf_true = np.tanh((ls_p - Xpf @ fs_beta_true) / sig_true)
    rich = rng.normal(0, 1, (n_pos, 4))
    ls_c = ls_p - ls_p.mean()
    # constant design: [1 | camp dummies | camp dummy * centered log-spend | rich | shared slope col]
    Xc = np.zeros((n_pos, 1 + 2 * C + 4 + 1))
    Xc[:, 0] = 1.0
    Xc[np.arange(n_pos), 1 + camp_p] = 1.0
    Xc[np.arange(n_pos), 1 + C + camp_p] = ls_c
    Xc[:, 1 + 2 * C: 1 + 2 * C + 4] = rich
    Xc[:, -1] = ls_c
    beta_true = np.concatenate([[1.2], rng.normal(0, 0.4, C), rng.normal(0, 0.1, C),
                                rng.normal(0, 0.2, 4), [0.5]])
    rho_true, ddrho_true = 0.6, 0.3
    alpha_true = 0.5
    eta = np.clip(Xc @ beta_true + cf_true * (1 - dd_p) * rho_true + cf_true * dd_p * ddrho_true,
                  -18, 18)
    mu_nb = np.exp(eta)
    size = 1.0 / alpha_true
    y = np.zeros(n_pos)
    for i in range(n_pos):  # truncated NB via rejection
        for _ in range(200):
            v = rng.negative_binomial(size, size / (size + mu_nb[i]))
            if v > 0:
                y[i] = v
                break
        else:
            y[i] = 1.0

    n_fs = C + 3
    n_c = Xc.shape[1]
    d = n_fs + 1 + n_c + 2 + 1  # fs_beta, log_sigma, pos_beta(+rho,+ddrho), log_alpha
    Xf_j, ls_f_j = jnp.asarray(Xf), jnp.asarray(ls_f)
    Xpf_j, ls_p_j = jnp.asarray(Xpf), jnp.asarray(ls_p)
    Xc_j, dd_j, y_j = jnp.asarray(Xc), jnp.asarray(dd_p), jnp.asarray(y)
    from jax.scipy.special import gammaln

    shared_idx = n_c - 1
    slope_start = 1 + C
    lo, hi = -1.0, 2.5  # slope clip window (CURVATURE_CLIP analogue)

    def logp(p):
        fs_beta = p[:n_fs]
        log_sigma = p[n_fs]
        pos_beta = p[n_fs + 1: n_fs + 1 + n_c]
        rho = p[n_fs + 1 + n_c]
        ddrho = p[n_fs + 2 + n_c]
        log_alpha = p[-1]
        sigma = jnp.exp(log_sigma)
        alpha = jnp.exp(log_alpha)
        std = (ls_f_j - Xf_j @ fs_beta) / sigma
        spend_nll = jnp.sum(log_sigma + 0.5 * std**2)
        cf = jnp.tanh((ls_p_j - Xpf_j @ fs_beta) / sigma)
        eta = jnp.clip(Xc_j @ pos_beta + cf * (1 - dd_j) * rho + cf * dd_j * ddrho, -18.0, 18.0)
        m = jnp.exp(eta)
        sz = 1.0 / alpha
        log_p0 = -jnp.log1p(alpha * m) / alpha
        log_nz = jnp.log(jnp.maximum(-jnp.expm1(log_p0), 1e-300))
        ll = (gammaln(y_j + sz) - gammaln(sz) - gammaln(y_j + 1.0)
              + sz * (jnp.log(sz) - jnp.log(sz + m))
              + y_j * (jnp.log(m) - jnp.log(sz + m)) - log_nz)
        pen = 0.5 * jnp.sum(fs_beta**2) + 0.5 * jnp.sum(pos_beta**2) + 0.5 * sigma**2 \
            + 0.5 * (rho**2 + ddrho**2) + 0.5 * log_alpha**2
        eff = pos_beta[shared_idx] + pos_beta[slope_start: slope_start + C]
        wall = 1e4 * (jnp.square(jnp.maximum(lo - pos_beta[shared_idx], 0.0))
                      + jnp.square(jnp.maximum(pos_beta[shared_idx] - hi, 0.0))
                      + jnp.sum(jnp.square(jnp.maximum(lo - eff, 0.0)))
                      + jnp.sum(jnp.square(jnp.maximum(eff - hi, 0.0))))
        return -(spend_nll - jnp.sum(ll) + pen + wall)

    return CountedTarget("vilya_bcf", d, logp, init_scale=0.5)


# ---------------------------------------------------------------------------
# Extended suite factories
# ---------------------------------------------------------------------------
def gauss_corr(d=30, rho=0.9):
    cov = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    prec = jnp.asarray(np.linalg.inv(cov))

    def logp(x):
        return -0.5 * x @ (prec @ x)

    t = CountedTarget(f"gauss_corr{d}", d, logp, truth_mean=np.zeros(d),
                      truth_var=np.diag(cov).copy(), init_scale=3.0)
    t.native_spec = (1, [], np.linalg.inv(cov), None)
    return t


def student_t(d=20, nu=4.0):
    def logp(x):
        return -0.5 * (nu + d) * jnp.log1p(jnp.sum(x**2) / nu)

    tv = np.full(d, nu / (nu - 2)) if nu > 2 else None
    t = CountedTarget(f"student_t{d}_nu{nu:g}", d, logp,
                      truth_mean=np.zeros(d) if nu > 1 else None,
                      truth_var=tv, init_scale=3.0, defensive=True)
    t.native_spec = (6, [nu], None, None)
    return t


def logcosh(d=20):
    def logp(x):
        return -jnp.sum(jnp.logaddexp(x, -x))  # log(2cosh) tails ~ Laplace

    t = CountedTarget("logcosh", d, logp, truth_mean=np.zeros(d),
                      truth_var=np.full(d, np.pi**2 / 4), init_scale=3.0)
    t.native_spec = (7, [], None, None)
    return t


def mixture3(d=10):
    mus = np.zeros((3, d))
    mus[0, 0], mus[1, 0], mus[1, 1], mus[2, 1] = 5.0, -4.0, 3.0, -5.0
    lws = np.log(np.array([0.5, 0.3, 0.2]))
    mj = jnp.asarray(mus)

    def logp(x):
        lc = jnp.asarray(lws) - 0.5 * jnp.sum((x[None, :] - mj) ** 2, axis=1)
        return jax.scipy.special.logsumexp(lc)

    w = np.array([0.5, 0.3, 0.2])
    tm = w @ mus
    tv = w @ (mus**2 + 1.0) - tm**2
    return CountedTarget("mixture3", d, logp, truth_mean=tm, truth_var=tv, init_scale=6.0)


def rosenbrock(pairs=5, a=1.0, b=5.0):
    d = 2 * pairs

    def logp(x):
        xe, xo = x[0::2], x[1::2]
        return -jnp.sum((a - xe) ** 2 / 20.0 + b * (xo - xe**2) ** 2 / 10.0)

    t = CountedTarget(f"rosenbrock{d}", d, logp, init_scale=3.0, defensive=True)
    t.native_spec = (8, [a, b], None, None)
    return t


def shell(d=20, R=8.0, w=0.5):
    """3-sphere shell in coords 0..2 — no current transport family fits it."""

    def logp(x):
        r = jnp.sqrt(jnp.sum(x[:3] ** 2) + 1e-12)
        return -((r - R) ** 2) / (2 * w**2) - 0.5 * jnp.sum(x[3:] ** 2)

    tv = np.ones(d)
    tv[:3] = (R**2 + w**2) / 3.0
    return CountedTarget("shell", d, logp, truth_mean=np.zeros(d), truth_var=tv,
                         init_scale=6.0, defensive=True)


def squiggle(d=10, freq=3.0):
    def logp(x):
        return -(x[0] ** 2) / 200.0 - 0.5 * (x[1] + jnp.sin(freq * x[0])) ** 2 \
            - 0.5 * jnp.sum(x[2:] ** 2)

    tm = np.zeros(d)
    tv = np.ones(d)
    tv[0] = 100.0
    tv[1] = 1.0 + 0.5  # Var(sin(f x0)) -> 1/2 for dispersed x0
    t = CountedTarget("squiggle", d, logp, truth_mean=tm, truth_var=tv,
                      init_scale=6.0, defensive=True)
    t.native_spec = (9, [freq], None, None)
    return t


def hier_gauss(J=40, seed=17):
    """Centered Gaussian random effects with unknown obs noise. d = J + 3."""
    rng = np.random.default_rng(seed)
    alpha_true = rng.normal(2.0, 1.2, J)
    n_per = 5
    yb = jnp.asarray(np.array([rng.normal(a, 1.5, n_per).mean() for a in alpha_true]))

    def logp(x):
        mu, log_tau, log_sy = x[0], x[1], x[2]
        a = x[3:]
        tau, sy = jnp.exp(log_tau), jnp.exp(log_sy)
        se = sy / np.sqrt(n_per)
        ll = -0.5 * jnp.sum(((yb - a) / se) ** 2) - J * jnp.log(se)
        lp = -0.5 * jnp.sum(((a - mu) / tau) ** 2) - J * log_tau
        return ll + lp - 0.5 * (mu / 5) ** 2 - 0.5 * (log_tau / 1.5) ** 2 - 0.5 * (log_sy / 1.5) ** 2

    return CountedTarget("hier_gauss", J + 3, logp, init_scale=1.5, defensive=True)


def horseshoe_reg(p=10, n=100, seed=19):
    """Sparse regression with per-coefficient local scales: p coupled funnels.
    x = [log_tau, log_lam_1..p, b_1..p, log_sig]. d = 2p + 2."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    b_true = np.zeros(p)
    b_true[:3] = [2.0, -1.5, 1.0]
    yv = X @ b_true + rng.normal(0, 1.0, n)
    Xj, yj = jnp.asarray(X), jnp.asarray(yv)

    def logp(x):
        log_tau = x[0]
        log_lam = x[1:1 + p]
        b = x[1 + p:1 + 2 * p]
        log_sig = x[-1]
        tau, lam, sig = jnp.exp(log_tau), jnp.exp(log_lam), jnp.exp(log_sig)
        ll = -0.5 * jnp.sum(((yj - Xj @ b) / sig) ** 2) - n * log_sig
        lb = -0.5 * jnp.sum((b / (tau * lam)) ** 2) - jnp.sum(log_lam) - p * log_tau
        return ll + lb - 0.5 * jnp.sum(log_lam**2) - 0.5 * (log_tau + 1.0) ** 2 \
            - 0.5 * (log_sig / 1.5) ** 2

    return CountedTarget("horseshoe_reg", 2 * p + 2, logp, init_scale=1.0, defensive=True)


def poisson_reg(p=20, n=300, seed=23):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * 0.5
    b_true = np.zeros(p)
    b_true[:4] = [0.6, -0.4, 0.3, 0.5]
    yv = rng.poisson(np.exp(0.5 + X @ b_true))
    Xj, yj = jnp.asarray(X), jnp.asarray(yv.astype(float))

    def logp(x):
        eta = jnp.clip(x[0] + Xj @ x[1:], -20, 20)
        return jnp.sum(yj * eta - jnp.exp(eta)) - 0.5 * jnp.sum(x**2) / 2.5**2

    t = CountedTarget("poisson_reg", p + 1, logp, init_scale=0.8)
    t.native_spec = (11, [], X, yv.astype(float))
    return t


def diabetes_linreg():
    """Bayesian linear regression on sklearn diabetes (442 x 10), unknown noise."""
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X = (data.data - data.data.mean(0)) / data.data.std(0)
    yv = (data.target - data.target.mean()) / data.target.std()
    Xj, yj = jnp.asarray(X), jnp.asarray(yv)
    n, p = X.shape

    def logp(x):
        b0, b, log_sig = x[0], x[1:1 + p], x[-1]
        sig = jnp.exp(log_sig)
        ll = -0.5 * jnp.sum(((yj - b0 - Xj @ b) / sig) ** 2) - n * log_sig
        return ll - 0.5 * jnp.sum(b**2) / 4.0 - 0.5 * b0**2 / 4.0 - 0.5 * (log_sig / 1.5) ** 2

    return CountedTarget("diabetes_linreg", p + 2, logp, init_scale=0.8)


def gp_hyper(n=60, seed=29):
    """GP marginal-likelihood posterior over (log_ls, log_amp, log_noise) — curved
    ridges from length-scale/amplitude trade-offs. d = 3."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 10, n))
    K0 = np.exp(-0.5 * (t[:, None] - t[None, :]) ** 2 / 1.5**2)
    yv = np.linalg.cholesky(K0 + 0.05 * np.eye(n)) @ rng.standard_normal(n)
    tj, yj = jnp.asarray(t), jnp.asarray(yv)

    def logp(x):
        ls, amp, noise = jnp.exp(x[0]), jnp.exp(x[1]), jnp.exp(x[2])
        K = amp**2 * jnp.exp(-0.5 * (tj[:, None] - tj[None, :]) ** 2 / ls**2) \
            + (noise**2 + 1e-6) * jnp.eye(n)
        L = jnp.linalg.cholesky(K)
        a = jax.scipy.linalg.solve_triangular(L, yj, lower=True)
        return -0.5 * a @ a - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * jnp.sum(x**2) / 2.0**2

    return CountedTarget("gp_hyper", 3, logp, init_scale=0.7, defensive=True)


def suite():
    """33-posterior benchmark suite, ordered roughly by trickiness tier."""
    return [
        # Gaussian tier
        ("gauss_iid10", lambda: gauss_iid(10)), ("gauss_iid50", lambda: gauss_iid(50)),
        ("gauss_corr30", lambda: gauss_corr(30)),
        ("gauss_ill_1e2", lambda: gauss_ill(30, 1e2)), ("gauss_ill_1e4", lambda: gauss_ill(50, 1e4)),
        ("gauss_ill_1e6", lambda: gauss_ill(30, 1e6)),
        # heavy tails
        ("student_t20_nu4", lambda: student_t(20, 4.0)), ("student_t10_nu2.5", lambda: student_t(10, 2.5)),
        ("logcosh20", logcosh),
        # multimodal
        ("mixture2_s4", lambda: mixture2(10, 4.0)), ("mixture2_s8", lambda: mixture2(10, 8.0)),
        ("mixture3", mixture3), ("mixture2_d40", lambda: mixture2(40, 6.0)),
        # curved
        ("banana20", lambda: banana(20)), ("banana50", lambda: banana(50)),
        ("banana_strong", lambda: banana(20, b=0.3)), ("rosenbrock10", rosenbrock),
        ("ring20", ring), ("shell20", shell), ("squiggle10", squiggle),
        # funnels / hierarchical
        ("funnel10", lambda: funnel(10)), ("funnel25", lambda: funnel(25)),
        ("eight_schools", eight_schools), ("hier_binom52", hier_binom),
        ("hier_gauss43", hier_gauss), ("horseshoe_reg22", horseshoe_reg),
        # GLMs / real data
        ("logreg26", logreg), ("breast_logreg31", breast_logreg),
        ("poisson_reg21", poisson_reg), ("diabetes_linreg12", diabetes_linreg),
        ("gp_hyper3", gp_hyper),
        # real-structure production model
        ("vilya_bcf37", vilya_bcf),
        ("mixture2_base", lambda: mixture2(10, 6.0)),
    ]


def all_targets():
    return [gauss_iid(), gauss_ill(), mixture2(), banana(), funnel(), ring(), logreg()]


def probit_reg(p=15, n=250, seed=31):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * 0.8
    b_true = np.zeros(p); b_true[:4] = [1.2, -0.8, 0.6, 0.9]
    yv = (rng.random(n) < 0.5 * (1 + jax.scipy.special.erf((X @ b_true) / np.sqrt(2)))).astype(float)
    Xj, yj = jnp.asarray(X), jnp.asarray(np.asarray(yv, dtype=float))

    def logp(x):
        t = Xj @ x
        lc = jax.scipy.stats.norm.logcdf(t)
        lcn = jax.scipy.stats.norm.logcdf(-t)
        return jnp.sum(yj * lc + (1 - yj) * lcn) - 0.5 * jnp.sum(x**2) / 4.0

    return CountedTarget(f"probit{p}_{seed}", p, logp, init_scale=0.8)


def negbin_reg(p=12, n=300, seed=37):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * 0.5
    b_true = np.zeros(p); b_true[:3] = [0.7, -0.5, 0.4]
    mu = np.exp(0.8 + X @ b_true)
    r_disp = 3.0
    yv = rng.negative_binomial(r_disp, r_disp / (r_disp + mu))
    Xj, yj = jnp.asarray(X), jnp.asarray(yv.astype(float))
    from jax.scipy.special import gammaln as gl

    def logp(x):
        b, lr = x[:p], x[p]
        r = jnp.exp(lr)
        eta = jnp.clip(0.8 + Xj @ b, -15, 15)
        m = jnp.exp(eta)
        ll = (gl(yj + r) - gl(r) - gl(yj + 1) + r * (jnp.log(r) - jnp.log(r + m))
              + yj * (eta - jnp.log(r + m)))
        return jnp.sum(ll) - 0.5 * jnp.sum(b**2) / 4.0 - 0.5 * lr**2 + 0.0

    return CountedTarget(f"negbin{p}_{seed}", p + 1, logp, init_scale=0.6)


def cauchy_reg(p=8, n=150, seed=41):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    b_true = np.zeros(p); b_true[:3] = [1.5, -1.0, 0.7]
    yv = X @ b_true + rng.standard_cauchy(n) * 0.5
    Xj, yj = jnp.asarray(X), jnp.asarray(yv)

    def logp(x):
        b, ls = x[:p], x[p]
        s = jnp.exp(ls)
        resid = (yj - Xj @ b) / s
        return (-jnp.sum(jnp.log1p(resid**2)) - n * ls
                - 0.5 * jnp.sum(b**2) / 25.0 - 0.5 * ls**2 + 0.0)

    return CountedTarget(f"cauchyreg{p}_{seed}", p + 1, logp, init_scale=0.8, defensive=True)


def skew_gauss(d=15, alpha=4.0):
    def logp(x):
        base = -0.5 * jnp.sum(x**2)
        skew = jnp.sum(jax.scipy.stats.norm.logcdf(alpha * x) + jnp.log(2.0))
        return base + skew

    return CountedTarget(f"skew{d}_a{alpha:g}", d, logp, init_scale=2.0)


def mega_suite():
    """~120 parametric targets across 18 families for the wide PMR-vs-NUTS sweep."""
    out = []
    for d in (5, 20, 100):
        out.append((f"m_gauss_iid{d}", lambda d=d: gauss_iid(d)))
    for cond in (1e3, 1e5):
        for d in (20, 80):
            out.append((f"m_gauss_ill{d}_k{cond:.0e}", lambda d=d, c=cond: gauss_ill(d, c, seed=d)))
    for rho in (0.5, 0.8, 0.95):
        for d in (15, 40):
            out.append((f"m_corr{d}_r{rho}", lambda d=d, r=rho: gauss_corr(d, r)))
    for nu in (2.5, 5.0, 10.0):
        for d in (10, 30):
            out.append((f"m_t{d}_nu{nu:g}", lambda d=d, n=nu: student_t(d, n)))
    for d in (10, 25):
        out.append((f"m_logcosh{d}", lambda d=d: logcosh(d)))
    for b in (0.05, 0.1, 0.2):
        for d in (10, 30):
            out.append((f"m_banana{d}_b{b}", lambda d=d, b=b: banana(d, b)))
    for sep in (4.0, 8.0):
        for d in (10, 30):
            out.append((f"m_mix2_{d}_s{sep:g}", lambda d=d, s=sep: mixture2(d, s)))
    out.append(("m_mix3", mixture3))
    for d in (8, 20):
        out.append((f"m_funnel{d}", lambda d=d: funnel(d)))
    for R in (6.0, 12.0):
        out.append((f"m_ring_R{R:g}", lambda R=R: ring(20, R)))
    out.append(("m_shell", shell))
    for f in (2.0, 4.0):
        out.append((f"m_squig_f{f:g}", lambda f=f: squiggle(10, f)))
    for pr in (2, 5):
        out.append((f"m_rosen{2*pr}", lambda p=pr: rosenbrock(p)))
    for sd in (11, 12, 13):
        for p in (10, 30):
            out.append((f"m_logreg{p}_{sd}", lambda p=p, s=sd: logreg(n=15*p, p=p, seed=s)))
    for sd in (23, 24):
        for p in (10, 25):
            out.append((f"m_pois{p}_{sd}", lambda p=p, s=sd: poisson_reg(p=p, n=12*p, seed=s)))
    for sd in (31, 32, 33):
        out.append((f"m_probit_{sd}", lambda s=sd: probit_reg(seed=s)))
    for sd in (37, 38):
        out.append((f"m_negbin_{sd}", lambda s=sd: negbin_reg(seed=s)))
    for sd in (41, 42):
        out.append((f"m_cauchyreg_{sd}", lambda s=sd: cauchy_reg(seed=s)))
    for a in (2.0, 6.0):
        out.append((f"m_skew_a{a:g}", lambda a=a: skew_gauss(15, a)))
    for J in (20, 60):
        out.append((f"m_hgauss{J}", lambda J=J: hier_gauss(J)))
    for J in (25, 75):
        out.append((f"m_hbinom{J}", lambda J=J: hier_binom(J)))
    for p in (5, 15):
        out.append((f"m_horse{p}", lambda p=p: horseshoe_reg(p=p)))
    for sd in (17, 18):
        out.append((f"m_eightsch_{sd}", eight_schools) if sd == 17 else (f"m_hg43_{sd}", hier_gauss))
    for sd in (29, 30):
        out.append((f"m_gph_{sd}", lambda s=sd: gp_hyper(seed=s)))
    for C in (5, 12):
        out.append((f"m_vilya{C}", lambda C=C: vilya_bcf(C=C, n_first=60*C, n_pos=40*C)))
    # extended grids
    out.append(("m_gauss_iid300", lambda: gauss_iid(300)))
    out.append(("m_gauss_ill200_k1e4", lambda: gauss_ill(200, 1e4, seed=9)))
    for d in (60, 150):
        out.append((f"m_corr{d}_r0.9", lambda d=d: gauss_corr(d, 0.9)))
    for d in (60,):
        out.append((f"m_t{d}_nu4", lambda d=d: student_t(d, 4.0)))
        out.append((f"m_banana{d}_b0.1", lambda d=d: banana(d, 0.1)))
    for d in (40, 60):
        out.append((f"m_funnel{d}", lambda d=d: funnel(d)))
    for w1 in (0.5, 0.9):
        out.append((f"m_mix2_w{w1}", lambda w=w1: mixture2(10, 6.0, w)))
    for sd in (14, 15, 16):
        out.append((f"m_logreg50_{sd}", lambda s=sd: logreg(n=600, p=50, seed=s)))
    for sd in (25, 26):
        out.append((f"m_pois40_{sd}", lambda s=sd: poisson_reg(p=40, n=500, seed=s)))
    for sd in (34, 35, 36):
        out.append((f"m_probit30_{sd}", lambda s=sd: probit_reg(p=30, n=450, seed=s)))
    for sd in (39, 40):
        out.append((f"m_negbin25_{sd}", lambda s=sd: negbin_reg(p=25, n=400, seed=s)))
    for sd in (43, 44):
        out.append((f"m_cauchyreg15_{sd}", lambda s=sd: cauchy_reg(p=15, n=250, seed=s)))
    for a_ in (3.0,):
        for d in (40,):
            out.append((f"m_skew{d}_a{a_:g}", lambda d=d, a=a_: skew_gauss(d, a)))
    for J in (120,):
        out.append((f"m_hbinom{J}", lambda J=J: hier_binom(J)))
    for J in (100,):
        out.append((f"m_hgauss{J}", lambda J=J: hier_gauss(J)))
    for p in (25,):
        out.append((f"m_horse{p}", lambda p=p: horseshoe_reg(p=p)))
    for sd in (19, 21):
        out.append((f"m_diab_{sd}", diabetes_linreg) if sd == 19 else (f"m_breast_{sd}", breast_logreg))
    for R in (20.0,):
        out.append((f"m_ring_R{R:g}", lambda R=R: ring(30, R)))
    for pr in (8,):
        out.append((f"m_rosen{2*pr}", lambda p=pr: rosenbrock(p)))
    return out
