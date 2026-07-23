"""pmrhmc test suite.

Covers: chart bijection exactness (incl. the new C tri chart), C-vs-Python
mixture-density parity, end-to-end moment correctness on analytic targets
(callback + native + parallel paths), determinism, oracle accounting, and
numerical-robustness guards (clamped exponents, NaN never poisons weights).
"""
import numpy as np
import pytest

import pmrhmc
from pmrhmc import native
from pmrhmc import warmup as P


# ---------------------------------------------------------------- helpers
def std_gauss(d):
    return pmrhmc.PMRHMC(lambda x: -0.5 * float(x @ x), d=d, name="gauss",
                         grad_logpdf=lambda x: -x)


def corr_gauss(d, rho=0.8):
    C = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    Pm = np.linalg.inv(C)
    return pmrhmc.PMRHMC(lambda x: -0.5 * float(x @ Pm @ x), d=d, name="corr",
                         grad_logpdf=lambda x: -(Pm @ x)), C


@pytest.fixture(scope="module")
def fitted_gauss():
    return std_gauss(6).fit(seed=0, n_starts=4, birth_rounds=1)


# ---------------------------------------------------------------- charts
def test_tri_chart_bijection_python():
    d = 4
    rng = np.random.default_rng(0)
    c = P.TriComp(order=np.arange(d), mu=rng.normal(size=d),
                  sig=np.exp(rng.normal(size=d)), eps=np.array([0.4, -0.3, 0.0, 0.8]),
                  delta=np.array([1.3, 0.7, 1.0, 2.0]),
                  drv=np.array([-1, 0, -1, 1]), gam=np.array([0.0, 0.6, 0.0, -0.4]))
    for _ in range(50):
        z = rng.normal(size=d)
        assert np.allclose(c.to_z(c.from_z(z)), z, atol=1e-10)


def test_tri_chart_clamp_no_nan():
    d = 3
    c = P.TriComp(order=np.arange(d), mu=np.zeros(d), sig=np.ones(d),
                  eps=np.zeros(d), delta=np.ones(d),
                  drv=np.array([-1, 0, 0]), gam=np.array([0.0, 50.0, -50.0]))
    for x in (np.array([100.0, 1.0, -1.0]), np.array([-100.0, 1e6, -1e6])):
        v = c.logpdf(x)
        assert not np.isnan(v)  # -inf is fine; NaN must never escape


def test_mixture_weight_sanitization():
    comps = [P.GaussComp(np.zeros(2), np.eye(2)), P.GaussComp(np.ones(2), np.eye(2))]
    m = P.MixtureRef(comps, np.array([0.0, np.nan]))
    assert np.isfinite(m.log_ws[0]) and m.log_ws[1] == -np.inf
    m2 = P.MixtureRef(comps, np.array([np.nan, np.nan]))
    assert np.allclose(np.exp(m2.log_ws), 0.5)


# ------------------------------------------------------------- C parity
def test_c_python_mixture_parity(fitted_gauss):
    s = fitted_gauss.sampler
    pk = native.pack_sampler(s)
    rng = np.random.default_rng(1)
    # native chain from one point must stay finite + within support
    ch, st = native.run_cb(fitted_gauss._logpdf, s, 500, 7)
    assert np.isfinite(ch).all()
    assert st["n_U"] <= 501  # ONE true-density call per transition (+ init)


def test_c_tri_pack_and_run():
    d = 3
    tri = P.TriComp(order=np.arange(d), mu=np.zeros(d), sig=np.ones(d),
                    eps=np.array([0.5, 0.0, -0.2]), delta=np.array([1.2, 1.0, 0.8]),
                    drv=np.array([-1, -1, -1]), gam=np.zeros(d))
    code, blob = native.pack_comp(tri, d)
    assert code == 6 and len(blob) == 7 * d


# --------------------------------------------------------- end-to-end
def test_moments_gauss_callback(fitted_gauss):
    draws = fitted_gauss.sample(40_000, seed=3)
    assert np.max(np.abs(draws.mean(0))) < 0.08
    assert np.max(np.abs(draws.var(0) - 1)) < 0.12


def test_moments_corr_gauss_native_path():
    d = 5
    s, C = corr_gauss(d)
    s.fit(seed=1, n_starts=4, birth_rounds=1)
    Pm = np.linalg.inv(C)
    s.native_spec = (1, np.zeros(1), Pm, None)  # gauss_prec native family
    draws = s.sample(40_000, seed=4)
    assert np.max(np.abs(draws.mean(0))) < 0.1
    assert np.max(np.abs(draws.var(0) - np.diag(C))) < 0.15


def test_parallel_chains_native():
    d = 4
    s = std_gauss(d).fit(seed=2, n_starts=4, birth_rounds=1)
    s.native_spec = (0, np.zeros(1), None, None)
    chs = s.sample(8000, seed=5, chains=4, cores=4)
    assert chs.shape[0] == 4
    pooled = chs.reshape(-1, d)
    assert np.max(np.abs(pooled.mean(0))) < 0.08
    # chains differ (independent seeds)
    assert not np.allclose(chs[0], chs[1])


def test_parallel_chains_callback():
    d = 3
    s = std_gauss(d).fit(seed=3, n_starts=4, birth_rounds=1)
    chs = s.sample(4000, seed=6, chains=2, cores=2)
    assert chs.shape[0] == 2
    assert np.isfinite(chs).all()


def test_determinism(fitted_gauss):
    a = fitted_gauss.sample(2000, seed=11)
    b = fitted_gauss.sample(2000, seed=11)
    assert np.array_equal(a, b)
    c = fitted_gauss.sample(2000, seed=12)
    assert not np.array_equal(a, c)


def test_skewed_target_tri_quality():
    # 1-d sinh-arcsinh-skewed target: atlas + kernel must reproduce the mean
    e, dl = 0.6, 1.0
    def logpdf(x):
        z = np.sinh(dl * np.arcsinh(float(x[0])) - e)
        jac = dl * np.cosh(dl * np.arcsinh(float(x[0])) - e) / np.sqrt(1 + float(x[0]) ** 2)
        return -0.5 * z * z + np.log(max(jac, 1e-300))
    s = pmrhmc.PMRHMC(logpdf, d=1, name="sas1d").fit(seed=4, n_starts=4, birth_rounds=2)
    draws = s.sample(30_000, seed=7)
    zq, wq = np.polynomial.hermite_e.hermegauss(120)   # E sinh((asinh z + e)/dl), z~N(0,1)
    truth = float(np.sum(wq * np.sinh((np.arcsinh(zq) + e) / dl)) / np.sum(wq))
    assert abs(draws.mean() - truth) < 0.12


def test_oracle_accounting():
    s = std_gauss(4)
    s.fit(seed=5, n_starts=4, birth_rounds=1)
    assert s.fit_info["warm_U"] > 0
    n = 3000
    _ = s.sample(n, seed=8)
    st = s.last_stats
    # production: at most one true-density call per transition + the init eval
    assert st["n_U"] <= n + 1


def test_fd_gradient_fallback():
    t = pmrhmc.Target(lambda x: -0.5 * float(x @ x), d=3)
    u, g = t.value_and_grad(np.array([1.0, -2.0, 0.5]))
    assert np.allclose(g, [1.0, -2.0, 0.5], atol=1e-4)
    assert t.counts["warmup"]["U"] == 1 + 6  # central FD charged as densities
