"""Simulation-based calibration + synthetic-data efficiency: theta* ~ prior,
y ~ model(theta*), sample posterior(theta|y), check rank uniformity of theta*
among the draws across replications. End-to-end exactness gold standard —
no NUTS reference needed."""
import json
import numpy as np
import jax
import jax.numpy as jnp
import targets as T
from pmr_hmc import run_pmr


def make_logreg(rep_rng, p=8, n=120):
    X = rep_rng.standard_normal((n, p - 1))
    th = rep_rng.normal(0, 1.5, p)
    yv = (rep_rng.random(n) < 1 / (1 + np.exp(-(th[0] + X @ th[1:])))).astype(float)
    Xj, yj = jnp.asarray(X), jnp.asarray(yv)

    def logp(x):
        t = x[0] + Xj @ x[1:]
        return (jnp.sum(yj * jax.nn.log_sigmoid(t) + (1 - yj) * jax.nn.log_sigmoid(-t))
                - 0.5 * jnp.sum(x**2) / 1.5**2)
    return T.CountedTarget("sbc_logreg", p, logp, init_scale=1.0), th


def make_hier_binom(rep_rng, J=15):
    mu = rep_rng.normal(0, 1.5)
    lt = rep_rng.normal(0, 1.0)
    a = rep_rng.normal(mu, np.exp(lt), J)
    nj = rep_rng.integers(20, 80, J)
    yv = rep_rng.binomial(nj, 1 / (1 + np.exp(-a)))
    njj, yj = jnp.asarray(nj, dtype=jnp.float64), jnp.asarray(yv, dtype=jnp.float64)

    def logp(x):
        m, l = x[0], x[1]
        al = x[2:]
        ll = jnp.sum(yj * jax.nn.log_sigmoid(al) + (njj - yj) * jax.nn.log_sigmoid(-al))
        lp = -0.5 * jnp.sum(((al - m) / jnp.exp(l)) ** 2) - J * l
        return ll + lp - 0.5 * (m / 1.5) ** 2 - 0.5 * l**2
    return T.CountedTarget("sbc_hier_binom", J + 2, logp, init_scale=1.5, defensive=True), \
        np.concatenate([[mu, lt], a])


def make_schools(rep_rng, J=8):
    mu = rep_rng.normal(0, 5)
    lt = rep_rng.normal(0, 1.0)
    th = rep_rng.normal(mu, np.exp(lt), J)
    s = rep_rng.uniform(8, 18, J)
    yv = rep_rng.normal(th, s)
    yj, sj = jnp.asarray(yv), jnp.asarray(s)

    def logp(x):
        m, l = x[0], x[1]
        t = x[2:]
        ll = -0.5 * jnp.sum(((yj - t) / sj) ** 2)
        lp = -0.5 * jnp.sum(((t - m) / jnp.exp(l)) ** 2) - J * l
        return ll + lp - 0.5 * (m / 5) ** 2 - 0.5 * l**2
    return T.CountedTarget("sbc_schools", J + 2, logp, init_scale=2.0, defensive=True), \
        np.concatenate([[mu, lt], th])


FAMILIES = [("logreg", make_logreg), ("hier_binom", make_hier_binom), ("schools", make_schools)]
R = 30
out = {}
for name, maker in FAMILIES:
    ranks, effs = [], []
    for rep in range(R):
        rr = np.random.default_rng(1000 + rep)
        tgt, th_star = maker(rr)
        try:
            ch, info = run_pmr(tgt, seed=2000 + rep, n_samples=2500, n_tune=250,
                               n_tune2=150, birth_rounds=3, transport=True, t_defense=True)
        except Exception as e:
            print(name, rep, "FAIL", repr(e), flush=True)
            continue
        draws = ch[300::10]  # thin
        ranks.append([float(np.mean(draws[:, i] < th_star[i])) for i in range(tgt.d)])
        u = info["warm_U"] + info["prod_U"] + 2.5 * (info["warm_grad"] + info["prod_grad"])
        effs.append(u)
        if rep % 10 == 0:
            print(name, rep, flush=True)
    A = np.array(ranks)
    from scipy.stats import kstest
    ks = [float(kstest(A[:, i], "uniform").pvalue) for i in range(A.shape[1])]
    out[name] = dict(reps=len(ranks), ks_min=round(min(ks), 4),
                     ks_median=round(float(np.median(ks)), 4),
                     frac_ks_below_05=round(float(np.mean(np.asarray(ks) < 0.05)), 3),
                     med_eval_units=int(np.median(effs)))
    print(name, out[name], flush=True)
    json.dump(out, open("sbc_results.json", "w"), indent=2)
print(json.dumps(out, indent=2))
