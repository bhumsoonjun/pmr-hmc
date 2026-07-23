"""Whale-shop Vilya wall-clock benchmark THROUGH the pmrhmc library.

The production question: how fast can the DataGlass Vilya joint BCF hurdle-NB
posterior be sampled for whale shops? Scales C (campaign count) through
realistic tiers up to whale size (C=120, d~490). PMR-HMC runs via
pmrhmc.PMRHMC with a JAX-jitted log-density callback (ONE callback per
transition); the baseline is jitted NumPyro NUTS on the identical potential.
Wall-clock is the metric (time to fit + time per 1000 min-ESS), so run this
on an OTHERWISE IDLE machine.
Usage: vilya_whale.py [C ...]   (default tiers 30 60 120)
"""
import json
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import targets as T
import pmrhmc


def essmin(c):
    import arviz as az
    return float(np.min(az.ess(az.convert_to_dataset(np.asarray(c)[None])).to_array().values))


def bench_pmr(tgt, n_prod=20000, chains=4, cores=4, seed=0):
    jU = jax.jit(tgt.U_jax)
    jG = jax.jit(jax.grad(tgt.U_jax))
    jU(jnp.zeros(tgt.d)).block_until_ready()   # compile outside the clock
    jG(jnp.zeros(tgt.d)).block_until_ready()
    logpdf = lambda x: -float(jU(jnp.asarray(x)))
    grad_logpdf = lambda x: -np.asarray(jG(jnp.asarray(x)), dtype=float)
    s = pmrhmc.PMRHMC(logpdf, d=tgt.d, grad_logpdf=grad_logpdf,
                      init_scale=tgt.init_scale, name=tgt.name)
    t0 = time.perf_counter()
    s.fit(seed=seed, transport=True, t_defense=True)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    chs = s.sample(n_prod, seed=seed + 1, chains=chains, cores=cores, burn=n_prod // 10)
    t_smp = time.perf_counter() - t0
    e = float(np.sum([essmin(c) for c in chs]))
    acc = float(np.mean([st["acc_local"] for st in s.last_stats]))
    return dict(fit_s=round(t_fit, 1), sample_s=round(t_smp, 1),
                ess=round(e, 1), s_per_kess=round(t_smp / max(e, 1e-9) * 1000, 2),
                acc_local=round(acc, 3), draws=int(chains * n_prod),
                us_per_draw=round(1e6 * t_smp / (chains * n_prod), 2),
                moments=[chs.reshape(-1, tgt.d).mean(0)[:3].tolist(),
                         chs.reshape(-1, tgt.d).var(0)[:3].tolist()])


def bench_nuts(tgt, n_warm=700, n_prod=1500, seed=0):
    from numpyro.infer import MCMC, NUTS
    kern = NUTS(potential_fn=tgt.U_jax)
    m = MCMC(kern, num_warmup=n_warm, num_samples=n_prod, progress_bar=False)
    rng = np.random.default_rng(seed)
    init = jnp.asarray(rng.standard_normal(tgt.d) * tgt.init_scale)
    t0 = time.perf_counter()
    m.warmup(jax.random.PRNGKey(seed), init_params=init)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.run(jax.random.PRNGKey(seed + 1))
    t_smp = time.perf_counter() - t0
    ch = np.asarray(m.get_samples())
    e = essmin(ch)
    return dict(fit_s=round(t_fit, 1), sample_s=round(t_smp, 1), ess=round(e, 1),
                s_per_kess=round(t_smp / max(e, 1e-9) * 1000, 2), draws=n_prod,
                moments=[ch.mean(0)[:3].tolist(), ch.var(0)[:3].tolist()])


if __name__ == "__main__":
    tiers = [int(a) for a in sys.argv[1:]] or [30, 60, 120]
    out = {}
    for C in tiers:
        tgt = T.vilya_bcf(C=C, n_first=60 * C, n_pos=40 * C)
        print(f"=== C={C} d={tgt.d} ===", flush=True)
        try:
            r_p = bench_pmr(tgt)
            print(" pmr :", {k: v for k, v in r_p.items() if k != "moments"}, flush=True)
        except Exception as ex:
            r_p = dict(error=repr(ex)[:200])
            print(" pmr FAILED:", r_p["error"], flush=True)
        tgt2 = T.vilya_bcf(C=C, n_first=60 * C, n_pos=40 * C)
        r_n = bench_nuts(tgt2)
        print(" nuts:", {k: v for k, v in r_n.items() if k != "moments"}, flush=True)
        out[C] = dict(d=tgt.d, pmr=r_p, nuts=r_n)
        json.dump(out, open("vilya_whale.json", "w"), indent=1)
    print("done -> vilya_whale.json")
