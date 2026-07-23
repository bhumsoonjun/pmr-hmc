"""SOTA baseline battery: dense-mass NUTS (numpyro), ChEES-HMC and MCLMC
(blackjax) vs pmr_rta, oracle-eval framing (grad=2.5 units) + quality."""
import json, time
import numpy as np
import jax, jax.numpy as jnp

import targets as T
import pdb_targets as PT
from pmr_hmc import run_pmr
from baselines import run_nuts


def essmin(c):
    import arviz as az
    return float(np.min(az.ess(az.convert_to_dataset(np.asarray(c)[None])).to_array().values))


def quality(tgt, chain, ref=None):
    if tgt.truth_mean is not None:
        tv = tgt.truth_var
        return (round(float(np.max(np.abs(chain.mean(0) - tgt.truth_mean) / np.sqrt(tv))), 3),
                round(float(np.max(np.abs(np.log(chain.var(0) / tv)))), 3))
    if ref is not None:
        c = tgt.constrained(chain)
        me = max(abs(c[k].mean() - ref[k].mean()) / (ref[k].std() + 1e-12) for k in ref)
        ve = max(abs(np.log((c[k].var() + 1e-12) / (ref[k].var() + 1e-12))) for k in ref)
        return round(float(me), 3), round(float(ve), 3)
    return None, None


def run_nuts_dense(tgt, seed):
    from numpyro.infer import MCMC, NUTS
    kern = NUTS(potential_fn=tgt.U_jax, dense_mass=True)
    m = MCMC(kern, num_warmup=800, num_samples=2500, progress_bar=False)
    rng = np.random.default_rng(seed)
    init = jnp.asarray(rng.standard_normal(tgt.d) * tgt.init_scale)
    m.warmup(jax.random.PRNGKey(seed), init_params=init, extra_fields=("num_steps",), collect_warmup=True)
    ws = int(np.sum(np.asarray(m.get_extra_fields()["num_steps"])))
    m.run(jax.random.PRNGKey(seed + 1), extra_fields=("num_steps",))
    ps = int(np.sum(np.asarray(m.get_extra_fields()["num_steps"])))
    ch = np.asarray(m.get_samples())
    units = 2500 + 2.5 * (ws + ps)
    return ch, units


def run_chees(tgt, seed, num_chains=16, n_warm=500, n_samp=1500):
    import blackjax, optax
    logdens = lambda x: -tgt.U_jax(x)
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    init_pos = jax.random.normal(k1, (num_chains, tgt.d)) * tgt.init_scale
    warm = blackjax.chees_adaptation(logdens, num_chains)
    (last, params), info_w = warm.run(k2, init_pos, 0.1, optax.adam(0.02), n_warm)
    steps_w = int(np.sum(np.asarray(info_w.info.num_integration_steps)))
    kern = blackjax.dynamic_hmc(logdens, **params).step

    def one(carry, k):
        st, = carry
        keys = jax.random.split(k, num_chains)
        st, inf = jax.vmap(kern)(keys, st)
        return (st,), (st.position, inf.num_integration_steps)

    (_,), (pos, nsteps) = jax.lax.scan(one, (last,), jax.random.split(k3, n_samp))
    ch = np.asarray(pos[:, 0, :])  # chain 0 for ESS; evals counted over all
    units = 2.5 * (steps_w + int(np.sum(np.asarray(nsteps)))) / num_chains  # per-chain units
    return ch, units


def run_mclmc(tgt, seed, n=8000):
    import blackjax
    from blackjax.mcmc import mclmc
    from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
    logdens = lambda x: -tgt.U_jax(x)
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    x0 = jax.random.normal(k1, (tgt.d,)) * tgt.init_scale
    st = mclmc.init(x0, logdens, k2)
    kern = mclmc.build_kernel()
    st2, prm, _ = mclmc_find_L_and_step_size(
        mclmc_kernel=kern, num_steps=2000, state=st, rng_key=k3, logdensity_fn=logdens)

    def one(s, k):
        s2, _ = kern(rng_key=k, state=s, logdensity_fn=logdens,
                     inverse_mass_matrix=prm.inverse_mass_matrix, L=prm.L,
                     step_size=prm.step_size)
        return s2, s2.position

    _, pos = jax.lax.scan(one, st2, jax.random.split(jax.random.PRNGKey(seed + 9), n))
    ch = np.asarray(pos)
    units = 2.5 * (n + 2000)
    return ch, units


CASES = [
    ("gauss_ill_1e4", lambda: T.gauss_ill(50, 1e4), None),
    ("mixture2", lambda: T.mixture2(10, 6.0), None),
    ("banana20", lambda: T.banana(20), None),
    ("funnel10", lambda: T.funnel(10), None),
    ("ring20", lambda: T.ring(20), None),
    ("student_t20", lambda: T.student_t(20, 4.0), None),
    ("logreg26", T.logreg, None),
    ("vilya_bcf37", T.vilya_bcf, None),
    ("pdb_es_nc", PT.es_nc, "ref"),
    ("pdb_nes1992", lambda: PT.nes(1992), "ref"),
]
import os
METHODS = ([("mclmc", run_mclmc)] if os.environ.get("MCLMC_ONLY") else [("nuts_dense", run_nuts_dense), ("chees", run_chees), ("mclmc", run_mclmc)])
rows = []
for name, factory, refflag in CASES:
    t0f = factory()
    ref = PT.reference(t0f.ref_name) if refflag else None
    for seed in (1, 2, 3, 4):
        r = dict(target=name, seed=seed)
        try:
            if os.environ.get("MCLMC_ONLY"):
                raise RuntimeError("skip pmr")
            tgt = factory()
            ch, info = run_pmr(tgt, seed=seed, n_samples=8000, transport=True, t_defense=True)
            u = info["warm_U"] + info["prod_U"] + 2.5 * (info["warm_grad"] + info["prod_grad"])
            e = essmin(ch[800:])
            me, ve = quality(tgt, ch[800:], ref)
            r["pmr"] = (round(1000 * e / u, 2), me, ve)
        except Exception as ex:
            r["pmr"] = ("ERR", repr(ex)[:60], None)
        for mname, fn in METHODS:
            try:
                tgt = factory()
                ch, units = fn(tgt, seed)
                burn = min(300, len(ch) // 10)
                e = essmin(ch[burn:])
                me, ve = quality(tgt, ch[burn:], ref)
                r[mname] = (round(1000 * e / units, 2), me, ve)
            except Exception as ex:
                r[mname] = ("ERR", repr(ex)[:80], None)
        rows.append(r)
        print(r, flush=True)
        json.dump(rows, open(os.environ.get("SOTA_OUT", "sota_results.json"), "w"))

by = {}
for r in rows:
    by.setdefault(r["target"], []).append(r)
lines = ["# SOTA battery (ess/kunit, mean/var err; medians over 4 seeds)", "",
         "| target | pmr | nuts_dense | chees | mclmc |", "|---|---|---|---|---|"]
for n, rs in by.items():
    def med(m):
        vals = [x[m][0] for x in rs if isinstance(x.get(m, ("ERR",))[0], (int, float))]
        if not vals:
            return "ERR"
        q = [x[m] for x in rs if isinstance(x[m][0], (int, float))][0]
        return f"{float(np.median(vals)):.1f} ({q[1]}/{q[2]})"
    lines.append(f"| {n} | {med('pmr')} | {med('nuts_dense')} | {med('chees')} | {med('mclmc')} |")
open("SOTA.md", "w").write("\n".join(lines))
print("\n".join(lines))
