"""Full SOTA panel over either suite: dense-mass NUTS, Pathfinder-init dense
NUTS, ChEES, MCLMC, TESS, flow-IMH, flowMC-style, NeuTra. 4 seeds, sharded by
(target x method) cell, cell-cached, judged exactly like the host suite (mega:
truth or the cached NUTS-seed1 reference moments; pdb: posteriordb gold draws
on constrained parameters).
Runners are copied from pdb_bench2/sota_bench/sota2_bench because those scripts
have no __main__ guard (importing them would launch a benchmark).
Usage: panel_bench.py <suite:mega|pdb> <shard> <nshards>   [PANEL_SMOKE=1]"""
import glob
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from flows import run_tess, run_flow_imh, run_flowmc, run_neutra

SUITE, SHARD, NSH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
SMOKE = os.environ.get("PANEL_SMOKE") == "1"
OUT = f"panel_smoke_{SUITE}.json" if SMOKE else f"{SUITE}_panel_shard{SHARD}.json"


def dump_atomic(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)


def essmin(c):
    import arviz as az
    return float(np.min(az.ess(az.convert_to_dataset(np.asarray(c)[None])).to_array().values))


def gold_err(dc, ref):
    me = ve = 0.0
    for k, rv in ref.items():
        if k not in dc:
            continue
        dv = dc[k]
        me = max(me, abs(dv.mean() - rv.mean()) / (rv.std() + 1e-12))
        ve = max(ve, abs(np.log((dv.var() + 1e-12) / (rv.var() + 1e-12))))
    return round(me, 3), round(ve, 3)


def run_dense(tgt, seed):
    from numpyro.infer import MCMC, NUTS
    kern = NUTS(potential_fn=tgt.U_jax, dense_mass=True)
    m = MCMC(kern, num_warmup=800, num_samples=2500, progress_bar=False)
    rng = np.random.default_rng(seed)
    init = jnp.asarray(rng.standard_normal(tgt.d) * tgt.init_scale)
    m.warmup(jax.random.PRNGKey(seed), init_params=init, extra_fields=("num_steps",),
             collect_warmup=True)
    ws = int(np.sum(np.asarray(m.get_extra_fields()["num_steps"])))
    m.run(jax.random.PRNGKey(seed + 1), extra_fields=("num_steps",))
    ps = int(np.sum(np.asarray(m.get_extra_fields()["num_steps"])))
    ch = np.asarray(m.get_samples())
    return ch, 2500 + 2.5 * (ws + ps)


def run_pf_dense(tgt, seed):
    from scipy.optimize import minimize
    from numpyro.infer import MCMC, NUTS
    rng = np.random.default_rng(seed)
    res = minimize(lambda x: tgt.value_and_grad(x), rng.standard_normal(tgt.d) * tgt.init_scale,
                   jac=True, method="L-BFGS-B", options={"maxiter": 200})
    kern = NUTS(potential_fn=tgt.U_jax, dense_mass=True)
    m = MCMC(kern, num_warmup=700, num_samples=2500, progress_bar=False)
    m.warmup(jax.random.PRNGKey(seed), init_params=jnp.asarray(res.x),
             extra_fields=("num_steps",), collect_warmup=True)
    ws = int(np.sum(np.asarray(m.get_extra_fields()["num_steps"])))
    m.run(jax.random.PRNGKey(seed + 1), extra_fields=("num_steps",))
    ps = int(np.sum(np.asarray(m.get_extra_fields()["num_steps"])))
    ch = np.asarray(m.get_samples())
    tgt.counts["production"]["grad"] += ws + ps
    tgt.counts["production"]["U"] += 2500
    return ch, None  # units from tgt.counts (includes the L-BFGS warm-up)


def run_chees(tgt, seed, num_chains=16, n_warm=500, n_samp=1500):
    import blackjax
    import optax
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
    ch = np.asarray(pos[:, 0, :])  # chain 0 for ESS; evals counted per-chain
    units = 2.5 * (steps_w + int(np.sum(np.asarray(nsteps)))) / num_chains
    return ch, units


def run_mclmc(tgt, seed, n=8000):
    import blackjax  # noqa: F401
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
    return np.asarray(pos), 2.5 * (n + 2000)


METHODS = [("dense", run_dense), ("pf_dense", run_pf_dense), ("chees", run_chees),
           ("mclmc", run_mclmc), ("tess", run_tess), ("flow_imh", run_flow_imh),
           ("flowmc", run_flowmc), ("neutra", run_neutra)]

if SUITE == "mega":
    import targets as T
    suite = T.mega_suite()
    _ref = {}
    for f_ in glob.glob("mega_results_shard*.json"):
        try:
            loaded_ = json.load(open(f_))
        except Exception:
            continue
        for r_ in loaded_:
            if "_refm" in r_:
                _ref[r_["target"]] = (np.asarray(r_["_refm"]), np.asarray(r_["_refv"]))
elif SUITE == "pdb":
    import pdb_targets as PT
    suite = []
    _refname = {}
    for fac_ in PT.pdb_suite_v3():
        t0_ = fac_()
        suite.append((t0_.name, fac_))
        _refname[t0_.name] = t0_.ref_name
        del t0_
else:
    sys.exit(f"unknown suite {SUITE}")

tf = os.environ.get("PANEL_TARGETS_FILE")
if tf:
    want = {ln.strip() for ln in open(tf) if ln.strip()}
    missing = want - {n for n, _ in suite}
    if missing:
        sys.exit(f"PANEL_TARGETS_FILE names not in suite: {sorted(missing)}")
    suite = [(n, f) for n, f in suite if n in want]

rows = json.load(open(OUT)) if os.path.exists(OUT) else []
done = set()
for f_ in glob.glob(f"{SUITE}_panel_shard*.json"):
    try:
        loaded_ = json.load(open(f_))
    except Exception:
        continue
    for r_ in loaded_:
        done.add((r_["target"], r_["method"], r_["seed"]))

if SMOKE:
    suite = suite[:1]
    METHODS = [m for m in METHODS if m[0] in ("dense", "tess")]

idx = 0
for name, factory in suite:
    ref = None
    if SUITE == "pdb":
        ref = None  # loaded lazily below only if a cell of this target runs here
    for mname, fn in METHODS:
        idx += 1
        if not SMOKE and idx % NSH != SHARD:
            continue
        for seed in ((1,) if SMOKE else (1, 2, 3, 4)):
            if (name, mname, seed) in done:
                continue
            if SUITE == "pdb" and ref is None:
                ref = PT.reference(_refname[name])
            r = dict(target=name, method=mname, seed=seed)
            try:
                tgt = factory()
                t0 = time.perf_counter()
                ch, second = fn(tgt, seed)
                wall = time.perf_counter() - t0
                units = second if isinstance(second, (int, float)) else None
                if units is None:
                    c = tgt.counts
                    units = c["warmup"]["U"] + c["production"]["U"] \
                        + 2.5 * (c["warmup"]["grad"] + c["production"]["grad"])
                burn = max(200, len(ch) // 10)
                cc = ch[burn:]
                e = essmin(cc)
                r.update(ess_ku=round(1000 * e / max(units, 1), 3), ess=round(e, 1),
                         wall_s=round(wall, 1), d=tgt.d)
                if SUITE == "pdb":
                    me, ve = gold_err(tgt.constrained(cc), ref)
                    r["me"], r["ve"] = me, ve
                else:
                    tm, tv = tgt.truth_mean, tgt.truth_var
                    if tm is None and name in _ref:
                        tm, tv = _ref[name]
                    if tm is not None:
                        r["me"] = round(float(np.max(np.abs(cc.mean(0) - tm)
                                                     / np.sqrt(np.maximum(tv, 1e-12)))), 3)
                        r["ve"] = round(float(np.max(np.abs(np.log(
                            np.maximum(cc.var(0), 1e-300) / np.maximum(tv, 1e-300))))), 3)
            except Exception as ex:
                r["error"] = repr(ex)[:120]
            rows.append(r)
            dump_atomic(rows, OUT)
        print(f"s{SHARD}: {name}/{mname}", flush=True)
print(f"s{SHARD} COMPLETE", flush=True)
