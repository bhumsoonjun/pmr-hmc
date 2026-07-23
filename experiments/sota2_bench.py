"""Learned-transport SOTA battery: TESS, flow-IMH, flowMC-style, NeuTra,
Pathfinder-init dense NUTS over the SOTA target set, 4 seeds, sharded.
Usage: sota2_bench.py <shard> <nshards>."""
import json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
import targets as T
import pdb_targets as PT
from flows import run_tess, run_flow_imh, run_flowmc, run_neutra

SHARD, NSH = int(sys.argv[1]), int(sys.argv[2])
OUT = f"sota2_shard{SHARD}.json"


def essmin(c):
    import arviz as az
    return float(np.min(az.ess(az.convert_to_dataset(np.asarray(c)[None])).to_array().values))


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
    return ch, dict(warm_time=0.0, prod_time=0.0)


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
METHODS = [("tess", run_tess), ("flow_imh", run_flow_imh),
           ("flowmc", run_flowmc), ("neutra", run_neutra),
           ("pf_dense", run_pf_dense)]

rows = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {(r["target"], r["method"], r["seed"]) for r in rows}
idx = 0
for name, factory, refflag in CASES:
    ref = PT.reference(factory().ref_name) if refflag else None
    for mname, fn in METHODS:
        idx += 1
        if idx % NSH != SHARD:
            continue
        for seed in (1, 2, 3, 4):
            if (name, mname, seed) in done: continue
            r = dict(target=name, method=mname, seed=seed)
            try:
                tgt = factory()
                t0 = time.perf_counter()
                ch, info = fn(tgt, seed)
                wall = time.perf_counter() - t0
                burn = max(200, len(ch) // 10)
                cc = ch[burn:]
                u = tgt.counts["warmup"]["U"] + tgt.counts["production"]["U"] \
                    + 2.5 * (tgt.counts["warmup"]["grad"] + tgt.counts["production"]["grad"])
                e = essmin(cc)
                r.update(ess_ku=round(1000 * e / max(u, 1), 3),
                         ess=round(e, 1), wall_s=round(wall, 1),
                         ess_per_s=round(e / wall, 2))
                tm, tv = tgt.truth_mean, tgt.truth_var
                if tm is None and ref is not None:
                    c = tgt.constrained(cc)
                    r["me"] = round(float(max(abs(c[k].mean() - ref[k].mean()) / (ref[k].std() + 1e-12) for k in ref)), 3)
                    r["ve"] = round(float(max(abs(np.log((c[k].var() + 1e-12) / (ref[k].var() + 1e-12))) for k in ref)), 3)
                elif tm is not None:
                    r["me"] = round(float(np.max(np.abs(cc.mean(0) - tm) / np.sqrt(tv))), 3)
                    r["ve"] = round(float(np.max(np.abs(np.log(np.maximum(cc.var(0), 1e-300) / tv)))), 3)
                if name == "mixture2":
                    s_ = np.sign(cc[:, 0]); r["switches"] = int(np.sum(s_[1:] != s_[:-1]))
            except Exception as ex:
                r["error"] = repr(ex)[:100]
            rows.append(r)
            json.dump(rows, open(OUT, "w"))
        print(f"s{SHARD}: {name}/{mname}", flush=True)
print(f"s{SHARD} COMPLETE", flush=True)
