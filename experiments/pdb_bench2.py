"""Expanded posteriordb battery: 37 posteriors x 8 seeds x
{pmr, diagonal-NUTS, dense-NUTS}, gold-judged, with cross-seed rank-Rhat."""
import json, os, time
import numpy as np
import pdb_targets as PT
from pmr_hmc import run_pmr
from baselines import run_nuts


def essmin(c):
    import arviz as az
    return float(np.min(az.ess(az.convert_to_dataset(np.asarray(c)[None])).to_array().values))


def gold_err(dc, ref):
    me = ve = 0.0
    for k, rv in ref.items():
        if k not in dc: continue
        dv = dc[k]
        me = max(me, abs(dv.mean() - rv.mean()) / (rv.std() + 1e-12))
        ve = max(ve, abs(np.log((dv.var() + 1e-12) / (rv.var() + 1e-12))))
    return round(me, 3), round(ve, 3)


def run_dense(tgt, seed):
    import jax, jax.numpy as jnp
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


path = "pdb2_results.json"
rows = json.load(open(path)) if os.path.exists(path) else []
done = {(r["target"], r["sampler"], r["seed"]) for r in rows}
draws_store = {}
for factory in PT.pdb_suite_v3():
    tgt0 = factory()
    name, ref = tgt0.name, PT.reference(tgt0.ref_name)
    for seed in range(1, 9):
        for sampler in ("pmr", "nuts", "dense"):
            if (name, sampler, seed) in done: continue
            r = dict(target=name, sampler=sampler, seed=seed)
            try:
                tgt = factory()
                if sampler == "pmr":
                    ch, info = run_pmr(tgt, seed=seed, n_samples=8000, transport=True, t_defense=True)
                    u = info["warm_U"] + info["prod_U"] + 2.5 * (info["warm_grad"] + info["prod_grad"])
                    burn = 800
                elif sampler == "nuts":
                    ch, info = run_nuts(tgt, seed=seed, num_warmup=800, num_samples=2500)
                    u = info["warm_U"] + info["prod_U"] + 2.5 * (info["warm_grad"] + info["prod_grad"])
                    burn = 250
                else:
                    ch, u = run_dense(tgt, seed)
                    burn = 250
                cc = ch[burn:]
                e = essmin(cc)
                me, ve = gold_err(tgt.constrained(cc), ref)
                r.update(ess_ku=round(1000 * e / u, 3), gold_mean=me, gold_var=ve)
                draws_store.setdefault((name, sampler), []).append(cc[:: max(1, len(cc) // 400)][:400])
            except Exception as ex:
                r["error"] = repr(ex)[:120]
            rows.append(r)
            json.dump(rows, open(path, "w"))
        print(name, seed, flush=True)
    # cross-seed rank-Rhat per sampler
    import arviz as az
    for sampler in ("pmr", "nuts", "dense"):
        key = (name, sampler)
        if key in draws_store and len(draws_store[key]) >= 4:
            n = min(len(c) for c in draws_store[key])
            stack = np.stack([c[:n] for c in draws_store[key]])
            rh = float(np.max(az.rhat(az.convert_to_dataset(stack)).to_array().values))
            rows.append(dict(target=name, sampler=sampler, seed=0, rhat_max=round(rh, 4)))
            json.dump(rows, open(path, "w"))
            del draws_store[key]

# summary
by = {}
for r in rows:
    if "ess_ku" in r:
        by.setdefault((r["target"], r["sampler"]), []).append(r)
rh = {(r["target"], r["sampler"]): r["rhat_max"] for r in rows if "rhat_max" in r}
lines = ["# posteriordb v2 (37 posteriors, 8 seeds, gold-judged, cross-seed rank-Rhat)", "",
         "| posterior | pmr | nuts | dense | pmr/nuts | pmr/dense | pmr err | pmr Rhat |",
         "|---|---|---|---|---|---|---|---|"]
targets = sorted({t for t, _ in by})
for t in targets:
    m = lambda s, k: float(np.median([x[k] for x in by.get((t, s), [{k: np.nan}])]))
    p, n_, d_ = m("pmr", "ess_ku"), m("nuts", "ess_ku"), m("dense", "ess_ku")
    lines.append(f"| {t} | {p:.1f} | {n_:.1f} | {d_:.1f} | {p/max(n_,1e-9):.1f}x | "
                 f"{p/max(d_,1e-9):.1f}x | {m('pmr','gold_mean'):.2f}/{m('pmr','gold_var'):.2f} | "
                 f"{rh.get((t,'pmr'), float('nan')):.3f} |")
open("PDB2.md", "w").write("\n".join(lines))
print("done")
