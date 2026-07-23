"""PMR rerun over the 37 posteriordb posteriors x 8 seeds (post-TriComp code),
sharded by target. Baseline (nuts/dense) cells stay cached in pdb2_results.json;
this writes fresh pmr cells to pdb2r_shard<N>.json with the same judging as
pdb_bench2 (gold reference on constrained draws, cross-seed rank-Rhat).
Usage: pdb_rerun.py <shard> <nshards>"""
import glob
import json
import os
import sys
import time

import numpy as np

import pdb_targets as PT
from pmr_hmc import run_pmr

SHARD, NSH = int(sys.argv[1]), int(sys.argv[2])
OUT = f"pdb2r_shard{SHARD}.json"


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


rows = json.load(open(OUT)) if os.path.exists(OUT) else []
done = set()
rhat_done = set()
for f_ in glob.glob("pdb2r_shard*.json"):
    try:
        loaded_ = json.load(open(f_))
    except Exception:
        continue
    for r_ in loaded_:
        done.add((r_["target"], r_["seed"]))
        if "rhat_max" in r_:
            rhat_done.add(r_["target"])

for ti, factory in enumerate(PT.pdb_suite_v3()):
    if ti % NSH != SHARD:
        continue
    tgt0 = factory()
    name, ref = tgt0.name, PT.reference(tgt0.ref_name)
    del tgt0
    store = []
    for seed in range(1, 9):
        if (name, seed) in done:
            continue
        r = dict(target=name, sampler="pmr", seed=seed)
        try:
            tgt = factory()
            t0 = time.perf_counter()
            ch, info = run_pmr(tgt, seed=seed, n_samples=8000, transport=True, t_defense=True)
            u = info["warm_U"] + info["prod_U"] + 2.5 * (info["warm_grad"] + info["prod_grad"])
            cc = ch[800:]
            e = essmin(cc)
            me, ve = gold_err(tgt.constrained(cc), ref)
            r.update(ess_ku=round(1000 * e / u, 3), gold_mean=me, gold_var=ve,
                     t=round(time.perf_counter() - t0, 1))
            store.append(cc[:: max(1, len(cc) // 400)][:400])
        except Exception as ex:
            r["error"] = repr(ex)[:120]
        rows.append(r)
        dump_atomic(rows, OUT)
        print(f"s{SHARD}: {name} seed{seed}", flush=True)
    if len(store) >= 4 and name not in rhat_done:
        import arviz as az
        n = min(len(c) for c in store)
        stack = np.stack([c[:n] for c in store])
        rh = float(np.max(az.rhat(az.convert_to_dataset(stack)).to_array().values))
        rows.append(dict(target=name, sampler="pmr", seed=0, rhat_max=round(rh, 4)))
        dump_atomic(rows, OUT)
print(f"s{SHARD} COMPLETE", flush=True)
