"""Mega battery: 102 parametric targets x 4 seeds x {pmr, nuts}, sharded for
parallelism. Usage: mega_bench.py <shard> <nshards>. Truthless targets are
judged against the seed-1 NUTS run's moments."""
import json, os, sys, time
import numpy as np
import targets as T
from pmr_hmc import run_pmr
from baselines import run_nuts

SHARD, NSHARDS = int(sys.argv[1]), int(sys.argv[2])
OUT = f"mega_results_shard{SHARD}.json"


def essmin(c):
    import arviz as az
    return float(np.min(az.ess(az.convert_to_dataset(np.asarray(c)[None])).to_array().values))


import glob


def dump_atomic(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)


rows = json.load(open(OUT)) if os.path.exists(OUT) else []
done = set()
_ref_cache = {}
for f_ in glob.glob("mega_results_shard*.json"):
    try:
        loaded_ = json.load(open(f_))
    except Exception:
        continue  # another shard may be mid-write at our startup instant
    for r_ in loaded_:
        done.add((r_["target"], r_["sampler"], r_["seed"]))
        if "_refm" in r_:
            _ref_cache[r_["target"]] = (np.asarray(r_["_refm"]), np.asarray(r_["_refv"]))
suite = T.mega_suite()
_tf = os.environ.get("MEGA_TARGETS_FILE")
if _tf:
    _want = {ln.strip() for ln in open(_tf) if ln.strip()}
    _missing = _want - {n for n, _ in suite}
    if _missing:
        sys.exit(f"MEGA_TARGETS_FILE names not in suite: {sorted(_missing)}")
    suite = [(n, f) for n, f in suite if n in _want]
for idx, (name, factory) in enumerate(suite):
    if idx % NSHARDS != SHARD:
        continue
    ref = _ref_cache.get(name)
    for seed in (1, 2, 3, 4):
        for sampler in ("nuts", "pmr"):
            if (name, sampler, seed) in done:
                continue
            r = dict(target=name, sampler=sampler, seed=seed)
            try:
                tgt = factory()
                t0 = time.perf_counter()
                if sampler == "nuts":
                    ch, info = run_nuts(tgt, seed=seed, num_warmup=700, num_samples=2200)
                    burn = 220
                else:
                    ch, info = run_pmr(tgt, seed=seed, n_samples=6000, n_tune=300,
                                       n_tune2=200, birth_rounds=4, transport=True,
                                       t_defense=True)
                    burn = 600
                u = info["warm_U"] + info["prod_U"] + 2.5 * (info["warm_grad"] + info["prod_grad"])
                cc = ch[burn:]
                e = essmin(cc)
                r.update(ess_ku=round(1000 * e / u, 3), t=round(time.perf_counter() - t0, 1),
                         d=tgt.d)
                tm, tv = tgt.truth_mean, tgt.truth_var
                if tm is None and sampler == "nuts" and seed == 1:
                    ref = (cc.mean(0), cc.var(0))
                    r["_refm"], r["_refv"] = ref[0].tolist(), ref[1].tolist()
                if tm is None and ref is not None:
                    tm, tv = ref
                if tm is not None:
                    r["me"] = round(float(np.max(np.abs(cc.mean(0) - tm) / np.sqrt(np.maximum(tv, 1e-12)))), 3)
                    r["ve"] = round(float(np.max(np.abs(np.log(np.maximum(cc.var(0), 1e-300) / np.maximum(tv, 1e-300))))), 3)
            except Exception as ex:
                r["error"] = repr(ex)[:100]
            rows.append(r)
            dump_atomic(rows, OUT)
    print(f"shard{SHARD}: {name} done", flush=True)
print(f"shard{SHARD} COMPLETE", flush=True)
