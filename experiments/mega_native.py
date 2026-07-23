"""Wall-clock pass over the mega-suite subset with native C targets:
C-kernel production ESS/s, END-TO-END time-to-1000-ESS (PMR warmup included),
vs JIT NUTS end-to-end. 4 seeds."""
import json, os, time
import numpy as np
import targets as T
from pmr_hmc import run_pmr
from baselines import run_nuts
from native import run_native, _py_native_U


def essmin(c):
    import arviz as az
    return float(np.min(az.ess(az.convert_to_dataset(np.asarray(c)[None])).to_array().values))


suite = [(n, f) for n, f in T.mega_suite()]
rows = json.load(open("mega_native.json")) if os.path.exists("mega_native.json") else []
done = {(r["target"], r["seed"]) for r in rows}
for name, factory in suite:
    probe = factory()
    if not hasattr(probe, "native_spec"):
        continue
    for seed in (1, 2, 3, 4):
        if (name, seed) in done: continue
        r = dict(target=name, seed=seed, d=probe.d)
        try:
            tgt = factory()
            t0 = time.perf_counter()
            ch, info, smp = run_pmr(tgt, seed=seed, n_samples=1500, n_tune=300,
                                    n_tune2=200, birth_rounds=4, transport=True,
                                    t_defense=True, return_sampler=True)
            warm_wall = time.perf_counter() - t0
            # parity check: SAME point for both formulas
            rng = np.random.default_rng(0)
            _pts = [smp.mix.sample(rng, 1)[0] for _ in range(3)]
            par = max(abs(_py_native_U(tgt, p_) - tgt.U(p_)) for p_ in _pts)
            t1 = time.perf_counter()
            cch, cst = run_native(tgt, smp, 30000, seed + 500)
            c_wall = time.perf_counter() - t1
            cc = cch[1500:]
            e = essmin(cc[::3])
            ess_per_s = e / c_wall
            tt1000 = warm_wall + 1000.0 / max(ess_per_s, 1e-9)
            r.update(pmr_warm_s=round(warm_wall, 2), c_ess_per_s=round(ess_per_s, 1),
                     c_us_per_draw=round(c_wall * 1e6 / 30000, 2),
                     pmr_t1000=round(tt1000, 2), c_acc=round(cst["acc_local"], 3),
                     parity=float(par))
            tgt2 = factory()
            t2 = time.perf_counter()
            nch, ninfo = run_nuts(tgt2, seed=seed, num_warmup=700, num_samples=2500)
            n_wall = time.perf_counter() - t2
            ne = essmin(nch[250:])
            n_rate = ne / n_wall
            r.update(nuts_ess_per_s=round(n_rate, 1),
                     nuts_t1000=round(700 / 2500 * n_wall + 1000.0 / max(n_rate, 1e-9), 2))
        except Exception as ex:
            r["error"] = repr(ex)[:100]
        rows.append(r)
        json.dump(rows, open("mega_native.json", "w"))
    print(name, "done", flush=True)
print("COMPLETE", flush=True)
