"""33-posterior suite: NUTS vs pmr_rta vs pmr_pc_rta, oracle-eval framing.

Resumable: results appended to suite_results.json after every target; rerun
skips completed (target, sampler) cells. Writes SUITE.md summary + win/loss
tally at the end.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import targets as T
from baselines import run_nuts
from pmr_hmc import run_pmr

PMR_KW = dict(n_samples=8000, n_tune=400, n_tune2=300, birth_rounds=5,
              transport=True, t_defense=True)  # full warmup budget: the lighter
# settings flipped ring/breast from wins to losses (detection needs coverage)
SPECIAL = {
    "gauss_ill_1e6": dict(lbfgs_max_iter=20000),
}


def ess_stats(chain):
    import arviz as az

    ds = az.convert_to_dataset(np.asarray(chain)[None])
    ess = az.ess(ds).to_array().values.ravel()
    return float(np.min(ess)), float(np.median(ess))


def summarize(name, sampler, chain, info, tgt, ref=None):
    burn = min(500, len(chain) // 10)
    ch = chain[burn:]
    e_min, e_med = ess_stats(ch)
    units = info["warm_U"] + info["prod_U"] + 2.5 * (info["warm_grad"] + info["prod_grad"])
    row = dict(target=name, sampler=sampler, d=tgt.d,
               ess_min=round(e_min, 1), ess_med=round(e_med, 1),
               eval_units=int(units),
               ess_per_kunit=round(1000.0 * e_min / units, 3),
               acc_local=round(info.get("acc_local", float("nan")), 3),
               charts=str(info.get("chart_kinds", "")),
               time_s=round(info["warm_time"] + info["prod_time"], 1))
    tm, tv = tgt.truth_mean, tgt.truth_var
    src = "truth"
    if tm is None and ref is not None:
        tm, tv = ref
        src = "nuts_ref"
    if tm is not None:
        m, v = ch.mean(0), ch.var(0)
        row["mean_err_sd"] = round(float(np.max(np.abs(m - tm) / np.sqrt(np.maximum(tv, 1e-12)))), 3)
        row["var_err_log"] = round(float(np.max(np.abs(np.log(np.maximum(v, 1e-300) / np.maximum(tv, 1e-300))))), 3)
        row["err_src"] = src
    return row


def main(selected=None):
    path = "suite_results.json"
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["target"], r["sampler"]) for r in rows}
    for name, factory in T.suite():
        if selected and name not in selected:
            continue
        print(f"=== {name} ===", flush=True)
        ref = None
        # NUTS first (reference for truthless targets)
        if (name, "nuts") not in done:
            tgt = factory()
            t0 = time.perf_counter()
            try:
                ch, info = run_nuts(tgt, seed=42, num_warmup=1000, num_samples=3000)
                rows.append(summarize(name, "nuts", ch, info, tgt))
                if tgt.truth_mean is None:
                    ref = (ch[300:].mean(0), ch[300:].var(0))
                    rows[-1]["_ref_mean"] = ref[0].tolist()
                    rows[-1]["_ref_var"] = ref[1].tolist()
                print(f"  nuts {time.perf_counter()-t0:.0f}s ess/ku={rows[-1]['ess_per_kunit']}", flush=True)
            except Exception as e:
                rows.append(dict(target=name, sampler="nuts", error=repr(e)))
                print(f"  nuts FAILED {e!r}", flush=True)
        else:
            for r in rows:
                if r["target"] == name and r["sampler"] == "nuts" and "_ref_mean" in r:
                    ref = (np.asarray(r["_ref_mean"]), np.asarray(r["_ref_var"]))
        for vname, kw in [("pmr_rta", dict()), ("pmr_pc_rta", dict(pareto_cma=True))]:
            if (name, vname) in done:
                continue
            tgt = factory()
            t0 = time.perf_counter()
            try:
                ch, info = run_pmr(tgt, seed=42, **PMR_KW, **SPECIAL.get(name, {}), **kw)
                rows.append(summarize(name, vname, ch, info, tgt, ref=ref))
                print(f"  {vname} {time.perf_counter()-t0:.0f}s ess/ku={rows[-1]['ess_per_kunit']} "
                      f"acc={rows[-1]['acc_local']}", flush=True)
            except Exception as e:
                rows.append(dict(target=name, sampler=vname, error=repr(e)))
                print(f"  {vname} FAILED {e!r}", flush=True)
        with open(path, "w") as f:
            json.dump(rows, f)

    write_summary(rows)


def write_summary(rows):
    by = {}
    for r in rows:
        by.setdefault(r["target"], {})[r["sampler"]] = r
    lines = ["# 33-posterior suite — oracle evals (min-ESS per 1k units)", "",
             "| target | d | nuts | pmr_rta | pmr_pc_rta | best/nuts | pmr quality (mean/var err) |",
             "|---|---|---|---|---|---|---|"]
    wins = losses = ties = 0
    for name, s in by.items():
        n = s.get("nuts", {})
        a = s.get("pmr_rta", {})
        b = s.get("pmr_pc_rta", {})
        nv = n.get("ess_per_kunit")
        best = max([x.get("ess_per_kunit", 0) or 0 for x in (a, b)] or [0])
        ratio = round(best / nv, 1) if nv else None
        bb = a if (a.get("ess_per_kunit") or 0) >= (b.get("ess_per_kunit") or 0) else b
        q = f"{bb.get('mean_err_sd', '—')}/{bb.get('var_err_log', '—')}"
        if ratio is not None:
            if ratio >= 1.5:
                wins += 1
            elif ratio <= 0.67:
                losses += 1
            else:
                ties += 1
        lines.append(f"| {name} | {n.get('d', bb.get('d','—'))} | {nv} | "
                     f"{a.get('ess_per_kunit', a.get('error','ERR'))} | "
                     f"{b.get('ess_per_kunit', b.get('error','ERR'))} | {ratio} | {q} |")
    lines += ["", f"**Tally (best-PMR vs NUTS, oracle evals/ESS): {wins} wins / {ties} ties / "
              f"{losses} losses** (win = >=1.5x, loss = <=0.67x)"]
    with open("SUITE.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
