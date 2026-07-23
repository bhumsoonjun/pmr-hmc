"""Production-scale Vilya BCF: does PMR's cost track the shared nonlinear core
(d-independence via conditionally-Gaussian campaign blocks) while NUTS pays in
full dimension? C=8 (d=37) -> C=50 (d=163) -> C=120 (d=373), data rows scaled
with C (60 first-stage / 40 positive rows per campaign)."""

from __future__ import annotations

import json

import numpy as np

import targets as T
from baselines import run_nuts
from pmr_hmc import run_pmr


def ess_min(chain):
    import arviz as az

    return float(np.min(az.ess(az.convert_to_dataset(np.asarray(chain)[None])).to_array().values))


rows = []
for C in (8, 50):
    n_first, n_pos = 60 * C, 40 * C
    print(f"=== vilya C={C} (d~{3*C+13}) ===", flush=True)
    for seed_ in (1, 2, 3, 4):
     for sampler in ("nuts", "pmr_rta"):
        tgt = T.vilya_bcf(C=C, n_first=n_first, n_pos=n_pos)
        try:
            if sampler == "nuts":
                ch, info = run_nuts(tgt, seed=seed_, num_warmup=1000, num_samples=3000)
            else:
                ch, info = run_pmr(tgt, seed=seed_, n_samples=8000, n_tune=300, n_tune2=200,
                                   birth_rounds=4, transport=True, t_defense=True,
                                   n_starts=6, pareto_cma=(sampler == "pmr_pc_rta"))
            e = ess_min(ch[300:])
            units = info["warm_U"] + info["prod_U"] + 2.5 * (info["warm_grad"] + info["prod_grad"])
            r = dict(C=C, d=tgt.d, sampler=sampler, seed=seed_, ess_min=round(e, 1),
                     ess_per_kunit=round(1000 * e / units, 3),
                     units_per_ess=round(units / max(e, 1e-3), 1),
                     acc=round(info.get("acc_local", float("nan")), 3),
                     steps=round(info.get("mean_L", 0), 1),
                     time_s=round(info["warm_time"] + info["prod_time"], 1))
        except Exception as ex:
            r = dict(C=C, sampler=sampler, error=repr(ex))
        rows.append(r)
        print(" ", r, flush=True)
        with open("vilya_scale4.json", "w") as f:
            json.dump(rows, f, indent=2)
print("done")
