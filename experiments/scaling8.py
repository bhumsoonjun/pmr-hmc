"""Scaling-exponent sweeps: where PMR beats NUTS in complexity, not constants.

  kappa : gauss_ill condition sweep     — NUTS O(sqrt(kappa)) vs PMR O(1) per ESS
          (PMR pays O(sqrt(kappa)) ONCE in L-BFGS warm-up, amortized)
  dim   : embedded banana ambient-d     — NUTS O(d^{1/4}) vs PMR O(1) evals/ESS
          (exact harmonic integration => no discretization error in the
           Gaussian directions; acceptance set by the s=2 residual only)
  barrier: mixture2 mode-separation     — NUTS e^{-sep^2/2}-ish switch rate vs
          PMR O(1) via the calibrated global branch (uniform ergodicity)
"""

from __future__ import annotations

import json
import sys

import numpy as np

import targets as T
from baselines import run_nuts
from pmr_hmc import run_pmr


def ess_min(chain):
    import arviz as az

    ds = az.convert_to_dataset(np.asarray(chain)[None])
    return float(np.min(az.ess(ds).to_array().values))


def switches(chain):
    s = np.sign(chain[:, 0])
    return int(np.sum(s[1:] != s[:-1]))


def row(sweep, sampler, x, chain, info, extra=None):
    burn = min(500, len(chain) // 10)
    ch = chain[burn:]
    e = max(ess_min(ch), 1e-3)
    units = info["warm_U"] + info["prod_U"] + 2.5 * (info["warm_grad"] + info["prod_grad"])
    prod_units = info["prod_U"] + 2.5 * info["prod_grad"]
    r = dict(sweep=sweep, sampler=sampler, x=x,
             ess_min=round(e, 1),
             units_per_ess=round(units / e, 2),
             prod_units_per_ess=round(prod_units / e, 3),
             warm_units=int(units - prod_units),
             acc=round(info.get("acc_local", float("nan")), 3))
    if extra:
        r.update(extra)
    return r


def kappa_sweep(out):
    for cond in [1e2, 1e3, 1e4, 1e5]:
      for seed in range(1, 9):
        tgt = T.gauss_ill(cond=cond)
        ch, info = run_nuts(tgt, seed=seed, num_warmup=1000, num_samples=3000)
        out.append(row("kappa", "nuts", cond, ch, info))
        print(out[-1], flush=True)
        tgt = T.gauss_ill(cond=cond)
        ch, info = run_pmr(tgt, seed=seed, n_samples=8000, transport=True, t_defense=True,
                           lbfgs_max_iter=8000)
        out.append(row("kappa", "pmr", cond, ch, info))
        print(out[-1], flush=True)


def dim_sweep(out):
    for d in [20, 50, 100, 200]:
      for seed in range(1, 9):
        tgt = T.banana(d=d)
        ch, info = run_nuts(tgt, seed=seed, num_warmup=1000, num_samples=3000)
        out.append(row("dim", "nuts", d, ch, info))
        print(out[-1], flush=True)
        tgt = T.banana(d=d)
        ch, info = run_pmr(tgt, seed=seed, n_samples=8000, transport=True, t_defense=True)
        out.append(row("dim", "pmr", d, ch, info))
        print(out[-1], flush=True)


def barrier_sweep(out):
    for sep in [3.0, 4.0, 6.0, 8.0]:
      for seed in range(1, 9):
        for sampler in ("nuts", "pmr"):
            tgt = T.mixture2(sep=sep)
            if sampler == "nuts":
                ch, info = run_nuts(tgt, seed=seed, num_warmup=1000, num_samples=4000)
            else:
                ch, info = run_pmr(tgt, seed=seed, n_samples=8000, transport=True, t_defense=True)
            burn = min(500, len(ch) // 10)
            m_err = abs(float(ch[burn:, 0].mean()) - tgt.truth_mean[0]) / np.sqrt(tgt.truth_var[0])
            out.append(row("barrier", sampler, sep, ch, info,
                           extra=dict(switches=switches(ch[burn:]),
                                      mean_err=round(m_err, 3))))
            print(out[-1], flush=True)


def main(which=None):
    out = []
    sweeps = dict(kappa=kappa_sweep, dim=dim_sweep, barrier=barrier_sweep)
    for name, fn in sweeps.items():
        if which and name not in which:
            continue
        print(f"=== {name} sweep ===", flush=True)
        fn(out)
        with open("scaling8_results.json", "w") as f:
            json.dump(out, f, indent=2)

    lines = ["# Scaling sweeps — NUTS vs PMR (units = density evals, grad = 2.5)", ""]
    for name in sweeps:
        rows = [r for r in out if r["sweep"] == name]
        if not rows:
            continue
        lines.append(f"## {name}")
        cols = list(rows[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        lines.append("")
    with open("SCALING8.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
