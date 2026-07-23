# pmrhmc

**Learned transport atlases for exact, gradient-free production MCMC.**

`pmrhmc` learns a *mixture-of-charts* reference (affine / Student-t /
quadratic-shear / conditional-scale / polar-winding / hierarchical /
sinh-arcsinh-triangular transports) for your posterior during a one-time
warm-up, then runs production sampling in a portable C kernel with:

- **one true density evaluation per transition** — zero gradients in production,
- **exactness by construction** — true-endpoint Metropolis–Hastings on the
  chart-augmented target; every transport is an exact bijection with an exact
  Jacobian,
- **oracle-cost scaling that is flat in condition number and ambient
  dimension** at fixed residual dimension (measured; see the paper),
- production speeds of **0.2–50 µs/draw** on the built-in target families.

## Install

```bash
pip install .        # compiles the C kernel for your machine on first import
```

Dependencies: `numpy`, `scipy`, a C compiler (`cc`/`clang`/`gcc`).
Everything is float64; densities are handled in log space end to end.

## Use

```python
import numpy as np, pmrhmc

# any unnormalized log-density (gradient optional — used only in warm-up;
# without it, warm-up falls back to finite differences)
def logpdf(x):
    return -0.5 * float(x @ x)

s = pmrhmc.PMRHMC(logpdf, d=20, grad_logpdf=lambda x: -x)
s.fit(seed=0)                                     # learn the atlas (one-time)
draws = s.sample(100_000, chains=4, cores=4)      # C kernel, parallel chains
print(s.diagnostics["acc_local"], s.last_stats)
```

- `chains`/`cores`: independent chains over the shared frozen atlas.
  Built-in target families are threaded in C (pthreads); Python-callback
  targets run one C chain per Python thread (the kernel releases the GIL
  between callbacks — one callback per *transition*, not per step, so the
  Python boundary costs microseconds per draw).
- `s.sample_py(n)`: pure-Python reference kernel (identical semantics; used
  by the test suite for parity).
- `pmrhmc.Target` exposes the phase-tagged oracle counters
  (`warmup/production × U/grad`) used for all cost accounting.

## Numerical robustness

- All chart exponents are clamped with map + Jacobian sharing the clamp, so
  bijections stay exact while overflow is impossible.
- A chart that cannot evaluate a point contributes zero reference density
  (−∞), never NaN; mixture weights are sanitized before normalization.
- d is capped at 4096 per chain (fixed scratch); float64 throughout.

## Layout

- `src/pmrhmc/warmup.py` — atlas learning (pure numpy/scipy): multi-start
  L-BFGS charts, forward-KL weights on a provenance-correct pool,
  residual-guided birth, transport detection, error-knee step-size.
- `src/pmrhmc/_csrc/pmr_kernel.c` — C11 production kernel (xoshiro256++,
  all 7 chart types, scalar-RBF + local-affine kNN residual caches,
  t-defended global moves, pthreads multi-chain).
- `src/pmrhmc/native.py` — build-on-first-use loader + frozen-atlas packing.
- `tests/` — chart bijection exactness, C/Python parity, end-to-end moment
  correctness on analytic targets, parallel-chain checks, determinism,
  oracle accounting, NaN-robustness guards.

## Paper

*Learned Transport Atlases for Exact, Gradient-Free Production MCMC* —
see `paper/` in the repository for the full construction, exactness proofs,
complexity separations, and the 10-method benchmark on posteriordb.

## Headline results

On the official **posteriordb** benchmark (37 gold-referenced posteriors,
symmetric accuracy gate: mean err ≤ 0.25σ and |log-var err| ≤ 0.5), PMR-HMC
is the accuracy-gated best of a **ten-method panel** on **34 of 37**
posteriors, against diagonal- and dense-mass NUTS, Pathfinder-initialized
dense NUTS, ChEES-HMC, MCLMC, TESS, flow-based independence MH, a
flowMC-style hybrid, and NeuTra. Median advantages over rows where both
samplers pass the gate: **38× (diag NUTS), 4.1× (dense NUTS), 12× (ChEES),
10× (MCLMC), 29–95× (flow-based samplers, which fail the gate on most real
posteriors)**. A curated 30-target controlled stress panel that retains
every known failure family (funnels, horseshoe, extreme skew, deep
hierarchies) localizes where the method wins and loses. Measured scaling:
production oracle cost flat in condition number 10²–10⁵, in dimension
50–200 at fixed residual dimension, and in mode separation. Full tables in
`results/`, per-cell raw data in `results/raw/`, and the complete
experimental harness in `experiments/`.

Honest limits, documented in the paper: three posteriordb losses; at small
ESS budgets JIT-compiled NUTS wins end-to-end wall clock (median crossover
≈ 4,500 ESS); warm-up has a dimension ceiling near d ≈ 400; cheap fully
compiled likelihoods favor gradient samplers in wall clock.
