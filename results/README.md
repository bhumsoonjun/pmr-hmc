# Results index

**Start with [PROPER.md](PROPER.md)** — the canonical final tables: all
ten methods on both suites (posteriordb-37 gold-judged; mega-30
controlled stress panel), accuracy-gated, matching the paper. Raw
per-cell data for every number in the paper is under `raw/`
(`pdb2r_shard*` = final PMR posteriordb cells; `pdb2_results.json` =
NUTS/dense cells; `*_panel_shard*` = the other seven methods;
`mega_results_shard*` = mega PMR/NUTS; `scaling8` / `sbc` /
`mega_native` / `vilya_whale` = sweeps, calibration, wall-clock).

Historical reports kept for provenance (earlier code versions or
subsumed subsets — do not quote): `MEGA.md` (102-target vs-NUTS run,
superseded), `PDB2.md` (3-method posteriordb), `SUITE.md` (33-target
early suite), `SOTA2.md` (10-target learned-method battery),
`NATIVE.md` (13-target parity study, still the source of the C-vs-JAX
parity claim), `SCALING.md` (superseded by scaling8 raw data).
