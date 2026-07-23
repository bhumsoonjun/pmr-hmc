# Referee report (simulated TPAMI Reviewer 2) — recommendation: Major Revision

See git history / task artifacts for full text. Blocking issues and status:

- M1 single-seed scaling sweeps described as multi-seed  -> TEXT FIXED (honest disclosure); 8-seed rerun QUEUED
- M2 accuracy-blind, asymmetric win accounting -> FIXED (symmetric accuracy-gated criterion, 24/26 -> 23/26, kilpisjarvi = PMR failure, funnel10 no-winner, quality-free rows marked)
- M3 headline vs diagonal-mass NUTS -> FIXED in abstract/text; dense-NUTS on all 26 pdb targets QUEUED
- M4 subset selection bias -> inclusion table present; radon-family ports QUEUED
- M5 winding lemma hypothesis gap -> FIXED (Jacobian-weighted branch rule + fiber-constant reduction); sheet/truncation disclosure QUEUED
- M6 dimension-independence overreach -> FIXED (target-oracle qualification + arithmetic remark)
- M7 wall-clock warmup asymmetry -> DISCLOSED; end-to-end-at-ESS-budget bench QUEUED
- M8 ChEES unpreconditioned -> ANNOTATED; preconditioned rerun QUEUED
- M9 single-chain ESS, no Rhat -> DISCLOSED; multi-chain runs QUEUED
- M10 missing split-HMC / pCN / SurVAE citations, novelty rescoping -> FIXED
- Minor: number drift (22-27, nes 8 sets 51-71x, 0.013 sigma, 0.3-100us), Alg-1 consistency (round/L_max, flip remark), hyperparameters disclosed, Prop-1 non-conservative wording, Fig-1b label, PDB.md header regen QUEUED
