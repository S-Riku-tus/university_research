# 2026-07-06 v2 GAP 5-fold candidate check

## Purpose

The 30-condition `cnntf_v2_gap` learning-rate and batch-size tuning run is
complete. The next step is not to expand the grid, but to confirm whether the
best candidates remain stable under 5-fold cross validation.

## Active Conditions

`code/run_ensemble_regression_onb.py` now runs only:

- `r2_best_lr0p001_bs12`: highest overall 3-fold R2
- `balanced_lr0p0005_bs32`: strong R2 with the smallest fold variation among
  the top candidates and better ONB-neighborhood behavior
- `r2_onb_lr0p0005_bs12`: high overall R2 with better high-heat-flux behavior

## Fixed Scope

- experiment: `2025.06.18_0.3_3`
- data source: `waterflow_20260622_1s`
- max frequency: `maxfreq=22kHz`
- noise condition: `heatflux_no_noise`
- model: `cnntf_v2_gap`
- epochs: 300
- folds: 5
- result directory suffix: `_v2gap_5fold_candidates`

## Next Reading

After this run finishes, compare the three candidates by:

- overall `r2_mean` and `r2_se`
- `r2_high_mean`
- `rmse_onb_mean`
- continuous ROC-AUC and PR-AUC

The likely final candidate should balance global heat-flux regression and ONB
neighborhood behavior, not simply maximize overall R2.

