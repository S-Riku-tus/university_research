# 2026-07-06 v2 GAP parameter confirmation before full grid

## Purpose

Before expanding to the full 3 experiment x 6 frequency x 7 noise grid, confirm
which of the three selected `cnntf_v2_gap` parameter sets is most stable across
experiment days.

This avoids mixing two questions:

- which parameter set should be fixed
- how the fixed model behaves under frequency and noise changes

## Active Script

`code/run_ensemble_regression_onb.py`

## Active Scope

- model: `cnntf_v2_gap`
- parameter sets:
  - `r2_best_lr0p001_bs12`
  - `balanced_lr0p0005_bs32`
  - `r2_onb_lr0p0005_bs12`
- experiments:
  - `2025.06.18_0.3_3`
  - `2025.07.09_0.3_1`
  - `2025.06.11_0.3_2`
- max frequency: `maxfreq=22kHz`
- noise: `heatflux_no_noise`
- folds: 5
- epochs: 300

## Verified Plan

- dataset jobs: 3 / 3
- parameter sets: 3
- planned model fits: 45
- result directory suffix: `_v2gap_5fold_param_confirm_3exp`

## Next Step After This Run

Select one parameter set by comparing:

- overall `r2_mean` and `r2_se`
- `r2_high_mean`
- `rmse_onb_mean`
- continuous ROC-AUC / PR-AUC
- consistency across the three experiment days

After the parameter is fixed, run the full frequency/noise grid with only that
one parameter set.

