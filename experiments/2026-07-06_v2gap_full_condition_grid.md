# 2026-07-06 v2 GAP full condition grid

## Purpose

Expand the selected `cnntf_v2_gap` candidate check from one clean condition to
the full planned condition grid:

- 3 selected parameter sets
- 3 experiment days
- 6 max-frequency settings
- 7 water-flow noise conditions

## Active Script

`code/run_ensemble_regression_onb.py`

## Active Model

- `cnntf_v2_gap`

## Parameter Sets

- `r2_best_lr0p001_bs12`
- `balanced_lr0p0005_bs32`
- `r2_onb_lr0p0005_bs12`

## Data Grid

- experiments:
  - `2025.06.18_0.3_3`
  - `2025.07.09_0.3_1`
  - `2025.06.11_0.3_2`
- max frequency:
  - `maxfreq=2kHz`
  - `maxfreq=3kHz`
  - `maxfreq=5kHz`
  - `maxfreq=10kHz`
  - `maxfreq=15kHz`
  - `maxfreq=22kHz`
- noise:
  - `heatflux_no_noise`
  - `heatflux_SNR=0`
  - `heatflux_SNR=-4`
  - `heatflux_SNR=-8`
  - `heatflux_SNR=-12`
  - `heatflux_SNR=-16`
  - `heatflux_SNR=-20`

## Verified Plan

- existing dataset jobs: 126 / 126
- parameter sets: 3
- folds: 5
- planned model fits: 1890
- result directory suffix: `_v2gap_5fold_3exp_6freq_7noise`

`resume_completed_runs=True` is left enabled so interrupted runs can be
continued without recomputing completed parameter-set directories.

