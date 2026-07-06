# 2026-07-06 fixed three-model ensemble full grid

## Purpose

The full-condition validation is now configured as a fixed three-model ensemble
instead of the CNN+Transformer v2 GAP single-model run.

## Active plan

- Models:
  - `rf`
  - `cnntf_v2_gap`
  - `alexnet`
- Ensemble:
  - enabled: `True`
  - strategy: `fixed`
  - weights: `rf=0.90`, `cnntf_v2_gap=0.05`, `alexnet=0.05`
  - combine: `mean`
- Fixed model parameters:
  - RF: `n_estimators=300`, `max_depth=8`, `subsample=0.8`, `colsample_bynode=0.6`
  - CNN+Transformer v2 GAP: `lr=0.0005`, `batch_size=32`
  - AlexNet: `lr=0.005`, `batch_size=32`
- Data grid:
  - 3 experiments
  - 6 max-frequency conditions
  - 7 noise conditions
- Folds: 3
- Planned model fits: `3 * 6 * 7 * 3 folds * 3 models = 1134`

## Verification

Import-based configuration check:

- datasets: `126 / 126`
- parameter sets: `["rf_v2_alex"]`
- result group: `ensemble`
- result directory: `20260706_fixed_ensemble_full_grid`
- explainability enabled: `False`

The edited Python files passed `python -m py_compile`.
