# 2026-06-29 Conformer Cleanup

## Purpose

The previous `cnn_transformer` models were useful for comparison, but they were
not strict Conformer models. This cleanup makes the main experiment path use a
canonical Conformer-style regressor instead of calling a plain CNN+Transformer
"Conformer".

## Code Changes

- Added `conformer_block()` in `code/utils/models/regression/base_regression.py`.
- Added `RegressionModelMaker.conformer()`.
- The Conformer block order is:
  `FFN -> self-attention -> convolution module -> FFN`.
- The main runner now uses model key `conformer`.
- Main active setting:
  - model: `conformer`
  - parameter_set: `main`
  - lr: `0.0001`
  - batch size: `64`
  - result folder suffix: `_conformer`
- Removed main-runner registrations for temporary comparison models:
  - `cnntf_axis`
  - `cnntf_legacy`
  - `cnntf_v2_gap`
  - `cnntf_v2_attn`
  - `cnntf_v2_dim256_attn`

## Verification

- Python compile check passed for:
  - `code/run_ensemble_regression_onb.py`
  - `code/utils/models/regression/base_regression.py`
  - `code/utils/plotting/regression_plots.py`
- Model construction passed:
  - model name: `conformer_regressor`
  - input shape: `(None, 224, 224, 1)`
  - output shape: `(None, 1)`
  - parameters: `14483009`
- Smoke run passed through data loading, 2-fold training, prediction, metrics,
  fold-prediction CSV saving, and tuning-summary saving.

## Archived

The failed partial path-length run was moved to:

`trush_box/2026-06-29_conformer_cleanup`

Archived items:

- `20260629_v2d256_legacy_lrbs`
- `v2d256_legacy_lrbs_20260629_163443.log`

The successful comparison summaries and successful result folders were kept as
evidence.
