# 2026-07-06 explainability implementation

## Purpose

Add a first runnable explainability pipeline for the current heat-flux
regression models without mixing explanation code into the training script.

## Added Code

- `code/run_explainability_analysis.py`
  - trains `rf`, `alexnet`, and `cnntf_v2_gap` on one CV fold
  - selects representative validation samples
  - saves explanation maps and summary CSVs
- `code/utils/explainability/spectrogram_explainers.py`
  - Integrated Gradients
  - Grad-CAM for scalar regression outputs
  - grouped frequency/time occlusion
  - deletion curves
  - CSV/PNG/NPY save helpers

## Default Model Conditions

- RF: `n_estimators=300`, `max_depth=8`, `subsample=0.8`,
  `colsample_bynode=0.6`
- AlexNet: `lr=0.005`, `batch_size=32`
- CNN+Transformer v2 GAP: `lr=0.0005`, `batch_size=32`

## Output

The full run saves under:

`Pool_boiling/.../2025.06.18_0.3_3/explainability_result/npy/<date>_xai_fold1/`

Each model has:

- `validation_metrics.csv`
- `explained_samples.csv`
- sample-wise explanation maps as `.npy` and `.png`
- frequency/time profiles as `.csv`
- grouped occlusion CSVs
- deletion-curve CSVs

RF additionally writes PCA-space feature importance and a TreeSHAP availability
status file.

## Verification

Smoke command:

`EXPLAIN_SMOKE=1 EXPLAIN_SMOKE_EPOCHS=1 python code/run_explainability_analysis.py`

The smoke run completed and produced 185 files under:

`.../explainability_result/npy/20260706_xai_fold1_smoke/`

The smoke metrics are not meaningful because the Keras models train for only
one epoch. The check is only for code-path and file-output verification.

