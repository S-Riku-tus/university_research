# 2026-07-02 CNN+Transformer v1 lr/batch tuning

## Purpose

Keep the current `cnntf_v1` architecture fixed and tune only training
parameters. This separates architecture diagnosis from optimizer/batch-size
effects.

## Current Code Setting

- Script: `code/run_ensemble_regression_onb.py`
- Active model: `cnntf_v1`
- Dataset: `2025.07.09_0.3_1`
- Data source: `waterflow_20251219_1s`
- Frequency condition: `maxfreq=22kHz`
- Noise condition: `heatflux_no_noise`
- Epochs: `500`
- Folds: `5`
- Result directory suffix: `_v1_lrbs_tune`

## Tuning Grid

- Learning rates: `0.05`, `0.01`, `0.005`, `0.001`, `0.0005`, `0.0001`, `0.00005`
- Batch sizes: `12`, `24`, `32`, `48`, `64`, `128`
- Total parameter sets: `42`
- Keras fit progress: enabled with `fit_verbose=1`

## Interpretation Plan

First compare `r2_mean`, `rmse_all_mean`, and fold stability in
`tuning_summary.csv`. If all conditions remain far below the historical
CNN+Tf result, the next investigation should compare the actual model graph,
input axes, and label scaling/evaluation path against the older saved run.
