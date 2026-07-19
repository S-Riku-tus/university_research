# 2026-07-02 CNN+Transformer v1 No Early-Stopping Reproduction

## Purpose

Previously, `CNN+Tf (AttnPool)` reached high accuracy under a historical
condition. The current check narrows the run to that condition so the model can
be compared without changing dataset, fold count, epoch count, or training-stop
behavior at the same time.

## Current Code Setting

- script: `code/run_ensemble_regression_onb.py`
- active model: `cnntf_v1`
- dataset: `2025.07.09_0.3_1`
- data source: `waterflow_20251219_1s`
- max frequency: `maxfreq=22kHz`
- noise: `heatflux_no_noise`
- epochs: `500`
- folds: `5`
- parameter set: `old_v1_reg`
- learning rate: `0.01`
- batch size: `24`
- result folder suffix: `_v1_noes_repro`

## Code Change

The automatic training-stop callback was removed from
`code/utils/training/model_training.py` and from the run metadata/tuning summary
fields in `code/run_ensemble_regression_onb.py`. Keras training now runs for the
configured epoch count unless an out-of-memory retry path is triggered.

## Interpretation Plan

If this run recovers the older `CNN+Tf (AttnPool)` performance, the likely cause
of the recent drop is the training protocol rather than the model architecture
itself. If it still fails, the next checks should compare generated `.npy`
content, label order, and old result fold predictions against the current run.
