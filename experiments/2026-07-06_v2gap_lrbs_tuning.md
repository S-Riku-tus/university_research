# 2026-07-06 CNN+Transformer v2 GAP lr/batch tuning

> **履歴・適用範囲（2026-09-15追記）**: この文書は当時の条件・判断の記録です。本文の「現在」「次にやること」は作成時点を指します。最新の設定・完了範囲・優先順位は[研究の現在地](../docs/research_status.md)、更新関係は[文書案内](../docs/document_index.md)を参照してください。

## Purpose

The `cnntf_v1` lr/batch-size tuning did not recover the historical regression
accuracy. The next check is to use the AlexNet-front-end CNN+Transformer
architecture that matched the user's older successful code more closely.

## Current Code Setting

- Script: `code/run_ensemble_regression_onb.py`
- Active model: `cnntf_v2_gap`
- Model function: `RegressionModelMaker.cnn_transformer_v2(pooling="gap")`
- Dataset: `2025.07.09_0.3_1`
- Data source: `waterflow_20251219_1s`
- Frequency condition: `maxfreq=22kHz`
- Noise condition: `heatflux_no_noise`
- Epochs: `500`
- Folds: `5`
- Result directory suffix: `_v2gap_lrbs_tune`

## Architecture

- AlexNet-like Conv2D front-end
- Reshape 2D CNN feature map to a sequence
- `TimeDistributed(Dense(32, activation="relu"))`
- Positional embedding
- 4 Transformer encoder blocks
- `GlobalAveragePooling1D`
- Linear regression output

This is intended to match the shared older `cnn_transformer_v2` structure with
GAP pooling.

## Tuning Grid

- Learning rates: `0.05`, `0.01`, `0.005`, `0.001`, `0.0005`, `0.0001`, `0.00005`
- Batch sizes: `12`, `24`, `32`, `48`, `64`, `128`
- Total parameter sets: `42`
- Keras fit progress: enabled with `fit_verbose=1`

## Interpretation Plan

If `cnntf_v2_gap` recovers R2 while `cnntf_v1` did not, the likely issue is
the v1 architecture or its input-axis interpretation. If `cnntf_v2_gap` also
fails, the next check should compare the old successful run against current
code in label scaling, prediction inverse transform, and exact model graph.
