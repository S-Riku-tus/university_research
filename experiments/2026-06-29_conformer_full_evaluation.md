# 2026-06-29 Canonical Conformer Full Evaluation

## Purpose

Run the cleaned-up canonical `conformer` model after replacing the temporary
CNN+Transformer comparison path with a Conformer-style block.

## Settings

- dataset: `2025.06.11_0.3_2`
- data source: `waterflow_20251126_1s`
- max frequency: `maxfreq=22kHz`
- noise: `heatflux_no_noise`
- threshold: `266907.6965`
- model: `conformer`
- architecture: AlexNet-like CNN front end, `conformer_block`, `AttentionPooling`
- `model_dim`: `256`
- `attention_key_dim`: `64`
- `num_heads`: `4`
- `ff_dim`: `2048`
- `num_conformer_blocks`: `4`
- `conv_kernel_size`: `31`
- dropout: `0.2`
- lr: `0.0001`
- batch size: `64`
- epochs: `300`
- folds: `3`

## Result Folder

`Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/conformer/20260629_conformer`

The first interrupted direct-run output was archived to:

`trush_box/2026-06-29_conformer_cleanup/aborted_conformer_full_20260629_171226`

## Result Snapshot

| model | R2 | RMSE all | MAE all | ONB RMSE | ROC-AUC cont | PR-AUC cont | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `conformer` | 0.4375 | 194125.8 | 167993.1 | 67840.0 | 0.9774 | 0.9878 | 0.7706 |

Fold-level behavior:

| fold | R2 | RMSE all | ONB RMSE | AUC binary | ROC-AUC cont | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.6994 | 149324.9 | 18379.1 | 0.8834 | 0.9792 | 0.8680 |
| 2 | 0.0739 | 249233.4 | 148263.3 | 0.5000 | 0.9790 | 0.7004 |
| 3 | 0.5392 | 183819.1 | 36877.7 | 0.5000 | 0.9739 | 0.7435 |

Epochs completed by fold: `171|152|169`.

## Additional Prediction Check

The continuous ROC/PR metrics stayed high, but fold 2 and fold 3 predicted all
validation samples above the fixed ONB threshold.

| fold | true positive rate | predicted positive rate at ONB threshold | low-true prediction mean |
|---|---:|---:|---:|
| 1 | 0.536 | 0.411 | 245711.4 |
| 2 | 0.539 | 1.000 | 412497.7 |
| 3 | 0.592 | 1.000 | 302249.6 |

Best per-fold prediction thresholds were much higher or lower than the physical
ONB threshold:

| fold | best F1 | best prediction threshold |
|---|---:|---:|
| 1 | 0.984 | 246031.9 |
| 2 | 0.979 | 412758.2 |
| 3 | 0.969 | 302379.5 |

This means the model is often ranking low/high heat-flux samples correctly, but
the absolute heat-flux scale is not calibrated consistently across folds.

## Comparison With Previous Candidate

The previous strongest CNN+Transformer candidate was
`cnntf_v2_dim256_attn`, `lr=0.0001`, `batch_size=64`:

- R2: `0.7737`
- RMSE all: `126715.5`
- ONB RMSE: `74002.5`
- F1: `0.8237`

The canonical Conformer has a similar ONB-neighborhood error scale, but it is
worse for global regression and fixed-threshold ONB detection. Therefore this
cleanup should not yet replace the v2 dim256 candidate as the strongest
empirical baseline.

## Interpretation

- The Conformer block itself runs correctly and saves all expected artifacts.
- The current failure mode is not inability to rank samples: continuous
  ROC-AUC and PR-AUC are high.
- The main failure mode is absolute-scale calibration. In fold 2 and fold 3,
  the low-heat-flux region is shifted above the physical ONB threshold, so the
  binary decision collapses to all-positive.
- Current Keras training uses `EarlyStopping(monitor="loss")` on training loss
  only. That is not enough to select weights that are well calibrated on held-out
  validation data.

## Next Step

Before more architecture changes, add an inner validation split inside each
training fold for Keras models and use:

- `validation_data` or `validation_split`
- `EarlyStopping(monitor="val_loss")`
- `restore_best_weights=True`

Then rerun the same `conformer` condition and compare:

1. R2 / RMSE all
2. ONB RMSE
3. fixed-threshold predicted positive rate by fold
4. ROC-AUC / PR-AUC continuous

If scale calibration improves but regression remains weaker than
`cnntf_v2_dim256_attn`, keep the canonical Conformer as a clearly named
comparison model and keep the v2 dim256 model as the stronger empirical
baseline for now.
