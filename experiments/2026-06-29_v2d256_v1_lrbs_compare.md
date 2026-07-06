# 2026-06-29 v2 Dim256 With Old v1 LR/Batch Settings

## Purpose

This run checks whether the old high-performing `cnn_transformer_v1` learning
conditions also help the current strongest candidate:

- model: `cnntf_v2_dim256_attn`
- architecture: v2 AlexNet-like CNN front-end, `model_dim=256`,
  `attention_key_dim=64`, `num_heads=4`, `ff_dim=2048`,
  `num_transformer_blocks=4`, `AttentionPooling`

The architecture-side old v1 settings are already represented by
`cnntf_v2_dim256_attn`, so this run focuses on learning rate and batch size.

## Settings

- dataset: `2025.06.11_0.3_2`
- data source: `waterflow_20251126_1s`
- max frequency: `maxfreq=22kHz`
- noise: `heatflux_no_noise`
- threshold: `266907.6965`
- epochs: `300`
- folds: `3`
- result group: `cnntf_v2d256`
- result folder suffix: `_v2d256_v1lr`

## Parameter Sets

| parameter_set | meaning | lr | batch_size |
|---|---|---:|---:|
| `cur64` | current candidate | 0.0001 | 64 |
| `onb128` | old v1 ONB/ROC-favorable setting | 0.0001 | 128 |
| `reg24` | old v1 regression-favorable setting | 0.01 | 24 |

## Result Folder

`Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/cnntf_v2d256/20260629_v2d256_v1lr`

## Result Snapshot

| parameter_set | R2 | RMSE all | MAE all | ONB RMSE | ROC-AUC cont | PR-AUC cont | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cur64` | 0.6979 | 145506.8 | 112362.5 | 69415.7 | 0.9738 | 0.9867 | 0.7417 |
| `onb128` | 0.2571 | 222800.8 | 181252.1 | 58532.1 | 0.9741 | 0.9857 | 0.4661 |
| `reg24` | 0.5665 | 153111.7 | 130598.0 | 135587.9 | 0.9576 | 0.9778 | 0.8005 |

## Interpretation

- The old v1 `batch_size=128` ONB-oriented setting did not transfer cleanly to
  this model. It reduced mean ONB RMSE compared with `cur64`, but global
  regression and binary threshold detection collapsed.
- The old v1 `lr=0.01, batch_size=24` regression setting gave reasonable F1,
  but fold 2 had negative R2, so it is not stable enough to use as the default.
- The current `lr=0.0001, batch_size=64` setting remains the safest default
  among these three for balanced regression/detection behavior.
- This supports the interpretation that the old v1 architectural settings
  mattered more than directly copying the old v1 learning-rate/batch-size
  choices.
