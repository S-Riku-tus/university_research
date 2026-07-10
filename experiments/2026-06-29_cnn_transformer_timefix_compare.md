# 2026-06-29 CNN+Transformer Time-Axis Fix Compare

## Purpose

This experiment checks two follow-up hypotheses after the v2 pooling comparison.

1. `cnntf_v2_dim256_attn`
   - Same AlexNet-like v2 front-end.
   - Uses legacy-like Transformer dimensions: `model_dim=256`, `attention_key_dim=64`.
   - Uses `AttentionPooling`.
   - Tests whether the main issue was the small/inconsistent v2 Transformer dimension.

2. `cnntf_timeaxis_attn`
   - Uses a CNN front-end that mainly pools the frequency axis.
   - Keeps the npy time axis as the Transformer sequence.
   - Expected sequence length after CNN: `224`.
   - Uses `model_dim=128`, `attention_key_dim=32`, two Transformer blocks, and `AttentionPooling`.
   - Tests whether the main issue was collapsing the time axis too aggressively before the Transformer.

## Current Run Settings

- dataset: `2025.06.11_0.3_2`
- data source: `waterflow_20251126_1s`
- max frequency: `maxfreq=22kHz`
- noise: `heatflux_no_noise`
- threshold: `266907.6965`
- epochs: `300`
- folds: `2`
- lr: `0.0001`
- batch sizes: `48`, `64`
- result folder suffix: `_timefix`

## Interpretation

- If `cnntf_v2_dim256_attn` is best, the likely issue is the v2 Transformer projection/attention dimension.
- If `cnntf_timeaxis_attn` is best, the likely issue is excessive time-axis compression before the Transformer.
- If both are worse than the previous v2 AttnPool/legacy results, the added complexity is not helping under the current training setup.

## Result Snapshot

Run folder:

`Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/cnntf_timefix/20260629_timefix`

Best result in this run:

- model: `cnntf_v2_dim256_attn`
- parameter set: `lr0p0001_bs48`
- R2: `0.8020`
- RMSE: `118540.5920`
- MAE: `94873.3511`
- ONB RMSE: `41626.6422`
- ROC-AUC continuous: `0.9821`
- PR-AUC continuous: `0.9902`
- F1: `0.8564`

Second candidate:

- model: `cnntf_v2_dim256_attn`
- parameter set: `lr0p0001_bs64`
- R2: `0.7780`
- RMSE: `125593.0386`
- ONB RMSE: `30914.9593`
- F1: `0.8596`

The time-axis-preserving model did not work under the current training setup:

- `lr0p0001_bs48`: R2 `-5.0092`, RMSE `580957.4432`
- `lr0p0001_bs64`: R2 `-45.7033`, RMSE `1446782.0786`

Current interpretation:

- The stronger explanation is not "keeping all 224 time steps" as implemented here.
- The stronger explanation is that v2 needed legacy-like Transformer dimensionality plus `AttentionPooling`.
- The next candidate should keep `cnntf_v2_dim256_attn` and validate it with more folds/seeds and then other experiment dates.
