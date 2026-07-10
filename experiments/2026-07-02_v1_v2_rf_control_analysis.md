# 2026-07-02 v1/v2 and RF control analysis

## Fixed data condition

- experiments: `2025.06.18_0.3_3`, `2025.07.09_0.3_1`, `2025.06.11_0.3_2`
- max frequency: `maxfreq=22kHz`
- noise: `heatflux_no_noise`
- current data sources:
  - `2025.06.18_0.3_3`: `waterflow_20260622_1s`
  - `2025.07.09_0.3_1`: `waterflow_20251219_1s`
  - `2025.06.11_0.3_2`: `waterflow_20260629_1s`

The temporary Conformer architecture sweep parameters
(`time_reduction`, `frontend_filters`) were removed from the regular
Conformer function. The Conformer front end is back to the fixed AlexNet-like
7-token sequence before the Conformer blocks.

## Run 1: historical v1/v2 candidates

Result directory: `20260702_v12b`

| model | mean R2 | mean RMSE | mean ONB RMSE | mean Acc | mean F1 |
|---|---:|---:|---:|---:|---:|
| `cnntf_v2_gap` | 0.6520 | 154120.8 | 100754.1 | 0.7571 | 0.7405 |
| `cnntf_v2_attn` | 0.3464 | 210644.3 | 104783.9 | 0.6652 | 0.5451 |
| `cnntf_v1` | -3.7050 | 515058.8 | 497944.0 | 0.5832 | 0.5623 |

Interpretation: v2 GAP is the strongest among the historical v1/v2 baseline
settings. v1 is unstable and sometimes collapses to large bias.

## Run 2: legacy-like v2 Dim256 Attention

Result directory: `20260702_v2d256`

| condition | mean R2 | mean RMSE | mean ONB RMSE | mean Acc | mean F1 |
|---|---:|---:|---:|---:|---:|
| `d256_b48` | 0.7552 | 133651.1 | 80331.2 | 0.7533 | 0.7739 |
| `d256_b64` | 0.7255 | 141878.9 | 55981.2 | 0.7666 | 0.7846 |
| `d256_l5e5_b64` | 0.6203 | 162950.5 | 41287.9 | 0.6597 | 0.7363 |

Interpretation: restoring the legacy-like larger Transformer dimension
substantially improves the neural model. `d256_b48` is best for global
heat-flux regression, while `d256_b64` is better balanced for ONB metrics.
However, neither returns to R2/accuracy around 0.9.

## Run 3: RF control on the same current data

Result directory: `20260702_rfcur`

| experiment | R2 | RMSE | ONB RMSE | Acc | F1 |
|---|---:|---:|---:|---:|---:|
| `2025.06.11_0.3_2` | 0.9663 | 48920.2 | 93057.1 | 0.9731 | 0.9751 |
| `2025.06.18_0.3_3` | 0.9799 | 38122.9 | 32826.8 | 0.9361 | 0.9388 |
| `2025.07.09_0.3_1` | 0.9234 | 77523.4 | 67039.6 | 0.9049 | 0.9024 |
| mean | 0.9565 | 54855.5 | 64307.8 | 0.9381 | 0.9388 |

Interpretation: the current data condition can still produce 0.9-class
performance, but it is RF that achieves it, not the current CNN/Transformer
models.

## Historical check

Across the three result trees for `22kHz/no_noise`, no CNN/Transformer or
Conformer `tuning_summary.csv` rows with R2 >= 0.85 were found. The R2 >= 0.9
rows found in the same search were RF rows, especially for
`2025.06.18_0.3_3`.

## Current conclusion

1. The recent low Conformer result is not fixed by increasing time tokens or by
   returning to v1.
2. The best neural candidate found here is the legacy-like
   `cnn_transformer_v2` with `model_dim=256`, `attention_key_dim=64`, and
   `AttentionPooling`.
3. The 0.9-class performance is currently explained by RF, not by the neural
   model family.
4. Next, treat RF as the main baseline and use the best v2 Dim256 model as the
   neural comparison/ensemble member. Claims about neural superiority should
   not be made from the present results.
