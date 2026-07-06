# 2026-07-02 Analysis of 20260701 Conformer Full Run

## Scope

This note summarizes the completed `20260701_conformer` run from the current
`code/run_ensemble_regression_onb.py` settings.

Current active conditions:

- model: `conformer`
- parameter set: `main`
- lr: `0.0001`
- batch size: `64`
- epochs: `300`
- folds: `3`
- experiments: `2025.06.18_0.3_3`, `2025.07.09_0.3_1`,
  `2025.06.11_0.3_2`
- max frequencies: `2`, `3`, `5`, `10`, `15`, `22 kHz`
- noise conditions: `no_noise`, `SNR=0`, `-4`, `-8`, `-12`, `-16`, `-20`

Expected total: `3 * 6 * 7 = 126` conditions.

## Completion Check

All expected outputs were present.

| experiment | manifests | validation txt | metrics csv | fold prediction csv | loss png | bar png |
|---|---:|---:|---:|---:|---:|---:|
| `2025.06.11_0.3_2` | 42 | 42 | 42 | 126 | 126 | 84 |
| `2025.06.18_0.3_3` | 42 | 42 | 42 | 126 | 126 | 84 |
| `2025.07.09_0.3_1` | 42 | 42 | 42 | 126 | 126 | 84 |

Other checks:

- `tuning_summary.csv` files: `3`
- total summary rows: `126`
- missing experiment/frequency/noise combinations: `0`
- `stopped_by_memory_error`: `0`
- actual batch size: always `64`
- fold prediction files found for all rows: `3/3`

## Overall Results

Aggregate by experiment:

| experiment | n | mean R2 | median R2 | mean RMSE | mean ONB RMSE | mean F1 | mean ROC-AUC | best R2 | best F1 | best ONB RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2025.06.11_0.3_2` | 42 | 0.4796 | 0.5841 | 176016.2 | 96235.1 | 0.7752 | 0.9951 | 0.8452 | 0.8901 | 24835.8 |
| `2025.06.18_0.3_3` | 42 | 0.4081 | 0.4861 | 194933.3 | 107455.2 | 0.7441 | 0.9786 | 0.7491 | 0.8786 | 12973.3 |
| `2025.07.09_0.3_1` | 42 | 0.2889 | 0.5534 | 209888.1 | 118639.0 | 0.7504 | 0.9870 | 0.7887 | 0.9267 | 24535.2 |

Aggregate by max frequency:

| max frequency | n | mean R2 | mean RMSE | mean ONB RMSE | mean F1 | mean ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `2 kHz` | 21 | 0.0478 | 227984.9 | 164538.8 | 0.7419 | 0.9843 |
| `3 kHz` | 21 | 0.6160 | 161431.5 | 85690.7 | 0.7780 | 0.9846 |
| `5 kHz` | 21 | 0.3669 | 199914.9 | 100049.1 | 0.7542 | 0.9828 |
| `10 kHz` | 21 | 0.2881 | 211740.5 | 128656.8 | 0.7393 | 0.9871 |
| `15 kHz` | 21 | 0.5272 | 178427.9 | 78729.3 | 0.7655 | 0.9912 |
| `22 kHz` | 21 | 0.5073 | 182175.6 | 86993.8 | 0.7605 | 0.9914 |

Aggregate by noise condition:

| noise | n | mean R2 | mean RMSE | mean ONB RMSE | mean F1 | mean ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `no_noise` | 18 | 0.2328 | 208502.8 | 131791.6 | 0.7558 | 0.9366 |
| `SNR=0` | 18 | 0.0501 | 234123.7 | 170471.8 | 0.7188 | 0.9846 |
| `SNR=-4` | 18 | 0.2348 | 217584.1 | 139932.3 | 0.7313 | 0.9909 |
| `SNR=-8` | 18 | 0.4974 | 185141.9 | 97955.4 | 0.7440 | 0.9971 |
| `SNR=-12` | 18 | 0.4549 | 192162.0 | 93066.5 | 0.7615 | 0.9993 |
| `SNR=-16` | 18 | 0.5581 | 176295.6 | 63807.9 | 0.7630 | 0.9999 |
| `SNR=-20` | 18 | 0.7174 | 141477.7 | 55076.1 | 0.8217 | 0.9999 |

## Best Conditions

Top R2:

| experiment | max frequency | noise | R2 | RMSE | ONB RMSE | F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| `2025.06.11_0.3_2` | `3 kHz` | `SNR=-20` | 0.8452 | 104302.9 | 94517.6 | 0.8461 | 1.0000 |
| `2025.06.11_0.3_2` | `22 kHz` | `SNR=-20` | 0.8338 | 107991.9 | 61286.9 | 0.8901 | 1.0000 |
| `2025.07.09_0.3_1` | `15 kHz` | `no_noise` | 0.7887 | 128793.8 | 73134.7 | 0.8307 | 0.9146 |

Top F1:

| experiment | max frequency | noise | F1 | R2 | RMSE | ONB RMSE | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| `2025.07.09_0.3_1` | `3 kHz` | `SNR=-20` | 0.9267 | 0.7690 | 133351.1 | 43556.9 | 1.0000 |
| `2025.06.11_0.3_2` | `22 kHz` | `SNR=-20` | 0.8901 | 0.8338 | 107991.9 | 61286.9 | 1.0000 |
| `2025.06.11_0.3_2` | `2 kHz` | `SNR=-12` | 0.8893 | 0.7262 | 139218.5 | 24835.8 | 1.0000 |

Lowest ONB RMSE:

| experiment | max frequency | noise | ONB RMSE | R2 | RMSE | F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| `2025.06.18_0.3_3` | `3 kHz` | `SNR=-20` | 12973.3 | 0.7167 | 143591.1 | 0.8786 | 0.9998 |
| `2025.06.18_0.3_3` | `5 kHz` | `SNR=-16` | 24207.6 | 0.6149 | 168682.2 | 0.7491 | 0.9994 |
| `2025.07.09_0.3_1` | `15 kHz` | `SNR=-8` | 24535.2 | 0.6846 | 157129.1 | 0.8052 | 1.0000 |

## Fixed-Threshold Calibration Check

The fold prediction CSVs show that fixed ONB-threshold classification is still
strongly affected by absolute prediction-scale drift.

- conditions with overall predicted positive rate >= 0.95: `17/126`
- conditions with overall predicted positive rate <= 0.05: `0/126`
- conditions where at least one fold is almost all-positive: `111/126`
- missing fold prediction rows: `0`

Average predicted-positive behavior:

| experiment | true positive rate | predicted positive rate | mean absolute bias | all-positive conditions |
|---|---:|---:|---:|---:|
| `2025.06.11_0.3_2` | 0.5556 | 0.6858 | 121522.3 | 5 |
| `2025.06.18_0.3_3` | 0.5556 | 0.7225 | 123899.8 | 5 |
| `2025.07.09_0.3_1` | 0.5294 | 0.6957 | 119385.0 | 7 |

This means the model often ranks high/low heat-flux samples correctly, but the
absolute heat-flux scale is not stable enough for a fixed physical ONB
threshold.

## Interpretation

1. The run appears complete and mechanically valid.
   All 126 planned conditions were executed, all expected result files were
   saved, and there was no recorded memory-error fallback.

2. The model is not uniformly bad, but its reliability depends strongly on
   frequency/noise condition.
   The best region is around `3 kHz`, `15 kHz`, and `22 kHz`. The weakest
   region is `2 kHz`, followed by `10 kHz`.

3. The noise trend is suspicious.
   Results improve as SNR becomes more negative, with `SNR=-20` giving the best
   average R2/F1. If more negative SNR means stronger noise, this is not a
   physically natural result and should be treated as a warning sign.

4. Continuous ROC-AUC can be misleading here.
   ROC-AUC is frequently near 1.0, but many folds are nearly all-positive under
   the fixed ONB threshold. The model may be learning relative ordering while
   failing absolute calibration.

5. The current training setup still monitors training loss.
   `EarlyStopping(monitor="loss", restore_best_weights=False)` is not ideal for
   fold-wise calibration. A validation split and `val_loss` monitoring should be
   tested before drawing final conclusions about the Conformer architecture.

## Next Step

Recommended order:

1. Verify the SNR/noise generation meaning and confirm that `SNR=-20` is really
   the harshest condition.
2. Add inner validation for Keras training and use:
   - `EarlyStopping(monitor="val_loss")`
   - `restore_best_weights=True`
3. Rerun a reduced but diagnostic set:
   - `3 kHz`, `15 kHz`, `22 kHz`
   - `no_noise`, `SNR=0`, `SNR=-20`
   - all three experiment dates
4. Compare fixed-threshold predicted-positive rate, not only R2/F1/ROC-AUC.
5. Only after calibration improves, compare this canonical `conformer` against
   the previous `cnntf_v2_dim256_attn` baseline.
