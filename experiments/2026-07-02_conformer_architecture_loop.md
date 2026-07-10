# Conformer architecture loop on 22kHz/no_noise

Date: 2026-07-02

## Fixed data condition

The data subset was kept fixed as requested:

- Experiments: `2025.06.18_0.3_3`, `2025.07.09_0.3_1`, `2025.06.11_0.3_2`
- Frequency: `maxfreq=22kHz`
- Noise: `heatflux_no_noise`
- Model: Conformer only
- Training: 3-fold CV, 300 epochs, `lr=0.0001`, `batch_size=64`

## Architecture candidates

Baseline from `20260701_conformer`:

- `base_t7_d256_k31_b4`
- Time tokens after CNN: 7
- Approx. parameters: 14.48M
- Conformer convolution kernel: 31

Iteration 1, `20260702_cfa`:

- `t28d128k15b3`: 28 time tokens, 1.79M params, kernel 15
- `t56d64k15b2`: 56 time tokens, 0.52M params, kernel 15

Iteration 2, `20260702_cfa2`:

- `t14d256k15b4`: 14 time tokens, 8.64M params, kernel 15
- `t28d256k15b3`: 28 time tokens, 6.01M params, kernel 15

Iteration 3, `20260702_cfa3`:

- `t7d256k7b4`: 7 time tokens, 14.46M params, kernel 7

## Mean results over the three experiments

| group | parameter_set | R2 | RMSE all | RMSE ONB | ROC-AUC cont | F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_7token | base_t7_d256_k31_b4 | 0.5039 | 183974.95 | 78270.01 | 0.9417 | 0.7817 |
| arch_iter3_k7 | t7d256k7b4 | 0.2095 | 230782.66 | 119126.18 | 0.9409 | 0.7232 |
| arch_iter2_mid | t14d256k15b4 | 0.2035 | 233069.78 | 109336.58 | 0.9360 | 0.7358 |
| arch_iter2_mid | t28d256k15b3 | 0.1897 | 234810.38 | 92571.46 | 0.9405 | 0.7093 |
| arch_iter1_light | t28d128k15b3 | 0.0912 | 253149.52 | 155599.12 | 0.9376 | 0.7048 |
| arch_iter1_light | t56d64k15b2 | -0.5459 | 299881.28 | 180619.28 | 0.9508 | 0.7600 |

## Interpretation

The original 7-token Conformer remained the best overall architecture for this
fixed condition. Increasing the number of time tokens did not improve the
average regression result. Reducing the Conformer convolution kernel from 31 to
7 also did not improve the result, so the large kernel alone is unlikely to be
the main cause of the current behavior.

The new variants sometimes improved ONB RMSE for a single experiment, but this
did not generalize across the three experiment dates. Several variants also
increased the predicted-positive rate above the true positive rate, indicating
that fixed-threshold ONB calibration is still the weak point.

## Current decision

The active code was restored to the stable baseline-style `main` parameter set.
The new `conformer()` options remain available:

- `time_reduction`
- `frontend_filters`

These make future architecture checks easier without changing the fixed data
condition lists.

## Next step

The next logical target is not another architecture sweep. The evidence points
more strongly to training/calibration:

1. Add an inner validation split for Keras models.
2. Monitor `val_loss` instead of training `loss`.
3. Use `restore_best_weights=True`.
4. Re-run the same fixed condition and compare R2, ONB RMSE, predicted-positive
   rate, and fold-level collapse.
