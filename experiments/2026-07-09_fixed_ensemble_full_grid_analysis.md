# 2026-07-09 fixed ensemble full-grid result analysis

## Result location

Completed result directory:

`Pool_boiling/Subcooling_20_degrees/0.3/<experiment>/regression_result/npy/ensemble/20260706_fixed_ensemble_full_grid`

Although the current script date would now generate `20260709_fixed_ensemble_full_grid`,
the completed files analyzed here are under `20260706_fixed_ensemble_full_grid`.

## Completeness

All three experiments have complete outputs:

- `2025.06.18_0.3_3`: 42 metrics files, 42 ensemble weight files, 42 manifests
- `2025.07.09_0.3_1`: 42 metrics files, 42 ensemble weight files, 42 manifests
- `2025.06.11_0.3_2`: 42 metrics files, 42 ensemble weight files, 42 manifests

Total evaluated conditions:

- 3 experiments
- 6 max-frequency settings
- 7 noise conditions
- 126 data conditions
- 4 reported model rows per condition: `rf`, `cnntf_v2_gap`, `alexnet`, `ensemble`

## Important caveat

The run directory is:

`e300_rf_v2_alex_rf-cnntf_v2__vleg`

Therefore this run used `weight_strategy="val_fold_legacy"`, not the fixed
`rf=0.90, cnntf_v2_gap=0.05, alexnet=0.05` weights. The result is useful for
legacy comparison, but it should not be used as final evidence because the
validation fold labels are used to choose the ensemble weights.

## Overall mean across 126 conditions

| model | R2 | RMSE all | RMSE ONB | ROC-AUC cont | F1 |
|---|---:|---:|---:|---:|---:|
| rf | 0.9706 | 44687.9 | 42917.0 | 0.9967 | 0.9775 |
| ensemble | 0.9647 | 49083.2 | 52641.7 | 0.9967 | 0.9493 |
| cnntf_v2_gap | 0.8420 | 105831.1 | 91268.4 | 0.9856 | 0.8523 |
| alexnet | 0.6934 | 111117.2 | 120903.8 | 0.9873 | 0.8525 |

## Interpretation

RF is still the strongest model. It had the best R2 in 124 of 126 conditions.
The ensemble exceeded RF in only 2 conditions, both on `2025.06.11_0.3_2` with
`SNR=-20` and high-frequency settings, and the margin was very small.

The ensemble is stable and much better than either deep model alone, but it is
not improving on RF overall. Because this is the legacy validation-fold weighting
strategy, even this small improvement in two conditions should be treated as
diagnostic rather than final evidence.

Noise-added conditions produced higher mean R2 than no-noise conditions for RF
and ensemble. This likely reflects dataset/label distribution differences and
should not be interpreted simply as noise improving the model without checking
the target distribution and fold predictions.

## Next actions

1. Re-run the same full grid with `weight_strategy="fixed"` for a leakage-free
   final comparison.
2. Keep RF as the main performance baseline.
3. Use the ensemble as a robustness/diagnostic comparison only if fixed weights
   stay close to RF across noise and frequency conditions.
4. After fixed-weight results are complete, compare RF vs fixed ensemble on:
   R2, RMSE ONB, F1/recall, and high-noise stability.
5. Then run explainability on selected representative conditions rather than on
   all 126 conditions.
