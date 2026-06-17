# 2026-06-17 legacy val-fold comparison analysis

## Scope

- Result path: `Pool_boiling/Subcooling_20_degrees/0.3/2025.07.09_0.3_1/regression_result/npy/ensemble/20260617/heatflux_no_noise/maxfreq=22kHz/ep500_legacy_val_fold_legacy_3m`
- Data: `2025.07.09_0.3_1`, `heatflux_no_noise`, `maxfreq=22kHz`
- Threshold: `275174.6641`
- Models: `RandomForest`, `CNN+Tf (AttnPool)`, `AlexNet`, `Ensemble`
- Keras parameters: `CNN+Tf lr=0.01 batch_size=24`, `AlexNet lr=0.005 batch_size=32`
- Legacy settings: `early_stopping=False`, `WEIGHT_STRATEGY="val_fold_legacy"`

`val_fold_legacy` uses the validation fold error to decide ensemble weights. This is useful for comparison with the old script, but should be treated as a leakage condition and not used as final evidence.

## Main Metrics

| model | R2 | RMSE all | RMSE ONB | AUC binary | ROC-AUC cont | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9260 +/- 0.0043 | 75568.9 +/- 1631.4 | 79986.6 +/- 6009.6 | 0.9080 +/- 0.0123 | 0.9688 +/- 0.0035 | 0.8990 +/- 0.0147 |
| CNN+Tf (AttnPool) | 0.6239 +/- 0.2023 | 153852.0 +/- 38275.1 | 105784.3 +/- 21402.1 | 0.8154 +/- 0.0635 | 0.8813 +/- 0.0057 | 0.7336 +/- 0.1284 |
| AlexNet | 0.8454 +/- 0.0097 | 109116.0 +/- 1707.5 | 97645.9 +/- 8044.0 | 0.8886 +/- 0.0120 | 0.8899 +/- 0.0106 | 0.8738 +/- 0.0151 |
| Ensemble | 0.9051 +/- 0.0049 | 85581.4 +/- 1194.5 | 87996.7 +/- 6657.7 | 0.8983 +/- 0.0120 | 0.9690 +/- 0.0035 | 0.8860 +/- 0.0149 |

## Weight Behavior

Fold-wise legacy weights were:

| fold | RF | CNN+Tf | AlexNet |
| --- | ---: | ---: | ---: |
| 1 | 0.490 | 0.249 | 0.261 |
| 2 | 0.508 | 0.253 | 0.239 |
| 3 | 0.630 | 0.043 | 0.327 |
| 4 | 0.579 | 0.153 | 0.268 |
| 5 | 0.534 | 0.241 | 0.225 |

Average weights were approximately `(RF, CNN+Tf, AlexNet)=(0.548, 0.188, 0.264)`. The legacy strategy gave much larger weights to the neural models than the later fixed RF-dominant setting.

## Interpretation

- RandomForest remains the strongest model for heat-flux regression in this condition. It has the best R2, RMSE all, RMSE ONB, binary AUC, and F1.
- The legacy ensemble did not beat RandomForest, even though its weights were chosen from the validation fold. This suggests that the neural models are not correcting RF errors enough in this condition.
- The ensemble ROC-AUC continuous is almost identical to RF and slightly higher numerically (`0.9690` vs `0.9688`), but regression and binary detection metrics are lower than RF. This means the ranking of boiling/non-boiling samples is preserved, while thresholded heat-flux accuracy is not improved.
- CNN+Tf is unstable with `early_stopping=False`. Fold 3 dropped to `R2=-0.1840`, which strongly inflates the mean error and standard error. AlexNet is more stable than CNN+Tf but still below RF.
- Compared with the 2026-06-16 fixed RF-dominant run `(0.90, 0.05, 0.05)`, the fixed ensemble is more defensible: `R2=0.9243`, `RMSE all=76406.2`, `RMSE ONB=81620.3`, and `F1=0.8963`. It stays close to RF while the legacy ensemble falls to `R2=0.9051`.

## Current Takeaway

For this 22 kHz no-noise condition, the result supports using RandomForest as the main performance baseline. The 3-model ensemble is useful for comparison and interpretation, but the safest ensemble setting is RF-dominant fixed weighting rather than legacy validation-fold weighting.

Before making a final research claim, this should still be checked on another experimental day, another noise condition, or a stricter split.
