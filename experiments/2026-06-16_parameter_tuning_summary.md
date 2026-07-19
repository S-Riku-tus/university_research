# 2026-06-16 parameter tuning summary

## Scope

- Data: `heatflux_no_noise`, `maxfreq=22kHz`
- Models: `RandomForest`, `CNN+Tf (AttnPool)`, `AlexNet`
- RF: fixed at `{n_estimators: 300, max_depth: 8, subsample: 0.8, colsample_bynode: 0.6}`
- Keras grid: 42 conditions from 6 batch sizes x 7 learning rates
- Run split: the first 37 conditions were saved under `20260615_...`; the restarted last 5 conditions were saved under `20260616_...`.
- Aggregation rule: use the latest complete result for each `parameter_set`; ignore the interrupted partial `20260615_ep500_k_lr0p00005_bs32_simple_3m`.

## Main Results

| target | best condition | R2 | RMSE all | RMSE ONB | ROC-AUC cont | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RF fixed baseline | all Keras conditions share the same RF | 0.9260 | 75568.9 | 79986.6 | 0.9688 | 0.8990 |
| CNN+Tf regression | `lr=0.01, batch_size=24` | 0.8611 | 103321.8 | 128867.4 | 0.8891 | 0.8895 |
| CNN+Tf ONB/ROC | `lr=0.0001, batch_size=128` | 0.7655 | 134656.2 | 20724.7 | 0.9235 | 0.7909 |
| AlexNet regression | `lr=0.005, batch_size=32` | 0.8522 | 106553.9 | 118443.8 | 0.8856 | 0.8682 |
| AlexNet ONB | `lr=0.0001, batch_size=128` | 0.7467 | 140147.4 | 27453.3 | 0.8944 | 0.8427 |
| same-grid simple ensemble regression | `lr=0.005, batch_size=12` | 0.8945 | 90119.4 | 116413.9 | 0.9693 | 0.8872 |

## Post-hoc Ensemble Check

The saved `fold_predictions` files allow reweighting without retraining. Combining the per-model regression-best predictions gave:

| ensemble setting | R2 | RMSE all | RMSE ONB | ROC-AUC cont | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF only | 0.9260 | 75568.9 | 79986.6 | 0.9688 | 0.8990 |
| per-model best, equal weights | 0.8966 | 89211.4 | 106888.1 | 0.9693 | 0.8851 |
| per-model best, weights `(rf, cnntf, alex)=(0.90,0.05,0.05)` | 0.9245 | 76304.2 | 83190.9 | 0.9689 | 0.8953 |
| per-model best, weights `(0.95,0.05,0.00)` | 0.9255 | 75811.8 | 81784.1 | 0.9688 | 0.8987 |

## Interpretation

- The strongest heat-flux regressor is still RF. A simple 3-model average reduces RF's regression accuracy.
- CNN+Tf and AlexNet each learned useful signals, but their best regression settings do not beat RF.
- Very low ONB RMSE appears at low learning rates and large batch sizes, but those settings often sacrifice global R2/RMSE. Treat ONB-only wins as diagnostic, not final model selection.
- For a 3-model ensemble demonstration, RF-dominant fixed weights are more defensible than equal weights.

## Next Run Prepared

`code/run_ensemble_regression_onb.py` is now set to one confirmation condition:

- `cnntf_v1`: `lr=0.01`, `batch_size=24`
- `alexnet`: `lr=0.005`, `batch_size=32`
- `WEIGHT_STRATEGY="fixed"`
- `FIXED_WEIGHTS={"rf": 0.90, "cnntf_v1": 0.05, "alexnet": 0.05}`

This run tests whether a 3-model RF-dominant ensemble can stay close to the RF baseline while keeping CNN/AlexNet available for model comparison and later interpretation.

## Recommended Next Steps

1. Do not rerun the full 42-condition grid immediately.
2. Run the prepared one-condition confirmation if a fresh, single-folder result is needed.
3. For final reporting, keep RF as the main performance baseline and treat CNN+Tf/AlexNet as comparison and explanation models unless a later ensemble beats RF on a held-out condition.
4. Before claiming final performance, validate on another experimental condition or a stricter split, because the current choices were selected from the same 5-fold tuning results.
5. Use the CNN+Tf and AlexNet runs for Grad-CAM/IG-style ONB-region analysis, especially around cases where RF is accurate but neural predictions differ.
