# 2026-07-10 AlexNet 22kHz noise tuning analysis

## Scope

Current `code/run_ensemble_regression_onb.py` output analyzed here:

- model: `alexnet` only
- experiments: `2025.06.18_0.3_3`, `2025.07.09_0.3_1`, `2025.06.11_0.3_2`
- frequency: `maxfreq=22kHz`
- noise: `heatflux_no_noise`, `SNR=0/-4/-8/-12/-16/-20`
- parameter candidates:
  - `lr=0.005, batch_size=32`
  - `lr=0.001, batch_size=32`
  - `lr=0.0005, batch_size=32`
  - `lr=0.0001, batch_size=128`

All three experiments produced 28 `metrics_summary` files, for 84 rows total.

## Parameter comparison

| parameter | n | mean R2 | median R2 | min R2 | max R2 | R2>=0.9 | mean ONB RMSE | mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lr=0.005, bs=32 | 21 | 0.7927 | 0.8843 | 0.1375 | 0.9422 | 8 | 121264 | 0.8453 |
| lr=0.001, bs=32 | 21 | 0.7146 | 0.7310 | 0.4655 | 0.8552 | 0 | 62084 | 0.7922 |
| lr=0.0005, bs=32 | 21 | 0.6142 | 0.6501 | 0.2323 | 0.8492 | 0 | 86434 | 0.7563 |
| lr=0.0001, bs=128 | 21 | 0.5898 | 0.7373 | 0.0491 | 0.8338 | 0 | 60178 | 0.7775 |

## Main findings

- `lr=0.005, bs=32` is the strongest candidate by R2. It wins 17 of 21
  experiment/noise conditions and is the only candidate that reaches R2 >= 0.9.
- Lower learning rates reduce the worst ONB RMSE in some conditions, but their
  regression R2 is clearly weaker.
- The current run did not reproduce the previous AlexNet 22kHz average exactly.
  Compared with the previous full-grid AlexNet 22kHz result, the best current
  candidate is almost neutral on average: mean R2 difference `-0.0112`, improved
  in 10 of 21 conditions.
- The largest current failure is `2025.06.18_0.3_3 / SNR=-4`: previous AlexNet
  R2 was `0.8996`, but the best current R2 is `0.6738`. This condition should
  be checked before finalizing AlexNet.

## Interpretation

For final ensemble use, `lr=0.005, batch_size=32` is still the most defensible
single fixed AlexNet setting if the priority is overall heat-flux regression.
However, AlexNet remains less stable than RF and `cnntf_v2_gap`, especially when
judged by ONB-neighborhood error.

The next step should not be a wider blind AlexNet search. A more useful check is
to rerun only a small stability subset around the failure conditions, especially:

- `2025.06.18_0.3_3 / maxfreq=22kHz / SNR=-4`
- `2025.06.11_0.3_2 / maxfreq=22kHz / SNR=0`
- `2025.06.18_0.3_3 / maxfreq=22kHz / no_noise`

If these remain unstable, keep AlexNet at a low ensemble weight and treat it as
a complementary model rather than a main predictor.
