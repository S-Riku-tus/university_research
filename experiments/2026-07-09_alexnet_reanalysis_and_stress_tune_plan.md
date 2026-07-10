# 2026-07-09 AlexNet reanalysis and stress-tune plan

## Why this check is needed

The previous full-grid ensemble result completed all 126 conditions, but AlexNet
was weaker than expected:

- mean R2 across 126 conditions: `0.6934`
- median R2: `0.8778`
- 42 / 126 conditions had R2 above `0.90`
- 6 / 126 conditions had negative R2

This means AlexNet is not uniformly poor. It works well on many conditions but
collapses on a small set of stress conditions, which pulls down the mean.

## Where AlexNet failed

The weakest patterns were:

- `SNR=0`: mean R2 `0.3080`
- `SNR=-4`: mean R2 `0.4528`
- `maxfreq=2kHz`: mean R2 `0.4204`
- `maxfreq=5kHz`: mean R2 `0.3787`

The strongest patterns were:

- `SNR=-20`: mean R2 `0.9128`
- `SNR=-16`: mean R2 `0.8800`
- `maxfreq=15kHz`: mean R2 `0.8868`
- `maxfreq=22kHz`: mean R2 `0.8826`

All AlexNet runs completed 300 epochs with actual batch size 32. Therefore the
main issue is not OOM or interruption; it is likely a mismatch between the
current AlexNet learning setting and some input/noise/frequency conditions.

## Current code setting

`code/run_ensemble_regression_onb.py` is now set to an AlexNet-only stress tune:

- active model: `alexnet`
- experiments: all 3 current experiments
- frequencies: `2kHz`, `5kHz`, `15kHz`, `22kHz`
- noise conditions: `no_noise`, `SNR=0`, `SNR=-4`, `SNR=-20`
- parameter candidates:
  - `alex_lr0p005_bs32`
  - `alex_lr0p001_bs32`
  - `alex_lr0p0005_bs32`
  - `alex_onb_lr0p0001_bs128`
- planned fits: `48 conditions * 4 candidates * 3 folds = 576`
- output group: `alexnet`
- result directory: `20260709_alexnet_stress_tune`

## Conformer handling

The current stable Conformer/CNN+Transformer setting remains:

- `cnntf_v2_gap`: `lr=0.0005`, `batch_size=32`

It is not re-run during this AlexNet-only check to avoid wasting time. After
AlexNet is selected, the final ensemble should use RF fixed parameters,
`cnntf_v2_gap lr=0.0005 bs=32`, and the best AlexNet setting from this tune.

## Next decision rule

Choose the AlexNet setting that improves the failure cases without sacrificing
the already-good high-noise/high-frequency cases. Prioritize:

1. higher median R2 and fewer negative R2 cases,
2. better `SNR=0` and `SNR=-4`,
3. lower ONB RMSE,
4. stable F1/recall.
