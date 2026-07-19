# 2026-06-22 storage cleanup

## Purpose

The C drive had reached 0 MB free space. This cleanup removed data that was clearly outside the current research line or duplicated large generated artifacts.

The current research line is:

- acoustic spectrogram or STFT-like `.npy`
- heat-flux regression
- ONB threshold evaluation
- main condition: `Pool_boiling/Subcooling_20_degrees/0.3`

## Deleted

Large folders removed as unused for the current thesis direction:

- `High_speed_compare/`
- `Pool_boiling/Subcooling_10_degrees/`
- `Pool_boiling/Subcooling_20_degrees/0.4/`
- `Pool_boiling/Subcooling_20_degrees/0.5/`

Large duplicated or intermediate weight folders removed:

- `Pool_boiling/Subcooling_20_degrees/0.3/2024.11.12_1_2.13_1/regression_result/npy/ensemble/heatflux_no_noise/weight_average/100%/all_weights`
- `Pool_boiling/Subcooling_20_degrees/0.3/2024.11.12_1/all_weights`
- low-epoch, optimizer-comparison, `last*`, `tune`, empty, and wave trial folders under `Pool_boiling/Subcooling_20_degrees/0.3/2024.11.12_1_2.13_1/all_weights`

Old individual `0.3` experiment folders removed after keeping the larger combined folder:

- `2024.09.26_1`
- `2024.9.30_1`
- `2024.9.30_2`
- `2024.10.2_1_120s`
- `2024.10.30_1`
- `2024.11.05_1_矩形水槽`
- `2024.11.05_2`
- `2024.11.08_1`
- `2024.11.11_2`
- `2024.11.12_1`
- `2024.11.12_2`
- `2024.11.13_1`
- `2024.11.13_4_マイク向き変更`

## Kept

These folders were intentionally kept:

- `Pool_boiling/Subcooling_20_degrees/0.3/2025.07.09_0.3_1`
  - current main dataset used by the recent regression and ONB evaluation work
- `Pool_boiling/Subcooling_20_degrees/0.3/2024.11.12_1_2.13_1`
  - older combined dataset and model-weight source; still useful for comparison or reproduction
- `Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2`
  - possible cross-day validation candidate referenced by current scripts/config comments
- `Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3`
  - small possible validation candidate
- `water_flow/`
  - source water-flow noise audio
- `研究進捗報告/`
  - progress reports and presentation material

## Current Notes

After cleanup, the main remaining storage pressure is still:

- `.npy` generated features
- `.h5` model weights
- generated `.png` spectrogram or result images

Do not remove `2025.07.09_0.3_1/data/npy/waterflow_20251219_1s` unless the current baseline/ONB experiments have been migrated or regenerated elsewhere.

If more space is needed, next candidates should be selected in this order:

1. unused max-frequency folders under generated `.npy` data, especially low-performing frequency limits
2. non-final `.h5` weight sets not tied to a reported result
3. generated `.png` spectrogram folders, while keeping only figures used in reports or papers

Old scripts under `code/trush_box/` may still contain paths to deleted historical folders. Treat them as obsolete notes rather than runnable current scripts.
