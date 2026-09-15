# 2026-07-06 v2 GAP tuning resume note

> **履歴・適用範囲（2026-09-15追記）**: この文書は当時の条件・判断の記録です。本文の「現在」「次にやること」は作成時点を指します。最新の設定・完了範囲・優先順位は[研究の現在地](../docs/research_status.md)、更新関係は[文書案内](../docs/document_index.md)を参照してください。

## Context

The `cnntf_v2_gap` learning-rate and batch-size tuning run stopped after
`lr0p0005_bs24` because NumPy could not allocate the next train split array.
The completed results were already written to `tuning_summary.csv`.

## Change

`code/run_ensemble_regression_onb.py` now has
`VALIDATION_CONFIG["output"]["resume_completed_runs"] = True`.

When enabled, each parameter-set run is skipped only if both files are present:

- the row for `run_dir` exists in `tuning_summary.csv`
- `metrics_summary_<snr>.csv` exists under that run directory

This lets the same script resume from the first unfinished condition without
recomputing completed parameter sets.

The CSV/text save paths now use a Windows long-path fallback, and fold
prediction files are written under `fold_pred/` instead of `fold_predictions/`
to keep the deepest paths shorter.

## Verified State

Initial check for `2025.06.18_0.3_3 / maxfreq=22kHz / heatflux_no_noise`:

- completed runs: 17
- remaining runs: 13
- first remaining run: `e300_lr0p0005_bs32_cnntf_v2_gap`

After the next interruption at `lr0p00005_bs12`:

- completed runs: 25
- remaining runs: 5
- first remaining run: `e300_lr0p00005_bs12_cnntf_v2_gap`
