# 2026-07-06 v2 GAP fixed-parameter full grid with explainability hook

> **履歴・適用範囲（2026-09-15追記）**: この文書は当時の条件・判断の記録です。本文の「現在」「次にやること」は作成時点を指します。最新の設定・完了範囲・優先順位は[研究の現在地](../docs/research_status.md)、更新関係は[文書案内](../docs/document_index.md)を参照してください。

## Purpose

The CNN+Transformer v2 GAP model recovered usable regression performance with
`lr=0.0005` and `batch_size=32`. Because time is limited, this run fixes that
parameter set and expands the evaluation to the full condition grid before doing
another parameter confirmation pass.

## Active learning plan

- Model: `cnntf_v2_gap`
- Parameter set: `lr=0.0005`, `batch_size=32`
- Experiments:
  - `2025.06.18_0.3_3`
  - `2025.07.09_0.3_1`
  - `2025.06.11_0.3_2`
- Frequencies:
  - `maxfreq=2kHz`
  - `maxfreq=3kHz`
  - `maxfreq=5kHz`
  - `maxfreq=10kHz`
  - `maxfreq=15kHz`
  - `maxfreq=22kHz`
- Noise conditions:
  - `heatflux_no_noise`
  - `heatflux_SNR=0`
  - `heatflux_SNR=-4`
  - `heatflux_SNR=-8`
  - `heatflux_SNR=-12`
  - `heatflux_SNR=-16`
  - `heatflux_SNR=-20`
- Folds: 3
- Planned model fits: `3 * 6 * 7 * 1 * 3 = 378`

## Explainability integration

Explainability is now called from `code/run_ensemble_regression_onb.py` after a
trained fold model has produced validation predictions. The main script only
passes model, validation data, predictions, and metadata; the internal
explainability processing lives in `code/utils/explainability/`.

Default explainability setting is disabled for this full grid because
Integrated Gradients, Grad-CAM, and occlusion maps on every condition would add a
large runtime and storage cost. To save explanations during training, set
`VALIDATION_CONFIG["explainability"]["enabled"] = True` and start with
`target_folds = [1]` and a small `max_samples_per_fold`.

## Verification

The configured plan was checked by importing `run_ensemble_regression_onb.py`:

- existing datasets: 126 / intended datasets: 126
- active model keys: `["cnntf_v2_gap"]`
- parameter sets: `["lr0p0005_bs32"]`
- explainability enabled: `False`
- result directory: `20260706_v2gap_lr0p0005_bs32_full_grid`

The edited Python files passed `python -m py_compile`.
