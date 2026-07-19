# ファイル名・フォルダ名のゆるいルール

今までの名前に慣れている前提で、急に全部を英語化・標準化しない。新しく作るものだけ、少しずつ揃える。

## 基本ルール

- 新しく作る実験フォルダは `YYYY-MM-DD_short-name` にする。
- 研究報告や発表資料は、今まで通り日本語名でよい。
- コードや設定ファイルは、できるだけ英数字・アンダースコアを使う。
- 生成データの条件は、ファイル名だけでなく設定ファイルにも残す。

## おすすめの名前

### 実験結果フォルダ

```text
experiments/2026-05-21_rf_cnntransformer_22khz/
```

### 実験設定

```text
configs/experiments/2026-05-21_rf_cnntransformer_22khz.yaml
```

### データ設定

```text
configs/datasets/2025-07-09_0.3_1_waterflow_1s_22khz.yaml
```

### 結果まとめ

```text
experiments/2026-05-21_rf_cnntransformer_22khz/run_summary.md
experiments/2026-05-21_rf_cnntransformer_22khz/metrics.csv
```

## 既存名の扱い

- `code/3.run_ensemble_ROC_100%_analysis.py` のような慣れている中心コードは、急に改名しない。
- 古いコードは、移動する前に `docs/code_map.md` に役割を書く。
- `trush_box` はすぐに変えなくてよい。将来的に整理するときは `legacy/` に寄せる。

## 避けたい名前

- `tmp`, `new`, `final`, `final2`, `最新版` だけの名前。
- 実験条件が分からない `result`, `output`, `graph` だけの名前。
- 日付なしの重要結果フォルダ。

## 迷ったときの型

```text
YYYY-MM-DD_対象_手法_条件
```

例:

```text
2026-05-21_heatflux_rf-cnntransformer_waterflow-1s-22khz
```
