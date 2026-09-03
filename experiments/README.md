# experiments

実験結果の「要約」を置く場所です。巨大な生成データや全画像をここに入れる必要はありません。各実験について、条件、結果、解釈、関連ファイルだけを残します。

## 使い方

新しい実験を行ったら、次のようなフォルダを作る。

```text
experiments/YYYY-MM-DD_short-name/
  config.yaml
  run_summary.md
  metrics.csv
  figures/
```

実際の巨大データや重みは `Pool_boiling/` 側に置いたままでよいです。ここには、研究報告や修論で使うための地図を残します。

## 主要な結果

- [`2026-09-02_selected_log_architecture/`](2026-09-02_selected_log_architecture/README.md): 現行データ・選定済みlog-power構造による最新ONB実行。56/105条件まで完了。
- [`2026-08-17_waterflow_dataset_snapshot/`](2026-08-17_waterflow_dataset_snapshot/README.md): 現行データ105条件の件数、実現SNR、paired-noise、元WAV RMSの監査記録。
- [`2026-08-30_log_power_architecture_study.md`](2026-08-30_log_power_architecture_study.md): 現行CNN+Transformer・AlexNet構造の選定根拠。
- [`2026-07-24_noise_shortcut_diagnostic/`](2026-07-24_noise_shortcut_diagnostic/README.md): ノイズと精度の逆転現象の診断記録。

## 保存するものと保存しないもの

Gitには、実行条件、完了範囲、主要指標、比較表、解釈上の制約、元結果を識別するハッシュを保存する。モデル重み、全画像、巨大なfold予測、再生成可能な中間ファイルは保存しない。

重要な実行結果が `Pool_boiling/**/regression_result/` に出た場合は、`code/export_ensemble_result_snapshot.py`のような抽出処理を使い、専用ディレクトリへ固定スナップショットを作る。実行日だけで内容が変わる「最新結果」ファイルにはせず、一度報告・判断に使った数値を後から再確認できる形にする。
