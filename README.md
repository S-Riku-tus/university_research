# 研究ワークスペース

このディレクトリは、沸騰実験の音響データ解析、機械学習モデルによる熱流束推定、沸騰開始点検知、研究進捗報告資料をまとめた研究用ワークスペースです。

## 研究の中心

現在の主題は、実験音から作成したスペクトログラムを入力として、熱流束を回帰予測し、予測値をもとに沸騰開始点を検知できるかを検証することです。学部研究では、スペクトログラムとCNN回帰による沸騰検知の基本的成立性を確認しました。修士研究では、沸騰開始点近傍で見られる特徴を、入力表現やモデルがどのように捉えるかを解析し、ノイズ下や条件変化下でも物理的に妥当な根拠に基づいて早期検知できるかを検証します。

## 主なディレクトリ

- `code/`: 解析・前処理・機械学習・評価に使うコード。
- `Pool_boiling/`: プール沸騰実験に関係するデータや結果。
- `water_flow/`: 水流音など、ノイズ付与に使う音源。
- `研究進捗報告/`: 週次進捗、発表資料、論文輪講、学会資料、卒論・修論関連資料。
- `docs/`: Codexや自分が研究文脈を素早く把握するための要約・索引・運用メモ。
- `configs/`: 新しい実験やデータセット条件を残すための設定テンプレート。
- `experiments/`: 実験結果の要約、評価指標、図、考察を残す場所。
- `templates/`: 週次報告など、繰り返し作成する文書のテンプレート。

## Codexに読ませる入口

Codexに研究を手伝わせるときは、まず `AGENTS.md` と `docs/research_context.md` を読ませます。週次進捗報告を作るときは、`docs/progress/weekly_report_guide.md` と `templates/weekly_progress_SOAP.md` を使います。過去報告を参照するときは、重いPDFやPPTXを直接すべて読ませる前に、`docs/progress_index.md` を更新してから使います。

## 現在の主結果

- [2026-09-02 selected log architecture](experiments/2026-09-02_selected_log_architecture/README.md): 修正済み水流音データを用いた単体モデル・アンサンブル・ONB検知・説明性の最新スナップショット。実行は56/105条件で中断している。
- [waterflow_20260817_1s](configs/datasets/waterflow_20260817_1s.yaml): 現行学習データの生成条件、件数、整合性確認、旧バグデータの除外方針。条件別監査値は[データ監査スナップショット](experiments/2026-08-17_waterflow_dataset_snapshot/README.md)に保存。
- [log-powerモデル構造選定](experiments/2026-08-30_log_power_architecture_study.md): CNN+TransformerとAlexNetにlog-power入力を採用した根拠。

`Pool_boiling/` 以下のデータ・重み・詳細画像は容量上Git対象外です。研究上必要な設定と数値は `configs/` と `experiments/` に軽量スナップショットとして保存します。

## 管理用の地図

- `docs/repository_management.md`: このリポジトリ全体の管理方針。
- `docs/code_map.md`: `code/` のどこに何があるかの地図。
- `docs/data_inventory.md`: 巨大データの場所・意味・規模の索引。
- `docs/naming_rules.md`: 新しく作るファイル名・フォルダ名のゆるいルール。
- `configs/README.md`: 実験条件を設定ファイルとして残すための入口。
- `experiments/README.md`: 実験結果の要約を残すための入口。

## よく使う作業

- 週次進捗報告の下書き作成。
- 実験結果の整理と次の計画作成。
- モデル構成、評価指標、結果図の説明文作成。
- 学会・中間発表・論文輪講のストーリー整理。
- コードの処理内容、問題点、改善方針の確認。
