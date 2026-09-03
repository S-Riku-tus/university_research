# リポジトリ管理方針

このリポジトリは、今まで慣れてきた配置を大きく崩さずに、研究の再現性と見通しを少しずつ上げる方針で管理する。

## 基本方針

- 既存の `code/`, `Pool_boiling/`, `High_speed_compare/`, `研究進捗報告/` は急に移動しない。
- 新しい整理は、まず `docs/`, `configs/`, `experiments/`, `templates/` に追加する。
- ファイルを探すときは、まず `README.md`, `AGENTS.md`, `docs/code_map.md`, `docs/data_inventory.md` を見る。
- 巨大データや生成物はGitで直接管理せず、場所・条件・意味を軽量なMarkdownやYAMLで記録する。
- コードを大きく書き換える前に、実験条件と出力先を `configs/` か `experiments/` に残す。

## 役割分担

```text
code/              実行するコード、過去コード、モデル定義
docs/              研究の文脈、データ索引、コード地図、運用ルール
configs/           実験条件・データ条件のテンプレートや設定
experiments/       実験結果の要約、指標、図、考察の置き場
templates/         進捗報告など繰り返し作る文書の型
研究進捗報告/       週次報告・発表資料・論文輪講・学会資料の原本
Pool_boiling/      実験データ・生成データ・解析結果の本体
High_speed_compare/過去または比較用の大規模データ
```

## 研究中のおすすめ作業手順

1. 新しい実験を思いついたら、まず `configs/experiments/` に設定ファイルを作る。
2. 実験を実行したら、`experiments/YYYY-MM-DD_short-name/` に `run_summary.md` を作る。
3. 結果の図や数値を出したら、`docs/experiment_log.md` に要点だけ追記する。
4. 週次報告前に、`docs/progress_index.md` を更新する。
5. 修論や発表資料を書くときは、`docs/` と `experiments/` から材料を集める。

## Gitで管理するもの

- Pythonコード、Notebook、Markdown、設定YAML。
- 週次報告や発表資料の完成版。
- 小さな表、重要な最終図、研究の根拠になる軽量ファイル。

## Gitで管理しないもの

- `.npy`, 大量の `.png`, 学習済み重み, ログ, キャッシュ。
- 再生成できる中間データ。
- Pythonの `__pycache__` や `.pyc`。

## Git対象外の重要結果を残す方法

`Pool_boiling/`、`regression_result/`、学習データ本体はGit対象外でも、そこから得た研究上の判断までローカルだけに置かない。重要な実行ごとに `experiments/YYYY-MM-DD_short-name/` を作り、次を保存する。

- `README.md`: 目的、完了範囲、主要結果、結論、解釈上の制約。
- `run_config.yaml`: データ、分割、モデル、しきい値、実行条件。
- 小さなCSV: 条件別指標、集計値、比較差。全予測ではなく主張を検証できる列を残す。
- `snapshot_manifest.json`: 元結果の相対パス、行数、SHA-256、生成ファイル一覧。
- 必要な場合だけ代表図。全fold・全サンプル画像は保存しない。

数値はMarkdownへ手入力するだけで終わらせず、可能なら抽出スクリプトも `code/` に残す。修論で使用しない診断値やリークを含む結果は削除せず、`claim_safe: false`などで明示して主結果から区別する。

## すぐに整理しないもの

- `trush_box` など、名前は気になるが使い慣れているフォルダ。
- 年度別の `研究進捗報告/`。
- 既存コード中の絶対パス。

これらは一気に直すより、触るタイミングで少しずつ `docs/code_map.md` や `configs/` に情報を移す。
