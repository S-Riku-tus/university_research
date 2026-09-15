# 研究ワークスペース

音響スペクトログラムから熱流束を予測し、プール沸騰のONB（核沸騰開始）近傍を検知する研究。予測精度、ノイズ下の安定性、判断根拠の物理的妥当性を調べます。

## まず読むページ

| 知りたいこと | 入口 |
|---|---|
| **今どこまで終わり、次に何をするか** | **[研究の現在地](docs/research_status.md)** |
| 年間計画・目指す成果 | [2026年度研究計画](docs/research_plan/2026_annual_plan.md) |
| 9月18日の進捗発表準備 | [説明性の出力・解釈・まとめ](docs/research_plan/2026-09-18_xai_progress_brief.md) |
| 研究目的と用語 | [研究コンテキスト](docs/research_context.md) |
| 過去の判断・古い資料の扱い | [文書案内](docs/document_index.md)、[進捗履歴](docs/progress_index.md) |
| 最新の確認済み数値 | [結果スナップショット一覧](experiments/README.md) |

9月15日に、文書・コード・保存結果の更新差を整理しました。今週は説明性の出力と解釈を優先し、ONB近傍の検知設計・新実験・同時計測は金曜の発表後に進めます。日々変わる設定・完了状況は「研究の現在地」で管理します。

## 配置

- `code/`: 実行コードと共通処理。現在の主実行は [run_ensemble_regression_onb.py](code/run_ensemble_regression_onb.py)。
- `Pool_boiling/`: 生データ、生成特徴、モデル結果。大量データはGit対象外。
- `water_flow/`: ノイズ音源。
- `研究進捗報告/`: 週次報告・発表・論文輪講・卒論等の原資料。
- `docs/`: 目的、現在地、年間計画、索引、運用メモ。
- `experiments/`: 日付付きの結果・解釈・軽量な根拠スナップショット。
- `configs/`: データ仕様と実験条件の記録。現行コードが自動でYAMLを読むという意味ではありません。
- `templates/`: 文書テンプレート。
- `archive/`、`trush_box/`: 過去コード・退避物。現在の主経路とは区別。

## 運用の地図

[コード地図](docs/code_map.md) / [データ索引](docs/data_inventory.md) / [管理方針](docs/repository_management.md) / [今回の確認範囲と変更理由](docs/audits/2026-09-15_workspace_review/README.md)

Codexは最初に [AGENTS.md](AGENTS.md) と現在地を読み、作業に必要な根拠へ進みます。過去の「次にやること」を、そのまま現在の未完了作業として再実行しない運用にします。
