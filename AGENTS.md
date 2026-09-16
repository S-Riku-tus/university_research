# Codex向け研究コンテキスト

このリポジトリは、音響スペクトログラムからプール沸騰の熱流束を回帰し、ONBを検知する研究の資料と解析コードを扱う。精度だけでなく、モデルが見る時間・周波数特徴の物理的妥当性も重視する。

## 読む順序

- 研究判断、結果解析、実験、計画、報告に関わる作業では、最初に `docs/research_status.md` を読む。
- 狭いコード修正やリポジトリ整備では、検索で対象を特定して必要なファイルから読む。研究条件や現在方針に影響すると分かった時点で `docs/research_status.md` を確認する。
- 長期計画は `docs/research_plan/2026_annual_plan.md`、文書の更新先は `docs/document_index.md`、修論の中心仮説は `docs/research_plan/2026-09-15_master_thesis_hypothesis.md` を参照する。
- 追加結果の解析は `docs/analysis_workflow.md` に従い、確認事実・原因仮説・次の識別比較を分ける。現在の対象外runを自動追加しない。
- 本人の最新指示を、古い計画やAIの提案より優先する。設定済み・実装済み・出力済み・研究上の検証済みを区別する。

## コンテキストと作業範囲

- まず `rg` で関連する記号・パスを絞り、必要なファイルと該当範囲だけ読む。不足する根拠がある場合に探索範囲を広げる。
- 大きなデータ、生成物、ログ、Office/PDFを一括で読み込まない。索引、manifest、要約、必要ページ、末尾などから確認する。
- 既知の内容を再読・再掲しない。無関係なリファクタリング、ファイル移動、依存追加、全生成物の再作成を行わない。
- 変更後は最小の関連チェックから実行する。共有基盤への影響、対象チェックの失敗、または明示的な要件がある場合だけ範囲を広げ、冗長なログを避ける。
- 成功条件を満たしたら終了する。追加調査や改善は、未解決の根拠または本人の依頼がある場合に行う。
- サブエージェントや並列エージェントは、本人が明示的に依頼した場合だけ使用する。
- 進捗報告と最終報告は簡潔にし、変更、確認結果、未解決事項だけを伝える。

## 配置と変更方針

- 原資料と既存配置を保ち、AI用の要約・索引・運用メモは `docs/` に置く。
- 繰り返し使う文書形式は `templates/`、新しい実験条件の記録は `configs/`、結果と解釈の固定記録は `experiments/YYYY-MM-DD_.../` に置く。
- 実験結果を確認したら、条件、評価単位、根拠への参照を `experiments/` に残し、必要に応じて現在地を更新する。過去の「次にやること」を未完了と決めつけて再実行しない。
- コード変更では、データの流れ、モデルの入出力、分割、評価指標、保存先を確認する。データリークを避け、既存の再現性とインターフェースを保つ。
- 研究資料の原本、提出済み文書、過去結果は原則として移動・上書きしない。研究文では断定しすぎず、事実、解釈、未確認事項を分ける。

## 主要な入口

- 現行実行: `code/run_ensemble_regression_onb.py`（設定は `VALIDATION_CONFIG`）
- モデル: `code/utils/models/regression/base_regression.py`
- 入力とラベル: `code/utils/dataloading/dataloading_and_conversion.py`
- STFT特徴生成: `code/2.run_npy_waterflow_2つhighpass.py`
- コード地図: `docs/code_map.md`
- データ索引: `docs/data_inventory.md`
- 管理方針: `docs/repository_management.md`
- Codex/Cursor運用: `docs/codex_token_efficiency.md`

## 特定の文書作業

- 週次報告は `docs/research_status.md`、`docs/research_context.md`、`docs/progress_index.md` と直近の変更を確認し、`templates/weekly_progress_SOAP.md` に沿う。「先生に相談したいこと」と「次週やること」を明確にし、発表準備メモと対象run・結論・次工程を揃える。
- 論文輪講は、対象PDFと直近pptxを確認してから `templates/論文輪講_スライド構成案.md` に従う。図表番号・式番号を原論文で確定し、原則として実図を使う。自研究との接続を冒頭とまとめに入れ、過去の提出原本は保持する。
