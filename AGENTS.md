# Codex向け研究コンテキスト

このリポジトリでは、研究資料と解析コードの両方を扱う。作業時は、既存ファイルを大きく移動せず、まず `docs/` に要約・索引・運用メモを追加して、AIが読みやすい入口を整える。

## 研究テーマ

音響データからプール沸騰の状態を推定する。中心タスクは、音声から作成したスペクトログラムを使って熱流束を回帰予測し、閾値を用いて沸騰開始点を検知すること。単なる分類精度だけでなく、モデルが沸騰現象に関係する時間・周波数特徴をどのように見ているか、物理的に妥当な説明ができるかを重視する。

## 重要ファイル

- `code/3.run_ensemble_ROC_100%_analysis.py`: 現在の中心的な回帰・アンサンブル・評価スクリプト。
- `code/utils/models/regression/base_regression.py`: AlexNet系、CNN+Transformer、RandomForest系などの回帰モデル定義。
- `code/utils/dataloading/dataloading_and_conversion.py`: `.npy` や画像から入力 `x` と熱流束ラベル `y` を作る処理。
- `code/2.run_npy_waterflow_2つhighpass.py`: 音声からSTFT特徴を作り、水流音ノイズ条件別の `.npy` を保存する処理。
- `研究進捗報告/`: 過去の週次報告、発表資料、論文輪講、卒論・学会資料。

## 作業方針

- 研究資料の原本は基本的に動かさない。
- AI用の要約は `docs/` に置く。
- 繰り返し使う文書形式は `templates/` に置く。
- 週次報告はSOAP形式、つまり `S: Subjective`, `O: Objective`, `A: Assessment`, `P: Plan` を基本形にする。
- コード改善時は、データの流れ、モデルの入出力、評価指標、保存先を必ず確認する。
- 研究報告文では、断定しすぎず、「現時点では」「今後確認する」「検討中」といった研究進行中の文体を自然に使う。

## 週次報告を作るとき

1. `docs/research_context.md` で研究の全体像を確認する。
2. `docs/progress_index.md` で直近の報告内容を確認する。
3. `git status --short` や最近更新されたファイルから今週の作業候補を拾う。
4. 必要に応じて直近の `.docx` 報告書やコード差分を読む。
5. `templates/weekly_progress_SOAP.md` に沿って下書きを作る。
6. 最後に「先生に相談したいこと」「次週やること」を明確にする。
