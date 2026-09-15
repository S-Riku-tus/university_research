# configs

実験条件やデータ条件を残す場所です。既存コードの絶対パスやパラメータをすぐ全部移す必要はありません。新しい実験から、ここに設定を残していきます。

## 現行コードとの関係

2026-09-15時点の実行設定の正本は [run_ensemble_regression_onb.py](../code/run_ensemble_regression_onb.py) の `VALIDATION_CONFIG`。このフォルダのYAMLは記録用で、編集するだけでは実行設定に反映されない。5〜6月の日付付きYAMLは当時の条件例として保持する。

新しい条件の記録には[実験テンプレート](experiments/experiment_template.yaml)を使う。モデル・分割・評価単位・要求/実際の学習条件を残し、run manifestと照合する。現在の選択条件と予定は[現在地](../docs/research_status.md)を参照する。

## 使い方

- データセット条件は `configs/datasets/` に置く。
- 学習・評価実験の条件は `configs/experiments/` に置く。
- 最初はテンプレートをコピーして、分かる範囲だけ埋める。

## 目的

- どのデータ、どの前処理、どのモデル、どの閾値で実験したかを後から追えるようにする。
- 週次報告や修論執筆時に、条件を思い出さなくてもよい状態にする。

## 現行データセット

- [`datasets/waterflow_20260817_1s.yaml`](datasets/waterflow_20260817_1s.yaml): 修正済み水流音データの生成条件、実験別件数、manifest照合結果、旧バグデータの除外方針。

データ本体はGitへ入れず、この設定と `experiments/` の結果スナップショットを対応づけて管理する。
