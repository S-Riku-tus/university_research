# configs

実験条件やデータ条件を残す場所です。既存コードの絶対パスやパラメータをすぐ全部移す必要はありません。新しい実験から、ここに設定を残していきます。

## 現行コードとの関係

現行設定の既定値は[run_ensemble_regression_onb.py](../code/run_ensemble_regression_onb.py)の`VALIDATION_CONFIG`。`configs/experiments/`のJSONは、[`run_from_condition.py`](../experiments/2026-09-19_b_clean_only/run_from_condition.py)へ`--condition`で渡した場合に既定値へ上書きして実行できる。YAMLは記録用のものがあり、編集だけでは実行設定に反映されない。

`learning_policy.training_noise="matched"`ではnoiseごとにモデル・前処理・epoch・アンサンブル重みを再fitする。`"clean_only"`ではcleanでfitした同じ状態を全評価noiseへ共有する。別の実験データへ切り替える場合は、`train_experiments`と`test_experiments`を変更し、過去の重みを手作業で移植しない。

今後の主方式は`performance_kfold`。次の条件は[`2026-09-25_matched_performance_kfold_fixed_epoch.json`](experiments/2026-09-25_matched_performance_kfold_fixed_epoch.json)で、元WAV非共有5-fold、200 epoch固定とする。深層モデルは常に`run.epochs`の値を使用し、`run.folds`は`performance_kfold`の重み用内部K-fold数である。

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
