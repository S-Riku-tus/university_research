# code フォルダの見方

## 現在の主経路

- [run_ensemble_regression_onb.py](run_ensemble_regression_onb.py): 現行の主実行。設定は冒頭の `VALIDATION_CONFIG`。3モデル、統合、1秒chunk評価、説明性、一般化方針を指定する。
- [2.run_npy_waterflow_2つhighpass.py](2.run_npy_waterflow_2つhighpass.py): 音声から現行STFT powerデータを生成する。
- [utils/](utils/): データ読込、モデル、学習、指標、統合、XAI、作図の共通処理。
- [check_gpu.py](check_gpu.py): GPU認識の確認。

現在の`explicit_days`では学習日・テスト日を`learning_policy`で指定し、`data.experiment_names`は自動算出する。旧`within_day / leave_one_day_out`では`data.experiment_names`の指定が必要。詳細は[コード地図](../docs/code_map.md)。

学習を起動する前に[研究の現在地](../docs/research_status.md)と設定・保存済み結果を照合する。設定済みと実行完了は分ける。通常の主経路はモデル重みを永続保存しないため、過去モデルの推論・XAI再計算を案内するときは実際の重みの有無を確認する。

## 過去コード

[3.run_ensemble_ROC_100%_analysis.py](3.run_ensemble_ROC_100%_analysis.py)は旧版・再現用。現在の主実行ではない。`regression_analysis/`、`6-class classification/`、`dBdata/`、`trush_box/`にも過去の比較・試行がある。`compare_predict_heatflux.py`は保存済みモデル用の過去の推論処理で、現行runの結果CSVと同じ入力経路とは限らない。

2026-09-16に、旧PCの絶対パスを持ち主経路から参照されない`analysis_channel.py`、`calc_channel_num.py`、`crop_spectrogram.py`を`trush_box/`へ退避した。

詳細は[コード地図](../docs/code_map.md)。条件の記録は[configs](../configs/README.md)、結果と解釈は[experiments](../experiments/README.md)へ残す。YAMLの条件記録は現在の実行コードへ自動適用されない。
