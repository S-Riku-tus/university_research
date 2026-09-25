# コード地図

更新日: 2026-09-25。現在の設定・完了runは[研究の現在地](research_status.md)。通常の主実行は[run_ensemble_regression_onb.py](../code/run_ensemble_regression_onb.py)。

## 主経路と入出力

| 段階 | 主なコード | 入出力・役割 |
|---|---|---|
| 前処理 | [水流音STFT生成](../code/2.run_npy_waterflow_2つhighpass.py)、[waterflow_preprocessing.py](../code/utils/dataloading/waterflow_preprocessing.py) | 元WAV→固定基準ノイズ・STFT power→224×224 NPY、manifest |
| 読込 | [dataloading_and_conversion.py](../code/utils/dataloading/dataloading_and_conversion.py) | 時間×周波数×channelのx、熱流束y、元WAV/chunk情報 |
| 条件・分割 | [dataset_jobs.py](../code/utils/experiment/dataset_jobs.py)、[learning_policy.py](../code/utils/experiment/learning_policy.py) | 実験日/ノイズ方針、元WAV分離、ノイズ間の対応検査 |
| 学習・評価実行 | [learning_runner.py](../code/utils/experiment/learning_runner.py) | 分割・ノイズ方針ごとの学習・予測・XAI・指標・保存をまとめる |
| モデル | [base_regression.py](../code/utils/models/regression/base_regression.py)、[onb_defaults.py](../code/utils/config/onb_defaults.py) | RandomForest=XGBRF、Conformer、log-power AlexNet。固定registryと出力・評価・XAI既定値 |
| 学習器 | [model_training.py](../code/utils/training/model_training.py) | 学習側PCA、Keras/RF学習、元スケールへの予測復元 |
| 統合 | [strategy_catalog.py](../code/utils/ensemble/strategy_catalog.py)、[ensemble_runtime.py](../code/utils/ensemble/ensemble_runtime.py)、[ensemble_weighting.py](../code/utils/ensemble/ensemble_weighting.py) | 選択式の統合、主方式performance K-foldのOOF chunk R²逆誤差重み、過去inner holdout、crossfitのWAV目的 |
| 回帰・二値指標 | [regression_detection_metrics.py](../code/utils/calculation/regression_detection_metrics.py) | R²/RMSE/MAE、連続ROC/PR-AUC、二値分類指標 |
| chunk予測記録 | [prediction_records.py](../code/utils/calculation/prediction_records.py) | foldごとの1秒予測と元WAV・時刻情報をCSVへ保存 |
| ONB閾値 | [onb_thresholds.py](../code/utils/experiment/onb_thresholds.py) | 3日分の正確な閾値と原資料の出典 |
| 説明性 | [training_integration.py](../code/utils/explainability/training_integration.py)、[spectrogram_explainers.py](../code/utils/explainability/spectrogram_explainers.py) | TreeSHAP/IG/Grad-CAM/マスク、整合性・安定性・最終層ランダム化 |
| 作図 | [regression_plots.py](../code/utils/plotting/regression_plots.py)、[noise_trend_plots.py](../code/utils/plotting/noise_trend_plots.py) | 損失・散布図・比較図・ノイズ別曲線 |
| 保存 | [result_paths.py](../code/utils/experiment/result_paths.py)、[run_helpers.py](../code/utils/experiment/run_helpers.py) | 実行日と実行ハッシュ/周波数/ノイズ、manifest、条件hash、再開判定 |

## 設定の正本

主実行の`VALIDATION_CONFIG`にはデータ、学習条件、モデル別parameter grid、統合など実験ごとに変える項目を置く。`acoustic_selection`はピーク高さ閾値だけを置き、`None`なら選別なしとする。特徴CSV・帯域・対象範囲など通常固定する条件と、output/evaluation/explainability、モデルregistryは[onb_defaults.py](../code/utils/config/onb_defaults.py)で補完し、解決後の全設定をmanifestへ保存する。`configs/experiments/`のJSONは`run_from_condition.py`で実行時上書きでき、YAMLは記録用で自動読込しない。現在の`explicit_days`では`learning_policy.train_experiments`と`test_experiments`の和集合からデータ対象日を自動決定し、`data.experiment_names`は指定しない。旧`within_day / leave_one_day_out`へ切り替える場合のみ`data.experiment_names`に対象日を指定する。[日付指定の監査](../experiments/2026-09-16_onb_experiment_day_audit/README.md)。

現行の有効3モデルは`randomforest / conformer / alexnet`。主設定の統合方式は`performance_kfold`で、`simple_equal`を対照として同じ通常指標を出す。`inner_holdout`は過去run再現用に保持する。`subset_equal_cv / crossfit_wav_stack / crossfit_shrinkage_stack`も実装済みだが、次の主runでは無効。[重み学習の実装](../code/utils/ensemble/crossfit_stacking.py)、[次条件](../experiments/2026-09-25_matched_performance_kfold/README.md)、[6方式の手法と数式](ensemble_methods.md)。`prediction_max`は削除済み。`val_fold_legacy`は再現・診断用で主張不可。

`within_day / leave_one_day_out / explicit_days`と`matched / clean_only`を組み合わせる。matchedはnoiseごとに別familyを作り、モデル・PCA・scaler・epoch・重みをそのnoiseの学習データから再fitする。clean_onlyは同じcleanモデル・PCA・scaler・epoch・重みを評価noise間で共有する。明示分割では学習専用日は学習に要るノイズだけを探索する。一般化評価の[実装・制約](research_plan/2026-09-14_result_layout_and_generalization.md)も確認する。

通常の学習は要求epochまで行い、validation lossによるearly stoppingは現行主経路にない。`training_validation.enabled=true`の場合だけ、学習日内の元WAV非共有K-foldでcheckpointの`rmse_all`を比較し、深層モデル別のepochを選ぶ。現在の主条件はこれを無効にして200 epochへ固定する。OOM時にbatchを減らす再試行や、一定epoch以上の途中学習を受け入れる処理があるため、`tuning_summary.csv`等の実際のepoch/batchも確認する。要求epochの設定値だけで全モデルが必ず完走したと断定しない。

## 保存されるもの

各runは`fold_pred/`の1秒chunk予測、`explainability/`、損失/散布図/通常指標、manifestを持つ。新実行経路は`split_manifest.json`と完了時の`completed.json`も保存する。

主実行の保存先は各実験日の`regression_result/npy/<モデル群>/<実行日>/onb_<主要条件>_[p番号_]<HHMMSS>/<周波数>/<ノイズ>/`。日付直下は最大52文字で、学習・評価日、WAV/chunk内部検証、学習ノイズ、音響選別閾値、epochを短く表示し、末尾6桁は日付を含まない実行時刻とする。1起動内で複数parameter setを比較するときだけ`p01`等を付ける。実際の全条件と設定hashは`run_manifest.json`に記録する。`tuning_summary.csv`は条件フォルダの直下、ノイズ比較図はその下の`noise_trends/<統合方式>/<周波数>/`に置く。`RUN_ID`を明示して同じ条件で再実行した場合は同じフォルダを参照して完了判定する。2026-09-17に既存19系列も同じ日付／条件名階層へ移行済み。

通常runはモデル本体を永続保存しない。保存済み予測からの後処理と、モデルを必要とするIG再計算・新マスク推論は区別する。clean_onlyの一部ノイズだけ未完了の場合は、同じ学習モデルを揃えるため関連ノイズ一式を再計算する仕様。

## 後処理・監査

- [export_ensemble_result_snapshot.py](../code/export_ensemble_result_snapshot.py): 旧chunk集計とXAIから軽量記録を抽出する。過去runの評価定義を確認して使う。
- [export_waterflow_dataset_snapshot.py](../code/export_waterflow_dataset_snapshot.py): 現行データの件数・manifest・ノイズ条件を監査。
- [9/14結果の数値採取スクリプト](../experiments/2026-09-15_research_status_snapshot/collect_snapshot.py): 9/14の保存結果の検算・固定と9月のrun一覧。
- [reorganize_onb_results.py](../code/reorganize_onb_results.py): 指定runの保存階層移行。読取だけの確認と`--apply`による移動を区別する。
- [run_controlled_noise_curve_diagnostics.py](../code/run_controlled_noise_curve_diagnostics.py): 固定ノイズの診断。過去資料の別診断スクリプト名は現在存在しないものもある。

- [9/15・2帯域解析スクリプト](../experiments/2026-09-15_onb_frequency_comparison/analysis.md): 完了runの抽出、WAV/chunk指標の検算、卒論比較、54区間の帯域SNR。学習なしの後処理。
- [解析の進め方](analysis_workflow.md): 代表条件を選ぶ問いと、性能・ノイズ・XAIの因果検証を分ける手順。

## 過去コード・資料処理

`3.run_ensemble_ROC_100%_analysis.py`、`3.run_ensemble_100percent_classification.py`は旧実行。`regression_analysis/`、`6-class classification/`、`dBdata/`は個別回帰/分類の過去比較。`various_feature_values/`はSTFT/SWT/spectrumの試行。`code/trush_box/`と`archive/`は過去実装を保持する。旧PCの固定パスを持つ単発スクリプト3本は`code/trush_box/`へ退避した。

熱流束計算・名前対応は`0.run_auto_heatflux_analysis*.ipynb`、`1.run_rename_files.ipynb`等。過去Notebookを再実行する前に対象パスと名前変更の影響を確認する。研究資料内にも過去Notebook・スクリプトがあり、主実行と混同しない。
