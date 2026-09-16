# コード地図

更新日: 2026-09-16。現在の設定・完了runは[研究の現在地](research_status.md)。通常の主実行は[run_ensemble_regression_onb.py](../code/run_ensemble_regression_onb.py)。

## 主経路と入出力

| 段階 | 主なコード | 入出力・役割 |
|---|---|---|
| 前処理 | [水流音STFT生成](../code/2.run_npy_waterflow_2つhighpass.py)、[waterflow_preprocessing.py](../code/utils/dataloading/waterflow_preprocessing.py) | 元WAV→固定基準ノイズ・STFT power→224×224 NPY、manifest |
| 読込 | [dataloading_and_conversion.py](../code/utils/dataloading/dataloading_and_conversion.py) | 時間×周波数×channelのx、熱流束y、元WAV/chunk情報 |
| 条件・分割 | [dataset_jobs.py](../code/utils/experiment/dataset_jobs.py)、[learning_policy.py](../code/utils/experiment/learning_policy.py) | 実験日/ノイズ方針、元WAV分離、ノイズ間の対応検査 |
| 学習・評価実行 | [learning_runner.py](../code/utils/experiment/learning_runner.py) | 分割・ノイズ方針ごとの学習・予測・XAI・指標・保存をまとめる |
| モデル | [base_regression.py](../code/utils/models/regression/base_regression.py)、[onb_defaults.py](../code/utils/config/onb_defaults.py) | RF=XGBRF、log-power AlexNet、log-power CNN＋Transformer。固定registryと出力・評価・XAI既定値 |
| 学習器 | [model_training.py](../code/utils/training/model_training.py) | 学習側PCA、Keras/RF学習、元スケールへの予測復元 |
| 統合 | [strategy_catalog.py](../code/utils/ensemble/strategy_catalog.py)、[ensemble_runtime.py](../code/utils/ensemble/ensemble_runtime.py)、[ensemble_weighting.py](../code/utils/ensemble/ensemble_weighting.py) | 選択式の統合、元WAV非重複inner holdout、WAV medianでの重み用誤差 |
| 回帰・二値指標 | [regression_detection_metrics.py](../code/utils/calculation/regression_detection_metrics.py) | R²/RMSE/MAE、連続ROC/PR-AUC、二値分類指標 |
| WAV・遷移指標 | [wav_event_metrics.py](../code/utils/calculation/wav_event_metrics.py) | OOFをWAVに集約、測定点順ONB遷移、予測交差診断 |
| ONB閾値 | [onb_thresholds.py](../code/utils/experiment/onb_thresholds.py) | 3日分の正確な閾値と原資料の出典 |
| 説明性 | [training_integration.py](../code/utils/explainability/training_integration.py)、[spectrogram_explainers.py](../code/utils/explainability/spectrogram_explainers.py) | TreeSHAP/IG/Grad-CAM/マスク、整合性・安定性・最終層ランダム化 |
| 作図 | [regression_plots.py](../code/utils/plotting/regression_plots.py)、[noise_trend_plots.py](../code/utils/plotting/noise_trend_plots.py) | 損失・散布図・比較図・ノイズ別曲線 |
| 保存 | [result_paths.py](../code/utils/experiment/result_paths.py)、[run_helpers.py](../code/utils/experiment/run_helpers.py) | 実行日と実行ハッシュ/周波数/ノイズ、manifest、条件hash、再開判定 |

## 設定の正本

主実行の`VALIDATION_CONFIG`にはデータ、学習条件、モデル別parameter grid、統合など実験ごとに変える項目を置く。`acoustic_selection`はピーク高さ閾値だけを置き、`None`なら選別なしとする。特徴CSV・帯域・対象範囲など通常固定する条件と、output/evaluation/explainability、モデルregistryは[onb_defaults.py](../code/utils/config/onb_defaults.py)で補完し、解決後の全設定をmanifestへ保存する。`configs/`のYAMLは条件記録で、現在は自動読込しない。現在の`explicit_days`では`learning_policy.train_experiments`と`test_experiments`の和集合からデータ対象日を自動決定し、`data.experiment_names`は指定しない。旧`within_day / leave_one_day_out`へ切り替える場合のみ`data.experiment_names`に対象日を指定する。[日付指定の監査](../experiments/2026-09-16_onb_experiment_day_audit/README.md)。

現行の有効3モデルは`rf / cnntf_v2_gap / alexnet`。主設定で有効な統合方式は`inner_holdout`で、これを主表示にも使う。`performance_kfold`へ戻す場合は現在の`wav_kfold`とWAV中央値OOF R²を使う。`subset_equal_cv / crossfit_wav_stack / crossfit_shrinkage_stack`の固定ピーク選別との併用も実装済みだが、現在は無効。[重み学習の実装](../code/utils/ensemble/crossfit_stacking.py)、[診断と修正](../experiments/2026-09-16_onb_result_diagnosis/README.md)、[5方式の手法と数式](ensemble_methods.md)。`prediction_max`は削除済み。`val_fold_legacy`は再現・診断用で主張不可。

`within_day / leave_one_day_out / explicit_days`と`matched / clean_only`を組み合わせる。clean_onlyは同じモデル・PCA・scaler・重みをノイズ間で共有する。明示分割では学習専用日は学習に要るノイズだけを探索する。一般化評価の[実装・制約](research_plan/2026-09-14_result_layout_and_generalization.md)も確認する。

通常の学習は要求epochまで行い、validation lossによるearly stoppingは現行主経路にない。OOM時にbatchを減らす再試行や、一定epoch以上の途中学習を受け入れる処理があるため、`tuning_summary.csv`等の実際のepoch/batchも確認する。要求300という名前だけで全モデルが必ず300完走したと断定しない。

## 保存されるもの

各runは`fold_pred/`のchunk予測、`wav_eval/`のpooled WAV指標・ONB遷移、`explainability/`、損失/散布図/指標、manifestを持つ。新実行経路は`split_manifest.json`と完了時の`completed.json`も保存する。

主実行の保存先は各実験日の`regression_result/npy/<モデル群>/<実行日>/<方針名>__<12桁hash>/<周波数>/<ノイズ>/`。末尾にモデル・epoch名のrunフォルダは作らない。12桁hashは起動ごとのIDと条件hashから作るため、同日・同条件の別起動でも別フォルダとなり、1起動内のパラメータ候補も分かれる。実際の条件と`execution_id`は`run_manifest.json`に記録する。`tuning_summary.csv`はhash付き方針フォルダの直下、ノイズ比較図はその下の`<周波数>/noise_trends/`に置く。`RUN_ID`を明示して同じ条件で再実行した場合は同じフォルダを参照して完了判定する。旧階層の結果は移動せず、既存の読み取り経路を維持する。

通常runはモデル本体を永続保存しない。保存済み予測からの後処理と、モデルを必要とするIG再計算・新マスク推論は区別する。clean_onlyの一部ノイズだけ未完了の場合は、同じ学習モデルを揃えるため関連ノイズ一式を再計算する仕様。

## 後処理・監査

- [run_wav_event_evaluation.py](../code/run_wav_event_evaluation.py): 保存予測をWAV/ONB評価へ集約。旧重み学習の情報混在は後処理では取り除けない。
- [export_ensemble_result_snapshot.py](../code/export_ensemble_result_snapshot.py): 主に旧chunk集計とXAIから軽量記録を抽出。WAV主評価や安全性の監査範囲を確認して使う。
- [export_waterflow_dataset_snapshot.py](../code/export_waterflow_dataset_snapshot.py): 現行データの件数・manifest・ノイズ条件を監査。
- [9/14結果の数値採取スクリプト](../experiments/2026-09-15_research_status_snapshot/collect_snapshot.py): 9/14の保存結果の検算・固定と9月のrun一覧。
- [reorganize_onb_results.py](../code/reorganize_onb_results.py): 指定runの保存階層移行。読取だけの確認と`--apply`による移動を区別する。
- [run_controlled_noise_curve_diagnostics.py](../code/run_controlled_noise_curve_diagnostics.py): 固定ノイズの診断。過去資料の別診断スクリプト名は現在存在しないものもある。

- [9/15・2帯域解析スクリプト](../experiments/2026-09-15_onb_frequency_comparison/analysis.md): 完了runの抽出、WAV/chunk指標の検算、卒論比較、54区間の帯域SNR。学習なしの後処理。
- [解析の進め方](analysis_workflow.md): 代表条件を選ぶ問いと、性能・ノイズ・XAIの因果検証を分ける手順。

## 過去コード・資料処理

`3.run_ensemble_ROC_100%_analysis.py`、`3.run_ensemble_100percent_classification.py`は旧実行。`regression_analysis/`、`6-class classification/`、`dBdata/`は個別回帰/分類の過去比較。`various_feature_values/`はSTFT/SWT/spectrumの試行。`code/trush_box/`と`archive/`は過去実装を保持する。旧PCの固定パスを持つ単発スクリプト3本は`code/trush_box/`へ退避した。

熱流束計算・名前対応は`0.run_auto_heatflux_analysis*.ipynb`、`1.run_rename_files.ipynb`等。過去Notebookを再実行する前に対象パスと名前変更の影響を確認する。研究資料内にも過去Notebook・スクリプトがあり、主実行と混同しない。
