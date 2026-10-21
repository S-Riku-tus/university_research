# コード地図

**10/21の探索再開**：AlexNetはlr=0.01・batch32へ固定，ResNet50/VGG16は各36候補を保持。`models.tuning_resume` と `utils/experiment/training_oof_tuning.py` が保存済み24候補を照合して飛ばし，残る49候補だけを学習する。再停止後は固定された再開先の完了候補も再利用。[条件と再開手順](research_plan/2026-10-19_cnn_only_three_comparison_and_writing_policy.md)。今回の再開起動は未実施。

**10/19の通常入口更新**：[CNN 3種の実行手順](research_plan/2026-10-19_cnn_only_three_comparison_and_writing_policy.md)。現在はAlexNet/ResNet50/VGG16の候補探索設定。ResNet50/VGG16の入力をlog-powerへ統一し登録，既存のOOF重み処理へCNN3用方式を追加した。候補が複数なら学習側探索，すべて1個なら通常の7条件評価。変更前の5モデル設定は[JSON](../configs/experiments/2026-10-19_before_cnn_three_validation_config.json)に保持。以下の5モデル等の説明は利用可能な経路として残す。

更新日: 2026-10-15。現在の設定・完了runは[研究の現在地](research_status.md)。通常の主実行は[run_ensemble_regression_onb.py](../code/run_ensemble_regression_onb.py)。

## 主経路と入出力

| 段階 | 主なコード | 入出力・役割 |
|---|---|---|
| 前処理 | [水流音STFT生成](../code/2.run_npy_waterflow_2つhighpass.py)、[waterflow_preprocessing.py](../code/utils/dataloading/waterflow_preprocessing.py) | 元WAV→固定基準ノイズ・STFT power→224×224 NPY、manifest |
| 10/7新実験の準備 | [従来の熱流束Notebook](../code/0.run_auto_heatflux_analysis_v2.ipynb)、[heatflux_preprocessing.py](../code/utils/calculation/heatflux_preprocessing.py)、[実行手順](2026-10-07_new_experiment_processing.md) | GL860/旧CSV、時刻保持、明示印加順・35秒の共通区間。結果は従来階層へ保存し改名/STFT/ONBが同じCSVを読む。再実行前に結果をexperimentsへ退避。新21/旧18段階を検算済み |
| 統合データ | [build_combined_0611_0618_dataset.py](../code/build_combined_0611_0618_dataset.py) | 6/11・6/18の0.5秒／1秒NPY・元名／熱流束名付きWAV・測定メタデータを、出典を保持して独立コピー |
| 読込 | [dataloading_and_conversion.py](../code/utils/dataloading/dataloading_and_conversion.py) | 時間×周波数×channelのx、熱流束y、元WAV/chunk情報 |
| 条件・分割 | [dataset_jobs.py](../code/utils/experiment/dataset_jobs.py)、[learning_policy.py](../code/utils/experiment/learning_policy.py) | 実験日/ノイズ方針、元WAV分離、ノイズ間の対応検査 |
| 学習・評価実行 | [learning_runner.py](../code/utils/experiment/learning_runner.py) | 分割・ノイズ方針ごとの学習・予測・XAI・指標・保存をまとめる |
| 保存fitの別日適用 | [saved_fit_evaluation.py](../code/utils/experiment/saved_fit_evaluation.py)、[saved_fit_report.py](../code/utils/calculation/saved_fit_report.py) | 保存cleanモデル/PCA/scaler/重みを再学習せず固定適用。日・WAV非重複、入力対応、旧予測再現を確認し、日別/段階別/学習範囲内外・誤り補完・特徴比較を出力 |
| 固定HGBの追加比較 | [run_hgb_complementarity_validation.py](../code/run_hgb_complementarity_validation.py)、[acoustic_summary_features.py](../code/utils/dataloading/acoustic_summary_features.py) | 固定34特徴HGBと既存3を同じchunk splitで再fit、clean-fit雑音OOF、固定統合、内部/最終状態保存。主runnerとは独立した比較入口 |
| モデル | [base_regression.py](../code/utils/models/regression/base_regression.py)、[acoustic_regression.py](../code/utils/models/regression/acoustic_regression.py)、[onb_defaults.py](../code/utils/config/onb_defaults.py) | 元XGBRF、CNN＋Transformer、log-power AlexNetを保持し、raw→34特徴HGB/ExtraTreesを追加。通常主設定は5 |
| 学習器 | [model_training.py](../code/utils/training/model_training.py) | RFだけfit側PCA、Kerasとsummaryはraw入力。共通scaler、学習/元スケール復元 |
| 統合 | [strategy_catalog.py](../code/utils/ensemble/strategy_catalog.py)、[fixed_core_stacking.py](../code/utils/ensemble/fixed_core_stacking.py)、[ensemble_runtime.py](../code/utils/ensemble/ensemble_runtime.py) | 共有clean OOFで元3内部比率を保持し、元3/HGB4/ExtraTrees4/5を一度に比較。過去performance/inner holdout/crossfitも保持 |
| 回帰・二値指標 | [regression_detection_metrics.py](../code/utils/calculation/regression_detection_metrics.py)、[source_day_metrics.py](../code/utils/calculation/source_day_metrics.py) | 従来実験閾値の指標＋日別閾値の合算/各日/元3差。FN/FP等の件数と日別q100も保存 |
| chunk予測記録 | [prediction_records.py](../code/utils/calculation/prediction_records.py) | foldごとの1秒予測と元WAV・時刻情報をCSVへ保存 |
| ONB閾値 | [onb_thresholds.py](../code/utils/experiment/onb_thresholds.py) | 3日分の正確な閾値と原資料の出典 |
| 説明性 | [training_integration.py](../code/utils/explainability/training_integration.py)、[spectrogram_explainers.py](../code/utils/explainability/spectrogram_explainers.py) | TreeSHAP/IG/Grad-CAM/マスク、整合性・安定性・最終層ランダム化 |
| 作図 | [regression_plots.py](../code/utils/plotting/regression_plots.py)、[noise_trend_plots.py](../code/utils/plotting/noise_trend_plots.py) | 損失・散布図・比較図・ノイズ別曲線 |
| 棒グラフ再描画 | [replot_regression_bars.py](../code/replot_regression_bars.py) | 明示した完了runの保存CSVから、全方式・固定抜粋・拡大版を別フォルダへ保存。完了印/hashを検査し、学習・モデル推論なし |
| 保存 | [result_paths.py](../code/utils/experiment/result_paths.py)、[run_helpers.py](../code/utils/experiment/run_helpers.py) | 実行日と実行ハッシュ/周波数/ノイズ、manifest、条件hash、再開判定 |

## 設定の正本

**10/15の現行確認**：本人の追加指示で発表用の2025/6/11＋6/18統合・`within_wav_chunk`・clean_only・200 epochs・主比較表有効を通常入口へ反映。設定検査と全7条件36 WAV/1620学習/540評価のメタデータ確認が成功、新学習/推論は未実行。[適用条件・確認手順](notes/2026-10-15_onb_presentation_run_conditions.md)と[適用済み設定記録](../configs/experiments/2026-10-15_onb_presentation_run_applied.json)を参照。設定JSONは記録用で自動読込しない。通常入口には`execution.mode`切替や`--preflight`引数はない。barモデル名23 pt、5モデル統合の`Original3`／`+ HGB + ET`2行表示を確認済み。

**同日・本人実行開始後の更新**：本人のONB実行は分析待ちとし、結果を読まずbarの数字30 pt・固定抜粋3構成・各構成の拡大版を追加した。[仕様・48図の確認・再描画手順](../experiments/2026-10-15_bar_presentation_views/README.md)。起動済みPythonへの修正の自動反映は保証しない。終了後は`replot_regression_bars.py --run-dir ...`で保存指標から再学習せず再描画できる。

### 10/14撤回前の保存fit入口の記録（現行手順ではない）

10/14現在の入口は`VALIDATION_CONFIG.execution.mode="saved_fit"`。旧6月1620 chunkの200 epochs保存fitを、新日10/7の全660 chunkへ適用する。初回は`data.noise_dir_names`を無雑音だけとし、6雑音を比較するときは同欄のコメントを戻す。`execution.source_condition_dir`と`source_fold`で保存fitを明示し、フォルダの新旧から自動選択しない。推論batchは旧runnerと同じ32。旧予測の照合は指定probeを含む元の32標本batch全体を再現し、最後の部分batchも保持する。学習用epoch/params設定はsaved_fitでは使用せず、保存モデルの条件を出力する。

撤回前の保存fit入口には`--preflight`があったが、現行入口では解析されず、付けても事前確認専用にはならない。現在はこの引数を使わない。以下の保存fit出力は当時の記録：新日側の`regression_result/npy/ensemble/YYYYMM/DD/onb_saved_<hash>_<HHMMSS>/maxfreq=3kHz/`。

全体の`main_metrics.csv`はkW/m²換算の日別/全体指標、`main_deltas.csv`は元3/ET4/ET単体との差、`q100_by_source_day.csv`は日別到達段階。雑音を含めた場合は`noise_degradation.csv`で自身のcleanからの増加量を確認する。各雑音フォルダの`metrics_by_stage.csv`と`prediction_by_stage.png`は段階別分布、`metrics_by_training_range.csv`は旧学習熱流束範囲の内外、`chunk_error_diagnostics.csv`は全chunkの共通FN/統合で失った陽性/訂正・追加誤り、`residual_complementarity.csv`は領域別の単体残差と共通誤り。これらの誤差・予測CSVは列名どおりW/m²。`features_by_chunk.csv`/`features_by_wav.csv`/`features_across_days.png`は旧fitの正確な1620学習chunkと新日cleanを、絶対qとq/日別ONBの両軸で比較する。34特徴はresize後の近似周波数に基づく記述量で、物理音源の因果同定とは区別する。

`execution.mode="train"`へ変えると従来の学習入口を使用する。現在の`learning_policy.evaluation_mode="cross_day"`では旧2日全2160で新fitし、新日を評価するため、旧保存1620 fitの固定適用とは別条件。`within_wav_chunk`も従来どおり選択可能。生成データ・旧run・元重みは変更しない。[実装確認記録](../experiments/2026-10-14_onb_saved_fit_evaluation/software_verification.json)。

### 通常実行の共通設定

主実行の`VALIDATION_CONFIG`にはデータ、学習条件、モデル別parameter grid、統合など実験ごとに変える項目を置く。`data_source_dir_by_experiment`の`{chunk_tag}`は`chunk_seconds`から`0.5s`または`1s`へ自動解決される。`acoustic_selection`はピーク高さ閾値だけを置き、`None`なら選別なしとする。特徴CSV・帯域・対象範囲など通常固定する条件と、output/evaluation/explainability、モデルregistryは[onb_defaults.py](../code/utils/config/onb_defaults.py)で補完し、解決後の全設定をmanifestへ保存する。`configs/experiments/`のJSONは条件実行スクリプトで実行時上書きでき、YAMLは記録用で自動読込しない。外側評価は`learning_policy.evaluation_mode`で選び、`evaluation_settings`内の同名欄だけを編集する。`cross_day`は学習日・テスト日のリスト、`within_wav_chunk`は実験フォルダ・テスト割合・seedを指定し、対象実験名は自動算出する。

現行主設定は`models.enabled_keys`の`randomforest / conformer / alexnet / hgb / extra_trees`。主統合は`original3_hgb_extra_trees`、対照は`original3_mse / original3_performance / original3_hgb / original3_extra_trees / simple_equal`。同じ1回の内部chunk OOFを共有して比率/配分をfitする。追加2は3 kHz専用の34特徴であり、別帯域を選ぶ前に入力表現を再定義する。registryの引数を省略する旧比較入口には元3を返す。[通常組込み記録](../experiments/2026-10-06_onb_five_model_integration/README.md)、[モデル理由](notes/2026-10-06_five_model_rationale_and_next_steps.md)、[統合の数式と旧6方式](ensemble_methods.md)。過去performance/inner holdout/crossfit方式も実装を保持する。

外側評価は、別日`cross_day`と全WAVのchunkをそれぞれ分ける`within_wav_chunk`の2方式を明示選択する。本人の10/3方針により、外側のWAV単位日内holdoutと旧日内K-foldは実行経路から削除した。`within_wav_chunk`は各WAVから同率のchunkをランダムにテストへ回し、同一chunkを共有しない既知WAV内の未使用区間評価である。10/5本人指定により、内部`performance_kfold`と`training_oof`探索は、外側学習chunk全体の通常KFold、shuffle=True、seed42へ切り替えた。`run.internal_validation_split`の既定は`chunk_kfold`、明示した旧対照だけ`wav_kfold`を使う。実効方式を条件hashへ含め、通常保存先は`ic3`で旧`iw3`と識別する。これらと`matched / clean_only`を組み合わせる。matchedはnoiseごとに別familyを作り、モデル・PCA・scaler・epoch・重みをそのnoiseの学習データから再fitする。clean_onlyは同じcleanモデル・PCA・scaler・epoch・重みを評価noise間で共有する。明示分割では学習専用日は学習に要るノイズだけを探索する。過去資料と結果は保持し、旧分割の記述を現行仕様とは扱わない。

通常の学習は`run.epochs`で指定した回数まで行い、別validationによるepoch選択とvalidation lossによるearly stoppingはない。OOM時にbatchを減らす再試行や、一定epoch以上の途中学習を受け入れる処理があるため、`tuning_summary.csv`等の実際のepoch/batchも確認する。要求epochの設定値だけで全モデルが必ず完走したと断定しない。

固定HGB比較の条件は[専用JSON](../configs/experiments/2026-10-06_hgb_followup_validation.json)から読む。`--phase train/integrate/analyze`で全モデル学習、OOFによる統合固定、外側評価を分け、同じ条件hashの完了fitから再開できる。各内部foldも保存し、150 epochs完走を要求する。雑音OOFを重みへ使う方式は、基礎モデルがclean固定でも、純粋なclean_only統合とは分けて報告する。[比較記録と再現入口](../experiments/2026-10-06_hgb_followup_validation/README.md)。

## 保存されるもの

各runは`fold_pred/`の1秒chunk予測、`explainability/`、損失/散布図/通常指標、manifestを持つ。新実行経路は`split_manifest.json`と完了時の`completed.json`も保存する。

5モデル主設定では11予測列（5単体＋6統合）を保存し、`metrics_by_source_day.csv`と`metrics_source_day_deltas.csv`を追加する。統合日の従来平均閾値と日別ONB閾値を別の評価定義として残す。`ensemble_training_oof_fit_f1.json`と`ensemble_training_oof_f1.csv`は重みfitの学習側診断であり外側評価ではない。

主実行の保存先は各実験日の`regression_result/npy/<モデル群>/<YYYYMM>/<DD>/onb_<主要条件>_[p番号_]<HHMMSS>/<周波数>/<ノイズ>/`。解析年月と日を分け、月単位で結果を探せるようにする。日付直下は最大52文字で、学習・評価日、WAV/chunk内部検証、学習ノイズ、音響選別閾値、epochを短く表示し、末尾6桁は日付を含まない実行時刻とする。1起動内で複数parameter setを比較するときだけ`p01`等を付ける。実際の全条件と設定hashは`run_manifest.json`に記録する。`tuning_summary.csv`は条件フォルダの直下、ノイズ比較図はその下の`noise_trends/<統合方式>/<周波数>/`に置く。`RUN_ID`を明示して同じ条件で再実行した場合は同じフォルダを参照して完了判定する。2026-09-25に日付付き既存系列を`YYYYMM/DD`へ移行した。日付のない旧系列は元の場所に残す。

共通既定はモデル本体を永続保存しないが、10/5の主実行は`output.save_fitted_artifacts=True`で最終モデル・PCA・scaler・重みの保存と再読込確認を有効にした。[実行準備確認](../experiments/2026-10-05_chunk_clean_baseline_ready/README.md)。同日の[再読込エラー修正](../experiments/2026-10-05_rf_reload_pca_layout_fix/README.md)では共通PCA変換をC連続配置へ統一し、`pca_transform_version=2`を実行hashと保存状態へ記録した。これは最終状態の保存であり、内部OOFの一時モデルも保存する設定ではない。保存済み予測からの後処理と、モデルを必要とするIG再計算・新マスク推論は区別する。clean_onlyの一部ノイズだけ未完了の場合は、同じ学習モデルを揃えるため関連ノイズ一式を再計算する仕様。

## 後処理・監査

- [export_ensemble_result_snapshot.py](../code/export_ensemble_result_snapshot.py): 旧chunk集計とXAIから軽量記録を抽出する。過去runの評価定義を確認して使う。
- [export_waterflow_dataset_snapshot.py](../code/export_waterflow_dataset_snapshot.py): 現行データの件数・manifest・ノイズ条件を監査。
- [9/14結果の数値採取スクリプト](../experiments/2026-09-15_research_status_snapshot/collect_snapshot.py): 9/14の保存結果の検算・固定と9月のrun一覧。
- [reorganize_onb_results.py](../code/reorganize_onb_results.py): 指定runの保存階層移行。読取だけの確認と`--apply`による移動を区別する。
- [migrate_ensemble_date_layout.py](../code/migrate_ensemble_date_layout.py): `ensemble`直下の日付付き既存系列を`YYYYMM/DD`へ移す。既定はdry-runで、衝突検査後に`--apply`を指定する。
- [run_controlled_noise_curve_diagnostics.py](../code/run_controlled_noise_curve_diagnostics.py): 固定ノイズの診断。過去資料の別診断スクリプト名は現在存在しないものもある。

- [9/15・2帯域解析スクリプト](../experiments/2026-09-15_onb_frequency_comparison/analysis.md): 完了runの抽出、WAV/chunk指標の検算、卒論比較、54区間の帯域SNR。学習なしの後処理。
- [解析の進め方](analysis_workflow.md): 代表条件を選ぶ問いと、性能・ノイズ・XAIの因果検証を分ける手順。

## 過去コード・資料処理

`3.run_ensemble_ROC_100%_analysis.py`、`3.run_ensemble_100percent_classification.py`は旧実行。`regression_analysis/`、`6-class classification/`、`dBdata/`は個別回帰/分類の過去比較。`various_feature_values/`はSTFT/SWT/spectrumの試行。`code/trush_box/`と`archive/`は過去実装を保持する。旧PCの固定パスを持つ単発スクリプト3本は`code/trush_box/`へ退避した。

熱流束計算・名前対応は`0.run_auto_heatflux_analysis*.ipynb`、`1.run_rename_files.ipynb`等。過去Notebookを再実行する前に対象パスと名前変更の影響を確認する。研究資料内にも過去Notebook・スクリプトがあり、主実行と混同しない。
