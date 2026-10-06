# code フォルダの見方

## 現在の主経路

- [run_ensemble_regression_onb.py](run_ensemble_regression_onb.py): 現行の主実行。設定は冒頭の `VALIDATION_CONFIG`。元3＋HGB＋ExtraTreesの5モデル、同じOOFによる3/4/5対照、1秒chunk評価、説明性、一般化方針を指定する。
- [2.run_npy_waterflow_2つhighpass.py](2.run_npy_waterflow_2つhighpass.py): 音声から現行STFT powerデータを生成する。
- [utils/](utils/): データ読込、モデル、学習、指標、統合、XAI、作図の共通処理。
- [check_gpu.py](check_gpu.py): GPU認識の確認。

外側評価は`learning_policy.evaluation_mode`で`cross_day`または`within_wav_chunk`を選ぶ。前者は`evaluation_settings.cross_day`の学習日・テスト日リスト、後者は`evaluation_settings.within_wav_chunk`の実験フォルダ・テストchunk割合・seedを編集する。単日または統合フォルダを使え、`data.experiment_names`は自動算出する。内部は既定のshuffle chunk KFoldを学習側だけで行い、明示WAV対照も残す。詳細は[コード地図](../docs/code_map.md)。

学習を起動する前に[研究の現在地](../docs/research_status.md)と設定・保存済み結果を照合する。現在の主設定は最終5モデル・PCA/scaler・統合重みの保存/再読込確認を有効にした。過去runはモデル非保存のものもあるため、推論・XAI再計算を案内するときは実際の保存物を確認する。

追加2は[acoustic_regression.py](utils/models/regression/acoustic_regression.py)にあり、3 kHz・1 channelのraw power→固定34特徴を使う。`metrics_by_source_day.csv`は日別ONB閾値の各日と合算、`metrics_source_day_deltas.csv`は元3との差を保存する。従来の`metrics_summary_*.csv`の統合日平均ONB閾値と区別する。[通常実行と確認先](../experiments/2026-10-06_onb_five_model_integration/README.md)、[5モデルの採用理由](../docs/notes/2026-10-06_five_model_rationale_and_next_steps.md)。

10/6の[主張・評価・主表の正本](../docs/thesis/2026-10-06_main_claims_and_evaluation.md)に沿い、通常clean_onlyの終了時に`maxfreq=3kHz/onb_comparison/main_comparison.md`と全7条件の主表・差・劣化量・重み・検算を生成する。全5モデルの全評価予測を保存再読込で照合し、内部/最終CNNの実epoch・batchを記録する。欠落条件は学習前に検出する。[実行方法と読むファイル](../experiments/2026-10-06_onb_main_comparison_ready/README.md)。既定は150 epochs/3fold/seed42、50関連テスト・実main 1 epoch確認済み、新しい150 epochs本runは未実施。

## 過去コード

[3.run_ensemble_ROC_100%_analysis.py](3.run_ensemble_ROC_100%_analysis.py)は旧版・再現用。現在の主実行ではない。`regression_analysis/`、`6-class classification/`、`dBdata/`、`trush_box/`にも過去の比較・試行がある。`compare_predict_heatflux.py`は保存済みモデル用の過去の推論処理で、現行runの結果CSVと同じ入力経路とは限らない。

2026-09-16に、旧PCの絶対パスを持ち主経路から参照されない`analysis_channel.py`、`calc_channel_num.py`、`crop_spectrogram.py`を`trush_box/`へ退避した。

詳細は[コード地図](../docs/code_map.md)。条件の記録は[configs](../configs/README.md)、結果と解釈は[experiments](../experiments/README.md)へ残す。YAMLの条件記録は現在の実行コードへ自動適用されない。
