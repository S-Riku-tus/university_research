# コード地図

このファイルは、`code/` のどこに何があるかを短く整理するための地図です。コードを移動せず、まずここで意味を明確にする。

## 中心スクリプト

- `code/2.run_npy_waterflow_2つhighpass.py`  
  音声データにハイパスフィルタや水流音ノイズ付与を行い、STFT特徴を `.npy` として保存する前処理スクリプト。

- `code/3.run_ensemble_ROC_100%_analysis.py`  
  `.npy` データを読み込み、RandomForest系、CNN+Transformer系モデルを学習・評価し、R2, AUC, RMSE, MAE, 100%分類閾値などを出す中心スクリプト（旧版・再現用に保持）。

- `code/run_ensemble_regression_onb.py`  
  上記中心スクリプトの作り直し版（2026-06-12 計画 Phase 0 対応）。次の3点を修正済み：
  ①モデルをレジストリ(`MODEL_SPECS`)で定義しラベルが実体に追従（RFをAlexNetと誤記しない）、
  ②AUCを「連続スコア版(ROC/PR-AUC)」と「二値化後の分類指標」に分離、
  ③アンサンブル重みを `WEIGHT_STRATEGY` で選択式（simple/fixed/inner_holdout=リークなし/val_fold_legacy=旧リークあり）。
  データパスは別マシン運用のため旧版と同じハードコードのまま。
  再利用可能な処理（指標計算・学習/予測・重み付け・作図）は下記 utils に分離済みで、
  本ファイルにはこの実験固有の設定と `main()` のオーケストレーションだけを置く。
  実行モデルは `ACTIVE_MODEL_KEYS` の1行で切替（RF単体検証 `["rf"]` ⇄ 3モデル
  `["rf","cnntf_v1","cnntf_v2"]`）。`SMOKE_TEST=True` で epoch/fold を縮小し
  「最後まで通るか」だけを高速確認できる（出力先に `smoke_` が付き本番結果と混ざらない）。
  出力先 `SAVE_PATH` 末尾にモデルセットのタグが付くので、RF単体と3モデルの結果は別フォルダに残る。

- `code/compare_predict_heatflux.py`  
  学習済みモデルを使って、指定したサンプルの熱流束予測を比較する推論・確認用スクリプト。

- `code/check_gpu.py`  
  TensorFlowからGPUが見えているか確認する小さな確認用スクリプト。

## 共通処理

- `code/utils/dataloading/dataloading_and_conversion.py`  
  `.npy` や画像を読み込み、入力 `x` と熱流束ラベル `y` を作る。

- `code/utils/models/regression/base_regression.py`  
  AlexNet系、CNN+Transformer、RandomForest系などの回帰モデル定義。

- `code/utils/models/regression/new_regression.py`  
  CNN+LSTM、改善版CNN+Transformer、時系列処理系のモデル候補。

- `code/utils/models/regression/swin_transformer.py`  
  Swin Transformer系の回帰モデル。

- `code/utils/calculation/calc_r2_auc.py`  
  R2やAUC計算の補助（旧スクリプト用）。

- `code/utils/calculation/regression_detection_metrics.py`  
  熱流束回帰と ONB 検知の評価指標（`RegressionDetectionMetrics`）。回帰指標／連続スコアAUC（ROC・PR）／二値化後の分類指標を分けて算出する。`run_ensemble_regression_onb.py` 用。

- `code/utils/training/model_training.py`  
  1モデルの学習・予測と PCA 前処理（`ModelTrainer`）。MODEL_SPECS の kind（keras/sklearn）に応じて入力形態を切り替える。

- `code/utils/ensemble/ensemble_weighting.py`  
  アンサンブルの重み決定と予測統合（`EnsembleWeighting`）。simple/fixed/inner_holdout/val_fold_legacy の各戦略に対応。

- `code/utils/plotting/regression_plots.py`  
  回帰・アンサンブル評価まわりの作図（`RegressionPlotter`）。損失曲線・指標棒グラフ・予測散布図（100%分類閾値線つき）。

## 実験・過去コード

- `code/regression_analysis/`  
  AlexNet, VGG16, ResNet50などの個別回帰実験。

- `code/6-class classification/`  
  6クラス分類用の過去実験コード。

- `code/dBdata/`  
  dB化データや4クラス分類関係のコード。

- `code/various_feature_values/`  
  STFT, SWT, spectrumなど、特徴量作成・可視化の試行。

- `code/trush_box/`  
  古い試行コードや一時的に退避したコード。すぐには消さず、必要なものを見つけたら中心スクリプトや共通処理へ移す。

## 今後の整理方針

1. 新しい処理は、できるだけ `code/utils/` に共通関数として置く。
2. 実行用スクリプトは `code/` 直下に置いてもよいが、設定は `configs/` に逃がす。
3. 古いスクリプトを消す前に、役割をこのファイルに書く。
4. 同じ処理が複数ファイルにある場合は、よく使うものから共通化する。
