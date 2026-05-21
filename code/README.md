# code フォルダの見方

このフォルダは、研究で実際に使ってきた解析コードをそのまま残している場所です。使い慣れたファイル名は維持しつつ、役割を把握しやすくするためにこのREADMEを置いています。

## まず見るファイル

- `3.run_ensemble_ROC_100%_analysis.py`  
  現在の中心。`.npy` データを読み込み、回帰モデルとアンサンブルを評価する。

- `2.run_npy_waterflow_2つhighpass.py`  
  音声から水流音ノイズ付きSTFT特徴量を作成し、`.npy` として保存する。

- `compare_predict_heatflux.py`  
  学習済みモデルを使って、指定サンプルの予測値を比較する。

- `utils/`  
  データ読み込み、モデル定義、評価計算などの共通処理。

## フォルダの役割

- `utils/`: 今後も使う共通処理。
- `regression_analysis/`: 過去の個別回帰実験。
- `6-class classification/`: 6クラス分類の過去実験。
- `dBdata/`: dB変換や分類実験。
- `various_feature_values/`: STFT, SWT, spectrumなど特徴量作成の試行。
- `trush_box/`: 古い試行コードや退避コード。すぐに消さず、必要なものを拾う。

## 今後の方針

- 新しい実験条件は、コード内に直書きするだけでなく `configs/` にも残す。
- 結果を出したら、`experiments/` に `run_summary.md` を作る。
- どのコードが何をするか迷ったら、`docs/code_map.md` を更新する。
- 古いコードを移動・削除する前に、必要な処理が中心コードか `utils/` に残っているか確認する。
