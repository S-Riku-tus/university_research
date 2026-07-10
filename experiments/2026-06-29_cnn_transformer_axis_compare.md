# 2026-06-29 CNN+Transformer 軸比較メモ

## 目的

`.npy` 作成側では保存配列を `(time_frame, frequency_bin)` として扱っている。一方、現行の `cnn_transformer_v1` は入力を `(frequency, time, channel)` と仮定し、横軸方向を時間として分割している。

そのため、Conformer/CNN+Transformer 系の不調が「データそのもの」ではなく、モデル側の軸解釈のずれで起きている可能性を切り分ける。

## 比較対象

- `cnntf_v1`: 現行モデル。入力を `(frequency_bin, time_frame, channel)` と仮定する歴史的ベースライン。
- `cnntf_axis`: `.npy` の保存形式に合わせ、入力を `(time_frame, frequency_bin, channel)` として時間軸を分割する軸修正版。
- `cnntf_legacy`: 過去に良い結果が出ていた時期の AlexNet 前段 + Transformer 構造。初回の軸比較では実行対象から外し、必要に応じて次の A/B 比較に使う。

## 現在の実行設定

`code/run_ensemble_regression_onb.py` の初期設定は、まず `cnntf_v1` と `cnntf_axis` を同じ学習率・バッチサイズで比較する軽量条件にしている。

- active models: `["cnntf_v1", "cnntf_axis"]`
- lr: `0.0001`, `0.00005`
- batch size: `48`, `64`
- epochs: `300`
- folds: `2`
- result folder suffix: `_axiscmp`

## 結果の見方

1. `cnntf_axis` が `cnntf_v1` より明確に良い場合  
   主原因は、現行モデルが `.npy` の時間軸を周波数軸として扱っていたことだと考えられる。

2. 両者が同程度に悪い場合  
   軸だけでは説明できず、学習設定、出力スケール、EarlyStopping、モデル容量、または旧構造との差を次に調べる。

3. `cnntf_v1` の方が良い場合  
   現行の時間分割仮定が偶然有効だった可能性、または `cnntf_axis` の分割粒度・CNN設計が合っていない可能性がある。`num_time_patches` や旧構造との比較に進む。

## 次の候補

軸修正で改善しない場合は、`active_model_keys` に `cnntf_legacy` を追加し、過去構造との比較を行う。その際は、旧構造で過去に良かった条件も参考にしつつ、同じデータ・同じfold条件で比較する。
