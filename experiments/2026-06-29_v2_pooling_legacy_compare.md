# 2026-06-29 v2 Pooling / Legacy 比較メモ

## 目的

過去に `cnn_transformer_v1` で良い結果が出ていた要因が、最終の時系列集約方法 `AttentionPooling` にあるのか、それとも旧v1全体の構造にあるのかを切り分ける。

## 比較対象

- `cnntf_v2_gap`: 現在の `cnn_transformer_v2`。AlexNet風CNN前段 + Transformer + `GlobalAveragePooling1D`。
- `cnntf_v2_attn`: `cnn_transformer_v2` の最終poolingだけを `AttentionPooling` に変更したもの。
- `cnntf_legacy`: 過去の旧v1に近い構造。AlexNet風CNN前段 + Transformer + `AttentionPooling` で、`model_dim=256`、`key_dim=64`。

## 現在の実行設定

- active models: `["cnntf_v2_gap", "cnntf_v2_attn", "cnntf_legacy"]`
- lr: `0.0001`, `0.00005`
- batch size: `48`, `64`
- epochs: `300`
- folds: `2`
- dataset: `2025.06.11_0.3_2`
- max frequency: `maxfreq=22kHz`
- noise: `heatflux_no_noise`
- result folder suffix: `_v2cmp`

## 結果の見方

1. `cnntf_v2_attn` が `cnntf_v2_gap` より良い場合  
   `AttentionPooling` が改善に効いている可能性が高い。

2. `cnntf_legacy` が `cnntf_v2_attn` より良い場合  
   Poolingだけでなく、旧v1の `model_dim=256` や `key_dim=64` などの構造差が効いている可能性が高い。

3. 3モデルとも同程度に悪い場合  
   モデル構造よりも、学習条件、出力スケール、EarlyStopping、またはラベル/入力前処理側を優先して調べる。

## 注意

ROC-AUCやPR-AUCが高くても、予測値の絶対スケールがずれるとONB閾値判定は崩れる。結果を見るときは `R2`, `RMSE`, `ONB RMSE`, `pred positive rate`, fold間の予測スケールを合わせて確認する。
