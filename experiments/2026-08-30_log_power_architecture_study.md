# 生パワースペクトログラム向けモデル構造改善実験

実施日: 2026-08-30  
対象: `run_ensemble_regression_onb.py` の `cnntf_v2_gap` と `alexnet`  
対象外: RF、TreeSHAP変更、Guided Grad-CAM、物理帯域分担型アンサンブル、RFアンカー残差アンサンブル

## 1. 目的

現行の学習率 `0.001`、batch size `16` を維持し、CNN+TransformerとAlexNetの内部構造に改善余地があるかを調べた。主評価は元WAV単位GroupKFoldのOOF R²、全域RMSE、fold間SEとした。ONB帯はfoldによって該当群が0になるため、構造選択の主指標には使用していない。

## 2. データと比較条件

- データ源: 3実験とも `waterflow_20260817_1s`
- 実験: `2025.06.11_0.3_2`、`2025.06.18_0.3_3`、`2025.07.09_0.3_1`
- 条件: `heatflux_no_noise`
- 確認周波数上限: 5 kHz、10 kHz
- 入力: 224 × 224 × 1、生のSTFT power
- 外側検証: `GroupKFold(source_wav_id)`、3 fold
- 学習: 200 epoch、SGD、学習率0.001、momentum 0.9、batch size 16
- 比較時はアンサンブルとXAIを無効化
- 最終確認では各fold・各モデルの直前に `seed = 42 + fold` を再設定し、実行順による初期値差を除外

06.11と06.18は18元WAV群（各60チャンク）、07.09は13元WAV群である。

## 3. 実装前の問題

### AlexNet

- `Flatten → Dense(4096) → Dense(4096)` により46,729,281パラメータ。
- 入力は非常に小さい生power（確認例: 最大約`3.4e-9`）で、対数圧縮なし。

### CNN+Transformer

- CNN後の7 × 7 × 256特徴を7 tokenへ圧縮。
- token埋め込みは32次元だが、Keras `MultiHeadAttention.key_dim` は各head 256次元、4 head。
- FFN 2048、Transformer 4 blockで、1,080サンプル規模に対して次元比が不均衡。

Keras公式仕様では`key_dim`は「各attention headのquery/key次元」である。したがって、従来値は総埋め込み32次元に対し、attention内部投影が過大だった。

## 4. 追加した構造

### 共通: LogPowerCompression

モデル先頭で次を適用できるようにした。

`log1p(max(power, 0) / 1e-12)`

サンプルごとの平均・分散正規化ではないため、熱流束と関係し得る絶対音響強度差を消さずにdynamic rangeを圧縮する。

音響CNNではlog-mel spectrogramが一般的に使用され、Audio Spectrogram Transformerも音響スペクトログラムをtokenとして扱う。本研究ではmel変換までは行わず、既存の線形周波数powerを維持してlog圧縮だけを導入した。

### AlexNet候補

| variant | 回帰ヘッド | log | パラメータ数 |
|---|---|---:|---:|
| `legacy` | 4096 → 4096 | なし | 46,729,281 |
| `compact_flatten` | 512 → 128 | なし | 7,068,481 |
| `gap` | GAP → 256 | なし | 3,791,425 |
| `gap_log` | GAP → 256 | あり | 3,791,425 |
| `legacy_log` | 4096 → 4096 | あり | 46,729,281 |
| `compact_flatten_log` | 512 → 128 | あり | 7,068,481 |

### CNN+Transformer候補

| variant | token | Transformer | log | パラメータ数 |
|---|---:|---|---:|---:|
| `legacy` | 7 | D=32、key/head=256、FFN=2048、4 block | なし | 4,852,833 |
| `balanced_axis` | 7 | D=64、key/head=16、FFN=256、2 block | なし | 3,940,609 |
| `balanced_spatial` | 49 | D=64、key/head=16、FFN=256、2 block | なし | 3,844,993 |
| `balanced_spatial_log` | 49 | 同上 | あり | 3,844,993 |
| `legacy_log` | 7 | legacyと同じ | あり | 4,852,833 |
| `balanced_axis_log` | 7 | balanced axisと同じ | あり | 3,940,609 |

## 5. 一次選別結果（06.18、5 kHz）

### AlexNet

| variant | R² mean | R² SE | RMSE |
|---|---:|---:|---:|
| legacy | 0.7743 | 0.1005 | 118,391 |
| compact_flatten | 0.7574 | 0.0874 | 126,418 |
| gap | 0.7890 | 0.0389 | 120,759 |
| gap_log | 0.9349 | 0.0193 | 66,620 |
| legacy_log | **0.9488** | **0.0025** | **60,358** |
| compact_flatten_log | 0.8826 | 0.0156 | 91,237 |

GAP化や全結合小型化だけでは改善しなかった。改善の中心はlog power圧縮である。精度優先では`legacy_log`、計算量優先では`gap_log`が候補になる。

### CNN+Transformer

| variant | R² mean | R² SE | RMSE |
|---|---:|---:|---:|
| legacy | 0.7026 | 0.1328 | 138,533 |
| balanced_axis | 0.7110 | 0.1520 | 132,455 |
| balanced_spatial | 0.3159 | 0.3760 | 204,388 |
| balanced_spatial_log | 0.9475 | 0.0112 | 60,506 |
| legacy_log | 0.9591 | 0.0144 | 52,370 |
| balanced_axis_log | **0.9623** | **0.0089** | **51,151** |

49-token化だけでは大幅に悪化した。log圧縮後は7-tokenで十分で、次元を整合した`balanced_axis_log`を採用する。

## 6. 完全対比較（06.18、10 kHz、同一seed規則）

| モデル | 従来R² | 選定R² | ΔR² | 従来RMSE | 選定RMSE | RMSE削減率 |
|---|---:|---:|---:|---:|---:|---:|
| CNN+Transformer | 0.7140 | **0.9642** | +0.2502 | 141,632 | **49,386** | **65.1%** |
| AlexNet | 0.8027 | **0.9486** | +0.1459 | 117,572 | **59,793** | **49.1%** |

同じ200 epochでも従来構造を大幅に上回った。また、過去の従来構造300 epoch結果も上回っており、単なる学習epoch増加では説明できない。

## 7. 3実験・2周波数での確認

| 実験 | maxfreq | CNN+Transformer R² ± SE | AlexNet R² ± SE |
|---|---|---:|---:|
| 06.11 | 5 kHz | 0.9367 ± 0.0213 | 0.9005 ± 0.0282 |
| 06.11 | 10 kHz | 0.9237 ± 0.0203 | 0.8998 ± 0.0218 |
| 06.18 | 5 kHz | 0.9553 ± 0.0107 | 0.9429 ± 0.0051 |
| 06.18 | 10 kHz | **0.9642 ± 0.0098** | **0.9486 ± 0.0093** |
| 07.09 | 5 kHz | 0.4792 ± 0.1998 | 0.5461 ± 0.1182 |
| 07.09 | 10 kHz | 0.5446 ± 0.2132 | 0.5127 ± 0.1556 |

06.11と06.18ではモデルと周波数上限をまたいで高精度が再現した。一方、07.09は改善後も低精度かつfold変動が大きい。

## 8. 07.09を追加深層化しない根拠

| 実験 | 元WAV群数 | heat flux vs 群平均log-power Pearson | 周波数bin別最大Spearman |
|---|---:|---:|---:|
| 06.11 | 18 | 0.942 | 0.967 |
| 06.18 | 18 | 0.942 | 0.977 |
| 07.09 | 13 | 0.687 | 0.599 |

07.09では中熱流束域のlog-powerが逆転・平坦化している。3～22 kHzの全上限で群平均log-powerのPearson相関は0.686～0.701に留まった。さらに07.09へ49-token構造を適用しても、axis型R²=0.479に対しspatial型R²=0.433へ悪化した。

以上より、07.09の律速は層数不足より次の可能性が高い。

- 元WAV/熱流束群が13群と少ない。
- 音響強度と熱流束の関係が中熱流束域で非単調。
- 同じ熱流束を説明する物理条件が音響以外にも存在する。

07.09に対して層を増やす探索は打ち切り、追加の熱流束群、繰返し録音、壁面過熱度・気泡状態などの物理補助量を優先する。

## 9. 採用判断

### 採用

- 共通入力: `LogPowerCompression(scale=1e-12)`
- CNN+Transformer: `balanced_axis_log`
  - 7 time token
  - model dimension 64
  - 4 heads、key dimension/head 16
  - FFN 256
  - 2 blocks、dropout 0.1
- AlexNet: 精度優先の`legacy_log`
- 各fold・各モデル直前のseed再設定

### 不採用

- 層をさらに増やすこと
- 49-token化
- AlexNet compact head（精度低下が大きい）
- 07.09に対する追加の構造探索

### 省計算案

AlexNet `gap_log`はR²=0.9349で、`legacy_log`より少し低いがパラメータ数を約92%削減できる。将来、精度より実行時間・保存サイズを優先する場合の代替候補として残す。

## 10. 論文上の結論候補

現時点では、単純なモデル深層化よりも、物理量として広いdynamic rangeを持つ生スペクトルpowerを、絶対強度差を保持したまま対数圧縮して入力することが、回帰精度とfold安定性の改善に支配的だった。CNN+Transformerでは、データ規模に合わせてattention次元とFFNを縮小しても性能を維持・改善できた。一方、実験07.09では音響量と熱流束の対応自体が弱く、モデル構造だけでは他実験と同等の精度に到達しなかった。この結果は、モデル性能の上限がネットワーク容量だけでなく、実験ごとの音響—熱流束関係と独立な元WAV群数に制約されることを示唆する。

## 11. 根拠資料

- TensorFlow/Keras MultiHeadAttention API: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
- Hershey et al., CNN Architectures for Large-Scale Audio Classification: https://research.google/pubs/cnn-architectures-for-large-scale-audio-classification/
- Gong et al., Audio Spectrogram Transformer: https://arxiv.org/abs/2104.01778
- Dosovitskiy et al., Vision Transformer: https://arxiv.org/abs/2010.11929

## 12. 保存結果

- AlexNet一次比較: `20260830_architecture_alex_screen`
- CNN+Transformer一次比較: `20260830_architecture_cnntf_screen`
- CNN+Transformer logアブレーション: `20260830_architecture_cnntf_log_ablation`
- AlexNet logアブレーション: `20260830_architecture_alex_log_ablation`
- AlexNet compact+log: `20260830_architecture_alex_compact_log`
- 3実験5 kHz確認: `20260830_architecture_log_confirmation_5khz`
- 3実験10 kHz確認: `20260830_architecture_log_confirmation_10khz`
- 07.09 spatial確認: `20260830_architecture_0709_spatial_check`
- 10 kHz従来構造の完全対比較: `20260830_architecture_paired_legacy_10khz`

