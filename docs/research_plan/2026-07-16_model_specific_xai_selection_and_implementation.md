# 2026-07-16 モデル別説明性手法の再調査と実装方針

## 1. 結論

本研究では説明手法を1つに統一せず、モデル構造に合わせて主手法を変え、最後に同じ物理軸の摂動評価で比較する。

| モデル | 主手法 | 補助手法 | 物理的な横断比較 |
|---|---|---|---|
| `rf` (`XGBRFRegressor`, PCA 100成分入力) | PCA空間のTreeSHAP | PCA feature importance | 周波数帯・時間帯マスクによる性能低下 |
| `cnntf_v2_gap` | Integrated Gradients (IG) | 入力安定性・最終層ランダム化 | 同上 |
| `alexnet` | Integrated Gradients (IG) | Grad-CAM、入力安定性・最終層ランダム化 | 同上 |

RFのTreeSHAPはモデル自体には忠実だが、現状の入力特徴はPCA成分である。したがって「PC12が重要」という結果を「512 Hz以下が重要」と読み替えてはいけない。RFの物理的説明は、元のスペクトログラム上で周波数帯または時間帯を除去し、R2、ONB近傍RMSE、recall、false negativeがどれだけ悪化するかで示す。

## 2. 研究との相性

本研究の説明対象は分類確率ではなく、連続値である予測熱流束である。また、最終的に知りたいのは「どの時間・周波数成分が熱流束回帰とONB近傍の見逃しに関与するか」である。このため、説明は次の2層に分ける。

1. モデル固有の局所説明
   - RF: TreeSHAP
   - CNN系: IG
   - AlexNetの空間的な補助表示: Grad-CAM
2. モデル非依存の物理検証
   - 周波数帯マスク時のR2低下、RMSE増加
   - ONB近傍RMSE増加
   - recall低下、false negative増加
   - 時間区間マスク時の同じ指標

これにより、「説明画像がそれらしく見える」だけでなく、重要とされた領域を除くと実際に予測性能とONB検知性能が悪化するかを確認できる。

## 3. 手法ごとの選定理由と一般性

### 3.1 RF: TreeSHAP

- 木構造を利用してSHAP寄与を多項式時間で計算でき、`XGBRFRegressor`と構造的に一致する。
- XGBoost公式APIの `pred_contribs=True` は各特徴のSHAP値とbiasを返し、総和が予測marginに一致する。
- SHAP原論文とTreeSHAP論文はいずれも非常に一般的である。2026-07-16参照時点の索引例では、Semantic ScholarでSHAP原論文が約31,900件、Nature掲載ページでTreeSHAP論文が約8,680件引用されている。ただし引用数は索引によって異なる。
- 最大の制約はPCAである。今回の実装ではTreeSHAPを計算・保存するが、出力に「物理軸へ直接解釈不可」と明記する。

### 3.2 CNN+Transformer v2 GAP: Integrated Gradients

- CNN前段、Transformer encoder、GlobalAveragePooling、線形回帰出力まで全計算グラフを通して入力寄与を計算できる。
- 現行v2はAttentionPoolingではなくGAPであり、MultiHeadAttentionもattention scoreを出力する構成ではない。attention重みを主説明として後付けするのは構造と一致しない。
- IGはSensitivityとImplementation Invarianceを動機とする公理的手法で、completenessにより入力寄与の総和と「入力予測－baseline予測」を照合できる。
- Semantic ScholarではIG原論文が約7,500件引用されており、TensorFlowにも公式チュートリアルがあるため、十分に一般的で再現しやすい。

### 3.3 AlexNet: Integrated Gradients + Grad-CAM

- IGは入力の224 x 224解像度で寄与を得られ、符号付き寄与を熱流束単位へ戻せるため主手法とする。
- Grad-CAMは最終畳み込み層に依存する粗い局在図であり、原論文自身もcoarse localization mapと位置づけている。周波数帯の精密な定量には使わず、CNNが大まかにどこを見たかの補助図とする。
- Grad-CAMは非常に一般的で、参照索引によって約18,000～25,000件規模の引用がある。ただし分類向けに導入された手法なので、本研究ではscalar regression出力へ勾配を取る拡張であることを明記する。

### 3.4 全モデル共通: grouped frequency/time occlusion

- 元のスペクトログラム座標で帯域をゼロ化するため、PCAを使うRFにも同じ物理軸で適用できる。
- 0は現在のpower spectrogramでは「その領域のエネルギーがない」状態を表す。ただし人工的な入力であるため、因果効果そのものではなく摂動感度として解釈する。
- 低周波は `0-256`, `256-512`, `512-1000`, `1000-2000 Hz` に細分し、それ以上は `2-5`, `5-10`, `10-15`, `15-22 kHz` とする。最大周波数が低い条件では帯域を自動的に切り詰める。

## 4. 説明の信頼性評価

### 4.1 IG completeness

IGの符号を保持し、目的変数のMinMaxScalerを逆変換して寄与を熱流束単位へ戻す。次を各標本で保存する。

`completeness error = sum(IG heat-flux contributions) - (prediction(x) - prediction(baseline))`

従来実装はIGを絶対値化・0-1正規化していたため、この検証ができなかった。可視化用の絶対値mapと、定量用の符号付きmapを分離する。

### 4.2 Deletion / Insertion for regression

Hama, Mase, Owen (JMLR 2023)に従い、予測曲線のAUCだけでなく、両端を結ぶ直線との差 `area_between_curve` を保存する。分類確率向けの「大きい方がよい／小さい方がよい」をそのまま流用せず、熱流束の単位と符号を残す。

### 4.3 Stability

入力標本の標準偏差の1%に相当する小さな非負摂動を加え、IG絶対値mapのPearson相関、cosine similarity、relative L1を保存する。これは水流音SNR条件間の頑健性とは別で、同一標本近傍で説明が急変しないかを測る。

### 4.4 Sanity check

Adebayo et al. (NeurIPS 2018)は、見た目だけではsaliency mapがモデルやデータに依存しているか判断できないと示した。今回の実装では計算量を抑えるため、最終trainable layerだけをランダム化してIG mapが変化するかを確認する。これは完全なcascading randomizationではないので、出力名も `top_layer_randomization_sanity.csv` とし、部分的スクリーニングとして扱う。

## 5. 実装上の安全策

- XAI対象を3実験日 x `22 kHz` x `no_noise/-16/-20 dB` x fold 1に限定する。
- 完了済みの500 epoch runにXAIがないだけで、自動的に再学習しない。
- XAI不足runを意図的に再学習するときだけ `retrain_completed_runs_for_xai=True` にする。
- RFのTreeSHAPはXGBoost組み込みのSHAP contributionを使い、外部`shap`パッケージを必須依存にしない。
- モデル間の最終比較は局所ヒートマップの見た目ではなく、`group_mask_performance.csv` の物理軸と性能差で行う。

## 6. 主な出力

- 全モデル
  - `group_mask_performance.csv`
  - `group_occlusion_summary.csv`
  - `explainability_summary.csv`
- RF
  - `treeshap_pca_values.csv`
  - `treeshap_pca_summary.csv`
  - `treeshap_status.csv`
- CNN+Transformer / AlexNet
  - `integrated_gradients_signed.npy/.png`
  - `integrated_gradients_magnitude.npy/.png`
  - `integrated_gradients_signed_frequency_profile.csv`
  - `input_stability.csv`
  - `top_layer_randomization_sanity.csv`
- AlexNetのみ
  - `grad_cam.npy/.png`

## 7. 参考文献・実装資料

- Lundberg, Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- Lundberg et al., "From local explanations to global understanding with explainable AI for trees," Nature Machine Intelligence 2020. https://doi.org/10.1038/s42256-019-0138-9
- XGBoost Python API (`pred_contribs`). https://xgboost.readthedocs.io/en/stable/prediction.html
- Sundararajan, Taly, Yan, "Axiomatic Attribution for Deep Networks," ICML 2017. https://proceedings.mlr.press/v70/sundararajan17a.html
- TensorFlow, "Integrated gradients." https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
- Selvaraju et al., "Grad-CAM," ICCV 2017. https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
- Hama, Mase, Owen, "Deletion and Insertion Tests in Regression Models," JMLR 2023. https://jmlr.org/papers/v24/22-0560.html
- Adebayo et al., "Sanity Checks for Saliency Maps," NeurIPS 2018. https://research.google/pubs/sanity-checks-for-saliency-maps/
- Letzgus et al., "Toward Explainable Artificial Intelligence for Regression Models," IEEE Signal Processing Magazine 2022. https://arxiv.org/abs/2112.11407
- Cremades, Hoyas, Vinuesa, "Additive-feature-attribution methods: a review on explainable artificial intelligence for fluid dynamics and heat transfer," International Journal of Heat and Fluid Flow 2025. https://doi.org/10.1016/j.ijheatfluidflow.2024.109662
- Jain, Wallace, "Attention is not Explanation," NAACL 2019. https://aclanthology.org/N19-1357/

