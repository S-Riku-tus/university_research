# 2026-07-16 モデル別説明性手法の再調査と実装方針

## 1. 結論

本研究では説明手法を1つに統一せず、モデル構造に合わせて主手法を変え、最後に同じ物理軸の摂動評価で比較する。

| モデル | 主手法 | 補助手法 | 物理的な横断比較 |
|---|---|---|---|
| `rf` (`XGBRFRegressor`, PCA 100成分入力) | PCA空間のTreeSHAP | PCA feature importance | 周波数帯・時間帯マスクによる性能低下 |
| `cnntf_v2_gap` | Integrated Gradients (IG) | 入力安定性・最終層ランダム化 | 同上 |
| `alexnet` | Integrated Gradients (IG) | Grad-CAM、入力安定性・最終層ランダム化 | 同上 |

RFのTreeSHAPはモデル自体には忠実だが、現状の入力特徴はPCA成分である。したがって「PC12が重要」という結果を「512 Hz以下が重要」と読み替えてはいけない。RFの物理的説明は、元のスペクトログラム上で周波数帯または時間帯を除去し、R2、ONB近傍RMSE、recall、false negativeがどれだけ悪化するかで示す。

## 1.1 2026-07-17時点の確認事項

今回の実装確認では、`cnntf_v2_gap` に対して Integrated Gradients (IG) が設定され、実際のXAI出力にも保存されていることを確認した。設定上は `VALIDATION_CONFIG["explainability"]["methods_by_model"]["cnntf_v2_gap"] = ["integrated_gradients", "group_occlusion"]` であり、対象条件6条件（3実験日 x `maxfreq=22kHz` x `heatflux_no_noise` / `heatflux_SNR=-20`）に対して、各5サンプル、合計30行の `integrated_gradients` が `explainability_summary.csv` に記録されていた。

`cnntf_v2_gap` は厳密な意味での本家Conformerではなく、CNN前段 + Transformer encoder + GlobalAveragePooling1D + 線形回帰出力のKerasモデルである。したがって、論文・報告書では「Conformer」と断定せず、`CNN+Transformer v2 GAP` と記述する方が安全である。IGは、本モデルが入力スペクトログラムからスカラーの熱流束予測値まで微分可能な計算グラフを持つため適用できる。

ただし、IGを計算できることと、その説明が常に信頼できることは別である。今回の確認では、`SNR=-20` 条件では completeness relative error が小さいサンプルが多い一方、`no_noise` 条件では大きいサンプルも見られた。そのため、IGは主説明として採用するが、completeness、入力安定性、最終層ランダム化、周波数帯・時間帯マスクで信頼性を併せて確認する。

現時点の研究上の読みは次の通りである。

- RFは性能面の最強ベースラインである。ただしPCA入力のため、PCA TreeSHAPを周波数帯の物理説明として直接使わない。
- CNN+Transformer v2 GAPとAlexNetは、入力スペクトログラムに対するIGで符号付き寄与を確認できる。ただし性能が条件依存であるため、説明結果はマスク実験で裏取りする。
- 全モデル共通の周波数帯・時間帯マスクは、モデル固有XAIを物理軸で比較するための共通土台である。

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

### 3.5 なぜ他の手法ではなくこの組み合わせにするか

手法選定の基準は、(1) モデル構造との整合、(2) 熱流束回帰出力に直接使えること、(3) 時間・周波数の物理解釈に接続できること、(4) 全モデルを同じ物理軸で比較できることである。したがって、各モデル固有の説明手法と、全モデル共通のマスク実験を分けて使う。

#### CNN+Transformer v2 GAP

- 主手法を Integrated Gradients (IG) とする。CNN前段、Transformer encoder、GlobalAveragePooling、線形回帰出力までを一つの微分可能な関数として扱い、入力スペクトログラムの各binが回帰熱流束をどちら向きに変えるかを符号付きで評価できるためである。
- Attention weight は主説明にしない。現行の v2 GAP は AttentionPooling ではなく GlobalAveragePooling で集約しており、MultiHeadAttention 内部の attention score をそのまま「入力周波数・時間の寄与」と読む構造ではない。また、attention は説明として不十分であり得ることが知られているため、本研究では主根拠ではなく、必要な場合の補助確認に留める。
- Grad-CAM は CNN+Transformer v2 GAP の主説明にはしない。最終予測はCNN特徴だけでなくTransformer encoderとGAPを通った後の値で決まるため、最終畳み込み層の粗い局在だけではモデル全体の寄与を十分に表せない。
- DeepSHAP / GradientSHAP / KernelSHAP も候補にはなるが、背景データ選択、計算量、TensorFlow実装の安定性、スペクトログラム高次元入力での扱いを考えると、まずは実装が単純で公理的性質と completeness を説明しやすい IG を主手法とする。

#### AlexNet

- 主手法を Integrated Gradients (IG) とする。AlexNetは入力スペクトログラムから畳み込み層・全結合層を通してスカラーの熱流束を出すため、入力解像度のまま符号付き寄与を得られるIGが、回帰出力の説明に最も直接的である。
- Grad-CAM は補助手法とする。CNNで広く使われる代表的手法であり、どの領域を大まかに見たかを図として示しやすい。一方で、最終畳み込み層由来の粗い局在図であり、熱流束への符号付き・定量的寄与ではないため、単独で「この帯域が物理的に重要」とは主張しない。
- vanilla gradient や saliency map はノイズが出やすく、baselineからの出力差を分解する性質も弱い。SmoothGradはノイズ低減には使えるが、主説明の理論的根拠としてはIGの方が説明しやすい。LRPは候補だが、層ごとの伝播則選択に依存し、現在のKeras実装に追加するコストが大きい。
- CAM系だけにしない理由は、AlexNetの予測が条件によって不安定なためである。Grad-CAMで高周波領域が強く見えても、それが回帰値を上げる寄与なのか、単なる粗い局在なのかを判別しにくい。IGとマスク実験で裏取りできた領域だけを強く述べる。

#### 全モデル共通の周波数帯・時間帯マスク

- モデル固有のXAIは出力の意味がそろわない。RFのTreeSHAPはPCA成分、IGは入力binの符号付き寄与、Grad-CAMは粗い相対重要度である。そのままではモデル間比較ができない。
- 周波数帯・時間帯マスクは、すべてのモデルに同じ問いを投げられる。「2-5 kHzを消すとR2が落ちるか」「10-15 kHzを消すとONB近傍RMSEやfalse negativeが増えるか」という形で、物理座標に沿った横断比較ができる。
- 本研究の目的は、可視化図を出すことではなく、沸騰音に関係する時間・周波数成分が熱流束回帰とONB検知に実際に効いているかを確認することである。したがって、マスク実験は補助というより、物理的説明の共通土台である。
- ただし、マスク実験は因果効果そのものではなく、入力摂動に対するモデル感度である。ゼロマスクは分布外入力を作る可能性があるため、結論では「当該帯域を除去したとき予測性能が悪化した」と表現し、「自然現象としてその帯域だけが原因である」とは言い切らない。

#### RFのPCA空間TreeSHAPの扱い

- PCA後特徴にTreeSHAPを適用すること自体は、モデルが実際に受け取っている特徴への説明としては正当である。すなわち「RFがPC成分にどう依存したか」を見るモデル内部監査として使える。
- しかし、PCAは複数の元特徴を混合する多変量変換であるため、PCA成分のSHAP値を元の周波数帯・時間帯の寄与として直接読み替えることはできない。標準化のような単変量変換とは違い、PCAではShapley値の意味が元特徴空間と一致しない。
- したがって、現状のRFではTreeSHAPを「PCA空間での忠実な説明」として保存し、物理的な周波数帯解釈は grouped frequency/time occlusion を主根拠にする。

## 3.6 論文・報告書での最終的な書き分け

論文では、説明性評価を「モデル固有の説明」と「物理軸での横断検証」に分けて記述する。前者は各モデルが内部的にどの入力・特徴に反応したかを確認するため、後者はその反応が実際に熱流束回帰やONB検知性能に影響するかを確認するために用いる。

| モデル | モデル固有の説明 | 物理的解釈の扱い | 論文上の主張の強さ |
|---|---|---|---|
| RF (`XGBRFRegressor`) | PCA空間のTreeSHAP | PCA成分への寄与として扱い、周波数帯へ直接読み替えない。物理説明はマスク実験で行う。 | 性能評価とモデル内部監査として強い。物理帯域主張はマスク実験に限定する。 |
| CNN+Transformer v2 GAP | Integrated Gradients | 入力スペクトログラムbinごとの符号付き寄与として扱う。attention weightは主説明にしない。 | IG + completeness + マスク実験が整合した場合に物理的説明として述べる。 |
| AlexNet | Integrated Gradients | 入力スペクトログラムbinごとの符号付き寄与として扱う。 | 主説明。ただしモデル性能が不安定な条件では慎重に述べる。 |
| AlexNet | Grad-CAM | CNNの粗い局在確認として扱う。 | 補助図。単独で周波数帯の重要性を結論しない。 |
| 全モデル | 周波数帯・時間帯マスク | R2、RMSE、ONB近傍RMSE、recall、false negativeの変化で評価する。 | 物理的な横断比較の主根拠。 |

本文での表現例:

> 本研究では、各モデルの構造に応じて説明手法を選択した。木ベースモデルであるRFにはTreeSHAPを適用したが、RFの入力はPCA成分であるため、TreeSHAPの結果はPCA空間におけるモデル内部の寄与として扱い、周波数帯への直接的な物理解釈には用いなかった。CNN+Transformer v2 GAPおよびAlexNetには、入力スペクトログラムからスカラーの熱流束回帰出力までの符号付き寄与を評価できるIntegrated Gradientsを適用した。さらに、全モデルに対して同一の周波数帯・時間帯マスク実験を行い、各帯域を除去した際のR2、ONB近傍RMSE、recall、false negativeの変化を比較することで、モデル固有の説明結果が物理的に意味のある時間・周波数領域と対応するかを検証した。

避けるべき表現:

- 「RFのTreeSHAPにより、2-5 kHzが重要であることが分かった」
  - PCA入力の場合、この言い方は不正確である。
- 「attention weightによりCNN+Transformerの判断根拠を説明した」
  - 現行 v2 GAP ではattention weightを主説明として使っていない。
- 「Grad-CAMによりAlexNetの熱流束寄与を定量化した」
  - Grad-CAMは粗い局在図であり、符号付き・定量的な熱流束寄与ではない。

推奨する表現:

- 「RFのTreeSHAPはPCA特徴量に対するモデル内部の寄与を示す」
- 「RFの物理的な周波数帯依存性は、周波数帯マスク時の性能低下から評価した」
- 「CNN+Transformer v2 GAPとAlexNetでは、IGにより入力スペクトログラム上の符号付き寄与を評価した」
- 「Grad-CAMはAlexNetの粗い局在を確認する補助図として用いた」
- 「全モデル共通のマスク実験により、説明結果とONB検知性能の関係を横断的に比較した」

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
