# 2026-06-26 説明性手法・評価指標の選定（3モデル確定版＋選定理由）

作成日: 2026-06-26
位置づけ: 同日の [2026-06-26_xai_method_selection.md](2026-06-26_xai_method_selection.md)（4モデルの「当たりづけ」初稿）を踏まえ、**実際に使う3モデルに絞って手法を確定し、選定理由を反論されにくい形で言語化**し、さらに「説明の質を数値で検証する評価指標」まで含めて整理した版。初稿は履歴として残す（上書きしない）。
参照: `docs/notes/model_notes.md`, `docs/research_context.md`, `docs/research_plan/2026-06-19_after_presentation.md`

---

## 0. このメモの目的

- 6/19発表で本研究の主戦場は分類でなく**熱流束回帰**、説明性も**回帰出力に直接**付与する（横＝周波数帯で寄与を見る）と整理された。その③（説明性付与）に向けた**手法と評価指標の確定**。
- 動機: 「なぜその手法を選んだのか」が曖昧だと発表・査読で突かれる。**選定が場当たりに見えないよう、1つの原理から各モデルの最適手法を導く**形にする。
- 実装はまだ。本メモは選定の確定と根拠の記録。

## 1. 対象モデル（実際に使っている3つ）

| key | label（コード上） | 実体 | 入力 |
| --- | --- | --- | --- |
| `rf` | RandomForest | XGBRFRegressor。flatten→**PCA100次元**で学習 | PCA特徴 |
| `alexnet` | AlexNet | 微分可能CNN＋線形回帰出力1個 | `(224,224,1)` 周波数×時間 |
| `cnntf_v1` | "Conformer" | **CNN前段＋Transformer＋AttentionPooling**（後述: 厳密にはConformerでない） | `(224,224,1)` 周波数×時間 |

- Swin Transformerは対象外。
- `cnn_transformer_v2` は `base_regression.py` に定義が残っており旧2スクリプトから参照されるが、本番パイプライン `run_ensemble_regression_onb.py` では未使用（休眠）。**今回は名称修正もv2削除も行わない**（§9に記録のみ）。

## 2. 結論サマリ

| モデル | 主手法（確定） | 補助 | 説明の検証（評価指標） |
| --- | --- | --- | --- |
| RandomForest | **TreeSHAP** | permutation importance | sensitivity-n／SHAP安定性、Deletion/Insertion、周波数帯マスクΔR2 |
| AlexNet | **Integrated Gradients (IG)** | Grad-CAM（粗い可視化） | Deletion/Insertion AUC、Adebayo sanity check、周波数帯マスクΔR2 |
| Conformer/CNN+Tf (v1) | **Integrated Gradients (IG)** | AttentionPooling重み＋Attention Rollout | 同上＋時間パッチマスクΔ |

DL系で**SHAPに揃えたい場合**は IG の代わりに GradientSHAP / DeepSHAP も可（同じ加法的属性族）。

## 3. 全体を貫く1つの原理（「なぜこの3手法か」の芯）

> 本研究は **「加法的特徴属性（additive feature attribution＝efficiency/completeness公理を満たす属性法）」を主軸**とし、各モデルの構造に対して**厳密 or 最適な実装**を選ぶ。

この1原理が、定番性・回帰適合・分野整合をまとめて回収する。

- **分野整合**: 熱伝達・流体のXAIレビュー（Cremades et al. 2024）が、この分野の標準を additive feature attribution（kernel / tree / gradient / deep SHAP）として体系化し、「physics-aligned（物理整合的）」解釈を重視している。RF=TreeSHAP、DL=gradient系という割当はこのレビューに沿う。
- **回帰適合**: efficiency/completeness公理により、属性の総和が「予測熱流束 − 基準値」に一致する。→「周波数帯 i は熱流束予測に **+◯◯ W/m² 寄与**」と**単位付きで分解**できる（ヒートマップの見た目でなく数値で主張できる）。
- **選定の必然性**: 「同一原理の中で各構造に最も正確な実装を選んだ」と言えるので恣意性がない。

## 4. 用語: 「説明性指標」は2層に分かれる

| 層 | 中身 | 代表 | 本研究での扱い |
| --- | --- | --- | --- |
| 層A: 説明**手法** | 説明そのものを生成 | TreeSHAP, IG, Grad-CAM | §5で確定 |
| 層B: 説明の**評価指標** | 説明が信用できるかを採点 | Deletion/Insertion, sanity check, robustness | §6で確定 |

「説明性指標を踏まえた**検証**」の本体は層B。層Aと層Bを両方そろえて初めて「根拠を出し、その根拠が信用できることも数値で示した」と言える。

## 5. モデル別の選定ロジック（①構造整合→②回帰・数値分解→③定番性→④代替案の棄却）

④（なぜ他を選ばなかったか）が反論対策の本体。

### 5.1 RandomForest → TreeSHAP

- **①構造整合**: 木アンサンブルに対しTreeSHAPは木構造を使い**厳密なShapley値を多項式時間**で算出（汎用KernelSHAPは近似・低速）。木モデルに厳密版を使わない理由がない。
- **②回帰・数値分解**: 連続値出力に直接適用、efficiency公理で「属性総和 = 予測熱流束 − ベース値」。帯域別・時間別寄与をW/m²で出せる。
- **③定番性**: SHAP（Lundberg & Lee, NeurIPS 2017、被引用2万級）、TreeSHAP（Lundberg et al., Nature Machine Intelligence 2020）。熱伝達XAIレビューがtree SHAPを分野標準として明記。
- **④代替案の棄却**:
  - Gini/split-based importance（XGBoost標準）: 高カーディナリティ・相関へのバイアス、**一貫性公理を満たさない**（Lundbergが反例提示）、グローバルのみ。→不適。
  - permutation importance: グローバル＆相関に弱い。→交差確認の補助に留める。
  - LIME: 局所線形の近似で不安定、公理なし。木には厳密なTreeSHAPが上位互換。→不要。
- **🚩 最大の前提（先に潰す突っ込みどころ）**: 現状RFの入力は flatten→**PCA100次元**（`utils/training/model_training.py: make_pca`）。TreeSHAPは「PCA成分の寄与」しか出せず**物理的な周波数・時間に戻せない**。物理的主張には **② 解釈可能特徴（周波数帯×時間ビンのエネルギー）への置換が前提**。→ **② → ③ の順序依存**は確定事項。

### 5.2 AlexNet → Integrated Gradients（主）／Grad-CAM（補助）

- **①構造整合**: 微分可能CNN＋**線形回帰ニューロン1個**。IGは「微分可能性＋ベースライン」だけで成立し、線形出力なのでcompletenessが「予測熱流束 − ベースライン熱流束」として直接意味を持つ。
- **②回帰・数値分解**: IGは**入力解像度(224×224)**で属性 → 周波数軸・時間軸に細かく当たり、帯域集約でW/m²寄与に変換できる。周波数帯の同定が目的の本研究に直結。
- **③定番性**: Integrated Gradients（Sundararajan et al., ICML 2017、公理的）。熱伝達レビューのgradient/deep SHAP系と同族。
- **④代替案の棄却**:
  - Grad-CAMを主にしない理由: 最終conv層の特徴マップに依存し、AlexNetでは最終conv特徴マップが**約6×6**まで縮む→224×224へ拡大するので**周波数軸の局在が粗い**（Grad-CAMは"coarse localization"、音響XAIでも"diffuse, limited frequency localisation"）。「どの周波数帯か」を主張する本研究では粒度不足。かつclass-discriminativeだが**completenessが無く数値分解にならない**。→「どこを見ているか」の補助可視化に格下げ。
  - 生saliency: 勾配飽和・ノイズが強い。→IG（経路積分で飽和緩和）が上位。
- **補足**: SHAP族に統一したいなら GradientSHAP / DeepSHAP（IGとほぼ同じcompleteness系）。RFのTreeSHAPと「全モデルadditive attribution」と言い切れて横断比較が綺麗。

### 5.3 Conformer / CNN+Tf (v1) → Integrated Gradients（主）／Attention系（補助）

- **①構造整合**: 「共有CNN（局所）→ Transformer（大域）→ AttentionPooling」で全段微分可能。**IGは計算グラフ全体（conv＋attention＋pooling）を貫いて属性**するので、畳み込み（局所）と自己注意（大域）の両寄与を1つの数値に統合できる。"局所＋大域を両取りする"設計意図に合致。
- **②回帰・数値分解**: completenessで**時間パッチ別・周波数別の寄与**が出る。**時間パッチ寄与はONB近傍の早期検知（④）の物理的裏付け**に直結。
- **③定番性**: 同上、公理的・定番。
- **④代替案の棄却（重要）**:
  - Attention重みを主にしない理由: 「Attention is not Explanation」（Jain & Wallace 2019）＋層をまたぐtoken混合で信頼性低下（Abnar & Zuidema 2020）。さらに、**仮に本物のConformer（Conv module入り）にした場合、attention重みはConv moduleの寄与を一切捉えない**。→attention単体は主手法にできない。
  - **AttentionPoolingの重み**（最終段が出す時間パッチ重要度。実装上ただで得られる）と **Attention Rollout** は、IGの結論と突き合わせる**安価な整合確認・補助**として併用。

## 6. 説明の評価指標（層B）— どの手法でも信頼性を同じ土俵で採点する

説明を出すだけでは「見た目がそれっぽいだけ」と突かれる。生成した説明を以下で**数値検証**する。

- **Faithfulness（忠実性・最重要）**:
  - **Deletion / Insertion AUC**（Petsiuk et al., BMVC 2018）: 重要と言われた特徴から順に消す/足すと予測が大きく動くか。手法非依存の標準スコア。スコアは予測確率でなく**予測熱流束（またはΔR2/ONB近傍誤差）**に差し替える。
  - 補助: AOPC、sensitivity-n（Ancona et al., ICLR 2018）、infidelity（Yeh et al., NeurIPS 2019）。
- **Randomisation / Sanity check（健全性）**: Adebayo et al., NeurIPS 2018。モデル重みをランダム化して説明が壊れるか。**CNN/Transformerの勾配系は必須**（壊れない＝モデルを見ていない危険信号）。木モデルはラベルシャッフル版が有効。
- **Robustness（安定性）**: 入力微小変化に対する説明のブレ（max-sensitivity、SHAP値の標準偏差・相関>0.90目安）。
- **回帰特有の注意**（Letzgus et al. XAIR 2022, BEExAI 2024）:
  - **参照点（baseline / base value）を固定して明示**する。属性は「参照点からのズレ」の分解なので、参照（no_noiseの低熱流束 or 訓練平均熱流束など）で値が変わる。
  - 特徴を消す順位付けは**符号でなく絶対値**で（正負が打ち消すため）。
- **摂動のOOD問題**: ゼロ埋め/ノイズ付与は分布外入力を作り、「重要度」でなく「分布シフトへの反応」を測る恐れ。→ **ゼロ＋帯域平均値での置換の両方**で評価し、順位が摂動法に依存しないことを確認する。

## 7. 3モデル共通の数値比較軸（「同じ土俵」の担保）

1. **加法的寄与の帯域プロファイル**: RF=TreeSHAP、DL=IG（or GradientSHAP）。efficiency/completenessで「**熱流束への寄与 W/m²**」という**同一スケール**の周波数帯プロファイルを出す→直接比較可能。
2. **Deletion / Insertion AUC**＋**周波数帯マスクΔR2 / 時間帯マスクΔ**: 全モデルを同一指標で採点。
3. **文献整合**: 沸騰音→熱流束予測で**低周波(<512Hz)が重要**との報告がある。属性がそこに当たれば妥当性の傍証、外れれば考察すべき発見（localisation/期待整合の軸）。

→「手法は構造ごとに最適化したが、属性は同じ加法的分解として比較し、信頼性も同じ指標で採点した」と一貫説明できる。

## 8. 論文・発表用の選定理由テンプレ（1モデル4行）

> **[モデル]** はアーキテクチャ的に **[構造]** である。説明手法には **[手法]** を採用した。理由は、(1) 構造上 [手法] が[厳密/最適]に適用でき、(2) 回帰出力をefficiency/completeness公理で単位付き分解でき、(3) [被引用・分野標準]として確立しており、(4) 代替の [手法X] は [理由] のため本タスクに不適だからである。

- **RandomForest → TreeSHAP**:（1）木構造で厳密Shapley、（2）W/m²で帯域分解、（3）SHAP/Nature MI・熱伝達レビュー標準、（4）Gini重要度は一貫性公理を満たさずバイアス、LIMEは近似不安定。**前提=PCA→解釈可能特徴の置換**。
- **AlexNet → Integrated Gradients**:（1）微分可能＋線形出力でcompleteness直結、（2）入力解像度で周波数局在、（3）ICML2017公理的、（4）Grad-CAMは最終conv約6×6で周波数粒度不足→補助。
- **Conformer/CNN+Tf → Integrated Gradients**:（1）conv＋attentionの全寄与を統合、（2）時間パッチ寄与＝ONB裏付け、（3）公理的・定番、（4）attention単体は説明不十分(Jain&Wallace)＋Conv寄与を捉えない→補助。

## 9. 実装の足場（参考）

- **SHAP**（TreeSHAP）: RF用。厳密・高速。
- **Captum**: Integrated Gradients / GradientSHAP / Grad-CAM、Infidelity・Sensitivity を内蔵。ただしPyTorch前提。
- **Quantus**（Hedström et al., JMLR 2023）: faithfulness/robustness/randomisation等35+指標。Deletion/Insertionの「スコア」を熱流束/ΔR2へ差し替える改造が要る。
- 本研究はTensorFlow/Keras実装のため、PyTorch前提ライブラリは「Keras対応 or 自前実装」の見積もりが必要（IGはKerasでも自前実装が容易）。

## 10. コード上の注意（今回は記録のみ。修正は未実施）

- **命名**: `cnn_transformer_v1` は label="Conformer" だが、実体は **CNN前段＋Transformer＋AttentionPooling**。本家Conformer（Gulati et al. 2020）はブロック内で `½FFN→自己注意→Conv module→½FFN`（macaron）と畳み込みと注意を交互に通す構造で、v1にはブロック内Conv moduleが無い。発表で「Conformer」と書くと定義を知る相手に突かれうる。→ 将来 "CNN+Transformer" へ改名 or 本物のConformer化を検討（XAIの結論はどちらでもIGで不変）。`utils/plotting/regression_plots.py` には既に正直名 `"CNN+Tf (AttnPool)"` がある。
- **v2**: `cnn_transformer_v2` は定義が残り旧2スクリプト（`3.run_ensemble_ROC_100%_analysis.py`, `compare_predict_heatflux.py`）から参照されるが本番未使用。整理時に削除候補。

## 11. 先生に確認したいこと

1. 主手法を **RF=TreeSHAP / AlexNet=IG / Conformer=IG**（DLはSHAP族で揃えるなら GradientSHAP）で確定してよいか。
2. 信頼性は **Deletion/Insertion AUC＋sanity check（DL）＋SHAP安定性（RF）** という標準評価指標で担保し、加えて**沸騰音文献の低周波(<512Hz)期待値**と照合する方針でよいか。
3. 比較軸を「帯域別寄与(W/m²)／Deletion-Insertion／周波数帯マスクΔR2」の3数値で出す方針でよいか。
4. 回帰の**参照点**（no_noise低熱流束／訓練平均など）をどこに固定するか。
5. **② PCA→解釈可能特徴（周波数帯×時間ビン）への置換を ③ より先**に行う順序でよいか。

## 12. 参考文献

- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP), NeurIPS 2017.
- Lundberg et al., "From local explanations to global understanding with explainable AI for trees" (TreeSHAP), Nature Machine Intelligence 2020.
- Sundararajan, Taly, Yan, "Axiomatic Attribution for Deep Networks" (Integrated Gradients), ICML 2017. arXiv:1703.01365
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017. arXiv:1610.02391
- Ribeiro, Singh, Guestrin, "Why Should I Trust You?" (LIME), KDD 2016.
- Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition", Interspeech 2020. arXiv:2005.08100
- Jain & Wallace, "Attention is not Explanation", NAACL 2019 / Wiegreffe & Pinter, "Attention is not not Explanation", EMNLP 2019.
- Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (attention rollout), ACL 2020 / Chefer et al., "Transformer Interpretability Beyond Attention Visualization", CVPR 2021.
- Petsiuk, Das, Saenko, "RISE: Randomized Input Sampling for Explanation" (Deletion/Insertion), BMVC 2018.
- Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018. arXiv:1810.03292
- Letzgus et al., "Toward Explainable AI for Regression Models" (XAIR), IEEE Signal Processing Magazine 2022. arXiv:2112.11407
- Ancona et al., "Towards better understanding of gradient-based attribution methods" (sensitivity-n), ICLR 2018 / Yeh et al., "On the (In)fidelity and Sensitivity of Explanations", NeurIPS 2019.
- Hedström et al., "Quantus: An Explainable AI Toolkit for Responsible Evaluation", JMLR 2023. arXiv:2202.06861
- Cremades, Hoyas, Vinuesa, "Additive-feature-attribution methods: a review on explainable AI for fluid dynamics and heat transfer", Int. J. Heat and Fluid Flow 2024. arXiv:2409.11992
- 沸騰音→熱流束（低周波<512Hzが重要）: "Nonintrusive heat flux quantification using acoustic emissions during pool boiling", Applied Thermal Engineering 2023.
