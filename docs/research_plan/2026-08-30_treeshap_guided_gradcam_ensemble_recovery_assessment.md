# RF説明性・Guided Grad-CAM・アンサンブル再設計の総合評価

作成日: 2026-08-30  
対象: 現行 `run_ensemble_regression_onb.py`、現在までの保存済みOOF結果、学部卒業研究の論文・スライド・旧コード

## 0. 結論の要約

今回の3つの問いに対する結論は、次のとおりである。

1. **現行RFのまま、元スペクトログラム画素に「厳密なTreeSHAP」を直接付与することはできない。**
   現行の木モデルが実際に受け取る特徴はPCA 100成分であり、TreeSHAPが説明するプレイヤーもPCA成分である。PCAの負荷量を用いてSHAP値を元画素へ逆投影しても、それは一般にはShapley値にならない。元空間のTreeSHAPが必要なら、元画素または物理的な集約特徴を入力として木モデルを再学習する必要がある。

2. **Guided Grad-CAMは実装可能だが、Grad-CAMやIntegrated Gradientsの置き換えにはしない方がよい。**
   Guided Grad-CAMは、Grad-CAMの粗い領域とGuided Backpropagationの細かい入力勾配を掛け合わせた高解像度図である。見た目は細かくなるが、物理的・因果的により正しいとは限らない。特にGuided Backpropagation系は、学習済み重みを壊しても図が変わりにくいというsanity check上の問題が報告されている。今回のスペクトログラムでは、線や境界を強調した「もっともらしい図」になる危険がある。

3. **アンサンブルを最終的にRF単体より高精度にする余地は残っているが、現在の固定平均やR²逆数重みの微調整だけでは難しい。**
   本命は、深層モデルをまず安定・校正し、録音元を分離したcross-fittingで得た予測だけを使って、RFの残差を小さく補正する方式である。さらに、全モデルが同じ2–5 kHzを見るのではなく、主要帯域を扱うRFと、他帯域・時間変動・ノイズ状態からRF残差を説明する補助専門家に役割を分ける。

4. 学部研究の結果は「当時の評価手順内ではアンサンブルが良かった」という事実である。ただし、現在と同じ厳しさの独立評価ではない。旧コードは、0.5秒chunkを通常のKFoldでランダム分割し、さらに外側のvalidation foldの正解値から各モデルの重みを計算して、同じfoldを評価していた。このため、学部時代の絶対性能や改善幅を、現在のGroupKFold結果と直接比較してはいけない。

5. 修士研究として最も強い物語は、次の流れである。

   > 学部研究では同系統CNNの平均化による性能向上を確認した。修士研究では、RFとTransformerを加えた異種モデル統合へ拡張したが、モデルを増やすだけでは最良単体を上回らなかった。その原因を個別精度、予測校正、残差共分散、利用周波数帯から特定し、物理帯域分担型・RFアンカー残差アンサンブルを提案した。録音元・実験日を分離した評価により、その有効条件を検証した。

この物語なら、最終的に提案アンサンブルが勝てば希望している結論へ到達できる。勝たなかった場合にも、「なぜ学部時代には有効で、厳密評価では何が成立しなかったか」が研究成果として残る。

---

## 1. 学部卒業研究のアンサンブルを復元した結果

確認資料:

- `研究進捗報告/卒論_柴崎陸.pdf`
- `研究進捗報告/8121048_柴崎陸_最終投稿卒論.pdf`
- `研究進捗報告/2024/1月24日(最終卒論発表)/卒論最終発表_柴崎陸.pptx`
- `研究進捗報告/2024/1月24日(最終卒論発表)/卒論最終発表_前刷り_柴崎陸.docx`
- `研究進捗報告/2024/1月24日(最終卒論発表)/卒論_2024_AUC_R2棒グラフ.xlsx`
- `archive/code/3.run_ensemble_100percent_classification.py`

### 1.1 当時の方式

- モデル: AlexNet、ResNet50、VGG16
- 入力: 60秒音響を0.5秒chunkに分割し、STFT後に224×224へリサイズ
- ノイズ: no-noise、0、-4、-8、-12、-16、-20 dB
- 交差検証: 5分割KFold
- アンサンブル: 3モデルの重み付き平均
- 重み: 各モデルについてvalidation fold上で `1 - R²` を計算し、その逆数を正規化

すなわち、モデル (m) の重みは概ね次式だった。

\[
w_m = \frac{1/(1-R_m^2)}{\sum_j 1/(1-R_j^2)}
\]

### 1.2 卒論図に保存されているR²

| SNR | AlexNet | ResNet50 | VGG16 | Ensemble | Ensemble − best single |
|---:|---:|---:|---:|---:|---:|
| no-noise | 0.9918 | 0.9950 | 0.9929 | 0.9962 | +0.0012 |
| 0 dB | 0.9951 | 0.9923 | 0.9928 | 0.9968 | +0.0017 |
| -4 dB | 0.9800 | 0.9828 | 0.9815 | 0.9838 | +0.0010 |
| -8 dB | 0.9677 | 0.9741 | 0.9731 | 0.9774 | +0.0033 |
| -12 dB | 0.9473 | 0.9639 | 0.9611 | 0.9648 | +0.0009 |
| -16 dB | 0.9272 | 0.9379 | 0.9360 | 0.9378 | -0.0001 |
| -20 dB | 0.9107 | 0.9081 | 0.9104 | 0.9165 | +0.0058 |

平均改善幅は約+0.0020である。7条件中6条件では最良単体より高いが、-16 dBではResNet50を0.0001下回っている。

### 1.3 卒論図に保存されているAUC

| SNR | AlexNet | ResNet50 | VGG16 | Ensemble | Ensemble − best single |
|---:|---:|---:|---:|---:|---:|
| no-noise | 0.9974 | 0.9975 | 0.9974 | 0.9986 | +0.0011 |
| 0 dB | 0.9979 | 0.9974 | 0.9975 | 0.9987 | +0.0008 |
| -4 dB | 0.9913 | 0.9911 | 0.9919 | 0.9938 | +0.0019 |
| -8 dB | 0.9862 | 0.9865 | 0.9873 | 0.9901 | +0.0028 |
| -12 dB | 0.9678 | 0.9750 | 0.9778 | 0.9807 | +0.0029 |
| -16 dB | 0.9366 | 0.9312 | 0.9505 | 0.9544 | +0.0039 |
| -20 dB | 0.8871 | 0.8792 | 0.9019 | 0.9101 | +0.0082 |

AUCは7条件すべてでアンサンブルが最良であり、平均改善幅は約+0.0031だった。

ただし、旧コードは熱流束予測値を一度0/1に閾値化してから `roc_curve` に渡している。このAUCは連続スコアを使う通常のROC-AUCではなく、単一動作点に近い評価である。修士論文では現在の定義へそのまま引き継がない方がよい。

### 1.4 学部結果が現在よりアンサンブルに有利だった理由

#### 理由A: 3モデルがほぼ同程度に強かった

no-noiseのR²は0.9918、0.9950、0.9929であり、明確に一つだけが支配的ではなかった。この条件では、少し異なる誤りを平均して分散を下げやすい。

現在は、RFのR²が概ね0.90前後である一方、CNN+TransformerやAlexNetは条件によって大幅に低くなる。弱いモデルへ正の重みを与えるだけで、強いRFの予測が薄まる。

#### 理由B: validation foldの正解を重み計算と性能評価の両方に使用していた

旧コードでは、各モデルが外側validation foldを予測した後、その同じ `y_val` からR²を計算し、重みを決め、同じ `y_val` に対してensemble R²を評価している。

これは現在のコードでいう `val_fold_legacy` に相当する。再現実験としては意味があるが、未知データ性能を主張する方法にはできない。

#### 理由C: chunk単位の通常KFoldだった

旧コードは `KFold(...).split(x)` を用いており、録音元IDを分離していない。同じ60秒録音から作られた似た0.5秒chunkがtrainとvalidationへ分散する可能性がある。

一方、現行コードはmanifestがあれば `GroupKFold(source_wav_id)` を使用する。したがって、現行評価の方が「未知録音へ一般化するか」という問いに近く、R²が下がるのは不自然ではない。

#### 理由D: 改善幅自体は小さい

学部時代のR²改善は平均約0.0020である。結果の方向は一貫しているが、独立録音を単位にした信頼区間やpaired testは行われていないため、論文中の「有意性」は統計的有意差まで証明したものではない。

### 1.5 学部結論を否定する必要はない

適切な表現は次である。

> 学部研究のデータと評価手順では、3 CNNの重み付き平均がほぼ全ノイズ条件で最良だった。ただし、重み推定と評価に同じvalidation foldを用い、録音chunkをランダム分割していたため、修士研究では録音元を分離した、より厳しい評価で再検証する必要がある。

これは学部結論の撤回ではなく、修士研究で評価を一段強くする自然な発展である。

---

## 2. RFに元入力空間のTreeSHAPを適用できるか

### 2.1 現行モデルの実体

現在「RandomForest」と表示されるモデルは、厳密には `XGBRFRegressor` である。入力の流れは次のようになっている。

\[
X_{224\times224\times1}
\rightarrow X_{50176}
\rightarrow PCA_{100}
\rightarrow XGBRFRegressor
\rightarrow \hat y
\]

TreeSHAPは木が分岐に使う特徴を説明する。したがって現行の厳密なTreeSHAP値は、100個のPCA成分が予測を基準値からどれだけ動かしたかを表している。

[TreeSHAPの原論文](https://arxiv.org/abs/1802.03888)が保証する局所加法性・一貫性は、モデルへ入力された特徴に対する保証であり、前処理前の意味を自動的に復元する保証ではない。

### 2.2 PCA負荷量による逆投影がTreeSHAPにならない理由

PCAを (z=P(x-\mu))、木モデルを (g(z)) とすると、全体モデルは次式である。

\[
f(x)=g(P(x-\mu))
\]

TreeSHAPで得られる φ\(_{z_j}\) は、「PCA成分 (z_j) をゲームの一参加者としたとき」の寄与である。一方、元画素 (x_k) を参加者とするゲームでは、1画素を有無にするたびに多数のPCA成分が同時に変化し、木の分岐経路も非線形に変わる。

したがって、例えば

\[
\phi_x = P^T\phi_z
\]

のような単純逆変換には、一般に次の保証がない。

- 元画素をプレイヤーとしたShapley値であること
- 全画素寄与の和が予測差と一致すること
- 相関した画素間の寄与配分が妥当であること
- 符号と単位の物理解釈が保存されること

PCA負荷量とTreeSHAPを組み合わせた可視化を作ること自体は可能だが、名称は「PCA-SHAP loading projection」などとし、「元画素TreeSHAP」とは呼ばない方がよい。

### 2.3 元空間で説明する選択肢

| 選択肢 | 厳密性 | 物理解釈 | 計算・過学習リスク | 推奨度 |
|---|---|---|---|---|
| 現行PCA TreeSHAP | PCA空間では厳密 | PCA成分なので低い | 低い | モデル監査用に維持 |
| 物理集約特徴で木を再学習しTreeSHAP | 入力特徴に対して厳密 | 高い | 中 | 最有力 |
| 元224×224画素で木を再学習しTreeSHAP | 画素入力に対して厳密 | 個別画素で不安定 | 非常に高い | 低い |
| 現行合成モデルへgroup SHAP | group定義に対する近似 | 高い | 中～高 | 局所説明の候補 |
| 現行group mask/occlusion | Shapley値ではない | 高い | 既に実装済み | 物理主張の主方法 |
| PCA-SHAP逆投影 | heuristic | 見た目は得られる | 誤解リスク大 | 補助図のみ |

### 2.4 最も現実的な改善案

RF用に、次のような物理特徴表現を別候補として作るのがよい。

- 周波数帯: 0–0.5、0.5–1、1–2、2–5、5–10、10–15、15–22 kHz
- 時間: 1秒を4区間または8区間
- 各領域の平均、標準偏差、最大、上位percentile、時間変動量
- spectral centroid、帯域エネルギー比、スペクトル傾斜
- 2–5 kHz内のピーク数・間欠性・時間方向変動

例えば8帯域×4時間区間×3統計量なら96特徴程度であり、現在のPCA 100成分と同程度の次元数を保ちながら、TreeSHAPを物理名の付いた特徴へ直接適用できる。

ただし、これはモデルを変更するため、現在のPCA-RFと同じGroupKFoldで精度を比較する必要がある。説明しやすさのために精度を大きく失うなら、予測主モデルはPCA-RF、物理説明はgroup maskという現在の役割分担を維持する。

### 2.5 推奨結論

- PCA TreeSHAP: 「RF内部でどの潜在成分が予測を動かしたか」の監査
- group mask: 「どの物理周波数帯・時間帯が性能に必要か」の主張
- 物理集約特徴RF: 解釈可能性と精度が両立するかを比較する追加モデル

この三層構造が、無理にPCA SHAPを元画像へ戻すより科学的に強い。

---

## 3. Guided Grad-CAMは今回に合うか

### 3.1 まず現行設定の確認

現行 `VALIDATION_CONFIG` を読む限り、説明手法は次の割り当てである。

| モデル | 現行の局所説明 | 現行の物理領域評価 |
|---|---|---|
| AlexNet | Integrated Gradients、Grad-CAM | group occlusion |
| CNN+Transformer | Integrated Gradients | group occlusion |
| RF | PCA TreeSHAP | group occlusion |

したがって、**現行設定ではGrad-CAMはAlexNetだけであり、CNN+Transformerには保存しない設定**である。ただし実装関数は最後のConv2D層を自動選択するため、設定と検証を追加すればCNN+Transformerにも技術的には適用できる。

### 3.2 Guided Grad-CAMとは何か

[Grad-CAM原論文](https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)では、Guided Grad-CAMを次の積として提案している。

\[
\text{Guided Grad-CAM}
=
\text{Upsampled Grad-CAM}
\odot
\text{Guided Backpropagation}
\]

- Grad-CAM: 最終畳み込み特徴マップから得る粗い位置情報
- Guided Backpropagation: 入力解像度の細かい勾配パターン
- Guided Grad-CAM: 粗い領域内だけ細かい線や模様を表示

したがって、Guided Grad-CAMの主な利点は**高解像度化**であり、「Grad-CAMより因果的に正確」という意味ではない。

### 3.3 AlexNetでは何が起きそうか

現行AlexNetの最後のConv2D出力は、入力224×224より大幅に小さい約12×12の特徴マップである。現行Grad-CAMはそれを224×224へbilinear補間するため、広く滑らかな塊になる。

Guided Grad-CAMを加えると、次のような図になる可能性が高い。

- 2–5 kHz帯の広いGrad-CAM領域内で、細い時間周波数ridgeが強調される
- 気泡音に対応しそうな短時間の立ち上がりや境界が見えやすくなる
- 一方で、単にスペクトログラムの強い輪郭・縞・補間境界を描く可能性がある

発表用の「モデルがどの細かい模様を拾った可能性があるか」という補助図には有用である。

### 3.4 CNN+Transformerでは何を説明するか

現行CNN+Transformerは、AlexNet型CNNで空間特徴を抽出し、reshape後にTransformer encoderへ渡す。最後のConv2D出力は約14×14である。

その層へGrad-CAMを適用すれば、最終熱流束出力からTransformerを通ってCNN特徴へ戻る勾配を使うため、「最終予測に関係したCNN空間領域」の可視化はできる。

ただしGuided Grad-CAMでも、次は直接説明できない。

- Transformerが時間token間をどう結合したか
- attention headごとの役割
- どの長距離時間依存が予測を変えたか
- 正・負の証拠が最終値へどの程度寄与したか

したがって、CNN+TransformerにGuided Grad-CAMを使っても、「Transformerまで完全に説明した」とは書かない。

### 3.5 今回特に注意すべき限界

#### 限界A: 回帰出力の負方向寄与を失いやすい

現行Grad-CAMは最後にReLUを適用し、熱流束予測を上げる方向の局在を主に表示する。Guided BackpropagationもReLUの逆伝播を変更して負勾配を抑える。

今回知りたいのは、どの領域が熱流束を上げたかだけでなく、低熱流束やONB近傍で何が予測を下げたかでもある。符号付きIntegrated Gradientsの方が、この問いには適している。

#### 限界B: 見た目の細かさとfaithfulnessは別である

[Adebayo et al.のsanity checks](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)では、Guided BackpropagationとGuided Grad-CAMが、ネットワーク重みを段階的にランダム化してもほとんど変わらない例が示されている。入力のedge detectorに近い図が残るためである。

スペクトログラムには時間周波数の線・境界が多いため、この問題は自然画像以上に注意が必要である。

#### 限界C: 現在のsanity checkでは足りない

現行コードのsanity checkは最終trainable layerだけをランダム化する軽量版で、設定上はIntegrated Gradientsのみが対象である。Guided Grad-CAMを追加するなら、少なくとも次を行う必要がある。

- 出力層だけでなく上層からConv層までのcascading randomization
- 学習済みモデルとrandom-labelモデルの比較
- group maskまたはpatch occlusionとの順位相関
- deletion/insertion曲線
- 入力の単純edge mapとの類似度比較

### 3.6 推奨する説明手法の役割分担

| 問い | 主方法 | 補助方法 |
|---|---|---|
| どの物理帯域が性能に必要か | group mask/occlusion | 固定grid band ablation |
| 1サンプルで正負どちらへ寄与したか | signed Integrated Gradients | baseline感度解析 |
| 最終CNN層が粗くどこを局在したか | Grad-CAM | HiResCAM等との比較 |
| 局在領域内の細い模様を見たい | Guided Grad-CAM | input edge mapとの比較 |
| Transformerの時間依存を知りたい | token/attentionを対象にした別手法 | CNN側Grad-CAM |

### 3.7 最終判断

Guided Grad-CAMは**追加する価値はあるが、置き換えない**。

優先順位は次のとおりである。

1. group maskを物理主張の中心にする
2. Integrated Gradientsを符号付き局所寄与の中心にする
3. Grad-CAMを粗いCNN局在として残す
4. Guided Grad-CAMを高解像度の補助図として追加する
5. Guided Grad-CAMがsanity checkを通らなければ論文主張には使わない

なお、Grad-CAMの勾配平均により未使用位置まで広く見せる場合があるという問題に対し、[HiResCAM](https://arxiv.org/abs/2011.08891)も比較候補になる。ただし、その理論保証が現行AlexNetの複数Dense層やCNN+Transformer全体へそのまま当てはまるとは限らないため、これも実測faithfulnessで選ぶ必要がある。

---

## 4. なぜ現在の異種アンサンブルはRF単体を超えないのか

### 4.1 学部と現在の決定的な差

| 観点 | 学部研究 | 現在 |
|---|---|---|
| モデル | AlexNet、ResNet50、VGG16 | XGBRF、CNN+Transformer、AlexNet |
| 個別性能 | 3 CNNがほぼ同格 | RFが明確に優位 |
| 予測校正 | 3 CNNで比較的近い | 深層モデルにレンジ圧縮・bias |
| 分割 | chunk単位KFold | source wav単位GroupKFold |
| 重み | outer validationラベル使用 | fixed、inner holdout等を比較 |
| 誤差多様性 | 保存結果だけでは未測定 | 残差相関を測定済み |
| 主帯域 | 未検証 | 3モデルとも主に2–5 kHz |

学部の発想「異なるモデルを組み合わせれば改善する」は方向として正しい。ただし、**アーキテクチャが違うことと、誤りが相補的であることは同じではない。**

### 4.2 アンサンブル改善の数学的条件

RFを基準モデル、別モデルを補助モデルとし、補助モデルを割合 (a) だけ混ぜる。

\[
\hat y_{ens}=(1-a)\hat y_{RF}+a\hat y_D
\]

RF残差を (e_R=\hat y_{RF}-y)、モデル差を (d=\hat y_D-\hat y_{RF}) とすると、

\[
MSE_{ens}-MSE_{RF}
=2aE[e_Rd]+a^2E[d^2]
\]

である。

小さな正の重み (a) がRFを改善するには、補助モデルの修正方向 (d) がRF残差 (e_R) と十分に逆向きでなければならない。単に予測が違うだけでは不十分である。

現在の `1/(1-R²)` 重みは各モデル単体の誤差量しか見ず、重要な (E[e_Rd])、すなわちモデル間残差共分散を見ていない。これが方式上の主要な弱点である。

[回帰ensembleの研究](https://www.jmlr.org/papers/v6/brown05a.html)でも、個別精度だけでなくbias・variance・covarianceの釣り合いが重要であることが示されている。

### 4.3 現在までの実測結果

8/25の14条件におけるRF単体との差は、概ね次の傾向だった。

| 方式 | 平均ΔR² | R²改善条件 | 平均ΔRecall | 解釈 |
|---|---:|---:|---:|---|
| RF 98% / CTF 1% / Alex 1% | +0.00044 | 10/14 | -0.00675 | ごく小さい平滑化、見逃し悪化 |
| Equal mean | -0.9217 | 0/14 | -0.165 | 弱い深層runの影響が大きい |
| Prediction max | -0.0704 | 0/14 | +0.00992 | recall寄りだが回帰悪化 |
| Inner holdout | -0.130 | 0/14相当 | -0.0937 | 重み推定が不安定 |
| Legacy validation weight | -0.009 | 安定せず | ― | outerラベル使用でも救えない |

残差相関の平均は、RF–AlexNet約0.620、RF–CNN+Transformer約0.656、深層2モデル間約0.867だった。予測そのものの相関は約0.95–0.98である。

また、8/28の同一周波数OOF予測で楽観的に非負重みを求めてもRFへ80–96%が割り当てられ、別foldへ重みを移すと全周波数でRF単体を下回った。複数maxfreqのRF同士も残差相関0.922–0.978であり、22 kHz RFだけが選ばれた。

この結果から、次は優先度が低い。

- 98/1/1を97/2/1などへ細かく変える探索
- equal meanの再実行
- maxfreqモデルを単純平均
- `1/(1-R²)` の式だけを少し変える

### 4.4 現在のデータにアンサンブルが合わないのか

正確には、次のように分けるべきである。

- **固定の全体重み**: 現在の候補モデルには合っていない可能性が高い
- **同一情報を使う異種モデル平均**: 相補性が弱く、優先度が低い
- **条件付き・残差型ensemble**: まだ十分に試していない
- **same-architecture multi-seed ensemble**: 深層モデルの分散低減に合う可能性が高い
- **物理的に入力役割を分けたensemble**: 研究上最も未検証で、有望

保存済み予測をサンプルごとにoracle選択すると、非RFモデルが最良になるサンプルが約51–58%存在した。これは固定重みでは拾えない条件付き相補性があり得ることを示す。ただし、未知サンプルで「どのモデルが勝つか」を正解値なしで予測できるかは未証明である。

---

## 5. アンサンブルを勝たせるための優先施策

ここでいう「勝たせる」は、評価データを見て都合のよい方式を選ぶことではなく、**アンサンブルが改善しやすい学習条件を設計し、最後に未使用データで一度だけ判定すること**を意味する。

### 5.1 最優先: 深層モデルをensemble memberとして使える強さへ戻す

現在はRFと深層モデルの性能差が大きすぎる。アンサンブル方式を変える前に、次を行う。

1. 入力正規化方法をtrain fold内で固定し、validationへ同じ変換を適用
2. inner GroupKFoldでearly stoppingとepoch選択
3. 学習率、batch size、weight decay、dropoutをgroup-awareに調整
4. 各architectureを3–5 seedで学習し、平均と分散を保存
5. 予測値の傾き・切片をOOFで確認し、レンジ圧縮をaffine calibration
6. 学部で有効だったResNet50とVGG16を候補libraryへ戻す

重要なのは、RF、AlexNet、CNN+Transformerの3つを必ず同じ重み付き平均へ入れることではない。候補を多く作り、validationで役立つものだけを選ぶ。

[Deep Ensembles](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)は、異なる初期値から学習した複数ネットワークの平均が、予測と不確実性評価の安定化に有効であることを示している。まず各深層architecture内のseed ensembleを一つの安定した候補モデルとしてから、異種統合へ進む方がよい。

### 5.2 必須: 全重み・校正をgroup-aware cross-fittingで学習する

outer validationの正解を使わず、次の二重構造にする。

1. Outer GroupKFold: 最終性能評価用
2. Outer train内のInner GroupKFold: base modelのOOF予測作成用
3. Inner OOF予測だけで、校正・モデル選択・ensemble重みを学習
4. Outer train全体でbase modelを再学習
5. Outer validationを一度だけ予測

[Breimanのstacked regression](https://doi.org/10.1023/A:1018046112532)は、cross-validation予測を用い、非負制約下で結合係数を学ぶ方式を示している。[Caruanaらのensemble selection](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf)も、学習に使っていない予測でモデルを選択し、悪いモデルを強制的に混ぜないことを重視している。

現行 `inner_holdout` はouter label leakageを避けている点では改善だが、`train_test_split` がchunk単位であり、inner側ではsource groupを保っていない。ここもInner GroupKFoldへ直す必要がある。

### 5.3 第一候補: 非負・縮小付きstacking

校正後の各モデル予測を (p_m) とし、次を学習する。

\[
\hat y=\beta_0+\sum_m w_mp_m,
\quad w_m\ge 0
\]

さらに、現在のようにRFが強い場合は、重みをRF-onlyへ縮小するpenaltyを入れる。

\[
\min_{\beta_0,w}
\sum_g \frac{1}{n_g}\sum_{i\in g}(y_i-\hat y_i)^2
+\lambda\|w-w_{RF-only}\|_2^2
\]

特徴:

- interceptで全体biasを補正できる
- affine calibrationで深層予測の圧縮を先に直せる
- 非負制約で極端な相殺を防ぐ
- shrinkageで小標本の重み過学習を防ぐ
- RF-onlyを必ず候補に含める
- chunk数ではなくsource recordingを等重みで評価する

ただし、保存済みOOF診断では単純な線形stackingは既に弱かった。したがって、**同じ3候補のままstacking solverだけ変えることは本命ではない。** 深層モデルの改善と候補多様化が先である。

### 5.4 本命: 物理帯域分担型・RFアンカー残差アンサンブル

本研究に合わせた新方式候補を、次のように定義する。

仮称:

> Physics-guided RF-anchored residual ensemble  
> 物理帯域分担型・RFアンカー残差アンサンブル

#### 基本式

\[
\hat y(x)=\hat y_{RF}(x)+\gamma\,g(q(x))\,h(v(x))
\]

- \(\hat y_{RF}\): 現在の最強モデルを基準予測にする
- \(h\): RFのOOF残差 (y-\hat y_{RF}^{OOF}) を予測する補正器
- \(v(x)\): RFと重複しすぎない補助表現
- \(g(q)\in[0,1]\): 補正を使うべきサンプルだけ選ぶgate
- \(\gamma\): 補正量を0方向へ縮小する係数

#### 補正器へ入れる候補情報

- AlexNet、ResNet50、VGG16、CNN+Transformerのcalibrated OOF予測
- 各モデルとRFの予測差
- 各深層モデルのseed間標準偏差
- 0.5–2、5–10、10–22 kHzの帯域エネルギー
- 2–5 kHz内の時間間欠性・peak数・変動量
- 推定SNR、水流音比、低周波/高周波エネルギー比
- モデル間disagreement

#### 物理的な役割分担

- RF anchor: 既に有効性が高い2–5 kHzを中心に主熱流束を推定
- residual expert A: 低周波の流動・背景状態を利用
- residual expert B: 5 kHz以上の気泡・ノイズ補助情報を利用
- residual expert C: CNN/Transformerで時間的間欠性を利用
- gate: どの補助情報が信頼できる条件かを判断

これにより、全モデルが同じ2–5 kHzから同じ熱流束を再推定する状態を避け、「主予測」と「補正情報」に役割を分けられる。

#### safety mechanism

- inner validationで改善しなければ γ=0、すなわちRF単体へ戻す
- 補正値に上限を設け、深層モデルの破綻runで大きく動かさない
- gate入力に真の熱流束やouter validation誤差を使わない
- outer foldごとに方式を再学習する
- 最終独立testまで方式・閾値を固定する

この方法でも未知データでRF以上を保証はできない。しかし、equal meanと違って「補助モデルを必ず混ぜる」という構造上の不利がなく、現在見えている条件付き相補性を最も直接的に検証できる。

### 5.5 次点: Caruana型ensemble selection with replacement

候補libraryを次のように広げる。

- 5 architectures × 3–5 seeds
- calibrated/un-calibratedの両予測
- 複数STFT窓長
- 物理帯域別入力
- PCA-RFと物理特徴RF

Inner OOF上で、RF-onlyから開始し、追加したときにgroup-balanced RMSEが改善する候補だけを一つずつ加える。同じモデルを複数回選べるようにすれば、自然に離散的な重みになる。

利点:

- 悪いモデルを全部混ぜない
- RFを複数回選択し、大きい重みを与えられる
- 目的指標をRMSE、worst-group RMSEなどへ合わせられる

リスク:

- 候補が多すぎるとinner OOFへ過学習する
- 独立source数が18程度では選択が不安定

対策は、上位候補だけを残す、bagged selectionを使う、独立録音を増やすことである。

### 5.6 高リスク探索: negative-correlation residual training

深層モデルを単に真値へfitさせるのではなく、RFと同じ誤りを避けるpenaltyを与える方法も考えられる。

\[
L_D=MSE(y,\hat y_D)+\lambda\,Corr(e_D,e_{RF})
\]

または、最初からRFのcross-fitted residualを学習targetにする。

これはモデルの多様性を訓練段階で作る方式で、単にarchitectureを変えるより目的に直接的である。ただし、個別性能を下げすぎたり、18 source程度で見せかけの逆相関を学んだりする危険がある。最初の本命にはせず、データ追加後の探索候補とする。

### 5.7 mixture of expertsを使うなら条件を限定する

[Adaptive Mixtures of Local Experts](https://doi.org/10.1162/neco.1991.3.1.79)の考え方では、入力ごとにgateが適切な専門家を選ぶ。

今回なら、次の専門化は物理的に説明しやすい。

- no-noise / waterflow noise
- 推定SNR帯
- RF予測が低・中・高熱流束の領域
- 周波数エネルギー比が通常・異常の領域

ただし、真の熱流束でgateを切り替えることは推論時にできない。RF予測や観測可能な音響統計だけを使う。また、gateは固定重みより自由度が高いため、複数実験日が揃うまでは過学習しやすい。

---

## 6. 「アンサンブルが良い」という結論へ科学的に到達する実験計画

### Phase 1: 評価単位を固定する

1. Primary metricを全熱流束RMSEまたはR²に固定する
2. ONB指標はsecondaryとする
3. source wavを分離するGroupKFoldを維持する
4. 可能なら実験日を完全holdoutする
5. 各録音sourceを等重みで評価する
6. 最終比較はbest single model対ensembleのpaired比較にする

成功条件の例:

- ensemble RMSEがbest singleより低い
- 改善方向が大半のouter fold・実験日で一致する
- source-level bootstrap 95% CIが0をまたがない
- 工学的に意味のある最小改善幅を事前に決める。暫定的にはRMSE 1–2%低下を候補とする

fold間標準誤差だけでは不十分である。独立group数が少ないとcross-validation差の不確実性は大きいことが知られている。[Varoquaux, 2018](https://arxiv.org/abs/1706.07581)

### Phase 2: 強く安定したcandidate libraryを作る

1. PCA-RF
2. 物理特徴RF
3. AlexNet 3–5 seeds
4. ResNet50 3–5 seeds
5. VGG16 3–5 seeds
6. CNN+Transformer 3–5 seeds
7. 固定周波数grid上の帯域別入力モデル
8. 必要なら複数STFT窓長モデル

各モデルについて次を保存する。

- group OOF prediction
- RMSE、R²、MAE
- calibration slope/intercept
- sourceごとの残差
- 残差相関・差分方向
- seed variance

### Phase 3: 安価な方式から順に比較する

比較順:

1. best single
2. architecture内seed mean
3. calibrated nonnegative ridge stacking
4. Caruana型ensemble selection
5. RF-anchor residual stacking
6. physics-guided gated residual ensemble
7. negative-correlation training

同じouter folds、同じcandidate predictionsで比較し、方式ごとに都合のよいrunを選ばない。

### Phase 4: 最終独立testを一度だけ開ける

開発用データで次を固定する。

- 入力前処理
- candidate models
- seed数
- calibration
- ensemble式
- gate特徴
- hyperparameters
- primary metric
- 成功判定

固定後、新しい実験日または未使用repeatを一度だけ評価する。

ここでensembleがbest singleを上回れば、修士論文で強く主張できる。

> 異種モデルを単純平均するだけでは改善しなかったが、録音元を分離したcross-fitting、予測校正、物理帯域の役割分担、RFアンカー残差補正により、最良単体モデルを上回った。

### Phase 5: 上回らなかった場合の着地点も事前に持つ

結果を見て結論を変えるのではなく、次の分岐を事前に用意する。

#### 結論A: 全体精度でensembleが勝つ

希望している主結論。平均RMSEと独立実験日でRFを上回る。

#### 結論B: 平均は同等だが頑健性で勝つ

平均RMSE差は小さいが、worst noise、worst source、実験日間分散を改善する。これも実用上のensemble価値である。

#### 結論C: deep seed ensembleだけが単体deepを改善する

異種ensembleはRFを超えないが、深層モデルの不確実性・安定化にはensembleが有効だったと結論する。

#### 結論D: RF単体が最終的に最良

異種architectureであっても、同じ主要帯域を利用し、残差補正が未知録音へ一般化しないためensembleが成立しなかった、と説明する。

結論Dでも、学部研究との対比、厳密評価、XAI、残差共分散まで示せれば修士研究として十分な内容になる。

---

## 7. 論文として推奨する中心仮説

「アンサンブルは必ず良い」を仮説にすると反証不能になる。次の形にすると、希望する方向を保ちながら科学的に検証できる。

### Main hypothesis

> 録音元を分離した熱流束回帰において、個別モデルの予測校正と誤差共分散を考慮し、主要周波数帯と補助情報の役割を分離した残差型ensembleは、最良単体モデルより高い精度または頑健性を示す。

### Sub-hypotheses

1. 単純なarchitecture多様性だけでは、利用帯域と残差が相関するため改善しない。
2. 同一architectureのmulti-seed平均は深層モデルの分散を低減する。
3. 深層モデルのcalibration後はRF残差との補完性が増える。
4. 2–5 kHz主予測と、それ以外の帯域・時間変動を用いる残差補正は、full-bandモデル同士の平均より相補的である。
5. gateはmodel disagreement、seed variance、音響SNRから補正の信頼性を推定できる。

---

## 8. 推奨する最終論文ストーリー

### 8.1 導入

- 学部研究で3 CNN ensembleがノイズ環境下の精度を改善した
- 修士研究ではRFとTransformerを加え、異なる帰納バイアスによるさらなる改善を狙った
- しかし、異なるarchitectureを集めるだけで改善するとは限らない
- 予測根拠、残差共分散、物理帯域の観点からensemble成立条件を明らかにする

### 8.2 方法

- source-grouped validation
- XGBRF、CNN、CNN+Transformer
- TreeSHAP、Integrated Gradients、Grad-CAM、group mask
- prediction calibration、residual correlation
- RF-anchor residual ensemble

### 8.3 結果

- RFが単体で最も強い
- 3モデルは2–5 kHzを共通して重視
- 単純平均は相関した誤りと深層モデルの未校正により失敗
- 物理帯域分担・残差補正により、条件付き相補性を活用できたかを提示

### 8.4 結論候補

提案方式が勝った場合:

> 異種モデルの単純統合では最良単体を上回らなかった。一方、録音元を分離したcross-fittingで予測を校正し、2–5 kHzを中心とするRF予測を基準に、補助帯域と時間変動から残差を条件付き補正することで、独立録音に対する熱流束推定精度を改善した。したがって、本課題におけるensembleの価値はモデル数そのものではなく、物理的役割分担、誤差共分散、および安全な残差補正にある。

この結論は、「アンサンブルした方がよい」という希望を満たしつつ、学部研究より一段深い主張になる。

---

## 9. 今後の優先順位

### 最優先

1. 新しい独立録音・実験日を確保する
2. 深層モデルの正規化、early stopping、multi-seedを整える
3. ResNet50とVGG16をcandidate libraryへ戻す
4. inner `train_test_split` をgroup-aware cross-fittingへ置き換える設計を固める
5. OOF prediction calibrationとsource-level残差表を作る

### 次に行う

6. seed ensembleを評価する
7. nonnegative shrinkage stackingを評価する
8. RF-anchor residual stackingを評価する
9. 帯域分担型residual expertを評価する
10. 十分なgroup数が得られた後にgateを導入する

### XAI

11. TreeSHAPはPCA監査として維持する
12. 物理特徴RFを小規模に比較する
13. Guided Grad-CAMを補助図として追加し、full sanity checkを行う
14. Guided Grad-CAMがedge mapとほぼ同じなら主張から外す

### 優先度を下げる

- 固定重みの小数点探索
- 同じmaxfreqモデルの単純平均
- outer validationラベルを使うlegacy weighting
- 元50,176画素へそのままRFを学習し、説明しやすさだけを狙うこと
- Guided Grad-CAMの見た目だけで物理現象を断定すること

---

## 10. 最終提言

現時点で最も重要な示唆は、次の一文である。

> 学部研究で有効だった「同程度に強いCNNを平均して分散を下げるensemble」と、現在必要な「強いRFを基準に、独立した補助情報が説明できる残差だけを安全に加えるensemble」は、同じアンサンブルでも別の問題である。

したがって、今後は「RF、AlexNet、CNN+Transformerをどういう比率で混ぜるか」から離れ、次の順番で進める。

1. 深層候補を強く・安定・校正済みにする
2. source-grouped OOFでRF残差を分析する
3. 残差を説明できる物理帯域・時間特徴を探す
4. RF-onlyへ戻れる残差ensembleを学習する
5. 未使用実験日で一度だけ最終判定する

これが、希望している「アンサンブルの方が精度が高い」という結論へ到達する可能性を最大化しながら、論文としての信頼性も守る最短経路である。

