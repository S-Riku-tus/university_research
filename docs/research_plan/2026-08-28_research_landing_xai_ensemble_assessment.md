　# 研究の着地点・説明性・アンサンブル総合診断

作成日: 2026-08-28  
対象: `20260828_ensemble_strategy_comparison` を中心とする最新結果、`20260825_ensemble_strategy_comparison` の実アンサンブル結果、既存のXAI出力・研究計画・関連一次論文

## 0. 先に結論

現状の研究は行き詰まっているのではなく、精度を出す段階から「何を最終主張にするかを絞る段階」に入っている。

最も自然な着地点は、次の3点である。

1. 現在の装置・前処理・データ範囲では、音響スペクトログラムから熱流束を高精度に推定でき、RandomForest系モデルが最も安定している。
2. モデル固有の可視化だけでなく、全検証データに対する共通の周波数帯マスクでも、主な予測根拠が一貫して2–5 kHzへ集中する。ただし、これは現時点では「本装置・500 Hzハイパス・現行STFT処理における有効帯域」であり、沸騰音一般に普遍的な帯域とは断定しない。
3. 異種モデルの単純な予測平均は、強いRFの予測を弱い・未校正な深層モデルで薄めるため、現構成では単体RFを安定して上回らない。この負の結果は失敗ではなく、「アンサンブルには個別精度だけでなく、校正された予測と低い誤差相関が必要である」という成立条件を実データで示した結果として扱える。

したがって、論文の主語を「新しいアンサンブルを作った」ではなく、次のように置くのがよい。

> プール沸騰音響スペクトログラムによる熱流束推定について、モデル横断で再現する有効周波数帯と、その物理的解釈の限界を明らかにした。さらに、異種モデル統合を精度・誤差相関・校正・説明根拠の観点から検証し、単純なアンサンブルが有効となる条件と、有効でない条件を示した。

ONBの±10%帯は現時点の仮置きなので、主結論の中心からいったん外してよい。まず熱流束回帰、周波数根拠、条件間一般化を固め、その後にONB定義を確定して二次評価として接続する方が論理的である。

---

## 1. 今回の診断範囲

### 1.1 最新8/28結果

`20260828_ensemble_strategy_comparison` は、保存先名に `ensemble` が含まれるが、実際には `independent_model_grid` による単体モデル比較である。

- 周波数上限: 3, 5, 10, 15, 22 kHz
- データ: `2025.06.18_0.3_3 / heatflux_no_noise`
- モデル: RF、CNN+Transformer v2 GAP、AlexNet
- 3 GroupKFold
- 15条件、合計45回のモデル学習
- 実アンサンブル予測は含まれない

したがって、この実行から直接分かるのは、各周波数表現における単体モデル性能とXAIであり、アンサンブル性能ではない。

### 1.2 実アンサンブルの診断

アンサンブル方式の評価には、同一fold内で3モデルを学習し、複数の統合方式を同時比較した `20260825_ensemble_strategy_comparison` を用いた。

- 周波数上限: 3, 22 kHz
- ノイズ: no-noise、SNR 0, -4, -8, -12, -16, -20 dB
- 合計14データ条件
- 比較方式: 等重み、RF98/1/1、予測最大値、inner holdout、legacy validation-fold weighting

さらに、8/28の保存済みOOF予測から、同一周波数の異種モデルstackingと、複数周波数RFの統合余地を診断した。この後解析は方式候補の絞り込み用であり、最終性能の証明には使わない。

### 1.3 独立サンプル数の注意

見かけ上は1080チャンクだが、実質的な独立単位は18本の録音元であり、各録音から60チャンクを作っている。現在のGroupKFoldは録音元を分離しているため、通常のランダムチャンク分割より妥当である。一方、統計的な情報量が1080独立サンプルあるわけではない。

構造を持つデータでは依存単位を保ったblock/group cross-validationが必要であるという方法論上の指摘とも整合する。[Roberts et al., 2017](https://doi.org/10.1111/ecog.02881) また、小標本のcross-validationではfold間SEが不確実性を過小評価し得るため、小さな性能差を強く主張すべきではない。[Varoquaux, 2018](https://doi.org/10.1016/j.neuroimage.2017.06.061)

---

## 2. 予測性能から分かること

### 2.1 最新no-noise OOF性能

3 foldの全OOF予測をまとめて再計算した値は次の通りである。

| max frequency | RF R² / RMSE / F1 | CNN+Tf R² / RMSE / F1 | AlexNet R² / RMSE / F1 |
|---|---|---|---|
| 3 kHz | 0.899 / 86,738 / 0.892 | 0.842 / 108,249 / 0.868 | 0.833 / 111,401 / 0.860 |
| 5 kHz | 0.898 / 87,130 / 0.880 | 0.841 / 108,659 / 0.845 | 0.855 / 103,587 / 0.812 |
| 10 kHz | 0.896 / 88,010 / 0.856 | 0.604 / 171,305 / 0.703 | 0.803 / 120,974 / 0.857 |
| 15 kHz | 0.906 / 83,640 / 0.889 | 0.853 / 104,475 / 0.852 | 0.771 / 130,297 / 0.843 |
| 22 kHz | 0.916 / 78,958 / 0.893 | 0.776 / 128,798 / 0.793 | 0.746 / 137,348 / 0.759 |

主な読み方は次の通りである。

- RFは全周波数で最良か最良に近く、周波数変更に対しても変化が小さい。
- 22 kHz RFが今回の単一条件では最高だが、3–15 kHzとの差は大きくない。
- CNN+Transformerの10 kHzはfold 2で大きく崩れており、物理的な「10 kHzだけ悪い現象」より学習不安定性を疑うべきである。
- AlexNet 5 kHzは連続ROC-AUCが非常に高い一方でF1が低く、順位づけはできても閾値に対する予測値の校正が悪い。
- 両深層モデルは低熱流束を過大、高熱流束を過小評価する圧縮傾向があり、RFへの単純混合で閾値側の予測を下げやすい。

### 2.2 周波数上限比較が意味する範囲

現行前処理では、周波数上限で切った後に毎回224×224へresizeしている。

| max frequency | 元の周波数bin概数 | 224幅への処理 |
|---|---:|---|
| 3 kHz | 約92 | 拡大 |
| 5 kHz | 約153 | 拡大 |
| 10 kHz | 約305 | 縮小 |
| 15 kHz | 約458 | 縮小 |
| 22 kHz | 約673 | 強い縮小 |

このため、変わるのは最大周波数だけではない。

- 1 pixelが表すHz幅
- CNN kernelが覆う物理帯域
- 補間・平滑化の程度
- 画像内で2–5 kHzが占める幅
- 入力値の平均・分散
- PCAで得られる基底

したがって、現在の結果は「現行パイプラインでどの表現が機能したか」を答えるが、「5 kHz以上の物理信号が不要である」と因果的には証明しない。純粋な帯域比較には、共通22 kHz gridを維持したまま帯域だけを残す／消す再学習が必要である。

---

## 3. 説明性出力の総合評価

### 3.1 評価した出力数

最新結果には、各モデル・周波数・foldから最大5例ずつ、合計225例の説明対象がある。

- IG画像: 150例（AlexNet 75、CNN+Transformer 75）
- Grad-CAM画像: 75例（AlexNetのみ）
- 個別group occlusion画像: 225例
- 全検証集合へのgroup mask performance: 45 fold-model-frequency出力
- RF TreeSHAP: 15 fold-frequency出力
- input stability: 深層モデル30出力
- top-layer randomization sanity: 深層モデル30出力

個別画像は、high/low heat flux、ONB上下、false negative、worst predictionなどの意図的に選んだ例であり、無作為標本ではない。したがって、物理帯域の主結論には、画像平均より全検証集合のgroup maskを優先する。

説明指標の最終的な役割分担は次の通りである。

| output | AlexNet | CNN+Transformer | RF | 最終的な使い方 |
|---|---|---|---|---|
| IG magnitude/signed | あり | あり | なし | 高熱流束例の局所寄与。magnitudeは重要度、signedは予測を上げる／下げる方向 |
| Grad-CAM | あり | なし | なし | 最終畳み込み層の粗い局在化を示す補助図 |
| TreeSHAP | なし | なし | PCA空間であり | RF内部の寄与とadditivity確認。物理周波数へ直接対応させない |
| individual group occlusion | あり | あり | あり | 個別例で帯域・時間区間を消したときの予測変化 |
| validation-set group mask | あり | あり | あり | 全モデル共通の主な物理比較。最重要 |
| deletion/insertion | IG/Grad-CAM | IG | なし | local mapの順位が予測変化へ結び付くかを補助的に確認 |
| input stability | IG | IG | なし | 微小入力摂動への再現性。正しさそのものではない |
| top-layer sanity | IG | IG | なし | model dependenceの部分確認。cascading randomizationは未実施 |

### 3.2 全モデルで最も強い結果: 2–5 kHz帯域マスク

全モデル、全周波数、全foldで、最大のR²低下を生じた周波数グループは2–5 kHzであった。3 kHz入力だけは利用可能範囲に合わせて2–3 kHzである。

代表的なfold平均R²低下は次の通りである。

| model | 3 kHzの2–3 kHz mask | 5 kHzの2–5 kHz mask | 10 kHz | 15 kHz | 22 kHz |
|---|---:|---:|---:|---:|---:|
| RF | 0.351 | 0.357 | 0.304 | 0.329 | 0.343 |
| CNN+Tf | 2.109 | 0.505 | 0.244 | 0.393 | 0.514 |
| AlexNet | 1.464 | 0.661 | 0.370 | 0.590 | 0.614 |

一方、22 kHz RFでは5–10 kHzのmaskによるR²低下は約0.001で、10 kHz以上はほぼ0であった。深層モデルでも主効果は一貫して2–5 kHzである。

この結果は、3種類のモデルが異なる帯域を使って相補的になっているという仮説より、同じ主要帯域を異なる方法で読んでいるという仮説を支持する。これは後述する高い誤差相関と、アンサンブル効果の小ささにも整合する。

ただし、group幅は均一ではなく、2–5 kHzは3 kHz幅を持つ。またゼロmaskは学習分布外入力を作る可能性がある。そのため、次の最終確認は「2–5 kHzのみ」「fullから2–5 kHz除去」の再学習で行うべきである。

### 3.3 AlexNet: IGは強いが、Grad-CAMは補助図に限定する

高熱流束例に限定すると、AlexNetのIG magnitudeの2–5 kHz占有率は周波数上限ごとに約90–94%であった。入力スペクトログラムに見える約2–3 kHzの離散的な強い成分へ、時刻をまたいで局所的に寄与が集中している。

Grad-CAMも高熱流束では2–5 kHzを主に含むが、占有率は約38–70%で、周波数上限が高いほど5 kHz以上へ広くにじむ。低熱流束やONB近傍例では、上端周波数側へ広く分布する例が多い。

faithfulnessの方向をそろえたdeletion/insertion診断では、IGの中央値がおおむね0.43–0.52、0.47–0.48で正方向だったのに対し、Grad-CAMの中央値は多くの周波数で0付近または負だった。このため、Grad-CAMは「最後の畳み込み層がどこを粗く局在化したか」を示す補助図としては使えるが、物理帯域の主要根拠にはしない。

Grad-CAM自体も、原論文では最終畳み込み層から得るcoarse localization mapとして提案されており、入力pixelごとの符号付き寄与量ではない。[Selvaraju et al., 2017](https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)

### 3.4 CNN+Transformer: 高熱流束のIGは2–5 kHzだが、「時間関係を使った」とはまだ言えない

高熱流束例のIG magnitudeでは、2–5 kHzの占有率が約83–95%であった。AlexNetとほぼ同じ離散成分が強調されている。

しかし、全検証集合の時間4分割maskによるR²低下は、多くの条件で0前後から0.036程度であり、特定の時刻区間だけが決定的という結果ではない。現画像も、重要点が1秒の中に散在している。

したがって、現在言えるのは「CNN+Transformerも2–5 kHzに現れる局所イベントまたは帯域エネルギーを利用している」であり、「Transformerが長時間依存を捉えたため優れている」とは言えない。実際、予測精度もRFを上回っていない。

### 3.5 RF: 最も信頼できる主モデルだが、TreeSHAPはPCA空間の内部説明

RFのTreeSHAP additivity errorは、熱流束単位で中央値約0.12–0.19、最大でも約1.76であり、数値的な再構成は非常に良好である。

ただし、SHAPが説明しているのはPCA成分であり、その成分を単一の物理周波数・時間領域として解釈できない。TreeSHAPはモデル内部の整合性確認、物理帯域は共通group mask、という役割分担が必要である。TreeSHAPは木モデルに対する局所的整合性・一貫性を持つ寄与法として提案されているが、入力特徴自体の物理意味を自動で復元するものではない。[Lundberg et al., 2018](https://arxiv.org/abs/1802.03888)

RFの時間maskは、4番目の時間区間が最大となる条件が多いものの、R²低下は約0.043–0.054程度で、周波数2–5 kHz maskの約0.30–0.36よりかなり小さい。よって「最後の0.25秒が本質」というより、時間全体に存在する特徴のうち終端側がやや強い可能性、という程度に留める。

### 3.6 IGの数値整合性とbaseline問題

IGは入力とbaselineの差に対する寄与を定義する方法であり、SensitivityとImplementation Invarianceを満たすよう設計されている。[Sundararajan et al., 2017](https://proceedings.mlr.press/v70/sundararajan17a.html)

現在はゼロスペクトログラムをbaselineとしている。高熱流束例では `prediction - baseline prediction` が大きく、IGの2–5 kHz集中には解釈価値がある。一方、低熱流束・ONB近傍・worst predictionの多くでは、ゼロ入力に対するモデル出力がすでに15–30万程度あり、入力との差が数十程度しかない。

全選択例で計算したcompleteness relative errorの中央値は、AlexNetで約0.082–0.582、CNN+Transformerで約0.214–1.008とばらついた。ただし、これは分母の `prediction - baseline prediction` がほぼ0の例で大きくなる。高熱流束例だけでは、AlexNetの周波数別中央値は約2.1–6.5%、CNN+Transformerは約3.0–22.2%であり、AlexNetは比較的良好、CNN+Transformerは一部条件で積分近似またはモデル非線形性の影響が残る。

この場合、非常に小さいattribution mapを画像ごとに最大値1へ正規化すると、数値的に小さな模様まで強く見える。実際、低熱流束・ONB近傍では `|prediction-baseline| / |prediction|` の中央値がほぼ0であった。

したがって、現状の安全な読み方は次の通りである。

- 高熱流束例のIG: 有効な局所根拠として扱える。
- 低熱流束・ONB近傍のIG: ゼロbaselineに対する寄与が小さく、正規化画像だけでは強く解釈しない。
- 全体の物理結論: group maskを優先する。

baselineが「特徴の欠如」を何として表すかでpath attributionが変わることは既報でも指摘されている。[Sturmfels et al., 2020](https://distill.pub/2020/attribution-baselines/) 今後は、学習データから選ぶ複数baseline、低熱流束の代表スペクトログラム、blurred/mean baselineなどへの感度確認が必要である。

### 3.7 stabilityとsanity check

入力へサンプル標準偏差の1%ノイズを加えたIGの安定性は非常に高い。

- Pearson: ほぼ0.9999–1.0000
- relative L1: おおむね1–2%、最大約6%

これは「微小摂動で画像が急変しない」ことを示すが、「画像が正しい」ことまでは示さない。

最終層だけをrandomizeしたsanity checkでは、元mapとのPearsonが次の程度残った。

- AlexNet IG: 約0.69–0.875
- CNN+Transformer IG: 約0.903–0.953

一方でrelative L1はAlexNetで約0.93–0.97、CNN+Transformerで約0.32–0.75変化した。つまり寄与強度は大きく変わるが、空間パターンの一部は残る。特にCNN+Transformerでは、入力構造・勾配構造を強く反映したtemplateである可能性を無視できない。

saliency mapは見た目だけではモデル依存性を誤認し得るため、parameter/data randomizationが必要だという指摘と一致する。[Adebayo et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) 現在は最終層のみなので、論文用には層を上から順にrandomizeするcascading testを少数条件で追加する方がよい。特徴除去による予測性能変化を重視する評価方針は、ROAR型benchmarkの考え方とも整合する。[Hooker et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html)

### 3.8 物理的な意味

本データの高熱流束スペクトログラムには、約2–3 kHz付近の離散的な強い成分が時間全体に現れ、IGとmaskがその成分へ集中している。独立した別研究でも、wire-heaterのpool boilingにおいて約2 kHzのpeakがregime identificationに重要と報告されており、今回の2–5 kHz結果には物理的な先行例がある。[Barathula et al., 2023](https://doi.org/10.1016/j.applthermaleng.2023.120281)

一方、別のhydrophone研究では512 Hz未満が熱流束予測に最重要と報告されている。[Dunlap et al., 2023](https://doi.org/10.1016/j.applthermaleng.2023.120558) 本研究は500 Hzハイパスを適用しているため、512 Hz未満の有用性をそもそも評価できない。

この違いは矛盾というより、sensor、容器共鳴、heater geometry、subcooling、sampling、前処理によって有効帯域が変わる可能性を示す。高周波AE hitと時系列を組み合わせて熱流束を予測する別方式も成立しており、単一の普遍帯域を想定すべきではない。[Dunlap et al., 2025](https://doi.org/10.1016/j.aitf.2025.100002)

よって論文では、次の表現が安全である。

> 本装置・実験条件・500 Hzハイパス・現行STFT表現では、2–5 kHz帯がモデル横断で最も大きな予測寄与を示した。この結果は既報の約2 kHz peakとも整合するが、装置や前処理を越えた普遍的沸騰周波数を意味するものではない。

---

## 4. なぜアンサンブルがRFを上回らないのか

## 4.1 8/25の14条件における方式別結果

最良単体モデルとの差を14条件で集計した。正が改善である。

| strategy | R²差の平均 | R²改善条件 | recall差の平均 | recall改善条件 | F1差の平均 | F1改善条件 |
|---|---:|---:|---:|---:|---:|---:|
| RF98/1/1 | +0.00044 | 10/14 | -0.00675 | 0/14 | +0.00093 | 4/14 |
| equal mean | -0.92170 | 0/14 | -0.16511 | 0/14 | -0.11852 | 0/14 |
| prediction max | -0.07037 | 0/14 | +0.00992 | 5/14 | -0.00177 | 3/14 |
| inner holdout | -0.13037 | 2/14 | -0.09365 | 0/14 | -0.05661 | 1/14 |
| validation-fold legacy | -0.00899 | 3/14 | -0.06359 | 0/14 | -0.03041 | 1/14 |

RF98/1/1は平均R²を約0.0004だけ改善したが、RFのfalse negativeを0件回復し、新たに28件発生させた。これは「回帰のごく小さな平滑化」はあるが、「頑健化またはONB見逃し低減」を達成したとは言いにくい。

prediction maxは135件のRF false negativeを回復し、新規false negativeは0件だった一方、回帰R²を全14条件で悪化させた。これは回帰ensembleではなく、見逃しを減らす代わりに過大予測を許すdecision ruleとして分離して扱うべきである。

### 4.2 誤差の相関と個別モデルの強さ

8/25の全42 fold-conditionで、平均残差相関は次の通りだった。

- RF vs AlexNet: 0.620
- RF vs CNN+Transformer: 0.656
- CNN+Transformer vs AlexNet: 0.867

予測値そのものの相関は約0.95–0.98である。深層2モデルは特に似た誤りをしており、相互補完が弱い。

8/28の最新no-noise予測でも、同一周波数の全OOFを見た後で最適化する楽観的な非負重みは、RFへ80–96%を割り当てた。改善はR²で約0.0002–0.003に留まり、重みを別foldへ移すと全周波数でRF単体を下回った。

重要なのは、深層モデルが約半数のチャンクでRFより小さい絶対誤差を持つ場合でも、負けるチャンクでの損失が大きいため、平均で混ぜると改善しない点である。「勝つ回数」と「二乗誤差を減らす能力」は同じではない。

### 4.3 複数周波数RFにも線形統合余地はほぼない

3, 5, 10, 15, 22 kHz RF間の残差相関は0.922–0.978だった。

- 5周波数等平均: R² 0.9070
- fold外へ移した非負stacking: R² 0.9119
- 22 kHz RF単体: R² 0.9159
- 全OOFを見た後の楽観的最適重み: 22 kHz RFへ100%

つまり、現行maxfreq表現同士は違って見えても、RFの誤り方はほぼ同じである。複数周波数を平均するだけの方式は優先度が低い。

### 4.4 原因は「方式」と「現在のメンバー集合」の両方

#### A. 現在のメンバー集合の問題

1. RFが一貫して最も強く、混ぜる相手が弱い。
2. 3モデルが同じ2–5 kHz帯へ強く依存し、誤りが相関する。
3. 深層モデルに低値過大・高値過小の校正biasがある。
4. CNN+Transformer/AlexNetはseedや実行順で結果が変わり、メンバー自体が安定していない。
5. 入力正規化、validation loss、early stoppingが未整備で、RFとの公平な最終比較にまだ改善余地がある。

#### B. 現在の重み方式の問題

`inner_holdout`は学習チャンクを通常の `train_test_split` で20%へ分けているため、録音元groupを維持していない。同じ録音由来の近似チャンクがinner-fitとinner-validationへ分かれ得る。

また、重みは各モデルの `1 - R²` の逆数だけで決まり、次を扱わない。

- 予測の切片・傾きのずれ
- モデル間の残差共分散
- RF単体を選ぶゼロ補正
- 閾値近傍のfalse negative cost
- 重み推定の不確実性

実際、inner holdoutの平均重みはRF 0.639、CNN+Transformer 0.186、AlexNet 0.175となり、外側評価で圧倒的に強いRFへ十分寄らなかった。

stacked regressionはcross-validation予測から非負制約付きleast squaresで係数を求める方法として提案されている。[Breiman, 1996](https://doi.org/10.1023/A:1018046112532) Super LearnerもV-fold cross-validationで候補learnerの重みを選ぶ。[van der Laan et al., 2007](https://doi.org/10.2202/1544-6115.1309) 現実装の逆R²重みはこれらとは異なり、誤差構造を直接最小化するstackingではない。

ただし、保存OOFへ非負stackingを試した診断でもRFを安定して超えなかったため、「重み方式だけ直せば解決する」とも言えない。現時点では、方式の弱さより、メンバー精度・校正・多様性不足の方が根本原因である。

### 4.5 アンサンブルに価値が全くないわけではない

価値を「平均R²を上げること」に限定しなければ、次の役割は残る。

- 複数seedの深層ensembleによる予測不確実性推定
- モデル不一致が大きいサンプルの警告・棄却
- ノイズ状態が異なる場合の条件別expert selection
- RFの性能を主予測とし、検証済みの場合だけ小さな残差補正を加えるsafe ensemble

独立初期値で学習したdeep ensembleは、予測平均だけでなく不確実性推定に利用できる。[Lakshminarayanan et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) また、入力領域ごとにexpertを割り当てるmixture of expertsは、全サンプルへ同じ固定重みを使う方式とは異なる。[Jacobs et al., 1991](https://doi.org/10.1162/neco.1991.3.1.79)

しかし、現在は実質18録音groupしかなく、自由度の高いgateは過学習しやすい。まず独立録音数とcross-condition検証を増やすことが先である。

---

## 5. 推奨する新しいアンサンブル方式

### 5.1 第一候補: RF-anchor safe residual stacking

新方式を一つに絞るなら、固定平均ではなくRFを基準にした残差補正が最も研究に合う。

概念式は次の通りである。

`prediction = RF + g(x) * correction`

`correction = intercept + a * (calibrated CNN+Tf - RF) + b * (calibrated AlexNet - RF)`

必要な制約は次の通りである。

1. 各深層モデルをgroup-aware OOF予測だけでaffine calibrationする。
2. 補正係数はgroup-aware nested CVで求め、ridgeまたは非負・強いshrinkageを使う。
3. 「補正なし=RF単体」を必ず候補に含める。
4. inner splitも録音group単位にする。
5. 改善がinner groupで一貫しない場合、`g(x)=0`としてRFへ戻す。
6. 最終評価foldは重み・calibration・gateの決定に一切使わない。

この方式の利点は、弱いモデルを常時混ぜず、RFを壊さない方向へ設計できることである。新規性を出すなら、`g(x)`にモデル間disagreementと、物理的な2–5 kHz signal-to-noise指標を使う「physics-aware safe gate」が考えられる。

ただし、これは未検証の研究案である。現データ数では複雑なneural gateを使わず、最初は0/1の単純な事前規則またはregularized logistic/ridge gateに留める。

### 5.2 第二候補: 同一architectureのmulti-seed ensemble

CNN+TransformerとAlexNetは同一条件でもrun間変動がある。異種3モデルを混ぜる前に、各深層モデルを3–5 seedで学習し、平均と標準偏差を出す。

目的はRF超えだけではなく、次である。

- 深層モデルの性能順位がseedで入れ替わるか
- 重要帯域がseedを越えて再現するか
- prediction varianceが大誤差を検知するか

これは現在の「1 seedの深層モデルを固定メンバーとみなす」不安定さを先に解消する。

### 5.3 第三候補: noise-state mixture of experts

実運用でノイズ状態が変わるなら、全状態を同じ重みで平均するより、入力PSDからnoise stateを推定し、condition-specific expertへ振り分ける方が物理的である。

候補のgate特徴:

- 2–5 kHz energy / 10–15 kHz energy
- 学習時noise referenceとのspectral distance
- RFとdeep modelsのprediction disagreement
- seed ensemble variance

ただし、現状は各noise条件内で学習・評価しているため、未知noiseへの一般化は未検証である。train-noise/test-noiseを分離した実験後にのみ進める。

### 5.4 優先しない方式

- 等重み平均: 弱いモデルの大誤差を直接混ぜる。
- さらに細かい固定重み探索: RF98/1/1の改善が小さく、結果を見た後の選択になりやすい。
- 現方式のinverse-R² inner holdout: group非対応かつbias/covariance非対応。
- 複数maxfreq RFの単純平均: 残差相関が高く、楽観的最適化でも22 kHz単体が選ばれた。
- 複雑なneural gating: 独立録音18本では自由度が高すぎる。

---

## 6. 論文として成立させるために本当に不足しているもの

## 6.1 最優先: 録音・実験日を越えた一般化

現在の最大の弱点は、1実験日の18録音で高い性能を出していることである。各録音に熱流束ラベルが一つだけなら、モデルが熱流束に対応する物理特徴だけでなく、録音固有の状態を識別している可能性を完全には除けない。

必要な検証は次のどちらか、可能なら両方である。

1. 同一熱流束で独立録音を複数回取得し、録音repeatを完全holdoutする。
2. 既存の複数実験日を使い、leave-one-experiment/date-outで学習日と評価日を分ける。

ここが成立すれば、「同じ録音内のchunkを当てた」から「別録音・別日に一般化した」へ主張が一段上がる。

## 6.2 最優先: 2–5 kHzの必要性・十分性を再学習で確認

共通22 kHz gridを使い、resize条件を固定した最小比較を行う。

| input | 問い |
|---|---|
| full 0.5–22 kHz | 基準 |
| 2–5 kHz only | この帯域だけで十分か |
| full minus 2–5 kHz | この帯域を消すと崩れるか |
| 0.5–2 kHz only | 隣接低域との比較 |
| 5–10 kHz only | 高域との比較 |

RFを主モデルとして先に実施し、深層モデルは代表1種で確認すればよい。全モデル・全条件の巨大gridは不要である。

## 6.3 深層モデルの公平な最終比較

アンサンブルを最終判定する前に、深層モデル側を最低限整える。

- 入力のglobalまたはtraining-fold normalization
- training-fold内validation
- early stopping
- seedをモデル・fold・周波数ごとに固定
- 3–5 seed反復
- 同じouter group folds
- 予測のcalibration slope/interceptを報告

この後もRFが勝ち、safe stackingも勝たなければ、アンサンブル不成立を最終結論としてよい。

## 6.4 ノイズ頑健性の定義を変える

同じnoise条件で学習・評価するだけでは、「そのノイズ条件を学習できる」ことは示せても、「未知ノイズに頑健」とは言えない。

最低限、次を分ける。

- train no-noise → test noise
- train mixed noise → test held-out SNR
- train one waterflow recording → test別waterflow recording
- no-noise XAIとnoise XAIの帯域比較

ノイズで重要帯域が水流音の強い帯域へ移る場合、それは頑健性ではなくshortcutの可能性として扱う。

---

## 7. ここからの実行順序

### Phase A: 論文の芯を確定する

1. 共通gridの帯域再学習ablationを行う。
2. RFでleave-one-experiment/date-outを行う。
3. 2–5 kHzが別日でも残るか、性能がどこまで低下するかを確認する。

この3点が最重要である。

### Phase B: モデル比較を公平に閉じる

4. 深層2モデルへnormalization、validation、early stopping、multi-seedを適用する。
5. 回帰精度、calibration、XAI帯域のseed再現性を比較する。
6. RFを最終主モデルにするかを確定する。

### Phase C: アンサンブルを一度だけ最終判定する

7. RF-anchor safe residual stackingをgroup-aware nested CVで評価する。
8. RF単体よりR²/RMSEだけでなく、条件間最悪値または将来確定するONB指標も改善するかを見る。
9. 改善しなければ追加の固定重み探索は終了し、負の結果を論文へ採用する。

### Phase D: 必要なら頑健性へ拡張する

10. train/test noiseを分ける。
11. corrected noise dataでgroup maskとXAIを再実行する。
12. 十分な独立録音が得られた場合だけnoise-state gateを検討する。

---

## 8. 明確な停止条件

研究が再び無限に広がらないよう、次を停止条件にする。

### アンサンブルを終了してよい条件

次を満たしたら、RF単体を採用し、アンサンブル不成立を結論とする。

- 深層モデルをmulti-seedで公平に整えた。
- group-aware nested stackingを行った。
- RF単体を候補に含めた。
- 別録音または別日評価で、ensembleがRFを一貫して上回らない。

この条件で勝たないなら、さらに重みを細かく探しても論文上の価値は小さい。

### 2–5 kHzを主結果にしてよい条件

- `2–5 kHz only`がfullに近い性能を維持する。
- `full minus 2–5 kHz`が明確に悪化する。
- 別録音・別日でも傾向が再現する。
- 複数seed・複数モデルのmask/IGで方向が一致する。

ここまで確認できれば、「有効帯域」から「必要性・十分性を伴う主要帯域」へ主張を強められる。

---

## 9. 最終的な研究質問の再定義

現在の結果に合わせるなら、研究質問は4つで十分である。

### RQ1

音響スペクトログラムからの熱流束回帰は、録音元・実験日・ノイズ条件を越えてどこまで成立するか。

### RQ2

予測に必要な時間・周波数情報は何か。また、その根拠は異なるモデルと説明手法を越えて再現するか。

### RQ3

異種モデルensembleは最良単体モデルの精度または頑健性を改善するか。改善しない場合、それは個別精度、校正、誤差相関、重み推定のどれで説明できるか。

### RQ4

上記の成立条件と限界を踏まえ、実用的な主モデル、入力帯域、不確実性の扱いをどう選ぶべきか。

---

## 10. 論文で使える結論文案

### 現時点で安全に言える版

> 本研究では、プール沸騰音響スペクトログラムを用いた熱流束回帰について、録音元を分離したgroup cross-validationにより複数モデルと周波数表現を比較した。現条件ではRandomForest系モデルが最も高く安定した性能を示した。Integrated Gradients、Grad-CAM、TreeSHAPおよびモデル共通の時間・周波数帯maskを段階的に検証した結果、個別画像の解釈にはbaselineやmodel-dependenceに関する制約がある一方、全モデルの予測性能は2–5 kHz帯の除去に対して最も大きく低下した。異種モデルensembleについては、弱い校正と相関した残差により、単純平均および学習重みが最良単体RFを安定して上回らなかった。以上から、本装置条件における主要な予測帯域と、説明可能な音響熱流束推定および異種モデル統合の成立条件・限界を示した。

### 追加検証後に目指す版

> 共通周波数grid上の再学習ablationと独立実験日評価により、2–5 kHz帯が熱流束推定に対して必要かつ概ね十分であることを確認した。複数モデル・複数seedの説明結果も同帯域で一致した。一方、異種モデルensembleは、メンバー間の特徴利用と残差が高く相関し、最良単体RFを一貫して上回らなかった。この結果から、沸騰音響回帰における性能向上にはモデル数ではなく、独立な物理情報、予測校正、group-awareな重み推定が必要であることを示した。

後半の「必要かつ十分」は、帯域再学習と別日再現が完了するまでは使わない。

---

## 11. 今は行わなくてよいこと

- 現行resizeのままmaxfreq候補をさらに細かく増やすこと
- 結果を見ながらRF重みを0.981、0.982のように細かく探索すること
- Grad-CAMの見た目だけから物理現象を断定すること
- 同じ録音からchunkを増やして標本数が増えたとみなすこと
- アンサンブルを主成果にするためだけに複雑なgateを追加すること
- 暫定ONB±10%のRMSE最適化へ研究全体を寄せること

現在必要なのは手法を増やすことではなく、2–5 kHzの因果的な帯域検証と、別録音・別日への一般化を確認して、主張の強さを決めることである。
