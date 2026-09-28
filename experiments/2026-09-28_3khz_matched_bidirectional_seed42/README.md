# 3 kHz matched双方向・seed 42解析

## 結論

6/11→6/18と6/18→6/11の3 kHz、matched、seed 42、各7条件は14/14完了した。各方向で7 noiseが7個の別fitを持ち、noiseごとにモデル・PCA・scaler・`performance_kfold`重みを独立して学習している。cleanの予測は対応するclean_only runと全モデル・両統合で最大絶対差0であり、学習方針以外の比較条件は揃っている。

主要結果は次の通り。

1. **matchedの効果は方向で反転した。** `performance_kfold`のnoise平均RMSEは6/11→6/18でclean_only 92.6からmatched 97.5へ4.9 kW/m²悪化し、6/18→6/11では93.8から76.2へ17.6 kW/m²改善した。
2. **双方向平均の改善は主に逆方向・ONB前領域による。** performanceの双方向RMSEは93.2から86.9へ改善したが、ONB前RMSEは116.9から96.1へ改善する一方、ONB以上は67.7から79.1、ONB近傍は91.2から123.1へ悪化した。
3. **matchedは固定閾値での誤報を大きく減らした。** performanceの双方向平均FPRは.113から.001、Precisionは.950から.999、F1は.885から.912へ改善した。Recallは.844から.838へわずかに低下した。連続ROC-AUCは.931から.927、PR-AUCは.966から.964で改善していないため、主な効果は順位識別より予測値の校正移動と考えられる。
4. **現パラメータではperformance重みの明確な上積みはない。** matched双方向noise平均RMSEはRF 86.8、performance 86.9、等重み86.5 kW/m²で実質同等だった。performanceはRFを7/12条件、事後最良単体を5/12条件で上回った。
5. **matchedでも内部モデル順位の別日移転は解決しない。** noiseあり12 cellで内部OOF最良と外部最良単体は2/12一致、平均Spearman順位相関は−.333だった。`performance_kfold`は本人方針どおり主方式として継続するが、等重みとRFを必ず併記し、重みの有効性はチューニング後に再判定する。

## 対象runと監査

| 方向 | run | 完了 | fit共有 |
|---|---|---:|---|
| 6/11→6/18 | `onb_xd-t0611-v0618_iw3-nm_s0_e150_194237` | 7/7 | 7 noiseで7 fit |
| 6/18→6/11 | `onb_xd-t0618-v0611_iw3-nm_s0_e150_205608` | 7/7 | 7 noiseで7 fit |

共通条件は3 kHz、150 epochs、内部WAV非共有3-fold、選別なし、seed 42、`performance_kfold`＋等重み、XAIなし。学習日は評価日と完全分離している。比較するclean_onlyは同じ3 kHz・seed 42・モデル条件の既存runを用いた。

## noiseあり6条件の方向別平均

RMSE、bias、ONB RMSEはkW/m²。

| 方向 | 方針 | 手法 | RMSE | R² | bias | ONB RMSE | FPR | Recall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 6/11→6/18 | matched | RF | 98.1 | .870 | −43.8 | 151.8 | .004 | .842 |
|  |  | Conformer | 101.7 | .858 | −2.4 | **96.3** | .047 | .842 |
|  |  | AlexNet | 112.2 | .830 | −40.8 | 115.7 | .002 | .803 |
|  |  | performance | **97.5** | **.872** | −28.2 | 114.8 | .002 | .828 |
|  |  | 等重み | 97.8 | .871 | −29.0 | 115.3 | .002 | .826 |
| 6/11→6/18 | clean_only | RF | 104.8 | .850 | −39.4 | 150.2 | .100 | .863 |
|  |  | Conformer | 95.3 | .875 | +1.7 | **103.4** | .037 | .818 |
|  |  | AlexNet | 97.7 | .871 | −31.0 | 133.2 | .001 | .792 |
|  |  | performance | **92.6** | **.884** | −20.7 | 120.3 | .000 | .808 |
|  |  | 等重み | 92.9 | .883 | −22.9 | 122.7 | .001 | .807 |
| 6/18→6/11 | matched | RF | 75.4 | .921 | +30.6 | **97.5** | .002 | **.867** |
|  |  | Conformer | 86.3 | .896 | +16.0 | 126.0 | .007 | .841 |
|  |  | AlexNet | 83.4 | .903 | −15.9 | 164.0 | .000 | .827 |
|  |  | performance | 76.2 | .919 | +8.7 | 131.5 | .000 | .849 |
|  |  | 等重み | **75.2** | **.921** | +10.2 | 128.1 | .000 | .851 |
| 6/18→6/11 | clean_only | RF | 102.9 | .840 | +61.4 | 71.6 | .358 | .913 |
|  |  | Conformer | 101.3 | .846 | +54.8 | 74.7 | .296 | .880 |
|  |  | AlexNet | **89.6** | **.887** | +27.8 | **51.3** | .108 | .846 |
|  |  | performance | 93.8 | .871 | +45.4 | 62.1 | .227 | .879 |
|  |  | 等重み | 94.2 | .869 | +48.0 | 62.8 | .229 | .893 |

forwardではmatchedによりRFだけがnoise平均で6.7 kW/m²改善した。Conformerは6.4、AlexNetは14.6、performanceは4.9、等重みは4.9 kW/m²悪化した。noiseを学習へ入れれば全モデルが良くなるわけではない。

reverseでは全モデルが改善し、performanceは17.6、等重みは19.1 kW/m²改善した。とくにclean_onlyで強noise時に生じた正biasとONB前の大量誤報が解消された。ただし、matchedの改善を同一モデルの耐雑音性とは解釈しない。SNRごとに別モデルを学習した、noise既知条件への適応性能である。

## SNR別のperformance

| 方向 | SNR | clean_only RMSE | matched RMSE | matched−clean | 解釈 |
|---|---:|---:|---:|---:|---|
| 6/11→6/18 | 0 | 100.6 | 101.1 | +0.5 | ほぼ同等 |
|  | −4 | 90.2 | 96.8 | +6.6 | matched悪化 |
|  | −8 | 87.7 | 96.0 | +8.4 | matched悪化 |
|  | −12 | 91.2 | 99.3 | +8.1 | matched悪化 |
|  | −16 | 92.9 | 95.3 | +2.3 | matched悪化 |
|  | −20 | 92.9 | 96.5 | +3.6 | matched悪化 |
| 6/18→6/11 | 0 | 75.7 | 80.8 | +5.1 | matched悪化 |
|  | −4 | 75.9 | 77.5 | +1.6 | matchedやや悪化 |
|  | −8 | 79.5 | 79.1 | −0.4 | 同等 |
|  | −12 | 88.6 | 74.6 | −14.0 | matched改善 |
|  | −16 | 104.4 | 72.4 | −32.0 | matched大幅改善 |
|  | −20 | 138.6 | 73.1 | −65.5 | matched大幅改善 |

reverseの改善は−12 dB以降で急に大きくなる。WAV別ではperformanceが改善した本数は−12で8/18、−16で12/18、−20で11/18だった。−16・−20の改善は低熱流束WAVで大きく、ONB前誤報の解消と整合する。一方、確定ONBの221.5 kW/m² WAVは−4〜−16で最も悪化する側に入り、全域改善とONB近傍悪化が同時に起きた。

forwardでは改善WAVが0〜−20 dBで2〜7/18本に留まり、多くのWAVでmatchedが悪化した。−20 dBでも低熱流束の一部は改善したが、最高熱流束873.6 kW/m²などで悪化し、全域改善にはならなかった。

## ONB判定のトレードオフ

双方向noise平均を対応比較する。

| 方針・手法 | 全域RMSE | ONB前RMSE | ONB以上RMSE | ONB近傍RMSE | FPR | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched RF | 86.8 | **84.5** | 88.2 | 124.6 | .003 | .997 | **.855** | **.920** | .927 |
| matched performance | 86.9 | 96.1 | 79.1 | 123.1 | **.001** | **.999** | .838 | .912 | .927 |
| matched 等重み | **86.5** | 95.8 | **78.7** | 121.7 | .001 | .999 | .838 | .911 | .927 |
| clean_only RF | 103.8 | 120.1 | 84.6 | 110.9 | .229 | .895 | .888 | .879 | **.932** |
| clean_only performance | 93.2 | 116.9 | **67.7** | **91.2** | .113 | .950 | .844 | .885 | .931 |
| clean_only 等重み | 93.6 | 117.1 | 68.2 | 92.7 | .115 | .950 | .850 | .889 | .931 |

matchedは固定ONB閾値でのPrecision・F1を改善したが、連続順位指標はわずかに低下した。したがって「ONB識別情報が増えた」よりも、「clean_onlyの過大予測を下げ、既定閾値との位置関係を改善した」ことが中心である。

ONB近傍RMSEの悪化とF1の改善は矛盾しない。reverseのclean_onlyは予測を全体に高くずらし、ONB前に誤報しながらONB付近へ偶然近づいていた。matchedはこの正biasを抑えて誤報を消すが、ONB付近の連続値を低く外し、Recallと近傍RMSEを悪化させる。この結果では全域回帰、固定閾値判定、ONB近傍回帰のどれを主目的とするかで評価が変わる。

## アンサンブルと重み

noise 12 cellの結果:

| 方針 | 方式 | 最良単体に勝利 | 2%以内 | RFに勝利 | 最良単体との差 | RFとの差 |
|---|---|---:|---:|---:|---:|---:|
| matched | performance | 5/12 | 6/12 | 7/12 | +1.43 | +0.11 |
| matched | 等重み | 5/12 | 8/12 | 7/12 | +1.05 | −0.26 |
| clean_only | performance | 3/12 | 5/12 | 10/12 | +4.00 | −10.62 |
| clean_only | 等重み | 4/12 | 5/12 | 11/12 | +4.40 | −10.23 |

matchedではRF自体が大きく改善し、統合の追加利得がほぼ消えた。既知noiseへの適応学習が強い場合、アンサンブルがnoise劣化をさらに抑える余地は小さい可能性がある。clean_onlyでは統合のRF比改善が大きく、未知noiseへの固定モデル転送で補完の価値が出ている。

performanceと等重みが近い理由は、matchedの強noiseで重みが約1/3ずつへ近づくためである。reverseでは0 dBのRF重み.188、−4 dB .204、−8 dB .257で、外部評価ではRFが最良なのに内部OOFはConformerを最良とした。−20 dBではRF/Conformer/AlexNet=.335/.335/.330となり、performanceは実質等重みだった。

内部OOF最良と外部最良の一致はmatched全14 cellで2/14、noiseだけで2/12。forward −16 dBとreverse −20 dBだけ一致した。モデル順位が別日へ移らない問題はmatchedでも残る。パラメータチューニングで変わる余地はあるが、現時点で「性能に応じた重みが等重みより有効」とは言えない。

## 誤差補完の方向差

forward matchedでは−20 dBで3モデルのbiasが−38.8〜−46.6 kW/m²へ揃い、残差相関もRF–Conformer .910、RF–AlexNet .923、Conformer–AlexNet .915だった。同じ向きの誤りを平均しても相殺しにくい。

reverse matched −20 dBではRF bias +30.1、Conformer −8.4、AlexNet −8.2 kW/m²となり、RF–Conformer残差相関も.735まで下がった。この正負biasの補完により、performance 73.1、等重み73.1がRF75.1をわずかに上回った。これは内部順位が正しかったからではなく、予測誤差の相殺と整合する。

## 研究上の解釈

### 確認した事実

- 3 kHz matchedは強noiseのreverseでclean_onlyの過大予測・誤報を大きく抑えた。
- forwardではRF以外のnoise平均RMSEが悪化し、統合もclean_onlyを下回った。
- matched双方向平均ではRF・performance・等重みが86.5〜86.9 kW/m²に並んだ。
- 固定閾値F1は改善したが、ONB近傍RMSEと連続ROC/PR-AUCは改善しなかった。
- performanceの内部順位は別日外部順位へ十分移らなかった。

### 整合する原因仮説

- reverse clean_onlyでは水流noiseが全モデルを過大予測側へ動かしていたため、同じnoiseで学習したmatchedが校正差を抑えた。
- forward clean_onlyではnoiseがclean時の日付間過大予測を既に相殺していた。noise別再学習により別の負biasが生じ、deep modelの全域誤差が増えた可能性がある。
- matchedの学習noiseと評価noiseは生成方法・素材分布が対応しており、noise条件への適応を学びやすい。独立した水流noiseへの一般化を示す結果ではない。

これらはbias・領域別誤差と整合するが、入力特徴の因果的説明はまだない。同一日内評価と帯域摂動で識別する。

## 次に行うこと

当初計画どおり、次は日内固定holdoutをseed 42で行う。

1. 6/11日内clean_only、6/18日内clean_only。
2. 同じ外側テストWAVで6/11日内matched、6/18日内matched。
3. 日内で内部OOF順位と外部holdout順位が一致するか、matchedの方向差が残るかを確認する。

この比較により、matchedの非対称性が「学習noiseの効果」より「実験日をまたぐ校正差」に強く由来するかを判断できる。日内でも同じ非対称性が残れば、日ごとのnoise応答・入力分布差を優先して調べる。

現段階でパラメータ探索を先に行うと、日付間移転とモデル容量・学習率の影響が混ざる。日内4 runを読んだ後、学習側限定チューニングへ進む。matchedを修論の主要条件にする場合は、その後にcross-day matchedのseed 43・44を追加して今回の方向差を再現確認する。

3 kHzを主帯域、`performance_kfold`を主方式として進める本人方針は維持する。ただしRF・等重みを対照として残し、performance固有の利得、全域性能、ONB近傍、誤報・見逃しを別々に報告する。

## 解析出力

- [解析スクリプト](analyze.py)
- [全条件指標](metrics.csv)
- [方向別noise平均](noise_direction_average.csv)
- [双方向noise平均](noise_bidirectional_average.csv)
- [matched−clean_only差](matched_minus_clean.csv)
- [WAV別対応差](wav_paired_deltas.csv)
- [重みと内部・外部性能](weights.csv)
- [重み順位移転](weight_transfer.csv)
- [重み順位移転集計](weight_transfer_summary.csv)
- [アンサンブル成立数](ensemble_success.csv)
- [残差相関](residual_correlations.csv)
- [clean予測同一性](clean_identity.csv)
