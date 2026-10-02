# 3 kHz採用値の外側評価と22 kHz比較

解析日: 2026-10-01。

**2026-10-02優先順位の更新**：以下の数値・確認事実は保持する。「低FPRを保ちONB直後の見逃しを改善」と次のmatched比較は、解析に基づくAI提案であり、教授と共有・合意済みの課題ではない。本人の最新指示により、直近は[後期目標・教授の助言・実施状況の整理](../../docs/research_plan/2026-10-02_second_semester_goal_progress_and_professor_feedback_audit.md)を優先する。本書の追加run案を必須の次工程として自動再開しない。

## 結論

現行の研究範囲で主入力を3 kHzに固定する判断を、今回の外側評価は強く支持する。22 kHzはclean全域RMSEでは3 kHzより5.10 kW/m²良かったが、未学習の付加水流noiseで低熱流束を大きく過大予測し、0 dBでFPR 46.67%、−4 dBで88.00%、−8 dB以下で100%となった。3 kHzはnoise平均RMSE 69.30 kW/m²、FPR 0.15%で、22 kHzの189.55 kW/m²、89.11%より明確に頑健だった。

ただし、3 kHzもnoise不変ではない。performanceのRMSEはclean 35.07から−20 dB 82.46 kW/m²へ単調に悪化し、Recallは.863から.797へ低下した。強noise時には低熱流束からONB直後までの予測が約160 kW/m²付近へ圧縮される。これはONB前誤報を防ぐ一方、ONB直後の見逃しを生む。したがって、現在の主問題は「3 kHzで誤報をさらに減らすこと」ではなく、**低いFPRを保ったまま、noise時のONB直後の過小予測とRecall低下を改善できるか**である。

今回の3 kHzチューニングは、従来値に対してnoise平均RMSEを69.64から69.30、強noise平均を78.36から77.13 kW/m²へ僅かに改善しただけである。−20 dB全域は5.74 kW/m²改善した一方、強noise平均ONB近傍RMSEは94.78から103.54 kW/m²へ悪化し、q100は14/14の日×noiseセルで不変だった。よって、3 kHzの頑健性を「チューニングで獲得した」とは解釈せず、主に入力帯域の差として扱う。

## 対象runと監査

- 3 kHz: `202610/01/onb_wc-t0611-v0611_iw3-nc_c1s_s0_e150_170655`
- 22 kHz対照: `202609/30/onb_wc-t0611-v0611_iw3-nc_c1s_s0_e150_190945`
- 共通条件: 1秒、6/11＋6/18統合、`within_wav_chunk`、選別なし、`clean_only`、seed 42、150 epochs、内部WAV GroupKFold 3-fold
- 外側学習1,620 chunk、外側評価540 chunk（6/11と6/18を各270）、36 WAVすべてで学習45／評価15 chunk、同一chunk重複なし
- 3 kHzの7/7条件に完了印があり、全条件がfit ID `b6bde0ab4ebe`を共有した。つまりcleanで学習した同一モデルを7 noiseへ固定適用した比較である。
- 3 kHz採用値: RF `100 / depth 12 / 0.6 / 0.6`、Conformer `lr=.001 / batch 12`、AlexNet `lr=.003 / batch 8`
- 22 kHz採用値: RF `600 / depth 6 / 0.6 / 0.6`、Conformer `lr=.0003 / batch 8`、AlexNet `lr=.01 / batch 24`
- ONB判定は保存時の統合平均246.591 kW/m²ではなく、6/11の221.505、6/18の271.678 kW/m²を各sampleへ割り当てて再集計した。

この評価は既知WAV内の未使用chunkを対象とする。未知WAV・未知実験日一般化の証拠とはしないが、本人の現行修論範囲には合っている。

## performanceの3 kHz結果

| noise | 全域RMSE | ONB近傍RMSE | ONB前bias | FPR | Recall |
|---:|---:|---:|---:|---:|---:|
| clean | 35.07 | 45.46 | +1.22 | .004 | .863 |
| 0 dB | 53.36 | 61.07 | +32.19 | .004 | .854 |
| −4 dB | 62.86 | 80.47 | +34.42 | .004 | .851 |
| −8 dB | 68.16 | 103.21 | +27.51 | 0 | .829 |
| −12 dB | 72.20 | 119.10 | +25.36 | 0 | .816 |
| −16 dB | 76.72 | 105.59 | +50.21 | 0 | .806 |
| −20 dB | 82.46 | 85.94 | +75.13 | 0 | .797 |

単位はRMSE・biasがkW/m²。全域RMSEはnoiseが強くなるほど単調に悪化する。ONB近傍RMSEが−20 dBで見かけ上小さく戻るが、同時にRecallは低下しているため、性能回復とは読まない。予測分布の移動により誤差の位置が変わった結果である。

日別にも方向は共通していた。−20 dB全域RMSEは6/11が78.19、6/18が86.52 kW/m²、FPRは両日0、Recallは.818と.773だった。6/18のONB直後の見逃しがより大きいが、一方の日だけで全体傾向が生じたわけではない。

## モデルがどの入力でどう出力したか

### 3 kHz

cleanでは、低熱流束に対する平均予測は6/11でおよそ39–49、6/18で14–25 kW/m²から始まり、熱流束とともに概ね単調に増加した。ONBちょうどは6/11で184.6、6/18で235.9 kW/m²と両日とも過小予測し、clean時点でもONB直後は一部を見逃している。

−20 dBでは、両日の0–約150 kW/m²入力に対する平均予測が約156–164 kW/m²へ集まった。この値は両日のONB閾値より低いためFPRは0に保たれた。一方、ONBちょうどの平均予測は6/11で158.7、6/18で170.8 kW/m²、次の測定点でも208.2、184.8 kW/m²となり、ONB後にも陰性が残った。100%陽性になるq100は6/11で368.978、6/18で434.018 kW/m²で、全noiseを通じて変わらなかった。

すなわち3 kHzは、強noiseを「沸騰が強い」と誤認するのでなく、低〜ONB近傍を中間的な一定値へ圧縮する傾向を示す。このため誤報には強いが、早いONB検知には遅れが生じる。

### 22 kHz

22 kHzは−4 dBの時点で、ONB前入力に対する平均予測が6/11で約239–266、6/18で約269–307 kW/m²へ上がった。−20 dBでは両日のONB前入力が一様に約471–483 kW/m²へ押し上げられた。したがって−8 dB以下ではONB前225 chunkすべてが陽性となり、FPR 1.0、Recall 1.0になった。

このRecall 1.0は検知改善ではなく常時陽性化である。同様にq100が0付近まで早まることも改善ではない。q100は必ずFPRと併記する必要がある。

## 3 kHzと22 kHzの対応比較

| 集計 | 3 kHz RMSE | 22 kHz RMSE | 3 kHz差 | 3 kHz FPR | 22 kHz FPR |
|---|---:|---:|---:|---:|---:|
| clean | 35.07 | 29.97 | +5.10 | .004 | .004 |
| noise 6条件平均 | 69.30 | 189.55 | −120.25 | .0015 | .8911 |
| 強noise 3条件平均 | 77.13 | 240.08 | −162.95 | 0 | 1.0 |
| −20 dB | 82.46 | 271.47 | −189.01 | 0 | 1.0 |

対応36 WAVのcluster bootstrapでは、全域RMSEの3 kHz−22 kHz差はcleanで+5.10 kW/m²、95% CI [+0.59, +9.77]で22 kHzが良かった。noise 6条件はすべて差の95% CIが0未満で、3 kHzが良かった。たとえば−20 dBは−189.01、95% CI [−225.15, −146.62] kW/m²である。

ONB近傍には段階的な交差がある。0、−4 dBでは3 kHzが22 kHzより26.10、32.49 kW/m²悪く、−8 dBは差が未確定、−12 dB以下では3 kHzが良い。しかし22 kHzの中程度noiseにおけるONB近傍の小ささは、予測全体を上へ動かした結果を含み、同時にFPRが46.67–100%である。ONB近傍RMSE単独では22 kHzを採用できない。

## アンサンブルの評価

3 kHzのperformance重みはRF .220、Conformer .426、AlexNet .354だった。22 kHzは.158、.394、.448で、3 kHzよりAlexNetが重くRFが軽い。

3 kHz performanceは全7 noiseで等重みと事後最良単体の両方より全域RMSEが小さかった。cleanではperformance 35.07、Conformer 35.16、等重み38.54、−20 dBでは82.46、83.70、87.25 kW/m²だった。現状の全域回帰の主方式としてperformanceを維持する根拠はある。

ただしONB用途では一貫して最良ではない。強noise平均ONB近傍RMSEはRF 74.12、等重み93.46、Conformer98.52、performance 103.54、AlexNet165.66 kW/m²だった。performance重みはclean学習側OOFの全域誤差から決まり、noise下ONB性能を直接最適化していないためである。全域回帰とONB検知で同じ「最良」を主張しない。

## 従来3 kHz値との比較

従来run `202609/29/..._161952`とsampleを対応させた。従来値はRF `300 / depth 4`、Conformer `.001 / 12`、AlexNet `.001 / 12`である。

| 集計 | tuned | 従来 | tuned−従来 |
|---|---:|---:|---:|
| clean全域RMSE | 35.07 | 35.33 | −0.26 |
| noise平均RMSE | 69.30 | 69.64 | −0.34 |
| 強noise平均RMSE | 77.13 | 78.36 | −1.23 |
| −20 dB全域RMSE | 82.46 | 88.20 | −5.74 |
| 強noise平均ONB近傍RMSE | 103.54 | 94.78 | +8.76 |

全域差のWAV cluster bootstrap 95% CIはclean、0、−4、−8、−12、−16 dBで0をまたぎ、−20 dBだけが[−10.10, −0.25] kW/m²だった。q100は日別14/14セルで完全一致した。採用値は学習側OOFで事前選択済みなので維持するが、外側結果から「大幅に改善した」とは言わない。

## 原因仮説

確認事実と整合する第一仮説は、3 kHzより上の帯域がclean回帰には追加情報を与える一方、今回の付加水流noiseに対する大きな感度も持つことである。22 kHzではdeep 2モデルのnoise平均FPRがConformer .980、AlexNet .887で、RFも.484だったため、単にアンサンブル重みだけが原因ではない。入力帯域と各モデルの応答の両方に由来する。

3 kHzで低熱流束予測が約160 kW/m²へ収束する理由は、noiseが低周波の熱流束差を覆い、モデルが中間的な出力へ回帰している可能性がある。22 kHzで約480 kW/m²へ上がる理由は、追加高周波帯のnoise特徴を高熱流束・沸騰特徴として誤認している可能性がある。ただし、帯域遮蔽や周波数別noiseエネルギーとの対応を今回再検証していないため、物理的原因としては仮説に留める。

## 次に行うこと

1. 主条件は3 kHz、1秒、選別なし、6/11＋6/18統合、`within_wav_chunk`、`performance_kfold`に固定する。22 kHzはclean精度とのトレードオフを示す対照とし、主条件へ戻さない。
2. 外側結果を見て採用パラメータを選び直さない。現在値を3 kHzの固定候補として扱う。
3. 次の識別比較は、同じ3 kHz採用値・split・seed・選別なしで`clean_only`と`matched`だけを変える。判定課題は、matchedがFPRほぼ0を維持しながら、強noiseのONB直後の過小予測、Recall、ONB近傍RMSEを改善できるかである。全域RMSEだけで採否を決めない。
4. matchedでもONB直後が改善しなければ、学習方針を増やす前に、3 kHz強noiseで生じる約160 kW/m²への圧縮を周波数寄与・モデル別予測で説明する。未知WAV・未知日は本人方針どおり直近の必須課題には戻さない。
5. 最終表へ載せる段階で、固定済み3 kHz候補の学習seed 43・44を追加し、数値幅を確認する。これは候補再選択でなく再現性確認とする。

## 保存物

- `analyze.py`: 完了性監査、日別ONB再集計、周波数比較、従来3 kHz比較、WAV cluster bootstrap
- `completion_audit.csv`: run・split・fit監査
- `metrics_source_day_thresholds.csv`: 日別確定ONBによる全モデル指標
- `heatflux_metrics_performance.csv`: 日・熱流束別の平均予測、bias、陽性率
- `q100_source_day_thresholds.csv`: 日別q100/g100
- `aggregate_two_day.csv`: clean、noise平均、強noise平均、−20 dB集計
- `frequency_comparison.csv`: 3/22 kHzの対応差
- `ensemble_comparison.csv`: performance、等重み、事後最良単体の比較
- `weights.csv`: 保存アンサンブル重み
- `cluster_bootstrap_frequency_delta.csv`: 3/22 kHz対応WAV bootstrap
- `tuning_comparison.csv`: tuned 3 kHzと従来3 kHzの対応差
- `cluster_bootstrap_tuning_delta.csv`: tuned/従来の対応WAV bootstrap
- `tuning_q100_comparison.csv`: tuned/従来の日別q100一致監査
