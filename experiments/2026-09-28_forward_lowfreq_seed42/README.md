# 6/11 clean学習→6/18評価・3/5 kHz seed 42解析

## 結論

指定した6/11 clean学習→6/18評価、3/5 kHz、seed 42、各7 noiseの14条件はすべて完了した。7/9、選別、XAIの混入はなく、周波数ごとに1個のclean学習`fit_id`を7条件へ共有している。

低周波入力の絶対性能は順方向でも22 kHzより良く、noiseあり平均RMSEは3 kHz performance 92.6・等重み92.9、5 kHz AlexNet 94.4・等重み96.7に対し、22 kHz performance 145.3・等重み137.1 kW/m²だった。したがって3–5 kHz優位の方向再現は予備的に支持された。

ただし順方向の3/5 kHz deep modelはcleanで正biasとONB前誤報が大きく、noise追加により予測が低下してRMSEが改善した。逆方向ではnoiseとともに誤差が増えており、同じ低RMSEでも機序が異なる。現段階では「低周波モデルがnoise不変」とせず、「低周波では22 kHzより絶対性能が良いが、noiseが日付間校正差を相殺する場合を含む」と解釈する。

## 完了性

- run: `202609/28/onb_xd-t0611-v0618_iw3-nc_s0_e150_160352`
- seed 42、150 epochs、WAV非共有3-fold、`clean_only`、選別なし、XAIなし。
- 3 kHz・5 kHzともclean＋SNR 0、−4、−8、−12、−16、−20 dBの7/7条件完了。
- 3 kHzはfit `e84599cd87b9`、5 kHzはfit `f47d705abac3`を各7評価へ共有。

## 周波数別の比較

RMSEはkW/m²。noiseは6条件平均。

| 周波数 | 手法 | clean RMSE | noise平均RMSE | noise平均R² | noise平均FPR |
|---|---|---:|---:|---:|---:|
| 3 kHz | RF | **100.1** | 104.8 | .850 | .100 |
|  | Conformer | 164.2 | 95.3 | .875 | .037 |
|  | AlexNet | 138.3 | 97.7 | .871 | .001 |
|  | performance | 123.0 | **92.6** | **.884** | **.000** |
|  | 等重み | 121.2 | 92.9 | .883 | .001 |
| 5 kHz | RF | **98.5** | 108.5 | .837 | .171 |
|  | Conformer | 142.6 | 129.3 | .771 | .332 |
|  | AlexNet | 113.4 | **94.4** | **.880** | **.000** |
|  | performance | 105.9 | 97.6 | .870 | .019 |
|  | 等重み | 104.5 | 96.7 | .872 | .012 |
| 22 kHz | RF | 95.7 | **101.1** | **.860** | **.108** |
|  | Conformer | **75.2** | 210.4 | .346 | .841 |
|  | AlexNet | 106.8 | 158.7 | .652 | .755 |
|  | performance | 83.7 | 145.3 | .701 | .545 |
|  | 等重み | 82.9 | 137.1 | .733 | .458 |

cleanでは22 kHz Conformerが最良で、3/5 kHz deep modelは悪い。一方、noise平均では3/5 kHz deep・統合が大幅に良い。周波数上限の選択はclean精度とnoise下安定性のトレードオフを持つ。

## noiseで性能が改善した理由

### 3 kHz Conformer

| 条件 | RMSE | bias | ONB前FPR |
|---|---:|---:|---:|
| clean | 164.2 | +87.2 | .660 |
| 0 dB | 120.9 | +50.5 | .221 |
| −4 dB | 100.3 | +26.8 | .000 |
| −8 dB | 86.7 | +0.4 | .000 |
| −12 dB | 86.6 | −24.9 | .000 |
| −16 dB | 90.6 | −33.6 | .000 |
| −20 dB | 86.8 | −8.9 | .000 |

noise強度とともにcleanの正biasが減り、−8 dB付近でほぼ0になる。これは同一モデルがnoiseに不変なのではなく、日付間の過大予測を付加noiseが反対方向へ動かして相殺した結果である。

5 kHzでもConformer clean bias +58.2、AlexNet +13.3に対し、noiseで低下する。22 kHzでは反対にnoiseで正biasが増加した。入力帯域によりwaterflow noiseが予測を動かす方向が変わっている。

このため、無雑音からの劣化量だけをnoise耐性として読まない。絶対RMSE、bias、FPR、Recallを併記する。

## アンサンブルの成立範囲

順方向seed 42の7条件では、次の結果だった。

| 周波数 | 方式 | RFに勝利 | 最良単体に勝利 | 最良単体2%以内 | 平均RF差 |
|---|---|---:|---:|---:|---:|
| 3 kHz | performance | 5/7 | 1/7 | 2/7 | −7.16 |
| 3 kHz | 等重み | 5/7 | 1/7 | 2/7 | −7.11 |
| 5 kHz | performance | 6/7 | 4/7 | 5/7 | −8.29 |
| 5 kHz | 等重み | 6/7 | 4/7 | 5/7 | −9.26 |

3 kHz統合はnoise平均では良いが、単体Conformerが−8～−20 dBで良いため、最良単体にはほぼ勝たない。5 kHz統合は0～−16 dBで比較的有効だが、−20 dBではAlexNet 90.6に対しperformance 122.9、等重み120.6へ悪化する。

逆方向でもmoderate noiseでは統合が有効、−20 dBではAlexNet単体が優位だった。よって「低周波統合はmoderate noiseで有効、極強noiseでは単体AlexNetが有効」という境界は両方向で整合する。

## 双方向比較

seed 42のnoise平均RMSEを比較する。

| 周波数 | 方式 | 6/11→6/18 | 6/18→6/11 |
|---|---|---:|---:|
| 3 kHz | RF | 104.8 | 102.9 |
|  | performance | **92.6** | **93.8** |
|  | 等重み | 92.9 | 94.2 |
| 5 kHz | RF | 108.5 | **88.8** |
|  | performance | 97.6 | 103.8 |
|  | 等重み | **96.7** | 97.0 |

3 kHz統合は両方向で約93～94、5 kHz等重みは約97で、noise平均の方向差が小さい。一方、最良単体は順方向で3 kHz Conformer／5 kHz AlexNet、逆方向では3 kHz deep系／5 kHz RFとなり、モデル順位は方向で変わる。

## 重み移転

順方向cleanでは内部OOF最良が3/5 kHzともConformerだったが、外部clean最良はRFだった。3 kHz performanceはConformerへ.396、RFへ.323、5 kHzはConformerへ.374、RFへ.315を与え、cleanでは等重み・performanceともRFより悪い。

逆方向でも内部OOFはdeep modelを高く評価し、外部cleanはRFが最良だった。したがって低周波でも`performance_kfold`の順位移転問題は解消しない。performanceがnoise条件で良い場合があることと、重み決定根拠が外部へ移ることは分ける。

## 研究上の判断

### 今回支持されたこと

- 3–5 kHzのnoise下絶対性能が22 kHzより良い傾向は順方向でも確認された。
- 5 kHz等重みはmoderate noiseでRF・最良単体を上回る条件が多い。
- 極強noiseでは両方向ともAlexNet単体が統合より良い。
- 3/5 kHzのnoise平均統合性能はseed 42で方向差が小さい。

### まだ確定していないこと

- 順方向3/5 kHzの結果がseed 43・44でも再現すること。
- noiseによるRMSE改善が頑健性を意味すること。現結果はbias相殺を含む。
- 2.1–2.5 kHz帯域が改善原因であること。
- 3 kHzと5 kHzのどちらを最終採用するか。

## 次に行うこと

低周波優位は方向を反転しても確認できたため、計画どおり同じ順方向3/5 kHzをseed 43・44で追加する。条件はseed以外を変更しない。

3 seed完成後に、双方向×3 seedについて次を判定する。

1. 3 kHz統合と5 kHz統合の方向・seed安定性。
2. moderate noiseでの最良単体勝利数。
3. cleanからnoiseへの変化ではなく、各SNRの絶対RMSE・FPR・Recall。
4. 3/5 kHzのどちらを説明性対象に固定するか。

現行コードは順方向3/5 kHz・seed 42の完了済み条件であるため、そのまま再実行しない。次は`random_seed=43`へ変更し、ほかは固定する。

## 生成物

- `metrics.csv`: 双方向seed 42の3/5/22 kHz指標
- `direction_frequency_summary.csv`: clean・noise平均比較
- `forward_ensemble_success.csv`: RF・最良単体との比較
- `weights.csv`: 内部OOF重みと外部clean性能
- `deep_residual_correlations.csv`: 深層2モデル残差相関
- `analyze.py`: 保存予測から再生成するスクリプト

