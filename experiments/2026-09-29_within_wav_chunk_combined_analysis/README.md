# 6/11＋6/18統合within_wav_chunk解析

記録日: 2026-09-29

## 今回答える問い

6/11と6/18を統合してclean学習した1つのモデルは、各日を別々にclean学習したモデルより、同じテストchunk上で全域回帰・未知水流noise・ONB判定を改善するかを確認した。

## 対象と完了性

- 統合run: `onb_wc-t0611-v0611_iw3-nc_s0_e150_161952`
- 条件: 1秒、3 kHz、7 noise、選別なし、150 epochs、学習seed 42、外側split seed 42、内部WAV GroupKFold 3-fold、`clean_only`、XAIなし
- 7/7条件に`completed.json`、run hashは全条件`61b0c5c1`、fit IDは全条件`1ceaf64e10c0`の1つだけ
- 外側学習1,620 chunk、外側テスト540 chunk、36 WAVが両側、同一chunk重複0
- 単日runとのテスト集合は6/11・6/18とも270/270一致し、全noiseでラベル差0

したがって以下の単日／統合差には、評価chunkの変更やnoiseごとの再学習は混ざっていない。

## 結論

1. 統合学習は、2日全体のperformance RMSEをcleanで39.57→35.33、noise平均で74.77→69.64、強noise平均で87.88→78.36 kW/m²へ改善した。特に6/18の改善が大きい。
2. ただし改善はONB領域で一様ではない。強noiseの2日合算FPRは.006→.001、全域RMSEは改善したが、Recallは.825→.807、ONB近傍RMSEは78.42→94.78 kW/m²へ悪化した。
3. 6/18 −20 dBでは、低熱流束の過大予測が大きく減り、全域RMSEは116.94→90.92、FPRは.042→0へ改善した。一方、Recallは.833→.773、ONB近傍RMSEは25.85→85.08へ悪化した。統合の主効果は、誤報抑制と引き換えにONB付近を低めに予測する校正移動である。
4. 6/11ではclean RMSEが42.22→38.21へ改善したが、noise平均は64.35→65.16、強noise平均は73.16→73.14でほぼ不変だった。統合効果は日ごとに対称ではない。
5. 統合runのperformanceは等重みを14/14の日×noise条件で上回り、事後的な最良単体にも9/14で勝った。ただし−20 dBでは両日ともAlexNet単体に負けた。performance重みの利点は増えたが、強noiseの万能な統合ではない。
6. `noise_trends`のperformance RMSEはcleanから−20 dBまで35.33→53.41→62.20→67.12→70.77→76.10→88.20 kW/m²と単調に悪化した。固定cleanモデルへのnoise増加として自然で、学部時代と似た傾向という本人の読みと整合する。

## 単日学習との対応比較

performance、単位はkW/m²。差は統合−単日で、負が改善。

| 評価日 | 学習 | clean | noise平均 | 強noise平均 | −20 dB |
|---|---|---:|---:|---:|---:|
| 6/11 | 単日 | 42.22 | 64.35 | 73.16 | 85.65 |
| 6/11 | 統合 | 38.21 | 65.16 | 73.14 | 85.39 |
| 6/11 | 差 | −4.01 | +0.80 | −0.02 | −0.26 |
| 6/18 | 単日 | 36.73 | 83.81 | 100.46 | 116.94 |
| 6/18 | 統合 | 32.20 | 73.78 | 83.19 | 90.92 |
| 6/18 | 差 | −4.53 | −10.03 | −17.27 | −26.02 |

単体モデル別では、6/18のnoise平均がConformerで−8.10、AlexNetで−14.99 kW/m²改善した一方、RFは−0.22 kW/m²相当のほぼ不変だった。6/11はAlexNetが−2.84改善したが、Conformerは+3.89、RFは+6.76悪化した。統合のnoise利得は主に深層2モデル、とくに6/18側から生じた。

## ONB前誤報とONB近傍見逃しのトレードオフ

日別の確定ONB（6/11: 221.505、6/18: 271.678 kW/m²）で再集計した。

### 6/11 −20 dB

- 全域RMSE: 85.65→85.39 kW/m²
- ONB前RMSE: 127.04→124.64 kW/m²
- FPR: .029→0
- Recall: .836→.824
- ONB近傍RMSE: 44.53→47.95 kW/m²

全体としてほぼ同等で、FPRが僅かに下がる代わりにRecallが僅かに下がった。

### 6/18 −20 dB

- 全域RMSE: 116.94→90.92 kW/m²
- ONB前RMSE: 168.30→118.56 kW/m²
- FPR: .042→0
- Recall: .833→.773
- ONB近傍RMSE: 25.85→85.08 kW/m²

6/18の改善は低熱流束で明瞭だった。予測biasは0、2.38、21.24、58.39 kW/m²で、それぞれ約+240→+178、+246→+179、+216→+156、+178→+122 kW/m²へ縮小した。一方、271.678 kW/m²のONB測定点はbiasが−19.49→−82.49、322.114 kW/m²は−62.94→−118.20 kW/m²へ低下した。したがって「低熱流束の誤報だけを直した」のではなく、予測校正全体が下方へ動いた結果である。

## q100と卒論指標への接続

- 6/11 performanceのq100は単日・統合、全noiseで368.978 kW/m²のまま（ONBとの差147.473 kW/m²）。
- 6/18は単日cleanの376.320から統合cleanの434.018 kW/m²へ遅くなり、noise条件は単日・統合とも434.018だった。

統合学習は全域RMSEとFPRを改善しても、「以降100%をONB以上と判断できる最初の熱流束」を早めなかった。6/18 cleanではむしろ遅れた。q100は測定熱流束の刻みと15 chunk全一致条件に強く依存するため補助指標のままだが、早期ONB検知が改善したとは言えない根拠になる。

## アンサンブル

performance重みは、RF .217、Conformer .438、AlexNet .344だった。単日6/11の.311/.411/.278と単日6/18の.173/.399/.428の中間に移り、最大重みはConformerとなった。

- performanceは等重みを6/11・6/18とも7/7条件で上回った。平均差は6/11で−1.94、6/18で−2.51 kW/m²。
- performanceは事後的な最良単体に6/11で6/7、6/18で3/7勝った。
- −20 dBではAlexNetが6/11で73.30、6/18で85.28 kW/m²と最良で、performanceは85.39、90.92だった。
- Conformer–AlexNet残差相関は−20 dBで6/11 .952、6/18 .961まで上がった。強noiseでは深層2モデルが似た方向へ誤るため、統合による相殺余地が小さい。

このrunではperformanceの等重み比優位が明確になったが、既知WAV内評価、1 seed、事後的最良単体比較である。未知WAV・未知日一般化まで広げない。

## 平均ONBで保存された指標の扱い

run本体は統合ONB 246.591 kW/m²で図と判定指標を保存している。全域RMSE・R²・MAEとnoise trendは閾値に依存しないためそのまま使える。一方、FPR・Recall・ONB近傍RMSE・q100は日別ONBでの後処理を主とする。

例えば−20 dBのperformanceは、平均ONBではFPR 0、Recall .850、ONB近傍RMSE 54.10 kW/m²だが、日別ONBを使うとFPR 0、Recall .800、ONB近傍RMSE 69.06 kW/m²となる。平均ONBだけではONB見逃しを過小評価する。

## 確認事実・原因仮説・次の識別比較

### 確認した事実

- 統合はcleanと2日全体のnoise RMSEを改善した。
- noise改善は6/18、低熱流束、深層モデルに偏った。
- FPRは下がったが、強noiseのRecallとONB近傍RMSEは悪化した。
- q100は改善しなかった。
- performanceは等重みを全14条件で上回ったが、−20 dBの最良単体はAlexNetだった。

### 整合する仮説

- 2日分へ学習量を増やしたことで、深層モデルのclean表現と6/18の低熱流束校正が改善した可能性がある。
- 6/11と6/18を同じ損失で学習したため、両日の校正差を平均化し、6/18の低熱流束過大予測を抑える一方でONB付近を過小予測した可能性がある。
- 強noiseでモデル残差が高相関になることが、−20 dBでperformanceがAlexNetを超えない一因と整合する。

これらはseed 42の1 runから原因確定したものではない。

### 次の識別比較

1. 外側split seed 42とテスト540 chunkを固定したまま、学習seed 43・44で同じ統合`clean_only`を再現する。
2. clean・noise平均・強noiseRMSEの利得に加え、6/18低熱流束bias縮小、日別Recall、ONB近傍RMSE悪化が同じ向きに再現するか確認する。
3. 再現すれば、`performance_kfold`を固定したまま学習側OOFだけで小規模なモデルパラメータ調整へ進む。現在は強noise FPRがほぼ0なので、FPRをさらに下げることではなく、全域RMSEを維持しながらRecallとONB近傍誤差を回復することを目的にする。
4. `matched`と0.5秒は別要因なので同時に変更しない。`matched`は既知noise適応の副比較として、統合clean_onlyの再現性を確定した後に判断する。

## 研究上の判断

統合学習は、学部時代に近いnoise trendを再現し、全域回帰と低熱流束誤報を改善する候補として有望である。ただし現在の修士研究上の目的が「ONBを早く、見逃さず検知すること」を含むなら、現時点で最終方式とはしない。今回の結果は、主問題を「強noiseの過大予測」だけでなく「誤報を抑えたときにONB近傍の過小予測と見逃しが増えるトレードオフ」として具体化した。

## 出力

- [metrics_by_day.csv](metrics_by_day.csv): 日別確定ONBによる全指標
- [two_day_day_specific_onb_metrics.csv](two_day_day_specific_onb_metrics.csv): 2日合算・日別ONBの対応比較
- [pooled_minus_single.csv](pooled_minus_single.csv): 同じ270 chunk上の統合−単日差
- [aggregate_by_day.csv](aggregate_by_day.csv): clean、noise平均、強noise、−20 dB集約
- [overall_average_onb_metrics.csv](overall_average_onb_metrics.csv): run保存値と同じ平均ONBでの再集計
- [ensemble_comparison.csv](ensemble_comparison.csv): performance、等重み、事後最良単体
- [weights.csv](weights.csv): 統合・単日のperformance重み
- [residual_correlation.csv](residual_correlation.csv): 単体残差相関
- [wav_changes.csv](wav_changes.csv): WAV／熱流束別の統合−単日差
- [q100.csv](q100.csv): 日別ONBによるq100/g100
- [split_audit.csv](split_audit.csv): 270/270対応とラベル一致
- [completion_audit.csv](completion_audit.csv): 完了性、fit ID、分割件数
- [analysis_manifest.json](analysis_manifest.json): 対象runと条件
- [analyze.py](analyze.py): 再計算スクリプト

## 限界

- 学習seed 42の1 runだけである。
- `within_wav_chunk`なので、同一WAVの別chunk・録音条件・熱流束ラベルを学習側とテスト側で共有する。未知WAV・未知実験日性能ではない。
- 1秒chunkを独立した実験反復とは数えない。
- 最良単体は外側テストを見た事後記述であり、採用規則ではない。
- 2日の平均ONBは運用上の保存値であり、物理的な日別ONBを置き換えない。
