# 22 kHz・OOF採用パラメータの外側540 chunk評価

## 結論

22 kHzの学習側OOF探索で固定したパラメータは、cleanの既知WAV内未使用chunkでは良好に移送した。`performance_kfold`の全域RMSEは29.97 kW/m²で、等重み34.54、Conformer 32.30、AlexNet 39.11、RF 72.95 kW/m²を下回った。一方、未学習の付加水流noiseに対しては低熱流束の大きな過大予測が生じ、SNR 0 dBですでにFPR 46.67%、−8 dB以下ではFPR 100%になった。したがって、今回の結果は「22 kHzのclean回帰性能」を支持するが、「clean OOFで決めた`performance_kfold`がnoise劣化を防ぐ」という主張は支持しない。

## 対象と完了性

- 対象run: `Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2_6.18_0.3_3/regression_result/npy/ensemble/202609/30/onb_wc-t0611-v0611_iw3-nc_c1s_s0_e150_190945`
- 1秒、22 kHz、6/11＋6/18統合、`within_wav_chunk`、seed 42、150 epoch、`clean_only`、音響選別なし。
- 外側学習1620 chunk、外側評価540 chunk。評価は6/11・6/18各270 chunk、36 WAVすべてから未使用chunkを含む。
- clean、0、−4、−8、−12、−16、−20 dBの7/7条件が完了した。全条件で評価chunk・順序・正解値が一致し、同じ`fit_id=48decba34818`を共有する。
- 固定値はRF `n_estimators=600, max_depth=6, subsample=0.6, colsample_bynode=0.6`、Conformer `lr=0.0003, batch=8`、AlexNet `lr=0.01, batch=24`。
- `performance_kfold`重みは全noise共通で、RF .1575、Conformer .3942、AlexNet .4483。外側評価値による再選択・重み更新はしていない。

## 評価上の注意

run内の通常の`metrics_summary`は、統合フォルダの平均ONB値246.591 kW/m²を全標本へ使う。この記録では、出典日の物理的ONBを保つため、6/11は221.505、6/18は271.678 kW/m²として、ONB近傍、FPR、Recall、q100を保存予測から再計算した。全域RMSE・MAE・R²は閾値に依存しないのでrun内の値と一致する。

## 全域性能とnoise trend

出典日別ONB値で集計した`performance_kfold`は次のとおり。

| 評価入力 | 全域RMSE | ONB近傍RMSE | ONB前bias | FPR | Recall |
|---|---:|---:|---:|---:|---:|
| clean | 29.97 | 36.69 | −8.55 | 0.44% | 89.21% |
| 0 dB | 119.15 | 34.97 | +162.39 | 46.67% | 98.10% |
| −4 dB | 133.07 | 47.97 | +187.23 | 88.00% | 99.37% |
| −8 dB | 164.85 | 89.59 | +237.51 | 100% | 100% |
| −12 dB | 207.97 | 144.75 | +302.23 | 100% | 100% |
| −16 dB | 240.79 | 188.21 | +349.56 | 100% | 100% |
| −20 dB | 271.47 | 228.53 | +392.26 | 100% | 100% |

単位はRMSEとbiasがkW/m²。noiseを強くするほどONB前biasが正方向へ単調に増え、−8 dB以下ではONB前225 chunkをすべて沸騰と誤判定した。Recall 100%はONB検知が健全になったためではなく、ほぼ何でも沸騰と判定する側へ崩れたためである。

noiseあり6条件の平均RMSEはRF 111.36、等重み170.38、performance 189.55、AlexNet 193.31、Conformer 234.35 kW/m²だった。RFは0 dB 72.88、−4 dB 74.16 kW/m²と低noise側でほぼ維持したが、performanceは0 dB 119.15、−4 dB 133.07まで悪化した。clean OOFで高く評価された深層2モデルへ合計84.25%を配分したため、noise時の正biasを強く受けたと解釈できる。

## cleanでのアンサンブル効果

cleanではperformanceの全域RMSE 29.97 kW/m²が5方式中最小だった。WAVをclusterとして再標本化したRMSE差 `performance − 比較方式` の95%区間は、等重み比−4.56 [−7.76, −1.67]、AlexNet比−9.13 [−14.47, −3.71]、RF比−42.98 [−57.26, −28.55] kW/m²だった。Conformer比は−2.33 [−5.95, 1.65]で区間が0をまたいだ。

noiseでは逆転し、performanceは等重みより全6条件で15.28～23.39 kW/m²悪く、全条件のWAV cluster bootstrap区間も0より大きかった。RF比では0 dBで+46.27 [13.57, 73.96]、−20 dBで+110.83 [93.06, 126.44] kW/m²だった。ただしperformanceは、noiseに弱いConformerとAlexNetの各単体よりは概ね低いRMSEに抑えている。つまりアンサンブルは深層単体の崩壊を緩和したが、RFより頑健にはならなかった。

## 日別ONB指標

cleanのperformanceは、6/11で全域RMSE 32.26、ONB近傍41.78 kW/m²、FPR 0%、Recall 87.88%、6/18で27.49、30.75 kW/m²、FPR 0.83%、Recall 90.67%だった。clean時点では両日に大きな誤報はなく、全域性能も良好である。

0 dBでは6/11のONB近傍RMSEが22.59 kW/m²へ一見改善したが、FPRは41.90%だった。6/18もONB近傍43.99 kW/m²、FPR 50.83%である。−4 dBのFPRは6/11 97.14%、6/18 80.00%、−8 dB以下は両日100%だった。ONB近傍だけを見ると、ONB前誤報の崩壊を見落とす。

## q100の意味

cleanのperformanceは、6/11でq100 368.978 kW/m²、`q100 − ONB`は+147.473 kW/m²、6/18で376.320、+104.642 kW/m²だった。ところが0 dBの6/11では差が+45.403 kW/m²へ縮んだ一方、すでにONB前105 chunk中44 chunkを誤報した。−8 dB以下では全入力を早期から沸騰と判定するためq100が0となり、差は負になる。

したがって、q100の小ささだけを「ONBを早く予測できた」と評価してはいけない。q100はFPRと必ず対で示す。今回の`q100=0`は最良結果ではなく、常時陽性化した失敗を表す。修論では、`q100 − ONB`とONB前FPRを2指標として併記するのが妥当である。

## OOF結果との接続

学習側WAV非共有OOFでは、同じ22 kHz performanceの全域RMSEが53.50、ONB近傍65.51 kW/m²だった。今回の外側cleanは29.97、36.69 kW/m²で良い。ただし、OOFは学習対象WAVをfoldごとに未知とするのに対し、今回の外側評価は各WAVの別chunkを学習に含む既知WAV内評価であり、難易度が違う。差をチューニングの過学習や改善量として直接解釈しない。

同じ外側splitの既存3 kHz・選別なしrunは、旧パラメータではあるが、performanceのRMSEがclean 35.33、0 dB 53.41、−4 dB 62.20、−20 dB 88.20 kW/m²で、FPRは全条件0.4%以下だった。今回の22 kHzはcleanを29.97まで改善した一方、0 dB 119.15、−4 dB 133.07、−20 dB 271.47 kW/m²、FPR 46.67～100%である。これは22 kHzがclean精度とnoise耐性を交換している可能性を強く示す。ただし3 kHz側は今回のOOF採用パラメータではないため、この差を周波数上限だけの因果効果とはまだ断定しない。

## 判断と次の識別比較

1. 22 kHz採用パラメータを外側結果で選び直さない。今回のclean外側結果は、OOFで固定した候補が既知WAV内cleanへ移送した確認として保持する。
2. 現在の主問題はパラメータ不足より、`clean_only`で学んだ深層モデルが付加水流noiseを熱流束増加として読むことと、clean OOF重みがその2モデルを84.25%含むことである。
3. 次は3 kHzのOOF採用値（RF `100/深さ12/0.6/0.6`、Conformer `0.001/12`、AlexNet `0.003/8`）を各1要素へ固定し、同じ外側540 chunk・7 noise・clean_onlyを一度実行する。これにより、旧3 kHzとのパラメータ差を除いた上で、22 kHzのclean利得とnoise崩壊が周波数方向として残るかを確認する。
4. 3 kHz採用値でも同じ崩壊が起きれば、今回のチューニングがclean性能へ寄りすぎた可能性を優先する。3 kHzが従来どおり頑健なら、主入力を3 kHz、22 kHzをclean精度は高いがnoiseに弱い対照として整理できる。
5. 周波数比較後、noise学習を改善策として扱う場合だけ、採用周波数・split・seed・パラメータを固定した`matched`を行う。matchedは未知noise耐性の評価条件を変えるため、周波数比較と同時に変えない。

## 生成物

- `completion_audit.csv`: 7条件の完了性、split、fit共有、標本一致
- `metrics_source_day_thresholds.csv`: 日別ONB値で再計算した全域・ONB・分類指標
- `aggregate_two_day.csv`: モデル別clean・noise・強noise集約
- `ensemble_comparison.csv`: performance、等重み、最良単体の条件別比較
- `q100_source_day_thresholds.csv`: 日別q100とONBとの差
- `weights.csv`: 全noiseの保存重み
- `wav_metrics.csv`: 元WAV単位のRMSEとbias
- `cluster_bootstrap_ensemble_comparison.csv`: 元WAV cluster bootstrapによるRMSE差
- `analyze.py`: 上記の再計算スクリプト
