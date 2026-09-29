# 7/9単日within_wav_chunk解析

記録日: 2026-09-29

## 今回答える問い

7/9を1秒・3 kHz・`within_wav_chunk`・`clean_only`で実行したとき、6/11・6/18と同様のnoise trendが得られるか、7/9固有の571.7～720.7 kW/m²の予測変化が同日学習でも残るかを確認した。

## 完了性

- run: `onb_wc-t0709-v0709_iw3-nc_s0_e150_165729`
- 7/7条件完了、run hash `e09ae23f`、全noiseでfit ID `534b425f091a`を共有
- 13 WAV、各WAV 45学習／15テスト、外側学習585／テスト195 chunk
- 1秒、3 kHz、選別なし、150 epochs、seed 42、内部WAV GroupKFold 3-fold、`clean_only`、XAIなし

## 結論

1. 7/9は6月と同じ単調なnoise trendを再現しなかった。performance RMSEはclean 103.29から−12 dB 239.66 kW/m²まで悪化した後、−16で229.72、−20で219.06へ見かけ上回復した。
2. この回復はnoise耐性向上ではなくbias相殺である。performance biasは−12で−129.43、−16で−108.93、−20で−90.75 kW/m²となり、強noiseで過小予測が弱まったためRMSEが下がった。
3. cleanでもperformanceは確定ONB 571.694 kW/m²を−145.81、643.517 kW/m²を−194.56 kW/m²過小予測した一方、720.691 kW/m²では+22.73 kW/m²となった。以前の別日評価で確認した「予測と2.1–2.5 kHzピークが720.7 kW/m²で同時に変わる」構造が、同じ7/9のWAV内chunk学習でも残った。
4. performanceのq100は全7 noiseで720.691 kW/m²、確定ONBとの差は+148.997 kW/m²だった。7/9ではモデルとnoise条件によらず、100%ONB以上と判定できる地点が720.7 kW/m²まで遅れる。
5. performanceは等重みを1/7、事後的な最良単体を1/7条件でしか上回らなかった。7/9のnoise条件では現行performance重みの利点は確認できない。
6. 7/9は6月2日と異なる音響―熱流束対応を持つ対照事例として残し、6/11＋6/18の主学習やパラメータ選択へ混ぜない。

## noise trend

| noise | performance RMSE | R² | bias | FPR | Recall |
|---|---:|---:|---:|---:|---:|
| clean | 103.29 | .873 | −16.40 | 0 | .667 |
| 0 | 158.54 | .701 | −67.41 | 0 | .627 |
| −4 | 191.95 | .562 | −93.92 | 0 | .613 |
| −8 | 219.83 | .425 | −117.32 | 0 | .613 |
| −12 | 239.66 | .317 | −129.43 | 0 | .613 |
| −16 | 229.72 | .373 | −108.93 | 0 | .600 |
| −20 | 219.06 | .430 | −90.75 | 0 | .600 |

RMSEとbiasはkW/m²。FPRが常に0なのは良好な早期検知を意味しない。予測が全体に低く、cleanからONB以上の見逃しが多いためである。

## モデル別集約

| モデル | clean RMSE | noise平均 | 強noise平均 | −20 dB |
|---|---:|---:|---:|---:|
| RF | 187.45 | 204.37 | 218.15 | 221.17 |
| Conformer | 87.48 | 239.65 | 263.25 | 270.35 |
| AlexNet | 108.35 | 231.75 | 270.27 | 256.31 |
| performance | 103.29 | 209.79 | 229.48 | 219.06 |
| equal | 111.48 | 203.52 | 220.55 | 208.10 |

単位はkW/m²。cleanはConformerが最良だが、−8～−16 dBはRFが最良となった。clean学習側OOFから求めたperformance重み（RF .251、Conformer .409、AlexNet .340）は強noiseの外側順位を予測できず、noise平均では等重みより6.27 kW/m²、強noiseでは8.93 kW/m²悪かった。

## 571.7～720.7 kW/m²

performanceのWAV別biasは次のとおりだった。

| 熱流束 | clean | −12 dB | −20 dB |
|---:|---:|---:|---:|
| 505.101 | −99.09 | −371.53 | −330.72 |
| 571.694（確定ONB） | −145.81 | −404.52 | −362.85 |
| 643.517 | −194.56 | −420.10 | −402.32 |
| 720.691 | +22.73 | +20.86 | +14.12 |

単位はkW/m²。720.7 kW/m²を境に予測が急変し、noiseを加えても境界位置は変わらない。これは7/9の音響特徴が確定ONB 571.7 kW/m²ではなく720.7 kW/m²付近で大きく変わるという既存観察と整合する。ただし、物理的ONBが720.7だったと再定義する根拠にはせず、ラベル、音響応答、測定条件の不一致候補として扱う。

## 6月結果との関係

6/11＋6/18統合performanceはRMSEが35.33→53.41→62.20→67.12→70.77→76.10→88.20 kW/m²とcleanから−20 dBまで単調に悪化した。7/9は103.29→158.54→191.95→219.83→239.66→229.72→219.06であり、絶対誤差も曲線形状も異なる。

したがって「3日で同じ傾向が再現した」とはしない。7/9は主系列へ追加するデータではなく、日によって音響―熱流束関係が異なることを示す補助的な対照である。本人の方針どおり、今後の主比較は6/11＋6/18に限定できる。

## 研究上の扱い

- 修論の主結果: 6/11＋6/18、既知WAV内の未使用chunk、clean学習から付加水流noiseへの耐性
- 7/9: 720.7 kW/m²で音響・予測が切り替わる異なる実験系列の補助結果
- 7/9を主学習、チューニング、6月poolへ混ぜない
- 7/9のONB定義と音響変化の関係は、必要性が生じたときに別課題として扱う

## 出力

- [metrics.csv](metrics.csv): 全noise・全モデルの再計算指標
- [aggregate.csv](aggregate.csv): clean、noise平均、強noise、−20 dB集約
- [wav_metrics.csv](wav_metrics.csv): 熱流束別RMSE・bias
- [q100.csv](q100.csv): q100/g100
- [ensemble_comparison.csv](ensemble_comparison.csv): performance、等重み、事後最良単体
- [weights.csv](weights.csv): performance重み
- [residual_correlation.csv](residual_correlation.csv): 単体残差相関
- [completion_audit.csv](completion_audit.csv): 完了性とfit共有
- [analysis_manifest.json](analysis_manifest.json): 対象runと条件
- [analyze.py](analyze.py): 再計算スクリプト

## 限界

- seed 42の1 runである。
- `within_wav_chunk`なので未知WAV・未知日性能ではない。
- ONB正解と音響状態の関係は同期観察なしに確定できない。
- 13 WAV、各熱流束15テストchunkであり、chunkを独立した実験反復とは数えない。
