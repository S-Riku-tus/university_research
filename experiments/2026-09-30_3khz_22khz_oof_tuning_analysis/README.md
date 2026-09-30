# 3 kHz・22 kHz 学習側OOFパラメータ探索の解析

解析日: 2026-09-30。

## 結論

両探索は、6/11＋6/18統合、1秒、clean_only、within-WAV chunk holdout、seed 42、3-fold、150 epoch、選別なしで一致し、各190候補が完了した。候補選択に外側テスト540 chunkは使われていない。

主方式`performance_kfold`へ採用する推奨値は次のとおり。

| 周波数 | RandomForest | Conformer | AlexNet |
|---|---|---|---|
| 3 kHz | `n_estimators=100, max_depth=12, subsample=0.6, colsample_bynode=0.6` | `lr=0.001, batch_size=12` | `lr=0.003, batch_size=8` |
| 22 kHz | `n_estimators=600, max_depth=6, subsample=0.6, colsample_bynode=0.6` | `lr=0.0003, batch_size=8` | `lr=0.01, batch_size=24` |

22 kHzは3モデルとも有望で、上記推奨値の保存OOF予測を`performance_kfold`の式で統合すると、3 kHzに対し全域RMSE 65.29→53.50 kW/m²、ONB近傍RMSE 68.35→65.51 kW/m²、Recall .859→.866、FPR .0015→.0015となった。元WAV単位のpaired bootstrapでも全域差は−11.79 kW/m²、95% CI [−20.58, −2.48]、22 kHz改善確率.994だった。一方ONB差のCIは[−23.77, 19.52]で0をまたぐ。したがって、学習側だけの選択では全域回帰を根拠に22 kHzを次の外側評価対象とするが、ONB改善は未確定とする。これはclean OOFで選んだ結果であり、付加水流noiseへの最終性能ではない。

## 完了性と比較条件

| 項目 | 3 kHz | 22 kHz |
|---|---:|---:|
| 候補数 | 190 | 190 |
| RF / Conformer / AlexNet | 135 / 25 / 30 | 135 / 25 / 30 |
| 日別指標行 | 380 | 380 |
| 外側学習 | 1,620 chunk | 1,620 chunk |
| 未使用外側テスト | 540 chunk | 540 chunk |
| 選別 | なし | なし |
| 完了印 | あり | あり |

候補の全域RMSE最小値から2%以内を、ONB指標を確認する範囲とした。該当数は3 kHzがRF 35、Conformer 4、AlexNet 4、22 kHzがRF 52、Conformer 2、AlexNet 1だった。

## 単体モデルの結果と選定

### 3 kHz

| モデル | 候補 | 全域RMSE | ONB近傍RMSE | Recall | FPR | 判断 |
|---|---|---:|---:|---:|---:|---|
| RF | 100木・深さ12・0.6・0.6 | 91.03 | 154.41 | .858 | .004 | 全域最小。ONB差が候補間で小さいため採用 |
| Conformer | `1e-4`, batch 8 | **65.01** | 63.02 | .869 | .025 | 全域最小だがONB・FPRに弱い |
| Conformer | `1e-3`, batch 12 | 65.41 | **52.44** | **.877** | **.009** | 全域+0.62%でONB・Recall・FPRが改善するため採用 |
| AlexNet | `3e-3`, batch 8 | **71.73** | 62.20 | **.853** | .021 | 主アンサンブルでは採用 |
| AlexNet | `1e-3`, batch 24 | 72.06 | **55.77** | .840 | **.012** | AlexNet単体のONB重視候補。ただし統合全域RMSEを押し下げた |

3 kHzで各モデルの全域最小だけを統合すると、performance OOFは全域64.45、ONB 73.19 kW/m²だった。ConformerだけをONB均衡候補へ替えると、全域65.29（+1.30%）を保ちながらONB 68.35へ改善し、Recall .860→.859、FPRは.0015で不変だった。AlexNetもONB候補へ替えると全域66.67（+3.44%）となり、全域2%条件を外れる。そのため主アンサンブルはConformerだけを均衡候補へ替える。

### 22 kHz

| モデル | 採用候補 | 全域RMSE | ONB近傍RMSE | Recall | FPR | 判断 |
|---|---|---:|---:|---:|---:|---|
| RF | 600木・深さ6・0.6・0.6 | 89.88 | 145.71 | .857 | .003 | 全域最小。RF候補は広いplateau |
| Conformer | `3e-4`, batch 8 | 56.81 | 62.84 | .897 | .003 | 全域最小。2%内のbatch 12より全体として良い |
| AlexNet | `1e-2`, batch 24 | 53.27 | 62.67 | .867 | .001 | 2%内に残った唯一の候補 |

22 kHzでは単体全域最小と均衡判断が一致する。Conformerのbatch 8はbatch 12に対して、全域、poolしたONB、Recall、FPRで優位だった。AlexNetの最良値は次点56.99 kW/m²より6.5%低く、今回のgrid内では比較的明確だった。

## 周波数比較の読み方

確認した事実:

- 推奨`performance_kfold` OOFは、22 kHzが3 kHzより全域RMSEで11.79 kW/m²（18.1%）低い。
- 36 WAV中27 WAVで22 kHzの全域RMSEが低かった。WAV cluster bootstrapの全域差95% CIも0をまたがなかった。ただし、これは評価WAVの不確かさであり学習seedの不確かさではない。
- ONB近傍の点推定は22 kHzが2.84 kW/m²低く、Recallは0.0063高く、FPRは同じだった。しかしONB帯は日別1 WAVずつの計2 WAVで、22 kHzは6/11を改善、6/18を悪化させた。cluster bootstrapの差95% CIは[−23.77, 19.52]で、ONB優位は確認できない。
- 22 kHzの推定重みはRF .158、Conformer .394、AlexNet .448。3 kHzはRF .220、Conformer .426、AlexNet .354で、22 kHzではAlexNetの寄与が大きい。
- 22 kHz ConformerのONB近傍RMSEは日別26.60 / 84.80 kW/m²で、pool値62.84だけでは見えない日差がある。3 kHz採用Conformerは51.36 / 53.50で安定していた。

整合する解釈:

- 現行の224×224入力・log-powerモデルでは、3 kHzより上の情報が深層2モデルのclean全域回帰に有用だった可能性がある。
- 一方、22 kHz Conformerの日別ONB差から、広帯域化がONB近傍を一様に改善したとはまだ言えない。

この解析だけでは、22 kHzが付加水流noiseにも強いこと、追加帯域が気泡由来であること、別seedでも順位が不変であることは確認していない。

## 次の一手

1. 22 kHzの推奨値は主コードの各リスト1要素へ固定済み。同じ外側540 chunk・7 noiseを通常runで一度評価する。
2. `performance_kfold`と等重み、3単体について、全域・日別ONB近傍・Recall・FPR・q100・noise trendを読む。
3. この外側結果を見てパラメータを再選択しない。期待と異なった場合も汎化差として記録する。
4. 5/10/15 kHzの追加探索は、22 kHz外側結果で「帯域上限の途中を調べる必要」が生じた場合に行う。現時点で先に4周波数へ広げない。

## 保存物

- `analyze.py`: 長いWindowsパスを含む再集計スクリプト
- `completion_audit.csv`: 完了性・候補数・外側テスト未使用の監査
- `eligible_candidates.csv`: 全域最小から2%以内の98候補と日別最悪値
- `recommended_candidates.csv`: 単体全域最小と均衡推奨値
- `recommended_ensemble_oof.csv`: 推奨候補を保存OOF予測で統合した診断
- `wav_frequency_comparison.csv`: 対応36 WAVの3/22 kHz誤差差
- `cluster_bootstrap_frequency_delta.csv`: WAV cluster単位のpaired差5000反復

`recommended_ensemble_oof.csv`は候補探索を新たに全直積したものではなく、単体基準で事前に絞った厳密最小・均衡候補の組合せを診断したもの。外側テスト性能ではない。
