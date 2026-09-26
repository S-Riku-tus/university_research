# clean_only・matched完成runの比較解析

> **2026-09-26更新**：本書はseed 42完了時点の解析である。その後seed 43・44も完了したため、結論と次工程は[3 seed比較](../2026-09-26_clean_matched_performance_kfold_3seed/README.md)を優先する。

## 結論

2026年9月25日夜から26日未明に完了した22 kHz・選別なし・150 epochs・seed 42の3系列を確認した。同じ6/11学習の比較では、無雑音の予測は`matched`と`clean_only`で全モデル・両統合とも最大絶対差0であり、雑音学習方針だけを変えた比較として成立している。

確認できた主結果は次の三点である。

1. `matched`は雑音あり入力に対する深層モデルと統合の全域回帰性能を大きく改善した。ただしRFは`clean_only`でも−16 dBまではほぼ不変で、未知noiseへの頑健な対照である。
2. `performance_kfold`はmatchedの7条件中4条件で全域RMSE最良となったが、noiseあり6条件平均でRFに対する改善は0.87 kW/m²（0.9%）に留まった。ONB近傍RMSEは7条件すべてで最良単体を上回れなかった。
3. clean学習へ7/9を追加すると、6/18の全域性能は全モデルで悪化した。学習日追加を一般化改善とは扱えず、日別の熱流束分布・音響―熱流束対応の差を先に診断する必要がある。

## 対象runと比較可能性

| 系列 | 学習データ | 評価データ | 学習状態 |
|---|---|---|---|
| matched 6/11 | 6/11の各評価noiseと同じnoise | 6/18のclean＋6 SNR | noiseごとに7回独立fit |
| clean 6/11 | 6/11 clean | 6/18のclean＋6 SNR | 1回のfitを7条件へ固定適用 |
| clean 6/11＋7/9 | 6/11＋7/9 clean | 6/18のclean＋6 SNR | 1回のfitを7条件へ固定適用 |

3系列はいずれも22 kHz、ピーク選別なし、150 epochs、seed 42、`performance_kfold`＋等重み、6/18全1080秒の別日評価である。内部重み推定は3-foldの元WAV分離で、全foldの共有WAV数は0だった。内部OOF標本数は6/11学習で1080、6/11＋7/9学習で1860である。外側評価は1日だけなので、保存CSVの標準誤差0は不確かさ0を意味しない。説明性は無効であり、今回は性能・重み・予測だけを解析した。

## 同じ6/11学習におけるmatchedとclean_only

### noiseあり6条件平均

単位はRMSEがkW/m²。R²とRMSEは6条件の単純平均である。

| モデル | matched R² | matched RMSE | clean_only R² | clean_only RMSE | matchedによるRMSE改善 |
|---|---:|---:|---:|---:|---:|
| RandomForest | .8758 | 95.96 | .8596 | 101.14 | 5.18 |
| Conformer | .8551 | 103.46 | .3464 | 210.37 | 106.91 |
| AlexNet | .8167 | 116.33 | .6520 | 158.68 | 42.35 |
| performance_kfold | .8778 | 95.09 | .7007 | 145.26 | 50.18 |
| simple_equal | .8751 | 96.15 | .7331 | 137.10 | 40.94 |

matchedでは`performance_kfold`がRFより平均RMSEを0.87 kW/m²改善したが差は小さい。条件別ではSNR 0、−4、−12、−16 dBの4/7条件で全モデル中最良、cleanと−20 dBはConformer、−8 dBはRFが最良だった。等重みはnoiseあり平均96.15 kW/m²でRFの95.96 kW/m²を僅かに下回った。

clean_onlyではRFがnoiseあり6条件すべてで全域RMSE最良だった。RFのR²はcleanから−16 dBまで.8764〜.8800とほぼ一定で、−20 dBで.7677へ低下した。これに対しConformerはclean .9238から−20 dB −.2838、AlexNetは.8462から.4559へ低下した。固定cleanモデルへ雑音を加えると深層予測が上方へ移動し、−20 dBではperformance・等重みともONB前480秒を全てONB以上と判定した。両統合のprecision=.5556、recall=1.0、accuracy=.5556であり、recall上昇は検知改善ではなく全陽性化である。

### ONB近傍とのトレードオフ

matchedのnoiseあり平均ONB近傍RMSEは、Conformer 82.14、simple equal 118.60、performance 123.29、AlexNet 133.21、RF 152.79 kW/m²だった。`performance_kfold`は全域RMSEで4/7条件最良だった一方、ONB近傍RMSEでは7/7条件で最良単体を上回れなかった。

clean_onlyのONB近傍RMSEはAlexNet 37.04、等重み49.87、performance 51.11 kW/m²と数値上小さい。しかし、noise増加に伴う大きな正biasで予測がONB近傍へ偶然近づいた条件を含む。例えば−20 dBのperformanceはONB近傍bias +111.1 kW/m²、ONB前bias +265.4 kW/m²で、ONB前の誤報率は1.0だった。このためONB近傍RMSEだけでclean_onlyを優位と判断できない。

## performance_kfoldの重みと評価日順位

6/11 clean内部OOFの`1 - R²`はRF .1024、Conformer .0735、AlexNet .0676で、重みはRF .256、Conformer .357、AlexNet .387となった。clean評価日の全域RMSE最良はConformerであり、noiseあり6条件はすべてRFが最良だったため、この固定重みは未知noiseで頑健なRFを過小評価した。

matchedでは重みをnoiseごとに再計算し、noiseありのRF重みは.351〜.478だった。最大重みモデルと6/18の全域RMSE最良単体は7条件中5条件で一致した。その結果、全域では統合が比較的安定した。一方、ONB近傍の最良単体と最大重みモデルの一致は1/7条件だけである。これは全chunkのpooled R²から作る重みが全域誤差には対応しても、ONB近傍を保護しないことと整合する。

## 7/9をclean学習へ追加した影響

### noiseあり6条件平均

| モデル | 6/11 clean RMSE | 6/11＋7/9 clean RMSE | 悪化量 | ONB近傍RMSEの変化 |
|---|---:|---:|---:|---:|
| RandomForest | 101.14 | 133.75 | +32.61 | −59.11 |
| Conformer | 210.37 | 280.32 | +69.95 | +51.50 |
| AlexNet | 158.68 | 240.34 | +81.66 | +120.76 |
| performance_kfold | 145.26 | 251.22 | +105.95 | +105.09 |
| simple_equal | 137.10 | 210.54 | +73.44 | +64.82 |

RFのONB近傍RMSEだけは改善したが、全域RMSEは悪化し、−20 dBではRFを含む全モデル・両統合が全1080秒をONB陽性と判定した。したがって、RFのONB近傍改善を単独で採用根拠にはできない。

7/9は780秒・13熱流束点で、平均421.6、中央値442.2 kW/m²である。6/11は1080秒・18点、平均340.0、中央値291.2 kW/m²で、7/9は低～中熱流束点が少なく高熱流束側へ偏る。また登録ONBは6/11の221.5に対して7/9は571.7 kW/m²である。これは確認事実であり、追加日により共有回帰関数の校正が変わった原因候補になるが、性能悪化の因果はまだ確定できない。

内部OOFでは6/11＋7/9のRF `1 - R²`が.4025、Conformer .0523、AlexNet .0900となり、重みはRF .076、Conformer .584、AlexNet .340になった。しかし6/18の全域RMSEは7条件すべてRFが最良だった。学習日内部の順位が評価日へ移らず、重み推定がRFを大きく過小評価したことがperformance統合悪化の直接的な機構である。

## 解釈と次の比較

### 確認できたこと

- 既知のnoise条件と同じnoiseで学習できる設定では、matchedがdeep modelの全域性能低下を大きく抑える。
- 未知noiseへclean固定モデルを直接適用する設定では、RFが−16 dBまで非常に安定し、deep modelと両統合は正bias・誤報を増やす。
- 現行performance重みは全域pooled R²用であり、ONB近傍最適化ではない。
- 学習日を増やすだけでは一般化しない。今回の7/9追加ではむしろ日間校正差が強く現れた。

### まだ断定できないこと

- seed 42だけなので、matchedにおける0.87 kW/m²の平均改善が初期値を越えて再現するかは未確認である。
- 6/18は既に方式判断へ繰り返し使用しており、完全な独立最終テストとはいえない。
- matchedは評価noiseと同じ生成条件で再学習する比較であり、未知noiseへの頑健性を示さない。
- 7/9追加悪化がラベル分布、日別音響特性、装置状態、ONB差のどれに主に由来するかは未識別である。

### 推奨する順序

1. まず同一の6/11→6/18条件でseed 43・44をmatchedとclean_onlyの両方に追加し、matchedの小さい統合改善とRFのclean頑健性が再現するか確認する。
2. 主評価を全域RMSE/R²だけにせず、ONB前誤報率、recall、ONB近傍RMSEを併記する。ONB近傍RMSE単独の改善は採用条件にしない。
3. `performance_kfold`について、内部OOF順位と6/18順位の対応をseed別・noise別に固定記録する。全域用重みとONB用途を同じ結論にしない。
4. 7/9追加系列の追加seed実行は一旦後順位とし、日ごとのラベル分布を揃えた比較、日別bias、日を丸ごと外す内部診断のどれで悪化が説明できるか先に確認する。

## 根拠run

- matched 6/11: `Pool_boiling/.../2025.06.18_0.3_3/regression_result/npy/ensemble/202609/25/onb_xd-t0611-v0618_iw3-nm_s0_e150_170800/`
- clean 6/11: `Pool_boiling/.../2025.06.18_0.3_3/regression_result/npy/ensemble/202609/26/onb_xd-t0611-v0618_iw3-nc_s0_e150_000904/`
- clean 6/11＋7/9: `Pool_boiling/.../2025.06.18_0.3_3/regression_result/npy/ensemble/202609/26/onb_xd-t0611+0709-v0618_iw3-nc_s0_e150_004249/`

解析では各runの`run_manifest.json`、`completed.json`、`metrics_summary_*.csv`、`ensemble_weights_*.csv`、`internal_validation_fold1.json`、`fold_pred/pred_f1_*.csv`を使用した。再学習・予測再計算・既存結果の変更は行っていない。
