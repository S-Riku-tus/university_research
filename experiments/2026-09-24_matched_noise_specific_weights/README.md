# matched学習・ノイズ別重みの確認と次回条件

更新日: 2026-09-25。状態: **7 SNR×3 seed本比較・完了性監査・clean-only対応比較・重み移送診断まで完了**。

## 本人の指定

- 重みを研究全体・将来入力に共通する定数として固定しない。
- 別の実験データを用いる場合も、学習実験日を変更すれば、その学習データから重みを再計算できる形を保つ。
- `clean_only`は従来どおりcleanでモデルと重みを一度fitし、複数の評価noiseへ共有する。
- `matched`は各noiseの学習データでモデル、PCA、scaler、epoch、アンサンブル重みを独立にfitする。

## コード監査

現行`build_learning_families()`は、`matched`のfamily keyへ評価noiseを含める。このため各noiseが別familyとなり、`learning_runner.py`内の`fit_inner_holdout_errors()`および最終モデル学習はnoiseごとに再実行される。`clean_only`だけが全評価noiseを一つのclean familyへまとめる。

今回、挙動自体は変更せず、manifestの`learning_context.ensemble_weight_scope`へ次を保存し、意図しない将来の共有を例外とテストで防ぐようにした。

- `matched`: `per_training_noise`
- `clean_only`: `shared_clean_across_evaluation_noises`

## 保存済みmatched runの実証

2026-09-19の6/11学習→6/18評価、−8/−16 dB、seed 42〜44を再確認した。[監査表](saved_matched_weight_audit.csv)。各seedで−8と−16の`fit_id`は異なり、重みも異なる。

例としてseed 42では、

- −8 dB: RF `.1959`、Conformer `.5329`、AlexNet `.2712`
- −16 dB: RF `.2508`、Conformer `.4424`、AlexNet `.3067`

である。従って「matchedでも全noiseへ同じ重みを使っていた」という問題は保存結果には存在しない。

## 次回の本比較

設定は[`2026-09-24_matched_noise_specific_weights.json`](../../configs/experiments/2026-09-24_matched_noise_specific_weights.json)。完了済みclean-only本比較と揃え、音響選別なし、22 kHz、6/11学習→6/18評価、7 SNR、200 epochs、seed 42〜44とする。

`data.experiment_names`は`null`とし、対象日は`train_experiments`と`test_experiments`の和集合から自動決定する。別の実験日へ切り替えるときは、この学習日・評価日指定を変更すれば、新しい学習データからnoise別重みを再計算する。

各SNRについて、

1. 対応するnoiseの6/11データだけでepochを選ぶ。
2. 対応するnoiseの6/11元WAV非共有holdoutから`inner_holdout`重みを求める。
3. 同じnoiseの6/18へ一度だけ適用する。
4. RF、Conformer、AlexNet、等重み、`inner_holdout`を比較する。

実行コマンド:

```powershell
python experiments/2026-09-19_b_clean_only/run_from_condition.py --condition configs/experiments/2026-09-24_matched_noise_specific_weights.json --seed 42
python experiments/2026-09-19_b_clean_only/run_from_condition.py --condition configs/experiments/2026-09-24_matched_noise_specific_weights.json --seed 43
python experiments/2026-09-19_b_clean_only/run_from_condition.py --condition configs/experiments/2026-09-24_matched_noise_specific_weights.json --seed 44
```

実行後は、各seed内で7 SNRの`fit_id`が互いに異なること、manifestの`training_noise_dir`が評価noiseと一致すること、`ensemble_weights_<SNR>.csv`がSNRごとに保存され各noiseの学習familyから推定されていることを先に監査する。別々に推定した結果が偶然同じ数値になる可能性はあるため、数値が必ず異なること自体は要件にしない。

## 9/25 本比較の完了性監査

実行先は次の3件で、seed 42、43、44に対応する。

- `matched_nois_xd-t0611-v0618_iw3-nm_s0_e200_173431`
- `matched_nois_xd-t0611-v0618_iw3-nm_s0_e200_191324`
- `matched_nois_xd-t0611-v0618_iw3-nm_s0_e200_205321`

[条件監査表](condition_audit.csv)と[run監査表](run_audit.csv)で次を確認した。

- 3 seedとも7/7条件が`completed.json`まで完了している。
- seed内の7条件は全て別`fit_id`で、モデル、PCA、scaler、epoch、重みがnoiseごとに再fitされている。
- 全21条件で`training_noise_dir == evaluation_noise_dir`、`ensemble_weight_scope == per_training_noise`である。
- epoch選択は`scope=training_days_only`、`test_used=false`で、6/18をepoch選択に使用していない。
- 音響選別なし、22 kHz、6/11学習→6/18評価、200 epoch上限という比較条件も一致する。
- 本runは設定どおり`save_fitted_artifacts=false`である。指標・予測・重みは保存済みだが、保存モデル再読込の検証対象ではない。

従って、今回の結果を「matchedなのに同じ重みを全noiseへ流用した結果」と解釈してはいけない。意図したnoise別再学習・noise別重みの結果である。

## 全域RMSEの結果

値は3 seed平均、単位はkW/m²。

| SNR | RF | Conformer | AlexNet | 等重み | inner holdout |
|---|---:|---:|---:|---:|---:|
| clean | 95.77 | 85.00 | 90.62 | **78.18** | 79.29 |
| 0 | **95.06** | 101.16 | 121.85 | 95.20 | 96.64 |
| −4 | 95.60 | 107.34 | 116.46 | **95.54** | 96.37 |
| −8 | **95.75** | 103.27 | 127.11 | 101.39 | 101.03 |
| −12 | **96.79** | 115.22 | 112.31 | 97.47 | 100.99 |
| −16 | 97.15 | 96.35 | 104.07 | 92.89 | **92.82** |
| −20 | 97.07 | 128.91 | 106.59 | 94.15 | **94.01** |

7 SNR×3 seed全体の平均は、RF 96.17、等重み93.54、inner 94.45 kW/m²で、RF比は等重み−2.73%、inner−1.79%であった。ただし、この差には統合が大きく勝つcleanが含まれる。noiseあり6条件×3 seedだけではRF 96.24、等重み96.11、inner 96.98 kW/m²で、等重みは−0.13%の実質同等、innerは+0.77%の悪化である。

seed別のnoiseあり平均でも、等重みのRF比は−1.19%、−2.47%、+3.25%、innerは+1.30%、−0.89%、+1.90%となり、改善はseed間で安定していない。RFのnoiseあり各SNRにおけるseed標準偏差が0.18〜0.33 kW/m²なのに対し、統合は2.35〜9.66 kW/m²で、深層モデル由来のばらつきが残る。

## 事前の暫定基準との照合

[暫定成功基準](../../docs/research_plan/2026-09-24_ensemble_research_position_and_next_steps.md)との対応は次のとおり。

| 判定項目 | 等重み | inner holdout |
|---|---:|---:|
| 7 SNR平均RMSEがRFより小さい | 達成 | 達成 |
| 最良単体の2%以内 | 11/21 | 11/21 |
| 暫定基準14/21以上 | 未達 | 未達 |
| RFに勝つ | 11/21 | 13/21 |
| 最良単体に勝つ | 7/21 | 9/21 |
| 5方式中の最良 | 4/21 | 5/21 |
| noiseあり18条件だけでRFに勝つ | 8/18 | 10/18 |
| 最悪SNR平均がRFより10%以上悪化しない | 達成（最悪+5.89%） | 達成（最悪+5.52%） |

二つの統合のどちらかを6/18の結果を見て事後選択すれば9/21条件で最良になるが、これは運用可能な一方式の成績ではないため、成功数として合算採用しない。現在言えるのは「全域平均と最悪時安全性には価値があるが、noiseあり条件で広く最良または実質同等という目標には未到達」である。

## clean-onlyとの対応比較

[clean-only本比較](../2026-09-24_clean_train_noise_inference/README.md)と同じseed・SNR・評価秒で照合した。clean条件の全モデルは完全一致し、条件の対応は正しい。noiseありではmatchedにより、等重みのRMSEは0 dBで5.35、−20 dBで106.25 kW/m²、innerは同16.12、136.61 kW/m²改善した。Conformerも30.78〜177.32 kW/m²改善しており、clean学習のまま未知noiseへ入れたことが以前の大幅悪化の主因だったことは明確になった。

ただし領域別には、noiseあり6 SNR×3 seed平均で、clean-onlyからmatchedへ変えるとinnerのONB前RMSEは121.62 kW/m²改善した一方、ONB近傍は57.39、ONB以降は32.05 kW/m²悪化した。平均予測は全領域で低い側へ移り、clean-onlyで多発したONB前誤報を減らした代わりにONB見逃しが増えた。従ってmatchedは単純に全領域を改善したのではなく、過大予測を抑えて全域RMSEを大きく直しながら、ONB近傍・以降では過小予測側の別の誤差を生んでいる。

一方、今回のmatched同士でRFと比べると、21条件平均でinnerはONB前RMSEが25.81 kW/m²悪いが、ONB近傍は34.70、ONB以降は24.46 kW/m²良い。熱流束回帰として高熱流束側を改善する余地は確認できたが、閾値判定ではONB近傍と以降を合わせてRFより平均13.0秒多く見逃す。RMSE改善とONB recall改善は同じではないため、今後も別指標として扱う。

## inner holdoutが安定しない直接要因

全noise共通重みの問題は解消済みだが、現在の`inner_holdout`には次の問題が残る。

1. 6/11の18 WAV中、1回だけ抽出した4 WAVで単体性能を推定するため、seedで選ばれるWAVと深層学習結果の影響を受けやすい。
2. 重みは各モデルの`1/(1-R²)`、すなわち同一holdout上の逆MSE相当だけから決まり、モデル間残差の相関・相殺を直接評価しない。
3. 全モデルへ必ず正の重みを配るため、あるnoiseで有害なモデルを0にできない。
4. 学習日内順位が別日の順位へ移らない。noiseあり18条件では、最大重みモデルと6/18の最良単体が一致したのは18条件中6条件、重み順位と6/18性能順位の平均Spearman相関はちょうど0であった。

全21条件では最大重みモデルの一致は9/21。一致した9条件ではinnerは最良単体より平均1.09 kW/m²良かったが、不一致12条件では平均3.49 kW/m²悪かった。例えばseed 43・0 dBはAlexNetへ0.616、seed 42・−12 dBはConformerへ0.601を配り、評価日ではその順位が維持されず大きな悪化になった。これが、noise別に重みを計算してもinnerが常に等重みを上回らない中心原因である。

6/18を用いた採用禁止の事後診断では、連続凸結合の最適解がAlexNetを0とする条件が15/21あった。また7つの等重み部分集合ではRF+Conformerが12/21条件、全てnoiseあり条件で最良で、noiseあり平均93.66 kW/m²だった。これは「AlexNetを必ず混ぜない仕組み」に改善余地があることを示すが、6/18を見てRF+Conformerへ固定する根拠には使えない。

## 次に行う識別比較

次はモデル構造や音響選別を同時に変えず、同じmatched条件で重み決定だけを比較する。

1. `simple_equal`と現行`inner_holdout`を対照として残す。
2. `subset_equal_cv`を追加する。6/11の元WAV非共有OOF予測だけで7部分集合を比較し、有害なモデルを0にできるか確認する。
3. `crossfit_shrinkage_stack`を追加する。同じOOF予測で統合後のWAV誤差と残差相関を扱い、等重みへの縮小で極端な重みを抑える。
4. 小標本で過適合しやすい無正則化`crossfit_wav_stack`と、相関を扱わず正の重みを残すだけの`1/RMSE`・`1/MAE`は最初の比較へ入れない。
5. 6/11学習→6/18評価の結果は方式開発用と明記し、noiseあり平均RMSE、最良単体2%以内の割合、最悪SNR、ONB前・近傍・以降、誤報・見逃し、seed間の選択・重み安定性で一方式を選ぶ。
6. 方式を一つに固定した後、学習日を6/11+6/18、評価日を7/9へ変更し、同じコード・同じ条件で別日確認する。最終候補runでは保存モデルと再読込検証を有効にする。

7/9は既存データであり、この研究で完全に未観察の試験日ではない。そのため「未知日の保証」ではなく「重みfitに使わない別日での確認」と表現する。また、追加実験および同期映像の取得可能性はない。現在の3実験日の範囲で結論の適用範囲を明示する。

## 解析成果物

- [再解析コード](analyze_results.py)
- [3 run監査](run_audit.csv)、[21条件監査](condition_audit.csv)
- [seed別指標](metrics_by_seed.csv)、[3 seed集約](metrics_seed_aggregate.csv)
- [領域別指標](region_metrics_by_seed.csv)、[統合差分](ensemble_deltas_by_seed.csv)
- [noise別重み](weights_by_seed_snr.csv)、[重み安定性](weight_stability_by_snr.csv)
- [重み順位の別日移送診断](inner_weight_transfer_diagnostic.csv)、[要約](inner_weight_transfer_summary.csv)
- [clean-only対応比較](matched_vs_clean_only.csv)、[領域別対応比較](matched_vs_clean_only_by_region.csv)
- [暫定基準判定](provisional_success_criteria.csv)、[条件別勝者](winner_by_seed_snr.csv)
- 6/18を使うため採用禁止の[連続重み事後診断](evaluation_day_oracle_diagnostic.csv)、[部分集合事後診断](evaluation_day_equal_subset_aggregate.csv)
