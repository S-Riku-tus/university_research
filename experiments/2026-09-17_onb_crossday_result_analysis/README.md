# 9/17 ONB別日結果の解析（6/11学習・6/18評価）

日付: 2026-09-17

## 1. 対象と完了範囲

対象は、6/18実験フォルダ内の次のrunである。

- 保存先: `Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260917/selected_log_architecture__days_matched/maxfreq=3kHz/heatflux_no_noise/e150_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp_af1ea408`
- run instance/hash: `000410 / af1ea408`
- 完了状態: 3 kHzは`completed.json`、予測、WAV評価、説明性出力まで存在する。
- 22 kHzは`completed.json`と性能・予測CSVがなく、途中出力として今回の性能比較から除外する。

保存manifest上の実条件は、学習6/11のみ、評価6/18、無雑音、150 epochs、seed 42である。7/9は学習に含まれていない。学習・評価はいずれも18 WAV、1,080個の1秒chunkである。ピーク高さ選別は有効だが、6/11は`1,080 -> 1,080`秒で除外がないため、このrun単独では選別効果を評価できない。

根拠:

- [run manifest](../../Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260917/selected_log_architecture__days_matched/maxfreq=3kHz/heatflux_no_noise/e150_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp_af1ea408/run_manifest.json)
- [学習選別記録](../../Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260917/selected_log_architecture__days_matched/maxfreq=3kHz/heatflux_no_noise/e150_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp_af1ea408/training_selection_fold1.json)
- [完了印](../../Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260917/selected_log_architecture__days_matched/maxfreq=3kHz/heatflux_no_noise/e150_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp_af1ea408/completed.json)

## 2. 「学習日6/11の内部OOFのWAV平均二乗誤差」とは何か

### 2.1 一言での定義

これは音声波形の振幅誤差ではない。**6/11の各WAVについて、そのWAVを学習に使っていないモデルが出した熱流束予測を1個の代表値へまとめ、真の熱流束との差を二乗し、18 WAVで平均した値**である。

より誤解の少ない呼び方は、`WAV単位MSE（各WAVを1票とする内部OOF熱流束誤差）`である。

### 2.2 作り方

6/11には18 WAVがあり、各WAVは60個の1秒chunkを持つ。合計1,080 chunkである。

1. 18 WAVを元WAV単位で4 foldへ分ける。各foldでは13または14 WAVで学習し、残り4または5 WAVを予測する。
2. 同じWAVのchunkは学習側と保留側へ分断しない。したがって、各保留WAVの予測は、そのWAVを見ていないモデルから得られる。
3. 4 foldをつなぐと、6/11の全1,080 chunkに対して一度ずつOOF（out-of-fold）予測が得られる。
4. 候補重みで3モデルのchunk予測を加重平均する。
5. 各WAV内の60予測を中央値にまとめ、WAV代表予測を1個作る。
6. WAV代表予測と、そのWAVの真の熱流束との差を二乗し、18 WAVで平均する。

モデルを $m$、WAVを $i$、その中の1秒chunkを $j$、モデル重みを $w_m$ とすると、今回の値は次である。

$$
\mathrm{MSE}_{\mathrm{WAV}}(w)
=\frac{1}{18}\sum_{i=1}^{18}
\left[
\operatorname{median}_{j\in i}
\left(\sum_m w_m\hat q^{\mathrm{OOF}}_{ijm}\right)-q_i
\right]^2
$$

ここで、$q_i$はWAV $i$の真の熱流束、$\hat q^{\mathrm{OOF}}_{ijm}$はそのWAVを学習に含めずに得た1秒予測である。

「平均」は最後の18 WAVに対する二乗誤差の平均を指す。WAV内の60秒は平均ではなく**中央値**でまとめている。また、60秒を独立な60反復として重く数えず、各WAVを同じ1票として扱う。

### 2.3 今回の実数値

単体モデルは、対象モデルの重みだけを1、他を0にして同じ式で計算する。

| 6/11内部OOF | WAV単位MSE [(W/m²)²] | 平方根RMSE [kW/m²] |
|---|---:|---:|
| RF | 7.021 × 10⁹ | 83.8 |
| CNN+Transformer | **4.114 × 10⁹** | **64.1** |
| AlexNet | 7.066 × 10⁹ | 84.1 |
| 3モデル等重み | 5.125 × 10⁹ | 71.6 |
| crossfit WAV stack選択後 | **4.091 × 10⁹** | **64.0** |
| shrinkage stack選択後 | 4.130 × 10⁹ | 64.3 |

MSEは二乗単位で直感的に読みにくいため、平方根を取ったRMSEも併記した。内部OOFではCNNが最も小さかったため、重み推定がCNN中心になること自体は6/11内の計算として整合している。

### 2.4 この値が答える問いと、答えない問い

答える問い:

> 6/11と同じ日の別WAVを未知WAVとして予測するとき、どの単体モデルまたは重みがWAV代表熱流束をよく予測するか。

答えない問い:

> 6/18のような別実験日でも、同じモデル順位と重みが最良になるか。

内部OOF値は6/18ラベルを使わずに重みを決めるため、外側テストへの直接リークはない。一方で、録音日、装置状態、背景音などはすべて6/11の範囲内である。したがって、これは重みfit用の内部診断であり、別日一般化性能の不偏推定値ではない。保存JSONにも`inner_oof_weight_fit_only_not_outer_performance`と明記されている。

## 3. 内部OOF値から重みが決まる流れ

今回の重みは次の通りである。

| 方式 | RF | CNN+Transformer | AlexNet |
|---|---:|---:|---:|
| subset equal CV | 0.0% | 100.0% | 0.0% |
| crossfit WAV stack | 0.0% | 97.8% | 2.2% |
| crossfit shrinkage stack | 0.0% | 82.2% | 17.8% |

- `subset_equal_cv`は候補部分集合のうち、CNN単体を選んだ。
- `crossfit_wav_stack`はWAV単位MSEを直接最小化し、ほぼCNN単体になった。
- `crossfit_shrinkage_stack`はMSEだけでなく等重みから離れすぎることへの固定罰則も含むため、CNN単体よりMSEがわずかに大きくてもAlexNetを17.8%残した。
- RFは6/11内部OOFでCNNより悪かったため、3方式とも重み0になった。

根拠:

- [内部OOF予測](../../Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260917/selected_log_architecture__days_matched/maxfreq=3kHz/heatflux_no_noise/e150_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp_af1ea408/ensemble_inner_oof_f1.csv)
- [crossfit重み・診断](../../Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260917/selected_log_architecture__days_matched/maxfreq=3kHz/heatflux_no_noise/e150_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp_af1ea408/ensemble_crossfit_fit_f1.json)

## 4. 6/18別日評価の結果

主評価である元WAV予測中央値は次の通り。誤差はkW/m²へ換算した。

| モデル | R² | RMSE | MAE | ONB以上R² | ONB近傍RMSE | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RF | **0.842** | **108.4** | **93.9** | 0.717 | 78.5 | **1.000** | 0.750 | **0.857** |
| CNN+Transformer | 0.555 | 181.6 | 125.0 | **0.967** | 37.0 | 0.667 | **1.000** | 0.800 |
| AlexNet | 0.775 | 129.2 | 102.8 | 0.608 | 56.1 | **1.000** | 0.750 | **0.857** |
| subset equal CV | 0.555 | 181.6 | 125.0 | **0.967** | 37.0 | 0.667 | **1.000** | 0.800 |
| crossfit WAV stack | 0.564 | 179.8 | 123.4 | 0.966 | 35.0 | 0.667 | **1.000** | 0.800 |
| crossfit shrinkage stack | 0.621 | 167.7 | 113.7 | 0.952 | **22.2** | 0.667 | **1.000** | 0.800 |

6/11内部OOFではCNNが最良だったが、6/18全域ではRFが最良になった。これは重み最適化の計算ミスではなく、**同日未知WAVでのモデル順位が別日で逆転した**ことを示す。

shrinkage統合はCNN単体に対してRMSEを約7.7%、MAEを約9.0%改善したが、RFよりRMSEが約55%、MAEが約21%大きい。元WAV分離により内部リークは解消されたが、単一学習日だけのOOFでは別日で強いRFを選べなかった。

根拠:

- [WAV指標](../../Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260917/selected_log_architecture__days_matched/maxfreq=3kHz/heatflux_no_noise/e150_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp_af1ea408/wav_eval/wav_metrics_no_noise.csv)
- [WAV予測](../../Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260917/selected_log_architecture__days_matched/maxfreq=3kHz/heatflux_no_noise/e150_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp_af1ea408/wav_eval/wav_predictions_no_noise.csv)

## 5. 誤差の構造

### 確認した事実

- RFは全域回帰と誤警報抑制が最良だが、ONB直後を低く予測する。
- CNNはONB以上のWAVでMAE 26.3 kW/m²と良い一方、ONB前は平均で203.9 kW/m²高く予測する。
- shrinkage統合もCNNを82.2%含むため、ONB前を平均182.4 kW/m²高く予測する。
- 真値0 kW/m²のWAV中央値は、RF 191.5、CNN 427.0、AlexNet 238.9、shrinkage 393.1 kW/m²である。
- CNNの学習日内部OOFから別日への予測傾向は、概略直線の傾きが0.861から0.579へ、切片が57.8から257.1 kW/m²へ変化している。低域の上方シフトと予測範囲の圧縮が同時に起きている。

### 整合する仮説

3 kHz CNNは熱流束の大小順序を完全に失ったのではなく、日が変わったときの入力profile差により回帰出力の尺度・オフセットがずれた可能性が高い。単一日OOFは同日の未知WAVには対応できても、この日間校正ずれを重み決定へ反映できない。

## 6. ONB検知としての読み方

6/18の物理ONB閾値は376.32 kW/m²である。

chunk単位では次のトレードオフがある。

| モデル | ONB前の誤警報chunk | ONB後の見逃しchunk |
|---|---:|---:|
| RF | **1** | 99 |
| CNN+Transformer | 246 | **16** |
| AlexNet | **0** | 121 |
| shrinkage stack | 147 | 26 |

WAV中央値では、RF/AlexNetはONB前誤警報がない代わりに、最初の陽性判定が496.0 kW/m²となり、真のONBから2測定点、119.7 kW/m²遅れる。CNN系は真のONB測定点を陽性にするが、ONB前4 WAVも陽性にする。2 WAV連続陽性を要求しても225.7 kW/m²から始まるため、きれいなONB遷移とはいえない。

RF、AlexNet、shrinkageのWAV連続ROC-AUCは1.0である。これは、ONB前後の順位分離はできている一方、予測熱流束へ物理閾値を直接適用したときの校正がずれていることを示す。R²、ROC-AUC、固定閾値での見逃し・誤警報は別々に判断する必要がある。

根拠: [ONB遷移評価](../../Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260917/selected_log_architecture__days_matched/maxfreq=3kHz/heatflux_no_noise/e150_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp_af1ea408/wav_eval/onb_transition_summary_no_noise.csv)

## 7. 周波数・説明性との対応

ピーク選別に使った2.1–2.5 kHz閾値だけではCNNの誤警報を説明できない。

- 真値0のWAVではピーク閾値以上が0/60秒だが、CNNは58/60秒をONB以上と予測した。
- 225.7、271.7、322.1 kW/m²ではピーク閾値以上がそれぞれ1、5、15秒だが、CNN陽性は52、51、50秒だった。
- 全WAVの帯域マスクでは、RFは2–3 kHzを消すとWAV R²が0.842から0.347、Recallが0.75から0へ低下した。
- CNNは回帰R²に対して1–2 kHzのマスク影響が最大で、ONB Recallは256–512 Hzを消すと1から0へ低下した。
- CNNの低熱流束誤予測例では、IGの最大寄与位置が約475–529 Hzに現れた。

CNNは選別帯域だけでなく低周波側の定常的な装置音・背景調波に見える構造も利用している可能性がある。ただし音源の物理的同定には同期観測が必要である。

CNNのゼロ入力baseline出力は約855 kW/m²であり、実入力の寄与がそこから予測値を下げる形になっている。IGの数値積分が収束していても、ゼロbaselineは物理的な無沸騰状態ではない。帯域マスクも分布外入力を作るため、これらは原因確定ではなく、低周波依存仮説を支持する補助診断として扱う。

## 8. 9/16の6/11+7/9・300 epochs結果との関係

先行runでは3 kHz CNNのchunk R²が-0.084、WAV中央値R²が-0.053だった。今回の6/11のみ・150 epochsでは、それぞれ0.535、0.555まで改善した。これは7/9追加による負の転移仮説と整合する。

ただし、学習日だけでなくepoch数と統合方式も異なるため、7/9が原因と確定はできない。今回もCNNの低熱流束バイアスは残っており、7/9を外すだけでは問題は解消していない。

## 9. 現時点の結論と次の識別比較

### 現時点の結論

6/11内部OOFのWAV単位MSEは、同日の未知WAVに対して重みを決めるための学習側診断である。CNN中心の重みは6/11内では妥当だったが、6/18ではモデル順位が逆転し、RFが全域最良となった。新crossfitは内部WAV混在を解消したが、日間分布変化そのものを解消する方法ではない。

### 次の識別比較

1. 3 kHz、同一seed、300 epochs、全テスト秒で、6/11のみと6/11+7/9を比較する。
2. 同じ条件でピーク選別なし/ありを比較し、学習日効果と選別効果を分離する。
3. 単体予測を固定して3統合方式を比較し、全域誤差、ONB近傍、誤警報を別々に評価する。
4. 6/11と6/18の低熱流束について256–512 Hz、1–2 kHz、2–3 kHzのprofile差を確認する。
5. 22 kHz runは完了印と性能CSVが揃ってから別スナップショットとして比較する。
6. 複数seedと新しい独立日で再現してから、方式の一般化優位を主張する。

## 10. 再現性上の注意

この保存runは`subset_equal_cv / crossfit_wav_stack / crossfit_shrinkage_stack`を実行している。実行後に主コードの有効方式が編集されており、現在の主コードをそのまま再実行しても同じ方式構成になるとは限らない。再現時は、その時点のソース設定ではなく、このrunの`run_manifest.json`を条件の正本とする。
