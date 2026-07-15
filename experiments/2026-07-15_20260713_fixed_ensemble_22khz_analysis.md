# 2026-07-13 fixed ensemble 22kHz result analysis

## Scope

7/13に実行された現行 `code/run_ensemble_regression_onb.py` の出力を分析した。

- result: `Pool_boiling/Subcooling_20_degrees/0.3/<experiment>/regression_result/npy/ensemble/20260713_fixed_ensemble`
- experiments: `2025.06.18_0.3_3`, `2025.07.09_0.3_1`, `2025.06.11_0.3_2`
- frequency: `maxfreq=22kHz`
- noise: `heatflux_no_noise`, `SNR=0/-4/-8/-12/-16/-20`
- folds/epochs: 3-fold, 300 epochs
- models: `rf`, `cnntf_v2_gap`, `alexnet`, `ensemble`
- ensemble: fixed weights `rf=0.90`, `cnntf_v2_gap=0.05`, `alexnet=0.05`

All 21 conditions were completed. The three `tuning_summary.csv` files contain 84 rows total.

## Overall result

| model | n | mean R2 | min R2 | mean RMSE all | mean RMSE ONB | ROC-AUC cont | PR-AUC cont | mean F1 | mean recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rf | 21 | 0.9633 | 0.9234 | 50934 | 42496 | 0.9967 | 0.9978 | 0.9807 | 0.9642 |
| ensemble | 21 | 0.9633 | 0.9214 | 51068 | 43289 | 0.9967 | 0.9978 | 0.9639 | 0.9327 |
| cnntf_v2_gap | 21 | 0.8597 | 0.7957 | 101678 | 81846 | 0.9898 | 0.9944 | 0.8691 | 0.7797 |
| alexnet | 21 | 0.8846 | 0.7384 | 91016 | 99520 | 0.9891 | 0.9941 | 0.8746 | 0.7800 |

The fixed ensemble is very close to RF in average R2, but it does not clearly exceed RF. RF remains the strongest single baseline, especially for thresholded ONB detection.

## Experiment-wise view

| experiment | model | mean R2 | min R2 | mean RMSE ONB | mean F1 | mean recall |
|---|---|---:|---:|---:|---:|---:|
| 2025.06.11_0.3_2 | rf | 0.9488 | 0.9436 | 27178 | 0.9905 | 0.9813 |
| 2025.06.11_0.3_2 | ensemble | 0.9499 | 0.9454 | 25576 | 0.9615 | 0.9262 |
| 2025.06.11_0.3_2 | cnntf_v2_gap | 0.8780 | 0.8589 | 66272 | 0.8885 | 0.8029 |
| 2025.06.11_0.3_2 | alexnet | 0.9164 | 0.8943 | 78549 | 0.8991 | 0.8180 |
| 2025.06.18_0.3_3 | rf | 0.9801 | 0.9785 | 24905 | 0.9671 | 0.9385 |
| 2025.06.18_0.3_3 | ensemble | 0.9798 | 0.9776 | 26249 | 0.9465 | 0.9003 |
| 2025.06.18_0.3_3 | cnntf_v2_gap | 0.8565 | 0.8333 | 82049 | 0.8545 | 0.7506 |
| 2025.06.18_0.3_3 | alexnet | 0.8830 | 0.7384 | 103592 | 0.8617 | 0.7610 |
| 2025.07.09_0.3_1 | rf | 0.9611 | 0.9234 | 75407 | 0.9845 | 0.9727 |
| 2025.07.09_0.3_1 | ensemble | 0.9602 | 0.9214 | 78041 | 0.9838 | 0.9714 |
| 2025.07.09_0.3_1 | cnntf_v2_gap | 0.8447 | 0.7957 | 97215 | 0.8642 | 0.7856 |
| 2025.07.09_0.3_1 | alexnet | 0.8545 | 0.7907 | 116420 | 0.8631 | 0.7609 |

`2025.06.11_0.3_2` is the only experiment where the fixed ensemble slightly improves R2 and ONB RMSE over RF on average. However, even there F1 and recall fall clearly, so the improvement is more like regression smoothing than better ONB detection.

## Noise-wise R2

| noise | rf | ensemble | cnntf_v2_gap | alexnet |
|---|---:|---:|---:|---:|
| no_noise | 0.9565 | 0.9549 | 0.8432 | 0.8668 |
| SNR=0 | 0.9667 | 0.9658 | 0.8366 | 0.8117 |
| SNR=-4 | 0.9642 | 0.9642 | 0.8552 | 0.8873 |
| SNR=-8 | 0.9639 | 0.9641 | 0.8702 | 0.9019 |
| SNR=-12 | 0.9649 | 0.9653 | 0.8579 | 0.9021 |
| SNR=-16 | 0.9637 | 0.9645 | 0.8647 | 0.9080 |
| SNR=-20 | 0.9635 | 0.9642 | 0.8903 | 0.9145 |

RF is stable across all noise conditions. AlexNet improves under stronger waterflow-noise conditions, but remains weaker than RF. CNN+Transformer is also stable enough to be useful as a comparison model, but not enough to pull the ensemble above RF.

## RF vs fixed ensemble

The ensemble had higher R2 than RF in 10 of 21 conditions, but the margins were very small. RF had higher F1 and recall in all 21 conditions.

From the saved fold predictions:

| model | total positives | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rf | 12180 | 11741 | 13 | 439 | 0.9989 | 0.9640 | 0.9811 |
| ensemble | 12180 | 11346 | 11 | 834 | 0.9990 | 0.9315 | 0.9641 |

The fixed ensemble made only two negative samples better by changing RF false positives into true negatives, but it changed 395 RF true positives into ensemble false negatives. In other words, the 5% + 5% deep-model contribution tends to lower some positive-side predictions around the threshold. This explains why R2 remains almost unchanged while recall and F1 drop.

Within the ONB band (`|y - threshold| <= 0.10 * threshold`), there are 60 samples per condition. The mean ensemble-minus-RF prediction shift in that band was about `-6368`, so the current fixed ensemble is slightly conservative near ONB.

## Comparison with previous 22kHz rows

Compared with the previous `20260706_fixed_ensemble_full_grid` 22kHz subset:

| model | delta mean R2 | delta mean RMSE ONB | delta mean F1 |
|---|---:|---:|---:|
| rf | +0.0000 | 0 | +0.0000 |
| ensemble | +0.0064 | -12164 | +0.0161 |
| cnntf_v2_gap | +0.0128 | +1501 | -0.0022 |
| alexnet | +0.0020 | +1870 | -0.0046 |

The current fixed ensemble is better than the previous legacy-weight 22kHz ensemble, but this is still not enough to beat RF as the main model.

## Diagnostic weight check

Using saved fold predictions, several fixed weights were simulated without retraining. This is diagnostic only; it should not be used as final evidence because the weights are being chosen after seeing validation predictions.

| candidate | weights | mean R2 | min R2 | mean RMSE ONB | mean F1 | mean recall |
|---|---|---:|---:|---:|---:|---:|
| rf only | 1.00 / 0.00 / 0.00 | 0.96351 | 0.92320 | 43688 | 0.98080 | 0.96424 |
| current | 0.90 / 0.05 / 0.05 | 0.96343 | 0.92128 | 44373 | 0.96418 | 0.93285 |
| rf95 even | 0.95 / 0.025 / 0.025 | 0.96369 | 0.92242 | 43747 | 0.96892 | 0.94176 |
| rf98 even | 0.98 / 0.01 / 0.01 | 0.96363 | 0.92293 | 43643 | 0.97423 | 0.95177 |
| rf95 + alex05 | 0.95 / 0.00 / 0.05 | 0.96378 | 0.92267 | 44102 | 0.96822 | 0.94033 |
| rf95 + cnntf05 | 0.95 / 0.05 / 0.00 | 0.96356 | 0.92212 | 43435 | 0.97028 | 0.94430 |

The diagnostic pattern suggests that if the thesis needs an ensemble line, a very small deep-model weight such as `rf=0.98, cnntf_v2_gap=0.01, alexnet=0.01` may preserve ONB recall better than the current 0.90/0.05/0.05 setting. However, RF-only still has the best F1/recall in this diagnostic check.

## Interpretation

現時点では、7/13の結果は「固定重みアンサンブルがRF単体を大きく改善した」とは言いにくい。むしろ、RFの強い回帰性能をほぼ維持しながら、深層モデルを少し混ぜたときにONB判定のrecallが落ちることが確認できた結果として読むのが安全。

研究上の位置づけとしては、RFを最強ベースライン、固定アンサンブルを頑健性確認用の比較対象、CNN+Transformer/AlexNetを説明性・特徴利用の比較対象として扱うのがよい。アンサンブルは「平均精度を上げる主役」ではなく、「単体モデル依存を下げられるかを検証したが、現設定ではONB境界の見逃し増加が課題」とまとめるのが自然。

## Next actions

1. 7/13結果をそのまま主張する場合は、RF単体を主結果、fixed ensembleを比較結果として並べる。
2. アンサンブルを残すなら、次は `rf=0.98, cnntf_v2_gap=0.01, alexnet=0.01` など低混合率を事前固定して再実行する。
3. ONB主張ではR2ではなく、recall/F1、ONB RMSE、閾値近傍のfalse negative数を中心に見る。
4. `2025.07.09_0.3_1 / no_noise` は全体で最も弱い条件なので、散布図とfold予測を確認する。
5. 説明性解析はRFと、深層モデルが比較的高い `SNR=-16/-20` のAlexNet/CNN+Transformerから選ぶとよい。
