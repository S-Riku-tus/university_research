# 9/18発表用図：6/11学習 → 6/18評価

元のPPTXは変更せず、5–8ページに挿入するためのPNGを作成した。対象はrun hash `56fc786f`、22 kHz、無雑音、200 epochs、seed 42。評価は6/18の18 WAVから得た全1080個の1秒chunkで、修正ONB閾値は271677.6816 W/m²。逆方向のrunは含めていない。

| ページ | 図 | 主な根拠 |
|---|---|---|
| 5 | `slide05_regression_r2_mae.png` | `metrics_summary_no_noise.csv` のR²、MAE |
| 6 | `slide06_onb_auc_false_negatives.png` | 同CSVの連続予測ROC-AUCと `fold_pred/pred_f1_no_noise.csv` から修正ONB閾値で再計算した見逃し・誤検知数 |
| 7 | `slide07_rmse_before_after_onb.png` | 同予測CSVをONB前480秒とONB以上600秒に分けたRMSE、`ensemble_weights_no_noise.csv` の重み |
| 8 | `slide08_frequency_mask_top_bands.png` | `explainability/top_groups_by_model.csv` の帯域ゼロマスクによるR²低下最大の帯域 |

5・6ページのドット図は差を見るため横軸を絞っている。誤差棒を載せていないのは、今回の評価日が1日・1 foldであり、秒を独立実験日の反復とみなせないため。7ページの領域別差は観測事実であり、重みの決め方が一般に悪いと確定したものではない。8ページの帯域ゼロマスクは学習時の入力分布を変えるため、物理的な気泡音の位置を特定する図として扱わない。

生成スクリプト：`experiments/2026-09-17_bidirectional_onb_ensemble_xai_analysis/generate_forward_slide_figures.py`。原runの結果は `Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260917/onb_xd-t0611-v0618_iw3-nm_s1e-9_e200_123711/maxfreq=22kHz/heatflux_no_noise/` に保存されている。
