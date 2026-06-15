# 2026-06-15 AlexNet差し替え後の3モデルスモーク確認

## 目的

CNN+Transformer系が `cnntf_v1` と `cnntf_v2` の2種類入っており、モデルの多様性が低いため、精度が低かった `cnntf_v2` を素のAlexNetに差し替えた。

## 判断

- `cnntf_v1`: CNN+Transformer + AttentionPooling。前回の500 epoch実行では、CNN+Transformer系の中でR2、RMSE、ROC-AUC、PR-AUC、F1が高かった。
- `cnntf_v2`: CNN+Transformer + GlobalAveragePooling。ONB近傍RMSEは比較的よかったが、全体R2と高熱流束域R2が低く、単純平均アンサンブルでは全体性能を下げる可能性が高かった。
- そのため、`cnntf_v1` を残し、`cnntf_v2` を `AlexNet` に置き換えた。

## 変更後のモデル構成

- RandomForest
- CNN+Tf (AttnPool)
- AlexNet
- Ensemble: simple mean

## スモーク条件

- `SMOKE_TEST=True`
- epoch: 3
- fold: 2
- data: `heatflux_no_noise`
- save date: 実行日から自動生成
- output: `Pool_boiling/Subcooling_20_degrees/0.3/2025.07.09_0.3_1/regression_result/npy/ensemble/heatflux_no_noise/maxfreq=22kHz/20260615_ep3_bs48_lr0.005_simple_smoke_3models`

## スモーク結果

| model | R2 | RMSE all | MAE all | RMSE ONB | ROC-AUC cont | PR-AUC cont | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9207 | 78791.0 | 53676.4 | 63480.2 | 0.9574 | 0.9698 | 0.9000 | 0.9788 | 0.8301 | 0.8974 |
| CNN+Tf (AttnPool) | -1.8935 | 473266.3 | 399665.2 | 386126.9 | 0.5531 | 0.6055 | 0.4941 | 0.2618 | 0.5000 | 0.3436 |
| AlexNet | -0.0116 | 281769.6 | 239825.7 | 62634.6 | 0.6562 | 0.7772 | 0.5294 | 0.5294 | 1.0000 | 0.6923 |
| Ensemble simple mean | 0.3072 | 232365.9 | 192196.9 | 126498.8 | 0.9574 | 0.9698 | 0.6304 | 0.7618 | 0.7546 | 0.6810 |

## 解釈

この結果は3 epochの動作確認であり、本番精度としては使わない。重要なのは、AlexNet差し替え後に3モデルの学習、予測、評価、保存が最後まで通ったことと、保存フォルダの日付が `20260615` になったこと。

追加で、`fold_predictions/pred_f1_no_noise.csv`, `fold_predictions/pred_f2_no_noise.csv`, `ensemble_weights_no_noise.csv` が保存されることも確認した。これにより、本番実行後に重み付けやONB近傍評価を再学習なしで後処理しやすくなる。

本番比較では `SMOKE_TEST=False`、epoch 500、5 foldで再実行し、`RandomForest + CNN+Tf (AttnPool) + AlexNet` が旧構成より改善するかを確認する。
