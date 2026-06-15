# 2026-06-15 RF・3モデル基準実験

## 基本情報

- 実験日: 2026-06-15
- 実験名: RF単体と3モデル単純平均アンサンブルの基準確認
- 目的: RF単体が通常設定で動くか確認し、その後3モデル同時実行と単純平均アンサンブルの基準結果を得る。
- 関連コード: `code/run_ensemble_regression_onb.py`
- 関連データ: `Pool_boiling/Subcooling_20_degrees/0.3/2025.07.09_0.3_1/data/npy/waterflow_20251219_1s/maxfreq=22kHz/heatflux_no_noise`

## 条件

- 入力表現: STFT系 `.npy`
- chunk: 1 s
- maxfreq: 22 kHz
- noise: `heatflux_no_noise`
- SNR: no_noise
- 分割: `KFold(n_splits=5, shuffle=True, random_state=42)`
- RF前処理: 学習foldのみでPCA fit、`PCA_COMPONENTS=100`
- threshold: `275174.6641`
- ONB近傍: `threshold ± 10%`
- アンサンブル: `WEIGHT_STRATEGY=simple`, `ENSEMBLE_COMBINE=mean`

## 出力

- RF単体: `Pool_boiling/Subcooling_20_degrees/0.3/2025.07.09_0.3_1/regression_result/npy/ensemble/heatflux_no_noise/maxfreq=22kHz/20260208_ep500_bs48_lr0.005_simple_rf`
- 3モデル: `Pool_boiling/Subcooling_20_degrees/0.3/2025.07.09_0.3_1/regression_result/npy/ensemble/heatflux_no_noise/maxfreq=22kHz/20260208_ep500_bs48_lr0.005_simple_3models`

## 結果

| model | R2 | RMSE all | MAE all | RMSE ONB | ROC-AUC cont | PR-AUC cont | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9241 | 76575.9 | 52096.7 | 82426.8 | 0.9648 | 0.9734 | 0.9029 | 0.9816 | 0.8300 | 0.8991 |
| CNN+Tf (AttnPool) | 0.8252 | 115126.8 | 93489.7 | 95431.2 | 0.8989 | 0.9418 | 0.8637 | 1.0000 | 0.7388 | 0.8475 |
| CNN+Tf (GAP) | 0.6067 | 165105.7 | 138883.4 | 82595.8 | 0.8158 | 0.8656 | 0.7951 | 0.9147 | 0.7673 | 0.8073 |
| Ensemble simple mean | 0.8559 | 105473.3 | 86589.4 | 73610.8 | 0.9654 | 0.9741 | 0.8863 | 1.0000 | 0.7831 | 0.8775 |

## 解釈

- RF単体は通常設定で完走し、同一条件内ランダムKFoldでは全体R2と連続スコアAUCが高い。
- 3モデル同時実行も完走したため、アンサンブルの実行経路は成立している。
- 単純平均アンサンブルは、連続スコアROC-AUC/PR-AUCではRF単体とほぼ同等だが、R2、RMSE、Recall、F1はRF単体より悪化した。
- CNN+Transformer系2モデルがRFより弱いため、単純平均で混ぜると全体回帰性能を下げている可能性が高い。
- 一方で、EnsembleのONB近傍RMSEはRF単体より小さいため、ONB近傍だけを見ると補完的に働いている可能性がある。

## 課題

- 現状はno_noiseのみ、かつランダムKFoldであり、汎化性能や高ノイズ頑健性は未確認。
- 単純平均では弱いモデルの影響を受けるため、固定重み、RF重視重み、inner_holdout重みを比較する必要がある。
- CNN+Tf (GAP) はfoldによって高熱流束域のR2が不安定で、単純平均に入れるべきか検討が必要。

## 次に試すこと

- `WEIGHT_STRATEGY=fixed` でRF重視重みを試す。例: `rf=0.6`, `cnntf_v1=0.3`, `cnntf_v2=0.1`。
- `WEIGHT_STRATEGY=inner_holdout` を試し、検証foldを見ない重み決定で改善するか確認する。
- no_noiseだけでなく、SNR条件を有効化して高ノイズ条件でRF単体とアンサンブルの差を見る。
- ONB近傍の見逃し率・誤検知率・検知遅れを追加し、平均R2ではなくONB近傍検知として評価する。

## 週次報告に使う要点

- RF単体と3モデル同時実行は完走した。
- 現時点ではRF単体が最も全体性能が高く、単純平均アンサンブルは全体回帰ではRF単体を下回った。
- ただしONB近傍RMSEは単純平均で改善しており、ONB近傍だけに絞った重み付けや評価が次の検討対象になる。

## 2026-06-15 追記

この実験の3モデル構成は `rf / cnntf_v1 / cnntf_v2` だったが、`cnntf_v2` は全体性能が弱く、`cnntf_v1` とモデル種別も近かった。そのため、現在の実行構成は `rf / cnntf_v1 / alexnet` に変更した。今後の固定重み例は、旧 `cnntf_v2` ではなく `alexnet` に読み替える。
