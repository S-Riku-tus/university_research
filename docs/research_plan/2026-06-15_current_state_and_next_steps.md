# 2026-06-15 現在状態と次工程の整理

## ここまでに終わったこと

2026-06-14時点の短期計画では、まずRandomForest単体を安定して動かし、その後3モデルを同一分割・同一指標で比較できる状態にすることを優先していた。

現時点では、次は完了済みとして扱ってよい。

- RF単体の通常設定実行は完走した。
- RF、CNN+Transformer、CNN+Transformerの3モデル同時実行は完走した。
- 単純平均アンサンブルは全体R2/RMSEではRF単体を下回る一方、ONB近傍RMSEでは改善の可能性が見えた。
- CNN+Transformerの2枠は多様性が低く、`cnntf_v2` は全体性能が弱かったため、現在の3モデル構成は `RandomForest + CNN+Tf (AttnPool) + AlexNet` に変更した。
- 保存フォルダの日付が `20260208` になる問題は、固定値 `SAVE_DATE` が原因だったため、実行日 `YYYYMMDD` を使うように修正した。

## 元計画から見た位置づけ

Phase 1の「RF単体で回帰が最後まで通ること」は完了した。

Phase 2の「3モデルを同条件で同時実行できる状態にする」も、旧構成では完了している。ただし、旧構成はCNN+Transformer系が2つ並んでおり、アンサンブルの多様性という観点では弱かった。そのため、Phase 2は「実行経路の成立」までは完了、「研究上妥当な3モデル構成での本番比較」は次に確認する段階とする。

## リポジトリ側で改善したこと

重み付けやONB近傍評価を試すたびに500 epochの学習をやり直すのは効率が悪い。そこで、`run_ensemble_regression_onb.py` に foldごとの予測保存を追加した。

保存されるもの:

- `fold_predictions/pred_f{fold}_{snr}.csv`
  - `sample_index`
  - `y_true`
  - 各単体モデルの予測
  - `ensemble` の予測
- `ensemble_weights_{snr}.csv`
  - foldごとの各モデル重み

これにより、今後は保存済み予測を使って、重み、ONB近傍誤差、閾値判定、検知遅れ候補を後処理で比較しやすくなる。

また、`RANDOM_SEED` がKFoldとPCAだけでなく、Python、NumPy、TensorFlowの乱数にも効くようにした。GPU演算の完全な決定性までは保証しないが、Kerasモデルの初期値に起因する実行ごとの揺れを減らせる。

## 次にやるべきこと

1. `RandomForest + CNN+Tf (AttnPool) + AlexNet` の本番設定を回す。
   - `SMOKE_TEST=False`
   - epoch 500
   - fold 5
   - まず `WEIGHT_STRATEGY=simple`

2. 本番実行後、保存されたfold予測CSVから、再学習なしで次を比較する。
   - RF単体
   - 単純平均
   - RF重視固定重み
   - ONB近傍重視の重み候補

3. 単純平均でRF単体を下回る場合は、アンサンブル失敗として止めるのではなく、どの領域で補完が起きたかを見る。
   - 全体R2/RMSE
   - ONB近傍RMSE/MAE
   - 見逃し、Recall、F1
   - fold間ばらつき

4. その後、SNR条件または水流音ノイズ条件に広げる。

5. 性能比較が固まってから、Grad-CAM、RISE、Integrated Gradients、Attention Rollout、時間・周波数マスク実験へ進む。

## 注意

現在のスモーク結果は3 epochであり、精度の議論には使わない。使えるのは、モデル差し替え後に学習・予測・保存が最後まで通ったという実行確認だけである。
