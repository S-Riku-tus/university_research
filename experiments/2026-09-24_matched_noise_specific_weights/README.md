# matched学習・ノイズ別重みの確認と次回条件

更新日: 2026-09-24。状態: **コード監査・条件固定済み、本7 SNR比較は未実行**。

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
