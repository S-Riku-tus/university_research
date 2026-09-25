# performance_kfold主方式・固定200 epoch比較

日付: 2026-09-25。状態: **条件・実装方針を確定、未実行。**

## 目的

`inner_holdout`を今後の主方式から外し、6/11の全18 WAVを重み決定に使う`performance_kfold`を主方式とする。過去runの再現性を保つため、`inner_holdout`の実装と過去条件ファイルは削除しない。

直近の9/24本比較では、深層モデルのepochを学習日内3-foldで選んでいた。今回はアンサンブル重みの決め方だけを明確にするため、このepoch選択を無効にし、ConformerとAlexNetの全fitを`run.epochs=200`に固定する。

## 5-foldでの重み決定

6/11の18 WAVを元WAV単位で5分割する。18は5で割り切れないため、検証WAV数は4、4、4、3、3本、対応する学習WAV数は14、14、14、15、15本となる。

1. 1 foldを検証側、残りを一時学習側にする。
2. RF、Conformer、AlexNetを一時学習側だけで学習する。深層2モデルは200 epoch固定とする。
3. 学習に使わなかったWAVを予測する。
4. 5回繰り返し、18 WAV全てに「自分を学習に含めていない」OOF予測を1つずつ得る。
5. foldごとの重みは作らず、全OOF chunkを結合してモデル別に1個のR²を計算する。
6. `e_m = 1 - R_m^2`、`w_m = normalize(1 / max(e_m, 1e-6))`により1組の重みを作る。
7. 3モデルを6/11の全18 WAVで200 epoch固定により再学習し、その1組の重みで6/18を一度だけ評価する。

この方法では、単一の4-WAV holdoutだけで重みを決める偏りを避け、全18 WAVを重み推定へ一度ずつ使う。ただし、fold割当、深層学習、学習日と評価日の差による変動は残るため、seed 42～44を維持する。

## 比較対象と今回扱わないもの

- 主方式: `performance_kfold`
- 対照: `simple_equal`、RF、Conformer、AlexNet
- 今回外す方式: `inner_holdout`
- 今回まだ同時実行しない方式: `subset_equal_cv`、`crossfit_wav_stack`、`crossfit_shrinkage_stack`

`simple_equal`は追加の内部学習を必要としない。後ろ3方式は元WAV非共有OOF予測を使う点は共通だが、現行実装では`performance_kfold`とは別の4-fold OOFを作るため、同時に入れると内部学習を重複して実行する。まず主方式の結果を確定し、その後に同一OOFを共有する比較へ進むか判断する。

## 条件

- 学習日: 2025.06.11
- 評価日: 2025.06.18
- 周波数上限: 22 kHz
- 学習noise: `matched`。各SNRでモデル・前処理・重みを独立fitする
- noise: clean、0、−4、−8、−12、−16、−20 dB
- 音響選別: なし
- epoch: 200固定。`training_validation.enabled=false`
- 重み用内部検証: 元WAV非共有5-fold
- seed: 42、43、44
- 評価日ラベルによるepoch・重み・方式選択: なし

条件ファイル: [`2026-09-25_matched_performance_kfold_fixed_epoch.json`](../../configs/experiments/2026-09-25_matched_performance_kfold_fixed_epoch.json)

## 実行コマンド

```powershell
python experiments/2026-09-19_b_clean_only/run_from_condition.py --condition configs/experiments/2026-09-25_matched_performance_kfold_fixed_epoch.json --seed 42
python experiments/2026-09-19_b_clean_only/run_from_condition.py --condition configs/experiments/2026-09-25_matched_performance_kfold_fixed_epoch.json --seed 43
python experiments/2026-09-19_b_clean_only/run_from_condition.py --condition configs/experiments/2026-09-25_matched_performance_kfold_fixed_epoch.json --seed 44
```

## 実行後の最低監査

- 各seedで7 SNRが全て完了している。
- 各matched SNRの`fit_id`が別で、`ensemble_weight_scope=per_training_noise`である。
- `internal_validation_fold1.json`の`method`が`wav_kfold`、fold数が5である。
- 全内部foldで`shared_source_wavs=0`である。
- OOFの全chunkがちょうど一度だけ予測され、評価日データが内部検証へ入っていない。
- `ensemble_weights_*.csv`に`performance_kfold`と`simple_equal`の重みが保存される。
- manifest上で`training_validation.enabled=false`、最終深層fitが200 epochである。

性能判断は、単体・等重みとの全域RMSE、ONB前・近傍・以降、誤報・見逃し、SNR別の悪化、seed間の重み安定性を分けて行う。
