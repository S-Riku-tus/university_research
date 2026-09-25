# WAV固定内部検証と日付リストによる評価方式の自動判定

## 変更目的

現行ONB実行では、アンサンブル重み用の内部検証を元WAV非共有のK-foldへ固定し、設定から`internal_validation`の選択肢を削除した。また、学習日・評価日のリストだけで評価設計を表せるため、設定値`split_mode`も削除した。

## 現行ルール

`learning_policy`に必要なのは`train_experiments`、`test_experiments`、`training_noise`である。評価方式は次の規則で導出する。

| 日付リストの関係 | 導出する評価方式 | 実処理 |
|---|---|---|
| 同一の1日 | `within_day` | 同日の元WAVを外側GroupKFoldで分離 |
| 同一の複数日 | `leave_one_day_out` | 各評価日について、残りの日を学習に使用 |
| 完全に分離した集合 | `cross_day` | 指定した全学習日で学習し、指定評価日を評価 |
| 一部だけ重複 | 無効 | 意図が曖昧で日付リークの危険があるため停止 |

`performance_kfold`の重み推定は、上記の外側評価方式にかかわらず、学習側データの元WAV単位K-foldを常に使う。同じWAVの1秒chunkが内部fitとheld-outへ分かれる経路は削除した。監査JSONの`method`は常に`wav_kfold`となる。

## 互換性と保存方針

- 現行Python設定と実行用JSON 11件から旧2項目を削除した。
- 実験記録テンプレートは日付リストと導出済み`evaluation_scheme`を記録する形へ更新した。
- 2026-09-16のYAMLや過去run manifestは当時の条件を示す履歴なので書き換えていない。
- 新形式は実行設定のhashを変える。旧結果を新条件として誤再開しない。

## 検証

- 全自動テスト79件に成功した（日付解決、学習方針、音響選別、crossfitを含む）。
- `performance_kfold`の内部各foldで`shared_source_wavs == 0`を確認するテストへ変更した。
- 現行ONB設定の読込結果は、6/11学習・6/18評価から`cross_day`、外側fold数1と導出された。
- 実行用JSON 11件はすべて構文解析に成功した。

本変更では本学習を起動しておらず、既存の結果ファイルも移動・再計算していない。
