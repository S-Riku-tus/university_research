# ONB主コードの実験日指定と旧スクリプトの整理

日付: 2026-09-16。対象はコード設定・データ計画の監査であり、モデルの学習結果ではない。

## 確認した処理経路

1. `VALIDATION_CONFIG`の実験日・周波数・ノイズから`dataset_jobs.py`が入力候補を作り、存在確認とONB閾値確認をする。
2. `learning_policy.py`が学習日・テスト日を確定し、`learning_runner.py`がテスト日ごとに学習対象を組み立てる。
3. 解決後の`experiment_names`はXAIの対象確認、run manifest、実行hashにも記録される。

従来の`data.experiment_names`は、`explicit_days`の学習日・テスト日を選ぶ変数ではなかった。それでもデータ候補を作るために必要で、余分な日が入ると未使用データまで存在確認される。逆に明示した学習日・テスト日が一覧にないと実行前にエラーになる。二重指定は誤設定の余地があった。

## 修正と現在の使い方

- 現在の`explicit_days`では`train_experiments`と`test_experiments`の和集合から対象日を自動生成する。主設定の3日は従来と同じ順序（6/11、6/18、7/9）で解決し、保存manifestにも解決後の一覧を残す。既存の同一条件の実行hashを変えない構成。
- 旧`within_day`または`leave_one_day_out`を使うときだけ、`data.experiment_names`を明示する。明示分割で旧キーも残す場合は和集合と完全一致させ、未使用日を拒否する。
- `explicit_days`で`clean_only`を使うとき、学習専用日は無雑音データだけを確認する。テスト日は指定した評価ノイズを確認する。`matched`では各学習日とテスト日の指定ノイズを確認する。
- 例えば6/18だけをテスト、6/11だけを学習にする場合は、`learning_policy`の二つのリストを変更すればよい。7/9は対象から外れ、データ欠損の検査にも入らない。

## 退避したコード

`code/analysis_channel.py`、`code/calc_channel_num.py`、`code/crop_spectrogram.py`を既存の`code/trush_box/`へ移動した。いずれも2024年データや旧PCの絶対パスを持つ単発処理で、`code/`、`tests/`、`docs/`、`configs/`、`experiments/`の現行参照は見つからなかった。旧ONB実行本体や研究資料の原本は移動していない。

## 検証範囲

- `tests/test_experiment_day_resolution.py`: 4件成功。日付の自動算出、日付過不足・重複の拒否、旧分割との互換、`clean_only`の学習専用日データ計画を確認。
- 変更Pythonファイルの`compileall`成功、`git diff --check`成功。
- この環境には実験日の`Pool_boiling/.../data`とTensorFlowがないため、主コードの300 epochs学習と実データの全件探索は実行していない。性能への影響は未評価。

現在の研究計画は[研究の現在地](../../docs/research_status.md)を参照する。設定済みと学習完了を混同しない。
