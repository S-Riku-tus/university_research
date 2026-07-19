# Experiment Result Folder Naming

`code/run_ensemble_regression_onb.py` の結果フォルダは、Windows のパス長制限と再実行時の衝突を避けるため、短い実行名を使う。

## 基本方針

- フォルダ名には最低限の識別子だけを入れる。
- 詳細な条件は各結果フォルダ直下の `run_manifest.json` と `validation_results_*.txt` に保存する。
- 同じ条件を同じ日に再実行しても上書きしにくいよう、実行開始時刻ベースの `run_instance_id` を入れる。
- 条件が変わったことを見分けやすいよう、設定内容から作った短い `run_hash` も入れる。

## フォルダ名の形

```text
e{epochs}_{parameter_set}_{weight_strategy}_{models}_{run_hash}_{run_instance_id}
```

例:

```text
e10_legacy_vleg_3m_a1b2c3_153012
```

意味:

- `e10`: epoch 数
- `legacy`: parameter set 名
- `vleg`: `val_fold_legacy`
- `3m`: `rf`, `cnntf_v1`, `alexnet` の3モデル
- `a1b2c3`: 実行設定から作った短い hash
- `153012`: 実行開始時刻由来の ID

## 詳細確認

短いフォルダ名だけでは、学習率、バッチサイズ、RandomForest パラメータ、データソースなどは読めない。これらは次を見る。

- `run_manifest.json`: 機械的に読みやすい完全な実行条件
- `validation_results_*.txt`: 人が読みやすい実行条件と fold ごとの結果
- `metrics_summary_*.csv`: モデル別の平均指標
- `ensemble_weights_*.csv`: fold ごとのアンサンブル重み

## 任意の名前を付けたい場合

環境変数 `RUN_NAME_SUFFIX` を使うと、フォルダ名の末尾に短いメモを追加できる。

```powershell
$env:RUN_NAME_SUFFIX = "note1"
python code/run_ensemble_regression_onb.py
```

完全に同じ `run_instance_id` を使いたい場合だけ、環境変数 `RUN_ID` を指定する。

```powershell
$env:RUN_ID = "manual001"
python code/run_ensemble_regression_onb.py
```
