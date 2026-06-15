# 2026-06-15 モデル別パラメータチューニング方針

## 背景

以前は `BATCH_SIZES_ALL` と `LEARNING_RATE_ALL` に候補値を入れて外側ループで検証していた。しかし、リファクタ後に `MODEL_SPECS` 側へ `lr` と `batch_size` が入っていると、保存フォルダ名だけが変わり、実際のKeras学習条件が変わらない危険があった。

現在は、モデルの定義とチューニング条件を分離している。

- `MODEL_SPECS`: モデルの実体を定義する。
- `RF_FIXED_PARAMS`: RFに渡す固定パラメータを定義する。
- `KERAS_BATCH_SIZE_GRID` / `KERAS_LEARNING_RATE_GRID`: `cnntf_v1` と `AlexNet` のチューニング候補を定義する。
- `PARAMETER_SETS`: 上記のグリッドから自動生成される実行条件。

## 現在の実行設定

RFは単体探索で選んだ次のパラメータに固定する。

```python
RF_FIXED_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bynode": 0.6,
}
```

Keras系2モデルは、次の42条件を試す。

```python
KERAS_BATCH_SIZE_GRID = [12, 24, 32, 48, 64, 128]
KERAS_LEARNING_RATE_GRID = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
```

現在の有効モデルは次の3つである。

```python
ACTIVE_MODEL_KEYS = ["rf", "cnntf_v1", "alexnet"]
```

このため、直接実行すると `RF_FIXED_PARAMS` を使うRandomForestを固定し、`cnntf_v1` と `AlexNet` に同じ `lr` / `batch_size` を渡して42通りを評価する。

## 出力

保存フォルダ名にはKeras条件だけを入れる。

```text
20260615_ep500_k_lr0p005_bs48_simple_3m
```

RFの固定パラメータは、各結果フォルダの `validation_results_*.txt` にある `model_params` に記録される。これにより、保存名は短く保ちつつ、実際の学習条件は後から確認できる。

## 注意

`learning_rate=0.05` はスモーク実行で `cnntf_v1` のlossが大きく跳ねたため、性能が悪い可能性が高い。ただし、グリッド探索として含めておくこと自体は問題ない。

本番実行は `42条件 * 5fold * 2 Kerasモデル = 420回` の深層モデル学習になるため、長時間実行を前提にする。
