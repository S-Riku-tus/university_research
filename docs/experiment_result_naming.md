# 結果の保存階層と識別

更新日: 2026-09-17。実装の正本は [result_paths.py](../code/utils/experiment/result_paths.py) と [run_helpers.py](../code/utils/experiment/run_helpers.py)。

## 現行の保存階層

```text
Pool_boiling/Subcooling_20_degrees/0.3/<収録実験日>/
  regression_result/npy/ensemble/<解析実行日>/
    onb_<主要条件>_[parameter番号_]<HHMMSS>/
      <周波数上限>/
        <ノイズ条件>/
          run_manifest.json
          split_manifest.json
          completed.json
          fold_pred/
          explainability/
        noise_trends/<統合方式>/
      tuning_summary.csv
      ensemble_presentation_summary.csv
```

解析日と収録日を区別する。日付直下の名前は最大52文字で、典型例は次のとおり。

```text
onb_xd-t0611+0709-v0618_iw3-nm_s1e-9_e300_093015
```

- `wd / lo / xd`: 日内分割／leave-one-day-out／学習日・テスト日明示
- `t... / v...`: 学習日／評価日。通常は月日、長すぎる場合は`2d`のように件数へ縮約
- `iw3 / ic3`: WAV単位／chunk単位の内部3-fold検証
- `nm / nc`: matched-noise学習／clean-only学習
- `s0 / s1e-9`: 音響選別なし／ピーク高さ閾値
- `e300`: epoch数
- `p01 / p02`: 1起動で複数parameter setを比較するときだけ付ける候補番号
- 末尾6桁: 実行時刻`HHMMSS`（時分秒）。日付は上位にあるため重複させない

フォルダ名は比較時に重要な条件を優先して示す。モデル詳細、seed、全データ名、全ハイパーパラメータと設定hashは、長文化を避けるため`run_manifest.json`で確定する。同一条件の別起動は秒単位の実行時刻で分離する。

旧runにはノイズ/周波数の順の階層や、完了印・分割manifestのない世代がある。[9/14の移行記録](research_plan/2026-09-14_result_layout_and_generalization.md)を参照し、古い階層を欠落と即断しない。

## run名と実行条件

現行の短いrun名は概ね次の形になる。

```text
e{epochs}_{短縮parameter_tag}_{短縮model_tag}[_{weight_tag}]_{設定hash}_{実行ID}
e300_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp_a1b2c3d4_001530_d4e5f6a7
```

現行の明示日分割では日付直下の`onb_..._<HHMMSS>`が実行単位となり、周波数・ノイズ直下に余分なrun階層を作らない。旧形式などrun階層を使う場合は、run名の末尾へ設定hashと実行IDを付ける。完全な条件は`run_manifest.json`の`created_at / run_instance_id / run_hash / execution_config_hash / validation_config / learning_context`で確認する。

実行IDは起動時刻から6桁で自動生成する。中断した特定実行を再開したい場合だけ、開始時の6桁を`RUN_ID`環境変数へ明示して同じ保存先を使用する。異なる設定で同じ`RUN_ID`を指定した場合は、manifest照合で停止して上書きを防ぐ。

## 完了・再開・比較

- 新形式は完了時に`completed.json`を保存。manifestだけでは開始したことしか分からない。
- 旧形式は集計・条件別指標・manifest等で判定する。実装上の判定は`is_completed_run`を参照。
- clean_onlyの部分再開はモデル重みが保存されていないため、対応する学習とノイズ一式を再計算する仕様。
- 別runの欠測を以前の数値で埋めない。比較表に実行日・hash・評価方針を付ける。
- `metrics_summary_*.csv`は1秒chunkのfold集計。単体モデルと有効な全アンサンブル方式について同じ指標を保存する。

実行ごとの判断材料は[experiments](../experiments/README.md)に固定し、現在の状態は[研究の現在地](research_status.md)で更新する。
