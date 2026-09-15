# 結果の保存階層と識別

更新日: 2026-09-15。実装の正本は [result_paths.py](../code/utils/experiment/result_paths.py) と [run_helpers.py](../code/utils/experiment/run_helpers.py)。

## 現行の保存階層

```text
Pool_boiling/Subcooling_20_degrees/0.3/<収録実験日>/
  regression_result/npy/ensemble/<解析実行日>_selected_log_architecture[__方針]/
    <周波数上限>/
      <ノイズ条件>/<run>/
        run_manifest.json
        split_manifest.json
        completed.json
        fold_pred/
        wav_eval/
        explainability/
      noise_trends/<run>/<統合方式>/
    tuning_summary.csv
    ensemble_presentation_summary.csv
```

解析日と収録日を区別する。一般化方針の末尾は`__wd_clean / __lodo_matched / __lodo_clean`。日内matchedは従来のまま。

旧runにはノイズ/周波数の順の階層や、完了印・分割manifestのない世代がある。[9/14の移行記録](research_plan/2026-09-14_result_layout_and_generalization.md)を参照し、古い階層を欠落と即断しない。

## run名と実行条件

現行の短いrun名は概ね次の形になる。

```text
e{epochs}_{短縮parameter_tag}_{短縮model_tag}[_{weight_tag}]
e300_active_rf-ctf-alex_ed198_rf-cnntf_v2__cmp
```

**現在のフォルダ名には、run_instance_idや完全なrun_hashが必ず入るわけではない。** 完全な条件と区別は`run_manifest.json`の`created_at / run_instance_id / run_hash / execution_config_hash / validation_config / learning_context`で確認する。同じ名前だから同じ学習内容とは限らない。

主実行は`RUN_ID`環境変数から実行IDを指定できるが、それだけで出力フォルダが別になるとは限らない。旧文書の`RUN_NAME_SUFFIX`は現行主経路では使っていない。別の比較runを残すときは、出力系列・設定の識別と再開判定を確認する。

## 完了・再開・比較

- 新形式は完了時に`completed.json`を保存。manifestだけでは開始したことしか分からない。
- 旧形式は集計・条件別指標・manifest等で判定する。実装上の判定は`is_completed_run`を参照。
- 主方式の表示変更のみなら、保存済みの全方式から再利用できる場合がある。
- clean_onlyの部分再開はモデル重みが保存されていないため、対応する学習とノイズ一式を再計算する仕様。
- 別runの欠測を以前の数値で埋めない。比較表に実行日・hash・評価方針を付ける。
- `metrics_summary_*.csv`はchunkのfold集計、`wav_eval/wav_metrics_*.csv`はWAV評価。指標名だけで混同しない。

実行ごとの判断材料は[experiments](../experiments/README.md)に固定し、現在の状態は[研究の現在地](research_status.md)で更新する。
