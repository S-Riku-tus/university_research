# 元WAV・ONB遷移単位評価の実装記録（2026-09-08）

## 目的

従来の1秒chunk評価は、同じ元WAVから切り出した60個前後のchunkを別標本として数えるため、
収録単位の標本数と性能の安定性を過大に見せる可能性がある。そこで、chunk予測は局所的な揺れを
見る診断として残し、研究上の主評価を元WAV単位へ追加した。

## 実装した評価階層

1. **chunk単位**: 従来指標を保持。WAV内の予測分散・瞬間的なしきい値交差の確認に使う。
2. **元WAV単位**: outer GroupKFoldの全OOF予測を集め、同じ `source_wav_id` の予測を
   mean/median/p90で集約する。主値は外れchunkに比較的頑健なmedianとする。R2等はfold別R2の
   単純平均ではなく、全OOF WAVを一度だけpoolして計算する。
3. **ONB遷移単位**: WAVを真の熱流束順に並べ、ONB閾値を初めて超える測定点と予測遷移点の差を、
   operating-point stepおよび熱流束差で表す。単発超過（1 WAV）と持続超過（2 WAV連続）を併記する。

秒単位の検知遅れと気泡イベントprecision/recallは、同期した独立正解が存在しないため算出しない。
WAV内のしきい値交差CSVは、予測熱流束の診断であって物理イベントの正解ではない。

## 修正した妥当性上の問題

- ONB閾値を「ONB直前の測定点」から「ONBと確認された最初の測定点」へ修正し、出典と一元管理した。
  - 2025.06.11: `368978.105`（報告値 `3.69e5`）
  - 2025.06.18: `376320.1201`（報告値 `3.76e5`）
  - 2025.07.09: `442169.29243200656`（報告値 `4.42e5`）
- 旧fold CSVの `y_true` は10桁へ丸められ、ONBと等しい値が微小に閾値未満になる場合があった。
  後処理では `chunk_manifest.csv` の高精度 `heat_flux` を正として読み直し、新規CSVは17桁で保存する。
- `inner_holdout` の重み決定をsampleランダム分割から元WAV非重複分割へ変更し、重み用誤差も
  WAV median予測で計算するようにした。
- 周波数帯・時間帯マスク後のXAI性能比較も、同じ元WAV median単位で計算するようにした。
  これにより、マスク重要度だけがchunk数の多さに支配される不整合を避ける。
- outer-validation正解を使う `val_fold_legacy` を既定の比較・主方式から外し、
  ラベルを使わない `simple_equal` を既定主方式とした。旧結果の後処理時は、legacy方式と旧sample単位
  inner holdoutを `claim_safe=0` と明示する。

## 保存物

各runの `wav_eval/` に以下を保存する。

- `oof_chunk_predictions_*.csv`: 元WAV・時刻情報付きOOF chunk予測
- `wav_predictions_*.csv`: WAV集約予測、WAV内ばらつき、予測しきい値交差
- `wav_metrics_*.csv`: pooled OOF WAV回帰・ONB判定指標
- `onb_transition_summary_*.csv`: ONB遷移の早期/遅延、誤警報、持続条件
- `predicted_event_summary_*.csv`: WAV内の予測交差診断（イベント正解ではない）
- `evaluation_manifest_*.json`: 評価単位、閾値、主集約、解釈上の注意

## 適用・検証結果

9/3保存結果へ再学習なしで適用し、3実験日それぞれ35条件、合計105条件を処理した。

- 2025.06.11: 35/35、各条件 1080 chunks → 18 WAVs
- 2025.06.18: 35/35、各条件 1080 chunks → 18 WAVs
- 2025.07.09: 35/35、各条件 780 chunks → 13 WAVs
- skipped: 0

構文確認と `python -m unittest discover -s tests -p "test_*.py" -v` を実施した。
今後の学習runでは同じ出力が本体スクリプトから自動保存される。
XAIマスク性能のWAV集約は学習中のmasked推論を必要とするため、保存予測だけで作れる105条件の
後処理には追加しておらず、次回の対象条件実行から反映される。

## 解釈上の残課題

WAV単位化によりchunkの擬似反復は抑えられるが、同じ実験日のWAV同士まで独立になるわけではない。
未知実験日への一般化を主張するには、3日をまとめたleave-one-experiment-day-out評価が次段階として必要である。
また、秒単位の早期検知を主張するには、動画・目視・計測信号などと同期したONB時刻ラベルが必要である。
