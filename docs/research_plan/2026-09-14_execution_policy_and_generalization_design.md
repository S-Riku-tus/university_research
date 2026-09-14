# 今週の実行方針と一般化評価の実装案

追記: この設計は本人の承認後に実装済み。現在の保存先・config・検証結果は[保存階層の変更と一般化評価](2026-09-14_result_layout_and_generalization.md)を参照する。以下は承認前の方針確認の記録である。

2026年9月14日の方針確認では、モデル比較、ノイズ生成の修正、元WAVを分離する学習・検証、log-power入力への改善、chunkと元WAVの両評価の整備までを完了済みの前提とした。今後はこれらの作業をやり直す計画へ戻さず、現在の基盤で結果を出し、次の研究課題へ進む。

今週の発表では、現行ONBコードの全対象条件を実行し、ノイズ強度を変えたときの予測性能と説明性の出力を示す。対象は明示的な選択により、2025.06.11と2025.06.18の2実験日、5周波数上限、7ノイズ条件、合計70条件とする。07.09は今回は有効化しない。学習の起動はコード修正の完了後に本人が行う。今回の作業では学習を起動していない。

同じ学習・検証予測から、chunk単位と元WAV単位の両方を評価する。両評価のためにモデルを2回学習するわけではない。元WAV分離inner holdoutでは、アンサンブル重みを決める内部学習が別に存在するが、これは評価単位を二つ用意することとは別の処理である。

| 評価単位 | 計算方法 | 保存先 |
|---|---|---|
| 1秒chunk | 各foldの検証chunkで指標を計算し、fold平均・標準誤差を保存 | 各runの `metrics_summary_<SNR>.csv`、`validation_results_<SNR>.txt` |
| 元WAV | 全foldの学習外予測を集め、各元WAVのmean/median/p90予測で指標を計算 | 各runの `wav_eval/wav_metrics_<SNR>.csv` |

主WAV集約は `evaluation.primary_wav_aggregation = "median"`。`evaluation.wav_level_enabled = True` の現設定で両方出力する。WAV指標はfold別R²の平均とは異なるため、chunkの標準誤差をWAVの点へ付けない。

今回追加したノイズ別グラフは、[主実行コード](../../code/run_ensemble_regression_onb.py)から `RegressionPlotter.plot_noise_trends()` を呼び出し、[共通処理](../../code/utils/plotting/noise_trend_plots.py)で作成する。既定では単体3モデル＋主アンサンブルの4本で、実験日・周波数上限・パラメータ設定・評価単位ごとに分ける。

設定は `VALIDATION_CONFIG["output"]["noise_trend_plots"]` に置いた。

```python
"noise_trend_plots": {
    "enabled": True,
    "ensemble_strategy_names": "primary",
    "metrics": ["r2", "roc_auc_cont", "auc_binary"],
    "evaluation_units": ["chunk", "wav"],
    "formats": ["png", "pdf"],
}
```

`primary` は `ensemble.primary_strategy_name` に従い、現在は `simple_equal`。`all` にすると各統合方式を別図で保存し、それぞれの図を単体3モデル＋1統合方式の4本に保つ。方式名のリストを指定することもできる。

R²のほかにAUCを2種類出す。`roc_auc_cont` は予測熱流束を連続スコアとして使うROC-AUC、`auc_binary` は予測を閾値で二値化した卒論互換の参考値である。図の縦軸名とファイル名でも区別する。

横軸は `No_noise, 0, -4, -8, -12, -16, -20` とし、右ほどノイズが強い順に並べる。現行データでは各値は固定基準RMSに対するreference SNRである。値の並び方を再現するもので、単調に低下するよう数値や軸を加工しない。未完了条件は欠測とし、線を途切れさせる。

出力先は各実験日の次の場所になる。

```text
regression_result/npy/ensemble/<実行日>_selected_log_architecture/
  noise_trends/<maxfreq>/<run_dir>/<統合方式>/
    chunk_fold_mean_r2.png / .pdf / .csv
    chunk_fold_mean_roc_auc_cont.png / .pdf / .csv
    chunk_fold_mean_auc_binary.png / .pdf / .csv
    wav_median_r2.png / .pdf / .csv
    wav_median_roc_auc_cont.png / .pdf / .csv
    wav_median_auc_binary.png / .pdf / .csv
```

各ノイズ条件が完了するたびに保存済みCSVから曲線を更新する。完了条件をスキップする再開経路でも作図する。同じ実験・周波数・run設定の結果だけを使い、以前の設定と異なるrun hashは混ぜない。作図設定は出力設定として扱うため、それだけを変更しても学習済み結果の判定用hashは変えない。全条件を終えた既定設定では、2日×5周波数×6種類＝60図、それぞれPNG・PDF・数値CSVを得る。

既存の損失曲線、棒グラフ、散布図、説明性図も引き続き出力される。英語と文字化けが混在していたONB本体の説明コメント・冒頭説明を日本語にした。変数名、CSV列名、モデル識別子は互換性を維持するため従来の表記を使う。

未知条件への評価は**今回は未実装**であり、次の案を検討してから実装の可否を決める。

設定は「実験日の分け方」と「学習時のノイズ」の2軸に分ける。目的が異なるため、ひとつの長いmode名へまとめるより組み合わせを明示した方が理解しやすい。

```python
# 実装予定の例。まだ現行コードへ追加していない。
"learning_policy": {
    # within_day: 実験日内で元WAVを分離する現在の交差検証
    # leave_one_day_out: 1実験日全体を学習外にする
    "split_mode": "within_day",
    # matched: 評価と同じノイズ条件で学習する現在の方式
    # clean_only: 無雑音だけで学習し、その同じモデルを各ノイズ条件で評価する
    "training_noise": "matched",
}
```

| split_mode | training_noise | 学習・評価の意味 |
|---|---|---|
| within_day | matched | 現行方式。同日の元WAVを分離し、各SNRで学習・評価 |
| within_day | clean_only | 同日の学習側WAVの無雑音だけで学習し、学習外WAVの各ノイズ版を評価 |
| leave_one_day_out | matched | 他の実験日の同じSNRで学習し、1日丸ごと評価 |
| leave_one_day_out | clean_only | 他の実験日の無雑音だけで学習し、学習外の日の各ノイズ版を評価 |

既定値は `within_day + matched` とし、現在の実行を維持する。評価単位のchunk/WAVはこの設定と独立で、どの学習方針でも両方保存する。

実装では、現在の「1条件＝1つの読込フォルダ」というjobを、「学習用の条件一覧」「評価用の条件一覧」「元WAVの分割計画」を持つjobへ拡張する。[dataset_jobs.py](../../code/utils/experiment/dataset_jobs.py)で条件を組み立て、[データ読込処理](../../code/utils/dataloading/dataloading_and_conversion.py)を複数条件に対して利用する。モデル定義、指標計算、作図はできるだけそのまま再利用する。

元WAVの識別には `(実験名, source_wav_id)` の組を使う。日が違うのに同じID文字列だった場合の衝突と、同じWAVのノイズ違いが学習・検証へ分散することを防ぐ。元WAVの分割を先に決め、その後で各WAVのclean/noisy版を対応付ける。SNRごとに別々のランダム分割を作らない。

`clean_only` では、各fold・周波数・パラメータ設定について学習を1回行い、同じモデル・PCA・目的変数scaler・アンサンブル重みを固定したまま7ノイズ条件へ予測する。ノイズ条件ごとに再学習してしまうと現在方式と同じ問いになるため、この学習の再利用が中心となる。内部holdoutも学習側のcleanデータだけで行う。

`leave_one_day_out` では、対象実験日を順番にテスト側へ回す。3日指定なら2日で学習・1日で評価、2日指定なら1日で学習・もう1日で評価する。現在の `run.folds` は日内GroupKFold用とし、日ごとの外側分割数は指定した実験日数から決める。PCA、目的変数scaler、重みの選択は学習日に閉じ、テスト日の正解は指標計算だけに使う。

ONB判定の正解には評価対象日の出典付き閾値を使う。これは保存データに対する研究上の評価であり、未知の実験日に事前情報なしで閾値を決定できたという意味ではない。実運用の閾値決定は別の研究課題として区別する。

保存先とmanifestには、学習方針、学習日、評価日、学習ノイズ、評価ノイズ、元WAV分割の対応を追加する。再開判定のhashにも方針を含め、異なる方針の成績を同じ結果として再利用しない。既存のノイズ曲線は、同一学習方針・評価日ごとに集計するよう接続する。

実装を承認した後は、最初に `within_day + clean_only` を小さい代表条件で通し、学習回数がノイズ条件数に比例して増えないこと、各評価ノイズで同じ学習外WAVを使うことを検証する。その後に `leave_one_day_out` を追加し、実験日が学習・評価に重複しないことを確認する。どちらも小規模検証の後で全条件へ広げる。

今回の検証は作図用5件を含む24テスト、実データ7ノイズ条件からの6図出力、70条件の存在確認である。一般化評価のコード追加、本学習、説明手法の変更は行っていない。
