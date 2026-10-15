# 10/16発表用ONB出力の実行条件と棒グラフ

記録日：2026-10-15。条件の提案後、本人の「＋以降は同じ行」「ONBのコードもその条件に合うように修正」により、下記条件を通常入口へ反映した。棒グラフの書式と実設定の入力を確認済み。本学習・モデル推論は未実行。

**同日その後の完了結果**：本人がclean_onlyとmatchedを実行し、matchedは途中停止と連絡。最新依頼により[clean_only `143609`だけの分析](../../experiments/2026-10-15_clean_only_result_analysis/README.md)を完了した。全7条件、231指標行と保存統合予測が一致。分析図と、下記の抜粋/拡大bar168図も別フォルダへ保存済み。matchedの分析/再実行は今回行わない。下段は設定準備時点の経緯。

## 今回の実行の目的

今回行うなら、既存の主条件で発表用の出力を揃える通常実行とする。新しい研究条件の探索、入力からの自動重み切替、別日/新2026年検証、修論完成後の最終確認には広げない。

目的が配色・文字・配置の更新だけなら、**保存済み予測/指標からの再描画が第一候補**。新しい学習をせず、10/14に分析した保存fitと数値を保って図を更新できる。今回のbarプレビューも同じ方式で作成した。

本人が通常ONBを再実行する場合は、以下の最新200 epochs条件を保つ。同じseedでも保存値との完全一致を保証せず、新runの結果として保持する。主結果を新fitへ替える場合、元fitの入力加工や誤り分析を新fitの直接検証として混ぜず、runを明記して整合を確認する。新runが良かったか悪かったかで都合のよい数値を継ぎ接ぎしない。

## 推奨条件

| 項目 | 推奨値・理由 |
|---|---|
| 使用データ | 2025/6/11＋6/18の既存統合フォルダ`2025.06.11_0.3_2_6.18_0.3_3` |
| 入力生成版 | `waterflow_20260817_1s`を維持。再生成なし |
| 外側評価 | `within_wav_chunk`、各WAVの25%を評価、split seed42 |
| 入力 | 1秒・3 kHz・1ch、選別なし |
| 学習雑音 | `clean_only`。無雑音で学習したモデル/PCA/scaler/重みを全雑音へ固定 |
| 評価雑音 | clean、0、−4、−8、−12、−16、−20 dBの7条件 |
| 学習 | 最新主結果と同じ200 epochs、内部`chunk_kfold`3分割、学習seed42、CPU threads2 |
| 基モデル | RF、CNN＋Transformer、AlexNet、HGB、ExtraTreesの5種 |
| 統合 | 元3 MSE、元3 performance、元3＋HGB、元3＋ET、5、5等重みの6方式 |
| モデル設定 | 現在の3 kHz採用値を維持し、再tuningしない。PCA100、batch等は適用済み設定JSONへ記録 |
| 重み | 現行のclean学習側OOF MSEと保持制約を維持。目的・自由度の変更なし |
| ONB | 6/11＝221.5051102、6/18＝271.6776816 kW/m²の日別閾値で主表を読む |
| 保存 | fold予測・モデル・前処理・重みを保存、再読込照合を維持 |
| 主比較表 | `main_comparison_report=True`、`report_source_day_metrics=True` |
| 説明性 | `enabled=False`を維持。今回は通常結果の出力を先に揃える |

元3/4/5ごとにCNNを別々に再学習するのではなく、同じ5単体の予測・OOFを共有して統合する。matched追加、7/9追加、22 kHz追加、epoch/seed探索は今回の必須条件にしない。7雑音を評価するがclean_onlyなので雑音ごとに7回の学習をしない。

200 epochsを勧める理由は、10/14までに分析した最新主結果と合わせるため。旧主比較プロトコルv1は150 epochsなので、200では`standard_configuration=False`となるが、要求した200 epochsの完了とは別。表示に合わせて150へ戻したり、旧実験プロトコルを上書きしたりしない。

## 通常入口への反映と入力確認

実ファイルの`VALIDATION_CONFIG`に以下を反映した。非選択の`cross_day`欄は保持するが、今回の実行には使用しない。

1. `learning_policy.evaluation_mode`を`within_wav_chunk`。
2. `learning_policy.evaluation_settings.within_wav_chunk.experiment`を2025年統合フォルダ。
3. `output.main_comparison_report=True`。
4. 閾値設定は2025年の既存日別値と出典を使用し、新2026年CSV取得を今回必要にしない。

[元の提案JSON](../../configs/experiments/2026-10-15_onb_presentation_run_proposal.json)は提案時点の記録として保持し、[適用済みの設定・解決後設定](../../configs/experiments/2026-10-15_onb_presentation_run_applied.json)を別途保存した。通常入口はJSONを自動読込せず、Pythonファイル内の設定を使用する。適用記録は`applied_not_run`であり、新runの完了記録ではない。

実際の入口をimportし、設定検査・入力job生成・manifest/ファイル名の検査を実施。7/7条件、36 WAV、2160入力、1620学習/540評価、全雑音の評価ID/正解・日別ONBが一致し、提案時の入力確認と同一。[適用後の確認結果](../../experiments/2026-10-15_bar_label_readability/applied_run_input_check.json)。NPY配列の再読込、学習、モデル推論、学習結果保存なし。入口の関連テスト5件も合格。

再確認：`python experiments/2026-10-15_bar_label_readability/verify_applied_config.py`。設定とメタデータを確認し、適用記録のみ更新する。旧`prepare_run_advice.py`は提案時の記録を作るスクリプトで、現在の確認手順には使わない。

現行の通常入口は`--preflight`を解析しない。その引数を付けても事前確認専用にはならないため、撤回前の保存fit入口の説明を使わない。通常実行は`python code/run_ensemble_regression_onb.py`。

## 発表に使う出力

- `bar/`：通常のR²・ROC/PR-AUCの棒グラフ。横軸モデル名は今回拡大した。
- `noise_trends/`：現在のモデル別配色と線種で保存される雑音曲線。
- `onb_comparison/`：元3/4/5・単体の全域/近傍RMSE、日別q100、見逃し/誤報、悪化量を確認する主表。

通常のROC/PR棒グラフは従来の統合平均ONBに基づく指標で、日別ONBを使う主表とは正解基準が異なる。R²棒グラフはONB閾値に依存しない。発表の主ONB値は日別表から採り、見逃し・q100と平均閾値AUCを同じ定義として扱わない。

過去の逆転診断図と修正後matched曲線は別条件の既存結果を使う。今回のclean_only通常実行だけで過去の診断図まで新しくなるわけではない。

## 棒グラフの修正と確認

[plot_bar](../../code/utils/plotting/regression_plots.py)の横軸モデル名を、モデル数によらず縦軸ラベルと同じ23 ptにした。5モデル統合は`Original3`／`+ HGB + ET`の2行とし、＋以降を同じ行にまとめた。モデル数と行数に応じて図を広げ、使用フォントで実測した文字幅が必要とする場合だけ追加の横幅を確保する。配色・指標値・棒の並び・保存ファイル名は維持する。

保存済みの実数値で5/11方式×Times New Roman/DejaVu Sansの4図を確認。23 pt一致、ラベルの非重複、図内への収まり、棒の値不変、元CSV hash不変を検査した。[確認記録とプレビュー](../../experiments/2026-10-15_bar_label_readability/README.md)。ここまでは設定と表示の確認で、本学習は起動していない。その後、本人が通常ONBを実行開始した。

## 実行開始後の追加依頼：数値と比較構成

本人から、現在のONBの分析は実行終了後と指定された。今回の修正では実行中runを読まず、作図処理だけを更新した。[追加図・根拠・プレビュー・再描画手順](../../experiments/2026-10-15_bar_presentation_views/README.md)。

- 棒の数字を一律30 ptへ拡大。元の11方式表示に、`legacy4`（元3単体＋original3_performance）、`addition4`（元3 MSE→HGB4/ET4/5）、`integration5`（元3 MSE/HGB/ET/5等重み/5）を追加。
- 各構成に`*_zoom.png`を追加し、拡大表示と軸省略を明示。全平均/SEを含む範囲で表示し、保存値は変更しない。
- 10/6の既存完了結果だけで48図と関連5テストを確認した。今回の実行結果の分析は保留。
- 次回の通常ONBでは追加図を自動保存。起動済みPythonへの自動反映は保証しないため、今回の実行終了後は`python code/replot_regression_bars.py --run-dir "完了したrunのフォルダ"`で再学習せず別フォルダへ再描画できる。
