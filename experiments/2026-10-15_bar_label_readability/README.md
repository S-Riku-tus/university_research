# barのモデル名拡大と発表用ONB条件の読み取り確認

記録日：2026-10-15。本人の要望により、[通常棒グラフ](../../code/utils/plotting/regression_plots.py)のモデル名を23 ptへ拡大した。

- 縦軸ラベルと同じ23 pt。モデルが多いと13 ptへ縮める従来分岐を廃止。
- 長い名前は短縮名の意味を保って改行。例：`Original3`／`+ HGB`／`+ ET`。
- 水平表示、モデル数/改行数に応じた図サイズ、実測文字幅に応じた追加横幅。
- 元の色、数値、保存名、棒の順序は保持。

[11方式のプレビュー](times_11/bar/r2_ep200_no_noise.png)、[5モデルのプレビュー](times_5/bar/r2_ep200_no_noise.png)。保存済みの最新2025年runの通常R²棒グラフ指標を再描画したもので、新しい性能評価ではない。元CSVのSE表記をそのまま使っており、SE=0を不確かさゼロの新しい根拠にはしない。

[確認結果](verification.json)：5/11方式、Times New Roman/DejaVu Sansの計4図。モデル名23 pt＝縦軸23 pt、ラベル間隔最小19.52 px（描画時）、全ラベル図内、棒の値不変、元CSV hash不変。PNGを目視し、改行と横幅を確認。`py_compile`も成功。

再描画確認：`python experiments/2026-10-15_bar_label_readability/verify_layout.py`

別途、本人への条件説明に使う[提案JSON](../../configs/experiments/2026-10-15_onb_presentation_run_proposal.json)と[入力確認](proposed_run_input_check.json)を作成。2025年統合、日内clean_only、200 epochs、全7雑音の入力が36 WAV/2160入力/1620学習/540評価でそろい、全雑音の評価ID/正解が一致。これはファイル名・manifestによる確認で、入力配列の内容再走査ではない。通常入口はimport・変更せず、新学習/推論0。

推奨条件と現在の入口との差は[実行条件の説明](../../docs/notes/2026-10-15_onb_presentation_run_conditions.md)へ記録した。設定案は未適用で、現在の通常入口はまだ2026年を評価するcross_dayのまま。
