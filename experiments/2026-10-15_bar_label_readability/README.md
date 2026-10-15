# barのモデル名拡大と発表用ONB条件の読み取り確認

記録日：2026-10-15。本人の要望により、[通常棒グラフ](../../code/utils/plotting/regression_plots.py)のモデル名を23 ptへ拡大した。

- 縦軸ラベルと同じ23 pt。モデルが多いと13 ptへ縮める従来分岐を廃止。
- 長い名前は短縮名の意味を保って改行。本人の追加指定を反映し、5モデル統合は`Original3`／`+ HGB + ET`の2行。
- 水平表示、モデル数/改行数に応じた図サイズ、実測文字幅に応じた追加横幅。
- 元の色、数値、保存名、棒の順序は保持。

[11方式のプレビュー](times_11/bar/r2_ep200_no_noise.png)、[5モデルのプレビュー](times_5/bar/r2_ep200_no_noise.png)。保存済みの最新2025年runの通常R²棒グラフ指標を再描画したもので、新しい性能評価ではない。元CSVのSE表記をそのまま使っており、SE=0を不確かさゼロの新しい根拠にはしない。

[確認結果](verification.json)：5/11方式、Times New Roman/DejaVu Sansの計4図。モデル名23 pt＝縦軸23 pt、ラベル間隔最小22.10 px（描画時）、全ラベル図内、棒の値不変、元CSV hash不変。5モデル統合の2行表示を機械検査し、PNGの改行と横幅も目視確認。

再描画確認：`python experiments/2026-10-15_bar_label_readability/verify_layout.py`

別途、本人への条件説明に使う[提案JSON](../../configs/experiments/2026-10-15_onb_presentation_run_proposal.json)と[入力確認](proposed_run_input_check.json)を作成。2025年統合、日内clean_only、200 epochs、全7雑音の入力が36 WAV/2160入力/1620学習/540評価でそろい、全雑音の評価ID/正解が一致。これはファイル名・manifestによる確認で、入力配列の内容再走査ではない。通常入口はimport・変更せず、新学習/推論0。

その後、本人の追加指示で通常入口へ条件を適用。[適用済み設定](../../configs/experiments/2026-10-15_onb_presentation_run_applied.json)と[適用後入力確認](applied_run_input_check.json)を別途保存した。2025年統合・within_wav_chunk・clean_only・200 epochs・主比較表有効で、上記提案と全条件・入力検査が一致。関連テスト5件合格。学習・推論は未実行。

適用後の確認：`python experiments/2026-10-15_bar_label_readability/verify_applied_config.py`。[実行条件の説明](../../docs/notes/2026-10-15_onb_presentation_run_conditions.md)を参照。
