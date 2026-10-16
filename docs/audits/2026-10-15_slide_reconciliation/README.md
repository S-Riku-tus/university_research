# 別AI案・本人の1〜22採否と10/16スライドの照合

記録日：2026-10-15。添付テキスト全文と、会話での本人選択・後続方針・ローカルの結果記録を照合した。

- [判断理由・1〜22対応・11枚の文章と配置](../../progress/2026-10-16_slide_reconciled_proposal.md)
- [再検討案の配置見本PDF](../../progress/assets/2026-10-16/reconciled_preview/reconciled_slide_preview.pdf)
- [一覧](../../progress/assets/2026-10-16/reconciled_preview/contact_sheet.png)
- [根拠とhash](sources.json)／[図の値と高q領域の定義](figure_values.json)
- [配置と全文](layout_spec.json)／[検算](verification.json)

採用した提案は、進展の先出し、旧雑音問題1枚化、旧IGを主証拠にせず共通加工の定量結果を示すこと。q100の全面予備化、境界問題の主題化、教授指摘の明示をなくす変更、後期計画の縮小は採用せず、本人の指定へ調整した。新案も本人の最終採用・教授了承は未確定。

添付のIG「20例中7例」は対象集合を確認できず、そのまま流用しなかった。前回再掲予定の22 kHz・0 dB同一試料は、両モデルとも数値収束済みという[保存記録](../../../experiments/2026-09-18_onb_full_sweep_151715/README.md)がある。入力依存差の確認と、水流雑音利得の物理原因の確定も区別した。

既存CSVから領域別MSE減少、誤差方向群の寄与、既存時間並べ替えのモデル別RMSEの3図を描画した。高q側はONB以上315から近傍30を除いた285標本で、領域間のSSEの差から同じ指標を再描画した。旧図の切り抜き・改変や新しい入力加工・モデル計算はしていない。

全11枚の一覧と主な変更ページを目視確認。文字の指定範囲超過・画面外配置0、リンク26件を確認し、原PPTX・旧作成ガイド・添付原文・参照CSVのhash不変。旧PDF/PNGも保持した。新学習・推論0、停止matchedの再調査なし。実PPTX本文は未編集。

再生成：`python docs/audits/2026-10-15_slide_reconciliation/build_reconciled_proposal.py`
