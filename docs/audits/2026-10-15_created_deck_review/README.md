# 本人作成の10/16修正版を実描画してレビュー

対象は`研究進捗_1016_編集用_修正版.pptx`、11枚、確認時点の更新日時2026/10/15 23:18:19。本人が明示した修正版を選び、原本と同じhashの`source_readonly_snapshot.pptx`をPowerPointの読み取り専用・ウィンドウ非表示で開き、全ページをPNGへ書き出した。ユーザーの元のプレゼンテーションは閉じたり変更したりしていない。

- [レビュー本文と修正案](../../progress/2026-10-16_created_slide_review.md)
- [実スライドの一覧](actual_deck_contact_sheet.png)／[各ページ](rendered_slides/)
- [本文・表・ノート](deck_text.md)／[配置とメディアの抽出](deck_evidence.json)
- [数値23項目の照合](numeric_verification.json)
- [既存図の挿入位置](figure_insertion_map.csv)／[画像出典と寸法](figure_insertion_map.json)
- [原本不変・描画・確認範囲](verification.json)

4・6・7・8・9枚目の図6枠は挿入待ち、11枚目は旧後期計画。主要表示数値は既存CSVと一致。q100の範囲/無雑音条件、元3 MSE、3つの予測の内訳、MSE寄与、1秒内の時間加工、強雑音を−20へ限定する表現を修正提案にした。新しい研究課題や除外した論点を独立ページへ戻していない。

最初の通常サンドボックスではPowerPointのCOM接続がWindowsログオンセッションの制限で失敗した。読み取り専用コピーからの描画を通常ユーザーセッションで実行し、全11枚の書き出しが成功した。これは描画の実行環境の差であり、原PPTXの不具合ではない。

再抽出/描画：`python docs/audits/2026-10-15_created_deck_review/inspect_and_render.py`

数値と挿入位置の確認：`python docs/audits/2026-10-15_created_deck_review/verify_content_and_map_figures.py`
