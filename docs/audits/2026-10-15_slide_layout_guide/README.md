# 10/16発表：文章と配置の具体化

記録日：2026-10-15。本人の「出力を見てすぐ作れるように、各スライドの文章・図の位置・全体配置を詳しく」という依頼に対応。

- [貼り付け文章・cm座標・口頭説明・出典](../../progress/2026-10-16_slide_build_guide.md)
- [11枚の配置見本PDF](../../progress/assets/2026-10-16/layout_preview/slide_layout_preview.pdf)
- [全体の一覧画像](../../progress/assets/2026-10-16/layout_preview/contact_sheet.png)
- [全オブジェクトの位置と文章](layout_spec.json)
- [検算](verification.json)／[挿入した元画像のhash](source_hashes.json)

現在の10/16 PPTXを再読し、20×11.25 inch、見出し位置、結論位置、結果ページ・説明性ページの図配置を確認した。原本のhashは前回確認時と同じ。実図の内容と縦横比を確認し、全11枚を結論1文＋主図主体で具体化した。4枚目の右側は横長画像を縮小せず、読みやすい縦3段のnative図形として指定。9枚目には前回の教授指摘と今回の活用先を明記した。

配置見本は作成を助ける図であり、元PPTXの完全なOfficeレンダリングや完成PPTXではない。左の大学装飾を簡略化しているため、実作業では元のマスターを保持する。全11枚の一覧と主要ページを目視確認し、テキストの指定範囲超過・オブジェクトのスライド外配置は0件。元PPTX、元画像、主結果CSVのhash不変を確認した。

再実行194258と143609の数値同等性は[完了済み照合](../../../experiments/2026-10-15_presentation_readiness/run_consistency.json)を引き継いだ。研究結果の追加解析、学習、推論、停止matchedの調査、2026年新実験の評価は行っていない。残るのはPPTXへの実反映と実機での可読性・持ち時間の確認。

再生成：`python docs/audits/2026-10-15_slide_layout_guide/build_guide.py`
