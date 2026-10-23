# 最新方針へのLaTeX整合・原資料確認の記録

2026-10-23。本人の「最新文書を基準に、進められる工程を進め、研究全体をまとめ、LaTeX草案を修正してほしい」という追加指示に基づく。

## 方針と反映

- 現在の基準は[研究全体の最新まとめ](../../research_plan/2026-10-23_current_research_summary_and_actions.md)。全域熱流束推定・ONB判定を二つの主要課題とし、改善・悪化の説明を三つ目の問いに置く。
- 緒言の目的・三問と対応表、和英要旨、第3章の評価・上限診断、第4章の回答、第5章の制約と入力依存の説明、第6章の三問への回答を整合させた。
- 元3比の全7条件のRMSE・総FN改善を主要成果とし、開始段階・劣化量・物理的対応の限界を維持した。既存の10/21上限診断を反映したもので、新しいモデル性能実験ではない。
- Uekiら2025・植木2026・Huangら2026を関連研究と参考文献へ追加。公開抄録までの資料と公開本文を確認した資料を区別した。
- 主結果の数値・分割・ラベル・ONB定義・入力・通常学習設定は変更していない。新学習・推論・稼働操作・外部連絡なし。

## 保存した変更前の状態

- [変更前PDF](before_revision_main.pdf)：現在の草稿を改訂する直前のPDF。
- [変更前ソース等25ファイル](before_revision_sources.zip)、[hash一覧](before_revision_manifest.json)。章・要旨・参考文献・表・検査コード等を保持した。
- [変更前の出典manifest](before_revision_source_manifest.json)：元の期待hashを保持する。

本フォルダの変更前資料は10/19版の研究整理を含む、今回直前の草稿であり、提出済み原本ではない。過去の研究資料・発表PPTX・先輩の論文は変更していない。

## 原資料とCNNの一度の保存確認

- [source_metadata_trace.json](source_metadata_trace.json)：主2日のメタデータ6件が日別の元ファイルと全てhash一致。計測CSVの必要ヘッダーから所有者欄・ロガー・チャネルを確認。
- [workbook_formula_trace.json](workbook_formula_trace.json)：対象XLSXのシート構成・範囲・数式セル数。両ファイルとも数式セル0であり、別の算出コード等の確認が必要。
- [出典・原正解の確認票](../2026-10-23_data_provenance_and_confirmation_sheet.md)：確認済み情報、未回答事項、依頼文。取得者・原ONBを推測で補っていない。
- [cnn_saved_progress.json](cnn_saved_progress.json)：12:14 JST、対象2保存先の候補監査JSON 99件、重複除外73候補＝Alex36・ResNet36・VGG1。探索・外側評価の再計算やプロセス監視は行っていない。

## コード参照版の追跡性

草稿の既存検査は、10/19参照時のコードhashを現行ファイルと直接比較していた。後日のCNN対応で変わった2ファイルに不一致があったため、当時の期待hashへ一致する参照版を保存した。[回復記録](historical_code_recovery.json)

- `fixed_core_stacking.py`はGitの`d3589981…`の内容が期待hashと一致。
- `base_regression.py`は同Git版の内容と、現ファイルに残る改行を用いて当時の改行列を復元し、既存のSHA-256 `28ca74eb…`と完全一致した場合だけ採用した。
- 保存先は`LaTeX/master_thesis/data/source_snapshots/2026-10-19/`。元の期待hashは変更せず、manifestに保存先の対応を追加した。現行の研究コードは変更していない。
- 保存版だけに適用する`.gitattributes`で自動改行変換を無効にし、Git経由でも元の期待hashに対応するバイト列を維持する。
- これは草稿参照時点のコードの保存であり、10/15実行時の全ソース・環境を復元したという意味ではない。環境版D06は未確認を保持する。

検査コードへ`--preview-dir`を追加し、今回の画像を過去の10/19監査フォルダへ上書きしないようにした。保存参照版も元の期待hashで検査する。

## 組版・確認

[改訂PDF](../../../LaTeX/master_thesis/build/main.pdf)は57ページ。既存のローカルTectonicで再組版し、[今回の固定した本文検査結果](manuscript_validation.json)を保存した。再実行時の出力は[build内の検査結果](../../../LaTeX/master_thesis/build/validation.json)へ更新される。

- 6章、引用23件、ラベル45件、図4件、生成済み表10ファイル、要確認ID16件の参照が整合。
- 元の主要数値、77行の指標表の母数、出典25件のhash、原参考資料6件のhashが合格。後日変更の2コードは保存した参照版と照合。
- 未解決参照・重複ラベル・本文の文字欠落・ページ外テキスト・Overfullなし。
- PDFは空ページ、置換文字、未解決参照記号を検出しなかった。目的・上限診断・結言等の代表ページを目視確認した。
- TeXには参考文献のUnderfull等の警告が残るが、本文・図表の欠落やページ外表示は確認されていない。

再現する場合は、リポジトリ直下で次を実行する。素材の再生成・モデル実行は不要。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File LaTeX/master_thesis/build.ps1
python LaTeX/master_thesis/scripts/check_manuscript.py --pdf --preview-dir docs/thesis/2026-10-23_manuscript_alignment_audit/preview
```

教授の確認、原データ取得者とONB原根拠、未読文献本文、正式様式等は残る。今回の組版・検査合格は、研究上の全目標達成や提出稿の完成を意味しない。
