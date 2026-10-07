# 修論の目的と優先順位の監査資料

2026-10-07。[判断・18課題の台帳](../../research_plan/2026-10-07_thesis_alignment_and_priority_audit.md)の原資料照合と、参照確認の記録。

- `extract_primary_sources.py`：索引から選んだ目的・計画・発表・報告の5原資料について、関連段落/指定スライドだけを抽出する。
- [primary_source_extracts.md](primary_source_extracts.md)：元本文、段落番号/スライド番号。教授口頭発言の原メモとは区別する。
- [primary_source_inventory.json](primary_source_inventory.json)：原資料のパス、hash、抽出範囲。
- [research_task_ledger.csv](research_task_ledger.csv)：本監査本文の18課題を転記した一覧。判断の正本は本文。
- `finalize_audit.py`：課題ID・関連参照・原資料hashを確認し、CSVとinventoryを生成する。
- [source_inventory.json](source_inventory.json)、[verification.json](verification.json)：本文の参照先と確認結果。参照されたファイルすべての内容を全面再検証したという意味ではない。

実行：`python docs/audits/2026-10-07_thesis_alignment/extract_primary_sources.py`、続いて`python docs/audits/2026-10-07_thesis_alignment/finalize_audit.py`。

原Office/PDF、モデル、予測、通常設定は変更しない。本文の主要根拠は保存結果と研究記録であり、未記録の相談や承認を推定しない。
