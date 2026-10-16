# 本人再編集14枚版：モデル・評価方針・目次の確認

対象：10/16 11:29:55更新の修正版。原本SHA256は`853dee396cf3c7ecf3f23b8550c68e364d2f997f17a23540995ca2bbe95747e7`。

- `inspect_current.py`で同一コピーをPowerPointから読み取り専用で開き、全14枚を出力。5・6・11の表示と本文、13の目次・14の旧計画の本文を確認した。元PPTXは変更していない。
- `deck_evidence.json`、`deck_text.md`、`rendered_slides/`を保存。旧9のONB結果削除と、現在6の評価分割削除を確認。
- 本人の最新の採否に従い、ONBページや前回提案の到達点ページを自動で追加しない。
- 5の「二つの不具合」を、旧生成の修正と評価目的の選択に分ける案を作成。matched自体を不具合としない。
- 6は追加HGB/ETの入力と学習方式、採用理由を説明する案。11はその入力依存を確認し、誤り分析と併せて研究へ用いる説明案。
- 学部卒論PDFのSHA256は`d91cc8167d16c412cbe8c7223f218a8f913b3b70e119f25d26b1aa8d47a88e63`。以前の`source_text.jsonl`の同一hashの本文から目次PDF p.3〜4を確認し、`bachelor_toc_verified.txt`へ抜粋した。今回の添付領域では目次画像を取得できず、PDF画像の新規描画は行っていない。
- 既存完了matched／clean_only対照の2曲線図を作成。現在の200 epochs主結果と、旧150 epochs対照を混ぜない。途中停止matchedの読込・再実行なし。
- 新規モデル学習・推論・入力加工は0。元コード・元結果・PPTXは保持。

[本人向けの修正文・比較ページ・配置](../../progress/2026-10-16_models_policy_feedback_and_outline_revision.md)／[修論詳細目次](../../research_plan/2026-10-16_detailed_master_thesis_outline.md)／[比較図の根拠](../../../experiments/2026-10-16_presentation_policy_comparison/README.md)。
