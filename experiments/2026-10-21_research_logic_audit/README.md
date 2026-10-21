# 研究全体の再検討に伴う保存証拠の軽量監査

2026-10-21。[研究方針の再検討](../../docs/research_plan/2026-10-21_research_logic_reassessment_user_feedback.md)と[研究活動台帳](../../docs/research_plan/2026-10-21_research_activity_evidence_inventory.md)の数値根拠。原結果・設定・ラベルを変更せず、この新規フォルダだけへ集計を保存した。

## 対象と検証

主対象は10/15 clean_only `143609`、hash `da540f2d`、fit `7560205f8feb`。2025/6/11＋6/18、36 WAV、1秒3 kHz、各45学習/15評価、1620/540、内部shuffle chunk3-fold、seed42、200 epochs、選別なし、7評価条件。原保存先は[主条件書](../2026-10-15_clean_only_result_analysis/conditions.json)を参照。

[audit_saved_evidence.py](audit_saved_evidence.py)はモデルをimportせず、CSV/JSONと原DOCX/PPTXの選択テキストを読む。raw NPY・モデル重みを読み込まず、fit・推論・重みの選択を行わない。Windowsの長いパスは拡張パスで読み取る。

- 主11方式×7条件の回帰・判定主要値を原予測から再集計。77行、保存表との最大差4.2633×10⁻¹⁴。
- 6統合×7条件を保存単体予測と重みから再構成。最大差2.3284×10⁻¹⁰ W/m²。
- 日別q100の154件を再計算し保存値と一致。
- 元10/15検算の入力35件のhashを照合し全件一致。原本の変更はない。
- OOFの学習1620 IDと外側540 IDの同一chunk共有0。元36 WAVは共有。記録上ONBは2本、各45学習/15評価、計90/30。
- CNNの元停止59監査のhashは全件一致。17:27 JSTの実保存監査は、元run Alex36/ResNet23に加えて再開先ResNet1候補が増え、重複除外後Alex36/ResNet24/VGG0。稼働中か停止中かは判定していない。未来の進捗を表す固定値ではない。

詳しい対象パス・hash・照合結果・時点は[verification.json](verification.json)。matched停止run、新しい2026日等の原予測はこの再集計の対象外。

## 出力

| ファイル | 内容 |
|---|---|
| [recomputed_primary_metrics.csv](recomputed_primary_metrics.csv) | 77モデル/条件行。ONB以上315を`rmse_high`、近傍を除く高q285を別列として保存 |
| [recomputed_q100.csv](recomputed_q100.csv) | 154日別到達点。元定義のまま、誤報0条件は追加しない |
| [onb_stage_partition.csv](onb_stage_partition.csv) | ONBに対応する2 WAVの90学習/30評価区間のID。母数の由来を追跡 |
| [near_band_sensitivity_counts.csv](near_band_sensitivity_counts.csv) | 日別ONB±5/10/15/20/25/30%の件数と測定段階。主定義の変更・最適幅選択はしない |
| [onb_failure_feasibility.csv](onb_failure_feasibility.csv) | clean/−20の5統合FNについて単体陽性の有無・現制約の1標本上限 |
| [primary_document_extracts.md](primary_document_extracts.md) | 計画DOCX2件・5/22/6/19/7/24/10/16の選択スライドのXMLテキスト。原資料hashは検算JSONに保存。図を含む全ページ目視監査ではない |
| [document_validation.json](document_validation.json) | 新規Markdownのローカルリンク、必須構成、差分等の点検記録 |

## 制約下の見逃し診断の意味

5単体全ての陰性/一部陽性と、固定元3ブロックの重み変更で救えることは異なる。元3内部比率固定、元3総量≥0.25、HGB/ET各≥0.05、和1なら、各標本の最大予測は

`0.25 * original3_mse + 0.05 * HGB + 0.05 * ET + 0.65 * max(original3_mse, HGB, ET)`

となる。[現重み実装](../../code/utils/ensemble/fixed_core_stacking.py)と[主条件](../2026-10-15_clean_only_result_analysis/conditions.json)に対応する解析的な上限であり、採用可能な新方式の成績ではない。

| 条件・範囲 | FN | 全単体陰性 | 一部単体陽性 | 1件ずつの救済可能数の上限 |
|---|---:|---:|---:|---:|
| clean全FN | 27 | 11 | 16 | 8 |
| clean開始段階FN | 22 | 9 | 13 | 6 |
| −20全FN | 52 | 0 | 52 | 2 |
| −20開始段階FN | 30 | 0 | 30 | 0 |

この上限は標本ごとに異なる最良配分を許す。共通配分で同時に救えること、他のFN/FPを増やさないことは検証していない。外側正解を用いる診断であり、外側の救済配分を学習・採用していない。全単体陰性も音響情報不在の証明にはならない。

## 実施範囲

主実行・探索入口は起動していない。新学習、新推論、入力加工、新モデル・新統合・新XAIの実装、既存保存結果や設定の上書きは行っていない。以前の検算・説明性対照を未完了へ戻さず、原予測の軽量照合と今回の判断に必要な件数・制約診断だけを行った。
