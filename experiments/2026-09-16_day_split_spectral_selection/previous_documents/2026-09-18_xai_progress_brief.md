# 9月18日進捗発表：説明性の出力・解釈・まとめ

更新日: **2026-09-15**。本人の追加希望を受け、主材料を**06.11・3/22 kHz・各7ノイズの比較**へ更新した。完成PPTXや発表実施記録ではない。ONB専用設計・新実験・同時計測は発表後に置く。

根拠: [最新比較・原因分析・全図表](../../experiments/2026-09-15_onb_frequency_comparison/analysis.md)、[本人の研究仮説](2026-09-15_master_thesis_hypothesis.md)。[SOAP下書き](../progress/2026-09-18_weekly_progress_draft.md)も完了済み2帯域の比較・本人の仮説へ更新した。

## 今回伝える中心（追加比較後）

> 学部で観察したノイズによる劣化とアンサンブルによる劣化抑制が、どの条件で成立するかを調べている。3 kHzではCNNのR²が概ね低下し、inner holdoutは5/7条件で最良単体を上回った。22 kHzにも劣化と補完があるが、途中の回復が大きい。RFは両帯域でほぼ横ばいであり、帯域内SNRとマスク依存の違いが手がかりとなった。一方、回帰の統合改善はONB検知の改善と一致せず、IGの不整合も両帯域に残る。今回の結果と原因を確かめるための比較を分けて報告する。

## 最新の発表構成（7枚案）

| 枚 | 主題・示すもの | 一番伝えること |
|---|---|---|
| 1 | 本人の仮説、学部からの発展、同じ18 WAV・2帯域・各7ノイズ | アンサンブルで劣化を抑える条件と、その理由を明らかにしたい |
| 2 | [卒論と現在のchunk R²・二値AUC](../../experiments/2026-09-15_onb_frequency_comparison/chunk_comparison/thesis_chunk_comparison.png) | 指標定義・集計を揃えても曲線は異なる。分割・モデル・データ・ノイズ定義は変わっている |
| 3 | [WAVの性能比較](../../experiments/2026-09-15_onb_frequency_comparison/comparison/frequency_metric_comparison.png)、[低下量](../../experiments/2026-09-15_onb_frequency_comparison/comparison/degradation_comparison.png) | 3 kHzの−20でCNN 0.9194→inner 0.9234。ただし無雑音はCNNが高精度。統合の回帰改善とONB点の検知改善は別 |
| 4 | [信号とノイズの帯域差](../../experiments/2026-09-15_onb_frequency_comparison/input_probe/band_signal_noise.png) | 同じreference −20でも帯域内SNR中央値は2–3 kHzで+4 dB、10–15 kHzで−35 dB。224列への変換も同時に変わる |
| 5 | [3 kHz帯域マスク](../../experiments/2026-09-15_onb_frequency_comparison/snapshot_3khz/figures/frequency_mask_comparison.png)、[22 kHz帯域マスク](../../experiments/2026-09-15_onb_22khz_noise_sweep/snapshot_20260915/figures/frequency_mask_comparison.png) | RFは2–3 / 2–5 kHz、深層は別の依存。マスク感度から気泡の物理原因を直接同定しない |
| 6 | [説明性の役割と検算](../../experiments/2026-09-15_onb_frequency_comparison/analysis.md) §6 | SHAPはPCA成分、IGは予測差、CAMは粗い局在。IGは420/420で不整合。安定した画像と数値的な正しさは別 |
| 7 | 結論・相談 | 劣化抑制は条件付きで支持。回復の原因は未確定。複数seed・clean転送・共通周波数grid、IG少数例の収束、新実験の順序を相談 |

発表時間が短い場合は2と3、5と6をまとめる。図を増やすより、結果として分かったことと原因仮説を明確に区別する。詳細な話す文章と先生への相談項目は[最新解析の§8](../../experiments/2026-09-15_onb_frequency_comparison/analysis.md)を使う。


## 話すときの確認点

- 希望する結論は仮説として示し、今回支持された部分と未確認の部分を続けて述べる。
- 劣化量・強ノイズ時の絶対性能・ONB誤りを分ける。3 kHzの統合は回帰で改善したが最初のONB点は捉えていない。
- マスク結果と帯域内SNRは原因の手がかり。22 kHz回復の原因や気泡由来の同定が確定したとは話さない。
- 手法を選んだ理由は[説明性の役割表](../notes/model_notes.md)、追加検証の選び方は[解析手順](../analysis_workflow.md)へ戻る。
- 今回の対象外runや他実験日を、準備資料へ自動的に追加しない。

## 残る仕上げ

発表時間に合わせて図を選び、PPTXへ配置して話す時間を確認する。現時点で作成済みなのは解析・図・構成案・SOAP下書きで、完成PPTXや発表実施記録ではない。

先行する22 kHz単独の草稿は[文書監査の改訂前記録](../audits/2026-09-15_document_consistency/before/extracted_text.jsonl)に保持し、この準備メモでは現行案を一つにした。[22 kHz単独解析](../../experiments/2026-09-15_onb_22khz_noise_sweep/analysis.md)の当時の数値は保持している。
