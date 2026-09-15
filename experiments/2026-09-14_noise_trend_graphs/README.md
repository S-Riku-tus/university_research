# ノイズ強度別のモデル比較グラフ

> **履歴・適用範囲（2026-09-15追記）**: この文書は当時の条件・判断の記録です。本文の「現在」「次にやること」は作成時点を指します。最新の設定・完了範囲・優先順位は[研究の現在地](../../docs/research_status.md)、更新関係は[文書案内](../../docs/document_index.md)を参照してください。

卒業研究の図に合わせ、横軸をノイズ強度、縦軸をR²またはAUCとし、RF・CNN＋Transformer・AlexNet・主アンサンブルの4本の曲線を保存する機能を追加した。

実装は `code/utils/plotting/noise_trend_plots.py`。既存の `RegressionPlotter` から呼び出し、ONB本体では各条件の完了時と再開時に更新する。設定、保存先、今週の70条件の方針は[実行方針メモ](../../docs/research_plan/2026-09-14_execution_policy_and_generalization_design.md)を参照する。

| 見本 | 内容 |
|---|---|
| [chunkのR²](preview_20260903_0618_22khz/simple_equal/chunk_fold_mean_r2.png) | fold平均と標準誤差 |
| [chunkの連続ROC-AUC](preview_20260903_0618_22khz/simple_equal/chunk_fold_mean_roc_auc_cont.png) | 連続予測スコアで計算 |
| [chunkの二値化後AUC](preview_20260903_0618_22khz/simple_equal/chunk_fold_mean_auc_binary.png) | 卒論互換の参考指標 |
| [WAVのR²](preview_20260903_0618_22khz/simple_equal/wav_median_r2.png) | WAV内中央値、全OOFをまとめた評価 |
| [WAVの連続ROC-AUC](preview_20260903_0618_22khz/simple_equal/wav_median_roc_auc_cont.png) | 独自の誤差推定は追加せず、点推定を表示 |
| [WAVの二値化後AUC](preview_20260903_0618_22khz/simple_equal/wav_median_auc_binary.png) | ONB閾値による二値判定 |

各見本と同名のPDF、数値CSVを保存している。これらは**9月3日開始の保存結果を使った書式確認用**であり、今回の新しい学習結果ではない。対象は2025.06.18、22 kHz、7ノイズ条件、simple equal。chunk側と後処理済みWAV側では使用したONB閾値も異なるため、この見本を使って集約方法だけの効果を比較しない。各CSVに使用閾値と元CSVパスを記録した。

連続ROC-AUCが1で重なる箇所では複数の線が同じ位置になる。数値どおりの描画であり、4本のモデルが省略されたわけではない。

検証結果: `python -m unittest discover -s tests -p "test_*.py" -v` は24件成功。追加5件では、欠測条件、ノイズ順序、chunk/WAV集約、AUCの種類、異なる設定の除外、再開時の更新と画像保存を確認した。ONB設定の読込と70条件のデータ存在確認も実施。本学習は起動していない。
