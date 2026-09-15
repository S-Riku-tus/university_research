# 保存階層・一般化評価の検証記録

> **履歴・適用範囲（2026-09-15追記）**: この文書は当時の条件・判断の記録です。本文の「現在」「次にやること」は作成時点を指します。最新の設定・完了範囲・優先順位は[研究の現在地](../../docs/research_status.md)、更新関係は[文書案内](../../docs/document_index.md)を参照してください。

実装・運用の説明は [保存階層の変更と一般化評価](../../docs/research_plan/2026-09-14_result_layout_and_generalization.md)。

- `existing_result_verification.json`: 保存済み6条件を学習・配列読込0回で再利用し、36図を再作成した記録と保存先。
- `real_data_preflight.json`: 実データ70条件のmetadataによる4方針の分割検査。学習は行っていない。
- `verification_summary.json`: 自動テスト34件の成功、6条件の再利用、移動前後の一致確認をまとめた記録。
- `verify_real_data.py`: 上記の実データ検査を再実行するスクリプト。ONBのconfigは変更しない。
- `22khz_r2.png`: 22 kHz・WAV中央値・inner_holdout方式のR²図のコピー。
- `22khz_roc_auc_cont.png`: 同じ条件の連続ROC-AUC図。
- `22khz_auc_binary.png`: 同じ条件の二値化後AUC図。

図の原本は各実験日の `regression_result/npy/ensemble/<実行日>/<周波数>/noise_trends/<run>/<方式>/` にPNG・PDF・数値CSVで保存する。このフォルダの画像は実際の計算値から作成した確認用コピーで、模式図ではない。

自動検証は `python -m unittest discover -s tests -q`。一般化評価の試験データは一時フォルダ内に作り、完了後に片付ける。本研究の既存モデル結果へテスト値を書き込まない。
