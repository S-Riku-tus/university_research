# 実験・解析ログの入口

更新日: 2026-09-15。結果を二重に転記して更新差が生じないよう、実行ごとの数値と解釈は[experiments一覧](../../experiments/README.md)、時系列の判断は[進捗索引](../progress_index.md)へ集約する。現在の状態は[研究の現在地](../research_status.md)。

## 直近の固定記録

- [9/15採取：9/14の6条件・説明性と9月実行一覧](../../experiments/2026-09-15_research_status_snapshot/README.md)
- [9/14採取：9/3の105条件・9/8の3条件](../../experiments/2026-09-14_status_audit/README.md)
- [9/2の56条件記録](../../experiments/2026-09-02_selected_log_architecture/README.md)：当時の中断スナップショット。旧inner holdoutの安全性は9/8監査で更新。

## 新しい結果を残すとき

[実験サマリーテンプレート](../../experiments/_template/run_summary.md)を使い、収録日と解析日、データ版、コード版、学習/評価分割、評価単位、要求/実際の学習条件、出力範囲、根拠CSV・hash、解釈と限界を記録する。

日付付きの記録は後から別runの数値へ置き換えず、訂正または後継への注記を残す。このファイルに成績のコピーを増やさない。
