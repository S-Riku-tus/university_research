# ensemble結果の日付階層整理

実施日: 2026-09-25

## 目的

各収録実験日の`regression_result/npy/ensemble/`直下に`YYYYMMDD`フォルダが増え続け、解析結果を探しにくくなっていた。今後のONB実行と日付付き既存結果を、月単位で閲覧できる次の構成へ統一する。

```text
regression_result/npy/ensemble/YYYYMM/DD/<実行条件>/
```

## 実装

- `result_date_dir`は`YYYYMM/DD/系列名`を正規形式とする。
- 旧指定`YYYYMMDD/系列名`も実行時に自動的に正規化する。
- 旧フォルダ`YYYYMMDD_系列名`は`YYYYMM/DD/系列名`へ移す。
- 日付で始まらない古い結果は、日付を推測せず元の場所に残す。
- 移動先が存在する場合は統合・上書きせず、移動前に停止する。
- 移動前後で各系列のファイル数と総byte数を照合する。
- JSON、CSV、TXT、Markdown内の旧保存パスは移動先に合わせて更新する。

実装の入口は[`migrate_ensemble_date_layout.py`](../../code/migrate_ensemble_date_layout.py)、通常実行時の正規化は[`result_paths.py`](../../code/utils/experiment/result_paths.py)。

## 移行記録

- dry-run: 未実施
- 本移行: 未実施
- 詳細対応表: `migration_report.json`

移行後も日付情報のない`heatflux_no_noise`等の旧系列はそのまま残す。それらを新しい解析日へ推測配置しない。
