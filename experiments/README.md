# experiments

実験結果の「要約」を置く場所です。巨大な生成データや全画像をここに入れる必要はありません。各実験について、条件、結果、解釈、関連ファイルだけを残します。

## 使い方

新しい実験を行ったら、次のようなフォルダを作る。

```text
experiments/YYYY-MM-DD_short-name/
  config.yaml
  run_summary.md
  metrics.csv
  figures/
```

実際の巨大データや重みは `Pool_boiling/` 側に置いたままでよいです。ここには、研究報告や修論で使うための地図を残します。
