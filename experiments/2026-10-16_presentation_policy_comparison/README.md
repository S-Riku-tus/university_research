# 発表用：従来3モデル統合のmatched／clean_only比較図

作成日：2026-10-16。本人の「雑音評価の説明の後へ、従来方式とclean_onlyの結果を比較するページを入れたい」という依頼に対応。既存の完了済み・条件を揃えた対照から再描画した。新しい学習・推論・入力加工は行っていない。

## 対象と条件

- 正本：[10/3の対応解析](../2026-10-03_tuned_within_wav_matched_comparison/README.md)。cleanは10/1、matchedは10/3の完了run。
- 2025年6/11＋6/18・3 kHz・1秒・選別なし。同じ録音内の学習1620／評価540区間、150 epochs、学習側WAV3-fold。
- 元3モデル（RF・Conformer・AlexNet）と従来performance方式の統合を両方に使う。元runの名称は`performance_kfold`、最新runnerの同系統の名称は`original3_performance`。
- matchedは雑音ごとにモデル・前処理・統合重みを学習。clean_onlyは無雑音学習の状態を固定する。学習方針全体の比較であり、重み変更だけを分離した実験ではない。
- 元の`noise_trends`のR²と、保存予測から検算済みのR²は、元表の小数6桁への丸めの範囲で一致。描画には検算済みの全精度値を使用した。無雑音の予測差は0。
- R²は共通540区間をまとめた値。モデル・fold・実験日のR²の平均ではない。

## 読み方

matchedは0〜−12 dBで全域R²が高く、0→−4では0.97114→0.97241と小さな回復がある。clean_onlyはこの対照では雑音が強くなるほどR²が低下する。−16／−20ではclean_onlyの方がわずかに高い。

したがって、この図はclean_onlyが全条件で高精度になった証拠ではない。「固定モデルへの雑音影響」と「各雑音条件へ再学習した性能」を区別する図として使う。曲線の単調性は全条件で保証される性質ではない。

旧雑音生成の問題は別の診断で修正済み。ここでは両方とも修正後の生成を使うため、この図を旧生成の修正効果とは呼ばない。

## 出力

- [R²の2曲線PNG](../../docs/progress/assets/2026-10-16/policy_comparison/matched_vs_clean_only_original3_r2.png)／[SVG](../../docs/progress/assets/2026-10-16/policy_comparison/matched_vs_clean_only_original3_r2.svg)：発表の第一候補。
- [RMSEの2曲線PNG](../../docs/progress/assets/2026-10-16/policy_comparison/matched_vs_clean_only_original3_rmse_kw_m2.png)／[SVG](../../docs/progress/assets/2026-10-16/policy_comparison/matched_vs_clean_only_original3_rmse_kw_m2.svg)：RMSEへ統一したい場合の代替。両方を本編へ追加する必要はない。
- `comparison_values.csv`：2方針×7条件のR²・RMSE。
- `verification.json`：元CSVのhash、元noise_trendsとの照合、条件、元ファイル不変を記録。

10/15の最新主結果は200 epochs・内部chunk3-foldなので、本図へ混ぜない。今回停止したmatchedも読込・再実行していない。本図は以前の完了対照による補助比較と注記し、後続の最新3/4/5モデル結果と区別する。

再描画：`python experiments/2026-10-16_presentation_policy_comparison/build_figures.py`。
