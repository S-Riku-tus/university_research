# 元WAVを分離したアンサンブル追加

2026-09-16。本人の添付提案を現行コード・研究記録と照合して追加する方式の設計・使用メモ。

## 判断

現行 `inner_holdout` は外側学習12 WAVから3 WAVを重み決定に使い、単体の `1-R²` の逆数で統合する。少数の録音による重みの変動と、誤差の相殺を直接最適化していない点を改善対象とする。独立単位はchunk数ではなく元WAV。

追加するのは `subset_equal_cv`、`crossfit_wav_stack`、`crossfit_shrinkage_stack`。外側学習集合だけで元WAV分離4-foldの学習外予測（inner OOF）を一度作り、3方式で共有する。PCA・目的変数scalerもinner学習側だけでfitする。

現行の評価は「chunkで重み付き平均→WAV中央値」。中央値と加重平均は交換できないため、新方式の候補評価・最適化もこの順序で行う。各モデルのWAV中央値を先に加重平均する通常の線形stackingとは異なり、目的関数は区分的で非凸になる。多始点の制約付き数値最適化を用い、大域最適とは主張しない。

ノイズ全条件を使うrobust stackingは、ノイズ条件間の学習共有と未知ノイズ評価を別途設計する必要がある。今回の追加は現在の `matched / clean_only` の意味を維持する。XAI gating、ONBの切片補正、多表現expertは今回導入しない。現行のONB近傍標本数では、全域回帰の改善とONB改善を分けて検証する必要がある。

既存2方式と既定の実行選択を維持し、新方式を選択した場合だけ追加学習する。実装・小規模動作確認と、実データで精度向上を検証した状態は区別する。

## 3方式の違い

| 方式名 | 重みの決め方 | 狙い |
|---|---|---|
| `subset_equal_cv` | 単体・2モデル平均・3モデル平均の7候補をinner OOFのWAV MSEで比較 | 悪化を起こすモデルを除ける。単体の採用も許す |
| `crossfit_wav_stack` | 非負・合計1の制約下でWAV MSEを直接最小化 | 各モデルの単体スコアに加え、誤差の相殺を利用する |
| `crossfit_shrinkage_stack` | 上記に等重みからの距離の罰則を加える | 少数WAVから極端な重みを推定するのを抑える |

後2方式の目的関数は、`WAV_MSE(w) / scale + λ * sum((w - 1/M)^2)`。`scale` はinner OOFの各単体WAV MSEの平均。熱流束の単位変更で罰則の強さが変わらないようにする。通常stackはλ=0、shrinkageは事前固定λ=0.1。**この値を研究データで最良と確認したわけではない**。後から変更する場合は別profileとし、評価側の成績で調整しない。

各部分集合の等重みを初期候補に残し、SLSQPの多始点最適化で得た有効解も比較する。失敗した解は採用せず、失敗理由と候補へのfallbackをJSONに記録する。単体候補が含まれるのでλ=0とsubset方式の**重み学習集合内のMSE**は最良単体より悪くならない。外側評価での保証ではない。縮小方式では罰則との兼ね合いで学習集合内のMSEが大きくなる場合がある。

縮小は添付の固定γによる後処理ではなく、目的関数内のL2罰則で実装した。median目的関数をそのまま最適化するため、`nnls`とは命名していない。切片・負の重みは使用しない。3モデルすべてが同じchunkでONB閾値未満なら、そのchunkの統合値も閾値未満になる。

## 有効にする方法

[主実行](../code/run_ensemble_regression_onb.py)の `VALIDATION_CONFIG["ensemble"]` を次のようにする。新規3行は主実行内にコメントとして用意済み。

```python
"ensemble": {
    "enabled_strategy_names": [
        "simple_equal",
        "inner_holdout",
        "subset_equal_cv",
        "crossfit_wav_stack",
        "crossfit_shrinkage_stack",
    ],
    "primary_strategy_name": "inner_holdout",
},
```

新方式は任意に1つだけでも選べる。方式の設定は[カタログ](../code/utils/ensemble/strategy_catalog.py)で管理し、実際に選択した方式と設定だけをmanifest/hashに含める。旧選択の設定値・hash計算は維持する。新方式を選択したrunは既存の完了runと設定hashが異なり、既存の結果ディレクトリがある場合は別名へ保存される。

外側学習集合が12 WAVならinnerは9 WAV学習/3 WAV予測×4回。3モデル・外側1-foldあたり、従来は `inner_holdout 3回 + 最終学習3回 = 6 fit`。3新方式も併用すると共通inner OOFの12 fitが増えて18 fitになる。**全5方式併用はfit回数で約3倍**であり、wall timeの確約ではない。新方式を複数選んでもOOF生成は増えない。3〜4 WAVではfold数をWAV数まで下げる。inner学習が2 WAV未満になる設定は拒否する。

`matched` はその条件の外側学習WAVだけ、`clean_only` はcleanの外側学習WAVだけから重みを学習する。clean_onlyでは各評価ノイズに同じ重みを適用する。別日評価でも評価日はinner fitへ渡さない。これらはノイズ全体を学習したrobust stackingとは区別する。

## 保存物と読み方

各runの既存の重みCSV・chunk予測・WAV指標・ONB遷移・ノイズ比較へ、新しい `ensemble__<方式名>` が追加される。さらに外側foldごとに次を保存する。

- `ensemble_inner_oof_fN.csv`: 学習集合内の行番号、実験日を含むWAV group、inner fold、真値、3モデルの学習外予測。全chunkが1回ずつ現れる。
- `ensemble_crossfit_fit_fN.json`: innerの学習/予測WAV、seed、実際のepoch/batch、採用重み、単体/等重み/採用方式のWAV MSE、残差相関、部分集合候補、最適化の成否。

JSON内のMSEは**重みを決めた集合での診断値**。最終性能は従来どおり外側の `wav_eval/` で読む。`claim_safe` は外側ラベルを重み決定に使っていないことを意味し、統計的有意性や単体超えの保証ではない。WAV中央値を先に取る線形stackingへの変更や、評価側での最良方式への自動切替はしていない。

## 検証と次の比較

[実装・検証記録](../experiments/2026-09-16_crossfit_ensemble/README.md)に確認範囲を保存する。[代表比較の条件書](../configs/experiments/2026-09-16_crossfit_ensemble.yaml)は未実行の案。まず現在と同じモデル・外側分割・seedで既存2方式と新3方式を比較し、WAV R²/RMSE/MAE、ONB最初の点の見逃し・誤警報、重み変動を見る。その後に複数seed・clean_only・別日へ進める。

保存済みの**外側OOF予測を新方式の重み学習に流用し、その同じ集合で改善を報告してはいけない**。外側OOFを使ったoracleは診断専用とする。既存結果から、無雑音時の統合悪化と強ノイズでの改善が両方あることは確認済みだが、今回の新方式での実データ改善は未検証。

## 添付提案の出典を確認した範囲

- [Super Learner 原論文の抄録](https://pubmed.ncbi.nlm.nih.gov/17910531/): V-foldの学習外予測から結合重みを選ぶ基本方針を確認。今回の有限WAV数・実装での改善保証には使わない。
- [scikit-learn StackingRegressor公式資料](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html): 最終モデルの学習外予測による学習と、基底モデルを全学習集合で再fitする構成を確認。既製クラスをそのまま使うとWAV分離・中央値の順序を扱えないため、既存trainerへ追加した。
- [Brown et al., Managing Diversity in Regression Ensembles](https://www.jmlr.org/papers/v6/brown05a.html): 回帰アンサンブルで多様性を考える研究上の根拠。今回の残差相関やマスクから物理的原因が確定するという意味ではない。

添付の2025〜2026年の全引用先や「最適」「必ず」といった評価は一括して採用していない。今回の3方式は、確認済みのコード・標本数・評価順序に合わせた実装上の判断。
