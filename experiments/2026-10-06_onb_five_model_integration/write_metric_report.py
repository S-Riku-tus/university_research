"""Generate the metric-change tables directly from validated production rows."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from run_hgb_complementarity_validation import read_csv

OUT = Path(__file__).resolve().parent
ROWS = read_csv(OUT / "metrics_recomputed.csv")
BASELINE = "ensemble__original3_mse"
FIVE = "ensemble__original3_hgb_extra_trees"


def row(seed, noise, key, day="pooled_source_day_thresholds"):
    return next(r for r in ROWS if int(r["seed"])==seed and r["noise"]==noise and r["model_key"]==key and r["source_day"]==day)


def table(noise):
    lines = ["| 指標 | seed43 元3 → 5 | 変化 | seed44 元3 → 5 | 変化 |",
             "|---|---|---|---|---|"]
    for label, key, kind in [("見逃し FN", "fn", "count"), ("誤報 FP", "fp", "count"),
        ("Recall", "recall", "rate"), ("Precision", "precision", "rate"), ("Accuracy", "accuracy", "rate"),
        ("F1", "f1", "score"), ("R²", "r2", "score"), ("全域RMSE", "rmse_all", "error"),
        ("全域MAE", "mae_all", "error"), ("ONB以上RMSE", "rmse_high", "error"),
        ("ONB近傍RMSE", "rmse_onb", "error"), ("連続ROC-AUC", "roc_auc_cont", "score"),
        ("連続PR-AUC", "pr_auc_cont", "score")]:
        cells = []
        for seed in [43, 44]:
            before, after = float(row(seed, noise, BASELINE)[key]), float(row(seed, noise, FIVE)[key])
            if kind=="error":
                before, after = before/1000, after/1000
                cells += [f"{before:.2f} → {after:.2f}", f"{after-before:+.2f}"]
            elif kind=="rate":
                cells += [f"{before*100:.2f}% → {after*100:.2f}%", f"{(after-before)*100:+.2f} pp"]
            elif kind=="count":
                cells += [f"{int(before)} → {int(after)}", f"{int(after-before):+d}"]
            else:
                cells += [f"{before:.5f} → {after:.5f}", f"{after-before:+.5f}"]
        lines.append("| "+" | ".join([label, *cells])+" |")
    return "\n".join(lines)


def main():
    content = """# 元3から5モデルへ追加したときの各指標の変化

2026-10-06。保存済みseed43/44を新しい通常ONBの統合・指標コードで再計算した。同じ540未使用chunk、36 WAV、学習・重みともclean_only。元3にも同じMSEをfitした公平な対照。ONB判定は日別閾値で、陽性315/陰性225。誤差・q100の単位はkW/m²、ppはパーセントポイント。

ここに示すのは150 epochs完了済みの保存モデルに基づく対応結果。今回の通常mainの1 epoch動作確認値とは別であり、通常main既定seed42の新しい本実行結果でもない。

## clean

"""+table("clean")+"""

見逃しが7/9例減り、FP0を維持したため、Recallは7/315=2.22pp、9/315=2.86pp、Accuracyは7/540=1.30pp、9/540=1.67pp改善する。Precisionは陽性予測に誤報が含まれないので100%のまま。F1はPrecisionとRecallから計算される。

RMSE/MAE/R²は残差の大きさ、連続ROC/PR-AUCは予測順位にも依存する。FNの件数が減っただけでこれらの改善量が一意に決まるわけではない。表はそれぞれを保存予測から再計算した。

## −20 dB

"""+table("-20")+"""

seed44は全域回帰と見逃しが改善しても、ONB近傍RMSEが90.58→109.46へ悪化する。q100や領域別誤差も合わせて読んで、全指標が改善したとはしない。

## 日別q100

q100は「その段階以降の評価chunkがすべて陽性だった最初の測定熱流束」。有限の評価標本の到達段階であり、将来の検知確率100%ではない。

| seed | 条件 | 元3 6/11 | 5モデル 6/11 | 元3 6/18 | 5モデル 6/18 |
|---|---|---|---|---|---|
"""
    for seed in [43, 44]:
        for noise in ["clean", "-20"]:
            cells = []
            for day in ["2025.06.11_0.3_2", "2025.06.18_0.3_3"]:
                cells += [f"{float(row(seed, noise, key, day)['q100'])/1000:.2f}" for key in [BASELINE, FIVE]]
            content += "| "+" | ".join([str(seed), noise, *cells])+" |\n"
    content += """
seed43のclean6/18は、HGBだけを追加する4モデルでは322.11まで早まるが、5モデルでは376.32へ戻る。5が4の全指標を上回るという意味ではない。ExtraTrees4と5の回帰差も小さい。

## 通常ONBの表との対応

従来の`metrics_summary_*.csv`は、統合日の平均閾値246.59 kW/m²での評価を保持する。今回の見逃し数と一致する日別閾値評価は、新しい`metrics_by_source_day.csv`の`pooled_source_day_thresholds`行を見る。元3との差は`metrics_source_day_deltas.csv`で出る。日別の行にはq100/g100も保存される。

各ノイズの全数値：[再計算指標CSV](metrics_recomputed.csv)、[全指標差CSV](metric_changes.csv)。元3/HGB4/ExtraTrees4/5の重み：[production_weights.csv](production_weights.csv)。実装・検算：[実行記録](README.md)。
"""
    (OUT / "metric_changes_summary.md").write_text(content, encoding="utf-8")
    print("[report] clean/minus20 all-metric changes and day q100 generated from validated rows")


if __name__ == "__main__":
    main()
