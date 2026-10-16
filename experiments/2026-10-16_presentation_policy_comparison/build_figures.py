"""Replot completed, paired three-model runs; no training or inference."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ASSETS = ROOT / "docs/progress/assets/2026-10-16/policy_comparison"
ORDER = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]


def accessible(p):
    return Path("\\\\?\\" + str(p.resolve())) if sys.platform == "win32" else p


def read_csv(p):
    with accessible(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(accessible(p).read_bytes()).hexdigest()


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    source = ROOT / "experiments/2026-10-03_tuned_within_wav_matched_comparison"
    scope_path = source / "run_scope.json"
    metric_path = source / "metrics_source_day_thresholds.csv"
    identity_path = source / "clean_identity.csv"
    scope = json.loads(scope_path.read_text(encoding="utf-8-sig"))
    records = [r for r in read_csv(metric_path)
               if r["scope"] == "two_day" and r["model"] == "performance"]
    assert len(records) == 14
    assert all(int(r["n"]) == 540 for r in records)
    assert all(float(r["max_clean_prediction_difference"]) == 0
               for r in read_csv(identity_path))
    sources = [scope_path, metric_path, identity_path]
    trend_differences = []
    for policy, path in scope["runs"].items():
        trend = (ROOT / path / "noise_trends/performance_kfold/maxfreq=3kHz/"
                 "chunk_held_out_day_r2.csv")
        sources.append(trend)
        ens = [r for r in read_csv(trend) if r["model_key"].startswith("ensemble__")]
        assert len(ens) == 7
        for r in ens:
            noise = "clean" if r["noise"] == "no_noise" else r["noise"]
            match = [a for a in records if a["policy"] == policy and a["noise"] == noise]
            assert len(match) == 1
            # The original trend tables store R² rounded to six decimals.
            delta = abs(float(r["value"]) - float(match[0]["r2"]))
            assert delta <= 5.01e-7
            trend_differences.append(delta)
    before = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sources}
    rows = []
    for policy in ["matched", "clean_only"]:
        for noise in ORDER:
            r = next(r for r in records if r["policy"] == policy and r["noise"] == noise)
            rows.append(dict(policy=policy, noise=noise, n=540,
                             r2=float(r["r2"]), rmse_kw_m2=float(r["rmse"]) / 1000))
    with (OUT / "comparison_values.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    font = "C:/Windows/Fonts/YuGothM.ttc"
    fontManager.addfont(font)
    plt.rcParams.update({"font.family": FontProperties(fname=font).get_name(),
                         "font.size": 17, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    for metric, ylabel in [("r2", "決定係数 R²（大きいほど良い）"),
                            ("rmse_kw_m2", "RMSE［kW/m²］（小さいほど良い）")]:
        fig, ax = plt.subplots(figsize=(12.0, 4.65))
        for policy, color, marker, label in [
            ("matched", "#df9536", "s", "各雑音条件で学習（matched）"),
            ("clean_only", "#287f96", "o", "無雑音学習・固定（clean_only）")]:
            subset = [r for r in rows if r["policy"] == policy]
            ax.plot(range(7), [r[metric] for r in subset], color=color, marker=marker,
                    linewidth=2.7, markersize=7, label=label)
        ax.set_xticks(range(7), ["無雑音", "0", "−4", "−8", "−12", "−16", "−20"])
        ax.set_xlabel("付加雑音の参照SNR［dB］ → 右ほど雑音が強い")
        ax.set_ylabel(ylabel)
        if metric == "r2":
            ax.set_ylim(.895, 1.001)
            ax.text(1.55, .978, "0 → −4 dBで小さな回復", fontsize=13, color="#9c641d")
        else:
            ax.set_ylim(25, 91)
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="lower left" if metric == "r2" else "upper left", fontsize=15.5)
        fig.tight_layout()
        for ext in ["png", "svg"]:
            fig.savefig(ASSETS / f"matched_vs_clean_only_original3_{metric}.{ext}",
                        dpi=220, bbox_inches="tight")
        plt.close(fig)
    assert all(sha(ROOT / p) == v for p, v in before.items())
    record = dict(sources=before, sources_unchanged=True,
                  paired_policy_runs=scope["runs"], evaluation_n=540,
                  current_1015_run_not_mixed=True, old_noise_generator_not_mixed=True,
                  stopped_matched_run_not_read=True, ensemble="original three / performance weights",
                  epochs=150, internal_validation="WAV 3-fold", new_training=0,
                  new_inference=0, new_input_interventions=0,
                  trend_csv_matches_audited_metrics_at_saved_precision=True,
                  max_r2_difference_from_six_decimal_trend=max(trend_differences),
                  plotted_values="audited full-precision metrics",
                  no_noise_predictions_identical=True)
    (OUT / "verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps(dict(rows=len(rows), assets=str(ASSETS.relative_to(ROOT)),
                          source_files_unchanged=True, new_training=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
