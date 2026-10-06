"""Export a compact scientific figure from verified CSV outputs."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent


def read(name):
    with (OUTPUT / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def main():
    metrics, endpoints, weights = read("full_metrics.csv"), read("endpoints.csv"), read("weights.csv")
    labels = ("Old WAV CV", "New chunk CV")
    runs = ("old_wav_clean", "new_chunk_clean")
    colors = ("#87949b", "#277da8")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for run, label, color, offset in zip(runs, labels, colors, (-.18, .18)):
        selected = [next(row for row in metrics if row["run"] == run and row["unit"] == unit and row["noise"] == "clean"
                         and row["scope"] == "two_day" and row["model"] == "ensemble__performance_kfold")
                    for unit in ("training_oof", "outer")]
        bars = axes[0, 0].bar(np.arange(2) + offset, [float(row["rmse"]) / 1000 for row in selected], .36, color=color, label=label)
        axes[0, 0].bar_label(bars, fmt="%.2f", fontsize=9)
        selected_w = [next(float(row["weight"]) for row in weights if row["run"] == run and row["noise"] == "clean" and row["model"] == key)
                      for key in ("randomforest", "conformer", "alexnet")]
        axes[0, 1].bar(np.arange(3) + offset, selected_w, .36, color=color)
        for day, style in (("2025.06.11_0.3_2", "-"), ("2025.06.18_0.3_3", "--")):
            ep = next(row for row in endpoints if row["run"] == run and row["unit"] == "outer" and row["noise"] == "clean"
                      and row["scope"] == day and row["model"] == "ensemble__performance_kfold")
            axes[1, 0].plot([labels.index(label)], [float(ep["q100_kW_m2"])], marker="o", color=color)
            if run == runs[0]:
                axes[1, 0].axhline(float(ep["q100_kW_m2"]), color="#51575d", linestyle=style, label="6/11" if style == "-" else "6/18")
        selected_noise = [next(row for row in metrics if row["run"] == run and row["unit"] == "outer" and row["noise"] == noise
                              and row["scope"] == "two_day" and row["model"] == "ensemble__performance_kfold")
                          for noise in ("clean", "0", "-4", "-8", "-12", "-16", "-20")]
        axes[1, 1].plot(range(7), [int(row["fn"]) for row in selected_noise], marker="o", color=color, label=label)
    axes[0, 0].set(xticks=[0, 1], xticklabels=["Training OOF*", "Same 540 outer chunks"], ylabel="Clean RMSE (kW/m²)", ylim=(0, 75), title="Regression error")
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].set(xticks=range(3), xticklabels=["RF", "Conformer", "AlexNet"], ylabel="Weight", ylim=(0, .6), title="Training-derived performance weights")
    axes[1, 0].set(xticks=[0, 1], xticklabels=labels, ylabel="q100 (kW/m²)", ylim=(300, 475), xlim=(-.3, 1.3), title="Outer q100: unchanged in all 7 conditions")
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].set(xticks=range(7), xticklabels=["Clean", "0", "-4", "-8", "-12", "-16", "-20"], xlabel="Water-flow SNR (dB)", ylabel="False negatives / 315 post-ONB chunks", title="Fewer clean misses, more strong-noise misses")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Fixed-parameter clean-only comparison\n*Internal validation task changed; PCA layout fix also included", fontsize=12)
    fig.savefig(OUTPUT / "comparison_summary.png", dpi=180)
    fig.savefig(OUTPUT / "comparison_summary.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
