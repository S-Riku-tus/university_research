"""Plot B6 training loss only; no validation history exists in these runs."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent


def main():
    with (OUT / "matched_seed_training_loss.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = [r for r in csv.DictReader(source) if r["fit_role"] == "full_training"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for row_i, snr in enumerate(("-8", "-16")):
        for column_i, model in enumerate(("Conformer", "AlexNet")):
            ax = axes[row_i, column_i]
            for seed in (42, 43, 44):
                selected = [r for r in rows if r["snr"] == snr and r["model"] == model
                            and int(r["seed"]) == seed]
                ax.plot([int(r["epoch"]) for r in selected],
                        [float(r["training_loss"]) for r in selected],
                        label=f"seed {seed}", linewidth=1.4)
            ax.set_title(f"{model}, matched {snr} dB")
            ax.set_yscale("log")
            ax.grid(alpha=.25)
            if row_i == 1:
                ax.set_xlabel("Epoch")
            if column_i == 0:
                ax.set_ylabel("Training MSE (scaled target)")
            if row_i == 0 and column_i == 1:
                ax.legend(frameon=False)
    fig.suptitle("B6 full-training loss checkpoints (no validation loss recorded)")
    fig.tight_layout()
    fig.savefig(OUT / "matched_seed_training_loss.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
