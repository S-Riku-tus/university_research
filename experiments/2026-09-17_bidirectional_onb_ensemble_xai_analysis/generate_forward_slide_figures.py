"""Generate slide-ready figures for the 6/11 train -> 6/18 test run only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / (
    "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/"
    "regression_result/npy/ensemble/20260917/"
    "onb_xd-t0611-v0618_iw3-nm_s1e-9_e200_123711/"
    "maxfreq=22kHz/heatflux_no_noise"
)
OUT = ROOT / "研究進捗報告/2026/918（研究計画発表）/figures_forward_0611_to_0618"
ONB = 271677.6816  # W/m², corrected 6/18 threshold

MODELS = ["RandomForest", "Conformer", "AlexNet", "Ensemble"]
PRED_COLUMNS = {
    "RandomForest": "randomforest",
    "Conformer": "conformer",
    "AlexNet": "alexnet",
    "Ensemble": "ensemble__inner_holdout",
}
MASK_KEYS = {
    "RandomForest": "randomforest",
    "Conformer": "conformer",
    "AlexNet": "alexnet",
}
COLORS = {
    "RandomForest": "#546A7B",
    "Conformer": "#31A397",
    "AlexNet": "#3277C8",
    "Ensemble": "#E48434",
}


def native_path(path: Path) -> str:
    """Open long source paths on Windows without changing stored results."""
    absolute = str(path.resolve())
    return "\\\\?\\" + absolute if os.name == "nt" else absolute


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    with open(native_path(RUN / "run_manifest.json"), encoding="utf-8") as file:
        manifest = json.load(file)
    assert manifest["run_hash"] == "56fc786f", manifest["run_hash"]

    metrics = pd.read_csv(native_path(RUN / "metrics_summary_no_noise.csv"))
    predictions = pd.read_csv(native_path(RUN / "fold_pred/pred_f1_no_noise.csv"))
    top_bands = pd.read_csv(
        native_path(RUN / "explainability/top_groups_by_model.csv")
    )
    weights = pd.read_csv(native_path(RUN / "ensemble_weights_no_noise.csv"))
    assert len(predictions) == 1080
    assert set(predictions["experiment_name"]) == {"2025.06.18_0.3_3"}
    assert int((predictions["y_true"] >= ONB).sum()) == 600
    assert set(metrics["model"]) == set(MODELS)
    assert len(weights) == 1 and weights.iloc[0]["strategy_name"] == "inner_holdout"
    return metrics.set_index("model"), predictions, top_bands, weights.iloc[0]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Meiryo",
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 20,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def prepare_metric_axes() -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    y = np.arange(len(MODELS))[::-1]
    for ax in axes:
        ax.set_yticks(y, MODELS)
        ax.set_ylim(-0.7, 3.7)
        ax.grid(axis="x", color="#D7DEE3", linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=12)
        for name, yy in zip(MODELS, y):
            ax.axhline(yy, color="#F0F2F4", linewidth=1, zorder=0)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.16, top=0.87, wspace=0.48)
    return fig, axes, y


def plot_slide05(metrics: pd.DataFrame) -> None:
    fig, axes, y = prepare_metric_axes()
    panels = [
        (axes[0], "r2_mean", "R²（高いほど良い）", (0.82, 0.93), ".3f", 0.003),
        (axes[1], "mae_all_mean", "MAE [kW/m²]（低いほど良い）", (62, 92), ".1f", 0.8),
    ]
    for ax, column, title, xlim, fmt, offset in panels:
        ax.set_title(title, pad=23)
        ax.set_xlim(*xlim)
        for name, yy in zip(MODELS, y):
            value = float(metrics.loc[name, column])
            if column == "mae_all_mean":
                value /= 1000
            ax.scatter(value, yy, s=250, color=COLORS[name], zorder=3)
            ax.text(value + offset, yy, format(value, fmt), va="center", fontsize=17)
    fig.savefig(OUT / "slide05_regression_r2_mae.png", dpi=200)
    plt.close(fig)


def plot_slide06(metrics: pd.DataFrame, pred: pd.DataFrame) -> None:
    fig, axes, y = prepare_metric_axes()
    auc_ax, miss_ax = axes
    auc_ax.set_title("連続予測 ROC-AUC（高いほど良い）", pad=23)
    miss_ax.set_title("ONB見逃し数 [秒]（少ないほど良い）", pad=23)
    auc_ax.set_xlim(0.925, 0.975)
    miss_ax.set_xlim(95, 142)
    actual = pred["y_true"].to_numpy() >= ONB
    false_positives: dict[str, int] = {}
    for name, yy in zip(MODELS, y):
        auc = float(metrics.loc[name, "roc_auc_cont_mean"])
        detected = pred[PRED_COLUMNS[name]].to_numpy() >= ONB
        missed = int(np.sum(actual & ~detected))
        false_positives[name] = int(np.sum(~actual & detected))
        auc_ax.scatter(auc, yy, s=250, color=COLORS[name], zorder=3)
        auc_ax.text(auc + 0.0012, yy, f"{auc:.3f}", va="center", fontsize=17)
        miss_ax.scatter(missed, yy, s=250, color=COLORS[name], zorder=3)
        miss_ax.text(missed + 1.3, yy, str(missed), va="center", fontsize=17)
    assert false_positives == {
        "RandomForest": 2,
        "Conformer": 0,
        "AlexNet": 0,
        "Ensemble": 0,
    }
    fig.text(
        0.5,
        0.045,
        "6/18のONB以上600秒。誤検知：RF 2秒、他の3モデル 0秒",
        ha="center",
        fontsize=17,
        color="#46525B",
    )
    fig.savefig(OUT / "slide06_onb_auc_false_negatives.png", dpi=200)
    plt.close(fig)


def plot_slide07(pred: pd.DataFrame, weights: pd.Series) -> None:
    regions = [
        ("ONB前（480秒）", pred["y_true"].to_numpy() < ONB),
        ("ONB以上（600秒）", pred["y_true"].to_numpy() >= ONB),
    ]
    fig, ax = plt.subplots(figsize=(15, 5.8))
    ax.set_xlim(76, 92)
    ax.set_ylim(-0.6, 1.65)
    ax.set_yticks([1, 0], [regions[0][0], regions[1][0]])
    ax.set_xlabel("RMSE [kW/m²]（小さいほど良い）", labelpad=8)
    ax.grid(axis="x", color="#D7DEE3", linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=15)
    for index, (label, region) in enumerate(regions):
        yy = 1 - index
        true = pred.loc[region, "y_true"].to_numpy()
        points = []
        for name, shift in [("AlexNet", 0.14), ("Ensemble", -0.14)]:
            estimate = pred.loc[region, PRED_COLUMNS[name]].to_numpy()
            rmse = float(np.sqrt(np.mean((estimate - true) ** 2))) / 1000
            points.append(rmse)
            ax.scatter(rmse, yy + shift, s=270, color=COLORS[name], zorder=3)
            ax.text(rmse + 0.28, yy + shift, f"{rmse:.1f}", va="center", fontsize=19)
        ax.plot(points, [yy, yy], color="#B4BEC6", linewidth=2, zorder=1)
    ax.scatter([], [], s=180, color=COLORS["AlexNet"], label="AlexNet")
    ax.scatter([], [], s=180, color=COLORS["Ensemble"], label="統合")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.30, top=0.79)
    fig.text(
        0.5,
        0.045,
        "統合重み：RF {rf:.2f} ／ Conformer {con:.2f} ／ AlexNet {alex:.2f}".format(
            rf=weights["randomforest"],
            con=weights["conformer"],
            alex=weights["alexnet"],
        ),
        ha="center",
        fontsize=17,
        color="#46525B",
    )
    fig.savefig(OUT / "slide07_rmse_before_after_onb.png", dpi=200)
    plt.close(fig)


def plot_slide08(top_bands: pd.DataFrame) -> None:
    selected = top_bands[
        (top_bands["axis"] == "frequency")
        & (top_bands["ranking_metric"] == "r2_drop")
    ].set_index("model_key")
    assert set(MASK_KEYS.values()).issubset(set(selected.index))
    fig, ax = plt.subplots(figsize=(15, 5.9))
    y = [2, 1, 0]
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.65, 2.7)
    ax.set_yticks(y, ["RandomForest", "Conformer", "AlexNet"])
    ax.set_xticks([0, 2, 5, 10, 15, 22])
    ax.set_xlabel("周波数 [kHz]", labelpad=14)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.grid(axis="x", color="#E0E5E8", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    for name, yy in zip(MODELS[:3], y):
        item = selected.loc[MASK_KEYS[name]]
        low, high = float(item["low"]) / 1000, float(item["high"]) / 1000
        drop = float(item["metric_value"])
        ax.barh(yy, 22, left=0, height=0.35, color="#E8ECEF", zorder=1)
        ax.barh(yy, high - low, left=low, height=0.35, color=COLORS[name], zorder=2)
        ax.text(22.7, yy, f"{low:g}–{high:g} kHz  ΔR²={drop:.3f}", va="center", fontsize=16)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.23, top=0.92)
    fig.text(
        0.5,
        0.055,
        "各モデルで、帯域をゼロにした際のR²低下が最大の場所。物理的な音源の特定ではない。",
        ha="center",
        fontsize=15,
        color="#46525B",
    )
    fig.savefig(OUT / "slide08_frequency_mask_top_bands.png", dpi=200)
    plt.close(fig)


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, predictions, top_bands, weights = load_sources()
    plot_slide05(metrics)
    plot_slide06(metrics, predictions)
    plot_slide07(predictions, weights)
    plot_slide08(top_bands)
    print(f"Created four figures in {OUT}")


if __name__ == "__main__":
    main()
