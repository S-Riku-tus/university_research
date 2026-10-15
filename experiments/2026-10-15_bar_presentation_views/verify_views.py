"""Render completed October-6 summaries only; do not inspect the current run."""
from pathlib import Path
import hashlib
import json
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from utils.experiment.run_helpers import open_text
import utils.plotting.regression_plots as plots


def main():
    comparisons = pd.read_csv(ROOT / "experiments/2026-10-07_onb_200epoch_result_analysis/comparison_across_runs.csv")
    source = Path(comparisons.loc[comparisons.seed.eq(42)].iloc[0].source_result) / "metrics_summary_no_noise.csv"
    with open_text(source, "rb") as stream:
        before = hashlib.sha256(stream.read()).hexdigest()
    with open_text(source, "r", encoding="utf-8-sig") as stream:
        data = pd.read_csv(stream)
    labels = data.model.tolist()
    records = []
    original_save = plots._save_current_figure
    metric = None

    def check_and_save(path):
        fig, ax = plt.gcf(), plt.gca()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        view = Path(path).parent.name
        selected = plots.BAR_COMPARISON_GROUPS.get(view, labels)
        indices = [labels.index(label) for label in selected]
        heights = [bar.get_height() for bar in ax.patches]
        np.testing.assert_array_equal(heights, data[f"{metric}_mean"].iloc[indices].to_numpy())
        ticks = ax.get_xticklabels()
        assert all(t.get_fontsize() == ax.yaxis.label.get_fontsize() == 23 for t in ticks)
        numbers = [t for t in ax.texts if t.get_text() != "Zoomed y-axis"]
        assert len(numbers) == len(selected) and all(t.get_fontsize() == 30 for t in numbers)
        text_bounds = [t.get_window_extent(renderer) for t in [*ticks, *ax.texts]]
        assert all(b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height
                   for b in text_bounds), (path, text_bounds, fig.bbox)
        for texts in (ticks, numbers):
            bounds = [t.get_window_extent(renderer) for t in texts]
            assert all(right.x0-left.x1 >= 3 for left, right in zip(bounds, bounds[1:])), path
        if "Original three + HGB + ExtraTrees" in selected:
            assert "Original3\n+ HGB + ET" in [t.get_text() for t in ticks]
        errors = data[f"{metric}_se"].iloc[indices].to_numpy()
        expected = data[f"{metric}_mean"].iloc[indices].to_numpy()
        assert ax.get_ylim()[0] <= min(expected-errors) and ax.get_ylim()[1] >= max(expected+errors)
        original_save(path)
        records.append({"path": str(Path(path).relative_to(OUT)), "model_count": len(selected),
                        "labels": list(selected), "font": plt.rcParams["font.family"],
                        "axis_fontsize": 23, "value_fontsize": 30, "zoom": str(path).endswith("_zoom.png"),
                        "values_unchanged": True, "text_nonoverlap_and_inside_figure": True})

    for font, key in (("Times New Roman", "times"), ("DejaVu Sans", "sans")):
        for metric, title in (("r2", "R2 Score"), ("roc_auc_cont", "ROC-AUC (continuous)"),
                              ("pr_auc_cont", "PR-AUC (continuous)")):
            with plt.rc_context({"font.family": font, "savefig.dpi": 150}), \
                    patch.object(plots, "_save_current_figure", side_effect=check_and_save):
                plots.RegressionPlotter().plot_bar(title, labels, data[f"{metric}_mean"].to_numpy(),
                    data[f"{metric}_se"].to_numpy(), 200, OUT / key, "no_noise")
    with open_text(source, "rb") as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == before
    result = {"status": "passed", "source": str(source.relative_to(ROOT)), "source_sha256": before,
              "source_unchanged": True, "current_run_read": False, "new_fits": 0, "new_inference": 0,
              "scope": "Presentation layout check using previously analyzed completed results only",
              "renders": records}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "renders": len(records), "source_unchanged": True,
                      "current_run_read": False, "new_fits": 0, "new_inference": 0}))


if __name__ == "__main__":
    main()
