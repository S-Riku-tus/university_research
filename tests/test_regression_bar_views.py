"""Subsets must retain saved values/colors; zooms must not hide error bars."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import utils.plotting.regression_plots as plots
from replot_regression_bars import replot_completed_bars


LABELS = ["RandomForest", "Conformer", "AlexNet", "HGB frequency34",
          "ExtraTrees frequency34", "Original three + HGB + ExtraTrees",
          "Original three MSE", "Original three performance", "Original three + HGB",
          "Original three + ExtraTrees", "Ensemble simple equal"]


class RegressionBarViewsTest(unittest.TestCase):
    def test_fixed_subsets_preserve_values_errors_colors_and_zoom_envelope(self):
        values = np.array([.93, .98, .975, .986, .99, .992, .984, .981, .988, .989, .985])
        errors = np.linspace(.0001, .0011, len(values))
        snapshots = {}

        def capture(path):
            ax = plt.gca()
            bars = ax.containers[-1]
            snapshots[str(path)] = {
                "heights": np.array([bar.get_height() for bar in ax.patches]),
                "colors": [bar.get_facecolor() for bar in ax.patches],
                "segments": bars.errorbar.lines[2][0].get_segments(),
                "ylim": ax.get_ylim(),
                "texts": [(t.get_text(), t.get_fontsize()) for t in ax.texts],
            }

        with patch.object(plots, "_save_current_figure", side_effect=capture):
            paths = plots.RegressionPlotter().plot_bar("R2 Score", LABELS, values, errors, 200, "out", "clean")
        self.assertEqual(len(paths), 8)
        full_colors = snapshots[str(Path("out/bar/r2_ep200_clean.png"))]["colors"]
        for path, snapshot in snapshots.items():
            group = Path(path).parent.name
            selected = plots.BAR_COMPARISON_GROUPS.get(group, LABELS)
            indices = [LABELS.index(label) for label in selected]
            np.testing.assert_array_equal(snapshot["heights"], values[indices])
            self.assertEqual(snapshot["colors"], [full_colors[i] for i in indices])
            for segment, i in zip(snapshot["segments"], indices):
                np.testing.assert_allclose(segment[:, 1], [values[i]-errors[i], values[i]+errors[i]])
            if path.endswith("_zoom.png"):
                self.assertGreater(snapshot["ylim"][0], 0.)
                self.assertLess(snapshot["ylim"][0], min(values[indices]-errors[indices]))
                self.assertGreater(snapshot["ylim"][1], max(values[indices]+errors[indices]))
                self.assertIn(("Zoomed y-axis", 17), snapshot["texts"])
            else:
                self.assertEqual(snapshot["ylim"][0], 0.)
            self.assertTrue(all(size == 30 for text, size in snapshot["texts"] if text != "Zoomed y-axis"))

    def test_missing_comparison_members_are_not_replaced_or_partially_selected(self):
        with patch.object(plots, "_save_current_figure"):
            paths = plots.RegressionPlotter().plot_bar("R2 Score", LABELS[:3], [.96, .98, .99], None, 200, "out", "clean")
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(Path(path).parent.name == "bar" for path in paths))

    def test_negative_r2_is_visible_and_undefined_metrics_do_not_get_a_zoom(self):
        limits = []
        with patch.object(plots, "_save_current_figure", side_effect=lambda _: limits.append(plt.ylim())):
            paths = plots.RegressionPlotter().plot_bar("R2 Score", ["a", "b"], [-.4, .9], [.1, .02], 200, "out", "noise")
            missing = plots.RegressionPlotter().plot_bar("PR-AUC (continuous)", ["a", "b"], [np.nan, np.nan], None, 200, "out", "missing")
        self.assertLessEqual(limits[0][0], -.5)
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(missing), 1)

    def test_redraw_checks_all_completion_markers_before_plotting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, name in enumerate(("clean", "noise")):
                directory = root / name
                directory.mkdir()
                manifest = {"run_hash": "same", "dataset": {"snr_value": name},
                            "validation_config": {"run": {"epochs": 200}}}
                (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with (directory / f"metrics_summary_{name}.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["model", "r2_mean", "r2_se", "roc_auc_cont_mean", "roc_auc_cont_se", "pr_auc_cont_mean", "pr_auc_cont_se"])
                    writer.writerow(["RandomForest", .98, .001, .99, 0., .995, 0.])
                if i == 0:
                    (directory / "completed.json").write_text('{"run_hash":"same"}', encoding="utf-8")
            output = root / "redraw"
            with patch.object(plots.RegressionPlotter, "plot_bar") as draw:
                with self.assertRaisesRegex(ValueError, "has not completed"):
                    replot_completed_bars(root, output)
                draw.assert_not_called()
                self.assertFalse(output.exists())
            # A stale marker must also fail before plotting.
            (root / "noise/completed.json").write_text('{"run_hash":"different"}', encoding="utf-8")
            with patch.object(plots.RegressionPlotter, "plot_bar") as draw:
                with self.assertRaisesRegex(ValueError, "does not match"):
                    replot_completed_bars(root, output)
                draw.assert_not_called()
            (root / "noise/completed.json").write_text('{"run_hash":"same"}', encoding="utf-8")
            for name in ("clean", "noise"):
                path = root / name / "run_manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest["dataset"].update(noise_dir_name=name, max_freq_hz="maxfreq=3kHz")
                manifest["validation_config"]["data"] = {"noise_dir_names": ["clean", "noise", "not_created"]}
                path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(plots.RegressionPlotter, "plot_bar") as draw:
                with self.assertRaisesRegex(ValueError, "not all present"):
                    replot_completed_bars(root, output)
                draw.assert_not_called()

    def test_completed_csv_redraw_preserves_sources_and_needs_no_training_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "condition"
            root.mkdir()
            manifest = {"run_hash": "finished", "dataset": {"snr_value": "no_noise"},
                        "validation_config": {"run": {"epochs": 200}}}
            (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "completed.json").write_text('{"run_hash":"finished"}', encoding="utf-8")
            summary = root / "metrics_summary_no_noise.csv"
            summary.write_text("model,r2_mean,r2_se,roc_auc_cont_mean,roc_auc_cont_se,pr_auc_cont_mean,pr_auc_cont_se\n"
                               "RandomForest,0.981234,0.001,0.991234,0.002,0.995678,0.001\n", encoding="utf-8")
            before = hashlib.sha256(summary.read_bytes()).hexdigest()
            output = Path(tmp) / "redrawn"
            modules_before = set(sys.modules)
            result = replot_completed_bars(root, output)
            self.assertEqual(len(result["files"]), 6)
            self.assertTrue(all((output / path).is_file() for path in result["files"]))
            self.assertTrue((output / "redraw_manifest.json").is_file())
            self.assertEqual(hashlib.sha256(summary.read_bytes()).hexdigest(), before)
            self.assertFalse((root / "bar").exists())
            self.assertEqual(result["new_fits"], 0)
            self.assertEqual(result["new_model_inference"], 0)
            self.assertNotIn("run_ensemble_regression_onb", set(sys.modules) - modules_before)


if __name__ == "__main__":
    unittest.main()
