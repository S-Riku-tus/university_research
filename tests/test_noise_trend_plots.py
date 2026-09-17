"""ノイズ曲線で条件・評価単位・AUCの種類が混在しないことを検証する。"""

import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from utils.plotting.noise_trend_plots import collect_noise_trend_rows, plot_noise_trends_from_runs


MODELS = {"randomforest": "RandomForest", "conformer": "Conformer", "alexnet": "AlexNet"}
ENSEMBLE = "ensemble__simple_equal"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(root, noise, *, run_hash="same", frequency="maxfreq=22kHz", threshold=5.0):
    directory = root / noise / frequency / "e300_test"
    directory.mkdir(parents=True)
    manifest = {
        "run_hash": run_hash, "run_dir": "e300_test",
        "dataset": {"experiment_name": "test-day", "max_freq_hz": frequency,
                    "snr_value": noise, "threshold": threshold},
        "run_specs": [{"key": key, "label": label} for key, label in MODELS.items()],
        "validation_config": {"ensemble": {
            "primary_strategy": "simple_equal",
            "resolved_strategy_plan": [{"name": "simple_equal", "result_key": ENSEMBLE,
                                         "label": "Ensemble simple equal", "claim_safe": True}],
        }},
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    labels = {**MODELS, ENSEMBLE: "Ensemble simple equal"}
    write_csv(directory / f"metrics_summary_{noise}.csv", [
        {"model": label, "r2_mean": -0.2, "r2_se": 0.1,
         "roc_auc_cont_mean": 0.9, "roc_auc_cont_se": 0.02,
         "auc_binary_mean": 0.6, "auc_binary_se": 0.03} for label in labels.values()
    ])
    write_csv(directory / "wav_eval" / f"wav_metrics_{noise}.csv", [
        {"model_key": key, "aggregation": aggregation, "n_wavs": 18, "claim_safe": 1,
         "r2": 0.4 if aggregation == "median" else 0.8,
         "roc_auc_cont": 0.95, "auc_binary": 0.7}
        for key in labels for aggregation in ("mean", "median")
    ])
    (directory / "wav_eval" / f"evaluation_manifest_{noise}.json").write_text(
        json.dumps({"threshold": threshold}), encoding="utf-8")
    return directory


class NoiseTrendPlotsTest(unittest.TestCase):
    def test_missing_noise_stays_gap_and_evaluation_units_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [make_run(Path(temp), "-20"), make_run(Path(temp), "no_noise")]
            rows = collect_noise_trend_rows(
                paths, noise_order=["-20", "0", "no_noise"], model_keys=list(MODELS),
                metrics=["r2", "roc_auc_cont", "auc_binary"])
            chunk = [r for r in rows if r["model_key"] == "randomforest" and r["evaluation_unit"] == "chunk" and r["metric"] == "r2"]
            self.assertEqual([r["noise"] for r in chunk], ["no_noise", "0", "-20"])
            self.assertEqual(chunk[0]["value"], -0.2)
            self.assertTrue(math.isnan(chunk[1]["value"]))
            self.assertEqual(chunk[0]["standard_error"], 0.1)
            wav = next(r for r in rows if r["evaluation_unit"] == "wav" and r["metric"] == "r2" and r["noise"] == "no_noise")
            self.assertEqual(wav["value"], 0.4)
            self.assertTrue(math.isnan(wav["standard_error"]))
            for metric, expected in (("roc_auc_cont", 0.9), ("auc_binary", 0.6)):
                self.assertEqual(next(r["value"] for r in rows if r["metric"] == metric and r["evaluation_unit"] == "chunk" and r["noise"] == "no_noise"), expected)
            self.assertEqual({r["model_key"] for r in rows}, set(MODELS) | {ENSEMBLE})

    def test_previous_configuration_does_not_enter_resumed_curve(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [make_run(Path(temp), "no_noise"), make_run(Path(temp), "-20", run_hash="old")]
            rows = collect_noise_trend_rows(paths, noise_order=["no_noise", "-20"],
                                            model_keys=list(MODELS), expected_run_hash="same")
            self.assertTrue(all(math.isnan(r["value"]) for r in rows if r["noise"] == "-20"))

    def test_incompatible_frequency_and_threshold_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = make_run(root, "no_noise")
            different_frequency = make_run(root, "0", frequency="maxfreq=5kHz")
            with self.assertRaises(ValueError):
                collect_noise_trend_rows([first, different_frequency], noise_order=["no_noise", "0"], model_keys=list(MODELS))
            different_threshold = make_run(root, "-20", threshold=6)
            with self.assertRaises(ValueError):
                collect_noise_trend_rows([first, different_threshold], noise_order=["no_noise", "-20"], model_keys=list(MODELS))

    def test_render_and_resume_use_saved_metrics_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = make_run(root, "no_noise")
            options = dict(noise_order=["no_noise", "-20"], model_keys=list(MODELS), metrics=["r2"], formats=["png", "pdf"])
            artifacts = plot_noise_trends_from_runs([path], root / "plots", **options)
            self.assertEqual(len(artifacts), 2)
            for artifact in artifacts:
                self.assertTrue(Path(artifact["csv"]).is_file())
                for figure in artifact["figures"]:
                    self.assertGreater(Path(figure).stat().st_size, 1000)
            second = make_run(root, "-20")
            updated = plot_noise_trends_from_runs([path, second], root / "plots", **options)
            with Path(updated[0]["csv"]).open(encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(math.isfinite(float(r["value"])) for r in rows))

    def test_disabled_wav_evaluation_does_not_create_empty_wav_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = make_run(root, "no_noise")
            (path / "wav_eval" / "wav_metrics_no_noise.csv").unlink()
            rows = collect_noise_trend_rows([path], noise_order=["no_noise"], model_keys=list(MODELS))
            self.assertEqual({r["evaluation_unit"] for r in rows}, {"chunk"})


if __name__ == "__main__":
    unittest.main()
