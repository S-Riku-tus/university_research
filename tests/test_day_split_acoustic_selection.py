import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from test_learning_policy import fixture, ObservedTrainer, SilentPlotter
from utils.ensemble.ensemble_runtime import EnsembleManager
from utils.ensemble.ensemble_weighting import EnsembleWeighting
from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.experiment.learning_policy import normalize_learning_policy, checked_metadata
from utils.experiment.learning_runner import run_learning_experiments
from utils.training.internal_validation import internal_splits
from utils.experiment.spectral_peaks import PSD_UNIT, SPECTRUM_METHOD


class DaySelectionTest(unittest.TestCase):
    def test_performance_weights_stabilize_near_perfect_scores(self):
        result = EnsembleWeighting().compute_weights("performance_kfold",
            [{"key": "a"}, {"key": "b"}], {"a": 0., "b": 1e-12})
        self.assertEqual(result, {"a": .5, "b": .5})

    def policy(self):
        return {"split_mode": "explicit_days", "training_noise": "clean_only",
                "train_experiments": ["day-a", "day-c"], "test_experiments": ["day-b"],
                "internal_validation": "chunk_kfold"}

    def test_bad_day_selection_rejected(self):
        for updates in ({"test_experiments": ["day-a"]}, {"train_experiments": []},
                        {"test_experiments": ["missing"]}, {"train_experiments": ["day-a", "day-a"]}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                normalize_learning_policy({**self.policy(), **updates}, ["day-a", "day-b", "day-c"])

    def test_ordinary_kfold_modes_cover_once(self):
        rows = [{"experiment_name": "a", "source_wav_id": f"wav-{i // 4}", "chunk_index": i % 4}
                for i in range(24)]
        for mode in ("chunk_kfold", "wav_kfold"):
            splits = internal_splits(rows, 3, 42, mode)
            self.assertEqual(sorted(np.concatenate([held for _, held in splits]).tolist()), list(range(24)))
            shared = []
            for fit, held in splits:
                self.assertFalse(set(fit) & set(held))
                shared.append({rows[i]["source_wav_id"] for i in fit} & {rows[i]["source_wav_id"] for i in held})
            self.assertEqual(any(shared), mode == "chunk_kfold")

    def test_selection_kfold_training_and_full_test_end_to_end(self):
        self.run_selection_pipeline("background_quantile")

    def test_peak_selection_kfold_training_and_full_test_end_to_end(self):
        self.run_selection_pipeline("peak_height")

    def run_selection_pipeline(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.policy()
            jobs, specs, _, config = fixture(root, policy, days=["day-a", "day-b", "day-c"])
            features = []
            for job in jobs:
                if job["noise_dir_name"] != "heatflux_no_noise":
                    continue
                with (job["data_path"] / "chunk_manifest.csv").open(newline="", encoding="utf-8") as inp:
                    for row in csv.DictReader(inp):
                        q = float(row["sample_filename"].split("_")[0])
                        row["heat_flux"] = q
                        row["band_2000_3000_db"] = -80 if q < job["threshold"] else (-20 if row["chunk_index"] == "0" else -90)
                        row["peak_2100_2500_psd"] = .01e-9 if q < job["threshold"] else (10e-9 if row["chunk_index"] == "0" else .1e-9)
                        row["spectrum_method"] = SPECTRUM_METHOD
                        row["spectrum_unit"] = PSD_UNIT
                        # Adversarial test-day features must never set the fit thresholds.
                        if row["experiment_name"] == "day-b":
                            row["band_2000_3000_db"] = 10000
                            row["peak_2100_2500_psd"] = 10000
                        features.append(row)
            feature_path = root / "features.csv"
            with feature_path.open("w", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=list(features[0]))
                writer.writeheader(); writer.writerows(features)
            config["thresholds"]["by_experiment"] = {"day-a": 36, "day-b": 37, "day-c": 38}
            config["acoustic_selection"] = {"enabled": True, "features_csv": str(feature_path),
                                               "feature": "band_2000_3000_db", "background_quantile": 0.99}
            if mode == "peak_height":
                config["acoustic_selection"].update(mode=mode, feature="peak_2100_2500_psd", peak_height_threshold=1e-9)
            manager = EnsembleManager({"enabled_strategy_names": ["performance_kfold"],
                                       "primary_strategy_name": "performance_kfold"}, [s["key"] for s in specs])
            config["ensemble"] = manager.snapshot()
            trainer = ObservedTrainer()
            with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), patch("tensorflow.keras.backend.clear_session"):
                for _ in range(2):
                    run_learning_experiments(jobs, policy, config, specs, [{"name": "test"}],
                                             manager, trainer, SilentPlotter(), lambda *a: None)
            # 3 inner folds + final refit, 2 models; same models reused across noises, resume skips.
            self.assertEqual(len(trainer.fits), 8)
            for x in trainer.fits + trainer.pca_fits:
                values = x[:, 0, 0, 0]
                self.assertFalse(np.any((values >= 200) & (values < 300)))
                self.assertTrue(np.all(values < 1000))
            for job in jobs:
                runs = list(job["save_base_path"].glob("maxfreq=3kHz/*/*/completed.json"))
                if job["experiment_name"] != "day-b":
                    self.assertFalse(runs)
                    continue
                directory = next((job["save_base_path"] / job["max_freq_hz"] / job["noise_dir_name"]).iterdir())
                selection = json.loads((directory / "training_selection_fold1.json").read_text())
                self.assertEqual(selection["n_before"], 24)
                self.assertEqual(selection["n_after"], 18)
                self.assertNotIn("day-b", selection["by_experiment"])
                internal = json.loads((directory / "internal_validation_fold1.json").read_text())
                self.assertEqual(len(internal["samples"]), 24)
                self.assertTrue(all(f["shared_source_wavs"] > 0 for f in internal["folds"]))
                with (directory / "fold_pred" / f"pred_f1_{job['snr_value']}.csv").open() as inp:
                    self.assertEqual(len(list(csv.DictReader(inp))), 12)


if __name__ == "__main__":
    unittest.main()
