"""WAV isolation, actual median objective, fixed-weight transfer and compatibility."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from utils.ensemble.crossfit_stacking import fit_crossfit_strategy, group_crossfit_splits
from utils.ensemble.ensemble_runtime import EnsembleManager
from utils.experiment.learning_runner import run_learning_experiments
from utils.experiment.run_helpers import run_config_digest
from test_learning_policy import fixture, ObservedTrainer, SilentPlotter


NAMES = ["subset_equal_cv", "crossfit_wav_stack", "crossfit_shrinkage_stack"]


class CrossfitStackingTest(unittest.TestCase):
    def test_unequal_chunk_counts_never_split_wav_and_cover_exactly_once(self):
        groups = np.repeat([f"wav-{i}" for i in range(12)], np.arange(1, 13))
        splits = group_crossfit_splits(groups, 4, 42)
        seen = np.zeros(len(groups), dtype=int)
        for fit, held in splits:
            self.assertFalse(set(groups[fit]) & set(groups[held]))
            self.assertEqual(len(set(groups[held])), 3)
            seen[held] += 1
        np.testing.assert_array_equal(seen, 1)
        again = group_crossfit_splits(groups, 4, 42)
        for first, second in zip(splits, again):
            np.testing.assert_array_equal(first[1], second[1])
        with self.assertRaises(ValueError):
            group_crossfit_splits(["a", "a", "b"])

    def test_subset_drops_harmful_model_or_selects_single(self):
        y = np.arange(6, dtype=float)
        groups = np.arange(6)
        predictions = {"a": y + 1, "b": y - 1, "bad": y + 10}
        weights, diagnostic = fit_crossfit_strategy(y, predictions, groups, list(predictions), NAMES[0])
        self.assertEqual(weights, {"a": 0.5, "b": 0.5, "bad": 0.0})
        self.assertEqual(diagnostic["selected_wav_mse"], 0)
        predictions["a"] = y.copy()
        predictions["b"] = y + 3
        weights, _ = fit_crossfit_strategy(y, predictions, groups, list(predictions), NAMES[0])
        self.assertEqual(weights, {"a": 1.0, "b": 0.0, "bad": 0.0})

    def test_joint_error_cancellation_beats_each_single_and_equal(self):
        y = np.linspace(0, 10, 8)
        predictions = {"a": y + 1, "b": y - 3, "bad": y + 12}
        weights, diagnostic = fit_crossfit_strategy(y, predictions, np.arange(8), list(predictions), NAMES[1])
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertTrue(all(value >= 0 for value in weights.values()))
        self.assertLess(diagnostic["selected_wav_mse"], 1e-7)
        self.assertGreater(diagnostic["equal_wav_mse"], 1)

    def test_median_order_and_equal_wav_weighting_are_preserved(self):
        # median(a)=median(b)=0, but median((a+b)/2)=5.
        y = np.repeat([5., 15., 25.], 3)
        groups = np.repeat(["a", "b", "c"], 3)
        offset = np.repeat([0., 10., 20.], 3)
        predictions = {"left": offset + np.tile([0., 0., 10.], 3),
                       "right": offset + np.tile([0., 10., 0.], 3)}
        for name in NAMES:
            weights, diagnostic = fit_crossfit_strategy(y, predictions, groups, list(predictions), name)
            self.assertAlmostEqual(diagnostic["selected_wav_mse"], 0)
            self.assertEqual(weights, {"left": 0.5, "right": 0.5})
        # Duplicating an entire WAV's chunk sequence must not increase its weight.
        repeat = np.concatenate([np.tile(np.arange(3), 9), np.arange(3, 9)])
        for name in NAMES:
            first, _ = fit_crossfit_strategy(y, predictions, groups, list(predictions), name)
            second, _ = fit_crossfit_strategy(y[repeat], {k: v[repeat] for k, v in predictions.items()},
                                             groups[repeat], list(predictions), name)
            np.testing.assert_allclose(list(first.values()), list(second.values()), atol=1e-7)

    def test_shrinkage_moves_weights_towards_equal_and_is_scale_invariant(self):
        y = np.linspace(0, 20, 10)
        predictions = {"good": y, "bad": y + 5}
        plain, _ = fit_crossfit_strategy(y, predictions, np.arange(10), list(predictions), NAMES[1])
        regularized, _ = fit_crossfit_strategy(y, predictions, np.arange(10), list(predictions), NAMES[2],
                                               {"regularization": 0.1})
        self.assertGreater(regularized["bad"], plain["bad"] + 0.01)
        scaled, _ = fit_crossfit_strategy(y * 1e5, {k: v * 1e5 for k, v in predictions.items()},
                                          np.arange(10), list(predictions), NAMES[2], {"regularization": 0.1})
        np.testing.assert_allclose(list(regularized.values()), list(scaled.values()), atol=1e-5)

    def test_degenerate_predictions_invalid_data_and_solver_failure(self):
        y = np.ones(4)
        same = {"a": y.copy(), "b": y.copy()}
        weights, diagnostic = fit_crossfit_strategy(y, same, np.arange(4), list(same), NAMES[1])
        self.assertEqual(weights, {"a": 0.5, "b": 0.5})
        self.assertEqual(diagnostic["residual_correlation"], [[None, None], [None, None]])
        with self.assertRaises(ValueError):
            fit_crossfit_strategy(y, {"a": y, "b": [1, 2, 3, np.nan]}, np.arange(4), list(same), NAMES[1])
        with self.assertRaises(ValueError):
            fit_crossfit_strategy([1, 2, 3, 4], same, ["a", "a", "b", "c"], list(same), NAMES[1])
        failed = SimpleNamespace(x=np.asarray([0.5, 0.5]), success=False, message="iteration limit", nit=1)
        with patch("utils.ensemble.crossfit_stacking.minimize", return_value=failed):
            weights, diagnostic = fit_crossfit_strategy(y, {"a": y, "b": y + 10},
                                                        np.arange(4), list(same), NAMES[1])
        self.assertEqual(weights["a"], 1)
        self.assertTrue(diagnostic["solver_fallback"])

    def test_missing_weights_and_wrong_fold_fail_instead_of_using_equal(self):
        manager = EnsembleManager({"enabled_strategy_names": NAMES}, ["a", "b"])
        run = manager.create_run([{"key": "a"}, {"key": "b"}])
        with self.assertRaises(ValueError):
            run.combine_predictions({"a": np.ones(4), "b": np.zeros(4)}, {}, 1)
        with self.assertRaises(ValueError):
            run.combine_predictions({}, {}, 2, {"fold": 1, "weights": {}})

    def test_legacy_config_has_no_new_settings_or_extra_fitting(self):
        selection = {"enabled_strategy_names": ["simple_equal", "inner_holdout"],
                     "primary_strategy_name": "inner_holdout"}
        manager = EnsembleManager(selection, ["rf", "second"])
        self.assertNotIn("crossfit", str(manager.snapshot()))
        run = manager.create_run([{"key": "rf"}, {"key": "second"}])
        self.assertIsNone(run.fit_crossfit_weights(None, None, None, None, None, None, None, 1, 3))
        predictions = {"rf": np.asarray([2., 4.]), "second": np.asarray([8., 12.])}
        result = run.combine_predictions(predictions, {"rf": 1., "second": 3.}, 1)
        np.testing.assert_array_equal(result["ensemble__simple_equal"]["prediction"], [5, 8])
        np.testing.assert_array_equal(result["ensemble__inner_holdout"]["prediction"], [3.5, 6])


class CrossfitPipelineTest(unittest.TestCase):
    def test_actual_keras_crossfit_without_pca(self):
        from tensorflow import keras
        from utils.training.model_training import ModelTrainer
        groups = np.repeat(np.arange(4), 2)
        y = np.repeat(np.arange(4, dtype=float), 2)
        x = np.arange(32, dtype=np.float32).reshape(8, 2, 2, 1) / 32
        specs = [{"key": key, "label": key, "kind": "keras", "lr": 0.001,
                  "batch_size": 4, "builder": lambda mm: keras.Sequential([
                      keras.layers.Input(shape=(2, 2, 1)), keras.layers.Flatten(), keras.layers.Dense(1)])}
                 for key in ("first", "second")]
        run = EnsembleManager({"enabled_strategy_names": NAMES}, [s["key"] for s in specs]).create_run(specs)
        trainer = ModelTrainer()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(trainer, "make_pca", side_effect=AssertionError("Keras-only PCA")):
            result = run.fit_crossfit_weights(trainer, x, y, groups, 2, (2, 2, 1), 1, 1, 3)
        self.assertEqual(len(result["splits"]), 4)
        self.assertEqual(set(result["weights"]), set(NAMES))
        for split in result["splits"]:
            self.assertTrue(all(item["epochs_completed"] == 1 for item in split["model_training"].values()))
        self.assertTrue(all(np.isfinite(prediction).all() for prediction in result["oof_predictions"].values()))

    def test_all_policies_oof_sharing_audit_resume_and_legacy_predictions(self):
        for split in ("within_day", "leave_one_day_out"):
            for noise in ("matched", "clean_only"):
                with self.subTest(split=split, noise=noise), tempfile.TemporaryDirectory() as temp:
                    policy = {"split_mode": split, "training_noise": noise}
                    jobs, specs, baseline, config = fixture(Path(temp), policy)
                    manager = EnsembleManager({"enabled_strategy_names": ["simple_equal", "inner_holdout", *NAMES],
                                               "primary_strategy_name": "inner_holdout"}, [s["key"] for s in specs])
                    manager.validate(specs)
                    original_hash = run_config_digest(config, {}, specs, "models", True)
                    config["ensemble"] = manager.snapshot()
                    self.assertNotEqual(original_hash, run_config_digest(config, {}, specs, "models", True))
                    trainer = ObservedTrainer()
                    call = lambda: run_learning_experiments(jobs, policy, config, specs, [{"name": "test"}],
                                                            manager, trainer, SilentPlotter(), lambda *args: None)
                    with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), patch("tensorflow.keras.backend.clear_session"):
                        call()
                        count = len(trainer.fits)
                        call()
                    self.assertEqual(len(trainer.fits), count)
                    folds = 3 if split == "within_day" else 1
                    families = 2 * (2 if noise == "matched" else 1)
                    self.assertEqual(count, families * folds * 2 * 6)  # holdout + 4 shared OOF + outer
                    if noise == "clean_only":
                        self.assertTrue(all(np.max(x[:, 0, 0, 0]) < 1000 for x in trainer.fits + trainer.pca_fits))
                    by_day = {}
                    legacy_predictions = {}
                    for job in jobs:
                        directory = next((job["save_base_path"] / job["max_freq_hz"] / job["noise_dir_name"]).iterdir())
                        outer = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
                        audits = []
                        for fold in outer["folds"]:
                            number = fold["fold"]
                            audit = json.loads((directory / f"ensemble_crossfit_fit_f{number}.json").read_text(encoding="utf-8"))
                            audits.append(audit)
                            seen = []
                            for inner in audit["splits"]:
                                train, held = set(inner["training_wav_groups"]), set(inner["heldout_wav_groups"])
                                self.assertFalse(train & held)
                                self.assertEqual(train | held, set(fold["training_wav_groups"]))
                                self.assertFalse((train | held) & set(fold["evaluation_wav_groups"]))
                                seen.extend(held)
                            self.assertEqual(sorted(seen), sorted(fold["training_wav_groups"]))
                            with (directory / f"ensemble_inner_oof_f{number}.csv").open(encoding="utf-8") as source:
                                rows = list(csv.DictReader(source))
                            self.assertEqual(len(rows), fold["n_training_chunks"])
                            with (directory / "fold_pred" / f"pred_f{number}_{job['snr_value']}.csv").open(encoding="utf-8") as source:
                                rows = list(csv.DictReader(source))
                            for row in rows:
                                for name in NAMES:
                                    expected = sum(audit["weights"][name][key] * float(row[key]) for key in ("rf", "second"))
                                    self.assertAlmostEqual(float(row["ensemble__" + name]), expected)
                            legacy_predictions[(job["experiment_name"], job["snr_value"], number)] = rows
                        by_day.setdefault(job["experiment_name"], []).append(audits)
                        with (directory / "wav_eval" / f"wav_metrics_{job['snr_value']}.csv").open(encoding="utf-8") as source:
                            self.assertEqual(len(list(csv.DictReader(source))), 21)  # 7 outputs x 3 aggregations
                        trends = collect_trends(directory)
                        self.assertTrue(all(any(row["strategy"] == name for row in trends) for name in NAMES))
                    if noise == "clean_only":
                        for clean, noisy in by_day.values():
                            self.assertEqual(clean, noisy)
                    # Re-run old configuration into a separate directory, comparing all existing prediction columns.
                    config["ensemble"] = baseline.snapshot()
                    for job in jobs:
                        job["save_base_path"] = job["save_base_path"].parent / "baseline_results"
                    with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), patch("tensorflow.keras.backend.clear_session"):
                        run_learning_experiments(jobs, policy, config, specs, [{"name": "test"}], baseline,
                                                 ObservedTrainer(), SilentPlotter(), lambda *args: None)
                    for job in jobs:
                        directory = next((job["save_base_path"] / job["max_freq_hz"] / job["noise_dir_name"]).iterdir())
                        for number in range(1, folds + 1):
                            with (directory / "fold_pred" / f"pred_f{number}_{job['snr_value']}.csv").open(encoding="utf-8") as source:
                                rows = list(csv.DictReader(source))
                            previous = legacy_predictions[(job["experiment_name"], job["snr_value"], number)]
                            for old, new in zip(rows, previous):
                                for key in ("rf", "second", "ensemble__simple_equal", "ensemble__inner_holdout"):
                                    self.assertEqual(old[key], new[key])


def collect_trends(directory):
    from utils.plotting.noise_trend_plots import collect_noise_trend_rows
    return collect_noise_trend_rows([directory], noise_order=["no_noise", "-20"],
                                   model_keys=["rf", "second"], ensemble_strategy_names="all")


if __name__ == "__main__":
    unittest.main()
