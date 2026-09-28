"""固定テストWAVが内部検証・PCA・学習へ入らないことを検証する。"""

import contextlib
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
from utils.experiment.learning_policy import (
    checked_metadata, experiment_split_kind, normalize_learning_policy,
    outer_splits, resolve_experiment_names, wav_groups,
)
from utils.experiment.learning_runner import run_learning_experiments


class WithinDayHoldoutTest(unittest.TestCase):
    def policy(self, noise="clean_only"):
        return {"evaluation_mode": "within_day", "within_day_experiment": "day-a",
                "train_experiments": ["unused-a"], "test_experiments": ["unused-b"],
                "test_fraction": 0.25, "test_split_seed": 42, "training_noise": noise}

    def metadata(self):
        return checked_metadata([
            {"source_wav_id": f"wav-{wav}", "chunk_index": chunk,
             "sample_filename": f"{wav * 10}_{wav}_{chunk}.npy"}
            for wav in range(8) for chunk in range(2)
        ], "day-a")

    def test_explicit_modes_resolve_days_and_reject_invalid_config(self):
        self.assertEqual(resolve_experiment_names({}, self.policy()), ["day-a"])
        normalized = normalize_learning_policy(self.policy(), ["day-a"])
        self.assertEqual(normalized["train_experiments"], ["day-a"])
        self.assertEqual(normalized["test_experiments"], ["day-a"])
        self.assertEqual(experiment_split_kind(normalized), "within_day_holdout")
        for updates in ({"test_fraction": 0}, {"test_fraction": 1},
                        {"test_split_seed": -1}, {"within_day_experiment": ""},
                        {"evaluation_mode": "typo"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                normalize_learning_policy({**self.policy(), **updates}, ["day-a"])
        with self.assertRaisesRegex(ValueError, "完全に分離"):
            experiment_split_kind({"evaluation_mode": "cross_day",
                                   "train_experiments": ["day-a"], "test_experiments": ["day-a"]})

    def test_holdout_is_fixed_across_noise_order_and_inner_fold_count(self):
        metadata = self.metadata()
        fit, test = outer_splits(metadata, metadata, 1, self.policy())[0]
        groups = wav_groups(metadata)
        self.assertEqual(len(set(groups[test])), 2)
        self.assertEqual(len(set(groups[fit])), 6)
        self.assertFalse(set(groups[test]) & set(groups[fit]))
        reversed_metadata = list(reversed(metadata))
        fit2, test2 = outer_splits(reversed_metadata, reversed_metadata, 5, self.policy("matched"))[0]
        self.assertEqual(set(groups[test]), set(wav_groups(reversed_metadata)[test2]))
        self.assertEqual(set(groups[fit]), set(wav_groups(reversed_metadata)[fit2]))
        _, aligned_test = outer_splits(metadata, reversed_metadata, 1, self.policy())[0]
        self.assertEqual(set(groups[test]), set(wav_groups(reversed_metadata)[aligned_test]))

    def test_onb_stratification_keeps_both_classes_and_requires_valid_groups(self):
        metadata = self.metadata()
        policy = {**self.policy(), "test_stratify": "onb"}
        fit, test = outer_splits(metadata, metadata, 1, policy, threshold=40)[0]
        for indices in (fit, test):
            self.assertEqual({int(metadata[i]["sample_filename"].split("_")[0]) >= 40
                              for i in indices}, {False, True})
        with self.assertRaisesRegex(ValueError, "ONB閾値"):
            outer_splits(metadata, metadata, 1, policy)
        with self.assertRaisesRegex(ValueError, "2本以上"):
            outer_splits(metadata, metadata, 1, policy, threshold=70)

    def test_runner_excludes_test_from_inner_cv_preprocessing_and_fit(self):
        for noise in ("clean_only", "matched"):
            with self.subTest(noise=noise), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                policy = normalize_learning_policy({**self.policy(noise), "test_stratify": "onb"}, ["day-a"])
                jobs, specs, _, config = fixture(root, policy, days=("day-a",))
                manager = EnsembleManager({"enabled_strategy_names": ["performance_kfold", "simple_equal"]},
                                          [s["key"] for s in specs])
                config["ensemble"] = manager.snapshot()
                trainer = ObservedTrainer()
                call = lambda: run_learning_experiments(
                    jobs, policy, config, specs, [{"name": "test"}], manager,
                    trainer, SilentPlotter(), lambda *args: None)
                with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), patch("tensorflow.keras.backend.clear_session"):
                    call()
                    fit_count = len(trainer.fits)
                    call()
                self.assertEqual(len(trainer.fits), fit_count)
                self.assertEqual(fit_count, (1 if noise == "clean_only" else 2) * 4 * 2)
                heldout_sets, fit_ids = [], []
                for job in jobs:
                    directory = next((job["save_base_path"] / job["max_freq_hz"] / job["noise_dir_name"]).iterdir())
                    split = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(split["learning_context"]["evaluation_scheme"], "within_day_holdout")
                    self.assertEqual(len(split["folds"]), 1)
                    fold = split["folds"][0]
                    held = set(fold["evaluation_wav_groups"])
                    self.assertEqual(len(held), 2)  # ceil(6 * .25)
                    self.assertFalse(held & set(fold["training_wav_groups"]))
                    self.assertEqual(fold["n_evaluation_chunks"], 4)
                    inner = json.loads((directory / "internal_validation_fold1.json").read_text(encoding="utf-8"))
                    self.assertFalse(held & set(wav_groups(inner["samples"])))
                    self.assertTrue(all(f["shared_source_wavs"] == 0 for f in inner["folds"]))
                    heldout_sets.append(held)
                    fit_ids.append(fold["fit_id"])
                    self.assertTrue((directory / "completed.json").exists())
                self.assertEqual(heldout_sets[0], heldout_sets[1])
                self.assertEqual(len(set(fit_ids)), 1 if noise == "clean_only" else 2)
                held_wav_numbers = {int(json.loads(g)[1].split("-")[1]) for g in heldout_sets[0]}
                for x_fit in trainer.fits + trainer.pca_fits:
                    wav_numbers = set(x_fit[:, 1, 0, 0].astype(int))
                    self.assertFalse(wav_numbers & held_wav_numbers)


if __name__ == "__main__":
    unittest.main()
