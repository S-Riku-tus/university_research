"""WAV内chunk分割の再現性と、外側テストの学習非使用を検証する。"""

import contextlib
import hashlib
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


class WithinWavChunkTest(unittest.TestCase):
    def chunk_policy(self, noise="clean_only"):
        return {
            "evaluation_mode": "within_wav_chunk",
            "within_wav_chunk_experiment": "day-a",
            "train_experiments": ["unused-a"],
            "test_experiments": ["unused-b"],
            "test_fraction": 0.25,
            "test_split_seed": 42,
            "training_noise": noise,
        }

    def test_explicit_modes_resolve_days_and_reject_invalid_config(self):
        self.assertEqual(resolve_experiment_names({}, self.chunk_policy()), ["day-a"])
        normalized = normalize_learning_policy(self.chunk_policy(), ["day-a"])
        self.assertEqual(normalized["train_experiments"], ["day-a"])
        self.assertEqual(normalized["test_experiments"], ["day-a"])
        self.assertEqual(experiment_split_kind(normalized), "within_wav_chunk_holdout")
        for updates in ({"test_fraction": 0}, {"test_fraction": 1},
                        {"test_split_seed": -1}, {"within_wav_chunk_experiment": ""},
                        {"evaluation_mode": "typo"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                normalize_learning_policy({**self.chunk_policy(), **updates}, ["day-a"])
        with self.assertRaisesRegex(ValueError, "完全に分離"):
            experiment_split_kind({"evaluation_mode": "cross_day",
                                   "train_experiments": ["day-a"], "test_experiments": ["day-a"]})

    def test_mode_specific_settings_only_apply_selected_section(self):
        settings = {
            "cross_day": {
                "train_experiments": ["day-b"],
                "test_experiments": ["day-c"],
            },
            "within_wav_chunk": {
                "experiment": "day-a",
                "test_fraction": 0.25,
                "test_split_seed": 42,
            },
        }
        policy = normalize_learning_policy({
            "evaluation_mode": "within_wav_chunk",
            "evaluation_settings": settings,
            "training_noise": "clean_only",
        }, ["day-a"])
        self.assertEqual(policy["train_experiments"], ["day-a"])
        self.assertEqual(policy["test_experiments"], ["day-a"])
        self.assertEqual(policy["test_fraction"], 0.25)
        self.assertNotIn("evaluation_settings", policy)

        cross = normalize_learning_policy({
            "evaluation_mode": "cross_day",
            "evaluation_settings": settings,
            "training_noise": "matched",
        }, ["day-b", "day-c"])
        self.assertEqual(cross["train_experiments"], ["day-b"])
        self.assertEqual(cross["test_experiments"], ["day-c"])

        with self.assertRaisesRegex(ValueError, "重複指定"):
            normalize_learning_policy({
                "evaluation_mode": "within_wav_chunk",
                "evaluation_settings": settings,
                "within_wav_chunk_experiment": "day-a",
            }, ["day-a"])

    def test_removed_mode_and_implicit_single_day_cv_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "evaluation_mode"):
            normalize_learning_policy({**self.chunk_policy(), "evaluation_mode": "within_day"}, ["day-a"])
        with self.assertRaisesRegex(ValueError, "未知の評価方式"):
            normalize_learning_policy({
                "evaluation_mode": "within_wav_chunk",
                "evaluation_settings": {
                    "within_wav_chunk": {"experiment": "day-a"},
                    "within_day": {"experiment": "day-a"},
                },
            }, ["day-a"])
        with self.assertRaisesRegex(ValueError, "明示"):
            experiment_split_kind({"train_experiments": ["day-a"], "test_experiments": ["day-a"]})

    def test_within_wav_chunk_selects_every_wav_without_reusing_a_chunk(self):
        metadata = checked_metadata([
            {"source_wav_id": f"wav-{wav}", "chunk_index": chunk,
             "sample_filename": f"{wav * 10}_{wav}_{chunk}.npy"}
            for wav in range(4) for chunk in range(8)
        ], "day-a")
        policy = normalize_learning_policy(self.chunk_policy(), ["day-a"])
        self.assertEqual(experiment_split_kind(policy), "within_wav_chunk_holdout")
        fit, test = outer_splits(metadata, metadata, policy)[0]
        groups = wav_groups(metadata)
        self.assertEqual(len(fit), 24)
        self.assertEqual(len(test), 8)
        self.assertEqual(set(groups[fit]), set(groups[test]))
        fit_keys = {(metadata[i]["source_wav_id"], metadata[i]["chunk_index"]) for i in fit}
        test_keys = {(metadata[i]["source_wav_id"], metadata[i]["chunk_index"]) for i in test}
        self.assertFalse(fit_keys & test_keys)
        for group in set(groups):
            self.assertEqual(np.sum(groups[test] == group), 2)
            self.assertEqual(np.sum(groups[fit] == group), 6)

        reversed_metadata = list(reversed(metadata))
        fit2, test2 = outer_splits(metadata, reversed_metadata, policy)[0]
        self.assertEqual(
            test_keys,
            {(reversed_metadata[i]["source_wav_id"], reversed_metadata[i]["chunk_index"])
             for i in test2},
        )
        with self.assertRaisesRegex(ValueError, "未知のlearning_policy"):
            normalize_learning_policy({**self.chunk_policy(), "test_stratify": "onb"}, ["day-a"])

    def test_both_noise_policies_keep_the_same_chunk_split_and_result_names(self):
        from utils.experiment.learning_policy import policy_result_date_dir
        from utils.experiment.result_paths import result_scope_dir_name

        metadata = checked_metadata([
            {"source_wav_id": f"wav-{wav}", "chunk_index": chunk,
             "sample_filename": f"{wav * 10}_{wav}_{chunk}.npy"}
            for wav in range(4) for chunk in range(8)
        ], "day-a")
        clean = self.chunk_policy()
        noisy = self.chunk_policy("matched")
        for a, b in zip(outer_splits(metadata, metadata, clean)[0],
                        outer_splits(metadata, metadata, noisy)[0]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(policy_result_date_dir("onb", clean), "onb__wc_clean")
        job = {"experiment_name": "2025.06.11_0.3_2_6.18_0.3_3"}
        config = {"learning_policy": {
            **clean, "within_wav_chunk_experiment": job["experiment_name"],
        }, "run": {"folds": 3, "epochs": 150}, "data": {"chunk_seconds": 1},
            "acoustic_selection": {"peak_height_threshold": None}}
        self.assertEqual(
            result_scope_dir_name("onb", job, config, "170655", "hash"),
            "onb_wc-t0611-v0611_iw3-nc_c1s_s0_e150_170655",
        )

    def test_combined_within_wav_chunk_reuses_each_source_days_single_day_test_chunks(self):
        def single_day_metadata(day):
            return checked_metadata([
                {
                    "source_wav_id": f"wav-{wav:02d}",
                    "chunk_index": chunk,
                    "sample_filename": f"{wav * 10}_{wav}_{chunk}.npy",
                }
                for wav in range(18) for chunk in range(60)
            ], day)

        expected = set()
        for day in ("day-a", "day-b"):
            metadata = single_day_metadata(day)
            day_policy = {**self.chunk_policy(), "within_wav_chunk_experiment": day}
            _, test = outer_splits(metadata, metadata, day_policy)[0]
            selected = {
                (day, metadata[index]["source_wav_id"], int(metadata[index]["chunk_index"]))
                for index in test
            }
            self.assertEqual(len(selected), 270)
            # 削除前のseed 42の分割を固定し、方式削除による変化を検査する。
            original_keys = sorted((wav, chunk) for _, wav, chunk in selected)
            self.assertEqual(
                hashlib.sha256(json.dumps(original_keys).encode()).hexdigest(),
                "f226515355c8d0c952757d04364198c4b7139cc52833d5da67d3c3a080386776",
            )
            expected.update(selected)

        combined_name = "day-a_day-b"
        combined = checked_metadata([
            {
                "source_wav_id": f"{day}::{wav_id}",
                "chunk_index": row["chunk_index"],
                "sample_filename": row["sample_filename"],
                "source_experiment_name": day,
                "original_source_wav_id": wav_id,
            }
            for day in ("day-a", "day-b")
            for row in single_day_metadata(day)
            for wav_id in [row["source_wav_id"]]
        ], combined_name)
        combined_policy = {
            **self.chunk_policy(),
            "within_wav_chunk_experiment": combined_name,
        }
        _, combined_test = outer_splits(combined, combined, combined_policy)[0]
        actual = {
            (
                combined[index]["source_experiment_name"],
                combined[index]["original_source_wav_id"],
                int(combined[index]["chunk_index"]),
            )
            for index in combined_test
        }
        self.assertEqual(len(actual), 540)
        self.assertEqual(actual, expected)
        for day in ("day-a", "day-b"):
            self.assertEqual(sum(key[0] == day for key in actual), 270)

    def test_runner_within_wav_chunk_keeps_all_wavs_on_both_outer_sides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = normalize_learning_policy(self.chunk_policy(), ["day-a"])
            jobs, specs, _, config = fixture(
                root, policy, evaluated_noises=("heatflux_no_noise",), days=("day-a",)
            )
            manager = EnsembleManager(
                {"enabled_strategy_names": ["performance_kfold", "simple_equal"]},
                [s["key"] for s in specs],
            )
            config["ensemble"] = manager.snapshot()
            trainer = ObservedTrainer()
            with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), patch("tensorflow.keras.backend.clear_session"):
                run_learning_experiments(
                    jobs, policy, config, specs, [{"name": "test"}], manager,
                    trainer, SilentPlotter(), lambda *args: None,
                )
            job = jobs[0]
            directory = next(
                (job["save_base_path"] / job["max_freq_hz"] / job["noise_dir_name"]).iterdir()
            )
            saved = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                saved["learning_context"]["evaluation_scheme"],
                "within_wav_chunk_holdout",
            )
            fold = saved["folds"][0]
            self.assertEqual(len(fold["training_wav_groups"]), 6)
            self.assertEqual(fold["training_wav_groups"], fold["evaluation_wav_groups"])
            self.assertEqual(fold["n_training_chunks"], 6)
            self.assertEqual(fold["n_evaluation_chunks"], 6)

    def test_runner_excludes_test_from_inner_cv_preprocessing_and_fit(self):
        for noise in ("clean_only", "matched"):
            with self.subTest(noise=noise), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                policy = normalize_learning_policy(self.chunk_policy(noise), ["day-a"])
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
                training_sets, heldout_sets, fit_ids = [], [], []
                for job in jobs:
                    directory = next((job["save_base_path"] / job["max_freq_hz"] / job["noise_dir_name"]).iterdir())
                    split = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(split["learning_context"]["evaluation_scheme"], "within_wav_chunk_holdout")
                    self.assertEqual(len(split["folds"]), 1)
                    fold = split["folds"][0]
                    held = set(fold["evaluation_wav_groups"])
                    self.assertEqual(len(held), 6)
                    self.assertEqual(held, set(fold["training_wav_groups"]))
                    self.assertEqual(fold["n_evaluation_chunks"], 6)
                    inner = json.loads((directory / "internal_validation_fold1.json").read_text(encoding="utf-8"))
                    self.assertEqual(held, set(wav_groups(inner["samples"])))
                    self.assertTrue(all(f["shared_source_wavs"] == 0 for f in inner["folds"]))
                    training_sets.append({(row["source_wav_id"], int(row["chunk_index"]))
                                          for row in inner["samples"]})
                    heldout_sets.append(set(fold["evaluation_sample_indices"]))
                    fit_ids.append(fold["fit_id"])
                    self.assertTrue((directory / "completed.json").exists())
                self.assertEqual(heldout_sets[0], heldout_sets[1])
                self.assertEqual(len(set(fit_ids)), 1 if noise == "clean_only" else 2)
                self.assertEqual(training_sets[0], training_sets[1])
                training_chunks = training_sets[0]
                for x_fit in trainer.fits + trainer.pca_fits:
                    self.assertTrue({(f"wav-{int(wav)}", int(chunk))
                                     for wav, chunk in x_fit[:, 1, :, 0]} <= training_chunks)


if __name__ == "__main__":
    unittest.main()
