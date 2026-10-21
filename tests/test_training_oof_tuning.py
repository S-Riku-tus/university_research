import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_ROOT))
sys.path.insert(0, str(TEST_ROOT.parent / "code"))

from test_learning_policy import ObservedTrainer, fixture
from utils.config.parameter_sets import (
    build_frequency_parameter_execution_plans,
    build_parameter_execution_plan,
    expand_parameter_sets,
    resolve_parameter_set,
)
from utils.experiment.learning_policy import normalize_learning_policy
from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.experiment.training_oof_tuning import (
    oof_regression_metrics,
    run_training_oof_tuning,
    tuning_resume_context,
    validate_resume_audit,
)


class TrainingOofTuningTest(unittest.TestCase):
    def test_frequency_grids_resolve_independent_fixed_parameters(self):
        plans = build_frequency_parameter_execution_plans({
            "type": "max_freq_active_model_grid",
            "by_max_freq_hz": {
                "maxfreq=3kHz": {
                    "randomforest": {"max_depth": [12]},
                    "conformer": {"lr": [0.001], "batch_size": [12]},
                },
                "maxfreq=22kHz": {
                    "randomforest": {"max_depth": [6]},
                    "conformer": {"lr": [0.0003], "batch_size": [8]},
                },
            },
        }, ["randomforest", "conformer"], ["maxfreq=3kHz", "maxfreq=22kHz"])

        self.assertFalse(plans["maxfreq=3kHz"]["parameter_search_enabled"])
        self.assertFalse(plans["maxfreq=22kHz"]["parameter_search_enabled"])
        params_3k = plans["maxfreq=3kHz"]["parameter_sets"][0]["models"]
        params_22k = plans["maxfreq=22kHz"]["parameter_sets"][0]["models"]
        self.assertEqual(params_3k["randomforest"]["max_depth"], 12)
        self.assertEqual(params_3k["conformer"]["batch_size"], 12)
        self.assertEqual(params_22k["randomforest"]["max_depth"], 6)
        self.assertEqual(params_22k["conformer"]["batch_size"], 8)

    def test_frequency_grids_require_every_selected_frequency(self):
        with self.assertRaisesRegex(ValueError, "maxfreq=22kHz"):
            build_frequency_parameter_execution_plans({
                "type": "max_freq_active_model_grid",
                "by_max_freq_hz": {
                    "maxfreq=3kHz": {
                        "randomforest": {"max_depth": [12]},
                    },
                },
            }, ["randomforest"], ["maxfreq=3kHz", "maxfreq=22kHz"])

    def test_singleton_active_grid_selects_normal_joint_run(self):
        parameter_sets, search_enabled = build_parameter_execution_plan({
            "type": "active_model_grid",
            "model_grids": {
                "randomforest": {"max_depth": [4]},
                "conformer": {"lr": [0.001], "batch_size": [12]},
            },
        }, ["randomforest", "conformer"])
        self.assertFalse(search_enabled)
        self.assertEqual(len(parameter_sets), 1)
        self.assertNotIn("model_key", parameter_sets[0])
        self.assertEqual(set(parameter_sets[0]["models"]),
                         {"randomforest", "conformer"})

    def test_multiple_values_trigger_model_wise_search_not_model_product(self):
        parameter_sets, search_enabled = build_parameter_execution_plan({
            "type": "active_model_grid",
            "model_grids": {
                "randomforest": {"max_depth": [3, 4]},
                "conformer": {"lr": [0.0003, 0.001], "batch_size": [12]},
                "inactive": {"value": [1, 2, 3]},
            },
        }, ["randomforest", "conformer"])
        self.assertTrue(search_enabled)
        self.assertEqual(len(parameter_sets), 4)
        self.assertEqual(
            [candidate["model_key"] for candidate in parameter_sets],
            ["randomforest", "randomforest", "conformer", "conformer"],
        )
        self.assertNotIn("inactive", {
            key
            for candidate in parameter_sets
            for key in candidate["models"]
        })

    def test_independent_grid_resolves_only_its_selected_model(self):
        specs = [
            {"key": "randomforest", "kind": "sklearn", "label": "RF"},
            {"key": "conformer", "kind": "keras", "label": "CF"},
        ]
        candidates = expand_parameter_sets({
            "type": "independent_model_grid",
            "model_grids": {
                "randomforest": {"max_depth": [3, 4]},
                "conformer": {"lr": [0.001], "batch_size": [12]},
            },
        })
        self.assertEqual(len(candidates), 3)
        resolved = [resolve_parameter_set(specs, candidate) for candidate in candidates]
        self.assertEqual([[spec["key"] for spec in item] for item in resolved],
                         [["randomforest"], ["randomforest"], ["conformer"]])

    def test_oof_metrics_use_source_day_thresholds(self):
        metadata = [
            {"experiment_name": "merged", "source_experiment_name": "a",
             "source_wav_id": "a::w1", "chunk_index": 0},
            {"experiment_name": "merged", "source_experiment_name": "b",
             "source_wav_id": "b::w1", "chunk_index": 0},
        ]
        metrics = oof_regression_metrics(
            np.asarray([9.0, 21.0]), np.asarray([11.0, 19.0]), metadata,
            {"a": 10.0, "b": 20.0}, 0.1,
        )
        self.assertEqual(metrics["n_pre_onb"], 1)
        self.assertEqual(metrics["n_post_onb"], 1)
        self.assertEqual(metrics["false_positive_rate"], 1.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_runner_ranks_candidates_without_loading_outer_test_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = normalize_learning_policy({
                "evaluation_mode": "within_wav_chunk",
                "within_wav_chunk_experiment": "day-a",
                "training_noise": "clean_only",
                "test_fraction": 0.5,
                "test_split_seed": 42,
            }, ["day-a"])
            jobs, specs, _, config = fixture(
                root, policy, evaluated_noises=("heatflux_no_noise",), days=("day-a",))
            output_dir = root / "tuning-output"
            config.update({
                "acoustic_selection": {"enabled": False},
                "thresholds": {
                    "by_experiment": {"day-a": 36.0},
                    "onb_band_frac": 0.1,
                    "provenance_by_experiment": {},
                },
            })
            candidates = [{
                "name": "rf_depth3",
                "tag": "rf_d3",
                "model_key": "randomforest",
                "models": {"randomforest": {"max_depth": 3}},
            }]
            trainer = ObservedTrainer()
            with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), \
                    patch("tensorflow.keras.backend.clear_session"):
                result = run_training_oof_tuning(
                    jobs, policy, config, specs, candidates, trainer,
                    output_dir=output_dir)
            self.assertEqual(len(trainer.fits), 3)
            self.assertEqual(result["n_outer_training_samples"], 6)
            self.assertEqual(result["n_outer_test_samples_not_scored"], 6)
            self.assertFalse(result["outer_test_used"])
            self.assertEqual(result["internal_validation_split"], "chunk_kfold")
            self.assertIn("chunk_kfold", result["candidate_ranking_scope"])
            self.assertEqual(result["selected"]["randomforest"]["candidate_name"],
                             "rf_depth3")
            saved = json.loads((output_dir / "selected_candidates.json").read_text(
                encoding="utf-8"))
            self.assertFalse(saved["outer_test_used"])
            self.assertTrue((output_dir / "candidate_metrics.csv").is_file())
            self.assertTrue((output_dir / "parameter_search_config.json").is_file())
            audit = json.loads(next((output_dir / "candidate_audits").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(audit["internal_validation"]["method"], "chunk_kfold")
            self.assertTrue(all(f["shared_samples"] == 0 for f in audit["internal_validation"]["folds"]))

    def test_resume_reuses_imported_and_new_completed_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = normalize_learning_policy({
                "evaluation_mode": "within_wav_chunk", "within_wav_chunk_experiment": "day-a",
                "training_noise": "clean_only", "test_fraction": 0.5, "test_split_seed": 42,
            }, ["day-a"])
            jobs, specs, _, config = fixture(
                root, policy, evaluated_noises=("heatflux_no_noise",), days=("day-a",))
            config.update({"acoustic_selection": {"enabled": False}, "thresholds": {
                "by_experiment": {"day-a": 36.0}, "onb_band_frac": 0.1,
                "provenance_by_experiment": {},
            }})
            candidate = {"name": "old-name", "tag": "old-tag", "model_key": "randomforest",
                         "models": {"randomforest": {"max_depth": 3}}}
            second = {"name": "fresh", "tag": "fresh", "model_key": "randomforest",
                      "models": {"randomforest": {"max_depth": 4}}}

            def run(output, candidates):
                trainer = ObservedTrainer()
                with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), \
                        patch("tensorflow.keras.backend.clear_session"):
                    result = run_training_oof_tuning(
                        jobs, policy, config, specs, candidates, trainer, output_dir=output)
                return result, trainer

            original = root / "original"
            run(original, [candidate])
            audit_path = next((original / "candidate_audits").glob("*.json"))
            original_bytes = audit_path.read_bytes()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "context": tuning_resume_context(config, specs), "candidates": [{
                    "path": str(audit_path), "sha256": hashlib.sha256(original_bytes).hexdigest(),
                    "parameter_set": candidate,
                }],
            }), encoding="utf-8")
            config["models"]["tuning_resume"] = {"source_manifest": str(manifest)}
            renamed = {**candidate, "name": "adopted", "tag": "adopted"}
            recovered = root / "recovered"
            result, trainer = run(recovered, [renamed, second])
            self.assertEqual((result["reused_candidates"], result["trained_candidates"]), (1, 1))
            self.assertEqual(len(trainer.fits), 3)
            self.assertFalse(result["outer_test_used"])
            self.assertEqual(result["selected"]["randomforest"]["candidate_name"], "adopted")
            # A second interruption must also reuse candidates produced after
            # recovery. Unfinished atomic writes are never treated as results.
            (recovered / "candidate_audits" / "partial.json.tmp").write_text("{", encoding="utf-8")
            result, trainer = run(recovered, [renamed, second])
            self.assertEqual((result["reused_candidates"], result["trained_candidates"]), (2, 0))
            self.assertEqual(len(trainer.fits), 0)
            self.assertEqual(audit_path.read_bytes(), original_bytes)

            config["run"]["epochs"] = 2
            with self.assertRaisesRegex(ValueError, "resume context differs"):
                run(root / "changed-context", [renamed])
            config["run"]["epochs"] = 1
            audit_path.write_bytes(original_bytes + b" ")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                run(root / "changed-audit", [renamed])

            # Even a valid checksum must not allow a mismatched sample, fold,
            # or unfinished neural-model fit into the candidate ranking.
            saved = json.loads(original_bytes)
            samples = saved["internal_validation"]["samples"]
            metadata = [{k: sample[k] for k in ("experiment_name", "source_wav_id", "chunk_index")}
                        for sample in samples]
            targets = np.asarray([sample["heat_flux"] for sample in samples])
            selector = AcousticTrainingSelector({"enabled": False}, {"day-a": 36.0})
            options = (metadata, targets, selector, 3, 42, 1, "chunk_kfold")
            altered = deepcopy(saved)
            altered["internal_validation"]["samples"][0]["heat_flux"] += 1
            with self.assertRaisesRegex(ValueError, "identities or targets differ"):
                validate_resume_audit(altered, candidate, specs[:1], *options)
            altered = deepcopy(saved)
            altered["internal_validation"]["folds"][0]["fit_indices"] = []
            with self.assertRaisesRegex(ValueError, "partition differs"):
                validate_resume_audit(altered, candidate, specs[:1], *options)
            altered = deepcopy(saved)
            for fold in altered["internal_validation"]["folds"]:
                fold["model_training"] = {"randomforest": {
                    "requested_epochs": 1, "epochs_completed": 0, "requested_batch_size": 8,
                    "stopped_by_memory_error": True,
                }}
            with self.assertRaisesRegex(ValueError, "did not complete"):
                validate_resume_audit(altered, candidate, [{"kind": "keras", "batch_size": 8}], *options)


if __name__ == "__main__":
    unittest.main()
