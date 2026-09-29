import contextlib
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
    build_parameter_execution_plan,
    expand_parameter_sets,
    resolve_parameter_set,
)
from utils.experiment.learning_policy import normalize_learning_policy
from utils.experiment.training_oof_tuning import (
    oof_regression_metrics,
    run_training_oof_tuning,
)


class TrainingOofTuningTest(unittest.TestCase):
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
                "within_day_experiment": "day-a",
                "training_noise": "clean_only",
                "test_fraction": 0.5,
                "test_split_seed": 42,
                "test_stratify": "none",
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
            self.assertEqual(result["selected"]["randomforest"]["candidate_name"],
                             "rf_depth3")
            saved = json.loads((output_dir / "selected_candidates.json").read_text(
                encoding="utf-8"))
            self.assertFalse(saved["outer_test_used"])
            self.assertTrue((output_dir / "candidate_metrics.csv").is_file())
            self.assertTrue((output_dir / "parameter_search_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
