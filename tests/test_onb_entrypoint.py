"""Check that the ONB entry point forwards the selected internal split."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import run_ensemble_regression_onb as onb  # noqa: E402
from utils.experiment.run_helpers import run_config_digest  # noqa: E402


class OnbEntrypointTest(unittest.TestCase):
    def test_selected_internal_split_reaches_execution_and_changes_run_identity(self):
        frequency = onb.MAX_FREQ_HZ_LIST[0]
        parameter_set = onb.parameter_plan_for_max_freq(frequency)["parameter_sets"][0]
        specs = onb.resolve_parameter_set(onb.MODEL_SPECS, parameter_set)
        hashes = []
        for method in ("chunk_kfold", "wav_kfold"):
            with self.subTest(method=method), patch.dict(
                onb.VALIDATION_CONFIG["run"], {"internal_validation_split": method}
            ), patch.dict(onb.THRESHOLD_BY_EXPERIMENT, {day: 200000.0 for day in onb.EXPERIMENT_DIR_NAMES}):
                onb.validate_validation_config(onb.MODEL_SPECS)
                execution = onb.validation_config_snapshot(frequency)
                self.assertEqual(execution["run"]["internal_validation_split"], method)
                hashes.append(run_config_digest(
                    execution, parameter_set, specs,
                    "-".join(onb.MODEL_KEYS), execution["output"]["save_fold_predictions"],
                ))
        self.assertNotEqual(hashes[0], hashes[1])

    def test_current_day_and_cross_day_options_resolve_without_duplicate_day_lists(self):
        from utils.experiment.learning_policy import normalize_learning_policy, resolve_experiment_names
        self.assertEqual(onb.EXPERIMENT_DIR_NAMES, ["2026.10.07_0.3_1"])
        self.assertEqual(onb.LEARNING_POLICY["test_fraction"], .25)
        self.assertEqual(onb.NOISE_DIR_NAMES, onb.VALIDATION_CONFIG['data']['noise_dir_names'])
        self.assertIn('heatflux_no_noise', onb.NOISE_DIR_NAMES)
        self.assertFalse(onb.VALIDATION_CONFIG["output"]["main_comparison_report"])
        policy = {**onb.VALIDATION_CONFIG["learning_policy"], "evaluation_mode": "cross_day"}
        names = resolve_experiment_names(onb.VALIDATION_CONFIG["data"], policy)
        resolved = normalize_learning_policy(policy, names)
        self.assertEqual(resolved["train_experiments"], ["2025.06.11_0.3_2", "2025.06.18_0.3_3"])
        self.assertEqual(resolved["test_experiments"], ["2026.10.07_0.3_1"])

    def test_unready_onb_label_is_rejected_before_training(self):
        with patch.dict(onb.THRESHOLD_BY_EXPERIMENT, {"2026.10.07_0.3_1": None}):
            with self.assertRaisesRegex(ValueError, "Pending CSV sources"):
                onb.validate_validation_config(onb.MODEL_SPECS)

    def test_generated_labels_and_explicit_full_data_form_a_ready_learning_family(self):
        from utils.experiment.learning_policy import build_learning_families
        from utils.experiment.onb_thresholds import CSV_ONB_THRESHOLD_RECORDS, onb_threshold_by_experiment
        day = "2026.10.07_0.3_1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / CSV_ONB_THRESHOLD_RECORDS[day]["heat_flux_csv"]
            labels.parent.mkdir(parents=True)
            labels.write_text("volt,q\n1.1V,100000.0\n1.2V,200987.654321\n", encoding="utf-8")
            experiment = root.joinpath(*onb.VALIDATION_CONFIG["data"]["experiment_root_parts"]) / day
            expected = experiment / "data/npy/waterflow_20261007_1s/maxfreq=3kHz/heatflux_no_noise"
            expected.mkdir(parents=True)
            (expected / "200987.654321_src-index=2.200987_chunk-0000.npy").touch()
            for noise in onb.NOISE_DIR_NAMES:
                condition = expected.parent / noise
                condition.mkdir(exist_ok=True)
                (condition / '200987.654321_src-index=2.200987_chunk-0000.npy').touch()
            with patch("utils.experiment.onb_thresholds.REPO_ROOT", root):
                thresholds = onb_threshold_by_experiment(csv_experiments=[day])
            with patch.object(onb, "EXPERIMENT_ROOT", experiment.parent), patch.dict(onb.THRESHOLD_BY_EXPERIMENT, thresholds):
                onb.validate_validation_config(onb.MODEL_SPECS)
                jobs = onb.build_dataset_jobs()
            self.assertEqual(len(jobs), len(onb.NOISE_DIR_NAMES))
            self.assertEqual(jobs[0]["data_path"], expected)
            self.assertEqual(jobs[0]["threshold"], 200987.654321)
            families = build_learning_families(jobs, onb.LEARNING_POLICY, onb.EXPERIMENT_DIR_NAMES)
            self.assertEqual(len(families), 1)
            self.assertEqual(len(families[0]["training_jobs"]), 1)
            self.assertEqual(len(families[0]["evaluation_jobs"]), len(onb.NOISE_DIR_NAMES))

    def test_unknown_internal_split_is_rejected_before_training(self):
        with patch.dict(onb.VALIDATION_CONFIG["run"], {"internal_validation_split": "typo"}):
            with self.assertRaisesRegex(ValueError, "internal_validation_split"):
                onb.validate_validation_config(onb.MODEL_SPECS)


if __name__ == "__main__":
    unittest.main()
