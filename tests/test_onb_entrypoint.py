"""Check that the ONB entry point forwards the selected internal split."""

from pathlib import Path
import sys
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
            ):
                onb.validate_validation_config(onb.MODEL_SPECS)
                execution = onb.validation_config_snapshot(frequency)
                self.assertEqual(execution["run"]["internal_validation_split"], method)
                hashes.append(run_config_digest(
                    execution, parameter_set, specs,
                    "-".join(onb.MODEL_KEYS), execution["output"]["save_fold_predictions"],
                ))
        self.assertNotEqual(hashes[0], hashes[1])

    def test_unknown_internal_split_is_rejected_before_training(self):
        with patch.dict(onb.VALIDATION_CONFIG["run"], {"internal_validation_split": "typo"}):
            with self.assertRaisesRegex(ValueError, "internal_validation_split"):
                onb.validate_validation_config(onb.MODEL_SPECS)


if __name__ == "__main__":
    unittest.main()
