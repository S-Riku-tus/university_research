from datetime import datetime
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from utils.config.onb_defaults import apply_onb_defaults, onb_model_specs


class OnbDefaultsTest(unittest.TestCase):
    def test_fixed_sections_are_added_and_partial_overrides_are_merged(self):
        resolved = apply_onb_defaults(
            {"run": {"epochs": 3}, "explainability": {"enabled": False}},
            now=datetime(2026, 9, 16),
        )
        self.assertEqual(resolved["run"]["epochs"], 3)
        self.assertFalse(resolved["explainability"]["enabled"])
        self.assertEqual(resolved["explainability"]["ig_max_steps"], 4096)
        self.assertEqual(resolved["evaluation"]["primary_wav_aggregation"], "median")
        self.assertEqual(resolved["output"]["save_date"], "20260916")
        self.assertEqual(
            resolved["output"]["result_date_dir"],
            "20260916/onb",
        )
        self.assertFalse(resolved["acoustic_selection"]["enabled"])

    def test_only_acoustic_threshold_is_needed_in_run_config(self):
        resolved = apply_onb_defaults({
            "acoustic_selection": {"peak_height_threshold": 1.0e-9},
        })
        selection = resolved["acoustic_selection"]
        self.assertTrue(selection["enabled"])
        self.assertEqual(selection["mode"], "peak_height")
        self.assertEqual(selection["feature"], "peak_2100_2500_psd")
        self.assertEqual(selection["peak_height_threshold"], 1.0e-9)

        disabled = apply_onb_defaults({
            "acoustic_selection": {"peak_height_threshold": None},
        })
        self.assertFalse(disabled["acoustic_selection"]["enabled"])

    def test_model_registry_is_fresh_and_builders_remain_callable(self):
        first = onb_model_specs()
        second = onb_model_specs()
        first[0]["label"] = "changed"
        self.assertEqual(second[0]["label"], "RandomForest")
        self.assertEqual([item["key"] for item in second], [
            "rf", "cnntf_v2_gap", "alexnet"
        ])
        self.assertTrue(all(callable(item["builder"]) for item in second))


if __name__ == "__main__":
    unittest.main()
