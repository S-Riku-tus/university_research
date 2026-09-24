import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.experiment.onb_thresholds import (  # noqa: E402
    ONB_THRESHOLD_RECORDS,
    onb_threshold_by_experiment,
)


class OnbThresholdRegistryTest(unittest.TestCase):
    def test_registry_uses_confirmed_current_run_values(self):
        self.assertEqual(
            onb_threshold_by_experiment(),
            {
                "2025.06.11_0.3_2": 221505.1102,
                "2025.06.18_0.3_3": 271677.6816,
                "2025.07.09_0.3_1": 571694.252491167,
            },
        )

    def test_every_threshold_has_decision_record_provenance(self):
        for experiment_name, record in ONB_THRESHOLD_RECORDS.items():
            with self.subTest(experiment_name=experiment_name):
                self.assertEqual(
                    record["definition"],
                    "experiment-specific ONB confirmed for the current run",
                )
                self.assertIn("2026-09-24_selection_onb_ig_review", record["source"])
                self.assertGreater(record["threshold"], 0)


if __name__ == "__main__":
    unittest.main()
