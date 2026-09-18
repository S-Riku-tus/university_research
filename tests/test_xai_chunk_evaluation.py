import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.explainability.training_integration import (  # noqa: E402
    GROUP_MASK_PERFORMANCE_HEADER,
    _group_mask_performance,
    aggregate_group_mask_comparison,
)
from utils.explainability.spectrogram_explainers import make_frequency_groups  # noqa: E402


class XaiChunkEvaluationTest(unittest.TestCase):
    def test_group_mask_metrics_use_every_chunk(self):
        predictions = np.asarray([0.0, 1.0, 100.0, 9.0, 10.0, 11.0])
        x_val = np.repeat(predictions[:, None, None, None], 2, axis=1)
        x_val = np.repeat(x_val, 2, axis=2)
        y_val = np.repeat([0.0, 10.0], 3)
        mask_groups = [{
            "group": "freq_0_500Hz",
            "axis": "frequency",
            "low": 0.0,
            "high": 500.0,
            "unit": "Hz",
            "low_index": 0,
            "high_index": 1,
            "mask": np.asarray([[True, False], [True, False]]),
        }]
        rows = _group_mask_performance(
            lambda values: values[:, 0, 0, 0],
            x_val,
            y_val,
            predictions,
            mask_groups,
            threshold=5.0,
            config={"baseline_value": 0.0, "onb_band_frac": 0.10},
        )
        self.assertEqual(len(rows[0]), len(GROUP_MASK_PERFORMANCE_HEADER))
        as_dict = dict(zip(GROUP_MASK_PERFORMANCE_HEADER, rows[0]))
        self.assertEqual(as_dict["n_samples"], 6)
        self.assertNotIn("evaluation_unit", as_dict)
        self.assertNotIn("aggregation", as_dict)

    def test_frequency_groups_mask_every_time_frame(self):
        groups = make_frequency_groups(
            height=4, width=8, max_freq_hz=4000,
            frequency_bands_hz=[(0, 1000), (1000, 2000)],
        )
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(group["axis"] == "frequency" for group in groups))
        self.assertTrue(np.all(groups[0]["mask"][:, :2]))
        self.assertFalse(np.any(groups[0]["mask"][:, 2:]))

    def test_aggregate_ignores_legacy_time_mask_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "explainability" / "fold1" / "randomforest"
            source.mkdir(parents=True)
            rows = [
                {"group": "freq_0_1000Hz", "axis": "frequency", "low": 0,
                 "high": 1000, "unit": "Hz", "r2_drop": 0.1,
                 "rmse_onb_increase": 2, "recall_drop": 0.01},
                {"group": "time_1", "axis": "time", "low": 0,
                 "high": 0.25, "unit": "s", "r2_drop": 0.9,
                 "rmse_onb_increase": 20, "recall_drop": 0.1},
            ]
            with (source / "group_mask_performance.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=GROUP_MASK_PERFORMANCE_HEADER)
                writer.writeheader()
                writer.writerows(rows)

            self.assertTrue(aggregate_group_mask_comparison(
                temp_dir, {"enabled": True}, ["randomforest"], 1,
            ))
            root = Path(temp_dir) / "explainability"
            with (root / "model_group_mask_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                aggregated = list(csv.DictReader(stream))
            self.assertEqual(len(aggregated), 1)
            self.assertEqual(aggregated[0]["axis"], "frequency")
            self.assertFalse(list(root.glob("group_mask_comparison_time_*.png")))


if __name__ == "__main__":
    unittest.main()
