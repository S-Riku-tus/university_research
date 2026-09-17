import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.explainability.training_integration import (  # noqa: E402
    GROUP_MASK_PERFORMANCE_HEADER,
    _group_mask_performance,
)


class XaiChunkEvaluationTest(unittest.TestCase):
    def test_group_mask_metrics_use_every_chunk(self):
        predictions = np.asarray([0.0, 1.0, 100.0, 9.0, 10.0, 11.0])
        x_val = np.repeat(predictions[:, None, None], 2, axis=1)
        y_val = np.repeat([0.0, 10.0], 3)
        mask_groups = [{
            "group": "t0",
            "axis": "time",
            "low": 0.0,
            "high": 0.5,
            "unit": "s",
            "low_index": 0,
            "high_index": 1,
            "mask": np.asarray([True, False]),
        }]
        rows = _group_mask_performance(
            lambda values: values[:, 0, 0],
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


if __name__ == "__main__":
    unittest.main()
