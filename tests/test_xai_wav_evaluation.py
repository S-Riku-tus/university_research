import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.explainability.training_integration import (  # noqa: E402
    GROUP_MASK_PERFORMANCE_HEADER,
    _aggregate_metric_inputs,
    _group_mask_performance,
)


class XaiWavEvaluationTest(unittest.TestCase):
    def test_mask_metrics_aggregate_each_source_wav_once(self):
        groups = np.repeat(["wav-a", "wav-b"], 3)
        targets = np.repeat([0.0, 10.0], 3)
        predictions = np.asarray([0.0, 1.0, 100.0, 9.0, 10.0, 11.0])

        grouped_y, grouped_prediction = _aggregate_metric_inputs(
            targets, predictions, groups, aggregation="median"
        )

        np.testing.assert_allclose(grouped_y, [0.0, 10.0])
        np.testing.assert_allclose(grouped_prediction, [1.0, 10.0])

    def test_one_wav_cannot_contain_multiple_targets(self):
        with self.assertRaisesRegex(ValueError, "multiple targets"):
            _aggregate_metric_inputs(
                [0.0, 1.0], [0.0, 1.0], ["wav-a", "wav-a"]
            )

    def test_group_mask_output_schema_records_wav_unit(self):
        predictions = np.asarray([0.0, 1.0, 100.0, 9.0, 10.0, 11.0])
        x_val = np.repeat(predictions[:, None, None], 2, axis=1)
        y_val = np.repeat([0.0, 10.0], 3)
        source_groups = np.repeat(["wav-a", "wav-b"], 3)
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
            config={
                "performance_evaluation_unit": "source_wav",
                "performance_wav_aggregation": "median",
                "baseline_value": 0.0,
                "onb_band_frac": 0.10,
            },
            source_wav_groups=source_groups,
        )
        self.assertEqual(len(rows[0]), len(GROUP_MASK_PERFORMANCE_HEADER))
        as_dict = dict(zip(GROUP_MASK_PERFORMANCE_HEADER, rows[0]))
        self.assertEqual(as_dict["evaluation_unit"], "source_wav")
        self.assertEqual(as_dict["aggregation"], "median")
        self.assertEqual(as_dict["n_chunks"], 6)
        self.assertEqual(as_dict["n_source_wavs"], 2)


if __name__ == "__main__":
    unittest.main()
