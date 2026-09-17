import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.ensemble.ensemble_runtime import (  # noqa: E402
    _group_disjoint_holdout_indices,
)


class EnsembleGroupHoldoutTest(unittest.TestCase):
    def test_inner_holdout_never_splits_one_wav_across_partitions(self):
        groups = np.repeat([f"wav-{index}" for index in range(10)], 4)
        targets = np.repeat(np.arange(10, dtype=float), 4)
        fit_index, holdout_index = _group_disjoint_holdout_indices(
            targets,
            groups,
            fraction=0.20,
            random_state=42,
        )
        fit_groups = set(groups[fit_index])
        holdout_groups = set(groups[holdout_index])
        self.assertFalse(fit_groups & holdout_groups)
        self.assertEqual(len(holdout_groups), 2)

if __name__ == "__main__":
    unittest.main()
