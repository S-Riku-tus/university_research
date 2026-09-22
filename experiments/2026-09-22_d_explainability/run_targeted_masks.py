"""Reproduce one run-151715 condition with A-selected mask samples."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
CONDITION = ROOT / "configs/experiments/2026-09-22_d_targeted_masks.json"


def merge(target, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("no_noise", "-8"), required=True)
    args = parser.parse_args()
    spec = json.loads(CONDITION.read_text(encoding="utf-8"))
    arm = spec["conditions"][args.condition]

    sys.path.insert(0, str(CODE))
    import utils.config.onb_defaults as defaults
    import utils.explainability.training_integration as integration

    original_defaults = defaults.apply_onb_defaults
    original_selector = integration.selected_sample_indices

    def with_override(config, now=None):
        resolved = merge(original_defaults(config, now), spec["validation_override"])
        resolved["data"]["noise_dir_names"] = [arm["noise_dir"]]
        resolved["output"]["result_date_dir"] += f"_{args.condition.replace('-', 'minus')}"
        return resolved

    def selected_indices(y_val, pred, threshold, max_samples=5):
        if len(y_val) != 1080:
            raise ValueError(f"Expected the full 1080-sample test set, got {len(y_val)}")
        return list(zip(arm["sample_labels"], arm["sample_indices"]))

    defaults.apply_onb_defaults = with_override
    integration.selected_sample_indices = selected_indices
    try:
        runpy.run_path(str(CODE / "run_ensemble_regression_onb.py"), run_name="__main__")
    finally:
        defaults.apply_onb_defaults = original_defaults
        integration.selected_sample_indices = original_selector


if __name__ == "__main__":
    main()
