"""Run one predeclared C-selection arm and seed without editing the main entry point."""

import argparse
import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
CONDITION = ROOT / "configs/experiments/2026-09-20_c_selection_pair.json"


def merge(target, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("off", "on"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    condition = json.loads(CONDITION.read_text(encoding="utf-8"))
    if args.seed not in condition["replicate_seeds"]:
        raise ValueError("Seed is absent from the predeclared condition")
    sys.path.insert(0, str(CODE))
    import utils.config.onb_defaults as defaults

    original = defaults.apply_onb_defaults

    def with_override(config, now=None):
        resolved = merge(original(config, now), condition["validation_override"])
        resolved["acoustic_selection"] = merge(
            resolved["acoustic_selection"], condition["arms"][args.arm]
        )
        resolved["run"]["random_seed"] = args.seed
        resolved["output"]["result_date_dir"] += f"_{args.arm}_s{args.seed}"
        return resolved

    defaults.apply_onb_defaults = with_override
    try:
        runpy.run_path(str(CODE / "run_ensemble_regression_onb.py"), run_name="__main__")
    finally:
        defaults.apply_onb_defaults = original


if __name__ == "__main__":
    main()
