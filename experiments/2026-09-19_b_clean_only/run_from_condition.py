"""Run the ONB entry point with an archived JSON override, leaving its source untouched."""

import argparse
import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
DEFAULT_CONDITION = ROOT / "configs/experiments/2026-09-19_b_clean_only_22khz.json"


def merge(target, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=Path, default=DEFAULT_CONDITION)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    condition = json.loads(args.condition.read_text(encoding="utf-8"))
    sys.path.insert(0, str(CODE))
    import utils.config.onb_defaults as defaults

    original = defaults.apply_onb_defaults

    def with_override(config, now=None):
        resolved = original(config, now)
        resolved = merge(resolved, condition["validation_override"])
        if args.seed is not None:
            allowed = condition.get("replicate_seeds")
            if allowed is None or args.seed not in allowed:
                raise ValueError(f"Seed {args.seed} is not listed in the condition file")
            resolved["run"]["random_seed"] = args.seed
            resolved["output"]["result_date_dir"] += f"_s{args.seed}"
        return resolved

    defaults.apply_onb_defaults = with_override
    try:
        runpy.run_path(str(CODE / "run_ensemble_regression_onb.py"), run_name="__main__")
    finally:
        defaults.apply_onb_defaults = original


if __name__ == "__main__":
    main()
