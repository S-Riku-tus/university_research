"""Compare legacy green-line and current q100 definitions on saved 2025 data.

Both use identical source-day ONB thresholds here to isolate definitions.
No fits/inference; historical pooled thresholds and fold averages are not
recreated as though they were identical evaluation conditions.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from utils.experiment.run_helpers import open_text, windows_long_path
OUT = Path(__file__).resolve().parent
RUN = ROOT / ("Pool_boiling/Subcooling_20_degrees/0.3/"
    "2025.06.11_0.3_2_6.18_0.3_3/regression_result/npy/ensemble/202610/06/"
    "onb_wc-t0611-v0611_ic3-nc_c1s_s0_e200_193305/maxfreq=3kHz")
THRESHOLDS = {"20250611": 221505.1102, "20250618": 271677.6816}


def q100_legacy(y, p, threshold, inclusive=False):
    blocks = [(q, p[y == q]) for q in sorted(set(y))]
    flags = [bool(np.all(values >= threshold if inclusive else values > threshold))
        for _, values in blocks]
    for i, (q, _) in enumerate(blocks):
        if all(flags[i:]):
            return float(q)
    return None


def q100_current(y, p, threshold):
    reached = [float(q) for q in sorted(set(y)) if np.all(p[y >= q] >= threshold)]
    return reached[0] if reached else None


def main():
    rows, hashes = [], []
    for noise in ["clean", "0", "-4", "-8", "-12", "-16", "-20"]:
        tag = "no_noise" if noise == "clean" else noise
        folder = RUN / ("heatflux_no_noise" if noise == "clean" else "heatflux_reference_SNR=" + noise)
        path = folder / "fold_pred" / ("pred_f1_" + tag + ".csv")
        with open_text(path, "r", encoding="utf-8-sig", newline="") as stream:
            frame = pd.read_csv(stream)
        methods = [c for c in frame if c in ["randomforest", "conformer", "alexnet", "hgb", "extra_trees"]
            or c.startswith("ensemble__")]
        assert len(frame) == 540 and len(methods) == 11
        hashes.append({"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(Path(windows_long_path(path)).read_bytes()).hexdigest()})
        for day, threshold in THRESHOLDS.items():
            subset = frame[frame.source_wav_id.str.startswith(day + "::")]
            y = subset.y_true.to_numpy()
            assert len(subset) == 270
            for key in methods:
                prediction = subset[key].to_numpy()
                old = q100_legacy(y, prediction, threshold)
                normalized = q100_legacy(y, prediction, threshold, inclusive=True)
                current = q100_current(y, prediction, threshold)
                assert normalized == current
                rows.append({"noise": noise, "source_day": day, "model_key": key,
                    "threshold_W_m2": threshold, "legacy_strict_q100_W_m2": old,
                    "legacy_with_inclusive_boundary_W_m2": normalized,
                    "current_q100_W_m2": current,
                    "equal_predictions_at_threshold": int(np.sum(prediction == threshold)),
                    "strict_equals_current": old == current})
    with (OUT / "q100_definition_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "passed", "comparison_cells": len(rows),
        "legacy_green_line_with_inclusive_boundary_matches_current_all_cells": True,
        "legacy_strict_boundary_matches_current_cells": sum(row["strict_equals_current"] for row in rows),
        "scope": "Same saved June 2025 predictions and same source-day ONB threshold; no claim of identical historical fit, pooled threshold, or averaging.",
        "source_hashes": hashes,
        "code_locations": {"legacy_green_line": "archive/code/3.run_ensemble_100percent_classification.py:350",
            "legacy_complete_separation_search": "archive/code/3.run_ensemble_100percent_classification.py:251",
            "current_q100": "code/utils/calculation/source_day_metrics.py:66"},
        "correction": "Legacy plotted 100% classification threshold and current q100 have the same positive-tail detection meaning. Neither green-line calculation checks false positives below ONB. The old separate TPR=1/FPR=0 threshold search changes both truth and prediction thresholds and is a different calculation.",
        "toy_examples": []}
    for name, y, prediction in [
        ("all_positive_false_alarm", np.array([50., 100., 150.]), np.array([150., 160., 170.])),
        ("no_false_alarm_but_boundary_miss", np.array([50., 100., 150.]), np.array([60., 90., 160.]))]:
        actual, positive = y >= 100., prediction >= 100.
        result["toy_examples"].append({"name": name, "observed_onb": 100,
            "q_true": y.tolist(), "q_pred": prediction.tolist(),
            "legacy_green_line_q100": q100_legacy(y, prediction, 100.),
            "current_q100": q100_current(y, prediction, 100.),
            "fp": int(np.sum(~actual & positive)), "fn": int(np.sum(actual & ~positive)),
            "full_classification_perfect": bool(np.all(actual == positive))})
    (OUT / "q100_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["source_hashes", "code_locations"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
