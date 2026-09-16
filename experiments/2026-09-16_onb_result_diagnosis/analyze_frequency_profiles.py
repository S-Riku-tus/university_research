"""Compare held-out spectral profiles with training-day profiles."""

import csv
import json
from collections import defaultdict

import numpy as np

from analyze_input_shift import (
    DAYS,
    FREQUENCIES,
    OUTPUT_DIR,
    SELECTION_CONFIG,
    _source_rows,
)
from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.experiment.learning_policy import checked_metadata
from utils.experiment.onb_thresholds import onb_threshold_by_experiment


TEST_DAY = "2025.06.18_0.3_3"


def _profile_groups():
    thresholds = onb_threshold_by_experiment()
    selector = AcousticTrainingSelector(SELECTION_CONFIG, thresholds)
    grouped = defaultdict(list)
    for frequency in FREQUENCIES:
        for day in DAYS:
            directory, manifest = _source_rows(day, frequency)
            metadata = checked_metadata(manifest, day)
            retained = (
                set(range(len(metadata)))
                if day == TEST_DAY
                else set(selector.select(metadata)[0].tolist())
            )
            for index, row in enumerate(manifest):
                if index not in retained:
                    continue
                array = np.load(directory / row["sample_filename"])
                log_power = np.log1p(np.maximum(array, 0.0) / 1e-12)
                regime = (
                    "pre_onb"
                    if float(row["heat_flux"]) < thresholds[day]
                    else "onb_positive"
                )
                grouped[(frequency, day, regime)].append(
                    np.mean(log_power, axis=0, dtype=np.float64)
                )
    return {
        key: np.median(np.asarray(values), axis=0)
        for key, values in grouped.items()
    }, {key: len(values) for key, values in grouped.items()}


def _distance(reference, candidate):
    centered_reference = reference - np.mean(reference)
    centered_candidate = candidate - np.mean(candidate)
    return {
        "absolute_log_rmse": float(np.sqrt(np.mean((reference - candidate) ** 2))),
        "centered_shape_rmse": float(np.sqrt(np.mean(
            (centered_reference - centered_candidate) ** 2
        ))),
        "profile_pearson": float(np.corrcoef(reference, candidate)[0, 1]),
    }


def main():
    profiles, counts = _profile_groups()
    rows = []
    for frequency in FREQUENCIES:
        for test_regime in ("pre_onb", "onb_positive"):
            reference = profiles[(frequency, TEST_DAY, test_regime)]
            for train_day in DAYS:
                if train_day == TEST_DAY:
                    continue
                for train_regime in ("pre_onb", "onb_positive"):
                    key = (frequency, train_day, train_regime)
                    rows.append({
                        "frequency": frequency,
                        "test_day": TEST_DAY,
                        "test_regime": test_regime,
                        "train_day": train_day,
                        "train_regime": train_regime,
                        "train_chunks": counts[key],
                        **_distance(reference, profiles[key]),
                    })
    path = OUTPUT_DIR / "frequency_profile_distances.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
