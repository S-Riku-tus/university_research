"""Audit peak-height thresholds on the fixed outer-training partition.

This script reads metadata and the existing clean-source feature table only.
It never loads the outer-test arrays and does not train or score a model.
"""

import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import run_ensemble_regression_onb as run  # noqa: E402
from utils.calculation.prediction_records import (  # noqa: E402
    load_sample_metadata_without_arrays,
)
from utils.experiment.learning_policy import checked_metadata, outer_splits  # noqa: E402


def load_outer_training_records():
    jobs = run.build_dataset_jobs()
    job = next(item for item in jobs if item["noise_dir_name"] == "heatflux_no_noise")
    metadata = checked_metadata(
        load_sample_metadata_without_arrays(job["data_path"]),
        job["experiment_name"],
    )
    fit, _ = outer_splits(
        metadata,
        metadata,
        1,
        policy=run.LEARNING_POLICY,
        threshold=job["threshold"],
    )[0]
    feature_path = Path(run.VALIDATION_CONFIG["acoustic_selection"]["features_csv"])
    if not feature_path.is_absolute():
        feature_path = ROOT / feature_path
    with feature_path.open(encoding="utf-8-sig", newline="") as source:
        lookup = {
            (row["experiment_name"], row["source_wav_id"], int(row["chunk_index"])):
            float(row["peak_2100_2500_psd"])
            for row in csv.DictReader(source)
        }
    records = []
    for index in fit:
        row = metadata[int(index)]
        day = str(row.get("source_experiment_name") or row["experiment_name"])
        wav = str(row.get("original_source_wav_id") or row["source_wav_id"])
        records.append((
            day,
            float(row["sample_filename"].split("_")[0]),
            lookup[(day, wav, int(row["chunk_index"]))],
        ))
    return records


def main():
    records = load_outer_training_records()
    day = np.asarray([item[0] for item in records])
    heat_flux = np.asarray([item[1] for item in records], dtype=float)
    peak = np.asarray([item[2] for item in records], dtype=float)
    days = sorted(set(day))
    exact_masks = {
        name: ((day == name) & np.isclose(
            heat_flux, run.THRESHOLD_BY_EXPERIMENT[name], rtol=0, atol=1e-4))
        for name in days
    }
    above_masks = {
        name: ((day == name) &
               (heat_flux > run.THRESHOLD_BY_EXPERIMENT[name] + 1e-4))
        for name in days
    }
    eligible_masks = {
        name: ((day == name) &
               (heat_flux >= run.THRESHOLD_BY_EXPERIMENT[name] - 1e-4))
        for name in days
    }
    pre_masks = {
        name: ((day == name) &
               (heat_flux < run.THRESHOLD_BY_EXPERIMENT[name] - 1e-4))
        for name in days
    }

    print("PSD quantiles, expressed as multiples of 1e-9")
    quantiles = [0, 0.1, 0.2, 0.25, 0.5, 0.75, 1]
    for name in days:
        print(name)
        for region, mask in (
            ("exact_onb", exact_masks[name]),
            ("above_onb", above_masks[name]),
            ("pre_onb", pre_masks[name]),
        ):
            values = np.quantile(peak[mask], quantiles) / 1e-9
            print(region, int(mask.sum()), [round(float(value), 8) for value in values])

    def summarize(threshold):
        summary = {}
        for name in days:
            exact = exact_masks[name]
            above = above_masks[name]
            eligible = eligible_masks[name]
            summary[name] = {
                "exact_kept": int(np.sum(exact & (peak >= threshold))),
                "exact_total": int(exact.sum()),
                "above_removed": int(np.sum(above & (peak < threshold))),
                "above_total": int(above.sum()),
                "eligible_removed": int(np.sum(eligible & (peak < threshold))),
                "eligible_total": int(eligible.sum()),
            }
        return summary

    print("Candidate thresholds")
    for threshold in (1e-13, 3e-13, 1e-12, 3e-12, 6e-12, 6.5e-12,
                      6.8e-12, 7e-12, 7.4e-12, 7.8e-12, 7.9e-12,
                      8e-12, 9e-12, 9.5e-12, 1e-11, 3e-11,
                      1e-10, 2e-10, 3e-10, 1e-9):
        print(f"{threshold:.1e}", summarize(threshold))

    print("Largest common observed threshold satisfying exact-ONB retention in every day")
    for fraction in (0.95, 0.9, 0.8, 0.75, 0.67, 0.5):
        feasible = []
        for threshold in np.unique(peak):
            summary = summarize(float(threshold))
            if all(
                values["exact_kept"] / values["exact_total"] >= fraction
                for values in summary.values()
            ):
                feasible.append((float(threshold), summary))
        threshold, summary = feasible[-1]
        print(f"{fraction:.2f} {threshold:.10e} {summary}")

    print("Largest observed threshold per day under exact-ONB retention constraint")
    for name in days:
        for fraction in (0.95, 0.9, 0.8, 0.75):
            feasible = []
            for threshold in np.unique(peak[day == name]):
                summary = summarize(float(threshold))[name]
                if summary["exact_kept"] / summary["exact_total"] >= fraction:
                    feasible.append((float(threshold), summary))
            threshold, summary = feasible[-1]
            print(name, f"{fraction:.2f}", f"{threshold:.10e}", summary)

    print("Protect exact ONB and apply 1e-9 only above it")
    for name in days:
        exact = exact_masks[name]
        above = above_masks[name]
        print(name, {
            "exact_kept": int(exact.sum()),
            "exact_total": int(exact.sum()),
            "above_removed": int(np.sum(above & (peak < 1e-9))),
            "above_total": int(above.sum()),
        })
        for value in sorted(set(heat_flux[(day == name) & (heat_flux >= run.THRESHOLD_BY_EXPERIMENT[name] - 1e-4)])):
            level = (day == name) & np.isclose(heat_flux, value, rtol=0, atol=1e-4)
            print("  heat_flux", value,
                  "removed_below_1e-9", int(np.sum(level & (peak < 1e-9))),
                  "total", int(level.sum()))


if __name__ == "__main__":
    main()
