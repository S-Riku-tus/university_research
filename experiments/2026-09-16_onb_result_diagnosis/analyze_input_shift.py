"""Summarize input-domain differences behind the 2026-09-16 ONB run.

This reads only clean 1-second NPY inputs.  It does not retrain a model or use
the held-out day to select a threshold/model.  Statistics use the exact
log1p(power / 1e-12) transform used by the two neural networks.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from utils.experiment.onb_thresholds import onb_threshold_by_experiment
from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.experiment.learning_policy import checked_metadata


DAYS = [
    "2025.06.11_0.3_2",
    "2025.07.09_0.3_1",
    "2025.06.18_0.3_3",
]
FREQUENCIES = ["maxfreq=3kHz", "maxfreq=22kHz"]
DATA_VERSION = "waterflow_20260817_1s"
OUTPUT_DIR = Path(__file__).resolve().parent
SELECTION_CONFIG = {
    "enabled": True,
    "mode": "peak_height",
    "features_csv": (
        "experiments/2026-09-16_peak_height_selection/peak_features.csv"
    ),
    "feature": "peak_2100_2500_psd",
    "peak_height_threshold": 1e-9,
    "peak_height_threshold_by_experiment": {},
    "apply_max_heat_flux_by_experiment": {},
}


def _source_rows(day, frequency):
    directory = (
        ROOT / "Pool_boiling" / "Subcooling_20_degrees" / "0.3" / day
        / "data" / "npy" / DATA_VERSION / frequency / "heatflux_no_noise"
    )
    with (directory / "chunk_manifest.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source:
        manifest = list(csv.DictReader(source))
    return directory, manifest


def _pearson(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def build_profiles():
    profiles = []
    thresholds = onb_threshold_by_experiment()
    selector = AcousticTrainingSelector(SELECTION_CONFIG, thresholds)
    for frequency in FREQUENCIES:
        for day in DAYS:
            directory, manifest = _source_rows(day, frequency)
            metadata = checked_metadata(manifest, day)
            if day == "2025.06.18_0.3_3":
                retained = set(range(len(metadata)))
            else:
                retained = set(selector.select(metadata)[0].tolist())
            chunks = defaultdict(list)
            for index, row in enumerate(manifest):
                array = np.load(directory / row["sample_filename"])
                if array.shape != (224, 224):
                    raise ValueError(
                        f"Unexpected input shape {array.shape}: {row['sample_filename']}"
                    )
                log_power = np.log1p(np.maximum(array, 0.0) / 1e-12)
                chunks[row["source_wav_id"]].append({
                    "heat_flux": float(row["heat_flux"]),
                    "log_mean": float(np.mean(log_power, dtype=np.float64)),
                    "log_std": float(np.std(log_power, dtype=np.float64)),
                    "log_p95": float(np.percentile(log_power, 95)),
                    "raw_mean": float(np.mean(array, dtype=np.float64)),
                    "selected": index in retained,
                })
            for source_wav_id, rows in chunks.items():
                targets = np.asarray([row["heat_flux"] for row in rows])
                if not np.allclose(targets, targets[0], rtol=0, atol=1e-5):
                    raise ValueError(f"Multiple targets in {day}/{source_wav_id}")
                selected_rows = [row for row in rows if row["selected"]]
                profile = {
                    "frequency": frequency,
                    "experiment_name": day,
                    "role": "test" if day == "2025.06.18_0.3_3" else "train",
                    "source_wav_id": source_wav_id,
                    "heat_flux": float(targets[0]),
                    "chunks": len(rows),
                    "selected_chunks": len(selected_rows),
                    **{
                        key: float(np.median([row[key] for row in rows]))
                        for key in ("log_mean", "log_std", "log_p95", "raw_mean")
                    },
                }
                for key in ("log_mean", "log_std", "log_p95", "raw_mean"):
                    profile["selected_" + key] = (
                        float(np.median([row[key] for row in selected_rows]))
                        if selected_rows else ""
                    )
                profiles.append(profile)
    return profiles


def summarize(profiles):
    thresholds = onb_threshold_by_experiment()
    grouped = defaultdict(list)
    for row in profiles:
        grouped[(row["frequency"], row["experiment_name"])].append(row)
    summaries = []
    for (frequency, day), rows in sorted(grouped.items()):
        threshold = thresholds[day]
        negative = [row for row in rows if row["heat_flux"] < threshold]
        positive = [row for row in rows if row["heat_flux"] >= threshold]
        below_200k = [row for row in rows if row["heat_flux"] < 200_000]
        summaries.append({
            "frequency": frequency,
            "experiment_name": day,
            "role": rows[0]["role"],
            "n_wavs": len(rows),
            "heatflux_log_mean_pearson": _pearson(
                [row["heat_flux"] for row in rows],
                [row["log_mean"] for row in rows],
            ),
            "pre_onb_log_mean_median": float(np.median(
                [row["log_mean"] for row in negative]
            )),
            "onb_positive_log_mean_median": float(np.median(
                [row["log_mean"] for row in positive]
            )),
            "below_200kw_log_mean_median": float(np.median(
                [row["log_mean"] for row in below_200k]
            )),
            "selected_wavs": int(sum(row["selected_chunks"] > 0 for row in rows)),
            "selected_heatflux_log_mean_pearson": _pearson(
                [row["heat_flux"] for row in rows if row["selected_chunks"] > 0],
                [row["selected_log_mean"] for row in rows if row["selected_chunks"] > 0],
            ),
        })
    return summaries


def main():
    profiles = build_profiles()
    profile_path = OUTPUT_DIR / "input_wav_profiles.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(profiles[0]))
        writer.writeheader()
        writer.writerows(profiles)
    summaries = summarize(profiles)
    summary_path = OUTPUT_DIR / "input_shift_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
