"""Read-only diagnostics for steps 5 and 7 using already saved predictions.

This script does not import or execute the ONB training pipeline.  It only reads
the completed B/C CSV and JSON artifacts and writes compact audit tables beside
this file.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
B_RUNS = ROOT / "experiments/2026-09-19_b_clean_only/matched_seed_run_locations.csv"
C_RUNS = ROOT / "experiments/2026-09-20_c_selection/c_run_locations.csv"
SINGLES = ("randomforest", "conformer", "alexnet")
SNRS = ("-8", "-16")
EVAL_ONB = 271677.6816
TRAIN_RUN_ONB = 221505.1102
TRAIN_SOURCE_ONB = 369000.0


def long_path(path: Path) -> Path:
    absolute = path.resolve()
    return Path("\\\\?\\" + str(absolute)) if os.name == "nt" else absolute


def read_csv(path: Path) -> list[dict[str, str]]:
    with long_path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_matched_runs():
    runs = {}
    for record in read_csv(B_RUNS):
        seed = int(record["seed"])
        root = ROOT / record["run"]
        runs[seed] = {"root": root, "predictions": {}, "weights": {}}
        for snr in SNRS:
            directory = root / "maxfreq=22kHz" / f"heatflux_reference_SNR={snr}"
            prediction_rows = read_csv(directory / "fold_pred" / f"pred_f1_{snr}.csv")
            keyed = {
                (row["experiment_name"], row["source_wav_id"], int(row["chunk_index"])): row
                for row in prediction_rows
            }
            if len(prediction_rows) != len(keyed) or len(keyed) != 1080:
                raise ValueError(f"Unexpected prediction keys: seed={seed}, snr={snr}")
            weight_row, = read_csv(directory / f"ensemble_weights_{snr}.csv")
            weights = {key: float(weight_row[key]) for key in SINGLES}
            if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
                raise ValueError(f"Weights do not sum to one: seed={seed}, snr={snr}")
            runs[seed]["predictions"][snr] = keyed
            runs[seed]["weights"][snr] = weights
    return runs


def prediction_metrics(rows, keys, weights):
    truth = [float(rows[key]["y_true"]) for key in keys]
    predicted = [sum(weights[model] * float(rows[key][model]) for model in SINGLES) for key in keys]
    residuals = [estimate - actual for estimate, actual in zip(predicted, truth)]
    mean = sum(truth) / len(truth)
    tss = sum((value - mean) ** 2 for value in truth)
    sse = sum(error**2 for error in residuals)
    near = [error for error, actual in zip(residuals, truth)
            if abs(actual - EVAL_ONB) <= 0.1 * EVAL_ONB]
    return {
        "r2": 1.0 - sse / tss,
        "rmse_kw_m2": math.sqrt(sse / len(keys)) / 1000.0,
        "mae_kw_m2": sum(abs(error) for error in residuals) / len(keys) / 1000.0,
        "near_onb_rmse_kw_m2": math.sqrt(sum(error**2 for error in near) / len(near)) / 1000.0,
        "misses": sum(actual >= EVAL_ONB and estimate < EVAL_ONB
                      for actual, estimate in zip(truth, predicted)),
        "false_alarms": sum(actual < EVAL_ONB and estimate >= EVAL_ONB
                            for actual, estimate in zip(truth, predicted)),
    }


def audit_matched_weights():
    runs = load_matched_runs()
    rows = []
    for seed, record in sorted(runs.items()):
        keys = sorted(record["predictions"]["-8"])
        if set(keys) != set(record["predictions"]["-16"]):
            raise ValueError(f"Prediction keys differ across SNR: seed={seed}")
        w8 = record["weights"]["-8"]
        w16 = record["weights"]["-16"]
        schemes = {
            "condition_specific": None,
            "fixed_minus8": w8,
            "fixed_minus16": w16,
            "fixed_pair_mean": {model: (w8[model] + w16[model]) / 2.0 for model in SINGLES},
            "equal": {model: 1.0 / len(SINGLES) for model in SINGLES},
        }
        for snr in SNRS:
            for name, scheme in schemes.items():
                weights = record["weights"][snr] if scheme is None else scheme
                metrics = prediction_metrics(record["predictions"][snr], keys, weights)
                rows.append({
                    "seed": seed,
                    "snr": snr,
                    "weight_scheme": name,
                    **metrics,
                    **{f"weight_{model}": weights[model] for model in SINGLES},
                })
    write_csv(OUT / "matched_weight_counterfactual.csv", rows)

    delta_rows = []
    for seed in sorted(runs):
        for scheme in ("condition_specific", "fixed_minus8", "fixed_minus16", "fixed_pair_mean", "equal"):
            pair = {row["snr"]: row for row in rows
                    if row["seed"] == seed and row["weight_scheme"] == scheme}
            delta_rows.append({
                "seed": seed,
                "weight_scheme": scheme,
                "r2_minus8": pair["-8"]["r2"],
                "r2_minus16": pair["-16"]["r2"],
                "delta_r2_minus16_minus8": pair["-16"]["r2"] - pair["-8"]["r2"],
                "delta_rmse_kw_m2_minus16_minus8": (
                    pair["-16"]["rmse_kw_m2"] - pair["-8"]["rmse_kw_m2"]
                ),
                "delta_near_onb_rmse_kw_m2_minus16_minus8": (
                    pair["-16"]["near_onb_rmse_kw_m2"] - pair["-8"]["near_onb_rmse_kw_m2"]
                ),
                "delta_misses_minus16_minus8": pair["-16"]["misses"] - pair["-8"]["misses"],
            })
    write_csv(OUT / "matched_weight_delta_summary.csv", delta_rows)
    return delta_rows


def load_selection_audits():
    audits = {}
    for record in read_csv(C_RUNS):
        if record["arm"] != "on":
            continue
        seed = int(record["seed"])
        path = (ROOT / record["run"] / "maxfreq=22kHz" / "heatflux_no_noise"
                / "training_selection_fold1.json")
        audits[seed] = json.loads(long_path(path).read_text(encoding="utf-8"))
    if set(audits) != {42, 43, 44}:
        raise ValueError(f"Incomplete C selection audits: {sorted(audits)}")
    return audits


def audit_removed_samples():
    audits = load_selection_audits()
    decisions_by_seed = {}
    for seed, audit in audits.items():
        decisions_by_seed[seed] = {
            (item["experiment_name"], item["source_wav_id"], int(item["chunk_index"])):
            (float(item["heat_flux"]), float(item["peak_height_psd"]), bool(item["keep"]))
            for item in audit["decisions"]
        }
    reference = decisions_by_seed[42]
    if any(decisions != reference for seed, decisions in decisions_by_seed.items() if seed != 42):
        raise ValueError("C selection decisions are not identical across seeds")

    removed_rows = []
    grouped = defaultdict(list)
    for key, (heat_flux, peak, keep) in sorted(reference.items()):
        experiment, wav, chunk = key
        if keep:
            continue
        grouped[(wav, heat_flux)].append((chunk, peak))
        removed_rows.append({
            "experiment_name": experiment,
            "source_wav_id": wav,
            "chunk_index": chunk,
            "heat_flux_w_m2": heat_flux,
            "peak_2100_2500_psd": peak,
            "relative_to_run_onb_221505": "at_or_above" if heat_flux >= TRAIN_RUN_ONB else "below",
            "relative_to_source_onb_369000": "at_or_above" if heat_flux >= TRAIN_SOURCE_ONB else "below",
        })
    if len(removed_rows) != 94:
        raise ValueError(f"Expected 94 removed samples, got {len(removed_rows)}")
    write_csv(OUT / "c_removed_samples.csv", removed_rows)

    summary = []
    by_wav_all = defaultdict(list)
    for (_, wav, _), (heat_flux, peak, keep) in reference.items():
        by_wav_all[(wav, heat_flux)].append((peak, keep))
    for (wav, heat_flux), removed in sorted(grouped.items(), key=lambda item: item[0][1]):
        all_samples = by_wav_all[(wav, heat_flux)]
        removed_peaks = [peak for _, peak in removed]
        retained_peaks = [peak for peak, keep in all_samples if keep]
        summary.append({
            "source_wav_id": wav,
            "heat_flux_w_m2": heat_flux,
            "total_chunks": len(all_samples),
            "removed_chunks": len(removed),
            "retained_chunks": len(retained_peaks),
            "removed_fraction": len(removed) / len(all_samples),
            "removed_peak_min": min(removed_peaks),
            "removed_peak_median": statistics.median(removed_peaks),
            "removed_peak_max": max(removed_peaks),
            "retained_peak_min": min(retained_peaks) if retained_peaks else "",
            "distance_above_run_onb_w_m2": heat_flux - TRAIN_RUN_ONB,
            "distance_below_source_onb_w_m2": TRAIN_SOURCE_ONB - heat_flux,
            "relative_to_run_onb_221505": "at_or_above" if heat_flux >= TRAIN_RUN_ONB else "below",
            "relative_to_source_onb_369000": "at_or_above" if heat_flux >= TRAIN_SOURCE_ONB else "below",
        })
    write_csv(OUT / "c_removed_wav_summary.csv", summary)
    return summary


def main():
    deltas = audit_matched_weights()
    removed = audit_removed_samples()
    for row in deltas:
        print(row["seed"], row["weight_scheme"], round(row["delta_r2_minus16_minus8"], 6))
    print("removed_by_wav", [(row["heat_flux_w_m2"], row["removed_chunks"]) for row in removed])


if __name__ == "__main__":
    main()
