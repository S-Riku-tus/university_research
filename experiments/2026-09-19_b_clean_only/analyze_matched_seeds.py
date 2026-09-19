"""Summarize predeclared 42/43/44 matched -8/-16 pairs on 18 paired WAVs."""

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN_PARENT = ROOT / (
    "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/"
    "regression_result/npy/ensemble/20260919"
)
SEEDS = (42, 43, 44)
SNRS = ("-8", "-16")
SINGLES = ("randomforest", "conformer", "alexnet")
MODELS = (*SINGLES, "ensemble__inner_holdout", "equal_offline")
ONB = 271677.6816


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_runs():
    found = {}
    for manifest in RUN_PARENT.glob("*/maxfreq=22kHz/heatflux_reference_SNR=-8/run_manifest.json"):
        record = json.loads(manifest.read_text(encoding="utf-8"))
        policy = record.get("validation_config", {}).get("learning_policy", {})
        seed = record.get("validation_config", {}).get("run", {}).get("random_seed")
        if policy.get("training_noise") != "matched" or seed not in SEEDS:
            continue
        root = manifest.parents[2]
        if seed in found:
            raise RuntimeError(f"Duplicate seed {seed}: {found[seed]}, {root}")
        found[seed] = root
    if set(found) != set(SEEDS):
        raise RuntimeError(f"Missing completed matched seeds: found {found}")
    return found


def main():
    runs = find_runs()
    records, weights, manifests = {}, {}, {}
    for seed in SEEDS:
        records[seed], weights[seed], manifests[seed] = {}, {}, {}
        for snr in SNRS:
            directory = runs[seed] / "maxfreq=22kHz" / f"heatflux_reference_SNR={snr}"
            completion = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
            manifests[seed][snr] = completion
            rows = read_csv(directory / "fold_pred" / f"pred_f1_{snr}.csv")
            keyed = {(r["experiment_name"], r["source_wav_id"], int(r["chunk_index"])): r for r in rows}
            assert len(rows) == len(keyed) == 1080
            records[seed][snr] = keyed
            weight_row, = read_csv(directory / f"ensemble_weights_{snr}.csv")
            weights[seed][snr] = {model: float(weight_row[model]) for model in SINGLES}
            assert math.isclose(sum(weights[seed][snr].values()), 1, abs_tol=1e-10)
        assert manifests[seed]["-8"]["run_hash"] == manifests[seed]["-16"]["run_hash"]
        assert manifests[seed]["-8"]["fit_ids"] != manifests[seed]["-16"]["fit_ids"]
    keys = set(records[42]["-8"])
    assert len({key[1] for key in keys}) == 18
    assert all(set(records[seed][snr]) == keys for seed in SEEDS for snr in SNRS)
    for key in keys:
        target = float(records[42]["-8"][key]["y_true"])
        assert all(math.isclose(float(records[seed][snr][key]["y_true"]), target, abs_tol=1e-6)
                   for seed in SEEDS for snr in SNRS)
    keys = sorted(keys)
    ys = [float(records[42]["-8"][key]["y_true"]) for key in keys]
    mean = sum(ys) / len(ys)
    tss = sum((y - mean) ** 2 for y in ys)

    def predict(seed, snr, key, model):
        row = records[seed][snr][key]
        if model == "equal_offline":
            return sum(float(row[m]) for m in SINGLES) / 3
        return float(row[model])

    metrics, paired_wavs = [], []
    for seed in SEEDS:
        for snr in SNRS:
            for model in MODELS:
                residuals = [predict(seed, snr, key, model) - y for key, y in zip(keys, ys)]
                sse = sum(e ** 2 for e in residuals)
                near = [e for e, y in zip(residuals, ys) if abs(y - ONB) <= .1 * ONB]
                metrics.append({"seed": seed, "snr": snr, "model": model,
                                "r2": 1 - sse / tss,
                                "rmse_kw_m2": math.sqrt(sse / len(keys)) / 1000,
                                "mae_kw_m2": sum(abs(e) for e in residuals) / len(keys) / 1000,
                                "near_onb_rmse_kw_m2": math.sqrt(sum(e ** 2 for e in near) / len(near)) / 1000,
                                "misses": sum(y >= ONB and predict(seed, snr, key, model) < ONB
                                              for key, y in zip(keys, ys)),
                                "false_alarms": sum(y < ONB and predict(seed, snr, key, model) >= ONB
                                                    for key, y in zip(keys, ys)),
                                "sse_w_m2_squared": sse,
                                **{f"weight_{m}": weights[seed][snr][m] if model == "ensemble__inner_holdout" else ""
                                   for m in SINGLES}})
        for model in MODELS:
            for wav in sorted({key[1] for key in keys}):
                wav_keys = [key for key in keys if key[1] == wav]
                y = float(records[seed]["-8"][wav_keys[0]]["y_true"])
                delta = sum((predict(seed, "-8", key, model) - y) ** 2 -
                            (predict(seed, "-16", key, model) - y) ** 2
                            for key in wav_keys)
                paired_wavs.append({"seed": seed, "model": model, "source_wav_id": wav,
                                    "y_true_w_m2": y,
                                    "region": "near_onb" if abs(y - ONB) <= .1 * ONB else (
                                        "below_onb" if y < ONB else "above_onb"),
                                    "chunks": len(wav_keys), "sse_reduction_w_m2_squared": delta})
    write_csv(OUT / "matched_seed_metrics.csv", metrics)
    write_csv(OUT / "matched_seed_paired_wavs.csv", paired_wavs)
    write_csv(OUT / "matched_seed_run_locations.csv", [
        {"seed": seed, "run": str(runs[seed].relative_to(ROOT)),
         "run_hash": manifests[seed]["-8"]["run_hash"],
         "fit_id_minus8": manifests[seed]["-8"]["fit_ids"][0],
         "fit_id_minus16": manifests[seed]["-16"]["fit_ids"][0]}
        for seed in SEEDS])
    for seed in SEEDS:
        subset = [r for r in metrics if r["seed"] == seed and r["model"] == "ensemble__inner_holdout"]
        print(seed, [(r["snr"], round(r["r2"], 4), r["misses"], r["false_alarms"])
                     for r in subset], "delta R2", round(subset[1]["r2"] - subset[0]["r2"], 4))


if __name__ == "__main__":
    main()
