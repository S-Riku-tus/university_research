"""Audit the predeclared C1/C2 pairs on identical 6/18 WAVs and seconds."""

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN_PARENT = ROOT / (
    "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/"
    "regression_result/npy/ensemble/20260920"
)
SEEDS = (42, 43, 44)
ARMS = ("off", "on")
SINGLES = ("randomforest", "conformer", "alexnet")
MODELS = (*SINGLES, "ensemble__inner_holdout", "equal_offline")
THRESHOLDS = {"run_271677": 271677.6816, "source_text_376000": 376000.0}


def long_path(path):
    """Open deeply nested run files on Windows without the legacy 260-char limit."""
    absolute = Path(path).resolve()
    return Path("\\\\?\\" + str(absolute)) if os.name == "nt" else absolute


def read_csv(path):
    with long_path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_runs():
    found = {}
    for run in RUN_PARENT.iterdir():
        if not run.is_dir() or "c_selection" not in run.name:
            continue
        manifest = long_path(run / "maxfreq=22kHz/heatflux_no_noise/run_manifest.json")
        if not manifest.exists():
            continue
        record = json.loads(manifest.read_text(encoding="utf-8"))
        config = record["validation_config"]
        seed = int(config["run"]["random_seed"])
        arm = "on" if config["acoustic_selection"]["enabled"] else "off"
        if seed not in SEEDS:
            continue
        assert config["learning_policy"]["train_experiments"] == ["2025.06.11_0.3_2"]
        assert config["learning_policy"]["test_experiments"] == ["2025.06.18_0.3_3"]
        assert config["run"]["epochs"] == 200
        assert config["data"]["noise_dir_names"] == ["heatflux_no_noise"]
        if arm == "on":
            assert config["acoustic_selection"]["peak_height_threshold"] == 1e-9
        key = seed, arm
        if key in found:
            raise RuntimeError(f"Duplicate C run for {key}")
        found[key] = run
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(found) != expected:
        raise RuntimeError(f"Incomplete C pairs; found={sorted(found)} missing={sorted(expected-set(found))}")
    return found


def main():
    runs = find_runs()
    data, weights, selection, completion = {}, {}, {}, {}
    for key, run in sorted(runs.items()):
        folder = long_path(run / "maxfreq=22kHz/heatflux_no_noise")
        completion[key] = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
        rows = read_csv(folder / "fold_pred/pred_f1_no_noise.csv")
        indexed = {(r["experiment_name"], r["source_wav_id"], int(r["chunk_index"])): r for r in rows}
        assert len(rows) == len(indexed) == 1080
        data[key] = indexed
        weight_row, = read_csv(folder / "ensemble_weights_no_noise.csv")
        assert weight_row["strategy_name"] == "inner_holdout"
        weights[key] = {m: float(weight_row[m]) for m in SINGLES}
        assert math.isclose(sum(weights[key].values()), 1, abs_tol=1e-9)
        selection[key] = json.loads((folder / "training_selection_fold1.json").read_text(encoding="utf-8"))
        assert selection[key]["enabled"] == (key[1] == "on")
    keys = sorted(data[(42, "off")])
    assert len({k[1] for k in keys}) == 18
    assert all(set(data[key]) == set(keys) for key in data)
    truth = {k: float(data[(42, "off")][k]["y_true"]) for k in keys}
    for key in data:
        assert all(math.isclose(float(data[key][k]["y_true"]), truth[k], abs_tol=1e-6) for k in keys)
    mean = sum(truth.values()) / len(keys)
    tss = sum((y - mean) ** 2 for y in truth.values())

    def prediction(run_key, sample_key, model):
        row = data[run_key][sample_key]
        if model == "equal_offline":
            return sum(float(row[m]) for m in SINGLES) / 3
        return float(row[model])

    metrics, wavs, locations, regions = [], [], [], []
    for run_key, run in sorted(runs.items()):
        seed, arm = run_key
        n_before = int(selection[run_key]["n_before"])
        n_after = int(selection[run_key]["n_after"])
        assert n_before == 1080 and (n_after <= n_before)
        tuning = {row["model_key"]: row for row in read_csv(run / "tuning_summary.csv")}
        for model_key in ("conformer", "alexnet"):
            assert tuning[model_key]["epochs_completed"] == "200"
            assert tuning[model_key]["actual_batch_sizes"] == "12"
        if arm == "off":
            assert n_after == n_before
        else:
            assert n_after < n_before
        locations.append({"seed": seed, "arm": arm, "run": str(run.relative_to(ROOT)),
                          "run_hash": completion[run_key]["run_hash"],
                          "fit_id": completion[run_key]["fit_ids"][0],
                          "n_before": n_before, "n_after": n_after,
                          "n_removed": n_before-n_after,
                          "nominal_full_fit_updates_per_cnn": math.ceil(n_after / 12) * 200,
                          **{f"weight_{m}": weights[run_key][m] for m in SINGLES}})
        for k in keys:
            weighted = sum(weights[run_key][m] * prediction(run_key, k, m) for m in SINGLES)
            assert math.isclose(weighted, prediction(run_key, k, "ensemble__inner_holdout"), abs_tol=1e-4)
        for model in MODELS:
            residuals = {k: prediction(run_key, k, model) - truth[k] for k in keys}
            sse = sum(e*e for e in residuals.values())
            for threshold_name, threshold in THRESHOLDS.items():
                near = [e for k, e in residuals.items() if abs(truth[k] - threshold) <= .1*threshold]
                metrics.append({"seed": seed, "arm": arm, "model": model,
                                "threshold": threshold_name, "chunks": len(keys), "wavs": 18,
                                "r2": 1-sse/tss,
                                "rmse_kw_m2": math.sqrt(sse/len(keys))/1000,
                                "mae_kw_m2": sum(abs(e) for e in residuals.values())/len(keys)/1000,
                                "sse_w_m2_squared": sse,
                                "near_onb_chunks": len(near),
                                "near_onb_rmse_kw_m2": math.sqrt(sum(e*e for e in near)/len(near))/1000 if near else "",
                                "misses": sum(truth[k] >= threshold and prediction(run_key, k, model) < threshold for k in keys),
                                "false_alarms": sum(truth[k] < threshold and prediction(run_key, k, model) >= threshold for k in keys)})
                grouped = defaultdict(list)
                for k, error in residuals.items():
                    y = truth[k]
                    region = ("near_onb" if abs(y-threshold) <= .1*threshold else
                              ("below_onb" if y < threshold else "above_onb"))
                    grouped[region].append((k, error))
                for region, values in grouped.items():
                    errors = [error for _, error in values]
                    regions.append({"seed": seed, "arm": arm, "model": model,
                                    "threshold": threshold_name, "region": region,
                                    "chunks": len(values), "wavs": len({k[1] for k, _ in values}),
                                    "sse_w_m2_squared": sum(error*error for error in errors),
                                    "rmse_kw_m2": math.sqrt(sum(error*error for error in errors)/len(errors))/1000,
                                    "mean_residual_kw_m2": sum(errors)/len(errors)/1000})
    assert len({r["fit_id"] for r in locations}) == 6
    for seed in SEEDS:
        for model in MODELS:
            for wav in sorted({k[1] for k in keys}):
                wav_keys = [k for k in keys if k[1] == wav]
                assert len(wav_keys) == 60
                delta = sum((prediction((seed, "off"), k, model)-truth[k])**2 -
                            (prediction((seed, "on"), k, model)-truth[k])**2 for k in wav_keys)
                y = truth[wav_keys[0]]
                wavs.append({"seed": seed, "model": model, "source_wav_id": wav,
                             "y_true_w_m2": y,
                             "region_run_threshold": "near_onb" if abs(y-THRESHOLDS["run_271677"]) <= .1*THRESHOLDS["run_271677"] else ("below_onb" if y < THRESHOLDS["run_271677"] else "above_onb"),
                             "chunks": len(wav_keys), "sse_reduction_on_minus_off_w_m2_squared": delta})
    write_csv(OUT / "c_metrics.csv", metrics)
    write_csv(OUT / "c_region_metrics.csv", regions)
    write_csv(OUT / "c_paired_wav_sse.csv", wavs)
    write_csv(OUT / "c_run_locations.csv", locations)
    for seed in SEEDS:
        subset = [r for r in metrics if r["seed"] == seed and r["threshold"] == "run_271677" and r["model"] == "ensemble__inner_holdout"]
        print(seed, [(r["arm"], round(r["r2"], 4), r["misses"], r["false_alarms"]) for r in subset],
              "deltaR2", round(subset[1]["r2"]-subset[0]["r2"], 4))


if __name__ == "__main__":
    main()
