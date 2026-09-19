"""Compare the four clean-only outputs on paired 6/18 chunks and WAVs."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN_PARENT = ROOT / (
    "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/"
    "regression_result/npy/ensemble/20260919"
)
SNRS = ("no_noise", "-8", "-16", "-20")
SNR_DIRS = {"no_noise": "heatflux_no_noise", **{
    snr: f"heatflux_reference_SNR={snr}" for snr in SNRS[1:]}}
SINGLES = ("randomforest", "conformer", "alexnet")
MODELS = (*SINGLES, "ensemble__inner_holdout", "equal_offline")
ONB = 271677.6816
NEAR = 0.1 * ONB


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def locate_run():
    candidates = sorted(RUN_PARENT.glob("*/maxfreq=22kHz/heatflux_no_noise/completed.json"))
    candidates = [path.parents[2] for path in candidates
                  if "b_clean_only" in path.parents[2].name]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one completed B run, found {candidates}")
    return candidates[0]


def region(y):
    return "near_onb" if abs(y - ONB) <= NEAR else (
        "below_onb" if y < ONB else "above_onb")


def main():
    run = locate_run()
    data, weight_map, completion = {}, {}, {}
    for snr in SNRS:
        folder = run / "maxfreq=22kHz" / SNR_DIRS[snr]
        completion[snr] = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
        rows = read_csv(folder / "fold_pred" / f"pred_f1_{snr}.csv")
        keys = [(r["experiment_name"], r["source_wav_id"], int(r["chunk_index"])) for r in rows]
        assert len(rows) == len(set(keys)) == 1080
        data[snr] = dict(zip(keys, rows))
        weights = read_csv(folder / f"ensemble_weights_{snr}.csv")
        assert len(weights) == 1 and weights[0]["strategy_name"] == "inner_holdout"
        weight_map[snr] = {m: float(weights[0][m]) for m in SINGLES}
    assert len({tuple(value["fit_ids"]) for value in completion.values()}) == 1
    assert len({value["run_hash"] for value in completion.values()}) == 1
    assert all(weight_map[snr] == weight_map[SNRS[0]] for snr in SNRS)
    keys = sorted(data[SNRS[0]])
    assert len({key[1] for key in keys}) == 18
    for snr in SNRS:
        assert set(data[snr]) == set(keys)
        for key in keys:
            assert math.isclose(float(data[snr][key]["y_true"]),
                                float(data[SNRS[0]][key]["y_true"]), abs_tol=1e-6)
    predictions = {}
    for snr in SNRS:
        predictions[snr] = {}
        for key in keys:
            raw = data[snr][key]
            y = float(raw["y_true"])
            values = {m: float(raw[m]) for m in MODELS[:-1]}
            values["equal_offline"] = sum(values[m] for m in SINGLES) / 3
            weighted = sum(weight_map[snr][m] * values[m] for m in SINGLES)
            assert math.isclose(weighted, values["ensemble__inner_holdout"], abs_tol=1e-5)
            assert all(math.isfinite(value) for value in values.values())
            predictions[snr][key] = (y, values)

    truth = [predictions[SNRS[0]][key][0] for key in keys]
    mean = sum(truth) / len(truth)
    tss = sum((y - mean) ** 2 for y in truth)
    metric_rows, region_rows, wav_rows = [], [], []
    for snr in SNRS:
        for model in MODELS:
            residuals = [(key, predictions[snr][key][1][model] - predictions[snr][key][0])
                         for key in keys]
            sse = sum(e * e for _, e in residuals)
            by_region = defaultdict(list)
            for key, e in residuals:
                y = predictions[snr][key][0]
                by_region[region(y)].append((key, e))
            near = by_region["near_onb"]
            misses = sum(predictions[snr][key][0] >= ONB and
                         predictions[snr][key][1][model] < ONB for key in keys)
            alarms = sum(predictions[snr][key][0] < ONB and
                         predictions[snr][key][1][model] >= ONB for key in keys)
            metric_rows.append({"snr": snr, "model": model, "chunks": 1080,
                                "wavs": 18, "r2": 1 - sse / tss,
                                "rmse_kw_m2": math.sqrt(sse / 1080) / 1000,
                                "mae_kw_m2": sum(abs(e) for _, e in residuals) / 1080 / 1000,
                                "sse_w_m2_squared": sse,
                                "near_onb_chunks": len(near),
                                "near_onb_rmse_kw_m2": math.sqrt(sum(e * e for _, e in near) / len(near)) / 1000,
                                "misses": misses, "false_alarms": alarms})
            for name, group in by_region.items():
                region_rows.append({"snr": snr, "model": model, "region": name,
                                    "chunks": len(group), "wavs": len({key[1] for key, _ in group}),
                                    "sse_w_m2_squared": sum(e * e for _, e in group),
                                    "rmse_kw_m2": math.sqrt(sum(e * e for _, e in group) / len(group)) / 1000,
                                    "mean_residual_kw_m2": sum(e for _, e in group) / len(group) / 1000})
    for before, after in (("no_noise", "-8"), ("no_noise", "-16"),
                          ("no_noise", "-20"), ("-8", "-16"), ("-8", "-20")):
        for model in MODELS:
            by_wav = defaultdict(list)
            for key in keys:
                y = predictions[before][key][0]
                error_before = predictions[before][key][1][model] - y
                error_after = predictions[after][key][1][model] - y
                by_wav[key[1]].append(error_before ** 2 - error_after ** 2)
            for wav, changes in sorted(by_wav.items()):
                y = predictions[before][next(key for key in keys if key[1] == wav)][0]
                wav_rows.append({"before": before, "after": after, "model": model,
                                 "source_wav_id": wav, "y_true_w_m2": y,
                                 "region": region(y), "chunks": len(changes),
                                 "sse_reduction_w_m2_squared": sum(changes)})
    write_csv(OUT / "fixed_metrics.csv", metric_rows)
    write_csv(OUT / "fixed_region_metrics.csv", region_rows)
    write_csv(OUT / "fixed_paired_wav_sse.csv", wav_rows)
    (OUT / "run_location.txt").write_text(str(run.relative_to(ROOT)), encoding="utf-8")
    print(f"Verified same fit ID, run hash, weights, and 1080 labels/keys for four SNRs: {run}")
    for row in metric_rows:
        if row["model"] == "ensemble__inner_holdout":
            print(row["snr"], f"R2={row['r2']:.4f}",
                  f"RMSE={row['rmse_kw_m2']:.1f} kW/m2",
                  f"miss={row['misses']}", f"FA={row['false_alarms']}")


if __name__ == "__main__":
    main()
