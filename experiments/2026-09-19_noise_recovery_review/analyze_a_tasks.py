"""A1-A7 audit of paired saved predictions; source runs are read-only."""

import csv
import math
from collections import defaultdict

from analyze_saved_predictions import ONB, OUT, MODELS, read_predictions, source_dir, write_csv


SNRS = ("no_noise", "-8", "-16", "-20")
PREDICTION_SNRS = SNRS[:2]
NEAR_HALF_WIDTH = 0.10 * ONB  # Frozen from the completed run's onb_band_frac.


def weights_for(snr):
    path = source_dir(snr) / f"ensemble_weights_{snr}.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["strategy_name"] == "inner_holdout"
    weights = {model: float(rows[0][model]) for model in MODELS[:3]}
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
    return weights


def region(y):
    if abs(y - ONB) <= NEAR_HALF_WIDTH:
        return "near_onb"
    return "below_onb" if y < ONB else "above_onb"


def finite(value):
    result = float(value)
    assert math.isfinite(result)
    return result


def main():
    data = {snr: read_predictions(snr) for snr in SNRS}
    keys = set(data["no_noise"])
    assert all(set(rows) == keys for rows in data.values())
    assert len({key[1] for key in keys}) == 18
    assert all(sum(key[1] == wav for key in keys) == 60
               for wav in {key[1] for key in keys})
    weight_map = {snr: weights_for(snr) for snr in SNRS}
    alignment = []
    for snr in SNRS:
        labels = sum(not math.isclose(finite(data[snr][key]["y_true"]),
                                      finite(data["no_noise"][key]["y_true"]),
                                      abs_tol=1e-6) for key in keys)
        missing = sum(any(not row.get(column) for column in ("y_true", *MODELS))
                      for row in data[snr].values())
        assert labels == missing == 0
        alignment.append({"snr": snr, "rows": len(data[snr]), "wavs": 18,
                          "missing_keys": len(keys - set(data[snr])),
                          "duplicate_keys": 0, "label_mismatch": labels,
                          "missing_prediction_rows": missing,
                          **{f"weight_{m}": weight_map[snr][m] for m in MODELS[:3]}})
    write_csv(OUT / "a_alignment_weights.csv", list(alignment[0]), alignment)

    detail = []
    summary = defaultdict(lambda: {"n": 0, "wavs": set(), "sse": 0.0,
                                   "residual_sum": 0.0, "miss": 0, "false_alarm": 0})
    decomposition = defaultdict(lambda: {"diagonal": 0.0, "cross": 0.0,
                                         "cross_negative_chunks": 0, "cross_positive_chunks": 0,
                                         "weighted_worse_than_equal_chunks": 0})
    for snr in PREDICTION_SNRS:
        weights = weight_map[snr]
        for key in sorted(keys):
            original = data[snr][key]
            y = finite(original["y_true"])
            preds = {m: finite(original[m]) for m in MODELS}
            weighted = sum(weights[m] * preds[m] for m in MODELS[:3])
            assert math.isclose(weighted, preds["ensemble__inner_holdout"], abs_tol=1e-6)
            equal = sum(preds[m] for m in MODELS[:3]) / 3
            preds["equal_offline"] = equal
            residuals = {m: preds[m] - y for m in preds}
            diagonal = sum((weights[m] * residuals[m]) ** 2 for m in MODELS[:3])
            cross = 2 * sum(weights[MODELS[i]] * weights[MODELS[j]] *
                            residuals[MODELS[i]] * residuals[MODELS[j]]
                            for i, j in ((0, 1), (0, 2), (1, 2)))
            assert math.isclose(diagonal + cross,
                                residuals["ensemble__inner_holdout"] ** 2,
                                rel_tol=1e-8, abs_tol=1e-4)
            group = region(y)
            d = decomposition[snr]
            d["diagonal"] += diagonal
            d["cross"] += cross
            d["cross_negative_chunks"] += cross < 0
            d["cross_positive_chunks"] += cross > 0
            d["weighted_worse_than_equal_chunks"] += (residuals["ensemble__inner_holdout"] ** 2 >
                                                       residuals["equal_offline"] ** 2)
            row = {"snr": snr, "experiment_name": key[0], "source_wav_id": key[1],
                   "chunk_index": key[2], "y_true_w_m2": y, "region": group,
                   **{f"weight_{m}": weights[m] for m in MODELS[:3]},
                   "weighted_diagonal_w_m2_squared": diagonal,
                   "weighted_cross_w_m2_squared": cross,
                   "ensemble_minus_best_single_sse_w_m2_squared":
                       residuals["ensemble__inner_holdout"] ** 2 -
                       min(residuals[m] ** 2 for m in MODELS[:3]),
                   "ensemble_minus_equal_sse_w_m2_squared":
                       residuals["ensemble__inner_holdout"] ** 2 -
                       residuals["equal_offline"] ** 2}
            for model in (*MODELS, "equal_offline"):
                error = residuals[model]
                result = summary[(snr, group, model)]
                result["n"] += 1
                result["wavs"].add(key[1])
                result["sse"] += error ** 2
                result["residual_sum"] += error
                result["miss"] += y >= ONB and preds[model] < ONB
                result["false_alarm"] += y < ONB and preds[model] >= ONB
                row[f"prediction_{model}_w_m2"] = preds[model]
                row[f"residual_{model}_w_m2"] = error
                row[f"sse_{model}_w_m2_squared"] = error ** 2
                row[f"miss_{model}"] = int(y >= ONB and preds[model] < ONB)
                row[f"false_alarm_{model}"] = int(y < ONB and preds[model] >= ONB)
            detail.append(row)
    write_csv(OUT / "a_paired_chunk_residuals.csv", list(detail[0]), detail)

    region_rows = []
    for (snr, group, model), values in sorted(summary.items()):
        region_rows.append({"snr": snr, "region": group, "model": model,
                            "chunks": values["n"], "wavs": len(values["wavs"]),
                            "sse_w_m2_squared": values["sse"],
                            "rmse_kw_m2": math.sqrt(values["sse"] / values["n"]) / 1000,
                            "mean_residual_kw_m2": values["residual_sum"] / values["n"] / 1000,
                            "misses": values["miss"], "false_alarms": values["false_alarm"]})
    write_csv(OUT / "a_region_metrics.csv", list(region_rows[0]), region_rows)

    representative = []
    for snr in PREDICTION_SNRS:
        for wav in sorted({key[1] for key in keys}):
            candidates = [r for r in detail if r["snr"] == snr and r["source_wav_id"] == wav]
            for direction, selected in (
                ("ensemble_best_vs_all_singles", min(candidates, key=lambda r: r["ensemble_minus_best_single_sse_w_m2_squared"])),
                ("ensemble_worst_vs_best_single", max(candidates, key=lambda r: r["ensemble_minus_best_single_sse_w_m2_squared"])),
            ):
                representative.append({"selection": direction, **selected})
    write_csv(OUT / "a_representative_chunks_by_wav.csv", list(representative[0]), representative)

    decomposition_rows = []
    for snr in PREDICTION_SNRS:
        d = decomposition[snr]
        rows = [r for r in detail if r["snr"] == snr]
        weighted_sse = sum(r["sse_ensemble__inner_holdout_w_m2_squared"] for r in rows)
        equal_sse = sum(r["sse_equal_offline_w_m2_squared"] for r in rows)
        assert math.isclose(d["diagonal"] + d["cross"], weighted_sse, rel_tol=1e-10)
        decomposition_rows.append({"snr": snr, "chunks": len(rows), **d,
                                   "weighted_sse": weighted_sse, "equal_offline_sse": equal_sse,
                                   "weighted_minus_equal_sse": weighted_sse - equal_sse})
    write_csv(OUT / "a_error_decomposition.csv", list(decomposition_rows[0]), decomposition_rows)

    wav_rows = []
    for before, after in (("-8", "-16"), ("-8", "-20")):
        for wav in sorted({key[1] for key in keys}):
            wav_keys = [key for key in keys if key[1] == wav]
            y = finite(data[before][wav_keys[0]]["y_true"])
            for model in (*MODELS, "equal_offline"):
                def sse(snr):
                    total = 0.0
                    for key in wav_keys:
                        row = data[snr][key]
                        pred = (sum(finite(row[m]) for m in MODELS[:3]) / 3 if model == "equal_offline"
                                else finite(row[model]))
                        total += (pred - finite(row["y_true"])) ** 2
                    return total
                before_sse, after_sse = sse(before), sse(after)
                wav_rows.append({"before": before, "after": after, "model": model,
                                 "source_wav_id": wav, "y_true_w_m2": y,
                                 "region": region(y), "chunks": len(wav_keys),
                                 "sse_before": before_sse, "sse_after": after_sse,
                                 "sse_reduction_w_m2_squared": before_sse - after_sse})
    write_csv(OUT / "a_recovery_by_wav.csv", list(wav_rows[0]), wav_rows)
    print("A1-A7: 4 aligned conditions, 2 detailed conditions, 18 WAV, 1080 chunks each.")
    for snr in PREDICTION_SNRS:
        print(snr, "weights", weight_map[snr], "decomposition", dict(decomposition[snr]))


if __name__ == "__main__":
    main()
