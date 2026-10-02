"""Describe saved predictions for the adopted 3 kHz run; no model execution."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
REFERENCE = REPO / "experiments/2026-10-01_3khz_22khz_tuned_outer_comparison"
RUN = (
    REPO / "Pool_boiling/Subcooling_20_degrees/0.3"
    / "2025.06.11_0.3_2_6.18_0.3_3/regression_result/npy/ensemble"
    / "202610/01/onb_wc-t0611-v0611_iw3-nc_c1s_s0_e150_170655"
)
NOISES = [
    ("clean", "heatflux_no_noise", "no_noise"),
    *[(str(snr), f"heatflux_reference_SNR={snr}", str(snr))
      for snr in (0, -4, -8, -12, -16, -20)],
]
THRESHOLDS = {"20250611": 221505.1102, "20250618": 271677.6816}
MODELS = {
    "RF": "randomforest", "Conformer": "conformer", "AlexNet": "alexnet",
    "performance": "ensemble__performance_kfold", "equal": "ensemble__simple_equal",
}


def read_rows(path):
    path = Path(path).resolve()
    if not str(path).startswith("\\\\?\\"):
        path = Path("\\\\?\\" + str(path))
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def save(name, rows):
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(row):
    return row["source_wav_id"], int(float(row["chunk_index"]))


def source_day(row):
    return row["source_wav_id"].split("::", 1)[0]


def threshold(row):
    return THRESHOLDS[source_day(row)]


def region(row):
    y = float(row["y_true"])
    if y < 60000:
        return "below_60kW"
    if y < threshold(row):
        return "60kW_to_ONB"
    if y < 1.5 * threshold(row):
        return "ONB_to_1.5ONB"
    return "at_least_1.5ONB"


def mean(values):
    return sum(values) / len(values)


def metrics(rows, model="performance"):
    errors = [float(r[MODELS[model]]) - float(r["y_true"]) for r in rows]
    actual = [float(r["y_true"]) >= threshold(r) for r in rows]
    predicted = [float(r[MODELS[model]]) >= threshold(r) for r in rows]
    fp = sum(p and not a for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))
    n_post = sum(actual)
    n_pre = len(rows) - n_post
    return {
        "n": len(rows), "sse": sum(e * e for e in errors),
        "rmse_kW_m2": math.sqrt(mean([e * e for e in errors])) / 1000,
        "mae_kW_m2": mean([abs(e) for e in errors]) / 1000,
        "bias_kW_m2": mean(errors) / 1000,
        "n_pre": n_pre, "n_post": n_post, "fp": fp, "fn": fn,
        "fpr": fp / n_pre if n_pre else "",
        "recall": (n_post - fn) / n_post if n_post else "",
    }


def compare_to_conformer(rows):
    corrected, introduced, lower_sse = 0, 0, 0
    for row in rows:
        y, t = float(row["y_true"]), threshold(row)
        c, p = float(row["conformer"]), float(row["ensemble__performance_kfold"])
        actual = y >= t
        c_correct, p_correct = (c >= t) == actual, (p >= t) == actual
        corrected += int(not c_correct and p_correct)
        introduced += int(c_correct and not p_correct)
        lower_sse += int((p - y) ** 2 < (c - y) ** 2)
    return {
        "conformer_errors_corrected": corrected,
        "errors_introduced_vs_conformer": introduced,
        "ensemble_lower_squared_error_chunks": lower_sse,
    }


def main():
    reference_metrics = read_rows(REFERENCE / "metrics_source_day_thresholds.csv")
    region_rows, stage_rows, model_rows, largest_rows, audit_rows = [], [], [], [], []
    expected_ids = None
    for noise, directory, suffix in NOISES:
        source = RUN / "maxfreq=3kHz" / directory / "fold_pred" / f"pred_f1_{suffix}.csv"
        rows = read_rows(source)
        ids = sorted(identity(row) for row in rows)
        assert len(rows) == len(set(ids)) == 540, "Unexpected/duplicate evaluation chunks"
        if expected_ids is None:
            expected_ids = ids
        assert ids == expected_ids, "Noise conditions do not share evaluation identities"
        wav_counts = {key: sum(r["source_wav_id"] == key for r in rows)
                      for key in sorted({r["source_wav_id"] for r in rows})}
        assert len(wav_counts) == 36 and set(wav_counts.values()) == {15}
        total = metrics(rows)
        ref = next(r for r in reference_metrics
                   if r["frequency"] == "3kHz" and r["noise"] == noise
                   and r["model"] == "performance"
                   and r["scope"] == "two_day_source_threshold")
        assert math.isclose(total["rmse_kW_m2"] * 1000, float(ref["rmse"]), abs_tol=1e-6)
        assert total["fp"] == int(ref["fp"]) and total["fn"] == int(ref["fn"])
        audit_rows.append({"noise": noise, "n": len(rows), "wav_count": len(wav_counts),
                           "same_identities": True, "reference_metrics_match": True,
                           "input": source.relative_to(REPO).as_posix()})

        scopes = {"two_day": rows,
                  **{day: [r for r in rows if source_day(r) == day] for day in THRESHOLDS}}
        for scope, selected in scopes.items():
            for model in MODELS:
                model_rows.append({"noise": noise, "scope": scope, "model": model,
                                   **metrics(selected, model)})
            local_total = metrics(selected)
            categories = ("below_60kW", "60kW_to_ONB", "ONB_to_1.5ONB", "at_least_1.5ONB")
            region_sse = 0.0
            for category in categories:
                group = [r for r in selected if region(r) == category]
                values = metrics(group)
                region_sse += values["sse"]
                region_rows.append({
                    "noise": noise, "scope": scope, "region": category, **values,
                    "sse_share_pct": values["sse"] / local_total["sse"] * 100,
                    "conformer_rmse_kW_m2": metrics(group, "Conformer")["rmse_kW_m2"],
                    **compare_to_conformer(group),
                })
            assert math.isclose(region_sse, local_total["sse"], rel_tol=1e-12)
            assert not any(
                float(r["y_true"]) >= threshold(r)
                and float(r["ensemble__performance_kfold"]) < threshold(r)
                and region(r) != "ONB_to_1.5ONB"
                for r in selected
            ), "False negatives occurred outside the reported early-ONB region"

        for wav in sorted(wav_counts):
            group = [r for r in rows if r["source_wav_id"] == wav]
            values = metrics(group)
            predictions = [float(r["ensemble__performance_kfold"]) for r in group]
            prediction_mean = mean(predictions)
            assert len({float(r["y_true"]) for r in group}) == 1
            stage_rows.append({
                "noise": noise, "source_day": source_day(group[0]), "source_wav_id": wav,
                "heatflux_kW_m2": float(group[0]["y_true"]) / 1000,
                "onb_kW_m2": threshold(group[0]) / 1000, "region": region(group[0]),
                "prediction_mean_kW_m2": prediction_mean / 1000,
                "prediction_std_kW_m2": math.sqrt(mean([(p - prediction_mean) ** 2 for p in predictions])) / 1000,
                "predicted_positive_count": sum(p >= threshold(group[0]) for p in predictions),
                "sse_share_pct": values["sse"] / total["sse"] * 100, **values,
            })
        for rank, row in enumerate(sorted(
                rows, key=lambda r: abs(float(r["ensemble__performance_kfold"]) - float(r["y_true"])),
                reverse=True)[:10], 1):
            y, p = float(row["y_true"]), float(row["ensemble__performance_kfold"])
            t = threshold(row)
            decision = "FN" if y >= t and p < t else "FP" if y < t and p >= t else "correct"
            largest_rows.append({"noise": noise, "rank": rank, "source_wav_id": row["source_wav_id"],
                                 "chunk_index": row["chunk_index"], "region": region(row),
                                 "heatflux_kW_m2": y / 1000, "prediction_kW_m2": p / 1000,
                                 "error_kW_m2": (p - y) / 1000, "onb_decision": decision,
                                 "conformer_kW_m2": float(row["conformer"]) / 1000})

    save("input_audit.csv", audit_rows)
    save("region_metrics.csv", region_rows)
    save("stage_profiles.csv", stage_rows)
    save("model_metrics.csv", model_rows)
    save("largest_error_chunks.csv", largest_rows)
    summary = {
        "run": RUN.relative_to(REPO).as_posix(),
        "evaluation": "known WAV, unused 1s chunks, source-day ONB thresholds",
        "thresholds_W_m2": THRESHOLDS,
        "region_definition": "q<60000; 60000<=q<ONB; ONB<=q<1.5ONB; q>=1.5ONB",
        "conditions_audited": len(audit_rows),
        "two_day_performance": [r for r in model_rows if r["scope"] == "two_day" and r["model"] == "performance"],
        "two_day_regions_clean_minus20": [r for r in region_rows if r["scope"] == "two_day" and r["noise"] in {"clean", "-20"}],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Audited 7 conditions, 540 aligned chunks each, 36 WAVs, all reference metrics matched.")
    for row in summary["two_day_regions_clean_minus20"]:
        print(row["noise"], row["region"], "RMSE", round(row["rmse_kW_m2"], 2),
              "SSE%", round(row["sse_share_pct"], 2), "FP", row["fp"], "FN", row["fn"],
              "C errors corrected", row["conformer_errors_corrected"],
              "introduced", row["errors_introduced_vs_conformer"])


if __name__ == "__main__":
    main()
