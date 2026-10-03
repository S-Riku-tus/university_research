"""Diagnostic only: residual signs and feasible convex combinations of saved outputs."""

import csv
import json
import math
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUN = ROOT / (
    "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2_6.18_0.3_3/"
    "regression_result/npy/ensemble/202610/01/"
    "onb_wc-t0611-v0611_iw3-nc_c1s_s0_e150_170655/maxfreq=3kHz"
)
MODELS = ("randomforest", "conformer", "alexnet")
THRESHOLDS = {"20250611": 221505.1102, "20250618": 271677.6816}


def long_path(path):
    return Path("\\\\?\\" + str(path.resolve()))


def mean(values):
    return sum(values) / len(values)


def label(row):
    y = row["y"]
    if y < 60000:
        return "below_60kW"
    if y < row["onb"]:
        return "60kW_to_ONB"
    if y < 1.5 * row["onb"]:
        return "ONB_to_1.5ONB"
    return "at_least_1.5ONB"


def normalized(raw, target):
    return [{
        "source_wav_id": r["source_wav_id"], "chunk_index": int(r["chunk_index"]),
        "y": float(r[target]), "onb": THRESHOLDS[r["source_wav_id"].split("::")[0]],
        **{m: float(r[m]) for m in MODELS},
        **({"performance": float(r["ensemble__performance_kfold"])}
           if "ensemble__performance_kfold" in r else {}),
    } for r in raw]


def save(name, rows):
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    inputs = []
    identity_sets = []
    for noise, directory, suffix in [
        ("clean", "heatflux_no_noise", "no_noise"),
        ("-20", "heatflux_reference_SNR=-20", "-20"),
    ]:
        path = RUN / directory / "fold_pred" / f"pred_f1_{suffix}.csv"
        with long_path(path).open(encoding="utf-8-sig", newline="") as source:
            rows = normalized(list(csv.DictReader(source)), "y_true")
        ids = {(r["source_wav_id"], r["chunk_index"]) for r in rows}
        assert len(rows) == len(ids) == 540
        identity_sets.append(ids)
        inputs.append(("outer", noise, rows))
    assert identity_sets[0] == identity_sets[1]
    audit_path = RUN / "heatflux_no_noise/internal_validation_fold1.json"
    audit = json.loads(long_path(audit_path).read_text(encoding="utf-8-sig"))
    assert not audit["test_used"] and audit["method"] == "wav_kfold"
    assert all(r["shared_source_wavs"] == 0 for r in audit["folds"])
    oof = normalized(audit["samples"], "heat_flux")
    oof_ids = {(r["source_wav_id"], r["chunk_index"]) for r in oof}
    assert len(oof) == len(oof_ids) == 1620
    assert not oof_ids & identity_sets[0]
    inputs.append(("training_oof", "clean", oof))

    feasibility, biases, pairs = [], [], []
    for scope, noise, rows in inputs:
        for region in ["all", "below_60kW", "60kW_to_ONB", "ONB_to_1.5ONB", "at_least_1.5ONB"]:
            selected = rows if region == "all" else [r for r in rows if label(r) == region]
            n = len(selected)
            all_over, all_under, inside = 0, 0, 0
            fn = recoverable_fn = 0
            oracle_squared_error = []
            errors = {m: [r[m] - r["y"] for r in selected] for m in MODELS}
            for row in selected:
                y, t = row["y"], row["onb"]
                low, high = min(row[m] for m in MODELS), max(row[m] for m in MODELS)
                all_over += low > y
                all_under += high < y
                inside += low <= y <= high
                distance = max(low - y, y - high, 0.0)
                oracle_squared_error.append(distance * distance)
                if "performance" in row:
                    bad = y >= t and row["performance"] < t
                    fn += bad
                    recoverable_fn += bad and high >= t
            assert all_over + all_under + inside == n
            feasibility.append({
                "scope": scope, "noise": noise, "region": region, "n": n,
                "all_three_overpredict": all_over, "all_three_underpredict": all_under,
                "true_value_inside_prediction_range": inside,
                "oracle_lower_bound_rmse_kW_m2": math.sqrt(mean(oracle_squared_error)) / 1000,
                "performance_fn": fn if scope == "outer" else "",
                "fn_with_some_model_positive": recoverable_fn if scope == "outer" else "",
                "fn_all_models_negative": fn - recoverable_fn if scope == "outer" else "",
            })
            for m, values in errors.items():
                bias = mean(values)
                biases.append({"scope": scope, "noise": noise, "region": region, "model": m,
                               "n": n, "bias_kW_m2": bias / 1000,
                               "rmse_kW_m2": math.sqrt(mean([v * v for v in values])) / 1000})
            for a, b in combinations(MODELS, 2):
                ea, eb = errors[a], errors[b]
                ba, bb = mean(ea), mean(eb)
                product = mean([u * v for u, v in zip(ea, eb)])
                covariance = product - ba * bb
                variance_a = mean([(v - ba) ** 2 for v in ea])
                variance_b = mean([(v - bb) ** 2 for v in eb])
                correlation = covariance / math.sqrt(variance_a * variance_b) if variance_a * variance_b > 0 else ""
                pairs.append({"scope": scope, "noise": noise, "region": region, "model_a": a, "model_b": b,
                              "n": n, "opposite_sign_count": sum(u * v < 0 for u, v in zip(ea, eb)),
                              "residual_correlation": correlation,
                              "mean_residual_product_kW2_m4": product / 1e6})
    save("convex_feasibility.csv", feasibility)
    save("model_region_bias.csv", biases)
    save("residual_pairs.csv", pairs)
    (OUTPUT / "input_audit.json").write_text(json.dumps({
        "run": RUN.relative_to(ROOT).as_posix(), "outer_conditions": ["clean", "-20"],
        "outer_chunks_each": 540, "training_oof_chunks": 1620,
        "outer_identity_match": True, "training_outer_chunk_overlap": 0,
        "internal_shared_source_wavs": 0,
        "warning": "Gold-dependent oracle is diagnostic, not an implementable selector or measured new-method performance.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Two aligned 540-chunk outer conditions and 1620 clean training-OOF samples audited.")
    for row in feasibility:
        if row["region"] in {"below_60kW", "ONB_to_1.5ONB"}:
            print(row)


if __name__ == "__main__":
    main()
