"""Training-only endpoint review and label-balanced, WAV-disjoint fold audit."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "experiments/2026-10-04_training_oof_diversity_diagnosis"
CONFIG_PATH = ROOT / "configs/experiments/2026-10-05_onb_endpoint_and_grouped_fold_review.json"
THRESHOLDS = {"20250611": 221.5051102, "20250618": 271.6776816}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(name, rows):
    assert rows, name
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint(y, p, threshold):
    """Same measured-stage suffix definition as the adopted outer analyses."""
    levels = np.unique(y)
    failures = y[p < threshold]
    q = next((float(level) for level in levels if np.all(p[y >= level] >= threshold)), np.nan)
    return {
        "q100_kW_m2": q, "g100_kW_m2": q - threshold,
        "q100_over_onb": q / threshold,
        "last_failed_stage_kW_m2": float(failures.max()) if len(failures) else "",
        "n_suffix_chunks": int((y >= q).sum()) if np.isfinite(q) else 0,
        "n_suffix_stages": int((levels >= q).sum()) if np.isfinite(q) else 0,
        "n": len(y), "n_pre": int((y < threshold).sum()),
        "fp": int(((y < threshold) & (p >= threshold)).sum()),
        "fn": int(((y >= threshold) & (p < threshold)).sum()),
        "fpr": float((p[y < threshold] >= threshold).mean()) if (y < threshold).any() else "",
        "recall": float((p[y >= threshold] >= threshold).mean()) if (y >= threshold).any() else "",
        "rmse_kW_m2": float(np.sqrt(np.mean((p-y)**2))),
    }


def training_rows():
    rows = [r for r in read_csv(PREVIOUS / "oof_sample_diagnostics.csv") if r["noise"] == "clean"]
    assert len(rows) == 1620
    assert len({(r["source_wav_id"], r["chunk_index"]) for r in rows}) == 1620
    return rows


def group_records(rows):
    records = {}
    for row in rows:
        group = row["source_wav_id"]
        y, t = float(row["y_kW_m2"]), float(row["onb_kW_m2"])
        record = records.setdefault(group, {"source_wav_id": group, "day": group[:8],
            "heat_flux_kW_m2": y, "onb_kW_m2": t, "n": 0})
        assert record["heat_flux_kW_m2"] == y and record["onb_kW_m2"] == t
        record["n"] += 1
    return records


def rank_balanced_folds(records, n_splits=3, seed=42):
    """One WAV from each day-specific adjacent rank block per held-out fold.

    A label-only constraint prevents all near-ONB groups from being held together
    when at least two exist. No prediction, noise feature or outer-test data is used.
    With one such group, fully cross-fitted held-out coverage necessarily gives one
    fit without it; this function does not hide or duplicate that group.
    """
    near = [g for g, r in records.items() if .9 <= r["heat_flux_kW_m2"]/r["onb_kW_m2"] <= 1.1]
    rng = np.random.default_rng(seed)
    for attempt in range(1, 1001):
        assignment = {}
        for day in sorted({r["day"] for r in records.values()}):
            ordered = sorted([g for g, r in records.items() if r["day"] == day],
                             key=lambda g: (records[g]["heat_flux_kW_m2"], g))
            for start in range(0, len(ordered), n_splits):
                for group, fold in zip(ordered[start:start+n_splits], rng.permutation(n_splits)+1):
                    assignment[group] = int(fold)
        if len(near) < 2 or len({assignment[g] for g in near}) >= 2:
            return assignment, attempt
    raise RuntimeError("Could not satisfy near-ONB label coverage; inspect group feasibility")


def score_for_selection(y, p, days, objective):
    rmse = float(np.sqrt(np.mean((p-y)**2)))
    if objective == "rmse":
        return (rmse,)
    # Priority alternatives are exploratory; no numeric FP allowance is assumed.
    ends = [endpoint(y[days == day], p[days == day], THRESHOLDS[day]) for day in np.unique(days)]
    fp = sum(r["fp"] for r in ends)
    gaps = [float(r["g100_kW_m2"]) if np.isfinite(r["g100_kW_m2"]) else float("inf") for r in ends]
    if objective == "q100_then_fp":
        return (max(gaps), float(np.mean(gaps)), fp, rmse)
    assert objective == "fp_then_q100"
    return (fp, max(gaps), float(np.mean(gaps)), rmse)


def summarize_predictions(name, condition, rows, prediction, **fields):
    y = np.asarray([float(r["y_kW_m2"]) for r in rows])
    days = np.asarray([r["source_wav_id"][:8] for r in rows])
    result = []
    for day, threshold in THRESHOLDS.items():
        mask = days == day
        result.append({**fields, "method": name, "condition": condition, "day": day,
                       **endpoint(y[mask], prediction[mask], threshold)})
    return result


def main():
    rows = training_rows()
    identities = [(r["source_wav_id"], r["chunk_index"]) for r in rows]
    records = group_records(rows)
    original = {r["source_wav_id"]: int(r["fold"]) for r in rows}
    balanced, attempts = rank_balanced_folds(records)
    swap = json.loads((ROOT / "configs/experiments/2026-10-04_internal_onb_fold_sensitivity.json").read_text(encoding="utf-8"))
    swapped = {**original, **swap["swap"]}
    methods, stages = [], []
    previous_oof = read_csv(PREVIOUS / "oof_sample_diagnostics.csv")
    for noise in ["clean", "0", "-4", "-8", "-12", "-16", "-20"]:
        selected = [r for r in previous_oof if r["noise"] == noise]
        assert [(r["source_wav_id"], r["chunk_index"]) for r in selected] == identities
        policy = selected[0]["policy"]
        for name in ["randomforest", "conformer", "alexnet", "performance_oof_diagnostic"]:
            methods.extend(summarize_predictions(name, noise, rows,
                np.asarray([float(r[name]) for r in selected]), policy=policy))
    for prefix in ["svr", "svr_shape"]:
        selected = read_csv(PREVIOUS / f"{prefix}_training_oof_predictions.csv")
        assert [(r["source_wav_id"], r["chunk_index"]) for r in selected] == identities
        for key in ["SVR_clean", "fixed_4_equal", "fixed_old75_SVR25", "SVR_minus20_clean_fitted_transfer"]:
            condition = "-20" if "minus20" in key else "clean"
            p = np.asarray([float(r[key]) for r in selected])
            methods.extend(summarize_predictions(f"{prefix}:{key}", condition, rows, p, policy="clean_only"))
            for group in sorted(records):
                indices = [i for i, r in enumerate(rows) if r["source_wav_id"] == group]
                rec = records[group]
                stages.append({"method": f"{prefix}:{key}", "condition": condition,
                    "source_wav_id": group, "day": rec["day"], "heat_flux_kW_m2": rec["heat_flux_kW_m2"],
                    "n": len(indices), "n_positive": int((p[indices] >= rec["onb_kW_m2"]).sum()),
                    "minimum_prediction_margin_kW_m2": float(p[indices].min()-rec["onb_kW_m2"])})
    save_csv("saved_oof_endpoints.csv", methods)
    save_csv("saved_pilot_stage_profiles.csv", stages)
    audit, group_rows = [], []
    for strategy, mapping in [("original_random", original), ("single_swap", swapped), ("rank_balanced", balanced)]:
        coverage = np.zeros(len(rows), dtype=int)
        for fold in [1, 2, 3]:
            fit = [g for g in records if mapping[g] != fold]
            held = [g for g in records if mapping[g] == fold]
            assert set(fit).isdisjoint(held) and len(fit) == 24 and len(held) == 12
            for i, row in enumerate(rows):
                coverage[i] += int(mapping[row["source_wav_id"]] == fold)
            for day in THRESHOLDS:
                own_fit = [records[g] for g in fit if records[g]["day"] == day]
                own_held = [records[g] for g in held if records[g]["day"] == day]
                audit.append({"strategy": strategy, "fold": fold, "day": day,
                    "fit_wavs": len(fit), "held_wavs": len(held), "shared_wavs": 0,
                    "fit_wavs_this_day": len(own_fit), "held_wavs_this_day": len(own_held),
                    "fit_near_onb_wavs_all_days": sum(.9 <= records[g]["heat_flux_kW_m2"]/records[g]["onb_kW_m2"] <= 1.1 for g in fit),
                    "fit_near_onb_wavs_this_day": sum(.9 <= r["heat_flux_kW_m2"]/r["onb_kW_m2"] <= 1.1 for r in own_fit),
                    "held_exact_day_stages_missing_in_fit": sum(not any(r["heat_flux_kW_m2"] == h["heat_flux_kW_m2"] for r in own_fit) for h in own_held),
                    "fit_min_heat_flux_kW_m2": min(r["heat_flux_kW_m2"] for r in own_fit),
                    "fit_max_heat_flux_kW_m2": max(r["heat_flux_kW_m2"] for r in own_fit)})
        assert np.all(coverage == 1)
        for group, rec in records.items():
            group_rows.append({"strategy": strategy, **rec, "held_fold": mapping[group]})
    save_csv("fold_support_audit.csv", audit)
    save_csv("group_assignments.csv", group_rows)
    manifest = {"config": CONFIG_PATH.relative_to(ROOT).as_posix(), "training_chunks": 1620,
        "wavs": 36, "stages_per_day": {day: len([r for r in records.values() if r["day"] == day]) for day in THRESHOLDS},
        "training_identity_sha256": hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
        "source_csv_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [PREVIOUS / "oof_sample_diagnostics.csv", PREVIOUS / "training_input_features.csv"]},
        "rank_balanced_label_constraint_attempts": attempts, "folds": balanced,
        "outer_test_used": False, "main_runner_changed": False,
        "q100_evaluation": "Pooled training OOF, 45 chunks per WAV; not outer 15-chunk or final-fit endpoints"}
    (OUTPUT / "audit.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for row in methods:
        if row["condition"] == "clean" and (row["method"] == "performance_oof_diagnostic" or "SVR_clean" in row["method"] or "fixed_old75" in row["method"]):
            print(row["method"], row["day"], "q100", round(row["q100_kW_m2"], 2), "FP", row["fp"])
    print("Balanced WAV-disjoint folds audited; label-only attempts:", attempts)


if __name__ == "__main__":
    main()
