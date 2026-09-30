"""Audit and compare the completed 3/22 kHz training-only OOF searches."""

import csv
import json
import math
import os
import random
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    REPO
    / "Pool_boiling"
    / "Subcooling_20_degrees"
    / "0.3"
    / "2025.06.11_0.3_2_6.18_0.3_3"
    / "regression_result"
    / "npy"
    / "tuning"
)
RUNS = {
    "3kHz": RESULT_ROOT
    / "202609"
    / "29"
    / "tuning_wc-t0611-v0611_iw3-nc_c1s_s0_e150_210848",
    "22kHz": RESULT_ROOT
    / "202609"
    / "30"
    / "tuning_wc-t0611-v0611_iw3-nc_c1s_s0_e150_102621",
}
RECOMMENDED = {
    "3kHz": {
        "randomforest": (
            "randomforest__n_estimators=100__max_depth=12__subsample=0.6__"
            "colsample_bynode=0.6"
        ),
        "conformer": "conformer__lr=0.001__batch_size=12",
        "alexnet": "alexnet__lr=0.001__batch_size=24",
    },
    "22kHz": {
        "randomforest": (
            "randomforest__n_estimators=600__max_depth=6__subsample=0.6__"
            "colsample_bynode=0.6"
        ),
        "conformer": "conformer__lr=0.0003__batch_size=8",
        "alexnet": "alexnet__lr=0.01__batch_size=24",
    },
}
MODEL_ORDER = ("randomforest", "conformer", "alexnet")
PREFERRED_SELECTION = {
    "3kHz": "balanced_conformer_strict_alexnet",
    "22kHz": "strict_rmse_best",
}
OUT = Path(__file__).resolve().parent


def long_path(path):
    path = Path(path).resolve()
    text = str(path)
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return Path("\\\\?\\" + text)
    return path


def read_json(path):
    with long_path(path).open(encoding="utf-8") as source:
        return json.load(source)


def read_csv(path):
    with long_path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric(row):
    result = dict(row)
    for key in (
        "r2", "rmse_all", "mae_all", "bias_all", "rmse_pre_onb",
        "bias_pre_onb", "false_positive_rate", "rmse_post_onb",
        "bias_post_onb", "recall", "rmse_onb", "mae_onb",
    ):
        if key in result and result[key] not in (None, ""):
            result[key] = float(result[key])
    if "candidate_index" in result:
        result["candidate_index"] = int(result["candidate_index"])
    return result


def source_day(sample):
    wav = str(sample["source_wav_id"])
    if wav.startswith("20250611"):
        return "2025.06.11_0.3_2"
    if wav.startswith("20250618"):
        return "2025.06.18_0.3_3"
    raise ValueError(f"Unknown source WAV: {wav}")


def regression_metrics(target, prediction, samples, thresholds):
    errors = [p - y for y, p in zip(target, prediction)]
    sample_thresholds = [float(thresholds[source_day(s)]) for s in samples]
    pre = [y < threshold for y, threshold in zip(target, sample_thresholds)]
    post = [not value for value in pre]
    near = [
        abs(y - threshold) <= abs(threshold) * 0.10
        for y, threshold in zip(target, sample_thresholds)
    ]

    def rmse(mask):
        values = [e * e for e, keep in zip(errors, mask) if keep]
        return math.sqrt(sum(values) / len(values)) if values else None

    return {
        "n_samples": len(target),
        "rmse_all": rmse([True] * len(target)),
        "rmse_onb": rmse(near),
        "recall": (
            sum(p >= t for p, t, keep in zip(prediction, sample_thresholds, post) if keep)
            / sum(post)
        ),
        "false_positive_rate": (
            sum(p >= t for p, t, keep in zip(prediction, sample_thresholds, pre) if keep)
            / sum(pre)
        ),
    }


def load_candidate_prediction(run_path, row):
    pattern = f"{int(row['candidate_index']):02d}_*.json"
    matches = list(long_path(run_path / "candidate_audits").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one audit for {pattern}, found {len(matches)}")
    audit = read_json(matches[0])
    samples = audit["internal_validation"]["samples"]
    model = row["model_key"]
    prediction = [float(sample[model]) for sample in samples]
    return samples, prediction


def same_samples(left, right):
    fields = ("source_wav_id", "chunk_index", "heat_flux")
    return len(left) == len(right) and all(
        all(a[field] == b[field] for field in fields)
        for a, b in zip(left, right)
    )


def main():
    completion_rows = []
    eligible_rows = []
    recommendation_rows = []
    ensemble_rows = []
    preferred_predictions = {}
    loaded = {}

    for frequency, run_path in RUNS.items():
        config = read_json(run_path / "parameter_search_config.json")
        selected = read_json(run_path / "selected_candidates.json")
        completed = read_json(run_path / "completed.json")
        metrics = [numeric(row) for row in read_csv(run_path / "candidate_metrics.csv")]
        by_day = [numeric(row) for row in read_csv(
            run_path / "candidate_metrics_by_source_day.csv"
        )]
        loaded[frequency] = (run_path, config, selected, metrics, by_day)
        counts = {
            model: sum(row["model_key"] == model for row in metrics)
            for model in MODEL_ORDER
        }
        completion_rows.append({
            "frequency": frequency,
            "completed": bool(completed.get("completed")),
            "outer_test_used": bool(selected["outer_test_used"]),
            "n_outer_training_samples": selected["n_outer_training_samples"],
            "n_outer_test_samples_not_scored": selected["n_outer_test_samples_not_scored"],
            "folds": selected["folds"],
            "epochs": config["run"]["epochs"],
            "selection_enabled": config["acoustic_selection"]["enabled"],
            "candidate_rows": len(metrics),
            "day_rows": len(by_day),
            "randomforest_candidates": counts["randomforest"],
            "conformer_candidates": counts["conformer"],
            "alexnet_candidates": counts["alexnet"],
        })

        row_by_name = {row["candidate_name"]: row for row in metrics}
        days_by_name = {}
        strict_names = {}
        for row in by_day:
            days_by_name.setdefault(row["candidate_name"], []).append(row)

        for model in MODEL_ORDER:
            model_rows = [row for row in metrics if row["model_key"] == model]
            best = min(row["rmse_all"] for row in model_rows)
            limit = best * 1.02
            strict = min(model_rows, key=lambda row: row["rmse_all"])
            strict_names[model] = strict["candidate_name"]
            recommended_name = RECOMMENDED[frequency][model]
            recommended = row_by_name[recommended_name]
            if recommended["rmse_all"] > limit + 1e-9:
                raise RuntimeError(f"Recommended candidate is outside 2%: {recommended_name}")
            for row in model_rows:
                if row["rmse_all"] > limit:
                    continue
                day_values = days_by_name[row["candidate_name"]]
                eligible_rows.append({
                    "frequency": frequency,
                    "model_key": model,
                    "candidate_name": row["candidate_name"],
                    "parameters_json": row["parameters_json"],
                    "rmse_all": row["rmse_all"],
                    "relative_to_best": row["rmse_all"] / best - 1.0,
                    "rmse_onb": row["rmse_onb"],
                    "recall": row["recall"],
                    "false_positive_rate": row["false_positive_rate"],
                    "worst_day_rmse_all": max(item["rmse_all"] for item in day_values),
                    "worst_day_rmse_onb": max(item["rmse_onb"] for item in day_values),
                    "minimum_day_recall": min(item["recall"] for item in day_values),
                    "maximum_day_fpr": max(item["false_positive_rate"] for item in day_values),
                    "strict_rmse_best": row is strict,
                    "recommended": row["candidate_name"] == recommended_name,
                })
            recommendation_rows.append({
                "frequency": frequency,
                "model_key": model,
                "selection_kind": "strict_rmse_best",
                "candidate_name": strict["candidate_name"],
                "parameters_json": strict["parameters_json"],
                "rmse_all": strict["rmse_all"],
                "rmse_onb": strict["rmse_onb"],
                "recall": strict["recall"],
                "false_positive_rate": strict["false_positive_rate"],
            })
            recommendation_rows.append({
                "frequency": frequency,
                "model_key": model,
                "selection_kind": "recommended_balanced",
                "candidate_name": recommended["candidate_name"],
                "parameters_json": recommended["parameters_json"],
                "rmse_all": recommended["rmse_all"],
                "rmse_onb": recommended["rmse_onb"],
                "recall": recommended["recall"],
                "false_positive_rate": recommended["false_positive_rate"],
            })

        selection_variants = [
            ("strict_rmse_best", strict_names),
            ("recommended_balanced", RECOMMENDED[frequency]),
        ]
        if frequency == "3kHz":
            selection_variants.extend([
                (
                    "balanced_conformer_strict_alexnet",
                    {**strict_names, "conformer": RECOMMENDED[frequency]["conformer"]},
                ),
                (
                    "strict_conformer_balanced_alexnet",
                    {**strict_names, "alexnet": RECOMMENDED[frequency]["alexnet"]},
                ),
            ])
        for parameter_selection, selected_names in selection_variants:
            samples_ref = None
            predictions = {}
            individual_errors = {}
            target = None
            for model in MODEL_ORDER:
                row = row_by_name[selected_names[model]]
                samples, prediction = load_candidate_prediction(run_path, row)
                if samples_ref is None:
                    samples_ref = samples
                    target = [float(sample["heat_flux"]) for sample in samples]
                elif not same_samples(samples_ref, samples):
                    raise RuntimeError(f"OOF sample order differs for {frequency}/{model}")
                predictions[model] = prediction
                mean_target = sum(target) / len(target)
                ss_res = sum((y - p) ** 2 for y, p in zip(target, prediction))
                ss_tot = sum((y - mean_target) ** 2 for y in target)
                individual_errors[model] = ss_res / ss_tot

            inverse = {model: 1.0 / max(individual_errors[model], 1e-6)
                       for model in MODEL_ORDER}
            total_inverse = sum(inverse.values())
            weights = {model: inverse[model] / total_inverse for model in MODEL_ORDER}
            performance_prediction = [
                sum(weights[model] * predictions[model][index] for model in MODEL_ORDER)
                for index in range(len(target))
            ]
            equal_prediction = [
                sum(predictions[model][index] for model in MODEL_ORDER) / len(MODEL_ORDER)
                for index in range(len(target))
            ]
            for strategy, prediction, strategy_weights in (
                ("performance_kfold", performance_prediction, weights),
                ("simple_equal", equal_prediction, {model: 1 / 3 for model in MODEL_ORDER}),
            ):
                result = regression_metrics(
                    target,
                    prediction,
                    samples_ref,
                    config["thresholds"]["by_experiment"],
                )
                ensemble_rows.append({
                    "frequency": frequency,
                    "parameter_selection": parameter_selection,
                    "strategy": strategy,
                    **result,
                    **{f"weight_{model}": strategy_weights[model] for model in MODEL_ORDER},
                })
                if (strategy == "performance_kfold"
                        and parameter_selection == PREFERRED_SELECTION[frequency]):
                    preferred_predictions[frequency] = {
                        "samples": samples_ref,
                        "target": target,
                        "prediction": prediction,
                        "thresholds": config["thresholds"]["by_experiment"],
                    }

    left = preferred_predictions["3kHz"]
    right = preferred_predictions["22kHz"]
    if not same_samples(left["samples"], right["samples"]):
        raise RuntimeError("3/22 kHz preferred OOF samples are not paired.")
    if left["target"] != right["target"]:
        raise RuntimeError("3/22 kHz preferred OOF targets differ.")
    target = left["target"]
    samples = left["samples"]
    thresholds = left["thresholds"]
    wav_ids = sorted({str(sample["source_wav_id"]) for sample in samples})
    wav_rows = []
    indices_by_wav = {
        wav: [index for index, sample in enumerate(samples)
              if str(sample["source_wav_id"]) == wav]
        for wav in wav_ids
    }

    def subset_rmse(prediction, indices, onb_only=False):
        kept = []
        for index in indices:
            threshold = float(thresholds[source_day(samples[index])])
            if onb_only and abs(target[index] - threshold) > abs(threshold) * 0.10:
                continue
            kept.append((prediction[index] - target[index]) ** 2)
        return math.sqrt(sum(kept) / len(kept)) if kept else None

    for wav in wav_ids:
        indices = indices_by_wav[wav]
        rmse_3 = subset_rmse(left["prediction"], indices)
        rmse_22 = subset_rmse(right["prediction"], indices)
        onb_3 = subset_rmse(left["prediction"], indices, onb_only=True)
        onb_22 = subset_rmse(right["prediction"], indices, onb_only=True)
        wav_rows.append({
            "source_wav_id": wav,
            "source_experiment_name": source_day(samples[indices[0]]),
            "heat_flux": target[indices[0]],
            "n_chunks": len(indices),
            "rmse_3khz": rmse_3,
            "rmse_22khz": rmse_22,
            "delta_22_minus_3": rmse_22 - rmse_3,
            "rmse_onb_3khz": onb_3,
            "rmse_onb_22khz": onb_22,
            "delta_onb_22_minus_3": (
                onb_22 - onb_3 if onb_3 is not None and onb_22 is not None else None
            ),
        })

    rng = random.Random(42)
    bootstrap = []
    for _ in range(5000):
        sampled_wavs = [rng.choice(wav_ids) for _ in wav_ids]
        sampled_indices = [
            index
            for wav in sampled_wavs
            for index in indices_by_wav[wav]
        ]
        rmse_3 = subset_rmse(left["prediction"], sampled_indices)
        rmse_22 = subset_rmse(right["prediction"], sampled_indices)
        onb_3 = subset_rmse(left["prediction"], sampled_indices, onb_only=True)
        onb_22 = subset_rmse(right["prediction"], sampled_indices, onb_only=True)
        bootstrap.append((
            rmse_22 - rmse_3,
            onb_22 - onb_3 if onb_3 is not None and onb_22 is not None else None,
        ))

    def interval(values):
        ordered = sorted(values)
        lower = ordered[int(0.025 * (len(ordered) - 1))]
        upper = ordered[int(0.975 * (len(ordered) - 1))]
        return lower, upper

    bootstrap_rows = []
    for metric, position in (("rmse_all", 0), ("rmse_onb", 1)):
        values = [item[position] for item in bootstrap if item[position] is not None]
        lower, upper = interval(values)
        bootstrap_rows.append({
            "metric": metric,
            "comparison": "22kHz_minus_3kHz",
            "point_delta": (
                regression_metrics(target, right["prediction"], samples, thresholds)[metric]
                - regression_metrics(target, left["prediction"], samples, thresholds)[metric]
            ),
            "cluster_bootstrap_replicates": len(values),
            "ci95_lower": lower,
            "ci95_upper": upper,
            "probability_22khz_better": sum(value < 0 for value in values) / len(values),
        })

    write_csv("completion_audit.csv", completion_rows)
    write_csv("eligible_candidates.csv", eligible_rows)
    write_csv("recommended_candidates.csv", recommendation_rows)
    write_csv("recommended_ensemble_oof.csv", ensemble_rows)
    write_csv("wav_frequency_comparison.csv", wav_rows)
    write_csv("cluster_bootstrap_frequency_delta.csv", bootstrap_rows)
    print("analysis complete")
    print(f"eligible rows: {len(eligible_rows)}")
    print(f"recommendation rows: {len(recommendation_rows)}")
    print(f"ensemble rows: {len(ensemble_rows)}")
    print(f"wav rows: {len(wav_rows)}")


if __name__ == "__main__":
    main()
