"""Extract saved research-status evidence without training or modifying source runs."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = ("2025.06.11_0.3_2", "2025.06.18_0.3_3", "2025.07.09_0.3_1")
RUNS = ("20260903_selected_log_architecture", "20260908_selected_log_architecture")
SOURCES = {}


def long_path(path):
    value = str(Path(path).resolve())
    return Path("\\\\?\\" + value) if os.name == "nt" and not value.startswith("\\\\?\\") else Path(value)


def relative(path):
    value = str(path)
    if value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value).relative_to(ROOT).as_posix()


def read_bytes(path):
    data = path.read_bytes()
    SOURCES[relative(path)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return data


def read_csv(path):
    return list(csv.DictReader(read_bytes(path).decode("utf-8-sig").splitlines()))


def read_json(path):
    return json.loads(read_bytes(path).decode("utf-8-sig"))


def write_csv(name, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    inventory, metrics, transitions, masks, ig, sanity = [], [], [], [], [], []
    verification = []
    for experiment in EXPERIMENTS:
        for run in RUNS:
            root = long_path(ROOT / "Pool_boiling/Subcooling_20_degrees/0.3" / experiment / "regression_result/npy/ensemble" / run)
            for manifest_path in sorted(root.glob("*/*/*/run_manifest.json")):
                manifest = read_json(manifest_path)
                dataset = manifest["dataset"]
                context = {"run": run, "experiment": experiment, "maxfreq": dataset["max_freq_hz"], "snr": dataset["snr_value"]}
                directory = manifest_path.parent
                eval_paths = list((directory / "wav_eval").glob("evaluation_manifest_*.json"))
                eval_manifest = read_json(eval_paths[0]) if eval_paths else {}
                inventory.append({**context, "created_at": manifest["created_at"], "epochs_requested": manifest["validation_config"]["run"]["epochs"], "folds": manifest["validation_config"]["run"]["folds"], "n_wavs": eval_manifest.get("n_wavs"), "n_chunks": eval_manifest.get("n_oof_chunks"), "original_threshold": dataset["threshold"], "evaluated_threshold": eval_manifest.get("threshold"), "run_manifest": relative(manifest_path), "wav_evaluation_present": bool(eval_paths)})
                current_metrics = []
                for path in (directory / "wav_eval").glob("wav_metrics_*.csv"):
                    current_metrics += [{**context, **row, "source_csv": relative(path)} for row in read_csv(path) if row["aggregation"] == "median"]
                metrics += current_metrics
                for path in (directory / "wav_eval").glob("onb_transition_summary_*.csv"):
                    transitions += [{**context, **row, "source_csv": relative(path)} for row in read_csv(path) if row["aggregation"] == "median"]
                if run != RUNS[-1]:
                    continue
                for path in (directory / "wav_eval").glob("wav_predictions_*.csv"):
                    predictions = read_csv(path)
                    for metric in current_metrics:
                        model = metric["model_key"]
                        truth = [float(row["y_true"]) for row in predictions]
                        pred = [float(row[f"{model}_pred_median"]) for row in predictions]
                        squared_error = sum((a - b) ** 2 for a, b in zip(truth, pred))
                        mean_truth = statistics.mean(truth)
                        r2 = 1 - squared_error / sum((a - mean_truth) ** 2 for a in truth)
                        rmse = math.sqrt(squared_error / len(truth))
                        threshold = float(eval_manifest["threshold"])
                        tp = sum(a >= threshold and b >= threshold for a, b in zip(truth, pred))
                        positive = sum(a >= threshold for a in truth)
                        recall = tp / positive
                        assert abs(r2 - float(metric["r2"])) < 1e-10
                        assert abs(rmse - float(metric["rmse_all"])) < 1e-5
                        assert abs(recall - float(metric["recall"])) < 1e-10
                        verification.append({**context, "model": model, "n_wavs": len(truth), "r2_rmse_recall_recomputed": True})
                for path in (directory / "explainability").glob("fold*/*/*.csv"):
                    extra = {**context, "fold": path.parent.parent.name, "model": path.parent.name, "source_csv": relative(path)}
                    if path.name == "group_mask_performance.csv":
                        masks += [{**extra, **row} for row in read_csv(path)]
                    elif path.name == "explainability_summary.csv":
                        ig += [{**extra, **row} for row in read_csv(path) if row["method"] == "integrated_gradients"]
                    elif path.name == "top_layer_randomization_sanity.csv":
                        sanity += [{**extra, **row} for row in read_csv(path)]
    for name, rows in (("run_inventory.csv", inventory), ("wav_median_metrics.csv", metrics), ("onb_transitions_median.csv", transitions), ("latest_group_mask_performance.csv", masks), ("latest_ig_diagnostics.csv", ig), ("latest_randomization_sanity.csv", sanity)):
        write_csv(name, rows)

    aggregate = []
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row["run"], row["experiment"], row["model_key"], row["claim_safe"])].append(row)
    for key, rows in sorted(grouped.items()):
        aggregate.append({"run": key[0], "experiment": key[1], "model": key[2], "claim_safe": key[3], "conditions": len(rows), **{f"mean_{metric}": statistics.mean(float(row[metric]) for row in rows) for metric in ("r2", "rmse_all", "recall", "precision")}})
    write_csv("metrics_condition_means.csv", aggregate)
    summary = {"collected_at_utc": datetime.now(timezone.utc).isoformat(), "sources": SOURCES, "verification": verification, "row_counts": {"runs": len(inventory), "median_metrics": len(metrics), "median_transitions": len(transitions), "latest_mask_rows": len(masks), "latest_ig_rows": len(ig)}, "caveat": "Condition averages are descriptive, not independent replicates. claim_safe is the saved weighting/split audit flag, not proof of unseen-day validity or unbiased model selection."}
    (OUTPUT / "snapshot_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["row_counts"]))
    print("Recomputed latest R2, RMSE and recall:", len(verification), "model-condition rows")
    for row in aggregate:
        print(row["run"][:8], row["experiment"], row["model"], row["conditions"], "R2", round(row["mean_r2"], 4), "safe", row["claim_safe"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
