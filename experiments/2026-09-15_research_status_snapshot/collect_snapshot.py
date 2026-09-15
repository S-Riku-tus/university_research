"""Read saved runs; export a new evidence snapshot without training or changing runs.

Usage: python collect_snapshot.py --output <new empty directory>
The September 14 runs are fixed evidence. Later runs are inventoried only.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ("2025.06.11_0.3_2", "2025.06.18_0.3_3", "2025.07.09_0.3_1")
CURRENT_KEYS = {"rf", "cnntf_v2_gap", "alexnet", "ensemble__simple_equal", "ensemble__inner_holdout"}


def long_path(path):
    value = str(path.absolute())
    return Path("\\\\?\\" + value) if os.name == "nt" and not value.startswith("\\\\?\\") else Path(value)


def relative(path):
    value = str(path)
    return Path(value[4:] if value.startswith("\\\\?\\") else value).relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.csv")) or (output / "snapshot_manifest.json").exists():
        raise SystemExit("Use a new output directory; an existing snapshot must remain fixed.")
    sources, counts, verification = {}, {}, []

    def read(path):
        data = path.read_bytes()
        sources[relative(path)] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data.decode("utf-8-sig")

    def rows(path):
        return list(csv.DictReader(io.StringIO(read(path))))

    def write(name, data):
        columns = list(dict.fromkeys(k for row in data for k in row))
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
        counts[name] = len(data)

    inventory, metrics, transitions, xai_status = [], [], [], []
    xai = {"ig_diagnostics.csv": [], "group_mask_performance.csv": [],
           "randomization_sanity.csv": [], "input_stability.csv": [], "treeshap_summary.csv": []}
    names = {"explainability_summary.csv": "ig_diagnostics.csv",
             "group_mask_performance.csv": "group_mask_performance.csv",
             "top_layer_randomization_sanity.csv": "randomization_sanity.csv",
             "input_stability.csv": "input_stability.csv", "treeshap_pca_summary.csv": "treeshap_summary.csv"}
    for experiment in EXPERIMENTS:
        base = long_path(ROOT / "Pool_boiling/Subcooling_20_degrees/0.3" / experiment / "regression_result/npy/ensemble")
        for run_root in sorted(base.glob("202609*_selected_log_architecture*")):
            for path in sorted(run_root.glob("*/*/*/run_manifest.json")):
                manifest = json.loads(read(path))
                dataset, directory = manifest["dataset"], path.parent
                context = {"run": run_root.name, "experiment": experiment,
                           "maxfreq": dataset["max_freq_hz"], "snr": dataset["snr_value"]}
                metric_path = directory / "wav_eval" / f"wav_metrics_{context['snr']}.csv"
                chunk_path = directory / f"metrics_summary_{context['snr']}.csv"
                marker = directory / "completed.json"
                new_schema = bool(manifest.get("execution_schema_version"))
                complete = metric_path.exists() and chunk_path.exists() and (marker.exists() or not new_schema)
                state = "saved_metrics_present" if complete else "incomplete_at_snapshot"
                if not new_schema and chunk_path.exists() and not metric_path.exists():
                    state = "legacy_chunk_metrics_only"
                inventory.append({**context, "created_at": manifest["created_at"], "run_hash": manifest["run_hash"],
                                  "completion": state,
                                  "completion_marker": marker.exists(), "new_schema": new_schema,
                                  "wav_metrics_present": metric_path.exists(),
                                  "fold_prediction_files": len(list((directory / "fold_pred").glob("*.csv"))),
                                  "run_manifest": relative(path)})
                if run_root.name != "20260914_selected_log_architecture" or not complete:
                    continue
                current = [{**context, **r, "current_strategy": r["model_key"] in CURRENT_KEYS,
                            "source_csv": relative(metric_path)} for r in rows(metric_path) if r["aggregation"] == "median"]
                metrics.extend(current)
                eval_path = directory / "wav_eval" / f"evaluation_manifest_{context['snr']}.json"
                threshold = float(json.loads(read(eval_path))["threshold"])
                pred_path = directory / "wav_eval" / f"wav_predictions_{context['snr']}.csv"
                predictions = rows(pred_path)
                truth = [float(r["y_true"]) for r in predictions]
                for metric in current:
                    key = metric["model_key"]
                    pred = [float(r[f"{key}_pred_median"]) for r in predictions]
                    mse = statistics.mean((a-b)**2 for a, b in zip(truth, pred))
                    r2 = 1 - mse / statistics.mean((a-statistics.mean(truth))**2 for a in truth)
                    recall = sum(a >= threshold and b >= threshold for a,b in zip(truth,pred))/sum(a >= threshold for a in truth)
                    if not (abs(r2-float(metric["r2"])) < 1e-10 and abs(math.sqrt(mse)-float(metric["rmse_all"])) < 1e-5 and abs(recall-float(metric["recall"])) < 1e-10):
                        raise ValueError(f"Saved metrics mismatch: {context}, {key}")
                    verification.append({**context, "model": key, "r2_rmse_recall_match": True})
                for q in (directory / "wav_eval").glob("onb_transition_summary_*.csv"):
                    transitions.extend({**context, **r, "source_csv": relative(q)} for r in rows(q) if r["aggregation"] == "median")
                for fold in sorted((directory / "explainability").glob("fold*")):
                    for model in sorted(fold.iterdir()):
                        if not model.is_dir():
                            continue
                        extra = {**context, "fold": fold.name, "model": model.name}
                        status = {**extra}
                        for source_name, dest in names.items():
                            q = model / source_name
                            status[source_name] = q.exists()
                            if q.exists():
                                values = rows(q)
                                if source_name == "explainability_summary.csv":
                                    values = [r for r in values if r["method"] == "integrated_gradients"]
                                xai[dest].extend({**extra, **r, "source_csv": relative(q)} for r in values)
                        xai_status.append(status)
    write("run_inventory.csv", inventory)
    write("wav_median_metrics.csv", metrics)
    write("onb_transitions_median.csv", transitions)
    write("xai_output_inventory.csv", xai_status)
    for name, values in xai.items():
        write(name, values)
    snapshot = {"collected_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
                "row_counts": counts, "verification": verification,
                "scope": "September 14 six-run numeric evidence; September run inventory at collection time. No training.",
                "caveats": ["Retired prediction_max values are retained as historical evidence and marked current_strategy=False.",
                            "File presence does not establish scientific validity. Incomplete runs may later finish.",
                            "XAI mask metrics use WAV medians within folds; they are not pooled OOF R2."]}
    (output / "snapshot_manifest.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
