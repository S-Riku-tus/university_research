"""Redraw completed ONB metric summaries, without importing models or training.

Pass one explicit condition directory, frequency directory, or study directory.
All selected conditions must be completed before any plots are written.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.experiment.run_helpers import open_text, windows_long_path
from utils.plotting.regression_plots import RegressionPlotter


METRICS = (("r2", "R2 Score"), ("roc_auc_cont", "ROC-AUC (continuous)"),
           ("pr_auc_cont", "PR-AUC (continuous)"))


def _read_json(path):
    with open_text(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def replot_completed_bars(run_dir, output_dir=None):
    root = Path(windows_long_path(Path(run_dir).resolve()))
    if not root.is_dir():
        raise ValueError(f"Result directory does not exist: {root}")
    manifests = ([root / "run_manifest.json"] if (root / "run_manifest.json").is_file()
                 else sorted(set(root.glob("*/run_manifest.json"))
                             | set(root.glob("*/*/run_manifest.json"))))
    if not manifests:
        raise ValueError("No result manifests found in the explicitly selected directory")
    sources = []
    for path in manifests:
        manifest = _read_json(path)
        marker = path.parent / "completed.json"
        if not marker.is_file():
            raise ValueError(f"Condition has not completed; redraw postponed: {path.parent}")
        completion = _read_json(marker)
        if not manifest.get("run_hash") or completion.get("run_hash") != manifest["run_hash"]:
            raise ValueError(f"Completion record does not match the result manifest: {path.parent}")
        snr = manifest["dataset"]["snr_value"]
        summary = path.parent / f"metrics_summary_{snr}.csv"
        with open_text(summary, "r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Empty metric summary: {summary}")
        for row in rows:
            for metric, _ in METRICS:
                float(row[f"{metric}_mean"])
                float(row[f"{metric}_se"])
        sources.append((path, manifest, summary, rows))
    if not (root / "run_manifest.json").is_file():
        # At study/frequency scope, a condition not yet created by the running
        # job must not disappear silently from the redraw selection.
        observed = {(manifest["dataset"].get("max_freq_hz"),
                     manifest["dataset"].get("noise_dir_name"))
                    for _, manifest, _, _ in sources}
        for _, manifest, _, _ in sources:
            frequency = manifest["dataset"].get("max_freq_hz")
            expected = manifest["validation_config"].get("data", {}).get("noise_dir_names", [])
            missing = [noise for noise in expected if (frequency, noise) not in observed]
            if missing:
                raise ValueError(f"Configured conditions are not all present; redraw postponed: {missing}")
    destination = Path(windows_long_path(Path(output_dir).resolve())) if output_dir else (
        root / "bar_redraw" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if any(destination == path.parent for path, _, _, _ in sources):
        raise ValueError("Choose a separate redraw output directory to preserve original graphs")
    files, records = [], []
    with plt.rc_context({"font.family": "Times New Roman", "xtick.direction": "in",
                         "ytick.direction": "in"}):
        plotter = RegressionPlotter()
        for path, manifest, summary, rows in sources:
            output = destination / path.parent.relative_to(root)
            for metric, title in METRICS:
                files.extend(plotter.plot_bar(
                    title, [row["model"] for row in rows],
                    [float(row[f"{metric}_mean"]) for row in rows],
                    [float(row[f"{metric}_se"]) for row in rows],
                    manifest["validation_config"]["run"]["epochs"], output,
                    manifest["dataset"]["snr_value"]))
            with open_text(summary, "rb") as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
            records.append({"summary": str(summary), "sha256": digest,
                            "run_hash": manifest["run_hash"]})
    record = {"source_directory": str(root), "output_directory": str(destination),
              "sources": records, "new_fits": 0, "new_model_inference": 0,
              "aggregation": "Saved CSV means and SE, without recalculation",
              "files": [str(Path(path).relative_to(destination)) for path in files]}
    with open_text(destination / "redraw_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Explicit completed result directory")
    parser.add_argument("--output-dir", help="Optional separate output directory")
    args = parser.parse_args()
    try:
        record = replot_completed_bars(args.run_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({"conditions": len(record["sources"]), "plots": len(record["files"]),
                      "output_directory": record["output_directory"], "new_fits": 0,
                      "new_model_inference": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
