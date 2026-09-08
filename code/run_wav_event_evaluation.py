"""Post-process saved outer-fold predictions without retraining models.

Examples
--------
One run directory::

    python code/run_wav_event_evaluation.py --run-dir <run-directory>

Every run below a result-date directory::

    python code/run_wav_event_evaluation.py --results-root <result-directory>

The experiment-threshold registry is used by default.  ``--use-saved-threshold``
exists only for reproducing the threshold stored in an old run manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from utils.calculation.wav_event_metrics import (
    load_sample_metadata_without_arrays,
    read_saved_fold_predictions,
    save_wav_event_evaluation,
)
from utils.experiment.onb_thresholds import (
    onb_threshold_by_experiment,
    onb_threshold_provenance_by_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _long_path(path):
    path = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


def _open_json(path):
    with open(_long_path(path), encoding="utf-8-sig") as source:
        return json.load(source)


def _resolve_data_path(dataset):
    saved_path = Path(dataset.get("data_path", ""))
    if os.path.isdir(_long_path(saved_path)):
        return saved_path

    # Some historical manifests contain a mojibake repository-directory name.
    # Reconstruct the path from stable experiment/dataset condition fields.
    source_name = Path(dataset.get("source_dir", "")).name
    reconstructed = (
        REPO_ROOT
        / "Pool_boiling"
        / "Subcooling_20_degrees"
        / "0.3"
        / str(dataset.get("experiment_name", ""))
        / "data"
        / "npy"
        / source_name
        / str(dataset.get("max_freq_hz", ""))
        / str(dataset.get("noise_dir_name", ""))
    )
    if os.path.isdir(_long_path(reconstructed)):
        return reconstructed
    raise FileNotFoundError(
        "Dataset path is missing both as saved and reconstructed: "
        f"saved={saved_path}, reconstructed={reconstructed}"
    )


def _find_run_manifests(root):
    manifests = []
    for current, _, filenames in os.walk(root):
        if "run_manifest.json" in filenames:
            manifests.append(Path(current) / "run_manifest.json")
    return sorted(manifests)


def _claim_status_from_manifest(manifest, model_keys):
    safe = {key: True for key in model_keys}
    notes = {key: "outer-fold prediction" for key in model_keys}
    ensemble = manifest.get("validation_config", {}).get("ensemble", {})
    grouped_inner_holdout = "inner_holdout_aggregation" in ensemble
    for strategy in ensemble.get("resolved_strategy_plan", []):
        result_key = strategy.get("result_key")
        if result_key not in safe:
            continue
        strategy_name = strategy.get("strategy")
        if not bool(strategy.get("claim_safe", True)):
            safe[result_key] = False
            notes[result_key] = "outer-validation labels were used to choose weights"
        elif strategy_name == "inner_holdout" and not grouped_inner_holdout:
            safe[result_key] = False
            notes[result_key] = (
                "legacy sample-level inner holdout; source-WAV leakage"
            )
        elif strategy_name == "inner_holdout":
            notes[result_key] = (
                "group-disjoint inner holdout with WAV-level weight metric"
            )
        else:
            notes[result_key] = "claim-safe ensemble strategy"
    return safe, notes


def process_run(
    run_dir,
    aggregations,
    primary_aggregation,
    transition_persistence_wavs=(1, 2),
    use_saved_threshold=False,
    metadata_cache=None,
):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    if not os.path.isfile(_long_path(manifest_path)):
        raise FileNotFoundError(f"run_manifest.json not found: {manifest_path}")
    manifest = _open_json(manifest_path)
    dataset = manifest.get("dataset", {})
    experiment_name = dataset.get("experiment_name")
    data_path = _resolve_data_path(dataset)
    snr_value = dataset.get("snr_value")

    registry = onb_threshold_by_experiment()
    provenance_registry = onb_threshold_provenance_by_experiment()
    if use_saved_threshold:
        threshold = dataset.get("threshold")
        threshold_source = "saved_run_manifest"
        threshold_provenance = {"source": "saved_run_manifest"}
    else:
        if experiment_name not in registry:
            raise KeyError(
                f"No registered ONB threshold for experiment={experiment_name!r}."
            )
        threshold = registry[experiment_name]
        threshold_source = "current_experiment_threshold_registry"
        threshold_provenance = provenance_registry[experiment_name]

    cache = metadata_cache if metadata_cache is not None else {}
    cache_key = str(data_path.resolve())
    if cache_key not in cache:
        cache[cache_key] = load_sample_metadata_without_arrays(data_path)
    chunk_rows, model_keys = read_saved_fold_predictions(
        run_dir / "fold_pred",
        snr_value,
        cache[cache_key],
    )
    claim_safe, claim_notes = _claim_status_from_manifest(manifest, model_keys)
    result = save_wav_event_evaluation(
        save_path=run_dir,
        snr_value=snr_value,
        chunk_rows=chunk_rows,
        model_keys=model_keys,
        threshold=threshold,
        band_frac=float(
            manifest.get("validation_config", {})
            .get("thresholds", {})
            .get("onb_band_frac", 0.10)
        ),
        aggregations=aggregations,
        primary_aggregation=primary_aggregation,
        save_predicted_event_summary=True,
        onb_transition_persistence_wavs=transition_persistence_wavs,
        claim_safe_by_model=claim_safe,
        claim_note_by_model=claim_notes,
        threshold_provenance=threshold_provenance,
    )
    evaluation_manifest_path = Path(result["manifest_path"])
    evaluation_manifest = _open_json(evaluation_manifest_path)
    evaluation_manifest.update({
        "source_run_manifest": str(manifest_path),
        "threshold_source": threshold_source,
        "experiment_name": experiment_name,
    })
    with open(_long_path(evaluation_manifest_path), "w", encoding="utf-8") as output:
        json.dump(evaluation_manifest, output, ensure_ascii=False, indent=2)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create pooled OOF WAV/event summaries from saved fold predictions."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", type=Path)
    target.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=["mean", "median", "p90"],
        choices=["mean", "median", "p90", "p95"],
    )
    parser.add_argument(
        "--primary-aggregation",
        default="median",
        choices=["mean", "median", "p90", "p95"],
    )
    parser.add_argument(
        "--use-saved-threshold",
        action="store_true",
        help="Reproduce an old manifest threshold instead of the corrected registry.",
    )
    parser.add_argument(
        "--transition-persistence-wavs",
        nargs="+",
        type=int,
        default=[1, 2],
        help=(
            "Consecutive threshold-positive WAV counts used to define an ONB "
            "transition (default: 1 2)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.primary_aggregation not in args.aggregations:
        raise ValueError("--primary-aggregation must be listed in --aggregations.")
    if any(value <= 0 for value in args.transition_persistence_wavs):
        raise ValueError("--transition-persistence-wavs values must be positive.")
    if args.run_dir is not None:
        run_directories = [args.run_dir]
    else:
        run_directories = [
            path.parent for path in _find_run_manifests(args.results_root)
        ]
    if not run_directories:
        raise FileNotFoundError("No run_manifest.json files were found.")

    metadata_cache = {}
    completed = 0
    skipped = []
    for index, run_dir in enumerate(run_directories, start=1):
        try:
            result = process_run(
                run_dir,
                aggregations=args.aggregations,
                primary_aggregation=args.primary_aggregation,
                transition_persistence_wavs=args.transition_persistence_wavs,
                use_saved_threshold=args.use_saved_threshold,
                metadata_cache=metadata_cache,
            )
        except (FileNotFoundError, KeyError, ValueError, OSError, csv.Error) as exc:
            skipped.append((str(run_dir), str(exc)))
            print(f"[{index}/{len(run_directories)}] skipped: {run_dir} | {exc}")
            continue
        completed += 1
        print(
            f"[{index}/{len(run_directories)}] saved: "
            f"{result['metrics_path']} | "
            f"chunks={result['n_chunk_rows']} "
            f"wavs={len(result['wav_rows'])}"
        )

    print(f"completed={completed} skipped={len(skipped)}")
    if completed == 0:
        raise RuntimeError("No run directories could be post-processed.")


if __name__ == "__main__":
    main()
