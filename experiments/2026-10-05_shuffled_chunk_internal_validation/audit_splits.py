"""Audit the configured chunk partitions against saved outer IDs; load no arrays."""

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from utils.calculation.prediction_records import load_sample_metadata_without_arrays
from utils.experiment.learning_policy import checked_metadata, outer_splits, sample_key, wav_groups
from utils.training.internal_validation import internal_splits


def read_json(path):
    path = Path("\\\\?\\" + str(path.resolve())) if os.name == "nt" else path
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(filename, rows):
    with (OUTPUT / filename).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    tree = ast.parse((ROOT / "code/run_ensemble_regression_onb.py").read_text(encoding="utf-8-sig"))
    config_node = next(node.value.args[0] for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "VALIDATION_CONFIG" for target in node.targets))
    sections = {key.value: value for key, value in zip(config_node.keys, config_node.values)}
    run = ast.literal_eval(sections["run"])
    policy = ast.literal_eval(sections["learning_policy"])
    assert run["internal_validation_split"] == "chunk_kfold"
    assert policy["evaluation_mode"] == "within_wav_chunk"
    scope = read_json(ROOT / "experiments/2026-10-03_tuned_within_wav_matched_comparison/run_scope.json")
    source_dir = ROOT / scope["runs"]["clean_only"] / "maxfreq=3kHz" / "heatflux_no_noise"
    manifest = read_json(source_dir / "run_manifest.json")
    saved_outer = read_json(source_dir / "split_manifest.json")["folds"][0]
    dataset = manifest["dataset"]
    metadata = checked_metadata(load_sample_metadata_without_arrays(dataset["data_path"]), dataset["experiment_name"])
    fit, test = outer_splits(metadata, metadata, policy)[0]
    assert len(metadata) == 2160 and len(fit) == 1620 and len(test) == 540
    assert set(test) == set(saved_outer["evaluation_sample_indices"])
    fit_keys, test_keys = {sample_key(metadata[i]) for i in fit}, {sample_key(metadata[i]) for i in test}
    assert fit_keys.isdisjoint(test_keys)
    training = [metadata[int(i)] for i in fit]
    groups = wav_groups(training)
    unique_groups = np.unique(groups)
    assert len(unique_groups) == 36
    labels = np.asarray([float(row["sample_filename"].split("_")[0]) for row in training])
    thresholds = {day[:10].replace(".", ""): value for day, value in scope["thresholds"].items()}
    summaries, wav_rows, membership = [], [], []
    coverage = np.zeros(len(training), int)
    for fold, (inner_fit, held) in enumerate(internal_splits(training, run["folds"], run["random_seed"],
                                                          run["internal_validation_split"]), 1):
        assert set(inner_fit).isdisjoint(held)
        assert {sample_key(training[i]) for i in inner_fit}.isdisjoint(test_keys)
        assert {sample_key(training[i]) for i in held}.isdisjoint(test_keys)
        coverage[held] += 1
        counts = [int((groups[inner_fit] == group).sum()) for group in unique_groups]
        assert min(counts) > 0
        summaries.append({"fold": fold, "fit_chunks": len(inner_fit), "validation_chunks": len(held),
            "shared_samples": 0, "shared_source_wavs": len(set(groups[inner_fit]) & set(groups[held])),
            "fit_wavs": len(set(groups[inner_fit])), "validation_wavs": len(set(groups[held])),
            "fit_chunks_per_wav_min": min(counts), "fit_chunks_per_wav_max": max(counts)})
        for group in unique_groups:
            positions = np.flatnonzero(groups == group)
            row = training[int(positions[0])]
            day = row.get("source_experiment_name", row["source_wav_id"][:8])
            day_key = day[:10].replace(".", "")
            threshold = thresholds[day_key]
            q = float(labels[positions[0]])
            wav_rows.append({"fold": fold, "source_wav_id": row["source_wav_id"], "source_day": day,
                "heat_flux_W_m2": q, "onb_W_m2": threshold,
                "onb_neighborhood": .9 <= q/threshold <= 1.1,
                "fit_chunks": int((groups[inner_fit] == group).sum()),
                "validation_chunks": int((groups[held] == group).sum())})
        for role, indices in [("fit", inner_fit), ("validation", held)]:
            membership.extend({"fold": fold, "role": role, "outer_training_index": int(i),
                "source_wav_id": training[i]["source_wav_id"], "chunk_index": training[i]["chunk_index"]} for i in indices)
    assert np.all(coverage == 1)
    write_csv("internal_fold_summary.csv", summaries)
    write_csv("heat_flux_support.csv", wav_rows)
    write_csv("internal_chunk_membership.csv", membership)
    audit = {"config": "configs/experiments/2026-10-05_shuffled_chunk_internal_validation.json",
        "effective_internal_method": run["internal_validation_split"], "shuffle": True,
        "random_seed": run["random_seed"], "folds": run["folds"], "metadata_only": True,
        "source_run": scope["runs"]["clean_only"],
        "outer_train": len(fit), "outer_test": len(test), "source_wavs": len(unique_groups),
        "outer_test_ids_identical_to_20261001": True, "outer_shared_samples": 0,
        "all_internal_fits_and_validations_exclude_outer_test": True,
        "all_internal_fits_retain_all_36_wavs": True,
        "internal_held_out_coverage_once_per_chunk": True,
        "training_identity_sha256": hashlib.sha256(json.dumps(sorted(fit_keys)).encode()).hexdigest(),
        "summaries": summaries, "research_models_trained": False}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ["effective_internal_method", "outer_train", "outer_test",
        "outer_test_ids_identical_to_20261001", "all_internal_fits_retain_all_36_wavs"]}, indent=2))
    print(summaries)


if __name__ == "__main__":
    main()
