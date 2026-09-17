"""Audit the completed spectra, all 105 manifests and real-data smoke outputs."""
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np

from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.experiment.learning_policy import checked_metadata, sample_key
from utils.experiment.onb_thresholds import onb_threshold_by_experiment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/2026-09-16_day_split_spectral_selection"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as inp:
        return list(csv.DictReader(inp))


def main():
    base = ROOT / "Pool_boiling/Subcooling_20_degrees/0.3"
    cfg = {"enabled": True, "features_csv": str(OUT / "spectral_features.csv"),
           "feature": "band_2000_3000_db", "background_quantile": .99}
    selector = AcousticTrainingSelector(cfg, onb_threshold_by_experiment())
    dataset_records = []
    for day in onb_threshold_by_experiment():
        reference_keys = None
        for path in sorted((base / day / "data/npy/waterflow_20260817_1s").glob("*/*/chunk_manifest.csv")):
            metadata = checked_metadata(read_csv(path), day)
            retained, audit = selector.select(metadata)
            keys = {sample_key(metadata[i]) for i in retained}
            if reference_keys is None:
                reference_keys = keys
            assert keys == reference_keys
            dataset_records.append({"day": day, "path": str(path.relative_to(ROOT)),
                                    "n_samples": len(metadata), "same_clean_selection_n": len(keys),
                                    "applied_in_training": day != "2025.06.18_0.3_3"})
        images = list((base / day / "data/spectrum_png/waterflow_20260817_1s").glob("*/*.png"))
        expected = 780 if day.startswith("2025.07.09") else 1080
        assert len(images) == expected, (day, len(images))
    assert len(dataset_records) == 105
    smoke_base = base / "2025.06.18_0.3_3/regression_result/npy/day_split_smoke"
    if os.name == "nt":
        smoke_base = Path("\\\\?\\" + str(smoke_base))
    runs = []
    for path in smoke_base.rglob("completed.json"):
        directory = path.parent
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["validation_config"].get("acoustic_selection", {}).get("mode") == "peak_height":
            continue  # Later peak-height runs are audited by audit_peak_height_outputs.py.
        strategies = manifest["validation_config"]["ensemble"]["strategies"]
        if not any(s["strategy"] == "performance_kfold" for s in strategies):
            continue  # development run before the final strategy type was named
        selected = manifest["validation_config"]["acoustic_selection"]["enabled"]
        split = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))["folds"][0]
        assert split["n_evaluation_chunks"] == 1080
        assert len(split["evaluation_wav_groups"]) == 18
        assert not set(split["training_wav_groups"]) & set(split["evaluation_wav_groups"])
        train_days = {json.loads(g)[0] for g in split["training_wav_groups"]}
        assert train_days == {"2025.06.11_0.3_2", "2025.07.09_0.3_1"}
        assert split["n_training_chunks"] == (1648 if selected else 1860)
        internal = json.loads((directory / "internal_validation_fold1.json").read_text(encoding="utf-8"))
        assert len(internal["samples"]) == 1860
        seen = [i for fold in internal["folds"] for i in fold["validation_indices"]]
        assert sorted(seen) == list(range(1860))
        predictions = read_csv(directory / "fold_pred/pred_f1_no_noise.csv")
        assert len(predictions) == 1080
        keys = ["randomforest", "conformer", "alexnet", "ensemble__performance_kfold"]
        assert all(np.isfinite(float(row[key])) for row in predictions for key in keys)
        weights = read_csv(directory / "ensemble_weights_no_noise.csv")[0]
        expected_weights = np.asarray([1 / max(internal["individual_errors"][key], 1e-6) for key in keys[:3]])
        expected_weights /= expected_weights.sum()
        np.testing.assert_allclose([float(weights[key]) for key in keys[:3]], expected_weights)
        record = {"selection_enabled": selected, "run_hash": manifest["run_hash"],
                  "path": str(directory), "n_train": split["n_training_chunks"], "n_test": 1080,
                  "test_wavs": 18, "finite_predictions_all_models": True,
                  "weights_recomputed_from_training_oof": True,
                  "internal_cv_shared_wavs": [f["shared_source_wavs"] for f in internal["folds"]],
                  "scope": "2-epoch integration verification, not scientific model ranking"}
        runs.append(record)
        snapshot = OUT / "smoke_snapshot" / ("selected" if selected else "baseline")
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in ["completed.json", "run_manifest.json", "split_manifest.json", "training_selection_fold1.json",
                     "internal_validation_fold1.json", "ensemble_weights_no_noise.csv", "metrics_summary_no_noise.csv"]:
            shutil.copyfile(directory / name, snapshot / name)
        shutil.copyfile(directory / "wav_eval/wav_metrics_no_noise.csv", snapshot / "wav_metrics_no_noise.csv")
        shutil.copyfile(directory / "fold_pred/pred_f1_no_noise.csv", snapshot / "test_predictions.csv")
    assert len(runs) == 2
    result = {"all_seconds_image_count": 2940, "dataset_manifest_count": 105,
              "same_clean_selection_across_all_variants": True, "runs": runs, "datasets": dataset_records}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifests": 105, "images": 2940, "runs": runs}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
