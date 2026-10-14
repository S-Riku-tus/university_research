"""Frozen-day evaluation must reject overlap, changed state and mispaired noise."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from utils.calculation.saved_fit_report import transfer_diagnostics, BASELINE, PRIMARY
from utils.config.onb_defaults import onb_model_specs
from utils.experiment.saved_fit_evaluation import prepare_saved_fit, validate_saved_weights, input_batches
from utils.training.fitted_artifacts import _file_digest
from utils.training.model_training import PCA_FEATURE_DECIMALS, PCA_TRANSFORM_VERSION


KEYS = ["randomforest", "conformer", "alexnet", "hgb", "extra_trees"]
STRATEGIES = ["original3_mse", "original3_hgb", "original3_extra_trees", "original3_hgb_extra_trees"]


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def dump_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample(day, q, chunk=0):
    return {"experiment_name": day, "source_wav_id": f"{day}-wav-{q}", "chunk_index": chunk,
            "chunk_start_seconds": chunk, "chunk_duration_seconds": 1, "heat_flux": q,
            "sample_filename": f"{q}_src-wav-{q}_chunk-{chunk:04}.npy"}


def fixture(root):
    source = root / "source"
    artifacts = source / "fitted_state/fold1"
    artifacts.mkdir(parents=True)
    weights = {"ensemble__"+name: {"weights": {k: .2 for k in KEYS}} for name in STRATEGIES}
    records = {}
    for name in ["scaler", "pca", "weights", *KEYS]:
        path = artifacts / name
        path.write_bytes(name.encode())
        records[name] = {"path": name, "sha256": _file_digest(path)}
    artifact = {"fit_id": "fixed", "fold": 1, "pca_transform_version": PCA_TRANSFORM_VERSION,
        "pca_feature_decimals": PCA_FEATURE_DECIMALS, "ensemble_weights": weights,
        "preprocessors": {"target_scaler": records["scaler"], "pca": records["pca"]},
        "ensemble_weights_file": records["weights"], "models": {}}
    for spec in onb_model_specs(KEYS):
        artifact["models"][spec["key"]] = {"kind": spec["kind"], "builder_params": {},
            "feature_version": spec.get("feature_version"), "artifact": records[spec["key"]]}
    dump_json(artifacts / "artifact_manifest.json", artifact)
    source_data = root / "old_data"
    old = [sample("old", q) for q in [100000, 200000, 400000]]
    dump_csv(source_data / "chunk_manifest.csv", old)
    for r in old:
        np.save(source_data / r["sample_filename"], np.ones((224, 224), np.float32))
    dump_json(source / "run_manifest.json", {"run_hash": "source_hash",
        "dataset": {"noise_dir_name": "heatflux_no_noise", "experiment_name": "old", "data_path": str(source_data), "max_freq_hz": "maxfreq=3kHz"},
        "validation_config": {"data": {"chunk_seconds": 1, "data_source_dir_by_experiment": {"old": "waterflow_1s"}},
                              "run": {"color_channel": 1}},
        "learning_context": {"training_noise": "clean_only"}})
    dump_json(source / "completed.json", {"run_hash": "source_hash", "fit_ids": ["fixed"]})
    dump_json(source / "internal_validation_fold1.json", {"samples": [old[0], old[2]]})
    dump_json(source / "split_manifest.json", {"folds": [{"fold": 1, "fit_id": "fixed", "n_training_chunks": 2, "evaluation_sample_indices": [1]}]})
    dump_csv(source / "fold_pred/pred_f1_no_noise.csv", [{**old[1], "sample_index": 1, "fold": 1,
        "y_true": old[1]["heat_flux"], **{k: 199000 for k in KEYS}}])
    jobs = []
    preprocessing = {"samplerate_hz": 44100, "chunk_seconds": 1, "max_freq_hz": 3000,
        "stft": {"resize_shape": [224, 224]}, "highpass_filter": {"fp_hz": 500}, "noise": {"scaling_mode": "fixed_global_rms"}}
    dump_json(root / "old/data/npy/waterflow_1s/maxfreq=3kHz/preprocess_manifest.json", preprocessing)
    dump_json(root / "new_data/preprocess_manifest.json", preprocessing)
    for noise in ["heatflux_no_noise", "heatflux_reference_SNR=-20"]:
        path = root / "new_data" / noise
        new = [sample("new", q) for q in [150000, 330000]]
        dump_csv(path / "chunk_manifest.csv", new)
        for r in new:
            np.save(path / r["sample_filename"], np.ones((224, 224), np.float32))
        jobs.append({"experiment_name": "new", "noise_dir_name": noise, "data_path": path})
    config = {"execution": {"source_condition_dir": "source", "source_fold": 1, "prediction_batch_size": 32, "source_probe_samples": 1},
        "learning_policy": {"evaluation_mode": "cross_day", "training_noise": "clean_only"},
        "data": {"chunk_seconds": 1, "max_freq_hz_list": ["maxfreq=3kHz"], "experiment_root": str(root)}, "run": {"color_channel": 1},
        "acoustic_selection": {"enabled": False}, "models": {"fixed_model_keys": KEYS}, "explainability": {"enabled": False},
        "ensemble": {"resolved_strategy_plan": [{"name": n} for n in STRATEGIES]}, "thresholds": {"by_experiment": {"new": 300000}}}
    return jobs, config


class SavedFitEvaluationTest(unittest.TestCase):
    def test_preflight_keeps_exact_fit_and_aligns_reordered_noise_without_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs, config = fixture(root)
            prepared = prepare_saved_fit(jobs, config, root)
            self.assertEqual(prepared["summary"]["source_training_chunks"], 2)
            self.assertEqual(prepared["summary"]["fit_id"], "fixed")
            self.assertEqual(prepared["summary"]["source_days"], ["old"])
            self.assertFalse(prepared["summary"]["training_performed"])
            self.assertEqual(prepared["weights"][PRIMARY]["hgb"], .2)
            # Hash checks run before model deserialization and before scoring.
            (root / "source/fitted_state/fold1/hgb").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare_saved_fit(jobs, config, root)

    def test_rejects_day_overlap_and_missing_noisy_chunk_before_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs, config = fixture(root)
            with patch("utils.experiment.saved_fit_evaluation.source_experiment", return_value="same-day"):
                with self.assertRaisesRegex(ValueError, "days overlap"):
                    prepare_saved_fit(jobs, config, root)
            next(jobs[1]["data_path"].glob("*.npy")).unlink()
            with self.assertRaisesRegex(ValueError, "input files and manifest differ"):
                prepare_saved_fit(jobs, config, root)

    def test_preprocessing_change_is_rejected_before_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs, config = fixture(root)
            path = root / "new_data/preprocess_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["highpass_filter"]["fp_hz"] = 1000
            dump_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "STFT/highpass preprocessing differs"):
                prepare_saved_fit(jobs, config, root)

    def test_invalid_weights_and_input_shape_are_rejected(self):
        artifact = {"ensemble_weights": {"ensemble__x": {"weights": {"a": .4, "b": .4}}}}
        with self.assertRaisesRegex(ValueError, "invalid saved weights"):
            validate_saved_weights(artifact, ["a", "b"], ["x"])
        artifact["ensemble_weights"]["ensemble__x"]["weights"] = {"a": -.1, "b": 1.1}
        with self.assertRaisesRegex(ValueError, "invalid saved weights"):
            validate_saved_weights(artifact, ["a", "b"], ["x"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            row = sample("new", 300000)
            np.save(path / row["sample_filename"], np.zeros((16, 16), np.float32))
            with self.assertRaisesRegex(ValueError, "invalid power array"):
                list(input_batches(path, [row], 2))

    def test_common_misses_ensemble_losses_corrections_and_extrapolation_stay_distinct(self):
        y = np.array([100., 300., 300., 500.])
        metadata = [sample("new", q, i) for i, q in enumerate(y)]
        predictions = {"a": np.array([100., 100., 400., 500.]), "b": np.array([100., 100., 100., 500.]),
            BASELINE: np.array([100., 100., 400., 100.]), PRIMARY: np.array([250., 100., 180., 500.])}
        training = [sample("old", 50.), sample("old", 400.)]
        stages, cases, paired = transfer_diagnostics(y, predictions, metadata, {"new": 200.}, training)
        self.assertEqual([r["primary_outcome"] for r in cases], ["false_positive", "all_base_models_negative",
            "positive_member_lost_in_ensemble", "true_positive"])
        self.assertTrue(cases[-1]["corrected_fn_vs_original3"])
        self.assertTrue(cases[2]["added_fn_vs_original3"])
        self.assertEqual(cases[-1]["training_range"], "above")
        primary = next(r for r in paired if r["model_key"] == PRIMARY and r["source_day"] == "new")
        self.assertEqual((primary["corrected_fn"], primary["added_fn"], primary["added_fp"]), (1, 1, 1))
        self.assertEqual(len(stages), len(np.unique(y))*len(predictions))


if __name__ == "__main__":
    unittest.main()
