"""Execute the actual ONB main with all five production models at one epoch.

144 existing clean chunks (four per WAV) and their minus20 counterparts are
used only for software validation. Results are retained outside temporary
inputs, distinctly marked smoke; they are not research performance results.
"""
import argparse
import csv
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import run_ensemble_regression_onb as onb
from run_hgb_complementarity_validation import read_json, save_json

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/2026-10-06_hgb_followup_validation"


class QuietPlotter:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def verify_smoke(execution_id):
    # The standard runner scopes the study directory beside save_base_path,
    # so resolve it from its manifest rather than assuming the input base.
    manifests = [p for p in OUT.rglob("run_manifest.json")
                 if read_json(p).get("run_instance_id")==execution_id]
    completed = [p.parent/"completed.json" for p in manifests]
    assert len(completed)==2 and all(p.exists() for p in completed)
    artifacts = list(manifests[0].parent.parent.rglob("artifact_manifest.json"))
    assert len(artifacts)==1
    saved = read_json(artifacts[0])
    assert set(saved["models"])==set(onb.MODEL_KEYS)
    assert all(m["verification"]["passed"] for m in saved["models"].values())
    primary = saved["ensemble_weights"]["ensemble__original3_hgb_extra_trees"]["weights"]
    assert all(primary[k]>0 for k in onb.MODEL_KEYS)
    for marker in completed:
        path = marker.parent
        assert (path / "metrics_by_source_day.csv").exists()
        assert (path / "metrics_source_day_deltas.csv").exists()
        assert (path / "ensemble_training_oof_fit_f1.json").exists()
        with (path / "metrics_by_source_day.csv").open(encoding="utf-8-sig") as stream:
            metric_rows = list(csv.DictReader(stream))
        assert len(metric_rows)==33
        assert {r["model_key"] for r in metric_rows if r["source_day"]=="pooled_source_day_thresholds"}==set(onb.MODEL_KEYS+onb.ENSEMBLE_MANAGER.create_run(onb.MODEL_SPECS).result_keys)
    save_json(OUT / "main_smoke_verification.json", {"status": "passed", "actual_onb_main_called": True,
        "execution_id": execution_id, "models": onb.MODEL_KEYS, "original_architectures_used": True, "epochs_per_cnn": 1,
        "inner_folds": 2, "source_wavs": 36, "input_chunks": 144, "outer_fit_chunks": 108, "outer_eval_chunks": 36,
        "noise_conditions": ["clean", "-20"], "base_fits": 15, "persisted_final_models": 5,
        "persisted_model_reload_verified": True, "source_day_metric_rows_per_condition": 33,
        "prediction_columns": 11, "research_performance_claim": False,
        "normal_150_epoch_run_executed": False,
        "results": [str(p.parent.relative_to(ROOT)) for p in completed]})
    print("[main smoke] actual five-model ONB main, OOF weights, both metric bases, persistence and two-noise evaluation passed")


def main():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Actual CNN smoke validation requires the existing GPU")
    manifest = read_json(SOURCE / "seed43/manifest.json")
    by_wav = {}
    for i, row in enumerate(manifest["samples"]):
        by_wav.setdefault(row["source_wav_id"], []).append(i)
    selected = [i for wav in sorted(by_wav) for i in by_wav[wav][:4]]
    assert len(selected)==144 and len(by_wav)==36
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke_inputs_", dir=OUT) as temporary:
        input_root = Path(temporary).resolve()
        # Verify the absolute target before TemporaryDirectory's recursive cleanup.
        assert input_root.parent == OUT.resolve()
        source = input_root / "source"
        jobs = []
        for noise, directory in [("clean", "heatflux_no_noise"), ("-20", "heatflux_reference_SNR=-20")]:
            data = source / "maxfreq=3kHz" / directory
            data.mkdir(parents=True)
            raw = np.load(SOURCE / f"cache/raw_{noise}.npy", mmap_mode="r")
            rows = [manifest["samples"][i] for i in selected]
            for i, row in zip(selected, rows):
                np.save(data / row["sample_filename"], np.asarray(raw[i]))
            with (data / "chunk_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            name = onb.EXPERIMENT_DIR_NAMES[0]
            jobs.append({"experiment_name": name, "source_dir": source, "data_path": data,
                "experiment_root": OUT, "max_freq_hz": "maxfreq=3kHz", "noise_dir_name": directory,
                "snr_value": "no_noise" if noise=="clean" else noise,
                "threshold": onb.THRESHOLD_BY_EXPERIMENT[name], "save_base_path": OUT / "smoke_results"})
        with patch.object(onb, "SMOKE_TEST", True), patch.object(onb, "EPOCH_NUM", 1), \
             patch.object(onb, "DIVISIONS", 2), patch.object(onb, "NOISE_DIR_NAMES", [j["noise_dir_name"] for j in jobs]), \
             patch.object(onb, "build_dataset_jobs", return_value=jobs), patch.object(onb, "RegressionPlotter", QuietPlotter):
            onb.main()
        verify_smoke(onb.EXECUTION_ID)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-execution-id")
    args = parser.parse_args()
    if args.verify_execution_id:
        verify_smoke(args.verify_execution_id)
    else:
        main()
