"""Four target chunks per condition: software verification, not performance.

Use the existing TFGPU environment from the repository root. All predictions,
figures and feature CSVs are written to a TemporaryDirectory and removed after
checking. Only provenance, counts and reload checks remain in this folder.
"""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import run_ensemble_regression_onb as onb
from utils.experiment.saved_fit_evaluation import prepare_saved_fit, run_saved_fit_evaluation, read_json


def main():
    noises = ["heatflux_no_noise", "heatflux_reference_SNR=-20"]
    with patch.object(onb, "NOISE_DIR_NAMES", noises):
        jobs = onb.build_dataset_jobs(evaluation_only=True)
        config = onb.validation_config_snapshot("maxfreq=3kHz")
    prepared = prepare_saved_fit(jobs, config, ROOT)
    for item in prepared["evaluations"]:
        selected = sorted({float(r["heat_flux"]) for r in item["metadata"]})[:4]
        rows = [next(r for r in item["metadata"] if float(r["heat_flux"]) == q) for q in selected]
        item["metadata"] = rows
        item["targets"] = np.asarray([float(r["heat_flux"]) for r in rows])
    prepared["summary"]["evaluation_chunks_per_condition"] = 4
    prepared["summary"]["scope"] = "Four target chunks per condition; software verification only, not research performance."
    with tempfile.TemporaryDirectory(prefix="onb_saved_fit_check_") as temp, threadpool_limits(limits=2):
        for job in jobs:
            job["save_base_path"] = Path(temp)
        with patch("utils.experiment.saved_fit_evaluation.prepare_saved_fit", return_value=prepared), \
             patch("utils.training.model_training.ModelTrainer.train_one_model", side_effect=AssertionError("must not train")):
            output = run_saved_fit_evaluation(jobs, config, ROOT)
        verification = read_json(output / "completed.json")
        verification["output_files"] = sorted(p.name for p in output.iterdir() if p.is_file())
        verification["condition_files"] = sorted(p.name for p in (output / noises[0]).iterdir())
        for name in ["features_by_chunk.csv", "features_by_wav.csv", "main_metrics.csv", "noise_degradation.csv"]:
            with (output / name).open(encoding="utf-8-sig", newline="") as stream:
                verification[name+"_rows"] = sum(1 for _ in csv.DictReader(stream))
        assert verification["base_predictions"] == 40
        assert verification["features_by_chunk.csv_rows"] == 1624
        assert verification["features_by_wav.csv_rows"] == 40
        assert verification["main_metrics.csv_rows"] == 44
        assert verification["noise_degradation.csv_rows"] == 22
        with (output / "noise_overview.csv").open(encoding="utf-8-sig", newline="") as stream:
            assert all(r["complete_six_noise_mean"] == "False" for r in csv.DictReader(stream))
        for name in ["prediction_by_stage.png", "../features_across_days.png"]:
            assert (output / noises[0] / name).stat().st_size > 1000
        verification["temporary_predictions_retained"] = False
    destination = Path(__file__).parent / "software_verification.json"
    destination.write_text(json.dumps(verification, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Software verification passed; temporary predictions removed and source files unchanged.")


if __name__ == "__main__":
    main()
