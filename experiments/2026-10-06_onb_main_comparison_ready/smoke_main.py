"""Reuse the established real-input smoke harness in a new result directory."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT/"experiments/2026-10-06_onb_five_model_integration/smoke_main.py"


def main():
    spec = importlib.util.spec_from_file_location("onb_integration_smoke_harness", SOURCE)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.OUT = OUT/"smoke"
    harness.OUT.mkdir(exist_ok=True)
    harness.main()
    reports = [p for p in harness.OUT.rglob("onb_comparison/verification.json")
               if json.loads(p.read_text(encoding="utf-8"))["status"] == "passed"]
    assert len(reports) == 1
    path = reports[0]
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["smoke_test"] is True
    assert result["normal_150_epoch_main_run_completed"] is False
    assert result["epochs_complete"] is True
    assert result["source_day_metric_rows"] == 66
    assert result["reloaded_evaluation_predictions"] == 360
    assert result["fitted_models"] == 5
    with (path.parent/"fit_epoch_audit.csv").open(encoding="utf-8-sig") as stream:
        epoch_rows = list(csv.DictReader(stream))
    assert len(epoch_rows) == 6 and all(r["epochs_completed"] == "1" for r in epoch_rows)
    for name in ["main_comparison.md", "main_metrics.csv", "main_deltas.csv", "noise_degradation.csv",
                 "q100_by_source_day.csv", "ensemble_weights.csv", "fitted_reload_checks.csv",
                 "ensemble_reconstruction_checks.csv", "evaluation_protocol.json", "run_conditions.json"]:
        assert (path.parent/name).exists(), name
    record = {"status": "passed", "actual_onb_main_called": True, "epochs": 1,
              "internal_folds": 2, "source_wavs": 36, "input_chunks": 144,
              "models": 5, "base_fits": 15, "reload_conditions": 2, "reloaded_evaluation_predictions": 360,
              "source_day_metric_rows": 66, "epoch_audit_rows": 6,
              "main_report": str((path.parent/"main_comparison.md").relative_to(ROOT)),
              "normal_150_epoch_run_executed": False}
    (OUT/"main_smoke_verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=True))


if __name__ == "__main__":
    main()
