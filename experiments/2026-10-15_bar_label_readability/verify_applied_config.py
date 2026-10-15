"""Check the applied ONB presentation configuration and metadata without training."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import run_ensemble_regression_onb as onb


def check_proposed_values(expected, actual, path="VALIDATION_CONFIG"):
    for key, value in expected.items():
        if isinstance(value, dict):
            check_proposed_values(value, actual[key], f"{path}.{key}")
        else:
            assert actual[key] == value, f"{path}.{key} differs from the proposed conditions"


def main():
    proposal_path = ROOT / "configs/experiments/2026-10-15_onb_presentation_run_proposal.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    check_proposed_values(proposal["recommended_validation_config"], onb.VALIDATION_CONFIG)
    onb.validate_validation_config(onb.MODEL_SPECS)
    config = onb.validation_config_snapshot("maxfreq=3kHz")
    assert not config["models"]["parameter_search_enabled"]
    assert config["models"]["parameter_sets"][0]["models"] == proposal["preserve_model_parameters"]
    jobs = onb.build_dataset_jobs()
    checked = onb.validate_main_comparison_jobs(jobs, config)
    assert len(jobs) == 7 and checked["status"] == "passed"
    assert all(row["source_wavs"] == 36 and row["input_chunks"] == 2160
               and row["training_chunks"] == 1620 and row["evaluation_chunks"] == 540
               for row in checked["datasets"])
    assert len({row["evaluation_identity_sha256"] for row in checked["datasets"]}) == 1
    original_check = json.loads((OUT / "proposed_run_input_check.json").read_text(encoding="utf-8"))
    assert checked["datasets"] == original_check["datasets"]
    checked.update({"date": "2026-10-15", "proposal_applied_to_entrypoint": True,
                    "entrypoint_sha256": hashlib.sha256(Path(onb.__file__).read_bytes()).hexdigest(),
                    "new_fits": 0, "new_inference": 0, "input_arrays_loaded": False,
                    "matches_proposed_conditions": True, "matches_proposed_input_check": True})
    (OUT / "applied_run_input_check.json").write_text(
        json.dumps(checked, ensure_ascii=False, indent=2), encoding="utf-8")
    record = {"date": "2026-10-15", "status": "applied_not_run",
              "source_proposal": str(proposal_path.relative_to(ROOT)),
              "validation_config": onb.VALIDATION_CONFIG, "resolved_config": config,
              "entrypoint_sha256": checked["entrypoint_sha256"],
              "verification": str((OUT / "applied_run_input_check.json").relative_to(ROOT)),
              "protocol_note": proposal["protocol_note"], "training_or_inference_started": False}
    (ROOT / "configs/experiments/2026-10-15_onb_presentation_run_applied.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "datasets": len(jobs), "source_wavs": 36,
                      "training_chunks": 1620, "evaluation_chunks": 540,
                      "matches_proposed_conditions": True, "training_started": False}))


if __name__ == "__main__":
    main()
