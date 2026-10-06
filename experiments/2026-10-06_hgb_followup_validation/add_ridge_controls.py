"""Fit the same Ridge protocol to existing models using training OOF only."""
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import run_hgb_complementarity_validation as run


def main():
    specification = ROOT / "configs/experiments/2026-10-06_hgb_followup_ridge_controls.json"
    controls_hash = hashlib.sha256(specification.read_bytes()).hexdigest()
    for seed in run.CONFIG["replication_seeds"]:
        folder = run.OUT / f"seed{seed}"
        frozen = run.read_json(folder / "frozen_integration.json")
        manifest = run.read_json(folder / "manifest.json")
        assert frozen["config_sha256"] == manifest["config_sha256"] == run.CONFIG_HASH
        with np.load(folder / "base_predictions.npz") as saved:
            oof = saved["oof"].copy()
            train = saved["train_indices"].copy()
        y = np.asarray([float(manifest["samples"][int(i)]["sample_filename"].split("_")[0]) / 1000 for i in train])
        methods = json.loads(specification.read_text(encoding="utf-8"))["methods"]
        frozen["methods"] = [m for m in frozen["methods"] if m["method"] not in methods]
        for noisy, name in zip([False, True], methods):
            matrix = oof[:, :, :3].reshape(-1, 3) if noisy else oof[0, :, :3]
            target = np.tile(y, len(run.NOISES)) if noisy else y
            model = make_pipeline(StandardScaler(), Ridge(alpha=run.CONFIG["ridge_alpha"]))
            model.fit(matrix, target)
            artifact = folder / f"{name}.joblib"
            joblib.dump(model, artifact, compress=3)
            np.testing.assert_allclose(model.predict(matrix[:12]), joblib.load(artifact).predict(matrix[:12]), rtol=1e-12, atol=1e-10)
            frozen["methods"].append({"method": name, "indices": [0, 1, 2], "kind": "ridge", "artifact": artifact.name,
                "noisy_oof_used": noisy, "supplemental_control_sha256": controls_hash})
        frozen["supplemental_control_config"] = str(specification.relative_to(ROOT))
        frozen["supplemental_control_sha256"] = controls_hash
        run.save_json(folder / "frozen_integration.json", frozen)
        print(f"seed{seed}: two existing-model Ridge controls frozen, training OOF only", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
