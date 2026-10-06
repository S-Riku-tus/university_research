"""Verify saved RF predictions in the paired run's fixed numerical context."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import run_hgb_complementarity_validation as run
from utils.training.model_training import ModelTrainer


def main():
    trainer = ModelTrainer(random_seed=42)
    raw = {n: np.load(run.OUT / "cache" / f"raw_{n}.npy", mmap_mode="r") for n in run.NOISES}
    rows = []
    for seed in run.CONFIG["replication_seeds"]:
        for stage in ["fold1", "fold2", "fold3", "final"]:
            folder = run.OUT / f"seed{seed}" / stage
            model = joblib.load(folder / "randomforest.joblib")
            pca = joblib.load(folder / "pca.joblib")
            scaler = joblib.load(folder / "target_scaler.joblib")
            predict = lambda x: scaler.inverse_transform(model.predict(trainer.transform_pca(pca, x)).reshape(-1, 1)).ravel() / 1000
            with np.load(folder / "randomforest_predictions.npz") as saved:
                ids = saved["indices"]
                for noise in run.NOISES:
                    x = np.asarray(raw[noise][ids])
                    p = predict(x)
                    np.testing.assert_allclose(p, saved[noise], rtol=1e-7, atol=1e-5)
                    small = predict(x[:8])
                    rows.append({"seed": seed, "stage": stage, "noise": noise, "chunks": len(ids),
                        "reload_full540_max_abs_kW_m2": float(np.max(np.abs(p - saved[noise]))),
                        "first8_vs_full540_max_abs_kW_m2": float(np.max(np.abs(small - p[:8]))), "blas_threads": 2})
    run.save_csv(run.OUT / "rf_replication_reload.csv", rows)
    run.save_json(run.OUT / "rf_replication_reload_audit.json", {"full540_prediction_checks": len(rows), "full540_reload_passed": True,
        "blas_threads": 2, "probe8_batch_max_abs_kW_m2": max(r["first8_vs_full540_max_abs_kW_m2"] for r in rows),
        "claim_scope": "Saved paired-run models under fixed two-thread CPU PCA/RF context; no general batch/thread invariance claim"})
    print("RF full saved predictions verified:", len(rows), "sets; max eight-chunk batch difference:", max(r["first8_vs_full540_max_abs_kW_m2"] for r in rows))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
