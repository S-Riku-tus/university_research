"""Shared inputs and evaluation for the three prespecified follow-up stages."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from utils.experiment.run_helpers import open_text, windows_long_path
from utils.calculation.source_day_metrics import source_day_metric_rows
from utils.calculation.onb_comparison_report import ERRORS

OUT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/experiments/2026-10-07_onb_input_and_integration_protocol.json"


def read_json(path):
    with open_text(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path):
    with open_text(path, "r", encoding="utf-8-sig", newline="") as stream:
        return pd.read_csv(stream)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def digest(path):
    return hashlib.sha256(Path(windows_long_path(path)).read_bytes()).hexdigest()


CONFIG = read_json(CONFIG_PATH)
RUN = ROOT / CONFIG["source_run"]
CLEAN = RUN / "heatflux_no_noise"
ARTIFACTS = CLEAN / "fitted_state/fold1"
MEMBERS = CONFIG["members"]
THRESHOLDS = CONFIG["source_day_thresholds_W_m2"]
FIVE = "ensemble__original3_hgb_extra_trees"
CORE = "ensemble__original3_mse"
ET4 = "ensemble__original3_extra_trees"
POOLED = "pooled_source_day_thresholds"


def noise_dir(noise):
    return "heatflux_no_noise" if noise == "clean" else "heatflux_reference_SNR="+noise


def source_predictions(noise):
    suffix = "no_noise" if noise == "clean" else noise
    return read_csv(RUN / noise_dir(noise) / f"fold_pred/pred_f1_{suffix}.csv")


def source_columns(frame):
    return [k for k in frame if k in MEMBERS or k.startswith("ensemble__")]


def source_metadata(frame):
    days = frame.source_wav_id.str[:8].map({"20250611": "2025.06.11_0.3_2", "20250618": "2025.06.18_0.3_3"})
    assert days.notna().all()
    return [{"experiment_name": d, "source_wav_id": w} for d, w in zip(days, frame.source_wav_id)]


def metric_rows(frame, predictions_W, noise, **context):
    rows = source_day_metric_rows(frame.y_true.to_numpy(), predictions_W,
        source_metadata(frame), THRESHOLDS, fold=1)
    for row in rows:
        for key in ERRORS:
            row[key+"_kW_m2"] = row.pop(key)/1000
        row.update(noise=noise, heat_flux_unit="kW/m2", **context)
    return rows


def deltas(rows, context_key="intervention", baseline_name="unchanged"):
    baseline = {(r["noise"], r["source_day"], r["model_key"]): r for r in rows if r[context_key] == baseline_name}
    result = []
    for r in rows:
        b = baseline[r["noise"], r["source_day"], r["model_key"]]
        if r[context_key] == baseline_name:
            continue
        keys = [k for k, v in b.items() if isinstance(v, (float, int, np.number)) and k not in ["fold"]]
        result.append({"noise": r["noise"], "source_day": r["source_day"], "model_key": r["model_key"],
            context_key: r[context_key], **{"delta_"+k: r[k]-b[k] for k in keys}})
    return result


def data_root():
    manifest = read_json(CLEAN / "run_manifest.json")
    assert manifest["run_hash"] == CONFIG["source_run_hash"]
    return Path(manifest["dataset"]["data_path"]).parent


def load_raw(frame, noise):
    folder = data_root() / noise_dir(noise)
    paths = [folder / name for name in frame.sample_filename]
    raw = np.asarray([np.load(windows_long_path(p)) for p in paths], dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[..., None]
    assert raw.shape == (len(frame), 224, 224, 1) and np.isfinite(raw).all() and raw.min() >= 0
    manifest = read_csv(folder / "chunk_manifest.csv").set_index("sample_filename")
    for row in frame.itertuples():
        current = manifest.loc[row.sample_filename]
        assert current.source_wav_id == row.source_wav_id and int(current.chunk_index) == int(row.chunk_index)
        assert np.isclose(float(current.heat_flux), row.y_true, rtol=1e-12, atol=1e-5)
    return raw


def training_data():
    audit = read_json(CLEAN / "internal_validation_fold1.json")
    rows = audit["samples"]
    manifest = read_csv(data_root() / "heatflux_no_noise/chunk_manifest.csv")
    filenames = {(r.source_wav_id, int(r.chunk_index)): r.sample_filename for r in manifest.itertuples()}
    frame = pd.DataFrame({"sample_filename": [filenames[r["source_wav_id"], int(r["chunk_index"])] for r in rows],
        "source_wav_id": [r["source_wav_id"] for r in rows], "chunk_index": [int(r["chunk_index"]) for r in rows],
        "y_true": [float(r["heat_flux"]) for r in rows]})
    oof = read_csv(CLEAN / "ensemble_training_oof_f1.csv")
    assert len(frame) == len(oof) == 1620
    np.testing.assert_allclose(frame.y_true, oof.y_true, rtol=1e-12, atol=1e-6)
    for key in MEMBERS:
        np.testing.assert_allclose([r[key] for r in rows], oof[key], rtol=1e-12, atol=1e-6)
    outer = source_predictions("clean")
    identities = set(zip(frame.source_wav_id, frame.chunk_index))
    assert not identities & set(zip(outer.source_wav_id, outer.chunk_index))
    assert len(identities) == 1620 and len(manifest) == 2160
    coverage = np.zeros(len(frame), int)
    for fold in audit["folds"]:
        fit, held = fold["fit_indices"], fold["validation_indices"]
        assert not set(fit)&set(held) and len(fit) == 1080 and len(held) == 540
        assert set(fit)|set(held) == set(range(1620))
        coverage[held] += 1
    assert np.all(coverage == 1)
    return frame, audit, oof


def frozen_fit():
    return read_json(CLEAN / "ensemble_training_oof_fit_f1.json")


def source_fingerprint():
    paths = [CONFIG_PATH, CLEAN / "ensemble_training_oof_fit_f1.json", CLEAN / "ensemble_training_oof_f1.csv",
        CLEAN / "internal_validation_fold1.json", ARTIFACTS / "artifact_manifest.json"]
    paths += [ARTIFACTS / name for name in ["randomforest.joblib", "conformer.weights.h5", "alexnet.weights.h5",
        "hgb.joblib", "extra_trees.joblib", "pca.joblib", "target_scaler.joblib"]]
    paths += [RUN / noise_dir(noise) / ("fold_pred/pred_f1_no_noise.csv" if noise == "clean" else f"fold_pred/pred_f1_{noise}.csv")
        for noise in CONFIG["evaluation_noise_conditions"]]
    return [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
