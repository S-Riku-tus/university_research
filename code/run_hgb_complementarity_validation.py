"""Fixed-candidate paired validation. Resume fits without changing conditions.

python code/run_hgb_complementarity_validation.py --phase train
python code/run_hgb_complementarity_validation.py --phase integrate
python code/run_hgb_complementarity_validation.py --phase analyze
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/2026-10-06_hgb_followup_validation"
CONFIG_PATH = ROOT / "configs/experiments/2026-10-06_hgb_followup_validation.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CONFIG_HASH = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
SCOPE = json.loads((ROOT / CONFIG["baseline_scope"]).read_text(encoding="utf-8"))
BASELINE = ROOT / SCOPE["runs"]["new_chunk_clean"] / "maxfreq=3kHz/heatflux_no_noise"
NOISES = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]
NOISE_DIRS = {"clean": "heatflux_no_noise", **{n: f"heatflux_reference_SNR={n}" for n in NOISES[1:]}}
KEYS = ["randomforest", "conformer", "alexnet", "hgb"]
THRESHOLDS = {"20250611": 221.5051102, "20250618": 271.6776816}


def lp(path):
    s = str(path.resolve())
    return Path(s if os.name != "nt" or s.startswith("\\\\?\\") else "\\\\?\\"+s)


def read_json(path):
    return json.loads(lp(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with lp(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    assert rows
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def identity(row):
    return row["source_wav_id"], int(float(row["chunk_index"]))


def dataset():
    from utils.experiment.learning_policy import checked_metadata, outer_splits
    manifest = read_json(BASELINE / "run_manifest.json")
    data_root = Path(manifest["dataset"]["source_dir"])/"maxfreq=3kHz"
    metadata = sorted(read_csv(data_root/NOISE_DIRS["clean"]/"chunk_manifest.csv"), key=lambda r: r["sample_filename"])
    metadata = checked_metadata(metadata, manifest["dataset"]["experiment_name"])
    assert len(metadata) == 2160 and len({identity(r) for r in metadata}) == 2160
    splits = {}
    for seed in [42, *CONFIG["replication_seeds"]]:
        policy = {"evaluation_mode": "within_wav_chunk", "within_wav_chunk_experiment": manifest["dataset"]["experiment_name"], "test_fraction": .25, "test_split_seed": seed, "training_noise": "clean_only"}
        train, test = outer_splits(metadata, metadata, policy)[0]
        assert len(train) == 1620 and len(test) == 540 and not set(train)&set(test)
        splits[seed] = train, test
    old_test = read_json(BASELINE/"split_manifest.json")["folds"][0]["evaluation_sample_indices"]
    np.testing.assert_array_equal(splits[42][1], old_test)
    y = np.asarray([float(r["sample_filename"].split("_")[0])/1000 for r in metadata])
    return data_root, metadata, y, splits


def prepare_cache(data_root, metadata):
    from utils.dataloading.acoustic_summary_features import AcousticFrequency34
    cache = OUT/"cache"; cache.mkdir(parents=True, exist_ok=True)
    ids = np.asarray([f"{r['source_wav_id']}|{int(r['chunk_index'])}" for r in metadata])
    for noise in NOISES:
        path = cache/f"raw_{noise}.npy"
        features = cache/f"features_{noise}.npz"
        if path.exists() and features.exists():
            with np.load(features) as saved:
                np.testing.assert_array_equal(saved["ids"], ids)
            continue
        lookup = {identity(r): r for r in read_csv(data_root/NOISE_DIRS[noise]/"chunk_manifest.csv")}
        assert set(lookup) == {identity(r) for r in metadata}
        temporary = cache/f"raw_{noise}.partial.npy"
        raw = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(2160, 224, 224, 1))
        for i, row in enumerate(metadata):
            sample = np.load(lp(data_root/NOISE_DIRS[noise]/lookup[identity(row)]["sample_filename"]), allow_pickle=False)
            if sample.shape == (224, 224): sample = sample[..., None]
            raw[i] = sample
        raw.flush()
        X = AcousticFrequency34().transform(raw)
        np.savez_compressed(features, X=X, ids=ids)
        del raw
        temporary.replace(path)
        print(f"[cache] {noise}: 2160 aligned raw inputs and frequency34", flush=True)
    return cache


def train(seeds):
    import tensorflow as tf
    from utils.config.onb_defaults import onb_model_specs
    from utils.experiment.run_helpers import set_global_seed
    from utils.models.regression.base_regression import RegressionModelMaker
    from utils.training.model_training import ModelTrainer
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Paired 150-epoch neural replication requires the available GPU; no CPU fallback was requested.")
    data_root, metadata, y, splits = dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = prepare_cache(data_root, metadata)
    X = {n: np.load(cache/f"raw_{n}.npy", mmap_mode="r") for n in NOISES}
    F = {n: np.load(cache/f"features_{n}.npz")["X"] for n in NOISES}
    original_specs = read_json(BASELINE/"run_manifest.json")["run_specs"]
    registry = {s["key"]: s for s in onb_model_specs()}
    specs = []
    for original in original_specs:
        spec = dict(registry[original["key"]]); spec.update(original); spec["progress_interval_epochs"] = 50
        spec["accept_partial_min_epochs"] = CONFIG["epochs"]+1
        specs.append(spec)
    for seed in seeds:
        folder = OUT/f"seed{seed}"; folder.mkdir(exist_ok=True)
        train_indices, test_indices = splits[seed]
        inner = [(train_indices[a], train_indices[b]) for a,b in KFold(3, shuffle=True, random_state=seed).split(train_indices)]
        manifest_path = folder/"manifest.json"
        if manifest_path.exists():
            assert read_json(manifest_path)["config_sha256"] == CONFIG_HASH
        else:
            save_json(manifest_path, {"config_sha256": CONFIG_HASH, "seed": seed, "train_indices": train_indices.tolist(), "test_indices": test_indices.tolist(), "samples": metadata,
                "folds": [{"fold": fold, "fit_indices": fit.tolist(), "held_indices": held.tolist()} for fold,(fit,held) in enumerate(inner,1)], "source_baseline": str(BASELINE.relative_to(ROOT)), "source_run_specs": original_specs,
                "evaluation_scope": "known-WAV unused chunks", "outer_train_chunk_overlap": 0})
        stages = [(f"fold{fold}", seed+fold, fit, held) for fold,(fit,held) in enumerate(inner,1)]
        stages.append(("final", seed+1, train_indices, test_indices))
        trainer = ModelTrainer(random_seed=seed)
        for name, model_seed, fit, held in stages:
            stage = folder/name; stage.mkdir(exist_ok=True)
            fit_x = np.asarray(X["clean"][fit])
            scaler_path, pca_path = stage/"target_scaler.joblib", stage/"pca.joblib"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
            else:
                scaler = MinMaxScaler().fit((y[fit]*1000).reshape(-1,1)); joblib.dump(scaler, scaler_path)
            y_scaled = scaler.transform((y[fit]*1000).reshape(-1,1))
            if pca_path.exists():
                pca = joblib.load(pca_path); fit_pca = trainer.transform_pca(pca, fit_x)
            else:
                fit_pca, _, pca = trainer.make_pca(fit_x, [], 100, return_pca=True); joblib.dump(pca, pca_path)
            for spec in [*specs, {"key": "hgb", "kind": "sklearn_summary"}]:
                key = spec["key"]; complete = stage/f"{key}_complete.json"; predicted = stage/f"{key}_predictions.npz"
                if complete.exists() and predicted.exists():
                    audit = read_json(complete); assert audit["config_sha256"] == CONFIG_HASH
                    print(f"[resume] seed{seed} {name} {key}", flush=True); continue
                set_global_seed(model_seed, deterministic_ops=True)
                started = time.monotonic()
                artifact = stage/(key+".weights.h5" if spec["kind"] == "keras" else key+".joblib")
                maker = RegressionModelMaker((224,224,1))
                if artifact.exists():
                    if spec["kind"] == "keras":
                        model = spec["builder"](maker, **spec.get("builder_params", {})); model.load_weights(artifact)
                    else:
                        model = joblib.load(artifact)
                    training_info = read_json(stage/f"{key}_fit.json")
                else:
                    print(f"[fit] seed{seed} {name} {key}; fit={len(fit)} held={len(held)} model_seed={model_seed}", flush=True)
                    if key == "hgb":
                        params = {k:v for k,v in CONFIG["candidate"].items() if k not in ["family", "representation"]}
                        model = HistGradientBoostingRegressor(**params, random_state=model_seed)
                        model.fit(F["clean"][fit], y_scaled.ravel())
                        training_info = {"iterations": int(model.n_iter_)}
                    else:
                        model, history = trainer.train_one_model(spec, maker, fit_x, y_scaled, fit_pca, CONFIG["epochs"])
                        training_info = dict(history.params) if history is not None else {}
                        if spec["kind"] == "keras":
                            completed_epochs = int(training_info.get("epochs_completed", len(history.history["loss"])))
                            assert completed_epochs == CONFIG["epochs"], training_info
                            training_info["epochs_completed"] = completed_epochs
                            save_json(stage/f"{key}_loss.json", history.history)
                    save_json(stage/f"{key}_fit.json", training_info)
                    if spec["kind"] == "keras": model.save_weights(artifact)
                    else: joblib.dump(model, artifact, compress=3)
                def predict(raw, features):
                    if key == "hgb": return scaler.inverse_transform(model.predict(features).reshape(-1,1)).ravel()/1000
                    xp = trainer.transform_pca(pca, raw) if key == "randomforest" else None
                    return trainer.predict_one_model(spec, model, raw, xp, scaler)/1000
                predictions = {noise: predict(np.asarray(X[noise][held]), F[noise][held]) for noise in NOISES}
                assert all(np.isfinite(values).all() for values in predictions.values())
                # Every persisted model is reloaded and checked on clean plus noisy held inputs.
                probe = held[:8]
                for noise in ["clean", "-20"]:
                    before = predict(np.asarray(X[noise][probe]), F[noise][probe])
                    if spec["kind"] == "keras":
                        restored = spec["builder"](maker, **spec.get("builder_params", {})); restored.load_weights(artifact)
                    else: restored = joblib.load(artifact)
                    old_model = model; model = restored
                    after = predict(np.asarray(X[noise][probe]), F[noise][probe]); model = old_model
                    np.testing.assert_allclose(after, before, rtol=1e-7, atol=1e-5)
                    del restored
                np.savez_compressed(predicted, indices=held, **predictions)
                save_json(complete, {"config_sha256": CONFIG_HASH, "seed": seed, "stage": name, "model": key, "model_seed": model_seed,
                    "fit_chunks": len(fit), "held_chunks": len(held), "shared_chunks": 0, "clean_fit": True, "noise_refit": False,
                    "training_info": training_info, "elapsed_seconds": time.monotonic()-started, "reload_verified": True,
                    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
                print(f"[done] seed{seed} {name} {key}, {time.monotonic()-started:.1f}s", flush=True)
                del model
                tf.keras.backend.clear_session(); gc.collect()
            del fit_x, fit_pca
        assemble_seed(folder, train_indices, test_indices)


def assemble_seed(folder, train_indices, test_indices):
    oof = np.full((len(NOISES),len(train_indices),len(KEYS)), np.nan)
    final = np.full((len(NOISES),len(test_indices),len(KEYS)), np.nan)
    positions = {int(idx):j for j,idx in enumerate(train_indices)}
    for j,key in enumerate(KEYS):
        for fold in [1,2,3]:
            with np.load(folder/f"fold{fold}"/f"{key}_predictions.npz") as saved:
                pos = [positions[int(idx)] for idx in saved["indices"]]
                for i,noise in enumerate(NOISES): oof[i,pos,j] = saved[noise]
        with np.load(folder/"final"/f"{key}_predictions.npz") as saved:
            np.testing.assert_array_equal(saved["indices"],test_indices)
            for i,noise in enumerate(NOISES): final[i,:,j] = saved[noise]
    assert np.isfinite(oof).all() and np.isfinite(final).all()
    np.savez_compressed(folder/"base_predictions.npz", oof=oof, outer=final, train_indices=train_indices, test_indices=test_indices,
        keys=np.asarray(KEYS), noises=np.asarray(NOISES))
    save_json(folder/"training_complete.json", {"config_sha256": CONFIG_HASH, "base_fits": 16, "keras_fits": 8, "epochs_per_keras_fit": CONFIG["epochs"],
        "clean_fitted_noisy_oof": True, "fold_models_saved": True, "outer_train_overlap": 0})


def simplex(matrix, target, penalty=0):
    n = matrix.shape[1]
    corr = np.zeros((n,n))
    if penalty:
        residual = matrix-target[:,None]
        with np.errstate(divide="ignore",invalid="ignore"):
            corr = np.nan_to_num(np.maximum(np.corrcoef(residual,rowvar=False),0),nan=0)
        np.fill_diagonal(corr,0)
    scale = float(np.var(target))
    result = minimize(lambda w: (np.mean((matrix@w-target)**2)+penalty*(w@corr@w))/scale,
        np.full(n,1/n), jac=lambda w: (2*matrix.T@(matrix@w-target)/len(target)+2*penalty*corr@w)/scale,
        method="SLSQP",bounds=[(0,1)]*n,constraints={"type":"eq","fun":lambda w:w.sum()-1,"jac":lambda w:np.ones(n)},options={"ftol":1e-12,"maxiter":1000})
    assert result.success,result.message
    weight=np.maximum(result.x,0);weight/=weight.sum();return weight


def integrate(seeds):
    for seed in seeds:
        folder=OUT/f"seed{seed}"
        if not (folder/"training_complete.json").exists(): raise RuntimeError(f"seed{seed}: complete paired training first")
        manifest=read_json(folder/"manifest.json"); assert manifest["config_sha256"]==CONFIG_HASH
        with np.load(folder/"base_predictions.npz") as saved:
            oof=saved["oof"].copy();train_indices=saved["train_indices"].copy()
        y=np.asarray([float(manifest["samples"][int(i)]["sample_filename"].split("_")[0])/1000 for i in train_indices])
        mse=np.mean((oof[0,:,:3]-y[:,None])**2,axis=0);performance=1/mse;performance/=performance.sum()
        catalog=[{"method":"existing_performance","indices":[0,1,2],"weights":performance.tolist(),"kind":"linear"},
                 {"method":"existing_equal","indices":[0,1,2],"weights":[1/3]*3,"kind":"linear"},
                 {"method":"hgb4_equal","indices":[0,1,2,3],"weights":[.25]*4,"kind":"linear"}]
        for method,indices,noisy in [("existing3_clean_mse",[0,1,2],False),("hgb4_clean_mse",[0,1,2,3],False),
            ("ca_hgb_clean_mse",[1,2,3],False),("conformer_hgb_clean_mse",[1,3],False),
            ("existing3_noisy_mse",[0,1,2],True),("hgb4_noisy_mse",[0,1,2,3],True),("conformer_hgb_noisy_mse",[1,3],True)]:
            matrix=oof[:,:,indices].reshape(-1,len(indices)) if noisy else oof[0][:,indices]
            target=np.tile(y,len(NOISES)) if noisy else y
            weights=simplex(matrix,target)
            catalog.append({"method":method,"indices":indices,"weights":weights.tolist(),"kind":"linear","noisy_oof_used":noisy})
        penalty=CONFIG["diversity_penalty_fraction"]*float(np.mean((oof[0,:,:3]@performance-y)**2))
        catalog.append({"method":"hgb4_diversity","indices":[0,1,2,3],"weights":simplex(oof[0],y,penalty).tolist(),"kind":"linear","penalty":penalty})
        for noisy in [False,True]:
            matrix=oof.reshape(-1,4) if noisy else oof[0];target=np.tile(y,len(NOISES)) if noisy else y
            model=make_pipeline(StandardScaler(),Ridge(alpha=CONFIG["ridge_alpha"]))
            model.fit(matrix,target)
            name="hgb4_noisy_ridge" if noisy else "hgb4_clean_ridge"
            artifact=folder/f"{name}.joblib";joblib.dump(model,artifact,compress=3)
            np.testing.assert_allclose(model.predict(matrix[:10]),joblib.load(artifact).predict(matrix[:10]),rtol=1e-12,atol=1e-10)
            catalog.append({"method":name,"indices":[0,1,2,3],"kind":"ridge","artifact":artifact.name,"noisy_oof_used":noisy})
        save_json(folder/"frozen_integration.json",{"config_sha256":CONFIG_HASH,"fit_labels":"training OOF only","outer_labels_used":False,"methods":catalog})
        print(f"[integration] seed{seed}: {len(catalog)} frozen methods",flush=True)


def analyze(seeds):
    # Use the already-verified day-threshold metric definition from the saved analysis.
    sys.path.insert(0,str(ROOT/"experiments/2026-10-06_chunk_complementary_model_pilots"))
    import pilot as reference
    metrics,predictions,weights=[],[],[]
    for seed in seeds:
        folder=OUT/f"seed{seed}";manifest=read_json(folder/"manifest.json");frozen=read_json(folder/"frozen_integration.json")
        assert frozen["config_sha256"]==CONFIG_HASH
        with np.load(folder/"base_predictions.npz") as saved:
            outer=saved["outer"].copy();test=saved["test_indices"].copy();oof=saved["oof"].copy();train=saved["train_indices"].copy()
        for unit,indices,source in [("outer_unused_chunk",test,outer),("training_fit_diagnostic",train,oof)]:
            rows=[manifest["samples"][int(i)] for i in indices]
            y=np.asarray([float(r["sample_filename"].split("_")[0])/1000 for r in rows]);days=np.asarray([r["source_wav_id"][:8] for r in rows])
            for n,noise in enumerate(NOISES):
                pool={key:source[n,:,j] for j,key in enumerate(KEYS)}
                for method in frozen["methods"]:
                    matrix=source[n][:,method["indices"]]
                    pool[method["method"]]=matrix@method["weights"] if method["kind"]=="linear" else joblib.load(folder/method["artifact"]).predict(matrix)
                for name,prediction in pool.items():
                    assert np.isfinite(prediction).all()
                    for row in reference.record_summary(name,unit,noise,y,prediction,days):metrics.append({"seed":seed,**row})
                    if unit=="outer_unused_chunk":
                        for i,r in enumerate(rows):predictions.append({"seed":seed,"method":name,"noise":noise,"source_wav_id":r["source_wav_id"],"chunk_index":r["chunk_index"],"y_kW_m2":y[i],"onb_kW_m2":THRESHOLDS[days[i]],"prediction_kW_m2":prediction[i]})
        for method in frozen["methods"]:
            for i,w in zip(method["indices"],method.get("weights",[])):weights.append({"seed":seed,"method":method["method"],"model":KEYS[i],"weight":w})
    save_csv(OUT/"metrics.csv",metrics);save_csv(OUT/"outer_predictions.csv",predictions);save_csv(OUT/"weights.csv",weights)
    save_json(OUT/"completed.json",{"config_sha256":CONFIG_HASH,"seeds":seeds,"status":"paired training, clean-fit noisy OOF and frozen evaluation complete","metric_rows":len(metrics),"outer_labels_used_for_selection":False})
    print("[analysis] completed",len(metrics),"metric rows",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=["train","integrate","analyze"],required=True);parser.add_argument("--seeds",type=int,nargs="+")
    args=parser.parse_args();seeds=args.seeds or CONFIG["replication_seeds"]
    if not set(seeds)<=set(CONFIG["replication_seeds"]):raise ValueError("Use only the predefined replication seeds.")
    with threadpool_limits(limits=2):
        {"train":train,"integrate":integrate,"analyze":analyze}[args.phase](seeds)
