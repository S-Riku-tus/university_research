"""Ordinary KFold on training days; estimate individual-performance weights."""
import gc

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

from utils.experiment.learning_policy import wav_groups
from utils.experiment.run_helpers import set_global_seed
from utils.models.regression.base_regression import RegressionModelMaker


def internal_splits(metadata, folds, seed, mode="chunk_kfold"):
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    if mode == "chunk_kfold":
        return list(splitter.split(np.arange(len(metadata))))
    if mode == "wav_kfold":
        groups = wav_groups(metadata)
        unique = np.unique(groups)
        return [(np.flatnonzero(np.isin(groups, unique[fit])), np.flatnonzero(np.isin(groups, unique[held])))
                for fit, held in splitter.split(unique)]
    raise ValueError(f"Unknown internal validation: {mode}")


def _validation_errors(y, predictions, groups, mode):
    """Score the independent unit represented by the selected CV mode."""
    if mode == "chunk_kfold":
        return {
            key: float(1.0 - r2_score(y, prediction))
            for key, prediction in predictions.items()
        }, "pooled_chunk_oof_R2"

    unique_groups = np.unique(groups)
    grouped_y = []
    grouped_predictions = {key: [] for key in predictions}
    for group in unique_groups:
        index = np.flatnonzero(groups == group)
        targets = np.asarray(y)[index]
        tolerance = max(1e-6, float(np.max(np.abs(targets))) * 1e-9)
        if not np.allclose(targets, targets[0], rtol=0.0, atol=tolerance):
            raise ValueError(f"Source WAV group {group!r} has multiple targets.")
        grouped_y.append(float(targets[0]))
        for key, prediction in predictions.items():
            grouped_predictions[key].append(float(np.median(prediction[index])))
    grouped_y = np.asarray(grouped_y)
    if np.var(grouped_y) == 0:
        raise ValueError("WAV-level internal weighting requires varying heat flux")
    return {
        key: float(1.0 - r2_score(grouped_y, prediction))
        for key, prediction in grouped_predictions.items()
    }, "source_wav_median_oof_R2"


def fit_individual_performance_cv(trainer, specs, x, y, metadata, selector,
                                  folds, seed, mode, pca_components, epochs):
    """Test data never enter this API. Apply selection to inner-fit only.

    Peak-height selection is fixed by config; the legacy quantile estimator and
    PCA/scaler are fitted on inner-fit only. Validation seconds remain intact.
    """
    predictions = {spec["key"]: np.full(len(y), np.nan) for spec in specs}
    groups = wav_groups(metadata)
    records, coverage = [], np.zeros(len(y), dtype=int)
    print(
        f"[internal validation] {mode}, folds={folds}; "
        "inner epoch progress is hidden and each fold/model is reported.",
        flush=True,
    )
    for fold, (fit, held) in enumerate(internal_splits(metadata, folds, seed, mode), 1):
        selected, selection = selector.select([metadata[i] for i in fit])
        fit = fit[selected]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(y[fit].reshape(-1, 1))
        x_fit, x_held = x[fit], x[held]
        x_pca = held_pca = None
        if any(spec["kind"] == "sklearn" for spec in specs):
            x_pca, others, _ = trainer.make_pca(x_fit, [x_held], pca_components, return_pca=True)
            held_pca = others[0]
        for spec in specs:
            set_global_seed(seed + fold)
            inner_spec = {**spec, "fit_verbose": 0}
            print(f"Internal {mode} {fold}/{folds}: {spec['key']}", flush=True)
            model, history = trainer.train_one_model(inner_spec, RegressionModelMaker(tuple(x.shape[1:])),
                                                     x_fit, scaled, x_pca, epochs)
            predictions[spec["key"]][held] = trainer.predict_one_model(inner_spec, model, x_held, held_pca, scaler)
            del model, history
            K.clear_session()
            gc.collect()
        coverage[held] += 1
        records.append({"fold": fold, "fit_indices": fit.tolist(), "validation_indices": held.tolist(),
                        "shared_source_wavs": len(set(groups[fit]) & set(groups[held])), "selection": selection})
    if not np.all(coverage == 1) or any(not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("Internal CV did not produce one finite prediction per sample")
    if np.var(y) == 0:
        raise ValueError("Internal performance weighting requires varying heat flux")
    errors, score_unit = _validation_errors(y, predictions, groups, mode)
    audit = {"method": mode, "shuffle": True, "random_state": seed, "folds": records,
             "weight_formula": f"normalize(1 / max(1 - {score_unit}, 1e-6))",
             "score_unit": score_unit,
             "internal_score_scope": "training-day model selection only; not unknown-recording performance",
             "test_used": False, "individual_errors": errors,
             "samples": [{"experiment_name": row["experiment_name"], "source_wav_id": row["source_wav_id"],
                          "chunk_index": int(row["chunk_index"]), "heat_flux": float(y[i]),
                          **{key: float(p[i]) for key, p in predictions.items()}} for i, row in enumerate(metadata)]}
    return errors, audit
