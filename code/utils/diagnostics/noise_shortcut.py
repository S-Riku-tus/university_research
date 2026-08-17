import csv
import math
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRFRegressor


def load_npy_dataset_with_metadata(folder_path):
    """Load arrays in deterministic filename order with chunk provenance."""
    folder_path = Path(folder_path)
    manifest_path = folder_path / "chunk_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"chunk_manifest.csv is required for P0 diagnostics: {manifest_path}"
        )

    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        manifest_by_filename = {
            row["sample_filename"]: row
            for row in csv.DictReader(f)
            if row.get("sample_filename")
        }

    x, y, metadata = [], [], []
    for npy_path in sorted(folder_path.glob("*.npy"), key=lambda path: path.name):
        row = manifest_by_filename.get(npy_path.name)
        if row is None:
            raise ValueError(
                f"{npy_path.name} is missing from {manifest_path}."
            )
        x.append(np.load(npy_path))
        y.append(float(row.get("heat_flux") or npy_path.name.split("_", 1)[0]))
        metadata.append(row)

    if not x:
        raise FileNotFoundError(f"No .npy samples found in: {folder_path}")
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=float), metadata


def _safe_pearson(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _split_plan(sample_count, metadata, split_strategy, folds, random_seed):
    indices = np.arange(sample_count)
    if split_strategy == "kfold":
        splitter = KFold(
            n_splits=int(folds),
            shuffle=True,
            random_state=int(random_seed),
        )
        return list(splitter.split(indices))

    if split_strategy == "group_kfold":
        groups = np.asarray(
            [row.get("source_wav_id", "") for row in metadata], dtype=object
        )
        if any(not str(group).strip() for group in groups):
            raise ValueError(
                "source_wav_id is required for every sample in group_kfold."
            )
        group_count = len(np.unique(groups))
        if group_count < int(folds):
            raise ValueError(
                f"group_kfold needs at least {folds} groups, got {group_count}."
            )
        splitter = GroupKFold(n_splits=int(folds))
        return list(splitter.split(indices, groups=groups))

    raise ValueError(
        "split_strategy must be 'kfold' or 'group_kfold', got "
        f"{split_strategy!r}."
    )


def _make_rf(random_seed, rf_params):
    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bynode": 0.6,
        "learning_rate": 1.0,
        "random_state": int(random_seed),
        "n_jobs": -1,
        "tree_method": "auto",
        "device": "cpu",
        "objective": "reg:squarederror",
    }
    params.update(rf_params or {})
    return XGBRFRegressor(**params)


def old_style_magnitude_minmax(x):
    """Recreate the bachelor's per-chunk magnitude [0, 1] mapping."""
    transformed = np.sqrt(np.maximum(np.asarray(x, dtype=np.float32), 0.0))
    axes = tuple(range(1, transformed.ndim))
    sample_min = transformed.min(axis=axes, keepdims=True)
    sample_max = transformed.max(axis=axes, keepdims=True)
    denominator = sample_max - sample_min
    denominator[denominator == 0] = 1.0
    return (transformed - sample_min) / denominator


def _features_for_fold(
    x,
    train_index,
    val_index,
    feature_set,
    total_power_feature,
    pca_components,
    random_seed,
):
    if feature_set == "total_power_only":
        features = np.asarray(total_power_feature, dtype=float).reshape(-1, 1)
        return features[train_index], features[val_index]

    if feature_set in {
        "full_spectrogram_pca",
        "old_style_magnitude_minmax_pca",
    }:
        transformed = np.asarray(x, dtype=np.float32)
        if feature_set == "old_style_magnitude_minmax_pca":
            # The bachelor's pipeline used magnitude STFT and independently
            # mapped every chunk to [0, 1].  Current saved arrays are linear
            # power, so sqrt restores magnitude before the same normalization.
            transformed = old_style_magnitude_minmax(transformed)

        x_train_flat = transformed[train_index].reshape(
            len(train_index), -1
        )
        x_val_flat = transformed[val_index].reshape(len(val_index), -1)
        component_count = min(
            int(pca_components),
            x_train_flat.shape[0],
            x_train_flat.shape[1],
        )
        pca = PCA(
            n_components=component_count,
            svd_solver="randomized",
            random_state=int(random_seed),
        )
        return pca.fit_transform(x_train_flat), pca.transform(x_val_flat)

    raise ValueError(f"Unknown feature_set: {feature_set!r}.")


def evaluate_noise_shortcut_dataset(
    x,
    y,
    metadata,
    split_strategies=("kfold", "group_kfold"),
    feature_sets=("total_power_only", "full_spectrogram_pca"),
    folds=3,
    random_seed=42,
    pca_components=100,
    rf_params=None,
):
    """Evaluate scalar-power and full-spectrum RFs on identical outer splits."""
    fold_rows = []
    prediction_rows = []
    split_rows = []
    manifest_power = [row.get("model_input_power", "") for row in metadata]
    if all(value not in ("", None) for value in manifest_power):
        total_power_feature = np.asarray(manifest_power, dtype=float)
        total_power_feature_source = (
            "chunk_manifest.model_input_power (mean waveform power)"
        )
    else:
        total_power_feature = x.reshape(x.shape[0], -1).mean(
            axis=1, dtype=np.float64
        )
        total_power_feature_source = (
            "mean of saved 224x224 linear-power spectrogram"
        )

    for split_strategy in split_strategies:
        splits = _split_plan(
            len(y), metadata, split_strategy, folds, random_seed
        )
        for fold, (train_index, val_index) in enumerate(splits, start=1):
            train_filenames = {
                metadata[index]["sample_filename"] for index in train_index
            }
            for index, row in enumerate(metadata):
                split_rows.append(
                    {
                        "split_strategy": split_strategy,
                        "fold": fold,
                        "sample_filename": row["sample_filename"],
                        "source_wav_id": row.get("source_wav_id", ""),
                        "partition": (
                            "train"
                            if row["sample_filename"] in train_filenames
                            else "validation"
                        ),
                    }
                )

            y_scaler = MinMaxScaler()
            y_train_scaled = y_scaler.fit_transform(
                y[train_index].reshape(-1, 1)
            ).ravel()

            for feature_set in feature_sets:
                x_train, x_val = _features_for_fold(
                    x,
                    train_index,
                    val_index,
                    feature_set,
                    total_power_feature,
                    pca_components,
                    random_seed + fold,
                )
                model = _make_rf(random_seed + fold, rf_params)
                model.fit(x_train, y_train_scaled)
                pred_scaled = model.predict(x_val).reshape(-1, 1)
                prediction = y_scaler.inverse_transform(pred_scaled).ravel()
                y_val = y[val_index]

                fold_rows.append(
                    {
                        "split_strategy": split_strategy,
                        "feature_set": feature_set,
                        "fold": fold,
                        "train_samples": len(train_index),
                        "validation_samples": len(val_index),
                        "r2": float(r2_score(y_val, prediction)),
                        "rmse": float(
                            math.sqrt(mean_squared_error(y_val, prediction))
                        ),
                        "mae": float(mean_absolute_error(y_val, prediction)),
                        "prediction_pearson": _safe_pearson(
                            y_val, prediction
                        ),
                    }
                )
                for sample_index, true_value, predicted_value in zip(
                    val_index, y_val, prediction
                ):
                    row = metadata[int(sample_index)]
                    prediction_rows.append(
                        {
                            "split_strategy": split_strategy,
                            "feature_set": feature_set,
                            "fold": fold,
                            "sample_filename": row["sample_filename"],
                            "source_wav_id": row.get("source_wav_id", ""),
                            "y_true": float(true_value),
                            "y_pred": float(predicted_value),
                            "total_power_feature": float(
                                total_power_feature[int(sample_index)]
                            ),
                        }
                    )

    return {
        "fold_metrics": fold_rows,
        "predictions": prediction_rows,
        "split_assignments": split_rows,
        "input_summary": {
            "samples": int(len(y)),
            "source_wav_groups": int(
                len({row.get("source_wav_id", "") for row in metadata})
            ),
            "total_power_feature_source": total_power_feature_source,
            "total_power_heat_flux_pearson": _safe_pearson(
                total_power_feature, y
            ),
        },
    }


def summarize_fold_metrics(rows):
    grouped = {}
    for row in rows:
        key = (row["split_strategy"], row["feature_set"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (split_strategy, feature_set), group_rows in sorted(grouped.items()):
        item = {
            "split_strategy": split_strategy,
            "feature_set": feature_set,
            "folds": len(group_rows),
        }
        for metric in ("r2", "rmse", "mae", "prediction_pearson"):
            values = np.asarray(
                [float(row[metric]) for row in group_rows], dtype=float
            )
            finite = values[np.isfinite(values)]
            item[f"{metric}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            item[f"{metric}_std"] = (
                float(np.std(finite, ddof=1))
                if finite.size > 1
                else 0.0 if finite.size == 1 else float("nan")
            )
        summary.append(item)
    return summary


def summarize_realized_snr(metadata):
    values = []
    for row in metadata:
        text = row.get("realized_snr_db", "")
        if text not in ("", None):
            values.append(float(text))
    if not values:
        return {
            "count": 0,
            "mean_db": float("nan"),
            "std_db": float("nan"),
            "min_db": float("nan"),
            "max_db": float("nan"),
        }
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean_db": float(np.mean(values)),
        "std_db": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min_db": float(np.min(values)),
        "max_db": float(np.max(values)),
    }
