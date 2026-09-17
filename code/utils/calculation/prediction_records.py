"""Chunk-level prediction records with source/time provenance."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


PREDICTION_METADATA_COLUMNS = [
    "fold",
    "sample_index",
    "sample_filename",
    "experiment_name",
    "source_wav_id",
    "source_wav_name",
    "chunk_index",
    "chunk_start_seconds",
    "chunk_duration_seconds",
    "y_true",
]


def _long_path(path):
    value = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\" + value
    return value


def _to_float(value, default=np.nan):
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default=0):
    if value in (None, ""):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.17g}"
    if isinstance(value, np.integer):
        return int(value)
    return value


def build_fold_prediction_rows(
    val_indices,
    y_true,
    predictions,
    sample_metadata,
    fold,
):
    """Attach source-WAV/time provenance to one chunk prediction set."""
    val_indices = np.asarray(val_indices).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    prediction_arrays = {
        key: np.asarray(values, dtype=float).ravel()
        for key, values in predictions.items()
    }
    expected = len(val_indices)
    if len(y_true) != expected:
        raise ValueError("val_indices and y_true must have the same length.")
    for key, values in prediction_arrays.items():
        if len(values) != expected:
            raise ValueError(
                f"Prediction length mismatch for {key}: {len(values)} != {expected}."
            )

    rows = []
    for row_i, sample_index in enumerate(val_indices):
        sample_index = int(sample_index)
        if not 0 <= sample_index < len(sample_metadata):
            raise IndexError(f"sample_index outside metadata: {sample_index}")
        metadata = dict(sample_metadata[sample_index])
        source_wav_id = str(metadata.get("source_wav_id", "")).strip()
        if not source_wav_id:
            raise ValueError(
                f"source_wav_id is missing for sample_index={sample_index}."
            )
        duration = _to_float(metadata.get("chunk_duration_seconds"), default=1.0)
        chunk_index = _to_int(metadata.get("chunk_index"), default=row_i)
        start_seconds = _to_float(
            metadata.get("chunk_start_seconds"),
            default=chunk_index * duration,
        )
        row = {
            "fold": int(fold),
            "sample_index": sample_index,
            "sample_filename": metadata.get("sample_filename", ""),
            "experiment_name": metadata.get("experiment_name", ""),
            "source_wav_id": source_wav_id,
            "source_wav_name": metadata.get("source_wav_name", ""),
            "chunk_index": chunk_index,
            "chunk_start_seconds": start_seconds,
            "chunk_duration_seconds": duration,
            "y_true": float(y_true[row_i]),
        }
        row.update({
            key: float(values[row_i])
            for key, values in prediction_arrays.items()
        })
        rows.append(row)
    return rows


def write_fold_prediction_csv(path, rows, model_keys):
    """Write chunk predictions in a stable, post-processable schema."""
    path = Path(path)
    os.makedirs(_long_path(path.parent), exist_ok=True)
    with open(_long_path(path), "w", newline="", encoding="utf-8") as output:
        fieldnames = PREDICTION_METADATA_COLUMNS + list(model_keys)
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def load_sample_metadata_without_arrays(data_path):
    """Load metadata in the exact sorted-.npy order used by the training loader."""
    data_path = Path(data_path)
    manifest_path = data_path / "chunk_manifest.csv"
    if not os.path.isfile(_long_path(manifest_path)):
        raise FileNotFoundError(f"chunk_manifest.csv not found: {manifest_path}")
    with open(_long_path(manifest_path), newline="", encoding="utf-8-sig") as source:
        manifest_by_filename = {
            row["sample_filename"]: row
            for row in csv.DictReader(source)
            if row.get("sample_filename")
        }
    filenames = sorted(
        entry.name for entry in os.scandir(_long_path(data_path))
        if entry.is_file() and entry.name.endswith(".npy")
    )
    metadata = []
    for filename in filenames:
        if filename not in manifest_by_filename:
            raise ValueError(f"Manifest row missing for {filename} in {data_path}")
        metadata.append({"sample_filename": filename, **manifest_by_filename[filename]})
    return metadata
