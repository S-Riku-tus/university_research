"""Lightweight research diagnostics that do not require TensorFlow training."""

from .noise_shortcut import (
    evaluate_noise_shortcut_dataset,
    load_npy_dataset_with_metadata,
    summarize_fold_metrics,
    summarize_realized_snr,
)

__all__ = [
    "evaluate_noise_shortcut_dataset",
    "load_npy_dataset_with_metadata",
    "summarize_fold_metrics",
    "summarize_realized_snr",
]
