"""Training-only exclusion of acoustically quiet, ONB-positive seconds.

The feature is measured on the clean source, shared across all added-noise and
frequency variants. No target is changed; test/validation samples stay intact.
Peak mode uses a fixed horizontal line on the linear PSD ordinate. The former
background-quantile mode is retained only for reproducing earlier experiments.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np

from utils.experiment.learning_policy import sample_key, targets_from_metadata
from utils.experiment.spectral_peaks import PSD_UNIT, SPECTRUM_METHOD

ROOT = Path(__file__).resolve().parents[3]


class AcousticTrainingSelector:
    def __init__(self, config, onb_by_day):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.onb_by_day = onb_by_day
        self.rows = {}
        if not self.enabled:
            return
        path = Path(self.config["features_csv"])
        if not path.is_absolute():
            path = ROOT / path
        self.config["features_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.mode = self.config.get("mode", "background_quantile")
        if self.mode not in {"background_quantile", "peak_height"}:
            raise ValueError(f"Unknown acoustic selection mode: {self.mode}")
        self.feature = self.config["feature"]
        if self.mode == "peak_height":
            if not self.feature.startswith("peak_") or not self.feature.endswith("_psd"):
                raise ValueError("peak_height requires a linear peak PSD feature")
            for value in [self.config["peak_height_threshold"],
                          *self.config.get("peak_height_threshold_by_experiment", {}).values()]:
                if not np.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError("Peak height thresholds must be finite and strictly positive")
        else:
            quantile = float(self.config.get("background_quantile", 0.99))
            if not 0 < quantile < 1:
                raise ValueError("background_quantile must be between zero and one")
            if not np.isfinite(float(self.config.get("margin_db", 0))):
                raise ValueError("margin_db must be finite")
        with path.open(encoding="utf-8-sig", newline="") as inp:
            for row in csv.DictReader(inp):
                key = sample_key(row)
                if key in self.rows:
                    raise ValueError(f"Duplicate acoustic feature: {key}")
                if self.feature not in row or not np.isfinite(float(row[self.feature])):
                    raise ValueError(f"Missing/nonfinite spectral feature: {key}")
                if self.mode == "peak_height":
                    if row.get("spectrum_unit") != PSD_UNIT or row.get("spectrum_method") != SPECTRUM_METHOD:
                        raise ValueError("Peak height PSD units / estimation method do not match")
                    if float(row[self.feature]) < 0:
                        raise ValueError("Peak PSD cannot be negative")
                self.rows[key] = row

    def select(self, metadata):
        if not self.enabled:
            return np.arange(len(metadata)), {"enabled": False, "n_before": len(metadata), "n_after": len(metadata)}
        if not metadata:
            raise ValueError("Empty training partition")
        y = targets_from_metadata(metadata)
        values = []
        for row, target in zip(metadata, y):
            feature_row = self.rows[sample_key(row)]  # missing provenance must fail
            if not np.isclose(float(feature_row["heat_flux"]), target, rtol=0, atol=1e-5):
                raise ValueError("Acoustic feature / target mismatch")
            for field in ("chunk_start_seconds", "chunk_duration_seconds"):
                if not np.isclose(float(feature_row[field]), float(row[field]), rtol=0, atol=1e-6):
                    raise ValueError(f"Acoustic feature / {field} mismatch")
            values.append(float(feature_row[self.feature]))
        values = np.asarray(values)
        days = np.asarray([row["experiment_name"] for row in metadata])
        keep = np.ones(len(metadata), dtype=bool)
        details = {}
        decisions = []
        for day in sorted(set(days)):
            onb = float(self.onb_by_day[day])
            day_mask = days == day
            background = day_mask & (y < onb)
            if self.mode == "peak_height":
                # A horizontal line in the same linear PSD units as each plot.
                # No other recording/second changes its height, including folds.
                threshold = float(self.config.get("peak_height_threshold_by_experiment", {}).get(
                    day, self.config["peak_height_threshold"]))
            else:
                if not background.any():
                    raise ValueError(f"No pre-ONB reference in training partition: {day}")
                threshold = float(np.quantile(values[background], self.config.get("background_quantile", 0.99))
                                  + self.config.get("margin_db", 0.0))
            upper = self.config.get("apply_max_heat_flux_by_experiment", {}).get(day)
            eligible = day_mask & (y >= onb)
            if upper is not None:
                if float(upper) < onb:
                    raise ValueError("Selection upper heat flux cannot be below ONB")
                eligible &= y <= float(upper)
            below = values < threshold if self.mode == "peak_height" else values <= threshold
            keep[eligible & below] = False
            details[day] = {"onb_heat_flux": onb,
                            ("threshold_psd" if self.mode == "peak_height" else "threshold_db"): threshold,
                            "background_chunks": int(background.sum()) if self.mode == "background_quantile" else 0,
                            "eligible_chunks": int(eligible.sum()), "excluded_chunks": int((day_mask & ~keep).sum())}
        for i, row in enumerate(metadata):
            decisions.append({"experiment_name": row["experiment_name"], "source_wav_id": row["source_wav_id"],
                              "chunk_index": int(row["chunk_index"]), "heat_flux": float(y[i]),
                              ("peak_height_psd" if self.mode == "peak_height" else "feature_db"): float(values[i]),
                              "keep": bool(keep[i])})
        if not keep.any():
            raise ValueError("Acoustic selection removed all training data")
        return np.flatnonzero(keep), {"enabled": True, "mode": self.mode, "config": self.config,
            "threshold_source": "fixed_config" if self.mode == "peak_height" else "current_fit_background_quantile",
            "n_before": len(metadata), "n_after": int(keep.sum()), "by_experiment": details,
            "decisions": decisions, "test_filtering": False, "labels_changed": False}
