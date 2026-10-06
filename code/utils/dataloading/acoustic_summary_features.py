"""Fixed acoustic summaries for the 3 kHz, one-second HGB candidate."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_VERSION = 1
BASE_NAMES = ["log_power_512_1000", "log_power_1000_2000", "log_power_2000_3000", "log_power_2100_2500",
              "log_power_total", "log_ratio_2000_3000_to_1000_2000", "log_ratio_2100_2500_to_1000_2000",
              "spectral_centroid_hz", "normalized_spectral_entropy", "temporal_power_cv"]


def frequency34_features(sample, log_floor=1e-20):
    """Match the pilot's post-resize coordinates; sample axes are time/frequency."""
    x = np.asarray(sample, dtype=np.float64)
    if x.shape == (224, 224, 1):
        x = x[..., 0]
    if x.shape != (224, 224) or not np.isfinite(x).all() or np.min(x) < 0:
        raise ValueError("Expected finite nonnegative power with shape (224,224[,1]).")
    if not np.isfinite(log_floor) or log_floor <= 0:
        raise ValueError("log_floor must be positive and finite.")
    powers, values = {}, []
    for low, high in [(512, 1000), (1000, 2000), (2000, 3000), (2100, 2500)]:
        a, b = round(224*low/3000), round(224*high/3000)
        power = float(x[:, a:b].mean()); powers[(low, high)] = power
        values.append(float(np.log10(max(power, log_floor))))
    total = float(np.log10(max(float(x.mean()), log_floor)))
    values.append(total)
    for band in [(2000, 3000), (2100, 2500)]:
        values.append(float(np.log10(max(powers[band], log_floor)/max(powers[(1000, 2000)], log_floor))))
    spectral = x.mean(axis=0)
    weights = spectral/max(float(spectral.sum()), log_floor)
    nonzero = weights > 0
    values.extend([float(weights@np.linspace(0, 3000, 224)),
        float(-np.sum(weights[nonzero]*np.log(weights[nonzero]))/np.log(224))])
    temporal = x.mean(axis=1)
    values.append(float(temporal.std()/max(float(temporal.mean()), log_floor)))
    for low in range(0, 3000, 125):
        a, b = round(224*low/3000), round(224*(low+125)/3000)
        values.append(float(np.log10(max(float(x[:, a:b].mean()), log_floor))-total))
    return np.asarray(values)


class AcousticFrequency34(BaseEstimator, TransformerMixin):
    """Stateless transformer; no label, recording identity or clean reference."""
    def __init__(self, log_floor=1e-20):
        self.log_floor = log_floor

    def fit(self, X, y=None):
        # Validation is performed by transform; the representation learns no parameters.
        return self

    def transform(self, X):
        return np.asarray([frequency34_features(sample, self.log_floor) for sample in X])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*BASE_NAMES, *[f"relative_log_power_{low}_{low+125}" for low in range(0, 3000, 125)]])
