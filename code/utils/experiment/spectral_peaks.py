"""Linear, unnormalised peak heights on the existing one-second Welch PSD."""
import numpy as np

PEAK_WINDOWS_HZ = ((850, 1200), (1250, 1650), (2100, 2500))
PEAK_FEATURE = "peak_2100_2500_psd"
PSD_UNIT = "digital_amplitude_squared_per_hz"
SPECTRUM_METHOD = "welch_hann_2048_overlap_1024_fs_44100"


def peak_features(frequency_hz, psd, windows=PEAK_WINDOWS_HZ):
    """Return the maximum ordinate and its frequency, not the band integral.

    The window permits slight peak-frequency drift. No per-second normalization,
    reference waveform, labels, or other seconds are used in this computation.
    """
    f = np.asarray(frequency_hz, dtype=float)
    power = np.asarray(psd, dtype=float)
    if (f.ndim != 1 or power.shape != f.shape or len(f) < 2
            or not np.isfinite(f).all() or not np.isfinite(power).all()
            or np.any(np.diff(f) <= 0) or np.any(power < 0)):
        raise ValueError("PSD must be finite, nonnegative, and aligned to increasing frequencies")
    result = {"spectrum_method": SPECTRUM_METHOD, "spectrum_unit": PSD_UNIT}
    for lo, hi in windows:
        indices = np.flatnonzero((f >= lo) & (f <= hi))
        if lo >= hi or not len(indices):
            raise ValueError(f"Empty or invalid peak-frequency window: {lo}, {hi}")
        index = indices[np.argmax(power[indices])]
        result[f"peak_{lo}_{hi}_psd"] = float(power[index])
        result[f"peak_{lo}_{hi}_hz"] = float(f[index])
    return result


def classify_peak_height(values, threshold, strong_threshold):
    """Three descriptions; selection itself only depends on the lower line."""
    values = np.asarray(values, dtype=float)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or not np.isfinite(threshold) or threshold <= 0
            or not np.isfinite(strong_threshold) or strong_threshold < threshold):
        raise ValueError("Require finite nonnegative peaks and 0 < threshold <= strong_threshold")
    return np.where(values >= strong_threshold, "strong",
                    np.where(values >= threshold, "weak_included", "below_threshold"))
