import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.experiment.spectral_peaks import peak_features, classify_peak_height, PSD_UNIT, SPECTRUM_METHOD


class PeakHeightSelectionTest(unittest.TestCase):
    def test_tone_height_tracks_squared_amplitude_and_frequency_drift(self):
        time = np.arange(44100) / 44100
        first = 1e-4 * np.sin(2 * np.pi * 2310 * time)
        f, psd = welch(first, fs=44100, window="hann", nperseg=2048, noverlap=1024)
        _, louder = welch(first * 2, fs=44100, window="hann", nperseg=2048, noverlap=1024)
        quiet, strong = peak_features(f, psd), peak_features(f, louder)
        self.assertLess(abs(quiet["peak_2100_2500_hz"] - 2310), 22)
        self.assertAlmostEqual(strong["peak_2100_2500_psd"] / quiet["peak_2100_2500_psd"], 4)
        self.assertLess(quiet["peak_1250_1650_psd"], quiet["peak_2100_2500_psd"] * 1e-5)

    def test_peak_height_is_not_band_integral_or_per_second_normalization(self):
        f = np.arange(0, 3001, 10.)
        narrow = np.zeros_like(f); narrow[f == 2300] = 3e-9
        wide = np.zeros_like(f); wide[(f >= 2150) & (f <= 2450)] = 3e-9
        self.assertAlmostEqual(peak_features(f, narrow)["peak_2100_2500_psd"],
                               peak_features(f, wide)["peak_2100_2500_psd"])
        self.assertGreater(wide.sum(), narrow.sum() * 10)
        with self.assertRaises(ValueError):
            peak_features(f, np.full_like(f, np.nan))

    def test_lower_line_includes_shoulders_without_padding_neighbors(self):
        sequence = np.array([.04, .6, 2., 100., 3., .5, .04]) * 1e-9
        self.assertEqual(classify_peak_height(sequence, 1e-9, 10e-9).tolist(),
                         ["below_threshold", "below_threshold", "weak_included", "strong",
                          "weak_included", "below_threshold", "below_threshold"])
        np.testing.assert_array_equal(sequence >= 1e-9, [False, False, True, True, True, False, False])
        self.assertEqual(int((sequence >= .3e-9).sum()), 5)
        self.assertEqual(int((sequence >= 10e-9).sum()), 1)

    def fixture(self, directory, **updates):
        rows = []
        for i, (day, q, peak) in enumerate([("a", 5, 100), ("a", 10, .2), ("a", 10, 1),
                                           ("a", 10, 3), ("a", 20, .2), ("test", 5, 1e9)]):
            rows.append({"experiment_name": day, "source_wav_id": f"w{q}", "chunk_index": i,
                "sample_filename": f"{q}_chunk{i}.npy", "heat_flux": q, "chunk_start_seconds": i,
                "chunk_duration_seconds": 1, "peak_2100_2500_psd": peak * 1e-9,
                "spectrum_unit": PSD_UNIT, "spectrum_method": SPECTRUM_METHOD})
        file = Path(directory) / "peaks.csv"
        with file.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        config = {"enabled": True, "mode": "peak_height", "features_csv": str(file),
                  "feature": "peak_2100_2500_psd", "peak_height_threshold": 1e-9, **updates}
        return AcousticTrainingSelector(config, {"a": 10, "test": 10}), rows

    def test_fixed_threshold_fit_subset_test_independence_and_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector, rows = self.fixture(tmp)
            kept, audit = selector.select(rows[:5])
            np.testing.assert_array_equal(kept, [0, 2, 3])
            self.assertEqual(audit["by_experiment"]["a"]["threshold_psd"], 1e-9)
            self.assertEqual(audit["threshold_source"], "fixed_config")
            # No pre-ONB reference is needed; no recalculation when a fold drops it.
            retained, _ = selector.select(rows[1:4])
            np.testing.assert_array_equal(retained, [1, 2])
            self.assertEqual(audit["by_experiment"]["a"]["background_chunks"], 0)
            self.assertFalse(audit["labels_changed"])
            self.assertFalse(audit["test_filtering"])
            selector, rows = self.fixture(tmp, apply_max_heat_flux_by_experiment={"a": 15})
            np.testing.assert_array_equal(selector.select(rows[:5])[0], [0, 2, 3, 4])

    def test_explicit_day_override_and_invalid_unit_threshold_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector, rows = self.fixture(tmp, peak_height_threshold_by_experiment={"a": 2e-9})
            np.testing.assert_array_equal(selector.select(rows[:5])[0], [0, 3])
            for v in (0, -1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    self.fixture(tmp, peak_height_threshold=v)
            selector, rows = self.fixture(tmp)
            file = Path(tmp) / "peaks.csv"
            file.write_text(file.read_text().replace(PSD_UNIT, "unknown_power"))
            with self.assertRaises(ValueError):
                AcousticTrainingSelector(selector.config, {"a": 10})


if __name__ == "__main__":
    unittest.main()
