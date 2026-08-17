import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.dataloading.waterflow_preprocessing import (  # noqa: E402
    realized_snr_db,
    scale_noise_chunk,
    select_noise_chunk,
    stable_noise_seed,
    write_chunk_manifest,
)
from utils.diagnostics.noise_shortcut import (  # noqa: E402
    evaluate_noise_shortcut_dataset,
    old_style_magnitude_minmax,
)


class WaterflowNoiseDiagnosticsTest(unittest.TestCase):
    def test_relative_and_fixed_scaling_are_distinct(self):
        signal = np.ones(100)
        noise = np.ones(200)
        noise_chunk = noise[:100]

        relative, relative_scale = scale_noise_chunk(
            signal, noise, noise_chunk, snr=-20
        )
        fixed, fixed_scale = scale_noise_chunk(
            signal,
            noise,
            noise_chunk,
            snr=-20,
            scaling_mode="fixed_reference_rms",
            fixed_reference_signal_rms=2.0,
        )

        self.assertAlmostEqual(relative_scale, 10.0)
        self.assertAlmostEqual(fixed_scale, 20.0)
        self.assertAlmostEqual(realized_snr_db(signal, relative), -20.0)
        self.assertAlmostEqual(realized_snr_db(signal, fixed), -26.0205999)

    def test_random_noise_offset_is_reproducible(self):
        noise = np.arange(100, dtype=float)
        seed = stable_noise_seed(42, "source-1", -20, 3)
        first, first_offset = select_noise_chunk(
            noise,
            20,
            randomize_offset=True,
            random_seed=seed,
        )
        second, second_offset = select_noise_chunk(
            noise,
            20,
            randomize_offset=True,
            random_seed=seed,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_offset, second_offset)

    def test_fixed_global_rms_is_independent_of_source_power(self):
        quiet_signal = np.ones(100)
        loud_signal = np.ones(100) * 1000
        noise_reference = np.linspace(0.1, 3.0, 200)
        noise_chunk = noise_reference[20:120]

        quiet_scaled, _ = scale_noise_chunk(
            quiet_signal,
            noise_reference,
            noise_chunk,
            snr=-20,
            scaling_mode="fixed_global_rms",
            fixed_reference_signal_rms=2.0,
        )
        loud_scaled, _ = scale_noise_chunk(
            loud_signal,
            noise_reference,
            noise_chunk,
            snr=-20,
            scaling_mode="fixed_global_rms",
            fixed_reference_signal_rms=2.0,
        )

        np.testing.assert_allclose(quiet_scaled, loud_scaled)
        self.assertAlmostEqual(float(np.mean(quiet_scaled**2)), 400.0)

    def test_chunk_manifest_upserts_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = {
                "sample_filename": "1_a.npy",
                "source_wav_id": "a",
                "chunk_index": 0,
                "realized_snr_db": -19.0,
            }
            updated = {**first, "realized_snr_db": -20.0}
            write_chunk_manifest(tmp, [first])
            write_chunk_manifest(tmp, [updated])

            with (Path(tmp) / "chunk_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["realized_snr_db"]), -20.0)

    def test_same_outer_splits_support_scalar_and_group_diagnostics(self):
        x = (
            np.arange(12, dtype=np.float32)[:, None, None]
            + np.ones((12, 2, 2), dtype=np.float32)
        )
        y = np.repeat(np.arange(6, dtype=float) * 100, 2)
        metadata = [
            {
                "sample_filename": f"{index}.npy",
                "source_wav_id": f"wav-{index // 2}",
            }
            for index in range(12)
        ]
        result = evaluate_noise_shortcut_dataset(
            x,
            y,
            metadata,
            split_strategies=("kfold", "group_kfold"),
            feature_sets=("total_power_only",),
            folds=3,
            random_seed=42,
            rf_params={"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        )

        self.assertEqual(len(result["fold_metrics"]), 6)
        self.assertEqual(len(result["predictions"]), 24)
        self.assertEqual(len(result["split_assignments"]), 72)

    def test_old_style_normalization_removes_absolute_scale(self):
        base = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4)
        x = np.concatenate([base, base * 100.0], axis=0)
        normalized = old_style_magnitude_minmax(x)
        np.testing.assert_allclose(normalized[0], normalized[1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
