import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.dataloading.waterflow_preprocessing import (  # noqa: E402
    save_spectrogram_chunks_with_snr,
)


class CorrectedWaterflowGenerationTest(unittest.TestCase):
    def test_paired_fixed_global_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            context = {
                "experiment_name": "synthetic_experiment",
                "folder_path": str(tmp_path / "recordings"),
                "waterflow_path": str(tmp_path / "noise.wav"),
                "heat_flux_csv_path": str(tmp_path / "labels.csv"),
                "heat_flux_label_by_index": {},
                "base_npy_save_folder_path": str(tmp_path / "npy"),
                "base_png_save_folder_path": str(tmp_path / "png"),
                "save_date": 20260817,
                "script_path": str(REPO_ROOT / "code" / "2.run_npy_waterflow_2つhighpass.py"),
            }
            config = {
                "fp": 500,
                "fs": 400,
                "gpass": 1,
                "gstop": 40,
                "sample_number": 2,
                "audio_samples_used": 8,
                "save_npy": True,
                "save_spectrogram_png": False,
                "png_with_axes": False,
                "noise_scaling_mode": "fixed_global_rms",
                "fixed_reference_signal_rms": 0.5,
                "fixed_reference_provenance": "synthetic test scalar",
                "randomize_noise_offset_per_chunk": True,
                "pair_noise_across_snr": True,
                "noise_seed_scope": "shared_across_sources",
                "noise_level_folder_prefix": "reference_SNR",
                "noise_level_definition": "synthetic reference level",
                "random_seed": 42,
                "include_source_in_filename": True,
                "saved_array_dtype": "float32",
                "max_chunks_per_source": None,
                "audio_loading": "prefiltered synthetic arrays",
            }

            save_spectrogram_chunks_with_snr(
                file_path=str(tmp_path / "index=1.123.wav"),
                max_freq_hz=3000,
                chunk_seconds=8 / 44100,
                context=context,
                config=config,
                snr_db=[0, -20],
                prefiltered_signal=np.linspace(0.1, 0.8, 8),
                prefiltered_waterflow_noise=np.linspace(0.1, 6.4, 64),
            )

            maxfreq_dir = tmp_path / "npy" / "maxfreq=3kHz"
            rows_by_level = {}
            for level in (0, -20):
                level_dir = maxfreq_dir / f"heatflux_reference_SNR={level}"
                array_path = next(level_dir.glob("*.npy"))
                self.assertEqual(np.load(array_path).dtype, np.float32)
                with (level_dir / "chunk_manifest.csv").open(
                    newline="", encoding="utf-8"
                ) as file:
                    rows_by_level[level] = list(csv.DictReader(file))[0]

            self.assertEqual(
                rows_by_level[0]["noise_seed"],
                rows_by_level[-20]["noise_seed"],
            )
            self.assertEqual(
                rows_by_level[0]["noise_offset_samples"],
                rows_by_level[-20]["noise_offset_samples"],
            )
            self.assertAlmostEqual(
                float(rows_by_level[0]["scaled_noise_chunk_power"]),
                0.25,
            )
            self.assertAlmostEqual(
                float(rows_by_level[-20]["scaled_noise_chunk_power"]),
                25.0,
            )

            manifest = json.loads(
                (maxfreq_dir / "preprocess_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["noise"]["scaling_mode"], "fixed_global_rms"
            )
            self.assertTrue(manifest["noise"]["pair_noise_across_snr"])
            self.assertEqual(
                manifest["noise"]["noise_seed_scope"],
                "shared_across_sources",
            )


if __name__ == "__main__":
    unittest.main()
