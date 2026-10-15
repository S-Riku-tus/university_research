import csv
import json
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.dataloading import waterflow_preprocessing as preprocessing  # noqa: E402
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


class WaterflowResumeTest(unittest.TestCase):
    def _arguments(self, root, save_npy=True, save_png=True):
        return {
            "file_path": str(root / "index=1.123.wav"),
            "max_freq_hz": 3000,
            "chunk_seconds": 8 / 44100,
            "context": {
                "experiment_name": "synthetic_experiment",
                "folder_path": str(root / "recordings"),
                "waterflow_path": str(root / "noise.wav"),
                "heat_flux_csv_path": str(root / "labels.csv"),
                "heat_flux_label_by_index": {},
                "base_npy_save_folder_path": str(root / "npy"),
                "base_png_save_folder_path": str(root / "png"),
                "save_date": 20261014,
                "script_path": str(REPO_ROOT / "code" / "2.run_npy_waterflow_2つhighpass.py"),
            },
            "config": {
                "fp": 500, "fs": 400, "gpass": 1, "gstop": 40,
                "sample_number": 2, "audio_samples_used": 16,
                "save_npy": save_npy, "save_spectrogram_png": save_png,
                "png_with_axes": False, "noise_scaling_mode": "fixed_global_rms",
                "fixed_reference_signal_rms": 0.5,
                "fixed_reference_provenance": "synthetic test scalar",
                "randomize_noise_offset_per_chunk": True,
                "pair_noise_across_snr": True,
                "noise_seed_scope": "shared_across_sources",
                "noise_level_folder_prefix": "reference_SNR",
                "random_seed": 42, "include_source_in_filename": True,
                "saved_array_dtype": "float32", "max_chunks_per_source": None,
                "audio_loading": "prefiltered synthetic arrays",
                "resume_existing": True,
            },
            "snr_db": [None, 0, -20],
            "prefiltered_signal": np.linspace(0.1, 1.6, 16),
            "prefiltered_waterflow_noise": np.linspace(0.1, 6.4, 64),
        }

    def _snapshot(self, root):
        return {
            str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()
        }

    def _remove_manifest_row(self, path, chunk_index):
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames
            rows = [row for row in reader if int(row["chunk_index"]) != chunk_index]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_completed_resume_skips_without_rewriting_outputs_or_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._arguments(root)
            save_spectrogram_chunks_with_snr(**args)
            snapshot = self._snapshot(root)
            with patch.object(preprocessing, "calc_stft", side_effect=AssertionError("completed chunk recomputed")):
                counts = save_spectrogram_chunks_with_snr(**args)
            self.assertEqual(counts, {
                "skipped_chunks": 6, "npy_written": 0, "png_written": 0,
                "manifest_repaired": 0,
            })
            self.assertEqual(snapshot, self._snapshot(root))

    def test_resume_repairs_partial_outputs_and_unrecorded_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._arguments(root)
            save_spectrogram_chunks_with_snr(**args)
            expected = self._snapshot(root)
            for kind in ("npy", "png"):
                self._remove_manifest_row(root / kind / "maxfreq=3kHz" / "heatflux_no_noise" / "chunk_manifest.csv", 1)
                self._remove_manifest_row(root / kind / "maxfreq=3kHz" / "heatflux_reference_SNR=0" / "chunk_manifest.csv", 1)
            missing_png = root / "png/maxfreq=3kHz/heatflux_reference_SNR=0/123_src-index=1.123_chunk-0001.png"
            missing_png.unlink()
            for kind in ("npy", "png"):
                broken = root / kind / "maxfreq=3kHz" / "heatflux_reference_SNR=-20" / f"123_src-index=1.123_chunk-0000.{kind}"
                broken.write_bytes(broken.read_bytes()[:32])
            damaged = self._snapshot(root)
            with patch.object(preprocessing, "calc_stft", wraps=preprocessing.calc_stft) as stft:
                counts = save_spectrogram_chunks_with_snr(**args)
            self.assertEqual(stft.call_count, 1)
            self.assertEqual(counts, {
                "skipped_chunks": 3, "npy_written": 1, "png_written": 2,
                "manifest_repaired": 1,
            })
            restored = self._snapshot(root)
            self.assertEqual({name: item[0] for name, item in expected.items()},
                             {name: item[0] for name, item in restored.items()})
            for name, original in expected.items():
                if name.endswith((".npy", ".png")) and damaged.get(name, (None,))[0] == original[0]:
                    self.assertEqual(original, restored[name], name)

    def test_resume_with_only_one_output_format(self):
        for kind in ("npy", "png"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self._arguments(root, save_npy=kind == "npy", save_png=kind == "png")
                save_spectrogram_chunks_with_snr(**args)
                counts = save_spectrogram_chunks_with_snr(**args)
                self.assertEqual(counts["skipped_chunks"], 6)
                next((root / kind).rglob(f"*.{kind}")).unlink()
                counts = save_spectrogram_chunks_with_snr(**args)
                self.assertEqual(counts[f'{kind}_written'], 1)
                self.assertEqual(counts["skipped_chunks"], 5)

    def test_resume_rejects_changed_generation_settings_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._arguments(root)
            save_spectrogram_chunks_with_snr(**args)
            expected = self._snapshot(root)
            for key, value in (("fixed_reference_signal_rms", 0.6), ("random_seed", 43),
                               ("gpass", 2), ("png_with_axes", True)):
                with self.subTest(key=key):
                    changed = {**args, "config": {**args["config"], key: value}}
                    with self.assertRaisesRegex(ValueError, "Cannot resume"):
                        save_spectrogram_chunks_with_snr(**changed)
                    self.assertEqual(expected, self._snapshot(root))

    def test_disabling_resume_retains_regeneration_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._arguments(Path(tmp), save_png=False)
            save_spectrogram_chunks_with_snr(**args)
            args["config"].pop("resume_existing")
            with patch.object(preprocessing, "calc_stft", wraps=preprocessing.calc_stft) as stft:
                counts = save_spectrogram_chunks_with_snr(**args)
            self.assertEqual(stft.call_count, 6)
            self.assertEqual(counts["npy_written"], 6)
            self.assertEqual(counts["skipped_chunks"], 0)

    def test_resume_preserves_global_reference_and_rejects_different_rms(self):
        script = runpy.run_path(str(REPO_ROOT / "code" / "2.run_npy_waterflow_2つhighpass.py"))
        with tempfile.TemporaryDirectory() as tmp:
            context = {"base_npy_save_folder_path": tmp}
            rms_by_path = {str(REPO_ROOT / "synthetic.wav"): 0.5}
            write = script["_write_reference_manifest"]
            write(context, 0.5, rms_by_path)
            original = self._snapshot(Path(tmp))
            write(context, 0.5, rms_by_path)
            self.assertEqual(original, self._snapshot(Path(tmp)))
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                write(context, 0.6, rms_by_path)
            self.assertEqual(original, self._snapshot(Path(tmp)))

    def test_failed_manifest_update_keeps_previous_manifest_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"sample_filename": "123.npy", "source_wav_id": "source", "chunk_index": 0}
            preprocessing.write_chunk_manifest(tmp, [row])
            original = self._snapshot(root)
            with patch.object(preprocessing.os, "replace", side_effect=OSError("interrupted update")):
                with self.assertRaisesRegex(OSError, "interrupted update"):
                    preprocessing.write_chunk_manifest(tmp, [{**row, "chunk_index": 1}])
            self.assertEqual(original, self._snapshot(root))


if __name__ == "__main__":
    unittest.main()
