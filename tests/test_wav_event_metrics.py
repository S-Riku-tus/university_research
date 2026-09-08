import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.calculation.wav_event_metrics import (  # noqa: E402
    aggregate_oof_predictions_by_wav,
    build_fold_prediction_rows,
    load_sample_metadata_without_arrays,
    onb_transition_summary_rows,
    read_saved_fold_predictions,
    save_wav_event_evaluation,
    summarize_pooled_oof_wav_metrics,
)


class WavEventMetricsTest(unittest.TestCase):
    def _chunk_rows(self):
        sequences = {
            "wav-a": (0.0, [10.0, 20.0, 30.0, 40.0], 1),
            "wav-b": (100.0, [90.0, 110.0, 120.0, 80.0], 2),
            "wav-c": (200.0, [110.0, 80.0, 120.0, 130.0], 3),
        }
        rows = []
        sample_index = 0
        for source_wav_id, (target, predictions, fold) in sequences.items():
            metadata = []
            for chunk_index in range(4):
                metadata.append({
                    "sample_filename": f"{source_wav_id}-{chunk_index}.npy",
                    "experiment_name": "experiment",
                    "source_wav_id": source_wav_id,
                    "source_wav_name": f"{source_wav_id}.wav",
                    "chunk_index": chunk_index,
                    "chunk_start_seconds": float(chunk_index),
                    "chunk_duration_seconds": 1.0,
                })
            fold_rows = build_fold_prediction_rows(
                val_indices=np.arange(4),
                y_true=np.full(4, target),
                predictions={"model": predictions},
                sample_metadata=metadata,
                fold=fold,
            )
            for row in fold_rows:
                row["sample_index"] = sample_index
                sample_index += 1
            rows.extend(fold_rows)
        return rows

    def test_wav_aggregation_and_predicted_event_runs(self):
        wav_rows = aggregate_oof_predictions_by_wav(
            self._chunk_rows(),
            ["model"],
            threshold=100.0,
            aggregations=("mean", "median", "p90"),
        )
        self.assertEqual(len(wav_rows), 3)

        wav_b = next(row for row in wav_rows if row["source_wav_id"] == "wav-b")
        self.assertAlmostEqual(wav_b["model_pred_median"], 100.0)
        self.assertEqual(wav_b["model_predicted_positive_chunk_count"], 2)
        self.assertAlmostEqual(wav_b["model_predicted_positive_chunk_rate"], 0.5)
        self.assertEqual(wav_b["model_predicted_positive_run_count"], 1)
        self.assertAlmostEqual(wav_b["model_first_predicted_positive_seconds"], 1.0)
        self.assertAlmostEqual(
            wav_b["model_longest_predicted_positive_run_seconds"], 2.0
        )

        wav_c = next(row for row in wav_rows if row["source_wav_id"] == "wav-c")
        self.assertEqual(wav_c["model_predicted_positive_run_count"], 2)
        self.assertAlmostEqual(
            wav_c["model_longest_predicted_positive_run_seconds"], 2.0
        )

    def test_metrics_are_pooled_over_wavs(self):
        wav_rows = aggregate_oof_predictions_by_wav(
            self._chunk_rows(), ["model"], threshold=100.0
        )
        metrics = summarize_pooled_oof_wav_metrics(
            wav_rows,
            ["model"],
            threshold=100.0,
            band_frac=0.10,
            aggregations=("median",),
        )
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["evaluation_unit"], "source_wav_pooled_oof")
        self.assertEqual(metrics[0]["n_wavs"], 3)
        self.assertTrue(np.isfinite(metrics[0]["r2"]))

    def test_onb_transition_separates_early_alarm_and_post_onb_delay(self):
        wav_rows = []
        for index, (target, prediction) in enumerate(
            [(0.0, 110.0), (100.0, 80.0), (200.0, 120.0), (300.0, 130.0)]
        ):
            wav_rows.append({
                "experiment_name": "experiment",
                "source_wav_id": f"wav-{index}",
                "y_true": target,
                "model_pred_median": prediction,
            })

        rows = onb_transition_summary_rows(
            wav_rows,
            ["model"],
            threshold=100.0,
            aggregations=("median",),
            persistence_wavs=(1, 2),
        )
        raw = next(row for row in rows if row["persistence_wavs"] == 1)
        stable = next(row for row in rows if row["persistence_wavs"] == 2)

        self.assertEqual(raw["signed_transition_step_error"], -1)
        self.assertAlmostEqual(raw["signed_transition_heat_flux_error"], -100.0)
        self.assertEqual(raw["false_positive_wavs_before_onb"], 1)
        self.assertEqual(raw["post_onb_detection_delay_steps"], 1)
        self.assertEqual(stable["signed_transition_step_error"], 1)
        self.assertEqual(stable["post_onb_detection_delay_steps"], 1)

    def test_legacy_fold_csv_is_enriched_from_manifest_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            pred_dir = root / "fold_pred"
            data_dir.mkdir()
            pred_dir.mkdir()
            manifest_rows = []
            for index in range(2):
                filename = f"{index}_sample.npy"
                (data_dir / filename).touch()
                manifest_rows.append({
                    "sample_filename": filename,
                    "heat_flux": [0.0, 100.00000006][index],
                    "experiment_name": "experiment",
                    "source_wav_id": f"wav-{index}",
                    "source_wav_name": f"wav-{index}.wav",
                    "chunk_index": 0,
                    "chunk_start_seconds": 0,
                    "chunk_duration_seconds": 1,
                })
            with (data_dir / "chunk_manifest.csv").open(
                "w", newline="", encoding="utf-8"
            ) as output:
                writer = csv.DictWriter(output, fieldnames=list(manifest_rows[0]))
                writer.writeheader()
                writer.writerows(manifest_rows)
            with (pred_dir / "pred_f1_no_noise.csv").open(
                "w", newline="", encoding="utf-8"
            ) as output:
                writer = csv.writer(output)
                writer.writerow(["sample_index", "y_true", "model"])
                writer.writerow([0, 0, 1])
                writer.writerow([1, 100, 99])

            metadata = load_sample_metadata_without_arrays(data_dir)
            rows, model_keys = read_saved_fold_predictions(
                pred_dir, "no_noise", metadata
            )
            self.assertEqual(model_keys, ["model"])
            self.assertEqual(rows[1]["source_wav_id"], "wav-1")
            self.assertEqual(rows[1]["chunk_start_seconds"], 0.0)
            self.assertEqual(rows[1]["y_true"], 100.00000006)

    def test_output_labels_event_rows_as_predictions_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = save_wav_event_evaluation(
                save_path=temp_dir,
                snr_value="no_noise",
                chunk_rows=self._chunk_rows(),
                model_keys=["model"],
                threshold=100.0,
                band_frac=0.10,
                aggregations=("median",),
                primary_aggregation="median",
            )
            with Path(result["event_path"]).open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["ground_truth_event_annotation_available"], "0")
            self.assertEqual(
                rows[0]["interpretation"],
                "predicted_heat_flux_threshold_crossing_only",
            )
            with Path(result["transition_path"]).open(encoding="utf-8") as source:
                transition_rows = list(csv.DictReader(source))
            self.assertEqual(
                transition_rows[0]["evaluation_unit"],
                "threshold_defined_onb_transition",
            )
            self.assertEqual(
                transition_rows[0]["temporal_delay_seconds_available"], "0"
            )


if __name__ == "__main__":
    unittest.main()
