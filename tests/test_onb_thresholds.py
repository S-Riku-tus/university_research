import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.experiment.onb_thresholds import (  # noqa: E402
    ONB_THRESHOLD_RECORDS,
    CSV_ONB_THRESHOLD_RECORDS,
    onb_threshold_by_experiment,
    onb_threshold_provenance_by_experiment,
    resolve_csv_onb_threshold_record,
)


class OnbThresholdRegistryTest(unittest.TestCase):
    def test_registry_uses_confirmed_current_run_values(self):
        self.assertEqual(
            onb_threshold_by_experiment(),
            {
                "2025.06.11_0.3_2": 221505.1102,
                "2025.06.18_0.3_3": 271677.6816,
                "2025.06.11_0.3_2_6.18_0.3_3": 246591.3959,
                "2025.07.09_0.3_1": 571694.252491167,
            },
        )

    def test_every_threshold_has_decision_record_provenance(self):
        for experiment_name, record in ONB_THRESHOLD_RECORDS.items():
            with self.subTest(experiment_name=experiment_name):
                self.assertTrue(record["definition"])
                source_path = record["source"].split("#", 1)[0]
                self.assertTrue((REPO_ROOT / source_path).is_file())
                self.assertGreater(record["threshold"], 0)


class CsvOnbThresholdTest(unittest.TestCase):
    day = "2026.10.07_0.3_1"

    def csv_path(self, root):
        path = Path(root) / CSV_ONB_THRESHOLD_RECORDS[self.day]["heat_flux_csv"]
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def test_missing_labels_remain_pending_without_changing_old_audit_scope(self):
        with tempfile.TemporaryDirectory() as tmp, patch("utils.experiment.onb_thresholds.REPO_ROOT", Path(tmp)):
            self.assertNotIn(self.day, onb_threshold_by_experiment())
            self.assertIsNone(onb_threshold_by_experiment(csv_experiments=[self.day])[self.day])
            record = onb_threshold_provenance_by_experiment(csv_experiments=[self.day])[self.day]
            self.assertEqual(record["status"], "waiting_for_heat_flux_csv")

    def test_observed_voltage_uses_exact_q_not_previous_row_or_filename_integer(self):
        with tempfile.TemporaryDirectory() as tmp, patch("utils.experiment.onb_thresholds.REPO_ROOT", Path(tmp)):
            self.csv_path(tmp).write_text(
                "volt,q,q(e)\n1.1V,100123.456,1.00e5\n1.2v,200987.654321,2.01e5\n1.3V,300123.456,3.00e5\n",
                encoding="utf-8-sig",
            )
            self.assertEqual(onb_threshold_by_experiment(csv_experiments=[self.day])[self.day], 200987.654321)
            record = resolve_csv_onb_threshold_record(self.day, tmp)
            self.assertEqual(record["csv_row_index"], 2)
            self.assertEqual(record["status"], "resolved_from_heat_flux_csv")
            self.assertEqual(len(record["heat_flux_csv_sha256"]), 64)

    def test_ambiguous_missing_or_invalid_onb_rows_are_rejected(self):
        cases = [
            "volt,q\n1.1V,100\n",
            "volt,q\n1.2V,100\n1.2V,200\n",
            "volt,q\n1.2V,nan\n",
            "volt,q\n1.2V,0\n",
            "volt,q\n1.2V,-1\n",
            "volt,q\n1.2V,inf\n",
            "volt,q\n1.2V,not-a-number\n",
            "voltage,heat_flux\n1.2V,100\n",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.csv_path(tmp)
            for content in cases:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        resolve_csv_onb_threshold_record(self.day, tmp)

    def test_onb_uses_established_csv_location_with_processing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = self.csv_path(tmp)
            old.write_text('volt,q\n1.2V,111\n', encoding='utf-8')
            config = Path(tmp) / 'configs/datasets' / (self.day + '_heatflux_processing.json')
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({'experiment_name':self.day, 'stability_window_seconds':35}))
            record = resolve_csv_onb_threshold_record(self.day, tmp)
            self.assertEqual(record['threshold'], 111)
            self.assertEqual(record['heat_flux_csv'], CSV_ONB_THRESHOLD_RECORDS[self.day]['heat_flux_csv'])


if __name__ == "__main__":
    unittest.main()
