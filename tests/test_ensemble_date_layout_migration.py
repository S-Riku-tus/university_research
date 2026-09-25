import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from migrate_ensemble_date_layout import (
    migrate_ensemble_dates,
    migration_plan,
    rewrite_references_from_report,
)


class EnsembleDateLayoutMigrationTest(unittest.TestCase):
    def _ensemble_root(self, root):
        result = root / "experiment" / "regression_result" / "npy" / "ensemble"
        result.mkdir(parents=True)
        return result

    def test_exact_date_and_suffix_are_moved_without_merging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ensemble = self._ensemble_root(root)
            exact = ensemble / "20260626"
            suffix = ensemble / "20260626_cf3m"
            (exact / "condition").mkdir(parents=True)
            (suffix / "condition").mkdir(parents=True)
            (exact / "condition" / "run_manifest.json").write_text(
                json.dumps({"result_date_dir": "20260626/onb"}),
                encoding="utf-8",
            )
            (suffix / "condition" / "note.txt").write_text(
                "20260626_cf3m", encoding="utf-8"
            )

            planned = migration_plan(root)
            self.assertEqual(len(planned), 2)
            report = migrate_ensemble_dates(root, apply=True)

            self.assertEqual(report["status"], "complete")
            self.assertFalse(exact.exists())
            self.assertFalse(suffix.exists())
            day = ensemble / "202606" / "26"
            self.assertTrue((day / "condition" / "run_manifest.json").is_file())
            self.assertTrue((day / "cf3m" / "condition" / "note.txt").is_file())
            manifest = json.loads(
                (day / "condition" / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["result_date_dir"], "202606/26/onb")
            self.assertEqual(
                (day / "cf3m" / "condition" / "note.txt").read_text(encoding="utf-8"),
                "202606/26/cf3m",
            )
            self.assertEqual(migration_plan(root), [])

    def test_existing_destination_is_rejected_before_moving(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ensemble = self._ensemble_root(root)
            (ensemble / "20260924").mkdir()
            (ensemble / "202609" / "24").mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                migration_plan(root)
            self.assertTrue((ensemble / "20260924").is_dir())

    def test_external_rewrite_changes_paths_but_not_standalone_run_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "experiment" / "regression_result" / "npy" / "ensemble" / "20260924"
            target = root / "experiment" / "regression_result" / "npy" / "ensemble" / "202609" / "24"
            report = root / "migration_report.json"
            report.write_text(json.dumps({
                "status": "complete",
                "moves": [{"from": str(source), "to": str(target)}],
            }), encoding="utf-8")
            document = root / "note.md"
            document.write_text(
                "run_id=20260924\npath=ensemble/20260924/onb\n",
                encoding="utf-8",
            )

            rewritten = rewrite_references_from_report(report, [document])

            self.assertEqual(len(rewritten), 1)
            self.assertEqual(
                document.read_text(encoding="utf-8"),
                "run_id=20260924\npath=ensemble/202609/24/onb\n",
            )


if __name__ == "__main__":
    unittest.main()
