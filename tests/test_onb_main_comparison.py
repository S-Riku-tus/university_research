"""Check main-table label bases, comparisons, coverage and full reload checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code"))
from utils.calculation.onb_comparison_report import comparison_tables, write_comparison_report, POOLED
from utils.calculation.source_day_metrics import source_day_metric_rows
from utils.config.onb_defaults import onb_model_specs
from utils.config.parameter_sets import resolve_parameter_set
from utils.experiment.onb_main_comparison import validate_main_comparison_jobs
from utils.experiment.run_helpers import run_config_digest
from utils.models.regression.base_regression import RegressionModelMaker
from utils.training.fitted_artifacts import begin_fitted_state, save_and_verify_model, verify_reloaded_evaluation_predictions
from utils.training.model_training import ModelTrainer

PROTOCOL = json.loads((ROOT/"configs/experiments/2026-10-06_onb_main_comparison_protocol.json").read_text(encoding="utf-8"))


def fixture():
    y = np.asarray([210., 230., 260., 280.])*1000
    before, after = np.asarray([200., 210., 250., 260.])*1000, np.asarray([200., 230., 250., 280.])*1000
    metadata = [{"experiment_name": "combined", "source_experiment_name": day, "source_wav_id": day}
                for day in ["day-a", "day-a", "day-b", "day-b"]]
    predictions = {k: before for k in PROTOCOL["model_keys"]}
    predictions.update({"ensemble__"+k: after for k in PROTOCOL["ensemble_names"]})
    predictions[PROTOCOL["baseline_model_key"]] = before
    rows = []
    for noise in ["clean", "-20"]:
        current = {k: p+(1000 if noise == "-20" else 0) for k, p in predictions.items()}
        rows.extend({"seed": 42, "noise": noise, **r} for r in source_day_metric_rows(y, current, metadata, {"day-a": 220000., "day-b": 270000.}, 1))
    return rows


class OnbMainComparisonTest(unittest.TestCase):
    def test_old_probe_only_states_cannot_resume_as_full_evaluation_verified_states(self):
        spec = [{"key": "hgb", "kind": "sklearn_summary", "builder_params": {}}]
        config = {"output": {"save_fitted_artifacts": True, "verify_reloaded_artifacts": True}}
        old = run_config_digest(config, {}, spec, "hgb", True)
        full = {"output": {**config["output"], "verify_reloaded_evaluation_predictions": True}}
        new = run_config_digest(full, {}, spec, "hgb", True)
        self.assertNotEqual(old, new)
        report = {"output": {**full["output"], "main_comparison_report": True, "main_comparison_protocol": PROTOCOL}}
        self.assertNotEqual(new, run_config_digest(report, {}, spec, "hgb", True))

    def test_main_tables_keep_daily_thresholds_units_and_own_clean_degradation(self):
        tables = comparison_tables(fixture(), PROTOCOL)
        primary = PROTOCOL["primary_model_key"]
        delta = next(r for r in tables["main_deltas.csv"] if r["noise"] == "clean" and r["source_day"] == POOLED
                     and r["model_key"] == primary and r["baseline_key"] == PROTOCOL["baseline_model_key"])
        self.assertEqual(delta["delta_fn"], -2)
        self.assertEqual(delta["delta_recall_pp"], 100.)
        self.assertAlmostEqual(delta["delta_rmse_all_kW_m2"], np.sqrt(50)-np.sqrt(250))
        days = [r for r in tables["q100_by_source_day.csv"] if r["model_key"] == primary and r["noise"] == "clean"]
        self.assertEqual([r["q100_kW_m2"] for r in days], [230., 280.])
        self.assertTrue(all(r["g100_kW_m2"] == 10. for r in days))
        pooled = next(r for r in tables["main_metrics.csv"] if r["source_day"] == POOLED)
        self.assertTrue(np.isnan(pooled["q100_kW_m2"]))
        noisy = next(r for r in tables["noise_degradation.csv"] if r["model_key"] == primary and r["source_day"] == POOLED)
        self.assertAlmostEqual(noisy["clean_rmse_kW_m2"], np.sqrt(50))
        self.assertAlmostEqual(noisy["increase_rmse_from_clean_kW_m2"], np.sqrt(41)-np.sqrt(50))

    def test_missing_methods_duplicate_rows_and_average_threshold_basis_are_rejected(self):
        rows = fixture()
        with self.assertRaisesRegex(ValueError, "Missing comparison"):
            comparison_tables([r for r in rows if r["model_key"] != "extra_trees"], PROTOCOL)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            comparison_tables(rows+[rows[0]], PROTOCOL)
        with self.assertRaisesRegex(ValueError, "source-day"):
            comparison_tables([{**rows[0], "threshold_basis": "combined-average"}], PROTOCOL)

    def test_report_does_not_label_smoke_as_full_150_epoch_result(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_comparison_report(temp, fixture(), PROTOCOL, {}, [],
                {"status": "passed", "scope": "software smoke; not the standard 150-epoch research result"})
            self.assertIn("software smoke", path.read_text(encoding="utf-8"))
            self.assertTrue((path.parent/"noise_degradation.csv").exists())

    def test_missing_noise_is_rejected_before_reading_or_training(self):
        config = {"output": {"main_comparison_protocol": PROTOCOL}, "data": {"experiment_names": ["combined"],
                  "max_freq_hz_list": ["maxfreq=3kHz"], "noise_dir_names": ["heatflux_no_noise", "heatflux_reference_SNR=-20"]}}
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            validate_main_comparison_jobs([{"experiment_name": "combined", "max_freq_hz": "maxfreq=3kHz", "noise_dir_name": "heatflux_no_noise"}], config)

    def test_full_reload_checks_all_conditions_and_rejects_changed_predictions(self):
        rng = np.random.default_rng(19)
        x = rng.lognormal(-20, 1, (16, 224, 224, 1)).astype(np.float32)
        scaler = MinMaxScaler().fit(np.arange(16.).reshape(-1, 1)*10000)
        trainer, maker = ModelTrainer(42), RegressionModelMaker(x.shape[1:])
        spec = resolve_parameter_set(onb_model_specs(["extra_trees"]), {"models": {"extra_trees": {"n_estimators": 4}}})[0]
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
            manifest = begin_fitted_state(temp, "fit", 1, 42, ["wav"], {}, None, scaler)
            model, _ = trainer.train_one_model(spec, maker, x, np.linspace(0, 1, 16), None, 1)
            save_and_verify_model(temp, manifest, spec, model, maker, trainer, x[:2], scaler, None)
            p = trainer.predict_one_model(spec, model, x[:3], None, scaler)
            checks = verify_reloaded_evaluation_predictions(temp, manifest, spec, maker, trainer,
                [("clean", x[:3], p), ("-20", x[:3], p)])
            self.assertEqual(len(checks), 2)
            self.assertEqual(manifest["models"]["extra_trees"]["evaluation_verification"]["total_predictions"], 6)
            bad = p.copy()
            bad[0] += 10000
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                verify_reloaded_evaluation_predictions(temp, manifest, spec, maker, trainer, [("bad", x[:3], bad)])


if __name__ == "__main__":
    unittest.main()
