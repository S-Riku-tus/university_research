"""Verify raw-summary persistence, genuine fixed-core weights and day labels."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from utils.calculation.regression_detection_metrics import RegressionDetectionMetrics
from utils.calculation.source_day_metrics import source_day_metric_rows, metric_delta_rows
from utils.config.onb_defaults import onb_model_specs
from utils.config.parameter_sets import resolve_parameter_set
from utils.ensemble.ensemble_runtime import EnsembleManager
from utils.experiment.run_helpers import set_global_seed
from utils.models.regression.base_regression import RegressionModelMaker
from utils.training.fitted_artifacts import begin_fitted_state, save_and_verify_model
from utils.training.model_training import ModelTrainer
from utils.explainability.training_integration import maybe_explain_trained_model

KEYS = ["randomforest", "conformer", "alexnet", "hgb", "extra_trees"]
STRATEGIES = ["original3_mse", "original3_hgb", "original3_extra_trees", "original3_hgb_extra_trees"]


class OnbFiveModelTest(unittest.TestCase):
    def test_summary_models_train_on_raw_and_reload_without_pca(self):
        rng = np.random.default_rng(31)
        x = rng.lognormal(-20, 1, (24, 224, 224, 1)).astype(np.float32)
        y = np.linspace(10000, 500000, len(x))
        scaler = MinMaxScaler().fit(y.reshape(-1, 1))
        trainer, maker = ModelTrainer(42), RegressionModelMaker(x.shape[1:])
        set_global_seed(77)
        specs = resolve_parameter_set(onb_model_specs(["hgb", "extra_trees"]), {"models": {
            "hgb": {"max_iter": 3, "min_samples_leaf": 2},
            "extra_trees": {"n_estimators": 4}}})
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temp)
            manifest = begin_fitted_state(directory, "test", 1, 77, ["wav"], {}, None, scaler)
            for spec in specs:
                model, _ = trainer.train_one_model(spec, maker, x, scaler.transform(y.reshape(-1, 1)),
                    np.asarray(["PCA_MUST_NOT_BE_USED"]), 1)
                self.assertEqual(model.named_steps["regressor"].random_state, 77)
                prediction = trainer.predict_one_model(spec, model, x[:4], None, scaler)
                self.assertTrue(np.isfinite(prediction).all())
                record = save_and_verify_model(directory, manifest, spec, model, maker, trainer, x[:4], scaler, None)
                self.assertTrue(record["verification"]["passed"])
                self.assertEqual(record["fitted_random_state"], 77)
                self.assertEqual(record["input_representation"], "raw_power_to_frequency34")
                indices = np.arange(0, len(x), 4)
                pred = trainer.predict_one_model(spec, model, x[indices], None, scaler)
                maybe_explain_trained_model(spec, model, scaler, x[indices], y[indices], pred,
                    225000., directory, 1, "maxfreq=3kHz", {
                        "enabled": True, "methods_by_model": {spec["key"]: ["group_occlusion"]},
                        "frequency_bands_hz": [[512, 1000], [1000, 2000], [2000, 3000]],
                        "max_samples_per_fold": 1, "save_maps": False, "onb_band_frac": .1,
                    })
                explanation = directory / "explainability/fold1" / spec["key"]
                self.assertTrue((explanation / "group_mask_performance.csv").exists())
                if spec["key"] == "extra_trees":
                    self.assertTrue((explanation / "summary_feature_importance.csv").exists())
            self.assertFalse((directory / "pca.joblib").exists())

    def test_shared_oof_fit_preserves_three_ratios_and_member_counts(self):
        y = np.arange(1., 61.)*1000
        predictions = {"randomforest": y+20000+4000*np.sin(y), "conformer": y+1000,
                       "alexnet": y-400, "hgb": y+np.cos(y)*100,
                       "extra_trees": y-np.cos(y)*100}
        samples = [{"heat_flux": v, "source_wav_id": f"wav{i//4}", **{k: p[i] for k, p in predictions.items()}} for i, v in enumerate(y)]
        audit = {"test_used": False, "method": "chunk_kfold", "samples": samples, "folds": [
            {"fold": j+1, "fit_indices": [i for i in range(60) if i%3 != j],
             "validation_indices": [i for i in range(60) if i%3 == j]} for j in range(3)]}
        manager = EnsembleManager({"enabled_strategy_names": STRATEGIES}, KEYS)
        specs = onb_model_specs(KEYS)
        manager.validate(specs)
        run = manager.create_run(specs)
        fit = run.fit_weights_from_internal_audit(audit, 1)
        outputs = run.combine_predictions(predictions, {}, 1, fit)
        core = np.asarray([outputs["ensemble__original3_mse"]["weights"][k] for k in KEYS[:3]])
        for name, members in [("original3_hgb", 4), ("original3_extra_trees", 4), ("original3_hgb_extra_trees", 5)]:
            weights = outputs[f"ensemble__{name}"]["weights"]
            current = np.asarray([weights[k] for k in KEYS[:3]])
            np.testing.assert_allclose(current/current.sum(), core, atol=1e-9)
            self.assertGreaterEqual(current.sum(), .25-1e-10)
            self.assertTrue(np.all(current>0))
            self.assertEqual(sum(w>0 for w in weights.values()), members)
        # Outer targets are absent from the API, and tampered OOF coverage is rejected.
        audit["folds"][0]["fit_indices"].append(audit["folds"][0]["validation_indices"][0])
        with self.assertRaisesRegex(ValueError, "Shared"):
            run.fit_weights_from_internal_audit(audit, 1)

    def test_source_thresholds_do_not_collapse_to_combined_average(self):
        y = np.asarray([210., 230., 260., 280.])
        before, after = np.asarray([200., 210., 250., 260.]), np.asarray([200., 230., 250., 280.])
        metadata = [{"experiment_name": "combined", "source_experiment_name": day, "source_wav_id": f"{day}:{i}"}
                    for i, day in enumerate(["day-a", "day-a", "day-b", "day-b"])]
        rows = source_day_metric_rows(y, {"ensemble__original3_mse": before, "five": after}, metadata,
                                     {"day-a": 220., "day-b": 270.}, 1)
        pooled = [r for r in rows if r["source_day"] == "pooled_source_day_thresholds"]
        self.assertEqual([(r["fp"], r["fn"], r["recall"]) for r in pooled], [(0, 2, 0.), (0, 0, 1.)])
        changes = metric_delta_rows(rows)
        recall = next(r for r in changes if r["source_day"] == "pooled_source_day_thresholds" and r["metric"] == "recall")
        self.assertEqual(recall["percentage_point_change"], 100.)
        scalar = RegressionDetectionMetrics().detection_metrics_binary(y, after, 245.)
        self.assertEqual(scalar["tp"], 2)
        self.assertEqual(scalar["n_post_onb"], 2)
        self.assertEqual(scalar["fp"], 0)

    def test_missing_additions_rejected_before_training(self):
        manager = EnsembleManager({"enabled_strategy_names": ["original3_hgb_extra_trees"]}, KEYS)
        with self.assertRaisesRegex(ValueError, "Missing fixed-core"):
            manager.validate(onb_model_specs())


if __name__ == "__main__":
    unittest.main()
