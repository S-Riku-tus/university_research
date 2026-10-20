"""Check CNN-only candidate routing, OOF parity, and real model persistence."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_ensemble_regression_onb as onb
from utils.config.onb_defaults import onb_model_specs
from utils.config.parameter_sets import build_frequency_parameter_execution_plans, resolve_parameter_set
from utils.ensemble.ensemble_runtime import EnsembleManager
from utils.models.regression.base_regression import RegressionModelMaker
from utils.training.fitted_artifacts import begin_fitted_state, save_and_verify_model
from utils.training.model_training import ModelTrainer


class CnnThreeTest(unittest.TestCase):
    def test_candidates_stay_model_wise_and_singletons_enable_joint_run(self):
        onb.validate_validation_config(onb.MODEL_SPECS)
        plan = onb.parameter_plan_for_max_freq("maxfreq=3kHz")
        self.assertTrue(plan["parameter_search_enabled"])
        self.assertEqual(len(plan["parameter_sets"]), 72)  # Sum of grids, not 24**3.
        for key in onb.MODEL_KEYS:
            candidates = [p for p in plan["parameter_sets"] if p["model_key"] == key]
            self.assertEqual(len(candidates), 24)
            self.assertEqual(len({(p["models"][key]["lr"], p["models"][key]["batch_size"]) for p in candidates}), 24)
            for candidate in candidates:
                specs = resolve_parameter_set(onb.MODEL_SPECS, candidate)
                self.assertEqual([s["key"] for s in specs], [key])
                self.assertIn(specs[0]["batch_size"], [4, 8, 16, 32])

        config = deepcopy(onb.VALIDATION_CONFIG["models"]["parameter_sets"])
        for key in onb.MODEL_KEYS:
            config["by_max_freq_hz"]["maxfreq=3kHz"][key]["lr"] = [0.001]
            config["by_max_freq_hz"]["maxfreq=3kHz"][key]["batch_size"] = [8]
        fixed = build_frequency_parameter_execution_plans(config, onb.MODEL_KEYS, ["maxfreq=3kHz"])["maxfreq=3kHz"]
        self.assertFalse(fixed["parameter_search_enabled"])
        self.assertEqual(len(fixed["parameter_sets"]), 1)
        self.assertEqual([s["key"] for s in resolve_parameter_set(onb.MODEL_SPECS, fixed["parameter_sets"][0])], onb.MODEL_KEYS)
        previous = json.loads((ROOT / "configs/experiments/2026-10-19_before_cnn_three_validation_config.json").read_text(encoding="utf-8"))["validation_config"]
        for section in ("data", "learning_policy", "thresholds", "acoustic_selection", "run"):
            self.assertEqual(onb.VALIDATION_CONFIG[section], previous[section])

    def test_cnn_shared_oof_uses_the_same_weight_rules_as_current_three(self):
        old_keys = ["randomforest", "conformer", "alexnet"]
        new_keys = ["alexnet", "resnet50", "vgg16"]
        y = np.linspace(1000., 12000., 12)
        columns = [y + 500., y - 200., y + 100*np.sin(y)]
        results = []
        for keys, names in [(old_keys, ["original3_mse", "original3_performance", "simple_equal"]),
                            (new_keys, ["cnn3_mse", "cnn3_performance", "simple_equal"])]:
            manager = EnsembleManager({"enabled_strategy_names": names}, keys)
            specs = onb_model_specs(keys)
            manager.validate(specs)
            run = manager.create_run(specs)
            predictions = dict(zip(keys, columns))
            audit = {"test_used": False, "method": "chunk_kfold", "samples": [
                {"heat_flux": value, "source_wav_id": f"wav{i//3}", **{k: p[i] for k, p in predictions.items()}}
                for i, value in enumerate(y)], "folds": [
                    {"fold": f+1, "fit_indices": [i for i in range(12) if i%3 != f],
                     "validation_indices": [i for i in range(12) if i%3 == f]} for f in range(3)]}
            fitted = run.fit_weights_from_internal_audit(audit, 1)
            outputs = run.combine_predictions(predictions, {}, 1, fitted)
            results.append([(np.array([outputs["ensemble__"+name]["weights"][k] for k in keys]),
                             outputs["ensemble__"+name]["prediction"]) for name in names])
        for (old_weights, old_pred), (new_weights, new_pred) in zip(*results):
            np.testing.assert_allclose(new_weights, old_weights, rtol=0, atol=1e-12)
            np.testing.assert_allclose(new_pred, old_pred, rtol=0, atol=1e-10)
            self.assertAlmostEqual(float(new_weights.sum()), 1.)

    def test_new_regressors_accept_power_train_and_reload(self):
        # Small synthetic inputs exercise the real backbones and existing
        # regression heads without starting any research-data fit.
        x = np.random.default_rng(8).uniform(0, 2e-12, (2, 32, 32, 1)).astype("float32")
        y = np.array([100000., 200000.])
        scaler = MinMaxScaler().fit(y[:, None])
        for key in ("resnet50", "vgg16"):
            with self.subTest(model=key):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(42)
                maker = RegressionModelMaker(x.shape[1:])
                spec = onb_model_specs([key])[0]
                model = spec["builder"](maker)
                self.assertEqual(model.input_shape, (None, 32, 32, 1))
                self.assertEqual(model.output_shape, (None, 1))
                compressed = tf.keras.Model(model.input, model.get_layer("log_power").output)(x).numpy()
                np.testing.assert_allclose(compressed, np.log1p(x/1e-12), rtol=1e-6, atol=1e-7)
                model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0003, momentum=0.9, clipnorm=1.), loss="mse")
                loss = model.train_on_batch(x, scaler.transform(y[:, None]))
                self.assertTrue(np.isfinite(loss))
                trainer = ModelTrainer(42)
                with tempfile.TemporaryDirectory() as temp:
                    manifest = begin_fitted_state(temp, "cnn_test", 1, 42, ["synthetic"], {}, None, scaler)
                    record = save_and_verify_model(temp, manifest, spec, model, maker, trainer, x[:1], scaler, None)
                    self.assertTrue(record["verification"]["passed"])
                del model, compressed
                tf.keras.backend.clear_session()
                gc.collect()


if __name__ == "__main__":
    unittest.main()
