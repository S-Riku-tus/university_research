import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from utils.ensemble.ensemble_runtime import EnsembleManager  # noqa: E402
from utils.config.onb_defaults import apply_onb_defaults, onb_model_specs  # noqa: E402
from utils.experiment.run_helpers import set_global_seed  # noqa: E402
from utils.models.regression.base_regression import RegressionModelMaker  # noqa: E402
from utils.training.fitted_artifacts import (  # noqa: E402
    begin_fitted_state,
    finish_fitted_state,
    save_and_verify_model,
)
from utils.training.model_training import ModelTrainer  # noqa: E402


def tiny_builder(model_maker):
    return keras.Sequential(
        [
            keras.layers.Input(shape=model_maker.input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1),
        ]
    )


class TrainingStatePersistenceTest(unittest.TestCase):
    def setUp(self):
        keras.utils.set_random_seed(42)

    def test_defaults_do_not_add_epoch_selection_validation(self):
        self.assertNotIn("training_validation", apply_onb_defaults({}))

    def test_production_randomforest_uses_current_run_seed(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(8, 2, 2, 1)).astype(np.float32)
        y = np.linspace(0.0, 1.0, len(x)).reshape(-1, 1)
        trainer = ModelTrainer(42)
        x_pca, _ = trainer.make_pca(x, [], 2)
        spec = onb_model_specs()[0]
        spec["builder_params"] = {"n_estimators": 3, "max_depth": 2, "n_jobs": 1}
        set_global_seed(91)
        model, _ = trainer.train_one_model(
            spec, RegressionModelMaker((2, 2, 1)), x, y, x_pca, 1
        )
        self.assertEqual(model.get_params()["random_state"], 91)

    def test_stable_pca_transform_is_batch_invariant(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(12, 3, 2, 1)).astype(np.float32)
        trainer = ModelTrainer(42)
        whole, _, pca = trainer.make_pca(x, [], 4, return_pca=True)
        in_parts = np.concatenate(
            [trainer.transform_pca(pca, x[:5]), trainer.transform_pca(pca, x[5:])]
        )
        np.testing.assert_array_equal(whole, in_parts)

    def test_keras_and_randomforest_state_reload_predictions(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(12, 2, 2, 1)).astype(np.float32)
        y = np.linspace(10.0, 120.0, len(x))
        scaler = MinMaxScaler().fit(y.reshape(-1, 1))
        y_scaled = scaler.transform(y.reshape(-1, 1))
        trainer = ModelTrainer(42)
        maker = RegressionModelMaker((2, 2, 1))
        x_pca, _, pca = trainer.make_pca(x, [], 3, return_pca=True)
        keras_spec = {
            "key": "tiny",
            "label": "Tiny",
            "kind": "keras",
            "builder": tiny_builder,
            "lr": 1e-3,
            "batch_size": 4,
            "fit_verbose": 0,
        }
        rf_spec = {
            "key": "randomforest",
            "label": "RF",
            "kind": "sklearn",
            "builder": lambda mm: mm.random_forest(
                n_estimators=3, max_depth=2, n_jobs=1
            ),
        }
        keras_model, _ = trainer.train_one_model(
            keras_spec, maker, x, y_scaled, x_pca, 1
        )
        rf_model, _ = trainer.train_one_model(
            rf_spec, maker, x, y_scaled, x_pca, 1
        )
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "state"
            manifest = begin_fitted_state(
                directory,
                "fit-test",
                1,
                42,
                ["wav-a", "wav-b"],
                {"tiny": 1},
                pca,
                scaler,
            )
            for spec, model in ((keras_spec, keras_model), (rf_spec, rf_model)):
                record = save_and_verify_model(
                    directory,
                    manifest,
                    spec,
                    model,
                    maker,
                    trainer,
                    x[:4],
                    scaler,
                    pca,
                    verify=True,
                )
                self.assertTrue(record["verification"]["passed"])
            run = EnsembleManager(
                {"enabled_strategy_names": ["simple_equal"]},
                ["tiny", "randomforest"],
            ).create_run([keras_spec, rf_spec])
            outputs = run.combine_predictions(
                {"tiny": np.ones(4), "randomforest": np.zeros(4)}, {}, 1
            )
            manifest_path = finish_fitted_state(directory, manifest, outputs)
            self.assertTrue(manifest_path.is_file())
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(set(saved["models"]), {"tiny", "randomforest"})
            self.assertIn("ensemble__simple_equal", saved["ensemble_weights"])
            self.assertEqual(saved["training_wav_groups"], ["wav-a", "wav-b"])
            self.assertEqual(saved["model_epochs"], {"tiny": 1})
            self.assertEqual(saved["pca_feature_decimals"], 6)


if __name__ == "__main__":
    unittest.main()
