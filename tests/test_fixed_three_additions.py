"""Guard genuine member retention and fit-only candidate/integration selection."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
spec = importlib.util.spec_from_file_location("fixed_additions", ROOT / "code/run_fixed_three_additions.py")
addition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addition)


class FixedThreeAdditionTest(unittest.TestCase):
    def test_biased_rf_kept_with_exact_original_ratios(self):
        y = np.arange(1., 61.)
        # Perfect HGB would eliminate RF under unrestricted weighting.
        matrix = np.column_stack([y + 200 + 40*np.cos(y), y + 10, y - 4, y] +
            [y + np.sin(y) * (i+1) for i in range(len(addition.FAMILIES))])
        core = np.asarray([.15, .5, .35])
        original = {name: {"weights": core.tolist()} for name in addition.CONFIG["integration"]["anchors"]}
        catalog, selected = addition.fit_catalog(matrix, y, original)
        for method in catalog:
            if method["anchor"] == "none":
                continue
            w = np.asarray(method["weights"])
            np.testing.assert_allclose(w[:3] / w[:3].sum(), core)
            self.assertGreaterEqual(w[:3].sum(), .25 - 1e-8)
            self.assertTrue(np.all(w[:3] > 0))
            if method["kind"] in ["anchored5", "fixed5", "selected5"]:
                self.assertEqual(method["active_members"], 5)
        self.assertEqual(len(selected), 2)
        free = next(m for m in catalog if m["kind"] == "unrestricted")
        self.assertLess(free["weights"][0], 1e-7)

    def test_integration_does_not_read_outer_labels_or_predictions(self):
        y = np.linspace(1, 60, 18)
        matrix = np.column_stack([y + np.sin(y+i)*2 for i in range(len(addition.KEYS))])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            source = Path(directory) / "source"
            (output / "seed43").mkdir(parents=True)
            (source / "seed43").mkdir(parents=True)
            samples = [{"sample_filename": f"{v*1000}_sample.npy"} for v in y]
            samples += [{"sample_filename": "OUTER_LABEL_MUST_NOT_BE_PARSED"}]
            manifest = {"samples": samples}
            np.savez(output / "seed43/predictions.npz", oof=np.asarray([matrix]),
                train_indices=np.arange(len(y)), outer=np.asarray(["OUTER_MUST_NOT_BE_READ"]))
            methods = [{"method": name, "weights": [.15, .5, .35]} for name in addition.CONFIG["integration"]["anchors"]]
            (source / "seed43/frozen_integration.json").write_text(json.dumps({"methods": methods}), encoding="utf-8")
            with patch.object(addition, "OUT", output), patch.object(addition, "SOURCE", source), \
                patch.object(addition, "load_seed", return_value=(manifest, {})):
                addition.integrate([43])
            frozen = json.loads((output / "seed43/frozen_integration.json").read_text(encoding="utf-8"))
            self.assertFalse(frozen["outer_labels_used_for_selection"])
            self.assertFalse(frozen["noisy_oof_used_for_selection"])

    def test_nested_tuning_predicts_only_unseen_inner_rows(self):
        class CheckingModel:
            def fit(self, X, y):
                self.seen = set(X[:, 0])
                self.mean = float(np.mean(y))
                return self

            def predict(self, X):
                if not self.seen.isdisjoint(X[:, 0]):
                    raise AssertionError("Inner selection saw its held rows")
                return np.full(len(X), self.mean)

        X = np.arange(30.).reshape(-1, 1)
        y = np.arange(30.)**.5
        with patch.object(addition, "make_model", side_effect=lambda *args: CheckingModel()) as factory:
            model, audit = addition.nested_fit("ridge_direct", X, y, 43)
        self.assertEqual(len(model.seen), 30)
        self.assertEqual(factory.call_count, 10)
        self.assertEqual(audit["selection_labels"], "fit subset only")


if __name__ == "__main__":
    unittest.main()
