"""Check noisy-OOF integration without allowing access to outer truth."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("hgb_validation",ROOT/"code/run_hgb_complementarity_validation.py")
validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)


class HgbValidationTest(unittest.TestCase):
    def test_clean_fit_and_noisy_fit_can_choose_different_weights(self):
        y=np.linspace(1,12,12)
        clean=np.column_stack([y+.05,y-.07,y+.09,y+.1])
        oof=np.asarray([clean]+[clean+np.asarray([3*n,3*n,3*n,0])[None,:] for n in range(1,7)])
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);folder=output/"seed43";folder.mkdir()
            manifest={"config_sha256":validation.CONFIG_HASH,"samples":[{"sample_filename":f"{v*1000}_sample.npy"} for v in y]+[{"sample_filename":"OUTER_TRUTH_MUST_NOT_BE_PARSED"} for _ in range(4)]}
            (folder/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            (folder/"training_complete.json").write_text("{}",encoding="utf-8")
            np.savez(folder/"base_predictions.npz",oof=oof,train_indices=np.arange(12),outer=np.asarray(["OUTER_PREDICTIONS_NOT_NEEDED"]))
            with patch.object(validation,"OUT",output):validation.integrate([43])
            frozen=json.loads((folder/"frozen_integration.json").read_text(encoding="utf-8"))
            methods={m["method"]:m for m in frozen["methods"]}
            noisy=np.asarray(methods["hgb4_noisy_mse"]["weights"])
            clean_w=np.asarray(methods["hgb4_clean_mse"]["weights"])
            self.assertGreater(noisy[3],.99)
            self.assertLess(clean_w[3],noisy[3])
            self.assertFalse(frozen["outer_labels_used"])
            for method in methods.values():
                if method["kind"]=="linear":
                    self.assertAlmostEqual(sum(method["weights"]),1)
                    self.assertTrue(np.all(np.asarray(method["weights"])>=0))

    def test_simplex_can_ignore_a_strongly_biased_member(self):
        y=np.arange(1,21,dtype=float)
        matrix=np.column_stack([y+200,y+np.sin(y),y-np.sin(y)])
        weights=validation.simplex(matrix,y)
        self.assertLess(weights[0],1e-6)
        np.testing.assert_allclose(matrix@weights,y,atol=1e-4)


if __name__=="__main__":unittest.main()
