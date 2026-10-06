import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
from utils.dataloading.acoustic_summary_features import AcousticFrequency34, frequency34_features


class AcousticSummaryFeaturesTest(unittest.TestCase):
    def test_uniform_power_has_uniform_shape_and_no_temporal_variation(self):
        f = frequency34_features(np.full((224,224), 1e-12))
        np.testing.assert_allclose(f[:5], -12)
        np.testing.assert_allclose(f[5:7], 0)
        self.assertAlmostEqual(f[7], 1500)
        self.assertAlmostEqual(f[8], 1)
        self.assertAlmostEqual(f[9], 0)
        np.testing.assert_allclose(f[10:], 0)

    def test_amplitude_changes_absolute_power_but_preserves_relative_shape(self):
        x = np.random.default_rng(7).uniform(1e-10,1e-9,size=(224,224))
        original = frequency34_features(x)
        scaled = frequency34_features(x*.1)
        np.testing.assert_allclose(scaled[:5]-original[:5], -1, atol=1e-12)
        np.testing.assert_allclose(scaled[5:],original[5:],rtol=1e-12,atol=1e-12)

    def test_time_shuffle_preserves_distribution_features(self):
        x = np.random.default_rng(8).lognormal(-22,2,size=(224,224))
        shuffled = x[np.random.default_rng(9).permutation(224)]
        np.testing.assert_allclose(frequency34_features(x),frequency34_features(shuffled),rtol=1e-12,atol=1e-12)

    def test_transform_supports_single_channel_and_rejects_invalid_power(self):
        transformer = AcousticFrequency34()
        x = np.zeros((2,224,224,1))
        result = transformer.fit_transform(x)
        self.assertEqual(result.shape,(2,34))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(len(transformer.get_feature_names_out()),34)
        for invalid in [np.full((224,224),-1),np.full((224,224),np.nan),np.zeros((223,224))]:
            with self.assertRaises(ValueError):frequency34_features(invalid)


if __name__=="__main__":unittest.main()
