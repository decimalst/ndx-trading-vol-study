"""Generated checks written before the descriptive critique reconstruction."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import t


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('diagnostics.py')
        spec = importlib.util.spec_from_file_location('diagnostics', path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_scale_convention_and_known_variance(self):
        z = np.array([[-1., -2.], [1., 2.], [0., 0.], [3., -3.]])
        h = np.full_like(z, .04)
        mu = np.full_like(z, .003)
        actual = self.m.coordinates(mu+np.sqrt(.75*h)*z, mu, h)
        np.testing.assert_allclose(actual, z, atol=1e-14)
        result = self.m.describe(actual, np.array([.1, .2, .3, .4]))
        self.assertAlmostEqual(result['assets'][0]['variance_ddof0'], 2.1875)
        self.assertAlmostEqual(result['assets'][1]['variance_ddof0'], 3.6875)
        self.assertAlmostEqual(result['nominal_variance'], 8/6)

    def test_tail_indicators_use_exact_quantile_boundaries(self):
        lo, hi = t.ppf([.025, .975], 8)
        z = np.array([[lo-.01, hi+.01], [0., 0.], [lo+.01, hi-.01], [1., -1.]])
        result = self.m.describe(z, np.ones(4))
        self.assertEqual(result['assets'][0]['lower_tail_count'], 1)
        self.assertEqual(result['assets'][1]['upper_tail_count'], 1)
        self.assertEqual(result['assets'][0]['lower_tail_rate'], .25)

    def test_contribution_shares_are_signed_and_counts_preserved(self):
        z = np.array([[0., 0.], [.5, -.5], [1., 2.], [5., 0.]])
        result = self.m.describe(z, np.array([1., -2., 3., 2.]))
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['central']['n'], 2)
        self.assertEqual(result['extreme']['n'], 1)
        self.assertAlmostEqual(result['central']['share_of_total_gain'], -.25)
        self.assertAlmostEqual(result['extreme']['share_of_total_gain'], .5)

    def test_invalid_variance_or_values_fail(self):
        for h in (np.zeros((2,2)), -np.ones((2,2)), np.full((2,2), np.nan)):
            with self.assertRaises(ValueError):
                self.m.coordinates(np.ones((2,2)), np.zeros((2,2)), h)
        with self.assertRaises(ValueError):
            self.m.describe(np.array([[1., np.inf], [0., 0.]]), np.ones(2))


if __name__ == '__main__':
    unittest.main()
