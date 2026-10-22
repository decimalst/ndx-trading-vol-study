"""Contract tests written before the recent-mean diagnostic implementation."""
import unittest

import numpy as np

from src.tirex_recent_mean import evaluate_readout, fit_readout


class RecentMeanTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(782)
        self.x = self.rng.normal(size=(40, 2, 64))
        self.a = 2.5 + 1.7 * self.x[:, :, -32:].mean(axis=-1)

    def test_exact_affine_recovery_and_pair_differences(self):
        model = fit_readout(self.x, self.a)
        np.testing.assert_allclose(model['coefficients'], [2.5, 1.7], atol=1e-12)
        result = evaluate_readout(model, self.x, self.a)
        self.assertAlmostEqual(result['level_r2'], 1.)
        self.assertAlmostEqual(result['paired_difference_r2'], 1.)

    def test_evaluation_does_not_refit_or_mutate(self):
        model = fit_readout(self.x, self.a)
        saved = model['coefficients'].copy()
        result = evaluate_readout(model, self.x, self.a + 10)
        self.assertLess(result['level_r2'], 0.)
        self.assertEqual(saved, model['coefficients'])
        self.assertAlmostEqual(result['paired_difference_r2'], 1.)

    def test_only_last32_inputs_used(self):
        model = fit_readout(self.x, self.a)
        shifted = self.x.copy()
        shifted[:, :, :-32] += 1000
        self.assertEqual(evaluate_readout(model, self.x, self.a),
                         evaluate_readout(model, shifted, self.a))

    def test_constant_target_r2_is_undefined(self):
        model = fit_readout(self.x, np.zeros((40, 2)))
        result = evaluate_readout(model, self.x, np.zeros((40, 2)))
        self.assertIsNone(result['level_r2'])
        self.assertIsNone(result['paired_difference_r2'])

    def test_malformed_and_nonfinite_reject(self):
        for x, a in [(self.x[:, :, :20], self.a), (self.x, self.a[:, 0]),
                     (self.x * np.nan, self.a), (self.x, self.a * np.inf)]:
            with self.assertRaises(ValueError):
                fit_readout(x, a)


if __name__ == '__main__':
    unittest.main()
