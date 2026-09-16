"""Synthetic checks written before reference comparison implementation/scores."""
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml

from src import model_memory_reference as ref


class ReferenceContracts(unittest.TestCase):
    def test_fixed_combined_family_rejects_removed_model_or_contrast(self):
        p = yaml.safe_load(ref.PROTOCOL.read_text())
        ref.validate_protocol(p)
        bad = deepcopy(p)
        bad['models'].pop()
        with self.assertRaises(ValueError):
            ref.validate_protocol(bad)
        bad = deepcopy(p)
        bad['comparisons']['total_with_core'] = 32
        with self.assertRaises(ValueError):
            ref.validate_protocol(bad)

    def test_calibration_targets_must_be_complete_and_rows_match_gamma(self):
        dates = pd.bdate_range('2012-01-02', periods=20)
        target = pd.DataFrame({'y': np.ones(20), 'target_end': dates + pd.offsets.BDay(5)}, index=dates)
        complete = pd.Series(True, index=dates)
        got = ref.training_mask(complete, target, dates[10])
        np.testing.assert_array_equal(np.flatnonzero(got), np.arange(6))

    def test_gamma_feature_addition_learns_exact_conditional_mean(self):
        x = np.linspace(-2, 2, 401)[:, None]
        y = np.exp(-8.5 + .35*x[:, 0])
        query = np.array([[-1.5], [.7]])
        pred, audit = ref.fit_gamma(x, y, query)
        np.testing.assert_allclose(pred, np.exp(-8.5 + .35*query[:, 0]), rtol=2e-6)
        self.assertLess(audit['gradient_inf_norm'], 2e-7)

    def test_future_apply_distribution_cannot_fit_scaler_or_estimates(self):
        x = np.linspace(-2, 2, 401)[:, None]
        y = np.exp(-8.5 + .2*x[:, 0])
        one, a = ref.fit_gamma(x, y, np.array([[.1]]))
        many, b = ref.fit_gamma(x, y, np.array([[.1], [10.]]))
        np.testing.assert_allclose(one, many[:1], rtol=0, atol=0)
        self.assertEqual(a, b)

    def test_bad_features_targets_or_zero_scale_fail(self):
        x = np.arange(30.)[:, None]
        for bad in [np.zeros(30), np.full(30, np.nan), np.full(30, -1)]:
            with self.assertRaises(ValueError):
                ref.fit_gamma(x, bad, x[:1])
        with self.assertRaises(ValueError):
            ref.fit_gamma(np.ones((30, 1)), np.ones(30), np.ones((1, 1)))

    def test_combined_holm_retains_all_registered_arms_even_if_unavailable(self):
        rows = [{'p_conservative': .001}] * 32
        other = [{'p_conservative': 1.}] * 14
        p = ref.joint_holm(rows, other)
        np.testing.assert_allclose(p[:32], .046)
        np.testing.assert_array_equal(p[32:], np.ones(14))
        with self.assertRaises(ValueError):
            ref.joint_holm(rows, other[:-1])


if __name__ == '__main__':
    unittest.main()
