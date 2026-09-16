"""Pre-score numerical contracts for round-two dependence-aware inference."""
from __future__ import annotations

import unittest

import numpy as np
import statsmodels.api as sm

from src.orthogonal_round2 import bootstrap_means, hac_summary, holm_adjust


class TestInference(unittest.TestCase):
    def test_circular_mean_preserves_constant_and_exact_remainder(self):
        draws = bootstrap_means(np.full(137, 3.5), 21, 99, 123)
        np.testing.assert_allclose(draws, 3.5, rtol=0, atol=1e-14)

    def test_circular_bootstrap_pairs_columns_and_is_reproducible(self):
        x = np.random.default_rng(83).normal(size=301)
        values = np.column_stack([x, 2 * x + 7])
        draws = bootstrap_means(values, 63, 300, 881)
        np.testing.assert_allclose(draws[:, 1], 2 * draws[:, 0] + 7, atol=1e-13)
        np.testing.assert_array_equal(draws, bootstrap_means(values, 63, 300, 881))

    def test_hac_matches_independent_statsmodels_intercept_regression(self):
        rng = np.random.default_rng(554)
        x = np.zeros(800)
        for i in range(1, len(x)):
            x[i] = .8 * x[i - 1] + rng.normal()
        fit = sm.OLS(x, np.ones((len(x), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 126, "use_correction": False})
        actual = hac_summary(x)
        self.assertAlmostEqual(actual["se"], float(fit.bse[0]), places=12)
        self.assertAlmostEqual(actual["p"], float(fit.pvalues[0]), places=12)
        self.assertGreater(actual["se"], x.std() / np.sqrt(len(x)))

    def test_eight_way_holm_does_not_drop_unfavorable_hypotheses(self):
        p = np.array([.01, .02, .2, .3, .4, .5, .6, .7])
        adjusted = holm_adjust(p)
        self.assertAlmostEqual(adjusted[0], .08)
        self.assertFalse((adjusted < .05).any())

    def test_invalid_block_fails_closed(self):
        for block in (0, 11):
            with self.assertRaises(ValueError):
                bootstrap_means(np.ones(10), block, 10, 1)


if __name__ == "__main__":
    unittest.main()
