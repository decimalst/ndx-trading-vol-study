"""Synthetic second-moment estimator contracts, written before empirical fits."""

import unittest
from unittest.mock import patch

import numpy as np

from src.macro_second_moment import fit_second_moment, objective, proper_score


class TestSecondMoment(unittest.TestCase):
    def test_proper_score_accepts_exact_zero_and_requires_positive_forecast(self):
        np.testing.assert_allclose(proper_score([0, 2], [2, 2]), [np.log(2), np.log(2)+1])
        for y, h in [([0], [0]), ([-1], [2]), ([1], [np.inf]), ([np.nan], [1])]:
            with self.assertRaises(ValueError):
                proper_score(y, h)

    def test_paired_loss_is_invariant_to_target_units(self):
        y, h1, h2 = np.array([0., .1, 1., 3.]), np.array([.2, .3, .5, 1.]), np.array([.1, .2, .7, 2.])
        gap = proper_score(y, h1)-proper_score(y, h2)
        np.testing.assert_allclose(proper_score(y*10000, h1*10000)-proper_score(y*10000, h2*10000), gap)

    def test_objective_gradient_and_hessian_agree_with_finite_differences(self):
        x = np.c_[np.ones(5), np.array([-2., -1., 0., 1., 2.])]
        y, beta = np.array([0., .5, 2., 1., 1.5]), np.array([.2, -.1])
        val, grad, hess = objective(beta, x, y, .01)
        self.assertTrue(np.isfinite(val))
        step = 1e-5
        for j in range(2):
            change = np.eye(2)[j]*step
            vp, gp, _ = objective(beta+change, x, y, .01)
            vm, gm, _ = objective(beta-change, x, y, .01)
            self.assertAlmostEqual(grad[j], (vp-vm)/(2*step), places=8)
            np.testing.assert_allclose(hess[:, j], (gp-gm)/(2*step), atol=1e-8)

    def test_penalty_is_per_mean_loss_and_leaves_intercept_unpenalized(self):
        x = np.c_[np.ones(3), [-1., 0., 1.]]
        y, beta = np.array([0., 1., 2.]), np.array([.2, .3])
        a = objective(beta, x, y, .01)
        b = objective(beta, np.repeat(x, 4, axis=0), np.repeat(y, 4), .01)
        for left, right in zip(a, b, strict=True):
            np.testing.assert_allclose(left, right)
        unpenalized = objective(beta, x, y, 0)
        self.assertAlmostEqual(a[0]-unpenalized[0], .01*.3**2)
        np.testing.assert_allclose(a[1]-unpenalized[1], [0, .02*.3], atol=1e-15)

    def test_intercept_only_optimum_equals_historical_mean_even_with_zeros(self):
        y = np.array([0., 0., 1., 3.])
        result = fit_second_moment(np.empty((4, 0)), y, np.empty((3, 0)))
        np.testing.assert_allclose(result["prediction"], np.ones(3))
        self.assertLessEqual(result["gradient_max_abs"], 1e-8)

    def test_synthetic_signal_fit_converges_with_zero_targets(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(120, 2))
        y = np.exp(.2+.4*x[:, 0])
        y[::7] = 0
        result = fit_second_moment(x, y, x[:3])
        self.assertGreater(result["beta"][1], 0)
        self.assertTrue(np.isfinite(result["prediction"]).all())
        self.assertTrue((result["prediction"] > 0).all())
        self.assertLessEqual(result["gradient_max_abs"], 1e-8)

    def test_fit_scaling_uses_training_only_and_forecast_rows_do_not_change_fit(self):
        x, y = np.arange(10.)[:, None], np.arange(10.)+.1
        a = fit_second_moment(x, y, x[:1])
        b = fit_second_moment(x, y, np.array([[9.], [15.]]))
        np.testing.assert_array_equal(a["beta"], b["beta"])
        np.testing.assert_array_equal(a["means"], x.mean(axis=0))
        np.testing.assert_array_equal(a["scales"], x.std(axis=0, ddof=0))

    def test_fitted_forecasts_and_slopes_rescale_with_target_units(self):
        x = np.arange(8.)[:, None]
        y = np.array([0., .1, .4, .2, .9, .7, 1., 1.2])
        a, b = fit_second_moment(x, y, x), fit_second_moment(x, y*10000, x)
        np.testing.assert_allclose(b["prediction"], a["prediction"]*10000, rtol=1e-8)
        np.testing.assert_allclose(b["beta"][1:], a["beta"][1:], atol=1e-10)

    def test_replicating_training_rows_preserves_normalized_penalty_solution(self):
        x = np.arange(8.)[:, None]
        y = np.array([0., .1, .4, .2, .9, .7, 1., 1.2])
        a = fit_second_moment(x, y, x)
        b = fit_second_moment(np.repeat(x, 3, axis=0), np.repeat(y, 3), x)
        np.testing.assert_allclose(a["prediction"], b["prediction"], rtol=1e-8)

    def test_degenerate_data_and_nonconvergence_fail_without_fallback(self):
        x = np.arange(8.)[:, None]
        for bad_x, bad_y in [(x, np.zeros(8)), (x, np.full(8, -1)), (np.ones((8, 1)), np.ones(8))]:
            with self.assertRaises(ValueError):
                fit_second_moment(bad_x, bad_y, bad_x)
        with patch("src.macro_second_moment.MAX_ITER", 0), self.assertRaisesRegex(ValueError, "converge"):
            fit_second_moment(x, np.arange(8.)+.1, x)

    def test_prediction_overflow_fails_without_clipping(self):
        x = np.arange(8.)[:, None]
        with self.assertRaisesRegex(ValueError, "forecast"):
            fit_second_moment(x, np.exp(x[:, 0]), np.array([[1e9]]))


if __name__ == "__main__":
    unittest.main()
