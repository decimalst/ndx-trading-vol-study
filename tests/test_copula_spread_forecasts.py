"""Pre-run synthetic spread probability/integration/clock contracts."""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm, t

from src import copula_spread_forecasts as forecast


def identity():
    return {"location": [0., 0.], "scale": [1., 1.],
            "epsilon": [0., 0.], "delta": [1., 1.]}


class MarginalProbabilityContracts(unittest.TestCase):
    def test_identity_equals_direct_student_t_cdf_and_survival(self):
        mu, h = np.array([.001, -.002]), np.array([.0001, .0009])
        strikes = np.log(np.array([.97, 1.02]))
        z = (strikes - mu) / np.sqrt(.75 * h)
        np.testing.assert_allclose(forecast.marginal_probability(strikes, mu, h, identity()),
                                   t.cdf(z, 8), atol=2e-13, rtol=2e-12)
        np.testing.assert_allclose(forecast.marginal_probability(
            strikes, mu, h, identity(), upper=True), t.sf(z, 8), atol=2e-13, rtol=2e-12)

    def test_quad_independently_reconstructs_normalized_spread_mean(self):
        mu, h = np.array([.003, -.001]), np.array([.0002, .0007])
        shape = {"location": [-.2, .15], "scale": [.8, 1.3],
                 "epsilon": [.3, -.25], "delta": [1.6, .9]}
        for pars in (identity(), shape):
            for structure in forecast.STRUCTURES:
                actual = forecast.exact_marginal_risk(mu, h, pars, structure, .02)
                for j in range(2):
                    def cdf(price, j=j, pars=pars):
                        z = (math.log(price) - mu[j]) / math.sqrt(.75 * h[j])
                        w = norm.ppf(t.cdf(z, 8))
                        v = math.sinh(pars["delta"][j] * math.asinh(
                            (w - pars["location"][j]) / pars["scale"][j])
                            - pars["epsilon"][j])
                        return norm.cdf(v)

                    expected = 0.
                    if structure in ("put", "condor"):
                        expected += quad(cdf, .97, .98, epsabs=1e-13, epsrel=1e-11)[0] / .01
                    if structure in ("call", "condor"):
                        expected += quad(lambda s: 1 - cdf(s), 1.02, 1.03,
                                         epsabs=1e-13, epsrel=1e-11)[0] / .01
                    self.assertAlmostEqual(actual["debit"][j], expected, places=10)

    def test_condor_exact_risk_is_sum_of_disjoint_sides(self):
        mu, h = np.zeros(2), np.array([.0004, .0009])
        parts = {s: forecast.exact_marginal_risk(mu, h, identity(), s, .02)
                 for s in forecast.STRUCTURES}
        for metric in ("breach", "full", "debit"):
            np.testing.assert_allclose(parts["condor"][metric],
                                       parts["put"][metric] + parts["call"][metric], atol=1e-14)
        self.assertTrue(np.all(parts["condor"]["full"] <= parts["condor"]["breach"]))
        self.assertTrue(np.all(parts["condor"]["debit"] <= parts["condor"]["breach"]))

    def test_narrow_distribution_inside_wing_has_accurate_mean_debit(self):
        # A fixed-node rule can miss the CDF transition inside the strike interval.
        # This is a numerical-domain stress test, not a market parameter choice.
        mu = np.log(np.array([.973, .977]))
        h = np.full(2, 1e-10)
        actual = forecast.exact_marginal_risk(mu, h, identity(), "put", .02)["debit"]
        for j in range(2):
            expected = quad(lambda s, j=j: t.cdf((math.log(s) - mu[j]) /
                                           math.sqrt(.75 * h[j]), 8), .97, .98,
                            points=[math.exp(mu[j])], epsabs=1e-12, epsrel=1e-10)[0] / .01
            self.assertAlmostEqual(actual[j], expected, delta=1e-6)


class SamplingContracts(unittest.TestCase):
    def test_gaussian_independence_any_and_both_match_exact(self):
        normals, scales = forecast.base_draws(power=14, seed=20260913)
        w = forecast.copula_draws(0., "gaussian", normals, scales)
        mu, h = np.zeros(2), np.array([.0004, .0009])
        y = mu + np.sqrt(.75 * h) * t.ppf(norm.cdf(w), 8)
        for structure in forecast.STRUCTURES:
            exact = forecast.exact_marginal_risk(mu, h, identity(), structure, .02)
            sample = forecast.sample_risks(y, structure, .02)
            p0, p1 = exact["breach"]
            self.assertAlmostEqual(sample["p_any_breach"], p0 + p1 - p0 * p1, delta=.003)
            self.assertAlmostEqual(sample["p_both_breach"], p0 * p1, delta=.003)
            self.assertAlmostEqual(sample["sample_mean_debit"], exact["debit"].mean(), delta=.003)

    def test_both_copulas_preserve_marginals_at_nonzero_rho(self):
        normals, scales = forecast.base_draws(power=14, seed=20260913)
        mu, h = np.array([.001, -.001]), np.array([.0004, .0009])
        exact = forecast.exact_marginal_risk(mu, h, identity(), "put", .02)
        for rho in (-.8, .8):
            for family in ("gaussian", "t8"):
                w = forecast.copula_draws(rho, family, normals, scales)
                y = mu + np.sqrt(.75 * h) * t.ppf(norm.cdf(w), 8)
                empirical = np.mean(y < np.log(.98), axis=0)
                np.testing.assert_allclose(empirical, exact["breach"], atol=.003, rtol=0)
                self.assertAlmostEqual(forecast.sample_risks(y, "put", .02)["sample_mean_debit"],
                                       exact["debit"].mean(), delta=.003)

    def test_fractional_expected_shortfall_and_ties(self):
        # 60 observations, tail mass 1.5: largest value plus half of second-largest.
        # A half-paid put spread corresponds exactly to an underlying price in its wings.
        debit = np.r_[np.zeros(58), .2, .8]
        price = .98 - .01 * debit
        y = np.repeat(np.log(price)[:, None], 2, axis=1)
        with patch("src.copula_spread_backtest.terminal_debit", return_value=np.repeat(
                debit[:, None], 2, axis=1)):
            actual = forecast.sample_risks(y, "put", .02)
        self.assertAlmostEqual(actual["es97_5"], (.8 + .5 * .2) / 1.5)
        tied = np.r_[np.zeros(55), np.ones(5)]
        with patch("src.copula_spread_backtest.terminal_debit", return_value=np.repeat(
                tied[:, None], 2, axis=1)):
            self.assertEqual(forecast.sample_risks(y, "put", .02)["es97_5"], 1.)

    def test_extreme_log_returns_have_finite_bounded_liability(self):
        y = np.array([[-1e300, 1e300], [1e300, -1e300], [-1e300, -1e300], [0., 0.]])
        for structure in forecast.STRUCTURES:
            with np.errstate(over="raise", invalid="raise"):
                risk = forecast.sample_risks(y, structure, .02)
            for value in risk.values():
                self.assertTrue(np.isfinite(value))
                self.assertGreaterEqual(value, 0.)
                self.assertLessEqual(value, 1.)

    def test_scrambles_are_reproducible_and_distinct(self):
        a, scale_a = forecast.base_draws(7, 10)
        b, scale_b = forecast.base_draws(7, 10)
        c, _ = forecast.base_draws(7, 11)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(scale_a, scale_b)
        self.assertFalse(np.array_equal(a, c))
        self.assertTrue(np.all(scale_a > 0))
        for rho in (-1., 1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                forecast.copula_draws(rho, "gaussian", a, scale_a)


class CausalBuildContracts(unittest.TestCase):
    def fixture(self):
        calendar = pd.bdate_range("2030-01-01", periods=5)
        origins = calendar[[1, 3]]
        rows = []
        for origin in origins:
            row = {"origin": origin, "fit_origin": calendar[1], "training_cutoff": calendar[0],
                   "phase": "synthetic", "mu_qqq": 0., "mu_spx": .001,
                   "h_qqq": .0004, "h_spx": .0009,
                   "a_qqq": 0., "a_spx": 0., "b_qqq": 1., "b_spx": 1.,
                   "epsilon_qqq": 0., "epsilon_spx": 0., "delta_qqq": 1., "delta_spx": 1.}
            row.update({f"rho_{model}": .4 for model in forecast.MODELS})
            rows.append(row)
        applications = pd.DataFrame(rows)
        archive = pd.DataFrame({"origin": origins, "target_end": calendar[[2, 4]],
                                "phase": "synthetic", "y_qqq": [0., 1.], "y_spx": [2., 3.]})
        return applications, origins, archive, calendar

    def test_build_does_not_consult_future_outcomes(self):
        args = self.fixture()
        result, diagnostics = forecast.build_risks(*args, power=6)
        changed = args[2].copy()
        changed["y_qqq"] = [float("nan"), -1e100]
        changed["y_spx"] = [float("inf"), 1e100]
        result_again, diagnostics_again = forecast.build_risks(
            args[0], args[1], changed, args[3], power=6)
        pd.testing.assert_frame_equal(result, result_again)
        self.assertEqual(diagnostics, diagnostics_again)

    def test_exact_marginal_fields_do_not_depend_on_copula(self):
        args = self.fixture()
        result, _ = forecast.build_risks(*args, power=6)
        for margin in ("orig", "cal", "shape"):
            a = result[result.model == f"{margin}_gaussian"].reset_index(drop=True)
            b = result[result.model == f"{margin}_t8"].reset_index(drop=True)
            for field in ("mean_debit", "breach_qqq", "breach_spx", "full_qqq", "full_spx",
                          "debit_qqq", "debit_spx"):
                np.testing.assert_array_equal(a[field], b[field])

    def test_daily_cutoff_uses_full_calendar_not_retained_rows(self):
        args = self.fixture()
        result, _ = forecast.build_risks(*args, power=6)
        later = result[result.origin == args[3][3]]
        self.assertTrue((later.feature_cutoff_date == args[3][2]).all())
        self.assertTrue((later.target_end == args[3][4]).all())

    def test_monthly_marginal_parameters_cannot_change_silently(self):
        for field in ("a_qqq", "b_spx", "epsilon_qqq", "delta_spx"):
            args = list(self.fixture())
            args[0].loc[1, field] += .01
            with self.subTest(field=field), self.assertRaises(ValueError):
                forecast.build_risks(*args, power=4)

    def test_monthly_dependence_parameters_cannot_change(self):
        args = list(self.fixture())
        args[0].loc[1, "rho_orig_gaussian"] = .5
        with self.assertRaises(ValueError):
            forecast.build_risks(*args, power=4)

    def test_archive_phase_and_target_end_must_match(self):
        args = list(self.fixture())
        args[2].loc[1, "phase"] = "other_phase"
        with self.assertRaises(ValueError):
            forecast.build_risks(*args, power=4)
        args = list(self.fixture())
        args[2].loc[1, "target_end"] = args[3][3]
        with self.assertRaises(ValueError):
            forecast.build_risks(*args, power=4)


if __name__ == "__main__":
    unittest.main()
