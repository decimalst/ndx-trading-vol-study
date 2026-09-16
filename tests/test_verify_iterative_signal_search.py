"""Prewritten independent contracts for the next index/HF research wave."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import verify_iterative_signal_search as verify


def market_fixture(n=100):
    rng = np.random.default_rng(90121)
    dates = pd.bdate_range("2011-01-03", periods=n)
    opening = 100 * np.exp(np.cumsum(rng.normal(0, .005, n)))
    close = opening * np.exp(rng.normal(0, .006, n))
    daily = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * 1.01,
                          "low": np.minimum(opening, close) * .99, "close": close,
                          "adj close": close * np.linspace(.9, 1., n)}, index=dates)
    cross = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .01, (n, 5)), axis=0)),
                         index=dates, columns=["hyg", "tlt", "gld", "uso", "uup"])
    iv = pd.DataFrame({"vxn": rng.uniform(12, 40, n), "vix": rng.uniform(10, 30, n)}, index=dates)
    hf = pd.DataFrame({"rv5": np.exp(rng.normal(-8, .1, n))}, index=dates)
    hf["rsv"] = hf.rv5 * rng.uniform(.2, .8, n)
    hf["rk_parzen"] = hf.rv5 * rng.uniform(.8, 1.2, n)
    return daily, cross, iv, hf


class IndependentConstructionTests(unittest.TestCase):
    def test_forecast_schema_requires_explicit_horizon(self):
        with self.assertRaisesRegex(AssertionError, "horizon"):
            verify.verify_forecasts("index", pd.DataFrame(), {}, {}, pd.DataFrame(), [])

    def test_oxford_dates_preserve_local_session_without_utc_conversion(self):
        actual = verify.local_archive_dates(["2011-04-04 00:00:00+01:00", "2011-11-07 00:00:00+00:00"])
        pd.testing.assert_index_equal(actual, pd.DatetimeIndex(["2011-04-04", "2011-11-07"]))

    def test_hf_target_requires_every_day_and_an_extra_publication_session(self):
        dates = pd.bdate_range("2014-01-02", periods=12)
        rv = pd.Series(np.arange(1., 13.), index=dates)
        target = verify.hf_target(rv, dates, 5)
        self.assertEqual(target.loc[dates[0], "y"], 4.)
        self.assertEqual(target.loc[dates[0], "target_end"], dates[5])
        self.assertEqual(target.loc[dates[0], "available_date"], dates[6])
        complete = pd.Series(True, index=dates)
        self.assertFalse(verify.training_eligibility(complete, target, dates[5], availability=True).loc[dates[0]])
        self.assertTrue(verify.training_eligibility(complete, target, dates[6], availability=True).loc[dates[0]])
        rv.iloc[3] = np.nan
        self.assertTrue(np.isnan(verify.hf_target(rv, dates, 5).loc[dates[0], "y"]))

    def test_hf_candidates_are_sum_ratios_then_shifted_one_session(self):
        daily, _cross, iv, hf = market_fixture()
        features = verify.hf_features(daily, hf, iv.vix)
        row = 60
        for label, span in (("d", 1), ("w", 5), ("m", 22)):
            ix = slice(row - span, row)
            self.assertAlmostEqual(features.iloc[row][f"share_{label}"], hf.rsv.iloc[ix].sum() / hf.rv5.iloc[ix].sum(), places=13)
            self.assertAlmostEqual(features.iloc[row][f"kernel_{label}"], np.log(hf.rk_parzen.iloc[ix].sum() / hf.rv5.iloc[ix].sum()), places=13)
            self.assertAlmostEqual(features.iloc[row][f"hf_{label}"], np.log(hf.rv5.iloc[ix].mean()), places=13)
        changed = hf.copy()
        changed.iloc[row:] *= 100
        np.testing.assert_allclose(verify.hf_features(daily, changed, iv.vix).iloc[row], features.iloc[row])

    def test_index_zero_actual_return_is_valid_and_not_filtered(self):
        daily, cross, iv, _hf = market_fixture()
        daily.loc[daily.index[60], "close"] = daily.loc[daily.index[60], "open"]
        _, target = verify.index_features(daily, cross, iv)
        origin = daily.index[59]
        self.assertEqual(target.loc[origin, "y"], 0.)
        self.assertEqual(target.loc[origin, "target_end"], daily.index[60])
        eligible = verify.training_eligibility(pd.Series(True, index=daily.index), target, daily.index[61], availability=False)
        self.assertTrue(eligible.loc[origin])

    def test_index_calendar_uses_only_target_date_and_prior_observed_month_sessions(self):
        dates = pd.DatetimeIndex(["2019-11-29", "2019-12-02", "2019-12-03", "2019-12-04", "2019-12-05"])
        calendar = verify.calendar_features(dates)
        np.testing.assert_array_equal(calendar.target_first3.iloc[:4], [1, 1, 1, 0])
        np.testing.assert_array_equal(calendar.target_dow_1.iloc[:4], [0, 1, 0, 0])
        self.assertTrue(calendar.iloc[-1].isna().all())

    def test_index_decomposition_preserves_adjusted_total_return_identity(self):
        daily, cross, iv, _hf = market_fixture()
        daily.loc[daily.index[55]:, "adj close"] *= .5
        features, _ = verify.index_features(daily, cross, iv)
        total = np.log(daily["adj close"]).diff()
        daytime = np.log(daily.close / daily.open)
        for label, span in (("d", 1), ("w", 5), ("m", 22)):
            np.testing.assert_allclose(features[f"split_{label}"], (total - 2 * daytime).rolling(span).mean(), equal_nan=True)


class IndependentEstimatorAndInferenceTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(53109)
        x = np.column_stack([np.ones(180), rng.normal(size=(180, 3))])
        y = .001 + x[:, 1] * .002 + rng.normal(0, .008, len(x))
        return x, y, x[:9]

    def test_ridge_penalty_uses_mean_loss_and_zero_returns_are_supported(self):
        x, y, query = self.fixture()
        predicted, _audit = verify.ridge_prediction(x, y, query, alpha=.01)
        centered = x[:, 1:] - x[:, 1:].mean(axis=0)
        scaled = centered / x[:, 1:].std(axis=0)
        expected_beta = np.linalg.solve(scaled.T @ scaled + len(x) * .01 * np.eye(3), scaled.T @ (y - y.mean()))
        expected = y.mean() + ((query[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)) @ expected_beta
        np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-14)
        duplicated, _ = verify.ridge_prediction(np.tile(x, (2, 1)), np.tile(y, 2), query, alpha=.01)
        np.testing.assert_allclose(duplicated, predicted, rtol=1e-11, atol=1e-14)
        zero, _ = verify.ridge_prediction(x, np.zeros(len(x)), query, alpha=.01)
        np.testing.assert_array_equal(zero, np.zeros(len(query)))

    def test_ols_uses_exact_duan_smearing_and_training_only_scale(self):
        x, y, query = self.fixture()
        positive = np.exp(-8 + y * 50)
        predicted, _audit = verify.ols_prediction(x, positive, query)
        beta = np.linalg.lstsq(x, np.log(positive), rcond=None)[0]
        smear = np.exp(np.log(positive) - x @ beta).mean()
        np.testing.assert_allclose(predicted, np.exp(query @ beta) * smear, rtol=1e-11)
        changed = query.copy()
        changed[1:, 1:] += 5
        revised, _ = verify.ols_prediction(x, positive, changed)
        self.assertAlmostEqual(predicted[0], revised[0], places=14)

    def test_two_legs_paid_on_each_held_day_even_with_same_signal(self):
        predicted = np.array([.0005, .0004, .003, -.001])
        actual = np.array([0., .02, -.005, .01])
        got = verify.execution_returns(predicted, actual, cost_per_side=.0002)
        np.testing.assert_array_equal(got["position"], [1., 0., 1., 0.])
        np.testing.assert_allclose(got["net"], [-.0004, 0., -.0054, 0.], atol=1e-14)
        same = verify.execution_returns(np.ones(3), np.zeros(3), cost_per_side=.0005)
        np.testing.assert_allclose(same["net"], -.001, atol=1e-14)

    def test_independent_holm_retains_all_trials_including_p_one(self):
        p = [0.001, .02, 1., .04]
        np.testing.assert_allclose(verify.holm(p), [.004, .06, 1., .08])
        self.assertGreater(verify.holm([.001] + [1.] * 65)[0], .05)


if __name__ == "__main__":
    unittest.main()
