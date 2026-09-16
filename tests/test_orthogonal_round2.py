"""Pre-implementation synthetic contracts for the fixed round-two search.

These fixtures never read market files or empirical study forecasts. Numerical
expectations use direct definitions and independent least-squares calculations.

Pre-score clarification: weekly/monthly leverage is the negative part of the
rolling mean adjusted return, matching the fixed YAML and existing benchmark;
it is not the rolling mean of the daily negative part. Missing stays missing.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import orthogonal_round2 as study

BASELINE = [
    "const", "lrv_d", "lrv_w", "lrv_m", "lev_d", "lev_w", "lev_m",
    "liv", "lvix", "term", "xasset_stress", "market_stress",
]
CANDIDATES = ["vvix", "rv_dispersion", "stock_bond_corr", "close_pressure"]


def market_fixture(n=420):
    rng = np.random.default_rng(74201)
    index = pd.bdate_range("2018-01-02", periods=n, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.008, n)))
    opening = close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(opening, close) * np.exp(rng.uniform(0.001, 0.02, n))
    low = np.minimum(opening, close) * np.exp(-rng.uniform(0.001, 0.02, n))
    daily = pd.DataFrame({
        "open": opening, "high": high, "low": low, "close": close,
        "adj close": close * np.exp(np.linspace(-0.03, 0, n)),
        "volume": np.exp(rng.normal(16, 0.3, n)),
    }, index=index)
    cross = pd.DataFrame(
        80 * np.exp(np.cumsum(rng.normal(0, 0.012, (n, 5)), axis=0)),
        index=index, columns=["hyg", "tlt", "gld", "uso", "uup"],
    )
    iv = pd.DataFrame({
        "vxn": rng.uniform(16, 40, n), "vix": rng.uniform(12, 32, n),
        "vix9d": rng.uniform(10, 35, n), "vvix": rng.uniform(70, 140, n),
    }, index=index)
    return daily, cross, iv


def regression_fixture(n=160):
    rng = np.random.default_rng(48103)
    index = pd.bdate_range("2021-01-04", periods=n, name="date")
    values = rng.normal(size=(n, len(BASELINE) + len(CANDIDATES)))
    features = pd.DataFrame(values, index=index, columns=BASELINE + CANDIDATES)
    features["const"] = 1.0
    features["vvix"] += 0.8 * features["liv"]
    coefficients = rng.normal(0, 0.06, len(BASELINE))
    log_y = -8 + features[BASELINE].to_numpy() @ coefficients
    log_y += 0.2 * features["vvix"].to_numpy() + rng.normal(0, 0.3, n)
    return features, pd.Series(np.exp(log_y), index=index, name="y")


class FeatureContractTests(unittest.TestCase):
    def test_realized_variance_har_and_leverage_match_independent_definitions(self):
        daily, cross, iv = market_fixture()
        got = study.build_features(daily, cross, iv)
        log_hl = np.log(daily["high"] / daily["low"])
        log_co = np.log(daily["close"] / daily["open"])
        gk = (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).clip(lower=1e-10)
        overnight = np.log(daily["open"] / daily["close"].shift(1))
        rv = gk + overnight**2
        adjusted_return = np.log(daily["adj close"]).diff()
        negative_return = adjusted_return.clip(upper=0)
        expected = {
            "rv_total": rv, "log_rv": np.log(rv), "lrv_d": np.log(rv),
            "lrv_w": np.log(rv.rolling(5).mean()),
            "lrv_m": np.log(rv.rolling(22).mean()), "lev_d": negative_return,
            "lev_w": adjusted_return.rolling(5).mean().clip(upper=0),
            "lev_m": adjusted_return.rolling(22).mean().clip(upper=0),
        }
        pd.testing.assert_index_equal(got.index, daily.index)
        for column, values in expected.items():
            np.testing.assert_allclose(got[column], values, rtol=1e-11, atol=1e-13, equal_nan=True,
                                       err_msg=column)
        np.testing.assert_array_equal(got["const"], np.ones(len(daily)))

    def test_implied_volatility_aligns_to_sessions_before_one_session_delay(self):
        daily, cross, iv = market_fixture()
        saturday = pd.Timestamp("2018-01-06")
        iv.loc[saturday] = [901, 902, 903, 904]
        iv = iv.sort_index()
        got = study.build_features(daily, cross, iv)
        aligned = iv.reindex(daily.index).shift(1)
        for column, original in [("liv", "vxn"), ("lvix", "vix"), ("vvix", "vvix")]:
            np.testing.assert_allclose(got[column], np.log(aligned[original]), equal_nan=True)
        np.testing.assert_allclose(got["term"], np.log(aligned["vix9d"] / aligned["vix"]), equal_nan=True)
        self.assertTrue(got.loc[daily.index[0], ["liv", "lvix", "term", "vvix"]].isna().all())

    def test_realized_candidates_use_declared_trailing_windows(self):
        daily, cross, iv = market_fixture()
        got = study.build_features(daily, cross, iv)
        adjusted_return = np.log(daily["adj close"]).diff()
        bond_return = np.log(cross["tlt"]).diff()
        close_location = 2 * np.log(daily["close"] / daily["low"]) / np.log(daily["high"] / daily["low"]) - 1
        expected = {
            "rv_dispersion": np.log(got["rv_total"]).rolling(22).std(ddof=1),
            "stock_bond_corr": adjusted_return.rolling(22).corr(bond_return),
            "close_pressure": close_location.rolling(5).mean(),
        }
        for column, values in expected.items():
            np.testing.assert_allclose(got[column], values, rtol=1e-10, atol=1e-12, equal_nan=True)

    def test_stress_scales_only_use_preceding_observations(self):
        daily, cross, iv = market_fixture()
        got = study.build_features(daily, cross, iv)
        cross_returns = np.log(cross).diff()
        scale_mean = cross_returns.rolling(252, min_periods=126).mean().shift(1)
        scale_std = cross_returns.rolling(252, min_periods=126).std(ddof=1).shift(1)
        z_cross = (cross_returns - scale_mean) / scale_std
        expected_cross = np.sqrt((z_cross**2).mean(axis=1, skipna=False))
        overnight = np.log(daily["open"] / daily["close"].shift(1))**2
        raw_market = pd.DataFrame({"log_volume": np.log(daily["volume"]),
                                   "overnight_share": overnight / got["rv_total"]})
        mean_market = raw_market.rolling(252, min_periods=126).mean().shift(1)
        std_market = raw_market.rolling(252, min_periods=126).std(ddof=1).shift(1)
        z_market = ((raw_market - mean_market) / std_market).clip(lower=0)
        expected_market = np.sqrt((z_market**2).mean(axis=1, skipna=False))
        np.testing.assert_allclose(got["xasset_stress"], expected_cross, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(got["market_stress"], expected_market, rtol=1e-10, equal_nan=True)

    def test_missing_observations_and_flat_ranges_are_never_filled_or_reweighted(self):
        daily, cross, iv = market_fixture()
        date, next_date = daily.index[300:302]
        iv.loc[date, "vvix"] = np.nan
        cross.loc[date, "hyg"] = np.nan
        daily.loc[date, ["high", "low", "open"]] = daily.loc[date, "close"]
        got = study.build_features(daily, cross, iv)
        self.assertTrue(np.isnan(got.loc[next_date, "vvix"]))
        self.assertTrue(got.loc[daily.index[300:302], "xasset_stress"].isna().all())
        self.assertTrue(got.loc[daily.index[300:305], "close_pressure"].isna().all())
        self.assertTrue(np.isfinite(got.loc[daily.index[305], "close_pressure"]))

    def test_future_mutations_cannot_change_any_origin_feature(self):
        daily, cross, iv = market_fixture()
        origin = daily.index[320]
        expected = study.build_features(daily, cross, iv).loc[:origin]
        changed_daily, changed_cross, changed_iv = daily.copy(), cross.copy(), iv.copy()
        future = changed_daily.index > origin
        changed_daily.loc[future] *= np.linspace(2, 9, future.sum())[:, None]
        changed_cross.loc[future] *= np.linspace(0.1, 5, future.sum())[:, None]
        changed_iv.loc[future] = 999.0
        actual = study.build_features(changed_daily, changed_cross, changed_iv).loc[:origin]
        pd.testing.assert_frame_equal(actual, expected)
        truncated = study.build_features(daily.loc[:origin], cross.loc[:origin], iv.loc[:origin])
        pd.testing.assert_frame_equal(truncated, expected)


class TargetAndTrainingContractTests(unittest.TestCase):
    def test_targets_are_forward_arithmetic_means_with_actual_session_end_dates(self):
        dates = pd.DatetimeIndex(["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05",
                                  "2024-07-08", "2024-07-09", "2024-07-10"], name="date")
        rv = pd.Series([1, 2, 4, 8, 16, 32, 64], index=dates, dtype=float)
        got = study.make_targets(rv, 5)
        self.assertAlmostEqual(got.loc[dates[0], "y"], (2 + 4 + 8 + 16 + 32) / 5)
        self.assertEqual(got.loc[dates[0], "target_end"], dates[5])
        self.assertAlmostEqual(got.loc[dates[1], "y"], (4 + 8 + 16 + 32 + 64) / 5)
        self.assertEqual(got.loc[dates[1], "target_end"], dates[6])
        self.assertTrue(got.loc[dates[2:], "y"].isna().all())
        self.assertTrue(got.loc[dates[2:], "target_end"].isna().all())

    def test_target_windows_require_every_future_variance(self):
        dates = pd.bdate_range("2024-01-02", periods=12)
        rv = pd.Series(np.arange(1, 13, dtype=float), index=dates)
        rv.iloc[4] = np.nan
        got = study.make_targets(rv, 5)
        self.assertTrue(got.loc[dates[:4], "y"].isna().all())
        self.assertAlmostEqual(got.loc[dates[4], "y"], np.mean([6, 7, 8, 9, 10]))
        one_day = study.make_targets(rv, 1)
        self.assertTrue(np.isnan(one_day.loc[dates[3], "y"]))
        self.assertEqual(one_day.loc[dates[4], "target_end"], dates[5])

    def test_training_purges_unobserved_five_day_targets_and_uses_common_rows(self):
        features, y = regression_fixture()
        targets = study.make_targets(y, 5)
        origin = features.index[100]
        features.loc[features.index[40], "vvix"] = np.nan
        features.loc[features.index[41], "liv"] = np.inf
        targets.loc[features.index[42], "y"] = np.nan
        got = study.training_mask(features, targets, origin, min_train=20)
        expected = pd.Series(False, index=features.index)
        expected.iloc[:96] = True  # row 95's t+5 target ends exactly at origin.
        expected.iloc[[40, 41, 42]] = False
        np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())
        self.assertTrue(got.loc[features.index[95]])
        self.assertFalse(got.loc[features.index[96:101]].any())

    def test_insufficient_common_training_rows_raise(self):
        features, y = regression_fixture()
        targets = study.make_targets(y, 1)
        with self.assertRaises(ValueError):
            study.training_mask(features, targets, features.index[80], min_train=100)

    def test_any_zero_scale_candidate_rejects_entire_training_fold(self):
        features, y = regression_fixture()
        targets = study.make_targets(y, 1)
        origin = features.index[120]
        for candidate in CANDIDATES:
            with self.subTest(candidate=candidate):
                changed = features.copy()
                changed.loc[changed.index < origin, candidate] = 2.0
                with self.assertRaises(ValueError):
                    study.training_mask(changed, targets, origin, min_train=20)


class EstimatorAndInferenceContractTests(unittest.TestCase):
    def test_each_forecast_matches_independent_direct_ols_and_exact_duan_smearing(self):
        features, y = regression_fixture()
        train, origin = features.iloc[:140], features.iloc[140]
        train_y = y.iloc[:140]
        got = study.fit_predict(train, train_y, origin)
        self.assertEqual(set(got["forecasts"]), {"baseline", *CANDIDATES})
        self.assertEqual(set(got["orthogonal_r2"]), set(CANDIDATES))
        log_y = np.log(train_y.to_numpy())
        for candidate in [None, *CANDIDATES]:
            columns = BASELINE + ([] if candidate is None else [candidate])
            design = train[columns].to_numpy()
            beta = np.linalg.lstsq(design, log_y, rcond=None)[0]
            residuals = log_y - design @ beta
            expected = np.exp(origin[columns].to_numpy() @ beta) * np.exp(residuals).mean()
            self.assertAlmostEqual(got["forecasts"][candidate or "baseline"], expected, places=13)
            if candidate is not None:
                base = train[BASELINE].to_numpy()
                values = train[candidate].to_numpy()
                projection = np.linalg.lstsq(base, values, rcond=None)[0]
                expected_r2 = 1 - np.sum((values - base @ projection)**2) / np.sum((values - values.mean())**2)
                self.assertAlmostEqual(got["orthogonal_r2"][candidate], expected_r2, places=12)

    def test_prediction_ignores_future_rows_and_candidate_baseline_component(self):
        features, y = regression_fixture()
        cutoff = 140
        expected = study.fit_predict(features.iloc[:cutoff], y.iloc[:cutoff], features.iloc[cutoff])
        changed, changed_y = features.copy(), y.copy()
        changed.iloc[cutoff + 1:] = 1e6
        changed_y.iloc[cutoff + 1:] = 1e6
        future_changed = study.fit_predict(changed.iloc[:cutoff], changed_y.iloc[:cutoff], changed.iloc[cutoff])
        for output in ["forecasts", "orthogonal_r2"]:
            self.assertEqual(set(future_changed[output]), set(expected[output]))
            for name in expected[output]:
                np.testing.assert_allclose(future_changed[output][name], expected[output][name],
                                           rtol=1e-12, atol=1e-15)
        changed["vvix"] = changed["vvix"] + 13 * changed["liv"] - 4 * changed["lev_m"]
        projected = study.fit_predict(changed.iloc[:cutoff], changed_y.iloc[:cutoff], changed.iloc[cutoff])
        for arm in ["baseline", *CANDIDATES]:
            self.assertAlmostEqual(projected["forecasts"][arm], expected["forecasts"][arm], places=12)

    def test_qlike_is_scale_invariant_and_rejects_nonpositive_or_nonfinite_inputs(self):
        actual, predicted = np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 1.0])
        expected = actual / predicted - np.log(actual / predicted) - 1
        np.testing.assert_allclose(study.qlike(actual, predicted), expected)
        np.testing.assert_allclose(study.qlike(actual * 1000, predicted * 1000), expected)
        for invalid in [0.0, -1.0, np.nan, np.inf]:
            for which in ["actual", "predicted"]:
                with self.subTest(invalid=invalid, which=which), self.assertRaises(ValueError):
                    study.qlike(np.array([invalid]) if which == "actual" else np.array([1.0]),
                                np.array([invalid]) if which == "predicted" else np.array([1.0]))

    def test_holm_adjustment_preserves_order_and_monotone_stepdown(self):
        p_values = np.array([0.04, 0.001, 0.02, 0.5, 0.009, 0.03, 0.8, 0.015])
        # Sorted raw values .001,.009,.015,.02,.03,.04,.5,.8 times 8..1,
        # with cumulative maxima and a cap at one, mapped to original order.
        expected = np.array([0.12, 0.008, 0.1, 1.0, 0.063, 0.12, 1.0, 0.09])
        np.testing.assert_allclose(study.holm_adjust(p_values), expected)
        permutation = np.array([5, 2, 0, 7, 4, 1, 6, 3])
        np.testing.assert_allclose(study.holm_adjust(p_values[permutation]), expected[permutation])


if __name__ == "__main__":
    unittest.main()
