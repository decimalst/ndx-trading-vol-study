"""Market/label timing contracts before macro second-moment empirical fits."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.macro_overnight import (
    ALL_FEATURES,
    BASE,
    MODELS,
    build_features,
    forecast_panel,
    training_mask,
)


def sample(n=90):
    dates = pd.bdate_range("2014-01-01", periods=n)
    rng = np.random.default_rng(4)
    close = 100*np.exp(np.cumsum(rng.normal(0, .01, n)))
    op = close*np.exp(rng.normal(0, .004, n))
    daily = pd.DataFrame({"open": op, "high": np.maximum(op, close)*1.002,
                          "low": np.minimum(op, close)*.998, "close": close,
                          "adj close": close*.8}, index=dates)
    iv = pd.DataFrame({"vxn": 20+np.arange(n)/30, "vix": 15+np.arange(n)/50}, index=dates)
    plans = pd.DataFrame({"nominal_hours": np.where(dates.weekday == 4, 65.5, 17.5),
                          "cpi_plan": (np.arange(n)%22 == 0).astype(float),
                          "nfp_plan": (np.arange(n)%22 == 5).astype(float),
                          "fomc_plan": (np.arange(n)%33 == 8).astype(float)}, index=dates)
    return daily, iv, plans


class TestMacroOvernight(unittest.TestCase):
    def test_target_is_squared_log_proxy_and_matures_at_next_session_close(self):
        daily, iv, plans = sample()
        features, target = build_features(daily, iv, plans)
        t = 30
        log_return = np.log(daily.iloc[t+1]["adj close"]/daily.iloc[t]["adj close"])-np.log(daily.iloc[t+1].close/daily.iloc[t+1].open)
        self.assertAlmostEqual(target.iloc[t].y, log_return**2, places=14)
        self.assertEqual(target.iloc[t].available_date, daily.index[t+1])
        self.assertEqual(features.iloc[t].feature_cutoff_date, daily.index[t-1])

    def test_market_features_lag_a_full_observed_session(self):
        daily, iv, plans = sample()
        before, _ = build_features(daily, iv, plans)
        changed, other_iv = daily.copy(), iv.copy()
        changed.iloc[40:, :] *= 4
        other_iv.iloc[40:, :] *= 3
        after, _ = build_features(changed, other_iv, plans)
        pd.testing.assert_frame_equal(before.iloc[:41], after.iloc[:41])

    def test_next_closing_price_cancels_when_adjustment_factor_is_fixed(self):
        daily, iv, plans = sample()
        _, before = build_features(daily, iv, plans)
        changed = daily.copy()
        changed.loc[daily.index[41], ["close", "adj close"]] *= 1.3
        changed.loc[daily.index[41], "high"] *= 1.3
        _, after = build_features(changed, iv, plans)
        self.assertAlmostEqual(before.iloc[40].y, after.iloc[40].y, places=14)

    def test_zero_squared_target_is_retained_without_floor(self):
        daily, iv, plans = sample()
        daily.loc[:, ["open", "close", "adj close"]] = 100.
        daily.loc[:, "high"], daily.loc[:, "low"] = 101., 99.
        _, target = build_features(daily, iv, plans)
        self.assertTrue((target.y.iloc[:-1] == 0).all())

    def test_missing_market_row_stays_inside_strict_rolling_window(self):
        daily, iv, plans = sample()
        daily.loc[daily.index[40], "open"] = np.nan
        features, _ = build_features(daily, iv, plans)
        self.assertTrue(np.isnan(features.iloc[45].on_rms_w))
        self.assertTrue(np.isfinite(features.iloc[46].on_rms_w))

    def test_unknown_plan_is_missing_and_never_a_zero_event(self):
        daily, iv, plans = sample()
        plans.loc[daily.index[40], "cpi_plan"] = np.nan
        features, _ = build_features(daily, iv, plans)
        self.assertTrue(np.isnan(features.iloc[40].cpi_plan))
        self.assertTrue(np.isfinite(features.iloc[40].nfp_plan))

    def test_training_requires_completed_labels_and_all_three_calendar_arms(self):
        daily, iv, plans = sample()
        features, target = build_features(daily, iv, plans)
        mask = training_mask(features, target, daily.index[70], min_train=20)
        self.assertTrue((target.loc[mask, "available_date"] <= daily.index[69]).all())
        self.assertFalse(mask.iloc[69])
        self.assertTrue(mask.iloc[68])
        features.loc[daily.index[40], "fomc_plan"] = np.nan
        new_mask = training_mask(features, target, daily.index[70], min_train=20)
        self.assertFalse(new_mask.iloc[40])
        self.assertEqual(mask.sum()-new_mask.sum(), 1)

    def test_adjustment_changes_are_retained_for_retrospective_sensitivity(self):
        daily, iv, plans = sample()
        daily.loc[daily.index[40]:, "adj close"] *= 1.001
        features, target = build_features(daily, iv, plans)
        self.assertTrue(target.iloc[39].adjustment_event)
        mask = training_mask(features, target, daily.index[70], min_train=20)
        self.assertTrue(mask.iloc[39])

    def test_future_realized_session_removal_cannot_change_current_predictors(self):
        daily, iv, plans = sample()
        before, target_before = build_features(daily, iv, plans)
        keep = daily.index.delete(41)
        after, target_after = build_features(daily.loc[keep], iv.loc[keep], plans.loc[keep])
        pd.testing.assert_series_equal(before.iloc[40], after.iloc[40])
        self.assertNotEqual(target_before.iloc[40].target_end, target_after.iloc[40].target_end)

    def test_feature_schema_and_invalid_calendar_values(self):
        daily, iv, plans = sample()
        features, _ = build_features(daily, iv, plans)
        self.assertEqual(tuple(features.columns[:-1]), ALL_FEATURES)
        plans.iloc[30, 1] = 2
        with self.assertRaises(ValueError):
            build_features(daily, iv, plans)

    def test_monthly_refit_origin_does_not_depend_on_first_query_label(self):
        daily, iv, plans = sample()
        features, target = build_features(daily, iv, plans)
        dates = daily.index
        config = {"models": list(MODELS), "baseline": list(BASE), "minimum_train": 20,
                  "origin_start": str(dates[50].date()), "origin_end": str(dates[88].date()),
                  "latest_target": str(dates[89].date()),
                  "development": [str(dates[50].date()), str(dates[64].date())],
                  "development_target_available_by": str(dates[64].date()),
                  "evaluation": [str(dates[65].date()), str(dates[88].date())]}
        def fake_fit(train, y, application):
            return {name: np.full(len(application), float(y.mean())) for name in MODELS}, {}
        with patch("src.macro_overnight.fit_predict", side_effect=fake_fit):
            panel, fits = forecast_panel(features, target, config)
            changed = target.copy()
            changed.loc[dates[50], "y"] = np.nan
            after, new_fits = forecast_panel(features, changed, config)
        self.assertEqual(fits[0]["fit_origin"], str(dates[50].date()))
        self.assertEqual(new_fits[0]["fit_origin"], fits[0]["fit_origin"])
        self.assertTrue((panel.origin == dates[50]).any())
        self.assertFalse((after.origin == dates[50]).any())


if __name__ == "__main__":
    unittest.main()
