"""Prewritten independent overnight-label and pre-close information contracts."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import verify_overnight_index as verify


def fixture():
    rng = np.random.default_rng(192026)
    dates = pd.bdate_range("2010-01-04", periods=420)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .005, len(dates))))
    opening = close * np.exp(rng.normal(0, .002, len(dates)))
    daily = pd.DataFrame({"open": opening, "close": close,
                          "high": np.maximum(opening, close) * 1.003,
                          "low": np.minimum(opening, close) / 1.003,
                          "adj close": close * .8,
                          "volume": np.exp(rng.normal(15, .3, len(dates)))}, index=dates)
    cross = pd.DataFrame({name: np.exp(np.cumsum(rng.normal(0, .005, len(dates))))
                          for name in ("hyg", "tlt", "gld", "uso", "uup")}, index=dates)
    iv = pd.DataFrame({name: np.exp(rng.normal(3, .1, len(dates)))
                      for name in ("vxn", "vix", "vix9d", "vvix")}, index=dates)
    return daily, cross, iv


class OvernightTimingTests(unittest.TestCase):
    def test_adjusted_target_equals_raw_overnight_times_adjustment_ratio(self):
        daily, _, _ = fixture()
        daily.loc[daily.index[201]:, "adj close"] *= 1.01
        target = verify.overnight_targets(daily)
        position = 200
        factor = daily["adj close"] / daily.close
        expected = daily.open.iloc[position + 1] / daily.close.iloc[position] * factor.iloc[position + 1] / factor.iloc[position] - 1
        self.assertAlmostEqual(target.y.iloc[position], expected, places=13)
        self.assertAlmostEqual(target.adjustment_jump.iloc[position], np.log(1.01), places=13)
        self.assertEqual(target.target_end.iloc[position], daily.index[position + 1])
        self.assertEqual(target.available_date.iloc[position], daily.index[position + 1])

    def test_zero_return_is_valid_and_last_label_is_missing(self):
        daily, _, _ = fixture()
        daily.loc[daily.index[201], "open"] = daily.close.iloc[200]
        target = verify.overnight_targets(daily)
        self.assertAlmostEqual(target.y.iloc[200], 0., places=13)
        self.assertTrue(np.isnan(target.y.iloc[-1]))
        self.assertTrue(pd.isna(target.available_date.iloc[-1]))

    def test_all_price_volume_cross_and_iv_features_stop_at_previous_close(self):
        daily, cross, iv = fixture()
        entry = daily.index[300]
        original = verify.overnight_features(daily, cross, iv)
        revised_daily, revised_cross, revised_iv = daily.copy(), cross.copy(), iv.copy()
        revised_daily.loc[entry:] *= 1.2
        revised_cross.loc[entry:] *= 1.5
        revised_iv.loc[entry:] *= 2
        revised = verify.overnight_features(revised_daily, revised_cross, revised_iv)
        np.testing.assert_allclose(original.loc[entry], revised.loc[entry], rtol=0, atol=0)

    def test_strictly_prior_volume_normalization_uses_sample_sd_and_current_pressure(self):
        daily, _, _ = fixture()
        z = verify.prior_volume_z(daily.volume)
        position = 300
        historical = np.log(daily.volume.iloc[position - 252:position])
        expected = (np.log(daily.volume.iloc[position]) - historical.mean()) / historical.std(ddof=1)
        self.assertAlmostEqual(z.iloc[position], expected, places=12)
        self.assertTrue(np.isnan(z.iloc[125]))
        self.assertTrue(np.isfinite(z.iloc[126]))
        altered = daily.volume.copy()
        altered.iloc[position] *= 10
        difference = verify.prior_volume_z(altered).iloc[position] - z.iloc[position]
        self.assertAlmostEqual(difference, np.log(10) / historical.std(ddof=1), places=12)

    def test_nonpositive_volume_is_missing_and_does_not_get_logged(self):
        daily, _, _ = fixture()
        daily.loc[daily.index[300], "volume"] = 0
        self.assertTrue(np.isnan(verify.prior_volume_z(daily.volume).iloc[300]))

    def test_training_labels_must_be_available_by_previous_session(self):
        daily, _, _ = fixture()
        target = verify.overnight_targets(daily)
        complete = pd.Series(True, index=daily.index)
        entry = daily.index[300]
        mask = verify.training_mask(complete, target, entry, daily.index)
        self.assertTrue(mask.iloc[298])
        self.assertFalse(mask.iloc[299])
        self.assertFalse(mask.iloc[300])

    def test_entry_weekday_is_known_without_the_next_session(self):
        daily, cross, iv = fixture()
        full = verify.overnight_features(daily, cross, iv)
        entry = daily.index[300]
        truncated = verify.overnight_features(daily.loc[:entry], cross.loc[:entry], iv.loc[:entry])
        np.testing.assert_allclose(full.loc[entry], truncated.loc[entry], rtol=0, atol=0)

    def test_measurement_subset_boundary_is_inclusive_and_never_filters_training(self):
        jumps = np.array([0., 1e-5, -1e-5, 1.001e-5, np.nan])
        np.testing.assert_array_equal(verify.measurement_mask(jumps), [True, True, True, False, False])


class RidgeAndFamilyTests(unittest.TestCase):
    def test_ridge_objective_uses_mean_loss_and_preserves_zero_targets(self):
        x = np.column_stack([np.ones(30), np.arange(30), np.sin(np.arange(30))])
        y = np.linspace(-.01, .01, len(x))
        prediction, audit = verify.ridge_prediction(x, y, x, alpha=.01)
        z = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)
        beta = np.linalg.solve(z.T @ z / len(z) + .01 * np.eye(2), z.T @ (y - y.mean()) / len(z))
        np.testing.assert_allclose(prediction, y.mean() + z @ beta, rtol=1e-12, atol=1e-14)
        zero, _ = verify.ridge_prediction(x, y * 0, x, alpha=.01)
        np.testing.assert_array_equal(zero, np.zeros(len(x)))
        np.testing.assert_allclose(audit["beta"], beta, rtol=1e-12, atol=1e-14)

    def test_cumulative_family_retains_all78_trials(self):
        adjusted = verify.holm([.0001] + [1.] * 77)
        self.assertAlmostEqual(adjusted[0], .0078)
        self.assertEqual(len(adjusted), 78)


if __name__ == "__main__":
    unittest.main()
