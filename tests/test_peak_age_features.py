"""Generated, preimplementation contracts; never opens historical sources."""
from __future__ import annotations

import math
import tempfile
import unittest
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import peak_age_features as feature

RAW = ("const", "I", "R", "ret_d", "ret_w", "ret_m", "ret_q", "lr_d", "lr_w",
       "term", "lvvix", "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
NEW = ("peak_age", "drawdown", "drawdown_sq", "window_return")


def sample(n=600, close=None):
    dates = pd.bdate_range("2010-01-04", periods=n + 2, name="date").delete([15, 80])
    t = np.arange(n, dtype=float)
    if close is None:
        close = 100 * np.exp(np.cumsum(.001 + .008 * np.sin(.37 * t)))
    close = np.asarray(close, dtype=float)
    opening = close * np.exp(.004 * np.cos(.23 * t))
    daily = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * 1.006,
                          "low": np.minimum(opening, close) * .994, "close": close}, index=dates)
    iv = pd.DataFrame({"vix": 18 + 3 * np.sin(.17 * t), "vix9d": 16 + 2 * np.cos(.21 * t),
                       "vvix": 85 + 4 * np.cos(.19 * t)}, index=dates)
    return daily, iv


class PeakAgeFeatureTests(unittest.TestCase):
    def test_literal_schema_and_sole_target(self):
        self.assertEqual(feature.RAW, RAW)
        self.assertEqual(feature.BASE, RAW + ("I_square", "R_square"))
        self.assertEqual(feature.COMMON, RAW + NEW)
        f, target, state = feature.build_features(*sample())
        self.assertEqual(tuple(f), RAW + NEW + ("feature_cutoff_date",))
        self.assertEqual(tuple(target), ("y", "target_end", "available_date"))
        self.assertEqual(tuple(state), ("origin_position", "cutoff_position", "window_start_position",
                         "window_end_position", "peak_position", "peak_age_sessions", "window_start_date",
                         "window_end_date", "peak_date", "feature_cutoff_date", "window_complete"))
        self.assertTrue(f.index.equals(target.index) and f.index.equals(state.index))
        self.assertFalse(f.iloc[:252][list(NEW)].notna().any().any())

    def test_exact_252_closes_251_intervals_and_latest_ties(self):
        close = np.full(600, 100.)
        close[[20, 99, 200]] = 200.
        f, _, state = feature.build_features(*sample(close=close))
        row = state.iloc[252]
        self.assertEqual((row.window_start_position, row.window_end_position), (0, 251))
        self.assertEqual((row.peak_position, row.peak_age_sessions), (200, 51))
        self.assertEqual(row.peak_date, f.index[200])
        self.assertEqual(f.iloc[252].peak_age, 51 / 251)
        self.assertAlmostEqual(f.iloc[252].drawdown, math.log(2))
        self.assertEqual(f.iloc[252].window_return, 0.)
        close[201] = np.nextafter(200., 0.)
        _, _, state = feature.build_features(*sample(close=close))
        self.assertEqual(state.iloc[252].peak_position, 200)

    def test_flat_increasing_decreasing_and_expiring_maximum(self):
        for close, age in [(np.ones(600) * 100, 0), (np.arange(600) + 100., 0),
                           (1000. - np.arange(600), 251)]:
            with self.subTest(age=age):
                f, _, state = feature.build_features(*sample(close=close))
                self.assertTrue((state.peak_age_sessions.iloc[252:] == age).all())
                np.testing.assert_array_equal(f.peak_age.iloc[252:], np.full(348, age / 251))
        close = np.full(600, 100.)
        close[0], close[200] = 200., 150.
        f, _, state = feature.build_features(*sample(close=close))
        self.assertEqual(state.iloc[252].peak_age_sessions, 251)
        self.assertEqual(state.iloc[253].peak_age_sessions, 52)
        self.assertLess(f.iloc[253].drawdown, f.iloc[252].drawdown)
        self.assertEqual(close[251], close[252])

    def test_one_origin_algebraic_distinction_with_identical_recent_controls(self):
        daily, iv = sample()
        first, second = daily.copy(), daily.copy()
        for frame, peak in [(first, 60), (second, 130)]:
            value = 2 * daily.close.iloc[:252].max()
            frame.loc[frame.index[peak], ["open", "high", "low", "close"]] = value
        a, _, _ = feature.build_features(first, iv)
        b, _, _ = feature.build_features(second, iv)
        # The original rolling accumulator can retain sub-ulp history rounding
        # even after altered returns expire; do not change its frozen formula.
        np.testing.assert_allclose(a.iloc[252][list(RAW)].to_numpy(float),
                                   b.iloc[252][list(RAW)].to_numpy(float), rtol=1e-8, atol=1e-12)
        columns = ["drawdown", "drawdown_sq", "window_return"]
        np.testing.assert_array_equal(a.iloc[252][columns], b.iloc[252][columns])
        self.assertNotEqual(a.iloc[252].peak_age, b.iloc[252].peak_age)

    def test_original_formulas_and_previous_observed_session_cutoff(self):
        daily, iv = sample()
        f, _, _ = feature.build_features(daily, iv)
        returns = np.log(daily.close / daily.close.shift())
        gk = (.5 * np.log(daily.high / daily.low)**2
              - (2 * np.log(2) - 1) * np.log(daily.close / daily.open)**2).clip(lower=1e-10)
        variance = gk + np.log(daily.open / daily.close.shift())**2
        i = 350
        for suffix, width in [("d", 1), ("w", 5), ("m", 22), ("q", 63)]:
            self.assertAlmostEqual(f.iloc[i]["ret_" + suffix], returns.iloc[i-width:i].mean())
        for name, width in [("R", 22), ("lr_d", 1), ("lr_w", 5)]:
            self.assertAlmostEqual(f.iloc[i][name], np.log(252 * variance.iloc[i-width:i].mean()))
        self.assertEqual(f.iloc[i].feature_cutoff_date, daily.index[i-1])
        self.assertAlmostEqual(f.iloc[i].I, np.log((iv.iloc[i-1].vix / 100)**2))
        self.assertAlmostEqual(f.iloc[i].term, np.log(iv.iloc[i-1].vix9d / iv.iloc[i-1].vix))

    def test_current_and_future_values_cannot_change_already_cut_off_features(self):
        daily, iv = sample()
        before, old_target, old_state = feature.build_features(daily, iv)
        changed, changed_iv = daily.copy(), iv.copy()
        changed.iloc[350:] *= np.linspace(2, 4, len(changed) - 350)[:, None]
        changed_iv.iloc[350:] *= 2
        after, target, state = feature.build_features(changed, changed_iv)
        pd.testing.assert_frame_equal(before.iloc[:351], after.iloc[:351])
        pd.testing.assert_frame_equal(old_state.iloc[:351], state.iloc[:351])
        self.assertNotEqual(old_target.iloc[340].y, target.iloc[340].y)

    def test_missing_close_preserves_full_window_and_recovers_only_after_expiry(self):
        daily, iv = sample()
        daily.loc[daily.index[270], ["open", "high", "low", "close"]] = np.nan
        f, target, state = feature.build_features(daily, iv)
        self.assertEqual(len(f), 600)
        self.assertTrue(f.iloc[271:523][list(NEW)].isna().all().all())
        self.assertTrue(state.peak_age_sessions.iloc[271:523].isna().all())
        self.assertFalse(state.window_complete.iloc[271:523].any())
        self.assertTrue(state.iloc[523].window_complete)
        self.assertEqual(state.iloc[300].window_start_position, 48)
        self.assertTrue(np.isnan(target.iloc[249].y))
        self.assertEqual(target.iloc[249].target_end, daily.index[270])

    def test_missing_iv_does_not_fill_or_modify_peak_calendar(self):
        daily, iv = sample()
        f, _, state = feature.build_features(daily, iv.drop(iv.index[300]))
        self.assertTrue(f.iloc[301][["I", "term", "lvvix"]].isna().all())
        self.assertTrue(state.iloc[301].window_complete)
        self.assertTrue(np.isfinite(f.iloc[302].I))

    def test_observed_interval_targets_allow_zero_negative_and_missing_interiors(self):
        daily, iv = sample()
        daily.loc[daily.index[300], ["open", "high", "low", "close"]] = 100.
        daily.loc[daily.index[321], ["open", "high", "low", "close"]] = 90.
        daily.loc[daily.index[310], ["open", "high", "low", "close"]] = np.nan
        _, targets, _ = feature.build_features(daily, iv)
        self.assertAlmostEqual(targets.iloc[300].y, math.log(.9))
        self.assertEqual(targets.iloc[300].target_end, daily.index[321])
        self.assertEqual(targets.iloc[300].available_date, daily.index[321])
        self.assertTrue(targets.iloc[-21:].isna().all().all())
        daily.loc[daily.index[321], ["open", "high", "low", "close"]] = 100.
        _, targets, _ = feature.build_features(daily, iv)
        self.assertEqual(targets.iloc[300].y, 0.)

    def test_invalid_prices_and_calendar_shapes_fail(self):
        daily, iv = sample()
        for value in [0., -1., np.inf, -np.inf]:
            changed = daily.copy()
            changed.iloc[300, changed.columns.get_loc("close")] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                feature.build_features(changed, iv)
        for index in [daily.index[::-1], daily.index.tz_localize("UTC"),
                      daily.index + pd.Timedelta(hours=1), pd.Index(range(len(daily)))]:
            changed = daily.copy()
            changed.index = index
            with self.subTest(index=str(index.dtype)), self.assertRaises(ValueError):
                feature.build_features(changed, iv)
        with self.assertRaises(ValueError):
            feature.build_features(pd.concat([daily, daily.iloc[-1:]]), iv)

    def test_fully_observed_arithmetic_fault_is_not_relabelled_as_missing(self):
        daily, iv = sample()
        daily.loc[daily.index[299], ["open", "high", "low", "close"]] = np.nextafter(0., 1.)
        daily.loc[daily.index[300], ["open", "high", "low", "close"]] = np.finfo(float).max
        with self.assertRaises(ValueError):
            feature.build_features(daily, iv)
        with patch.object(feature, "stable_log_ratio", return_value=1e-300), self.assertRaises(ValueError):
            feature.build_features(*sample())

    def test_log_ratio_matches_separate_high_precision_oracle(self):
        smallest, largest = np.nextafter(0., 1.), np.finfo(float).max
        cases = [(1., 1.), (smallest, largest), (largest, smallest), (1., np.nextafter(1., 2.)),
                 (np.nextafter(2., 0.), 2.), (2., np.nextafter(2., 0.)),
                 (np.nextafter(largest, 0.), largest), (smallest * 2, smallest),
                 (math.ldexp(1., -1022), np.nextafter(math.ldexp(1., -1022), 0.))]
        with localcontext() as context:
            context.prec = 160
            for x, y in cases:
                with self.subTest(x=x, y=y):
                    expected = float((Decimal.from_float(float(x)) / Decimal.from_float(float(y))).ln())
                    got = feature.stable_log_ratio(x, y)
                    if x == y:
                        self.assertEqual(got, 0.)
                    else:
                        self.assertTrue(math.isfinite(got) and got != 0)
                        self.assertEqual(got > 0, x > y)
                        self.assertTrue(math.isclose(got, expected, rel_tol=7e-16, abs_tol=0.))

    def test_log_ratio_rejects_bad_inputs_and_detects_arithmetic_zero(self):
        for invalid in [0., -1., np.nan, np.inf]:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                feature.stable_log_ratio(invalid, 1.)
        with patch.object(feature.math, "log1p", return_value=0.), self.assertRaises(ValueError):
            feature.stable_log_ratio(1., np.nextafter(1., 2.))

    def test_reversible_power_of_two_scaling_preserves_literal_peak_features(self):
        daily, iv = sample()
        before, _, state = feature.build_features(daily, iv)
        scaled = daily * 8
        after, _, after_state = feature.build_features(scaled, iv)
        np.testing.assert_array_equal(before[list(NEW)].to_numpy(), after[list(NEW)].to_numpy())
        pd.testing.assert_frame_equal(state, after_state)

    def test_generated_ms_us_ns_parquet_transport_preserves_exact_state_semantics(self):
        daily, iv = sample()
        expected, target_expected, state_expected = feature.build_features(daily, iv)
        with tempfile.TemporaryDirectory() as directory:
            for unit in ["ms", "us", "ns"]:
                d, v = daily.copy(), iv.copy()
                d.index, v.index = d.index.as_unit(unit), v.index.as_unit(unit)
                f, target, state = feature.build_features(d, v)
                for label, actual, reference in [("features", f, expected), ("target", target, target_expected),
                                                  ("state", state, state_expected)]:
                    path = Path(directory) / f"{label}-{unit}.parquet"
                    actual.to_parquet(path)
                    loaded = pd.read_parquet(path)
                    for column in loaded.select_dtypes(include="datetime").columns:
                        loaded[column] = loaded[column].dt.as_unit("ns")
                    loaded.index = loaded.index.as_unit("ns")
                    pd.testing.assert_frame_equal(loaded, reference, check_freq=False)


if __name__ == "__main__":
    unittest.main()
