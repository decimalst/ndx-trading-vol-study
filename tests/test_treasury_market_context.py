"""Prewritten contracts for the Treasury study's matched market context."""

import unittest

import numpy as np
import pandas as pd

from src.treasury_market_context import build_market_context


class TreasuryMarketContextTests(unittest.TestCase):
    def setUp(self):
        self.days = pd.bdate_range("2009-12-21", periods=100)
        self.cross = pd.DataFrame(
            {
                c: 100 * np.exp(0.002 * np.arange(100) + 0.001 * np.sin(np.arange(100)))
                for c in ("hyg", "tlt", "gld", "uso", "uup")
            },
            index=self.days,
        )

    def test_independent_delayed_return_and_rolling_oracle(self):
        got = build_market_context(self.days, self.cross)
        for i in range(35, 100):
            rr = [
                np.log(self.cross.tlt.iloc[k]) - np.log(self.cross.tlt.iloc[k - 1])
                for k in range(i - 22, i)
            ]
            self.assertAlmostEqual(got.tlt_ret.iloc[i], rr[-1], places=14)
            self.assertAlmostEqual(got.tlt_r2.iloc[i], rr[-1] ** 2, places=18)
            self.assertAlmostEqual(
                got.tlt_lrv5.iloc[i], np.log(sum(x * x for x in rr[-5:]) / 5)
            )
            self.assertAlmostEqual(got.tlt_lrv22.iloc[i], np.log(sum(x * x for x in rr) / 22))
        pd.testing.assert_index_equal(got.index, self.days)

    def test_current_and_future_prices_cannot_change_earlier_output(self):
        initial = build_market_context(self.days, self.cross)
        changed = self.cross.copy()
        changed.loc[self.days[60] :, "tlt"] *= 3
        other = build_market_context(self.days, changed)
        pd.testing.assert_frame_equal(initial.iloc[:61], other.iloc[:61])

    def test_missing_session_never_compresses_history(self):
        changed = self.cross.drop(self.days[50])
        got = build_market_context(self.days, changed)
        self.assertTrue(got.tlt_ret.iloc[51:53].isna().all())
        self.assertTrue(got.tlt_lrv22.iloc[51:74].isna().all())
        self.assertTrue(np.isfinite(got.tlt_lrv22.iloc[74]))
        pd.testing.assert_index_equal(got.index, self.days)

    def test_source_floor_prevents_pre_2010_history(self):
        got = build_market_context(self.days, self.cross)
        first = self.days.get_indexer([pd.Timestamp("2010-01-01")])[0]
        self.assertTrue(got.tlt_ret.iloc[: first + 2].isna().all())
        self.assertTrue(got.tlt_lrv22.iloc[: first + 23].isna().all())
        self.assertTrue(np.isfinite(got.tlt_lrv22.iloc[first + 23]))
        self.assertTrue(got.treasury_cutoff_date.iloc[: first + 1].isna().all())
        self.assertEqual(got.treasury_cutoff_date.iloc[first + 1], self.days[first])

    def test_weekday_reference_and_zero_returns(self):
        cross = self.cross.copy()
        cross["tlt"] = 100.0
        got = build_market_context(self.days, cross)
        self.assertEqual(got.tlt_ret.iloc[-1], 0.0)
        self.assertEqual(got.tlt_r2.iloc[-1], 0.0)
        self.assertTrue(got.tlt_lrv5.isna().all())
        self.assertTrue(got.tlt_lrv22.isna().all())
        for day, row in got.iterrows():
            for weekday in range(1, 5):
                self.assertEqual(row[f"weekday_{weekday}"], float(day.weekday() == weekday))

    def test_other_cross_asset_values_are_opaque(self):
        changed = self.cross.copy()
        for name in ("hyg", "gld", "uso", "uup"):
            changed[name] = "unread"
        pd.testing.assert_frame_equal(
            build_market_context(self.days, self.cross),
            build_market_context(self.days, changed),
        )

    def test_positive_price_and_native_numeric_validation(self):
        for value in (0.0, -1.0, np.inf):
            bad = self.cross.copy()
            bad.loc[self.days[60], "tlt"] = value
            with self.assertRaises(ValueError):
                build_market_context(self.days, bad)
        for value in ("100", True):
            bad = self.cross.copy()
            bad["tlt"] = value
            with self.assertRaises(ValueError):
                build_market_context(self.days, bad)

    def test_full_date_envelopes_checked_before_values(self):
        extra = self.cross.copy()
        extra.loc[pd.Timestamp("2025-11-03")] = 100.0
        with self.assertRaises(ValueError):
            build_market_context(self.days, extra)
        for days in (
            self.days[::-1],
            self.days.append(self.days[:1]),
            self.days.tz_localize("UTC"),
            self.days + pd.Timedelta(hours=1),
        ):
            with self.assertRaises(ValueError):
                build_market_context(days, self.cross)
        with self.assertRaises(ValueError):
            build_market_context(self.days, self.cross.drop(columns="hyg"))

    def test_input_preservation_and_date_units(self):
        original = self.cross.copy(deep=True)
        expected = build_market_context(self.days, self.cross)
        for unit in ("ms", "us", "ns"):
            cross = self.cross.copy()
            cross.index = cross.index.as_unit(unit)
            result = build_market_context(self.days.as_unit(unit), cross)
            np.testing.assert_array_equal(
                result.iloc[:, :8].to_numpy(), expected.iloc[:, :8].to_numpy()
            )
            self.assertTrue(
                pd.DatetimeIndex(result.treasury_cutoff_date).equals(
                    pd.DatetimeIndex(expected.treasury_cutoff_date)
                )
            )
        pd.testing.assert_frame_equal(original, self.cross)


if __name__ == "__main__":
    unittest.main()
