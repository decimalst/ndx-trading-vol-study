"""Prewritten generated clocks, missingness and numerical commodity contracts."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.commodity_implied_features import _strict_square, build_commodity_features


def fixture():
    index = pd.bdate_range("2008-12-29", periods=65)
    x = np.arange(len(index), dtype=float)
    cross = pd.DataFrame(
        {
            "hyg": 90.0,
            "tlt": 100.0,
            "gld": 80 * np.exp(0.007 * x + 0.03 * np.sin(x / 3)),
            "uso": 40 * np.exp(-0.002 * x + 0.04 * np.cos(x / 4)),
            "uup": 20.0,
        },
        index=index,
    )
    implied = pd.DataFrame({"OVX": 20 + x / 3, "GVZ": 15 + x / 7}, index=index)
    return index, cross, implied


class CommodityFeatureTests(unittest.TestCase):
    def test_exact_schema_and_independent_scalar_lagged_history(self):
        index, cross, implied = fixture()
        actual = build_commodity_features(index, cross, implied)
        self.assertEqual(
            actual.columns.tolist(),
            [
                "uso_ret",
                "uso_r2",
                "uso_lrv5",
                "uso_lrv22",
                "gld_ret",
                "gld_r2",
                "gld_lrv5",
                "gld_lrv22",
                "lovx",
                "lgvz",
                "commodity_cutoff_date",
            ],
        )
        pd.testing.assert_index_equal(actual.index, index)
        origin = 40
        for asset in ("uso", "gld"):
            returns = [
                math.log(cross[asset].iloc[j] / cross[asset].iloc[j - 1])
                for j in range(origin - 22, origin)
            ]
            self.assertAlmostEqual(actual.iloc[origin][asset + "_ret"], returns[-1], places=13)
            self.assertAlmostEqual(
                actual.iloc[origin][asset + "_r2"], returns[-1] ** 2, places=13
            )
            for window in (5, 22):
                expected = math.log(math.fsum(r * r for r in returns[-window:]) / window)
                self.assertAlmostEqual(
                    actual.iloc[origin][asset + f"_lrv{window}"], expected, places=11
                )
        self.assertAlmostEqual(
            actual.lovx.iloc[origin], math.log((implied.OVX.iloc[origin - 1] / 100) ** 2 / 252)
        )
        self.assertEqual(actual.commodity_cutoff_date.iloc[origin], index[origin - 1])

    def test_2009_floor_blocks_both_prices_and_iv_before_any_history(self):
        index, cross, implied = fixture()
        actual = build_commodity_features(index, cross, implied)
        early = index < pd.Timestamp("2009-01-02")
        cross.loc[early, ["uso", "gld"]] = -np.inf
        implied.loc[early, :] = -123.0
        changed = build_commodity_features(index, cross, implied)
        assert_frame_equal(changed, actual)
        self.assertTrue(actual.loc[:"2009-01-02"].iloc[:, :10].isna().all().all())
        self.assertTrue(actual.loc[:"2009-01-02", "commodity_cutoff_date"].isna().all())
        self.assertTrue(pd.isna(actual.loc["2009-01-05", "uso_ret"]))
        self.assertTrue(np.isfinite(actual.loc["2009-01-05", "lovx"]))
        self.assertTrue(np.isfinite(actual.loc["2009-01-06", "uso_ret"]))
        first = index.get_loc(pd.Timestamp("2009-01-02"))
        self.assertTrue(pd.isna(actual.uso_lrv22.iloc[first + 22]))
        self.assertTrue(np.isfinite(actual.uso_lrv22.iloc[first + 23]))

    def test_same_day_mutation_cannot_change_current_or_earlier_features(self):
        index, cross, implied = fixture()
        expected = build_commodity_features(index, cross, implied)
        at = index[40]
        cross.loc[at, "uso"] *= 1.2
        implied.loc[at, "OVX"] *= 2
        changed = build_commodity_features(index, cross, implied)
        assert_frame_equal(changed.loc[:at], expected.loc[:at])
        self.assertNotEqual(changed.uso_ret.iloc[41], expected.uso_ret.iloc[41])
        self.assertNotEqual(changed.lovx.iloc[41], expected.lovx.iloc[41])

    def test_missing_dates_are_aligned_before_delay_and_windows_never_skip(self):
        index, cross, implied = fixture()
        cross = cross.drop(index[30])
        implied = implied.drop(index[35])
        actual = build_commodity_features(index, cross, implied)
        self.assertTrue(actual.uso_ret.iloc[31:33].isna().all())
        self.assertTrue(actual.uso_lrv5.iloc[31:37].isna().all())
        self.assertTrue(np.isfinite(actual.uso_lrv5.iloc[37]))
        self.assertTrue(np.isnan(actual.lovx.iloc[36]))
        self.assertTrue(np.isfinite(actual.lovx.iloc[35]))
        self.assertTrue(np.isfinite(actual.lovx.iloc[37]))
        self.assertEqual(actual.commodity_cutoff_date.iloc[36], index[35])

    def test_zero_returns_valid_and_all_zero_variance_windows_unknown(self):
        index, cross, implied = fixture()
        cross[["uso", "gld"]] = 50.0
        actual = build_commodity_features(index, cross, implied)
        self.assertEqual(actual.uso_ret.iloc[-1], 0.0)
        self.assertEqual(actual.uso_r2.iloc[-1], 0.0)
        self.assertTrue(
            actual[["uso_lrv5", "uso_lrv22", "gld_lrv5", "gld_lrv22"]].isna().all().all()
        )

    def test_iv_log_variance_avoids_square_underflow_for_positive_extremes(self):
        index, cross, implied = fixture()
        implied.loc[index[40], "OVX"] = np.nextafter(0.0, 1.0)
        implied.loc[index[40], "GVZ"] = np.finfo(float).max
        actual = build_commodity_features(index, cross, implied)
        for column, name in (("OVX", "lovx"), ("GVZ", "lgvz")):
            expected = 2 * (
                math.log(implied.loc[index[40], column]) - math.log(100)
            ) - math.log(252)
            self.assertTrue(np.isfinite(actual.loc[index[41], name]))
            self.assertAlmostEqual(actual.loc[index[41], name], expected)

    def test_nonzero_square_underflow_and_overflow_abort_instead_of_zero_or_missing(self):
        for value in (1e-200, 1e200):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _strict_square(np.array([value]))
        np.testing.assert_array_equal(_strict_square(np.array([0.0, -2.0])), [0.0, 4.0])

    def test_all_calendars_checked_before_numerical_arithmetic(self):
        index, cross, implied = fixture()
        extra = pd.DataFrame(
            {"OVX": [-1.0], "GVZ": [-1.0]}, index=pd.DatetimeIndex(["2025-10-21"])
        )
        implied = pd.concat([implied, extra])
        with patch("src.commodity_implied_features.np.log") as logged:
            with self.assertRaises(ValueError):
                build_commodity_features(index, cross, implied)
            logged.assert_not_called()

    def test_invalid_shapes_types_values_and_dates_rejected(self):
        for defect in (
            "negative",
            "zero",
            "inf",
            "bool",
            "complex",
            "duplicate",
            "timezone",
            "midday",
            "wrong_iv_case",
            "missing_cross_name",
        ):
            index, cross, implied = fixture()
            if defect == "negative":
                cross.loc[index[40], "uso"] = -1.0
            elif defect == "zero":
                implied.loc[index[40], "OVX"] = 0.0
            elif defect == "inf":
                implied.loc[index[40], "GVZ"] = np.inf
            elif defect == "bool":
                cross["uso"] = True
            elif defect == "complex":
                implied["OVX"] = implied.OVX.astype(complex)
            elif defect == "duplicate":
                cross = pd.concat([cross, cross.iloc[-1:]])
            elif defect == "timezone":
                implied.index = implied.index.tz_localize("UTC")
            elif defect == "midday":
                index = index + pd.Timedelta(hours=1)
            elif defect == "wrong_iv_case":
                implied = implied.rename(columns={"OVX": "ovx"})
            else:
                cross = cross.drop(columns="hyg")
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                build_commodity_features(index, cross, implied)

    def test_unrelated_asset_values_opaque_price_units_and_inputs_unchanged(self):
        index, cross, implied = fixture()
        old_cross, old_iv = cross.copy(deep=True), implied.copy(deep=True)
        expected = build_commodity_features(index, cross, implied)
        assert_frame_equal(cross, old_cross)
        assert_frame_equal(implied, old_iv)
        for name in ("hyg", "tlt", "uup"):
            cross[name] = [object() for _ in index]
        cross["uso"] *= 1000
        cross["gld"] /= 100
        changed = build_commodity_features(index, cross, implied)
        np.testing.assert_allclose(
            changed.iloc[:, :10], expected.iloc[:, :10], rtol=1e-10, atol=1e-11, equal_nan=True
        )


if __name__ == "__main__":
    unittest.main()
