"""Generated market contracts and independent formula oracles; no source files."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import claims_release_market as market

MARKET = (
    "const",
    "lrv_d",
    "lrv_w",
    "lrv_m",
    "lev_d",
    "lev_w",
    "lev_m",
    "liv",
    "lvix",
    "term",
    "xasset_stress",
    "market_stress",
)


def fixture(n=340):
    rng = np.random.default_rng(7301)
    index = pd.bdate_range("2019-01-02", periods=n, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    opening = close * np.exp(rng.normal(0, 0.006, n))
    daily = pd.DataFrame(
        {
            "open": opening,
            "high": np.maximum(opening, close) * np.exp(rng.uniform(0.001, 0.01, n)),
            "low": np.minimum(opening, close) * np.exp(-rng.uniform(0.001, 0.01, n)),
            "close": close,
            "adj close": close * np.exp(np.linspace(-0.03, 0, n)),
            "volume": np.exp(rng.normal(10, 0.25, n)),
        },
        index=index,
    )
    cross = pd.DataFrame(
        {
            name: (40 + i) * np.exp(np.cumsum(rng.normal(0, 0.003 + i * 0.002, n)))
            for i, name in enumerate(("hyg", "tlt", "gld", "uso", "uup"))
        },
        index=index,
    )
    iv = pd.DataFrame(
        {
            "vxn": rng.uniform(12, 30, n),
            "vix": rng.uniform(10, 25, n),
            "vix9d": rng.uniform(10, 25, n),
        },
        index=index,
    )
    return daily, cross, iv


def prior_z_at(values, position):
    prior = np.asarray(values[max(0, position - 252) : position], dtype=float)
    prior = prior[np.isfinite(prior)]
    if len(prior) < 126 or not np.isfinite(values[position]):
        return np.nan
    scale = np.std(prior, ddof=1)
    return (values[position] - np.mean(prior)) / scale if scale > 0 else np.nan


class MarketFeaturesTests(unittest.TestCase):
    def test_exact_output_columns_full_index_and_no_unused_features(self):
        frames = fixture()
        actual = market.build_market_features(*frames)
        self.assertEqual(market.MARKET, MARKET)
        self.assertEqual(tuple(actual.columns), MARKET + ("rv_total",))
        pd.testing.assert_index_equal(actual.index, frames[0].index)
        np.testing.assert_array_equal(actual["const"], np.ones(len(actual)))
        self.assertTrue(np.isfinite(actual.iloc[-1]).all())

    def test_gk_raw_overnight_and_har_arithmetic_variance_means(self):
        daily, cross, iv = fixture()
        actual = market.build_market_features(daily, cross, iv)
        expected = np.full(len(daily), np.nan)
        for i in range(1, len(daily)):
            row = daily.iloc[i]
            intraday = max(
                0.5 * np.log(row.high / row.low) ** 2
                - (2 * np.log(2) - 1) * np.log(row.close / row.open) ** 2,
                1e-10,
            )
            expected[i] = intraday + np.log(row.open / daily.close.iloc[i - 1]) ** 2
        np.testing.assert_allclose(actual.rv_total, expected, rtol=1e-13, equal_nan=True)
        self.assertTrue(np.isnan(actual.rv_total.iloc[0]))
        for position in (22, 170, 339):
            self.assertAlmostEqual(
                actual.lrv_d.iloc[position], np.log(expected[position]), places=13
            )
            self.assertAlmostEqual(
                actual.lrv_w.iloc[position],
                np.log(np.mean(expected[position - 4 : position + 1])),
                places=13,
            )
            self.assertAlmostEqual(
                actual.lrv_m.iloc[position],
                np.log(np.mean(expected[position - 21 : position + 1])),
                places=13,
            )
        self.assertNotAlmostEqual(
            actual.lrv_w.iloc[170], np.mean(np.log(expected[166:171])), places=6
        )

    def test_gk_floor_applies_before_raw_overnight_is_added(self):
        daily, cross, iv = fixture(3)
        daily.loc[:, ["open", "high", "low", "close", "adj close"]] = np.array(
            [[100], [110], [110]]
        )
        actual = market.build_market_features(daily, cross, iv)
        self.assertAlmostEqual(actual.rv_total.iloc[1], 1e-10 + np.log(1.1) ** 2, places=14)
        self.assertEqual(actual.rv_total.iloc[2], 1e-10)

    def test_leverage_clips_after_rolling_adjusted_returns(self):
        daily, cross, iv = fixture(30)
        returns = np.tile([0.03, -0.01, 0.02, -0.03, 0.01], 6)
        daily["adj close"] = 100 * np.exp(np.cumsum(returns))
        actual = market.build_market_features(daily, cross, iv)
        r = np.diff(np.log(daily["adj close"].to_numpy()), prepend=np.nan)
        for position in (22, 28):
            self.assertAlmostEqual(actual.lev_d.iloc[position], min(r[position], 0), places=14)
            self.assertAlmostEqual(
                actual.lev_w.iloc[position],
                min(np.mean(r[position - 4 : position + 1]), 0),
                places=14,
            )
            self.assertAlmostEqual(
                actual.lev_m.iloc[position],
                min(np.mean(r[position - 21 : position + 1]), 0),
                places=14,
            )
        self.assertEqual(actual.lev_w.iloc[22], 0)
        self.assertLess(np.mean(np.minimum(r[18:23], 0)), 0)

    def test_iv_is_aligned_then_delayed_one_observed_daily_session(self):
        daily, cross, iv = fixture(10)
        iv = iv.drop(iv.index[4])
        actual = market.build_market_features(daily, cross, iv)
        self.assertTrue(actual.loc[daily.index[0], ["liv", "lvix", "term"]].isna().all())
        self.assertAlmostEqual(actual.liv.iloc[4], np.log(iv.vxn.iloc[3]), places=14)
        self.assertTrue(actual.loc[daily.index[5], ["liv", "lvix", "term"]].isna().all())
        self.assertAlmostEqual(
            actual.term.iloc[7],
            np.log(iv.loc[daily.index[6], "vix9d"] / iv.loc[daily.index[6], "vix"]),
            places=14,
        )

    def test_cross_stress_matches_past_only_252_sample_std_oracle(self):
        daily, cross, iv = fixture()
        actual = market.build_market_features(daily, cross, iv)
        returns = np.diff(np.log(cross.to_numpy()), axis=0, prepend=np.full((1, 5), np.nan))
        self.assertTrue(actual.xasset_stress.iloc[:127].isna().all())
        for position in (127, 200, 339):
            z = [prior_z_at(returns[:, j], position) for j in range(5)]
            self.assertAlmostEqual(
                actual.xasset_stress.iloc[position], np.sqrt(np.mean(np.square(z))), places=12
            )
        position = 200
        inclusive = (
            returns[position] - np.mean(returns[position - 125 : position + 1], axis=0)
        ) / np.std(returns[position - 125 : position + 1], axis=0, ddof=1)
        self.assertNotAlmostEqual(
            actual.xasset_stress.iloc[position], np.sqrt(np.mean(inclusive**2)), places=6
        )

    def test_market_stress_uses_positive_parts_of_two_prior_standardizations(self):
        daily, cross, iv = fixture()
        actual = market.build_market_features(daily, cross, iv)
        overnight = np.log(daily.open.to_numpy() / daily.close.shift(1).to_numpy()) ** 2
        share = np.clip(overnight / actual.rv_total.to_numpy(), 0, 1)
        logged_volume = np.log(daily.volume.to_numpy())
        checked_zero = False
        for position in (127, 170, 200, 280, 339):
            zv, zo = prior_z_at(logged_volume, position), prior_z_at(share, position)
            expected = np.sqrt((max(zv, 0) ** 2 + max(zo, 0) ** 2) / 2)
            self.assertAlmostEqual(actual.market_stress.iloc[position], expected, places=12)
            checked_zero |= zv < 0 or zo < 0
        self.assertTrue(checked_zero)

    def test_missing_cross_constituent_is_not_skipped_or_forward_filled(self):
        daily, cross, iv = fixture()
        cross.loc[cross.index[170], "hyg"] = np.nan
        actual = market.build_market_features(daily, cross, iv)
        self.assertTrue(actual.xasset_stress.iloc[170:172].isna().all())
        self.assertTrue(np.isfinite(actual.xasset_stress.iloc[172]))
        self.assertTrue(np.isfinite(actual.market_stress.iloc[170]))
        reduced = market.build_market_features(daily, cross.drop(cross.index[180]), iv)
        self.assertTrue(reduced.xasset_stress.iloc[180:182].isna().all())

    def test_missing_volume_overnight_and_adjusted_prices_remain_undefined(self):
        daily, cross, iv = fixture()
        daily.loc[daily.index[170], "volume"] = 0
        daily.loc[daily.index[180], "close"] = np.nan
        daily.loc[daily.index[190], "adj close"] = np.nan
        actual = market.build_market_features(daily, cross, iv)
        self.assertTrue(np.isnan(actual.market_stress.iloc[170]))
        self.assertTrue(actual.rv_total.iloc[180:182].isna().all())
        self.assertTrue(actual.lrv_w.iloc[180:186].isna().all())
        self.assertTrue(actual.lev_w.iloc[190:196].isna().all())
        self.assertTrue(np.isfinite(actual.rv_total.iloc[190]))

    def test_genuine_nullable_numeric_missing_values_are_supported(self):
        daily, cross, iv = (x.astype("Float64") for x in fixture())
        daily.loc[daily.index[170], "volume"] = pd.NA
        iv.loc[iv.index[170], "vxn"] = pd.NA
        cross.loc[cross.index[170], "hyg"] = pd.NA
        actual = market.build_market_features(daily, cross, iv)
        self.assertTrue(np.isnan(actual.market_stress.iloc[170]))
        self.assertTrue(np.isnan(actual.liv.iloc[171]))
        self.assertTrue(np.isnan(actual.xasset_stress.iloc[170]))

    def test_zero_prior_scales_produce_missing_stresses_without_epsilon_fallback(self):
        daily, cross, iv = fixture()
        cross.loc[:, :] = 100.0
        daily["volume"] = 100.0
        actual = market.build_market_features(daily, cross, iv)
        self.assertTrue(actual.xasset_stress.isna().all())
        self.assertTrue(actual.market_stress.isna().all())
        self.assertTrue(np.isfinite(actual.lrv_m.iloc[-1]))

    def test_future_mutation_and_prefix_truncation_leave_features_unchanged(self):
        frames = fixture()
        actual = market.build_market_features(*frames)
        changed = copy.deepcopy(frames)
        for frame in changed:
            frame.iloc[201:] *= 1.7
        mutant = market.build_market_features(*changed)
        pd.testing.assert_frame_equal(actual.iloc[:201], mutant.iloc[:201], check_exact=True)
        prefix = market.build_market_features(*(f.iloc[:201] for f in frames))
        pd.testing.assert_frame_equal(actual.iloc[:201], prefix, check_exact=True)

    def test_missing_prior_observations_count_toward_minimum_without_calendar_compression(
        self,
    ):
        daily, cross, iv = fixture(140)
        cross.loc[cross.index[10], "hyg"] = np.nan
        actual = market.build_market_features(daily, cross, iv)
        self.assertTrue(actual.xasset_stress.iloc[:129].isna().all())
        self.assertTrue(np.isfinite(actual.xasset_stress.iloc[129]))
        pd.testing.assert_index_equal(actual.index, daily.index)

    def test_all_calendar_units_keep_the_same_values_and_session_order(self):
        frames = fixture(140)
        expected = market.build_market_features(*frames)
        for unit in ("ms", "us", "ns"):
            transported = [f.copy() for f in frames]
            for f in transported:
                f.index = f.index.as_unit(unit)
            actual = market.build_market_features(*transported)
            pd.testing.assert_index_equal(actual.index, transported[0].index)
            np.testing.assert_allclose(
                actual.to_numpy(), expected.to_numpy(), rtol=0, atol=0, equal_nan=True
            )

    def test_empty_auxiliary_calendars_mean_unavailable_inputs(self):
        daily, cross, iv = fixture(140)
        actual = market.build_market_features(daily, cross.iloc[:0], iv.iloc[:0])
        self.assertTrue(actual[["liv", "lvix", "term", "xasset_stress"]].isna().all().all())
        self.assertTrue(np.isfinite(actual.rv_total.iloc[-1]))

    def test_malformed_calendars_are_rejected_for_all_three_sources(self):
        for source in range(3):
            for mode in ("duplicate", "unsorted", "timezone", "intraday", "nat", "strings"):
                frames = list(fixture(5))
                frame = frames[source]
                if mode == "duplicate":
                    frame.index = frame.index[:1].append(frame.index[:-1])
                elif mode == "unsorted":
                    frames[source] = frame.iloc[::-1]
                elif mode == "timezone":
                    frame.index = frame.index.tz_localize("UTC")
                elif mode == "intraday":
                    frame.index = frame.index + pd.Timedelta(hours=1)
                elif mode == "nat":
                    frame.index = pd.DatetimeIndex([pd.NaT, *frame.index[1:]])
                else:
                    frame.index = frame.index.strftime("%Y-%m-%d")
                with self.subTest(source=source, mode=mode), self.assertRaises(ValueError):
                    market.build_market_features(*frames)

    def test_ceiling_on_every_calendar_is_checked_before_logarithms(self):
        for source in range(3):
            frames = list(fixture(3))
            frame = frames[source]
            frame.index = pd.DatetimeIndex(["2025-10-17", "2025-10-20", "2025-10-21"])
            with (
                self.subTest(source=source),
                patch.object(
                    np,
                    "log",
                    side_effect=AssertionError("Arithmetic before ceiling rejection"),
                ),
                self.assertRaises(ValueError),
            ):
                market.build_market_features(*frames)

    def test_nonpositive_prices_iv_and_infinity_negative_volume_are_invalid(self):
        for source, column in (
            (0, "open"),
            (0, "high"),
            (0, "low"),
            (0, "close"),
            (0, "adj close"),
            (1, "hyg"),
            (2, "vxn"),
            (2, "vix"),
            (2, "vix9d"),
        ):
            for value in (0, -1, np.inf, -np.inf):
                frames = list(fixture(3))
                frames[source].loc[frames[source].index[1], column] = value
                with (
                    self.subTest(source=source, column=column, value=value),
                    self.assertRaises(ValueError),
                ):
                    market.build_market_features(*frames)
        for value in (-1, np.inf, -np.inf):
            frames = list(fixture(3))
            frames[0].loc[frames[0].index[1], "volume"] = value
            with self.subTest(volume=value), self.assertRaises(ValueError):
                market.build_market_features(*frames)

    def test_impossible_ohlc_geometry_rejected_even_with_another_missing_price(self):
        for column, value in (("high", 1), ("low", 1000)):
            daily, cross, iv = fixture(3)
            daily.loc[daily.index[1], column] = value
            daily.loc[daily.index[1], "open"] = np.nan
            with self.subTest(column=column), self.assertRaises(ValueError):
                market.build_market_features(daily, cross, iv)

    def test_exact_required_columns_real_numeric_types_and_nonempty_daily(self):
        for source in range(3):
            for mode in ("missing", "extra", "duplicate", "text", "bool", "complex"):
                frames = list(fixture(3))
                frame = frames[source]
                if mode == "missing":
                    frames[source] = frame.iloc[:, 1:]
                elif mode == "extra":
                    frame["unused"] = 1.0
                elif mode == "duplicate":
                    frame.columns = [frame.columns[0]] * len(frame.columns)
                else:
                    frame[frame.columns[0]] = {"text": "1", "bool": True, "complex": 1 + 2j}[
                        mode
                    ]
                with self.subTest(source=source, mode=mode), self.assertRaises(ValueError):
                    market.build_market_features(*frames)
        daily, cross, iv = fixture(3)
        with self.assertRaises(ValueError):
            market.build_market_features(daily.iloc[:0], cross, iv)

    def test_inputs_are_not_mutated_or_aliased(self):
        frames = fixture(140)
        before = copy.deepcopy(frames)
        actual = market.build_market_features(*frames)
        actual.iloc[-1, :] = 0
        for original, saved in zip(frames, before):
            pd.testing.assert_frame_equal(original, saved, check_exact=True)


class FiveSessionTargetsTests(unittest.TestCase):
    def series(self):
        index = pd.DatetimeIndex(
            [
                "2020-01-02",
                "2020-01-03",
                "2020-01-07",
                "2020-01-08",
                "2020-01-10",
                "2020-01-13",
                "2020-01-14",
                "2020-01-16",
            ],
            name="date",
        )
        return pd.Series(np.arange(1, 9, dtype=float), index=index, name="rv_total")

    def test_exact_next_five_full_sessions_and_end_dates(self):
        rv = self.series()
        actual = market.make_five_session_targets(rv)
        self.assertEqual(tuple(actual.columns), ("y", "target_end"))
        pd.testing.assert_index_equal(actual.index, rv.index)
        np.testing.assert_allclose(
            actual.y, [4, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan], equal_nan=True
        )
        self.assertEqual(actual.target_end.iloc[0], pd.Timestamp("2020-01-13"))
        self.assertEqual(actual.target_end.iloc[2], pd.Timestamp("2020-01-16"))
        self.assertTrue(actual.target_end.iloc[3:].isna().all())

    def test_missing_forward_value_invalidates_mean_without_changing_endpoint(self):
        rv = self.series()
        rv.iloc[3] = np.nan
        actual = market.make_five_session_targets(rv)
        self.assertTrue(actual.y.iloc[:3].isna().all())
        self.assertEqual(actual.target_end.iloc[0], rv.index[5])
        self.assertEqual(actual.target_end.iloc[2], rv.index[7])
        pd.testing.assert_index_equal(actual.index, rv.index)

    def test_target_does_not_use_current_value_or_values_after_its_endpoint(self):
        rv = self.series()
        expected = market.make_five_session_targets(rv)
        rv.iloc[0] = np.nan
        rv.iloc[6:] = 1000
        actual = market.make_five_session_targets(rv)
        self.assertEqual(actual.y.iloc[0], expected.y.iloc[0])
        self.assertEqual(actual.target_end.iloc[0], expected.target_end.iloc[0])

    def test_nonpositive_or_infinite_observed_rv_is_rejected(self):
        for value in (0, -1, np.inf, -np.inf):
            rv = self.series()
            rv.iloc[4] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                market.make_five_session_targets(rv)

    def test_target_calendar_validation_and_ceiling_precede_arithmetic(self):
        for mode in (
            "duplicate",
            "unsorted",
            "timezone",
            "intraday",
            "nat",
            "strings",
            "ceiling",
        ):
            rv = self.series()
            if mode == "duplicate":
                rv.index = rv.index[:1].append(rv.index[:-1])
            elif mode == "unsorted":
                rv = rv.iloc[::-1]
            elif mode == "timezone":
                rv.index = rv.index.tz_localize("UTC")
            elif mode == "intraday":
                rv.index = rv.index + pd.Timedelta(minutes=1)
            elif mode == "nat":
                rv.index = pd.DatetimeIndex([pd.NaT, *rv.index[1:]])
            elif mode == "strings":
                rv.index = rv.index.strftime("%Y-%m-%d")
            else:
                rv.index = pd.bdate_range("2025-10-20", periods=len(rv))
            with (
                self.subTest(mode=mode),
                patch.object(
                    pd.Series,
                    "shift",
                    side_effect=AssertionError("Shift before calendar validation"),
                ),
                self.assertRaises(ValueError),
            ):
                market.make_five_session_targets(rv)

    def test_target_input_types_and_short_history(self):
        rv = self.series()
        for bad in (
            rv.to_frame(),
            rv.astype(str),
            rv.astype(bool),
            rv.astype(complex),
            rv.iloc[:0],
        ):
            with (
                self.subTest(kind=type(bad), dtype=getattr(bad, "dtype", None)),
                self.assertRaises(ValueError),
            ):
                market.make_five_session_targets(bad)
        short = market.make_five_session_targets(rv.iloc[:4])
        self.assertEqual(len(short), 4)
        self.assertTrue(short.y.isna().all())
        self.assertTrue(short.target_end.isna().all())

    def test_target_transport_units_and_no_input_mutation(self):
        for unit in ("ms", "us", "ns"):
            rv = self.series().astype("Float64")
            rv.index = rv.index.as_unit(unit)
            before = rv.copy(deep=True)
            actual = market.make_five_session_targets(rv)
            pd.testing.assert_index_equal(actual.index, rv.index)
            self.assertEqual(actual.target_end.iloc[0], rv.index[5])
            actual.iloc[0, 0] = 999
            pd.testing.assert_series_equal(rv, before, check_exact=True)


if __name__ == "__main__":
    unittest.main()
