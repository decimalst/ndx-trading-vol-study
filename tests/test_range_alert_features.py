"""Prewritten generated contracts for range-location risk-alert inputs.

No historical source values, event counts, relationships or fits are read.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import range_alert_features as features
from src import tail_shape_features as tail
from tests.test_tail_shape_features import design, sample, sources_fixture


def raw_design(n=300, seed=5):
    frame = design(n, seed)
    rng = np.random.default_rng(seed + 40)
    frame["intraday"] = rng.normal(0, 0.02, n)
    frame["intraday_sq"] = frame.intraday**2
    frame["overnight"] = rng.normal(0, 0.01, n)
    frame["overnight_sq"] = frame.overnight**2
    frame["range_extremity"] = rng.uniform(0.01, 0.95, n)
    return frame


class RangeAlertFeatures(unittest.TestCase):
    def test_exact_raw_baseline_and_output_columns(self):
        nuisance = ("intraday", "intraday_sq", "overnight", "overnight_sq")
        self.assertEqual(features.RAW, tail.RAW + nuisance + ("range_extremity",))
        self.assertEqual(features.BASE, tail.BASE + nuisance)
        self.assertEqual(features.ALL_FEATURES, features.RAW)
        self.assertEqual(features.MODELS, ("baseline", "recent_frequency", "location"))
        f, t = features.build_features(*sample())
        self.assertEqual(tuple(f.columns), features.RAW + ("feature_cutoff_date",))
        self.assertEqual(tuple(t.columns), ("y", "target_end", "available_date"))
        self.assertEqual((len(features.RAW), len(features.BASE)), (24, 26))

    def test_original_raw_controls_are_unchanged(self):
        daily, iv = sample()
        old, _ = tail.build_features(daily, iv)
        new, _ = features.build_features(daily, iv)
        pd.testing.assert_frame_equal(new.loc[:, tail.RAW], old.loc[:, tail.RAW])
        pd.testing.assert_series_equal(new.feature_cutoff_date, old.feature_cutoff_date)

    def test_nuisances_location_and_risk_match_independent_bar_arithmetic(self):
        daily, iv = sample()
        f, targets = features.build_features(daily, iv)
        t = 90
        bar, previous = daily.iloc[t - 1], daily.close.iloc[t - 2]
        intraday = np.log(bar.close / bar.open)
        overnight = np.log(bar.open / previous)
        z = (np.log(bar.close / bar.low) - np.log(bar.high / bar.close)) / np.log(
            bar.high / bar.low
        )
        for name, expected in {
            "intraday": intraday,
            "intraday_sq": intraday**2,
            "overnight": overnight,
            "overnight_sq": overnight**2,
            "range_extremity": z**2,
        }.items():
            self.assertAlmostEqual(f.iloc[t][name], expected, places=14)
        day = np.log(daily.close / daily.open)
        gk = np.maximum(
            0.5 * np.log(daily.high / daily.low) ** 2 - (2 * np.log(2) - 1) * day**2,
            1e-10,
        )
        variance = gk + np.log(daily.open / daily.close.shift()) ** 2
        threshold = 2 * math.fsum(variance.iloc[t - 22 : t]) / 22
        self.assertEqual(targets.iloc[t].y, float(variance.iloc[t + 1] > threshold))
        self.assertEqual(f.iloc[t].feature_cutoff_date, daily.index[t - 1])

    def test_same_gk_and_returns_can_have_different_location(self):
        dates = pd.bdate_range("2014-01-02", periods=70, name="date")

        def bars(a):
            return pd.DataFrame(
                {
                    "open": 100.0,
                    "close": 100.0,
                    "high": 100 * np.exp(a),
                    "low": 100 * np.exp(a - 0.04),
                },
                index=dates,
            )

        center = features.session_components(bars(0.02))
        edge = features.session_components(bars(0.005))
        np.testing.assert_allclose(center.variance, edge.variance, equal_nan=True, atol=1e-17)
        np.testing.assert_array_equal(center.intraday, edge.intraday)
        np.testing.assert_array_equal(center.overnight.iloc[1:], edge.overnight.iloc[1:])
        self.assertLess(center.range_extremity.iloc[30], 1e-25)
        self.assertAlmostEqual(edge.range_extremity.iloc[30], 0.5625, places=12)

    def test_target_ties_strict_prior22_window_and_maturity(self):
        dates = pd.bdate_range("2014-01-02", periods=80, name="date").delete([31])
        variance = pd.Series(1.0, index=dates)
        variance.iloc[40] = 1000.0  # Entry bar must not enter the reference at entry40.
        variance.iloc[41] = 2.0
        target = features.build_targets(variance)
        self.assertEqual(target.iloc[40].y, 0.0)
        variance.iloc[41] = np.nextafter(2.0, np.inf)
        self.assertEqual(features.build_targets(variance).iloc[40].y, 1.0)
        self.assertEqual(target.iloc[40].target_end, dates[41])
        self.assertEqual(target.iloc[40].available_date, dates[41])
        self.assertTrue(target.y.iloc[:22].isna().all())
        self.assertTrue(target.iloc[-1].isna().all())

    def test_checked_compensated_reference_is_authoritative(self):
        dates = pd.bdate_range("2014-01-02", periods=50, name="date")
        values = np.array([1e-10] * 21 + [1.0])
        variance = pd.Series(1.0, index=dates)
        variance.iloc[:22] = values
        threshold = 2 * (math.fsum(values) / 22)
        variance.iloc[23] = threshold
        self.assertEqual(features.build_targets(variance).iloc[22].y, 0.0)
        variance.iloc[23] = np.nextafter(threshold, np.inf)
        self.assertEqual(features.build_targets(variance).iloc[22].y, 1.0)

    def test_label_known_even_when_zero_range_or_iv_makes_predictors_unknown(self):
        daily, iv = sample()
        daily.loc[:, ["open", "high", "low", "close"]] = 100.0
        iv.loc[:, :] = np.nan
        f, targets = features.build_features(daily, iv)
        self.assertTrue(f.range_extremity.isna().all())
        self.assertTrue(f.I.isna().all())
        self.assertTrue(targets.y.iloc[23:-1].eq(0).all())
        self.assertEqual(len(targets), len(daily))
        self.assertTrue(targets.index.equals(daily.index))

    def test_missing_bar_retains_calendar_and_invalidates_full_required_windows(self):
        daily, iv = sample()
        daily.loc[daily.index[80], ["open", "high", "low", "close"]] = np.nan
        f, target = features.build_features(daily, iv)
        self.assertEqual(len(f), len(daily))
        self.assertTrue(np.isnan(f.range_extremity.iloc[81]))
        self.assertTrue(np.isfinite(f.range_extremity.iloc[82]))
        self.assertTrue(target.y.iloc[79:104].isna().all())
        self.assertTrue(np.isfinite(target.y.iloc[104]))
        self.assertEqual(target.iloc[79].target_end, daily.index[80])

    def test_location_requires_complete_bar_but_primitive_checks_do_not_hide_behind_missingness(
        self,
    ):
        daily, _ = sample()
        daily.loc[daily.index[40], "open"] = np.nan
        components = features.session_components(daily)
        self.assertTrue(np.isnan(components.range_extremity.iloc[40]))
        bad = daily.copy()
        bad.loc[bad.index[40], ["low", "high", "close"]] = [1e-300, 1e300, 1.0]
        with self.assertRaises(ValueError):
            features.session_components(bad)

    def test_entry_and_future_values_cannot_change_current_predictors(self):
        daily, iv = sample()
        before, _ = features.build_features(daily, iv)
        changed, changed_iv = daily.copy(), iv.copy()
        changed.loc[changed.index[90:], ["open", "high", "low", "close"]] *= 3
        changed_iv.iloc[90:] *= 2
        after, _ = features.build_features(changed, changed_iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])

    def test_future_observed_calendar_is_only_used_for_target_end(self):
        daily, iv = sample()
        before, target = features.build_features(daily, iv)
        after, other = features.build_features(daily.drop(daily.index[91]), iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])
        self.assertEqual(target.iloc[90].target_end, daily.index[91])
        self.assertEqual(other.iloc[90].target_end, daily.index[92])

    def test_positive_price_units_and_adjusted_fields_do_not_change_evidence(self):
        daily, iv = sample()
        before, target = features.build_features(daily, iv)
        changed = daily.copy()
        changed.loc[:, ["open", "high", "low", "close"]] *= 7
        changed["adj close"] = np.nan
        after, other = features.build_features(changed, iv)
        np.testing.assert_allclose(
            before.loc[:, features.RAW],
            after.loc[:, features.RAW],
            atol=3e-12,
            rtol=1e-11,
            equal_nan=True,
        )
        pd.testing.assert_frame_equal(target, other)

    def test_valid_close_at_range_end_has_extremity_one_without_clipping(self):
        daily, _ = sample()
        daily["close"] = daily.high
        up = features.session_components(daily)
        np.testing.assert_array_equal(up.range_extremity, np.ones(len(daily)))
        daily["close"] = daily.low
        down = features.session_components(daily)
        np.testing.assert_array_equal(down.range_extremity, np.ones(len(daily)))

    def test_invalid_observed_ohlc_pairs_and_values_reject_even_on_partial_bars(self):
        daily, iv = sample()
        for value in [0.0, -1.0, np.inf, -np.inf]:
            bad = daily.copy()
            bad.loc[bad.index[40], "low"] = value
            with self.assertRaises(ValueError):
                features.build_features(bad, iv)
        bad = daily.copy()
        bad.loc[bad.index[40], "open"] = np.nan
        bad.loc[bad.index[40], "high"] = 0.1
        with self.assertRaises(ValueError):
            features.build_features(bad, iv)
        with self.assertRaises(ValueError):
            features.build_features(daily.iloc[::-1], iv)

    def test_computed_log_ratio_and_implied_variance_overflow_are_not_missing(self):
        daily, iv = sample()
        bad = daily.copy()
        bad.loc[bad.index[40], ["open", "close", "high", "low"]] = [
            1e-300,
            1e300,
            1e300,
            1e-300,
        ]
        with self.assertRaises(ValueError):
            features.build_features(bad, iv)
        bad_iv = iv.copy()
        bad_iv.loc[bad_iv.index[40], "vix"] = 1e300
        with self.assertRaises(ValueError):
            features.build_features(daily, bad_iv)
        bad_iv.loc[bad_iv.index[40], "vix"] = np.nextafter(0.0, 1.0)
        with self.assertRaises(ValueError):
            features.build_features(daily, bad_iv)

    def test_invalid_variance_inputs_or_reference_overflow_reject(self):
        dates = pd.bdate_range("2014-01-02", periods=60, name="date")
        for bad in [0.0, -1.0, np.inf, -np.inf, 1e308]:
            variance = pd.Series(1.0, index=dates)
            variance.iloc[:30] = bad
            with self.assertRaises(ValueError):
                features.build_targets(variance)

    def test_curvature_centers_and_all25_population_scales_are_training_only(self):
        train, apply = raw_design(), raw_design(40, 8)
        x, a, e, ae, audit = features.transform(train, apply)
        expected, query = train.copy(), apply.copy()
        for name in ["I", "R", "skew"]:
            expected[name + "_square"] = (train[name] - train[name].mean()) ** 2
            query[name + "_square"] = (apply[name] - train[name].mean()) ** 2
        means = expected.loc[:, features.BASE].mean()
        scales = expected.loc[:, features.BASE].std(ddof=0)
        means["const"], scales["const"] = 0.0, 1.0
        pd.testing.assert_frame_equal(x, (expected.loc[:, features.BASE] - means) / scales)
        pd.testing.assert_frame_equal(a, (query.loc[:, features.BASE] - means) / scales)
        np.testing.assert_allclose(e, train.range_extremity - train.range_extremity.mean())
        np.testing.assert_allclose(ae, apply.range_extremity - train.range_extremity.mean())
        self.assertEqual(audit["columns"], list(features.BASE))
        self.assertEqual(audit["means"], means.tolist())
        self.assertEqual(audit["scales"], scales.tolist())
        self.assertEqual(
            audit["curvature_means"], {n: train[n].mean() for n in ["I", "R", "skew"]}
        )

    def test_query_mutation_cannot_change_centers_scales_or_training_transform(self):
        train, apply = raw_design(), raw_design(40, 8)
        original = train.copy()
        x, _, e, _, audit = features.transform(train, apply)
        changed = apply.copy()
        changed["I"] += 50
        changed["intraday"] *= 4
        changed["range_extremity"] = 1 - changed.range_extremity
        xx, _, ee, _, other = features.transform(train, changed)
        pd.testing.assert_frame_equal(x, xx)
        pd.testing.assert_series_equal(e, ee)
        self.assertEqual(audit, other)
        pd.testing.assert_frame_equal(train, original)

    def test_constant_extremity_has_exact_zero_centered_history_without_fold_drop(self):
        train, apply = raw_design(), raw_design(40, 8)
        train["range_extremity"] = 0.1
        x, a, e, ae, audit = features.transform(train, apply)
        self.assertEqual(len(x), len(train))
        self.assertEqual(len(a), len(apply))
        np.testing.assert_array_equal(e, np.zeros(len(train)))
        np.testing.assert_allclose(ae, apply.range_extremity - 0.1)
        self.assertTrue(audit["range_extremity_constant"])
        self.assertEqual(audit["range_extremity_mean"], 0.1)
        self.assertEqual(audit["range_extremity_scale"], 1.0)

    def test_collinear_baseline_columns_are_retained_but_zero_population_scale_aborts(self):
        train, apply = raw_design(), raw_design(40, 8)
        train["intraday"], apply["intraday"] = train.ret_d, apply.ret_d
        x, a, _, _, _ = features.transform(train, apply)
        self.assertEqual(tuple(x.columns), features.BASE)
        np.testing.assert_array_equal(x.ret_d, x.intraday)
        np.testing.assert_array_equal(a.ret_d, a.intraday)
        train["intraday_sq"] = 0.0
        with self.assertRaisesRegex(ValueError, "scale"):
            features.transform(train, apply)
        train = raw_design()
        tr = train.intraday - train.intraday.mean()
        train["intraday"] = tr / tr.std(ddof=0) * 0.9995e-12
        self.assertGreater(train.intraday.std(ddof=1), 1e-12)
        with self.assertRaisesRegex(ValueError, "scale"):
            features.transform(train, apply)

    def test_all_raw_features_finite_real_literal_intercept_and_bounded_e_required(self):
        train, apply = raw_design(), raw_design(40, 8)
        for name, bad in [
            ("const", 0.0),
            ("intraday", np.nan),
            ("range_extremity", -1e-16),
            ("range_extremity", np.nextafter(1.0, np.inf)),
        ]:
            altered = train.copy()
            altered.loc[altered.index[10], name] = bad
            with self.assertRaises(ValueError):
                features.transform(altered, apply)
        for altered in [train.astype(complex), train.astype(str), train.astype(object)]:
            with self.assertRaises(ValueError):
                features.transform(altered, apply)

    def test_staged_source_loader_reuses_exact_admission_and_bounds_before_numeric_parse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol, digest = sources_fixture(root)
            with patch.object(tail, "SKEW_SHA256", digest):
                daily, iv, audit = features.load_sources(protocol, root)
            self.assertLessEqual(daily.index.max(), pd.Timestamp("2025-10-20"))
            self.assertLessEqual(iv.index.max(), pd.Timestamp("2025-10-20"))
            self.assertFalse(audit["numeric_post_cutoff_values_parsed"])
            self.assertEqual(audit["raw_columns"], list(features.RAW))
            self.assertEqual(audit["baseline_columns"], list(features.BASE))
            self.assertNotIn("u<-1.5", audit["target_interpretation"])
            self.assertNotIn("gap_interpretation", audit)
            self.assertTrue(audit["skew_provenance"]["raw_derived_bounded_exact_equal"])
            (root / protocol["sources"]["skew"]).write_bytes(b"tampered")
            with patch.object(tail, "_read_skew_csv") as parser:
                with self.assertRaises(ValueError):
                    features.load_sources(protocol, root)
                parser.assert_not_called()


if __name__ == "__main__":
    unittest.main()
