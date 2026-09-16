"""Wave 7 source/feature contracts, written before the producer exists."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import tail_shape_features as features


def sample(n=180):
    dates = pd.bdate_range("2014-01-02", periods=n, name="date").delete([12, 37])
    z = np.arange(len(dates), dtype=float)
    close = 100*np.exp(np.cumsum(.001+.008*np.sin(z*.37)))
    opening = close*np.exp(.004*np.cos(z*.23))
    daily = pd.DataFrame({"open": opening, "high": np.maximum(opening, close)*1.006,
                          "low": np.minimum(opening, close)*.994, "close": close,
                          "adj close": close*.7}, index=dates)
    iv = pd.DataFrame({"vix": 18+3*np.sin(z*.17), "vix9d": 16+2*np.cos(z*.21),
                       "vvix": 85+4*np.cos(z*.19), "skew": 120+5*np.sin(z*.13)}, index=dates)
    return daily, iv


def design(n=300, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n, name="date")
    frame = pd.DataFrame(rng.normal(size=(n, len(features.RAW))), index=dates, columns=features.RAW)
    frame["const"] = 1.
    frame["I"] -= 3
    frame["R"] -= 4
    frame["skew"] = 120+10*frame["skew"]
    for weekday in range(1, 5):
        frame[f"entry_dow_{weekday}"] = (dates.weekday == weekday).astype(float)
    return frame


def sources_fixture(root, raw_value="159.03", derived_value=159.03):
    dates = pd.DatetimeIndex(["2018-08-10", "2018-08-13", "2025-10-20", "2025-11-03"], name="date")
    daily = pd.DataFrame({"open": [100., 101., 102., np.inf], "high": [102., 103., 104., np.inf],
                          "low": [99., 100., 101., np.inf], "close": [101., 102., 103., np.inf],
                          "adj close": [1., 2., 3., 4.]}, index=dates)
    daily.to_parquet(root/"daily.parquet")
    paths = {"daily": "daily.parquet", "skew": "SKEW.csv", "skew_source": "skew_source.json",
             "skew_derived": "skew.parquet"}
    for name, column in [("vix", "CLOSE"), ("vix9d", "CLOSE"), ("vvix", "VVIX")]:
        paths[name] = name+".csv"
        (root/paths[name]).write_text(f"DATE,{column}\n08/10/2018,20\n08/13/2018,21\n10/20/2025,22\n11/03/2025,FORBIDDEN\n")
    raw = f"DATE,SKEW\n08/10/2018,130\n08/13/2018,{raw_value}\n10/20/2025,140\n11/03/2025,FORBIDDEN\n".encode()
    (root/paths["skew"]).write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    metadata = {"url": "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
                "fetched_at_utc": "2026-08-10T12:00:00+00:00", "sha256": digest, "rows": 4,
                "first_date": "2018-08-10", "last_date": "2025-11-03",
                "anchor_date": "2018-08-13", "anchor_close": 159.03}
    (root/paths["skew_source"]).write_text(json.dumps(metadata))
    pd.DataFrame({"close": [130., derived_value, 140., np.inf]}, index=dates).to_parquet(root/paths["skew_derived"])
    protocol = {"sources": paths, "index": {"source_end": "2025-10-20", "sealed_start": "2025-11-03"}}
    return protocol, digest


class TestTailShapeFeatures(unittest.TestCase):
    def test_fixed_feature_model_target_schema(self):
        raw = ("const", "I", "R", "ret_d", "ret_w", "ret_m", "ret_q", "lr_d", "lr_w",
               "term", "lvvix", "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4",
               "neg_d", "neg_w", "neg_m", "skew")
        self.assertEqual(features.RAW, raw)
        self.assertEqual(features.BASE, raw+("I_square", "R_square", "skew_square"))
        self.assertEqual(features.ALL_FEATURES, features.BASE)
        self.assertEqual(features.MODELS, ("frequency", "constant_shape", "skew_shape"))
        f, target = features.build_features(*sample())
        self.assertEqual(tuple(f.columns), raw+("feature_cutoff_date", "normalization_mean", "normalization_scale"))
        self.assertEqual(tuple(target.columns), ("y", "raw_return", "event", "target_end", "available_date"))

    def test_market_controls_and_negative_histories_match_explicit_past_windows(self):
        daily, iv = sample()
        f, _ = features.build_features(daily, iv)
        ret = np.log(daily.close/daily.close.shift())
        negative = np.maximum(-ret, 0)
        t = 90
        for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
            self.assertAlmostEqual(f.iloc[t]["neg_"+suffix], negative.iloc[t-width:t].mean())
        for suffix, width in [("d", 1), ("w", 5), ("m", 22), ("q", 63)]:
            self.assertAlmostEqual(f.iloc[t]["ret_"+suffix], ret.iloc[t-width:t].mean())
        self.assertAlmostEqual(f.iloc[t].I, np.log((iv.vix.iloc[t-1]/100)**2))
        self.assertAlmostEqual(f.iloc[t].term, np.log(iv.vix9d.iloc[t-1]/iv.vix.iloc[t-1]))
        self.assertAlmostEqual(f.iloc[t].lvvix, np.log(iv.vvix.iloc[t-1]))
        self.assertEqual(f.iloc[t]["skew"], iv["skew"].iloc[t-1])
        self.assertEqual(f.iloc[t].feature_cutoff_date, daily.index[t-1])

    def test_strict_normalization_uses_daily_variance_without_annualization(self):
        daily, iv = sample()
        daily.loc[daily.index[91:], ["open", "high", "low", "close"]] *= .8
        f, target = features.build_features(daily, iv)
        ret = np.log(daily.close/daily.close.shift())
        gk = np.maximum(.5*np.log(daily.high/daily.low)**2
                        -(2*np.log(2)-1)*np.log(daily.close/daily.open)**2, 1e-10)
        variance = gk+np.log(daily.open/daily.close.shift())**2
        t = 90
        mean = ret.iloc[t-22:t].mean()
        scale = np.sqrt(variance.iloc[t-22:t].mean())
        self.assertAlmostEqual(f.iloc[t].normalization_mean, mean)
        self.assertAlmostEqual(f.iloc[t].normalization_scale, scale)
        self.assertAlmostEqual(f.iloc[t].R, np.log(252*scale**2))
        for name, width in [("lr_d", 1), ("lr_w", 5)]:
            self.assertAlmostEqual(f.iloc[t][name], np.log(252*variance.iloc[t-width:t].mean()))
        future = np.log(daily.close.iloc[t+1]/daily.close.iloc[t])
        self.assertAlmostEqual(target.iloc[t].raw_return, future)
        self.assertAlmostEqual(target.iloc[t].y, (future-mean)/scale)
        self.assertEqual(target.iloc[t].event, float((future-mean)/scale < -1.5))
        self.assertEqual(target.iloc[t].event, 1.)

    def test_next_observed_close_is_target_and_availability_with_missing_last_label(self):
        daily, iv = sample()
        _, target = features.build_features(daily, iv)
        self.assertEqual(target.iloc[40].target_end, daily.index[41])
        self.assertEqual(target.iloc[40].available_date, daily.index[41])
        self.assertTrue(target.iloc[-1].isna().all())
        self.assertTrue(target.index.equals(daily.index))

    def test_zero_return_is_valid_zero_event_and_floor_preserves_positive_scale(self):
        daily, iv = sample()
        daily.loc[:, ["open", "high", "low", "close"]] = 100.
        f, target = features.build_features(daily, iv)
        self.assertAlmostEqual(f.iloc[90].normalization_scale, 1e-5)
        self.assertEqual(target.iloc[90].raw_return, 0.)
        self.assertEqual(target.iloc[90].y, 0.)
        self.assertEqual(target.iloc[90].event, 0.)

    def test_unknown_normalizer_makes_event_unknown_not_zero(self):
        daily, iv = sample()
        f, target = features.build_features(daily, iv)
        self.assertTrue(f.normalization_scale.iloc[:23].isna().all())
        self.assertTrue(target.event.iloc[:23].isna().all())
        self.assertTrue(np.isfinite(target.raw_return.iloc[10]))

    def test_entry_and_future_mutations_cannot_change_predictors_or_normalizers(self):
        daily, iv = sample()
        before, _ = features.build_features(daily, iv)
        changed, changed_iv = daily.copy(), iv.copy()
        changed.iloc[90:] *= 3
        changed_iv.iloc[90:] *= 2
        after, _ = features.build_features(changed, changed_iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])

    def test_future_actual_calendar_changes_only_target_not_current_features(self):
        daily, iv = sample()
        before, target_before = features.build_features(daily, iv)
        after, target_after = features.build_features(daily.drop(daily.index[91]), iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])
        self.assertNotEqual(target_before.iloc[90].target_end, target_after.iloc[90].target_end)

    def test_raw_price_units_and_adjusted_close_do_not_change_features_or_target(self):
        daily, iv = sample()
        before, target_before = features.build_features(daily, iv)
        changed = daily.copy()
        changed.loc[:, ["open", "high", "low", "close"]] *= 7
        changed["adj close"] = np.nan
        after, target_after = features.build_features(changed, iv)
        for columns in [features.RAW, ("normalization_mean", "normalization_scale")]:
            np.testing.assert_allclose(before.loc[:, columns], after.loc[:, columns], atol=2e-13, equal_nan=True)
        np.testing.assert_allclose(target_before[["y", "raw_return", "event"]],
                                   target_after[["y", "raw_return", "event"]], atol=2e-12, equal_nan=True)

    def test_missing_prices_propagate_through_negative_and_normalization_windows(self):
        daily, iv = sample()
        daily.loc[daily.index[80], ["open", "high", "low", "close"]] = np.nan
        f, target = features.build_features(daily, iv)
        self.assertEqual(len(f), len(daily))
        self.assertTrue(f.neg_w.iloc[81:87].isna().all())
        self.assertTrue(np.isfinite(f.iloc[87].neg_w))
        self.assertTrue(f.normalization_mean.iloc[81:104].isna().all())
        self.assertTrue(f.normalization_scale.iloc[81:104].isna().all())
        self.assertTrue(target.event.iloc[81:104].isna().all())
        self.assertTrue(np.isfinite(f.iloc[104].normalization_scale))
        self.assertTrue(np.isnan(target.iloc[79].raw_return))

    def test_missing_skew_is_not_filled_or_calendar_compressed(self):
        daily, iv = sample()
        iv.loc[daily.index[80], "skew"] = np.nan
        f, _ = features.build_features(daily, iv)
        self.assertTrue(np.isnan(f.iloc[81]["skew"]))
        self.assertTrue(np.isfinite(f.iloc[82]["skew"]))
        self.assertEqual(f.iloc[81].feature_cutoff_date, daily.index[80])

    def test_invalid_prices_or_calendars_are_rejected(self):
        daily, iv = sample()
        for bad in [0., -1., np.inf]:
            changed = iv.copy()
            changed.loc[changed.index[40], "skew"] = bad
            with self.assertRaises(ValueError):
                features.build_features(daily, changed)
        with self.assertRaises(ValueError):
            features.build_features(daily.iloc[::-1], iv)
        changed = daily.copy()
        changed.loc[changed.index[40], "high"] = .1
        with self.assertRaises(ValueError):
            features.build_features(changed, iv)

    def test_training_centers_and_population_scales_use_exact_admitted_rows(self):
        train, apply = design(), design(30, 8)
        tr, ap, audit = features.transform(train, apply)
        self.assertEqual(tuple(tr.columns), features.BASE)
        self.assertEqual(tuple(ap.columns), features.BASE)
        self.assertEqual(audit, {"train_n": len(train), "I_mean": train.I.mean(),
                                 "R_mean": train.R.mean(), "skew_mean": train["skew"].mean()})
        for name in ["I", "R", "skew"]:
            np.testing.assert_allclose(tr[name+"_square"], (train[name]-train[name].mean())**2)
            np.testing.assert_allclose(ap[name+"_square"], (apply[name]-train[name].mean())**2)
        self.assertTrue((tr.loc[:, features.BASE[1:]].std(ddof=0) > 1e-12).all())

    def test_query_values_do_not_change_transform_centers_or_mutate_inputs(self):
        train, apply = design(), design(30, 8)
        original_train, original_apply = train.copy(), apply.copy()
        tr, _, audit = features.transform(train, apply)
        changed = apply.copy()
        changed["skew"] += 100
        other, _, other_audit = features.transform(train, changed)
        pd.testing.assert_frame_equal(tr, other)
        self.assertEqual(audit, other_audit)
        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(apply, original_apply)

    def test_all_common_slopes_including_derived_square_require_population_scale(self):
        train, apply = design(), design(30, 8)
        train["neg_w"] = 0.
        with self.assertRaisesRegex(ValueError, "scale"):
            features.transform(train, apply)
        train = design()
        train["skew"] = np.tile([110., 130.], len(train)//2)
        with self.assertRaisesRegex(ValueError, "scale"):
            features.transform(train, apply)
        train = design()
        centered = train.neg_w-train.neg_w.mean()
        train["neg_w"] = centered/centered.std(ddof=0)*(.9995e-12)
        self.assertGreater(train.neg_w.std(ddof=1), 1e-12)
        with self.assertRaisesRegex(ValueError, "scale"):
            features.transform(train, apply)

    def test_nonfinite_design_and_nonunit_intercept_do_not_trigger_row_selection(self):
        train, apply = design(), design(30, 8)
        for column in ["skew", "neg_m", "I"]:
            changed = train.copy()
            changed.loc[changed.index[10], column] = np.nan
            with self.assertRaises(ValueError):
                features.transform(changed, apply)
        changed = apply.copy()
        changed["const"] = 0.
        with self.assertRaises(ValueError):
            features.transform(train, changed)

    def test_source_hash_mismatch_aborts_before_skew_numeric_parser(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol, _ = sources_fixture(root)
            with patch.object(features, "_read_skew_csv") as parser:
                with self.assertRaisesRegex(ValueError, "hash"):
                    features.load_sources(protocol, root)
                parser.assert_not_called()

    def test_source_manifest_mismatch_is_rejected_even_when_raw_pin_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol, digest = sources_fixture(root)
            path = root/protocol["sources"]["skew_source"]
            metadata = json.loads(path.read_text())
            metadata["sha256"] = "0"*64
            path.write_text(json.dumps(metadata))
            with patch.object(features, "SKEW_SHA256", digest), self.assertRaisesRegex(ValueError, "hash"):
                features.load_sources(protocol, root)

    def test_bounded_source_parser_preserves_raw_values_and_checks_derived_equality(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol, digest = sources_fixture(root)
            with patch.object(features, "SKEW_SHA256", digest):
                daily, iv, audit = features.load_sources(protocol, root)
            self.assertEqual(tuple(daily.columns), ("open", "high", "low", "close"))
            self.assertEqual(iv["skew"].dropna().tolist(), [130., 159.03, 140.])
            self.assertLessEqual(iv.index.max(), pd.Timestamp("2025-10-20"))
            self.assertFalse(audit["numeric_post_cutoff_values_parsed"])
            self.assertTrue(audit["skew_provenance"]["raw_matches_manifest"])
            self.assertTrue(audit["skew_provenance"]["raw_derived_bounded_exact_equal"])
            self.assertEqual(audit["sources"]["skew"]["source_sha256"], digest)
            self.assertEqual(audit["sources"]["skew"]["bounded_rows"], 3)

    def test_bounded_raw_derived_value_difference_aborts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol, digest = sources_fixture(root, derived_value=159.031)
            with patch.object(features, "SKEW_SHA256", digest), self.assertRaisesRegex(ValueError, "derived"):
                features.load_sources(protocol, root)

    def test_historical_anchor_is_fixed_even_if_metadata_and_derived_are_changed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol, digest = sources_fixture(root, raw_value="140", derived_value=140.)
            path = root/protocol["sources"]["skew_source"]
            metadata = json.loads(path.read_text())
            metadata["anchor_close"] = 140.
            path.write_text(json.dumps(metadata))
            with patch.object(features, "SKEW_SHA256", digest), self.assertRaisesRegex(ValueError, "anchor"):
                features.load_sources(protocol, root)


if __name__ == "__main__":
    unittest.main()
