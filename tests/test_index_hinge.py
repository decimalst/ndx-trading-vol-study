"""Synthetic contracts written before the index-hinge producer exists."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import index_hinge as model


def sample(n=180):
    dates = pd.bdate_range("2014-01-02", periods=n, name="date").delete([12, 37])
    z = np.arange(len(dates), dtype=float)
    close = 100*np.exp(np.cumsum(.001+.008*np.sin(z*.37)))
    opening = close*np.exp(.004*np.cos(z*.23))
    daily = pd.DataFrame({"open": opening, "high": np.maximum(opening, close)*1.006,
                          "low": np.minimum(opening, close)*.994, "close": close,
                          "adj close": close*.7}, index=dates)
    iv = pd.DataFrame({"vix": 18+3*np.sin(z*.17), "vix9d": 16+2*np.cos(z*.21),
                       "vvix": 85+4*np.cos(z*.19)}, index=dates)
    return daily, iv


def design(n=300, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n, name="date")
    frame = pd.DataFrame(rng.normal(size=(n, len(model.RAW))), index=dates, columns=model.RAW)
    frame["const"] = 1.
    frame["I"] -= 3
    frame["R"] -= 4
    for weekday in range(1, 5):
        frame[f"entry_dow_{weekday}"] = (dates.weekday == weekday).astype(float)
    return frame


class TestIndexHinge(unittest.TestCase):
    def test_fixed_schema(self):
        raw = ("const", "I", "R", "ret_d", "ret_w", "ret_m", "ret_q", "lr_d", "lr_w",
               "term", "lvvix", "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
        self.assertEqual(model.RAW, raw)
        self.assertEqual(model.BASE, raw+("I_square", "R_square"))
        self.assertEqual(model.ALL_FEATURES, model.BASE+("hinge",))
        self.assertEqual(model.MODELS, ("mean", "baseline", "hinge"))
        self.assertEqual(model.HORIZONS, (21, 63))
        features, targets = model.build_features(*sample())
        self.assertEqual(tuple(features.columns), raw+("feature_cutoff_date",))
        self.assertEqual(tuple(targets), (21, 63))
        for target in targets.values():
            self.assertEqual(tuple(target.columns), ("y", "target_end", "available_date"))

    def test_lagged_market_features_match_explicit_windows(self):
        daily, iv = sample()
        features, _ = model.build_features(daily, iv)
        returns = np.log(daily.close/daily.close.shift())
        gk = np.maximum(.5*np.log(daily.high/daily.low)**2
                        -(2*np.log(2)-1)*np.log(daily.close/daily.open)**2, 1e-10)
        variance = gk+np.log(daily.open/daily.close.shift())**2
        t = 90
        for suffix, width in [("d", 1), ("w", 5), ("m", 22), ("q", 63)]:
            self.assertAlmostEqual(features.iloc[t]["ret_"+suffix], returns.iloc[t-width:t].mean())
        for name, width in [("lr_d", 1), ("lr_w", 5), ("R", 22)]:
            self.assertAlmostEqual(features.iloc[t][name], np.log(252*variance.iloc[t-width:t].mean()))
        self.assertAlmostEqual(features.iloc[t].I, np.log((iv.vix.iloc[t-1]/100)**2))
        self.assertAlmostEqual(features.iloc[t].term, np.log(iv.vix9d.iloc[t-1]/iv.vix.iloc[t-1]))
        self.assertAlmostEqual(features.iloc[t].lvvix, np.log(iv.vvix.iloc[t-1]))
        self.assertEqual(features.iloc[t].feature_cutoff_date, daily.index[t-1])
        for weekday in range(1, 5):
            self.assertEqual(features.iloc[t][f"entry_dow_{weekday}"], float(daily.index[t].weekday() == weekday))

    def test_targets_use_raw_close_exact_session_horizons_and_same_close_availability(self):
        daily, iv = sample()
        _, targets = model.build_features(daily, iv)
        for horizon in model.HORIZONS:
            target = targets[horizon]
            self.assertAlmostEqual(target.iloc[40].y, np.log(daily.close.iloc[40+horizon]/daily.close.iloc[40]))
            self.assertEqual(target.iloc[40].target_end, daily.index[40+horizon])
            self.assertEqual(target.iloc[40].available_date, daily.index[40+horizon])
            self.assertTrue(target.iloc[-horizon:].isna().all().all())
            self.assertTrue(target.index.equals(daily.index))

    def test_zero_return_targets_and_gk_floor_are_valid(self):
        daily, iv = sample()
        daily.loc[:, ["open", "high", "low", "close"]] = 100.
        features, targets = model.build_features(daily, iv)
        self.assertAlmostEqual(features.iloc[90].R, np.log(252e-10))
        for target in targets.values():
            self.assertEqual(target.iloc[40].y, 0.)

    def test_price_unit_changes_and_adjusted_close_do_not_change_outputs(self):
        daily, iv = sample()
        before, targets_before = model.build_features(daily, iv)
        changed = daily.copy()
        changed.loc[:, ["open", "high", "low", "close"]] *= 7
        changed["adj close"] = np.nan
        after, targets_after = model.build_features(changed, iv)
        np.testing.assert_allclose(before.loc[:, model.RAW], after.loc[:, model.RAW], atol=2e-13, equal_nan=True)
        for horizon in model.HORIZONS:
            np.testing.assert_allclose(targets_before[horizon].y, targets_after[horizon].y, atol=1e-14, equal_nan=True)

    def test_entry_and_future_market_mutations_cannot_change_current_features(self):
        daily, iv = sample()
        before, _ = model.build_features(daily, iv)
        changed, changed_iv = daily.copy(), iv.copy()
        changed.iloc[90:] *= 3
        changed_iv.iloc[90:] *= 2
        after, _ = model.build_features(changed, changed_iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])
        self.assertNotEqual(before.iloc[91].I, after.iloc[91].I)

    def test_future_observed_calendar_does_not_enter_predictors(self):
        daily, iv = sample()
        before, targets_before = model.build_features(daily, iv)
        after, targets_after = model.build_features(daily.drop(daily.index[91]), iv)
        pd.testing.assert_frame_equal(before.iloc[:91], after.iloc[:91])
        self.assertNotEqual(targets_before[21].iloc[90].target_end, targets_after[21].iloc[90].target_end)

    def test_missing_prices_remain_in_strict_rolling_windows(self):
        daily, iv = sample()
        daily.loc[daily.index[80], ["open", "high", "low", "close"]] = np.nan
        features, targets = model.build_features(daily, iv)
        self.assertEqual(len(features), len(daily))
        self.assertTrue(features.ret_w.iloc[81:87].isna().all())
        self.assertTrue(np.isfinite(features.iloc[87].ret_w))
        self.assertTrue(features.R.iloc[81:104].isna().all())
        self.assertTrue(np.isfinite(features.iloc[104].R))
        self.assertTrue(np.isnan(targets[21].iloc[59].y))
        self.assertEqual(targets[21].iloc[59].target_end, daily.index[80])

    def test_missing_iv_is_not_filled_or_calendar_compressed(self):
        daily, iv = sample()
        features, _ = model.build_features(daily, iv.drop(daily.index[80]))
        self.assertTrue(features.iloc[81][["I", "term", "lvvix"]].isna().all())
        self.assertTrue(np.isfinite(features.iloc[82].I))
        self.assertEqual(features.iloc[81].feature_cutoff_date, daily.index[80])

    def test_invalid_observed_market_data_and_calendar_are_rejected(self):
        daily, iv = sample()
        for bad in [0., -1., np.inf]:
            with self.subTest(value=bad):
                changed = daily.copy()
                changed.loc[changed.index[80], "close"] = bad
                with self.assertRaises(ValueError):
                    model.build_features(changed, iv)
        invalid_range = daily.copy()
        invalid_range.loc[invalid_range.index[80], "high"] = .5
        with self.assertRaises(ValueError):
            model.build_features(invalid_range, iv)
        for invalid in [daily.iloc[::-1], pd.concat([daily, daily.iloc[-1:]])]:
            with self.assertRaises(ValueError):
                model.build_features(invalid, iv)

    def test_transform_is_training_centered_with_separate_component_squares(self):
        train, apply = design(), design(30, 8)
        tr, ap, audit = model.transform(train, apply)
        self.assertEqual(tuple(tr.columns), model.ALL_FEATURES)
        self.assertEqual(tuple(ap.columns), model.ALL_FEATURES)
        self.assertEqual(audit["train_n"], len(train))
        for name in ["I", "R"]:
            self.assertEqual(audit[name+"_mean"], train[name].mean())
            np.testing.assert_allclose(tr[name+"_square"], (train[name]-train[name].mean())**2)
            np.testing.assert_allclose(ap[name+"_square"], (apply[name]-train[name].mean())**2)
        gap_mean = (train.I-train.R).mean()
        self.assertEqual(audit["gap_mean"], gap_mean)
        np.testing.assert_allclose(ap.hinge, np.maximum(apply.I-apply.R-gap_mean, 0))
        self.assertTrue((tr.hinge == 0).any())
        self.assertTrue((tr.hinge > 0).any())

    def test_apply_values_do_not_change_training_transform_or_audit(self):
        train, apply = design(), design(30, 8)
        original_train, original_apply = train.copy(), apply.copy()
        tr, _, audit = model.transform(train, apply)
        changed = apply.copy()
        changed["I"] += 100
        tr_changed, _, audit_changed = model.transform(train, changed)
        pd.testing.assert_frame_equal(tr, tr_changed)
        self.assertEqual(audit, audit_changed)
        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(apply, original_apply)

    def test_gap_is_redundant_but_hinge_is_not_a_component_span_on_synthetic_design(self):
        train = design()
        transformed, _, _ = model.transform(train, train.iloc[:2])
        base = transformed.loc[:, model.BASE].to_numpy(float)
        self.assertEqual(np.linalg.matrix_rank(np.column_stack([base, train.I-train.R])), np.linalg.matrix_rank(base))
        self.assertEqual(np.linalg.matrix_rank(transformed.to_numpy(float)), np.linalg.matrix_rank(base)+1)

    def test_ridge_matches_independent_augmented_least_squares_and_mean(self):
        train, apply = design(), design(30, 8)
        y = pd.Series(.1+train.I*.04+np.sin(train.R)*.1, index=train.index)
        result = model.fit_predict(train, y, apply)
        tr, ap, audit = model.transform(train, apply)
        self.assertEqual(result["transform_audit"], audit)
        np.testing.assert_array_equal(result["predictions"]["mean"], np.full(len(apply), y.mean()))
        self.assertEqual(result["model_audit"]["mean"]["alpha"], 0.)
        for name, columns in [("baseline", model.BASE), ("hinge", model.ALL_FEATURES)]:
            a, b = tr.loc[:, columns[1:]].to_numpy(float), ap.loc[:, columns[1:]].to_numpy(float)
            means, scales = a.mean(axis=0), a.std(axis=0, ddof=0)
            z = (a-means)/scales
            matrix = np.vstack([z, np.sqrt(len(z)*.01)*np.eye(z.shape[1])])
            rhs = np.r_[y-y.mean(), np.zeros(z.shape[1])]
            beta = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
            expected = y.mean()+(b-means)/scales@beta
            np.testing.assert_allclose(result["predictions"][name], expected, rtol=1e-12, atol=1e-12)
            fit_audit = result["model_audit"][name]
            self.assertEqual(fit_audit["columns"], list(columns))
            np.testing.assert_allclose(fit_audit["means"], np.r_[0., means])
            np.testing.assert_allclose(fit_audit["scales"], np.r_[1., scales])
            np.testing.assert_allclose(fit_audit["beta"], np.r_[y.mean(), beta], atol=1e-12)
            self.assertEqual(fit_audit["alpha"], .01)
            self.assertLess(fit_audit["gradient_max_abs"], 1e-12)

    def test_ridge_mean_loss_is_invariant_to_replication_of_training_rows(self):
        train, apply = design(), design(30, 8)
        y = np.sin(train.I.to_numpy())
        first = model.fit_predict(train, y, apply)
        repeated = model.fit_predict(pd.concat([train, train]), np.tile(y, 2), apply)
        for name in model.MODELS:
            np.testing.assert_allclose(first["predictions"][name], repeated["predictions"][name], atol=1e-12)

    def test_zero_scale_any_common_feature_aborts_all_models(self):
        train, apply = design(), design(30, 8)
        train["entry_dow_4"] = 0.
        with self.assertRaisesRegex(ValueError, "scale"):
            model.fit_predict(train, np.zeros(len(train)), apply)
        train = design()
        train["R"] = train.I-1
        with self.assertRaisesRegex(ValueError, "scale"):
            model.fit_predict(train, np.zeros(len(train)), apply)

    def test_invalid_fit_data_are_rejected_without_silent_row_selection(self):
        train, apply = design(), design(30, 8)
        y = pd.Series(np.zeros(len(train)), index=train.index)
        with self.assertRaises(ValueError):
            model.fit_predict(train, y.iloc[::-1], apply)
        with self.assertRaises(ValueError):
            model.fit_predict(train, np.zeros((len(train), 1)), apply)
        for column in ["I", "ret_q", "entry_dow_4"]:
            changed = train.copy()
            changed.loc[changed.index[10], column] = np.nan
            with self.assertRaises(ValueError):
                model.fit_predict(changed, y, apply)
        changed = apply.copy()
        changed["const"] = 0.
        with self.assertRaises(ValueError):
            model.fit_predict(train, y, changed)

    def test_source_loader_filters_dates_before_numeric_csv_conversion_and_selects_raw_ohlc(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dates = pd.DatetimeIndex(["2025-10-17", "2025-10-20", "2025-11-03"], name="date")
            daily = pd.DataFrame({"open": [100., 101., np.inf], "high": [102., 103., np.inf],
                                  "low": [99., 100., np.inf], "close": [101., 102., np.inf],
                                  "adj close": [1., 2., 3.]}, index=dates)
            daily.to_parquet(root/"daily.parquet")
            sources = {"daily": "daily.parquet"}
            for name, column, values in [("vix", "CLOSE", [20, 21]), ("vix9d", "CLOSE", [16, 17]),
                                         ("vvix", "VVIX", [80, 81])]:
                sources[name] = name+".csv"
                (root/sources[name]).write_text(f"DATE,{column}\n10/17/2025,{values[0]}\n10/20/2025,{values[1]}\n11/03/2025,FORBIDDEN_NUMERIC_TOKEN\n")
            protocol = {"sources": sources, "index": {"source_end": "2025-10-20", "sealed_start": "2025-11-03"}}
            bounded, iv, audit = model.load_sources(protocol, root)
            self.assertEqual(tuple(bounded.columns), ("open", "high", "low", "close"))
            self.assertEqual(bounded.index.tolist(), list(dates[:2]))
            self.assertEqual(iv.index.tolist(), list(dates[:2]))
            self.assertEqual(iv.iloc[0].to_dict(), {"vix": 20., "vix9d": 16., "vvix": 80.})
            self.assertFalse(audit["numeric_post_cutoff_values_parsed"])
            self.assertFalse(audit["historical_vintage_certified"])
            for name, source in sources.items():
                self.assertEqual(audit["sources"][name]["source_sha256"], hashlib.sha256((root/source).read_bytes()).hexdigest())
                self.assertEqual(audit["sources"][name]["bounded_rows"], 2)
            protocol["index"]["source_end"] = "2025-11-03"
            with self.assertRaisesRegex(ValueError, "cutoff|bound|fence"):
                model.load_sources(protocol, root)


if __name__ == "__main__":
    unittest.main()
