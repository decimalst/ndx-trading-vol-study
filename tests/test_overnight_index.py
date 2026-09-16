"""Pre-implementation synthetic contracts for entry-compatible overnight forecasts."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import overnight_index as study

BASE = ("const", "on_d", "on_w", "on_m", "day_d", "day_w", "day_m",
        "lrv_d", "lrv_w", "lrv_m", "liv", "lvix",
        "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
BLOCKS = {"cross_signed": ("x_hyg", "x_tlt", "x_gld", "x_uso", "x_uup"),
          "volume_pressure": ("pressure_d", "pressure_w"), "iv_shape": ("term", "lvvix")}
ALL = BASE + sum(BLOCKS.values(), ())
MODELS = ("mean", "baseline", "cross_signed", "volume_pressure", "iv_shape")


def fixture(n=1600):
    rng = np.random.default_rng(903201)
    index = pd.bdate_range("2010-01-04", periods=n, name="date")
    close = 100*np.exp(np.cumsum(rng.normal(.0002, .011, n)))
    opening = close*np.exp(rng.normal(0, .005, n))
    daily = pd.DataFrame({"open": opening, "close": close,
                          "high": np.maximum(opening, close)*np.exp(rng.uniform(.001, .02, n)),
                          "low": np.minimum(opening, close)*np.exp(-rng.uniform(.001, .02, n)),
                          "adj close": close*np.exp(np.linspace(-.04, 0, n)),
                          "volume": np.exp(rng.normal(17, .4, n))}, index=index)
    cross = pd.DataFrame(80*np.exp(np.cumsum(rng.normal(0, .012, (n, 5)), axis=0)),
                         index=index, columns=["hyg", "tlt", "gld", "uso", "uup"])
    iv = pd.DataFrame({"vxn": rng.uniform(15, 40, n), "vix": rng.uniform(12, 35, n),
                       "vix9d": rng.uniform(10, 35, n), "vvix": rng.uniform(65, 145, n)}, index=index)
    return daily, cross, iv


def config(index):
    return {"models": list(MODELS), "baseline": list(BASE), "horizons": [1],
            "ridge_alpha": .01, "minimum_train": 1000,
            "origin_start": str(index[1200].date()), "origin_end": str(index[1450].date()),
            "development": [str(index[1200].date()), str(index[1320].date())],
            "development_target_available_by": str(index[1320].date()),
            "evaluation": [str(index[1321].date()), str(index[1450].date())],
            "latest_target": str(index[1451].date())}


class OvernightFeatureTests(unittest.TestCase):
    def test_fixed_feature_and_model_family(self):
        self.assertEqual(study.BASE, BASE)
        self.assertEqual(study.BLOCKS, BLOCKS)
        self.assertEqual(study.ALL_FEATURES, ALL)
        self.assertEqual(study.MODELS, MODELS)

    def test_exact_target_feature_cutoff_and_availability(self):
        d, c, iv = fixture(400)
        f, target = study.build_features(d, c, iv)
        day = np.log(d.close/d.open)
        overnight = np.log(d["adj close"]).diff()-day
        np.testing.assert_allclose(target.y, np.expm1(overnight.shift(-1)), equal_nan=True)
        pd.testing.assert_series_equal(target.target_end, pd.Series(d.index, index=d.index, name="target_end").shift(-1))
        pd.testing.assert_series_equal(target.available_date, target.target_end.rename("available_date"))
        pd.testing.assert_series_equal(f.feature_cutoff_date, pd.Series(d.index, index=d.index, name="feature_cutoff_date").shift(1))
        gk = (.5*np.log(d.high/d.low)**2-(2*np.log(2)-1)*day**2).clip(lower=1e-10)
        variance = gk+np.log(d.open/d.close.shift(1))**2
        for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
            np.testing.assert_allclose(f["on_"+suffix], overnight.rolling(width).mean().shift(1), equal_nan=True)
            np.testing.assert_allclose(f["day_"+suffix], day.rolling(width).mean().shift(1), equal_nan=True)
            np.testing.assert_allclose(f["lrv_"+suffix], np.log(variance.rolling(width).mean()).shift(1), equal_nan=True)

    def test_iv_and_cross_only_use_previous_actual_qqq_session(self):
        d, c, iv = fixture(400)
        iv.loc[pd.Timestamp("2010-01-09")] = [999]*4
        iv = iv.sort_index()
        f, _ = study.build_features(d, c, iv)
        delayed = iv.reindex(d.index).shift(1)
        for output, input_column in [("liv", "vxn"), ("lvix", "vix"), ("lvvix", "vvix")]:
            np.testing.assert_allclose(f[output], np.log(delayed[input_column]), equal_nan=True)
        np.testing.assert_allclose(f.term, np.log(delayed.vix9d/delayed.vix), equal_nan=True)
        for col in c:
            np.testing.assert_allclose(f["x_"+col], np.log(c[col]).diff().shift(1), equal_nan=True)

    def test_volume_reference_excludes_current_observation_before_lag(self):
        d, c, iv = fixture(400)
        f, _ = study.build_features(d, c, iv)
        lv = np.log(d.volume)
        prior_mean = lv.rolling(252, min_periods=126).mean().shift(1)
        prior_sd = lv.rolling(252, min_periods=126).std(ddof=1).shift(1)
        pressure = np.log(d.close/d.open)*(lv-prior_mean)/prior_sd
        np.testing.assert_allclose(f.pressure_d, pressure.shift(1), equal_nan=True)
        np.testing.assert_allclose(f.pressure_w, pressure.rolling(5).mean().shift(1), equal_nan=True)
        self.assertTrue(np.isnan(f.pressure_d.iloc[126]))
        self.assertTrue(np.isfinite(f.pressure_d.iloc[127]))

    def test_all_entry_and_future_market_values_leave_preclose_features_unchanged(self):
        d, c, iv = fixture(400)
        f, _ = study.build_features(d, c, iv)
        entry = d.index[250]
        d.loc[entry:] *= 2
        c.loc[entry:] *= 3
        iv.loc[entry:] *= 4
        changed, _ = study.build_features(d, c, iv)
        pd.testing.assert_frame_equal(f.loc[:entry], changed.loc[:entry])

    def test_weekdays_use_known_entry_not_future_opening(self):
        d, c, iv = fixture(400)
        f, _ = study.build_features(d, c, iv)
        for day in range(1, 5):
            np.testing.assert_array_equal(f[f"entry_dow_{day}"], (d.index.weekday == day).astype(float))
        entry = d.index[250]
        # Removing a later session changes the next actual opening, not today's predictors.
        keep = d.index != d.index[251]
        g, _ = study.build_features(d.loc[keep], c.loc[keep], iv.loc[keep])
        pd.testing.assert_series_equal(f.loc[entry], g.loc[entry])

    def test_missing_and_nonpositive_volume_stays_missing(self):
        for invalid in [0., -1., np.nan, np.inf]:
            d, c, iv = fixture(400)
            d.loc[d.index[250], "volume"] = invalid
            f, _ = study.build_features(d, c, iv)
            with self.subTest(invalid=invalid):
                self.assertTrue(np.isnan(f.loc[d.index[251], "pressure_d"]))
                self.assertTrue(f.loc[d.index[251:256], "pressure_w"].isna().all())

    def test_zero_price_rejected_and_missing_price_not_replaced(self):
        d, c, iv = fixture(400)
        d.loc[d.index[200], "open"] = 0.
        with self.assertRaises(ValueError):
            study.build_features(d, c, iv)
        d.loc[d.index[200], "open"] = np.nan
        _, target = study.build_features(d, c, iv)
        self.assertTrue(np.isnan(target.loc[d.index[199], "y"]))


class CorporateActionTests(unittest.TestCase):
    def fixture_action(self, opening=99., entry_adjusted=99., next_close=101.):
        d, c, iv = fixture(400)
        entry, next_date = d.index[200:202]
        d.loc[entry, ["open", "high", "low", "close", "adj close"]] = [100, 101, 99, 100, entry_adjusted]
        d.loc[next_date, ["open", "high", "low", "close", "adj close"]] = [opening, max(opening, next_close)+1, min(opening, next_close)-1, next_close, next_close]
        return d, c, iv, entry, next_date

    def test_pure_dividend_drop_has_zero_proxy_return_and_action_flag(self):
        d, c, iv, entry, _ = self.fixture_action()
        _, target = study.build_features(d, c, iv)
        self.assertAlmostEqual(target.loc[entry, "y"], 0., places=14)
        self.assertTrue(target.loc[entry, "adjustment_event"])

    def test_dividend_proxy_is_not_claimed_equal_to_cash_pnl(self):
        d, c, iv, entry, _ = self.fixture_action(opening=100.)
        _, target = study.build_features(d, c, iv)
        self.assertAlmostEqual(target.loc[entry, "y"], 100/99-1, places=14)
        self.assertNotAlmostEqual(target.loc[entry, "y"], .01, places=6)

    def test_consistent_split_generates_no_fake_overnight_return(self):
        d, c, iv, entry, _ = self.fixture_action(opening=50., entry_adjusted=50., next_close=51.)
        _, target = study.build_features(d, c, iv)
        self.assertAlmostEqual(target.loc[entry, "y"], 0., places=14)

    def test_next_close_cancels_when_adjustment_factor_is_fixed(self):
        d, c, iv, entry, next_date = self.fixture_action()
        _, before = study.build_features(d, c, iv)
        d.loc[next_date, ["close", "adj close"]] *= 1.03
        d.loc[next_date, "high"] = d.loc[next_date, "close"]+1
        _, after = study.build_features(d, c, iv)
        self.assertAlmostEqual(before.loc[entry, "y"], after.loc[entry, "y"], places=14)
        self.assertAlmostEqual(before.loc[entry, "adjustment_log_change"], after.loc[entry, "adjustment_log_change"], places=14)

    def test_common_later_adjustment_multiplier_does_not_change_returns_or_features(self):
        d, c, iv = fixture(400)
        f, target = study.build_features(d, c, iv)
        d["adj close"] *= .72
        g, changed = study.build_features(d, c, iv)
        np.testing.assert_allclose(f.loc[:, ALL], g.loc[:, ALL], equal_nan=True, atol=2e-12)
        np.testing.assert_allclose(target.y, changed.y, equal_nan=True, atol=2e-14)
        pd.testing.assert_series_equal(target.adjustment_event, changed.adjustment_event)

    def test_consistent_share_unit_change_leaves_all_return_features_invariant(self):
        d, c, iv = fixture(400)
        f, target = study.build_features(d, c, iv)
        d.loc[:, ["open", "high", "low", "close", "adj close"]] *= .5
        g, changed = study.build_features(d, c, iv)
        np.testing.assert_allclose(f.loc[:, ALL], g.loc[:, ALL], equal_nan=True, atol=2e-12)
        np.testing.assert_allclose(target.y, changed.y, equal_nan=True, atol=2e-14)

    def test_adjustment_threshold_is_exact_and_last_missing_label_not_zero(self):
        d, c, iv = fixture(400)
        d["adj close"] = d.close
        d.loc[d.index[200]:, "adj close"] *= np.exp(.00002)
        _, target = study.build_features(d, c, iv)
        self.assertTrue(target.loc[d.index[199], "adjustment_event"])
        self.assertFalse(target.loc[d.index[198], "adjustment_event"])
        self.assertTrue(pd.isna(target.adjustment_event.iloc[-1]))
        self.assertTrue(np.isnan(target.y.iloc[-1]))


class OvernightModelTests(unittest.TestCase):
    def regression(self):
        rng = np.random.default_rng(321)
        x = pd.DataFrame(rng.normal(size=(150, len(ALL))), columns=ALL)
        x["const"] = 1.
        y = pd.Series(rng.normal(.0005, .007, len(x)), index=x.index)
        return x, y

    def test_every_ridge_and_mean_prediction_matches_independent_solution(self):
        x, y = self.regression()
        tr, ap = x.iloc[:120], x.iloc[120:]
        result = study.fit_predict(tr, y.iloc[:120], ap)
        self.assertEqual(tuple(result["predictions"]), MODELS)
        for model, block in [("baseline", ())] + list(BLOCKS.items()):
            cols = BASE[1:]+block
            a, b = tr.loc[:, cols].to_numpy(), ap.loc[:, cols].to_numpy()
            mean, sd = a.mean(0), a.std(0, ddof=0)
            a, b = (a-mean)/sd, (b-mean)/sd
            target = y.iloc[:120].to_numpy()
            slopes = np.linalg.solve(a.T@a/len(a)+.01*np.eye(len(cols)), a.T@(target-target.mean())/len(a))
            np.testing.assert_allclose(result["predictions"][model], target.mean()+b@slopes, atol=1e-13)
            audit = result["model_audit"][model]
            self.assertEqual(audit["columns"], ["const", *cols])
            np.testing.assert_allclose(audit["beta"], np.r_[target.mean(), slopes])
            np.testing.assert_allclose(audit["means"], np.r_[0., mean])
            np.testing.assert_allclose(audit["scales"], np.r_[1., sd])
        np.testing.assert_allclose(result["predictions"]["mean"], y.iloc[:120].mean())

    def test_application_values_cannot_change_training_fit(self):
        x, y = self.regression()
        a = study.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120:])
        changed = x.iloc[120:].copy()
        changed.loc[:, ALL[1:]] *= 100
        b = study.fit_predict(x.iloc[:120], y.iloc[:120], changed)
        self.assertEqual(a["model_audit"], b["model_audit"])

    def test_mean_loss_penalty_and_unpenalized_intercept(self):
        x, y = self.regression()
        a = study.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120:])
        b = study.fit_predict(pd.concat([x.iloc[:120]]*3), pd.concat([y.iloc[:120]]*3)+.02, x.iloc[120:])
        for model in MODELS:
            np.testing.assert_allclose(b["predictions"][model]-a["predictions"][model], .02, atol=1e-13)

    def test_zero_target_valid_and_zero_scale_not_silently_removed(self):
        x, y = self.regression()
        result = study.fit_predict(x.iloc[:120], y.iloc[:120]*0, x.iloc[120:])
        for model in MODELS:
            np.testing.assert_array_equal(result["predictions"][model], np.zeros(30))
        x["pressure_d"] = 0
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            study.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120:])

    def test_nonfinite_and_misaligned_labels_fail(self):
        x, y = self.regression()
        with self.assertRaises(ValueError):
            study.fit_predict(x.iloc[:120], y.iloc[:120].iloc[::-1], x.iloc[120:])
        y.iloc[10] = np.nan
        with self.assertRaises(ValueError):
            study.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120:])


class OvernightTimingTests(unittest.TestCase):
    def test_train_labels_are_available_by_previous_session_not_fit_entry_close(self):
        d, c, iv = fixture()
        f, target = study.build_features(d, c, iv)
        fit = d.index[1250]
        mask = study.training_mask(f, target, fit)
        self.assertTrue(mask.loc[d.index[1248]])
        self.assertFalse(mask.loc[d.index[1249]])
        self.assertFalse(mask.loc[d.index[1250]])
        self.assertTrue((target.loc[mask, "available_date"] <= d.index[1249]).all())

    def test_common_training_rows_and_unflagged_sensitivity_never_filter_training(self):
        d, c, iv = fixture()
        f, target = study.build_features(d, c, iv)
        entry = d.index[600]
        f.loc[entry, "x_hyg"] = np.nan
        mask = study.training_mask(f, target, d.index[1250])
        self.assertFalse(mask.loc[entry])
        target.loc[:, "adjustment_event"] = True
        pd.testing.assert_series_equal(mask, study.training_mask(f, target, d.index[1250]))
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            study.training_mask(f, target, d.index[600])

    def test_forecast_common_samples_monthly_fits_metadata_and_phase_fences(self):
        d, c, iv = fixture()
        f, target = study.build_features(d, c, iv)
        cfg = config(d.index)
        frame, fits = study.forecast_panel(f, target, cfg)
        self.assertEqual(set(frame.model), set(MODELS))
        self.assertTrue((frame.groupby("origin").size() == 5).all())
        self.assertTrue((frame.horizon == 1).all())
        self.assertTrue((frame.feature_cutoff_date < frame.origin).all())
        self.assertTrue((frame.train_last_available <= frame.fit_cutoff_date).all())
        self.assertTrue((frame.fit_cutoff_date < frame.fit_origin).all())
        self.assertTrue((frame.target_end > frame.origin).all())
        pd.testing.assert_series_equal(frame.target_end, frame.available_date.rename("target_end"))
        development = frame.loc[frame.phase == "development"]
        self.assertTrue((development.available_date <= cfg["development_target_available_by"]).all())
        self.assertNotIn(d.index[1320], set(development.origin))
        self.assertEqual(frame.groupby(frame.origin.dt.to_period("M")).fit_origin.nunique().max(), 1)
        self.assertEqual(len(fits), frame.fit_origin.nunique())
        self.assertTrue(np.isfinite(frame[["y", "prediction", "adjustment_log_change"]]).all().all())

    def test_future_market_data_cannot_change_earlier_issued_forecasts(self):
        d, c, iv = fixture()
        cfg = config(d.index)
        f, target = study.build_features(d, c, iv)
        before, _ = study.forecast_panel(f, target, cfg)
        entry = d.index[1350]
        d.loc[entry:] *= 1.4
        c.loc[entry:] *= 1.8
        iv.loc[entry:] *= .7
        f, target = study.build_features(d, c, iv)
        after, _ = study.forecast_panel(f, target, cfg)
        cols = ["origin", "model", "prediction", "fit_origin", "fit_cutoff_date", "train_n", "train_last_available"]
        pd.testing.assert_frame_equal(before.loc[before.origin <= entry, cols].reset_index(drop=True),
                                      after.loc[after.origin <= entry, cols].reset_index(drop=True))

    def test_loader_masks_sealed_rows_before_validating_values(self):
        d, c, iv = fixture(400)
        sealed = pd.DatetimeIndex(["2025-11-03"], name="date")
        d_bad = pd.concat([d, pd.DataFrame(0., index=sealed, columns=d.columns)])
        c_bad = pd.concat([c, pd.DataFrame(0., index=sealed, columns=c.columns)])
        iv_bad = pd.concat([iv, pd.DataFrame(0., index=sealed, columns=iv.columns)])
        sources = {"daily": "daily.parquet", "cross": "cross.parquet", **{s: s+".csv" for s in iv}}
        protocol = {"sources": sources, "index": {"source_end": "2025-10-20", "sealed_start": "2025-11-03"}}

        def fake_parquet(path, **kwargs):
            return c_bad.copy() if "cross" in str(path) else d_bad.copy()

        def fake_csv(path):
            name = str(path).split("/")[-1].split(".")[0]
            column = "VVIX" if name == "vvix" else "CLOSE"
            return pd.DataFrame({"DATE": iv_bad.index.strftime("%m/%d/%Y"), column: iv_bad[name].to_numpy()})

        with patch.object(study.pd, "read_parquet", side_effect=fake_parquet), \
                patch.object(study.pd, "read_csv", side_effect=fake_csv):
            actual_d, actual_c, actual_iv = study.load_inputs(protocol)
        f, _ = study.build_features(actual_d, actual_c, actual_iv)
        expected, _ = study.build_features(d, c, iv)
        pd.testing.assert_frame_equal(f, expected, check_freq=False)


if __name__ == "__main__":
    unittest.main()
