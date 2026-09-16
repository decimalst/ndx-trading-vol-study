"""Synthetic contracts written before the index-return implementation or scores."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import iterative_index as study

BASE = ("const", "cc_d", "cc_w", "cc_m", "lrv_d", "lrv_w", "lrv_m", "liv", "lvix")
BLOCKS = {
    "session_split": ("split_d", "split_w", "split_m"),
    "cross_signed": ("x_hyg", "x_tlt", "x_gld", "x_uso", "x_uup"),
    "calendar": ("target_dow_1", "target_dow_2", "target_dow_3", "target_dow_4", "target_first3"),
}
ALL = BASE + sum(BLOCKS.values(), ())


def fixture(n=420):
    rng = np.random.default_rng(60817)
    idx = pd.bdate_range("2012-01-03", periods=n, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0, .011, n)))
    opening = close * np.exp(rng.normal(0, .004, n))
    daily = pd.DataFrame({
        "open": opening, "close": close,
        "high": np.maximum(opening, close) * np.exp(rng.uniform(.001, .02, n)),
        "low": np.minimum(opening, close) * np.exp(-rng.uniform(.001, .02, n)),
        "adj close": close * np.exp(np.linspace(-.04, 0, n)),
    }, index=idx)
    cross = pd.DataFrame(80 * np.exp(np.cumsum(rng.normal(0, .012, (n, 5)), axis=0)),
                         index=idx, columns=["hyg", "tlt", "gld", "uso", "uup"])
    iv = pd.DataFrame({"vxn": rng.uniform(15, 45, n), "vix": rng.uniform(12, 35, n)}, index=idx)
    return daily, cross, iv


def regression_fixture(n=160):
    rng = np.random.default_rng(23892)
    index = pd.bdate_range("2014-01-02", periods=n)
    x = pd.DataFrame(rng.normal(size=(n, len(ALL))), index=index, columns=ALL)
    x["const"] = 1.
    y = pd.Series(.003 + .002 * x["cc_d"] + rng.normal(0, .01, n), index=index)
    return x, y


class FeatureTests(unittest.TestCase):
    def test_exact_family(self):
        self.assertEqual(study.BASE, BASE)
        self.assertEqual(study.BLOCKS, BLOCKS)
        self.assertEqual(study.ALL_FEATURES, ALL)
        self.assertEqual(study.MODELS, ("mean", "baseline", "session_split", "cross_signed", "calendar"))

    def test_returns_variance_and_signed_split_match_definitions(self):
        d, c, iv = fixture()
        f, y = study.build_features(d, c, iv)
        cc = np.log(d["adj close"]).diff()
        day = np.log(d.close / d.open)
        gk = (.5 * np.log(d.high / d.low)**2 - (2 * np.log(2) - 1) * day**2).clip(lower=1e-10)
        rv = gk + np.log(d.open / d.close.shift(1))**2
        for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
            np.testing.assert_allclose(f[f"cc_{suffix}"], cc.rolling(width).mean(), equal_nan=True)
            np.testing.assert_allclose(f[f"split_{suffix}"], (cc - 2*day).rolling(width).mean(), equal_nan=True)
            np.testing.assert_allclose(f[f"lrv_{suffix}"], np.log(rv.rolling(width).mean()), equal_nan=True)
        np.testing.assert_allclose(y.y, (d.close / d.open - 1).shift(-1), equal_nan=True)
        pd.testing.assert_series_equal(y.target_end, pd.Series(d.index, index=d.index, name="target_end").shift(-1))

    def test_dividend_and_split_decomposition_is_exact(self):
        d, c, iv = fixture(80)
        d.loc[d.index[40]:, ["open", "high", "low", "close"]] /= 2
        d["adj close"] = d.close * np.where(np.arange(len(d)) < 40, .5, 1.)
        d.loc[d.index[:60], "adj close"] *= .99
        f, _ = study.build_features(d, c, iv)
        day = np.log(d.close / d.open)
        adjusted_overnight = np.log(d["adj close"]).diff() - day
        np.testing.assert_allclose((f.split_d + 2*day), np.log(d["adj close"]).diff(), equal_nan=True)
        np.testing.assert_allclose(f.split_d, adjusted_overnight - day, equal_nan=True)

    def test_cboe_shift_is_after_qqq_alignment(self):
        d, c, iv = fixture()
        iv.loc[pd.Timestamp("2012-01-07")] = [999., 999.]
        iv = iv.sort_index()
        f, _ = study.build_features(d, c, iv)
        expected = iv.reindex(d.index).shift(1)
        np.testing.assert_allclose(f.liv, np.log(expected.vxn), equal_nan=True)
        np.testing.assert_allclose(f.lvix, np.log(expected.vix), equal_nan=True)

    def test_cross_missing_is_not_filled_and_direction_is_retained(self):
        d, c, iv = fixture()
        c.loc[c.index[60], "hyg"] = np.nan
        f, _ = study.build_features(d, c, iv)
        for name in c:
            np.testing.assert_allclose(f[f"x_{name}"], np.log(c[name]).diff(), equal_nan=True)
        self.assertTrue(f.loc[c.index[60:62], "x_hyg"].isna().all())

    def test_future_values_cannot_change_prior_features(self):
        d, c, iv = fixture()
        f, _ = study.build_features(d, c, iv)
        cut = d.index[200]
        changed_d, changed_c, changed_iv = d.copy(), c.copy(), iv.copy()
        changed_d.loc[changed_d.index > cut] *= 3
        changed_c.loc[changed_c.index > cut] *= .4
        changed_iv.loc[changed_iv.index >= cut] *= 9
        altered, _ = study.build_features(changed_d, changed_c, changed_iv)
        pd.testing.assert_frame_equal(f.loc[:cut], altered.loc[:cut])

    def test_current_iv_is_unavailable_but_current_qqq_is_available(self):
        d, c, iv = fixture()
        f, _ = study.build_features(d, c, iv)
        t = d.index[100]
        iv.loc[t] *= 5
        same, _ = study.build_features(d, c, iv)
        pd.testing.assert_series_equal(f.loc[t], same.loc[t])

    def test_calendar_uses_target_date_and_only_prior_session_count(self):
        history = pd.DatetimeIndex(["2021-01-28", "2021-01-29", "2021-02-01", "2021-02-02"])
        first = study.calendar_features(history, "2021-01-29", "2021-02-01")
        third = study.calendar_features(history, "2021-02-02", "2021-02-03")
        self.assertEqual(first["target_first3"], 1.)
        self.assertEqual(third["target_first3"], 1.)
        self.assertEqual(third["target_dow_2"], 1.)
        history = history.append(pd.DatetimeIndex(["2021-02-03", "2099-02-02"]))
        fourth = study.calendar_features(history, "2021-02-03", "2021-02-04")
        self.assertEqual(fourth["target_first3"], 0.)
        same = study.calendar_features(history, "2021-01-29", "2021-02-01")
        self.assertEqual(first, same)

    def test_calendar_rejects_target_not_after_origin(self):
        with self.assertRaises(ValueError):
            study.calendar_features(pd.bdate_range("2021-01-01", periods=10), "2021-01-04", "2021-01-04")

    def test_zero_target_return_is_valid(self):
        d, c, iv = fixture()
        d.loc[d.index[50], "close"] = d.loc[d.index[50], "open"]
        _, y = study.build_features(d, c, iv)
        self.assertEqual(y.loc[d.index[49], "y"], 0.)

    def test_bad_prices_and_duplicate_dates_fail(self):
        for which in ["zero", "negative", "infinite", "duplicate", "unsorted"]:
            d, c, iv = fixture()
            if which in ("zero", "negative", "infinite"):
                d.iloc[50, d.columns.get_loc("open")] = {"zero": 0, "negative": -1, "infinite": np.inf}[which]
            elif which == "duplicate":
                d = pd.concat([d, d.iloc[[-1]]])
            else:
                d = d.iloc[::-1]
            with self.subTest(which=which), self.assertRaises(ValueError):
                study.build_features(d, c, iv)

    def test_missing_target_open_does_not_become_zero(self):
        d, c, iv = fixture()
        d.loc[d.index[60], "open"] = np.nan
        _, y = study.build_features(d, c, iv)
        self.assertTrue(pd.isna(y.loc[d.index[59], "y"]))

    def test_loader_discards_sealed_values_before_feature_validation(self):
        d, c, iv = fixture()
        sealed = pd.Timestamp("2025-11-03")
        sealed_index = pd.DatetimeIndex([sealed], name="date")
        invalid_d = pd.concat([d, pd.DataFrame(0., index=sealed_index, columns=d.columns)])
        invalid_c = pd.concat([c, pd.DataFrame(0., index=sealed_index, columns=c.columns)])
        invalid_iv = pd.concat([iv, pd.DataFrame(0., index=sealed_index, columns=iv.columns)])

        def fake_parquet(path):
            return invalid_c.copy() if "cross" in str(path) else invalid_d.copy()

        def fake_csv(path):
            column = "vxn" if "VXN" in str(path) else "vix"
            return pd.DataFrame({"DATE": invalid_iv.index.strftime("%m/%d/%Y"),
                                 "CLOSE": invalid_iv[column].to_numpy()})

        with patch.object(study.pd, "read_parquet", side_effect=fake_parquet), \
                patch.object(study.pd, "read_csv", side_effect=fake_csv):
            actual_d, actual_c, actual_iv = study.load_inputs()
        for actual in (actual_d, actual_c, actual_iv):
            self.assertLess(actual.index.max(), sealed)
        f, _ = study.build_features(actual_d, actual_c, actual_iv)
        expected, _ = study.build_features(d, c, iv)
        pd.testing.assert_frame_equal(f, expected, check_freq=False)


class RegressionTests(unittest.TestCase):
    def test_exact_normalized_ridge_and_mean(self):
        x, y = regression_fixture()
        train, apply = x.iloc[:120], x.iloc[120:]
        got = study.fit_predict(train, y.iloc[:120], apply)
        for model, block in [("baseline", ())] + list(BLOCKS.items()):
            cols = BASE[1:] + block
            a = train.loc[:, cols].to_numpy()
            b = apply.loc[:, cols].to_numpy()
            mu, sd = a.mean(0), a.std(0, ddof=0)
            a, b = (a-mu)/sd, (b-mu)/sd
            target = y.iloc[:120].to_numpy()
            beta = np.linalg.solve(a.T@a/len(a)+.01*np.eye(len(cols)), a.T@(target-target.mean())/len(a))
            expected = target.mean()+b@beta
            np.testing.assert_allclose(got["forecasts"][model], expected, rtol=1e-10, atol=1e-12)
            audit = got["audit"][model]
            self.assertEqual(audit["columns"], ["const", *cols])
            np.testing.assert_allclose(audit["mean"], mu)
            np.testing.assert_allclose(audit["scale"], sd)
            np.testing.assert_allclose(audit["beta"], np.r_[target.mean(), beta])
        np.testing.assert_allclose(got["forecasts"]["mean"], y.iloc[:120].mean())

    def test_apply_rows_do_not_change_scaling_or_other_predictions(self):
        x, y = regression_fixture()
        a = study.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120:121])
        batch = pd.concat([x.iloc[120:121], x.iloc[121:]*100])
        batch["const"] = 1.
        b = study.fit_predict(x.iloc[:120], y.iloc[:120], batch)
        self.assertEqual(a["audit"], b["audit"])
        for model in study.MODELS:
            self.assertAlmostEqual(a["forecasts"][model][0], b["forecasts"][model][0], places=13)

    def test_intercept_unpenalized_and_target_shift_equivariant(self):
        x, y = regression_fixture()
        a = study.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120:])
        b = study.fit_predict(x.iloc[:120], y.iloc[:120]+.02, x.iloc[120:])
        for model in study.MODELS:
            np.testing.assert_allclose(b["forecasts"][model]-a["forecasts"][model], .02, atol=1e-14)

    def test_same_penalty_under_training_row_replication(self):
        x, y = regression_fixture()
        a = study.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120:])
        b = study.fit_predict(pd.concat([x.iloc[:120]]*3), pd.concat([y.iloc[:120]]*3), x.iloc[120:])
        for model in study.MODELS:
            np.testing.assert_allclose(a["forecasts"][model], b["forecasts"][model], atol=1e-13)

    def test_zero_scale_nonfinite_bad_intercept_and_alignment_fail(self):
        for which in ["scale", "nan", "intercept", "alignment"]:
            x, y = regression_fixture()
            if which == "scale":
                x["split_d"] = 1
            elif which == "nan":
                x.iloc[20, 2] = np.nan
            elif which == "intercept":
                x["const"] = 2
            else:
                y = y.iloc[::-1]
            with self.subTest(which=which), self.assertRaises(ValueError):
                study.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120:])

    def test_scalar_application(self):
        x, y = regression_fixture()
        a = study.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120])
        b = study.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120:121])
        for model in study.MODELS:
            self.assertIsInstance(a["forecasts"][model], float)
            self.assertEqual(a["forecasts"][model], b["forecasts"][model][0])


class TimingAndTradeTests(unittest.TestCase):
    def test_training_labels_must_be_completed_at_fit_origin(self):
        d, c, iv = fixture()
        f, y = study.build_features(d, c, iv)
        origin = d.index[200]
        mask = study.training_mask(f, y, origin, min_train=100)
        self.assertTrue(mask.loc[d.index[199]])
        self.assertFalse(mask.loc[origin])
        y.loc[d.index[180], "target_end"] = d.index[201]
        second = study.training_mask(f, y, origin, min_train=100)
        self.assertFalse(second.loc[d.index[180]])

    def test_common_rows_include_all_candidate_features(self):
        d, c, iv = fixture()
        f, y = study.build_features(d, c, iv)
        f.loc[d.index[100], "x_hyg"] = np.nan
        mask = study.training_mask(f, y, d.index[200], min_train=100)
        self.assertFalse(mask.loc[d.index[100]])
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            study.training_mask(f, y, d.index[60], min_train=100)

    def test_forecast_common_rows_monthly_refits_and_fences(self):
        d, c, iv = fixture()
        f, y = study.build_features(d, c, iv)
        start, end = d.index[210], d.index[330]
        f.loc[d.index[260], "split_d"] = np.nan
        panel, audit = study.forecast_panel(f, y, origin_start=start, origin_end=end,
                                           latest_target=d.index[331], min_train=100)
        self.assertEqual(set(panel.model), set(study.MODELS))
        np.testing.assert_array_equal(panel.horizon.to_numpy(), np.ones(len(panel), dtype=int))
        self.assertTrue(np.isfinite(panel[["y", "prediction"]].to_numpy()).all())
        self.assertTrue((panel.groupby("origin").size() == 5).all())
        self.assertNotIn(d.index[260], set(panel.origin))
        self.assertTrue((panel.target_end > panel.origin).all())
        self.assertTrue((panel.train_last_target <= panel.fit_origin).all())
        self.assertTrue((panel.fit_origin <= panel.origin).all())
        self.assertEqual(len(audit), panel.fit_origin.nunique())
        for _, group in panel.groupby(panel.origin.dt.to_period("M")):
            self.assertEqual(group.fit_origin.nunique(), 1)
            self.assertEqual(group.fit_origin.iloc[0], group.origin.min())

    def test_historical_forecasts_invariant_to_later_values(self):
        d, c, iv = fixture()
        f, y = study.build_features(d, c, iv)
        kwargs = {"origin_start": d.index[210], "origin_end": d.index[270],
                  "latest_target": d.index[271], "min_train": 100}
        a, _ = study.forecast_panel(f, y, **kwargs)
        f.loc[f.index > d.index[260], [col for col in ALL if col != "const"]] *= 12
        y.loc[y.index > d.index[260], "y"] *= 10
        b, _ = study.forecast_panel(f, y, **kwargs)
        cols = ["origin", "model", "prediction", "fit_origin", "train_n", "train_last_target"]
        pd.testing.assert_frame_equal(a.loc[a.origin <= d.index[260], cols].reset_index(drop=True),
                                      b.loc[b.origin <= d.index[260], cols].reset_index(drop=True))

    def test_development_targets_cannot_cross_into_evaluation(self):
        x, y = regression_fixture(600)
        x.index = pd.bdate_range("2018-01-02", periods=len(x))
        y.index = x.index
        targets = pd.DataFrame({"y": y, "target_end": pd.Series(x.index, index=x.index).shift(-1)})
        # BDay is a synthetic calendar; removing New Year's Day makes the next
        # observed target of Dec 31 be Jan 2, as in the empirical calendar.
        keep = x.index != pd.Timestamp("2020-01-01")
        x, targets = x.loc[keep], targets.loc[keep]
        targets.loc[pd.Timestamp("2019-12-31"), "target_end"] = pd.Timestamp("2020-01-02")
        panel, _ = study.forecast_panel(x, targets, origin_start="2019-12-02",
                                        origin_end="2020-02-20", latest_target="2020-02-21", min_train=100)
        self.assertNotIn(pd.Timestamp("2019-12-31"), set(panel.origin))
        development = panel.loc[panel.origin <= "2019-12-31"]
        self.assertTrue((development.target_end <= "2019-12-31").all())
        self.assertIn(pd.Timestamp("2020-01-02"), set(panel.origin))

    def test_roundtrip_costs_apply_each_day_and_threshold_is_fixed(self):
        index = pd.bdate_range("2020-01-02", periods=5)
        panel = pd.DataFrame({"origin": index[:-1], "target_end": index[1:],
                              "model": ["baseline"]*4, "prediction": [.001, .002, .0004, -.001],
                              "y": [.01, -.02, .02, .01]})
        trade = study.trade_screen(panel, costs=(0., .0002, .0005))
        self.assertIn("always_day_long", set(trade.model))
        expected_position = np.array([1., 1., 0., 0.])
        for cost in [0., .0002, .0005]:
            rows = trade[(trade.model == "baseline") & (trade.cost_per_side == cost)]
            np.testing.assert_array_equal(rows.position, expected_position)
            np.testing.assert_allclose(rows.net_return, expected_position*(panel.y.to_numpy()-2*cost))
        self.assertEqual(len(trade), 24)

    def test_costs_and_duplicate_panel_keys_fail(self):
        index = pd.bdate_range("2020-01-02", periods=2)
        panel = pd.DataFrame({"origin": [index[0]], "target_end": [index[1]], "model": ["baseline"],
                              "prediction": [.001], "y": [.01]})
        with self.assertRaises(ValueError):
            study.trade_screen(panel, costs=(-.001,))
        with self.assertRaises(ValueError):
            study.trade_screen(pd.concat([panel, panel]))


if __name__ == "__main__":
    unittest.main()
