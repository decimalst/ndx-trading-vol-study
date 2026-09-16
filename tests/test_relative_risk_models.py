"""Prewritten signed-risk, common-row and fixed ridge contracts for wave9."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import relative_risk_models as model
from src.relative_risk_features import ALL_FEATURES, BASE, MODELS


def sample(n=120):
    rng = np.random.default_rng(814)
    dates = pd.bdate_range("2015-01-01", periods=n)
    f = pd.DataFrame(
        rng.normal(size=(n, len(ALL_FEATURES))), index=dates, columns=ALL_FEATURES
    )
    f["const"] = 1.0
    f["corr22"] = rng.uniform(-0.8, 0.8, n)
    f["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    y = rng.normal(0.4, 0.7, n)
    t = pd.DataFrame(
        {
            "y": y,
            "target_end": pd.Series(dates, index=dates).shift(-1),
            "available_date": pd.Series(dates, index=dates).shift(-1),
        },
        index=dates,
    )
    t.loc[dates[-1], "y"] = np.nan
    return f, t


def config(dates):
    return {
        "models": list(MODELS),
        "baseline": list(BASE),
        "minimum_train": 20,
        "origin_start": str(dates[50].date()),
        "origin_end": str(dates[-2].date()),
        "latest_target": str(dates[-1].date()),
        "development": [str(dates[50].date()), str(dates[74].date())],
        "development_target_available_by": str(dates[74].date()),
        "evaluation": [str(dates[75].date()), str(dates[-2].date())],
    }


class RelativeRiskModels(unittest.TestCase):
    def test_maturity_uses_previous_observed_session_and_all_features(self):
        f, t = sample()
        mask = model.training_mask(f, t, f.index[60], 20)
        self.assertTrue(mask.iloc[58])
        self.assertFalse(mask.iloc[59])
        f.loc[f.index[30], "corr22"] = np.nan
        after = model.training_mask(f, t, f.index[60], 20)
        self.assertEqual(int(mask.sum()) - int(after.sum()), 1)
        self.assertFalse(after.iloc[30])

    def test_minimum_cannot_be_silently_skipped(self):
        f, t = sample()
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            model.training_mask(f, t, f.index[60], 1000)

    def test_exact_next_session_maturity_is_required(self):
        f, t = sample()
        t.loc[f.index[40], ["target_end", "available_date"]] = f.index[42]
        with self.assertRaises(ValueError):
            model.training_mask(f, t, f.index[60], 20)

    def test_exact_previous_session_cutoff_is_required(self):
        f, t = sample()
        f.loc[f.index[60], "feature_cutoff_date"] = f.index[58]
        with self.assertRaises(ValueError):
            model.training_mask(f, t, f.index[60], 20)

    def test_infinite_target_is_rejected(self):
        f, t = sample()
        for bad in [np.inf, -np.inf]:
            altered = t.copy()
            altered.loc[f.index[30], "y"] = bad
            with self.assertRaises(ValueError):
                model.training_mask(f, altered, f.index[60], 20)

    def test_same_common_rows_and_one_joint_correlation(self):
        f, t = sample()
        p, audits = model.fit_predict(f.iloc[:80], t.y.iloc[:80], f.iloc[80:90])
        self.assertEqual(set(p), set(MODELS))
        self.assertTrue(all(np.isfinite(v).all() for v in p.values()))
        np.testing.assert_allclose(p["mean"], t.y.iloc[:80].mean())
        self.assertEqual(audits["baseline"]["columns"], list(BASE))
        self.assertEqual(audits["correlation"]["columns"], list(ALL_FEATURES))
        self.assertTrue(all(a["train_n"] == 80 for a in audits.values()))
        self.assertLessEqual(audits["correlation"]["gradient_max_abs"], 1e-10)

    def test_target_scale_equivariance(self):
        f, t = sample()
        p, _ = model.fit_predict(f.iloc[:80], t.y.iloc[:80], f.iloc[80:90])
        other, _ = model.fit_predict(f.iloc[:80], t.y.iloc[:80] * 100, f.iloc[80:90])
        for name in MODELS:
            np.testing.assert_allclose(other[name], p[name] * 100, rtol=1e-10)

    def test_zero_scale_candidate_aborts_even_baseline(self):
        f, t = sample()
        f["corr22"] = 0.0
        with self.assertRaisesRegex(ValueError, "zero-scale"):
            model.fit_predict(f.iloc[:80], t.y.iloc[:80], f.iloc[80:90])

    def test_label_order_is_checked(self):
        f, t = sample()
        with self.assertRaises(ValueError):
            model.fit_predict(f.iloc[:80], t.y.iloc[:80].iloc[::-1], f.iloc[80:90])

    def test_apply_rows_do_not_change_training_scaling(self):
        f, t = sample()
        _, before = model.fit_predict(f.iloc[:80], t.y.iloc[:80], f.iloc[80:90])
        altered = f.iloc[80:90].copy()
        altered["qqq_lg_d"] += 2
        _, after = model.fit_predict(f.iloc[:80], t.y.iloc[:80], altered)
        self.assertEqual(before, after)

    def test_refit_origin_is_independent_of_query_label(self):
        f, t = sample()
        c = config(f.index)

        def fake_fit(train, y, query):
            return {name: np.full(len(query), y.mean()) for name in MODELS}, {}

        with patch.object(model, "fit_predict", side_effect=fake_fit):
            panel, fits = model.forecast_panel(f, t, c)
            altered = t.copy()
            altered.loc[f.index[50], "y"] = np.nan
            other, new_fits = model.forecast_panel(f, altered, c)
        self.assertEqual(fits[0]["fit_origin"], str(f.index[50].date()))
        self.assertEqual(new_fits[0]["fit_origin"], fits[0]["fit_origin"])
        self.assertTrue(panel.origin.eq(f.index[50]).any())
        self.assertFalse(other.origin.eq(f.index[50]).any())
        dev = panel.loc[panel.phase == "development"]
        self.assertTrue(dev.available_date.le(c["development_target_available_by"]).all())
        self.assertFalse(dev.origin.eq(f.index[74]).any())
        self.assertTrue(panel.groupby("origin").model.nunique().eq(3).all())

    def test_future_labels_cannot_change_earlier_fit(self):
        f, t = sample()
        c = config(f.index)

        def fake_fit(train, y, query):
            return {name: np.full(len(query), y.mean()) for name in MODELS}, {}

        with patch.object(model, "fit_predict", side_effect=fake_fit):
            panel, _ = model.forecast_panel(f, t, c)
            altered = t.copy()
            altered.loc[f.index[60] :, "y"] *= 2
            other, _ = model.forecast_panel(f, altered, c)
        first = panel.fit_origin.min()
        np.testing.assert_array_equal(
            panel.loc[panel.fit_origin.eq(first), "prediction"],
            other.loc[other.fit_origin.eq(first), "prediction"],
        )

    def test_negative_and_zero_log_risk_ratios_are_valid(self):
        f, t = sample()
        t.loc[f.index[30], "y"] = -2.0
        t.loc[f.index[31], "y"] = 0.0
        mask = model.training_mask(f, t, f.index[60], 20)
        self.assertTrue(mask.loc[f.index[30]])
        self.assertTrue(mask.loc[f.index[31]])
        predictions, _ = model.fit_predict(f.iloc[:80], t.y.iloc[:80], f.iloc[80:90])
        self.assertTrue(all(np.isfinite(p).all() for p in predictions.values()))

    def test_component_ridge_subtraction_is_same_forecast_not_new_arm(self):
        f, t = sample()
        first = pd.Series(-8.0 + t.y.iloc[:80].to_numpy(), index=f.index[:80])
        second = pd.Series(-9.0 + np.cos(np.arange(80) / 9), index=f.index[:80])
        one, _ = model.fit_predict(f.iloc[:80], first, f.iloc[80:90])
        two, _ = model.fit_predict(f.iloc[:80], second, f.iloc[80:90])
        gap, _ = model.fit_predict(f.iloc[:80], first - second, f.iloc[80:90])
        for name in MODELS:
            np.testing.assert_allclose(
                gap[name], one[name] - two[name], rtol=1e-11, atol=1e-12
            )

    def test_saved_coefficients_satisfy_full_normalized_mse_gradient(self):
        f, t = sample()
        _, audit = model.fit_predict(f.iloc[:80], t.y.iloc[:80], f.iloc[80:90])
        for name in MODELS:
            a = audit[name]
            x = f.loc[f.index[:80], a["columns"]].to_numpy(float)
            z = (x - np.asarray(a["means"])) / np.asarray(a["scales"])
            beta = np.asarray(a["beta"])
            gradient = 2 * z.T @ (z @ beta - t.y.iloc[:80].to_numpy()) / 80
            gradient[1:] += 2 * a["alpha"] * beta[1:]
            np.testing.assert_allclose(gradient, 0.0, atol=1e-10)
            self.assertLessEqual(a["gradient_max_abs"], 1e-10)

    def test_constant_signed_target_keeps_unpenalized_intercept(self):
        f, _t = sample()
        y = pd.Series(-1.5, index=f.index[:80])
        predictions, audits = model.fit_predict(f.iloc[:80], y, f.iloc[80:90])
        for name in MODELS:
            np.testing.assert_allclose(predictions[name], -1.5, atol=1e-12)
            self.assertAlmostEqual(audits[name]["beta"][0], -1.5)


if __name__ == "__main__":
    unittest.main()
