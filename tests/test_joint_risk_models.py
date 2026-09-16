"""Prewritten shared-marginal, staging and chronological joint-risk contracts."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import joint_risk_models as model
from src.joint_risk_features import ALL_FEATURES, MODELS


def sample(n=160):
    rng = np.random.default_rng(614)
    dates = pd.bdate_range("2015-01-01", periods=n)
    f = pd.DataFrame(
        rng.normal(size=(n, len(ALL_FEATURES))), index=dates, columns=ALL_FEATURES
    )
    f["const"] = 1.0
    f["corr22"] = rng.uniform(-0.8, 0.8, n)
    f["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    u = rng.normal(size=(n, 2)) * 0.01
    u[:, 1] = 0.6 * u[:, 0] + 0.8 * u[:, 1]
    t = pd.DataFrame(u, columns=["y_qqq", "y_spx"], index=dates)
    for name in ["target_end", "available_date"]:
        t[name] = pd.Series(dates, index=dates).shift(-1)
    t.loc[dates[-1], ["y_qqq", "y_spx"]] = np.nan
    return f, t


def config(d):
    return {
        "models": list(MODELS),
        "all_features": list(ALL_FEATURES),
        "minimum_train": 20,
        "origin_start": str(d[60].date()),
        "origin_end": str(d[-2].date()),
        "latest_target": str(d[-1].date()),
        "development": [str(d[60].date()), str(d[94].date())],
        "development_target_available_by": str(d[94].date()),
        "evaluation": [str(d[95].date()), str(d[-2].date())],
    }


def fake_fit(train, targets, query):
    mean = targets[["y_qqq", "y_spx"]].mean().to_numpy()
    return {
        "forecasts": {
            name: {
                "mu": np.tile(mean, (len(query), 1)),
                "h": np.full((len(query), 2), 0.001),
                "rho": np.full(len(query), 0.2),
            }
            for name in MODELS
        },
        "audit": {},
    }


class JointRiskModels(unittest.TestCase):
    def test_training_requires_both_mature_labels_and_all_inputs(self):
        f, t = sample()
        mask = model.training_mask(f, t, f.index[60], 20)
        self.assertTrue(mask.iloc[58])
        self.assertFalse(mask.iloc[59])
        t.loc[f.index[30], "y_spx"] = np.nan
        f.loc[f.index[31], "qqq_day_d"] = np.nan
        after = model.training_mask(f, t, f.index[60], 20)
        self.assertEqual(int(mask.sum()) - int(after.sum()), 2)

    def test_zero_and_signed_returns_valid(self):
        f, t = sample()
        t.loc[f.index[30], ["y_qqq", "y_spx"]] = [0.0, -0.01]
        self.assertTrue(model.training_mask(f, t, f.index[60], 20).iloc[30])

    def test_minimum_not_relaxed(self):
        f, t = sample()
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            model.training_mask(f, t, f.index[60], 1000)

    def test_alignment_and_nonfinite_labels_rejected(self):
        for fault in ["cutoff", "maturity", "infinite", "order"]:
            f, t = sample()
            if fault == "cutoff":
                f.loc[f.index[40], "feature_cutoff_date"] = f.index[38]
            if fault == "maturity":
                t.loc[f.index[40], "available_date"] = f.index[42]
            if fault == "infinite":
                t.loc[f.index[40], "y_spx"] = np.inf
            if fault == "order":
                t = t.iloc[::-1]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                model.training_mask(f, t, f.index[60], 20)

    def test_transform_square_uses_training_center_only(self):
        f, _ = sample()
        a, b, audit = model.transform(f.iloc[:100], f.iloc[100:110])
        self.assertEqual(list(a.columns), [*ALL_FEATURES, "corr22_centered_sq"])
        center = f.corr22.iloc[:100].mean()
        np.testing.assert_allclose(a.corr22_centered_sq, (f.corr22.iloc[:100] - center) ** 2)
        np.testing.assert_allclose(
            b.corr22_centered_sq, (f.corr22.iloc[100:110] - center) ** 2
        )
        self.assertEqual(audit["corr22_mean"], center)
        altered = f.iloc[100:110].copy()
        altered["corr22"] = 0.99
        c, _, other = model.transform(f.iloc[:100], altered)
        pd.testing.assert_frame_equal(a, c)
        self.assertEqual(audit, other)

    def test_all_means_shared_and_dependence_diagonals_identical(self):
        f, t = sample()
        out = model.fit_predict(f.iloc[:120], t.iloc[:120], f.iloc[120:130])
        self.assertEqual(set(out["forecasts"]), set(MODELS))
        constant = out["forecasts"]["constant_correlation"]
        dynamic = out["forecasts"]["dynamic_correlation"]
        np.testing.assert_array_equal(constant["h"], dynamic["h"])
        for name in MODELS:
            p = out["forecasts"][name]
            np.testing.assert_array_equal(p["mu"], constant["mu"])
            self.assertTrue(np.isfinite(p["h"]).all())
            self.assertTrue((p["h"] > 0).all())
            self.assertTrue((np.abs(p["rho"]) <= 1 - 1e-6).all())
        for asset in ["qqq", "spx"]:
            a = out["audit"]["moments"][asset]
            self.assertIn("corr22", a["mean"]["columns"])
            self.assertIn("corr22_centered_sq", a["variance"]["columns"])
            self.assertLessEqual(a["mean"]["gradient_max_abs"], 1e-10)
            self.assertLessEqual(a["variance"]["gradient_max_abs"], 1e-8)
        self.assertEqual(out["audit"]["residual_staging"], "current_fit_training_residuals")

    def test_constant_matrix_is_uncentered_training_residual_second_moment(self):
        f, t = sample()
        out = model.fit_predict(f.iloc[:120], t.iloc[:120], f.iloc[120:130])
        transformed, _, _ = model.transform(f.iloc[:120], f.iloc[120:130])
        residuals = []
        for asset in ["qqq", "spx"]:
            a = out["audit"]["moments"][asset]["mean"]
            x = transformed[a["columns"]].to_numpy()
            z = (x - np.asarray(a["means"])) / np.asarray(a["scales"])
            residuals.append(t["y_" + asset].iloc[:120].to_numpy() - z @ np.asarray(a["beta"]))
        e = np.column_stack(residuals)
        matrix = e.T @ e / len(e)
        np.testing.assert_allclose(
            out["audit"]["constant_matrix"]["matrix"], matrix, rtol=1e-12
        )
        p = out["forecasts"]["constant_matrix"]
        np.testing.assert_allclose(p["h"], np.tile(np.diag(matrix), (10, 1)))
        np.testing.assert_allclose(
            p["rho"], matrix[0, 1] / np.sqrt(matrix[0, 0] * matrix[1, 1])
        )

    def test_zero_scale_or_zero_training_risk_aborts(self):
        f, t = sample()
        for fault in ["scale", "zero", "collinear"]:
            x, y = f.copy(), t.copy()
            if fault == "scale":
                x["spx_day_w"] = 0.0
            if fault == "zero":
                y[["y_qqq", "y_spx"]] = 0.0
            if fault == "collinear":
                y["y_spx"] = y.y_qqq
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                model.fit_predict(x.iloc[:120], y.iloc[:120], x.iloc[120:130])

    def test_label_order_and_invalid_application_rejected(self):
        f, t = sample()
        with self.assertRaises(ValueError):
            model.fit_predict(f.iloc[:120], t.iloc[:120].iloc[::-1], f.iloc[120:130])
        q = f.iloc[120:130].copy()
        q.iloc[0, 1] = np.inf
        with self.assertRaises(ValueError):
            model.fit_predict(f.iloc[:120], t.iloc[:120], q)

    def test_units_and_asset_order_preserve_dependence(self):
        f, t = sample()
        a = model.fit_predict(f.iloc[:120], t.iloc[:120], f.iloc[120:130])
        other = t.copy()
        other["y_qqq"] *= 10
        other["y_spx"] *= 100
        b = model.fit_predict(f.iloc[:120], other.iloc[:120], f.iloc[120:130])
        swapped = t.copy()
        swapped[["y_qqq", "y_spx"]] = t[["y_spx", "y_qqq"]].to_numpy()
        c = model.fit_predict(f.iloc[:120], swapped.iloc[:120], f.iloc[120:130])
        for name in MODELS:
            p, q, r = [x["forecasts"][name] for x in [a, b, c]]
            np.testing.assert_allclose(q["mu"], p["mu"] * [10, 100], atol=1e-10)
            np.testing.assert_allclose(q["h"], p["h"] * [100, 10000], rtol=1e-7)
            np.testing.assert_allclose(q["rho"], p["rho"], atol=1e-7)
            np.testing.assert_allclose(r["h"], p["h"][:, ::-1], rtol=1e-7)
            np.testing.assert_allclose(r["rho"], p["rho"], atol=1e-7)

    def test_application_does_not_change_any_fit(self):
        f, t = sample()
        query = f.iloc[120:130].copy()
        before = model.fit_predict(f.iloc[:120], t.iloc[:120], query)
        query["qqq_lg_d"] += 0.1
        query["corr22"] = 0.1
        after = model.fit_predict(f.iloc[:120], t.iloc[:120], query)
        self.assertEqual(before["audit"], after["audit"])

    def test_query_label_filter_cannot_move_monthly_refit(self):
        f, t = sample()
        c = config(f.index)
        with patch.object(model, "fit_predict", side_effect=fake_fit):
            panel, fits = model.forecast_panel(f, t, c)
            t.loc[f.index[60], "y_qqq"] = np.nan
            other, after = model.forecast_panel(f, t, c)
        self.assertEqual(fits[0]["fit_origin"], str(f.index[60].date()))
        self.assertEqual(fits[0]["fit_origin"], after[0]["fit_origin"])
        self.assertTrue(panel.origin.eq(f.index[60]).any())
        self.assertFalse(other.origin.eq(f.index[60]).any())
        self.assertFalse(
            panel.loc[panel.phase.eq("development"), "origin"].eq(f.index[94]).any()
        )
        self.assertTrue(panel.groupby("origin").model.nunique().eq(3).all())

    def test_entire_unscored_month_still_retains_scheduled_fit(self):
        f, t = sample()
        c = config(f.index)
        month = f.index[60].to_period("M")
        keep_month = f.index.to_period("M") == month
        t.loc[keep_month, ["y_qqq", "y_spx"]] = np.nan
        with patch.object(model, "fit_predict", side_effect=fake_fit):
            panel, fits = model.forecast_panel(f, t, c)
        self.assertEqual(fits[0]["fit_origin"], str(f.index[60].date()))
        self.assertGreater(fits[0]["application_n"], 0)
        self.assertFalse(panel.origin.dt.to_period("M").eq(month).any())

    def test_future_label_perturbation_leaves_first_fit_unchanged(self):
        f, t = sample()
        c = config(f.index)
        with patch.object(model, "fit_predict", side_effect=fake_fit):
            a, _ = model.forecast_panel(f, t, c)
            t.loc[f.index[70] :, ["y_qqq", "y_spx"]] *= 2
            b, _ = model.forecast_panel(f, t, c)
        first = a.fit_origin.min()
        np.testing.assert_array_equal(
            a.loc[a.fit_origin.eq(first), ["mu_qqq", "mu_spx"]],
            b.loc[b.fit_origin.eq(first), ["mu_qqq", "mu_spx"]],
        )


if __name__ == "__main__":
    unittest.main()
