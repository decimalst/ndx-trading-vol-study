"""Prewritten mixed-geometry, convex quarter increment and chronology tests."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from src import civil_quarter_features as features
from src import civil_quarter_models as model


def sample(n=1500):
    rng = np.random.default_rng(20260921)
    dates = pd.bdate_range("2010-01-04", periods=n)
    old = pd.DataFrame(
        rng.normal(size=(n, len(features.OLD_FEATURES))),
        index=dates,
        columns=features.OLD_FEATURES,
    )
    old["const"] = 1.0
    old["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    f = features.augment_features(old)
    y = np.exp(-10 + 0.2 * f.lrv_d + 0.35 * f.quarter_end5 + rng.normal(0, 0.3, n))
    future = pd.Series(dates, index=dates).shift(-1)
    t = pd.DataFrame({"y": y, "target_end": future, "available_date": future}, index=dates)
    t.loc[dates[-1], "y"] = np.nan
    return f, t


def config(dates):
    return {
        "models": list(features.MODELS),
        "baseline": list(features.BASE),
        "all_features": list(features.ALL_FEATURES),
        "minimum_train": 1000,
        "origin_start": str(dates[1300].date()),
        "origin_end": str(dates[-2].date()),
        "latest_target": str(dates[-1].date()),
        "development": [str(dates[1300].date()), str(dates[1400].date())],
        "development_target_available_by": str(dates[1400].date()),
        "evaluation": [str(dates[1401].date()), str(dates[-2].date())],
    }


class CivilQuarterModels(unittest.TestCase):
    def test_positive_objective_gradient_hessian_against_independent_differences(self):
        rng = np.random.default_rng(15)
        x = np.c_[np.ones(120), rng.normal(size=(120, 4))]
        y = np.exp(rng.normal(0, 0.3, 120))
        beta = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
        value, gradient, hessian = model.objective(beta, x, y)
        self.assertAlmostEqual(
            value, np.mean(x @ beta + y * np.exp(-x @ beta)) + 0.01 * np.sum(beta[1:] ** 2)
        )
        for index in range(len(beta)):
            step = np.eye(len(beta))[index] * 1e-5
            plus = model.objective(beta + step, x, y)
            minus = model.objective(beta - step, x, y)
            self.assertAlmostEqual(gradient[index], (plus[0] - minus[0]) / 2e-5, delta=1e-9)
            np.testing.assert_allclose(
                hessian[:, index], (plus[1] - minus[1]) / 2e-5, rtol=1e-8, atol=1e-9
            )
        self.assertGreater(np.linalg.eigvalsh(hessian).min(), 0)

    def test_scalar_derivatives_strict_curvature_and_zero_nesting(self):
        eta = np.linspace(-0.2, 0.4, 100)
        y = np.linspace(0.5, 1.5, 100)
        z = (np.arange(100) % 4 == 0).astype(float) - 0.25
        b = 0.3
        value, g, h = model.quarter_objective(b, eta, y, z)
        plus, minus = (
            model.quarter_objective(b + 1e-5, eta, y, z),
            model.quarter_objective(b - 1e-5, eta, y, z),
        )
        self.assertAlmostEqual(g, (plus[0] - minus[0]) / 2e-5, delta=1e-9)
        self.assertAlmostEqual(h, (plus[1] - minus[1]) / 2e-5, delta=1e-9)
        self.assertGreaterEqual(h, 0.02)
        self.assertAlmostEqual(
            model.quarter_objective(0.0, eta, y, z)[0], np.mean(eta + y * np.exp(-eta))
        )
        self.assertTrue(np.isfinite(value))

    def test_log_ratio_arithmetic_avoids_avoidable_intermediate_overflow(self):
        value, gradient, _ = model.objective(
            np.array([-730.0]), np.ones((2, 1)), np.full(2, 1e-300)
        )
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(gradient).all())
        for eta in [-1000.0, 1000.0]:
            with self.subTest(eta=eta), self.assertRaises(ValueError):
                model.objective(np.array([eta]), np.ones((2, 1)), np.ones(2))

    def test_train_only_mixed_geometry_and_exact_quarter_center(self):
        f, _ = sample()
        x, query, z, qz, audit = model.transform(f.iloc[:1200], f.iloc[1200:1210])
        self.assertEqual(tuple(x.columns), features.BASE)
        self.assertEqual(audit["means"][0], 0.0)
        self.assertEqual(audit["scales"][0], 1.0)
        self.assertEqual(audit["scales"][-13:], [1.0] * 13)
        np.testing.assert_allclose(x.iloc[:, 1:18].std(ddof=0), 1, atol=1e-14)
        np.testing.assert_allclose(
            x.month_end5, f.month_end5.iloc[:1200] - f.month_end5.iloc[:1200].mean()
        )
        self.assertEqual(audit["quarter_mean"], f.quarter_end5.iloc[:1200].mean())
        np.testing.assert_array_equal(
            qz, f.quarter_end5.iloc[1200:1210] - audit["quarter_mean"]
        )
        changed = f.iloc[1200:1210].copy()
        changed.loc[:, features.OLD_FEATURES[1:]] *= 100
        xx, _, zz, _, other = model.transform(f.iloc[:1200], changed)
        pd.testing.assert_frame_equal(x, xx)
        np.testing.assert_array_equal(z, zz)
        self.assertEqual(audit, other)
        self.assertTrue(np.isfinite(query).all().all())

    def test_candidate_identification_is_conditional_on_all_baseline_controls(self):
        f, t = sample()
        f["lrv_d"] = f.quarter_end5
        features.civil_support(f.iloc[:1200])
        features.require_civil_rank(f.iloc[:1200])
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            model.fit_predict(f.iloc[:1200], t.y.iloc[:1200], f.iloc[1200:1210])

    def test_market_collinearity_alone_is_not_an_extra_rank_gate(self):
        f, t = sample()
        f["lrv_w"] = f.lrv_d
        forecasts, audits = model.fit_predict(
            f.iloc[:1200], t.y.iloc[:1200], f.iloc[1200:1210]
        )
        self.assertLess(audits["quarter"]["identification"]["baseline_rank"], 31)
        self.assertGreater(audits["quarter"]["identification"]["relative_residual_norm"], 1e-8)
        self.assertTrue(np.isfinite(forecasts["quarter"]).all())

    def test_scientific_support_and_civil_rank_fail_without_dropping_columns(self):
        for fault in ["support", "rank", "old_scale"]:
            f, t = sample()
            if fault == "support":
                f["quarter_end5"] = 0.0
            elif fault == "rank":
                f["quarter_end5"] = f.year_end5
            else:
                f["lvix"] = 0.1
            with (
                self.subTest(fault=fault),
                self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"),
            ):
                model.fit_predict(f.iloc[:1200], t.y.iloc[:1200], f.iloc[1200:1210])

    def test_baseline_and_scalar_match_independent_convex_optimizers(self):
        f, t = sample()
        forecasts, audits = model.fit_predict(
            f.iloc[:1200], t.y.iloc[:1200], f.iloc[1200:1210]
        )
        x, application, z, appz, _ = model.transform(f.iloc[:1200], f.iloc[1200:1210])
        design, y = x.to_numpy(), t.y.iloc[:1200].to_numpy()
        mean = y.mean()
        yn = y / mean

        def objective(beta):
            eta = design @ beta
            ratio = yn * np.exp(-eta)
            penalty = beta.copy()
            penalty[0] = 0.0
            return np.mean(eta + ratio) + 0.01 * (penalty @ penalty), design.T @ (
                1 - ratio
            ) / len(y) + 0.02 * penalty

        fit = minimize(
            objective,
            np.zeros(31),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-10, "maxiter": 1000},
        )
        baseline = audits["baseline"]
        np.testing.assert_allclose(baseline["scaled_beta"], fit.x, rtol=1e-7, atol=1e-6)
        eta = design @ np.array(baseline["scaled_beta"])
        derivative = lambda b: float(np.mean(z * (1 - yn * np.exp(-eta - b * z))) + 0.02 * b)
        quarter = audits["quarter"]
        independent = brentq(derivative, *quarter["bracket"], xtol=1e-12)
        self.assertAlmostEqual(quarter["b"], independent, delta=1e-6)
        self.assertLessEqual(quarter["bracket_gradients"][0], 0)
        self.assertGreaterEqual(quarter["bracket_gradients"][1], 0)
        logmean = np.log(mean)
        appeta = application.to_numpy() @ np.array(baseline["scaled_beta"])
        np.testing.assert_array_equal(forecasts["baseline"], np.exp(logmean + appeta))
        np.testing.assert_array_equal(
            forecasts["quarter"], np.exp(logmean + appeta + quarter["b"] * appz)
        )
        self.assertEqual(forecasts["mean"][0], mean)
        self.assertLessEqual(baseline["gradient_max_abs"], 1e-8)
        self.assertLessEqual(quarter["gradient_max_abs"], 1e-8)
        json.dumps(audits, allow_nan=False)

    def test_constant_target_and_zero_scalar_exactly_nest(self):
        f, _ = sample()
        forecasts, audits = model.fit_predict(
            f.iloc[:1200], np.full(1200, 0.0001), f.iloc[1200:1210]
        )
        self.assertEqual(audits["quarter"]["b"], 0.0)
        np.testing.assert_array_equal(forecasts["baseline"], forecasts["quarter"])
        self.assertEqual(audits["quarter"]["iterations"], 0)

    def test_target_units_and_old_feature_units_do_not_change_normalized_forecasts(self):
        f, t = sample()
        a, _ = model.fit_predict(f.iloc[:1200], t.y.iloc[:1200], f.iloc[1200:1210])
        changed = f.copy()
        changed.loc[:, features.OLD_FEATURES[1:]] = (
            3 * changed.loc[:, features.OLD_FEATURES[1:]] + 2
        )
        b, _ = model.fit_predict(
            changed.iloc[:1200], 100 * t.y.iloc[:1200], changed.iloc[1200:1210]
        )
        for name in features.MODELS:
            np.testing.assert_allclose(a[name], b[name] / 100, rtol=1e-10, atol=0)

    def test_application_changes_do_not_refit_either_stage(self):
        f, t = sample()
        _, a = model.fit_predict(f.iloc[:1200], t.y.iloc[:1200], f.iloc[1200:1210])
        app = f.iloc[1200:1210].copy()
        app["lrv_d"] += 3
        _, b = model.fit_predict(f.iloc[:1200], t.y.iloc[:1200], app)
        self.assertEqual(a, b)

    def test_nonpositive_nonfinite_unaligned_and_unrepresentable_targets_reject(self):
        f, t = sample()
        for fault in ["zero", "negative", "nan", "infinite", "underflow", "order"]:
            y = t.y.iloc[:1200].copy()
            if fault == "order":
                y = y.iloc[::-1]
            else:
                y.iloc[0] = {
                    "zero": 0.0,
                    "negative": -1.0,
                    "nan": np.nan,
                    "infinite": np.inf,
                    "underflow": np.nextafter(0.0, 1.0),
                }[fault]
            if fault == "underflow":
                y.iloc[1:] = 1e300
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                model.fit_predict(f.iloc[:1200], y, f.iloc[1200:1210])

    def test_fixed_optimizer_and_nonfinite_bracket_fail_without_fallback(self):
        f, t = sample()
        with (
            patch.object(model, "MAX_ITERATIONS", 0),
            self.assertRaisesRegex(ValueError, "converge"),
        ):
            model.fit_predict(f.iloc[:1200], t.y.iloc[:1200], f.iloc[1200:1210])
        with (
            patch.object(model.np.linalg, "solve", side_effect=np.linalg.LinAlgError("test")),
            self.assertRaises(ValueError),
        ):
            model.fit_predict(f.iloc[:1200], t.y.iloc[:1200], f.iloc[1200:1210])
        with self.assertRaises(ValueError):
            model.fit_quarter(
                np.full(100, -100.0), np.ones(100), np.r_[np.zeros(50), np.ones(50)] - 0.5
            )

    def test_monthly_mask_is_common_mature_and_zero_targets_invalid(self):
        f, t = sample()
        mask = model.training_mask(f, t, f.index[1300], 1000)
        self.assertTrue(mask.iloc[1298])
        self.assertFalse(mask.iloc[1299])
        f.loc[f.index[200], "quarter_end5"] = np.nan
        self.assertFalse(model.training_mask(f, t, f.index[1300], 1000).iloc[200])
        t.loc[f.index[201], "y"] = 0.0
        with self.assertRaises(ValueError):
            model.training_mask(f, t, f.index[1300], 1000)

    def test_monthly_walkforward_preserves_unscored_queries_and_fits(self):
        f, t = sample()
        cfg = config(f.index)
        first = pd.Timestamp(cfg["origin_start"])
        month = first.to_period("M")
        t.loc[t.index.to_period("M") == month, "y"] = np.nan
        panel, fits = model.forecast_panel(f, t, cfg)
        self.assertEqual(fits[0]["fit_origin"], str(first.date()))
        self.assertGreater(fits[0]["application_n"], 0)
        self.assertFalse((panel.origin.dt.to_period("M") == month).any())
        self.assertEqual(tuple(panel), model.PANEL_COLUMNS)
        self.assertTrue(panel.groupby("origin").size().eq(3).all())
        self.assertTrue(
            (
                panel.loc[panel.phase == "development", "available_date"]
                <= pd.Timestamp(cfg["development"][1])
            ).all()
        )
        for fit in fits:
            self.assertLessEqual(fit["model_audit"]["baseline"]["gradient_max_abs"], 1e-8)
            self.assertLessEqual(fit["model_audit"]["quarter"]["gradient_max_abs"], 1e-8)

    def test_preflight_uses_all_months_support_before_any_optimizer(self):
        f, t = sample(3600)
        cfg = config(f.index)
        cfg["development"][1] = str(f.index[2250].date())
        cfg["development_target_available_by"] = cfg["development"][1]
        cfg["evaluation"][0] = str(f.index[2251].date())
        cfg["evaluation_stability"] = [
            [str(f.index[2251].date()), str(f.index[2900].date())],
            [str(f.index[2901].date()), str(f.index[-2].date())],
        ]
        first_month = f.index[1300].to_period("M")
        t.loc[t.index.to_period("M") == first_month, "y"] = np.nan
        with patch.object(
            model, "fit_predict", side_effect=AssertionError("optimizer called")
        ):
            audit = model.preflight(f, t, cfg)
        self.assertEqual(audit["fits"][0]["fit_origin"], str(f.index[1300].date()))
        self.assertEqual(audit["monthly_fits"], len(audit["fits"]))
        self.assertGreater(audit["common_application_origins"], audit["common_scored_origins"])
        self.assertEqual(len(audit["phases"]), 2)
        self.assertEqual(len(audit["phases"][1]["slices"]), 2)
        for fit in audit["fits"]:
            self.assertEqual(fit["civil_rank"]["rank"], 15)
            self.assertGreater(fit["identification"]["relative_residual_norm"], 1e-8)
        json.dumps(audit, allow_nan=False)
        bad = f.copy()
        bad.loc[bad.index >= f.index[2251], "year_end5"] = 0.0
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            model.preflight(bad, t, cfg)

    def test_validation_rejects_panel_metadata_or_positive_domain_tampering(self):
        f, t = sample()
        panel, _ = model.forecast_panel(f, t, config(f.index))
        self.assertIs(model.validate_panel(panel), panel)
        for fault in [
            "zero",
            "nan",
            "missing_model",
            "duplicate",
            "label",
            "month",
            "cutoff",
            "train_count",
        ]:
            bad = panel.copy()
            if fault == "zero":
                bad.loc[0, "prediction"] = 0.0
            elif fault == "nan":
                bad.loc[0, "prediction"] = np.nan
            elif fault == "missing_model":
                bad = bad.loc[bad.model != "quarter"]
            elif fault == "duplicate":
                bad = pd.concat([bad, bad.iloc[:1]], ignore_index=True)
            elif fault == "label":
                bad.loc[0, "y"] *= 2
            elif fault == "month":
                bad.loc[0, "fit_origin"] -= pd.Timedelta(days=40)
            elif fault == "cutoff":
                bad.loc[0, "fit_cutoff_date"] = bad.loc[0, "origin"]
            else:
                bad["train_n"] = bad.train_n.astype(float)
                bad.loc[0, "train_n"] = 1000.5
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                model.validate_panel(bad)

    def test_missing_predecessor_cutoff_never_enters_training(self):
        f, t = sample()
        self.assertTrue(np.isfinite(f.loc[f.index[0], features.ALL_FEATURES]).all())
        self.assertTrue(pd.isna(f.feature_cutoff_date.iloc[0]))
        admitted = model.training_mask(f, t, f.index[1300], 1000)
        self.assertFalse(admitted.iloc[0])
        self.assertTrue(admitted.iloc[1])
        self.assertTrue(admitted.iloc[1298])

    def test_invalid_forecast_in_wholly_unscored_month_is_not_hidden(self):
        f, t = sample()
        cfg = config(f.index)
        first_month = f.index[1300].to_period("M")
        t.loc[t.index.to_period("M") == first_month, "y"] = np.nan
        original = model.fit_predict

        def invalid(train, y, application):
            predictions, audit = original(train, y, application)
            predictions["quarter"][:] = np.nan
            return predictions, audit

        with (
            patch.object(model, "fit_predict", side_effect=invalid),
            self.assertRaises(ValueError),
        ):
            model.forecast_panel(f, t, cfg)


if __name__ == "__main__":
    unittest.main()
