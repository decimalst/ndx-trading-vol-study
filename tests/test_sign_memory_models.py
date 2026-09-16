"""Prewritten synthetic contracts for staged sign-agreement probabilities."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src import sign_memory_models as model
from src.joint_risk_features import ALL_FEATURES as OLD

BOUNDED = ("qqq_pos22", "qqq_neg22", "spx_pos22", "spx_neg22", "independent22")
ALL = (*OLD, *BOUNDED, "excess22")
MODELS = ("frequency", "baseline", "memory")


def sample(n=480):
    rng = np.random.default_rng(20260918)
    dates = pd.bdate_range("2014-01-02", periods=n)
    f = pd.DataFrame(rng.normal(size=(n, len(ALL))), index=dates, columns=ALL)
    f["const"] = 1.0
    f["corr22"] = rng.uniform(-0.85, 0.85, n)
    f.loc[:, BOUNDED] = rng.uniform(0, 1, (n, len(BOUNDED)))
    f["excess22"] = rng.uniform(-0.45, 0.45, n)
    f["feature_cutoff_date"] = pd.Series(dates, index=dates).shift(1)
    # Balanced labels guarantee the fixed 50-per-class support contract.
    y = np.tile([0.0, 1.0], n // 2 + 1)[:n]
    rng.shuffle(y)
    t = pd.DataFrame({"y": y}, index=dates)
    t["target_end"] = pd.Series(dates, index=dates).shift(-1)
    t["available_date"] = t["target_end"]
    t.loc[dates[-1], "y"] = np.nan
    return f, t


def config(dates):
    return {
        "models": list(MODELS),
        "all_features": list(ALL),
        "minimum_train": 120,
        "minimum_train_per_class": 50,
        "origin_start": str(dates[220].date()),
        "origin_end": str(dates[-2].date()),
        "latest_target": str(dates[-1].date()),
        "development": [str(dates[220].date()), str(dates[330].date())],
        "development_target_available_by": str(dates[330].date()),
        "evaluation": [str(dates[331].date()), str(dates[-2].date())],
    }


def fake_fit(train, targets, query):
    value = float(targets.y.mean())
    return {
        "forecasts": {name: np.full(len(query), value) for name in MODELS},
        "audit": {"train_n": len(train), "application_n": len(query)},
    }


class SignMemoryModels(unittest.TestCase):
    def test_analytic_gradient_and_hessian(self):
        rng = np.random.default_rng(81)
        x = np.c_[np.ones(120), rng.normal(size=(120, 4))]
        y = np.tile([0.0, 1.0], 60)
        beta = np.array([0.3, -0.2, 0.1, 0.5, -0.4])
        value, gradient, hessian = model.logistic_state(beta, x, y)
        eps = 1e-5
        numerical_g, numerical_h = [], []
        for j in range(len(beta)):
            shift = np.eye(len(beta))[j] * eps
            vp, gp, _ = model.logistic_state(beta + shift, x, y)
            vm, gm, _ = model.logistic_state(beta - shift, x, y)
            numerical_g.append((vp - vm) / (2 * eps))
            numerical_h.append((gp - gm) / (2 * eps))
        self.assertTrue(np.isfinite(value))
        np.testing.assert_allclose(gradient, numerical_g, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(hessian, np.array(numerical_h).T, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(hessian).min(), 0)

    def test_stable_extreme_logits_and_unpenalized_intercept(self):
        x = np.ones((2, 1))
        for value in [-1000.0, 1000.0]:
            objective, gradient, hessian = model.logistic_state(
                np.array([value]), x, np.array([0.0, 1.0])
            )
            self.assertEqual(objective, 500.0)
            self.assertEqual(gradient[0], np.sign(value) * 0.5)
            self.assertEqual(hessian[0, 0], 0.0)
        # Small but representable tail derivatives must survive expit rounding to 1.
        _, gradient, hessian = model.logistic_state(np.array([40.0]), x, np.ones(2))
        self.assertLess(gradient[0], 0)
        self.assertGreater(hessian[0, 0], 0)

    def test_memory_offset_derivative_penalizes_its_only_slope(self):
        m = np.linspace(-0.4, 0.4, 120)[:, None]
        y = np.tile([0.0, 1.0], 60)
        offset = np.linspace(-3, 3, 120)
        b = np.array([0.7])
        value, gradient, hessian = model.logistic_state(
            b, m, y, offset=offset, penalize_intercept=True
        )
        eta = offset + m[:, 0] * b[0]
        self.assertAlmostEqual(
            value, np.logaddexp(0, (1 - 2 * y) * eta).mean() + 0.01 * b[0] ** 2
        )
        self.assertAlmostEqual(gradient[0], np.mean((expit(eta) - y) * m[:, 0]) + 0.02 * b[0])
        self.assertAlmostEqual(
            hessian[0, 0], np.mean(expit(eta) * expit(-eta) * m[:, 0] ** 2) + 0.02
        )

    def test_transform_uses_training_only_and_fixed_bounded_scales(self):
        f, _ = sample()
        train, query = f.iloc[:300], f.iloc[300:310].copy()
        x, app, audit = model.transform(train, query)
        self.assertEqual(tuple(x.columns), (*OLD, "corr22_centered_sq", *BOUNDED))
        self.assertEqual(audit["scales"][-5:], [1.0] * 5)
        center = float(train.corr22.mean())
        self.assertEqual(audit["corr22_mean"], center)
        raw = (train.corr22 - center) ** 2
        np.testing.assert_allclose(
            x["corr22_centered_sq"], (raw - raw.mean()) / raw.std(ddof=0)
        )
        np.testing.assert_allclose(app["qqq_pos22"], query.qqq_pos22 - train.qqq_pos22.mean())
        query.loc[:, ALL[1:]] *= 100
        other, _, changed = model.transform(train, query)
        pd.testing.assert_frame_equal(x, other)
        self.assertEqual(audit, changed)

    def test_exact_constant_nonzero_bounded_columns_are_retained(self):
        f, t = sample()
        f["qqq_pos22"] = 0.1
        out = model.fit_predict(f.iloc[:300], t.iloc[:300], f.iloc[300:310])
        a = out["audit"]["baseline"]
        column = a["columns"].index("qqq_pos22")
        self.assertEqual(a["beta"][column], 0.0)
        self.assertEqual(a["means"][column], 0.1)
        self.assertEqual(a["scales"][column], 1.0)
        self.assertIn("qqq_pos22", out["audit"]["transform"]["bounded_constant_columns"])

    def test_exact_constant_memory_nests_without_dropping_arm(self):
        f, t = sample()
        f.loc[f.index[:300], "excess22"] = 0.1
        out = model.fit_predict(f.iloc[:300], t.iloc[:300], f.iloc[300:310])
        self.assertEqual(set(out["forecasts"]), set(MODELS))
        np.testing.assert_array_equal(out["forecasts"]["baseline"], out["forecasts"]["memory"])
        a = out["audit"]["memory"]
        self.assertEqual(a["b"], 0.0)
        self.assertEqual(a["status"], "EXACT_CONSTANT_INPUT")
        self.assertEqual(a["gradient_max_abs"], 0.0)

    def test_solver_matches_independent_bfgs_and_json_replay(self):
        f, t = sample()
        train, query, y = f.iloc[:300], f.iloc[300:325], t.y.iloc[:300].to_numpy()
        out = model.fit_predict(train, t.iloc[:300], query)
        x, app, _ = model.transform(train, query)
        z = x.to_numpy()

        def objective(beta):
            eta = z @ beta
            error = expit(eta) - y
            penalty = beta.copy()
            penalty[0] = 0.0
            return np.logaddexp(0, (1 - 2 * y) * eta).mean() + 0.01 * (
                penalty @ penalty
            ), z.T @ error / len(y) + 0.02 * penalty

        other = minimize(
            objective,
            np.zeros(z.shape[1]),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-10, "maxiter": 1000},
        )
        a = out["audit"]["baseline"]
        np.testing.assert_allclose(a["beta"], other.x, atol=1e-6, rtol=1e-7)
        np.testing.assert_allclose(
            out["forecasts"]["baseline"],
            expit(app.to_numpy() @ np.array(a["beta"])),
            atol=0,
            rtol=0,
        )
        memory = out["audit"]["memory"]
        m = query.excess22.to_numpy() - memory["mean"]
        np.testing.assert_array_equal(
            out["forecasts"]["memory"],
            expit(app.to_numpy() @ np.array(a["beta"]) + memory["b"] * m),
        )
        self.assertLessEqual(a["gradient_max_abs"], 1e-8)
        self.assertLessEqual(memory["gradient_max_abs"], 1e-8)
        self.assertEqual(out["audit"]["frequency"]["probability"], y.mean())
        json.dumps(out["audit"], allow_nan=False)

    def test_query_mutation_does_not_refit_baseline_or_memory(self):
        f, t = sample()
        first = model.fit_predict(f.iloc[:300], t.iloc[:300], f.iloc[300:310])
        query = f.iloc[300:310].copy()
        query["corr22"] += 20
        query["excess22"] -= 5
        second = model.fit_predict(f.iloc[:300], t.iloc[:300], query)
        self.assertEqual(first["audit"], second["audit"])
        self.assertFalse(
            np.array_equal(first["forecasts"]["memory"], second["forecasts"]["memory"])
        )

    def test_binary_complement_symmetry(self):
        f, t = sample()
        original = model.fit_predict(f.iloc[:300], t.iloc[:300], f.iloc[300:310])
        altered = t.iloc[:300].copy()
        altered["y"] = 1 - altered.y
        complement = model.fit_predict(f.iloc[:300], altered, f.iloc[300:310])
        for name in MODELS:
            np.testing.assert_allclose(
                complement["forecasts"][name], 1 - original["forecasts"][name], atol=1e-12
            )
        np.testing.assert_allclose(
            complement["audit"]["baseline"]["beta"],
            -np.array(original["audit"]["baseline"]["beta"]),
            atol=1e-12,
        )

    def test_empirical_feature_units_invariance(self):
        f, t = sample()
        original = model.fit_predict(f.iloc[:300], t.iloc[:300], f.iloc[300:310])
        changed = f.copy()
        changed.loc[:, OLD[1:]] = 3 * changed.loc[:, OLD[1:]] + 7
        altered = model.fit_predict(changed.iloc[:300], t.iloc[:300], changed.iloc[300:310])
        for name in MODELS:
            np.testing.assert_allclose(
                original["forecasts"][name], altered["forecasts"][name], atol=1e-12
            )

    def test_bad_labels_and_support_are_failures(self):
        for replacement in [0.5, np.inf, np.nan, 0.0]:
            f, t = sample()
            train = t.iloc[:300].copy()
            if replacement == 0.0:
                train["y"] = 0.0
            else:
                train.iloc[0, 0] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                model.fit_predict(f.iloc[:300], train, f.iloc[300:310])

    def test_nonfinite_and_zero_scale_existing_features_rejected(self):
        for value in [0.1, np.inf, np.nan]:
            f, t = sample()
            f["qqq_lg_d"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                model.fit_predict(f.iloc[:300], t.iloc[:300], f.iloc[300:310])

    def test_solver_failure_is_not_retried(self):
        f, t = sample()
        with (
            patch.object(
                model.np.linalg, "solve", side_effect=np.linalg.LinAlgError("synthetic")
            ) as solve,
            self.assertRaisesRegex(ValueError, "Newton"),
        ):
            model.fit_predict(f.iloc[:300], t.iloc[:300], f.iloc[300:310])
        self.assertEqual(solve.call_count, 1)

    def test_iteration_budget_is_fixed(self):
        f, t = sample()
        with (
            patch.object(model, "MAX_ITERATIONS", 0),
            self.assertRaisesRegex(ValueError, "converge"),
        ):
            model.fit_predict(f.iloc[:300], t.iloc[:300], f.iloc[300:310])

    def test_training_maturity_common_rows_and_zero_class(self):
        f, t = sample()
        mask = model.training_mask(f, t, f.index[220], 120)
        self.assertTrue(mask.iloc[218])
        self.assertFalse(mask.iloc[219])
        zero = t.index[(t.y == 0) & mask][0]
        self.assertTrue(mask.loc[zero])
        t.loc[f.index[50], "y"] = np.nan
        f.loc[f.index[51], "excess22"] = np.nan
        after = model.training_mask(f, t, f.index[220], 120)
        self.assertEqual(mask.sum() - after.sum(), 2)

    def test_full_calendar_maturity_and_alignment_required(self):
        for fault in ["cutoff", "target", "available", "order", "nonbinary"]:
            f, t = sample()
            if fault == "cutoff":
                f.loc[f.index[50], "feature_cutoff_date"] = f.index[48]
            elif fault == "order":
                t = t.iloc[::-1]
            elif fault == "nonbinary":
                t.loc[f.index[50], "y"] = 2
            else:
                t.loc[f.index[50], "target_end" if fault == "target" else "available_date"] = (
                    f.index[52]
                )
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                model.training_mask(f, t, f.index[220], 120)

    def test_training_minimum_and_class_support_not_relaxed(self):
        f, t = sample()
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            model.training_mask(f, t, f.index[220], 1000)
        t.loc[f.index[:219], "y"] = 1.0
        t.loc[f.index[:49], "y"] = 0.0
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            model.training_mask(f, t, f.index[220], 120)

    def test_monthly_origin_selected_before_future_query_label(self):
        f, t = sample()
        cfg = config(f.index)
        with patch.object(model, "fit_predict", side_effect=fake_fit):
            _, before = model.forecast_panel(f, t, cfg)
            first = pd.Timestamp(before[0]["fit_origin"])
            t.loc[first, "y"] = np.nan
            panel, after = model.forecast_panel(f, t, cfg)
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[0]["fit_origin"], str(first.date()))
        self.assertNotIn(first, panel.origin.to_list())

    def test_entire_unscored_month_keeps_its_fit_audit(self):
        f, t = sample()
        cfg = config(f.index)
        month = f.index[220].to_period("M")
        t.loc[f.index.to_period("M") == month, "y"] = np.nan
        with patch.object(model, "fit_predict", side_effect=fake_fit):
            panel, fits = model.forecast_panel(f, t, cfg)
        self.assertEqual(pd.Timestamp(fits[0]["fit_origin"]).to_period("M"), month)
        self.assertGreater(fits[0]["application_n"], 0)
        self.assertFalse((panel.origin.dt.to_period("M") == month).any())

    def test_invalid_unscored_month_forecasts_cannot_bypass_validation(self):
        f, t = sample()
        cfg = config(f.index)
        month = f.index[220].to_period("M")
        t.loc[f.index.to_period("M") == month, "y"] = np.nan

        def invalid_first_month(train, labels, query):
            result = fake_fit(train, labels, query)
            if query.index[0].to_period("M") == month:
                result["forecasts"]["memory"][:] = np.nan
            return result

        with (
            patch.object(model, "fit_predict", side_effect=invalid_first_month),
            self.assertRaises(ValueError),
        ):
            model.forecast_panel(f, t, cfg)

    def test_development_boundary_and_all_three_models_exactly_paired(self):
        f, t = sample()
        cfg = config(f.index)
        with patch.object(model, "fit_predict", side_effect=fake_fit):
            panel, _ = model.forecast_panel(f, t, cfg)
        self.assertEqual(tuple(panel.columns), model.PANEL_COLUMNS)
        self.assertTrue(panel.groupby("origin").size().eq(3).all())
        self.assertFalse((panel.origin == pd.Timestamp(cfg["development"][1])).any())
        self.assertTrue(
            (
                panel.loc[panel.phase == "development", "available_date"]
                <= pd.Timestamp(cfg["development"][1])
            ).all()
        )
        np.testing.assert_array_equal(panel.loss, (panel.probability - panel.y) ** 2)
        model.validate_panel(panel)

    def test_panel_rejects_missing_duplicate_or_unshared_metadata(self):
        f, t = sample()
        with patch.object(model, "fit_predict", side_effect=fake_fit):
            original, _ = model.forecast_panel(f, t, config(f.index))
        for fault in ["missing", "duplicate", "label", "fit", "loss", "probability", "target"]:
            panel = original.copy()
            if fault == "missing":
                panel = panel.iloc[1:]
            elif fault == "duplicate":
                panel = pd.concat([panel, panel.iloc[[0]]])
            elif fault == "label":
                panel.loc[0, "y"] = 1 - panel.loc[0, "y"]
            elif fault == "fit":
                panel.loc[0, "train_n"] += 1
            elif fault == "loss":
                panel.loc[0, "loss"] += 0.01
            elif fault == "target":
                panel.loc[0, "target_end"] += pd.Timedelta(days=1)
            else:
                panel.loc[0, "probability"] = -1e-9
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                model.validate_panel(panel)

    def test_brier_endpoints_and_nonzero_underflow(self):
        np.testing.assert_array_equal(
            model.brier_loss(np.array([0.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0])), [0, 0, 1]
        )
        with self.assertRaises(ValueError):
            model.brier_loss(np.array([1e-200]), np.array([0.0]))

    def test_panel_rejects_monthly_fit_or_cutoff_drift(self):
        f, t = sample()
        with patch.object(model, "fit_predict", side_effect=fake_fit):
            original, _ = model.forecast_panel(f, t, config(f.index))
        first_origin = original.origin.iloc[0]
        cohort = original.origin.eq(first_origin)
        for fault in ["train_n", "fit_month", "feature_cutoff"]:
            panel = original.copy()
            if fault == "train_n":
                panel.loc[cohort, "train_n"] += 1
            elif fault == "fit_month":
                previous_month = first_origin.to_period("M").start_time - pd.Timedelta(days=1)
                panel.loc[cohort, "fit_origin"] = previous_month
                for column in ["fit_cutoff_date", "train_last_target", "train_last_available"]:
                    panel.loc[cohort, column] = previous_month - pd.Timedelta(days=1)
            else:
                panel.loc[cohort, "feature_cutoff_date"] = panel.loc[
                    cohort, "fit_cutoff_date"
                ] - pd.Timedelta(days=1)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                model.validate_panel(panel)

    def test_full_synthetic_monthly_pipeline(self):
        f, t = sample(320)
        cfg = config(sample()[0].index)
        dates = f.index
        cfg.update(
            origin_start=str(dates[220].date()),
            origin_end=str(dates[-2].date()),
            latest_target=str(dates[-1].date()),
            development=[str(dates[220].date()), str(dates[265].date())],
            development_target_available_by=str(dates[265].date()),
            evaluation=[str(dates[266].date()), str(dates[-2].date())],
        )
        panel, fits = model.forecast_panel(f, t, cfg)
        self.assertGreater(len(fits), 1)
        self.assertTrue(panel.probability.between(0, 1).all())
        for fit in fits:
            self.assertGreaterEqual(fit["model_audit"]["support"]["events"], 50)
            self.assertGreaterEqual(fit["model_audit"]["support"]["nonevents"], 50)
            self.assertLessEqual(fit["model_audit"]["baseline"]["gradient_max_abs"], 1e-8)
            self.assertLessEqual(fit["model_audit"]["memory"]["gradient_max_abs"], 1e-8)


if __name__ == "__main__":
    unittest.main()
