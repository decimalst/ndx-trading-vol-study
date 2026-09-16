"""Prewritten generated contracts for four matched return-model arms."""
from __future__ import annotations

import json
import math
import unittest

import numpy as np
import pandas as pd

from src import peak_age_features as feature
from src import peak_age_models as model


def design(n=1100, seed=61, start="2001-01-02"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n, name="date")
    f = pd.DataFrame(rng.normal(size=(n, len(feature.RAW))), index=dates, columns=feature.RAW)
    f["const"], f["I"], f["R"] = 1., f.I - 3, f.R - 4
    for weekday in range(1, 5):
        f[f"entry_dow_{weekday}"] = (dates.weekday == weekday).astype(float)
    f["peak_age"] = rng.integers(0, 252, n) / 251
    f["drawdown"] = rng.uniform(.001, .4, n)
    f["drawdown_sq"] = f.drawdown**2
    f["window_return"] = rng.normal(.04, .2, n)
    return f.loc[:, feature.COMMON]


def target(train):
    return pd.Series(.02 + .04 * train.I + .03 * np.sin(train.R)
                     + .05 * train.peak_age, index=train.index, name="y")


def independent_ridge(train, y, apply, columns):
    tr, ap = train.copy(), apply.copy()
    for name in ["I", "R"]:
        center = float(train[name].mean())
        tr[name + "_square"] = (train[name] - center)**2
        ap[name + "_square"] = (apply[name] - center)**2
    a, b = tr[list(columns[1:])].to_numpy(), ap[list(columns[1:])].to_numpy()
    means, scales = a.mean(axis=0), a.std(axis=0, ddof=0)
    z, query = (a - means) / scales, (b - means) / scales
    matrix = np.vstack([z, np.sqrt(len(z) * .01) * np.eye(z.shape[1])])
    rhs = np.r_[np.asarray(y) - np.mean(y), np.zeros(z.shape[1])]
    coefficient = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    return np.mean(y) + query @ coefficient, np.mean(y) + z @ coefficient, coefficient, means, scales, z


class PeakAgeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = design()
        cls.apply = design(30, 73, "2020-01-02")
        cls.y = target(cls.train)

    def test_fixed_four_arms_16_19_slopes_and_training_only_curvature(self):
        self.assertEqual(model.MODELS, ("mean", "baseline", "depth", "peak_age"))
        self.assertEqual(model.BASE, feature.BASE)
        self.assertEqual(model.DEPTH, feature.BASE + ("drawdown", "drawdown_sq", "window_return"))
        tr, ap, audit = model.transform(self.train, self.apply)
        self.assertEqual(tuple(tr), model.DEPTH)
        for name in ["I", "R"]:
            self.assertEqual(audit[name + "_mean"], self.train[name].mean())
            np.testing.assert_allclose(tr[name + "_square"], (self.train[name] - audit[name + "_mean"])**2)
            np.testing.assert_allclose(ap[name + "_square"], (self.apply[name] - audit[name + "_mean"])**2)
        changed = self.apply.copy()
        changed["I"] += 100
        altered, _, changed_audit = model.transform(self.train, changed)
        pd.testing.assert_frame_equal(tr, altered)
        self.assertEqual(audit, changed_audit)

    def test_all_predictions_ridge_coefficients_objectives_and_full_gradients(self):
        result = model.fit_predict(self.train, self.y, self.apply)
        self.assertEqual(set(result), {"predictions", "model_audit", "transform_audit", "scalar_audit"})
        self.assertEqual(tuple(result["predictions"]), model.MODELS)
        np.testing.assert_array_equal(result["predictions"]["mean"], np.full(len(self.apply), self.y.mean()))
        for name, columns in [("baseline", model.BASE), ("depth", model.DEPTH)]:
            expected, fitted, beta, means, scales, z = independent_ridge(self.train, self.y, self.apply, columns)
            audit = result["model_audit"][name]
            self.assertEqual(audit["columns"], list(columns))
            np.testing.assert_allclose(result["predictions"][name], expected, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(audit["beta"], np.r_[self.y.mean(), beta], rtol=1e-7, atol=1e-12)
            np.testing.assert_array_equal(audit["means"], np.r_[0., means])
            np.testing.assert_array_equal(audit["scales"], np.r_[1., scales])
            self.assertEqual(audit["alpha"], .01)
            residual = fitted - self.y.to_numpy()
            gradient = np.r_[residual.mean(), z.T @ residual / len(z) + .01 * beta]
            np.testing.assert_allclose(audit["gradient"], gradient, atol=1e-12)
            self.assertLessEqual(audit["gradient_max_abs"], 1e-10)
            self.assertAlmostEqual(audit["objective"], np.mean(residual**2) + .01 * np.sum(beta**2), places=14)
        baseline, depth = result["model_audit"]["baseline"], result["model_audit"]["depth"]
        np.testing.assert_array_equal(baseline["means"], depth["means"][:17])
        np.testing.assert_array_equal(baseline["scales"], depth["scales"][:17])

    def test_scalar_matches_independent_conditional_augmented_least_squares(self):
        result = model.fit_predict(self.train, self.y, self.apply)
        _, fitted, _, _, _, _ = independent_ridge(self.train, self.y, self.apply, model.DEPTH)
        center = math.fsum(self.train.peak_age) / len(self.train)
        a = self.train.peak_age.to_numpy() - center
        residual = self.y.to_numpy() - fitted
        b = np.linalg.lstsq(np.r_[a, np.sqrt(len(a) * .01)][:, None], np.r_[residual, 0.], rcond=None)[0][0]
        audit = result["scalar_audit"]
        self.assertEqual(audit["age_center"], center)
        self.assertEqual(audit["age_scale"], 1.)
        self.assertAlmostEqual(audit["coefficient"], b, places=12)
        expected = result["predictions"]["depth"] + b * (self.apply.peak_age.to_numpy() - center)
        np.testing.assert_allclose(result["predictions"]["peak_age"], expected, rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(audit["objective"], np.mean((fitted + b * a - self.y)**2) + .01 * b*b, places=14)
        gradient = np.mean(a * (fitted + b * a - self.y)) + .01 * b
        self.assertAlmostEqual(audit["gradient"], gradient, places=12)
        self.assertLessEqual(abs(gradient), 1e-10)

    def test_exact_constant_age_copies_depth_even_for_changed_query_ages(self):
        train, apply = self.train.copy(), self.apply.copy()
        train["peak_age"] = 13 / 251
        apply["peak_age"] = np.linspace(0, 1, len(apply))
        result = model.fit_predict(train, self.y, apply)
        self.assertEqual(result["scalar_audit"]["status"], "EXACT_CONSTANT_INPUT")
        self.assertEqual(result["scalar_audit"]["age_center"], 13 / 251)
        self.assertEqual(result["scalar_audit"]["coefficient"], 0.)
        np.testing.assert_array_equal(result["predictions"]["depth"], result["predictions"]["peak_age"])

    def test_balanced_zero_numerator_and_both_coefficient_signs(self):
        age = np.array([0., 1., 0., 1.])
        for residual, sign in [(np.ones(4), 0), (np.array([-1., 1., -1., 1.]), 1),
                               (np.array([1., -1., 1., -1.]), -1)]:
            beta, audit = model.fit_scalar(age, residual, np.zeros(4))
            self.assertEqual(np.sign(beta), sign)
            if sign == 0:
                self.assertEqual(audit["status"], "BALANCED_AT_ZERO")
                self.assertEqual(beta, 0.)

    def test_scalar_finite_difference_half_gradient(self):
        age = np.linspace(0, 1, 31)
        offset, y = np.sin(age), np.cos(age)
        b = .07
        center = math.fsum(age) / len(age)
        a = age - center
        objective, gradient = model.scalar_objective(b, a, y, offset)
        epsilon = 1e-6
        plus = model.scalar_objective(b + epsilon, a, y, offset)[0]
        minus = model.scalar_objective(b - epsilon, a, y, offset)[0]
        self.assertAlmostEqual(gradient, (plus - minus) / (4 * epsilon), places=9)
        self.assertGreater(objective, 0.)

    def test_ridge_span_counterexample_is_accepted_without_orthogonality_claim(self):
        train = self.train.copy()
        train["peak_age"] = (np.arange(len(train)) % 252) / 251
        train["ret_d"] = train.peak_age
        apply = self.apply.copy()
        apply["ret_d"] = apply.peak_age
        y = pd.Series(train.peak_age.to_numpy(), index=train.index)
        result = model.fit_predict(train, y, apply)
        self.assertGreater(result["scalar_audit"]["coefficient"], 1e-5)
        self.assertGreater(np.max(np.abs(result["predictions"]["peak_age"] - result["predictions"]["depth"])), 1e-7)
        columns = result["model_audit"]["depth"]["columns"]
        self.assertIn("ret_d", columns)
        self.assertNotIn("peak_age", columns)

    def test_common_complete_cohort_minimum1000_and_exact_target_order(self):
        for n in [999, 1000]:
            train = self.train.iloc[:n]
            if n == 999:
                with self.assertRaisesRegex(ValueError, "1000|INSUFFICIENT"):
                    model.fit_predict(train, self.y.iloc[:n], self.apply)
            else:
                result = model.fit_predict(train, self.y.iloc[:n], self.apply)
                self.assertTrue(all(audit["train_n"] == n for audit in result["model_audit"].values()))
                self.assertEqual(result["scalar_audit"]["train_n"], n)
        with self.assertRaises(ValueError):
            model.fit_predict(self.train, self.y.iloc[::-1], self.apply)
        changed = self.train.copy()
        changed.iloc[10, changed.columns.get_loc("peak_age")] = np.nan
        with self.assertRaises(ValueError):
            model.fit_predict(changed, self.y, self.apply)

    def test_all_declared_nuisance_scales_fail_closed_but_no_unused_hinge(self):
        for name in ["entry_dow_4", "drawdown", "drawdown_sq", "window_return"]:
            changed = self.train.copy()
            changed[name] = 0.
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "scale"):
                model.fit_predict(changed, self.y, self.apply)
        changed, apply = self.train.copy(), self.apply.copy()
        changed["R"], apply["R"] = changed.I - 1, apply.I - 1
        result = model.fit_predict(changed, self.y, apply)
        self.assertEqual(len(result["predictions"]), 4)
        self.assertNotIn("hinge", result["model_audit"])

    def test_nonfinite_curvature_and_nonzero_scalar_product_underflow_fail(self):
        changed = self.train.copy()
        changed.iloc[0, changed.columns.get_loc("I")] = 1e200
        with self.assertRaises(ValueError):
            model.fit_predict(changed, self.y, self.apply)
        with self.assertRaises(ValueError):
            model.fit_scalar(np.array([0., 1e-300]), np.array([1e-300, -1e-300]), np.zeros(2))

    def test_query_changes_leave_all_training_audits_fixed_and_no_joint_refit(self):
        before = model.fit_predict(self.train, self.y, self.apply)
        changed = self.apply.copy()
        changed["I"] += 1
        changed["peak_age"] = 1 - changed.peak_age
        after = model.fit_predict(self.train, self.y, changed)
        for key in ["model_audit", "transform_audit", "scalar_audit"]:
            self.assertEqual(before[key], after[key])
        self.assertFalse(np.array_equal(before["predictions"]["peak_age"], after["predictions"]["peak_age"]))

    def test_generated_ms_us_ns_dates_and_json_audits_replay_numerically(self):
        expected = model.fit_predict(self.train, self.y, self.apply)
        for unit in ["ms", "us", "ns"]:
            train, apply, y = self.train.copy(), self.apply.copy(), self.y.copy()
            train.index, y.index = train.index.as_unit(unit), y.index.as_unit(unit)
            apply.index = apply.index.as_unit(unit)
            actual = model.fit_predict(train, y, apply)
            for name in model.MODELS:
                np.testing.assert_allclose(actual["predictions"][name], expected["predictions"][name], atol=1e-12)
            for key in ["model_audit", "transform_audit", "scalar_audit"]:
                self.assertEqual(json.loads(json.dumps(actual[key], allow_nan=False)), actual[key])


if __name__ == "__main__":
    unittest.main()
