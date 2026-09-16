"""Synthetic staged mean/variance and shape fit contracts before empirical execution."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from src import tail_shape_features as features
from src import tail_shape_models as model


def fixture():
    rng = np.random.default_rng(711)
    dates = pd.bdate_range("2011-01-03", periods=1250)
    f = pd.DataFrame(
        rng.normal(size=(len(dates), len(features.RAW))), index=dates, columns=features.RAW
    )
    f["const"] = 1.0
    u = 0.1 * f.I + 0.4 * rng.standard_t(8, len(f))
    t = pd.DataFrame({"y": u, "event": (u < -1.5).astype(float)}, index=dates)
    return f, t


class TestTailShapeModels(unittest.TestCase):
    def test_shape_objective_matches_analytic_chain_gradient(self):
        rng = np.random.default_rng(113)
        z, s = rng.standard_t(8, 170), rng.normal(size=170)
        for theta in [np.array([0.2]), np.array([0.2, -0.1]), np.array([-0.3, 0.4])]:
            error = check_grad(
                lambda p: model.shape_objective(p, z, s)[0],
                lambda p: model.shape_objective(p, z, s)[1],
                theta,
            )
            self.assertLess(error, 1e-6)

    def test_constant_and_candidate_start_rules_and_kkt(self):
        rng = np.random.default_rng(15)
        z, s = rng.standard_t(8, 600) * np.sqrt(6 / 8), rng.normal(size=600)
        baseline = model.fit_shape(z, s)
        candidate = model.fit_shape(z, s, constant=baseline["theta"][0])
        self.assertEqual(baseline["start"], [0.0])
        self.assertEqual(candidate["start"], [baseline["theta"][0], 0.0])
        for result in [baseline, candidate]:
            self.assertTrue(result["success"])
            self.assertLessEqual(result["projected_gradient_max_abs"], 1e-7)
            self.assertEqual(result["nu"], 8.0)
        self.assertLessEqual(candidate["objective"], baseline["objective"] + 1e-8)

    def test_bound_projection_has_correct_sign(self):
        np.testing.assert_array_equal(
            model.projected_gradient(
                np.array([-3.0, 3.0, 0.0, -3.0, 3.0]), np.array([2.0, -2.0, 1.0, -1.0, 1.0])
            ),
            [0.0, 0.0, 1.0, -1.0, 1.0],
        )

    def test_optimizer_failure_and_false_success_are_not_retried(self):
        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        s = z.copy()
        for success, x in [(False, np.array([0.0])), (True, np.array([2.0]))]:
            result = SimpleNamespace(success=success, x=x, nit=1, message="fixture")
            with patch.object(model, "minimize", return_value=result) as mocked:
                with self.assertRaises(ValueError):
                    model.fit_shape(z, s)
                self.assertEqual(mocked.call_count, 1)

    def test_common_moments_and_fitted_frequency_are_exact(self):
        f, t = fixture()
        train, apply = f.iloc[:1100], f.iloc[1100:]
        fit = model.fit_predict(train, t.loc[train.index], apply)
        for label in ["mu", "variance"]:
            np.testing.assert_array_equal(
                fit["conditional"][label]["constant_shape"],
                fit["conditional"][label]["skew_shape"],
            )
        self.assertTrue((fit["conditional"]["variance"]["constant_shape"] > 0).all())
        for probability in fit["predictions"].values():
            self.assertTrue(np.isfinite(probability).all())
            self.assertTrue(((probability > 0) & (probability < 1)).all())
        np.testing.assert_array_equal(
            fit["predictions"]["frequency"],
            np.repeat(t.loc[train.index, "event"].mean(), len(apply)),
        )
        self.assertEqual(fit["transform_audit"]["train_n"], 1100)
        self.assertEqual(fit["moment_audit"]["mean"]["train_n"], 1100)
        self.assertEqual(fit["moment_audit"]["variance"]["train_n"], 1100)

    def test_application_inputs_cannot_change_training_parameters(self):
        f, t = fixture()
        train, apply = f.iloc[:1100], f.iloc[1100:].copy()
        before = model.fit_predict(train, t.loc[train.index], apply)
        apply[["I", "R", "skew"]] += 5.0
        after = model.fit_predict(train, t.loc[train.index], apply)
        for key in [
            "moment_audit",
            "shape_audit",
            "transform_audit",
            "skew_mean",
            "skew_scale",
            "frequency",
        ]:
            self.assertEqual(before[key], after[key])

    def test_event_definition_and_alignment_are_validated(self):
        f, t = fixture()
        bad = t.iloc[:1100].copy()
        bad.iloc[2, 1] = 1 - bad.iloc[2, 1]
        with self.assertRaises(ValueError):
            model.fit_predict(f.iloc[:1100], bad, f.iloc[1100:])
        with self.assertRaises(ValueError):
            model.fit_predict(f.iloc[:1100], t.iloc[:1100].iloc[::-1], f.iloc[1100:])

    def test_likelihood_includes_scale_jacobian(self):
        u = np.array([-2.0, 0.0, 1.0])
        mu = np.array([0.2, 0.3, -0.1])
        h = np.array([0.5, 1.0, 3.0])
        lam = np.array([-0.1, 0.0, 0.2])
        expected = model.density.logpdf((u - mu) / np.sqrt(h), 8.0, lam) - 0.5 * np.log(h)
        np.testing.assert_allclose(model.log_density(u, mu, h, lam), expected)


if __name__ == "__main__":
    unittest.main()
