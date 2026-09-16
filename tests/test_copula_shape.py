"""Prewritten generated contracts for shape-sensitive marginal calibration."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm, t

import src.copula_shape as subject
from src.copula_shape import fit_shape, inverse_shape, shape_coordinates
from tests.test_verify_copula_calibration import synthetic_archive


def parameters(location=(0.0, 0.0), scale=(1.0, 1.0), epsilon=(0.0, 0.0), delta=(1.0, 1.0)):
    return {
        "location": list(location),
        "scale": list(scale),
        "epsilon": list(epsilon),
        "delta": list(delta),
        "train_n": 0,
        "optimizer_audits": [],
    }


class ShapeCoreTests(unittest.TestCase):
    def test_identity_recovers_original_t8_coordinates_and_density(self):
        z = np.array([[-2.0, 0.0], [0.3, 1.2], [4.0, -0.7]])
        h = np.array([[0.2, 0.8], [1.0, 2.0], [0.5, 4.0]])
        got = shape_coordinates(z, h, parameters())
        self.assertEqual(set(got), {"z", "normal", "log_marginal", "pit"})
        assert_array_equal(got["z"], z)
        assert_allclose(got["normal"], norm.ppf(t.cdf(z, 8)), atol=2e-12)
        assert_allclose(
            got["log_marginal"], t.logpdf(z, 8) - 0.5 * np.log(0.75 * h), atol=2e-12
        )

    def test_zero_shape_adjustment_exactly_nests_affine_calibration(self):
        z = np.array([[-1.5, 0.5], [0.2, 1.3], [2.1, -0.7]])
        h = np.array([[0.2, 0.8], [1.0, 2.0], [0.5, 4.0]])
        spec = parameters((0.2, -0.3), (0.8, 1.2))
        w = norm.ppf(t.cdf(z, 8))
        v = (w - spec["location"]) / spec["scale"]
        got = shape_coordinates(z, h, spec)
        expected = (
            t.logpdf(z, 8)
            - 0.5 * np.log(0.75 * h)
            + norm.logpdf(v)
            - norm.logpdf(w)
            - np.log(spec["scale"])
        )
        assert_allclose(got["normal"], v, atol=2e-12)
        assert_allclose(got["log_marginal"], expected, atol=2e-11)
        assert_allclose(got["z"], t.ppf(norm.cdf(v), 8), rtol=1e-9, atol=1e-11)

    def test_jacobian_matches_cdf_derivative_for_skew_and_tail_adjustments(self):
        z = np.array([[-1.3, 0.4], [0.2, 1.2]])
        h = np.array([[0.4, 0.9], [1.2, 0.6]])
        spec = parameters((0.1, -0.2), (0.8, 1.1), (0.35, -0.4), (0.9, 1.3))
        w = norm.ppf(t.cdf(z, 8))
        x = (w - spec["location"]) / spec["scale"]
        u = np.asarray(spec["delta"]) * np.arcsinh(x) - spec["epsilon"]
        v = np.sinh(u)
        log_j = (
            np.log(spec["delta"])
            + np.log(np.cosh(u))
            - np.log(spec["scale"])
            - 0.5 * np.log1p(x * x)
        )
        expected = (
            t.logpdf(z, 8) - 0.5 * np.log(0.75 * h) + norm.logpdf(v) - norm.logpdf(w) + log_j
        )
        got = shape_coordinates(z, h, spec)
        assert_allclose(got["log_marginal"], expected, atol=2e-10)
        step = 1e-5

        def cdf(value):
            x = (norm.ppf(t.cdf(value, 8)) - spec["location"]) / spec["scale"]
            return norm.cdf(
                np.sinh(np.asarray(spec["delta"]) * np.arcsinh(x) - spec["epsilon"])
            )

        derivative = (cdf(z + step) - cdf(z - step)) / (2 * step * np.sqrt(0.75 * h))
        assert_allclose(np.exp(got["log_marginal"]), derivative, rtol=5e-8)

    def test_inverse_roundtrip_at_bounded_risk_simulation_extremes(self):
        spec = parameters((0.1, -0.2), (1.4, 1.3), (0.75, -0.75), (0.75, 0.75))
        v = np.array([[-4.8, 4.8], [4.8, -4.8], [0.0, 0.0]])
        z = inverse_shape(v, spec)
        self.assertTrue(np.isfinite(z).all())
        got = shape_coordinates(z, np.ones_like(z), spec)
        assert_allclose(got["normal"], v, atol=1e-9, rtol=1e-9)
        self.assertGreater(abs(z[1, 0]), 1e20)

    def test_fit_rejects_missing_and_degenerate_training_pairs(self):
        for z in (
            np.empty((0, 2)),
            np.zeros((1, 2)),
            np.zeros((8, 2)),
            np.array([[0.0, 1.0], [np.nan, 2.0]]),
            np.ones((3, 3)),
        ):
            with self.subTest(shape=z.shape), self.assertRaises(ValueError):
                fit_shape(z)

    def test_normalization_under_skew_and_heavier_and_lighter_tails(self):
        nodes, weights = np.polynomial.legendre.leggauss(220)
        w = 18 * nodes
        z = np.sign(w) * t.isf(norm.sf(np.abs(w)), 8)
        spec = parameters((0.1, -0.2), (0.8, 1.1), (0.35, -0.3), (0.9, 1.2))
        result = shape_coordinates(np.column_stack([z, z]), np.full((len(z), 2), 4 / 3), spec)
        for j in range(2):
            mass = float(
                (18 * weights)
                @ np.exp(result["log_marginal"][:, j] + norm.logpdf(w) - t.logpdf(z, 8))
            )
            self.assertAlmostEqual(mass, 1.0, delta=2e-7)

    def test_objective_derivatives_and_positive_semidefinite_hessian(self):
        x = np.array([-2.0, -0.8, -0.3, 0.1, 0.7, 1.3, 2.2])
        for theta in ([-0.6, 0.8], [0.2, 1.15], [0.6, 1.9]):
            theta = np.array(theta)
            value, gradient, hessian = subject.objective(theta, x)
            u = theta[1] * np.arcsinh(x) - theta[0]
            expected = np.mean(0.5 * np.sinh(u) ** 2 - np.log(np.cosh(u))) - np.log(theta[1])
            self.assertAlmostEqual(value, expected, places=12)
            step = 1e-5
            basis = np.eye(2) * step
            fd = np.array(
                [
                    (subject.objective(theta + b, x)[0] - subject.objective(theta - b, x)[0])
                    / (2 * step)
                    for b in basis
                ]
            )
            hd = np.column_stack(
                [
                    (subject.objective(theta + b, x)[1] - subject.objective(theta - b, x)[1])
                    / (2 * step)
                    for b in basis
                ]
            )
            assert_allclose(gradient, fd, atol=2e-8, rtol=2e-8)
            assert_allclose(hessian, hd, atol=2e-7, rtol=2e-8)
            self.assertGreater(np.linalg.eigvalsh(hessian).min(), 0.0)

    def test_generated_skew_is_detected_with_better_marginal_training_score(self):
        p = (np.arange(1001) + 0.5) / 1001
        g = norm.ppf(p)
        skew = np.sinh(np.arcsinh(g) + 0.6)
        w = np.column_stack([skew, -skew[::-1]])
        z = np.sign(w) * t.isf(norm.sf(np.abs(w)), 8)
        fit = fit_shape(z)
        self.assertGreater(fit["epsilon"][0], 0.0)
        self.assertLess(fit["epsilon"][1], 0.0)
        for j, audit in enumerate(fit["optimizer_audits"]):
            x = (w[:, j] - fit["location"][j]) / fit["scale"][j]
            self.assertLess(audit["objective"], subject.objective([0.0, 1.0], x)[0] - 1e-5)

    def test_fit_audit_reconstructs_convex_box_certificate_and_affine_archive(self):
        rng = np.random.default_rng(6401)
        w = rng.normal(size=(101, 2)) * [0.9, 1.2] + [0.2, -0.1]
        z = np.sign(w) * t.isf(norm.sf(np.abs(w)), 8)
        fit = fit_shape(z)
        self.assertEqual(
            set(fit), {"location", "scale", "epsilon", "delta", "train_n", "optimizer_audits"}
        )
        self.assertEqual(fit["train_n"], len(z))
        assert_allclose(fit["location"], w.mean(0), atol=1e-10)
        assert_allclose(fit["scale"], w.std(0), atol=1e-10)
        lower, upper = np.array([-0.75, 0.75]), np.array([0.75, 2.0])
        for j, audit in enumerate(fit["optimizer_audits"]):
            theta = np.array([fit["epsilon"][j], fit["delta"][j]])
            self.assertTrue(((theta >= lower) & (theta <= upper)).all())
            x = (w[:, j] - fit["location"][j]) / fit["scale"][j]
            value, gradient, hessian = subject.objective(theta, x)
            corner = np.where(gradient >= 0, lower, upper)
            gap = float(gradient @ (theta - corner))
            assert_allclose(audit["theta"], theta, atol=0, rtol=0)
            assert_allclose(audit["gradient"], gradient, atol=1e-9)
            assert_allclose(audit["hessian"], hessian, atol=1e-9)
            self.assertAlmostEqual(audit["objective"], value, places=9)
            self.assertAlmostEqual(audit["box_gap"], max(gap, 0), delta=1e-9)
            self.assertAlmostEqual(audit["lower_bound"], value - max(gap, 0), delta=1e-9)
            self.assertLessEqual(gap, 1e-7)
            self.assertEqual(audit["bounds"], [[-0.75, 0.75], [0.75, 2.0]])

    def test_invalid_forecast_parameters_and_unrepresentable_inverse_fail(self):
        for spec in (
            parameters(scale=(0.0, 1.0)),
            parameters(delta=(0.0, 1.0)),
            parameters(epsilon=(np.nan, 0.0)),
            parameters(delta=(0.7, 1.0)),
            parameters(epsilon=(0.8, 0.0)),
        ):
            with self.subTest(parameters=spec), self.assertRaises(ValueError):
                shape_coordinates(np.zeros((2, 2)), np.ones((2, 2)), spec)
        with self.assertRaises(ValueError):
            inverse_shape(np.array([[200.0, 0.0]]), parameters())

    def test_inputs_and_parameter_audits_are_not_mutated(self):
        z = np.array([[-1.0, 0.7], [0.2, -0.4], [1.5, 0.3]])
        h = np.ones_like(z)
        before = z.copy()
        fit = fit_shape(z)
        audit = copy.deepcopy(fit)
        shape_coordinates(z, h, fit)
        inverse_shape(np.array([[0.1, -0.2]]), fit)
        assert_array_equal(z, before)
        assert_array_equal(h, np.ones_like(h))
        self.assertEqual(fit, audit)


def dependency_stub(z, family):
    return 0.25, {"family": family, "train_n": len(z), "generated_stub": True}


class ShapeTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.copula_calibration import run_crossed

        cls.archive, cls.calendar = synthetic_archive()
        cls.baseline = run_crossed(cls.archive, minimum_train=12)

    def test_future_labels_cannot_change_issued_monthly_parameters(self):
        first = self.baseline["applications"].fit_origin.iloc[0]
        apps = self.baseline["applications"].loc[lambda f: f.fit_origin == first].copy()
        panel = self.baseline["panel"].loc[lambda f: f.fit_origin == first].copy()
        with patch.object(subject, "fit_dependence", side_effect=dependency_stub):
            original = subject.run_shape(self.archive, apps, panel, minimum_train=12)
            modified = self.archive.copy(deep=True)
            modified.loc[modified.origin >= first, ["y_qqq", "y_spx"]] += np.array([0.1, -0.2])
            changed = subject.run_shape(modified, apps, panel, minimum_train=12)
        self.assertEqual(original["fits"][0], changed["fits"][0])
        left = original["applications"].query("fit_origin == @first")
        right = changed["applications"].query("fit_origin == @first")
        import pandas as pd

        pd.testing.assert_frame_equal(left, right)

    def test_supported_wholly_unscored_month_preserves_issued_fits_and_other_scores(self):
        apps = self.baseline["applications"]
        first = apps.fit_origin.iloc[0]
        panel = self.baseline["panel"].loc[lambda f: f.fit_origin != first].copy()
        with patch.object(subject, "fit_dependence", side_effect=dependency_stub):
            result = subject.run_shape(self.archive, apps, panel, minimum_train=12)
        self.assertEqual(result["applications"].origin.tolist(), apps.origin.tolist())
        self.assertEqual(result["panel"].origin.tolist(), panel.origin.tolist())
        self.assertEqual(len(result["fits"]), apps.fit_origin.nunique())


if __name__ == "__main__":
    unittest.main()
