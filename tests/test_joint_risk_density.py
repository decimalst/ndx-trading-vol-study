"""Prewritten synthetic matrix-score and scalar global-search contracts."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src import joint_risk_density as density


def exact_moments(rho):
    left = np.array([-1.0, -1.0, 1.0, 1.0])
    other = np.array([-1.0, 1.0, -1.0, 1.0])
    return np.column_stack([left, rho * left + np.sqrt(1 - rho * rho) * other])


def varying_fixture():
    values = np.array([-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    blocks, signals = [], []
    for z in values:
        blocks.append(exact_moments(0.995 * np.tanh(0.3 + 0.2 * z)))
        signals.extend([z] * 4)
    return np.vstack(blocks), np.asarray(signals)


class MatrixScoreTests(unittest.TestCase):
    def test_vectorized_score_matches_independent_cholesky_without_gaussian_half(self):
        rng = np.random.default_rng(2847)
        residual = rng.normal(size=(35, 2))
        h = np.exp(rng.normal(size=(35, 2)))
        rho = rng.uniform(-0.994, 0.994, len(h))
        expected = []
        matrices = []
        for e, variance, correlation in zip(residual, h, rho, strict=True):
            cov = np.diag(variance)
            cov[0, 1] = cov[1, 0] = correlation * np.sqrt(variance[0] * variance[1])
            chol = np.linalg.cholesky(cov)
            standardized = np.linalg.solve(chol, e)
            expected.append(2 * np.log(chol.diagonal()).sum() + standardized @ standardized)
            matrices.append(cov)
        np.testing.assert_allclose(
            density.matrix_score(residual, h, rho), expected, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            density.full_matrix_score(residual, np.asarray(matrices)),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rank_one_and_zero_residual_targets_need_no_target_determinant(self):
        e = np.array([[0.0, 0.0], [1.0, -2.0], [-1.0, 2.0]])
        score = density.matrix_score(e, np.ones((3, 2)), 0.7)
        self.assertAlmostEqual(score[0], np.log(1 - 0.7**2))
        self.assertEqual(score[1], score[2])
        self.assertTrue(np.isfinite(score).all())

    def test_asset_swap_invariance_and_unit_change_adds_only_known_constant(self):
        residual, _ = varying_fixture()
        h = np.exp(np.linspace(-1, 1, residual.size)).reshape(residual.shape)
        rho = np.linspace(-0.7, 0.9, len(residual))
        initial = density.matrix_score(residual, h, rho)
        np.testing.assert_allclose(
            density.matrix_score(residual[:, ::-1], h[:, ::-1], rho), initial, rtol=1e-12
        )
        units = np.array([100.0, 0.001])
        changed = density.matrix_score(residual * units, h * units**2, rho)
        np.testing.assert_allclose(
            changed - initial, 2 * np.log(units).sum(), rtol=1e-12, atol=1e-12
        )

    def test_extreme_representable_marginal_units_do_not_require_variance_product(self):
        residual = np.array([[1e-125, -2e-125], [1e125, 2e125]])
        h = np.array([[1e-250, 1e-250], [1e250, 1e250]])
        result = density.matrix_score(residual, h, 0.5)
        expected = np.log(h).sum(axis=1) + np.log(0.75) + np.array([7.0, 3.0]) / 0.75
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_invalid_diagonal_correlation_matrix_and_nonfinite_outputs_reject(self):
        e, h = np.ones((2, 2)), np.ones((2, 2))
        for bad_h, rho in ((h * 0, 0.2), (h * -1, 0.2), (h, 1.0), (h, np.nan)):
            with self.assertRaises(ValueError):
                density.matrix_score(e, bad_h, rho)
        with self.assertRaises(ValueError):
            density.matrix_score(e * 1e200, h, 0.2)
        for covariance in (
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            np.array([[1.0, 0.1], [0.2, 1.0]]),
        ):
            with self.assertRaises(ValueError):
                density.full_matrix_score(e, covariance)


class DependenceObjectiveTests(unittest.TestCase):
    def test_analytic_derivative_matches_independent_central_difference(self):
        residual, z = varying_fixture()
        for constant, parameter, signal in ((None, 0.4, None), (0.2, -0.7, z), (0.2, 1.3, z)):
            value, gradient = density.dependence_objective(
                parameter, residual, signal, constant=constant
            )
            plus = density.dependence_objective(
                parameter + 1e-6, residual, signal, constant=constant
            )[0]
            minus = density.dependence_objective(
                parameter - 1e-6, residual, signal, constant=constant
            )[0]
            self.assertTrue(np.isfinite(value))
            self.assertAlmostEqual(gradient, (plus - minus) / 2e-6, delta=2e-7)

    def test_zero_slope_nests_constant_exactly_and_penalty_is_meanloss_scaled(self):
        residual, z = varying_fixture()
        constant = density.dependence_objective(0.4, residual)[0]
        candidate = density.dependence_objective(0.0, residual, z, constant=0.4)[0]
        self.assertEqual(constant, candidate)
        first = density.dependence_objective(0.3, residual, z, constant=0.4)
        repeated = density.dependence_objective(
            0.3, np.tile(residual, (2, 1)), np.tile(z, 2), constant=0.4
        )
        np.testing.assert_allclose(first, repeated, rtol=1e-12, atol=1e-12)

    def test_saturated_transforms_keep_finite_spd_correlation_and_gradient(self):
        residual = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 3.0]])
        z = np.array([-1000.0, 0.0, 1000.0])
        value, gradient = density.dependence_objective(4.0, residual, z, constant=4.0)
        self.assertTrue(np.isfinite([value, gradient]).all())
        rho = density.RHO_MAX * np.tanh(4 + 4 * z)
        self.assertGreaterEqual(np.min(1 - np.abs(rho)), 0.005 - 1e-15)

    def test_invalid_residual_signal_and_bounds_reject(self):
        residual, z = varying_fixture()
        for bad in (residual * np.nan, residual * np.inf, residual[:, :1]):
            with self.assertRaises(ValueError):
                density.fit_dependence(bad)
        with self.assertRaises(ValueError):
            density.fit_dependence(residual, z[:-1], constant=0.2)
        with self.assertRaises(ValueError):
            density.fit_dependence(residual, z, constant=4.1)
        with self.assertRaisesRegex(ValueError, "Unrepresentable"):
            density.fit_dependence(np.full((4, 2), 1e200))
        with self.assertRaisesRegex(ValueError, "Unrepresentable"):
            density.fit_dependence(residual, np.full(len(residual), 1e308), constant=0.2)


class GlobalOptimizationTests(unittest.TestCase):
    def test_curvature_bounds_dominate_finite_difference_and_taylor_lower_envelope(self):
        residual, signal = varying_fixture()
        squares = np.square(residual).sum(axis=1)
        product = np.prod(residual, axis=1)
        for left, right, constant in ((-4.0, 4.0, 0.3), (-0.7, 0.8, -0.4), (2.9, 3.1, 3.8)):
            bound = density.interval_curvature_bound(
                left, right, squares, product, signal, constant
            )
            middle, radius = (left + right) / 2, (right - left) / 2
            value, gradient = density.dependence_objective(
                middle, residual, signal, constant=constant
            )
            lower = value - abs(gradient) * radius - 0.5 * bound * radius**2
            for point in np.linspace(left, right, 101):
                point_value = density.dependence_objective(
                    point, residual, signal, constant=constant
                )[0]
                plus = density.dependence_objective(
                    point + 1e-5, residual, signal, constant=constant
                )[1]
                minus = density.dependence_objective(
                    point - 1e-5, residual, signal, constant=constant
                )[1]
                self.assertLessEqual(abs((plus - minus) / 2e-5), bound * (1 + 1e-7))
                self.assertLessEqual(lower, point_value + 1e-12)

    def test_zero_signal_curvature_is_fixed_penalty_and_constant_boundary_is_valid(self):
        residual = exact_moments(0.3)
        bound = density.interval_curvature_bound(
            -4.0,
            4.0,
            np.square(residual).sum(axis=1),
            np.prod(residual, axis=1),
            np.zeros(4),
            0.4,
        )
        self.assertAlmostEqual(bound, 0.02, delta=1e-12)
        boundary = density.fit_dependence(exact_moments(1.0))
        self.assertEqual(boundary["parameter"], 4.0)
        self.assertEqual(boundary["projected_gradient"], 0.0)
        self.assertLess(boundary["gradient"], 0.0)

    def test_constant_solver_recovers_exact_unit_variance_correlation(self):
        for correlation in (-0.8, 0.0, 0.35, 0.9):
            audit = density.fit_dependence(exact_moments(correlation))
            fitted = density.RHO_MAX * np.tanh(audit["parameter"])
            self.assertAlmostEqual(fitted, correlation, delta=1e-9)
            self.assertLessEqual(abs(audit["projected_gradient"]), 1e-7)
            self.assertEqual(audit["penalty"], 0.0)
            self.assertLessEqual(audit["global_value_gap"], 1e-8)

    def test_constant_solver_compares_both_minima_of_nonconvex_misspecified_case(self):
        residual = exact_moments(0.0) * 0.5
        audit = density.fit_dependence(residual)
        rho = density.RHO_MAX * np.tanh(audit["parameter"])
        self.assertAlmostEqual(abs(rho), np.sqrt(0.5), delta=1e-9)
        self.assertLess(audit["objective"], density.dependence_objective(0.0, residual)[0])
        self.assertGreaterEqual(len(audit["stationary_rho_roots"]), 3)

    def test_candidate_recovers_positive_slope_and_certifies_against_dense_grid(self):
        residual, z = varying_fixture()
        a0 = density.fit_dependence(residual)["parameter"]
        audit = density.fit_dependence(residual, z, constant=a0)
        self.assertAlmostEqual(audit["parameter"], 0.2, delta=0.04)
        self.assertLessEqual(audit["global_value_gap"], 1e-8)
        self.assertLessEqual(abs(audit["projected_gradient"]), 1e-7)
        self.assertLessEqual(audit["interval_splits"], 32768)
        grid = np.linspace(-4, 4, 4001)
        oracle = min(
            density.dependence_objective(b, residual, z, constant=a0)[0] for b in grid
        )
        self.assertLessEqual(audit["objective"], oracle + 1e-8)
        self.assertTrue(audit["local_attempts"])
        leaves = audit["certificate_intervals"]
        self.assertEqual(leaves[0]["left"], -4.0)
        self.assertEqual(leaves[-1]["right"], 4.0)
        for first, second in zip(leaves[:-1], leaves[1:], strict=True):
            self.assertEqual(first["right"], second["left"])
        self.assertEqual(
            audit["global_lower_bound"], min(row["lower_bound"] for row in leaves)
        )
        self.assertEqual(len(leaves), audit["interval_splits"] + 1)

    def test_wrong_marginal_scales_can_create_gain_with_true_conditional_correlation_zero(
        self,
    ):
        # Each state has independent Rademacher signs, true correlation zero,
        # but the supplied marginal second moments are incorrectly fixed at1.
        residual = np.vstack(
            [exact_moments(0.0) * np.sqrt(0.1), exact_moments(0.0) * np.sqrt(0.4)]
        )
        signal = np.repeat([-1.0, 1.0], 4)
        self.assertEqual(np.mean(residual[:4, 0] * residual[:4, 1]), 0.0)
        self.assertEqual(np.mean(residual[4:, 0] * residual[4:, 1]), 0.0)
        constant = density.fit_dependence(residual)
        candidate = density.fit_dependence(residual, signal, constant=constant["parameter"])
        self.assertLess(candidate["objective"], constant["objective"] - 0.01)

    def test_certificate_budget_exhaustion_is_an_explicit_failure(self):
        residual, z = varying_fixture()
        with (
            patch.object(density, "MAX_INTERVAL_SPLITS", 0),
            self.assertRaisesRegex(ValueError, "certificate|budget|global"),
        ):
            density.fit_dependence(residual, z, constant=0.3)

    def test_candidate_rejects_inconsistent_lower_bound_above_upper_bound(self):
        residual, signal = varying_fixture()
        with (
            patch.object(density, "interval_curvature_bound", return_value=-1e6),
            self.assertRaisesRegex(ValueError, "certificate|bound|gap"),
        ):
            density.fit_dependence(residual, signal, constant=0.3)

    def test_constant_rejects_negative_or_nonfinite_certificate_gap(self):
        for inflation in (-1.0, np.nan):
            with (
                self.subTest(inflation=inflation),
                patch.object(density, "FLOAT_INFLATION", inflation),
                self.assertRaisesRegex(ValueError, "certificate|bound|gap"),
            ):
                density.fit_dependence(exact_moments(0.3))


if __name__ == "__main__":
    unittest.main()
