"""Generated mathematical contracts written before the copula implementation."""

import importlib
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal, multivariate_t, norm, t


class JointCopulaDensityTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module("src.joint_copula_density")
        self.z = np.array([[-2.0, -0.3], [0.0, 0.0], [0.2, 1.3], [3.0, -4.0]])

    def test_t_density_matches_independent_distribution_ratio(self):
        for rho in (-0.995, -0.7, 0.0, 0.6, 0.995):
            expected = multivariate_t.logpdf(
                self.z, shape=[[1, rho], [rho, 1]], df=8
            ) - t.logpdf(self.z, 8).sum(axis=1)
            np.testing.assert_allclose(
                self.m.log_copula(self.z, rho, "t8"), expected, atol=2e-12, rtol=2e-12
            )

    def test_gaussian_density_matches_transformed_distribution_ratio(self):
        w = norm.ppf(t.cdf(self.z, 8))
        for rho in (-0.995, -0.7, 0.0, 0.6, 0.995):
            expected = multivariate_normal.logpdf(w, cov=[[1, rho], [rho, 1]]) - norm.logpdf(
                w
            ).sum(axis=1)
            np.testing.assert_allclose(
                self.m.log_copula(self.z, rho, "gaussian"), expected, atol=1e-10, rtol=1e-11
            )

    def test_zero_correlation_t_is_not_independence(self):
        self.assertGreater(np.max(np.abs(self.m.log_copula(self.z, 0.0, "t8"))), 0.01)
        np.testing.assert_array_equal(
            self.m.log_copula(self.z, 0.0, "independence"), np.zeros(4)
        )
        np.testing.assert_allclose(self.m.log_copula(self.z, 0.0, "gaussian"), 0.0, atol=1e-14)

    def test_each_dependence_family_preserves_marginals(self):
        for family in ("t8", "gaussian"):
            for rho in (-0.6, 0.0, 0.6):
                for first in (-1.0, 0.0, 2.0):
                    value, error = quad(
                        lambda second, first=first, rho=rho, family=family: np.exp(
                            self.m.log_copula([[first, second]], rho, family)[0]
                            + t.logpdf(second, 8)
                        ),
                        -np.inf,
                        np.inf,
                        epsabs=2e-8,
                    )
                    self.assertAlmostEqual(value, 1.0, delta=2e-7)
                    self.assertLess(error, 2e-7)

    def test_fixed_marginal_has_specified_variance_and_location(self):
        variance = 0.023
        scale = np.sqrt(variance * 0.75)
        value, _ = quad(
            lambda r: (
                r
                * r
                * np.exp(
                    self.m.marginal_logpdf(np.array([r / scale]), np.array([variance]))[0]
                )
            ),
            -np.inf,
            np.inf,
        )
        self.assertAlmostEqual(value, variance, delta=1e-9)

    def test_shared_marginal_scores_cancel_exactly_up_to_roundoff(self):
        h = np.full_like(self.z, 0.003)
        marginal = self.m.marginal_logpdf(self.z, h).sum(axis=1)
        a = self.m.log_copula(self.z, 0.7, "t8")
        b = self.m.log_copula(self.z, 0.4, "gaussian")
        np.testing.assert_allclose(-(marginal + a) + (marginal + b), b - a, atol=2e-14)

    def test_stable_logtail_transform_does_not_saturate(self):
        z = np.array([-1e300, -1e100, -1e10, -1.0, 0.0, 1.0, 1e10, 1e100, 1e300])
        w = self.m.normal_scores(z)
        self.assertTrue(np.isfinite(w).all())
        self.assertTrue((np.diff(w) > 0).all())
        np.testing.assert_allclose(w, -w[::-1], atol=0.0)
        np.testing.assert_allclose(w[3:6], norm.ppf(t.cdf(z[3:6], 8)), atol=1e-13)

    def test_extreme_finite_values_and_vector_rho_remain_finite(self):
        z = np.array([[1e300, -1e300], [1e100, 1e100], [0.0, -1e200]])
        for family in ("t8", "gaussian", "independence"):
            rho = np.zeros(3) if family == "independence" else np.array([-0.995, 0.995, 0.0])
            self.assertTrue(np.isfinite(self.m.log_copula(z, rho, family)).all())
        self.assertTrue(np.isfinite(self.m.marginal_logpdf(z, np.ones_like(z))).all())

    def test_symmetry_and_rowwise_parameters(self):
        for family in ("t8", "gaussian"):
            rho = np.array([-0.9, -0.1, 0.2, 0.8])
            got = self.m.log_copula(self.z, rho, family)
            np.testing.assert_allclose(got, self.m.log_copula(-self.z, rho, family))
            np.testing.assert_allclose(got, self.m.log_copula(self.z[:, ::-1], rho, family))
            expected = np.array(
                [self.m.log_copula(self.z[i : i + 1], rho[i], family)[0] for i in range(4)]
            )
            np.testing.assert_allclose(got, expected)

    def test_analytic_objective_derivative(self):
        for family in ("t8", "gaussian"):
            for rho in (-0.8, -0.1, 0.7):
                value, gradient = self.m.objective_gradient(self.z, rho, family)
                self.assertAlmostEqual(
                    value, -self.m.log_copula(self.z, rho, family).mean(), delta=1e-12
                )
                step = 1e-6
                finite_difference = (
                    -self.m.log_copula(self.z, rho + step, family).mean()
                    + self.m.log_copula(self.z, rho - step, family).mean()
                ) / (2 * step)
                self.assertAlmostEqual(gradient, finite_difference, delta=2e-7)

    def test_generated_optima_are_no_worse_than_independent_grid(self):
        rng = np.random.default_rng(9027)
        z = rng.standard_t(8, (160, 2))
        z[:, 1] += 0.65 * z[:, 0]
        for family in ("t8", "gaussian"):
            rho, audit = self.m.fit_dependence(z, family)
            grid = np.linspace(-0.995, 0.995, 1001)
            values = [-self.m.log_copula(z, r, family).mean() for r in grid]
            self.assertLessEqual(audit["objective"], min(values) + 1e-9)
            self.assertLessEqual(audit["projected_gradient"], 1e-7)
            self.assertEqual(audit["status"], "CERTIFIED_GLOBAL_NUMERICAL_OPTIMUM")
            self.assertLessEqual(audit["certificate"]["gap"], 1e-8)
            self.assertLessEqual(abs(rho), 0.995)

    def test_global_certificate_covers_whole_interval_and_bounds_interior(self):
        _, audit = self.m.fit_dependence(self.z, "t8")
        leaves = sorted(audit["certificate"]["leaves"], key=lambda q: q["left"])
        self.assertEqual(leaves[0]["left"], -0.995)
        self.assertEqual(leaves[-1]["right"], 0.995)
        for i, leaf in enumerate(leaves):
            if i:
                self.assertEqual(leaves[i - 1]["right"], leaf["left"])
            for point in np.linspace(leaf["left"], leaf["right"], 5):
                actual = -self.m.log_copula(self.z, point, "t8").mean()
                self.assertLessEqual(leaf["lower_bound"], actual + 1e-11)

    def test_boundary_tie_is_deterministic_and_not_local_stationary_maximum(self):
        for family in ("t8", "gaussian"):
            rho, audit = self.m.fit_dependence(np.zeros((5, 2)), family)
            self.assertEqual(rho, -0.995)
            self.assertLess(
                audit["objective"], -self.m.log_copula(np.zeros((5, 2)), 0.0, family).mean()
            )

    def test_certificate_budget_exhaustion_is_failure(self):
        with self.assertRaisesRegex(ValueError, "certificate|certification"):
            self.m.fit_dependence(self.z, "t8", max_splits=0)

    def test_invalid_data_domain_and_nonzero_independence_fail(self):
        for z in ([], [1.0, 2.0], [[1.0, np.nan]], [[1.0, np.inf]]):
            with self.assertRaises(ValueError):
                self.m.log_copula(z, 0.2, "t8")
        for rho in (-1.0, 1.0, 0.996, np.nan):
            with self.assertRaises(ValueError):
                self.m.log_copula(self.z, rho, "t8")
        with self.assertRaises(ValueError):
            self.m.log_copula(self.z, 0.1, "independence")
        for h in (0.0, -1.0, np.nan):
            with self.assertRaises(ValueError):
                self.m.marginal_logpdf(self.z, np.full_like(self.z, h))


if __name__ == "__main__":
    unittest.main()
