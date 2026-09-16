"""Preimplementation mathematical contracts for Hansen's standardized skew-t."""
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import t

from src import tail_shape_density as density


def reference_constants(nu, lam):
    """Scalar Gamma expression from Hansen (1994), independent of the module."""
    c = math.gamma((nu+1)/2)/(math.sqrt(math.pi*(nu-2))*math.gamma(nu/2))
    a = 4*lam*c*(nu-2)/(nu-1)
    b = math.sqrt(1+3*lam**2-a**2)
    return a, b, c


def integrated_moment(order, nu, lam):
    a, b, _ = reference_constants(nu, lam)
    join = -a/b

    def integrand(z):
        return z**order*math.exp(float(density.logpdf(z, nu, lam)))

    left = quad(integrand, -np.inf, join, epsabs=2e-10, epsrel=2e-10, limit=200)[0]
    right = quad(integrand, join, np.inf, epsabs=2e-10, epsrel=2e-10, limit=200)[0]
    return left+right


class TestTailShapeDensity(unittest.TestCase):
    def test_invalid_parameter_domain_and_nan_observations_are_rejected(self):
        for function in (density.logpdf, density.cdf, density.dlogpdf_dlambda):
            for nu, lam in [(2., 0.), (1., 0.), (np.inf, 0.), (np.nan, 0.),
                            (8., -1.), (8., 1.), (8., 1.1), (8., np.nan), (8., np.inf)]:
                with self.subTest(function=function.__name__, nu=nu, lam=lam), self.assertRaises(ValueError):
                    function(0., nu, lam)
            with self.assertRaises(ValueError):
                function([0., np.nan], 8., .2)

    def test_parameters_and_observations_broadcast_together(self):
        z = np.array([-3., 0., 2.])[:, None]
        nu = np.array([4., 8., 30.])[:, None]
        lam = np.array([-.95, -.6, 0., .6, .95])[None, :]
        for function in (density.logpdf, density.cdf, density.dlogpdf_dlambda):
            actual = function(z, nu, lam)
            self.assertEqual(actual.shape, (3, 5))
            expected = np.array([[float(function(x, v, skew)) for skew in lam[0]]
                                 for x, v in zip(z[:, 0], nu[:, 0], strict=True)])
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_inputs_are_not_mutated_and_scalar_results_are_scalar_shaped(self):
        z, nu, lam = np.array([-1., 0., 3.]), np.array([4., 8., 30.]), np.array([-.6, 0., .6])
        before = [item.copy() for item in (z, nu, lam)]
        for function in (density.logpdf, density.cdf, density.dlogpdf_dlambda):
            function(z, nu, lam)
            self.assertEqual(np.ndim(function(0., 8., .6)), 0)
        for first, last in zip(before, (z, nu, lam), strict=True):
            np.testing.assert_array_equal(first, last)

    def test_symmetric_case_matches_variance_standardized_students_t(self):
        z = np.array([-12., -3., -1., 0., .4, 2., 9.])
        for nu in (4., 8., 30.):
            scale = math.sqrt((nu-2)/nu)
            np.testing.assert_allclose(density.logpdf(z, nu, 0.), t.logpdf(z, nu, scale=scale), rtol=1e-13, atol=2e-13)
            np.testing.assert_allclose(density.cdf(z, nu, 0.), t.cdf(z, nu, scale=scale), rtol=2e-13, atol=2e-14)

    def test_symmetric_inverse_quantiles_match_known_student_distribution(self):
        probabilities = np.array([.0001, .01, .05, .25, .5, .75, .95, .99, .9999])
        for nu in (4., 8., 30.):
            z = t.ppf(probabilities, nu)*math.sqrt((nu-2)/nu)
            np.testing.assert_allclose(density.cdf(z, nu, 0.), probabilities, rtol=1e-9, atol=1e-11)

    def test_reflecting_lambda_reflects_density_and_cdf(self):
        z = np.array([-5., -1.5, -.2, 0., .7, 4.])
        for lam in (-.95, -.6, 0., .6, .95):
            np.testing.assert_allclose(density.logpdf(z, 8., lam), density.logpdf(-z, 8., -lam), atol=2e-13)
            np.testing.assert_allclose(density.cdf(z, 8., lam), 1-density.cdf(-z, 8., -lam), atol=2e-14)

    def test_piecewise_join_has_correct_height_and_probability_mass(self):
        for lam in (-.95, -.6, 0., .6, .95):
            a, b, c = reference_constants(8., lam)
            join = -a/b
            self.assertAlmostEqual(float(density.logpdf(join, 8., lam)), math.log(b*c), places=13)
            self.assertAlmostEqual(float(density.cdf(join, 8., lam)), (1-lam)/2, places=13)
            for epsilon in (-1e-9, 1e-9):
                self.assertAlmostEqual(float(density.logpdf(join+epsilon, 8., lam)), math.log(b*c), places=12)

    def test_unit_integral_zero_mean_and_unit_variance_at_fixed_nu8(self):
        for lam in (-.95, -.6, 0., .6, .95):
            for order, expected in [(0, 1.), (1, 0.), (2, 1.)]:
                with self.subTest(lam=lam, order=order):
                    self.assertAlmostEqual(integrated_moment(order, 8., lam), expected, delta=2e-8)

    def test_standardization_also_holds_for_nu4_and_nu30(self):
        for nu in (4., 30.):
            for lam in (-.6, .6):
                for order, expected in [(0, 1.), (1, 0.), (2, 1.)]:
                    with self.subTest(nu=nu, lam=lam, order=order):
                        self.assertAlmostEqual(integrated_moment(order, nu, lam), expected, delta=2e-8)

    def test_cdf_matches_independent_quadrature_on_both_sides_of_join(self):
        for lam in (-.95, -.6, 0., .6, .95):
            a, b, _ = reference_constants(8., lam)
            join = -a/b
            for end in (join-2., join, join+.8):
                integrand = lambda z, skew=lam: math.exp(float(density.logpdf(z, 8., skew)))
                if end <= join:
                    expected = quad(integrand, -np.inf, end, epsabs=2e-10, epsrel=2e-10)[0]
                else:
                    expected = quad(integrand, -np.inf, join, epsabs=2e-10, epsrel=2e-10)[0]
                    expected += quad(integrand, join, end, epsabs=2e-10, epsrel=2e-10)[0]
                self.assertAlmostEqual(float(density.cdf(end, 8., lam)), expected, delta=2e-9)

    def test_cdf_derivative_is_the_density_and_values_are_monotone(self):
        z = np.linspace(-6., 6., 401)
        step = 1e-5
        for lam in (-.95, -.6, 0., .6, .95):
            cdf = density.cdf(z, 8., lam)
            self.assertTrue(((cdf >= 0) & (cdf <= 1)).all())
            self.assertTrue((np.diff(cdf) >= 0).all())
            numerical = (density.cdf(z+step, 8., lam)-density.cdf(z-step, 8., lam))/(2*step)
            np.testing.assert_allclose(numerical, np.exp(density.logpdf(z, 8., lam)), rtol=5e-6, atol=3e-9)

    def test_lambda_derivative_matches_finite_differences_for_variable_lambda(self):
        z = np.array([-8., -2., -.1, .3, 1.2, 7.])[:, None]
        lam = np.array([-.95, -.6, -.01, 0., .01, .6, .95])[None, :]
        step = 1e-6
        for nu in (4., 8., 30.):
            numerical = (density.logpdf(z, nu, lam+step)-density.logpdf(z, nu, lam-step))/(2*step)
            np.testing.assert_allclose(density.dlogpdf_dlambda(z, nu, lam), numerical, rtol=3e-6, atol=2e-7)

    def test_lambda_derivative_is_continuous_at_the_moving_join(self):
        nu = 8.
        for lam in (-.95, -.6, 0., .6, .95):
            a, b, c = reference_constants(nu, lam)
            a_prime = 4*c*(nu-2)/(nu-1)
            expected = (3*lam-a*a_prime)/b**2
            join = -a/b
            self.assertAlmostEqual(float(density.dlogpdf_dlambda(join, nu, lam)), expected, delta=2e-12)
            numerical = (density.logpdf(join, nu, lam+1e-7)-density.logpdf(join, nu, lam-1e-7))/2e-7
            self.assertAlmostEqual(float(numerical), expected, delta=2e-5)
            self.assertAlmostEqual(float(density.dlogpdf_dlambda(join-1e-9, nu, lam)), expected, delta=2e-5)
            self.assertAlmostEqual(float(density.dlogpdf_dlambda(join+1e-9, nu, lam)), expected, delta=2e-5)

    def test_integrated_lambda_score_is_zero(self):
        for lam in (-.6, 0., .6):
            a, b, _ = reference_constants(8., lam)
            join = -a/b

            def integrand(z, skew=lam):
                return math.exp(float(density.logpdf(z, 8., skew)))*float(density.dlogpdf_dlambda(z, 8., skew))

            result = quad(integrand, -np.inf, join, epsabs=2e-9, epsrel=2e-9)[0]
            result += quad(integrand, join, np.inf, epsabs=2e-9, epsrel=2e-9)[0]
            self.assertAlmostEqual(result, 0., delta=2e-8)

    def test_fixed_external_location_and_scale_preserve_moments_for_all_skews(self):
        mu, sigma = .7, 1.8
        for lam in (-.95, 0., .95):
            mean = integrated_moment(1, 8., lam)
            second = integrated_moment(2, 8., lam)
            self.assertAlmostEqual(mu+sigma*mean, mu, delta=2e-8)
            self.assertAlmostEqual(sigma**2*(second-mean**2), sigma**2, delta=5e-8)

    def test_infinite_observation_limits_and_extreme_finite_log_densities(self):
        np.testing.assert_array_equal(density.cdf([-np.inf, np.inf], 8., [.6, -.6]), [0., 1.])
        np.testing.assert_array_equal(density.logpdf([-np.inf, np.inf], 8., [.6, -.6]), [-np.inf, -np.inf])
        z = np.array([-1e250, 1e250])
        self.assertTrue(np.isfinite(density.logpdf(z, 8., [-.95, .95])).all())
        self.assertTrue(np.isfinite(density.dlogpdf_dlambda(z, 8., [-.95, .95])).all())


if __name__ == "__main__":
    unittest.main()
