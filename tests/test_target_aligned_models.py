"""Prewritten synthetic contracts for target-aligned bounded scalar quadratics."""

from __future__ import annotations

import unittest
from copy import deepcopy

import numpy as np

from src import target_aligned_models as models


def sample(a=0.2, b=0.35):
    z = np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0])
    diagonal = np.array([0.3, 4.0, 0.5, 2.0, 0.3, 4.0])
    target = diagonal * (a + b * (0.995 - abs(a)) * np.tanh(z))
    return np.column_stack([target, np.ones(6)]), np.column_stack([diagonal, diagonal]), z


class ScalarAlgebraTests(unittest.TestCase):
    def test_signed_numerator_cancellation_retains_the_small_identifiable_component(self):
        residual = np.column_stack([np.array([1e16, 1.0, -1e16]), np.ones(3)])
        result = models.fit_cross_moment(residual, np.ones((3, 2)), np.array([-1.0, 0.0, 1.0]))
        self.assertEqual(result["constant"]["numerator"], 1 / 3)
        self.assertEqual(result["a"], 1 / 3)

    def test_exact_two_stage_coefficients_sufficient_statistics_and_objectives(self):
        residual, h, z = sample()
        result = models.fit_cross_moment(residual, h, z)
        D = np.sqrt(h[:, 0]) * np.sqrt(h[:, 1])
        Y = np.prod(residual, axis=1)
        k = D.mean()
        d = D / k
        y = Y / k
        a = np.clip(np.mean(d * y) / np.mean(d * d), -0.995, 0.995)
        w = (0.995 - abs(a)) * d * np.tanh(z)
        b = np.clip(np.mean(w * (y - a * d)) / np.mean(w * w), -1.0, 1.0)
        self.assertAlmostEqual(result["a"], 0.2, delta=1e-14)
        self.assertAlmostEqual(result["b"], 0.35, delta=1e-14)
        self.assertEqual(result["train_n"], 6)
        self.assertAlmostEqual(result["common_unit"], k)
        for name, den, num, coef, pred in (
            ("constant", np.mean(d * d), np.mean(d * y), a, a * d),
            ("dynamic", np.mean(w * w), np.mean(w * (y - a * d)), b, a * d + b * w),
        ):
            audit = result[name]
            self.assertAlmostEqual(audit["denominator"], den, delta=1e-14)
            self.assertAlmostEqual(audit["numerator"], num, delta=1e-14)
            self.assertAlmostEqual(audit["coefficient"], coef, delta=1e-14)
            self.assertAlmostEqual(audit["objective"], np.mean((y - pred) ** 2), delta=1e-14)
            self.assertLessEqual(abs(audit["projected_gradient"]), audit["gradient_tolerance"])

    def test_scalar_solutions_dominate_fixed_grid_without_penalty(self):
        residual, h, z = sample()
        audit = models.fit_cross_moment(residual, h, z)
        D = np.sqrt(h[:, 0]) * np.sqrt(h[:, 1])
        d = D / audit["common_unit"]
        y = np.prod(residual, axis=1) / audit["common_unit"]
        w = audit["headroom"] * d * np.tanh(z)
        baseline = min(np.mean((y - a * d) ** 2) for a in np.linspace(-0.995, 0.995, 1001))
        dynamic = min(
            np.mean((y - audit["a"] * d - b * w) ** 2) for b in np.linspace(-1, 1, 1001)
        )
        self.assertLessEqual(audit["constant"]["objective"], baseline + 1e-15)
        self.assertLessEqual(audit["dynamic"]["objective"], dynamic + 1e-15)

    def test_single_common_unit_preserves_unweighted_product_mse(self):
        D = np.array([1.0, 10.0])
        Y = np.array([0.9, 0.0])
        z = np.array([-1.0, 1.0])
        residual = np.column_stack([Y, np.ones(2)])
        h = np.column_stack([D, D])
        audit = models.fit_cross_moment(residual, h, z)
        self.assertAlmostEqual(audit["a"], np.dot(D, Y) / np.dot(D, D), delta=1e-14)
        self.assertGreater(abs(audit["a"] - np.mean(Y / D)), 0.4)
        self.assertAlmostEqual(
            audit["constant"]["objective"] * audit["common_unit"] ** 2,
            np.mean((Y - audit["a"] * D) ** 2),
            delta=1e-14,
        )

    def test_asset_swap_positive_unit_rescaling_and_repeated_rows_preserve_coefficients(self):
        residual, h, z = sample()
        original = models.fit_cross_moment(residual, h, z)
        changed = [
            (residual[:, ::-1], h[:, ::-1], z),
            (np.tile(residual, (2, 1)), np.tile(h, (2, 1)), np.tile(z, 2)),
        ]
        units = np.array([100.0, 0.02])
        changed.append((residual * units, h * units**2, z))
        for e, hh, zz in changed:
            result = models.fit_cross_moment(e, hh, zz)
            np.testing.assert_allclose(
                [result["a"], result["b"]],
                [original["a"], original["b"]],
                rtol=1e-12,
                atol=1e-14,
            )

    def test_constant_boundary_removes_headroom_but_preserves_flat_model(self):
        for sign in (-1.0, 1.0):
            residual = np.column_stack([np.full(4, 2.0 * sign), np.ones(4)])
            h = np.ones((4, 2))
            z = np.array([-2.0, -1.0, 1.0, 2.0])
            result = models.fit_cross_moment(residual, h, z)
            self.assertEqual(result["a"], sign * 0.995)
            self.assertEqual(result["headroom"], 0.0)
            self.assertEqual(result["b"], 0.0)
            self.assertEqual(result["slope_status"], "FLAT_OBJECTIVE")
            self.assertIsNone(result["dynamic"]["unconstrained_coefficient"])
            self.assertEqual(result["dynamic"]["denominator"], 0.0)
            self.assertEqual(result["dynamic"]["projected_gradient"], 0.0)
            prediction = models.predict_cross_moment(h, z, result)
            np.testing.assert_array_equal(
                prediction["aligned_constant"], prediction["aligned_dynamic"]
            )

    def test_zero_signal_is_flat_even_with_nonzero_headroom(self):
        residual, h, z = sample()
        audit = models.fit_cross_moment(residual, h, np.zeros_like(z))
        self.assertGreater(audit["headroom"], 0.0)
        self.assertEqual(audit["slope_status"], "FLAT_OBJECTIVE")
        self.assertEqual(audit["b"], 0.0)

    def test_slope_boundary_gradient_signs_are_valid(self):
        z = np.array([-2.0, -1.0, 1.0, 2.0])
        h = np.ones((4, 2))
        for sign in (-1.0, 1.0):
            residual = np.column_stack([sign * 3 * np.tanh(z), np.ones(4)])
            audit = models.fit_cross_moment(residual, h, z)
            self.assertEqual(audit["b"], sign)
            self.assertEqual(audit["dynamic"]["projected_gradient"], 0.0)
            self.assertLess(sign * audit["dynamic"]["gradient"], 0.0)

    def test_individual_signed_zero_products_and_all_zero_targets_remain_valid(self):
        residual, h, z = sample()
        residual[:, 0] = 0.0
        result = models.fit_cross_moment(residual, h, z)
        self.assertEqual(result["a"], 0.0)
        self.assertEqual(result["b"], 0.0)
        self.assertEqual(result["slope_status"], "IDENTIFIED")
        self.assertEqual(result["constant"]["objective"], 0.0)
        self.assertEqual(result["dynamic"]["objective"], 0.0)

    def test_projected_kkt_detects_boundary_sign_and_interior_residual(self):
        self.assertEqual(models.projected_gradient(-1.0, 2.0, [-1.0, 1.0]), 0.0)
        self.assertEqual(models.projected_gradient(1.0, -2.0, [-1.0, 1.0]), 0.0)
        self.assertEqual(models.projected_gradient(-1.0, -2.0, [-1.0, 1.0]), -2.0)
        self.assertEqual(models.projected_gradient(0.2, 0.01, [-1.0, 1.0]), 0.01)


class PredictionAndArithmeticTests(unittest.TestCase):
    def test_zero_slope_nesting_and_uniform_psd_bound_for_extreme_finite_query(self):
        residual, h, z = sample(b=0.0)
        audit = models.fit_cross_moment(residual, h, z)
        self.assertAlmostEqual(audit["b"], 0.0, delta=1e-14)
        query = np.array([-1e308, -1.0, 0.0, 1.0, 1e308])
        hh = np.ones((5, 2))
        predicted = models.predict_cross_moment(hh, query, audit)
        np.testing.assert_allclose(
            predicted["aligned_dynamic"], predicted["aligned_constant"], atol=1e-14
        )
        e, h, z = sample(b=1.0)
        audit = models.fit_cross_moment(e, h, z)
        predicted = models.predict_cross_moment(hh, query, audit)
        for rho in predicted.values():
            self.assertTrue((abs(rho) <= 0.995).all())
            self.assertGreaterEqual(np.min(1 - abs(rho)), 0.005 - 1e-15)

    def test_query_changes_cannot_mutate_training_audit(self):
        residual, h, z = sample()
        audit = models.fit_cross_moment(residual, h, z)
        before = deepcopy(audit)
        models.predict_cross_moment(h, z * 100, audit)
        self.assertEqual(audit, before)

    def test_extreme_representable_asset_units_do_not_multiply_variances_directly(self):
        residual, h, z = sample()
        units = np.array([1e150, 1e-150])
        original = models.fit_cross_moment(residual, h, z)
        scaled = models.fit_cross_moment(residual * units, h * units**2, z)
        np.testing.assert_allclose(
            [scaled["a"], scaled["b"]], [original["a"], original["b"]], rtol=1e-12, atol=1e-14
        )

    def test_nonzero_tiny_slope_norm_is_not_classified_flat(self):
        residual, h, z = sample()
        with self.assertRaisesRegex(ValueError, "underflow|represent"):
            models.fit_cross_moment(residual, h, np.full(len(z), 1e-200))

    def test_nonzero_product_underflow_and_nonfinite_intermediates_reject(self):
        residual, h, z = sample()
        for e, hh, zz in (
            (np.full_like(residual, 1e-200), h, z),
            (np.full_like(residual, 1e200), h, z),
            (residual, np.full_like(h, np.finfo(float).max), z),
        ):
            with self.assertRaises(ValueError):
                models.fit_cross_moment(e, hh, zz)

    def test_bad_shapes_complex_values_nonpositive_h_and_invalid_coefficients_reject(self):
        residual, h, z = sample()
        for e, hh, zz in (
            (residual[:, :1], h, z),
            (residual, h, z[:-1]),
            (residual, np.zeros_like(h), z),
            (residual, h, np.full_like(z, np.nan)),
            (residual.astype(complex) + 1j, h, z),
        ):
            with self.assertRaises(ValueError):
                models.fit_cross_moment(e, hh, zz)
        audit = models.fit_cross_moment(residual, h, z)
        for field, value in (("a", 1.0), ("b", 1.1), ("headroom", 0.1), ("a", np.nan)):
            changed = deepcopy(audit)
            changed[field] = value
            with self.assertRaises(ValueError):
                models.predict_cross_moment(h, z, changed)


if __name__ == "__main__":
    unittest.main()
