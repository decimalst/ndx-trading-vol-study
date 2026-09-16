"""Synthetic contracts written before the new independent profiled solver."""

import json
import math
import unittest
from decimal import Decimal, localcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src import profiled_quarter_solver as solver


def two_rows(level=0.4, scale=1.0, xscale=1.0):
    design = np.zeros((2, 31))
    design[:, 0] = 1
    design[:, 1] = [-xscale, xscale]
    return design, scale * np.exp(np.array([-level, level]))


def decimal_solution(q, xscale=1.0):
    """Separate scalar derivative root in 75-digit Decimal arithmetic."""
    with localcontext() as ctx:
        ctx.prec = 75
        lo, hi = Decimal(-50), Decimal(50)
        qm, qp = map(Decimal.from_float, map(float, q))
        x = Decimal.from_float(float(xscale))
        for _ in range(270):
            b = (lo + hi) / 2
            left, right = qm * (x * b).exp(), qp * (-x * b).exp()
            g = x * (left - right) / (left + right) + Decimal(".02") * b
            if g < 0:
                lo = b
            else:
                hi = b
        b = (lo + hi) / 2
        a = ((qm * (x * b).exp() + qp * (-x * b).exp()) / 2).ln()
        return float(a), float(b)


def general_sample():
    rng = np.random.default_rng(20260925)
    x = rng.normal(size=(180, 30))
    x[:, 2] = x[:, 1] + 1e-9 * rng.normal(size=len(x))
    x += np.linspace(-0.7, 0.8, 30)
    design = np.c_[np.ones(len(x)), x]
    q = 1.7 * np.exp(0.2 * x[:, 0] - 0.1 * x[:, 4])
    return design, q


class ProfiledQuarterSolver(unittest.TestCase):
    def test_scalar_solution_matches_high_precision_oracle(self):
        x, q = two_rows()
        beta, audit = solver.solve_baseline(x, q)
        a, b = decimal_solution(q)
        np.testing.assert_allclose(beta[:2], [a, b], rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(beta[2:], 0)
        self.assertLessEqual(audit["gradient_max_abs"], 1e-8 + 1e-12)
        self.assertLessEqual(audit["profile_gradient_max_abs"], 1e-10)
        json.dumps(audit, allow_nan=False)

    def test_no_target_renormalization_and_intercept_unpenalized(self):
        x, q = two_rows()
        first, _ = solver.solve_baseline(x, q)
        second, audit = solver.solve_baseline(x, q * 19)
        np.testing.assert_allclose(second[1:], first[1:], rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(second[0] - first[0], math.log(19), delta=1e-12)
        self.assertAlmostEqual(audit["start_intercept"], math.log(float(q.mean() * 19)))
        self.assertEqual(audit["start_slopes"], [0.0] * 30)

    def test_constant_target_terminates_before_direction_normalization(self):
        x, _ = general_sample()
        beta, audit = solver.solve_baseline(x, np.full(len(x), 7.0))
        np.testing.assert_allclose(beta, np.r_[math.log(7), np.zeros(30)], atol=1e-13)
        self.assertEqual(audit["iterations"], 0)
        self.assertEqual(audit["history"], [])

    def test_noncentered_design_profile_full_identity(self):
        x, q = general_sample()
        b = np.linspace(-0.025, 0.035, 30)
        state = solver.profile_objective(b, x, q)
        value, g, _ = solver.full_objective(np.r_[state["intercept"], b], x, q)
        self.assertAlmostEqual(value, state["objective"], delta=1e-12)
        np.testing.assert_allclose(g[1:], state["gradient"], rtol=1e-11, atol=1e-12)
        self.assertLess(abs(g[0]), 1e-13)
        wrong = 1 + state["intercept"] + 0.01 * np.dot(b, b)
        self.assertGreater(abs(wrong - value), 0.01)

    def test_profile_derivatives_and_covariance_curvature(self):
        x, q = general_sample()
        b = np.linspace(-0.02, 0.03, 30)
        state = solver.profile_objective(b, x, q)
        for j in (0, 1, 2, 10, 29):
            step = np.eye(30)[j] * 1e-5
            plus = solver.profile_objective(b + step, x, q)
            minus = solver.profile_objective(b - step, x, q)
            self.assertAlmostEqual(
                state["gradient"][j],
                (plus["objective"] - minus["objective"]) / 2e-5,
                delta=1e-9,
            )
            np.testing.assert_allclose(
                state["hessian"][:, j],
                (plus["gradient"] - minus["gradient"]) / 2e-5,
                rtol=1e-7,
                atol=1e-9,
            )
        self.assertGreaterEqual(np.linalg.eigvalsh(state["hessian"]).min(), 0.02 - 1e-12)

    def test_near_collinearity_has_unique_regularized_solution(self):
        x, q = general_sample()
        beta, audit = solver.solve_baseline(x, q)
        _, g, _ = solver.full_objective(beta, x, q)
        self.assertLessEqual(np.max(np.abs(g)), 1e-8 + 1e-12)
        self.assertAlmostEqual(beta[2], beta[3], delta=1e-8)
        self.assertGreaterEqual(audit["slope_distance_bound"], 0)
        self.assertGreaterEqual(audit["objective_gap_bound"], 0)

    def test_identical_rounded_objectives_do_not_hide_nonstationarity(self):
        x, q = two_rows(level=1e-7, scale=math.exp(100))
        beta, audit = solver.solve_baseline(x, q)
        start = np.r_[audit["start_intercept"], np.zeros(30)]
        value0, g0, _ = solver.full_objective(start, x, q)
        value1, _, _ = solver.full_objective(beta, x, q)
        self.assertEqual(value0, value1)
        self.assertGreater(np.max(np.abs(g0)), 1e-8 + 1e-12)
        self.assertGreater(abs(beta[1]), 5e-8)
        self.assertLessEqual(audit["profile_gradient_max_abs"], 1e-10)
        self.assertLess(audit["history"][0]["objective_difference"], 0)

    def test_small_difference_matches_decimal_not_total_subtraction(self):
        x, q = two_rows(level=1e-7, scale=math.exp(100))
        delta = np.zeros(30)
        delta[0] = 1e-7 / 1.02
        actual = solver.objective_difference(np.zeros(30), delta, x, q)
        with localcontext() as ctx:
            ctx.prec = 75
            qm, qp = map(Decimal.from_float, map(float, q))
            d = Decimal.from_float(delta[0])
            expected = ((qm * d.exp() + qp * (-d).exp()) / (qm + qp)).ln() + Decimal(
                ".01"
            ) * d * d
        self.assertLess(actual, 0)
        self.assertAlmostEqual(actual, float(expected), delta=abs(float(expected)) * 2e-7)

    def test_large_analytic_bracket_can_fail_but_bounded_local_rule_succeeds(self):
        x, q = two_rows(level=10, xscale=10)
        start = solver.profile_objective(np.zeros(30), x, q)
        radius = 2 * abs(start["gradient"][0]) / 0.02
        far = np.zeros(30)
        far[0] = radius
        with self.assertRaises(ValueError):
            solver.profile_objective(far, x, q)
        beta, audit = solver.solve_baseline(x, q)
        a, b = decimal_solution(q, 10)
        np.testing.assert_allclose(beta[:2], [a, b], rtol=1e-10, atol=1e-11)
        self.assertGreater(sum(row["invalid_bracket_trials"] for row in audit["history"]), 0)
        for row in audit["history"]:
            self.assertLess(row["bracket"][0], row["bracket"][1])
            self.assertLess(row["bracket_derivatives"][0], 0)
            self.assertGreaterEqual(row["bracket_derivatives"][1], 0)
            invalid = [
                trial["distance"] for trial in row["bracket_trials"] if not trial["valid"]
            ]
            if invalid:
                self.assertLess(row["bracket"][1], min(invalid))

    def test_finite_rejection_cannot_skip_initial_or_permanent_domain_failure(self):
        x, q = two_rows()
        original = solver._profile

        def no_nonzero_state(b, prepared, **kwargs):
            if np.any(b != 0):
                raise ValueError("synthetic unrepresentable trial")
            return original(b, prepared, **kwargs)

        with (
            patch.object(solver, "_profile", side_effect=no_nonzero_state),
            self.assertRaisesRegex(ValueError, "bracket budget"),
        ):
            solver.solve_baseline(x, q)
        with (
            patch.object(solver, "_profile", side_effect=ValueError("initial domain")),
            self.assertRaisesRegex(ValueError, "initial domain"),
        ):
            solver.solve_baseline(x, q)

    def test_extreme_representable_constant_targets_and_zero_step(self):
        x, _ = two_rows()
        for target in [1e-300, 1e300]:
            with self.subTest(target=target):
                beta, audit = solver.solve_baseline(x, np.full(2, target))
                self.assertAlmostEqual(beta[0], math.log(target), delta=1e-12)
                self.assertEqual(audit["iterations"], 0)
                self.assertEqual(
                    solver.objective_difference(
                        np.zeros(30), np.zeros(30), x, np.full(2, target)
                    ),
                    0,
                )

    def test_matrix_product_nonzero_underflow_is_not_hidden_by_regularization(self):
        x, q = two_rows()
        x[:, 3] = [-1e-200, 1e-200]
        with self.assertRaisesRegex(ValueError, "underflow"):
            solver.profile_objective(np.zeros(30), x, q)

    def test_synthetic_large_sample_reaches_unchanged_full_gate(self):
        rng = np.random.default_rng(927)
        x = np.c_[np.ones(3600), rng.normal(size=(3600, 30))]
        q = np.exp(0.3 * x[:, 1] - 0.2 * x[:, 7] + 0.15 * rng.normal(size=len(x)))
        beta, audit = solver.solve_baseline(x, q)
        self.assertLessEqual(audit["gradient_max_abs"], 1e-8 + 1e-12)
        self.assertLessEqual(audit["profile_gradient_max_abs"], 1e-10)
        self.assertTrue(np.isfinite(beta).all())

    def test_row_permutation_does_not_change_objective_optimum(self):
        x, q = general_sample()
        first, _ = solver.solve_baseline(x, q)
        second, _ = solver.solve_baseline(x[::-1], q[::-1])
        np.testing.assert_allclose(first, second, rtol=1e-9, atol=1e-11)

    def test_zero_and_signed_design_values_are_valid(self):
        x, q = two_rows(level=-0.3)
        beta, _ = solver.solve_baseline(x, q)
        self.assertLess(beta[1], 0)
        self.assertTrue(np.isfinite(beta).all())

    def test_invalid_domains_shapes_and_literal_intercept_reject(self):
        x, q = two_rows()
        faults = [
            (x[:, :30], q),
            (x, q[:, None]),
            (x, [0, 1]),
            (x, [-1, 1]),
            (x, [np.inf, 1]),
            (x, [np.nan, 1]),
            (x.astype(complex), q),
        ]
        bad = x.copy()
        bad[:, 0] = 2
        faults.append((bad, q))
        bad = x.copy()
        bad[0, 2] = np.inf
        faults.append((bad, q))
        faults.append((x[:0], q[:0]))
        for xx, yy in faults:
            with self.subTest(shape=np.shape(xx)), self.assertRaises(ValueError):
                solver.solve_baseline(xx, yy)

    def test_boolean_string_and_object_arrays_are_not_numeric_admission(self):
        x, q = two_rows()
        for bad in [q.astype(str), q.astype(object), np.ones(2, dtype=bool)]:
            with self.subTest(dtype=str(bad.dtype)), self.assertRaises(ValueError):
                solver.solve_baseline(x, bad)
        for bad in [x.astype(str), x.astype(object), x.astype(bool)]:
            with self.subTest(dtype=str(bad.dtype)), self.assertRaises(ValueError):
                solver.solve_baseline(bad, q)

    def test_exact_endpoint_root_has_deterministic_zero_iteration_audit(self):
        x, q = two_rows()
        prepared = solver._inputs(x, q)
        state = solver.profile_objective(np.zeros(30), x, q)
        desired = np.r_[1.0, np.zeros(29)]
        state["gradient"] = -desired
        state["hessian"] = np.eye(30)

        def quadratic(b, *_args, **_kwargs):
            return {"gradient": b - desired}

        with (
            patch.object(solver, "_profile", side_effect=quadratic),
            patch.object(solver, "_difference", return_value=-0.5),
            patch.object(
                solver,
                "brentq",
                side_effect=AssertionError("exact endpoint needs no root iterations"),
            ),
        ):
            beta, audit = solver._line_step(np.zeros(30), state, prepared)
        np.testing.assert_array_equal(beta, desired)
        self.assertEqual(audit["root_iterations"], 0)
        self.assertEqual(audit["root_function_calls"], 0)

    def test_nonzero_multiplication_underflow_rejects(self):
        x, q = two_rows()
        x[:, 3] = 1e-300
        b = np.zeros(30)
        b[2] = 1e-100
        with self.assertRaisesRegex(ValueError, "underflow"):
            solver.profile_objective(b, x, q)

    def test_budget_exhaustion_is_not_success(self):
        x, q = two_rows()
        with patch.object(solver, "MAX_ITERATIONS", 0), self.assertRaises(ValueError):
            solver.solve_baseline(x, q)
        with patch.object(solver, "MAX_BRACKET_EVALUATIONS", 0), self.assertRaises(ValueError):
            solver.solve_baseline(x, q)

    def test_success_flag_and_unchanged_root_do_not_override_gradient(self):
        x, q = two_rows()
        fake = (0.0, SimpleNamespace(converged=True, iterations=0, function_calls=1))
        with patch.object(solver, "brentq", return_value=fake), self.assertRaises(ValueError):
            solver.solve_baseline(x, q)

    def test_full_intercept_gradient_is_recomputed_and_authoritative(self):
        x, _ = two_rows()
        original = solver.full_objective

        def broken(beta, design, q):
            value, g, h = original(beta, design, q)
            g[0] = 2e-8
            return value, g, h

        with (
            patch.object(solver, "full_objective", side_effect=broken),
            self.assertRaises(ValueError),
        ):
            solver.solve_baseline(x, np.ones(2))

    def test_tolerances_and_scalar_penalty_are_fixed(self):
        self.assertEqual(solver.ALPHA, 0.01)
        self.assertEqual(solver.INTERNAL_TOLERANCE, 1e-10)
        self.assertEqual(solver.FULL_TOLERANCE, 1e-8 + 1e-12)
        self.assertEqual(solver.MAX_ITERATIONS, 500)
        self.assertEqual(solver.MAX_ROOT_ITERATIONS, 200)


if __name__ == "__main__":
    unittest.main()
