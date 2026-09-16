"""Prewritten invented-data contracts for a convex precision mixture."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult, minimize, minimize_scalar

from src.precision_gate import _objective_gradient, fit_gate, predict_gate


def literal_objective(c, base, adaptive, y, state):
    z = np.tanh(state)
    w = (1 - z) / 2 * c[0] + (1 + z) / 2 * c[1]
    p = 1 - w + w * (base / adaptive)
    return float(np.mean((y / base) * p - np.log(p)) + 0.005 * np.dot(c, c))


def synthetic():
    rng = np.random.default_rng(731249)
    base = np.exp(rng.normal(-6, 0.4, 200))
    adaptive = base * np.exp(rng.normal(0, 0.6, len(base)))
    state = rng.normal(0, 1.5, len(base))
    w = (1 - np.tanh(state)) / 2 * 0.2 + (1 + np.tanh(state)) / 2 * 0.7
    truth = 1 / ((1 - w) / base + w / adaptive)
    y = truth * np.exp(rng.normal(-0.02, 0.2, len(base)))
    return base, adaptive, y, state


class PrecisionGateTests(unittest.TestCase):
    def test_independently_constructed_interior_optimum(self):
        # Exactly solve the desired first-order equations, not producer math.
        base = np.ones(2)
        adaptive = base / 2
        state = np.arctanh(np.array([-0.5, 0.5]))
        x = np.array([[0.75, 0.25], [0.25, 0.75]])
        expected = np.array([0.25, 0.65])
        y = 1 / (1 + x @ expected) - 0.02 * np.linalg.solve(x.T, expected)
        result = fit_gate(base, adaptive, y, state)
        np.testing.assert_allclose(result["coefficients"], expected, atol=1e-7, rtol=0)
        self.assertLessEqual(result["projected_gradient_max_abs"], 1e-8)
        self.assertTrue(result["success"])

    def test_scalar_brent_and_separate_constrained_solver_oracles(self):
        b, a, y, s = synthetic()
        constant = fit_gate(b, a, y, s, constant=True)
        oracle = minimize_scalar(
            lambda c: literal_objective([c, c], b, a, y, s),
            bounds=(0, 1),
            method="bounded",
            options={"xatol": 1e-13},
        )
        np.testing.assert_allclose(constant["coefficients"], [oracle.x] * 2, atol=1e-7)
        candidate = fit_gate(b, a, y, s)
        independent = minimize(
            lambda c: literal_objective(c, b, a, y, s),
            np.array([0.5, 0.5]),
            method="SLSQP",
            bounds=[(0, 1), (0, 1)],
            options={"ftol": 1e-13, "maxiter": 1000},
        )
        self.assertTrue(independent.success)
        np.testing.assert_allclose(candidate["coefficients"], independent.x, atol=2e-6)
        self.assertLessEqual(candidate["objective"], constant["objective"] + 1e-12)

    def test_analytic_gradient_hessian_and_exact_qlike_offset(self):
        b, a, y, s = synthetic()
        x = np.column_stack([(1 - np.tanh(s)) / 2, (1 + np.tanh(s)) / 2])
        c = np.array([0.31, 0.64])
        value, gradient = _objective_gradient(c, y / b, b / a, x)
        self.assertAlmostEqual(value, literal_objective(c, b, a, y, s), places=14)
        step = 1e-5
        numeric_gradient = []
        numeric_hessian = []
        for axis in np.eye(2):
            plus, minus = c + step * axis, c - step * axis
            numeric_gradient.append(
                (literal_objective(plus, b, a, y, s) - literal_objective(minus, b, a, y, s))
                / (2 * step)
            )
            gp = _objective_gradient(plus, y / b, b / a, x)[1]
            gm = _objective_gradient(minus, y / b, b / a, x)[1]
            numeric_hessian.append((gp - gm) / (2 * step))
        np.testing.assert_allclose(gradient, numeric_gradient, atol=1e-10, rtol=1e-7)
        p = 1 - x @ c + (x @ c) * (b / a)
        g = (b / a - 1)[:, None] * x
        hessian = (g / p[:, None]).T @ (g / p[:, None]) / len(b) + 0.01 * np.eye(2)
        np.testing.assert_allclose(np.array(numeric_hessian).T, hessian, atol=1e-9, rtol=1e-7)
        self.assertGreaterEqual(np.linalg.eigvalsh(hessian).min(), 0.01)
        v = predict_gate(b, a, s, c)
        qlike = np.mean(y / v - np.log(y / v) - 1)
        self.assertAlmostEqual(
            qlike, value - 0.005 * np.dot(c, c) - np.log(y / b).mean() - 1, places=14
        )

    def test_constant_is_exactly_nested_with_correct_penalty_and_gradient(self):
        b, a, y, s = synthetic()
        result = fit_gate(b, a, y, s, constant=True)
        c = result["coefficients"]
        self.assertEqual(c[0], c[1])
        self.assertAlmostEqual(result["objective"], literal_objective(c, b, a, y, s))
        self.assertEqual(len(result["optimization_gradient"]), 1)
        self.assertAlmostEqual(result["optimization_gradient"][0], sum(result["gradient"]))
        np.testing.assert_allclose(
            predict_gate(b, a, s, c), predict_gate(b, a, s * 10 + 8, c), rtol=1e-15
        )

    def test_boundary_and_identical_expert_optima(self):
        b, a, _, s = synthetic()
        for constant in (False, True):
            for multiplier, expected in ((10.0, 0.0), (0.1, 1.0)):
                result = fit_gate(
                    np.ones(40),
                    np.full(40, 0.5),
                    np.full(40, multiplier),
                    np.linspace(-2, 2, 40),
                    constant=constant,
                )
                np.testing.assert_allclose(
                    result["coefficients"], [expected] * 2, atol=0, rtol=0
                )
                self.assertLessEqual(result["projected_gradient_max_abs"], 1e-8)
            identical = fit_gate(b, b, a, s, constant=constant)
            self.assertEqual(identical["coefficients"], [0.0, 0.0])

    def test_harmonic_endpoints_bounds_and_tiny_ratio_cancellation(self):
        b, a, _, s = synthetic()
        np.testing.assert_array_equal(predict_gate(b, a, s, [0, 0]), b)
        np.testing.assert_allclose(predict_gate(b, a, s, [1, 1]), a, rtol=2e-15)
        v = predict_gate(b, a, s, [0.2, 0.8])
        self.assertTrue(np.all(v >= np.minimum(b, a)))
        self.assertTrue(np.all(v <= np.maximum(b, a)))
        np.testing.assert_allclose(
            predict_gate([1e-300], [1.0], [0.0], [1, 1]), [1.0], rtol=1e-15
        )
        self.assertEqual(len(predict_gate([], [], [], [0, 0])), 0)

    def test_common_units_and_row_permutation(self):
        b, a, y, s = synthetic()
        expected = fit_gate(b, a, y, s)
        for scale in (1e-80, 1e80):
            actual = fit_gate(b * scale, a * scale, y * scale, s)
            np.testing.assert_allclose(
                actual["coefficients"], expected["coefficients"], atol=1e-8, rtol=0
            )
            np.testing.assert_allclose(
                predict_gate(b * scale, a * scale, s, actual["coefficients"]),
                predict_gate(b, a, s, expected["coefficients"]) * scale,
                rtol=1e-8,
            )
        perm = np.random.default_rng(72).permutation(len(b))
        actual = fit_gate(b[perm], a[perm], y[perm], s[perm])
        np.testing.assert_allclose(
            actual["coefficients"], expected["coefficients"], atol=1e-8, rtol=0
        )

    def test_audit_reconstruction_input_preservation_and_application_independence(self):
        b, a, y, s = synthetic()
        old = [x.copy() for x in (b, a, y, s)]
        fit = fit_gate(b, a, y, s)
        first = predict_gate(b[:8], a[:8], s[:8], fit["coefficients"])
        changed = s[:8].copy()
        changed[-1] += 5
        second = predict_gate(b[:8], a[:8], changed, fit["coefficients"])
        np.testing.assert_array_equal(first[:-1], second[:-1])
        self.assertEqual(fit["n_train"], len(b))
        self.assertEqual(fit["solver"], "L-BFGS-B")
        self.assertFalse(fit["constant"])
        self.assertGreaterEqual(fit["n_iterations"], 0)
        self.assertAlmostEqual(
            fit["objective"], literal_objective(fit["coefficients"], b, a, y, s), places=14
        )
        for actual, original in zip((b, a, y, s), old):
            np.testing.assert_array_equal(actual, original)

    def test_bad_inputs_flags_and_unrepresentable_ratios_rejected(self):
        for defect in ([], [True], ["1"], [complex(1, 0)], [np.nan], [np.inf], [[1.0]]):
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                fit_gate(defect, [1.0], [1.0], [0.0])
        for field in range(3):
            for defect in (0.0, -1.0, np.nan, np.inf):
                args = [np.ones(2), np.ones(2), np.ones(2), np.zeros(2)]
                args[field][0] = defect
                with self.subTest(field=field, defect=defect), self.assertRaises(ValueError):
                    fit_gate(*args)
        for flag in (1, None, "constant"):
            with self.assertRaises(ValueError):
                fit_gate([1], [1], [1], [0], constant=flag)
        for b, a in ((1e308, 1e-308), (1e-308, 1e308)):
            with self.assertRaises(ValueError):
                predict_gate([b], [a], [0], [0, 0])
        with self.assertRaises(ValueError):
            fit_gate([1e-308], [1e-308], [1e308], [0])
        with self.assertRaises(ValueError):
            fit_gate([1], [1], [1], [np.nan])

    def test_bad_coefficients_and_prediction_shapes_rejected(self):
        for c in (
            [0],
            [0, 0, 0],
            [-1, 0],
            [0, 1.0001],
            [np.nan, 0],
            [True, False],
            [[0, 0]],
            ["0", "0"],
        ):
            with self.subTest(c=c), self.assertRaises(ValueError):
                predict_gate([1], [2], [0], c)
        with self.assertRaises(ValueError):
            predict_gate([1, 2], [2], [0], [0, 0])

    def test_exact_optimizer_contract_no_failure_or_stationarity_fallback(self):
        b, a, y, s = synthetic()
        with patch("src.precision_gate.minimize", wraps=minimize) as wrapped:
            fit_gate(b, a, y, s)
        self.assertEqual(wrapped.call_count, 1)
        kwargs = wrapped.call_args.kwargs
        self.assertEqual(kwargs["method"], "L-BFGS-B")
        self.assertEqual(kwargs["bounds"], [(0.0, 1.0), (0.0, 1.0)])
        self.assertEqual(kwargs["options"], {"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000})
        self.assertIs(kwargs["jac"], True)
        for result in (
            OptimizeResult(success=False, x=np.zeros(2), nit=1, message="injected failure"),
            OptimizeResult(success=True, x=np.zeros(2), nit=1, message="false convergence"),
            OptimizeResult(success=True, x=np.array([1.1, 0.0]), nit=1, message="bad bound"),
        ):
            with patch("src.precision_gate.minimize", return_value=result) as mocked:
                with self.assertRaises(ValueError):
                    fit_gate(b, a, y, s)
                self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
