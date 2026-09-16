"""Prewritten generated weighted expert arithmetic contracts."""

import unittest

import numpy as np
import pandas as pd

from src.precision_gate_models import MARKET, fit_experts


def model_fixture():
    rng = np.random.default_rng(715)
    x = np.column_stack((np.ones(90), rng.normal(size=(90, 11))))
    frame = pd.DataFrame(x, columns=MARKET)
    y = np.exp(-5 + x[:, 1] * 0.2 - x[:, 3] * 0.13 + rng.normal(0, 0.3, 90))
    return frame, y, np.arange(90) * 2


class PrecisionExpertTests(unittest.TestCase):
    def test_weighted_raw_normal_equations_and_exact_separate_smearing(self):
        frame, y, positions = model_fixture()
        query = frame.iloc[-4:].copy()
        got = fit_experts(
            frame,
            y,
            query,
            train_positions=positions,
            cutoff_position=181,
            adaptive_half_life=30,
        )
        x, z = frame.to_numpy(), np.log(y)
        for arm, weights in (
            ("base", np.ones(90)),
            ("adaptive", np.exp2(-(181 - positions) / 30)),
        ):
            beta = np.linalg.solve(x.T @ (weights[:, None] * x), x.T @ (weights * z))
            residual = z - x @ beta
            smear = np.sum(weights * np.exp(residual)) / np.sum(weights)
            expected = np.exp(query.to_numpy() @ beta) * smear
            np.testing.assert_allclose(got["predictions"][arm], expected, rtol=2e-12)
            audit = got["fits"][arm]
            np.testing.assert_allclose(audit["weights"], weights, rtol=0, atol=0)
            self.assertAlmostEqual(audit["log_smearing"], np.log(smear), places=12)
            means = np.average(x[:, 1:], axis=0, weights=weights)
            scales = np.sqrt(np.average((x[:, 1:] - means) ** 2, axis=0, weights=weights))
            np.testing.assert_allclose(list(audit["feature_means"].values()), means)
            np.testing.assert_allclose(list(audit["feature_scales"].values()), scales)

    def test_exact_log_linear_reconstruction_and_no_application_fit_influence(self):
        frame, _, pos = model_fixture()
        y = np.exp(-3 + frame.iloc[:, 1].to_numpy() * 0.4)
        query = frame.iloc[:3].copy()
        got = fit_experts(
            frame, y, query, train_positions=pos, cutoff_position=181, adaptive_half_life=30
        )
        query.iloc[:, 1:] *= 5
        changed = fit_experts(
            frame, y, query, train_positions=pos, cutoff_position=181, adaptive_half_life=30
        )
        self.assertEqual(got["fits"], changed["fits"])
        for arm in ("base", "adaptive"):
            np.testing.assert_allclose(got["predictions"][arm], y[:3], rtol=2e-12)

    def test_rank_scale_and_position_failures_have_no_fallback(self):
        frame, y, pos = model_fixture()
        for kind in ("rank", "scale", "future", "duplicate", "underflow"):
            x, p, cutoff = frame.copy(), pos.copy(), 181
            if kind == "rank":
                x.iloc[:, 2] = x.iloc[:, 1]
            if kind == "scale":
                x.iloc[:, 2] = 2.0
            if kind == "future":
                p[-1] = 182
            if kind == "duplicate":
                p[-1] = p[-2]
            if kind == "underflow":
                cutoff = 1000000
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                fit_experts(
                    x,
                    y,
                    x.iloc[:2],
                    train_positions=p,
                    cutoff_position=cutoff,
                    adaptive_half_life=30,
                )

    def test_typed_and_aligned_inputs_and_immutability(self):
        frame, y, pos = model_fixture()
        before = frame.copy(deep=True)
        fit_experts(
            frame,
            y,
            frame.iloc[:2],
            train_positions=pos,
            cutoff_position=181,
            adaptive_half_life=30,
        )
        pd.testing.assert_frame_equal(frame, before)
        for bad in (np.zeros(90), np.full(90, np.inf), np.array([True] * 90)):
            with self.assertRaises(ValueError):
                fit_experts(
                    frame,
                    bad,
                    frame.iloc[:2],
                    train_positions=pos,
                    cutoff_position=181,
                    adaptive_half_life=30,
                )


if __name__ == "__main__":
    unittest.main()
