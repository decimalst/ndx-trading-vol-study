"""Prewritten proper-variance score and stable-gap contracts."""

import unittest
from decimal import Decimal, localcontext

import numpy as np

from src import civil_quarter_score as score


class CivilQuarterScore(unittest.TestCase):
    def test_loss_elicits_positive_conditional_mean(self):
        y = np.array([0.2, 0.8, 2.0])
        mean = float(y.mean())
        optimum = score.proper_score(y, np.full(3, mean)).mean()
        for factor in [0.2, 0.8, 1.2, 5.0]:
            self.assertGreater(
                score.proper_score(y, np.full(3, mean * factor)).mean(), optimum
            )

    def test_paired_gap_matches_decimal_functional_not_rounded_loss_subtraction(self):
        cases = [
            (1e-4, 1.00000001e-4, 1.00000002e-4),
            (1.0, 1e150, 1e-150),
            (1e-250, 1e-320, 2e-320),
            (0.02, 0.003, 0.01),
            (1.0, 1.0, np.nextafter(1.0, 2.0)),
        ]
        for y, a, b in cases:
            with localcontext() as ctx:
                ctx.prec = 100
                q, x, z = map(Decimal.from_float, [y, a, b])
                expected = float(x.ln() - z.ln() + q / x - q / z)
            actual = score.paired_difference([a], [b], [y])[0]
            with self.subTest(y=y, a=a, b=b):
                self.assertAlmostEqual(actual / expected, 1.0, delta=2e-7)

    def test_identical_predictions_give_exact_zero_and_swapping_reverses_gap(self):
        y = np.array([0.01, 0.04, 0.03])
        a = np.array([0.02, 0.01, 0.04])
        b = np.array([0.01, 0.04, 0.02])
        np.testing.assert_array_equal(score.paired_difference(a, a, y), np.zeros(3))
        np.testing.assert_array_equal(
            score.paired_difference(a, b, y), -score.paired_difference(b, a, y)
        )

    def test_common_units_change_levels_not_pair_gaps(self):
        y, a, b = np.array([0.2, 0.8]), np.array([0.3, 0.6]), np.array([0.25, 0.7])
        gap = score.paired_difference(a, b, y)
        for scale in [1e-100, 1e100]:
            np.testing.assert_allclose(
                score.paired_difference(a * scale, b * scale, y * scale),
                gap,
                rtol=1e-12,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                score.proper_score(y * scale, a * scale) - np.log(scale),
                score.proper_score(y, a),
                rtol=1e-12,
                atol=1e-14,
            )

    def test_native_subnormal_forecasts_do_not_require_inverse_forecasts(self):
        h = np.array([1e-320, 2e-320])
        y = np.array([1e-250, 1e-250])
        self.assertTrue(np.isfinite(score.proper_score(y, h)).all())
        self.assertTrue(np.isfinite(score.paired_difference(h, h[::-1], y)).all())

    def test_invalid_real_domain_alignment_and_underflow_reject(self):
        for y, h in [
            ([0.0], [1.0]),
            ([-1.0], [1.0]),
            ([1.0], [0.0]),
            ([1.0], [np.inf]),
            ([np.nan], [1.0]),
            ([1 + 1j], [1.0]),
            ([1e-300], [1e300]),
            ([1e300], [1e-300]),
            ([1.0, 2.0], [1.0]),
            ([[1.0]], [[1.0]]),
        ]:
            with self.subTest(y=y, h=h), self.assertRaises(ValueError):
                score.proper_score(y, h)
        with self.assertRaises(ValueError):
            score.paired_difference([1.0], [1.0, 2.0], [1.0])

    def test_coherence_is_scaled_to_components_without_unit_floor(self):
        score.require_coherence(np.array([1e-200]), np.array([1e-200]), [np.array([1e-200])])
        with self.assertRaises(ValueError):
            score.require_coherence(
                np.array([2e-200]), np.array([1e-200]), [np.array([1e-200])]
            )


if __name__ == "__main__":
    unittest.main()
