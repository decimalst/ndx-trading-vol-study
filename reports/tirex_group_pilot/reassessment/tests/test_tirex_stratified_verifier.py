"""Tests precede second-implementation follow-up verifier."""

import math
import unittest

import numpy as np

from scripts.verify_tirex_stratified_pilot import (
    independent_holm,
    independent_mean_test,
    independent_tau,
)


class VerifierTests(unittest.TestCase):
    def test_effect_raw_sign_no_orientation(self):
        out = independent_mean_test(np.array([-2.0, -4.0, -6.0]), tau=0)
        self.assertEqual(out["mean"], -4)
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["se"], 2 / math.sqrt(3))

    def test_reference_two_sided_student_probability(self):
        # df2: P(|T|>=1)=1-1/sqrt(3)
        e = np.array([0.0, 1.0, 2.0]) / math.sqrt(3)
        # e mean=1/sqrt3 and SE=1/3; adjust tau to make t exactly1.
        out = independent_mean_test(e, tau=e.mean() - 1 / 3)
        self.assertAlmostEqual(out["p"], 1 - 1 / math.sqrt(3), places=12)

    def test_inside_band_is_one(self):
        self.assertEqual(
            independent_mean_test(np.array([-1.0, 0.0, 1.0]), tau=0.1)["p"], 1
        )

    def test_zero_variance_convention(self):
        self.assertEqual(independent_mean_test(np.ones(4), tau=1)["p"], 1)
        self.assertEqual(independent_mean_test(np.ones(4), tau=0.5)["p"], 0)

    def test_holm_dependency_agnostic_and_order_preserved(self):
        np.testing.assert_allclose(
            independent_holm([0.02, 0.001, 0.04]), [0.04, 0.003, 0.04]
        )
        np.testing.assert_allclose(independent_holm([0.9, 0.8]), [1, 1])

    def test_tau_largest_absolute_jitter_or_float32_spacing(self):
        baseline = np.array([-9.0, -8.0], dtype=np.float32)
        shifted = baseline.astype(float) + 0.00001
        result = independent_tau(baseline, [(baseline, shifted)])
        self.assertAlmostEqual(result["tau"], 0.0001, places=10)
        same = independent_tau(baseline, [(baseline, baseline)])
        self.assertAlmostEqual(same["tau"], 10 * np.max(np.abs(np.spacing(baseline))))

    def test_invalid_inputs_fail(self):
        for values in [[1], [1, np.nan]]:
            with self.assertRaises(ValueError):
                independent_mean_test(values, 0)
        with self.assertRaises(ValueError):
            independent_mean_test([1, 2], -1)


class ReadoutVerifierTests(unittest.TestCase):
    def test_closed_form_affine_fit_and_paired_scoring(self):
        from scripts.verify_tirex_stratified_pilot import independent_readout

        feature = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        actual = 7 + 3 * feature
        result = independent_readout(feature, actual, feature, actual)
        np.testing.assert_allclose(result["coefficients"], [7, 3])
        self.assertEqual(result["level_r2"], 1)
        # Paired activation differences are constant; R2 is undefined.
        self.assertIsNone(result["paired_difference_r2"])
        self.assertEqual(result["paired_difference_rmse"], 0)

    def test_heldout_shift_is_not_refitted(self):
        from scripts.verify_tirex_stratified_pilot import independent_readout

        feature = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 8.0]])
        actual = 7 + 3 * feature
        result = independent_readout(feature, actual, feature, actual + 1)
        self.assertAlmostEqual(result["level_rmse"], 1)
        self.assertAlmostEqual(result["paired_difference_r2"], 1)


if __name__ == "__main__":
    unittest.main()
