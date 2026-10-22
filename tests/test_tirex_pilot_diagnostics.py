"""Tests written before retrospective diagnostics implementation."""

import math
import unittest

import numpy as np

from scripts.tirex_pilot_diagnostics import (
    align,
    power_audit,
    summarize,
    summarize_singletons,
)


class DiagnosticTests(unittest.TestCase):
    def test_ids_align_and_reject_missing_or_duplicate(self):
        values = np.array([5.0, 7.0])
        np.testing.assert_array_equal(align(values, ["b", "a"], ["a", "b"]), [7, 5])
        for source, target in [
            (["b", "b"], ["a", "b"]),
            (["a", "b"], ["c"]),
            (["a", "b"], ["a", "a"]),
        ]:
            with self.assertRaises(ValueError):
                align(values, source, target)

    def test_both_opposite_is_not_complement_of_both_expected(self):
        baseline = np.zeros((4, 2, 3))
        swap = baseline.copy()
        swap[:, :, 1] = [[2, -2], [-2, 2], [2, 2], [0, 0]]
        out = summarize(baseline, swap, direction=1, floor=1)
        self.assertEqual(out["both_expected_count"], 1)
        self.assertEqual(out["both_opposite_count"], 1)
        self.assertEqual(out["mixed_or_zero_count"], 2)
        self.assertEqual(out["influence_successes"], 1)
        self.assertEqual(out["influence_raw_p"], 15 / 16)
        self.assertEqual(out["raw_bidirectional_mean"], 0)

    def test_direction_is_only_frozen_orientation_not_raw_sign(self):
        baseline = np.zeros((2, 2, 3))
        swap = baseline.copy()
        swap[:, :, 1] = [[-2, 2], [-4, 4]]
        out = summarize(baseline, swap, direction=-1, floor=1)
        self.assertEqual(out["raw_bidirectional_mean"], -3)
        self.assertEqual(out["oriented_bidirectional_mean"], 3)
        self.assertEqual(out["recipient_mean"], -3)
        self.assertEqual(out["donor_mean"], 3)
        self.assertEqual(out["both_expected_count"], 2)

    def test_floor_is_strict_and_binary_test_not_mean_test(self):
        baseline = np.zeros((4, 2, 3))
        swap = baseline.copy()
        swap[:, :, 1] = [[1, -1], [1, -1], [1, -1], [10, -10]]
        out = summarize(baseline, swap, direction=1, floor=1)
        self.assertEqual(out["influence_successes"], 1)
        self.assertEqual(out["influence_raw_p"], 15 / 16)
        self.assertEqual(out["oriented_bidirectional_mean"], 3.25)

    def test_float32_is_cast_before_subtraction(self):
        baseline = np.full((1, 2, 3), 100000000, dtype=np.float32)
        swap = baseline.copy()
        swap[0, 0, 1] = 100000008
        out = summarize(baseline, swap, direction=1)
        self.assertEqual(out["recipient_mean"], 8)

    def test_binomial_cutoff_exact_tail_and_power(self):
        result = power_audit(32, alpha=0.05, claims=18, power=0.8)
        k = result["first_holm_min_k"]
        exact = lambda j: sum(math.comb(32, t) for t in range(j, 33)) / 2**32
        self.assertLessEqual(exact(k), 0.05 / 18)
        self.assertGreater(exact(k - 1), 0.05 / 18)
        q = result["exceedance_probability_for_80pct_power"]
        actual = sum(
            math.comb(32, t) * q**t * (1 - q) ** (32 - t) for t in range(k, 33)
        )
        self.assertAlmostEqual(actual, 0.8, places=9)
        self.assertIsNone(result["log_variance_minimum_detectable_mean"])

    def test_singleton_component_sums_and_envelope_are_distinct(self):
        baseline = np.zeros((2, 2, 3))
        singles = np.zeros((2, 2, 2, 3))
        singles[0, :, :, 1] = [[2, -2], [0, 0]]
        singles[1, :, :, 1] = [[0, 0], [4, -4]]
        result = summarize_singletons(baseline, singles, [3, 4], direction=1)
        self.assertEqual(result["dominant_unit_by_mean_magnitude"], 4)
        self.assertEqual(result["mean_pairwise_envelope"], 3)
        self.assertEqual(result["largest_singleton_mean_magnitude"], 2)
        self.assertEqual(result["sum_raw_bidirectional_mean"], 3)

    def test_shape_and_direction_guards(self):
        base = np.zeros((2, 2, 3))
        for bad in [np.zeros((2, 3)), np.full((2, 2, 3), np.nan)]:
            with self.assertRaises(ValueError):
                summarize(base, bad, 1)
        with self.assertRaises(ValueError):
            summarize(base, base, 0)


if __name__ == "__main__":
    unittest.main()
