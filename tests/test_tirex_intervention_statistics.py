"""Prewritten statistical contracts, independent of model loading or core helpers."""

import unittest

import numpy as np

from src.tirex_intervention_statistics import analyze_candidate, holm_adjust


def example_forecasts():
    base = np.tile(np.array([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]), (4, 1, 1))
    swapped = base.copy()
    shifts = np.array([[1.0, -1.0], [2.0, -2.0], [-1.0, 1.0], [0.0, 0.0]])
    swapped += shifts[:, :, None]
    controls = np.repeat(base[None, :, :, :], 2, axis=0)
    controls[0] += np.array([0.5, -0.5])[None, :, None]
    controls[1] += np.array([0.25, -0.25])[None, :, None]
    return base, swapped, controls


class CandidateStatisticsTests(unittest.TestCase):
    def test_exact_bidirectional_effect_and_binomial_tail(self):
        base, swapped, controls = example_forecasts()
        result = analyze_candidate(base, swapped, controls, direction=1, seed=19)
        self.assertEqual(result["n_pairs"], 4)
        self.assertEqual(result["n_controls"], 2)
        self.assertEqual(result["influence_effects"], [1.0, 2.0, -1.0, 0.0])
        self.assertEqual(result["influence_successes"], 2)
        self.assertEqual(result["influence_pvalue"], 0.6875)
        self.assertEqual(result["mean_influence_effect"], 0.5)
        self.assertEqual(result["both_directions_expected_sign_fraction"], 0.5)

    def test_controls_are_matched_by_pair_before_averaging(self):
        base, swapped, controls = example_forecasts()
        result = analyze_candidate(base, swapped, controls, direction=1, seed=19)
        self.assertEqual(result["advantage_effects"], [0.625, 1.625, 0.625, -0.375])
        self.assertEqual(result["advantage_successes"], 3)
        self.assertEqual(result["advantage_pvalue"], 0.3125)
        self.assertEqual(result["mean_advantage_effect"], 0.625)
        self.assertEqual(result["mean_abs_control_median_shift"], 0.375)

    def test_negative_direction_reverses_influence_but_not_advantage(self):
        base, swapped, controls = example_forecasts()
        positive = analyze_candidate(base, swapped, controls, direction=1, seed=19)
        negative = analyze_candidate(base, swapped, controls, direction=-1, seed=19)
        self.assertEqual(negative["influence_effects"], [-1.0, -2.0, 1.0, 0.0])
        self.assertEqual(negative["influence_successes"], 1)
        self.assertEqual(negative["influence_pvalue"], 0.9375)
        self.assertEqual(negative["advantage_effects"], positive["advantage_effects"])
        self.assertEqual(negative["both_directions_expected_sign_fraction"], 0.25)

    def test_success_uses_strict_floor_including_zero_swaps(self):
        base, _, controls = example_forecasts()
        swapped = base.copy()
        # Exact binary fractions avoid accidental rounding around the floor.
        swapped += np.array([[0.5, -0.5], [1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]])[:, :, None]
        result = analyze_candidate(base, swapped, controls, direction=1, floor=0.5)
        self.assertEqual(result["influence_successes"], 1)
        sham = analyze_candidate(base, base.copy(), controls, direction=1)
        self.assertEqual(sham["influence_successes"], 0)
        self.assertEqual(sham["influence_pvalue"], 1.0)
        self.assertEqual(sham["both_directions_expected_sign_fraction"], 0.0)

    def test_width_and_raw_direction_summaries_are_distinct(self):
        base, swapped, controls = example_forecasts()
        swapped[:, :, 0] -= 0.5
        swapped[:, :, 2] += 0.5
        result = analyze_candidate(base, swapped, controls, direction=1)
        self.assertEqual(result["mean_native_donor_recipient_contrast"], 4.0)
        self.assertEqual(result["mean_recipient_median_shift"], 0.5)
        self.assertEqual(result["mean_donor_median_shift"], -0.5)
        self.assertEqual(result["mean_abs_median_shift"], 1.0)
        self.assertEqual(result["mean_abs_width_shift"], 1.0)
        self.assertEqual(result["crossing_count"], 0)

    def test_quantile_crossings_are_reported_without_repair(self):
        base, swapped, controls = example_forecasts()
        swapped[0, 0] = [3.0, 2.0, 1.0]
        controls[0, 0, 0] = [3.0, 2.0, 1.0]
        result = analyze_candidate(base, swapped, controls, direction=1)
        self.assertEqual(result["crossing_count"], 1)
        self.assertEqual(result["base_crossing_count"], 0)
        self.assertEqual(result["control_crossing_count"], 1)
        np.testing.assert_array_equal(swapped[0, 0], [3.0, 2.0, 1.0])

    def test_bootstrap_reproducibility_and_degenerate_interval(self):
        base, swapped, controls = example_forecasts()
        first = analyze_candidate(base, swapped, controls, direction=1, seed=91)
        second = analyze_candidate(base, swapped, controls, direction=1, seed=91)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_replicates"], 5000)
        swapped = base + np.array([1.0, -1.0])[None, :, None]
        result = analyze_candidate(base, swapped, controls, direction=1, seed=91)
        self.assertEqual(result["influence_mean_ci95"], [1.0, 1.0])
        self.assertEqual(result["advantage_mean_ci95"], [0.625, 0.625])

    def test_all_outputs_are_json_serializable_without_nan(self):
        import json

        base, swapped, controls = example_forecasts()
        result = analyze_candidate(base, swapped, controls, direction=1)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_invalid_shape_nonfinite_values_and_parameters_reject(self):
        base, swapped, controls = example_forecasts()
        for malformed in (swapped[:3], np.zeros((4, 3)), np.zeros((0, 2, 3))):
            with self.subTest(shape=malformed.shape), self.assertRaises(ValueError):
                analyze_candidate(base, malformed, controls, direction=1)
        for malformed in (controls[:, :3], np.zeros((0, 4, 2, 3)), np.zeros((4, 2, 3))):
            with self.subTest(shape=malformed.shape), self.assertRaises(ValueError):
                analyze_candidate(base, swapped, malformed, direction=1)
        bad = swapped.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            analyze_candidate(base, bad, controls, direction=1)
        for direction in (0, 2, np.nan, True):
            with self.subTest(direction=direction), self.assertRaises((ValueError, TypeError)):
                analyze_candidate(base, swapped, controls, direction=direction)
        for floor in (-1.0, np.nan, np.inf):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                analyze_candidate(base, swapped, controls, direction=1, floor=floor)


class HolmAdjustmentTests(unittest.TestCase):
    def test_unsorted_values_preserve_input_order_and_monotonicity(self):
        self.assertEqual(holm_adjust([0.03, 0.01, 0.04, 0.2]), [0.09, 0.04, 0.09, 0.2])

    def test_ties_edges_and_empty_family(self):
        self.assertEqual(holm_adjust([0.0, 0.02, 0.02, 1.0]), [0.0, 0.06, 0.06, 1.0])
        self.assertEqual(holm_adjust([0.4, 0.5, 0.9]), [1.0, 1.0, 1.0])
        self.assertEqual(holm_adjust([]), [])
        self.assertEqual(holm_adjust([0.123]), [0.123])

    def test_invalid_pvalues_reject(self):
        for invalid in ([np.nan], [np.inf], [-0.01], [1.01], [[0.1, 0.2]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                holm_adjust(invalid)


if __name__ == "__main__":
    unittest.main()
