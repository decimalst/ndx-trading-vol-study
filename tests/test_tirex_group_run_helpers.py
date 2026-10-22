"""Prewritten row-alignment contracts for grouped donor activation patches."""

import unittest

import numpy as np

from src.tirex_group_run import donor_values


class DonorValuesTests(unittest.TestCase):
    def setUp(self):
        self.activations = np.arange(30, dtype=np.float64).reshape(3, 2, 5)

    def test_bidirectional_interleaving_preserves_unsorted_member_order(self):
        result = donor_values(self.activations, [3, 1])
        expected = np.array([[8, 6], [3, 1], [18, 16], [13, 11], [28, 26], [23, 21]])
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (6, 2))

    def test_pair_mask_selects_both_directions_in_original_pair_order(self):
        result = donor_values(self.activations, [4, 0], mask=np.array([True, False, True]))
        expected = np.array([[9, 5], [4, 0], [29, 25], [24, 20]])
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (4, 2))

    def test_input_values_and_member_and_mask_arrays_are_not_mutated(self):
        units = [3, 1]
        mask = np.array([False, True, True])
        saved_activations, saved_mask, saved_units = self.activations.copy(), mask.copy(), units.copy()
        result = donor_values(self.activations, units, mask=mask)
        result[0, 0] = -1000
        np.testing.assert_array_equal(self.activations, saved_activations)
        np.testing.assert_array_equal(mask, saved_mask)
        self.assertEqual(units, saved_units)

    def test_no_unrequested_pair_filtering_occurs(self):
        self.activations[1] = 0.0
        result = donor_values(self.activations, [2])
        self.assertEqual(result.shape, (6, 1))
        np.testing.assert_array_equal(result[2:4], np.zeros((2, 1)))
        np.testing.assert_array_equal(result[:, 0], [7, 2, 0, 0, 27, 22])

    def test_malformed_or_nonfinite_activations_and_bad_members_reject(self):
        bad = self.activations.copy()
        bad[1, 1, 2] = np.inf
        invalid = [np.zeros((3, 5)), np.zeros((3, 3, 5)), np.zeros((0, 2, 5)), bad]
        for values in invalid:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                donor_values(values, [1, 2])
        for units in ([], [1, 1], [True], [5], [-1], [1.5]):
            with self.subTest(units=units), self.assertRaises((TypeError, ValueError)):
                donor_values(self.activations, units)

    def test_mask_requires_boolean_shape_and_at_least_one_pair(self):
        masks = [np.array([True, False]), np.array([[True, False, True]]),
                 np.array([1, 0, 1]), np.array([False, False, False])]
        for mask in masks:
            with self.subTest(mask=mask.tolist()), self.assertRaises(ValueError):
                donor_values(self.activations, [1, 2], mask=mask)


if __name__ == "__main__":
    unittest.main()
