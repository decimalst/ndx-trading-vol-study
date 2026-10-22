"""Tests before the explicitly retrospective sensitivity extractor."""

import unittest

import numpy as np

from scripts.tirex_stratified_postdiagnostics import effect_diagnostics


class PostdiagnosticTests(unittest.TestCase):
    def test_negative_paired_effect_does_not_mean_both_swap_directions(self):
        base = np.zeros((3, 2, 3))
        q = base.copy()
        q[:, :, 1] = [[-1, 1], [-1, 0], [-1, -0.5]]
        row = effect_diagnostics(base, q, ["a", "b", "c"], tau=0.1)
        self.assertEqual(row["negative_raw_paired_effect_count"], 3)
        self.assertEqual(row["both_recipient_negative_donor_positive_strict_zero"], 1)
        self.assertEqual(row["both_negative_raw_orientation_beyond_tau"], 1)
        self.assertEqual(row["mixed_or_one_neutral_beyond_tau"], 2)

    def test_tau_boundary_is_neutral(self):
        base = np.zeros((2, 2, 3))
        q = base.copy()
        q[:, :, 1] = [[-0.1, 0.1], [-0.2, 0.2]]
        row = effect_diagnostics(base, q, ["a", "b"], tau=0.1)
        self.assertEqual(row["both_recipient_negative_donor_positive_strict_zero"], 2)
        self.assertEqual(row["both_negative_raw_orientation_beyond_tau"], 1)

    def test_largest_absolute_exclusion_is_named_as_descriptive(self):
        base = np.zeros((3, 2, 3))
        q = base.copy()
        q[:, :, 1] = [[-100, 100], [-1, 1], [-1, 1]]
        row = effect_diagnostics(base, q, ["large", "b", "c"], tau=0.1)
        self.assertEqual(row["mean_e"], -34)
        self.assertEqual(row["mean_without_one_largest_abs"], -1)
        self.assertEqual(row["largest_abs_pair_ids"][0], "large")
        self.assertIn("Post-result", row["descriptive_note"])


if __name__ == "__main__":
    unittest.main()
