"""Prospective exact-family and protected-boundary contracts."""

import copy
import unittest

from src.claims_release_protocol import CONTRACT, validate


class ClaimsReleaseProtocolTests(unittest.TestCase):
    def test_fixed_two_comparison_wave_and_date_fences(self):
        validate(copy.deepcopy(CONTRACT))
        self.assertEqual(CONTRACT["wave"], 23)
        self.assertEqual(CONTRACT["comparisons"]["controls"], ["matched", "market"])
        self.assertEqual(CONTRACT["comparisons"]["cumulative_hypotheses"], 142)
        self.assertEqual(CONTRACT["forecast"]["source_end"], "2025-10-20")
        self.assertEqual(CONTRACT["sealed_start"], "2025-11-03")
        self.assertEqual(CONTRACT["forecast"]["minimum_train_releases"], 200)
        self.assertEqual(CONTRACT["inference"]["wave_alpha"], 0.05 / (23 * 24))

    def test_no_grid_timing_or_gate_mutation(self):
        for section, key, value in (
            ("forecast", "minimum_train", 999),
            ("forecast", "source_end", "2025-11-03"),
            ("inference", "bootstrap_draws", 999),
            ("inference", "minimum_phase_releases", 52),
            ("comparisons", "controls", ["market"]),
        ):
            changed = copy.deepcopy(CONTRACT)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(changed)


if __name__ == "__main__":
    unittest.main()
