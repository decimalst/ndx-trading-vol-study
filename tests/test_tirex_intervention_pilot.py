"""Prewritten contracts for the synthetic TiRex intervention pilot.

These checks exercise synthetic design and analysis without loading a model.
Pooled activation scales use population SD over all discovery pairs and sides.
Swap direction metrics exclude only declared near-zero original contrasts.
"""

import copy
import unittest

import numpy as np

from src.tirex_intervention_pilot import (
    make_pairs,
    rank_candidates,
    select_controls,
    summarize_swaps,
)


class PairDesignTests(unittest.TestCase):
    def test_shapes_and_family_balance(self):
        pairs = make_pairs("discovery", pairs_per_family=3, length=128)
        self.assertEqual(pairs["recipient"].shape, (9, 128))
        self.assertEqual(pairs["donor"].shape, (9, 128))
        self.assertEqual(pairs["family"].shape, (9,))
        self.assertEqual(pairs["pair_id"].shape, (9,))
        self.assertEqual(pairs["split"], "discovery")
        self.assertEqual(set(pairs["family"]), {"persistence", "recency", "recovery"})
        for family in ("persistence", "recency", "recovery"):
            self.assertEqual(np.count_nonzero(pairs["family"] == family), 3)

    def test_matching_preserves_distribution_and_last_observation(self):
        pairs = make_pairs("discovery", pairs_per_family=3, length=128)
        recipient, donor = pairs["recipient"], pairs["donor"]
        self.assertEqual(recipient.dtype, np.float64)
        self.assertEqual(donor.dtype, np.float64)
        np.testing.assert_array_equal(np.sort(recipient, axis=1), np.sort(donor, axis=1))
        np.testing.assert_array_equal(recipient[:, -1], donor[:, -1])
        for values in (recipient, donor):
            np.testing.assert_allclose(values.mean(axis=1), -9.0, rtol=0, atol=1e-10)
            np.testing.assert_allclose(values.std(axis=1), 0.6, rtol=0, atol=1e-10)
            self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(np.any(recipient != donor, axis=1)))

    def test_reproducible_and_disjoint_splits(self):
        first = make_pairs("discovery", pairs_per_family=3, length=128)
        second = make_pairs("discovery", pairs_per_family=3, length=128)
        held_out = make_pairs("validation", pairs_per_family=3, length=128)
        for key in ("recipient", "donor", "family", "pair_id"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(set(first["pair_id"]) & set(held_out["pair_id"]))
        self.assertEqual(len(set(first["pair_id"])), 9)
        discovery_rows = {row.tobytes() for row in first["recipient"]}
        self.assertFalse(any(row.tobytes() in discovery_rows for row in held_out["recipient"]))

    def test_malformed_generation_arguments_reject(self):
        invalid = [
            {"split": "test"},
            {"split": "discovery", "pairs_per_family": 0},
            {"split": "discovery", "pairs_per_family": -1},
            {"split": "discovery", "pairs_per_family": 1.5},
            {"split": "discovery", "pairs_per_family": True},
            {"split": "discovery", "length": 0},
            {"split": "discovery", "length": 3},
            {"split": "discovery", "length": 128.5},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                make_pairs(**kwargs)


def ranking_fixture():
    families = np.array(["persistence"] * 2 + ["recency"] * 2 + ["recovery"] * 2)
    pairs = {
        "family": families,
        "split": "discovery",
        "pair_id": np.array([f"discovery-{i}" for i in range(6)]),
        "recipient": np.zeros((6, 128)),
        "donor": np.ones((6, 128)),
    }
    values = np.zeros((6, 2, 6))
    for family_index, amplitude in enumerate((2.0, -3.0, 4.0)):
        values[2 * family_index:2 * family_index + 2, 1, family_index] = amplitude
    return pairs, {7: values}, {7: np.ones(6)}


class CandidateSelectionTests(unittest.TestCase):
    def test_one_unit_per_family_with_correct_orientation_and_scale(self):
        pairs, activations, norms = ranking_fixture()
        selected = rank_candidates(pairs, activations, norms)
        self.assertEqual(len(selected), 3)
        by_family = {candidate["family"]: candidate for candidate in selected}
        for family, unit, orientation in (
            ("persistence", 0, 1), ("recency", 1, -1), ("recovery", 2, 1)
        ):
            candidate = by_family[family]
            self.assertEqual((candidate["layer"], candidate["unit"]), (7, unit))
            self.assertEqual(candidate["orientation"], orientation)
            expected_scale = activations[7][:, :, unit].std()
            expected_delta = np.mean(
                activations[7][pairs["family"] == family, 1, unit]
                - activations[7][pairs["family"] == family, 0, unit]
            )
            self.assertAlmostEqual(candidate["activation_scale"], expected_scale)
            self.assertAlmostEqual(
                candidate["standardized_effect"], expected_delta / expected_scale
            )

    def test_rank_is_invariant_to_positive_unit_rescaling_and_zero_units(self):
        pairs, activations, norms = ranking_fixture()
        first = rank_candidates(pairs, activations, norms)
        scaled = {7: activations[7] * np.array([2.0, 3.0, 4.0, 1.0, 1.0, 1.0])}
        second = rank_candidates(pairs, scaled, norms)
        for before, after in zip(first, second, strict=True):
            self.assertEqual((before["layer"], before["unit"]), (after["layer"], after["unit"]))
            self.assertAlmostEqual(before["standardized_effect"], after["standardized_effect"])
            self.assertTrue(np.isfinite(after["activation_scale"]))
            self.assertGreater(after["activation_scale"], 0)
        self.assertEqual(rank_candidates(pairs, {7: np.zeros_like(activations[7])}, norms), [])

    def test_ties_are_deterministic_and_units_unique(self):
        pairs, activations, norms = ranking_fixture()
        activations[7][:] = 0
        activations[7][:, 1, :3] = 1
        first = rank_candidates(pairs, activations, norms)
        second = rank_candidates(copy.deepcopy(pairs), copy.deepcopy(activations), norms)
        self.assertEqual(first, second)
        self.assertEqual([candidate["unit"] for candidate in first], [0, 1, 2])
        self.assertEqual(len({candidate["family"] for candidate in first}), 3)
        self.assertEqual(len(rank_candidates(pairs, activations, norms, max_units=2)), 2)

    def test_validation_split_cannot_select_candidates(self):
        pairs, activations, norms = ranking_fixture()
        pairs["split"] = "validation"
        with self.assertRaises(ValueError):
            rank_candidates(pairs, activations, norms)

    def test_invalid_activation_shapes_and_nonfinite_values_reject(self):
        pairs, activations, norms = ranking_fixture()
        malformed = [np.zeros((5, 2, 6)), np.zeros((6, 3, 6)), np.zeros((6, 6))]
        nonfinite = activations[7].copy()
        nonfinite[0, 0, 0] = np.nan
        malformed.append(nonfinite)
        for values in malformed:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                rank_candidates(pairs, {7: values}, norms)
        with self.assertRaises(ValueError):
            rank_candidates(pairs, activations, {7: np.ones(5)})


class ControlSelectionTests(unittest.TestCase):
    @staticmethod
    def fixture():
        base = np.array([[-1.0, 1.0], [-1.0, 1.0]])[:, :, None]
        values = base * np.array([1.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0])
        candidate = {"layer": 7, "unit": 0, "activation_scale": 1.0}
        return candidate, {7: values, 9: np.ones_like(values)}, {
            7: np.array([1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0]),
            9: np.ones(7),
        }

    def test_matches_scale_and_output_norm_with_deterministic_ties(self):
        candidate, activations, norms = self.fixture()
        selected = select_controls(candidate, activations, norms, n_controls=4)
        self.assertEqual(selected, [1, 6, 2, 3])
        self.assertEqual(selected, select_controls(candidate, activations, norms, n_controls=4))

    def test_excludes_candidate_and_zero_scale_or_zero_norm_units(self):
        candidate, activations, norms = self.fixture()
        selected = select_controls(candidate, activations, norms, n_controls=20)
        self.assertEqual(len(selected), 4)
        self.assertFalse({0, 4, 5} & set(selected))

    def test_missing_layer_and_invalid_count_reject(self):
        candidate, activations, norms = self.fixture()
        with self.assertRaises((ValueError, KeyError)):
            select_controls({"layer": 99, "unit": 0}, activations, norms)
        with self.assertRaises((ValueError, TypeError)):
            select_controls(candidate, activations, norms, n_controls=-1)


class SwapSummaryTests(unittest.TestCase):
    @staticmethod
    def fixture():
        # Pair contrasts have opposite signs to test orientation, not just movement.
        base = np.array([
            [[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]],
            [[4.0, 5.0, 6.0], [0.0, 1.0, 2.0]],
        ])
        swapped = base.copy()
        swapped[0, 0] += 1.0
        swapped[0, 1] -= 2.0
        swapped[1, 0] -= 1.0
        swapped[1, 1] += 2.0
        return base, swapped

    def test_bidirectional_median_effect_metrics(self):
        base, swapped = self.fixture()
        result = summarize_swaps(base, swapped)
        self.assertAlmostEqual(result["mean_signed_median_shift"], 1.5)
        self.assertAlmostEqual(result["mean_abs_median_shift"], 1.5)
        self.assertAlmostEqual(result["mean_abs_width_shift"], 0.0)
        self.assertAlmostEqual(result["directional_agreement"], 1.0)
        self.assertAlmostEqual(result["both_direction_effect"], 1.0)
        self.assertEqual(result["near_zero_contrast_count"], 0)
        self.assertEqual(result["directional_pair_count"], 2)

    def test_width_and_unit_scaling_are_separate_from_median_direction(self):
        base, swapped = self.fixture()
        swapped[:, :, 0] -= 0.5
        swapped[:, :, 2] += 0.5
        result = summarize_swaps(base, swapped, unit_scale=2.0)
        self.assertAlmostEqual(result["mean_signed_median_shift"], 0.75)
        self.assertAlmostEqual(result["mean_abs_median_shift"], 0.75)
        self.assertAlmostEqual(result["mean_abs_width_shift"], 0.5)
        self.assertAlmostEqual(result["directional_agreement"], 1.0)

    def test_near_zero_contrast_is_reported_and_sham_is_not_an_effect(self):
        base, _ = self.fixture()
        base[1, 1] = base[1, 0] + 1e-14
        result = summarize_swaps(base, base.copy())
        self.assertEqual(result["near_zero_contrast_count"], 1)
        self.assertEqual(result["directional_pair_count"], 1)
        self.assertEqual(result["mean_abs_median_shift"], 0.0)
        self.assertEqual(result["directional_agreement"], 0.0)
        self.assertEqual(result["both_direction_effect"], 0.0)
        all_equal = np.zeros((2, 2, 3))
        result = summarize_swaps(all_equal, all_equal.copy())
        self.assertEqual(result["near_zero_contrast_count"], 2)
        self.assertEqual(result["directional_pair_count"], 0)
        self.assertIsNone(result["directional_agreement"])
        self.assertIsNone(result["both_direction_effect"])

    def test_malformed_or_nonfinite_forecasts_and_invalid_scale_reject(self):
        base, swapped = self.fixture()
        invalid = [np.zeros((2, 3)), np.zeros((2, 3, 3)), np.zeros((0, 2, 3))]
        nonfinite = swapped.copy()
        nonfinite[0, 0, 0] = np.inf
        invalid.append(nonfinite)
        for values in invalid:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                summarize_swaps(base, values)
        for scale in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                summarize_swaps(base, swapped, unit_scale=scale)


if __name__ == "__main__":
    unittest.main()
