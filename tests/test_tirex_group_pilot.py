"""Prewritten contracts for fresh synthetic groups and singleton comparisons."""

import copy
import unittest
from types import MappingProxyType
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment

from src import tirex_intervention_pilot as prior
from src.tirex_group_pilot import analyze_group, make_group_pairs, select_groups

FAMILIES = ("persistence", "recency", "recovery")
LAYERS = {"persistence": 11, "recency": 0, "recovery": 5}


class FreshPairTests(unittest.TestCase):
    def test_pair_matching_distribution_and_shape(self):
        pairs = make_group_pairs("discovery", pairs_per_family=3, length=128)
        self.assertEqual(pairs["recipient"].shape, (9, 128))
        self.assertEqual(pairs["recipient"].dtype, np.float64)
        self.assertEqual(pairs["split"], "discovery")
        self.assertEqual(set(pairs["family"]), set(FAMILIES))
        self.assertEqual(len(set(pairs["pair_id"])), 9)
        np.testing.assert_array_equal(
            np.sort(pairs["recipient"], axis=1), np.sort(pairs["donor"], axis=1)
        )
        np.testing.assert_array_equal(pairs["recipient"][:, -1], pairs["donor"][:, -1])
        for values in (pairs["recipient"], pairs["donor"]):
            np.testing.assert_allclose(values.mean(axis=1), -9, atol=1e-10, rtol=0)
            np.testing.assert_allclose(values.std(axis=1), 0.6, atol=1e-10, rtol=0)

    def test_new_splits_reproduce_and_are_disjoint_from_both_old_splits(self):
        old_rows, old_ids = set(), set()
        for split in ("discovery", "validation"):
            old = prior.make_pairs(split, 3, 128)
            old_rows.update(row.tobytes() for row in old["recipient"])
            old_rows.update(row.tobytes() for row in old["donor"])
            old_ids.update(old["pair_id"])
        generated = []
        for split in ("discovery", "validation"):
            first = make_group_pairs(split, 3, 128)
            second = make_group_pairs(split, 3, 128)
            for key in ("recipient", "donor", "family", "variant", "pair_id"):
                np.testing.assert_array_equal(first[key], second[key])
            self.assertFalse(set(first["pair_id"]) & old_ids)
            self.assertFalse(any(row.tobytes() in old_rows for row in first["recipient"]))
            self.assertFalse(any(row.tobytes() in old_rows for row in first["donor"]))
            generated.append(first)
        self.assertFalse(set(generated[0]["pair_id"]) & set(generated[1]["pair_id"]))
        discovery_rows = {row.tobytes() for row in generated[0]["recipient"]}
        self.assertFalse(any(row.tobytes() in discovery_rows for row in generated[1]["recipient"]))

    def test_declared_fresh_seeds_do_not_mutate_prior_seed_mapping(self):
        original_seeds = dict(prior.SEEDS)
        factory = np.random.default_rng
        with patch.object(prior, "SEEDS", MappingProxyType(original_seeds)):
            with patch("numpy.random.default_rng", side_effect=factory) as observed:
                make_group_pairs("discovery", 2, 128)
                make_group_pairs("validation", 2, 128)
            self.assertEqual([call.args[0] for call in observed.call_args_list], [641203, 908117])
            self.assertEqual(dict(prior.SEEDS), original_seeds)
        self.assertEqual(prior.SEEDS, original_seeds)

    def test_malformed_generation_arguments_reject(self):
        invalid = [("other", 3, 128), ("discovery", 0, 128), ("discovery", True, 128),
                   ("discovery", 2.5, 128), ("validation", 2, 3), ("validation", 2, 128.5)]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                make_group_pairs(*arguments)


def selection_fixture():
    families = np.repeat(FAMILIES, 2)
    pairs = {"split": "discovery", "family": families,
             "pair_id": np.array([f"group-discovery-{i}" for i in range(6)])}
    means = np.concatenate((np.arange(8, 0, -1, dtype=float), np.linspace(0.08, 0.7, 40)))
    baseline = np.array([-1, 1, -1, 1, -1, 1], dtype=float)
    values = np.empty((6, 2, 48))
    values[:, 0, :] = baseline[:, None]
    values[:, 1, :] = baseline[:, None] + means[None, :]
    return pairs, {layer: values.copy() for layer in LAYERS.values()}, {
        layer: np.ones(48) for layer in LAYERS.values()
    }


class GroupSelectionTests(unittest.TestCase):
    def test_fixed_layer_records_and_nested_sizes(self):
        pairs, activations, norms = selection_fixture()
        groups = select_groups(pairs, activations, norms)
        self.assertEqual([(g["family"], g["size"]) for g in groups],
                         [(family, size) for family in FAMILIES for size in (4, 8)])
        for small, large in zip(groups[::2], groups[1::2], strict=True):
            self.assertEqual(small["layer"], LAYERS[small["family"]])
            self.assertEqual(large["layer"], LAYERS[large["family"]])
            self.assertEqual(small["units"], list(range(4)))
            self.assertEqual(large["units"], list(range(8)))
            for smaller, larger in zip(small["control_groups"], large["control_groups"], strict=True):
                self.assertEqual(smaller, larger[:4])

    def test_controls_exclude_all_candidates_and_all_prior_controls(self):
        pairs, activations, norms = selection_fixture()
        groups = select_groups(pairs, activations, norms)
        for group in groups[1::2]:
            self.assertEqual(len(group["control_groups"]), 4)
            used = set(group["units"])
            for controls in group["control_groups"]:
                self.assertEqual(len(controls), 8)
                self.assertEqual(len(set(controls)), 8)
                self.assertFalse(used & set(controls))
                used.update(controls)

    def test_standardized_ranking_and_deterministic_ties(self):
        pairs, activations, norms = selection_fixture()
        first = select_groups(pairs, activations, norms)
        factors = np.linspace(0.2, 3.0, 48)
        scaled = {layer: values * factors[None, None, :] for layer, values in activations.items()}
        second = select_groups(pairs, scaled, norms)
        self.assertEqual([g["units"] for g in first], [g["units"] for g in second])
        for values in activations.values():
            values[:, :, :8] = values[:, :, 0, None]
        first = select_groups(pairs, activations, norms)
        second = select_groups(copy.deepcopy(pairs), copy.deepcopy(activations), copy.deepcopy(norms))
        self.assertEqual(first, second)
        self.assertEqual(first[1]["units"], list(range(8)))

    def test_hungarian_controls_minimize_all_three_matching_costs(self):
        pairs, activations, norms = selection_fixture()
        # Paired sign changes alter RMS contrasts independently of mean contrasts.
        for layer, values in activations.items():
            values[::2, 1, 20:30] -= 0.9
            norms[layer][8:] = np.linspace(0.3, 2.0, 40)
        groups = select_groups(pairs, activations, norms)
        for group in groups[1::2]:
            values = activations[group["layer"]]
            mask = pairs["family"] == group["family"]
            sd = values.std(axis=(0, 1))
            rms = np.sqrt(np.mean((values[mask, 1] - values[mask, 0]) ** 2, axis=0))
            features = np.column_stack((np.log(sd), np.log(norms[group["layer"]]), np.log(rms)))
            targets = np.asarray(group["units"])
            used = set(targets)
            for matched in group["control_groups"]:
                available = np.array([u for u in range(48) if u not in used])
                costs = np.sum((features[targets, None, :] - features[available][None, :, :]) ** 2, axis=2)
                rows, cols = linear_sum_assignment(costs)
                optimal = costs[rows, cols].sum()
                actual = np.sum((features[targets] - features[matched]) ** 2)
                self.assertAlmostEqual(actual, optimal, places=10)
                used.update(matched)

    def test_degenerate_units_excluded_and_insufficient_controls_reject(self):
        pairs, activations, norms = selection_fixture()
        for layer, values in activations.items():
            values[:, :, 40] = 2.0
            norms[layer][41] = 0.0
            values[:, 1, 42] = values[:, 0, 42]
        groups = select_groups(pairs, activations, norms)
        for group in groups:
            selected = group["units"] + [u for controls in group["control_groups"] for u in controls]
            self.assertFalse(set(selected) & {40, 41, 42})
        with self.assertRaises(ValueError):
            select_groups(pairs, {k: v[:, :, :39] for k, v in activations.items()},
                          {k: v[:39] for k, v in norms.items()})

    def test_validation_selection_and_malformed_arrays_reject(self):
        pairs, activations, norms = selection_fixture()
        invalid_pairs = {**pairs, "split": "validation"}
        with self.assertRaises(ValueError):
            select_groups(invalid_pairs, activations, norms)
        bad = copy.deepcopy(activations)
        bad[11][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            select_groups(pairs, bad, norms)
        with self.assertRaises(ValueError):
            select_groups(pairs, {**activations, 11: activations[11][:5]}, norms)
        with self.assertRaises(ValueError):
            select_groups(pairs, activations, {**norms, 11: norms[11][:47]})


def forecast_fixture():
    base = np.tile(np.array([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]), (4, 1, 1))
    shifts = np.array([[[0.5, -0.5], [1.0, -1.0], [0.0, 0.0], [-0.5, 0.5]],
                       [[0.25, -0.25], [0.5, -0.5], [0.25, -0.25], [0.0, 0.0]]])
    singles = base[None] + shifts[:, :, :, None]
    joint = base + shifts.sum(axis=0)[:, :, None]
    controls = np.repeat(base[None], 4, axis=0)
    return base, joint, controls, singles


class GroupAnalysisTests(unittest.TestCase):
    def test_additive_joint_has_zero_joint_minus_sum_residual(self):
        base, joint, controls, singles = forecast_fixture()
        result = analyze_group(base, joint, controls, singles, direction=1)
        np.testing.assert_array_equal(result["joint_minus_sum_median"], np.zeros((4, 2)))
        self.assertEqual(result["mean_abs_joint_minus_sum_median"], 0)
        self.assertEqual(result["largest_singleton_mean_abs"], 0.5)
        self.assertEqual(result["singleton_crossing_count"], 0)

    def test_nonadditivity_preserves_recipient_and_donor_signs(self):
        base, joint, controls, singles = forecast_fixture()
        joint += np.array([0.5, -0.25])[None, :, None]
        result = analyze_group(base, joint, controls, singles, direction=1)
        np.testing.assert_array_equal(result["joint_minus_sum_median"],
                                      np.tile([0.5, -0.25], (4, 1)))
        self.assertEqual(result["mean_abs_joint_minus_sum_median"], 0.375)

    def test_singleton_advantage_uses_pairwise_max_before_aggregation(self):
        base, joint, controls, singles = forecast_fixture()
        result = analyze_group(base, joint, controls, singles, direction=1)
        self.assertEqual(result["singleton_advantage_effects"], [0.25, 0.5, 0.0, 0.0])
        self.assertEqual(result["singleton_advantage_successes"], 2)
        self.assertEqual(result["singleton_advantage_pvalue"], 0.6875)
        self.assertEqual(result["singleton_advantage_mean"], 0.1875)
        self.assertEqual(len(result["singleton_advantage_ci95"]), 2)
        # The best singleton changes from unit 0 to unit 1 on the third pair.
        self.assertEqual(result["singleton_advantage_effects"][2], 0.0)

    def test_member_and_control_permutation_do_not_change_summary(self):
        base, joint, controls, singles = forecast_fixture()
        first = analyze_group(base, joint, controls, singles, direction=1, seed=29)
        second = analyze_group(base, joint, controls[::-1], singles[::-1], direction=1, seed=29)
        self.assertEqual(first, second)
        self.assertEqual(first, analyze_group(base, joint, controls, singles, direction=1, seed=29))

    def test_strict_floor_and_zero_swaps_do_not_count_as_success(self):
        base, joint, controls, singles = forecast_fixture()
        result = analyze_group(base, joint, controls, singles, direction=1, floor=0.25)
        self.assertEqual(result["singleton_advantage_successes"], 1)
        all_zero = analyze_group(base, base.copy(), controls,
                                 np.repeat(base[None], 2, axis=0), direction=-1)
        self.assertEqual(all_zero["singleton_advantage_successes"], 0)
        self.assertEqual(all_zero["singleton_advantage_pvalue"], 1.0)
        self.assertEqual(all_zero["singleton_advantage_ci95"], [0.0, 0.0])

    def test_invalid_singleton_inputs_and_parameters_reject(self):
        base, joint, controls, singles = forecast_fixture()
        invalid = [singles[:, :3], np.zeros((0, 4, 2, 3)), np.zeros((4, 2, 3))]
        bad = singles.copy()
        bad[0, 0, 0, 0] = np.inf
        invalid.append(bad)
        for values in invalid:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                analyze_group(base, joint, controls, values, direction=1)
        with self.assertRaises((TypeError, ValueError)):
            analyze_group(base, joint, controls, singles, direction=0)
        with self.assertRaises(ValueError):
            analyze_group(base, joint, controls, singles, direction=1, floor=-1)


if __name__ == "__main__":
    unittest.main()
