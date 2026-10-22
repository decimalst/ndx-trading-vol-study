"""Prewritten runner contracts: untouched donors, deterministic reference, row order."""

import unittest

import numpy as np

from src.tirex_stratified_run import (
    dose_values,
    first_per_cell,
    forecast_swap,
    make_random_reference,
    numerical_band_from_flat,
    save_cell_baselines,
)


class DoseValuesTests(unittest.TestCase):
    def setUp(self):
        self.acts = np.arange(48, dtype=np.float32).reshape(4, 2, 6)

    def test_zero_is_same_value_and_one_is_original_donor(self):
        units = [4, 1]
        np.testing.assert_array_equal(dose_values(self.acts, units, 0), self.acts[:, :, units].reshape(-1, 2))
        np.testing.assert_array_equal(dose_values(self.acts, units, 1), self.acts[:, ::-1, :][:, :, units].reshape(-1, 2))

    def test_half_and_two_are_recipient_relative_doses_not_scaled_donors(self):
        original = self.acts[:, :, [2, 0]].reshape(-1, 2)
        donor = self.acts[:, ::-1, :][:, :, [2, 0]].reshape(-1, 2)
        for alpha in (.25, .5, 2):
            np.testing.assert_array_equal(dose_values(self.acts, [2, 0], alpha), original + alpha * (donor - original))

    def test_boolean_pair_mask_preserves_bidirectional_and_member_order(self):
        mask = np.array([False, True, False, True])
        expected = self.acts[mask, ::-1, :][:, :, [5, 1]].reshape(-1, 2)
        np.testing.assert_array_equal(dose_values(self.acts, [5, 1], 1, mask), expected)

    def test_never_mutates_original_values_or_reuses_output_view(self):
        before = self.acts.copy()
        result = dose_values(self.acts, [5, 1], 2)
        result[:] = -100
        np.testing.assert_array_equal(self.acts, before)

    def test_rejects_nonfinite_negative_and_nonnumeric_dose(self):
        for alpha in (-1, np.nan, np.inf, True, "1", None):
            with self.subTest(alpha=alpha), self.assertRaises((ValueError, TypeError)):
                dose_values(self.acts, [1], alpha)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.groups = [
            {"group_id": "recency_k8", "family": "recency", "layer": 0, "units": list(range(8))},
            {"group_id": "persistence_k8", "family": "persistence", "layer": 11, "units": list(range(8, 16))},
            {"group_id": "recovery_k8", "family": "recovery", "layer": 5, "units": list(range(16, 24))},
        ]

    def test_uniform_reference_is_reproducible_nested_and_excludes_fixed_candidates(self):
        first = make_random_reference(self.groups, 19, 8119)
        self.assertEqual(first, make_random_reference(self.groups, 19, 8119))
        self.assertEqual(len(first), 114)
        for i in range(0, len(first), 2):
            four, eight = first[i:i+2]
            self.assertEqual(four['units'], eight['units'][:4])
            self.assertEqual(len(set(eight['units'])), 8)
            forbidden = set(next(g['units'] for g in self.groups if g['family'] == eight['family']))
            if eight['layer'] == 0:
                forbidden.add(103)
            self.assertFalse(forbidden.intersection(eight['units']))
            self.assertTrue(all(0 <= u < 1408 for u in eight['units']))

    def test_seed_changes_reference_and_no_source_mutation(self):
        before = [dict(g, units=g['units'].copy()) for g in self.groups]
        self.assertNotEqual(make_random_reference(self.groups, 2, 7), make_random_reference(self.groups, 2, 8))
        self.assertEqual(self.groups, before)

    def test_rng_is_uniform_choice_without_replacement_from_allowed_units(self):
        actual = make_random_reference(self.groups, 1, 27)
        rng = np.random.default_rng(27)
        for offset, group in enumerate(self.groups):
            excluded = set(group['units']) | ({103} if group['layer'] == 0 else set())
            allowed = np.array([u for u in range(1408) if u not in excluded])
            expected = rng.choice(allowed, 8, replace=False).tolist()
            self.assertEqual(actual[offset*2+1]['units'], expected)

    def test_rejects_invalid_count_seed_or_missing_top8(self):
        for count in (0, True, 1.5):
            with self.assertRaises(ValueError):
                make_random_reference(self.groups, count, 1)
        with self.assertRaises(ValueError):
            make_random_reference(self.groups[:2], 1, 1)


class SelectionAndRunTests(unittest.TestCase):
    def test_numeric_band_reshapes_interleaved_forecasts_into_complete_pairs(self):
        baseline = np.array([[-10., -9., -8.], [-11., -10., -9.]], dtype=np.float32)
        moved = baseline.copy()
        moved[0, 1] += np.float32(.0002)
        result = numerical_band_from_flat(baseline, {'repeat': moved})
        self.assertAlmostEqual(result['deadband'], 10 * float(abs(moved[0, 1] - baseline[0, 1])))
        self.assertEqual(result['n_null_evaluations'], 1)
        self.assertEqual(result['null_max_abs_errors']['repeat'], float(abs(moved[0, 1] - baseline[0, 1])))

    def test_first_per_cell_never_uses_forecast_or_activation_values(self):
        family = np.array(['a', 'b', 'a', 'a', 'b', 'b', 'a', 'b'])
        variant = np.array(['s', 'l', 'l', 's', 's', 'l', 'l', 's'])
        chosen = first_per_cell({'family': family, 'variant': variant}, 1)
        np.testing.assert_array_equal(chosen, [True, True, True, False, True, False, False, False])

    def test_cell_selection_rejects_insufficient_pairs_and_bad_count(self):
        pairs = {'family': np.array(['a', 'a']), 'variant': np.array(['s', 'l'])}
        for n in (0, True, 2):
            with self.assertRaises(ValueError):
                first_per_cell(pairs, n)

    def test_forecast_swap_uses_original_pair_rows_and_slice_matched_patches(self):
        class FakeAdapter:
            layers = (0,)

            def __init__(self):
                self.calls = []

            def forecast(self, contexts, patch=None, capture=True):
                self.calls.append((contexts.copy(), patch['values'].copy(), capture))
                result = np.column_stack((contexts[:, 0], patch['values'].sum(axis=1), contexts[:, 1]))
                return {'quantiles': result, 'activations': {}, 'residual': None}

        adapter = FakeAdapter()
        contexts = np.arange(32, dtype=np.float32).reshape(8, 4)
        acts = np.arange(48, dtype=np.float32).reshape(4, 2, 6)
        mask = np.array([False, True, False, True])
        result = forecast_swap(adapter, contexts, acts, layer=0, units=[5, 1], mask=mask, batch_size=3, alpha=.5)
        self.assertEqual(result.shape, (2, 2, 3))
        expected_rows = contexts[np.repeat(mask, 2)]
        np.testing.assert_array_equal(np.concatenate([x[0] for x in adapter.calls]), expected_rows)
        np.testing.assert_array_equal(np.concatenate([x[1] for x in adapter.calls]), dose_values(acts, [5, 1], .5, mask))
        self.assertTrue(all(not x[2] for x in adapter.calls))
        self.assertEqual([len(x[0]) for x in adapter.calls], [3, 1])

    def test_cell_archives_keep_every_baseline_row_and_matching_metadata(self):
        import tempfile
        from pathlib import Path
        pairs = {'family': np.array(['a', 'b', 'a', 'b']), 'variant': np.array(['s', 's', 'l', 'l']),
                 'pair_id': np.array(['p0', 'p1', 'p2', 'p3']), 'recipient': np.arange(16).reshape(4, 4),
                 'donor': np.arange(16, 32).reshape(4, 4), 'split': 'validation', 'window': np.array([1, 1, 2, 2])}
        output = {'quantiles': np.arange(24).reshape(8, 3), 'activations': {0: np.arange(40).reshape(8, 5)},
                  'residual': np.arange(32).reshape(8, 4)}
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_cell_baselines(Path(tmp), pairs, output)
            self.assertEqual(len(paths), 4)
            found = set()
            for path in paths:
                with np.load(path, allow_pickle=False) as saved:
                    index = int(saved['pair_id'][0][1:])
                    found.add(index)
                    np.testing.assert_array_equal(saved['quantiles'][0], output['quantiles'].reshape(4, 2, 3)[index])
                    np.testing.assert_array_equal(saved['layer_0'][0], output['activations'][0].reshape(4, 2, 5)[index])
                    np.testing.assert_array_equal(saved['recipient'][0], pairs['recipient'][index])
            self.assertEqual(found, {0, 1, 2, 3})


if __name__ == '__main__':
    unittest.main()
