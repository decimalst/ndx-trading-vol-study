"""Contracts written before the fresh, motif-stratified follow-up implementation."""

import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import binomtest, norm, t

from src import tirex_group_pilot as previous_group
from src import tirex_intervention_pilot as previous_single
from src.tirex_stratified_core import (
    calibrate_deadband,
    make_stratified_pairs,
    minimum_detectable_mean,
    summarize_swaps,
)


class StratifiedGeneratorTests(unittest.TestCase):
    def test_exact_balance_fixed_windows_and_unique_ids(self):
        for split in ("calibration", "validation"):
            pairs = make_stratified_pairs(split, per_cell=3, length=256)
            self.assertEqual(pairs["split"], split)
            self.assertEqual(pairs["recipient"].shape, (18, 256))
            self.assertEqual(pairs["donor"].shape, (18, 256))
            self.assertEqual(len(set(pairs["pair_id"])), 18)
            for family in ("persistence", "recency", "recovery"):
                for variant, width in (("short", 76), ("long", 180)):
                    mask = (pairs["family"] == family) & (pairs["variant"] == variant)
                    self.assertEqual(int(mask.sum()), 3)
                    np.testing.assert_array_equal(pairs["window"][mask], width)
                    self.assertTrue(all(split in identifier for identifier in pairs["pair_id"][mask]))

    def test_histogram_last_point_and_affine_moments_are_conditioned(self):
        pairs = make_stratified_pairs("validation", per_cell=4, length=2048)
        recipient, donor = pairs["recipient"], pairs["donor"]
        self.assertEqual(recipient.dtype, np.float64)
        np.testing.assert_array_equal(np.sort(recipient, axis=1), np.sort(donor, axis=1))
        np.testing.assert_array_equal(recipient[:, -1], donor[:, -1])
        for values in (recipient, donor):
            np.testing.assert_allclose(values.mean(axis=1), -9, atol=1e-12, rtol=0)
            np.testing.assert_allclose(values.std(axis=1), .6, atol=1e-12, rtol=0)
        for index, width in enumerate(pairs["window"]):
            np.testing.assert_array_equal(recipient[index, :-width - 1], donor[index, :-width - 1])

    def test_splits_use_same_parameter_law_and_recency_quiet_tail(self):
        for split in ("calibration", "validation"):
            pairs = make_stratified_pairs(split, per_cell=5, length=256)
            self.assertTrue(np.all((pairs["amplitude"] >= .8) & (pairs["amplitude"] <= 2.8)))
            self.assertTrue(np.all((pairs["noise_sd"] >= .04) & (pairs["noise_sd"] <= .3)))
            self.assertGreater(np.unique(pairs["amplitude"]).size, 1)
            self.assertGreater(np.unique(pairs["noise_sd"]).size, 1)
            for index in np.flatnonzero(pairs["family"] == "recency"):
                tail = max(2, int(pairs["window"][index]) // 6)
                np.testing.assert_array_equal(pairs["recipient"][index, -tail - 1:],
                                              pairs["donor"][index, -tail - 1:])

    def test_reproducible_disjoint_from_both_previous_pilots(self):
        old_hashes, old_ids = set(), set()
        for factory in (previous_single.make_pairs, previous_group.make_group_pairs):
            for split in ("discovery", "validation"):
                old = factory(split, 3, 2048)
                old_ids.update(old["pair_id"])
                old_hashes.update(hashlib.sha256(row.tobytes()).hexdigest()
                                  for side in ("recipient", "donor") for row in old[side])
        fresh_sets = []
        for split in ("calibration", "validation"):
            first = make_stratified_pairs(split, per_cell=3)
            second = make_stratified_pairs(split, per_cell=3)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            fresh_hashes = {hashlib.sha256(row.tobytes()).hexdigest()
                            for side in ("recipient", "donor") for row in first[side]}
            self.assertFalse(fresh_hashes & old_hashes)
            self.assertFalse(set(first["pair_id"]) & old_ids)
            fresh_sets.append(fresh_hashes)
        self.assertFalse(fresh_sets[0] & fresh_sets[1])

    def test_new_seeds_and_unchanged_old_seed_mappings(self):
        old_single, old_group = copy.deepcopy(previous_single.SEEDS), copy.deepcopy(previous_group.SEEDS)
        factory = np.random.default_rng
        with patch("numpy.random.default_rng", side_effect=factory) as observed:
            make_stratified_pairs("calibration", per_cell=1, length=256)
            make_stratified_pairs("validation", per_cell=1, length=256)
        self.assertEqual([call.args[0] for call in observed.call_args_list], [837194, 529086])
        self.assertEqual(old_single, previous_single.SEEDS)
        self.assertEqual(old_group, previous_group.SEEDS)

    def test_invalid_arguments_reject(self):
        invalid = [("discovery", 2, 256), ("validation", 0, 256), ("calibration", True, 256),
                   ("validation", 2.5, 256), ("validation", 2, 180), ("validation", 2, 256.0)]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                make_stratified_pairs(*arguments)


def swap_fixture(effects):
    effects = np.asarray(effects, dtype=float)
    base = np.tile(np.array([[-10., -9., -8.], [-10., -9., -8.]]), (len(effects), 1, 1))
    swapped = base.copy()
    swapped[:, 0] += effects[:, None]
    swapped[:, 1] -= effects[:, None]
    return base, swapped


class StratifiedSummaryTests(unittest.TestCase):
    def test_raw_sign_and_bidirectional_effects_have_no_frozen_orientation(self):
        base, swapped = swap_fixture([.1, .2, -.1, 0])
        result = summarize_swaps(base, swapped, .01, seed=4, n_bootstrap=100)
        np.testing.assert_allclose(result["e"], [.1, .2, -.1, 0])
        self.assertAlmostEqual(result["mean_e"], .05)
        self.assertAlmostEqual(result["mean_recipient_median_shift"], .05)
        self.assertAlmostEqual(result["mean_donor_median_shift"], -.05)
        self.assertAlmostEqual(result["mean_abs_median_shift"], .1)
        self.assertEqual(result["n_pairs"], 4)
        self.assertEqual(result["sign_counts"], {"positive": 2, "negative": 1, "neutral": 1,
                                                "nonneutral": 3, "total": 4})
        self.assertEqual(result["sign_pvalue"], binomtest(2, 3, .5).pvalue)

    def test_primary_t_test_against_band_reconstructs_exactly(self):
        base, swapped = swap_fixture([.01, .02, .03, .05, .06])
        result = summarize_swaps(base, swapped, .005, n_bootstrap=100)
        effects = np.asarray(result["e"])
        sd, mean = effects.std(ddof=1), effects.mean()
        se = sd / np.sqrt(len(effects))
        expected = min(1., 2 * min(t.sf((mean - .005) / se, 4),
                                   t.sf((-mean - .005) / se, 4)))
        self.assertAlmostEqual(result["mean_e_sd"], sd)
        self.assertAlmostEqual(result["mean_e_se"], se)
        self.assertAlmostEqual(result["primary_pvalue"], expected)

    def test_reversed_effect_has_same_primary_pvalue_opposite_estimate(self):
        base, swapped = swap_fixture([.01, .02, .03, .05])
        positive = summarize_swaps(base, swapped, .001, n_bootstrap=100)
        negative = summarize_swaps(base, 2 * base - swapped, .001, n_bootstrap=100)
        self.assertAlmostEqual(positive["mean_e"], -negative["mean_e"])
        self.assertAlmostEqual(positive["primary_pvalue"], negative["primary_pvalue"])
        self.assertEqual(negative["sign_counts"]["negative"], 4)

    def test_degenerate_cases_and_exact_deadband_boundaries(self):
        for effect, deadband, expected in ((.125, .125, 1.), (0., 0., 1.), (.25, .125, 0.),
                                           (-.25, .125, 0.)):
            base, swapped = swap_fixture([effect] * 4)
            result = summarize_swaps(base, swapped, deadband, n_bootstrap=100)
            self.assertEqual(result["primary_pvalue"], expected)
            self.assertEqual(result["mean_e_se"], 0.)
            if abs(effect) <= deadband:
                self.assertEqual(result["sign_counts"]["neutral"], 4)
                self.assertEqual(result["sign_pvalue"], 1.)
        base, swapped = swap_fixture([-.125, .125, -.25, .25])
        result = summarize_swaps(base, swapped, .125, n_bootstrap=100)
        self.assertEqual(result["sign_counts"], {"positive": 1, "negative": 1, "neutral": 2,
                                                "nonneutral": 2, "total": 4})

    def test_quantiles_width_crossings_and_separate_side_metrics(self):
        base, swapped = swap_fixture([0, 0, 0, 0])
        swapped[:, 0] += [-.25, .5, .75]
        swapped[:, 1] += [.25, -.5, -.75]
        result = summarize_swaps(base, swapped, .01, n_bootstrap=100)
        np.testing.assert_allclose(result["mean_quantile_e"], [-.25, .5, .75])
        np.testing.assert_allclose(result["mean_recipient_quantile_shift"], [-.25, .5, .75])
        np.testing.assert_allclose(result["mean_donor_quantile_shift"], [.25, -.5, -.75])
        self.assertAlmostEqual(result["mean_width_e"], 1.)
        self.assertAlmostEqual(result["mean_abs_width_shift"], 1.)
        self.assertEqual(result["base_crossing_count"], 0)
        self.assertEqual(result["crossing_count"], 0)
        swapped[0, 0, 0] = 100
        result = summarize_swaps(base, swapped, .01, n_bootstrap=100)
        self.assertEqual(result["crossing_count"], 1)

    def test_bootstrap_resamples_pairs_preserving_sides_quantiles_and_seed(self):
        base, swapped = swap_fixture([-.25, .125, .25, .5])
        saved_base, saved_swapped = base.copy(), swapped.copy()
        result = summarize_swaps(base, swapped, .01, seed=7, n_bootstrap=199)
        indices = np.random.default_rng(7).integers(0, 4, size=(199, 4))
        effects = np.asarray(result["e"])
        expected_ci = np.quantile(effects[indices].mean(axis=1), [.025, .975])
        np.testing.assert_array_equal(result["mean_e_ci95"], expected_ci)
        np.testing.assert_array_equal(result["recipient_median_ci95"], expected_ci)
        np.testing.assert_array_equal(result["donor_median_ci95"], -expected_ci[::-1])
        self.assertEqual(result, summarize_swaps(base, swapped, .01, seed=7, n_bootstrap=199))
        np.testing.assert_array_equal(base, saved_base)
        np.testing.assert_array_equal(swapped, saved_swapped)

    def test_strata_kept_separate_even_with_unequal_counts(self):
        base, swapped = swap_fixture([.25] * 3 + [-.125] * 7)
        short = summarize_swaps(base[:3], swapped[:3], .01, n_bootstrap=100)
        long = summarize_swaps(base[3:], swapped[3:], .01, n_bootstrap=100)
        self.assertEqual((short["n_pairs"], long["n_pairs"]), (3, 7))
        self.assertEqual((short["mean_e"], long["mean_e"]), (.25, -.125))
        self.assertEqual((short["primary_pvalue"], long["primary_pvalue"]), (0., 0.))

    def test_malformed_statistics_inputs_reject(self):
        base, swapped = swap_fixture([.1, .2, .3, .4])
        bad = swapped.copy()
        bad[0, 0, 0] = np.nan
        invalid = [(base[:1], swapped[:1], .01), (base, bad, .01), (base, swapped[:-1], .01),
                   (base[:, 0], swapped[:, 0], .01), (base, swapped, -.01),
                   (base, swapped, np.inf), (base, swapped, True)]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                summarize_swaps(*args, n_bootstrap=100)
        for kwargs in ({"seed": -1}, {"seed": True}, {"n_bootstrap": 0}, {"n_bootstrap": 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                summarize_swaps(base, swapped, .01, **kwargs)


class CalibrationTests(unittest.TestCase):
    def test_deadband_covers_repeat_noop_batch_jitter_and_float32_spacing(self):
        base, _ = swap_fixture([0, 0, 0])
        repeat, noop = base.copy(), base.copy()
        repeat[0, 0, 1] += .0001
        noop[1, 1, 2] -= .0002
        result = calibrate_deadband(base, repeat, noop)
        observed = max(np.abs(repeat - base).max(), np.abs(noop - base).max())
        spacing = np.abs(np.spacing(base.astype(np.float32))).max()
        self.assertEqual(result["observed_max_jitter"], observed)
        self.assertEqual(result["max_float32_spacing"], float(spacing))
        self.assertEqual(result["deadband"], max(10 * observed, 10 * float(spacing)))
        exact = calibrate_deadband(base, base.copy())
        self.assertEqual(exact["deadband"], 10 * float(spacing))

    def test_calibration_inputs_reject_missing_nonfinite_or_mismatched_nulls(self):
        base, _ = swap_fixture([0, 0, 0])
        for nulls in ((), (base[:-1],), (base * np.nan,)):
            with self.subTest(nulls=nulls), self.assertRaises(ValueError):
                calibrate_deadband(base, *nulls)

    def test_normal_approximation_mde_uses_calibration_sd_and_two_sided_bonferroni(self):
        expected = .001 + (norm.ppf(1 - .05 / (2 * 18)) + norm.ppf(.8)) * .03 / np.sqrt(320)
        self.assertEqual(minimum_detectable_mean(.03, n=320, tau=.001), expected)
        self.assertEqual(minimum_detectable_mean(0, n=320, tau=.001), .001)
        self.assertLess(minimum_detectable_mean(.03, n=640, tau=.001), expected)
        for kwargs in ({"calibration_sd": -.1}, {"calibration_sd": .1, "tau": -.1},
                       {"calibration_sd": .1, "n": 0}, {"calibration_sd": .1, "claims": True},
                       {"calibration_sd": .1, "power": .5}, {"calibration_sd": .1, "alpha": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                minimum_detectable_mean(**kwargs)


if __name__ == "__main__":
    unittest.main()
