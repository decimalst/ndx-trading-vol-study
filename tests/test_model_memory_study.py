"""Pre-implementation synthetic contracts for modeling and causal memory.

These tests do not read market data, latent caches, or empirical forecasts.
Expected answers follow direct definitions rather than production helpers.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import model_memory_study as study


class FrozenProtocolContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "model_memory_study.yaml"
        cls.protocol = yaml.safe_load(path.read_text())

    def test_declared_unscored_protocol_is_accepted(self):
        study.validate_protocol(deepcopy(self.protocol))

    def test_sample_cannot_cross_protected_phase_or_change_horizons(self):
        for key, value in [("score_end", "2025-11-03"),
                           ("latest_target", "2025-11-03"),
                           ("horizons", [1, 5, 22])]:
            changed = deepcopy(self.protocol)
            changed["sample"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.validate_protocol(changed)

    def test_strong_benchmark_and_complete_comparison_family_cannot_be_relaxed(self):
        mutations = [
            ("baseline_columns", self.protocol["baseline_columns"][:-1]),
            ("models", self.protocol["models"][:-1]),
            ("component_models", self.protocol["component_models"][:-1]),
        ]
        for key, value in mutations:
            changed = deepcopy(self.protocol)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.validate_protocol(changed)
        changed = deepcopy(self.protocol)
        changed["comparisons"]["total_hypotheses"] = 20
        with self.assertRaises(ValueError):
            study.validate_protocol(changed)

    def test_fixed_memory_and_ensemble_choices_cannot_drift(self):
        mutations = [
            ("memory", "neighbors", 32),
            ("memory", "additional_gap_after_target_sessions", 0),
            ("memory", "lookback_sessions", 252),
            ("memory", "pca_components", 16),
            ("memory", "shrinkage", 1.0),
            ("ensemble", "half_life_sessions", 21),
            ("ensemble", "temperature", 10),
            ("ensemble", "equal_weight_floor", 0),
            ("inference", "blocks", [5]),
            ("inference", "bootstrap_draws", 99),
        ]
        for section, key, value in mutations:
            changed = deepcopy(self.protocol)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                study.validate_protocol(changed)


class MemoryAvailabilityContracts(unittest.TestCase):
    def setUp(self):
        # Deliberately omit real-world non-trading days and retain sparse rows.
        self.sessions = pd.bdate_range("2022-01-03", periods=180).delete([5, 19, 40])

    def test_gap_exact_boundary_and_origin_lookback_use_session_positions(self):
        positions = np.array([19, 20, 21, 72, 73, 74, 99, 100, 101])
        origins = self.sessions[positions]
        ends = self.sessions[positions + 5]
        got = study.eligible_memory(origins, ends, self.sessions[100], self.sessions,
                                    lookback=80, gap=22)
        expected = ((positions < 100) & (positions >= 20) & (positions + 5 <= 78))
        np.testing.assert_array_equal(got, expected)
        self.assertEqual(np.asarray(got).dtype, np.dtype(bool))
        self.assertTrue(got[4])   # Its five-session target ends at the gap boundary.
        self.assertFalse(got[5])  # One session later is unavailable.

    def test_sparse_retained_rows_do_not_redefine_lookback_or_gap(self):
        positions = np.array([0, 9, 10, 25, 45, 67, 68, 69, 89, 90])
        origins = self.sessions[positions]
        ends = self.sessions[positions + 1]
        got = study.eligible_memory(origins, ends, self.sessions[90], self.sessions,
                                    lookback=80, gap=22)
        np.testing.assert_array_equal(got, [False, False, True, True, True,
                                            True, False, False, False, False])

    def test_old_target_ends_cannot_make_future_origins_eligible(self):
        origins = self.sessions[[49, 50, 51]]
        ends = self.sessions[[49, 49, 49]]
        # This malformed fixture has targets ending at/before their origins.
        # Fail closed instead of admitting the first malformed row as memory.
        with self.assertRaises(ValueError):
            study.eligible_memory(origins, ends, self.sessions[50], self.sessions,
                                  lookback=100, gap=0)

    def test_future_target_values_cannot_influence_selected_memory(self):
        origins = self.sessions[:120]
        ends = self.sessions[5:125]
        eligible = study.eligible_memory(origins, ends, self.sessions[100], self.sessions,
                                         lookback=70, gap=22)
        original = np.linspace(0.5, 1.5, len(origins))
        changed = original.copy()
        changed[~eligible] = 1e12
        before = study.ratio_correction(original[eligible])
        after = study.ratio_correction(changed[eligible])
        self.assertEqual(before, after)
        self.assertEqual(int(np.sum(eligible)), 44)  # origins 30 through 73 inclusive.

    def test_default_gap_and_lookback_are_fixed(self):
        sessions = pd.bdate_range("2000-01-03", periods=1500)
        positions = np.array([39, 40, 41, 1277, 1278, 1279, 1299, 1300])
        got = study.eligible_memory(sessions[positions], sessions[positions + 1],
                                    sessions[1300], sessions)
        expected = (positions >= 40) & (positions + 1 <= 1278) & (positions < 1300)
        np.testing.assert_array_equal(got, expected)

    def test_unknown_query_origin_or_target_end_fails_instead_of_imputing_dates(self):
        missing = pd.Timestamp("2022-01-08")  # Saturday is not a trading session.
        with self.subTest(field="query"), self.assertRaises(ValueError):
            study.eligible_memory(self.sessions[:3], self.sessions[1:4], missing, self.sessions)
        with self.subTest(field="origin"), self.assertRaises(ValueError):
            study.eligible_memory(pd.DatetimeIndex([missing]), self.sessions[:1],
                                   self.sessions[100], self.sessions)
        with self.subTest(field="target_end"), self.assertRaises(ValueError):
            study.eligible_memory(self.sessions[:1], pd.DatetimeIndex([missing]),
                                   self.sessions[100], self.sessions)
        with self.subTest(field="missing_target_end"), self.assertRaises(ValueError):
            study.eligible_memory(self.sessions[:1], pd.DatetimeIndex([pd.NaT]),
                                   self.sessions[100], self.sessions)

    def test_output_remains_aligned_with_supplied_origin_order(self):
        positions = np.array([90, 40, 0, 70, 20])
        got = study.eligible_memory(self.sessions[positions], self.sessions[positions + 1],
                                    self.sessions[100], self.sessions, lookback=80, gap=22)
        np.testing.assert_array_equal(got, [False, True, False, True, True])


class StateGeometryContracts(unittest.TestCase):
    def test_raw_scaler_uses_training_population_scale_and_never_modifies_inputs(self):
        train = np.array([[1., 4., -4.], [2., 10., 0.], [6., 16., 5.], [7., 2., 11.]])
        apply = np.array([[100., 200., 300.], [-40., -80., -90.]])
        train_before, apply_before = train.copy(), apply.copy()
        got_train, got_apply = study.raw_keys(train, apply)
        mean = np.mean(train, axis=0)
        scale = np.sqrt(np.mean((train - mean) ** 2, axis=0))
        np.testing.assert_allclose(got_train, (train - mean) / scale, atol=1e-14)
        np.testing.assert_allclose(got_apply, (apply - mean) / scale, atol=1e-14)
        np.testing.assert_array_equal(train, train_before)
        np.testing.assert_array_equal(apply, apply_before)

    def test_raw_transform_is_invariant_to_other_application_rows(self):
        rng = np.random.default_rng(28193)
        train, query = rng.normal(size=(80, 5)), rng.normal(size=(1, 5))
        base_train, base_query = study.raw_keys(train, query)
        changed_train, changed_queries = study.raw_keys(train, np.vstack([query, np.full((7, 5), 1e10)]))
        np.testing.assert_allclose(changed_train, base_train, atol=0)
        np.testing.assert_allclose(changed_queries[0], base_query[0], atol=0)

    def test_any_zero_scale_raw_coordinate_rejects_the_whole_state(self):
        for column in range(3):
            train = np.arange(18, dtype=float).reshape(6, 3)
            train[:, column] = 12
            with self.subTest(column=column), self.assertRaises(ValueError):
                study.raw_keys(train, np.ones((1, 3)))

    def test_nonfinite_or_incompatible_raw_states_fail(self):
        train = np.arange(18, dtype=float).reshape(6, 3)
        for bad in [np.nan, np.inf, -np.inf]:
            changed = train.copy()
            changed[1, 1] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                study.raw_keys(changed, np.ones((1, 3)))
        with self.assertRaises(ValueError):
            study.raw_keys(train, np.ones((1, 4)))
        with self.assertRaises(ValueError):
            study.raw_keys(train, np.array([[1., np.nan, 2.]]))

    def test_pca_uses_training_center_and_covariance_with_one_global_scale(self):
        rng = np.random.default_rng(83011)
        train = rng.normal(size=(64, 12)) * np.arange(1, 13) + np.arange(12) * 20
        apply = rng.normal(size=(3, 12)) * 7 + 400
        original_train, original_apply = train.copy(), apply.copy()
        got_train, got_apply = study.latent_keys(train, apply, n_components=8)
        centered = train - train.mean(axis=0)
        eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / (len(train) - 1))
        selected = np.argsort(-eigenvalues, kind="stable")[:8]
        projection = vectors[:, selected]
        global_scale = np.sqrt(eigenvalues[selected].sum())
        expected_train = centered @ projection / global_scale
        expected_apply = (apply - train.mean(axis=0)) @ projection / global_scale
        # PCA column signs are arbitrary; Gram matrices and pairwise distances are not.
        np.testing.assert_allclose(got_train @ got_train.T, expected_train @ expected_train.T,
                                   rtol=1e-11, atol=1e-12)
        expected_distances = np.sum((expected_train[:, None, :] - expected_apply[None, :, :]) ** 2, axis=2)
        actual_distances = np.sum((got_train[:, None, :] - got_apply[None, :, :]) ** 2, axis=2)
        np.testing.assert_allclose(actual_distances, expected_distances, rtol=1e-11, atol=1e-12)
        self.assertEqual(got_train.shape, (64, 8))
        self.assertEqual(got_apply.shape, (3, 8))
        np.testing.assert_allclose(np.mean(got_train, axis=0), 0, atol=1e-13)
        self.assertAlmostEqual(float(np.trace(np.cov(got_train, rowvar=False))), 1., places=12)
        np.testing.assert_array_equal(train, original_train)
        np.testing.assert_array_equal(apply, original_apply)

    def test_pca_does_not_standardize_input_dimensions_or_whiten_components(self):
        train = np.array([[-3., 0.], [3., 0.], [0., -1.], [0., 1.]])
        apply = np.array([[0., 80.], [3., 0.]])
        _, projected = study.latent_keys(train, apply, n_components=1)
        self.assertAlmostEqual(float(projected[0, 0]), 0., places=14)
        self.assertGreater(abs(float(projected[1, 0])), 0.)
        scores, _ = study.latent_keys(train, apply, n_components=2)
        variances = np.var(scores, axis=0, ddof=1)
        self.assertAlmostEqual(float(variances[0] / variances[1]), 9., places=12)

    def test_pca_training_fit_and_existing_query_ignore_future_application_states(self):
        rng = np.random.default_rng(12092)
        train, query = rng.normal(size=(60, 10)), rng.normal(size=(1, 10))
        before_train, before_query = study.latent_keys(train, query)
        after_train, after_queries = study.latent_keys(train, np.vstack([query, np.full((5, 10), 1e8)]))
        np.testing.assert_allclose(after_train, before_train, atol=0)
        np.testing.assert_allclose(after_queries[0], before_query[0], atol=0)
        self.assertEqual(before_train.shape[1], 8)

    def test_pca_is_repeatable_including_tied_eigenvalues(self):
        train = np.vstack([np.eye(10), -np.eye(10)])
        apply = np.arange(20, dtype=float).reshape(2, 10)
        first = study.latent_keys(train, apply, n_components=8)
        second = study.latent_keys(train, apply, n_components=8)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_insufficient_latent_rank_fails_without_silently_reducing_components(self):
        rng = np.random.default_rng(8013)
        train = rng.normal(size=(40, 3)) @ rng.normal(size=(3, 12))
        with self.assertRaises(ValueError):
            study.latent_keys(train, np.ones((1, 12)), n_components=4)
        with self.assertRaises(ValueError):
            study.latent_keys(np.zeros((40, 12)), np.ones((1, 12)))

    def test_nonfinite_latent_training_or_query_fails(self):
        rng = np.random.default_rng(8311)
        train = rng.normal(size=(50, 12))
        bad_train = train.copy()
        bad_train[0, 0] = np.nan
        with self.assertRaises(ValueError):
            study.latent_keys(bad_train, np.zeros((1, 12)))
        with self.assertRaises(ValueError):
            study.latent_keys(train, np.full((1, 12), np.inf))


class RetrievalAndCorrectionContracts(unittest.TestCase):
    def test_neighbors_match_direct_euclidean_distance_and_stable_ties(self):
        keys = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [0., 0.], [2., 0.]])
        query = np.array([0., 0.])
        got = study.neighbor_indices(keys, query, k=4)
        np.testing.assert_array_equal(got, [4, 0, 1, 2])
        # Caller supplies eligibility; an exact state match is a valid historical neighbor.
        self.assertEqual(int(got[0]), 4)

    def test_neighbor_default_count_and_inputs_remain_unchanged(self):
        rng = np.random.default_rng(29382)
        keys, query = rng.normal(size=(100, 8)), rng.normal(size=8)
        original_keys, original_query = keys.copy(), query.copy()
        expected = np.argsort(np.sum((keys - query) ** 2, axis=1), kind="stable")[:64]
        np.testing.assert_array_equal(study.neighbor_indices(keys, query), expected)
        np.testing.assert_array_equal(keys, original_keys)
        np.testing.assert_array_equal(query, original_query)

    def test_neighbors_reject_nonfinite_or_mismatched_states(self):
        keys = np.ones((70, 3))
        with self.assertRaises(ValueError):
            study.neighbor_indices(keys, np.array([np.nan, 1., 1.]))
        with self.assertRaises(ValueError):
            study.neighbor_indices(keys, np.ones(4))
        changed = keys.copy()
        changed[12, 0] = np.inf
        with self.assertRaises(ValueError):
            study.neighbor_indices(changed, np.ones(3))

    def test_correction_uses_arithmetic_ratio_mean_and_declared_shrinkage(self):
        ratios = np.array([0.25, 1., 4., 10.])
        expected = 0.75 + 0.25 * (15.25 / 4)
        self.assertAlmostEqual(study.ratio_correction(ratios), expected, places=14)
        geometric = 0.75 + 0.25 * np.exp(np.log(ratios).mean())
        self.assertNotAlmostEqual(study.ratio_correction(ratios), geometric, places=5)
        self.assertEqual(study.ratio_correction(ratios, shrinkage=0), 1.)
        self.assertEqual(study.ratio_correction(ratios, shrinkage=1), ratios.mean())

    def test_correction_preserves_positive_variance_and_has_no_hidden_caps(self):
        self.assertEqual(study.ratio_correction(np.array([100., 300.])), 50.75)
        for scale in [1e-8, 0.25, 1., 1e6]:
            for shrinkage in [0., 0.25, 1.]:
                with self.subTest(scale=scale, shrinkage=shrinkage):
                    correction = study.ratio_correction(np.array([scale, 2 * scale]), shrinkage=shrinkage)
                    self.assertGreater(correction * 1e-4, 0.)

    def test_bad_ratio_or_shrinkage_fails_instead_of_filtering(self):
        for ratios in [[], [1., 0.], [1., -0.1], [1., np.nan], [1., np.inf]]:
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                study.ratio_correction(np.array(ratios))
        for shrinkage in [-0.01, 1.01, np.nan, np.inf]:
            with self.subTest(shrinkage=shrinkage), self.assertRaises(ValueError):
                study.ratio_correction(np.array([1., 2.]), shrinkage=shrinkage)


class DynamicEnsembleContracts(unittest.TestCase):
    def test_ensemble_matches_independent_exponential_age_and_softmax_definition(self):
        losses = np.array([[0.1, 1., 2.], [0.4, 0.3, 0.8], [2., 0.6, 0.2]])
        ages = np.array([126., 63., 0.])
        time_weights = np.array([0.25, 0.5, 1.])
        mean_losses = np.sum(losses * time_weights[:, None], axis=0) / time_weights.sum()
        logits = -5 * mean_losses
        softmax = np.exp(logits - logits.max())
        expected = 0.1 / 3 + 0.9 * softmax / softmax.sum()
        before_losses, before_ages = losses.copy(), ages.copy()
        actual = study.ensemble_weights(losses, ages)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        self.assertAlmostEqual(float(np.sum(actual)), 1., places=14)
        self.assertTrue((actual >= 0.1 / 3).all())
        np.testing.assert_array_equal(losses, before_losses)
        np.testing.assert_array_equal(ages, before_ages)

    def test_equal_losses_are_uniform_and_combined_variance_remains_positive(self):
        losses = np.repeat(np.array([0.1, 4., 0.3])[:, None], 4, axis=1)
        weights = study.ensemble_weights(losses, np.array([200., 30., 0.]))
        np.testing.assert_allclose(weights, np.full(4, 0.25), atol=1e-14)
        forecasts = np.array([1e-10, 0.1, 2., 100.])
        forecast = float(weights @ forecasts)
        self.assertGreaterEqual(forecast, float(forecasts.min()))
        self.assertLessEqual(forecast, float(forecasts.max()))
        self.assertGreater(forecast, 0.)

    def test_recent_completed_losses_outweigh_opposite_old_losses(self):
        losses = np.array([[0., 10.], [10., 0.]])
        recent_favors_second = study.ensemble_weights(losses, np.array([630., 0.]))
        recent_favors_first = study.ensemble_weights(losses, np.array([0., 630.]))
        self.assertGreater(recent_favors_second[1], recent_favors_second[0])
        self.assertGreater(recent_favors_first[0], recent_favors_first[1])
        np.testing.assert_allclose(recent_favors_second, recent_favors_first[::-1], atol=1e-14)

    def test_temperature_zero_and_floor_one_recover_uniform_controls(self):
        losses = np.array([[0.1, 2., 80.], [4., 1., 10.]])
        ages = np.array([63., 0.])
        for kwargs in [{"temperature": 0.}, {"floor": 1.}]:
            with self.subTest(kwargs=kwargs):
                np.testing.assert_allclose(study.ensemble_weights(losses, ages, **kwargs),
                                           np.full(3, 1 / 3), atol=1e-14)

    def test_softmax_is_stable_for_large_finite_losses(self):
        weights = study.ensemble_weights(np.array([[1e6, 1e6 + 1, 1e6 + 100]]), np.array([0.]))
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.sum()), 1., places=14)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_bad_losses_ages_or_parameters_fail_instead_of_dropping_experts(self):
        losses, ages = np.array([[0.1, 0.2], [0.5, 0.3]]), np.array([1., 0.])
        for value in [np.nan, np.inf, -np.inf]:
            changed = losses.copy()
            changed[1, 1] = value
            with self.subTest(loss=value), self.assertRaises(ValueError):
                study.ensemble_weights(changed, ages)
        for bad_ages in [np.array([-1., 0.]), np.array([np.nan, 0.]), np.array([np.inf, 0.]), np.array([0.])]:
            with self.subTest(ages=bad_ages), self.assertRaises(ValueError):
                study.ensemble_weights(losses, bad_ages)
        for kwargs in [{"half_life": 0.}, {"half_life": -1.}, {"half_life": np.inf},
                       {"temperature": -1.}, {"temperature": np.nan},
                       {"floor": -0.1}, {"floor": 1.1}, {"floor": np.nan}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                study.ensemble_weights(losses, ages, **kwargs)


if __name__ == "__main__":
    unittest.main()
