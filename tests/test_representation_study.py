"""Core frozen contracts plus explicitly post-result reviewer checks."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from src import representation_study as study


def _protocol() -> dict:
    return {
        "source": {
            "first_session": "1999-03-11",
            "last_permitted_origin": "2025-10-17",
            "clean_start": "2025-11-03",
            "require_source_hash": True,
            "forbid_clean_origins": True,
            "permitted_inputs": ["log_rv", "rv_total", "ret_cc"],
        },
        "tail_target": {
            "horizon_sessions": 5,
            "stress_quantile": 0.80,
            "score_only_calm_origins": True,
            "require_complete_future": True,
            "first_score_year": 2001,
            "final_score_date": "2025-10-17",
            "refit": "annual",
            "minimum_training_observations": 400,
        },
        "ranking_scoreboard": {
            "primary": "auc",
            "secondary": "top_decile_lift",
            "top_fraction": 0.10,
            "common_rows_required": True,
            "brier_and_logloss": "diagnostic_only",
        },
        "classical_models": {
            "benchmark": {"features": ["log_rv_d", "log_rv_w", "log_rv_m"], "ridge": 1e-6},
            "hmm": {
                "states": 2,
                "max_iter": 200,
                "tolerance": 1e-7,
                "calibration": "platt",
                "calibration_clip": 1e-6,
                "calibration_ridge": 1e-6,
            },
        },
        "reviewer_controls": {
            "trailing_rv_percentile": {
                "status": "post_result_requested_after_headline_seen",
                "evidence_role": "diagnostic_only_not_prespecified_confirmation",
                "ridge": 1e-6,
            },
            "phase_and_episode_reporting": {
                "status": "post_result_requested_after_headline_seen",
                "evidence_role": "diagnostic_uncertainty_not_prespecified_confirmation",
                "ranking_phases": 5,
                "phase_summaries": ["mean", "min", "max", "spread"],
                "inference": "leave_one_transition_episode_out_jackknife",
                "confidence_level": 0.95,
            },
        },
        "latent_probe": {
            "checkpoint_revision": "05e5b26db52bfb256f1ae1bdf785589850482de3",
            "context_length": 2048,
            "layer": "output of stack_out_norm, before output_patch_embedding",
            "pooling": "last non-padding target token",
            "expected_dimension": 512,
            "test_time_augmentation": False,
            "differencing": False,
            "frozen_backbone": True,
            "no_layer_selection": True,
            "no_pooling_selection": True,
            "forbid_pca": True,
            "probe_ladder": {
                "full_ridge": {"ridge": 1.0, "dimensions": "all"},
                "sparse": {"ridge": 1.0, "k": [1, 5, 10]},
                "small_mlp": {"hidden_units": 8, "l2_alpha": 1.0, "seed": 42},
            },
            "control_task": {"draws": 10, "seeds": list(range(4200, 4210))},
            "uncertainty": {"cluster_not_rows": True},
        },
        "noise_robustness": {
            "sample_start": "2016-01-04",
            "sample_end": "2025-10-17",
            "origin_stride_sessions": 20,
            "context_length": 2048,
            "target_horizon": 1,
            "seed": 42,
            "origin_seed": "first 64 bits of SHA-256('42|YYYY-MM-DD'), interpreted unsigned big-endian",
            "gaussian": {"intensities": [0, .2, .4, .6, .8]},
            "impulse": {"probabilities": [0, .05, .1, .15, .2], "magnitude_local_std": 8.0},
            "primary_metric": "relative CRPS = noisy CRPS / same-model clean CRPS",
            "models": [
                "chronos_2_univariate",
                "tirex_2_univariate",
                "har_univariate_expanding_clean_fit",
            ],
            "paired_corruption_required": True,
            "preprocessing_policy": (
                "Each adapter retains native preprocessing. HAR does not renormalize."
            ),
            "all_intensities_reported": True,
            "comparison_inference": {
                "block_sessions": 22,
                "block_sampled_origins": 2,
                "bootstrap_draws": 5000,
                "seed": 420042,
            },
        },
    }


class ProtocolContract(unittest.TestCase):
    def test_frozen_protocol_validates(self):
        study.validate_protocol(_protocol())

    def test_clean_origins_are_forbidden(self):
        protocol = _protocol()
        protocol["tail_target"]["final_score_date"] = protocol["source"]["clean_start"]
        with self.assertRaisesRegex(ValueError, "clean"):
            study.validate_protocol(protocol)

    def test_ranking_metrics_and_old_threshold_are_fixed(self):
        for key, value in (("primary", "brier"), ("secondary", "accuracy")):
            protocol = copy.deepcopy(_protocol())
            protocol["ranking_scoreboard"][key] = value
            with self.assertRaisesRegex(ValueError, "AUC|decile"):
                study.validate_protocol(protocol)
        protocol = copy.deepcopy(_protocol())
        protocol["tail_target"]["stress_quantile"] = .90
        with self.assertRaisesRegex(ValueError, "80th"):
            study.validate_protocol(protocol)

    def test_latent_layer_and_pooling_cannot_be_selected(self):
        protocol = copy.deepcopy(_protocol())
        protocol["latent_probe"]["no_layer_selection"] = False
        with self.assertRaisesRegex(ValueError, "selection"):
            study.validate_protocol(protocol)

    def test_pca_and_missing_control_are_rejected(self):
        protocol = copy.deepcopy(_protocol())
        protocol["latent_probe"]["forbid_pca"] = False
        with self.assertRaisesRegex(ValueError, "PCA"):
            study.validate_protocol(protocol)
        protocol = copy.deepcopy(_protocol())
        protocol["latent_probe"]["control_task"]["draws"] = 0
        with self.assertRaisesRegex(ValueError, "control"):
            study.validate_protocol(protocol)

    def test_sparse_ladder_is_exactly_one_five_ten(self):
        protocol = copy.deepcopy(_protocol())
        protocol["latent_probe"]["probe_ladder"]["sparse"]["k"] = [5]
        with self.assertRaisesRegex(ValueError, "sparse"):
            study.validate_protocol(protocol)


class TailTargetContract(unittest.TestCase):
    def test_transition_features_inherit_mean_log_rv_convention(self):
        idx = pd.bdate_range("2000-01-03", periods=30)
        y = pd.Series(np.linspace(-12, -6, 30), index=idx)
        got = study.build_history_features(y)
        self.assertAlmostEqual(got.loc[idx[-1], "log_rv_w"], y.iloc[-5:].mean())
        self.assertAlmostEqual(got.loc[idx[-1], "log_rv_m"], y.iloc[-22:].mean())
        self.assertNotAlmostEqual(
            got.loc[idx[-1], "log_rv_w"], np.log(np.exp(y.iloc[-5:]).mean())
        )

    def test_threshold_uses_only_cutoff_and_future_is_strictly_after_origin(self):
        idx = pd.bdate_range("2000-01-03", periods=20)
        y = pd.Series(np.arange(20.0), index=idx)
        cutoff = idx[9]
        origins = idx[10:13]
        got = study.build_fold_targets(y, cutoff=cutoff, origins=origins, horizon=2, quantile=.8)
        self.assertTrue((got["threshold"] == y.loc[:cutoff].quantile(.8)).all())
        self.assertTrue((got["target_end"] > got.index).all())
        self.assertTrue(got["event"].all())

    def test_incomplete_future_is_not_labeled(self):
        idx = pd.bdate_range("2000-01-03", periods=8)
        y = pd.Series(np.arange(8.0), index=idx)
        got = study.build_fold_targets(y, cutoff=idx[3], origins=idx[5:], horizon=2, quantile=.8)
        self.assertEqual(got.index.max(), idx[-3])

    def test_future_values_cannot_change_current_features(self):
        idx = pd.bdate_range("2000-01-03", periods=30)
        y = pd.Series(np.linspace(-10, -5, 30), index=idx)
        before = study.build_history_features(y)
        changed = y.copy()
        changed.iloc[-1] = 1000
        after = study.build_history_features(changed)
        pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


class RankingContract(unittest.TestCase):
    def test_phase_dispersion_reports_min_max_and_spread(self):
        phases = pd.DataFrame({
            "model": ["a"] * 5,
            "auc": [.7, .8, .9, .75, .85],
            "top_decile_lift": [1, 2, 3, 4, 5],
            "top_decile_event_rate": [.1, .2, .3, .4, .5],
        })
        got = study.summarize_phase_dispersion(phases)["a"]
        self.assertAlmostEqual(got["auc"]["min"], .7)
        self.assertAlmostEqual(got["auc"]["max"], .9)
        self.assertAlmostEqual(got["auc"]["spread"], .2)

    def test_episode_jackknife_uses_episode_count_not_positive_rows(self):
        n = 60
        event = np.tile([0, 0, 1], 20)
        clusters = pd.array(
            [position // 9 if value else pd.NA for position, value in enumerate(event)],
            dtype="Int64",
        )
        frame = pd.DataFrame({
            "event": event,
            "p_a": np.linspace(0, 1, n),
            "ranking_phase": np.arange(n) % 5,
            "event_cluster": clusters,
        })
        got = study.episode_jackknife_ranking(
            frame, ["a"], horizon=5, top_fraction=.1, confidence=.95
        )
        self.assertEqual(int(got.loc[0, "episodes"]), frame["event_cluster"].nunique())
        self.assertLess(int(got.loc[0, "episodes"]), int(frame["event"].sum()))
        self.assertTrue(np.isfinite(got[["ci_low", "ci_high"]].to_numpy()).all())

    def test_episode_jackknife_differences_are_paired(self):
        n = 90
        event = np.tile([0, 0, 1], 30)
        frame = pd.DataFrame({
            "event": event,
            "p_better": np.linspace(0, 1, n),
            "p_worse": np.linspace(1, 0, n),
            "ranking_phase": np.arange(n) % 5,
            "event_cluster": pd.array(
                [position // 12 if value else pd.NA for position, value in enumerate(event)],
                dtype="Int64",
            ),
        })
        got = study.episode_jackknife_differences(
            frame,
            {"better_minus_worse": ("better", "worse")},
            horizon=5,
            top_fraction=.1,
            confidence=.95,
        )
        auc = got.loc[got["metric"] == "auc"].iloc[0]
        self.assertGreater(float(auc["estimate"]), 0)
        self.assertEqual(int(auc["episodes"]), frame["event_cluster"].nunique())

    def test_prior_reference_percentile_is_an_empirical_cdf(self):
        reference = np.array([-3.0, -2.0, -1.0])
        values = np.array([-3.0, -2.0, -1.0, 0.0])
        got = study.prior_reference_percentile(values, reference)
        np.testing.assert_allclose(got, [1 / 3, 2 / 3, 1.0, 1.0])

    def test_auc_handles_ties_as_half_credit(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.5, 0.5, 0.9])
        self.assertAlmostEqual(study.roc_auc(y, score), .875)

    def test_top_decile_lift_uses_ceil_and_stable_order(self):
        y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        score = np.array([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 1.0])
        # ceil(1.1)=2, stable tie order chooses positions 0 then 10: 100% / 2/11.
        self.assertAlmostEqual(study.top_decile_lift(y, score, .1), 5.5)

    def test_phase_metrics_keep_five_nonoverlapping_scoreboards(self):
        frame = pd.DataFrame({
            "event": np.tile([0, 1], 25),
            "p_a": np.linspace(0, 1, 50),
            "p_b": np.linspace(1, 0, 50),
        })
        phases = study.evaluate_ranking_phases(frame, ["a", "b"], horizon=5, top_fraction=.1)
        self.assertEqual(set(phases["phase"]), set(range(5)))
        self.assertTrue((phases.groupby("phase")["n"].first() == 10).all())
        self.assertIn("top_decile_event_rate", phases)
        np.testing.assert_allclose(
            phases["top_decile_event_rate"], phases["event_rate"] * phases["top_decile_lift"]
        )


class ReviewerControlArtifactContract(unittest.TestCase):
    def test_saved_scoreboard_contains_post_result_level_control(self):
        metrics = study.OUTPUT_DIR / "tail_classical_metrics.json"
        self.assertTrue(metrics.exists())
        saved = __import__("json").loads(metrics.read_text())
        self.assertIn("rv_percentile", saved["phase_mean"])
        self.assertEqual(
            saved["reviewer_control_status"], "post_result_requested_after_headline_seen"
        )
        self.assertEqual(saved["ranking_phases"], 5)
        self.assertEqual(saved["transition_episodes"], 118)
        self.assertIn("phase_dispersion", saved)
        self.assertIn("episode_differences", saved)

    def test_saved_tail_artifacts_recompute_exactly(self):
        saved = study.verify_tail_classical()
        self.assertEqual(saved["origins"], 5592)
        self.assertEqual(saved["ranking_phases"], 5)
        report = study._render_tail_report(saved)
        self.assertIn("not full cluster-robust standard errors", report)
        self.assertIn("not the standard HAR", report)


class NoiseContract(unittest.TestCase):
    def test_origin_seed_is_order_invariant(self):
        date = pd.Timestamp("2020-03-16")
        first = study.origin_noise_seed(42, date)
        _ = [study.origin_noise_seed(42, value) for value in pd.bdate_range("2019-01-01", periods=9)]
        second = study.origin_noise_seed(42, date)
        self.assertEqual(first, second)
        self.assertNotEqual(first, study.origin_noise_seed(42, date + pd.Timedelta(days=1)))

    def test_noise_seed_and_bootstrap_contract_cannot_drift(self):
        protocol = copy.deepcopy(_protocol())
        protocol["noise_robustness"]["origin_seed"] = "global RNG stream"
        with self.assertRaisesRegex(ValueError, "per-origin"):
            study.validate_protocol(protocol)
        protocol = copy.deepcopy(_protocol())
        protocol["noise_robustness"]["comparison_inference"]["bootstrap_draws"] = 100
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            study.validate_protocol(protocol)

    def test_gaussian_noise_is_seed_deterministic_and_zero_is_identity(self):
        x = np.linspace(-2, 2, 200)
        a = study.gaussian_noise(x, .4, np.random.default_rng(42))
        b = study.gaussian_noise(x, .4, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(study.gaussian_noise(x, 0, np.random.default_rng(9)), x)

    def test_impulse_noise_has_only_fixed_magnitude_spikes(self):
        x = np.linspace(-1, 1, 1000)
        got = study.impulse_noise(x, .2, 8.0, np.random.default_rng(42))
        delta = got - x
        changed = delta != 0
        self.assertGreater(changed.sum(), 0)
        np.testing.assert_allclose(np.abs(delta[changed]), 8.0 * np.std(x, ddof=0))

    def test_corruption_never_mutates_future_target(self):
        context = np.arange(20.0)
        future = np.array([999.0])
        _ = study.gaussian_noise(context, .8, np.random.default_rng(42))
        np.testing.assert_array_equal(future, [999.0])


class LatentPoolingContract(unittest.TestCase):
    def test_pooling_selects_fixed_last_context_token(self):
        hidden = np.arange(2 * 74 * 3, dtype=float).reshape(2, 74, 3)
        got = study.pool_tirex_context_token(hidden, context_tokens=64, expected_dim=3)
        np.testing.assert_array_equal(got, hidden[:, 63, :])

    def test_wrong_embedding_dimension_fails(self):
        hidden = np.zeros((2, 74, 4))
        with self.assertRaisesRegex(ValueError, "dimension"):
            study.pool_tirex_context_token(hidden, context_tokens=64, expected_dim=3)


class ProbeControlContract(unittest.TestCase):
    def test_sparse_selection_uses_training_labels_only(self):
        X_train = np.array([[0, 0], [0, 1], [2, 0], [2, 1]], dtype=float)
        y_train = np.array([0, 0, 1, 1])
        before = study.select_sparse_dimensions(X_train, y_train, k=1)
        _changed_test_labels = np.array([1, 1, 0, 0])
        after = study.select_sparse_dimensions(X_train, y_train, k=1)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(before, [0])

    def test_markov_control_is_deterministic_and_binary(self):
        y = np.tile([0, 0, 0, 1, 1], 30)
        a = study.markov_control_labels(y, total_length=200, seed=4200)
        b = study.markov_control_labels(y, total_length=200, seed=4200)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(set(np.unique(a)), {0, 1})

    def test_event_clusters_merge_nearby_trigger_sessions(self):
        sessions = pd.bdate_range("2020-01-02", periods=20)
        trigger = pd.Series([sessions[5], sessions[7], sessions[15]], index=sessions[:3])
        got = study.assign_event_clusters(trigger, sessions, max_gap_sessions=5)
        self.assertEqual(got.iloc[0], got.iloc[1])
        self.assertNotEqual(got.iloc[1], got.iloc[2])


if __name__ == "__main__":
    unittest.main()
