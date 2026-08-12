"""Pre-result contracts for the frozen TiRex latent-probe ladder.

These tests were written before any latent embedding was extracted or any
financial probe score was produced.  They deliberately exercise leakage and
scoreboard boundaries with synthetic data only.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import latent_probe_study as study


class LatentExtractionContract(unittest.TestCase):
    def test_chunk_reuse_fails_without_or_with_stale_run_signature(self):
        expected = {
            "protocol_sha256": "a" * 64,
            "history_panel_sha256": "b" * 64,
            "checkpoint_revision": "c" * 40,
            "origins_sha256": "d" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "unbound|manifest"):
            study.validate_chunk_manifest(expected, None)
        stale = {"signature": {**expected, "protocol_sha256": "e" * 64}}
        with self.assertRaisesRegex(RuntimeError, "signature"):
            study.validate_chunk_manifest(expected, stale)
        study.validate_chunk_manifest(expected, {"signature": expected, "chunks": {}})

    def test_pools_exactly_full_context_token_63_and_all_512_coordinates(self):
        hidden = np.arange(3 * 68 * 512, dtype=np.float32).reshape(3, 68, 512)
        got = study.pool_final_context_token(hidden, token_index=63, expected_dim=512)
        self.assertEqual(got.shape, (3, 512))
        np.testing.assert_array_equal(got, hidden[:, 63, :])

    def test_wrong_token_or_dimension_fails_closed(self):
        hidden = np.zeros((2, 64, 511), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "dimension"):
            study.pool_final_context_token(hidden, token_index=63, expected_dim=512)
        with self.assertRaisesRegex(ValueError, "token"):
            study.pool_final_context_token(
                np.zeros((2, 63, 512), dtype=np.float32),
                token_index=63,
                expected_dim=512,
            )

    def test_context_is_trailing_only_and_includes_the_origin(self):
        index = pd.bdate_range("2000-01-03", periods=20)
        values = pd.Series(np.arange(20.0), index=index)
        context = study.causal_context(values, index[12], context_length=8)
        np.testing.assert_array_equal(context, np.arange(5.0, 13.0))
        changed = values.copy()
        changed.loc[index[13]:] = 10_000
        np.testing.assert_array_equal(
            study.causal_context(changed, index[12], context_length=8), context
        )

    def test_embedding_origin_axis_has_one_canonical_persisted_name(self):
        index = pd.bdate_range("2000-01-03", periods=40, name="date")
        history = pd.DataFrame({"log_rv": np.arange(40.0)}, index=index)
        classical = pd.DataFrame(index=index[30:])
        got = study.embedding_origins(history, classical)
        self.assertEqual(got.name, "origin")


class FoldAndProbeContract(unittest.TestCase):
    def test_mlp_exposes_optimizer_convergence_instead_of_suppressing_it(self):
        rng = np.random.default_rng(91)
        X = rng.normal(size=(100, 16))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        fitted = study.fit_probe(
            "small_mlp",
            X,
            y,
            ridge=1.0,
            seed=42,
            mlp_config={
                "hidden_units": 8,
                "activation": "tanh",
                "solver": "lbfgs",
                "l2_alpha": 1.0,
                "max_iter": 1,
                "seed": 42,
            },
        )
        self.assertEqual(fitted.n_iter, 1)
        self.assertFalse(fitted.converged)
        self.assertTrue(fitted.convergence_warning)

    def test_strict_fold_reconstruction_ignores_only_index_axis_metadata(self):
        index = pd.bdate_range("1999-01-04", periods=650, name="date")
        y = pd.Series(np.sin(np.arange(650) / 13.0), index=index)
        cutoff = index[499]
        origins = index[520:550]
        rebuilt = study.base.build_fold_targets(
            y, cutoff=cutoff, origins=origins, horizon=5, quantile=.8
        )
        classical = rebuilt.loc[rebuilt["calm"]].copy()
        classical.index = classical.index.rename(None)
        # Parquet may restore equal timestamps at a different datetime unit.
        classical["target_end"] = classical["target_end"].astype("datetime64[ms]")
        classical["trigger_date"] = classical["trigger_date"].astype("datetime64[ms]")
        study._strict_validate_classical_fold(
            y,
            classical,
            cutoff=cutoff,
            horizon=5,
            quantile=.8,
        )

    def test_fold_training_labels_are_completed_before_cutoff_and_test_rows_are_exact(self):
        index = pd.bdate_range("1999-01-04", periods=800)
        y = pd.Series(np.sin(np.arange(800) / 17.0), index=index)
        features = pd.DataFrame(
            {
                "log_rv_d": y,
                "log_rv_w": y.rolling(5).mean(),
                "log_rv_m": y.rolling(22).mean(),
            },
            index=index,
        )
        cutoff = index[499]
        requested = index[520:550]
        classical = pd.DataFrame(
            {
                "event": np.tile([0, 1], 15),
                "threshold": 0.3,
                "target_end": index[525:555],
                "trigger_date": pd.NaT,
                "fold_year": 2001,
                "p_benchmark": np.linspace(.1, .9, 30),
                "p_hmm_augmented": np.linspace(.2, .8, 30),
            },
            index=requested,
        )
        embeddings = pd.DataFrame(
            np.random.default_rng(7).normal(size=(len(index), 512)),
            index=index,
            columns=study.latent_columns(512),
        )
        train, test = study.build_annual_fold_tables(
            y=y,
            features=features,
            embeddings=embeddings,
            classical_fold=classical,
            cutoff=cutoff,
            horizon=5,
            quantile=.8,
        )
        self.assertTrue((train["target_end"] <= cutoff).all())
        self.assertTrue((train.index < requested.min()).all())
        pd.testing.assert_index_equal(test.index, requested)
        self.assertEqual(len(test), len(classical))

    def test_full_ridge_consumes_every_coordinate(self):
        X = np.random.default_rng(8).normal(size=(80, 512))
        y = np.tile([0, 1], 40)
        fitted = study.fit_probe("full_ridge", X, y, ridge=1.0, seed=42)
        self.assertEqual(fitted.n_input_dimensions, 512)

    def test_sparse_selection_is_training_only_standardized_mean_difference(self):
        X = np.zeros((8, 4), dtype=float)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X[y == 1, 2] = 3.0
        X[:, 1] = np.arange(8) * 100.0
        before = study.select_sparse_dimensions(X, y, k=1)
        unrelated_test_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        _ = unrelated_test_labels  # Test labels are intentionally never passed.
        after = study.select_sparse_dimensions(X, y, k=1)
        np.testing.assert_array_equal(before, [2])
        np.testing.assert_array_equal(after, before)
        # Arbitrarily changing held-out embeddings and labels cannot affect a
        # selector whose interface accepts training inputs only.
        X_test = np.full((20, 4), 1e12)
        y_test = np.ones(20, dtype=int)
        X_test[:, 0] = -1e12
        y_test[:] = 0
        _ = (X_test, y_test)
        np.testing.assert_array_equal(
            study.select_sparse_dimensions(X, y, k=1), before
        )


class ControlAndScoreboardContract(unittest.TestCase):
    def test_ten_controls_cannot_supply_five_percent_randomization_evidence(self):
        actual = .80
        controls = np.linspace(.45, .60, 10)
        got = study.randomization_evidence(actual, controls, alpha=.05)
        self.assertAlmostEqual(got["exact_randomization_p_lower_bound"], 1 / 11)
        self.assertAlmostEqual(got["exact_randomization_p"], 1 / 11)
        self.assertFalse(got["formal_evidence"])

    def test_phase_dispersion_reports_each_frozen_endpoint(self):
        phases = pd.DataFrame(
            {
                "phase": range(5),
                "auc": [.6, .7, .8, .9, .5],
                "top_decile_lift": [1, 2, 3, 4, 5],
            }
        )
        got = study.phase_dispersion(phases)
        self.assertEqual(got["auc"], {"min": .5, "max": .9, "spread": .4})
        self.assertEqual(
            got["top_decile_lift"], {"min": 1.0, "max": 5.0, "spread": 4.0}
        )

    def test_markov_control_uses_training_labels_and_continues_into_test(self):
        train = np.tile([0, 0, 0, 1, 1], 30)
        a = study.markov_surrogate_path(train, train_length=len(train), test_length=40, seed=4200)
        b = study.markov_surrogate_path(train, train_length=len(train), test_length=40, seed=4200)
        np.testing.assert_array_equal(a.train, b.train)
        np.testing.assert_array_equal(a.test, b.test)
        self.assertEqual(len(a.train), len(train))
        self.assertEqual(len(a.test), 40)
        self.assertEqual(set(np.unique(np.r_[a.train, a.test])), {0, 1})

    def test_phase_mean_is_unweighted_and_keeps_all_five_phases(self):
        frame = pd.DataFrame(
            {
                "event": np.tile([0, 1], 25),
                "score": np.linspace(0, 1, 50),
            },
            index=pd.bdate_range("2000-01-03", periods=50),
        )
        got = study.phase_ranking_metrics(frame, score_column="score", horizon=5, top_fraction=.1)
        self.assertEqual(set(got["phase"]), set(range(5)))
        summary = study.phase_mean(got)
        self.assertAlmostEqual(summary["auc"], got["auc"].mean())
        self.assertAlmostEqual(summary["top_decile_lift"], got["top_decile_lift"].mean())

    def test_episode_clusters_use_trigger_session_distance(self):
        sessions = pd.bdate_range("2020-01-02", periods=30)
        trigger = pd.Series(
            [sessions[5], sessions[8], sessions[17], pd.NaT],
            index=sessions[:4],
        )
        got = study.assign_transition_episodes(trigger, sessions, max_gap_sessions=5)
        self.assertEqual(got.iloc[0], got.iloc[1])
        self.assertNotEqual(got.iloc[1], got.iloc[2])
        self.assertTrue(pd.isna(got.iloc[3]))

    def test_control_episodes_are_clustered_on_positive_origins_within_each_fold(self):
        sessions = pd.bdate_range("2020-01-02", periods=30)
        origins = sessions[[2, 4, 7, 9, 10, 14]]
        event = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
        folds = np.array([2020, 2020, 2020, 2020, 2021, 2021])
        got = study.assign_control_episodes(
            event,
            origins,
            folds,
            sessions,
            max_gap_sessions=5,
        )
        self.assertEqual(got.iloc[0], got.iloc[1])
        self.assertEqual(got.iloc[1], got.iloc[3])
        # A fold reset is a hard boundary even for adjacent source sessions.
        self.assertNotEqual(got.iloc[3], got.iloc[4])
        self.assertEqual(got.iloc[4], got.iloc[5])
        self.assertTrue(pd.isna(got.iloc[2]))

    def test_selectivity_interval_has_actual_and_control_cluster_components(self):
        n = 100
        index = pd.bdate_range("2020-01-02", periods=n)
        frame = pd.DataFrame(index=index)
        frame["phase"] = np.arange(n) % 5
        frame["event"] = (np.arange(n) * 7 % 13) < 5
        frame["candidate"] = np.linspace(0, 1, n)
        frame["episode"] = pd.array(
            [value // 10 if event else pd.NA for value, event in enumerate(frame["event"])],
            dtype="Int64",
        )
        controls = []
        for seed, offset in ((1, 0), (2, 2)):
            event_col = f"control_event_{seed}"
            score_col = f"control_score_{seed}"
            episode_col = f"control_episode_{seed}"
            frame[event_col] = np.roll(frame["event"].to_numpy(), offset)
            frame[score_col] = np.random.default_rng(seed).normal(size=n)
            frame[episode_col] = pd.array(
                [value // 12 if event else pd.NA for value, event in enumerate(frame[event_col])],
                dtype="Int64",
            )
            controls.append((event_col, score_col, episode_col, seed))
        protocol = {
            "tail_target": {"horizon_sessions": 5},
            "ranking_scoreboard": {"top_fraction": .1},
        }
        got = study.clustered_selectivity_interval(
            frame,
            protocol,
            candidate_score="candidate",
            controls=controls,
            metric="auc",
        )
        self.assertGreater(got["actual_episodes"], 1)
        self.assertEqual(set(got["control_episodes_by_seed"]), {"1", "2"})
        self.assertTrue(all(value > 1 for value in got["control_episodes_by_seed"].values()))
        self.assertEqual(
            set(got["variance_components"]), {"actual", "control_seed_1", "control_seed_2"}
        )
        self.assertLess(got["lower"], got["upper"])

    def test_leave_one_episode_out_jackknife_deletes_clusters_not_rows(self):
        frame = pd.DataFrame(
            {
                "event": [1, 1, 0, 1, 0, 0, 1, 0],
                "score": [.9, .8, .2, .7, .3, .1, .6, .4],
                "episode": pd.array([0, 0, pd.NA, 1, pd.NA, pd.NA, 2, pd.NA], dtype="Int64"),
            }
        )
        calls: list[int] = []

        def statistic(sample: pd.DataFrame) -> float:
            calls.append(len(sample))
            return float(sample["score"].mean())

        got = study.leave_one_episode_out(frame, episode_column="episode", statistic=statistic)
        self.assertEqual(got["episodes"], 3)
        self.assertEqual(sorted(calls), [6, 7, 7, 8])
        self.assertLess(got["lower"], got["upper"])


if __name__ == "__main__":
    unittest.main()
