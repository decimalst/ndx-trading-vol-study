"""Pre-written contract tests for the frozen GBM functional-form study."""
from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import config, models
from src.gbm_study import (
    DEFAULT_PROTOCOL,
    FEATURES,
    apply_locked_term,
    assert_leakage_fences,
    build_design,
    calculate_metrics,
    complete_rows,
    exact_interaction_probe,
    load_fenced_master,
    load_protocol,
    make_gbm,
    moving_block_means,
    paired_comparison,
    protocol_sha256,
    qlike,
    render_report,
    resolve_repo_path,
    smearing_variance,
    split_origins,
    training_rows,
    validate_artifacts,
    validate_lock,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _dummy_lock() -> dict:
    pairs = []
    for a_i, a in enumerate(FEATURES):
        for b in FEATURES[a_i + 1:]:
            pairs.append({
                "pair": [a, b], "score": 0.01, "grid_a": [0.0],
                "grid_b": [0.0], "surface": [[0.0]], "interaction": [[0.0]],
                "location": [0, 0], "location_value": 0.0,
            })
    return {
        "protocol_sha256": protocol_sha256(),
        "feature_order": list(FEATURES),
        "confirmation_results_read": False,
        "method": "exact quantile-grid two-way functional-ANOVA partial-dependence interaction",
        "pair_scores": pairs,
        "selected_pair": ["lrv_d", "liv"],
        "selected_score": 0.01,
        "selected_location": [0, 0],
        "selected_interaction_value": 0.0,
        "locked_term": {
            "kind": "product_of_one_sided_iqr_scaled_hinges",
            "features": [
                {"name": "lrv_d", "threshold": -9.0, "direction": 1,
                 "scale": 1.0, "discovery_median": -9.5},
                {"name": "liv", "threshold": 3.0, "direction": -1,
                 "scale": 0.5, "discovery_median": 3.2},
            ],
        },
    }


def _comparison(n: int = 100) -> dict:
    return {
        "n": n,
        "baseline_qlike": 0.30,
        "candidate_qlike": 0.29,
        "mean_difference": -0.01,
        "improvement_pct": 3.333,
        "win_count": 55,
        "win_rate": 0.55,
        "dm_statistic": -1.5,
        "dm_p_value": 0.13,
        "block_bootstrap_p_value": 0.12,
        "block_ci95": [-0.025, 0.004],
        "block_ci90": [-0.020, 0.001],
        "equivalence_margin": 0.009,
        "equivalent_3pct": False,
    }


class FrozenProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_protocol()

    def test_protocol_is_machine_readable_and_frozen_before_results(self):
        self.assertEqual(self.spec["protocol_version"], 1)
        self.assertEqual(self.spec["specified_on"], "2026-08-12")
        self.assertEqual(self.spec["status"], "frozen_before_first_empirical_run")
        self.assertTrue(self.spec["fences"]["forbid_clean_origins"])
        self.assertTrue(self.spec["fences"]["forbid_clean_targets"])
        self.assertEqual(len(protocol_sha256()), 64)

    def test_information_set_is_exactly_har_iv(self):
        self.assertEqual(tuple(self.spec["information_set"]["feature_order"]), FEATURES)
        self.assertEqual(FEATURES, ("lrv_d", "lrv_w", "lrv_m", "liv"))
        self.assertNotIn("lev_d", FEATURES)
        self.assertNotIn("earnings_wt", FEATURES)

    def test_one_fixed_tree_and_no_search(self):
        tree = self.spec["estimators"]["gbm"]
        self.assertEqual(tree["hyperparameter_search"], "none")
        self.assertFalse(tree["early_stopping"])
        self.assertEqual(tree["refit"], "expanding at every origin")
        model = make_gbm(self.spec)
        params = model.get_params()
        for key in ("learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
                    "min_samples_leaf", "l2_regularization", "max_bins",
                    "early_stopping", "random_state"):
            self.assertEqual(params[key], tree[key])

    def test_discovery_and_confirmation_are_fixed_and_disjoint(self):
        discovery = self.spec["splits"]["discovery"]
        confirmation = self.spec["splits"]["confirmation"]
        self.assertLess(pd.Timestamp(discovery["end"]), pd.Timestamp(confirmation["start"]))
        self.assertTrue(self.spec["splits"]["no_split_selection_after_results"])
        self.assertTrue(self.spec["interaction_probe"]["confirmation_only"])
        self.assertTrue(self.spec["interaction_probe"]["no_pair_or_threshold_reselection"])

    def test_outputs_are_isolated(self):
        for value in self.spec["outputs"].values():
            if isinstance(value, str) and ("/" in value):
                self.assertTrue(
                    value.startswith("data/gbm_study")
                    or value.startswith("reports/gbm_study")
                )


class LeakageAndDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_protocol()
        cls.master = load_fenced_master(cls.spec)
        cls.design = build_design(cls.master)

    def test_design_matches_existing_har_iv_builder_exactly(self):
        cfg = config.load()
        existing = models._har_design(
            self.master, cfg, with_x=False, with_iv=True
        )
        pd.testing.assert_frame_equal(
            self.design.loc[:, FEATURES], existing.loc[:, FEATURES],
            check_names=True, check_dtype=True,
        )
        pd.testing.assert_series_equal(
            self.design["y_next"], existing["y_next"], check_names=True,
        )

    def test_diagnostic_common_sample_is_exact_and_stops_before_clean(self):
        origins = split_origins(self.design, self.spec, "all_diagnostic")
        self.assertEqual(len(origins), 2463)
        self.assertEqual(origins.min(), pd.Timestamp("2016-01-04"))
        self.assertEqual(origins.max(), pd.Timestamp("2025-10-17"))
        rows = complete_rows(self.design).loc[origins]
        self.assertEqual(rows["target_date"].max(), pd.Timestamp("2025-10-20"))
        self.assertLess(rows["target_date"].max(), pd.Timestamp("2025-11-03"))
        assert_leakage_fences(rows, self.spec)

    def test_parquet_scan_itself_stops_at_last_permitted_target(self):
        self.assertEqual(self.master.index.max(), pd.Timestamp("2025-10-20"))
        self.assertLess(self.master.index.max(), pd.Timestamp("2025-11-03"))

    def test_training_rows_exclude_current_unknown_target(self):
        origin = pd.Timestamp("2016-01-04")
        train = training_rows(self.design, origin, 500)
        self.assertTrue((train.index < origin).all())
        self.assertNotIn(origin, train.index)
        self.assertGreaterEqual(len(train), 500)
        # The latest training target is realized at the origin or earlier.
        self.assertLessEqual(train["target_date"].max(), origin)

    def test_fence_raises_on_clean_origin(self):
        bad = pd.DataFrame(
            {"target_date": [pd.Timestamp("2025-11-04")]},
            index=pd.DatetimeIndex(["2025-11-03"], name="origin"),
        )
        with self.assertRaisesRegex(ValueError, "clean origin"):
            assert_leakage_fences(bad, self.spec)


class EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_protocol()

    def test_smearing_matches_definition(self):
        mu = -9.0
        residuals = np.array([-0.4, 0.0, 0.6])
        expected = np.exp(mu) * np.mean(np.exp(residuals))
        self.assertAlmostEqual(smearing_variance(mu, residuals), expected, places=15)

    def test_qlike_matches_hand_calculation(self):
        actual = np.array([1.0, 2.0])
        forecast = np.array([1.0, 1.0])
        expected = actual / forecast - np.log(actual / forecast) - 1.0
        np.testing.assert_allclose(qlike(actual, forecast), expected)

    def test_registered_interaction_fallback_detects_known_pair(self):
        class ProductModel:
            def predict(self, values):
                values = np.asarray(values)
                return values[:, 0] * values[:, 1] + 0.1 * values[:, 2]

        rng = np.random.default_rng(77)
        background = pd.DataFrame(rng.normal(size=(500, 4)), columns=FEATURES)
        result = exact_interaction_probe(ProductModel(), background, self.spec)
        self.assertEqual(result["selected_pair"], ["lrv_d", "lrv_w"])
        self.assertEqual(len(result["pair_scores"]), 6)
        scores = {tuple(x["pair"]): x["score"] for x in result["pair_scores"]}
        self.assertGreater(scores[("lrv_d", "lrv_w")], 0.5)
        for pair, score in scores.items():
            if pair != ("lrv_d", "lrv_w"):
                self.assertLess(score, 1e-20)

    def test_locked_hinge_term_is_deterministic(self):
        lock = _dummy_lock()
        frame = pd.DataFrame({
            "lrv_d": [-10.0, -8.0, -7.0],
            "lrv_w": [0.0, 0.0, 0.0],
            "lrv_m": [0.0, 0.0, 0.0],
            "liv": [3.5, 2.5, 2.0],
        })
        expected = np.array([0.0, 1.0, 4.0])
        np.testing.assert_allclose(apply_locked_term(frame, lock), expected)

    def test_block_bootstrap_is_deterministic_and_paired(self):
        values = np.arange(100, dtype=float) - 50
        a = moving_block_means(values, block=10, draws=100, seed=5)
        b = moving_block_means(values, block=10, draws=100, seed=5)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (100,))

    def test_paired_comparison_rewards_exact_candidate(self):
        spec = copy.deepcopy(self.spec)
        spec["scoreboard"]["robust_inference"]["draws"] = 200
        dates = pd.bdate_range("2020-01-02", periods=120)
        actual = np.exp(np.linspace(-10.0, -8.0, len(dates)))
        frame = pd.DataFrame({
            "actual_var": actual,
            "har_iv_var": actual * 1.25,
            "gbm_var": actual,
        }, index=dates)
        result = paired_comparison(frame, "gbm_var", spec)
        self.assertEqual(result["n"], 120)
        self.assertEqual(result["candidate_qlike"], 0.0)
        self.assertGreater(result["improvement_pct"], 99.9)
        self.assertEqual(result["win_rate"], 1.0)


class ArtifactAndReportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_protocol()

    def test_lock_rejects_protocol_or_future_contamination(self):
        lock = _dummy_lock()
        validate_lock(lock, self.spec)
        bad = copy.deepcopy(lock)
        bad["confirmation_results_read"] = True
        with self.assertRaisesRegex(ValueError, "before confirmation"):
            validate_lock(bad, self.spec)
        bad = copy.deepcopy(lock)
        bad["protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen protocol"):
            validate_lock(bad, self.spec)

    def test_generated_artifact_schema_and_exact_sample_contract(self):
        master = load_fenced_master(self.spec)
        design = build_design(master)
        origins = split_origins(design, self.spec, "all_diagnostic")
        frame = complete_rows(design).loc[origins, ["target_date", "actual_var"]].copy()
        frame["har_iv_var"] = frame["actual_var"]
        frame["gbm_var"] = frame["actual_var"]
        split_at = pd.Timestamp(self.spec["splits"]["confirmation"]["start"])
        frame["phase"] = np.where(frame.index < split_at, "discovery", "confirmation")
        frame["augmented_var"] = frame["actual_var"]
        frame.loc[frame["phase"] == "discovery", "augmented_var"] = np.nan
        metrics = {
            "protocol_sha256": protocol_sha256(),
            "sample": {"all_n": len(frame)},
        }
        validate_artifacts(frame, metrics, _dummy_lock(), self.spec)

    def test_report_contract_contains_every_required_outcome(self):
        comp = _comparison()
        metrics = {
            "protocol_sha256": protocol_sha256(),
            "gbm": {
                "all_diagnostic": comp,
                "discovery": comp,
                "confirmation": comp,
            },
            "augmented_term": {"confirmation": comp},
            "verdicts": {
                "gbm_functional_form": "INCONCLUSIVE",
                "locked_interpretable_term": "DOES_NOT_ADD",
            },
        }
        report = render_report(metrics, _dummy_lock())
        for required in (
            "all_diagnostic", "discovery", "confirmation", "HAR-IV QLIKE",
            "GBM QLIKE", "DM p", "block p", "win rate",
            "equivalent within 3%", "Exact interaction fallback",
            "Interpretable-term confirmation", "INCONCLUSIVE", "DOES_NOT_ADD",
            "sealed NDX clean origins were not read",
        ):
            self.assertIn(required, report)

    def test_saved_metrics_recompute_exactly_from_saved_forecasts(self):
        outputs = self.spec["outputs"]
        forecasts = pd.read_parquet(resolve_repo_path(outputs["forecasts"]))
        lock = json.loads(resolve_repo_path(outputs["interactions"]).read_text())
        saved = json.loads(resolve_repo_path(outputs["metrics"]).read_text())
        self.assertEqual(calculate_metrics(forecasts, lock, self.spec), saved)


if __name__ == "__main__":
    unittest.main()
