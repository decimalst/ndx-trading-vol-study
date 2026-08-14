"""Pre-written tests for the GBM post-result reviewer diagnostics."""
from __future__ import annotations

import json
import pathlib
import unittest

import numpy as np
import pandas as pd

from src.gbm_post_result import (
    DEFAULT_PROTOCOL,
    assign_realized_deciles,
    build_timing_safe_design,
    decompose_pair,
    load_protocol,
    qlike,
    resolve_repo_path,
    score_shap_interactions,
    timing_safe_assessment,
    validate_input_hashes,
    verify,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


class FrozenPostResultProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_protocol()

    def test_protocol_is_explicitly_post_result_and_parent_stays_inconclusive(self):
        self.assertEqual(self.spec["status"], "frozen_before_diagnostic_query")
        self.assertIn("post-result", self.spec["evidence_class"])
        self.assertEqual(self.spec["immutability"]["frozen_parent_verdict"], "INCONCLUSIVE")
        self.assertIn("data/gbm_study/metrics.json", self.spec["immutability"]["do_not_modify"])

    def test_exact_five_pairs_and_realized_deciles_are_frozen(self):
        self.assertEqual(set(self.spec["loss_decomposition"]["pairs"]), {
            "gbm_confirmation", "locked_term_confirmation",
            "timing_safe_gbm_confirmation",
            "earnings_with_iv_diagnostic", "earnings_without_iv_diagnostic",
        })
        self.assertEqual(self.spec["loss_decomposition"]["deciles"], 10)
        self.assertIn("Outcome-conditioned", self.spec["loss_decomposition"]["interpretation"])

    def test_earnings_inputs_are_tracked_non_smoothed_artifacts(self):
        inputs = self.spec["inputs"]["earnings_forecasts"]
        self.assertEqual(set(inputs), {"har_iv_x", "har_iv", "har_x", "har"})
        for item in inputs.values():
            self.assertNotIn("_sm", item["path"])
            self.assertTrue(item["path"].endswith("_all.parquet"))

    def test_shap_method_has_no_search_or_confirmation_use(self):
        shap = self.spec["correlation_audit"]["shap"]
        self.assertEqual(shap["parameter_search"], "none")
        self.assertEqual(shap["confirmation_data_use"], "none")
        self.assertEqual(shap["feature_perturbation"], "tree_path_dependent")

    def test_timing_sensitivity_changes_only_vxn_and_has_no_reselection(self):
        timing = self.spec["timing_safe_sensitivity"]
        self.assertIn("preceding", timing["information_change"])
        self.assertEqual(timing["tuning_or_reselection"], "none")
        self.assertIn("all three", timing["substantive_assessment_rule"])
        self.assertIn("not established as", self.spec["immutability"]["parent_timing_status"])

    def test_locked_input_hashes_match_current_files(self):
        validate_input_hashes(self.spec)


class LossDecompositionTests(unittest.TestCase):
    def test_deciles_are_equal_count_and_stably_break_ties(self):
        index = pd.date_range("2020-01-01", periods=100)
        actual = pd.Series(np.repeat(np.arange(10), 10), index=index)
        deciles = assign_realized_deciles(actual)
        self.assertEqual(deciles.value_counts().sort_index().tolist(), [10] * 10)
        self.assertTrue((deciles.iloc[:10] == 1).all())
        self.assertTrue((deciles.iloc[-10:] == 10).all())

    def test_qlike_penalizes_large_underforecast_more_than_overforecast(self):
        under = qlike(np.array([10.0]), np.array([1.0]))[0]
        over = qlike(np.array([1.0]), np.array([10.0]))[0]
        self.assertGreater(under, over)

    def test_decomposition_exactly_reconciles_mean_sum_and_fraction(self):
        index = pd.date_range("2020-01-01", periods=100)
        actual = np.linspace(1.0, 10.0, 100)
        frame = pd.DataFrame({
            "actual_var": actual,
            "baseline": np.full(100, 4.0),
            "candidate": np.r_[np.full(90, 4.1), np.full(10, 2.0)],
        }, index=index)
        rows, result = decompose_pair(
            frame, pair="synthetic", baseline_col="baseline", candidate_col="candidate"
        )
        decile_sum = sum(x["sum_loss_difference"] for x in result["deciles"].values())
        weighted = sum(
            x["mean_loss_difference"] * x["n"] for x in result["deciles"].values()
        ) / result["n"]
        self.assertAlmostEqual(decile_sum, result["sum_loss_difference"], places=13)
        self.assertAlmostEqual(weighted, result["mean_loss_difference"], places=13)
        self.assertAlmostEqual(
            sum(x["fraction_total_gap"] for x in result["deciles"].values()), 1.0, places=13
        )
        self.assertEqual(rows["realized_decile"].value_counts().tolist(), [10] * 10)

    def test_zero_total_gap_uses_null_fractions_without_division(self):
        index = pd.date_range("2020-01-01", periods=20)
        actual = np.linspace(1.0, 2.0, 20)
        frame = pd.DataFrame({
            "actual_var": actual, "baseline": actual, "candidate": actual,
        }, index=index)
        _, result = decompose_pair(
            frame, pair="zero", baseline_col="baseline", candidate_col="candidate"
        )
        self.assertTrue(all(x["fraction_total_gap"] is None for x in result["deciles"].values()))


class TimingSafeSensitivityTests(unittest.TestCase):
    def test_timing_safe_design_lags_only_vxn_feature(self):
        index = pd.bdate_range("2020-01-01", periods=30)
        master = pd.DataFrame({
            "rv_total": np.linspace(0.01, 0.03, 30),
            "vxn": np.linspace(10.0, 39.0, 30),
        }, index=index)
        from src.gbm_study import build_design
        original = build_design(master)
        safe = build_timing_safe_design(master)
        for column in ("lrv_d", "lrv_w", "lrv_m", "y_next", "actual_var", "target_date"):
            pd.testing.assert_series_equal(safe[column], original[column])
        expected = np.log(master["vxn"].shift(1))
        pd.testing.assert_series_equal(safe["liv"], expected, check_names=False)

    def test_substantive_assessment_requires_point_no_better_on_all_splits(self):
        no_better = {
            name: {"mean_difference": 0.01, "block_bootstrap_p_value": 0.2}
            for name in ("all_diagnostic", "discovery", "confirmation")
        }
        self.assertEqual(
            timing_safe_assessment(no_better), "SURVIVES_AT_1600_SAFE_BOUNDARY"
        )
        no_better["discovery"]["mean_difference"] = -0.001
        self.assertEqual(
            timing_safe_assessment(no_better), "DOES_NOT_SURVIVE_AT_1600_SAFE_BOUNDARY"
        )


class ShapScoreTests(unittest.TestCase):
    def test_off_diagonal_mean_absolute_interaction_selects_known_pair(self):
        cube = np.zeros((20, 4, 4))
        cube[:, 1, 2] = cube[:, 2, 1] = np.linspace(-2.0, 2.0, 20)
        cube[:, 0, 3] = cube[:, 3, 0] = 0.1
        scores = score_shap_interactions(cube)
        self.assertEqual(scores[0]["pair"], ["lrv_w", "lrv_m"])
        self.assertGreater(scores[0]["mean_absolute_interaction"], scores[1]["mean_absolute_interaction"])

    def test_bad_interaction_shape_fails_transparently(self):
        with self.assertRaisesRegex(ValueError, "unexpected SHAP interaction shape"):
            score_shap_interactions(np.zeros((5, 4)))


@unittest.skipUnless(
    resolve_repo_path(load_protocol()["outputs"]["metrics"]).exists(),
    "post-result artifacts not generated yet",
)
class SavedArtifactTests(unittest.TestCase):
    def test_saved_metrics_keep_parent_verdict_and_reconcile(self):
        spec = load_protocol()
        metrics = json.loads(resolve_repo_path(spec["outputs"]["metrics"]).read_text())
        self.assertEqual(metrics["parent_frozen_verdict"], "INCONCLUSIVE")
        self.assertEqual(set(metrics["pairs"]), set(spec["loss_decomposition"]["pairs"]))
        for result in metrics["pairs"].values():
            self.assertLessEqual(
                result["reconciliation"]["sum_absolute_error"],
                result["reconciliation"]["tolerance"],
            )
            self.assertLessEqual(
                result["reconciliation"]["mean_absolute_error"],
                result["reconciliation"]["tolerance"],
            )

    def test_saved_report_reproduces_exactly(self):
        verify(load_protocol())

    def test_verifier_recomputes_timing_and_decile_metrics(self):
        # The verifier rebuilds every decomposition from locked parent inputs
        # and rescoring the saved lagged-VXN forecasts; this is intentionally
        # stronger than a schema or rendered-report check.
        verify(load_protocol())


if __name__ == "__main__":
    unittest.main()
