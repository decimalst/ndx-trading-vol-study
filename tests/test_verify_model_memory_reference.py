"""Pre-score corruption checks for reference-model independent auditing."""
import unittest

import numpy as np
import pandas as pd

from src import verify_model_memory_reference as v


class ReferencePanelTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2022-01-03", periods=20)
        rows = []
        for horizon in (1, 5):
            for model in ("baseline", "gamma"):
                for i in range(10):
                    rows.append({"origin": dates[i], "horizon": horizon, "model": model,
                                 "y": 0.0001, "prediction": 0.00011, "target_end": dates[i + horizon]})
        self.core = pd.DataFrame(rows)
        combined = [self.core]
        for name in v.REFERENCE_MODELS:
            more = self.core.loc[self.core["model"] == "baseline"].copy()
            more["model"], more["prediction"] = name, 0.00012
            combined.append(more)
        self.combined = pd.concat(combined, ignore_index=True)

    def test_exact_same_origins_targets_and_preserved_core_accepted(self):
        result = v.verify_common_panel(self.core, self.combined)
        self.assertEqual(result["reference_rows"], 80)

    def test_missing_reference_origin_or_reference_model_is_rejected(self):
        for bad in (self.combined.iloc[:-1], self.combined.loc[self.combined["model"] != "xlstm"]):
            with self.assertRaises(AssertionError):
                v.verify_common_panel(self.core, bad)

    def test_changed_core_prediction_and_reference_target_are_rejected(self):
        for model, column in (("baseline", "prediction"), ("xlstm", "y")):
            bad = self.combined.copy()
            bad.loc[bad["model"] == model, column] *= 1.01
            with self.subTest(model=model), self.assertRaises(AssertionError):
                v.verify_common_panel(self.core, bad)

    def test_protected_target_timestamp_and_foreign_horizon_rejected(self):
        for column, value in (("target_end", pd.Timestamp("2026-01-01")), ("horizon", 21)):
            bad = self.combined.copy()
            bad.loc[bad.index[-1], column] = value
            with self.subTest(column=column), self.assertRaises(AssertionError):
                v.verify_common_panel(self.core, bad)


class JointMultiplicityTests(unittest.TestCase):
    def make_metrics(self):
        rows = [{"horizon": 1, "candidate": f"arm{i}", "control": "baseline", "p_conservative": 0.001 if i == 0 else 0.5,
                 "improvement_pct": 2, "periods": [{"delta": -0.01}] * 3} for i in range(46)]
        return {"rows": rows[:32]}, {"rows": rows[32:]}

    def test_entire_46_hypothesis_penalty_is_retained(self):
        core, reference = self.make_metrics()
        rows = v.joint_holm_rows(core, reference)
        self.assertAlmostEqual(rows[0]["p_holm_joint"], 0.046)
        self.assertEqual(rows[0]["verdict_joint"], "EXPLORATORY_SHORTLIST")

    def test_missing_reference_comparison_or_duplicate_hypothesis_is_rejected(self):
        core, reference = self.make_metrics()
        reference["rows"].pop()
        with self.assertRaises(AssertionError):
            v.joint_holm_rows(core, reference)
        core, reference = self.make_metrics()
        reference["rows"][-1] = core["rows"][0]
        with self.assertRaises(AssertionError):
            v.joint_holm_rows(core, reference)

    def test_joint_significance_without_period_stability_is_inconclusive(self):
        core, reference = self.make_metrics()
        core["rows"][0]["periods"] = [{"delta": 0.01}, {"delta": -0.01}, {"delta": -0.01}]
        rows = v.joint_holm_rows(core, reference)
        self.assertEqual(rows[0]["verdict_joint"], "INCONCLUSIVE")


class ReferenceCausalityTests(unittest.TestCase):
    def test_joint_neural_target_completion_and_actual_contexts(self):
        dates = pd.bdate_range("2010-01-04", periods=730)
        rng = np.random.default_rng(104)
        columns = list(v.independent.BASELINE[1:])
        features = pd.DataFrame(rng.normal(size=(730, 11)), index=dates, columns=columns)
        features["rv_total"] = np.exp(rng.normal(-8, 0.3, 730))
        origins, windows, target, end = v.neural_training_data(features, dates, dates[700])
        self.assertEqual(origins[-1], dates[695])
        self.assertEqual(end, dates[700])
        np.testing.assert_array_equal(windows[-1], features.loc[dates[674]:dates[695], columns])
        self.assertAlmostEqual(target[-1, 1], features["rv_total"].iloc[696:701].mean())
        poisoned = features.copy()
        poisoned.iloc[701:] = 1000
        changed = v.neural_training_data(poisoned, dates, dates[700])
        self.assertTrue(changed[0].equals(origins))
        np.testing.assert_array_equal(changed[1], windows)
        np.testing.assert_array_equal(changed[2], target)

    def test_neural_missing_actual_session_excludes_entire_22_session_context(self):
        dates = pd.bdate_range("2010-01-04", periods=730)
        features = pd.DataFrame(np.ones((730, 11)), index=dates, columns=v.independent.BASELINE[1:])
        features["rv_total"] = 0.0001
        features.loc[dates[600], "liv"] = np.nan
        origins, _, _, _ = v.neural_training_data(features, dates, dates[700])
        self.assertFalse(origins.isin(dates[600:622]).any())
        self.assertIn(dates[622], origins)

    def test_moirai_context_fence_and_consecutive_session_metadata(self):
        dates = pd.bdate_range("2022-01-03", periods=8)
        features = pd.DataFrame({"rv_total": np.arange(1, 9)}, index=dates)
        summaries = pd.DataFrame({"moirai_log_h1": np.ones(6), "moirai_log_h5": np.ones(6)}, index=dates[2:])
        contexts = pd.DataFrame({"context_start": dates[:6], "context_end": dates[2:], "context_rows": 3}, index=dates[2:])
        protocol = {"moirai": {"extraction_start": str(dates[2].date()), "context_sessions": 3},
                    "sample": {"score_end": str(dates[-1].date())}}
        result = v.verify_moirai_contexts(features, summaries, contexts, protocol)
        self.assertEqual(result["moirai_contexts_checked"], 6)
        wrong = contexts.copy()
        wrong.loc[dates[2], "context_end"] = dates[3]
        with self.assertRaises(AssertionError):
            v.verify_moirai_contexts(features, summaries, wrong, protocol)
        wrong = contexts.copy()
        wrong.loc[dates[2], "context_rows"] = 2
        with self.assertRaises(AssertionError):
            v.verify_moirai_contexts(features, summaries, wrong, protocol)

    def test_nlinear_independent_architecture_matches_direct_weight_algebra(self):
        import torch
        torch.manual_seed(83)
        model = v.make_neural_replay("nlinear")
        inputs = torch.randn(3, 22, 11)
        state = model.state_dict()
        last = inputs[:, -1:, :]
        normalized = (inputs - last).transpose(1, 2)
        compressed = normalized @ state["temporal.weight"].T + state["temporal.bias"] + last.transpose(1, 2)
        encoded = compressed @ state["pre_encoding.weight"].T + state["pre_encoding.bias"]
        expected = encoded.reshape(3, -1) @ state["readout.weight"].T + state["readout.bias"]
        torch.testing.assert_close(model(inputs), expected)


if __name__ == "__main__":
    unittest.main()
