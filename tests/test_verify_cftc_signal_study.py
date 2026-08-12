"""Contracts for the independent CFTC positioning-study verifier."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import verify_cftc_signal_study as verifier


class IndependenceContract(unittest.TestCase):
    def test_verifier_does_not_import_the_study_implementation(self):
        tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertFalse(
            any(name.endswith("free_signal_study") for name in imported), imported
        )


class TimingAndLabelContract(unittest.TestCase):
    def test_release_is_delayed_ten_days_and_mapped_once(self):
        protocol = verifier.load_protocol()
        reports = pd.DataFrame(
            {
                "report_date": ["2024-01-02", "2024-01-09"],
                "contract_code": ["20974+", "20974+"],
                "lev_long": [30, 35],
                "lev_short": [10, 15],
                "open_interest": [100, 100],
            }
        )
        sessions = pd.bdate_range("2024-01-01", "2024-01-31")
        got = verifier.reconstruct_release_origins(reports, sessions, protocol)
        self.assertEqual(
            got["available_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-12", "2024-01-19"],
        )
        self.assertEqual(
            got["origin"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-12", "2024-01-19"],
        )
        self.assertTrue(got["origin"].is_unique)

    def test_targets_use_exactly_the_next_five_sessions(self):
        index = pd.bdate_range("2024-01-02", periods=10)
        log_rv = pd.Series(
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=index,
        )
        got = verifier.reconstruct_fold_targets(
            log_rv,
            cutoff=index[1],
            origins=pd.DatetimeIndex([index[2], index[3], index[4]]),
            horizon=5,
            quantile=0.8,
        )
        self.assertEqual(got.loc[index[2], "target_end"], index[7])
        self.assertTrue(bool(got.loc[index[2], "event"]))
        self.assertEqual(got.loc[index[2], "trigger_date"], index[3])
        self.assertFalse(bool(got.loc[index[3], "calm"]))
        self.assertFalse(bool(got.loc[index[4], "event"]))
        self.assertTrue(pd.isna(got.loc[index[4], "trigger_date"]))

    def test_negative_label_with_nat_trigger_is_not_dropped(self):
        index = pd.DatetimeIndex(["2024-01-05", "2024-01-12"])
        labels = pd.DataFrame(
            {
                "event": [False, True],
                "trigger_date": [pd.NaT, pd.Timestamp("2024-01-16")],
            },
            index=index,
        )
        features = pd.DataFrame({"log_rv_d": [1.0, 2.0]}, index=index)
        got = verifier.join_scorable(labels, features, ["log_rv_d"])
        self.assertEqual(got["event"].tolist(), [False, True])
        self.assertTrue(pd.isna(got.iloc[0]["trigger_date"]))


class ScoreContract(unittest.TestCase):
    def test_auc_and_top_fraction_lift_are_recomputed_from_rows(self):
        frame = pd.DataFrame(
            {
                "event": [False, True, False, True],
                "p_model": [0.1, 0.8, 0.2, 0.9],
            }
        )
        got = verifier.ranking_summary(frame, "model", top_fraction=0.25)
        self.assertEqual(got["n"], 4)
        self.assertEqual(got["positives"], 2)
        self.assertAlmostEqual(got["auc"], 1.0)
        self.assertAlmostEqual(got["top_decile_event_rate"], 1.0)
        self.assertAlmostEqual(got["top_decile_lift"], 2.0)

    def test_nonfinite_probability_is_rejected(self):
        frame = pd.DataFrame({"event": [False, True], "p_model": [0.1, np.nan]})
        with self.assertRaisesRegex(AssertionError, "probability"):
            verifier.ranking_summary(frame, "model", top_fraction=0.1)


class CurrentArtifactContract(unittest.TestCase):
    def test_current_artifacts_pass_full_independent_verification(self):
        got = verifier.verify_artifacts()
        self.assertEqual(got["status"], "PASS")
        self.assertEqual(got["rows"], 591)
        self.assertEqual(got["positives"], 99)
        self.assertEqual(got["negative_labels_with_nat_trigger"], 492)
        self.assertEqual(len(got["source_sha256"]), 64)
        self.assertGreaterEqual(got["checks"], 12)


if __name__ == "__main__":
    unittest.main()
