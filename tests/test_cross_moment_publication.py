"""Prewritten new-wave failure accounting; upstream artifacts are read-only."""

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src import cross_moment_search as study


class CrossMomentPublication(unittest.TestCase):
    def exercise(self, fault):
        scored = {
            "rows": [
                {
                    "study": "cross_moment",
                    "candidate": a,
                    "control": b,
                    "score": s,
                    "horizon": 1,
                    "p_conservative": 1e-5,
                    "phases": [],
                }
                for a, b, s in study.CONTRASTS
            ],
            "leads": ["dynamic_correlation"],
            "hypothesis_count": 2,
            "cumulative_hypothesis_count": 114,
            "new_model_fits": 0,
            "new_forecasts": 0,
        }
        if fault == "nonfinite":
            scored["rows"][0]["p_conservative"] = float("nan")
        prior = [
            {
                "study": "synthetic",
                "candidate": str(i),
                "control": "baseline",
                "horizon": 1,
                "p_conservative": 1.0,
            }
            for i in range(112)
        ]
        proof = {"status": "UPSTREAM_VERIFIED_READ_ONLY", "synthetic": True}
        original_ledger = study.ledger
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            report = root / "report"
            output = root / "output"
            protocol = root / "new.yaml"
            old = root / "upstream.txt"
            old.write_text("untouched original source/report bytes")
            old_bytes = old.read_bytes()
            for name in ["old.parquet", "features.parquet"]:
                (root / name).write_bytes(b"synthetic pinned parquet bytes")
            protocol.write_text(
                json.dumps(
                    {
                        "upstream": {
                            "forecasts": "old.parquet",
                            "features": "features.parquet",
                        },
                        "index": {},
                    }
                )
            )

            def score(*_args):
                if fault == "protocol":
                    protocol.write_text(protocol.read_text() + "\n")
                return scored

            def admit(_root):
                if fault == "admission":
                    raise ValueError("UPSTREAM_NOT_ADMITTED: injected mismatch")
                if fault in {"forecast_changed", "calendar_changed"}:
                    name = "old.parquet" if fault == "forecast_changed" else "features.parquet"
                    (root / name).write_bytes(b"changed after successful admission")
                return proof

            def render(_metrics):
                (report / "results.md").write_text("Synthetic provisional result")
                if fault == "report":
                    raise OSError("Injected report error")

            def write_ledger(rows):
                if rows and rows[0]["event"] == "registered" and fault == "registered":
                    original_ledger(rows[:1])
                    raise OSError("Injected partial registration")
                if rows and rows[0]["event"] == "evaluated" and fault == "evaluated":
                    original_ledger(rows[:1])
                    raise OSError("Injected partial evaluated ledger")
                original_ledger(rows)

            def save_frame(_frame, path, *_args, **_kwargs):
                Path(path).write_bytes(b"synthetic scored rows")

            def hash_file(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()

            for name, value in {
                "ROOT": root,
                "REPORT": report,
                "OUT": output,
                "PROTOCOL": protocol,
                "validate": lambda p: None,
                "input_paths": lambda p: {"old.parquet", "features.parquet"},
                "inherited": lambda p: prior,
                "validate_upstream": admit,
                "evaluate": score,
                "report": render,
                "ledger": write_ledger,
                "version": lambda name: "synthetic",
            }.items():
                stack.enter_context(patch.object(study, name, value))
            stack.enter_context(
                patch.object(
                    study.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        returncode=0, stdout="synthetic tests pass", stderr=""
                    ),
                )
            )
            stack.enter_context(patch.object(study.np, "show_config", return_value=None))
            stack.enter_context(patch.object(study.inference, "digest", side_effect=hash_file))
            stack.enter_context(
                patch.object(
                    pd,
                    "read_parquet",
                    return_value=pd.DataFrame(index=pd.bdate_range("2020-01-02", periods=3)),
                )
            )
            newscore = stack.enter_context(
                patch.object(
                    study.cs, "score_panel", return_value=pd.DataFrame({"synthetic": [1]})
                )
            )
            stack.enter_context(patch.object(pd.DataFrame, "to_parquet", new=save_frame))
            exception = (
                ValueError
                if fault
                in {
                    "admission",
                    "protocol",
                    "nonfinite",
                    "forecast_changed",
                    "calendar_changed",
                }
                else OSError
            )
            with self.assertRaises(exception):
                study.run()
            canonical = json.loads((report / "metrics.json").read_text())
            self.assertEqual(canonical, json.loads((report / "failure.json").read_text()))
            self.assertEqual(canonical["status"], "UNEVALUABLE")
            self.assertEqual(canonical["leads"], [])
            self.assertEqual(
                (canonical["hypothesis_count"], canonical["cumulative_hypothesis_count"]),
                (2, 114),
            )
            self.assertTrue(
                all(
                    r["p_conservative"] == r["p_holm_wave"] == r["p_holm_cumulative"] == 1
                    for r in canonical["rows"]
                )
            )
            events = [
                json.loads(s) for s in (report / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(sum(x["event"] == "inherited" for x in events), 112)
            self.assertEqual(
                sum(x["event"] == "registered" for x in events),
                1 if fault == "registered" else 2,
            )
            self.assertEqual(
                sum(x["event"] == "evaluated" for x in events), int(fault == "evaluated")
            )
            self.assertEqual([x["event"] for x in events[-2:]], ["unevaluable"] * 2)
            self.assertTrue(all(x["p_conservative"] == 1 for x in events[-2:]))
            self.assertEqual(old.read_bytes(), old_bytes)
            self.assertIn("UNEVALUABLE", (report / "results.md").read_text())
            if fault in {"registered", "admission", "forecast_changed", "calendar_changed"}:
                newscore.assert_not_called()
                self.assertFalse((output / "scored_forecasts.parquet").exists())
                self.assertFalse((report / "unpublished_scored_metrics.json").exists())
            else:
                self.assertEqual(
                    (output / "scored_forecasts.parquet").read_bytes(),
                    b"synthetic scored rows",
                )
                self.assertEqual(
                    json.loads((output / "upstream_admission.json").read_text()), proof
                )
                backup = json.loads((report / "unpublished_scored_metrics.json").read_text())
                self.assertTrue(backup["not_for_inherited_inference_or_promotion"])
                if fault == "nonfinite":
                    self.assertIn("nan", backup["nonserializable_metrics_repr"])
                else:
                    self.assertEqual(backup["scored_metrics"], scored)

    def test_admission_failure_records_trials_and_never_scores(self):
        self.exercise("admission")

    def test_partial_registration_retains_all_hypotheses(self):
        self.exercise("registered")

    def test_partial_evaluated_ledger_retains_all_failed_trials(self):
        self.exercise("evaluated")

    def test_report_failure_preserves_diagnostics(self):
        self.exercise("report")

    def test_protocol_mutation_invalidates_new_scores_only(self):
        self.exercise("protocol")

    def test_nonfinite_score_payload_cannot_skip_terminal_ledger(self):
        self.exercise("nonfinite")

    def test_forecast_changed_after_admission_never_reaches_scorer(self):
        self.exercise("forecast_changed")

    def test_calendar_changed_after_admission_never_reaches_scorer(self):
        self.exercise("calendar_changed")


if __name__ == "__main__":
    unittest.main()
