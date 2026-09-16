"""Pre-fit publication failures preserve the whole two-contrast joint-risk family."""

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src import joint_risk_search as study


class TestJointRiskPublication(unittest.TestCase):
    def _assert_failed_publication(self, fault):
        # All measurements, fits and scores are synthetic stubs.
        # Only persistence and failure transitions run, in a temporary tree.
        scored = {
            "rows": [
                {
                    "study": "joint_risk",
                    "candidate": candidate,
                    "control": control,
                    "score": score,
                    "horizon": 1,
                    "p_conservative": 0.00001,
                    "phases": [],
                }
                for candidate, control, score in study.CONTRASTS
            ],
            "leads": ["dynamic_correlation"],
            "hypothesis_count": 2,
            "cumulative_hypothesis_count": 112,
        }
        if fault == "nonfinite_scored":
            scored["rows"][0]["p_conservative"] = float("nan")
        prior = [
            {
                "study": "synthetic",
                "candidate": f"old_{i}",
                "control": "baseline",
                "horizon": 1,
                "p_conservative": 1.0,
            }
            for i in range(110)
        ]
        source_audit = {"synthetic": True, "diagnostic": "preserve source audit"}
        measurement_audit = {
            "synthetic": True,
            "status": "PASS",
            "diagnostic": "preserve measurement audit",
        }
        fit_audit = [{"synthetic": True, "diagnostic": "preserve completed fits"}]
        original_ledger = study.ledger
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, output, protocol = root / "report", root / "output", root / "protocol.yaml"
            protocol.write_text(
                json.dumps(
                    {
                        "sources": {
                            "qqq": "qqq.parquet",
                            "daily": "spx.parquet",
                            "vxn": "vxn.csv",
                            "vix": "vix.csv",
                            "vix9d": "vix9d.csv",
                            "vvix": "vvix.csv",
                        },
                        "index": {},
                    }
                )
            )

            def publish_report(_metrics):
                (report / "results.md").write_text("Synthetic scored lead, partly published.")
                if fault == "report":
                    raise OSError("Injected report publication failure")

            def publish_ledger(rows):
                if rows and rows[0]["event"] == "evaluated" and fault == "evaluated_ledger":
                    original_ledger(rows[:1])
                    raise OSError("Injected evaluated-ledger publication failure")
                if rows and rows[0]["event"] == "registered" and fault == "registered_ledger":
                    original_ledger(rows[:1])
                    raise OSError("Injected registered-ledger publication failure")
                original_ledger(rows)

            def require_measurement(_audit):
                if fault == "measurement":
                    raise ValueError("INSUFFICIENT_DATA: injected measurement failure")

            if fault == "measurement":
                measurement_audit["status"] = "INSUFFICIENT_MEASUREMENT"

            def score_synthetic(*_args):
                if fault == "protocol_mutation":
                    protocol.write_text(protocol.read_text() + "\n")
                return scored

            def source_hash(path):
                return (
                    hashlib.sha256(protocol.read_bytes()).hexdigest()
                    if Path(path) == protocol
                    else "0" * 64
                )

            def save_synthetic_parquet(_frame, path, *args, **kwargs):
                Path(path).write_bytes(b"synthetic retained row artifact")

            replacements = {
                "ROOT": root,
                "REPORT": report,
                "OUT": output,
                "PROTOCOL": protocol,
                "validate": lambda _p: None,
                "inherited": lambda _p: prior,
                "evaluate": score_synthetic,
                "report": publish_report,
                "ledger": publish_ledger,
                "version": lambda _name: "synthetic",
            }
            for name, value in replacements.items():
                stack.enter_context(patch.object(study, name, value))
            stack.enter_context(
                patch.object(
                    study.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        returncode=0, stdout="synthetic pre-run success", stderr=""
                    ),
                )
            )
            stack.enter_context(patch.object(study.np, "show_config", return_value=None))
            stack.enter_context(
                patch.object(study.inference, "digest", side_effect=source_hash)
            )
            stack.enter_context(
                patch.object(
                    study.cf, "load_sources", return_value=(None, None, None, source_audit)
                )
            )
            stack.enter_context(
                patch.object(study.cf, "measurement_audit", return_value=measurement_audit)
            )
            stack.enter_context(
                patch.object(study.cf, "require_measurement", side_effect=require_measurement)
            )
            build = stack.enter_context(
                patch.object(
                    study.cf, "build_features", return_value=(pd.DataFrame(), pd.DataFrame())
                )
            )
            fit = stack.enter_context(
                patch.object(
                    study.cm,
                    "forecast_panel",
                    return_value=(pd.DataFrame({"synthetic": [1]}), fit_audit),
                )
            )
            stack.enter_context(
                patch.object(pd.DataFrame, "to_parquet", new=save_synthetic_parquet)
            )
            expected_error = (
                ValueError
                if fault in {"protocol_mutation", "measurement", "nonfinite_scored"}
                else OSError
            )
            expected_message = (
                "[Pp]rotocol"
                if fault == "protocol_mutation"
                else ("INSUFFICIENT_DATA" if fault == "measurement" else "Injected")
            )
            if fault == "nonfinite_scored":
                expected_message = "JSON"
            with self.assertRaisesRegex(expected_error, expected_message):
                study.run()

            failed = json.loads((report / "metrics.json").read_text())
            self.assertEqual(failed["status"], "UNEVALUABLE")
            self.assertTrue(failed["whole_wave_aborted"])
            self.assertEqual(
                (failed["hypothesis_count"], failed["cumulative_hypothesis_count"]), (2, 112)
            )
            self.assertEqual(len(failed["rows"]), 2)
            self.assertEqual(
                {(r["candidate"], r["control"], r["score"]) for r in failed["rows"]},
                set(study.CONTRASTS),
            )
            expected_status = "INSUFFICIENT_DATA" if fault == "measurement" else "INVALID_RUN"
            self.assertTrue(
                all(
                    r["p_conservative"] == 1 and r["status"] == expected_status
                    for r in failed["rows"]
                )
            )
            self.assertTrue(
                all(
                    r["p_holm_wave"] == 1 and r["p_holm_cumulative"] == 1
                    for r in failed["rows"]
                )
            )
            self.assertEqual(failed["leads"], [])
            self.assertEqual(study.candidate_leads(failed["rows"]), [])
            self.assertEqual(json.loads((report / "failure.json").read_text()), failed)

            if fault in {"registered_ledger", "measurement"}:
                self.assertFalse((report / "unpublished_scored_metrics.json").exists())
                self.assertFalse((output / "fits.json").exists())
                self.assertFalse((output / "forecasts.parquet").exists())
                build.assert_not_called()
                fit.assert_not_called()
                self.assertFalse((output / "features.parquet").exists())
                self.assertFalse((output / "targets.parquet").exists())
                if fault == "measurement":
                    self.assertEqual(
                        json.loads((output / "measurement_audit.json").read_text()),
                        measurement_audit,
                    )
                    self.assertEqual(
                        json.loads((output / "source_audit.json").read_text()), source_audit
                    )
                else:
                    self.assertFalse((output / "measurement_audit.json").exists())
            else:
                diagnostic = json.loads(
                    (report / "unpublished_scored_metrics.json").read_text()
                )
                self.assertEqual(diagnostic["status"], "UNPUBLISHED_DIAGNOSTIC_ONLY")
                self.assertTrue(diagnostic["not_for_inherited_inference_or_promotion"])
                if fault == "nonfinite_scored":
                    self.assertIn("nan", diagnostic["nonserializable_metrics_repr"])
                    self.assertNotIn("scored_metrics", diagnostic)
                else:
                    self.assertEqual(diagnostic["scored_metrics"], scored)
                self.assertEqual(
                    json.loads((output / "source_audit.json").read_text()), source_audit
                )
                self.assertEqual(json.loads((output / "fits.json").read_text()), fit_audit)
                self.assertEqual(
                    json.loads((output / "measurement_audit.json").read_text()),
                    measurement_audit,
                )
                for name in ("features", "targets", "forecasts"):
                    self.assertEqual(
                        (output / f"{name}.parquet").read_bytes(),
                        b"synthetic retained row artifact",
                    )

            result_text = (report / "results.md").read_text()
            self.assertIn("UNEVALUABLE", result_text)
            self.assertNotIn("Synthetic scored lead", result_text)
            events = [
                json.loads(line)
                for line in (report / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(sum(r["event"] == "inherited" for r in events), 110)
            self.assertEqual(
                sum(r["event"] == "registered" for r in events),
                1 if fault == "registered_ledger" else 2,
            )
            self.assertEqual(
                sum(r["event"] == "evaluated" for r in events),
                int(fault == "evaluated_ledger"),
            )
            self.assertTrue(
                all(
                    r["event"] == "unevaluable" and r["p_conservative"] == 1
                    for r in events[-2:]
                )
            )
            self.assertEqual(
                {(r["candidate"], r["control"], r["score"]) for r in events[-2:]},
                set(study.CONTRASTS),
            )

    def test_nonfinite_scored_payload_still_records_both_failed_trials(self):
        self._assert_failed_publication("nonfinite_scored")

    def test_measurement_failure_retains_audit_and_skips_all_builds_and_fits(self):
        self._assert_failed_publication("measurement")

    def test_report_failure_invalidates_all_two_and_preserves_diagnostics(self):
        self._assert_failed_publication("report")

    def test_partial_evaluated_ledger_failure_invalidates_all_two_and_preserves_diagnostics(
        self,
    ):
        self._assert_failed_publication("evaluated_ledger")

    def test_partial_registration_failure_still_accounts_for_all_two(self):
        self._assert_failed_publication("registered_ledger")

    def test_protocol_mutation_invalidates_scores_and_preserves_diagnostics(self):
        self._assert_failed_publication("protocol_mutation")


if __name__ == "__main__":
    unittest.main()
