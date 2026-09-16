"""Synthetic publication faults cannot leave an eligible measurement candidate."""
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src import measurement_search as study


class TestMeasurementPublication(unittest.TestCase):
    def _assert_failed_publication(self, fault):
        # The complete I/O transition is exercised in a temporary directory.
        # No real sources, numerical fits, or score calculations are invoked.
        scored = {
            "rows": [
                {"study": "measurement_memory", "candidate": candidate,
                 "control": control, "measure": measure, "horizon": 1,
                 "p_conservative": .00001, "phases": []}
                for candidate, control, measure in study.CONTRASTS
            ],
            "leads": ["width", "quality"],
            "hypothesis_count": 15,
            "cumulative_hypothesis_count": 99,
        }
        prior = [{"study": "synthetic", "candidate": f"old_{i}",
                  "control": "baseline", "horizon": 1, "p_conservative": 1.}
                 for i in range(84)]
        source_audit = {"synthetic": True, "diagnostic": "preserve source audit"}
        fit_audit = [{"synthetic": True, "diagnostic": "preserve completed fits"}]
        original_ledger = study.ledger
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, output, protocol = root/"report", root/"output", root/"protocol.yaml"
            protocol.write_text(json.dumps({"sources": {
                "risklab": "risklab.txt", "daily": "daily.parquet",
                "vix": "vix.csv", "vix9d": "vix9d.csv", "vvix": "vvix.csv"},
                "index": {}}))

            def publish_report(_metrics):
                (report/"results.md").write_text("Synthetic scored lead, partly published.")
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

            def score_synthetic(*_args):
                if fault == "protocol_mutation":
                    protocol.write_text(protocol.read_text()+"\n")
                return scored

            def source_hash(path):
                return hashlib.sha256(protocol.read_bytes()).hexdigest() if Path(path) == protocol else "0"*64

            def save_synthetic_parquet(_frame, path, *args, **kwargs):
                Path(path).write_bytes(b"synthetic retained row artifact")

            replacements = {
                "ROOT": root, "REPORT": report, "OUT": output, "PROTOCOL": protocol,
                "validate": lambda _p: None, "inherited": lambda _p: prior,
                "forecast_panel": lambda *_args: (pd.DataFrame({"synthetic": [1]}), fit_audit),
                "evaluate": score_synthetic, "report": publish_report,
                "ledger": publish_ledger, "version": lambda _name: "synthetic",
            }
            for name, value in replacements.items():
                stack.enter_context(patch.object(study, name, value))
            stack.enter_context(patch.object(study.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="synthetic pre-run success", stderr="")))
            stack.enter_context(patch.object(study.np, "show_config", return_value=None))
            stack.enter_context(patch.object(study.inference, "digest", side_effect=source_hash))
            stack.enter_context(patch.object(study.mm, "load_sources", return_value=(None, None, None, source_audit)))
            stack.enter_context(patch.object(study.mm, "build_features", return_value=(pd.DataFrame(), pd.DataFrame())))
            stack.enter_context(patch.object(pd.DataFrame, "to_parquet", new=save_synthetic_parquet))
            expected_error = ValueError if fault == "protocol_mutation" else OSError
            expected_message = "[Pp]rotocol" if fault == "protocol_mutation" else "Injected"
            with self.assertRaisesRegex(expected_error, expected_message):
                study.run()

            failed = json.loads((report/"metrics.json").read_text())
            self.assertEqual(failed["status"], "UNEVALUABLE")
            self.assertTrue(failed["whole_wave_aborted"])
            self.assertEqual((failed["hypothesis_count"], failed["cumulative_hypothesis_count"]), (15, 99))
            self.assertEqual(len(failed["rows"]), 15)
            self.assertEqual({(r["candidate"], r["control"], r["measure"]) for r in failed["rows"]},
                             set(study.CONTRASTS))
            self.assertTrue(all(r["p_conservative"] == 1 and r["status"] == "INVALID_RUN"
                                for r in failed["rows"]))
            self.assertEqual(failed["leads"], [])
            self.assertEqual(study.candidate_leads(failed["rows"]), [])
            self.assertEqual(json.loads((report/"failure.json").read_text()), failed)

            if fault == "registered_ledger":
                self.assertFalse((report/"unpublished_scored_metrics.json").exists())
                self.assertFalse((output/"fits.json").exists())
                self.assertFalse((output/"forecasts.parquet").exists())
            else:
                diagnostic = json.loads((report/"unpublished_scored_metrics.json").read_text())
                self.assertEqual(diagnostic["status"], "UNPUBLISHED_DIAGNOSTIC_ONLY")
                self.assertTrue(diagnostic["not_for_inherited_inference_or_promotion"])
                self.assertEqual(diagnostic["scored_metrics"], scored)
                self.assertEqual(json.loads((output/"source_audit.json").read_text()), source_audit)
                self.assertEqual(json.loads((output/"fits.json").read_text()), fit_audit)
                for name in ("features", "targets", "forecasts"):
                    self.assertEqual((output/f"{name}.parquet").read_bytes(), b"synthetic retained row artifact")

            result_text = (report/"results.md").read_text()
            self.assertIn("UNEVALUABLE", result_text)
            self.assertNotIn("Synthetic scored lead", result_text)
            events = [json.loads(line) for line in (report/"trial_ledger.jsonl").read_text().splitlines()]
            self.assertEqual(sum(r["event"] == "inherited" for r in events), 84)
            self.assertEqual(sum(r["event"] == "registered" for r in events), 1 if fault == "registered_ledger" else 15)
            self.assertEqual(sum(r["event"] == "evaluated" for r in events), int(fault == "evaluated_ledger"))
            self.assertTrue(all(r["event"] == "unevaluable" and r["p_conservative"] == 1 for r in events[-15:]))
            self.assertEqual({(r["candidate"], r["control"], r["measure"]) for r in events[-15:]},
                             set(study.CONTRASTS))

    def test_report_failure_invalidates_all_fifteen_and_preserves_diagnostics(self):
        self._assert_failed_publication("report")

    def test_partial_evaluated_ledger_failure_invalidates_all_fifteen_and_preserves_diagnostics(self):
        self._assert_failed_publication("evaluated_ledger")

    def test_partial_registration_failure_still_accounts_for_all_fifteen(self):
        self._assert_failed_publication("registered_ledger")

    def test_protocol_mutation_invalidates_scores_and_preserves_diagnostics(self):
        self._assert_failed_publication("protocol_mutation")


if __name__ == "__main__":
    unittest.main()
