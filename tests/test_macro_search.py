"""Pre-score proper-loss inference and complete-family contracts."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import macro_search
from src.macro_search import adjust, failure_rows, paired_inference, passes, sensitivity


class TestMacroSearch(unittest.TestCase):
    def test_negative_raw_scores_have_absolute_gaps_without_percentage(self):
        control = -10+np.arange(140)/1000
        candidate = control-.02
        result = paired_inference(candidate, control, {"inference": {"blocks": [21, 63, 126], "bootstrap_draws": 199}}, 17)
        self.assertAlmostEqual(result["delta"], -.02)
        self.assertNotIn("improvement_pct", result)
        self.assertLess(result["control_loss"], 0)

    def test_common_score_constants_leave_inference_unchanged(self):
        rng = np.random.default_rng(2)
        control = np.ones(140)*-5
        candidate = control+rng.normal(-.02, .1, 140)
        p = {"inference": {"blocks": [21, 63, 126], "bootstrap_draws": 199}}
        first = paired_inference(candidate, control, p, 17)
        second = paired_inference(candidate+10, control+10, p, 17)
        self.assertAlmostEqual(first["delta"], second["delta"])
        self.assertEqual(first["p_conservative"], second["p_conservative"])

    def test_gate_uses_absolute_gap_both_phases_and_sensitivities(self):
        row = {"p_holm_wave": .001, "p_holm_cumulative": .01, "phases": [
            {"name": "development", "delta": -.005, "action_sensitivity": {"n": 10, "delta": -.001}},
            {"name": "evaluation", "delta": -.006, "action_sensitivity": {"n": 10, "delta": -.001},
             "stability": [{"delta": -.001}, {"delta": -.002}]}]}
        self.assertTrue(passes(row))
        row["phases"][0]["delta"] = -.0049
        self.assertFalse(passes(row))
        row["phases"][0]["delta"] = -.005
        row["phases"][1]["action_sensitivity"]["delta"] = 0
        self.assertFalse(passes(row))

    def test_every_failed_hypothesis_remains_in_both_families(self):
        rows = failure_rows(ValueError("INSUFFICIENT_DATA: no complete calendar"))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["p_conservative"] == 1 for row in rows))
        wave, total = adjust([{"p_conservative": 1.}]*78, rows)
        np.testing.assert_array_equal(wave, np.ones(6))
        np.testing.assert_array_equal(total, np.ones(6))
        with self.assertRaises(ValueError):
            adjust([{"p_conservative": 1.}]*78, rows[:-1])

    def test_sensitivity_accepts_negative_scores_and_excludes_only_flagged_labels(self):
        result = sensitivity(np.array([-6., -100., -5.]), np.array([-5., -1., -4.]), np.array([False, True, False]))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["delta"], -1)
        self.assertNotIn("improvement_pct", result)
        with self.assertRaises(ValueError):
            sensitivity(np.array([-1.]), np.array([-2.]), np.array([True]))

    def _assert_failed_publication(self, fault):
        # This exercises only publication using synthetic precomputed values.
        # Source readers, feature construction, fits, and evaluation are mocked.
        scored = {"rows": [{"study": "macro_overnight", "horizon": 1, "candidate": candidate,
                            "control": control, "p_conservative": .00001, "phases": []}
                           for candidate, control in macro_search.CONTRASTS],
                  "leads": [{"candidate": "cpi", "horizon": 1}]}
        prior = [{"study": "synthetic", "candidate": "old", "horizon": 1,
                  "control": "baseline", "p_conservative": 1.}]*78
        original_ledger = macro_search.ledger
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            report, output, protocol = root/"report", root/"output", root/"protocol.yaml"
            protocol.write_text(json.dumps({"sources": {"vxn": "vxn.csv", "vix": "vix.csv"}, "index": {}}))

            def fake_report(metrics):
                (report/"results.md").write_text("Synthetic scored lead, possibly only partially written.")
                if fault == "report":
                    raise OSError("Injected report publication failure")

            def injected_ledger(rows):
                if rows and rows[0]["event"] == "evaluated" and fault == "evaluated_ledger":
                    original_ledger(rows[:1])
                    raise OSError("Injected evaluated-ledger publication failure")
                original_ledger(rows)

            replacements = {
                "ROOT": root, "REPORT": report, "OUT": output, "PROTOCOL": protocol,
                "validate": lambda p: None, "load_plan_records": lambda p: None,
                "load_inputs": lambda p: (None, None, None, {}), "inherited": lambda p: prior,
                "evaluate": lambda *args: scored, "report": fake_report, "ledger": injected_ledger,
                "version": lambda name: "synthetic",
            }
            for name, value in replacements.items():
                stack.enter_context(patch.object(macro_search, name, value))
            stack.enter_context(patch.object(macro_search.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")))
            stack.enter_context(patch.object(macro_search.inference, "digest", return_value="0"*64))
            stack.enter_context(patch.object(macro_search.macro_overnight, "build_features", return_value=(pd.DataFrame(), None)))
            stack.enter_context(patch.object(macro_search.macro_overnight, "forecast_panel", return_value=(pd.DataFrame({"synthetic": [1]}), [{}])))
            stack.enter_context(patch.object(pd.DataFrame, "to_parquet", return_value=None))
            with self.assertRaisesRegex(OSError, "Injected"):
                macro_search.run()
            failed = json.loads((report/"metrics.json").read_text())
            self.assertEqual(len(failed["rows"]), 6)
            self.assertTrue(all(row["p_conservative"] == 1 for row in failed["rows"]))
            self.assertTrue(all(row["status"] == "INVALID_RUN" for row in failed["rows"]))
            self.assertEqual(failed["leads"], [])
            self.assertTrue(failed["whole_wave_aborted"])
            self.assertEqual(json.loads((report/"failure.json").read_text())["rows"], failed["rows"])
            archived = json.loads((report/"unpublished_scored_metrics.json").read_text())
            self.assertEqual(archived["status"], "UNPUBLISHED_DIAGNOSTIC_ONLY")
            self.assertEqual(archived["scored_metrics"], scored)
            self.assertIn("UNEVALUABLE", (report/"results.md").read_text())
            self.assertNotIn("Synthetic scored lead", (report/"results.md").read_text())
            events = [json.loads(line) for line in (report/"trial_ledger.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["event"] == "unevaluable" and row["p_conservative"] == 1 for row in events[-6:]))

    def test_report_failure_replaces_promotable_scores_with_six_unevaluable_rows(self):
        self._assert_failed_publication("report")

    def test_partial_evaluated_ledger_failure_replaces_promotable_scores_with_six_unevaluable_rows(self):
        self._assert_failed_publication("evaluated_ledger")


if __name__ == "__main__":
    unittest.main()
