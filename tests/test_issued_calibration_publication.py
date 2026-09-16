"""Whole-family failure records, registration ordering and immutable sources."""

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from src import issued_calibration_search as s


class PublicationTests(unittest.TestCase):
    def fixture(self, root, stack):
        p = root / "issued_calibration.yaml"
        p.write_text(yaml.safe_dump(s.CONTRACT))
        report = root / "report"
        report.mkdir()
        for k, v in {
            "ROOT": root,
            "PROTOCOL": p,
            "REPORT": report,
            "OUT": root / "data",
        }.items():
            stack.enter_context(patch.object(s, k, v))
        stack.enter_context(patch.object(s, "validate_freeze", return_value={}))
        stack.enter_context(patch.object(s, "input_paths", return_value=set()))
        stack.enter_context(
            patch.object(
                s.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            )
        )
        prior = [
            {"candidate": str(i), "control": "old", "horizon": 1, "p_conservative": 1.0}
            for i in range(129)
        ]
        stack.enter_context(patch.object(s, "inherited", return_value=prior))
        return report

    def test_source_failure_keeps_all_registration_and_p1(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            report = self.fixture(root, stack)
            with (
                patch.object(s, "validate_upstream", side_effect=ValueError("source failed")),
                patch.object(s.sm, "forecast_panel") as build,
            ):
                with self.assertRaisesRegex(ValueError, "source failed"):
                    s.run()
                build.assert_not_called()
            metrics = json.loads((report / "metrics.json").read_bytes())
            self.assertEqual(metrics, json.loads((report / "failure.json").read_bytes()))
            self.assertEqual(len(metrics["rows"]), 2)
            self.assertEqual(metrics["leads"], [])
            events = [
                json.loads(x)["event"]
                for x in (report / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events.count("inherited"), 129)
            self.assertEqual(events.count("registered"), 2)
            self.assertEqual(events.count("unevaluable"), 2)
            self.assertEqual(len(events), 133)

    def test_manifest_admission_failure_retains_both_and_blocks_rerun(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            report = self.fixture(root, stack)
            with (
                patch.object(s, "input_paths", side_effect=ValueError("closure failed")),
                self.assertRaisesRegex(ValueError, "closure failed"),
            ):
                s.run()
            records = [
                json.loads(x) for x in (report / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [x["event"] for x in records], ["registered"] * 2 + ["unevaluable"] * 2
            )
            failure = json.loads((report / "failure.json").read_bytes())
            self.assertFalse(failure["inherited_rows_available"])
            self.assertTrue(all(x["p_conservative"] == 1 for x in failure["rows"]))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                s.run()

    def test_existing_registration_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            report = self.fixture(root, stack)
            (report / "manifest.json").write_text("prior")
            with (
                patch.object(s, "validate_upstream") as admission,
                self.assertRaisesRegex(ValueError, "overwrite"),
            ):
                s.run()
            admission.assert_not_called()
            self.assertEqual((report / "manifest.json").read_text(), "prior")

    def test_existing_private_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            self.fixture(root, stack)
            (root / "data").mkdir()
            (root / "data/output").write_text("prior")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                s.run()
            self.assertEqual((root / "data/output").read_text(), "prior")

    def test_pretests_fail_before_registration(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            report = self.fixture(root, stack)
            with (
                patch.object(
                    s.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=1, stdout="failure", stderr=""),
                ),
                self.assertRaisesRegex(RuntimeError, "Prewritten"),
            ):
                s.run()
            self.assertFalse((report / "manifest.json").exists())
            self.assertFalse((report / "trial_ledger.jsonl").exists())

    def test_protocol_mutation_during_checks_prevents_registration(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            report = self.fixture(root, stack)
            with (
                patch.object(
                    s, "validate_freeze", side_effect=[{}, ValueError("freeze changed")]
                ),
                self.assertRaisesRegex(ValueError, "freeze changed"),
            ):
                s.run()
            self.assertFalse((report / "manifest.json").exists())

    def test_inherited_failure_still_registers_both_and_declares_missing_history(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            report = self.fixture(root, stack)
            with (
                patch.object(s, "inherited", side_effect=ValueError("history unavailable")),
                self.assertRaisesRegex(ValueError, "history unavailable"),
            ):
                s.run()
            records = [
                json.loads(x) for x in (report / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [x["event"] for x in records], ["registered"] * 2 + ["unevaluable"] * 2
            )
            failure = json.loads((report / "failure.json").read_bytes())
            self.assertFalse(failure["inherited_rows_available"])
            self.assertEqual(failure["inherited_rows_reconstructed"], 0)

    def test_partial_registration_preserved_and_terminal_family_repaired(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            report = self.fixture(root, stack)

            def partial(rows):
                (report / "trial_ledger.jsonl").write_text(json.dumps(rows[0]) + "\n{partial")
                raise OSError("recoverable partial write")

            with (
                patch.object(s, "ledger", side_effect=partial),
                self.assertRaisesRegex(OSError, "partial write"),
            ):
                s.run()
            self.assertTrue(
                (report / "interrupted_trial_ledger.jsonl").read_text().endswith("{partial")
            )
            records = [
                json.loads(x) for x in (report / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [x["event"] for x in records], ["registered"] * 2 + ["unevaluable"] * 2
            )

    def test_atomic_serialization_failure_keeps_prior_ledger(self):
        with tempfile.TemporaryDirectory() as d, patch.object(s, "REPORT", Path(d)):
            path = Path(d) / "trial_ledger.jsonl"
            s.ledger([{"event": "registered"}])
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                s.ledger([{"p": float("nan")}])
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(Path(d).glob(".trial-ledger-*")))

    def test_upstream_failure_created_during_scoring_blocks_publication(self):
        import pandas as pd

        from tests.test_plot_issued_calibration import fixture as metric_fixture

        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            report = self.fixture(root, stack)
            frame = pd.DataFrame({"origin": pd.to_datetime(["2016-01-04"])})
            loaded = {
                "features": frame,
                "targets": frame,
                "forecasts": frame,
                "fits": [],
                "states": frame,
                "protocol": {"index": {}},
            }
            metric, _ = metric_fixture()
            metric["leads"] = []
            for row in metric["rows"]:
                row["verdict"] = "DOES_NOT_QUALIFY"

            def scoring(*args, **kwargs):
                marker = root / "reports/range_alert/failure.json"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}")
                return metric

            with (
                patch.object(s, "validate_upstream", return_value=({}, loaded)),
                patch.object(s.sm, "forecast_panel", return_value=(frame, frame, {})),
                patch.object(s, "evaluate", side_effect=scoring),
                self.assertRaisesRegex(ValueError, "upstream failure"),
            ):
                s.run()
            failure = json.loads((report / "metrics.json").read_bytes())
            self.assertEqual(failure["status"], "UNEVALUABLE")
            self.assertEqual(failure["leads"], [])
            self.assertTrue((report / "unpublished_scored_metrics.json").exists())

    def test_report_and_partial_evaluated_failures_preserve_diagnostics(self):
        import pandas as pd

        from tests.test_plot_issued_calibration import fixture as metric_fixture

        for mode in ["report", "evaluated"]:
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as d,
                ExitStack() as stack,
            ):
                root = Path(d)
                report = self.fixture(root, stack)
                frame = pd.DataFrame({"origin": pd.to_datetime(["2016-01-04"])})
                metric, _ = metric_fixture()
                metric["leads"] = []
                for row in metric["rows"]:
                    row["verdict"] = "DOES_NOT_QUALIFY"
                loaded = {
                    "features": frame,
                    "targets": frame,
                    "forecasts": frame,
                    "fits": [],
                    "states": frame,
                    "protocol": {"index": {}},
                }
                stack.enter_context(
                    patch.object(s, "validate_upstream", return_value=({}, loaded))
                )
                stack.enter_context(
                    patch.object(s.sm, "forecast_panel", return_value=(frame, frame, {}))
                )
                stack.enter_context(patch.object(s, "evaluate", return_value=metric))
                if mode == "report":
                    stack.enter_context(
                        patch.object(s, "report", side_effect=OSError("report failure"))
                    )
                else:
                    original = s.ledger

                    def partial(rows, report=report, original=original):
                        if rows[0]["event"] == "evaluated":
                            with (report / "trial_ledger.jsonl").open("a") as stream:
                                stream.write(json.dumps(rows[0]) + "\n{partial")
                            raise OSError("evaluated failure")
                        original(rows)

                    stack.enter_context(patch.object(s, "ledger", side_effect=partial))
                with self.assertRaisesRegex(OSError, mode + " failure"):
                    s.run()
                failure = json.loads((report / "failure.json").read_bytes())
                self.assertEqual(failure, json.loads((report / "metrics.json").read_bytes()))
                self.assertTrue(failure["inherited_rows_available"])
                self.assertEqual(failure["inherited_rows_reconstructed"], 129)
                records = [
                    json.loads(x)
                    for x in (report / "trial_ledger.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(records), 133)
                self.assertEqual([x["event"] for x in records[-2:]], ["unevaluable"] * 2)
                self.assertTrue((report / "unpublished_scored_metrics.json").exists())
                self.assertTrue((report / "interrupted_trial_ledger.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
