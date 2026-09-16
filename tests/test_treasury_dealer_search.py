"""Prewritten one-shot registration, preservation and failure accounting tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.treasury_dealer_search import _full_test_count, execute_registered, verify_pins


class TreasuryDealerSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prior = [
            {
                "study": "generated",
                "candidate": "candidate",
                "control": f"control{i}",
                "horizon": 5,
                "score": "qlike",
                "source": "reports/generated/metrics.json",
                "source_sha256": "b" * 64,
                "source_row_index": i,
                "p_conservative": 1.0,
            }
            for i in range(144)
        ]

    @staticmethod
    def result():
        return {
            "status": "COMPLETED",
            "hypothesis_count": 2,
            "cumulative_hypothesis_count": 146,
            "leads": [],
            "rows": [
                {
                    "study": "treasury_dealer",
                    "candidate": "candidate",
                    "control": control,
                    "horizon": 5,
                    "score": "qlike",
                    "p_conservative": 1.0,
                }
                for control in ("matched", "market")
            ],
        }

    def test_full_family_is_durable_before_numerical_callback(self):
        def work():
            events = [
                json.loads(line)
                for line in (self.root / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(events), 146)
            self.assertEqual([row["event"] for row in events[:2]], ["registered"] * 2)
            self.assertEqual([row["control"] for row in events[:2]], ["matched", "market"])
            self.assertEqual(
                [{k: v for k, v in row.items() if k != "event"} for row in events[2:]],
                self.prior,
            )
            return self.result()

        result = execute_registered(self.root, "a" * 64, self.prior, work)
        self.assertEqual(result["status"], "COMPLETED")
        events = [
            json.loads(line)
            for line in (self.root / "trial_ledger.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(events), 148)
        self.assertEqual([row["event"] for row in events[-2:]], ["evaluated"] * 2)

    def test_failure_keeps_both_registered_hypotheses_at_one(self):
        def work():
            raise ValueError("INSUFFICIENT_DATA: generated unsupported scheduled month")

        result = execute_registered(self.root, "a" * 64, self.prior, work)
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertEqual(result["cumulative_hypothesis_count"], 146)
        self.assertEqual([row["p_conservative"] for row in result["rows"]], [1.0, 1.0])
        self.assertEqual(len(result["inherited_rows"]), 144)
        self.assertTrue((self.root / "failure.json").exists())
        self.assertEqual(len((self.root / "trial_ledger.jsonl").read_text().splitlines()), 148)

    def test_existing_attempt_is_never_restarted(self):
        (self.root / "trial_ledger.jsonl").write_text("existing\n")
        with self.assertRaises(ValueError):
            execute_registered(
                self.root,
                "a" * 64,
                self.prior,
                lambda: self.fail("Numerical work must not run"),
            )
        self.assertEqual((self.root / "trial_ledger.jsonl").read_text(), "existing\n")

    def test_missing_comparison_invalidates_whole_attempt(self):
        result = self.result()
        result["rows"].pop()
        observed = execute_registered(self.root, "a" * 64, self.prior, lambda: result)
        self.assertEqual([row["status"] for row in observed["rows"]], ["INVALID_RUN"] * 2)
        self.assertEqual(observed["leads"], [])

    def test_invalid_prior_never_registers(self):
        with self.assertRaises(ValueError):
            execute_registered(
                self.root, "a" * 64, self.prior[:-1], lambda: self.fail("Must not run")
            )
        self.assertFalse((self.root / "trial_ledger.jsonl").exists())

    def test_callback_cannot_rewrite_saved_inherited_family(self):
        def work():
            self.prior[0]["p_conservative"] = 0.0
            return self.result()

        result = execute_registered(self.root, "a" * 64, self.prior, work)
        self.assertEqual(result["inherited_rows"][0]["p_conservative"], 1.0)

    def test_late_pin_failure_invalidates_published_attempt_without_losing_it(self):
        checks = []

        def final_check():
            checks.append(1)
            if len(checks) == 2:
                raise ValueError("Frozen artifact changed after provisional publication")

        result = execute_registered(
            self.root, "a" * 64, self.prior, self.result, final_check=final_check
        )
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertEqual([r["p_conservative"] for r in result["rows"]], [1.0, 1.0])
        self.assertEqual(
            json.loads((self.root / "metrics.json").read_text())["status"], "UNEVALUABLE"
        )
        self.assertEqual(
            json.loads((self.root / "attempted_metrics.json").read_text())["status"],
            "COMPLETED",
        )
        events = [
            json.loads(s) for s in (self.root / "trial_ledger.jsonl").read_text().splitlines()
        ]
        self.assertEqual([r["event"] for r in events[-2:]], ["invalidated"] * 2)

    def test_publication_failure_retains_both_rows_at_one(self):
        from src import treasury_dealer_search as search

        real_events = search._events
        failed = []

        def events(path, rows, mode):
            if mode == "a" and not failed:
                failed.append(True)
                raise OSError("generated journal write failure")
            return real_events(path, rows, mode)

        with patch.object(search, "_events", side_effect=events):
            result = execute_registered(self.root, "a" * 64, self.prior, self.result)
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertEqual(result["inherited_rows"], self.prior)
        self.assertEqual(
            json.loads((self.root / "metrics.json").read_text())["status"], "UNEVALUABLE"
        )

    def test_every_outcome_carries_qualified_clock_limitation(self):
        result = execute_registered(self.root, "a" * 64, self.prior, self.result)
        self.assertEqual(result["source_clock_class"], "QUALIFIED_REPORTED_CLOCK_ONLY_FOR_Z52")
        self.assertIn("unknown", result["source_clock_limitation"])

    def test_freeze_with_empty_current_code_or_test_evidence_is_rejected(self):
        from src.treasury_dealer_protocol import EXECUTION, PROTOCOL_SHA256, load_protocol
        from src.treasury_dealer_search import _verify_freeze

        forged = {
            "status": "FROZEN_BEFORE_TREASURY_MARKET_COHORT_OR_FITS",
            "protocol_sha256": PROTOCOL_SHA256,
            "execution": EXECUTION,
            "code": {},
            "inputs": {},
            "preserved": {},
            "prefit": {},
        }
        with self.assertRaises(ValueError):
            _verify_freeze(self.root, load_protocol(), forged)

    def test_buffered_stdout_does_not_mask_failed_or_empty_full_suite(self):
        log = "test_generated ... ok\n\nRan 2700 tests in 1.234s\n\nOK\nbuffered stdout\n"
        self.assertEqual(_full_test_count(log, 0), 2700)
        for text, code in [
            (log, 1),
            (log + log, 0),
            (log.replace("OK", "FAILED"), 0),
            ("Ran 0 tests in 0.0s\n\nOK\n", 0),
        ]:
            with self.assertRaises(ValueError):
                _full_test_count(text, code)

    def test_every_pin_is_readonly_enforced(self):
        path = self.root / "evidence"
        path.write_bytes(b"original")
        pins = {"evidence": hashlib.sha256(b"original").hexdigest()}
        verify_pins(self.root, pins)
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            verify_pins(self.root, pins)
        self.assertEqual(path.read_bytes(), b"changed")

    def test_real_runner_registers_before_decode_and_stops_before_fit_on_support_failure(self):
        from src.treasury_dealer_protocol import EXECUTION
        from src.treasury_dealer_search import run

        report = self.root / EXECUTION["reports"]
        report.mkdir(parents=True)
        (report / "freeze_record.json").write_text("{}")
        dates = pd.bdate_range("2019-01-02", periods=8, name="date")
        features = pd.DataFrame({"auction_count": [0.0] * 8}, index=dates)
        targets = pd.DataFrame(
            {
                "y": [1.0] * 3 + [float("nan")] * 5,
                "target_end": list(dates[5:]) + [pd.NaT] * 5,
            },
            index=dates,
        )
        support = {"passed": False, "status": "INSUFFICIENT_DATA"}

        def decode(*args):
            registered = [
                json.loads(s) for s in (report / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(registered), 146)
            self.assertEqual(
                [r["event"] for r in registered[:2]], ["registered", "registered"]
            )
            return {}, {"events": []}

        with (
            patch("src.treasury_dealer_search._verify_freeze"),
            patch("src.treasury_dealer_search._source_pins", return_value=({}, self.prior)),
            patch("src.treasury_dealer_inputs.read_inputs", side_effect=decode) as reader,
            patch(
                "src.treasury_dealer_inputs.build_inputs",
                return_value={"features": features, "targets": targets, "event_audit": []},
            ),
            patch("src.treasury_dealer_support.audit_support", return_value=support),
            patch(
                "src.verify_treasury_dealer_forecasts.verify_inputs_and_support",
                return_value={"status": "VERIFIED", "support_passed": False},
            ) as verifier,
            patch("src.treasury_dealer_pipeline.build_panel") as fitter,
            patch("src.treasury_dealer_search.evaluate") as scorer,
        ):
            result = run(self.root)
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertEqual(result["cumulative_hypothesis_count"], 146)
        reader.assert_called_once()
        verifier.assert_called_once()
        fitter.assert_not_called()
        scorer.assert_not_called()
        self.assertEqual([r["p_conservative"] for r in result["rows"]], [1.0, 1.0])
        self.assertTrue((report / "input_support_verification.json").is_file())
        self.assertTrue((report / "terminal.json").is_file())


if __name__ == "__main__":
    unittest.main()
