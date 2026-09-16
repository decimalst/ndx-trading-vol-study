"""Prewritten one-shot registration, preservation and failure accounting tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.commodity_implied_search import _full_test_count, execute_registered, verify_pins


class CommodityImpliedSearchTests(unittest.TestCase):
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
            for i in range(142)
        ]

    @staticmethod
    def result():
        return {
            "status": "COMPLETED",
            "hypothesis_count": 2,
            "cumulative_hypothesis_count": 144,
            "leads": [],
            "rows": [
                {
                    "study": "commodity_implied",
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
            self.assertEqual(len(events), 144)
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
        self.assertEqual(len(events), 146)
        self.assertEqual([row["event"] for row in events[-2:]], ["evaluated"] * 2)

    def test_failure_keeps_both_registered_hypotheses_at_one(self):
        def work():
            raise ValueError("INSUFFICIENT_DATA: generated unsupported scheduled month")

        result = execute_registered(self.root, "a" * 64, self.prior, work)
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertEqual(result["cumulative_hypothesis_count"], 144)
        self.assertEqual([row["p_conservative"] for row in result["rows"]], [1.0, 1.0])
        self.assertEqual(len(result["inherited_rows"]), 142)
        self.assertTrue((self.root / "failure.json").exists())
        self.assertEqual(len((self.root / "trial_ledger.jsonl").read_text().splitlines()), 146)

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


if __name__ == "__main__":
    unittest.main()
