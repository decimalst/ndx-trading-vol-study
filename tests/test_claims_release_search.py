"""Generated registration, preservation and whole-family failure contracts."""

import json
import tempfile
import unittest
from pathlib import Path

from src.claims_release_search import execute_registered, verify_pins


class ClaimsReleaseSearchTests(unittest.TestCase):
    def test_full_suite_summary_allows_buffered_stdout_after_success(self):
        from src.claims_release_search import _full_test_count

        text = "test_generated ... ok\n\nRan 2500 tests in 1.234s\n\nOK\nbuffered generated output\n"
        self.assertEqual(_full_test_count(text, 0), 2500)
        for payload, status in (
            (text, 1),
            ("Ran 0 tests in 0.0s\n\nOK\n", 0),
            (text.replace("OK", "FAILED (failures=1)"), 0),
            (text + text, 0),
        ):
            with self.assertRaises(ValueError):
                _full_test_count(payload, status)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prior = [
            {"study": "generated", "id": i, "p_conservative": 1.0} for i in range(140)
        ]

    def test_registration_is_durable_before_numerical_callback(self):
        def work():
            events = [
                json.loads(x)
                for x in (self.root / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual([x["event"] for x in events[:2]], ["registered"] * 2)
            self.assertEqual([x["control"] for x in events[:2]], ["matched", "market"])
            self.assertEqual(
                [{k: v for k, v in x.items() if k != "event"} for x in events[2:]],
                self.prior,
            )
            return {
                "status": "COMPLETED",
                "rows": [
                    {
                        "study": "claims_release",
                        "candidate": "candidate",
                        "control": c,
                        "horizon": 5,
                        "score": "qlike",
                        "p_conservative": 1.0,
                    }
                    for c in ("matched", "market")
                ],
                "hypothesis_count": 2,
                "cumulative_hypothesis_count": 142,
                "leads": [],
            }

        result = execute_registered(self.root, "a" * 64, self.prior, work)
        self.assertEqual(result["status"], "COMPLETED")
        events = [
            json.loads(x) for x in (self.root / "trial_ledger.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(events), 144)
        self.assertEqual([x["event"] for x in events[-2:]], ["evaluated"] * 2)

    def test_registered_error_retains_both_comparisons_at_one(self):
        def work():
            raise ValueError("INSUFFICIENT_DATA: generated missing support")

        result = execute_registered(self.root, "a" * 64, self.prior, work)
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertEqual(result["cumulative_hypothesis_count"], 142)
        self.assertEqual([x["p_conservative"] for x in result["rows"]], [1.0, 1.0])
        self.assertTrue((self.root / "failure.json").exists())
        self.assertEqual(len((self.root / "trial_ledger.jsonl").read_text().splitlines()), 144)

    def test_no_overwrite_or_repeated_attempt_callback(self):
        ledger = self.root / "trial_ledger.jsonl"
        ledger.write_text("existing attempt\n")
        with self.assertRaises(ValueError):
            execute_registered(
                self.root, "a" * 64, self.prior, lambda: self.fail("Must not run")
            )
        self.assertEqual(ledger.read_text(), "existing attempt\n")

    def test_incomplete_output_family_becomes_invalid_whole_wave(self):
        result = execute_registered(
            self.root,
            "a" * 64,
            self.prior,
            lambda: {"status": "COMPLETED", "rows": [], "hypothesis_count": 0},
        )
        self.assertEqual([x["status"] for x in result["rows"]], ["INVALID_RUN"] * 2)
        self.assertEqual(result["leads"], [])

    def test_invalid_prior_stops_before_registration_and_callback(self):
        with self.assertRaises(ValueError):
            execute_registered(
                self.root, "a" * 64, self.prior[:-1], lambda: self.fail("Must not run")
            )
        self.assertFalse((self.root / "trial_ledger.jsonl").exists())

    def test_every_pin_enforced_without_rewriting_evidence(self):
        import hashlib

        file = self.root / "evidence.txt"
        file.write_bytes(b"original")
        pins = {"evidence.txt": hashlib.sha256(b"original").hexdigest()}
        verify_pins(self.root, pins)
        file.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "evidence.txt"):
            verify_pins(self.root, pins)
        self.assertEqual(file.read_bytes(), b"changed")


if __name__ == "__main__":
    unittest.main()
