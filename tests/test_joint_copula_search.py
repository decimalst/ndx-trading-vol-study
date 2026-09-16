"""Prewritten registration, preservation and protocol-boundary contracts."""

import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class JointCopulaSearchTests(unittest.TestCase):
    def setUp(self):
        self.search = importlib.import_module("src.joint_copula_search")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prior = [
            {
                "study": "generated",
                "candidate": "old",
                "control": f"c{i}",
                "horizon": 1,
                "score": "negative_joint_log_density",
                "source": "reports/generated/metrics.json",
                "source_sha256": "b" * 64,
                "source_row_index": i,
                "p_conservative": 1.0,
            }
            for i in range(149)
        ]

    def result(self):
        return {
            "status": "COMPLETED",
            "hypothesis_count": 2,
            "cumulative_hypothesis_count": 151,
            "leads": [],
            "rows": [
                {
                    "study": "joint_copula",
                    "candidate": "t8_copula",
                    "control": c,
                    "horizon": 1,
                    "score": "negative_joint_log_density",
                    "p_conservative": 1.0,
                }
                for c in ("gaussian_copula", "independence")
            ],
        }

    def test_registration_and_inherited_identities_are_durable_before_work(self):
        def work():
            rows = [
                json.loads(s)
                for s in (self.root / "trial_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 151)
            self.assertEqual(
                [r["control"] for r in rows[:2]], ["gaussian_copula", "independence"]
            )
            self.assertEqual([r["event"] for r in rows[:2]], ["registered"] * 2)
            self.assertEqual(
                [{k: v for k, v in r.items() if k != "event"} for r in rows[2:]], self.prior
            )
            return self.result()

        result = self.search.execute_registered(self.root, "a" * 64, self.prior, work)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(len((self.root / "trial_ledger.jsonl").read_text().splitlines()), 153)
        self.assertEqual(result["inherited_rows"], self.prior)
        self.assertIn("not untouched", result["evidence_limitation"])

    def test_support_failure_preserves_both_as_unevaluable(self):
        def work():
            raise ValueError("INSUFFICIENT_DATA: generated support failure")

        result = self.search.execute_registered(self.root, "a" * 64, self.prior, work)
        self.assertEqual(result["status"], "UNEVALUABLE")
        self.assertEqual(result["cumulative_hypothesis_count"], 151)
        self.assertEqual([r["p_conservative"] for r in result["rows"]], [1.0] * 2)
        self.assertEqual(result["inherited_rows"], self.prior)
        self.assertEqual(result["leads"], [])
        self.assertTrue((self.root / "failure.json").exists())

    def test_existing_attempt_and_invalid_family_prevent_callback(self):
        for prior in (self.prior[:-1], self.prior + [self.prior[0]]):
            with self.assertRaises(ValueError):
                self.search.execute_registered(
                    self.root, "a" * 64, prior, lambda: self.fail("work ran")
                )
        self.assertFalse((self.root / "trial_ledger.jsonl").exists())
        (self.root / "trial_ledger.jsonl").write_text("preserve\n")
        with self.assertRaises(ValueError):
            self.search.execute_registered(
                self.root, "a" * 64, self.prior, lambda: self.fail("work ran")
            )
        self.assertEqual((self.root / "trial_ledger.jsonl").read_text(), "preserve\n")

    def test_missing_or_reordered_control_invalidates_whole_attempt(self):
        for mutate in (lambda r: r["rows"].pop(), lambda r: r["rows"].reverse()):
            with tempfile.TemporaryDirectory() as directory:
                result = self.result()
                mutate(result)
                got = self.search.execute_registered(
                    directory, "a" * 64, self.prior, lambda result=result: result
                )
                self.assertEqual(got["status"], "UNEVALUABLE")
                self.assertEqual([r["p_conservative"] for r in got["rows"]], [1.0] * 2)

    def test_late_preservation_failure_retains_provisional_result(self):
        calls = []

        def check():
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("generated late pin failure")

        got = self.search.execute_registered(
            self.root, "a" * 64, self.prior, self.result, check
        )
        self.assertEqual(got["status"], "UNEVALUABLE")
        self.assertEqual(
            json.loads((self.root / "attempted_metrics.json").read_text())["status"],
            "COMPLETED",
        )
        self.assertEqual(
            json.loads((self.root / "metrics.json").read_text())["status"], "UNEVALUABLE"
        )
        events = [
            json.loads(s) for s in (self.root / "trial_ledger.jsonl").read_text().splitlines()
        ]
        self.assertEqual([r["event"] for r in events[-2:]], ["invalidated"] * 2)

    def test_publication_write_failure_and_mutable_caller_prior_preserve_family(self):
        original = copy.deepcopy(self.prior)
        actual_events = self.search._events
        failed = []

        def events(path, rows, mode):
            if mode == "a" and not failed:
                failed.append(True)
                raise OSError("generated journal failure")
            return actual_events(path, rows, mode)

        def work():
            self.prior[0]["p_conservative"] = 0.0
            return self.result()

        with patch.object(self.search, "_events", side_effect=events):
            got = self.search.execute_registered(self.root, "a" * 64, self.prior, work)
        self.assertEqual(got["status"], "UNEVALUABLE")
        self.assertEqual(got["inherited_rows"], original)

    def test_empty_current_inventory_or_missing_selected_receipts_cannot_freeze(self):
        from src.joint_copula_protocol import EXECUTION, PROTOCOL_SHA256, load_protocol

        frozen = {
            "status": "FROZEN_BEFORE_JOINT_COPULA_MARKET_COHORT_OR_FITS",
            "protocol_sha256": PROTOCOL_SHA256,
            "execution": EXECUTION,
            "code": {},
            "inputs": {},
            "prefit": {},
            "preserved": {},
        }
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.search._verify_freeze(self.root, load_protocol(), frozen)
        with self.assertRaisesRegex(ValueError, "receipts"):
            self.search._test_evidence(self.root, frozen)

    def test_changed_freeze_record_bytes_after_registration_invalidate_attempt(self):
        report, _ = self.search._paths(self.root)
        report.mkdir(parents=True)
        path = report / "freeze_record.json"
        path.write_text('{"generated": true}\n')
        signature = self.search._sha(path)
        initial = json.loads(path.read_text())
        # Assert the binding helper exists before the registration wrapper can
        # turn a missing implementation into an apparently expected failure.
        check_snapshot = self.search._check_frozen_snapshot
        with patch.object(self.search, "_verify_freeze") as contents_check:
            check_snapshot(self.root, {}, initial, signature)
            contents_check.assert_called_once_with(self.root, {}, initial)

            def work():
                path.write_text('{"generated": false}\n')
                return self.result()

            got = self.search.execute_registered(
                report,
                "a" * 64,
                self.prior,
                work,
                final_check=lambda: check_snapshot(self.root, {}, initial, signature),
            )
        self.assertEqual(got["status"], "UNEVALUABLE")
        self.assertEqual([r["p_conservative"] for r in got["rows"]], [1.0] * 2)
        self.assertEqual(got["inherited_rows"], self.prior)
        self.assertEqual(json.loads(path.read_text()), {"generated": False})

    def test_changed_output_during_verification_invalidates_saved_result(self):
        report, out = self.search._paths(self.root)
        out.mkdir(parents=True)
        for name in (
            "features.parquet",
            "targets.parquet",
            "applications.parquet",
            "panel.parquet",
            "coverage.parquet",
            "schedules.parquet",
            "fits.json",
        ):
            (out / name).write_text("generated bytes, not historical arrays\n")
        pins = {str(p.relative_to(self.root)): self.search._sha(p) for p in out.iterdir()}
        check_outputs = self.search._check_output_snapshot
        check_outputs(self.root, pins)

        def work():
            # Models and coefficients have already been saved; this represents
            # drift while an independent verifier is working in memory.
            (out / "panel.parquet").write_text("changed during verifier callback\n")
            return self.result()

        got = self.search.execute_registered(
            report,
            "a" * 64,
            self.prior,
            work,
            final_check=lambda: check_outputs(self.root, pins),
        )
        self.assertEqual(got["status"], "UNEVALUABLE")
        self.assertEqual([r["p_conservative"] for r in got["rows"]], [1.0] * 2)
        self.assertIn("changed during", (out / "panel.parquet").read_text())

    def test_metrics_changed_after_publication_cannot_keep_completed_status(self):
        events = self.search._events
        changed = []

        def mutate_metrics_after_journal(path, rows, mode):
            events(path, rows, mode)
            if mode == "a" and not changed:
                changed.append(True)
                metrics = self.root / "metrics.json"
                value = json.loads(metrics.read_text())
                value["rows"][0]["p_conservative"] = 0.0
                metrics.write_text(json.dumps(value))

        with patch.object(self.search, "_events", side_effect=mutate_metrics_after_journal):
            got = self.search.execute_registered(self.root, "a" * 64, self.prior, self.result)
        self.assertEqual(got["status"], "UNEVALUABLE")
        self.assertEqual([r["p_conservative"] for r in got["rows"]], [1.0] * 2)
        self.assertTrue((self.root / "attempted_metrics.json").exists())


class JointCopulaProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = importlib.import_module("src.joint_copula_protocol")

    def test_fixed_design_and_projection(self):
        spec = self.protocol.load_protocol()
        self.protocol.validate(spec)
        self.assertEqual(spec["comparisons"]["cumulative"], 151)
        self.assertEqual(spec["comparisons"]["wave_alpha"], 0.05 / (27 * 28))
        self.assertEqual(spec["forecast"]["horizon"], 1)
        config = self.protocol.pipeline_config(spec)
        self.assertEqual(
            set(config),
            {
                "origin_start",
                "origin_end",
                "source_end",
                "development",
                "evaluation",
                "minimum_train",
            },
        )
        self.assertEqual(config["origin_end"], "2025-10-17")
        self.assertEqual(config["source_end"], "2025-10-20")
        self.assertEqual(config["minimum_train"], 1000)

    def test_altered_rule_or_protocol_bytes_rejected(self):
        spec = self.protocol.load_protocol()
        spec["forecast"]["minimum_train"] = 999
        with self.assertRaises(ValueError):
            self.protocol.validate(spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "joint_copula.yaml").write_text("study_id: changed\n")
            with patch.object(self.protocol, "ROOT", path), self.assertRaises(ValueError):
                self.protocol.load_protocol()

    def test_exact_original_six_sources_and_prior_are_bound(self):
        execution = self.protocol.EXECUTION
        expected = {
            "data/research_paths/spx_daily.parquet",
            "data/raw/daily_ohlc.parquet",
            *[
                "data/free_sources/raw/cboe/" + s + "_History.csv"
                for s in ("VIX", "VIX9D", "VVIX", "VXN")
            ],
        }
        self.assertEqual(set(execution["market_pins"]), expected)
        self.assertEqual(
            execution["prior_sha256"],
            "3ad635e31bf81f224ccf690bb0efd109b8f0b23dea6fc7bfb30d6073b7ffde58",
        )
        self.assertIn("not untouched", self.protocol.EVIDENCE_LIMITATION)
        self.assertIn("QQQ ETF", self.protocol.EVIDENCE_LIMITATION)


if __name__ == "__main__":
    unittest.main()
