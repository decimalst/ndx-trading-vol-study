"""Synthetic contracts written before the prospective ledger implementation."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from src.prospective_benchmark import ProspectiveLedger


class LedgerContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = datetime(2030, 1, 2, 12, tzinfo=UTC)
        self.ledger = ProspectiveLedger(self.tmp.name, clock=lambda: self.now)
        self.spec = {
            "kind": "log_return", "instrument": "SYNTH:MNQH30", "unit": "log_return",
            "session": "synthetic_rth", "definition_id": "open_close_v1",
            "provider": "synthetic", "dataset": "trades", "feed": "synthetic",
            "adjustment": "raw",
        }
        self.protocol = {
            "benchmark_group": "synthetic_pair", "target_spec": self.spec,
            "training_cutoff": "2030-01-01T20:00:00Z", "input_schema": "synthetic_v1",
            "code_sha256": "a" * 64, "evaluation_plan_sha256": "b" * 64,
            "required_input_datasets": ["trades"],
        }
        self.baseline = self.ledger.freeze_model("baseline", b"baseline", self.protocol)
        self.candidate = self.ledger.freeze_model("candidate", b"candidate", self.protocol)
        self.target = {**self.spec, "start": "2030-01-02T14:30:00Z",
                       "end": "2030-01-02T21:00:00Z", "session_id": "2030-01-02"}
        self.ledger.schedule_target(self.target, [self.baseline, self.candidate])
        self.source = self.source_event(b"initial", "2030-01-01T20:00:00Z")
        self.prediction = {"family": "normal", "location": 0.0, "scale": 2.0}

    def source_event(self, raw, event_end):
        return self.ledger.ingest_source("synthetic/trades", raw, {
            "provider": "synthetic", "dataset": "trades", "instrument": "SYNTH:MNQH30",
            "feed": "synthetic", "adjustment": "raw", "event_end": event_end,
            "released_at": event_end, "endpoint": "synthetic://fixture",
        })

    def issue(self, freeze=None, prediction=None):
        return self.ledger.issue_forecast(freeze or self.baseline, self.target,
                                          prediction or self.prediction, [self.source])

    def observe(self, value=1.0, raw=b"outcome"):
        self.now = datetime(2030, 1, 2, 22, tzinfo=UTC)
        source = self.source_event(raw, self.target["end"])
        return self.ledger.record_observation(self.target, value, [source])

    def test_receipts_deduplicate_and_preserve_revised_bytes(self):
        count = len(self.ledger.verify()["events"])
        same = self.source_event(b"initial", "2030-01-01T20:00:00Z")
        self.assertEqual(same, self.source)
        self.assertEqual(len(self.ledger.verify()["events"]), count)
        revision = self.source_event(b"revision", "2030-01-01T20:00:00Z")
        self.assertNotEqual(revision, same)
        sources = [e for e in self.ledger.verify()["events"] if e["kind"] == "source"]
        self.assertEqual(sources[-1]["payload"]["supersedes"], same)
        self.assertEqual((Path(self.tmp.name) / "blobs" /
                          hashlib.sha256(b"initial").hexdigest()).read_bytes(), b"initial")

    def test_source_key_cannot_silently_switch_feed(self):
        source = next(r for r in self.ledger.verify()["events"]
                      if r["event_hash"] == self.source)
        metadata = {**source["payload"]["metadata"], "feed": "different_venue"}
        with self.assertRaises(ValueError):
            self.ledger.ingest_source("synthetic/trades", b"different", metadata)

    def test_incomplete_append_is_rejected_before_next_write(self):
        path = Path(self.tmp.name) / "events.jsonl"
        with path.open("ab") as output:
            output.write(b'{"unfinished":')
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            self.issue()
        self.assertEqual(before, path.read_bytes())

    def test_normal_score_and_idempotent_maturation(self):
        forecast = self.issue()
        self.assertEqual(self.ledger.mature_scores(), [])
        self.observe()
        scores = self.ledger.mature_scores()
        self.assertEqual(len(scores), 1)
        p = scores[0]["payload"]
        self.assertEqual(p["forecast_id"], forecast)
        self.assertAlmostEqual(p["loss"], .5 * math.log(2 * math.pi) + math.log(2) + .125)
        self.assertAlmostEqual(p["pit"], .5 * (1 + math.erf(.5 / math.sqrt(2))))
        count = len(self.ledger.verify()["events"])
        self.assertEqual(scores, self.ledger.mature_scores())
        self.assertEqual(len(self.ledger.verify()["events"]), count)

    def test_student_t_scale_includes_density_normalization(self):
        self.issue(prediction={"family": "student_t", "location": 1., "scale": 3., "df": 8.})
        self.observe(value=1.)
        p = self.ledger.mature_scores()[0]["payload"]
        expected = math.log(3) + .5 * math.log(8 * math.pi) + math.lgamma(4) - math.lgamma(4.5)
        self.assertAlmostEqual(p["loss"], expected)
        self.assertAlmostEqual(p["pit"], .5)

    def test_exact_start_and_backdated_forecasts_never_qualify(self):
        self.now = datetime(2030, 1, 2, 14, 30, tzinfo=UTC)
        f = self.issue()
        self.observe()
        self.assertEqual(self.ledger.mature_scores(), [])
        row = next(r for r in self.ledger.coverage() if r["freeze_id"] == self.baseline)
        self.assertEqual(row["status"], "late_forecast")
        event = next(e for e in self.ledger.verify()["events"] if e["event_hash"] == f)
        self.assertFalse(event["payload"]["timing_eligible"])
        with self.assertRaises(TypeError):
            self.ledger.issue_forecast(self.candidate, self.target, self.prediction,
                                       [self.source], received_at="2030-01-01T00:00:00Z")

    def test_forecast_payload_is_immutable(self):
        first = self.issue()
        self.assertEqual(first, self.issue())
        with self.assertRaises(ValueError):
            self.issue(prediction={"family": "normal", "location": 9., "scale": 2.})

    def test_unknown_missing_inputs_and_unscheduled_target_fail(self):
        with self.assertRaises(ValueError):
            self.ledger.issue_forecast(self.baseline, self.target, self.prediction, ["c" * 64])
        with self.assertRaises(ValueError):
            self.ledger.issue_forecast(self.baseline, self.target, self.prediction, [])
        target = {**self.target, "session_id": "undeclared"}
        with self.assertRaises(ValueError):
            self.ledger.issue_forecast(self.baseline, target, self.prediction, [self.source])

    def test_model_and_target_scope_must_match(self):
        p = copy.deepcopy(self.protocol)
        p["target_spec"]["instrument"] = "SPX"
        different = self.ledger.freeze_model("other", b"other", p)
        with self.assertRaises(ValueError):
            self.ledger.schedule_target(self.target, [self.baseline, different])
        with self.assertRaises(ValueError):
            self.ledger.paired_benchmark(self.baseline, different)

    def test_first_observation_is_frozen_primary_and_revisions_retained(self):
        self.issue()
        first = self.observe(value=1., raw=b"v1")
        first_scores = self.ledger.mature_scores()
        second = self.observe(value=5., raw=b"v2")
        self.assertNotEqual(first, second)
        self.assertEqual(first_scores, self.ledger.mature_scores())
        observations = [e for e in self.ledger.verify()["events"] if e["kind"] == "observation"]
        self.assertEqual(observations[-1]["payload"]["supersedes"], first)

    def test_outcome_before_end_or_wrong_instrument_is_rejected(self):
        with self.assertRaises(ValueError):
            self.ledger.record_observation(self.target, 1., [self.source])
        self.now = datetime(2030, 1, 2, 22, tzinfo=UTC)
        with self.assertRaises(ValueError):
            self.ledger.record_observation(self.target, 1., [self.source])

    def test_paired_coverage_keeps_missing_candidate(self):
        self.issue()
        self.observe()
        self.ledger.mature_scores()
        report = self.ledger.paired_benchmark(self.baseline, self.candidate)
        self.assertEqual(report["n"], 0)
        self.assertEqual(report["expected_targets"], 1)
        self.assertEqual(report["coverage"]["candidate"], {"missing_forecast": 1})

    def test_paired_scores_use_same_first_vintage(self):
        self.issue()
        self.issue(self.candidate, {"family": "normal", "location": 1., "scale": 2.})
        self.observe()
        self.ledger.mature_scores()
        report = self.ledger.paired_benchmark(self.baseline, self.candidate)
        self.assertEqual(report["status"], "DESCRIPTIVE_ONLY")
        self.assertEqual(report["n"], 1)
        self.assertAlmostEqual(report["mean_loss_difference"], -.125)

    def test_asof_cannot_read_future_receipts(self):
        self.issue()
        self.observe()
        self.ledger.mature_scores()
        self.assertEqual(self.ledger.mature_scores(as_of="2030-01-02T20:00:00Z"), [])
        with self.assertRaises(ValueError):
            self.ledger.mature_scores(as_of="2031-01-01T00:00:00Z")

    def test_timestamp_validation_and_clock_regression(self):
        with self.assertRaises(ValueError):
            self.source_event(b"future", "2031-01-01T00:00:00Z")
        self.now = datetime(2030, 1, 2, 11, tzinfo=UTC)
        with self.assertRaises(ValueError):
            self.issue()
        self.now = datetime(2030, 1, 2, 15, tzinfo=UTC)
        with self.assertRaises(ValueError):
            self.ledger.schedule_target(self.target, [self.baseline, self.candidate])

    def test_nan_zero_scale_and_naive_target_fail(self):
        for scale in (0., -1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.issue(prediction={"family": "normal", "location": 0., "scale": scale})
        target = {**self.target, "start": "2030-01-02T14:30:00"}
        with self.assertRaises(ValueError):
            self.ledger.schedule_target(target, [self.baseline, self.candidate])

    def test_hash_chain_and_raw_blob_corruption_fail_closed(self):
        path = Path(self.tmp.name) / "events.jsonl"
        original = path.read_bytes()
        rows = original.decode().splitlines()
        event = json.loads(rows[0])
        event["received_at"] = "2029-01-01T00:00:00Z"
        rows[0] = json.dumps(event)
        path.write_text("\n".join(rows) + "\n")
        with self.assertRaises(ValueError):
            self.ledger.verify()
        path.write_bytes(original)
        blob = Path(self.tmp.name) / "blobs" / hashlib.sha256(b"initial").hexdigest()
        blob.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self.ledger.verify()

    def test_checkpoint_detects_valid_suffix_deletion(self):
        head = self.ledger.verify()["head"]
        path = Path(self.tmp.name) / "events.jsonl"
        rows = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(rows[:-1]))
        with self.assertRaises(ValueError):
            self.ledger.verify(expected_head=head)

    def test_variance_qlike_has_correct_units_and_zero_target_policy(self):
        spec = {**self.spec, "kind": "variance", "unit": "log_return_squared",
                "definition_id": "synthetic_gk_v1"}
        p = {**self.protocol, "target_spec": spec}
        freeze = self.ledger.freeze_model("variance", b"var", p)
        target = {**self.target, **spec}
        self.ledger.schedule_target(target, [freeze])
        self.ledger.issue_forecast(freeze, target, {"family": "variance_mean", "mean": 2.},
                                   [self.source])
        self.now = datetime(2030, 1, 2, 22, tzinfo=UTC)
        source = self.source_event(b"variance", target["end"])
        with self.assertRaises(ValueError):
            self.ledger.record_observation(target, 0., [source])
        self.ledger.record_observation(target, 4., [source])
        result = self.ledger.mature_scores()[0]["payload"]
        self.assertAlmostEqual(result["loss"], 2 - math.log(2) - 1)
        self.assertIsNone(result["pit"])


if __name__ == "__main__":
    unittest.main()
