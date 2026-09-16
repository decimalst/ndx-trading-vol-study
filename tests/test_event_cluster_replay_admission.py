"""Synthetic failed-output snapshot contracts, written before implementation."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import event_cluster_admission as old
from src import event_cluster_replay_admission as source
from tests.test_event_cluster_admission import Fixture as OriginalFixture

REPORT = "reports/event_cluster/"
DATA = "data/event_cluster/"
ERROR = (
    "Independent verification failed: AssertionError: every saved application state "
    "differs from independent reconstruction"
)
CONTROLS = ("baseline", "nuisance", "recent_frequency")
EVENT_COUNTS = {"registered": 3, "inherited": 131, "evaluated": 3, "verification_failed": 3}


class Fixture(OriginalFixture):
    def __init__(self, root):
        super().__init__(root)
        self.original_expected = dict(self.expected)
        original, loaded = old.admit_upstream(root, self.original_expected)
        self.put("event_cluster.yaml", {"upstream": {"anchors": self.original_expected}})
        self.put("src/event_cluster_pipeline.py", b"# frozen synthetic producer\n")
        self.put("src/verify_event_cluster.py", b"# frozen synthetic verifier\n")
        self.put("private/wave20/outside.bin", b"preserved beyond output directory")
        self.put(
            "reports/wave20_prior/manifest.json",
            {"existing_artifacts_sha256": self.pins("private/wave20/outside.bin")},
        )
        for name in (
            "DESIGN.md",
            "SUMMARY.md",
            "FAILURE_REVIEW.md",
            "full_repository_tests.txt",
            "pre_run_checks.txt",
        ):
            self.put(REPORT + name, ("synthetic " + name).encode())
        code = {
            **self.documents["reports/issued_calibration/manifest.json"]["code"],
            **self.pins("src/event_cluster_pipeline.py", "src/verify_event_cluster.py"),
        }
        self.put(
            REPORT + "freeze_record.json",
            {
                "protocol_sha256": self.hash("event_cluster.yaml"),
                "code": code,
                "prefit_design": self.pins(
                    REPORT + "DESIGN.md", REPORT + "pre_run_checks.txt"
                ),
                "checks": {
                    "full_repository_tests": 1,
                    "full_log_sha256": self.hash(REPORT + "full_repository_tests.txt"),
                },
            },
        )
        self.put(
            REPORT + "manifest.json",
            {
                "protocol_sha256": self.hash("event_cluster.yaml"),
                "code": code,
                "inputs": {**original["files"], **self.pins(REPORT + "freeze_record.json")},
                "preserved": self.pins("reports/wave20_prior/manifest.json"),
            },
        )
        self.frame(DATA + "forecasts.parquet", loaded["forecasts"])
        self.frame(DATA + "states.parquet", loaded["states"])
        self.frame(DATA + "memory.parquet", loaded["features"])
        self.put(DATA + "fits.json", [{"synthetic_unverified_fit": True}])
        self.put(DATA + "support_audit.json", {"synthetic_unverified_support": True})
        self.put(DATA + "upstream_admission.json", original)
        self.put(
            REPORT + "verification.json",
            {
                "status": "FAILED",
                "error": ERROR,
                "protocol_sha256": self.hash("event_cluster.yaml"),
            },
        )
        self.rows = [
            {
                "study": "event_cluster",
                "candidate": "cluster",
                "control": control,
                "score": "brier",
                "horizon": 1,
                "phases": [],
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "status": "INVALID_RUN",
                "verdict": "UNEVALUABLE",
                "error": ERROR,
            }
            for control in CONTROLS
        ]
        failure = {
            "status": "UNEVALUABLE",
            "whole_wave_aborted": True,
            "leads": [],
            "hypothesis_count": 3,
            "cumulative_hypothesis_count": 134,
            "protocol_sha256": self.hash("event_cluster.yaml"),
            "rows": self.rows,
        }
        self.put(REPORT + "metrics.json", failure)
        self.put(REPORT + "failure.json", failure)
        registrations = [
            {
                "event": "registered",
                "study": "event_cluster",
                "candidate": "cluster",
                "control": c,
                "horizon": 1,
                "score": "brier",
                "protocol_sha256": self.hash("event_cluster.yaml"),
            }
            for c in CONTROLS
        ]
        events = registrations + [{"event": "inherited", "opaque": i} for i in range(131)]
        events += [{"event": "evaluated", "opaque_unpublished_payload": i} for i in range(3)]
        events += [{"event": "verification_failed", **r} for r in self.rows]
        self.put(
            REPORT + "trial_ledger.jsonl",
            ("\n".join(json.dumps(r) for r in events) + "\n").encode(),
        )
        # Deliberately invalid JSON: preservation must not decode this private payload.
        self.put(
            REPORT + "unpublished_scored_metrics.json",
            b"opaque unpublished scored bytes, not for JSON decoding",
        )
        self.put(
            REPORT + "failure_review.json",
            {
                "status": "FAILURE_ACCOUNTING_REVIEWED",
                "research_status": "UNEVALUABLE",
                "numerical_verification_status": "FAILED",
                "not_a_successful_numerical_verification": True,
                "historical_producer_reruns": 0,
                "whole_verifier_reruns": 0,
                "additional_optimizer_runs": 0,
                "unpublished_candidate_score_values_inspected": False,
                "error": {
                    "canonical": ERROR,
                    "column": "feature_cutoff_date",
                    "saved_dtype": "datetime64[us]",
                    "expected_dtype": "datetime64[ms]",
                    "stage": "exact full application-state frame comparison",
                    "independent_new_stage_solves_completed": 0,
                    "independent_inference_reached": False,
                    "all_other_values_proven_correct": False,
                },
                "canonical_failure": {
                    "metrics_equal_failure": True,
                    "whole_wave_aborted": True,
                    "leads": [],
                    "new_hypotheses": 3,
                    "inherited_hypotheses": 131,
                    "cumulative_hypotheses": 134,
                    "rows": [
                        {
                            k: r[k]
                            for k in (
                                "candidate",
                                "control",
                                "p_conservative",
                                "p_holm_wave",
                                "p_holm_cumulative",
                                "phases",
                                "verdict",
                            )
                        }
                        for r in self.rows
                    ],
                },
                "ledger": {
                    "events": 140,
                    "event_counts": EVENT_COUNTS,
                    "order": [{"event": k, "count": n} for k, n in EVENT_COUNTS.items()],
                    "terminal_rows_equal_canonical": True,
                    "evaluated_score_values_inspected": False,
                },
                "output_metadata": {
                    "combined_rows": 4,
                    "new_generated_unverified_forecasts": 2,
                    "reused_control_forecasts": 2,
                    "producer_reported_monthly_schedules": 1,
                    "state_rows": 2,
                    "memory_rows": 3,
                    "new_fits_independently_verified": False,
                },
            },
        )
        self.put(
            REPORT + "publication_audit.json",
            {
                "status": "FAILED_RESEARCH_RECORD_PRESERVED_AND_AUDITED",
                "research_verification_status": "FAILED",
                "canonical_metrics_status": "UNEVALUABLE",
                "new_hypotheses": 3,
                "cumulative_hypotheses": 134,
                "ledger_events": 140,
                "ledger_event_counts": EVENT_COUNTS,
                "canonical_all_three_p_one": True,
                "current_wave_generated_scored_forecasts_not_independently_verified": 2,
                "current_wave_reused_control_forecasts": 2,
                "current_wave_combined_saved_forecasts": 4,
                "current_wave_producer_reported_monthly_schedules_not_independently_verified": 1,
                "current_wave_completed_independently_verified_forecasts": 0,
                "full_repository_tests": 1,
                "unpublished_scores_used_for_promotion": False,
                "post_failure_code_or_protocol_changes": False,
                "post_failure_empirical_retry": False,
            },
        )
        self.seal()

    def seal(self):
        """Rebind only temporary documentary links after a semantic mutation."""
        refs = {
            n: self.hash(REPORT + n)
            for n in (
                "manifest.json",
                "freeze_record.json",
                "verification.json",
                "metrics.json",
                "failure.json",
                "trial_ledger.jsonl",
                "SUMMARY.md",
                "FAILURE_REVIEW.md",
                "full_repository_tests.txt",
                "pre_run_checks.txt",
                "unpublished_scored_metrics.json",
            )
        }
        outputs = self.pins(*source.OUTPUT_PATHS)
        review = copy.deepcopy(self.documents[REPORT + "failure_review.json"])
        for key, name in (
            ("protocol_sha256", "event_cluster.yaml"),
            ("manifest_sha256", REPORT + "manifest.json"),
            ("freeze_sha256", REPORT + "freeze_record.json"),
        ):
            review[key] = self.hash(name)
        review["report_artifact_sha256"] = {REPORT + n: h for n, h in refs.items()}
        review["current_private_output_sha256"] = outputs
        review["failure_review_document_sha256"] = refs["FAILURE_REVIEW.md"]
        lines = (self.root / (REPORT + "trial_ledger.jsonl")).read_bytes().splitlines()
        review["ledger"]["inherited_record_sha256"] = [
            hashlib.sha256(x).hexdigest() for x in lines[3:134]
        ]
        review["ledger"]["opaque_evaluated_record_sha256"] = [
            hashlib.sha256(x).hexdigest() for x in lines[134:137]
        ]
        self.put(REPORT + "failure_review.json", review)
        pub = copy.deepcopy(self.documents[REPORT + "publication_audit.json"])
        pub.update(
            {
                "protocol_sha256": self.hash("event_cluster.yaml"),
                "manifest_sha256": refs["manifest.json"],
                "freeze_sha256": refs["freeze_record.json"],
                "verification_sha256": refs["verification.json"],
            }
        )
        pub["failed_output_hashes_not_numerically_verified"] = outputs
        pub["report_artifact_hashes"] = {
            **refs,
            "failure_review.json": self.hash(REPORT + "failure_review.json"),
        }
        pub["manifest_entries_checked"] = {
            k: len(self.documents[REPORT + "manifest.json"][k])
            for k in ("code", "inputs", "preserved")
        }
        self.put(REPORT + "publication_audit.json", pub)
        self.expected = self.pins(*source.ANCHOR_PATHS)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.f = Fixture(self.root)

    def admit(self, pins=None):
        return source.admit_upstream(self.root, self.f.expected, pins)

    def test_three_anchor_and_six_unverified_output_contract(self):
        self.assertEqual(
            set(source.ANCHOR_PATHS),
            {
                "event_cluster.yaml",
                REPORT + "publication_audit.json",
                REPORT + "failure_review.json",
            },
        )
        self.assertEqual(set(source.DEFAULT_EXPECTED), set(source.ANCHOR_PATHS))
        self.assertEqual(len(source.OUTPUT_PATHS), 6)

    def test_exact_original_and_retained_tables_no_transform(self):
        pins = source.collect_input_pins(self.root, self.f.expected)
        audit, original, retained = self.admit(pins)
        self.assertEqual(list(pins), sorted(pins))
        self.assertEqual(audit["files"], pins)
        self.assertEqual(audit["status"], "PINNED_FAILED_WAVE20_WITH_VERIFIED_WAVE19_WAVE18")
        self.assertFalse(audit["retained_outputs_numerically_verified"])
        self.assertFalse(audit["historical_models_refitted"])
        self.assertFalse(audit["retained_outputs_transformed"])
        self.assertEqual(
            set(retained),
            {"forecasts", "states", "memory", "fits", "support", "upstream_admission"},
        )
        for key in ("forecasts", "states", "memory"):
            pd.testing.assert_frame_equal(
                retained[key],
                pd.read_parquet(self.root / (DATA + key + ".parquet")),
                check_exact=True,
            )
        self.assertEqual(retained["fits"], self.f.documents[DATA + "fits.json"])
        self.assertEqual(retained["support"], self.f.documents[DATA + "support_audit.json"])
        self.assertEqual(retained["upstream_admission"], audit["original_admission"])
        self.assertTrue(pd.isna(original["targets"].iloc[-1].y))
        self.assertIn("private/wave20/outside.bin", pins)
        self.assertIn("reports/early_session_feasibility/SUMMARY.md", pins)
        self.assertEqual(audit["failed_wave_counts"]["ledger_events"], 140)

    def test_collection_is_metadata_only_and_unpublished_payload_is_opaque(self):
        with (
            patch.object(pd, "read_parquet", side_effect=AssertionError("table decode")),
            patch.object(old, "admit_upstream", side_effect=AssertionError("table admission")),
        ):
            pins = source.collect_input_pins(self.root, self.f.expected)
        self.assertIn(REPORT + "unpublished_scored_metrics.json", pins)

    def test_no_producer_or_verifier_writer_called(self):
        with (
            patch(
                "src.event_cluster_pipeline.forecast_panel", side_effect=AssertionError("fit")
            ),
            patch("src.verify_event_cluster.verify", side_effect=AssertionError("writer")),
            patch(
                "src.range_alert_models.forecast_panel", side_effect=AssertionError("old fit")
            ),
        ):
            self.admit()

    def test_all_retained_and_report_identity_tampers_block_before_inner_load(self):
        for name in (
            *source.OUTPUT_PATHS,
            REPORT + "unpublished_scored_metrics.json",
            "private/wave20/outside.bin",
            REPORT + "full_repository_tests.txt",
        ):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"tamper")
            with self.subTest(name=name), patch.object(old, "admit_upstream") as loader:
                with self.assertRaises((ValueError, OSError)):
                    self.admit()
                loader.assert_not_called()
            path.write_bytes(original)

    def test_only_exact_identified_failure_is_accepted(self):
        for field, value in (
            ("status", "VERIFIED"),
            ("error", "some other verification failure"),
        ):
            original = copy.deepcopy(self.f.documents[REPORT + "verification.json"])
            changed = {**original, field: value}
            self.f.put(REPORT + "verification.json", changed)
            self.f.seal()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.admit()
            self.f.put(REPORT + "verification.json", original)
            self.f.seal()

    def test_failure_review_unit_stage_and_no_solve_declarations_are_exact(self):
        original = copy.deepcopy(self.f.documents[REPORT + "failure_review.json"])
        for field, value in (
            ("saved_dtype", "datetime64[ns]"),
            ("independent_new_stage_solves_completed", 1),
            ("independent_inference_reached", True),
        ):
            changed = copy.deepcopy(original)
            changed["error"][field] = value
            self.f.put(REPORT + "failure_review.json", changed)
            self.f.seal()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.admit()
        self.f.put(REPORT + "failure_review.json", original)
        self.f.seal()

    def test_canonical_three_p_one_and_empty_phases_cannot_be_promoted(self):
        original = copy.deepcopy(self.f.documents[REPORT + "metrics.json"])
        for field, value in (
            ("p_conservative", 0.1),
            ("p_holm_wave", 1),
            ("phases", [{"unpublished": True}]),
            ("verdict", "PASS"),
        ):
            changed = copy.deepcopy(original)
            changed["rows"][0][field] = value
            self.f.put(REPORT + "metrics.json", changed)
            self.f.put(REPORT + "failure.json", changed)
            self.f.seal()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.admit()
        self.f.put(REPORT + "metrics.json", original)
        self.f.put(REPORT + "failure.json", original)
        self.f.seal()

    def test_missing_required_failure_or_failed_verified_ancestors_reject(self):
        path = self.root / (REPORT + "failure.json")
        content = path.read_bytes()
        path.unlink()
        with self.assertRaises((OSError, ValueError)):
            self.admit()
        path.write_bytes(content)
        for name in (
            "reports/range_alert/failure.json",
            "reports/issued_calibration/failure.json",
        ):
            self.f.put(name, b"new failure")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.admit()
            (self.root / name).unlink()

    def test_registered_pins_require_every_transitive_source(self):
        pins = source.collect_input_pins(self.root, self.f.expected)
        pins.pop("private/wave20/outside.bin")
        with patch.object(old, "admit_upstream") as loader:
            with self.assertRaises(ValueError):
                self.admit(pins)
            loader.assert_not_called()

    def test_original_audit_identity_checked_before_original_table_load(self):
        changed = copy.deepcopy(self.f.documents[DATA + "upstream_admission.json"])
        changed["historical_models_refitted"] = True
        self.f.put(DATA + "upstream_admission.json", changed)
        self.f.seal()
        with patch.object(old, "admit_upstream") as loader:
            with self.assertRaises(ValueError):
                self.admit()
            loader.assert_not_called()

    def test_ledger_total_role_counts_and_terminal_rows_are_separate(self):
        review = copy.deepcopy(self.f.documents[REPORT + "failure_review.json"])
        review["ledger"]["event_counts"] = 140
        self.f.put(REPORT + "failure_review.json", review)
        self.f.seal()
        with self.assertRaises(ValueError):
            self.admit()

    def test_wrong_terminal_ledger_row_rejected_even_when_new_hashes_bound(self):
        name = REPORT + "trial_ledger.jsonl"
        lines = (self.root / name).read_bytes().splitlines()
        row = json.loads(lines[-1])
        row["p_conservative"] = 0.5
        lines[-1] = json.dumps(row).encode()
        self.f.put(name, b"\n".join(lines) + b"\n")
        self.f.seal()
        with self.assertRaises(ValueError):
            self.admit()

    def test_conflicting_or_unsafe_documentary_refs_reject(self):
        original = copy.deepcopy(self.f.documents[REPORT + "publication_audit.json"])
        for key, signature in (
            ("../escape", "a" * 64),
            ("/absolute", "a" * 64),
            ("SUMMARY.md", "a" * 64),
        ):
            changed = copy.deepcopy(original)
            changed["report_artifact_hashes"][key] = signature
            self.f.put(REPORT + "publication_audit.json", changed)
            self.f.expected = self.f.pins(*source.ANCHOR_PATHS)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.admit()
        self.f.put(REPORT + "publication_audit.json", original)

    def test_retained_frames_decode_only_checked_byte_buffers(self):
        reader = pd.read_parquet
        seen = []

        def inspect(path, *args, **kwargs):
            self.assertIsInstance(path, io.BytesIO)
            seen.append(hashlib.sha256(path.getvalue()).hexdigest())
            return reader(path, *args, **kwargs)

        with patch.object(pd, "read_parquet", side_effect=inspect):
            self.admit()
        for key in ("forecasts", "states", "memory"):
            self.assertIn(self.f.hash(DATA + key + ".parquet"), seen)

    def test_late_successful_ancestor_failure_marker_is_rechecked(self):
        real = source._Closure.read
        marker = self.root / "reports/issued_calibration/failure.json"

        def read(c, name, **kwargs):
            result = real(c, name, **kwargs)
            if kwargs.get("fresh") and name == sorted(c.files)[-1]:
                marker.write_text("late failure")
            return result

        with patch.object(source._Closure, "read", read), self.assertRaises(ValueError):
            self.admit()

    def test_required_failed_marker_is_rehashed_at_final_boundary(self):
        real = source._Closure.read
        marker = self.root / (REPORT + "failure.json")

        def read(c, name, **kwargs):
            result = real(c, name, **kwargs)
            if kwargs.get("fresh") and name == sorted(c.files)[-1]:
                marker.write_text("a different newly introduced failure")
            return result

        with patch.object(source._Closure, "read", read), self.assertRaises(ValueError):
            self.admit()

    def test_scored_middle_ledger_payloads_are_never_decoded(self):
        real = source.checked._json

        def decode(payload, **kwargs):
            if b'"opaque_unpublished_payload"' in payload or b'"opaque":' in payload:
                raise AssertionError("Opaque middle ledger payload was decoded")
            return real(payload, **kwargs)

        with patch.object(source.checked, "_json", side_effect=decode):
            self.admit()

    def test_retained_timestamp_units_and_unknowns_are_not_normalized(self):
        dates = pd.bdate_range("2001-02-01", periods=3).as_unit("ms")
        memory = pd.DataFrame({"date": dates, "unknown": [float("nan"), 0.0, 1.0]})
        states = pd.DataFrame(
            {"date": dates.as_unit("us"), "unknown": [float("nan"), 0.0, 1.0]}
        )
        self.f.frame(DATA + "memory.parquet", memory)
        self.f.frame(DATA + "states.parquet", states)
        self.f.seal()
        _, _, retained = self.admit()
        pd.testing.assert_frame_equal(retained["memory"], memory, check_exact=True)
        pd.testing.assert_frame_equal(retained["states"], states, check_exact=True)
        self.assertEqual(str(retained["memory"].date.dtype), "datetime64[ms]")
        self.assertEqual(str(retained["states"].date.dtype), "datetime64[us]")

    def test_symlink_and_duplicate_documentary_json_keys_are_rejected(self):
        name = "private/wave20/outside.bin"
        path = self.root / name
        payload = path.read_bytes()
        path.unlink()
        target = self.root / "private/wave20/alternate.bin"
        target.write_bytes(payload)
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            self.admit()
        path.unlink()
        path.write_bytes(payload)
        name = REPORT + "publication_audit.json"
        self.f.put(name, b'{"status": "FAILED", "status": "FAILED"}')
        self.f.expected = self.f.pins(*source.ANCHOR_PATHS)
        with self.assertRaises(ValueError):
            self.admit()


if __name__ == "__main__":
    unittest.main()
