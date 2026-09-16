"""Synthetic two-layer admission contracts written before implementation."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import event_cluster_admission as admission
from src import issued_calibration_admission as old
from tests.test_issued_calibration_admission import Fixture as OldFixture

REPORT = "reports/issued_calibration/"
DATA = "data/issued_calibration/"
DOC = "reports/early_session_feasibility/"


class Fixture(OldFixture):
    def __init__(self, root):
        super().__init__(root)
        self.old_expected = dict(self.expected)
        inner, loaded = old.admit_upstream(root, self.old_expected)
        self.put("issued_calibration.yaml", {"upstream": {"anchors": self.old_expected}})
        self.put("src/verify_issued_calibration.py", b"# synthetic wave19 verifier\n")
        for name in (
            "DESIGN.md",
            "full_repository_tests.txt",
            "SUMMARY.md",
            "NEXT_RESEARCH_DIRECTION.md",
        ):
            self.put(REPORT + name, ("synthetic " + name).encode())
        self.put(REPORT + "comparison_intervals.png", b"synthetic wave19 figure")
        self.put("private/wave19/preserved.bin", b"wave19 preservation-only source")
        self.put(
            "reports/wave19_older/manifest.json",
            {"preserved": self.pins("private/wave19/preserved.bin")},
        )
        code = {
            **self.documents[old.REPORT + "manifest.json"]["code"],
            **self.pins("src/verify_issued_calibration.py"),
        }
        self.put(
            REPORT + "freeze_record.json",
            {
                "protocol_sha256": self.hash("issued_calibration.yaml"),
                "code": code,
                "prefit_design": self.pins(REPORT + "DESIGN.md"),
                "checks": {
                    "full_repository_tests": 1,
                    "full_log_sha256": self.hash(REPORT + "full_repository_tests.txt"),
                },
            },
        )
        self.put(
            REPORT + "manifest.json",
            {
                "protocol_sha256": self.hash("issued_calibration.yaml"),
                "code": code,
                "inputs": {**inner["files"], **self.pins(REPORT + "freeze_record.json")},
                "preserved": self.pins("reports/wave19_older/manifest.json"),
            },
        )
        self.put(DATA + "upstream_admission.json", inner)
        self.frame(DATA + "forecasts.parquet", loaded["forecasts"])
        self.frame(DATA + "states.parquet", loaded["states"])
        self.put(DATA + "calibration_audit.json", {"synthetic": True})
        rows = [
            {"candidate": "calibrated", "control": c, "p_conservative": 1.0}
            for c in ("baseline", "recent_frequency")
        ]
        self.put(
            REPORT + "metrics.json",
            {
                "protocol_sha256": self.hash("issued_calibration.yaml"),
                "hypothesis_count": 2,
                "cumulative_hypothesis_count": 131,
                "new_forecasts": 1,
                "reused_control_forecasts": 2,
                "combined_forecasts": 3,
                "new_monthly_fits": 0,
                "common_scored_origins": 1,
                "common_application_origins": 2,
                "rows": rows,
                "inherited_rows": [{"p_conservative": 1.0}] * 129,
                "leads": [],
            },
        )
        ledger = (
            [{"event": "registered", **r} for r in rows]
            + [{"event": "inherited"}] * 129
            + [{"event": "evaluated", **r} for r in rows]
        )
        self.put(
            REPORT + "trial_ledger.jsonl",
            ("\n".join(json.dumps(r) for r in ledger) + "\n").encode(),
        )
        outputs = self.pins(*admission.OUTPUT_PATHS)
        counts = {
            k: len(self.documents[REPORT + "manifest.json"][k])
            for k in ("code", "inputs", "preserved")
        }
        self.put(
            REPORT + "verification.json",
            {
                "status": "VERIFIED",
                "protocol_sha256": self.hash("issued_calibration.yaml"),
                "verifier_sha256": self.hash("src/verify_issued_calibration.py"),
                "verified_output_hashes": outputs,
                "forecast_reconstruction": {
                    "forecasts_verified": 3,
                    "new_forecasts": 1,
                    "reused_control_forecasts": 2,
                    "new_monthly_fits": 0,
                    "original_monthly_fits_replayed": 1,
                    "common_scored_origins": 1,
                    "common_application_origins": 2,
                },
                "inference": {
                    "new_hypotheses_verified": 2,
                    "cumulative_hypotheses_verified": 131,
                },
                "artifact_hashes_checked": {**counts, "outputs": 6},
                "ledger_events_verified": {
                    "inherited": 129,
                    "registered": 2,
                    "evaluated": 2,
                },
            },
        )
        self.put(
            REPORT + "publication_review.json",
            {
                "status": "APPROVED_VERIFIED_INTERPRETATION",
                "protocol_sha256": self.hash("issued_calibration.yaml"),
                "summary_sha256": self.hash(REPORT + "SUMMARY.md"),
                "figure_visually_reviewed": True,
                "document_hashes": self.pins(REPORT + "SUMMARY.md"),
                "figure_hashes": self.pins(REPORT + "comparison_intervals.png"),
            },
        )
        self.put(
            REPORT + "publication_audit.json",
            {
                "status": "VERIFIED_REPORT_AUDITED",
                "protocol_sha256": self.hash("issued_calibration.yaml"),
                "manifest_sha256": self.hash(REPORT + "manifest.json"),
                "freeze_sha256": self.hash(REPORT + "freeze_record.json"),
                "verification_sha256": self.hash(REPORT + "verification.json"),
                "verifier_sha256": self.hash("src/verify_issued_calibration.py"),
                "verified_output_hashes": outputs,
                "manifest_entries_checked": counts,
                "cumulative_hypotheses": 131,
                "current_wave_verified_forecasts": 1,
                "current_wave_reused_forecasts": 2,
                "current_wave_combined_forecasts": 3,
                "current_wave_verified_monthly_fits": 0,
                "current_wave_verified_applications": 2,
                "current_wave_scored_origins": 1,
                "ledger_events": 133,
                "ledger_event_counts": {"registered": 2, "inherited": 129, "evaluated": 2},
                "report_artifact_hashes": {
                    n: self.hash(REPORT + n)
                    for n in (
                        "SUMMARY.md",
                        "NEXT_RESEARCH_DIRECTION.md",
                        "verification.json",
                        "metrics.json",
                        "trial_ledger.jsonl",
                        "publication_review.json",
                        "comparison_intervals.png",
                    )
                },
            },
        )
        self.put(DOC + "SUMMARY.md", b"No newly admitted providers or experiment")
        self.refresh_documentary()
        self.expected = self.pins(*admission.ANCHOR_PATHS)

    def refresh_documentary(self):
        self.put(
            DOC + "preservation_audit.json",
            {
                "status": "DOCUMENTARY_REVIEW_COMPLETE_SOURCE_NOT_ADMITTED",
                "new_hypotheses": 0,
                "cumulative_hypotheses_unchanged": 131,
                "prior_anchors_verified": {
                    n: self.hash(REPORT + n)
                    for n in (
                        "publication_audit.json",
                        "freeze_record.json",
                        "manifest.json",
                        "NEXT_RESEARCH_DIRECTION.md",
                    )
                },
                "new_document_sha256": {"SUMMARY.md": self.hash(DOC + "SUMMARY.md")},
            },
        )

    def republish(self):
        """Update only synthetic proof links after an intentional semantic mutation."""
        outputs = self.pins(*admission.OUTPUT_PATHS)
        verification = dict(self.documents[REPORT + "verification.json"])
        verification["verified_output_hashes"] = outputs
        self.put(REPORT + "verification.json", verification)
        publication = dict(self.documents[REPORT + "publication_audit.json"])
        publication["verified_output_hashes"] = outputs
        publication["verification_sha256"] = self.hash(REPORT + "verification.json")
        publication["report_artifact_hashes"] = {
            n: self.hash(REPORT + n) for n in publication["report_artifact_hashes"]
        }
        self.put(REPORT + "publication_audit.json", publication)
        self.refresh_documentary()
        self.expected = self.pins(*admission.ANCHOR_PATHS)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.f = Fixture(self.root)

    def test_fixed_seven_anchor_six_output_contract(self):
        self.assertEqual(len(admission.ANCHOR_PATHS), 7)
        self.assertEqual(set(admission.DEFAULT_EXPECTED), set(admission.ANCHOR_PATHS))
        self.assertEqual(len(admission.OUTPUT_PATHS), 6)
        self.assertIn(DOC + "preservation_audit.json", admission.ANCHOR_PATHS)

    def test_frozen_verification_ledger_uses_typed_role_count_map(self):
        verified = dict(self.f.documents[REPORT + "verification.json"])
        verified["ledger_events_verified"] = {
            "inherited": 129,
            "registered": 2,
            "evaluated": 2,
        }
        self.f.put(REPORT + "verification.json", verified)
        self.f.republish()
        pins = admission.collect_input_pins(self.root, self.f.expected)
        audit, _ = admission.admit_upstream(self.root, self.f.expected, pins)
        self.assertEqual(audit["upstream_verified_counts"]["cumulative_hypotheses"], 131)
        # A scalar total or swapped same-total roles is not the frozen schema.
        for incorrect in (
            133,
            {"inherited": 129, "registered": 3, "evaluated": 1},
            {"inherited": 129, "registered": 2.0, "evaluated": 2},
        ):
            verified["ledger_events_verified"] = incorrect
            self.f.put(REPORT + "verification.json", verified)
            self.f.republish()
            with self.assertRaises(ValueError):
                admission.collect_input_pins(self.root, self.f.expected)

    def test_two_layers_preserve_exact_original_tables_without_fitting(self):
        pins = admission.collect_input_pins(self.root, self.f.expected)
        audit, loaded = admission.admit_upstream(self.root, self.f.expected, pins)
        self.assertEqual(audit["files"], pins)
        self.assertEqual(list(pins), sorted(pins))
        for name in (
            "private/older/artifact.bin",
            "private/outside/source.bin",
            "private/wave19/preserved.bin",
            DOC + "SUMMARY.md",
        ):
            self.assertIn(name, pins)
        self.assertEqual(audit["status"], "PINNED_WAVE19_WITH_WAVE18_RECORDS")
        self.assertEqual(
            audit["wave18_admission"], self.f.documents[DATA + "upstream_admission.json"]
        )
        self.assertEqual(audit["upstream_verified_counts"]["cumulative_hypotheses"], 131)
        self.assertFalse(audit["historical_models_refitted"])
        self.assertFalse(audit["source_values_reparsed"])
        self.assertEqual(len(loaded["states"]), 2)  # One unscored application survives.
        self.assertTrue(pd.isna(loaded["targets"].iloc[-1].y))
        self.assertEqual(loaded["protocol"], {"study_id": "range_alert_wave18"})
        self.assertEqual(audit["upstream_verified_counts"]["monthly_fits"], 0)
        self.assertEqual(
            audit["wave18_admission"]["upstream_verified_counts"]["monthly_fits"], 1
        )

    def test_metadata_collection_never_decodes_output_tables(self):
        with (
            patch.object(pd, "read_parquet", side_effect=AssertionError("not metadata")),
            patch.object(old, "admit_upstream", side_effect=AssertionError("no load")),
        ):
            admission.collect_input_pins(self.root, self.f.expected)

    def test_old_fit_and_verification_writers_are_never_called(self):
        with (
            patch(
                "src.range_alert_models.forecast_panel",
                side_effect=AssertionError("old refit"),
            ),
            patch(
                "src.issued_calibration_models.forecast_panel",
                side_effect=AssertionError("old calibration"),
            ),
            patch(
                "src.verify_range_alert.verify",
                side_effect=AssertionError("old verifier writer"),
            ),
            patch(
                "src.verify_issued_calibration.verify",
                side_effect=AssertionError("old verifier writer"),
            ),
        ):
            admission.admit_upstream(self.root, self.f.expected)

    def test_conflicting_preservation_binding_is_not_superseded(self):
        review = dict(self.f.documents[REPORT + "publication_review.json"])
        review["document_hashes"] = {"private/outside/source.bin": "0" * 64}
        self.f.put(REPORT + "publication_review.json", review)
        self.f.republish()
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            admission.collect_input_pins(self.root, self.f.expected)

    def test_all_outer_outputs_and_original_inputs_tamper_before_decode(self):
        for name in admission.OUTPUT_PATHS + (
            old.DATA + "features.parquet",
            old.DATA + "fits.json",
            old.DATA + "states.parquet",
        ):
            with self.subTest(name=name):
                p = self.root / name
                before = p.read_bytes()
                p.write_bytes(before + b"mutation")
                with (
                    patch.object(pd, "read_parquet") as decoder,
                    self.assertRaises(ValueError),
                ):
                    admission.admit_upstream(self.root, self.f.expected)
                decoder.assert_not_called()
                p.write_bytes(before)

    def test_saved_inner_audit_must_match_before_any_table_load(self):
        inner = dict(self.f.documents[DATA + "upstream_admission.json"])
        inner["historical_models_refitted"] = True
        self.f.put(DATA + "upstream_admission.json", inner)
        self.f.republish()
        with patch.object(pd, "read_parquet") as decoder, self.assertRaises(ValueError):
            admission.admit_upstream(self.root, self.f.expected)
        decoder.assert_not_called()

    def test_preserved_and_documentary_refs_are_active_hashes(self):
        for name in (
            "private/wave19/preserved.bin",
            DOC + "SUMMARY.md",
            REPORT + "comparison_intervals.png",
        ):
            with self.subTest(name=name):
                p = self.root / name
                before = p.read_bytes()
                p.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    admission.collect_input_pins(self.root, self.f.expected)
                p.write_bytes(before)

    def test_both_successful_upstream_failure_markers_block(self):
        for report in (REPORT, old.REPORT):
            with self.subTest(report=report):
                self.f.put(report + "failure.json", {"status": "UNEVALUABLE"})
                with self.assertRaisesRegex(ValueError, "failure"):
                    admission.admit_upstream(self.root, self.f.expected)
                (self.root / (report + "failure.json")).unlink()

    def test_old_failed_study_markers_are_preserved_not_new_failures(self):
        self.f.put("reports/civil_quarter/failure.json", {"status": "UNEVALUABLE"})
        admission.collect_input_pins(self.root, self.f.expected)

    def test_missing_registered_outer_or_inner_pin_before_decode(self):
        pins = admission.collect_input_pins(self.root, self.f.expected)
        for key in (
            DATA + "forecasts.parquet",
            old.DATA + "targets.parquet",
            DOC + "SUMMARY.md",
        ):
            bad = dict(pins)
            bad.pop(key)
            with patch.object(pd, "read_parquet") as decoder, self.assertRaises(ValueError):
                admission.admit_upstream(self.root, self.f.expected, bad)
            decoder.assert_not_called()

    def test_checked_buffer_decoding_and_outer_final_hash_guard(self):
        decoder = pd.read_parquet

        def mutated(value, *args, **kwargs):
            self.assertIsInstance(value, io.BytesIO)
            (self.root / (DATA + "states.parquet")).write_bytes(b"changed outer output")
            return decoder(value, *args, **kwargs)

        with (
            patch.object(pd, "read_parquet", side_effect=mutated),
            self.assertRaises(ValueError),
        ):
            admission.admit_upstream(self.root, self.f.expected)

    def test_failure_during_outer_last_rehash_cannot_escape(self):
        original = admission._Closure.read

        def read(c, name, **kwargs):
            result = original(c, name, **kwargs)
            if kwargs.get("fresh") and name == sorted(c.files)[-1]:
                self.f.put(REPORT + "failure.json", {"status": "UNEVALUABLE"})
            return result

        with (
            patch.object(admission._Closure, "read", new=read),
            self.assertRaisesRegex(ValueError, "failure"),
        ):
            admission.collect_input_pins(self.root, self.f.expected)

    def test_outer_anchor_hash_precedes_json_decoding(self):
        (self.root / (REPORT + "verification.json")).write_bytes(b"invalid JSON")
        with self.assertRaisesRegex(ValueError, "hash"):
            admission.collect_input_pins(self.root, self.f.expected)

    def test_semantic_outer_proof_and_documentary_identity_checks(self):
        for filename, key, value in (
            (REPORT + "verification.json", "status", "FAILED"),
            (REPORT + "publication_review.json", "figure_visually_reviewed", 1),
            (DOC + "preservation_audit.json", "new_hypotheses", 1),
            (DOC + "preservation_audit.json", "cumulative_hypotheses_unchanged", 132),
        ):
            with self.subTest(filename=filename, key=key):
                self.f = Fixture(self.root)
                j = dict(self.f.documents[filename])
                j[key] = value
                self.f.put(filename, j)
                if filename.startswith(REPORT):
                    self.f.republish()
                else:
                    self.f.expected[filename] = self.f.hash(filename)
                with self.assertRaises(ValueError):
                    admission.collect_input_pins(self.root, self.f.expected)

    def test_bad_output_set_and_typed_zero_fit_counts(self):
        for field, value in (("new_monthly_fits", False), ("new_forecasts", 2)):
            self.f = Fixture(self.root)
            j = dict(self.f.documents[REPORT + "verification.json"])
            j["forecast_reconstruction"] = {**j["forecast_reconstruction"], field: value}
            self.f.put(REPORT + "verification.json", j)
            self.f.republish()
            with self.assertRaises(ValueError):
                admission.collect_input_pins(self.root, self.f.expected)
        self.f = Fixture(self.root)
        j = dict(self.f.documents[REPORT + "verification.json"])
        j["verified_output_hashes"] = dict(j["verified_output_hashes"])
        j["verified_output_hashes"].pop(DATA + "states.parquet")
        self.f.update_anchor(REPORT + "verification.json", j)
        with self.assertRaises(ValueError):
            admission.collect_input_pins(self.root, self.f.expected)

    def test_path_hash_json_and_symlink_guards(self):
        for name in ("../bad", "/bad", "a//b", "a\\b"):
            self.f = Fixture(self.root)
            j = dict(self.f.documents[REPORT + "publication_review.json"])
            j["document_hashes"] = {name: "0" * 64}
            self.f.put(REPORT + "publication_review.json", j)
            self.f.republish()
            with self.assertRaises(ValueError):
                admission.collect_input_pins(self.root, self.f.expected)
        self.f = Fixture(self.root)
        path = self.root / "private/wave19/preserved.bin"
        payload = path.read_bytes()
        path.unlink()
        path.with_name("copy.bin").write_bytes(payload)
        path.symlink_to(path.with_name("copy.bin"))
        with self.assertRaisesRegex(ValueError, "symlink"):
            admission.collect_input_pins(self.root, self.f.expected)

    def test_duplicate_and_nonfinite_json_reject(self):
        for payload in (
            b'{"status":"VERIFIED","status":"FAILED"}',
            b'{"x":NaN}',
            b'{"x":1e999}',
        ):
            self.f.put(REPORT + "verification.json", payload)
            self.f.expected[REPORT + "verification.json"] = self.f.hash(
                REPORT + "verification.json"
            )
            with self.assertRaises(ValueError):
                admission.collect_input_pins(self.root, self.f.expected)

    def test_outer_terminal_failure_cannot_be_repromoted(self):
        j = dict(self.f.documents[REPORT + "metrics.json"])
        j["status"] = "UNEVALUABLE"
        self.f.put(REPORT + "metrics.json", j)
        self.f.republish()
        with self.assertRaises(ValueError):
            admission.admit_upstream(self.root, self.f.expected)

    def test_missing_conflicting_or_malformed_pins_reject(self):
        for expected in (
            {},
            {**self.f.expected, "extra": "0" * 64},
            {**self.f.expected, "issued_calibration.yaml": "bad"},
        ):
            with self.assertRaises(ValueError):
                admission.collect_input_pins(self.root, expected)
        (self.root / "private/wave19/preserved.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            admission.collect_input_pins(self.root, self.f.expected)

    def test_deterministic_read_only_operation(self):
        before = {
            str(p.relative_to(self.root)): p.read_bytes()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        first, _ = admission.admit_upstream(self.root, self.f.expected)
        second, _ = admission.admit_upstream(self.root, self.f.expected)
        self.assertEqual(first, second)
        after = {
            str(p.relative_to(self.root)): p.read_bytes()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
