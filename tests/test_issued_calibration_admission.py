"""Synthetic immutable-admission contracts, written before implementation."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import issued_calibration_admission as admission

REPORT = "reports/range_alert/"
DATA = "data/range_alert/"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class Fixture:
    def __init__(self, root):
        self.root = root
        self.documents = {}
        self.expected = {}
        self.put("range_alert.yaml", b"study_id: range_alert_wave18\n")
        self.put("src/verify_range_alert.py", b"# synthetic frozen verifier\n")
        self.put("src/frozen_prior.py", b"# synthetic old code\n")
        self.put("private/outside/source.bin", b"opaque source bytes, never numeric parsing")
        self.put("private/older/artifact.bin", b"older preserved evidence")
        self.put(REPORT + "DESIGN.md", b"synthetic frozen design\n")
        self.put(REPORT + "full_repository_tests.txt", b"synthetic tests passed\n")
        self.put(REPORT + "SUMMARY.md", b"synthetic verified summary\n")
        self.put(REPORT + "NEXT_RESEARCH_DIRECTION.md", b"synthetic unregistered proposal\n")
        self.put(REPORT + "comparison_intervals.png", b"synthetic figure bytes")
        self.put(
            "reports/older/manifest.json",
            {"existing_artifacts_sha256": self.pins("private/older/artifact.bin")},
        )
        self.put(
            "reports/prior/manifest.json",
            {
                "inputs": self.pins(
                    "private/outside/source.bin", "reports/older/manifest.json"
                ),
                "sources": {
                    "old": {
                        "path": "private/outside/source.bin",
                        "bounded_content_sha256": "0" * 64,
                    }
                },
            },
        )
        code = self.pins("src/verify_range_alert.py", "src/frozen_prior.py")
        self.put(
            REPORT + "freeze_record.json",
            {
                "protocol_sha256": self.hash("range_alert.yaml"),
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
                "protocol_sha256": self.hash("range_alert.yaml"),
                "code": code,
                "inputs": self.pins(
                    "reports/prior/manifest.json", REPORT + "freeze_record.json"
                ),
                "preserved": {},
            },
        )
        dates = pd.date_range("2020-01-01", periods=4)
        self.frame(DATA + "features.parquet", pd.DataFrame({"const": 1.0}, index=dates))
        self.frame(
            DATA + "targets.parquet",
            pd.DataFrame({"y": [0.0, 1.0, 0.0, float("nan")]}, index=dates),
        )
        self.frame(DATA + "forecasts.parquet", pd.DataFrame({"origin": [dates[1]] * 3}))
        self.frame(DATA + "states.parquet", pd.DataFrame({"origin": dates[1:3]}))
        self.put(DATA + "fits.json", [{"fit_origin": str(dates[1].date())}])
        for name in ("upstream_admission", "source_audit", "support_audit"):
            self.put(DATA + name + ".json", {"synthetic": True})
        rows = [
            {
                "study": "range_alert",
                "candidate": "location",
                "control": control,
                "horizon": 1,
                "score": "brier",
                "p_conservative": 1.0,
            }
            for control in ("baseline", "recent_frequency")
        ]
        self.put(
            REPORT + "metrics.json",
            {
                "protocol_sha256": self.hash("range_alert.yaml"),
                "hypothesis_count": 2,
                "cumulative_hypothesis_count": 129,
                "new_forecasts": 3,
                "new_monthly_fits": 1,
                "common_scored_origins": 1,
                "common_application_origins": 2,
                "rows": rows,
                "inherited_rows": [{"p_conservative": 1.0}] * 127,
                "leads": [],
            },
        )
        ledger = (
            [{"event": "registered", **row} for row in rows]
            + [{"event": "inherited", "p_conservative": 1.0}] * 127
            + [{"event": "evaluated", **row} for row in rows]
        )
        self.put(
            REPORT + "trial_ledger.jsonl",
            ("\n".join(json.dumps(x) for x in ledger) + "\n").encode(),
        )
        outputs = self.pins(*admission.OUTPUT_PATHS)
        self.put(
            REPORT + "verification.json",
            {
                "status": "VERIFIED",
                "protocol_sha256": self.hash("range_alert.yaml"),
                "verifier_sha256": self.hash("src/verify_range_alert.py"),
                "verified_output_hashes": outputs,
                "forecast_reconstruction": {
                    "forecasts_verified": 3,
                    "monthly_fits_verified": 1,
                    "common_scored_origins": 1,
                    "common_application_origins": 2,
                },
                "source_reconstruction": {"bounded_reference_rows": 4},
                "inference": {
                    "new_hypotheses_verified": 2,
                    "cumulative_hypotheses_verified": 129,
                },
                "artifact_hashes_checked": {
                    "code": 2,
                    "inputs": 2,
                    "preserved": 0,
                    "outputs": 10,
                },
            },
        )
        self.put(
            REPORT + "publication_review.json",
            {
                "status": "APPROVED_VERIFIED_INTERPRETATION",
                "protocol_sha256": self.hash("range_alert.yaml"),
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
                "protocol_sha256": self.hash("range_alert.yaml"),
                "manifest_sha256": self.hash(REPORT + "manifest.json"),
                "freeze_sha256": self.hash(REPORT + "freeze_record.json"),
                "verification_sha256": self.hash(REPORT + "verification.json"),
                "verifier_sha256": self.hash("src/verify_range_alert.py"),
                "verified_output_hashes": outputs,
                "manifest_entries_checked": {"code": 2, "inputs": 2, "preserved": 0},
                "cumulative_hypotheses": 129,
                "current_wave_verified_forecasts": 3,
                "current_wave_verified_monthly_fits": 1,
                "current_wave_verified_applications": 2,
                "current_wave_scored_origins": 1,
                "report_artifact_hashes": {
                    name: self.hash(REPORT + name)
                    for name in (
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
        self.expected = self.pins(*admission.ANCHOR_PATHS)

    def put(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(value, bytes):
            self.documents[name] = value
            value = (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()
        p.write_bytes(value)

    def hash(self, name):
        return sha((self.root / name).read_bytes())

    def pins(self, *names):
        return {name: self.hash(name) for name in names}

    def frame(self, name, frame):
        stream = io.BytesIO()
        frame.to_parquet(stream)
        self.put(name, stream.getvalue())

    def update_anchor(self, name, value):
        self.put(name, value)
        self.expected[name] = self.hash(name)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))
        self.root = self.fixture.root

    def test_default_anchor_paths_and_output_set_fixed(self):
        self.assertEqual(set(admission.DEFAULT_EXPECTED), set(admission.ANCHOR_PATHS))
        self.assertEqual(len(admission.ANCHOR_PATHS), 7)
        self.assertEqual(len(admission.OUTPUT_PATHS), 10)

    def test_complete_transitive_closure_and_no_model_refitting(self):
        pins = admission.collect_input_pins(self.root, self.fixture.expected)
        self.assertIn("private/outside/source.bin", pins)
        self.assertIn("private/older/artifact.bin", pins)
        audit, loaded = admission.admit_upstream(self.root, self.fixture.expected, pins)
        self.assertEqual(pins, audit["files"])
        self.assertEqual(list(pins), sorted(pins))
        self.assertEqual(audit["counts"]["manifests"], 3)
        self.assertEqual(
            audit["manifest_groups"]["reports/older/manifest.json"][
                "existing_artifacts_sha256"
            ],
            1,
        )
        self.assertEqual(audit["status"], "PINNED_WAVE18_VERIFIED_OUTPUTS")
        self.assertFalse(audit["historical_models_refitted"])
        self.assertFalse(audit["source_values_reparsed"])
        self.assertEqual(
            set(loaded),
            {
                "protocol",
                "features",
                "targets",
                "forecasts",
                "fits",
                "states",
                "support_audit",
                "source_audit",
                "upstream_admission",
                "metrics",
                "ledger",
            },
        )
        self.assertEqual(len(loaded["features"]), 4)
        self.assertEqual(len(loaded["ledger"]), 131)
        self.assertEqual(len(loaded["states"]), 2)

    def test_collect_metadata_never_decodes_parquet(self):
        with patch.object(pd, "read_parquet", side_effect=AssertionError("forbidden")):
            admission.collect_input_pins(self.root, self.fixture.expected)

    def test_any_verified_output_tamper_precedes_all_parquet_decoding(self):
        for name in admission.OUTPUT_PATHS:
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b"corrupt")
                with patch.object(pd, "read_parquet") as decoder:
                    with self.assertRaises(ValueError):
                        admission.admit_upstream(self.root, self.fixture.expected)
                    decoder.assert_not_called()
                path.write_bytes(original)

    def test_anchor_hash_checked_before_json_decode(self):
        (self.root / (REPORT + "verification.json")).write_bytes(b"not JSON")
        with self.assertRaisesRegex(ValueError, "hash|Hash"):
            admission.admit_upstream(self.root, self.fixture.expected)

    def test_recursive_old_preservation_hash_and_documentary_hash_enforced(self):
        for name in (
            "private/older/artifact.bin",
            REPORT + "comparison_intervals.png",
            REPORT + "full_repository_tests.txt",
        ):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    admission.collect_input_pins(self.root, self.fixture.expected)
                path.write_bytes(original)

    def test_missing_and_conflicting_declared_files_reject(self):
        (self.root / "private/outside/source.bin").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            admission.collect_input_pins(self.root, self.fixture.expected)

    def test_pin_conflict_rejected_even_if_reference_is_not_decoded(self):
        name = REPORT + "publication_review.json"
        value = self.fixture.documents[name].copy()
        value["document_hashes"] = {"private/outside/source.bin": "0" * 64}
        self.fixture.update_anchor(name, value)
        with self.assertRaises(ValueError):
            admission.collect_input_pins(self.root, self.fixture.expected)

    def test_missing_extra_or_malformed_anchor_map_rejected(self):
        for expected in (
            {},
            {**self.fixture.expected, "extra.json": "0" * 64},
            {**self.fixture.expected, "range_alert.yaml": "bad"},
        ):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                admission.collect_input_pins(self.root, expected)

    def test_noncanonical_and_escaping_metadata_paths_reject(self):
        for name in (
            "../outside",
            "/outside",
            "a/../outside",
            "a//outside",
            "a\\outside",
            "./outside",
        ):
            with self.subTest(name=name):
                key = REPORT + "publication_review.json"
                value = self.fixture.documents[key].copy()
                value["document_hashes"] = {name: "0" * 64}
                self.fixture.update_anchor(key, value)
                with self.assertRaises(ValueError):
                    admission.collect_input_pins(self.root, self.fixture.expected)

    def test_symlink_reference_rejected_even_inside_root(self):
        path = self.root / "private/outside/source.bin"
        contents = path.read_bytes()
        path.unlink()
        (path.parent / "same.bin").write_bytes(contents)
        path.symlink_to(path.parent / "same.bin")
        with self.assertRaisesRegex(ValueError, "symlink"):
            admission.collect_input_pins(self.root, self.fixture.expected)

    def test_duplicate_json_and_nonfinite_json_rejected(self):
        name = REPORT + "verification.json"
        for payload in (b'{"status":"VERIFIED","status":"FAILED"}', b'{"x":NaN}'):
            self.fixture.put(name, payload)
            self.fixture.expected[name] = self.fixture.hash(name)
            with self.assertRaises(ValueError):
                admission.collect_input_pins(self.root, self.fixture.expected)

    def test_success_and_identity_crosschecks_cannot_be_bypassed_by_repinning(self):
        for name, field, value in (
            ("verification.json", "status", "FAILED"),
            ("publication_audit.json", "status", "UNVERIFIED"),
            ("freeze_record.json", "protocol_sha256", "0" * 64),
            ("publication_audit.json", "verification_sha256", "0" * 64),
            ("publication_review.json", "figure_visually_reviewed", 1),
        ):
            with self.subTest(name=name, field=field):
                f = Fixture(self.root)
                full = REPORT + name
                document = f.documents[full].copy()
                document[field] = value
                f.update_anchor(full, document)
                with self.assertRaises(ValueError):
                    admission.collect_input_pins(self.root, f.expected)

    def test_complete_exact_ten_output_map_required(self):
        name = REPORT + "verification.json"
        doc = self.fixture.documents[name].copy()
        doc["verified_output_hashes"] = dict(doc["verified_output_hashes"])
        doc["verified_output_hashes"].pop(DATA + "states.parquet")
        self.fixture.update_anchor(name, doc)
        with self.assertRaises(ValueError):
            admission.collect_input_pins(self.root, self.fixture.expected)

    def test_registered_hash_coverage_before_output_decoding(self):
        pins = admission.collect_input_pins(self.root, self.fixture.expected)
        for key in (DATA + "targets.parquet", "private/older/artifact.bin"):
            damaged = dict(pins)
            damaged.pop(key)
            with patch.object(pd, "read_parquet") as decoder:
                with self.assertRaises(ValueError):
                    admission.admit_upstream(self.root, self.fixture.expected, damaged)
                decoder.assert_not_called()

    def test_decoding_uses_checked_bytes_and_final_mutation_rejects(self):
        original_decoder = pd.read_parquet
        seen = []

        def decode(value, *args, **kwargs):
            self.assertIsInstance(value, io.BytesIO)
            seen.append(value.getvalue())
            (self.root / (DATA + "states.parquet")).write_bytes(b"changed after checked read")
            return original_decoder(value, *args, **kwargs)

        with (
            patch.object(pd, "read_parquet", side_effect=decode),
            self.assertRaisesRegex(ValueError, "hash|Hash"),
        ):
            admission.admit_upstream(self.root, self.fixture.expected)
        self.assertTrue(seen)

    def test_failure_record_always_blocks_previously_verified_inputs(self):
        self.fixture.put(REPORT + "failure.json", {"status": "UNEVALUABLE"})
        with self.assertRaises(ValueError):
            admission.admit_upstream(self.root, self.fixture.expected)

    def test_failure_marker_created_during_last_rehash_blocks_return(self):
        original_read = admission._Closure.read

        def changed(closure, name, **kwargs):
            result = original_read(closure, name, **kwargs)
            if kwargs.get("fresh") and name == sorted(closure.files)[-1]:
                self.fixture.put(REPORT + "failure.json", {"status": "UNEVALUABLE"})
            return result

        with (
            patch.object(admission._Closure, "read", new=changed),
            self.assertRaisesRegex(ValueError, "failure"),
        ):
            admission.collect_input_pins(self.root, self.fixture.expected)

    def test_admission_deterministic_and_read_only(self):
        before = {
            str(p.relative_to(self.root)): sha(p.read_bytes())
            for p in self.root.rglob("*")
            if p.is_file()
        }
        first, _ = admission.admit_upstream(self.root, self.fixture.expected)
        second, _ = admission.admit_upstream(self.root, self.fixture.expected)
        self.assertEqual(first, second)
        after = {
            str(p.relative_to(self.root)): sha(p.read_bytes())
            for p in self.root.rglob("*")
            if p.is_file()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
