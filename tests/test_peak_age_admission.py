"""Generated wave21-envelope/raw-source contracts written before implementation."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import event_cluster_replay_admission as previous
from src import peak_age_admission as source
from tests.test_event_cluster_replay_admission import Fixture as FailedFixture

REPORT = "reports/event_cluster_replay/"
DATA = "data/event_cluster_replay/"
SOURCES = {
    "daily": "data/research_paths/spx_daily.parquet",
    "vix": "data/free_sources/raw/cboe/VIX_History.csv",
    "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
    "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
}
OUTPUTS = (
    DATA + "upstream_admission.json",
    DATA + "reconstruction_audit.json",
    REPORT + "metrics.json",
    REPORT + "trial_ledger.jsonl",
)


class Fixture(FailedFixture):
    def __init__(self, root):
        # The superclass calls its own seal dynamically before wave21 exists.
        super().__init__(root)
        self.failed_expected = dict(self.expected)
        prior = previous._collect(root, self.failed_expected)
        self.put("event_cluster_replay.yaml", {"upstream": {"anchors": self.failed_expected}})
        self.put("src/verify_event_cluster_replay.py", b"# synthetic verified replay\n")
        self.put("src/verify_index_hinge.py", b"# synthetic source baseline verifier\n")
        self.put(
            "index_hinge.yaml",
            {
                "sources": SOURCES,
                "index": {"source_end": "2025-10-20", "sealed_start": "2025-11-03"},
            },
        )
        self.daily = pd.DataFrame(
            {k: [101.0, np.nan, -99.0, -88.0] for k in ("open", "high", "low", "close")},
            index=pd.DatetimeIndex(
                ["2025-10-17", "2025-10-20", "2025-10-21", "2025-11-03"], name="date"
            ),
        )
        self.frame(SOURCES["daily"], self.daily)
        for name, column in [("vix", "CLOSE"), ("vix9d", "CLOSE"), ("vvix", "VVIX")]:
            self.put(
                SOURCES[name],
                (
                    "DATE," + column + "\n10/17/2025,20\n10/20/2025,\n"
                    "10/21/2025,not-numeric\n11/03/2025,protected\n"
                ).encode(),
            )
        self.put(
            "reports/index_hinge/manifest.json",
            {
                "protocol_sha256": self.hash("index_hinge.yaml"),
                "code": self.pins("src/verify_index_hinge.py"),
                "inputs": {},
                "preserved": {},
            },
        )
        self.put(
            "reports/index_hinge/verification.json",
            {
                "status": "VERIFIED",
                "protocol_sha256": self.hash("index_hinge.yaml"),
                "verifier_sha256": self.hash("src/verify_index_hinge.py"),
            },
        )
        for n in (
            "DESIGN.md",
            "SUMMARY.md",
            "NEXT_PEAK_AGE_DESIGN.md",
            "PUBLICATION_REVIEW.md",
            "full_repository_tests.txt",
            "pre_run_checks.txt",
        ):
            self.put(REPORT + n, ("synthetic " + n).encode())
        self.put(DATA + "upstream_admission.json", prior.audit)
        self.put(DATA + "reconstruction_audit.json", {"synthetic": True})
        rows = [
            {
                "study": "event_cluster_replay",
                "candidate": "cluster",
                "control": c,
                "horizon": 1,
                "score": "brier",
                "p_conservative": 1.0,
            }
            for c in ("baseline", "nuisance", "recent_frequency")
        ]
        self.put(
            REPORT + "metrics.json",
            {
                "protocol_sha256": self.hash("event_cluster_replay.yaml"),
                "rows": rows,
                "inherited_rows": [{"p_conservative": 1.0}] * 134,
                "hypothesis_count": 3,
                "cumulative_hypothesis_count": 137,
                "leads": [],
            },
        )
        events = [
            {
                "event": "registered",
                "study": "event_cluster_replay",
                "candidate": "cluster",
                "control": r["control"],
                "horizon": 1,
                "score": "brier",
                "protocol_sha256": self.hash("event_cluster_replay.yaml"),
            }
            for r in rows
        ]
        events += [{"event": "inherited", "p_conservative": 1.0}] * 134
        events += [{"event": "evaluated", **r} for r in rows]
        self.put(
            REPORT + "trial_ledger.jsonl",
            ("\n".join(json.dumps(r) for r in events) + "\n").encode(),
        )
        code = {**prior.files, **self.pins("src/verify_event_cluster_replay.py")}
        code = {k: v for k, v in code.items() if k.startswith("src/")}
        self.put(
            REPORT + "freeze_record.json",
            {
                "protocol_sha256": self.hash("event_cluster_replay.yaml"),
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
                "protocol_sha256": self.hash("event_cluster_replay.yaml"),
                "code": code,
                "inputs": {**prior.files},
                "preserved": {},
            },
        )
        self.put(
            REPORT + "verification.json",
            {
                "status": "VERIFIED",
                "protocol_sha256": self.hash("event_cluster_replay.yaml"),
                "verifier_sha256": self.hash("src/verify_event_cluster_replay.py"),
                "ledger_events_verified": {"registered": 3, "inherited": 134, "evaluated": 3},
                "inference": {
                    "new_hypotheses_verified": 3,
                    "cumulative_hypotheses_verified": 137,
                },
            },
        )
        self.put(
            REPORT + "publication_audit.json",
            {
                "status": "VERIFIED_REPLAY_PUBLICATION_AUDITED",
                "protocol_sha256": self.hash("event_cluster_replay.yaml"),
                "prior_failed_wave20_preserved": True,
                "prior_wave20_publication_sha256": self.failed_expected[
                    "reports/event_cluster/publication_audit.json"
                ],
                "new_hypotheses": 3,
                "cumulative_hypotheses": 137,
                "ledger_event_counts": {"registered": 3, "inherited": 134, "evaluated": 3},
                "new_producer_fits": 0,
                "newly_generated_forecasts": 0,
                "full_repository_tests": 1,
                "leads": [],
            },
        )
        self.seal_replay()

    def seal(self):
        FailedFixture.seal(self)

    def seal_replay(self):
        for name in ("vix", "vix9d", "vvix"):
            self.put(
                SOURCES[name] + ".manifest.json",
                {
                    "manifest_version": 1,
                    "sha256": self.hash(SOURCES[name]),
                    "bytes": (self.root / SOURCES[name]).stat().st_size,
                    "source_id": "cboe:daily-index:" + name.upper(),
                    "source_version": "daily-live",
                },
            )
        self.put(
            "data/research_paths/source_manifest.json",
            {
                "protocol_version": 1,
                "sources": {
                    "spx": {
                        "sha256": self.hash(SOURCES["daily"]),
                        "auto_adjust": False,
                        "ticker": "^GSPC",
                        "provider": "Yahoo Finance via yfinance",
                    }
                },
            },
        )
        refs = [
            "index_hinge.yaml",
            "reports/index_hinge/manifest.json",
            "reports/index_hinge/verification.json",
            "src/verify_index_hinge.py",
            "data/research_paths/source_manifest.json",
            *SOURCES.values(),
            *(SOURCES[n] + ".manifest.json" for n in ("vix", "vix9d", "vvix")),
        ]
        m = copy.deepcopy(self.documents[REPORT + "manifest.json"])
        m["inputs"].update(self.pins(*refs))
        self.put(REPORT + "manifest.json", m)
        v = copy.deepcopy(self.documents[REPORT + "verification.json"])
        v["verified_output_hashes"] = self.pins(*OUTPUTS)
        v["verified_retained_input_hashes"] = self.pins(*previous.OUTPUT_PATHS)
        v["artifact_hashes_checked"] = {k: len(m[k]) for k in ("code", "inputs", "preserved")}
        v["artifact_hashes_checked"]["outputs"] = 4
        self.put(REPORT + "verification.json", v)
        p = copy.deepcopy(self.documents[REPORT + "publication_audit.json"])
        p.update(
            verified_output_hashes=v["verified_output_hashes"],
            verified_retained_input_hashes=v["verified_retained_input_hashes"],
        )
        p["manifest_entries_checked"] = {k: len(m[k]) for k in ("code", "inputs", "preserved")}
        names = [
            "manifest.json",
            "freeze_record.json",
            "verification.json",
            "metrics.json",
            "trial_ledger.jsonl",
            "DESIGN.md",
            "SUMMARY.md",
            "NEXT_PEAK_AGE_DESIGN.md",
            "PUBLICATION_REVIEW.md",
            "full_repository_tests.txt",
            "pre_run_checks.txt",
        ]
        p["report_artifact_hashes"] = {n: self.hash(REPORT + n) for n in names}
        for key, name in [
            ("manifest_sha256", "manifest.json"),
            ("freeze_sha256", "freeze_record.json"),
            ("verification_sha256", "verification.json"),
        ]:
            p[key] = self.hash(REPORT + name)
        self.put(REPORT + "publication_audit.json", p)
        self.expected = self.pins(
            "event_cluster_replay.yaml", REPORT + "publication_audit.json"
        )


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.f = Fixture(self.root)

    def test_metadata_only_collection_no_table_csv_decoding(self):
        with (
            patch.object(pd, "read_parquet", side_effect=AssertionError("numerical decoding")),
            patch.object(pd, "read_csv", side_effect=AssertionError("numeric CSV reading")),
        ):
            pins = source.collect_input_pins(self.root, self.f.expected)
        self.assertTrue(set(SOURCES.values()).issubset(pins))
        self.assertIn("private/wave20/outside.bin", pins)
        self.assertIn("reports/event_cluster/failure.json", pins)

    def test_checked_raw_sources_bounded_and_missing_calendar_preserved(self):
        pins = source.collect_input_pins(self.root, self.f.expected)
        audit, daily, iv = source.admit_upstream(self.root, self.f.expected, pins)
        pd.testing.assert_frame_equal(daily, self.f.daily.iloc[:2], check_freq=False)
        self.assertTrue(iv.iloc[1].isna().all())
        self.assertEqual(list(iv), ["vix", "vix9d", "vvix"])
        self.assertEqual(audit["status"], "VERIFIED_WAVE21_BOUND_RAW_SPX_CBOE_SNAPSHOTS")
        self.assertEqual(audit["source_end"], "2025-10-20")
        self.assertFalse(audit["numeric_post_cutoff_values_parsed"])
        self.assertFalse(audit["prior_forecast_tables_decoded"])
        self.assertEqual(audit["reference_calendar"]["sessions"], 2)
        self.assertTrue(audit["loaded_from_checked_snapshots"])

    def test_raw_decodes_only_from_bytes_and_frozen_loader_not_called(self):
        original_pq, original_csv = pd.read_parquet, pd.read_csv

        def pq(buffer, **kwargs):
            self.assertIsInstance(buffer, io.BytesIO)
            self.assertEqual(kwargs["filters"], [("date", "<=", pd.Timestamp("2025-10-20"))])
            self.assertEqual(kwargs["columns"], ["open", "high", "low", "close"])
            return original_pq(buffer, **kwargs)

        def csv(buffer, **kwargs):
            self.assertIsInstance(buffer, io.BytesIO)
            self.assertEqual(kwargs["dtype"], str)
            return original_csv(buffer, **kwargs)

        with (
            patch.object(pd, "read_parquet", side_effect=pq) as p,
            patch.object(pd, "read_csv", side_effect=csv) as c,
            patch.object(
                previous, "admit_upstream", side_effect=AssertionError("old numerical loader")
            ),
        ):
            source.admit_upstream(self.root, self.f.expected)
        self.assertEqual(p.call_count, 1)
        self.assertEqual(c.call_count, 3)

    def test_anchor_set_and_hash_required_before_numeric_decode(self):
        variants = [dict(self.f.expected), dict(self.f.expected), dict(self.f.expected)]
        variants[0].pop("event_cluster_replay.yaml")
        variants[1]["unexpected.json"] = "a" * 64
        variants[2]["event_cluster_replay.yaml"] = "a" * 64
        for expected in variants:
            with self.subTest(expected=expected), patch.object(pd, "read_parquet") as decode:
                with self.assertRaises(ValueError):
                    source.admit_upstream(self.root, expected)
                decode.assert_not_called()

    def test_tampered_raw_or_nested_preserved_file_fails_before_decode(self):
        for name in [SOURCES["daily"], SOURCES["vvix"], "private/wave20/outside.bin"]:
            path = self.root / name
            before = path.read_bytes()
            path.write_bytes(before + b"tampered")
            with self.subTest(name=name), patch.object(pd, "read_parquet") as decode:
                with self.assertRaises(ValueError):
                    source.admit_upstream(self.root, self.f.expected)
                decode.assert_not_called()
            path.write_bytes(before)

    def test_registered_omission_or_replacement_fails_before_decode(self):
        pins = source.collect_input_pins(self.root, self.f.expected)
        for name in [
            SOURCES["daily"],
            REPORT + "verification.json",
            "private/older/artifact.bin",
        ]:
            p = dict(pins)
            p.pop(name)
            with self.subTest(name=name), patch.object(pd, "read_parquet") as decode:
                with self.assertRaises(ValueError):
                    source.admit_upstream(self.root, self.f.expected, p)
                decode.assert_not_called()

    def test_wave21_status_family_output_set_and_verifier_hash_are_checked(self):
        original = copy.deepcopy(self.f.documents[REPORT + "verification.json"])
        for kind in ["status", "family", "verifier", "output"]:
            v = copy.deepcopy(original)
            if kind == "status":
                v["status"] = "FAILED"
            elif kind == "family":
                v["inference"]["cumulative_hypotheses_verified"] = True
            elif kind == "verifier":
                v["verifier_sha256"] = "a" * 64
            else:
                v["verified_output_hashes"].pop(OUTPUTS[0])
            self.f.put(REPORT + "verification.json", v)
            self.f.seal_replay()
            # seal repairs output map, so remove it once more and rebind only publication.
            if kind == "output":
                v = copy.deepcopy(self.f.documents[REPORT + "verification.json"])
                v["verified_output_hashes"].pop(OUTPUTS[0])
                self.f.put(REPORT + "verification.json", v)
                p = copy.deepcopy(self.f.documents[REPORT + "publication_audit.json"])
                p["verification_sha256"] = self.f.hash(REPORT + "verification.json")
                p["report_artifact_hashes"]["verification.json"] = p["verification_sha256"]
                p["verified_output_hashes"] = v["verified_output_hashes"]
                self.f.put(REPORT + "publication_audit.json", p)
                self.f.expected = self.f.pins(*source.ANCHOR_PATHS)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                source.collect_input_pins(self.root, self.f.expected)
            self.f.put(REPORT + "verification.json", original)
            self.f.seal_replay()

    def test_wave21_canonical_ledger_roles_and_counts(self):
        name = REPORT + "trial_ledger.jsonl"
        before = (self.root / name).read_bytes()
        lines = before.splitlines()
        row = json.loads(lines[3])
        row["event"] = "evaluated"
        lines[3] = json.dumps(row).encode()
        self.f.put(name, b"\n".join(lines) + b"\n")
        self.f.seal_replay()
        with self.assertRaises(ValueError):
            source.collect_input_pins(self.root, self.f.expected)

    def test_required_old_failure_and_success_failure_absence(self):
        for report in [
            "range_alert",
            "issued_calibration",
            "event_cluster_replay",
            "index_hinge",
        ]:
            p = self.root / "reports" / report / "failure.json"
            p.write_bytes(b"{}")
            with self.subTest(report=report), self.assertRaises(ValueError):
                source.collect_input_pins(self.root, self.f.expected)
            p.unlink()
        p = self.root / "reports/event_cluster/failure.json"
        p.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            source.collect_input_pins(self.root, self.f.expected)

    def test_symlink_sources_and_dangling_failure_markers_rejected(self):
        name = SOURCES["daily"]
        p = self.root / name
        target = self.root / "duplicate.parquet"
        p.rename(target)
        p.symlink_to(target)
        with self.assertRaises(ValueError):
            source.collect_input_pins(self.root, self.f.expected)
        p.unlink()
        target.rename(p)
        marker = self.root / (REPORT + "failure.json")
        marker.symlink_to(self.root / "absent")
        with self.assertRaises(ValueError):
            source.collect_input_pins(self.root, self.f.expected)

    def test_in_cutoff_invalid_numbers_reject_while_post_cutoff_text_opaque(self):
        name = SOURCES["vix"]
        self.f.put(name, b"DATE,CLOSE\n10/17/2025,bad\n11/03/2025,protected\n")
        self.f.seal_replay()
        with self.assertRaises(ValueError):
            source.admit_upstream(self.root, self.f.expected)

    def test_nonpositive_observed_ohlc_rejected_missing_is_not_dropped(self):
        daily = self.f.daily.copy()
        daily.loc[daily.index[0], "close"] = 0.0
        self.f.frame(SOURCES["daily"], daily)
        self.f.seal_replay()
        with self.assertRaises(ValueError):
            source.admit_upstream(self.root, self.f.expected)

    def test_unsorted_duplicate_subday_or_timezone_reference_rejected(self):
        dates = self.f.daily.index
        for variant in [
            dates[[1, 0, 2, 3]],
            pd.DatetimeIndex([dates[0], dates[0], dates[2], dates[3]], name="date"),
            dates + pd.Timedelta(hours=1),
            dates.tz_localize("UTC"),
        ]:
            daily = self.f.daily.copy()
            daily.index = variant
            self.f.frame(SOURCES["daily"], daily)
            self.f.seal_replay()
            with (
                self.subTest(variant=str(variant)),
                self.assertRaises((ValueError, TypeError)),
            ):
                source.admit_upstream(self.root, self.f.expected)

    def test_float64_contract_and_ms_us_ns_transport(self):
        for unit in ["ms", "us", "ns"]:
            daily = self.f.daily.copy()
            daily.index = daily.index.as_unit(unit)
            self.f.frame(SOURCES["daily"], daily)
            self.f.seal_replay()
            _, got, _ = source.admit_upstream(self.root, self.f.expected)
            self.assertEqual(got.index.dtype, daily.index.dtype)
        daily = self.f.daily.astype("float32")
        self.f.frame(SOURCES["daily"], daily)
        self.f.seal_replay()
        with self.assertRaises(ValueError):
            source.admit_upstream(self.root, self.f.expected)

    def test_reference_calendar_identity_changes_for_omitted_source_date(self):
        first, _, _ = source.admit_upstream(self.root, self.f.expected)
        self.f.frame(SOURCES["daily"], self.f.daily.drop(self.f.daily.index[1]))
        self.f.seal_replay()
        second, _, _ = source.admit_upstream(self.root, self.f.expected)
        self.assertNotEqual(
            first["reference_calendar"]["date_sequence_sha256"],
            second["reference_calendar"]["date_sequence_sha256"],
        )
        self.assertEqual(second["reference_calendar"]["sessions"], 1)
        self.assertEqual(second["sources"]["vix"]["outside_reference_dates"], ["2025-10-20"])

    def test_source_manifest_hash_and_identity_checked_before_decode(self):
        name = SOURCES["vix"] + ".manifest.json"
        m = copy.deepcopy(self.f.documents[name])
        m["sha256"] = "a" * 64
        self.f.put(name, m)
        manifest = copy.deepcopy(self.f.documents[REPORT + "manifest.json"])
        manifest["inputs"][name] = self.f.hash(name)
        self.f.put(REPORT + "manifest.json", manifest)
        pub = copy.deepcopy(self.f.documents[REPORT + "publication_audit.json"])
        pub["manifest_sha256"] = self.f.hash(REPORT + "manifest.json")
        pub["report_artifact_hashes"]["manifest.json"] = pub["manifest_sha256"]
        self.f.put(REPORT + "publication_audit.json", pub)
        self.f.expected = self.f.pins(*source.ANCHOR_PATHS)
        with patch.object(pd, "read_parquet") as decode:
            with self.assertRaises(ValueError):
                source.admit_upstream(self.root, self.f.expected)
            decode.assert_not_called()

    def test_post_snapshot_mutation_rejected_at_final_hash_check(self):
        original = pd.read_csv

        def changing(buffer, **kwargs):
            result = original(buffer, **kwargs)
            p = self.root / SOURCES["daily"]
            p.write_bytes(p.read_bytes() + b"late mutation")
            return result

        with patch.object(pd, "read_csv", side_effect=changing), self.assertRaises(ValueError):
            source.admit_upstream(self.root, self.f.expected)

    def test_success_failure_appearing_during_final_required_trio_rejected(self):
        original = source._Closure.read
        state = {"armed": False}
        marker = self.root / (REPORT + "failure.json")

        def changing(instance, name, **kwargs):
            result = original(instance, name, **kwargs)
            if kwargs.get("fresh") and name == sorted(instance.files)[-1]:
                state["armed"] = True
            if state["armed"] and name == "reports/event_cluster/verification.json":
                marker.write_bytes(b"{}")
            return result

        with patch.object(source._Closure, "read", changing), self.assertRaises(ValueError):
            source.collect_input_pins(self.root, self.f.expected)

    def test_old_failure_removed_after_last_general_hash_still_rejected(self):
        original = source._Closure.read
        marker = self.root / "reports/event_cluster/failure.json"

        def changing(instance, name, **kwargs):
            result = original(instance, name, **kwargs)
            if kwargs.get("fresh") and name == sorted(instance.files)[-1] and marker.exists():
                marker.unlink()
            return result

        with (
            patch.object(source._Closure, "read", changing),
            self.assertRaises((ValueError, FileNotFoundError)),
        ):
            source.collect_input_pins(self.root, self.f.expected)


if __name__ == "__main__":
    unittest.main()
