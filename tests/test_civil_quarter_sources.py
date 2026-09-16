"""Synthetic source-closure contracts written before implementation or collection."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import civil_quarter_sources as source
from src.verify_macro_overnight import scan_source_admissions

CAPTURE = "data/source_discovery/bls_plan_capture/capture_manifest.json"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def write(root, name, value):
    payload = value if isinstance(value, bytes) else json.dumps(value).encode()
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha(payload)


def fixture(root):
    protocol = {
        "sources": {
            "cpi": "data/ledgers/cpi.json",
            "nfp": "data/ledgers/nfp.json",
            "fomc": "data/calendar/plans.csv",
            "fomc_coverage": "data/calendar/coverage.json",
            "daily": "NEVER_READ_MARKET.parquet",
        },
        "calendar": {
            "source_publication_start": "2010-01-01",
            "source_publication_end": "2025-10-20",
        },
    }
    records = {}
    for event, prefix in (("cpi", "cpi"), ("nfp", "empsit")):
        url = f"https://www.bls.gov/news.release/archives/{prefix}_01152010.htm"
        name = f"data/exemplars/{prefix}.txt"
        text = f"Original release ({url})\nL1: A synthetic original plan.\n"
        records[event] = [
            {
                "source_url": url,
                "parse_status": "VERIFIED_EXPLICIT_PLAN",
                "snapshot_path": name,
                "snapshot_sha256": write(root, name, text.encode()),
            }
        ]
    cap_hash = write(root, CAPTURE, {"records": [], "documentary_only": True})
    pins = {}
    for event in ("cpi", "nfp"):
        ledger = {"records": records[event]}
        if event == "nfp":
            ledger["artifact_sha256"] = {CAPTURE: cap_hash}
            ledger["superseded_intermediate_ledger_sha256"] = "a" * 64
        pins[protocol["sources"][event]] = write(root, protocol["sources"][event], ledger)
    name = "data/annual/original.txt"
    h = write(root, name, b"Synthetic annual source, no market observations.\n")
    annual = [
        {
            "annual_schedule_year": "2011",
            "source_url": "https://example.test/annual2011",
            "source_extraction_path": name,
            "source_extraction_sha256": h,
        }
        for _ in range(2)
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(annual[0]))
    writer.writeheader()
    writer.writerows(annual)
    pins[protocol["sources"]["fomc"]] = write(
        root, protocol["sources"]["fomc"], stream.getvalue().encode()
    )
    coverage = [
        {
            "annual_schedule_year": 2011,
            "coverage_start_date": "2011-01-01",
            "coverage_end_date": "2011-12-31",
            "plan_count": 2,
            "source_url": annual[0]["source_url"],
            "source_extraction_sha256": h,
        }
    ]
    pins[protocol["sources"]["fomc_coverage"]] = write(
        root, protocol["sources"]["fomc_coverage"], coverage
    )
    return protocol, pins


def change_json(root, protocol, pins, event, change):
    name = protocol["sources"][event]
    value = json.loads((root / name).read_bytes())
    change(value)
    pins[name] = write(root, name, value)


class SourceClosureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.protocol, self.pins = fixture(self.root)

    def collect(self):
        return source.collect_sources(self.root, self.protocol, self.pins)

    def test_outside_ledger_directory_closure_reaches_frozen_scanner(self):
        closure = self.collect()
        self.assertEqual(len(closure["runtime_files"]), 3)
        self.assertEqual(closure["counts"]["fomc_reference_rows"], 2)
        self.assertEqual(
            closure["documentary_files"], {CAPTURE: sha((self.root / CAPTURE).read_bytes())}
        )
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            snapshots = source.snapshot_sources(self.root, closure["files"])
            for name, payload in snapshots.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            audit = scan_source_admissions(stage, self.protocol)
        self.assertEqual(audit["bounded_source_sections_scanned"], 2)
        self.assertEqual(audit["mismatches"], [])

    def test_deterministic_json_and_no_market_path_read(self):
        first = self.collect()
        second = source.collect_sources(
            self.root, copy.deepcopy(self.protocol), dict(reversed(list(self.pins.items())))
        )
        self.assertEqual(json.dumps(first), json.dumps(second))
        for key in ("files", "metadata_files", "runtime_files", "documentary_files"):
            self.assertEqual(list(first[key]), sorted(first[key]))
        self.assertNotIn("NEVER_READ_MARKET.parquet", first["files"])

    def test_outside_fence_does_not_inspect_snapshot_path_or_hash(self):
        def change(d):
            d["records"].append(
                {
                    "source_url": "https://www.bls.gov/news.release/archives/cpi_11202025.htm",
                    "parse_status": "OUTSIDE_PUBLICATION_FENCE",
                }
            )

        change_json(self.root, self.protocol, self.pins, "cpi", change)
        result = self.collect()
        self.assertEqual(result["counts"]["excluded_bls_records"], 1)
        self.assertEqual(len(result["runtime_files"]), 3)

    def test_unmarked_outside_fence_is_rejected(self):
        change_json(
            self.root,
            self.protocol,
            self.pins,
            "cpi",
            lambda d: d["records"].append(
                {
                    "source_url": "https://www.bls.gov/news.release/archives/cpi_11202025.htm",
                    "parse_status": "VERIFIED_EXPLICIT_PLAN",
                }
            ),
        )
        with self.assertRaises(ValueError):
            self.collect()

    def test_excluded_bounded_reissue_still_requires_snapshot(self):
        change_json(
            self.root,
            self.protocol,
            self.pins,
            "cpi",
            lambda d: d["records"][0].update(parse_status="ORIGINAL_VINTAGE_UNCERTAIN"),
        )
        (self.root / "data/exemplars/cpi.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            self.collect()

    def test_metadata_hash_checked_before_decode(self):
        path = self.root / self.protocol["sources"]["cpi"]
        path.write_bytes(b"\xffnotjson")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.collect()

    def test_each_metadata_root_requires_registered_pin(self):
        del self.pins[self.protocol["sources"]["fomc_coverage"]]
        with self.assertRaises(ValueError):
            self.collect()

    def test_duplicate_json_keys_are_rejected_after_hash_check(self):
        name = self.protocol["sources"]["cpi"]
        self.pins[name] = write(self.root, name, b'{"records": [], "records": []}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.collect()

    def test_unsafe_reference_paths_reject(self):
        for path in (
            "../escape.txt",
            "/absolute.txt",
            "data/../escape.txt",
            "data//x.txt",
            "./x.txt",
            "data\\x.txt",
            "",
        ):
            with self.subTest(path=path):
                change_json(
                    self.root,
                    self.protocol,
                    self.pins,
                    "cpi",
                    lambda d, path=path: d["records"][0].update(snapshot_path=path),
                )
                with self.assertRaises(ValueError):
                    self.collect()

    def test_symlink_escape_rejects_but_confined_symlink_is_valid(self):
        name = self.root / "data/exemplars/cpi.txt"
        original = name.read_bytes()
        inside = self.root / "data/inside.txt"
        inside.write_bytes(original)
        name.unlink()
        name.symlink_to(inside)
        self.collect()
        with tempfile.TemporaryDirectory() as external:
            outside = Path(external) / "outside.txt"
            outside.write_bytes(original)
            name.unlink()
            name.symlink_to(outside)
            with self.assertRaises(ValueError):
                self.collect()

    def test_missing_malformed_or_conflicting_hash_rejects(self):
        for bad in (None, "x" * 64, "A" * 64, "a" * 63):
            with self.subTest(bad=bad):
                change_json(
                    self.root,
                    self.protocol,
                    self.pins,
                    "cpi",
                    lambda d, bad=bad: d["records"][0].update(snapshot_sha256=bad),
                )
                with self.assertRaises(ValueError):
                    self.collect()

    def test_same_path_conflicting_active_declarations_reject(self):
        cpi = json.loads((self.root / self.protocol["sources"]["cpi"]).read_bytes())[
            "records"
        ][0]
        change_json(
            self.root,
            self.protocol,
            self.pins,
            "nfp",
            lambda d: d["records"][0].update(
                snapshot_path=cpi["snapshot_path"], snapshot_sha256="b" * 64
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflict"):
            self.collect()

    def test_discovered_file_cannot_override_existing_registration(self):
        self.pins["data/exemplars/cpi.txt"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "conflict"):
            self.collect()

    def test_snapshot_tamper_after_collection_fails_before_staging(self):
        closure = self.collect()
        (self.root / "data/exemplars/cpi.txt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            source.snapshot_sources(self.root, closure["files"])

    def test_snapshot_returns_single_checked_bytes_even_if_path_changes_later(self):
        closure = self.collect()
        before = (self.root / "data/exemplars/cpi.txt").read_bytes()
        snapshots = source.snapshot_sources(self.root, closure["files"])
        (self.root / "data/exemplars/cpi.txt").write_bytes(b"later path mutation")
        self.assertEqual(snapshots["data/exemplars/cpi.txt"], before)
        self.assertEqual(
            sha(snapshots["data/exemplars/cpi.txt"]),
            closure["files"]["data/exemplars/cpi.txt"],
        )

    def test_capture_manifest_bound_by_final_ledger_and_not_superseded_metadata(self):
        change_json(
            self.root,
            self.protocol,
            self.pins,
            "nfp",
            lambda d: d.update(
                original_ledger_path="nonexistent_superseded.json",
                superseded_intermediate_ledger_sha256="f" * 64,
            ),
        )
        self.collect()
        (self.root / CAPTURE).write_bytes(b"\xffchanged manifest")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.collect()

    def test_capture_manifest_requires_exact_final_artifact_hash(self):
        change_json(
            self.root, self.protocol, self.pins, "nfp", lambda d: d.update(artifact_sha256={})
        )
        with self.assertRaises(ValueError):
            self.collect()

    def test_fomc_coverage_must_match_active_rows(self):
        original = json.loads(
            (self.root / self.protocol["sources"]["fomc_coverage"]).read_bytes()
        )
        for key, value in (
            ("source_extraction_sha256", "b" * 64),
            ("source_url", "wrong"),
            ("plan_count", 3),
            ("coverage_end_date", "2011-12-30"),
            ("annual_schedule_year", 2012),
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed[0][key] = value
                name = self.protocol["sources"]["fomc_coverage"]
                self.pins[name] = write(self.root, name, changed)
                with self.assertRaises(ValueError):
                    self.collect()

    def test_snapshot_content_is_not_decoded_during_collection(self):
        name = "data/exemplars/cpi.txt"
        h = write(self.root, name, b"\xffopaque checked source bytes")
        change_json(
            self.root,
            self.protocol,
            self.pins,
            "cpi",
            lambda d: d["records"][0].update(snapshot_sha256=h),
        )
        self.assertEqual(self.collect()["runtime_files"][name], h)

    def test_only_selected_metadata_and_active_snapshot_bytes_are_read(self):
        actual = Path.read_bytes
        accessed = []

        def read(path):
            accessed.append(path.relative_to(self.root).as_posix())
            return actual(path)

        with patch.object(Path, "read_bytes", read):
            result = self.collect()
        self.assertEqual(set(accessed), set(result["files"]))
        self.assertEqual(len(accessed), len(result["files"]))


if __name__ == "__main__":
    unittest.main()
