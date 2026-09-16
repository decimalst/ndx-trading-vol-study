"""Prewritten generated contracts for the bounded full-history capture."""

import copy
import hashlib
import json
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import treasury_history_acquire as acquire

PREFIX = "data/source_discovery/treasury_auction/inventory_capture_v2"
PILOT = "reports/next_signal_review/TREASURY_DOCUMENT_CAPTURE.json"
INVENTORY = "reports/next_signal_review/TREASURY_INVENTORY_CAPTURE.json"
AUDIT = "reports/next_signal_review/TREASURY_FULL_NOTICE_TERMINAL_AUDIT.json"
PRIVATE = "data/source_discovery/treasury_auction/history_capture_v1"
REPORT = "reports/next_signal_review/TREASURY_HISTORY_CAPTURE.json"
ROLES = ["announcement_pdf", "announcement_xml", "competitive_pdf", "competitive_xml"]
KINDS = ROLES + ["noncompetitive_pdf", "special_pdf"]
PDF = "https://www.treasurydirect.gov/instit/annceresult/press/preanre/2018/"
XML = "https://www.treasurydirect.gov/xml/"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def row(n, security_type="Note", bucket="2-Year"):
    return {
        "announcement_date": "2018-01-02",
        "auction_date": f"2018-01-{n + 3:02d}",
        "cusip": f"912ABC12{n}",
        "security_type": security_type,
        "security_term": "9-Year 10-Month" if bucket == "10-Year" else bucket,
        "term_bucket": bucket,
        "documents": {
            "announcement_pdf": [PDF + "shared.pdf"],
            "announcement_xml": [XML + f"a{n}.xml"],
            "competitive_pdf": [PDF + f"r{n}.pdf"],
            "competitive_xml": [XML + f"r{n}.xml"],
            "noncompetitive_pdf": [PDF + "n1.pdf"] if n == 1 else [],
            "special_pdf": [PDF + "unavailable_notice.pdf"],
        },
    }


def sources(root):
    rows = [row(2, bucket="10-Year"), row(1), row(3, "Bill"), row(4, "TIPS"), row(5, "FRN")]
    for year in range(2010, 2026):
        write(root / PREFIX / str(year) / "metadata.json", rows if year == 2018 else [])
    old = root / "data/pilot"
    old.mkdir(parents=True)
    (old / "body.pdf").write_bytes(b"%PDF-1.4\nPRIVATE ORIGINAL SENTINEL")
    (old / "headers.txt").write_text("HTTP/2 200\n")
    receipt = {
        "requested_url": PDF + "shared.pdf",
        "effective_url": PDF + "shared.pdf",
        "http_status": 200,
        "curl_exit": 0,
        "content_type": "application/pdf",
        "bytes": (old / "body.pdf").stat().st_size,
        "sha256": digest(old / "body.pdf"),
        "retrieved_utc": "2026-09-09T00:00:00+00:00",
    }
    write(old / "receipt.json", receipt)
    write(
        root / PILOT,
        {
            "status": "CAPTURED_SCHEMA_ONLY",
            "documents": [
                {
                    "url": PDF + "shared.pdf",
                    "format": "pdf",
                    "body_path": "data/pilot/body.pdf",
                    "body_sha256": digest(old / "body.pdf"),
                    "receipt_path": "data/pilot/receipt.json",
                    "receipt_sha256": digest(old / "receipt.json"),
                    "headers_sha256": digest(old / "headers.txt"),
                    "status": "VERIFIED_TRANSPORT_SCHEMA_ONLY",
                    "memberships": [
                        {
                            "auction_date": "2018-01-04",
                            "cusip": "912ABC121",
                            "kind": "announcement_pdf",
                        }
                    ],
                }
            ],
        },
    )
    write(root / INVENTORY, {"status": "VERIFIED_METADATA_INVENTORY"})
    write(
        root / AUDIT,
        {
            "status": "VERIFIED_FULL_NOTICE_REVIEW_TERMINAL_PRESERVATION",
            "source_history_admitted": False,
        },
    )
    paths = [f"{PREFIX}/{year}/metadata.json" for year in range(2010, 2026)] + [
        PILOT,
        INVENTORY,
        AUDIT,
    ]
    return {p: digest(root / p) for p in paths}


def freeze(root, pins):
    selection = acquire.build_history_selection(root, pins)
    sel = root / "reports/next_signal_review/TREASURY_HISTORY_SELECTION.json"
    cp = root / "reports/next_signal_review/TREASURY_HISTORY_CHECKPOINT.json"
    write(sel, selection)
    capture_pins = dict(pins)
    capture_pins[str(sel.relative_to(root))] = digest(sel)
    for path in (root / "data/pilot").iterdir():
        capture_pins[str(path.relative_to(root))] = digest(path)
    write(
        cp,
        {
            "status": "READY_FULL_HISTORY_CAPTURE",
            "pins": capture_pins,
            "expected_counts": selection["counts"],
        },
    )
    return sel, cp


class TreasuryHistoryAcquireTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pins = sources(self.root)

    def fake(self, args, **kwargs):
        self.assertEqual(args[0], "curl")
        self.assertFalse(set(args) & {"--insecure", "-k", "--location", "-L", "--retry"})
        self.assertEqual(args[args.index("--proto") + 1], "=https")
        self.assertEqual(args[args.index("--max-time") + 1], "45")
        self.assertEqual(kwargs["timeout"], 50)
        fmt = "xml" if args[-1].endswith(".xml") else "pdf"
        self.assertEqual(
            args[args.index("--max-filesize") + 1],
            str((2 if fmt == "xml" else 8) * 1024 * 1024),
        )
        self.assertTrue((self.root / PRIVATE / "started.json").is_file())
        body = Path(args[args.index("--output") + 1])
        body.write_bytes(
            b"<root>PRIVATE NUMERICAL SENTINEL</root>"
            if fmt == "xml"
            else b"%PDF-1.4 PRIVATE SENTINEL"
        )
        Path(args[args.index("--dump-header") + 1]).write_text("HTTP/2 200\n")
        return SimpleNamespace(
            returncode=0, stdout=f"200\n{args[-1]}\napplication/{fmt}", stderr=""
        )

    def test_selection_is_complete_deterministic_metadata_with_only_four_requested_roles(self):
        before = {p: digest(self.root / p) for p in self.pins}
        selected = acquire.build_history_selection(self.root, self.pins)
        self.assertEqual(
            selected,
            acquire.build_history_selection(
                self.root, dict(reversed(list(self.pins.items())))
            ),
        )
        self.assertEqual(selected["requested_roles"], ROLES)
        originals = sorted(
            [row(1), row(2, bucket="10-Year")], key=lambda r: (r["auction_date"], r["cusip"])
        )
        self.assertEqual([r["record"] for r in selected["records"]], originals)
        self.assertTrue(all(r["requested_roles"] == ROLES for r in selected["records"]))
        self.assertEqual(
            selected["counts"],
            {
                "metadata_rows": 5,
                "selected_events": 2,
                "original_role_associations": 11,
                "requested_role_associations": 8,
                "unique_requested_urls": 7,
                "reused_urls": 1,
                "new_requests": 6,
                "missing_requested_roles": 0,
            },
        )
        urls = [r["url"] for r in selected["requests"]]
        self.assertEqual(urls, sorted(set(urls)))
        self.assertFalse(any("unavailable_notice" in u or "n1.pdf" in u for u in urls))
        self.assertEqual(selected["source_pins"], self.pins)
        self.assertEqual(before, {p: digest(self.root / p) for p in self.pins})
        self.assertIs(selected["numerical_source_admitted"], False)
        self.assertEqual(selected["predictive_comparisons_added"], 0)

    def test_missing_role_is_explicit_and_never_guessed(self):
        path = self.root / PREFIX / "2018/metadata.json"
        rows = json.loads(path.read_text())
        rows[0]["documents"]["announcement_xml"] = []
        write(path, rows)
        self.pins[str(path.relative_to(self.root))] = digest(path)
        selected = acquire.build_history_selection(self.root, self.pins)
        self.assertEqual(selected["counts"]["missing_requested_roles"], 1)
        self.assertEqual(
            selected["missing_requested_roles"],
            [{"auction_date": "2018-01-05", "cusip": "912ABC122", "kind": "announcement_xml"}],
        )
        self.assertNotIn(XML + "a2.xml", [r["url"] for r in selected["requests"]])

    def test_exact_empty_cmb_bucket_is_preserved_without_a_tenor_and_filtered(self):
        path = self.root / PREFIX / "2018/metadata.json"
        rows = json.loads(path.read_text())
        rows[2].update(security_type="CMB", security_term="21-Day", term_bucket="")
        write(path, rows)
        self.pins[str(path.relative_to(self.root))] = digest(path)
        original_hash = digest(path)
        selected = acquire.build_history_selection(self.root, self.pins)
        self.assertEqual(selected["counts"]["metadata_rows"], 5)
        self.assertEqual(selected["counts"]["selected_events"], 2)
        self.assertEqual(selected["counts"]["unique_requested_urls"], 7)
        self.assertNotIn("912ABC123", [e["record"]["cusip"] for e in selected["records"]])
        self.assertEqual(digest(path), original_hash)
        self.assertEqual(json.loads(path.read_text())[2]["term_bucket"], "")
        sel, cp = freeze(self.root, self.pins)
        with patch.object(acquire.subprocess, "run", side_effect=self.fake) as call:
            result = acquire.capture_history(self.root, sel, cp)
        self.assertEqual(call.call_count, 6)
        self.assertEqual(result["status"], "CAPTURED_SCHEMA_ONLY")

    def test_no_other_blank_identity_or_nonliteral_empty_bucket_is_admitted(self):
        cases = [
            ("security_term", ""),
            ("security_term", " "),
            ("security_type", ""),
            ("security_type", " "),
            ("cusip", ""),
            ("announcement_date", ""),
            ("auction_date", ""),
            ("term_bucket", None),
            ("term_bucket", " "),
            ("term_bucket", 0),
        ]
        path = self.root / PREFIX / "2018/metadata.json"
        original = json.loads(path.read_text())
        for key, value in cases:
            with self.subTest(key=key, value=value):
                rows = copy.deepcopy(original)
                rows[2].update(security_type="CMB", security_term="21-Day", term_bucket="")
                rows[2][key] = value
                write(path, rows)
                self.pins[str(path.relative_to(self.root))] = digest(path)
                with self.assertRaises(ValueError):
                    acquire.build_history_selection(self.root, self.pins)

    def test_selection_requires_all_sixteen_pinned_years_and_checks_hashes_first(self):
        missing = dict(self.pins)
        del missing[f"{PREFIX}/2010/metadata.json"]
        with self.assertRaises(ValueError):
            acquire.build_history_selection(self.root, missing)
        path = self.root / PREFIX / "2025/metadata.json"
        path.write_text("not json")
        with (
            patch.object(acquire, "_json", side_effect=AssertionError("decoded before pins")),
            self.assertRaises(ValueError),
        ):
            acquire.build_history_selection(self.root, self.pins)

    def test_entire_metadata_date_identity_url_schema_preflight_including_filtered_rows(self):
        cases = [
            "future_filtered",
            "date_order",
            "bad_date",
            "duplicate",
            "wrong_year",
            "unsafe_unused_url",
            "unknown_key",
        ]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                pins = sources(root)
                path = root / PREFIX / "2018/metadata.json"
                rows = json.loads(path.read_text())
                if case == "future_filtered":
                    rows[2]["auction_date"] = "2025-10-21"
                if case == "date_order":
                    rows[0]["announcement_date"] = "2018-01-06"
                if case == "bad_date":
                    rows[0]["auction_date"] = "2018-02-30"
                if case == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                if case == "wrong_year":
                    rows[0]["auction_date"] = "2017-01-05"
                if case == "unsafe_unused_url":
                    rows[0]["documents"]["special_pdf"] = [PDF + "../evil.pdf"]
                if case == "unknown_key":
                    rows[0]["amount"] = "PRIVATE"
                write(path, rows)
                pins[str(path.relative_to(root))] = digest(path)
                with self.assertRaises(ValueError):
                    acquire.build_history_selection(root, pins)
                self.assertFalse((root / PRIVATE).exists())

    def test_capture_reuses_exact_original_and_retains_expanded_memberships(self):
        sel, cp = freeze(self.root, self.pins)
        originals = {p: digest(p) for p in (self.root / "data/pilot").iterdir()}
        with patch.object(acquire.subprocess, "run", side_effect=self.fake) as call:
            result = acquire.capture_history(self.root, sel, cp)
        self.assertEqual(call.call_count, 6)
        reused = next(x for x in result["documents"] if x["url"] == PDF + "shared.pdf")
        self.assertEqual(reused["source_capture_mode"], "reused_original_bytes")
        self.assertEqual(reused["body_path"], "data/pilot/body.pdf")
        self.assertEqual(len(reused["memberships"]), 2)
        self.assertEqual(len(reused["archive_memberships"]), 2)
        self.assertEqual(originals, {p: digest(p) for p in originals})
        self.assertFalse(
            (self.root / PRIVATE / hashlib.sha256(reused["url"].encode()).hexdigest()).exists()
        )
        self.assertEqual(result["status"], "CAPTURED_SCHEMA_ONLY")
        self.assertEqual(len(result["documents"]), 7)
        self.assertEqual(result["selection_sha256"], digest(sel))
        self.assertEqual(result["checkpoint_sha256"], digest(cp))
        self.assertEqual(json.loads((self.root / REPORT).read_text()), result)
        self.assertNotIn("SENTINEL", json.dumps(result))
        self.assertIs(result["numerical_source_admitted"], False)
        self.assertEqual(result["predictive_comparisons_added"], 0)
        for p in (self.root / PRIVATE).rglob("*"):
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o700 if p.is_dir() else 0o600)

    def test_preflight_rejects_checkpoint_counts_selection_and_pin_changes_without_outputs(
        self,
    ):
        for case in [
            "status",
            "count",
            "bool_count",
            "omit_record",
            "extra_request",
            "unpin_selection",
            "unpin_source",
            "drift",
        ]:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                pins = sources(root)
                sel, cp = freeze(root, pins)
                chk = json.loads(cp.read_text())
                selection = json.loads(sel.read_text())
                if case == "status":
                    chk["status"] = "NOT_READY"
                if case == "count":
                    chk["expected_counts"]["selected_events"] += 1
                if case == "bool_count":
                    chk["expected_counts"]["reused_urls"] = True
                if case == "omit_record":
                    selection["records"].pop()
                if case == "extra_request":
                    selection["requests"].append(copy.deepcopy(selection["requests"][0]))
                if case in ("omit_record", "extra_request"):
                    write(sel, selection)
                    chk["pins"][str(sel.relative_to(root))] = digest(sel)
                if case == "unpin_selection":
                    del chk["pins"][str(sel.relative_to(root))]
                if case == "unpin_source":
                    del chk["pins"][AUDIT]
                if case == "drift":
                    (root / PREFIX / "2010/metadata.json").write_text("changed")
                write(cp, chk)
                with (
                    patch.object(acquire.subprocess, "run") as call,
                    self.assertRaises(ValueError),
                ):
                    acquire.capture_history(root, sel, cp)
                call.assert_not_called()
                self.assertFalse((root / PRIVATE).exists())
                self.assertFalse((root / REPORT).exists())

    def test_reuse_requires_pinned_body_receipt_headers_and_valid_original_identity(self):
        for case in [
            "body_unpinned",
            "receipt_unpinned",
            "headers_unpinned",
            "body_drift",
            "receipt_identity",
            "pilot_membership",
        ]:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                pins = sources(root)
                if case == "receipt_identity":
                    path = root / "data/pilot/receipt.json"
                    r = json.loads(path.read_text())
                    r["effective_url"] = PDF + "other.pdf"
                    write(path, r)
                    pilot = json.loads((root / PILOT).read_text())
                    pilot["documents"][0]["receipt_sha256"] = digest(path)
                    write(root / PILOT, pilot)
                    pins[PILOT] = digest(root / PILOT)
                if case == "pilot_membership":
                    pilot = json.loads((root / PILOT).read_text())
                    pilot["documents"][0]["memberships"][0]["cusip"] = "999ABC999"
                    write(root / PILOT, pilot)
                    pins[PILOT] = digest(root / PILOT)
                sel, cp = freeze(root, pins)
                chk = json.loads(cp.read_text())
                for name in ("body", "receipt", "headers"):
                    if case == name + "_unpinned":
                        del chk["pins"][
                            f"data/pilot/{name}."
                            + (
                                "pdf"
                                if name == "body"
                                else "json"
                                if name == "receipt"
                                else "txt"
                            )
                        ]
                if case == "body_drift":
                    (root / "data/pilot/body.pdf").write_bytes(b"%PDF changed")
                write(cp, chk)
                with (
                    patch.object(acquire.subprocess, "run") as call,
                    self.assertRaises(ValueError),
                ):
                    acquire.capture_history(root, sel, cp)
                call.assert_not_called()
                self.assertFalse((root / PRIVATE).exists())

    def test_http_failure_and_timeout_preserve_receipts_and_finish_other_requests(self):
        sel, cp = freeze(self.root, self.pins)
        failed = {XML + "a1.xml", XML + "a2.xml"}

        def mixed(args, **kwargs):
            if args[-1] == XML + "a1.xml":
                raise subprocess.TimeoutExpired(args, 50)
            result = self.fake(args, **kwargs)
            if args[-1] == XML + "a2.xml":
                result.stdout = f"404\n{args[-1]}\napplication/xml"
            return result

        with patch.object(acquire.subprocess, "run", side_effect=mixed) as call:
            result = acquire.capture_history(self.root, sel, cp)
        self.assertEqual(call.call_count, 6)
        self.assertEqual(result["status"], "DOCUMENTS_REQUIRE_REVIEW")
        self.assertEqual(
            {d["url"] for d in result["documents"] if d["status"] == "REQUIRES_REVIEW"}, failed
        )
        for d in result["documents"]:
            self.assertEqual(d["receipt_sha256"], digest(self.root / d["receipt_path"]))
        timeout = next(d for d in result["documents"] if d["url"] == XML + "a1.xml")
        self.assertEqual(
            json.loads((self.root / timeout["receipt_path"]).read_text())["curl_exit"], 124
        )

    def test_replay_refuses_directory_or_report_without_modifying_sentinel(self):
        for existing in (PRIVATE, REPORT):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                pins = sources(root)
                sel, cp = freeze(root, pins)
                path = root / existing
                if existing == PRIVATE:
                    path.mkdir(parents=True)
                    path = path / "sentinel"
                path.write_bytes(b"PRESERVE")
                with (
                    patch.object(acquire.subprocess, "run") as call,
                    self.assertRaises(FileExistsError),
                ):
                    acquire.capture_history(root, sel, cp)
                call.assert_not_called()
                self.assertEqual(path.read_bytes(), b"PRESERVE")

    def test_three_worker_limit_and_final_input_drift_are_recorded(self):
        sel, cp = freeze(self.root, self.pins)

        def mutate(args, **kwargs):
            result = self.fake(args, **kwargs)
            if args[-1] == XML + "a1.xml":
                (self.root / AUDIT).write_text("changed during capture")
            return result

        executor = acquire.ThreadPoolExecutor
        with (
            patch.object(acquire, "ThreadPoolExecutor", wraps=executor) as pool,
            patch.object(acquire.subprocess, "run", side_effect=mutate),
        ):
            result = acquire.capture_history(self.root, sel, cp)
        pool.assert_called_once_with(max_workers=3)
        self.assertEqual(result["status"], "DOCUMENTS_REQUIRE_REVIEW")
        self.assertTrue(result["input_pin_errors"])
        self.assertEqual(len(result["documents"]), 7)


if __name__ == "__main__":
    unittest.main()
