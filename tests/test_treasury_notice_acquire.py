"""Prewritten generated notice-batch transport and preservation contracts."""

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import treasury_notice_acquire as acquire

BASE = "https://www.treasurydirect.gov/instit/annceresult/press/preanre/2018/"
CAPTURE = "reports/next_signal_review/TREASURY_DOCUMENT_CAPTURE.json"
PRIVATE = "data/source_discovery/treasury_auction/full_notices_v1"
REPORT = "reports/next_signal_review/TREASURY_FULL_NOTICE_CAPTURE.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def member(day="2018-01-04", cusip="912ABC123"):
    return {
        "auction_date": day,
        "announcement_date": "2018-01-02",
        "cusip": cusip,
        "security_type": "Note",
        "security_term": "2-Year",
        "term_bucket": "2-Year",
    }


def bundle(root):
    rep = root / "reports/next_signal_review"
    rep.mkdir(parents=True)
    old = root / "data/pilot"
    old.mkdir(parents=True)
    body = old / "body.pdf"
    body.write_bytes(b"%PDF-1.4\nPRIVATE REUSED SENTINEL")
    headers = old / "headers.txt"
    headers.write_text("HTTP/2 200\n")
    receipt = {
        "requested_url": BASE + "a.pdf",
        "effective_url": BASE + "a.pdf",
        "http_status": 200,
        "curl_exit": 0,
        "content_type": "application/pdf",
        "bytes": body.stat().st_size,
        "sha256": digest(body),
        "retrieved_utc": "2026-09-09T00:00:00+00:00",
    }
    write(old / "receipt.json", receipt)
    captured = {
        "documents": [
            {
                "url": BASE + "a.pdf",
                "format": "pdf",
                "status": "VERIFIED_TRANSPORT_SCHEMA_ONLY",
                "body_path": "data/pilot/body.pdf",
                "body_sha256": digest(body),
                "receipt_path": "data/pilot/receipt.json",
                "receipt_sha256": digest(old / "receipt.json"),
                "headers_sha256": digest(headers),
                "memberships": [
                    {"auction_date": "2018-01-04", "cusip": "912ABC123", "kind": "special_pdf"}
                ],
            }
        ]
    }
    write(root / CAPTURE, captured)
    source = root / "data/inventory.json"
    write(source, {"metadata_only": True})
    selection = {
        "status": "FIXED_KNOWN_SPECIAL_NOTICE_METADATA_MANIFEST",
        "source_floor": "2010-01-01",
        "source_ceiling": "2025-10-20",
        "source_admitted": False,
        "cumulative_comparisons": 144,
        "distinct_notice_urls": 3,
        "already_captured_and_reviewed": 1,
        "not_yet_captured": 2,
        "provisional_nominal_rows_with_notices": 4,
        "source_pins": {
            CAPTURE: digest(root / CAPTURE),
            "data/inventory.json": digest(source),
        },
        "notices": [
            {
                "url": BASE + "a.pdf",
                "status": "CAPTURED_REVIEWED_PILOT_ORIGINAL",
                "body_path": "data/pilot/body.pdf",
                "body_sha256": digest(body),
                "archive_memberships": [member(), member("2018-01-05", "912ABC124")],
            },
            {
                "url": BASE + "b.pdf",
                "status": "NOT_YET_CAPTURED",
                "archive_memberships": [member("2018-01-08", "912ABC125")],
            },
            {
                "url": BASE + "c.pdf",
                "status": "NOT_YET_CAPTURED",
                "archive_memberships": [member("2018-01-09", "912ABC126")],
            },
        ],
    }
    selection_path = rep / "selection.json"
    write(selection_path, selection)
    checkpoint_path = rep / "checkpoint.json"
    write(
        checkpoint_path,
        {
            "status": "READY_FULL_NOTICE_CAPTURE",
            "pins": selection["source_pins"]
            | {str(selection_path.relative_to(root)): digest(selection_path)},
        },
    )
    return selection_path, checkpoint_path


def repin_selection(root, selection_path, checkpoint_path, selection):
    write(selection_path, selection)
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["pins"][str(selection_path.relative_to(root))] = digest(selection_path)
    write(checkpoint_path, checkpoint)


class TreasuryNoticeAcquireTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.selection, self.checkpoint = bundle(self.root)

    def fake(self, args, **kwargs):
        self.assertEqual(args[0], "curl")
        self.assertFalse(set(args) & {"--insecure", "-k", "--location", "-L", "--retry"})
        self.assertEqual(args[args.index("--proto") + 1], "=https")
        self.assertEqual(args[args.index("--max-time") + 1], "45")
        self.assertEqual(args[args.index("--max-filesize") + 1], str(8 * 1024 * 1024))
        self.assertEqual(kwargs["timeout"], 50)
        body = Path(args[args.index("--output") + 1])
        private = body.parent.parent
        self.assertTrue((private / "started.json").is_file())
        body.write_bytes(b"%PDF-1.4\nPRIVATE NEW SENTINEL")
        Path(args[args.index("--dump-header") + 1]).write_text("HTTP/2 200\n")
        return SimpleNamespace(
            returncode=0, stdout=f"200\n{args[-1]}\napplication/pdf", stderr=""
        )

    def test_reuse_original_bytes_without_call_or_copy_and_preserve_full_memberships(self):
        old = {p: digest(p) for p in (self.root / "data/pilot").iterdir()}
        with patch.object(acquire.subprocess, "run", side_effect=self.fake) as call:
            result = acquire.capture_notice_batch(self.root, self.selection, self.checkpoint)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(
            {c.args[0][-1] for c in call.call_args_list}, {BASE + "b.pdf", BASE + "c.pdf"}
        )
        self.assertEqual(len(result["documents"]), 3)
        reused = next(d for d in result["documents"] if d["url"] == BASE + "a.pdf")
        self.assertEqual(reused["source_capture_mode"], "reused_original_bytes")
        self.assertEqual(reused["body_path"], "data/pilot/body.pdf")
        self.assertEqual(reused["receipt_path"], "data/pilot/receipt.json")
        self.assertEqual(len(reused["archive_memberships"]), 2)
        self.assertEqual(len(reused["memberships"]), 2)
        self.assertTrue(
            all(
                d["status"] == "VERIFIED_TRANSPORT_SCHEMA_ONLY" and d["format"] == "pdf"
                for d in result["documents"]
            )
        )
        self.assertTrue(
            all(
                d["source_capture_mode"] == "new_request"
                for d in result["documents"]
                if d["url"] != BASE + "a.pdf"
            )
        )
        self.assertEqual({p: digest(p) for p in old}, old)
        self.assertFalse(
            (
                self.root / PRIVATE / hashlib.sha256((BASE + "a.pdf").encode()).hexdigest()
            ).exists()
        )
        self.assertEqual(result["checkpoint_sha256"], digest(self.checkpoint))
        self.assertEqual(result["selection_sha256"], digest(self.selection))
        self.assertIs(result["numerical_source_admitted"], False)
        self.assertEqual(result["predictive_comparisons_added"], 0)
        self.assertNotIn("SENTINEL", json.dumps(result))
        self.assertEqual(json.loads((self.root / REPORT).read_text()), result)

    def test_all_checkpoint_pins_and_readiness_precede_outputs_or_requests(self):
        for case in (
            "status",
            "drift",
            "unpin_selection",
            "missing_source_pin",
            "wrong_source_pin",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sel, chk = bundle(root)
                cp = json.loads(chk.read_text())
                if case == "status":
                    cp["status"] = "NOT_READY"
                if case == "drift":
                    (root / "data/inventory.json").write_text("changed")
                if case == "unpin_selection":
                    del cp["pins"][str(sel.relative_to(root))]
                if case == "missing_source_pin":
                    del cp["pins"]["data/inventory.json"]
                if case == "wrong_source_pin":
                    cp["pins"]["data/inventory.json"] = "f" * 64
                write(chk, cp)
                with (
                    patch.object(acquire.subprocess, "run") as call,
                    self.assertRaises(ValueError),
                ):
                    acquire.capture_notice_batch(root, sel, chk)
                call.assert_not_called()
                self.assertFalse((root / PRIVATE).exists())
                self.assertFalse((root / REPORT).exists())

    def test_whole_manifest_dates_classes_counts_and_duplicates_preflight(self):
        cases = (
            "future",
            "before_floor",
            "announcement_after",
            "bad_class",
            "bad_term",
            "extra_member_key",
            "duplicate_url",
            "duplicate_association",
            "count",
            "event_count",
            "unknown_status",
            "unsafe_url",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sel, chk = bundle(root)
                s = json.loads(sel.read_text())
                last = s["notices"][-1]
                m = last["archive_memberships"][0]
                if case == "future":
                    m["auction_date"] = "2025-10-21"
                if case == "before_floor":
                    m["auction_date"] = "2009-12-31"
                if case == "announcement_after":
                    m["announcement_date"] = "2018-01-10"
                if case == "bad_class":
                    m["security_type"] = "Bill"
                if case == "bad_term":
                    m["term_bucket"] = "20-Year"
                if case == "extra_member_key":
                    m["yield"] = "opaque"
                if case == "duplicate_url":
                    last["url"] = s["notices"][0]["url"]
                if case == "duplicate_association":
                    last["archive_memberships"].append(copy.deepcopy(m))
                if case == "count":
                    s["not_yet_captured"] = 3
                if case == "event_count":
                    s["provisional_nominal_rows_with_notices"] = 3
                if case == "unknown_status":
                    last["status"] = "SKIP"
                if case == "unsafe_url":
                    last["url"] = BASE + "../c.pdf"
                repin_selection(root, sel, chk, s)
                with (
                    patch.object(acquire.subprocess, "run") as call,
                    patch.object(acquire, "validate_document_receipt") as receipt,
                    self.assertRaises(ValueError),
                ):
                    acquire.capture_notice_batch(root, sel, chk)
                call.assert_not_called()
                receipt.assert_not_called()
                self.assertFalse((root / PRIVATE).exists())

    def test_reused_body_and_pinned_receipt_must_match_original_transport(self):
        for case in (
            "body_drift",
            "receipt_drift",
            "receipt_http",
            "capture_url",
            "selected_hash",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sel, chk = bundle(root)
                s = json.loads(sel.read_text())
                cap = json.loads((root / CAPTURE).read_text())
                if case == "body_drift":
                    (root / "data/pilot/body.pdf").write_bytes(b"%PDF-1.4 changed")
                if case == "receipt_drift":
                    (root / "data/pilot/receipt.json").write_text("changed")
                if case == "selected_hash":
                    s["notices"][0]["body_sha256"] = "a" * 64
                if case == "capture_url":
                    cap["documents"][0]["url"] = BASE + "other.pdf"
                if case == "receipt_http":
                    rp = root / "data/pilot/receipt.json"
                    r = json.loads(rp.read_text())
                    r["http_status"] = 404
                    write(rp, r)
                    cap["documents"][0]["receipt_sha256"] = digest(rp)
                write(root / CAPTURE, cap)
                s["source_pins"][CAPTURE] = digest(root / CAPTURE)
                cp = json.loads(chk.read_text())
                cp["pins"][CAPTURE] = s["source_pins"][CAPTURE]
                write(chk, cp)
                repin_selection(root, sel, chk, s)
                with (
                    patch.object(acquire.subprocess, "run") as call,
                    self.assertRaises(ValueError),
                ):
                    acquire.capture_notice_batch(root, sel, chk)
                call.assert_not_called()
                self.assertFalse((root / PRIVATE).exists())

    def test_existing_directory_or_terminal_report_refuses_replay(self):
        for existing in (PRIVATE, REPORT):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sel, chk = bundle(root)
                if existing == PRIVATE:
                    (root / existing).mkdir(parents=True)
                else:
                    (root / existing).write_text("existing terminal")
                with (
                    patch.object(acquire.subprocess, "run") as call,
                    self.assertRaises(FileExistsError),
                ):
                    acquire.capture_notice_batch(root, sel, chk)
                call.assert_not_called()

    def test_individual_http_failure_keeps_terminal_receipt_and_remaining_requests(self):
        def response(args, **kwargs):
            result = self.fake(args, **kwargs)
            if args[-1] == BASE + "b.pdf":
                result.stdout = result.stdout.replace("200\n", "404\n", 1)
            return result

        with patch.object(acquire.subprocess, "run", side_effect=response) as call:
            result = acquire.capture_notice_batch(self.root, self.selection, self.checkpoint)
        self.assertEqual(call.call_count, 2)
        by_url = {d["url"]: d for d in result["documents"]}
        self.assertEqual(by_url[BASE + "b.pdf"]["status"], "REQUIRES_REVIEW")
        self.assertEqual(by_url[BASE + "c.pdf"]["status"], "VERIFIED_TRANSPORT_SCHEMA_ONLY")
        receipt = json.loads((self.root / by_url[BASE + "b.pdf"]["receipt_path"]).read_text())
        self.assertEqual(receipt["http_status"], 404)
        self.assertEqual(
            digest(self.root / by_url[BASE + "b.pdf"]["receipt_path"]),
            by_url[BASE + "b.pdf"]["receipt_sha256"],
        )
        self.assertTrue((self.root / REPORT).exists())

    def test_timeout_is_one_terminal_failure_without_retry_or_stopping_other_request(self):
        def response(args, **kwargs):
            if args[-1] == BASE + "b.pdf":
                raise subprocess.TimeoutExpired("curl", 50)
            return self.fake(args, **kwargs)

        with patch.object(acquire.subprocess, "run", side_effect=response) as call:
            result = acquire.capture_notice_batch(self.root, self.selection, self.checkpoint)
        self.assertEqual(call.call_count, 2)
        by_url = {d["url"]: d for d in result["documents"]}
        self.assertEqual(by_url[BASE + "b.pdf"]["status"], "REQUIRES_REVIEW")
        receipt = json.loads((self.root / by_url[BASE + "b.pdf"]["receipt_path"]).read_text())
        self.assertEqual(receipt["curl_exit"], 124)
        self.assertEqual(by_url[BASE + "c.pdf"]["status"], "VERIFIED_TRANSPORT_SCHEMA_ONLY")

    def test_repeated_event_across_notices_requires_identical_metadata(self):
        s = json.loads(self.selection.read_text())
        s["notices"][-1]["archive_memberships"] = [
            copy.deepcopy(s["notices"][0]["archive_memberships"][0])
        ]
        s["notices"][-1]["archive_memberships"][0]["term_bucket"] = "5-Year"
        s["provisional_nominal_rows_with_notices"] = 3
        repin_selection(self.root, self.selection, self.checkpoint, s)
        with patch.object(acquire.subprocess, "run") as call, self.assertRaises(ValueError):
            acquire.capture_notice_batch(self.root, self.selection, self.checkpoint)
        call.assert_not_called()
        self.assertFalse((self.root / PRIVATE).exists())


if __name__ == "__main__":
    unittest.main()
