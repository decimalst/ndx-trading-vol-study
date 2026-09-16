"""Generated required-input and exact document-inventory binding contracts."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.treasury_history_bindings import bind_inventory, pinned_bytes

ROLES = ("announcement_pdf", "announcement_xml", "competitive_pdf", "competitive_xml")
PDF = "https://www.treasurydirect.gov/instit/annceresult/press/preanre/2018/"
XML = "https://www.treasurydirect.gov/xml/"


def fixtures():
    identity = {
        "auction_date": "2018-01-04",
        "announcement_date": "2018-01-02",
        "cusip": "912ABC123",
        "security_type": "Note",
        "security_term": "2-Year",
        "term_bucket": "2-Year",
    }
    documents = {
        "announcement_pdf": [PDF + "a.pdf"],
        "announcement_xml": [XML + "a.xml"],
        "competitive_pdf": [PDF + "r.pdf"],
        "competitive_xml": [XML + "r.xml"],
        "noncompetitive_pdf": [PDF + "n.pdf"],
        "special_pdf": [PDF + "missing_notice.pdf"],
    }
    requests = []
    for kind in ROLES:
        requests.append(
            {
                "url": documents[kind][0],
                "format": kind.rsplit("_", 1)[1],
                "memberships": [
                    {
                        "auction_date": identity["auction_date"],
                        "cusip": identity["cusip"],
                        "kind": kind,
                    }
                ],
                "archive_memberships": [identity | {"kind": kind}],
                "source_capture_mode": "reused_original_bytes"
                if kind == "announcement_pdf"
                else "new_request",
            }
        )
    requests.sort(key=lambda x: x["url"])
    selection = {
        "records": [{"record": identity | {"documents": documents}}],
        "requests": requests,
    }
    captured = []
    inspected = []
    for i, request in enumerate(requests):
        doc = copy.deepcopy(request) | {
            "body_path": f"data/generated/body{i}.{request['format']}",
            "body_sha256": str(i + 1) * 64,
            "status": "VERIFIED_TRANSPORT_SCHEMA_ONLY",
        }
        captured.append(doc)
        inspected.append(
            {
                k: copy.deepcopy(doc[k])
                for k in (
                    "url",
                    "format",
                    "memberships",
                    "body_path",
                    "body_sha256",
                    "source_capture_mode",
                )
            }
            | {"status": "METADATA_REQUIRES_REVIEW"}
        )
    return selection, {"documents": captured}, {"documents": inspected}


class TreasuryHistoryBindingsTests(unittest.TestCase):
    def test_pinned_bytes_returns_the_exact_authenticated_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = b"PRIVATE GENERATED SENTINEL\x00\xff"
            (root / "body.bin").write_bytes(raw)
            signature = hashlib.sha256(raw).hexdigest()
            pins = {"body.bin": signature}
            self.assertEqual(pinned_bytes(root, pins, "body.bin"), raw)
            self.assertEqual(pinned_bytes(root, pins, "body.bin", signature), raw)
            self.assertEqual(pins, {"body.bin": signature})

    def test_missing_pin_or_expected_hash_disagreement_precedes_file_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("unbound read")):
                for pins, expected in (
                    ({}, None),
                    ({"x": "a" * 64}, "b" * 64),
                    ({"x": "bad"}, None),
                ):
                    with (
                        self.subTest(pins=pins, expected=expected),
                        self.assertRaises(ValueError),
                    ):
                        pinned_bytes(root, pins, "x", expected)

    def test_actual_drift_missing_file_and_unsafe_paths_reject(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
            root = Path(temp)
            (root / "body").write_bytes(b"changed")
            (Path(outside) / "secret").write_bytes(b"outside")
            (root / "escape").symlink_to(Path(outside) / "secret")
            for relative in (
                "body",
                "missing",
                "../secret",
                str(Path(outside) / "secret"),
                "escape",
                None,
            ):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    pinned_bytes(root, {relative: "a" * 64}, relative)

    def test_matching_four_role_inventory_binds_and_retains_failed_metadata(self):
        selection, capture, metadata = fixtures()
        snapshot = copy.deepcopy((selection, capture, metadata))
        self.assertIsNone(bind_inventory(selection, capture))
        self.assertIsNone(bind_inventory(selection, capture, metadata))
        self.assertEqual((selection, capture, metadata), snapshot)
        self.assertTrue(
            any(
                "missing_notice" in u
                for u in selection["records"][0]["record"]["documents"]["special_pdf"]
            )
        )
        self.assertFalse(any("missing_notice" in d["url"] for d in capture["documents"]))

    def test_duplicate_selected_event_or_requested_url_rejects(self):
        for where in ("records", "requests"):
            selection, capture, metadata = fixtures()
            selection[where].append(copy.deepcopy(selection[where][0]))
            with self.subTest(where=where), self.assertRaises(ValueError):
                bind_inventory(selection, capture, metadata)

    def test_requested_memberships_must_reconstruct_from_original_records(self):
        for case in (
            "wrong_event",
            "wrong_role",
            "missing_request",
            "unrequested_notice",
            "full_metadata",
            "duplicate_membership",
        ):
            selection, capture, _metadata = fixtures()
            request = selection["requests"][0]
            if case == "wrong_event":
                request["memberships"][0]["cusip"] = "999ABC999"
            if case == "wrong_role":
                request["memberships"][0]["kind"] = "competitive_pdf"
            if case == "missing_request":
                selection["requests"].pop()
            if case == "unrequested_notice":
                request["url"] = PDF + "missing_notice.pdf"
            if case == "full_metadata":
                request["archive_memberships"][0]["announcement_date"] = "2018-01-01"
            if case == "duplicate_membership":
                request["memberships"].append(copy.deepcopy(request["memberships"][0]))
            # Alter the capture the same way: source-to-request agreement must still reject.
            capture["documents"][0].update(copy.deepcopy(request))
            with self.subTest(case=case), self.assertRaises(ValueError):
                bind_inventory(selection, capture)

    def test_capture_requires_exact_unique_urls_and_all_request_bindings(self):
        for case in (
            "duplicate",
            "missing",
            "extra",
            "format",
            "memberships",
            "archive_memberships",
            "source_capture_mode",
        ):
            selection, capture, _metadata = fixtures()
            docs = capture["documents"]
            if case == "duplicate":
                docs.append(copy.deepcopy(docs[0]))
            elif case == "missing":
                docs.pop()
            elif case == "extra":
                docs[0]["url"] = PDF + "other.pdf"
            elif case in ("memberships", "archive_memberships"):
                docs[0][case] = []
            else:
                docs[0][case] = "changed"
            with self.subTest(case=case), self.assertRaises(ValueError):
                bind_inventory(selection, capture)

    def test_metadata_requires_exact_unique_capture_urls_and_body_bindings(self):
        for case in (
            "duplicate",
            "missing",
            "extra",
            "format",
            "memberships",
            "body_path",
            "body_sha256",
            "source_capture_mode",
        ):
            selection, capture, metadata = fixtures()
            docs = metadata["documents"]
            if case == "duplicate":
                docs.append(copy.deepcopy(docs[0]))
            elif case == "missing":
                docs.pop()
            elif case == "extra":
                docs[0]["url"] = PDF + "other.pdf"
            elif case == "memberships":
                docs[0][case] = []
            else:
                docs[0][case] = "changed"
            with self.subTest(case=case), self.assertRaises(ValueError):
                bind_inventory(selection, capture, metadata)

    def test_shared_url_preserves_each_distinct_event_association(self):
        selection, capture, metadata = fixtures()
        other = copy.deepcopy(selection["records"][0])
        other["record"].update(auction_date="2018-01-05", cusip="912ABC124")
        selection["records"].append(other)
        for request, doc, inspected in zip(
            selection["requests"], capture["documents"], metadata["documents"]
        ):
            member = request["memberships"][0] | {
                "auction_date": "2018-01-05",
                "cusip": "912ABC124",
            }
            full = request["archive_memberships"][0] | member
            request["memberships"].append(member)
            request["archive_memberships"].append(full)
            doc.update(copy.deepcopy(request))
            inspected["memberships"] = copy.deepcopy(request["memberships"])
        self.assertIsNone(bind_inventory(selection, capture, metadata))
        capture["documents"][0]["memberships"].pop()
        with self.assertRaises(ValueError):
            bind_inventory(selection, capture, metadata)


if __name__ == "__main__":
    unittest.main()
