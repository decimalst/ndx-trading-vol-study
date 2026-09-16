"""Generated acquisition tests before the first pilot document request."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.treasury_document_acquire import capture_documents
from tests.test_treasury_auction_documents import selection


class DocumentAcquireTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rep = self.root / "reports/next_signal_review"
        self.rep.mkdir(parents=True)
        self.selection = self.rep / "selection.json"
        self.selection.write_text(json.dumps(selection()))
        self.pin = self.root / "pinned.txt"
        self.pin.write_text("frozen")
        self.checkpoint = self.rep / "checkpoint.json"
        self.checkpoint.write_text(
            json.dumps(
                {
                    "status": "READY_DOCUMENT_SCHEMA_CAPTURE",
                    "pins": {
                        "pinned.txt": hashlib.sha256(self.pin.read_bytes()).hexdigest(),
                        str(self.selection.relative_to(self.root)): hashlib.sha256(
                            self.selection.read_bytes()
                        ).hexdigest(),
                    },
                }
            )
        )

    def fake(self, args, **kwargs):
        self.assertNotIn("--insecure", args)
        self.assertNotIn("-k", args)
        self.assertNotIn("--location", args)
        self.assertIn("--max-filesize", args)
        self.assertEqual(args[args.index("--proto") + 1], "=https")
        url = args[-1]
        fmt = "pdf" if url.endswith(".pdf") else "xml"
        raw = (
            b"%PDF-1.4\nsynthetic"
            if fmt == "pdf"
            else b'<Auction hidden="SECRET"><Amount>938383938383</Amount></Auction>'
        )
        Path(args[args.index("--output") + 1]).write_bytes(raw)
        Path(args[args.index("--dump-header") + 1]).write_text("HTTP/2 200\n")
        return SimpleNamespace(
            returncode=0, stdout=f"200\n{url}\napplication/{fmt}", stderr=""
        )

    def test_capture_without_numerical_export_and_refuse_replay(self):
        with patch(
            "src.treasury_document_acquire.subprocess.run", side_effect=self.fake
        ) as call:
            result = capture_documents(self.root, self.selection, self.checkpoint)
            self.assertEqual(result["status"], "CAPTURED_SCHEMA_ONLY")
            self.assertEqual(len(result["documents"]), 3)
            self.assertFalse(result["numerical_source_admitted"])
            self.assertEqual(call.call_count, 3)
            for item in result["documents"]:
                self.assertEqual(item["status"], "VERIFIED_TRANSPORT_SCHEMA_ONLY")
                if item["format"] == "xml":
                    schema = json.loads((self.root / item["schema_path"]).read_text())
                    self.assertIn("Auction/Amount", schema["tag_counts"])
                    self.assertNotIn("938383938383", json.dumps(schema))
                    self.assertNotIn("SECRET", json.dumps(schema))
            with self.assertRaises(FileExistsError):
                capture_documents(self.root, self.selection, self.checkpoint)
            self.assertEqual(call.call_count, 3)

    def test_pins_validated_before_transport(self):
        self.pin.write_text("changed")
        with patch("src.treasury_document_acquire.subprocess.run") as call:
            with self.assertRaises(ValueError):
                capture_documents(self.root, self.selection, self.checkpoint)
            call.assert_not_called()

    def test_failed_document_keeps_receipt_and_other_documents(self):
        def bad(args, **kwargs):
            result = self.fake(args, **kwargs)
            if args[-1].endswith(".xml"):
                result.stdout = result.stdout.replace("200\n", "404\n", 1)
            return result

        with patch("src.treasury_document_acquire.subprocess.run", side_effect=bad) as call:
            result = capture_documents(self.root, self.selection, self.checkpoint)
            self.assertEqual(call.call_count, 3)
        self.assertEqual(result["status"], "DOCUMENTS_REQUIRE_REVIEW")
        bads = [d for d in result["documents"] if d["status"] == "REQUIRES_REVIEW"]
        self.assertEqual(len(bads), 1)
        self.assertTrue((self.root / bads[0]["receipt_path"]).is_file())
        self.assertNotIn("schema_path", bads[0])

    def test_timeout_preserves_a_terminal_failure(self):
        import subprocess

        with patch(
            "src.treasury_document_acquire.subprocess.run",
            side_effect=subprocess.TimeoutExpired("curl", 50),
        ):
            result = capture_documents(self.root, self.selection, self.checkpoint)
        self.assertEqual(result["status"], "DOCUMENTS_REQUIRE_REVIEW")
        self.assertTrue(all(d["status"] == "REQUIRES_REVIEW" for d in result["documents"]))
        self.assertTrue((self.rep / "TREASURY_DOCUMENT_CAPTURE.json").is_file())


if __name__ == "__main__":
    unittest.main()
