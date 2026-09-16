"""Generated network substitute checks before any real annual capture."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from scripts import capture_treasury_inventory_metadata as capture


class CaptureIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reports = self.root / "reports/next_signal_review"
        self.reports.mkdir(parents=True)
        self.private = self.root / "private/capture"
        old = self.root / "reports/commodity_implied/predictive"
        old.mkdir(parents=True)
        (old / "freeze_record.json").write_text(json.dumps({"code": {}}))
        (self.root / "pinned.txt").write_text("before request")
        self.checkpoint = {
            "status": "READY_METADATA_CAPTURE",
            "pins": {"pinned.txt": hashlib.sha256(b"before request").hexdigest()},
        }
        (self.reports / "TREASURY_INVENTORY_CHECKPOINT.json").write_text(
            json.dumps(self.checkpoint)
        )
        from src.treasury_auction_capture import annual_requests

        legacy = self.root / "data/source_discovery/treasury_auction/archive_capture_v1/2010"
        legacy.mkdir(parents=True)
        raw = json.dumps(
            [
                dict(
                    a="912ABC123",
                    d="56-Day",
                    z3a="CMB",
                    t3a="",
                    h="2010-01-01",
                    i="2010-01-02",
                )
            ]
        ).encode()
        (legacy / "response.json").write_bytes(raw)
        (legacy / "headers.txt").write_text("HTTP/2 200\n")
        (legacy / "transport_stderr.txt").write_text("")
        (legacy / "receipt.json").write_text(
            json.dumps(
                dict(
                    requested_url=annual_requests()[0][2],
                    effective_url=annual_requests()[0][2],
                    http_status=200,
                    curl_exit=0,
                    content_type="application/json",
                    bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    retrieved_utc="2026-09-08T23:43:38+00:00",
                )
            )
        )
        self.context = patch.multiple(
            capture, ROOT=self.root, REPORTS=self.reports, PRIVATE=self.private
        )
        self.context.start()
        self.addCleanup(self.context.stop)

    def fake_http(self, args, **kwargs):
        self.assertEqual(args[0], "curl")
        self.assertNotIn("-k", args)
        self.assertNotIn("--insecure", args)
        self.assertNotIn("--location", args)
        self.assertEqual(args[args.index("--proto") + 1], "=https")
        self.assertLessEqual(int(args[args.index("--max-time") + 1]), 45)
        self.assertLessEqual(kwargs["timeout"], 50)
        url = args[-1]
        query = parse_qs(urlsplit(url).query)
        self.assertLessEqual(query["endDate"][0], "2025-10-20")
        day = query["startDate"][0]
        body = json.dumps(
            [
                dict(
                    a="912ABC123",
                    d="2-Year",
                    z3a="Note",
                    t3a="2-Year",
                    h=day,
                    i=day,
                    e3="announcement.pdf",
                    f3="result.pdf",
                    ignored_unadmitted_amount="938373938393938",
                )
            ]
        ).encode()
        Path(args[args.index("--output") + 1]).write_bytes(body)
        Path(args[args.index("--dump-header") + 1]).write_text("HTTP/2 200\n")
        return SimpleNamespace(stdout=f"200\n{url}\napplication/json", stderr="", returncode=0)

    def test_complete_capture_then_refuses_replay(self):
        with patch.object(capture.subprocess, "run", side_effect=self.fake_http) as net:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(capture.run(), 0)
            self.assertEqual(net.call_count, 15)
            result = json.loads((self.reports / "TREASURY_INVENTORY_CAPTURE.json").read_text())
            self.assertEqual(result["status"], "VERIFIED_METADATA_INVENTORY")
            self.assertFalse(result["numerical_source_admitted"])
            self.assertEqual(result["predictive_comparisons_added"], 0)
            for path in self.private.glob("*/metadata.json"):
                self.assertNotIn("938373938393938", path.read_text())
                self.assertNotIn("ignored_unadmitted_amount", path.read_text())
            with self.assertRaises(FileExistsError):
                capture.run()
            self.assertEqual(net.call_count, 15)

    def test_checkpoint_drift_prevents_any_network_or_new_capture(self):
        (self.root / "pinned.txt").write_text("changed")
        with patch.object(capture.subprocess, "run") as net:
            with self.assertRaises(ValueError):
                capture.run()
            net.assert_not_called()
        self.assertFalse(self.private.exists())

    def test_transport_failure_is_saved_and_stops_without_retry(self):
        def bad_http(args, **kwargs):
            result = self.fake_http(args, **kwargs)
            result.stdout = result.stdout.replace("200\n", "404\n", 1)
            return result

        with patch.object(capture.subprocess, "run", side_effect=bad_http) as net:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(capture.run(), 2)
            self.assertEqual(net.call_count, 1)
        result = json.loads((self.reports / "TREASURY_INVENTORY_CAPTURE.json").read_text())
        self.assertEqual(result["status"], "REQUIRES_SOURCE_REVIEW")
        self.assertTrue((self.private / "2011/receipt.json").is_file())
        self.assertFalse((self.private / "2011/metadata.json").exists())

    def test_existing_partial_capture_is_not_replaced(self):
        self.private.mkdir(parents=True)
        marker = self.private / "interrupted.txt"
        marker.write_text("preserve")
        with patch.object(capture.subprocess, "run") as net:
            with self.assertRaises(FileExistsError):
                capture.run()
            net.assert_not_called()
        self.assertEqual(marker.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
