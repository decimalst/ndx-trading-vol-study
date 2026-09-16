"""Generated URL/receipt guards before reading Treasury document bodies."""

import copy
import hashlib
import unittest

from src.treasury_auction_documents import document_requests, validate_document_receipt


def selection():
    return {
        "records": [
            {
                "record": {
                    "auction_date": "2010-01-13",
                    "announcement_date": "2010-01-07",
                    "cusip": "912ABC123",
                    "documents": {
                        "announcement_pdf": [
                            "https://www.treasurydirect.gov/instit/annceresult/press/preanre/2010/A_20100107_1.pdf"
                        ],
                        "competitive_pdf": [
                            "https://www.treasurydirect.gov/instit/annceresult/press/preanre/2010/R_20100113_1.pdf"
                        ],
                        "noncompetitive_pdf": [],
                        "special_pdf": [],
                        "announcement_xml": [],
                        "competitive_xml": [
                            "https://www.treasurydirect.gov/xml/R_20100113_1.xml"
                        ],
                    },
                }
            }
        ]
    }


class TreasuryDocumentTests(unittest.TestCase):
    def test_only_selected_urls_deduplicated_with_all_memberships(self):
        s = selection()
        row = copy.deepcopy(s["records"][0])
        row["record"]["cusip"] = "912ABC124"
        s["records"].append(row)
        req = document_requests(s)
        self.assertEqual(len(req), 3)
        self.assertEqual([x["url"] for x in req], sorted(x["url"] for x in req))
        for item in req:
            self.assertEqual(set(item), {"url", "format", "memberships"})
            self.assertEqual(len(item["memberships"]), 2)
            self.assertEqual(set(item["memberships"][0]), {"auction_date", "cusip", "kind"})

    def test_official_host_path_year_and_literal_filenames(self):
        for url in [
            "http://www.treasurydirect.gov/xml/R_20100113_1.xml",
            "https://evil.test/xml/R_20100113_1.xml",
            "https://www.treasurydirect.gov/xml/../R_20100113_1.xml",
            "https://www.treasurydirect.gov/xml/R_20100113_1.xml?x=1",
            "https://www.treasurydirect.gov/xml/R_20100113_1.xml#x",
            "https://www.treasurydirect.gov/xml/\u212a.xml",
        ]:
            s = selection()
            s["records"][0]["record"]["documents"]["competitive_xml"] = [url]
            with self.subTest(url=url), self.assertRaises(ValueError):
                document_requests(s)
        s = selection()
        s["records"][0]["record"]["documents"]["competitive_pdf"][0] = s["records"][0][
            "record"
        ]["documents"]["competitive_pdf"][0].replace("/2010/", "/2011/")
        with self.assertRaises(ValueError):
            document_requests(s)

    def test_full_selection_date_scope_and_future_filename_guards(self):
        for key, value in [
            ("auction_date", "2025-10-21"),
            ("auction_date", "2009-12-31"),
            ("auction_date", "2010-02-30"),
            ("announcement_date", "2010-01-14"),
        ]:
            s = selection()
            s["records"][0]["record"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                document_requests(s)
        for name in ["R_20261021_1.xml", "R_20100230_1.xml"]:
            s = selection()
            s["records"][0]["record"]["documents"]["competitive_xml"] = [
                "https://www.treasurydirect.gov/xml/" + name
            ]
            with self.subTest(name=name), self.assertRaises(ValueError):
                document_requests(s)

    def test_schema_duplicates_and_absence_are_explicit(self):
        for change in ["empty", "duplicate", "unknownkind", "null", "badcusip"]:
            s = selection()
            if change == "empty":
                s["records"] = []
            if change == "duplicate":
                s["records"] *= 2
            if change == "unknownkind":
                s["records"][0]["record"]["documents"]["other"] = []
            if change == "null":
                s["records"][0]["record"]["documents"]["special_pdf"] = None
            if change == "badcusip":
                s["records"][0]["record"]["cusip"] = "912abc123"
            with self.subTest(change=change), self.assertRaises(ValueError):
                document_requests(s)

    def receipt(self, body=b"%PDF-1.4\nsynthetic", **changes):
        url = selection()["records"][0]["record"]["documents"]["competitive_pdf"][0]
        r = dict(
            requested_url=url,
            effective_url=url,
            http_status=200,
            curl_exit=0,
            content_type="application/pdf",
            bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            retrieved_utc="2026-09-09T00:00:00+00:00",
        )
        r.update(changes)
        return body, r, url

    def test_valid_pdf_and_xml_receipts_without_value_decoding(self):
        body, r, url = self.receipt()
        self.assertIsNone(validate_document_receipt(body, r, url, "pdf"))
        for mime in ["application/xml", "text/xml; charset=UTF-8"]:
            body, r, url = self.receipt(b"<opaque>999999999999999</opaque>", content_type=mime)
            self.assertIsNone(validate_document_receipt(body, r, url, "xml"))

    def test_invalid_receipts_fail_closed(self):
        for changes in [
            dict(http_status=404),
            dict(http_status=True),
            dict(curl_exit=60),
            dict(curl_exit=False),
            dict(bytes=0),
            dict(sha256="f" * 64),
            dict(effective_url="https://example.test"),
            dict(content_type="text/html"),
            dict(retrieved_utc="2026-09-09"),
        ]:
            body, r, url = self.receipt(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_document_receipt(body, r, url, "pdf")
        body, r, url = self.receipt(b"<html>error</html>")
        with self.assertRaises(ValueError):
            validate_document_receipt(body, r, url, "pdf")
        body, r, url = self.receipt()
        with self.assertRaises(ValueError):
            validate_document_receipt(body + b" ", r, url, "pdf")
        with self.assertRaises(ValueError):
            validate_document_receipt(body, dict(r, extra=1), url, "pdf")

    def test_document_body_limits_and_format_are_fixed(self):
        for fmt, body in [
            ("xml", b"x" * (2 * 1024 * 1024 + 1)),
            ("pdf", b"%PDF-" + b"x" * (8 * 1024 * 1024)),
            ("html", b"x"),
            ("pdf", b""),
        ]:
            body, r, url = self.receipt(body)
            with self.subTest(fmt=fmt), self.assertRaises(ValueError):
                validate_document_receipt(body, r, url, fmt)


if __name__ == "__main__":
    unittest.main()
