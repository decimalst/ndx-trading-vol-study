"""Generated DOL index contracts; no historical files or network are used."""

import hashlib
import unittest
from unittest.mock import patch

from src import claims_release_archive as archive


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def page(*hrefs, year=2025):
    links = "".join(f'<a href="{href}">Release</a>' for href in hrefs)
    return f"<!doctype html><html><body><b>{year}</b>{links}</body></html>".encode()


class ClaimsReleaseArchiveTests(unittest.TestCase):
    def parse(self, payload=None, year=2025, ceiling="2025-10-20"):
        payload = page("/press/2025/010825.pdf") if payload is None else payload
        return archive.parse_archive_index(payload, year, digest(payload), ceiling)

    def test_exact_metadata_schema_keeps_wednesday_and_orders_dates(self):
        payload = page("/press/2025/061825.pdf", "/press/2025/010825.pdf")
        self.assertEqual(self.parse(payload), {
            "status": "METADATA_ONLY_NOT_RELEASE_RECONCILED",
            "year": 2025,
            "source_ceiling": "2025-10-20",
            "source_sha256": digest(payload),
            "records": [
                {"release_date": "2025-01-08", "release_url": "https://oui.doleta.gov/press/2025/010825.pdf"},
                {"release_date": "2025-06-18", "release_url": "https://oui.doleta.gov/press/2025/061825.pdf"},
            ],
            "excluded_postcutoff": [],
            "duplicate_links_removed": 0,
        })

    def test_actual_legacy_asp_pattern_is_preserved(self):
        result = self.parse(page("../press/2009/052809.asp", year=2009), year=2009)
        self.assertEqual(result["records"], [{"release_date": "2009-05-28", "release_url": "https://oui.doleta.gov/press/2009/052809.asp"}])

    def test_hash_failure_precedes_html_decoding(self):
        payload = page("/press/2025/010825.pdf")
        with patch.object(archive, "_decode_html", side_effect=AssertionError("early decode")) as decode:
            with self.assertRaisesRegex(ValueError, "hash"):
                archive.parse_archive_index(payload, 2025, "0" * 64)
            decode.assert_not_called()

    def test_typed_hash_payload_year_and_ceiling_contracts(self):
        payload = page("/press/2025/010825.pdf")
        for signature in (None, "bad", "A" * 64, 42):
            with self.subTest(signature=signature), self.assertRaises(ValueError):
                archive.parse_archive_index(payload, 2025, signature)
        for year in (True, "2025", 2025.0, 1899, 2100):
            with self.subTest(year=year), self.assertRaises(ValueError):
                self.parse(payload, year=year)
        for ceiling in (None, "20251020", "2025-02-30", "2025-10-21", "2025-01-08", "2025-1-08"):
            with self.subTest(ceiling=ceiling), self.assertRaises(ValueError):
                self.parse(payload, ceiling=ceiling)
        with self.assertRaises(ValueError):
            archive.parse_archive_index(bytearray(payload), 2025, digest(payload))

    def test_postcutoff_links_are_separate_even_when_all_links_are_late(self):
        payload = page("/press/2025/112025.pdf", "/press/2025/092525.pdf", "/press/2025/102025.pdf")
        result = self.parse(payload)
        self.assertEqual([r["release_date"] for r in result["records"]], ["2025-09-25", "2025-10-20"])
        self.assertEqual(result["excluded_postcutoff"], [{"release_date": "2025-11-20", "release_url": "https://oui.doleta.gov/press/2025/112025.pdf", "reason": "after_source_ceiling"}])
        late = self.parse(page("/press/2025/112025.pdf"))
        self.assertEqual(late["records"], [])
        self.assertEqual(len(late["excluded_postcutoff"]), 1)

    def test_equivalent_relative_http_https_links_deduplicate(self):
        payload = page("/press/2025/010825.pdf", "../press/2025/010825.pdf", "http://oui.doleta.gov/press/2025/010825.pdf", "https://oui.doleta.gov/press/2025/010825.pdf")
        result = self.parse(payload)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["duplicate_links_removed"], 3)

    def test_competing_same_date_paths_rejected_including_postcutoff(self):
        for stem in ("010825", "112025"):
            with self.subTest(stem=stem), self.assertRaisesRegex(ValueError, "competing"):
                self.parse(page(f"/press/2025/{stem}.asp", f"/press/2025/{stem}.pdf"))

    def test_invalid_calendar_dates_and_filename_years_rejected(self):
        for name in ("023025", "022925", "130125", "000125", "013225", "010824", "01082025", "01082"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.parse(page(f"/press/2025/{name}.pdf"))

    def test_valid_leap_day_is_not_restricted_to_thursday(self):
        result = self.parse(page("/press/2020/022920.pdf", year=2020), year=2020)
        self.assertEqual(result["records"][0]["release_date"], "2020-02-29")

    def test_requested_year_must_match_all_release_paths(self):
        for path in ("/press/2024/010825.pdf", "/press/2024/010424.pdf"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "year"):
                self.parse(page("/press/2025/010825.pdf", path))

    def test_release_shaped_foreign_credential_port_or_unsafe_urls_rejected(self):
        for prefix in ("https://example.org", "https://oui.doleta.gov.evil.example", "https://user@oui.doleta.gov", "https://oui.doleta.gov:443", "ftp://oui.doleta.gov"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                self.parse(page(prefix + "/press/2025/010825.pdf"))

    def test_release_query_fragment_encoding_and_unknown_extension_rejected(self):
        for suffix in ("010825.pdf?x=1", "010825.pdf#page=1", "010825.txt", "010825.PDF", "%30%31%30%38%32%35.pdf"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                self.parse(page("/press/2025/" + suffix))

    def test_unrelated_navigation_scripts_and_text_are_ignored(self):
        payload = page("/unemploy/claims_arch.asp", "https://www.dol.gov/ui/data.pdf", "mailto:webmaster@example.org", "/press/", "/press/2025/")
        payload = payload.replace(b"</body>", b'<script>var fake = \'<a href="/press/2025/112025.pdf">\';</script><p>/press/2025/120425.pdf</p><a href="/press/2025/010825.pdf">Wednesday</a></body>')
        result = self.parse(payload)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["excluded_postcutoff"], [])

    def test_reject_empty_error_and_non_html_responses(self):
        for payload in (b"", b"%PDF-1.7", b"upstream error", page("/unemploy/claims_arch.asp"), b'<a href="/press/2025/010825.pdf">release</a>'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.parse(payload)

    def test_legacy_non_ascii_surrounding_text_does_not_change_ascii_identity(self):
        payload = page("/press/2025/010825.pdf").replace(b"<body>", b"<body>Agency\x92s archive ")
        self.assertEqual(self.parse(payload)["records"][0]["release_date"], "2025-01-08")

    def test_duplicate_href_attributes_are_rejected(self):
        payload = b'<html><a href="/press/2025/010825.pdf" href="/press/2025/112025.pdf">ambiguous</a></html>'
        with self.assertRaises(ValueError):
            self.parse(payload)

    def test_base_href_cannot_silently_rebase_relative_release_identity(self):
        payload = page("../press/2025/010825.pdf").replace(b"<body>", b'<head><base href="https://example.org/"></head><body>')
        with self.assertRaises(ValueError):
            self.parse(payload)

    def test_observed_bold_year_header_must_match_requested_year(self):
        with self.assertRaisesRegex(ValueError, "year header"):
            self.parse(page("/press/2025/010825.pdf", year=2024))

    def test_observed_bold_year_header_is_required(self):
        payload = page("/press/2025/010825.pdf").replace(b"<b>2025</b>", b"<p>2025</p>")
        with self.assertRaisesRegex(ValueError, "year header"):
            self.parse(payload)

    def test_conflicting_or_duplicate_bold_year_headers_rejected(self):
        for extra in (b"<b>2024</b>", b"<b>2025</b>"):
            payload = page("/press/2025/010825.pdf").replace(b"</body>", extra + b"</body>")
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "year header"):
                self.parse(payload)


if __name__ == "__main__":
    unittest.main()
