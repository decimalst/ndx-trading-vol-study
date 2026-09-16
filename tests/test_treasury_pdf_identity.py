"""Generated PDF text-header checks before viewing source numerical tables."""

import unittest

from src.treasury_pdf_identity import check_pdf_identity, probe_pdf_header


class PDFIdentityTests(unittest.TestCase):
    def test_header_exports_only_dates_cusips_and_release_marker(self):
        text = "For Immediate Release CONTACT: SECRET\nJanuary 13, 2010 202-000-0000\nTreasury Auction Results\nCUSIP\n912ABC123\nHigh Yield 938383938\nAmount $9999999999"
        self.assertEqual(
            probe_pdf_header(text),
            {
                "has_release_label": True,
                "release_dates": ["2010-01-13"],
                "cusips": ["912ABC123"],
            },
        )

    def test_case_whitespace_and_repeated_identity(self):
        p = probe_pdf_header(
            "FOR IMMEDIATE RELEASE\n February 29, 2020\nCUSIP: 912ABC123\nCUSIP Number 912ABC123"
        )
        self.assertEqual(p["cusips"], ["912ABC123"])
        self.assertEqual(
            check_pdf_identity(p, expected_date="2020-02-29", expected_cusip="912ABC123"),
            "2020-02-29",
        )

    def test_body_dates_do_not_supply_missing_header(self):
        text = "For Immediate Release\n" + "body\n" * 12 + "January 13, 2010\nCUSIP 912ABC123"
        self.assertEqual(probe_pdf_header(text)["release_dates"], [])
        with self.assertRaises(ValueError):
            check_pdf_identity(probe_pdf_header(text))

    def test_invalid_dates_and_nonstring_input(self):
        for text in [
            None,
            b"text",
            "For Immediate Release\nFebruary 30, 2020",
            "For Immediate Release\nFebruary 29, 2019",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                probe_pdf_header(text)

    def test_unique_date_release_label_and_scope_required(self):
        for text in [
            "January 13, 2010",
            "For Immediate Release\nJanuary 13, 2010\nJanuary 14, 2010",
            "For Immediate Release\nOctober 21, 2025",
            "For Immediate Release\nDecember 31, 2009",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                check_pdf_identity(probe_pdf_header(text))

    def test_standard_document_date_and_identity_must_match(self):
        p = probe_pdf_header("For Immediate Release\nJanuary 13, 2010\nCUSIP 912ABC123")
        for d, c in [("2010-01-14", "912ABC123"), ("2010-01-13", "912ABC124")]:
            with self.subTest(d=d, c=c), self.assertRaises(ValueError):
                check_pdf_identity(p, expected_date=d, expected_cusip=c)
        p["cusips"].append("912ABC124")
        with self.assertRaises(ValueError):
            check_pdf_identity(p, expected_cusip="912ABC123")

    def test_special_notice_date_can_be_checked_without_assumed_event_date(self):
        p = probe_pdf_header("For Immediate Release\nJune 25, 2010\nSeveral offerings changed")
        self.assertEqual(check_pdf_identity(p), "2010-06-25")
        with self.assertRaises(ValueError):
            check_pdf_identity(p, expected_cusip="912ABC123")

    def test_lowercase_and_long_cusip_substrings_are_not_accepted(self):
        p = probe_pdf_header(
            "For Immediate Release\nJanuary 13, 2010\nCUSIP 912abc123\nCUSIP 912ABC1234"
        )
        self.assertEqual(p["cusips"], [])


if __name__ == "__main__":
    unittest.main()
