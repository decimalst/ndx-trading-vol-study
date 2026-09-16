"""Generated contracts for metadata inspection before full-history source reads."""

import unittest

from src.treasury_history_inspection import inspect_history_payload

XML = b"""<a:AuctionData xmlns:a="http://www.treasurydirect.gov/">
<AuctionAnnouncement><CUSIP>912828AAA</CUSIP><AnnouncedCUSIP>912828BBB</AnnouncedCUSIP>
<AnnouncementDate>2017-04-20T00:00:00</AnnouncementDate>
<AuctionDate>2017-04-25T00:00:00</AuctionDate></AuctionAnnouncement>
<AuctionResults><ReleaseTime>13:00:00</ReleaseTime>
<PrimaryDealerAccepted>not-a-number</PrimaryDealerAccepted></AuctionResults>
</a:AuctionData>"""
PDF = """For Immediate Release
April 25, 2017
TREASURY AUCTION RESULTS
CUSIP 912828AAA
Primary Dealer 999,999 111,111
"""


class HistoryInspectionTests(unittest.TestCase):
    def test_xml_preserves_distinct_announced_and_final_identity_without_admitting_it(self):
        got = inspect_history_payload(XML, "xml")
        self.assertEqual(got["status"], "XML_METADATA_EXTRACTED")
        self.assertEqual(got["metadata"]["announcement"]["CUSIP"], "912828AAA")
        self.assertEqual(got["metadata"]["announcement"]["AnnouncedCUSIP"], "912828BBB")
        self.assertNotIn("PrimaryDealerAccepted", str(got))
        self.assertFalse(got["identity_reconciled"])
        self.assertFalse(got["source_admitted"])

    def test_valid_pdf_header_reports_date_and_cusip_only(self):
        got = inspect_history_payload(b"%PDF-synthetic", "pdf", pdf_text=PDF)
        self.assertEqual(got["status"], "PDF_DATED_HEADER_EXTRACTED")
        self.assertEqual(got["release_date"], "2017-04-25")
        self.assertEqual(got["metadata"]["cusips"], ["912828AAA"])
        self.assertNotIn("999", str(got))
        self.assertFalse(got["identity_reconciled"])

    def test_bad_header_is_retained_as_failed_evidence(self):
        got = inspect_history_payload(
            b"%PDF-synthetic",
            "pdf",
            pdf_text=PDF.replace("For Immediate Release", "Auction details"),
        )
        self.assertEqual(got["status"], "PDF_HEADER_REQUIRES_REVIEW")
        self.assertEqual(got["metadata"]["cusips"], ["912828AAA"])
        self.assertNotIn("release_date", got)
        self.assertIn("error", got)

    def test_post_ceiling_header_cannot_be_a_supported_release(self):
        got = inspect_history_payload(
            b"%PDF-synthetic",
            "pdf",
            pdf_text=PDF.replace("April 25, 2017", "November 3, 2025"),
        )
        self.assertEqual(got["status"], "PDF_HEADER_REQUIRES_REVIEW")
        self.assertNotIn("release_date", got)

    def test_duplicate_xml_identity_leaves_fail_without_dropping_record(self):
        bad = XML.replace(
            b"<CUSIP>912828AAA</CUSIP>", b"<CUSIP>912828AAA</CUSIP><CUSIP>912828CCC</CUSIP>"
        )
        got = inspect_history_payload(bad, "xml")
        self.assertEqual(got["status"], "XML_METADATA_REQUIRES_REVIEW")
        self.assertIn("Duplicate", got["error"])
        self.assertFalse(got["source_admitted"])

    def test_body_format_and_required_extracted_text_fail_closed(self):
        for raw, fmt, text in [
            (b"html", "pdf", PDF),
            (b"%PDF-synthetic", "pdf", None),
            (XML, "json", None),
            ("xml", "xml", None),
        ]:
            with self.subTest(fmt=fmt, raw=raw), self.assertRaises(ValueError):
                inspect_history_payload(raw, fmt, pdf_text=text)


if __name__ == "__main__":
    unittest.main()
