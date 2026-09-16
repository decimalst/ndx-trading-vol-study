"""Header-layout cases found by masked metadata inspection, before table review."""

import unittest

from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header


class HeaderLayoutTests(unittest.TestCase):
    def test_embargoed_offering_and_corpus_identifier_separated(self):
        text = "Footnote\nEmbargoed Until 11:00 A.M. CONTACT: Office\nJanuary 07, 2010\nTREASURY OFFERING ANNOUNCEMENT\nCUSIP Number 912ABC123\nCorpus CUSIP Number 912XYZ123"
        p = probe_pdf_header(text)
        self.assertEqual(
            p,
            {
                "has_release_label": True,
                "release_dates": ["2010-01-07"],
                "cusips": ["912ABC123"],
            },
        )
        self.assertEqual(
            check_pdf_identity(p, expected_date="2010-01-07", expected_cusip="912ABC123"),
            "2010-01-07",
        )

    def test_dotted_release_marker_and_later_body_date(self):
        p = probe_pdf_header(
            "CONTACT: Treasury FOR I.MMEDIATE RELEASE:\nJune 20, 2019\nCONTINGENCY AUCTION\nJune 21, 2019\nCUSIP Number 912ABC123"
        )
        self.assertEqual(p["release_dates"], ["2019-06-20"])
        self.assertEqual(check_pdf_identity(p), "2019-06-20")

    def test_body_dates_do_not_supply_absent_or_conflicting_header_date(self):
        for text in [
            "For Immediate Release\nTITLE\nJanuary 13, 2010",
            "For Immediate Release\nJanuary 13, 2010\nJanuary 14, 2010\nTITLE",
            "January 13, 2010\nTITLE",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                check_pdf_identity(probe_pdf_header(text))

    def test_existing_identity_conflict_is_not_repaired(self):
        p = probe_pdf_header(
            "Embargoed Until 11:00 A.M.\nJanuary 07, 2016\nTITLE\nCUSIP Number 912828RP5\nCorpus CUSIP Number 912803EQ2"
        )
        with self.assertRaises(ValueError):
            check_pdf_identity(p, expected_cusip="912810RP5")

    def test_source_cutoff_and_invalid_date_still_fail(self):
        for text in [
            "For Immediate Release\nOctober 21, 2025\nTITLE",
            "Embargoed Until 11:00 A.M.\nFebruary 30, 2020\nTITLE",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                check_pdf_identity(probe_pdf_header(text))

    def test_only_metadata_exported_with_known_future_maturity_body(self):
        p = probe_pdf_header(
            "For Immediate Release\nJanuary 13, 2010\nTITLE\nCUSIP Number 912ABC123\nMaturity Date January 13, 2040\nAmount 987654321"
        )
        self.assertEqual(set(p), {"has_release_label", "release_dates", "cusips"})
        self.assertNotIn("987654321", str(p))
        self.assertNotIn("2040", str(p))
        self.assertEqual(check_pdf_identity(p), "2010-01-13")


if __name__ == "__main__":
    unittest.main()
