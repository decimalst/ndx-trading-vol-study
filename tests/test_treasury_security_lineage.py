"""Generated date-lineage contracts before application to source documents."""

import unittest

from src.treasury_security_lineage import reconcile_lineage

EVENT = {"auction_date": "2016-01-13", "announcement_date": "2016-01-07", "cusip": "AB1234567"}


def fixture(reopened=True):
    a = {
        "CUSIP": EVENT["cusip"],
        "AuctionDate": EVENT["auction_date"],
        "AnnouncementDate": EVENT["announcement_date"],
        "SecurityType": "NOTE",
        "SecurityTermWeekYear": "9-YEAR" if reopened else "3-YEAR",
        "SecurityTermDayMonth": "10-MONTH" if reopened else "0-MONTH",
        "InflationIndexSecurity": "N",
        "FloatingRate": "N",
        "ReOpeningIndicator": "Y" if reopened else "N",
        "IssueDate": "2016-01-15",
        "MaturityDate": "2025-11-15" if reopened else "2019-01-15",
        "DatedDate": "2015-11-15" if reopened else "2016-01-15",
        "OriginalDatedDate": "2015-11-15" if reopened else "",
        "OriginalIssueDate": "2015-11-16" if reopened else "",
    }
    body = "".join(f"<{k}>{v}</{k}>" for k, v in a.items())
    raw = (
        '<t:AuctionData xmlns:t="http://www.treasurydirect.gov/">'
        f"<AuctionAnnouncement>{body}</AuctionAnnouncement><AuctionResults/></t:AuctionData>"
    ).encode()
    term = "9-Year 10-Month Note" if reopened else "3-Year Note"
    pdf = (
        "For Immediate Release CONTACT: Treasury\nJanuary 13, 2016\n"
        f"TREASURY AUCTION RESULTS\nTerm and Type of Security {term}\n"
        "CUSIP Number AB1234567\nIssue Date January 15, 2016\n"
        + (
            "Maturity Date November 15, 2025\nOriginal Issue Date November 16, 2015\n"
            "Dated Date November 15, 2015\n"
            if reopened
            else "Maturity Date January 15, 2019\nOriginal Issue Date January 15, 2016\n"
            "Dated Date January 15, 2016\n"
        )
    )
    return raw, pdf


def parse(pair):
    return reconcile_lineage(*pair, **EVENT)


class TreasurySecurityLineageTests(unittest.TestCase):
    def test_reopening_uses_stated_original_dates_not_rounded_remaining_term(self):
        result = parse(fixture())
        self.assertEqual(result["original_tenor_years"], 10)
        self.assertEqual(result["remaining_term"], "9-Year 10-Month Note")
        self.assertEqual(result["dated_date"], "2015-11-15")
        self.assertTrue(result["reopening"])

    def test_new_issue_and_weekend_issue_date_do_not_change_dated_tenor(self):
        raw, pdf = fixture(False)
        self.assertEqual(parse((raw, pdf))["original_tenor_years"], 3)
        raw = raw.replace(b"<IssueDate>2016-01-15", b"<IssueDate>2016-01-18")
        pdf = pdf.replace("Issue Date January 15, 2016", "Issue Date January 18, 2016")
        self.assertEqual(parse((raw, pdf))["original_tenor_years"], 3)

    def test_actual_five_year_reopening_is_not_reclassified_as_two_year(self):
        raw, pdf = fixture()
        raw = raw.replace(b"9-YEAR", b"2-YEAR").replace(b"10-MONTH", b"0-MONTH")
        raw = raw.replace(b"2015-11-15", b"2013-01-31")
        raw = raw.replace(b"2015-11-16", b"2013-01-31")
        raw = raw.replace(b"2025-11-15", b"2018-01-31")
        pdf = pdf.replace("9-Year 10-Month Note", "2-Year Note")
        pdf = pdf.replace("November 15, 2015", "January 31, 2013")
        pdf = pdf.replace("November 16, 2015", "January 31, 2013")
        pdf = pdf.replace("November 15, 2025", "January 31, 2018")
        self.assertEqual(parse((raw, pdf))["original_tenor_years"], 5)

    def test_metadata_conflicts_missing_dates_and_duplicate_aliases_fail(self):
        raw, pdf = fixture()
        for changed in (
            raw.replace(b"<DatedDate>2015-11-15</DatedDate>", b""),
            raw.replace(b"<DatedDate>", b"<t:DatedDate>").replace(
                b"</DatedDate>", b"</t:DatedDate>"
            ),
            raw.replace(
                b"<AuctionResults/>",
                b"<AuctionResults><DatedDate>2015-11-15</DatedDate></AuctionResults>",
            ),
            raw.replace(
                b"</AuctionAnnouncement>",
                b"<DatedDate>2015-11-15</DatedDate></AuctionAnnouncement>",
            ),
            raw.replace(b"<OriginalDatedDate>2015-11-15", b"<OriginalDatedDate>2015-11-16"),
            raw.replace(b"<OriginalIssueDate>2015-11-16", b"<OriginalIssueDate>2017-01-15"),
        ):
            with self.subTest(changed=changed):
                self.assertNotEqual(changed, raw)
                with self.assertRaises(ValueError):
                    parse((changed, pdf))

    def test_pdf_date_type_term_identity_and_role_conflicts_fail(self):
        raw, pdf = fixture()
        for changed in (
            pdf.replace("CUSIP Number AB1234567", "CUSIP Number XY1234567"),
            pdf.replace("January 13, 2016", "January 14, 2016"),
            pdf.replace("TREASURY AUCTION RESULTS", "TREASURY OFFERING ANNOUNCEMENT"),
            pdf.replace("9-Year 10-Month Note", "9-Year 11-Month Note"),
            pdf.replace("9-Year 10-Month Note", "9-Year 10-Month Bond"),
            pdf.replace("Dated Date November 15, 2015", "Dated Date November 16, 2015"),
            pdf + "Dated Date November 15, 2015\n",
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                parse((raw, changed))

    def test_unknown_or_nonexact_original_tenor_is_not_rounded(self):
        raw, pdf = fixture()
        for y, d in (("2024", "15"), ("2025", "16")):
            changed = raw.replace(b"2025-11-15", f"{y}-11-{d}".encode())
            text = pdf.replace("November 15, 2025", f"November {d}, {y}")
            with self.subTest(y=y, d=d), self.assertRaises(ValueError):
                parse((changed, text))

    def test_new_issue_must_have_whole_original_term_and_consistent_original_fields(self):
        raw, pdf = fixture(False)
        for changed in (
            raw.replace(
                b"<OriginalIssueDate></OriginalIssueDate>",
                b"<OriginalIssueDate>2015-01-15</OriginalIssueDate>",
            ),
            raw.replace(b"<ReOpeningIndicator>N", b"<ReOpeningIndicator>"),
            raw.replace(b"3-YEAR", b"2-YEAR"),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                parse((changed, pdf))

    def test_no_financial_conversion_required_and_post_ceiling_auction_fails(self):
        raw, pdf = fixture()
        raw = raw.replace(
            b"<AuctionResults/>",
            b"<AuctionResults><HighYield>NaN</HighYield></AuctionResults>",
        )
        self.assertEqual(parse((raw, pdf))["original_tenor_years"], 10)
        with self.assertRaises(ValueError):
            reconcile_lineage(raw, pdf, **(EVENT | {"auction_date": "2025-10-21"}))

    def test_issue_cannot_precede_selected_auction_even_when_formats_agree(self):
        raw, pdf = fixture(False)
        raw = raw.replace(b"2016-01-15", b"2016-01-12").replace(b"2019-01-15", b"2019-01-12")
        pdf = pdf.replace("January 15, 2016", "January 12, 2016").replace(
            "January 15, 2019", "January 12, 2019"
        )
        with self.assertRaises(ValueError):
            parse((raw, pdf))


if __name__ == "__main__":
    unittest.main()
