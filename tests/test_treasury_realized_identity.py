"""Prewritten generated identity reconciliation contracts; no source histories."""

import copy
import unittest
from datetime import date

from src.treasury_realized_identity import UnsupportedIdentity, reconcile_realized_identity
from src.treasury_xml_identity import check_event_identity, read_xml_identity

ANNOUNCED = "AB1234567"
ACTUAL = "XY1234567"
CONTEXT = {"auction_date": "2017-01-13", "announcement_date": "2017-01-07"}


def pretty(value):
    d = date.fromisoformat(value)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def xml(a, result=False, extra=""):
    leaves = "".join(f"<{k}>{v}</{k}>" for k, v in a.items())
    results = (
        "<AuctionResults><OriginalCUSIP></OriginalCUSIP>"
        "<ReleaseTime>13:02</ReleaseTime><ResultsPDFName>R_fixture.pdf</ResultsPDFName>"
        f"{extra}</AuctionResults>"
        if result
        else ""
    )
    return (
        '<t:AuctionData xmlns:t="http://www.treasurydirect.gov/">'
        f"<AuctionAnnouncement>{leaves}</AuctionAnnouncement>{results}</t:AuctionData>"
    ).encode()


def pdf(a, *, result, series):
    return (
        "For Immediate Release CONTACT: Treasury\n"
        + pretty(CONTEXT["auction_date"] if result else CONTEXT["announcement_date"])
        + "\n"
        + ("TREASURY AUCTION RESULTS" if result else "TREASURY OFFERING ANNOUNCEMENT")
        + "\nTerm and Type of Security 2-Year Note\n"
        + f"CUSIP Number {a['CUSIP']}\nSeries {series}\n"
        + (f"Auction Date {pretty(a['AuctionDate'])}\n" if not result else "")
        + f"Issue Date {pretty(a['IssueDate'])}\n"
        + f"Maturity Date {pretty(a['MaturityDate'])}\n"
        + f"Original Issue Date {pretty(a['OriginalIssueDate'] or a['IssueDate'])}\n"
        + f"Dated Date {pretty(a['DatedDate'])}\n"
    )


def fixture(changed=False):
    announced = {
        "CUSIP": ANNOUNCED,
        "AnnouncedCUSIP": "",
        "AnnouncementDate": CONTEXT["announcement_date"],
        "AuctionDate": CONTEXT["auction_date"],
        "IssueDate": "2017-01-17",
        "DatedDate": "2017-01-15",
        "MaturityDate": "2019-01-15",
        "OriginalIssueDate": "",
        "OriginalDatedDate": "",
        "SecurityType": "NOTE",
        "SecurityTermWeekYear": "2-YEAR",
        "SecurityTermDayMonth": "0-MONTH",
        "ReOpeningIndicator": "N",
        "InflationIndexSecurity": "N",
        "FloatingRate": "N",
        "TypeOfAuction": "SINGLE PRICE",
        "AnnouncementPDFName": "A_fixture.pdf",
    }
    final = announced.copy()
    if changed:
        final.update(
            CUSIP=ACTUAL,
            AnnouncedCUSIP=ANNOUNCED,
            ReOpeningIndicator="Y",
            DatedDate="2012-01-15",
            OriginalDatedDate="2012-01-15",
            OriginalIssueDate="2012-01-17",
        )
    confirmation = {
        "kind": "REALIZED_REOPENING",
        "url": "https://www.treasurydirect.gov/instit/annceresult/press/preanre/2017/BPD_SPL_20170113_1.pdf",
        "body_sha256": "a" * 64,
        "text_sha256": "b" * 64,
        "release_date": CONTEXT["auction_date"],
        "auction_date": CONTEXT["auction_date"],
        "announcement_date": CONTEXT["announcement_date"],
        "offered_term": "2-Year Note",
        "actual_cusip": ACTUAL,
        "original_term_years": 7,
        "original_issue_date": "2012-01-17",
        "series": "K-2019",
    }
    return {
        "args": [
            xml(announced),
            pdf(announced, result=False, series="U-2019"),
            xml(final, True),
            pdf(final, result=True, series="K-2019" if changed else "U-2019"),
        ],
        "kwargs": CONTEXT
        | {
            "selected_final_cusip": ACTUAL if changed else ANNOUNCED,
            "confirmation": confirmation if changed else None,
        },
    }


def run(f):
    return reconcile_realized_identity(*f["args"], **f["kwargs"])


class TreasuryRealizedIdentityTests(unittest.TestCase):
    def test_unchanged_identity_needs_no_confirmation_and_keeps_settlement_shift(self):
        f = fixture()
        before = copy.deepcopy(f)
        out = run(f)
        self.assertEqual(out["status"], "RECONCILED_METADATA_ONLY")
        self.assertEqual(out["announced_cusip"], ANNOUNCED)
        self.assertEqual(out["actual_cusip"], ANNOUNCED)
        self.assertFalse(out["substitution"])
        self.assertIsNone(out["confirmation"])
        self.assertEqual(out["identity_available_date"], "2017-01-13")
        self.assertEqual(out["final_lineage"]["dated_date"], "2017-01-15")
        self.assertEqual(out["final_lineage"]["issue_date"], "2017-01-17")
        self.assertEqual(f, before)

    def test_realized_identity_can_differ_from_warning_without_rewriting_frozen_input(self):
        f = fixture(True)
        f["kwargs"]["conditional_alternatives"] = ("CD1234567",)
        before = copy.deepcopy(f)
        out = run(f)
        self.assertTrue(out["substitution"])
        self.assertEqual((out["announced_cusip"], out["actual_cusip"]), (ANNOUNCED, ACTUAL))
        self.assertEqual(out["announcement_lineage"]["original_tenor_years"], 2)
        self.assertEqual(out["final_lineage"]["original_tenor_years"], 7)
        self.assertEqual(out["conditional_alternatives"], ["CD1234567"])
        self.assertFalse(out["confirmation_provenance_verified"])
        self.assertEqual(f, before)
        with self.assertRaises(ValueError):
            check_event_identity(read_xml_identity(f["args"][2]), **CONTEXT, cusip=ACTUAL)

    def test_actual_substitution_requires_realized_confirmation_not_warning(self):
        for value in (None, {"kind": "CONDITIONAL_REOPENING"}):
            f = fixture(True)
            f["kwargs"].update(confirmation=value, conditional_alternatives=(ACTUAL,))
            with self.subTest(value=value), self.assertRaises(ValueError):
                run(f)

    def test_announcement_typo_and_final_pdf_identity_conflict_are_not_aliases(self):
        for index, old, new in ((1, ANNOUNCED, "ZZ1234567"), (3, ACTUAL, ANNOUNCED)):
            f = fixture(True)
            f["args"][index] = f["args"][index].replace(old, new)
            with self.subTest(index=index), self.assertRaises(ValueError):
                run(f)

    def test_announcement_pdf_auction_date_must_match_selected_and_xml_date(self):
        f = fixture(True)
        self.assertEqual(run(f)["auction_date"], CONTEXT["auction_date"])
        f["args"][1] = f["args"][1].replace(
            "Auction Date January 13, 2017", "Auction Date January 12, 2017"
        )
        with self.assertRaisesRegex(ValueError, "auction date"):
            run(f)

    def test_rescheduled_announcement_context_stays_explicitly_unsupported(self):
        f = fixture()
        f["args"][0] = f["args"][0].replace(b"2017-01-07", b"2017-01-10")
        f["args"][1] = f["args"][1].replace("January 7, 2017", "January 10, 2017")
        with self.assertRaises(UnsupportedIdentity) as caught:
            run(f)
        self.assertEqual(
            caught.exception.audit["status"], "UNSUPPORTED_ANNOUNCEMENT_DATE_CONTEXT"
        )
        self.assertEqual(caught.exception.audit["archive_announcement_date"], "2017-01-07")
        self.assertEqual(caught.exception.audit["source_announcement_date"], "2017-01-10")

    def test_final_pair_offered_context_and_document_roles_must_agree(self):
        mutations = [
            (2, b"<AnnouncedCUSIP>AB1234567", b"<AnnouncedCUSIP>CD1234567"),
            (2, b"<AuctionDate>2017-01-13", b"<AuctionDate>2017-01-12"),
            (2, b"<SecurityTermWeekYear>2-YEAR", b"<SecurityTermWeekYear>3-YEAR"),
            (3, "2-Year Note", "3-Year Note"),
            (3, "TREASURY AUCTION RESULTS", "TREASURY AUCTION NONCOMPETITIVE RESULTS"),
            (1, "TREASURY OFFERING ANNOUNCEMENT", "TREASURY AUCTION RESULTS"),
            (2, b"<FloatingRate>N", b"<FloatingRate>Y"),
        ]
        for index, old, new in mutations:
            f = fixture(True)
            f["args"][index] = f["args"][index].replace(old, new)
            with self.subTest(index=index, old=old), self.assertRaises(ValueError):
                run(f)

    def test_confirmation_context_lineage_and_provenance_schema_are_checked(self):
        mutations = {
            "actual_cusip": "CD1234567",
            "auction_date": "2017-01-12",
            "announcement_date": "2017-01-06",
            "offered_term": "3-Year Note",
            "original_term_years": 5,
            "original_issue_date": "2012-01-18",
            "series": "WRONG",
            "release_date": "2017-01-12",
            "url": "https://evil.example/BPD_SPL_20170113_1.pdf",
            "body_sha256": "a" * 63,
            "text_sha256": "B" * 64,
        }
        for key, value in mutations.items():
            f = fixture(True)
            f["kwargs"]["confirmation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                run(f)
        f = fixture(True)
        f["kwargs"]["confirmation"]["original_term_years"] = True
        with self.assertRaises(ValueError):
            run(f)
        f = fixture(True)
        f["kwargs"]["confirmation"]["release_date"] = "2025-10-21"
        with self.assertRaises(ValueError):
            run(f)

    def test_final_original_lineage_must_be_internally_consistent(self):
        mutations = [
            (2, b"<OriginalDatedDate>2012-01-15", b"<OriginalDatedDate>2012-01-16"),
            (2, b"<OriginalIssueDate>2012-01-17", b"<OriginalIssueDate>2018-01-17"),
            (
                3,
                "Original Issue Date January 17, 2012",
                "Original Issue Date January 18, 2012",
            ),
            (2, b"<MaturityDate>2019-01-15", b"<MaturityDate>2019-01-16"),
            (2, b"<ReOpeningIndicator>Y", b"<ReOpeningIndicator>N"),
        ]
        for index, old, new in mutations:
            f = fixture(True)
            f["args"][index] = f["args"][index].replace(old, new)
            with self.subTest(old=old), self.assertRaises(ValueError):
                run(f)

    def test_unknown_original_cusip_is_preserved_in_an_unsupported_audit(self):
        f = fixture(True)
        f["args"][2] = f["args"][2].replace(
            b"<OriginalCUSIP></OriginalCUSIP>", b"<OriginalCUSIP>CD1234567</OriginalCUSIP>"
        )
        with self.assertRaises(UnsupportedIdentity) as caught:
            run(f)
        self.assertEqual(caught.exception.audit["status"], "UNSUPPORTED_ORIGINAL_CUSIP")
        self.assertEqual(caught.exception.audit["original_cusip_lexeme"], "CD1234567")

    def test_future_selected_event_rejects_before_decoding_and_financial_values_stay_opaque(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "[Cc]eiling|[Ff]uture"):
            reconcile_realized_identity(
                None,
                None,
                None,
                None,
                auction_date="2025-10-21",
                announcement_date="bad",
                selected_final_cusip="bad",
            )
        f = fixture(True)
        f["args"][2] = f["args"][2].replace(
            b"</AuctionResults>",
            (
                "<CompetitiveAccepted>"
                + "9" * 7000
                + "</CompetitiveAccepted><HighYield>NaN</HighYield></AuctionResults>"
            ).encode(),
        )
        out = run(f)
        self.assertNotIn("CompetitiveAccepted", repr(out))
        self.assertNotIn("HighYield", repr(out))

    def test_latest_required_document_date_is_retained_without_backdating(self):
        f = fixture(True)
        f["kwargs"]["confirmation"]["release_date"] = "2017-01-16"
        self.assertEqual(run(f)["identity_available_date"], "2017-01-16")
        f = fixture()
        f["args"][3] = f["args"][3].replace("January 13, 2017", "January 16, 2017")
        self.assertEqual(run(f)["identity_available_date"], "2017-01-16")

    def test_unchanged_cusip_cannot_hide_changed_reopening_or_preauction_issue(self):
        f = fixture(True)
        f["args"][2] = f["args"][2].replace(ACTUAL.encode(), ANNOUNCED.encode())
        f["args"][3] = f["args"][3].replace(ACTUAL, ANNOUNCED)
        f["kwargs"].update(selected_final_cusip=ANNOUNCED, confirmation=None)
        with self.assertRaises(ValueError):
            run(f)
        f = fixture()
        for index in (0, 2):
            f["args"][index] = f["args"][index].replace(b"2017-01-17", b"2017-01-12")
        for index in (1, 3):
            f["args"][index] = f["args"][index].replace("January 17, 2017", "January 12, 2017")
        with self.assertRaises(ValueError):
            run(f)


if __name__ == "__main__":
    unittest.main()
