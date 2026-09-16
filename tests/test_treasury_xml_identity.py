"""Prewritten invented XML identity tests; no source documents or values."""

import copy
import unittest

from src.treasury_xml_identity import check_event_identity, read_xml_identity

NS = "http://www.treasurydirect.gov/"
CUSIP = "AB1234567"
ANNOUNCEMENT = {
    "CUSIP": CUSIP,
    "AnnouncedCUSIP": CUSIP,
    "AnnouncementDate": "2010-01-04T00:00:00",
    "AuctionDate": "2010-01-05",
    "IssueDate": "2030-01-01",
    "MaturityDate": "2040-01-01",
    "OriginalIssueDate": "2009-02-01",
    "OriginalDatedDate": "2009-02-01",
    "SecurityType": " Note ",
    "SecurityTermDayMonth": "9-Year 11-Month",
    "SecurityTermWeekYear": "10-Year",
    "ReOpeningIndicator": "Yes",
    "InflationIndexSecurity": "No",
    "FloatingRate": "No",
    "TypeOfAuction": "Single-Price",
    "AnnouncementPDFName": "A_fixture.pdf",
}
RESULTS = {
    "ReleaseTime": "13:00:00",
    "ResultsPDFName": "R_fixture.pdf",
    "OriginalCUSIP": "XY1234567",
}


def xml(announcement=None, results=None, *, extra="", include_results=True):
    announcement = ANNOUNCEMENT if announcement is None else announcement
    results = RESULTS if results is None else results
    content = "".join(f"<{key}>{value}</{key}>" for key, value in announcement.items())
    out = f'<t:AuctionData xmlns:t="{NS}"><AuctionAnnouncement>{content}{extra}</AuctionAnnouncement>'
    if include_results:
        content = "".join(f"<{key}>{value}</{key}>" for key, value in results.items())
        out += f"<AuctionResults>{content}</AuctionResults>"
    return (out + "</t:AuctionData>").encode()


def check(metadata, **changes):
    selected = {
        "auction_date": "2010-01-05",
        "announcement_date": "2010-01-04",
        "cusip": CUSIP,
    }
    selected.update(changes)
    return check_event_identity(metadata, **selected)


class TreasuryXMLIdentityTests(unittest.TestCase):
    def test_exact_whitelist_preserves_text_and_allows_future_issue_maturity(self):
        metadata = read_xml_identity(xml())
        self.assertEqual(metadata, {"announcement": ANNOUNCEMENT, "results": RESULTS})
        before = copy.deepcopy(metadata)
        self.assertIsNone(check(metadata))
        self.assertEqual(metadata, before)
        self.assertEqual(metadata["announcement"]["SecurityType"], " Note ")
        self.assertNotIn("publication_date", metadata["results"])

    def test_absent_containers_tags_and_empty_leaves_stay_distinct(self):
        announcement = {
            key: value for key, value in ANNOUNCEMENT.items() if key != "AnnouncedCUSIP"
        }
        announcement["OriginalIssueDate"] = ""
        metadata = read_xml_identity(xml(announcement, include_results=False))
        self.assertEqual(metadata, {"announcement": announcement, "results": {}})
        self.assertNotIn("AnnouncedCUSIP", metadata["announcement"])
        self.assertIsNone(check(metadata))
        raw = xml(announcement, {})
        self.assertEqual(read_xml_identity(raw)["results"], {})

    def test_financial_numbers_and_unknown_tags_remain_opaque(self):
        pathological = "9" * 7000
        extra = f"<CompetitiveAccepted>{pathological}</CompetitiveAccepted><HighYield>NaN</HighYield><Financial><Value>1e999999999</Value></Financial>"
        metadata = read_xml_identity(xml(extra=extra))
        self.assertEqual(metadata, {"announcement": ANNOUNCEMENT, "results": RESULTS})
        self.assertIsNone(check(metadata))

    def test_wrong_root_containers_duplicates_and_nested_metadata_rejected(self):
        valid = xml()
        invalid = [
            valid.replace(NS.encode(), b"http://evil.example/"),
            valid.replace(b"t:AuctionData", b"AuctionData"),
            valid.replace(b"AuctionAnnouncement", b"t:AuctionAnnouncement"),
            valid.replace(b"AuctionResults", b"t:AuctionResults"),
            f'<t:AuctionData xmlns:t="{NS}"><AuctionResults/></t:AuctionData>'.encode(),
            valid.replace(b"</t:AuctionData>", b"<AuctionAnnouncement/></t:AuctionData>"),
            valid.replace(b"</t:AuctionData>", b"<AuctionResults/></t:AuctionData>"),
            valid.replace(
                b"</AuctionAnnouncement>", b"<CUSIP>AB1234567</CUSIP></AuctionAnnouncement>"
            ),
            valid.replace(
                b"</AuctionResults>", b"<ReleaseTime>13:00:00</ReleaseTime></AuctionResults>"
            ),
            valid.replace(
                b"<CUSIP>AB1234567</CUSIP>", b"<CUSIP><Nested>AB1234567</Nested></CUSIP>"
            ),
            valid.replace(
                b"</AuctionAnnouncement>",
                b"<Wrapper><AuctionDate>2010-01-05</AuctionDate></Wrapper></AuctionAnnouncement>",
            ),
            valid.replace(
                b"</AuctionResults>",
                b"<Wrapper><ResultsPDFName>R_fixture.pdf</ResultsPDFName></Wrapper></AuctionResults>",
            ),
        ]
        for raw in invalid:
            with self.subTest(raw=raw[:100]), self.assertRaises(ValueError):
                read_xml_identity(raw)

    def test_dtd_entities_encoding_size_and_malformed_xml_rejected(self):
        valid = xml()
        invalid = [
            b'<!DOCTYPE AuctionData [<!ENTITY x "AB1234567">]>' + valid,
            b'<!DOCTYPE AuctionData SYSTEM "https://evil.example/schema.dtd">' + valid,
            b'<!ENTITY x SYSTEM "file:///tmp/sentinel">' + valid,
            valid[:-20],
            b"\xff" + valid,
            valid.decode().encode("utf-16"),
            b'<?xml version="1.0" encoding="ISO-8859-1"?>' + valid,
            b" " * (2 * 1024 * 1024 + 1),
        ]
        for raw in invalid:
            with self.subTest(prefix=raw[:60], size=len(raw)), self.assertRaises(ValueError):
                read_xml_identity(raw)
        with self.assertRaises(ValueError):
            read_xml_identity(valid.decode())

    def test_future_auction_rejected_before_identity_and_metadata_checks(self):
        with self.assertRaisesRegex(ValueError, "[Ff]uture|[Bb]ound|[Cc]eiling"):
            check(None, auction_date="2025-10-21", announcement_date="bad", cusip="bad")
        metadata = read_xml_identity(xml())
        metadata["announcement"].update(
            AuctionDate="2025-10-21", CUSIP="BAD", AnnouncementDate="bad"
        )
        with self.assertRaisesRegex(ValueError, "[Ff]uture|[Bb]ound|[Cc]eiling"):
            check(metadata)
        metadata = read_xml_identity(
            xml(
                ANNOUNCEMENT
                | {"AuctionDate": "2025-10-20", "AnnouncementDate": "2025-10-20T00:00:00"}
            )
        )
        self.assertIsNone(
            check(metadata, auction_date="2025-10-20", announcement_date="2025-10-20")
        )

    def test_cusips_conflicts_blank_missing_and_source_lexemes(self):
        metadata = read_xml_identity(xml())
        for fields in (
            {"CUSIP": "XY1234567"},
            {"AnnouncedCUSIP": "XY1234567"},
            {"CUSIP": "", "AnnouncedCUSIP": ""},
            {"CUSIP": " AB1234567"},
            {"AnnouncedCUSIP": " "},
            {"CUSIP": 123456789},
        ):
            changed = copy.deepcopy(metadata)
            changed["announcement"].update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                check(changed)
        changed = copy.deepcopy(metadata)
        changed["announcement"].pop("CUSIP")
        changed["announcement"].pop("AnnouncedCUSIP")
        with self.assertRaises(ValueError):
            check(changed)
        for selected in ("ab1234567", "AB123456", "AB12345678", " AB1234567"):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                check(metadata, cusip=selected)
        metadata["announcement"]["CUSIP"] = ""
        self.assertIsNone(check(metadata))

    def test_date_matching_floor_precision_and_announcement_chronology(self):
        metadata = read_xml_identity(xml())
        for field, values in (
            (
                "AuctionDate",
                (
                    "2009-12-31",
                    "2010-02-30",
                    "2010-01-06",
                    "2010-01-05T01:00:00",
                    "2010-01-05T00:00:00Z",
                    "2010-01-05T00:00:00.000",
                    " 2010-01-05",
                    "",
                ),
            ),
            (
                "AnnouncementDate",
                (
                    "2010-01-06",
                    "2010-01-03",
                    "2010-1-4",
                    "2010-01-04T00:00:00Z",
                    None,
                    20100104,
                ),
            ),
        ):
            for value in values:
                changed = copy.deepcopy(metadata)
                changed["announcement"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    check(changed)
        for key, value in (
            ("auction_date", "2010-01-05T00:00:00"),
            ("announcement_date", "2010-01-04T00:00:00"),
            ("auction_date", "2009-12-31"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check(metadata, **{key: value})
        metadata["announcement"]["AnnouncementDate"] = "2009-12-31"
        self.assertIsNone(check(metadata, announcement_date="2009-12-31"))


if __name__ == "__main__":
    unittest.main()
