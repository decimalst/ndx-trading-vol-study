"""Prewritten invented three-document offering checks; no real application."""

import unittest
from copy import deepcopy
from decimal import localcontext
from unittest.mock import patch

from src import treasury_offering_amounts as offering
from src.treasury_xml_identity import check_event_identity, read_xml_identity

NS = "http://www.treasurydirect.gov/"
PDF_ID = {
    "auction_date": "2020-03-18",
    "announcement_date": "2020-03-12",
    "cusip": "AB1234567",
}
XML_ID = PDF_ID | {"announced_cusip": None}
IDENTITIES = {
    "announcement_pdf": PDF_ID,
    "announcement_xml": XML_ID,
    "result_xml": XML_ID,
}


def xml(token="12.345678900", *, role="announcement", identity=None, extra=""):
    i = XML_ID if identity is None else identity
    announced = i["announced_cusip"] or ""
    head = (
        f'<t:AuctionData xmlns:t="{NS}"><AuctionAnnouncement>'
        f"<CUSIP>{i['cusip']}</CUSIP><AnnouncedCUSIP>{announced}</AnnouncedCUSIP>"
        f"<AuctionDate>{i['auction_date']}</AuctionDate>"
        f"<AnnouncementDate>{i['announcement_date']}</AnnouncementDate>"
        f"<OfferingAmount>{token}</OfferingAmount>{extra}</AuctionAnnouncement>"
    )
    results = (
        "<AuctionResults><TotalAccepted>777</TotalAccepted>"
        "<CompetitiveAccepted>555</CompetitiveAccepted></AuctionResults>"
        if role == "result"
        else ""
    )
    return (head + results + "</t:AuctionData>").encode()


def pdf(token="$12,345,678,900", *, reopening=False):
    footnotes = (
        "1 Governed by the Uniform Offering Circular and this offering announcement.\n"
        "2 An additional issue has the same maturity as the outstanding security.\n"
    )
    return (
        footnotes
        + "Embargoed Until 11:00 A.M. CONTACT: Invented Desk\nMarch 12, 2020\n"
        + "TREASURY OFFERING ANNOUNCEMENT\n1\nTerm and Type of Security "
        + ("9-Year 10-Month Note\n(Reopening)\n" if reopening else "3-Year Note\n")
        + f"Offering Amount {token}\nCurrently Outstanding $99,999,999,999\n"
        + "CUSIP Number AB1234567\nAuction Date March 18, 2020\n"
        + "SOMA Amounts Included in Offering Amount No\n"
        + "FIMA Amounts Included in Offering Amount\n3 Yes\n"
    )


def reconcile(pdf_text=None, ann=None, result=None, identities=None, unit="USD_BILLIONS"):
    return offering.reconcile_offering(
        pdf() if pdf_text is None else pdf_text,
        xml() if ann is None else ann,
        xml(role="result") if result is None else result,
        identities=IDENTITIES if identities is None else identities,
        xml_unit=unit,
    )


class TreasuryOfferingAmountsTests(unittest.TestCase):
    def test_three_sources_reconcile_exact_usd_and_disclose_scale(self):
        result = reconcile()
        self.assertEqual(result["status"], "VERIFIED_STATED_OFFERING_AGREEMENT")
        self.assertEqual(result["offering_amount_usd"], 12345678900)
        self.assertIs(type(result["offering_amount_usd"]), int)
        self.assertEqual(result["unit"], "USD")
        self.assertFalse(result["source_admitted"])
        self.assertEqual(set(result["sources"]), set(IDENTITIES))
        self.assertEqual(result["sources"]["announcement_pdf"]["source_unit"], "USD")
        for role in ("announcement_xml", "result_xml"):
            self.assertEqual(result["sources"][role]["source_unit"], "USD_BILLIONS")
            self.assertEqual(result["sources"][role]["offering_amount_usd"], 12345678900)
        self.assertIn("hypothesis", result["xml_scale_basis"])
        self.assertIn("PDF", result["xml_scale_basis"])

    def test_new_issue_and_reopening_footnotes_do_not_supply_amount(self):
        for reopened in (False, True):
            with self.subTest(reopening=reopened):
                result = reconcile(pdf_text=pdf(reopening=reopened))
                self.assertEqual(result["offering_amount_usd"], 12345678900)
                self.assertEqual(
                    result["sources"]["announcement_pdf"]["source_token"],
                    "$12,345,678,900",
                )
        missing = pdf().replace("Offering Amount $12,345,678,900\n", "")
        with self.assertRaises(ValueError):
            reconcile(pdf_text=missing)

    def test_substitution_preserves_each_expected_identity_without_relaxing_frozen_check(self):
        final = XML_ID | {"cusip": "XY7654321", "announced_cusip": "AB1234567"}
        identities = deepcopy(IDENTITIES)
        identities["result_xml"] = final
        result_xml = xml(role="result", identity=final)
        self.assertEqual(
            reconcile(result=result_xml, identities=identities)["offering_amount_usd"],
            12345678900,
        )
        with self.assertRaises(ValueError):
            reconcile(result=result_xml)
        with self.assertRaises(ValueError):
            check_event_identity(read_xml_identity(result_xml), **PDF_ID)
        # Explicit per-document old/new schedule identities also remain separate.
        moved = final | {"auction_date": "2020-03-19"}
        identities["result_xml"] = moved
        self.assertEqual(
            reconcile(result=xml(role="result", identity=moved), identities=identities)[
                "offering_amount_usd"
            ],
            12345678900,
        )

    def test_all_document_identities_checked_before_amount_conversion(self):
        wrong_result = xml(role="result", identity=XML_ID | {"cusip": "ZZ7654321"})
        with (
            patch.object(
                offering, "_usd_integer", side_effect=AssertionError("decoded early")
            ) as decode,
            self.assertRaises(ValueError),
        ):
            reconcile(result=wrong_result)
        decode.assert_not_called()
        for text in (
            pdf().replace("March 12, 2020", "March 11, 2020"),
            pdf().replace("Auction Date March 18, 2020", "Auction Date March 19, 2020"),
            pdf().replace("CUSIP Number AB1234567", "CUSIP Number ZZ7654321"),
            pdf().replace("TREASURY OFFERING ANNOUNCEMENT", "TREASURY AUCTION RESULTS"),
            pdf() + "CUSIP Number AB1234567\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                reconcile(pdf_text=text)

    def test_explicit_identity_and_unit_contracts_reject_missing_or_wrong_fields(self):
        bad_contracts = [None, {}, True, IDENTITIES | {"extra": XML_ID}]
        for contract in bad_contracts:
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                offering.reconcile_offering(
                    pdf(),
                    xml(),
                    xml(role="result"),
                    identities=contract,
                    xml_unit="USD_BILLIONS",
                )
        for field, value in (
            ("cusip", "ab1234567"),
            ("announced_cusip", True),
            ("auction_date", "2020-02-30"),
            ("auction_date", "2025-10-21"),
            ("announcement_date", "2020-03-19"),
        ):
            identities = deepcopy(IDENTITIES)
            identities["result_xml"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                reconcile(identities=identities)
        for unit in (None, "USD", "USD_MILLIONS", "auto", 1000000000, True):
            with self.subTest(unit=unit), self.assertRaises(ValueError):
                reconcile(unit=unit)

    def test_xml_direct_unique_exact_unaliased_announcement_leaf_and_role(self):
        raw = xml()
        leaf = b"<OfferingAmount>12.345678900</OfferingAmount>"
        cases = [
            raw.replace(leaf, b""),
            raw.replace(leaf, leaf + leaf),
            raw.replace(leaf, b"<Wrapper>" + leaf + b"</Wrapper>"),
            raw.replace(leaf, b"<OfferingAmount><Value>12.345678900</Value></OfferingAmount>"),
            raw.replace(leaf, b"<t:OfferingAmount>12.345678900</t:OfferingAmount>"),
            raw.replace(leaf, leaf + b"<t:OfferingAmount>12.345678900</t:OfferingAmount>"),
            raw.replace(leaf, b'<OfferingAmount unit="USD">12.345678900</OfferingAmount>'),
            raw.replace(leaf, b"<Offering_Amount>12.345678900</Offering_Amount>"),
            raw.replace(leaf, b"").replace(
                b"</t:AuctionData>",
                b"<AuctionResults>" + leaf + b"</AuctionResults></t:AuctionData>",
            ),
        ]
        for number, value in enumerate(cases):
            with self.subTest(number=number), self.assertRaises(ValueError):
                offering.parse_xml_offering(
                    value, role="announcement", identity=XML_ID, unit="USD_BILLIONS"
                )
        for value, role in [
            (xml(), "result"),
            (xml(role="result"), "announcement"),
            (xml(), "auto"),
        ]:
            with self.subTest(role=role), self.assertRaises(ValueError):
                offering.parse_xml_offering(
                    value, role=role, identity=XML_ID, unit="USD_BILLIONS"
                )

    def test_missing_duplicate_aliased_or_malformed_pdf_amount_rows_fail(self):
        for row in (
            "Offering Amount $12,34,567,890",
            "Offering Amount $12345678900",
            "Offering Amount $12,345,678,900.00",
            "Offering Amount -$12,345,678,900",
            "Offering Amount $NaN",
            "Offering Amount $Infinity",
            "Offering Amount $0",
            "OfferingAmount $12,345,678,900",
            "Offered Amount $12,345,678,900",
            "Offering Amount $12,345,678,900 trailing",
            "Offering Amount $12,345,678,900\nOffering Amount $12,345,678,900",
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                reconcile(pdf_text=pdf().replace("Offering Amount $12,345,678,900", row))

    def test_xml_scale_is_fixed_and_usd_must_be_positive_integral(self):
        for token in (
            "",
            "0",
            "0.0",
            "-1",
            "+1",
            "1e2",
            "NaN",
            "Infinity",
            " 12.0",
            "12.0 ",
            "12,000",
            ".5",
            "0.0000000001",
        ):
            with self.subTest(token=token), self.assertRaises(ValueError):
                offering.parse_xml_offering(
                    xml(token), role="announcement", identity=XML_ID, unit="USD_BILLIONS"
                )
        for token, expected in [
            ("0.000000001", 1),
            ("1.000000001", 1000000001),
            ("1.0000000010000", 1000000001),
        ]:
            with self.subTest(token=token):
                self.assertEqual(
                    offering.parse_xml_offering(
                        xml(token), role="announcement", identity=XML_ID, unit="USD_BILLIONS"
                    )["offering_amount_usd"],
                    expected,
                )

    def test_large_exact_values_ignore_float_and_decimal_context(self):
        with (
            localcontext() as context,
            patch("builtins.float", side_effect=AssertionError("no float conversion")),
        ):
            context.prec = 2
            value = reconcile(
                pdf_text=pdf("$123,456,789,012,345,678,900"),
                ann=xml("123456789012.345678900"),
                result=xml("123456789012.345678900", role="result"),
            )
        self.assertEqual(value["offering_amount_usd"], 123456789012345678900)

    def test_each_source_disagreement_fails_without_accepted_total_fallback(self):
        for kwargs in (
            {"pdf_text": pdf("$12,345,678,901")},
            {"ann": xml("12.345678901")},
            {"result": xml("12.345678901", role="result")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reconcile(**kwargs)
        with self.assertRaises(ValueError):
            reconcile(
                result=xml(role="result").replace(
                    b"<OfferingAmount>12.345678900</OfferingAmount>", b""
                )
            )

    def test_empty_duplicate_pdf_labels_cannot_hide_beside_valid_rows(self):
        for label in ("Offering Amount", "Auction Date"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                reconcile(pdf_text=pdf() + label + "\n")

    def test_raw_types_bounds_and_xml_entities_fail(self):
        for raw in (
            None,
            "XML",
            b"",
            b"not xml",
            b"x" * (2 * 1024 * 1024 + 1),
            b'<!DOCTYPE t [<!ENTITY amount "12.0">]>' + xml(),
        ):
            with self.subTest(kind=type(raw).__name__), self.assertRaises(ValueError):
                offering.parse_xml_offering(
                    raw, role="announcement", identity=XML_ID, unit="USD_BILLIONS"
                )
        for text in (None, b"PDF", "", "x" * (2 * 1024 * 1024 + 1)):
            with self.subTest(kind=type(text).__name__), self.assertRaises(ValueError):
                offering.parse_announcement_offering(text, identity=PDF_ID)


if __name__ == "__main__":
    unittest.main()
