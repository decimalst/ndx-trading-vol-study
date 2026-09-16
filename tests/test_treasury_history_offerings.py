"""Prewritten generated history-offering contracts; no source amounts."""

import unittest
from copy import deepcopy
from unittest.mock import patch

from src import treasury_offering_amounts as frozen
from src.treasury_history_offerings import reconcile_history_offering
from tests.test_treasury_history_pdf_layout import CUSIP, columns, standard

PDF_ID = {
    "auction_date": "2017-02-23",
    "announcement_date": "2017-02-16",
    "cusip": CUSIP,
}
XML_ID = PDF_ID | {"announced_cusip": None}
IDS = {"announcement_pdf": PDF_ID, "announcement_xml": XML_ID, "result_xml": XML_ID}


def pdf(token="$17,234,567,890", *, column=False, amended=False):
    if column:
        return columns(amended=amended).replace("opaque offering cell", token)
    return standard(amended=amended).replace(
        "Term and Type of Security 7-Year Note",
        "Term and Type of Security 7-Year Note\nOffering Amount " + token,
    )


def xml(token="17.234567890", *, result=False, identity=None):
    i = XML_ID if identity is None else identity
    return (
        '<t:AuctionData xmlns:t="http://www.treasurydirect.gov/">'
        f"<AuctionAnnouncement><CUSIP>{i['cusip']}</CUSIP>"
        f"<AnnouncedCUSIP>{i['announced_cusip'] or ''}</AnnouncedCUSIP>"
        f"<AuctionDate>{i['auction_date']}</AuctionDate>"
        f"<AnnouncementDate>{i['announcement_date']}</AnnouncementDate>"
        f"<OfferingAmount>{token}</OfferingAmount></AuctionAnnouncement>"
        + (
            "<AuctionResults><TotalAccepted>opaque</TotalAccepted></AuctionResults>"
            if result
            else ""
        )
        + "</t:AuctionData>"
    ).encode()


def run(text=None, ann=None, result=None, identities=None, unit="USD_BILLIONS"):
    return reconcile_history_offering(
        pdf() if text is None else text,
        xml() if ann is None else ann,
        xml(result=True) if result is None else result,
        identities=IDS if identities is None else identities,
        xml_unit=unit,
    )


class TreasuryHistoryOfferingTests(unittest.TestCase):
    def test_rows_and_columns_produce_exact_frozen_result_schema(self):
        expected = {
            "status": "VERIFIED_STATED_OFFERING_AGREEMENT",
            "offering_amount_usd": 17234567890,
            "unit": "USD",
            "sources": {
                "announcement_pdf": {
                    "offering_amount_usd": 17234567890,
                    "source_unit": "USD",
                    "source_token": "$17,234,567,890",
                },
                "announcement_xml": {
                    "offering_amount_usd": 17234567890,
                    "source_unit": "USD_BILLIONS",
                    "source_token": "17.234567890",
                },
                "result_xml": {
                    "offering_amount_usd": 17234567890,
                    "source_unit": "USD_BILLIONS",
                    "source_token": "17.234567890",
                },
            },
            "xml_scale_basis": "Fixed 1e9 mapping hypothesis checked against the explicitly USD-labelled PDF on this record; not a schema-authenticated universal XML unit.",
            "source_admitted": False,
        }
        for column in (False, True):
            with self.subTest(column=column):
                self.assertEqual(run(pdf(column=column)), expected)

    def test_amended_pdf_uses_own_release_and_preserves_original_xml_dates(self):
        identities = deepcopy(IDS)
        identities["announcement_pdf"]["announcement_date"] = "2017-02-20"
        for column in (False, True):
            text = pdf(column=column, amended=True).replace(
                "February 16, 2017", "February 20, 2017"
            )
            before = deepcopy(identities)
            self.assertEqual(
                run(text, identities=identities)["offering_amount_usd"], 17234567890
            )
            self.assertEqual(identities, before)
            with self.assertRaises(ValueError):
                run(text)

    def test_every_document_identity_precedes_any_amount_conversion(self):
        cases = [
            {"text": pdf().replace(CUSIP, "XY1234567")},
            {"ann": xml(identity=XML_ID | {"auction_date": "2017-02-22"})},
            {"result": xml(result=True, identity=XML_ID | {"cusip": "XY1234567"})},
            {
                "result": xml(
                    result=True, identity=XML_ID | {"announcement_date": "2017-02-20"}
                )
            },
            {
                "text": pdf().replace(
                    "Auction Date February 23, 2017", "Auction Date February 22, 2017"
                )
            },
        ]
        for kwargs in cases:
            with (
                self.subTest(kwargs=list(kwargs)),
                patch.object(
                    frozen, "_answer", side_effect=AssertionError("premature conversion")
                ) as convert,
            ):
                with self.assertRaises(ValueError):
                    run(**kwargs)
                convert.assert_not_called()

    def test_explicit_different_final_identity_does_not_rewrite_raw_sources(self):
        identities = deepcopy(IDS)
        identities["result_xml"] = XML_ID | {"cusip": "XY1234567", "announced_cusip": CUSIP}
        raw = xml(result=True, identity=identities["result_xml"])
        before = bytes(raw)
        self.assertEqual(
            run(result=raw, identities=identities)["offering_amount_usd"], 17234567890
        )
        self.assertEqual(raw, before)

    def test_three_way_disagreement_and_invalid_dollar_tokens_fail(self):
        for column in (False, True):
            for token in (
                "$17,234,567,891",
                "$0",
                "$17,23",
                "$17,234,567,890.0",
                "17234567890",
                "$NaN",
            ):
                with self.subTest(column=column, token=token), self.assertRaises(ValueError):
                    run(pdf(token, column=column))
        for kwargs in (
            {"ann": xml("17.234567891")},
            {"result": xml("17.234567891", result=True)},
        ):
            with self.assertRaises(ValueError):
                run(**kwargs)

    def test_duplicate_empty_alias_or_supplemental_conflicts_fail(self):
        for column in (False, True):
            for extra in (
                "Offering Amount\n",
                "Offering Amount $17,234,567,890\n",
                "Offering Amount: $17,234,567,890\n",
                "Offering Amount(s) $17,234,567,890\n",
                "CUSIP Number: AB1234567\n",
                "CUSIP Number(s)Alias AB1234567\n",
            ):
                with self.subTest(column=column, extra=extra), self.assertRaises(ValueError):
                    run(pdf(column=column) + extra)
        for text in (
            pdf().replace("Offering Amount $17,234,567,890", "Offering Amount"),
            pdf(column=True).replace("$17,234,567,890\n", "\n"),
        ):
            with self.assertRaises(ValueError):
                run(text)

    def test_fixed_scale_and_strict_xml_leaf_rules_remain_unchanged(self):
        for unit in (None, "auto", "USD", "USD_MILLIONS", 1000000000):
            with self.subTest(unit=unit), self.assertRaises(ValueError):
                run(unit=unit)
        leaf = b"<OfferingAmount>17.234567890</OfferingAmount>"
        for ann in (
            xml("0.0000000001"),
            xml("1e2"),
            xml().replace(leaf, leaf + leaf),
            xml().replace(leaf, b"<Wrapper>" + leaf + b"</Wrapper>"),
        ):
            with self.assertRaises(ValueError):
                run(ann=ann)
        self.assertEqual(
            run(pdf("$1", column=True), xml("0.000000001"), xml("0.000000001", result=True))[
                "offering_amount_usd"
            ],
            1,
        )

    def test_original_text_and_identity_contracts_remain_strict(self):
        text = pdf(column=True)
        from src import treasury_history_offerings as adapter

        with patch.object(adapter, "inspect_pdf", wraps=adapter.inspect_pdf) as inspect:
            run(text)
            inspect.assert_called_once_with(text, result=False, expected_cusip=CUSIP)
        for identities in (
            {},
            IDS | {"extra": PDF_ID},
            IDS | {"announcement_pdf": PDF_ID | {"extra": 1}},
        ):
            with self.assertRaises(ValueError):
                run(identities=identities)
        for text in (
            pdf() + "\x00",
            pdf().replace("TREASURY OFFERING ANNOUNCEMENT", "TREASURY AUCTION RESULTS"),
        ):
            with self.assertRaises(ValueError):
                run(text)


if __name__ == "__main__":
    unittest.main()
