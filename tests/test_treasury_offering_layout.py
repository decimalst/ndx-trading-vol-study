"""Prewritten generated regressions for own-security versus supplemental labels."""

import unittest
from copy import deepcopy
from unittest.mock import patch

from src import treasury_offering_amounts as frozen
from src import treasury_offering_layout as layout

PDF_ID = {
    "auction_date": "2021-04-21",
    "announcement_date": "2021-04-15",
    "cusip": "AB1234567",
}
XML_ID = PDF_ID | {"announced_cusip": None}
IDENTITIES = {
    "announcement_pdf": PDF_ID,
    "announcement_xml": XML_ID,
    "result_xml": XML_ID,
}


def pdf(token="$17,234,567,890", *, extra=""):
    return (
        "Embargoed Until 11:00 A.M. CONTACT: Invented Desk\nApril 15, 2021\n"
        "TREASURY OFFERING ANNOUNCEMENT\nTerm and Type of Security 5-Year Note\n"
        f"Offering Amount {token}\nCUSIP Number AB1234567\n"
        "Auction Date April 21, 2021\n" + extra
    )


def xml(token="17.234567890", *, role="announcement", identity=None):
    i = XML_ID if identity is None else identity
    return (
        '<t:AuctionData xmlns:t="http://www.treasurydirect.gov/">'
        f"<AuctionAnnouncement><CUSIP>{i['cusip']}</CUSIP>"
        f"<AnnouncedCUSIP>{i['announced_cusip'] or ''}</AnnouncedCUSIP>"
        f"<AuctionDate>{i['auction_date']}</AuctionDate>"
        f"<AnnouncementDate>{i['announcement_date']}</AnnouncementDate>"
        f"<OfferingAmount>{token}</OfferingAmount></AuctionAnnouncement>"
        + (
            "<AuctionResults><TotalAccepted>2</TotalAccepted></AuctionResults>"
            if role == "result"
            else ""
        )
        + "</t:AuctionData>"
    ).encode()


def reconcile(text=None, ann=None, result=None, identities=None, unit="USD_BILLIONS"):
    return layout.reconcile_offering(
        pdf() if text is None else text,
        xml() if ann is None else ann,
        xml(role="result") if result is None else result,
        identities=IDENTITIES if identities is None else identities,
        xml_unit=unit,
    )


class TreasuryOfferingLayoutTests(unittest.TestCase):
    def test_distinct_corpus_and_plural_labels_are_not_own_security(self):
        for value in ("", " None", " EF3456789"):
            text = pdf(extra="Corpus CUSIP Number CD2345678\nCUSIP Number(s)" + value + "\n")
            with self.subTest(value=value):
                self.assertEqual(reconcile(text)["offering_amount_usd"], 17234567890)
                self.assertEqual(
                    layout.parse_announcement_offering(text, identity=PDF_ID),
                    {
                        "offering_amount_usd": 17234567890,
                        "source_unit": "USD",
                        "source_token": "$17,234,567,890",
                    },
                )
                # Preserve evidence that the frozen helper is intentionally unchanged.
                with self.assertRaises(ValueError):
                    frozen.parse_announcement_offering(text, identity=PDF_ID)

    def test_expected_identity_in_supplement_cannot_replace_wrong_or_missing_own(self):
        for own in ("CUSIP Number ZZ7654321", "CUSIP Number", ""):
            text = pdf(extra="Corpus CUSIP Number AB1234567\nCUSIP Number(s) AB1234567\n")
            text = text.replace("CUSIP Number AB1234567\n", own + "\n", 1)
            with self.subTest(own=own), self.assertRaises(ValueError):
                reconcile(text)

    def test_duplicate_empty_and_unknown_own_labels_fail(self):
        extras = (
            "CUSIP Number AB1234567\n",
            "CUSIP Number\n",
            "CUSIP Number   \n",
            "CUSIP Number ZZ7654321\n",
            "CUSIP AB1234567\n",
            "CUSIP Number: AB1234567\n",
            "CUSIP Number(s)Alias AB1234567\n",
        )
        for extra in extras:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                reconcile(pdf(extra="Corpus CUSIP Number CD2345678\n" + extra))

    def test_unaffected_output_and_exact_financial_mapping_equal_frozen(self):
        for ptoken, xtoken, dollars in (
            ("$17,234,567,890", "17.234567890", 17234567890),
            ("$1", "0.000000001", 1),
            ("$123,456,789,012,345,678,900", "123456789012.345678900", 123456789012345678900),
        ):
            text, ann, result = pdf(ptoken), xml(xtoken), xml(xtoken, role="result")
            expected = frozen.reconcile_offering(
                text, ann, result, identities=IDENTITIES, xml_unit="USD_BILLIONS"
            )
            with self.subTest(token=ptoken):
                self.assertEqual(reconcile(text, ann, result), expected)
                adapted = reconcile(
                    text + "Corpus CUSIP Number CD2345678\nCUSIP Number(s) None\n", ann, result
                )
                self.assertEqual(adapted, expected)
                self.assertEqual(adapted["offering_amount_usd"], dollars)
                self.assertFalse(adapted["source_admitted"])
                self.assertIn("hypothesis", adapted["xml_scale_basis"])

    def test_frozen_xml_scale_and_strict_leaf_rules_still_apply(self):
        text = pdf(extra="Corpus CUSIP Number CD2345678\nCUSIP Number(s) None\n")
        leaf = b"<OfferingAmount>17.234567890</OfferingAmount>"
        for ann in (
            xml("17.234567891"),
            xml("0"),
            xml("NaN"),
            xml("1e2"),
            xml("0.0000000001"),
            xml().replace(leaf, b""),
            xml().replace(leaf, leaf + leaf),
            xml().replace(leaf, b"<Wrapper>" + leaf + b"</Wrapper>"),
        ):
            with self.subTest(ann=ann), self.assertRaises(ValueError):
                reconcile(text, ann=ann)
        for unit in ("USD", "USD_MILLIONS", "auto", None, 1000000000):
            with self.subTest(unit=unit), self.assertRaises(ValueError):
                reconcile(text, unit=unit)
        for token in ("$0", "$NaN", "$17,23", "$17,234,567,890.0", "$17,234,567,891"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                reconcile(pdf(token, extra="CUSIP Number(s) None\n"))

    def test_all_original_document_identities_precede_any_value_conversion(self):
        text = pdf(extra="Corpus CUSIP Number CD2345678\nCUSIP Number(s) None\n")
        wrong = xml(role="result", identity=XML_ID | {"cusip": "ZZ7654321"})
        with patch.object(
            frozen, "_usd_integer", side_effect=AssertionError("early conversion")
        ) as decode:
            with self.assertRaises(ValueError):
                reconcile(text, result=wrong)
            decode.assert_not_called()
        identities = deepcopy(IDENTITIES)
        identities["result_xml"] = XML_ID | {
            "cusip": "ZZ7654321",
            "announced_cusip": "AB1234567",
        }
        result = xml(role="result", identity=identities["result_xml"])
        self.assertEqual(
            reconcile(text, result=result, identities=identities)["offering_amount_usd"],
            17234567890,
        )

    def test_original_text_is_passed_intact_to_header_validation(self):
        text = pdf(extra="Corpus CUSIP Number CD2345678\nCUSIP Number(s) None\n")
        with patch.object(layout, "probe_pdf_header", wraps=frozen.probe_pdf_header) as probe:
            reconcile(text)
        probe.assert_called_once_with(text)
        self.assertIn("Corpus CUSIP Number CD2345678", text)
        self.assertIn("CUSIP Number(s) None", text)

    def test_date_title_and_duplicate_financial_rows_remain_strict(self):
        changes = (
            pdf().replace("April 15, 2021", "April 14, 2021"),
            pdf().replace("Auction Date April 21, 2021", "Auction Date April 22, 2021"),
            pdf().replace("TREASURY OFFERING ANNOUNCEMENT", "TREASURY AUCTION RESULTS"),
            pdf() + "Auction Date\n",
            pdf() + "Offering Amount\n",
            pdf() + "Offering Amount $17,234,567,890\n",
        )
        for text in changes:
            with self.subTest(text=text), self.assertRaises(ValueError):
                reconcile(text + "CUSIP Number(s) None\n")


if __name__ == "__main__":
    unittest.main()
