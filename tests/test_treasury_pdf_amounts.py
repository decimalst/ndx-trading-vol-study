"""Prewritten invented PDF table contracts, independent of the XML parser."""

import unittest
from decimal import Decimal
from unittest.mock import patch

from src import treasury_pdf_amounts as pdf

EXPECTED = {
    "PrimaryDealerAccepted": "200",
    "DirectBidderAccepted": "80",
    "IndirectBidderAccepted": "120",
    "CompetitiveAccepted": "400",
    "CompetitiveTendered": "1000",
    "PrimaryDealerTendered": "600",
    "DirectBidderTendered": "200",
    "IndirectBidderTendered": "200",
    "NonCompetitiveAccepted": "10",
    "FIMAAccepted": "5",
    "SOMAAccepted": "20",
    "SOMATendered": "20",
    "TotalAccepted": "435",
    "TotalTendered": "1035",
    "BidToCoverRatio": "2.45",
    "HighYield": "3.754",
}


def fixture():
    return """4Bid-to-Cover Ratio: $1,015/$415 = 2.45
For Immediate Release CONTACT: Treasury
January 13, 2010
TREASURY AUCTION RESULTS
Term and Type of Security 9-Year 10-Month Note
CUSIP Number 912ABC123
High Yield
1 3.754%
Tendered Accepted
Competitive $1,000 $400
Noncompetitive $10 $10
FIMA (Noncompetitive) $5 $5
Subtotal
4 $1,015 $415
5
SOMA $20 $20
Total $1,035 $435
Tendered Accepted
Primary Dealer
6 $600 $200
Direct Bidder
7 $200 $80
Indirect Bidder
8 $200 $120
Total Competitive $1,000 $400
"""


def parse(text=None, **changes):
    event = {"auction_date": "2010-01-13", "cusip": "912ABC123"} | changes
    return pdf.parse_pdf_amounts(fixture() if text is None else text, **event)


class TreasuryPDFAmountsTests(unittest.TestCase):
    def test_exact_sixteen_decimal_source_fields_and_units(self):
        actual = parse()
        self.assertEqual(actual, {key: Decimal(value) for key, value in EXPECTED.items()})
        self.assertEqual(set(actual), set(EXPECTED))
        self.assertTrue(all(type(value) is Decimal for value in actual.values()))
        self.assertEqual(actual["HighYield"], Decimal("3.754"))
        self.assertEqual(actual["CompetitiveTendered"], Decimal("1000"))

    def test_optional_linebreaks_superscripts_and_label_whitespace(self):
        text = fixture().replace("4Bid-to-Cover", "4 Bid-to-Cover")
        text = text.replace("High Yield\n1 ", "High Yield   ")
        text = text.replace("Subtotal\n4 ", "Subtotal ").replace("\n5\n", "\n")
        text = text.replace("Primary Dealer\n6 ", "Primary Dealer  ")
        text = text.replace("Direct Bidder\n7 ", "Direct Bidder\t")
        text = text.replace("Indirect Bidder\n8 ", "Indirect Bidder \n")
        text = text.replace("Tendered Accepted", "Tendered\nAccepted")
        text = text.replace("FIMA (Noncompetitive) ", "FIMA (Noncompetitive) \n")
        self.assertEqual(parse(text), parse())

    def test_identity_future_and_tentative_title_guards_precede_decimal_decode(self):
        cases = [
            (fixture(), {"auction_date": "2025-10-21"}),
            (
                fixture().replace("January 13, 2010", "October 21, 2025"),
                {"auction_date": "2025-10-21"},
            ),
            (fixture(), {"cusip": "912XYZ123"}),
            (fixture().replace("January 13, 2010", "January 14, 2010"), {}),
            (
                fixture().replace(
                    "TREASURY AUCTION RESULTS", "TREASURY AUCTION RESULTS (NONCOMPETITIVE)"
                ),
                {},
            ),
            (
                fixture().replace(
                    "TREASURY AUCTION RESULTS",
                    "TREASURY AUCTION RESULTS\nNONCOMPETITIVE RESULTS",
                ),
                {},
            ),
            (
                fixture().replace(
                    "TREASURY AUCTION RESULTS", "TREASURY TENTATIVE AUCTION RESULTS"
                ),
                {},
            ),
            (
                fixture().replace(
                    "TREASURY AUCTION RESULTS",
                    "TREASURY AUCTION RESULTS\nTREASURY AUCTION RESULTS",
                ),
                {},
            ),
            (fixture(), {"auction_date": None}),
            (fixture(), {"cusip": None}),
        ]
        for number, (text, event) in enumerate(cases):
            with (
                self.subTest(number=number),
                patch.object(
                    pdf,
                    "Decimal",
                    side_effect=AssertionError("financial decode before header guard"),
                ) as decode,
                self.assertRaises(ValueError),
            ):
                parse(text, **event)
            decode.assert_not_called()

    def test_missing_rows_fail_while_explicit_zero_soma_is_preserved(self):
        for row in (
            "Noncompetitive $10 $10\n",
            "FIMA (Noncompetitive) $5 $5\n",
            "SOMA $20 $20\n",
            "Direct Bidder\n7 $200 $80\n",
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                parse(fixture().replace(row, ""))
        text = fixture().replace("SOMA $20 $20", "SOMA $0 $0")
        text = text.replace("Total $1,035 $435", "Total $1,015 $415")
        actual = parse(text)
        self.assertEqual(actual["SOMAAccepted"], Decimal(0))
        self.assertEqual(actual["SOMATendered"], Decimal(0))
        self.assertEqual(actual["TotalAccepted"], Decimal(415))

    def test_column_headers_must_twice_be_tendered_then_accepted(self):
        for text in (
            fixture().replace("Tendered Accepted", "Accepted Tendered", 1),
            fixture().replace("Tendered Accepted", "Tendered", 1),
            fixture().replace("Tendered Accepted", "", 1),
            fixture() + "Tendered Accepted\n",
        ):
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError):
                parse(text)

    def test_duplicate_rows_swapped_columns_and_competitive_disagreement_rejected(self):
        for old, new in (
            (
                "Competitive $1,000 $400\n",
                "Competitive $1,000 $400\nCompetitive $1,000 $400\n",
            ),
            ("6 $600 $200", "6 $200 $600"),
            ("Total Competitive $1,000 $400", "Total Competitive $1,001 $400"),
            ("Direct Bidder\n7 $200 $80", "Direct Bidder\n7 $200 $80\nDirect Bidder $200 $80"),
            ("Indirect Bidder\n8 $200 $120", "Indirect Bidder\n8 $200 $121"),
        ):
            with self.subTest(old=old), self.assertRaises(ValueError):
                parse(fixture().replace(old, new, 1))

    def test_dollar_tokens_require_exact_grouping_and_unsigned_integer_units(self):
        for token in (
            "$1000",
            "$1,00",
            "$10,00",
            "$1,000,",
            "$1,000.5",
            "$+1,000",
            "-$1,000",
            "$1e3",
            "$NaN",
            "1,000",
            "$01,000",
        ):
            text = fixture().replace("Competitive $1,000 $400", f"Competitive {token} $400", 1)
            with self.subTest(token=token), self.assertRaises(ValueError):
                parse(text)

    def test_yield_requires_unique_nonnegative_plain_percentage_and_bcr_is_unique(self):
        for token in ("3.754", "-3.754%", "+3.754%", "3.754e0%", "NaN%", "3,754%"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                parse(fixture().replace("3.754%", token))
        for suffix in ("High Yield 3.754%\n", "4Bid-to-Cover Ratio: $1,015/$415 = 2.45\n"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                parse(fixture() + suffix)
        self.assertEqual(parse(fixture().replace("3.754%", "0%"))["HighYield"], Decimal(0))

    def test_footnote_text_cannot_replace_a_missing_table_row(self):
        text = fixture().replace("FIMA (Noncompetitive) $5 $5\n", "")
        text = "1 FIMA (Noncompetitive) $5 $5\n" + text
        with self.assertRaises(ValueError):
            parse(text)

    def test_public_subtotal_bcr_footnote_and_all_accounting_identities(self):
        cases = (
            ("Noncompetitive $10 $10", "Noncompetitive $11 $10"),
            ("FIMA (Noncompetitive) $5 $5", "FIMA (Noncompetitive) $6 $5"),
            ("4 $1,015 $415", "4 $1,016 $415"),
            ("SOMA $20 $20", "SOMA $19 $20"),
            ("Total $1,035 $435", "Total $1,035 $415"),
            ("$1,015/$415 = 2.45", "$1,000/$400 = 2.50"),
            ("$1,015/$415 = 2.45", "$1,035/$435 = 2.38"),
            ("$1,015/$415 = 2.45", "$1,015/$415 = 2.46"),
            ("$1,015/$415 = 2.45", "$1,015/$415 = 2.450"),
        )
        for old, new in cases:
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                parse(fixture().replace(old, new, 1))
        text = fixture().replace("SOMA $20 $20", "SOMA $30 $20")
        text = text.replace("Total $1,035 $435", "Total $1,045 $435")
        self.assertEqual(parse(text)["SOMATendered"], Decimal(30))


if __name__ == "__main__":
    unittest.main()
