"""Prewritten generated PDF metadata layout contracts; no historical application."""

import unittest

from src.treasury_history_pdf_layout import inspect_pdf

CUSIP = "AB1234567"
FIELDS = {
    "Term and Type of Security": "7-Year Note",
    "Auction Date": "February 23, 2017",
    "Original Issue Date": "February 28, 2017",
    "Issue Date": "February 28, 2017",
    "Maturity Date": "February 29, 2024",
    "Dated Date": "February 28, 2017",
    "Series": "H-2024",
}


def heading(*, result=False, amended=False):
    title = (
        "TREASURY AUCTION RESULTS"
        if result
        else "AMENDED ANNOUNCEMENT"
        if amended
        else "TREASURY OFFERING ANNOUNCEMENT"
    )
    return "FOR IMMEDIATE RELEASE: CONTACT: Treasury\nFebruary 16, 2017\n" + title + "\n"


def standard(*, result=False, amended=False):
    return (
        heading(result=result, amended=amended)
        + f"CUSIP Number {CUSIP}\n"
        + "\n".join(
            f"{k} {v}" for k, v in FIELDS.items() if not (result and k == "Auction Date")
        )
        + "\nCorpus CUSIP Number ZZ1234567\nCUSIP Number(s) XY1234567\n"
    )


def columns(*, amended=False):
    values = [
        FIELDS["Term and Type of Security"],
        "opaque offering cell",
        "opaque outstanding cell",
        CUSIP,
        FIELDS["Auction Date"],
        FIELDS["Original Issue Date"],
        FIELDS["Issue Date"],
        FIELDS["Maturity Date"],
        FIELDS["Dated Date"],
        FIELDS["Series"],
        "opaque yield cell",
        "opaque rate cell",
        "August 31 and February 28",
        "None",
    ]
    labels = [
        "Term and Type of Security",
        "Offering Amount",
        "Currently Outstanding",
        "CUSIP Number",
        "Auction Date",
        "Original Issue Date",
        "Issue Date",
        "Maturity Date",
        "Dated Date",
        "Series",
        "Yield",
        "Interest Rate",
        "Interest Payment Dates4",
        "Accrued Interest from 02/28/2017 to 02/28/2017",
    ]
    return (
        heading(amended=amended)
        + ("" if amended else "1\n")
        + "\n".join(values + labels)
        + "\nPremium or Discount\nDetermined at Auction\n"
        + "Corpus CUSIP Number ZZ1234567\nCUSIP Number(s) XY1234567\n"
    )


class TreasuryHistoryPdfLayoutTests(unittest.TestCase):
    def test_punctuated_own_cusip_conflicts_cannot_be_hidden_by_header_reconstruction(self):
        for base, result in ((standard(result=True), True), (columns(), False)):
            for label in ("CUSIP:", "CUSIP#", "CUSIP Number:", "CUSIP Number#"):
                for value in (CUSIP, "ZZ1234567"):
                    text = base + f"{label} {value}\n"
                    with (
                        self.subTest(result=result, label=label, value=value),
                        self.assertRaises(ValueError),
                    ):
                        inspect_pdf(text, result=result, expected_cusip=CUSIP)

    def test_standard_row_fields_and_own_identity_are_retained(self):
        text = standard()
        out = inspect_pdf(text, result=False, expected_cusip=CUSIP)
        self.assertEqual(set(out), {"header", "release_date", "cusip", "amended", "fields"})
        self.assertEqual(out["fields"], FIELDS)
        self.assertEqual(out["header"]["cusips"], [CUSIP])
        self.assertEqual(out["release_date"], "2017-02-16")
        self.assertFalse(out["amended"])
        self.assertEqual(text, standard())

    def test_observed_column_order_maps_dates_and_cusip_without_search(self):
        for amended in (False, True):
            with self.subTest(amended=amended):
                out = inspect_pdf(columns(amended=amended), result=False, expected_cusip=CUSIP)
                self.assertEqual(out["fields"], FIELDS)
                self.assertEqual(out["cusip"], CUSIP)
                self.assertEqual(out["amended"], amended)

    def test_opaque_financial_cells_are_never_converted_or_returned(self):
        text = (
            columns()
            .replace("opaque offering cell", "9" * 7000)
            .replace("opaque rate cell", "NaN")
        )
        out = inspect_pdf(text, result=False, expected_cusip=CUSIP)
        self.assertEqual(out["fields"], FIELDS)
        self.assertNotIn("9" * 7000, str(out))

    def test_expected_cusip_in_other_cells_cannot_override_own_mismatch(self):
        text = (
            columns()
            .replace(CUSIP + "\n", "XY1234567\n")
            .replace("opaque offering cell", CUSIP)
        )
        with self.assertRaises(ValueError):
            inspect_pdf(text, result=False, expected_cusip=CUSIP)

    def test_duplicate_empty_or_conflicting_own_labels_fail(self):
        examples = [
            standard() + f"CUSIP Number {CUSIP}\n",
            standard() + "CUSIP Number\n",
            standard() + "CUSIP XY1234567\n",
            standard().replace(f"CUSIP Number {CUSIP}", "CUSIP Number"),
            columns() + f"CUSIP Number {CUSIP}\n",
        ]
        for text in examples:
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError):
                inspect_pdf(text, result=False, expected_cusip=CUSIP)

    def test_supplemental_identifiers_never_supply_missing_own_field(self):
        text = (
            standard()
            .replace(f"CUSIP Number {CUSIP}\n", "")
            .replace("CUSIP Number(s) XY1234567", f"CUSIP Number(s) {CUSIP}")
        )
        with self.assertRaises(ValueError):
            inspect_pdf(text, result=False, expected_cusip=CUSIP)

    def test_column_shape_missing_duplicate_and_reordered_labels_fail(self):
        examples = [
            columns().replace("opaque outstanding cell\n", ""),
            columns().replace("Issue Date\nMaturity Date", "Maturity Date\nIssue Date"),
            columns().replace("Series\nYield", "Series\nSeries\nYield"),
            columns().replace("opaque outstanding cell\n", "\n"),
            columns().replace("Auction Date\n", "Auction Date wrong\n"),
        ]
        for text in examples:
            with self.subTest(text=text[-100:]), self.assertRaises(ValueError):
                inspect_pdf(text, result=False, expected_cusip=CUSIP)

    def test_amended_role_is_exposed_and_bogus_or_duplicate_titles_fail(self):
        out = inspect_pdf(standard(amended=True), result=False, expected_cusip=CUSIP)
        self.assertTrue(out["amended"])
        for text, result in [
            (standard(amended=True), True),
            (standard(), True),
            (standard(result=True), False),
            (standard().replace("TREASURY OFFERING ANNOUNCEMENT", "AMENDED RESULTS"), False),
            (standard() + "AMENDED ANNOUNCEMENT\n", False),
            (
                standard().replace(
                    "TREASURY OFFERING ANNOUNCEMENT", "TREASURY OFFERING ANNOUNCEMENT V2"
                ),
                False,
            ),
        ]:
            with self.subTest(result=result), self.assertRaises(ValueError):
                inspect_pdf(text, result=result, expected_cusip=CUSIP)

    def test_results_omit_absent_auction_date_and_optional_fields(self):
        text = standard(result=True)
        for key in ["Original Issue Date", "Series"]:
            text = text.replace(f"{key} {FIELDS[key]}\n", "")
        out = inspect_pdf(text, result=True, expected_cusip=CUSIP)
        self.assertEqual(
            set(out["fields"]), set(FIELDS) - {"Auction Date", "Original Issue Date", "Series"}
        )
        self.assertFalse(out["amended"])

    def test_empty_duplicate_invalid_or_missing_metadata_fail(self):
        for text in [
            standard() + f"Issue Date {FIELDS['Issue Date']}\n",
            standard().replace("Series H-2024", "Series"),
            standard().replace("February 29, 2024", "February 30, 2024"),
            standard().replace("Auction Date February 23, 2017\n", ""),
            standard().replace("Dated Date February 28, 2017", "Dated Date"),
        ]:
            with self.subTest(text=text[-60:]), self.assertRaises(ValueError):
                inspect_pdf(text, result=False, expected_cusip=CUSIP)

    def test_bounded_typed_input_and_release_header(self):
        for text, result, cusip in [
            (None, False, CUSIP),
            ("x" * (2 * 1024 * 1024 + 1), False, CUSIP),
            (standard(), 0, CUSIP),
            (standard(), False, CUSIP.lower()),
            (standard().replace("February 16, 2017", "October 21, 2025"), False, CUSIP),
            (standard().replace("FOR IMMEDIATE RELEASE:", "CONTACT ONLY:"), False, CUSIP),
        ]:
            with self.subTest(result=result, cusip=cusip), self.assertRaises(ValueError):
                inspect_pdf(text, result=result, expected_cusip=cusip)


if __name__ == "__main__":
    unittest.main()
