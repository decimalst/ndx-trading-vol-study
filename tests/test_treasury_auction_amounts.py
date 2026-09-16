"""Prewritten synthetic par-dollar accounting contracts; no source-value reads."""

import unittest
from decimal import Decimal
from unittest.mock import patch

from src import treasury_auction_amounts as amounts

NS = "http://www.treasurydirect.gov/"
EVENT = {"auction_date": "2010-01-05", "announcement_date": "2010-01-04", "cusip": "AB1234567"}
AMOUNTS = {
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
    "HighYield": "3.750",
}
PAR_KEYS = tuple(key for key in AMOUNTS if key not in {"BidToCoverRatio", "HighYield"})


def xml(values=None, announcement=None):
    fields = {
        "CUSIP": EVENT["cusip"],
        "AuctionDate": EVENT["auction_date"],
        "AnnouncementDate": EVENT["announcement_date"],
        "SecurityType": "NOTE",
        "InflationIndexSecurity": "N",
        "FloatingRate": "N",
    }
    fields.update(announcement or {})
    data = dict(AMOUNTS)
    data.update(values or {})
    head = "".join(
        f"<{key}>{value}</{key}>" for key, value in fields.items() if value is not None
    )
    body = "".join(
        f"<{key}>{value}</{key}>" for key, value in data.items() if value is not None
    )
    return (
        f'<t:AuctionData xmlns:t="{NS}"><AuctionAnnouncement>{head}</AuctionAnnouncement>'
        f"<AuctionResults>{body}</AuctionResults></t:AuctionData>"
    ).encode()


def parse(raw=None, **event):
    return amounts.parse_result_amounts(xml() if raw is None else raw, **(EVENT | event))


class TreasuryAuctionAmountsTests(unittest.TestCase):
    def test_exact_sixteen_decimal_values_keep_dollars_and_yield_percentage_points(self):
        result = parse()
        self.assertEqual(result, {key: Decimal(value) for key, value in AMOUNTS.items()})
        self.assertEqual(set(result), set(AMOUNTS))
        self.assertTrue(all(type(value) is Decimal for value in result.values()))
        self.assertEqual(result["CompetitiveAccepted"], Decimal("400"))
        self.assertEqual(result["HighYield"], Decimal("3.750"))

    def test_note_bond_and_only_pre_frn_absence_are_admissible(self):
        for security in ("NOTE", "BOND"):
            for floating in ("N", None):
                with self.subTest(security=security, floating=floating):
                    result = parse(
                        xml(announcement={"SecurityType": security, "FloatingRate": floating})
                    )
                    self.assertEqual(result["HighYield"], Decimal("3.750"))
        self.assertEqual(
            parse(
                xml(announcement={"FloatingRate": None, "AuctionDate": "2014-01-28"}),
                auction_date="2014-01-28",
            ),
            parse(),
        )

    def test_class_and_frn_absence_rejected_before_decimal_decode(self):
        cases = [
            ({"SecurityType": "BILL"}, {}),
            ({"SecurityType": None}, {}),
            ({"InflationIndexSecurity": "Y"}, {}),
            ({"InflationIndexSecurity": None}, {}),
            ({"FloatingRate": "Y"}, {}),
            ({"FloatingRate": ""}, {}),
            (
                {"FloatingRate": None, "AuctionDate": "2014-01-29"},
                {"auction_date": "2014-01-29"},
            ),
        ]
        for announcement, event in cases:
            with (
                self.subTest(announcement=announcement),
                patch.object(
                    amounts,
                    "Decimal",
                    side_effect=AssertionError("financial decode preceded class check"),
                ) as decode,
                self.assertRaises(ValueError),
            ):
                parse(xml({"HighYield": "NaN"}, announcement), **event)
            decode.assert_not_called()

    def test_date_and_event_identity_fail_before_any_decimal_decode(self):
        cases = [
            ({"AuctionDate": "2025-10-21"}, {"auction_date": "2025-10-21"}),
            ({"AuctionDate": "2009-12-31"}, {"auction_date": "2009-12-31"}),
            ({"AuctionDate": "2010-02-30"}, {}),
            ({"AuctionDate": "2010-01-06"}, {}),
            ({"AnnouncementDate": "2010-01-03"}, {}),
            ({"CUSIP": "XY1234567"}, {}),
            ({"AuctionDate": None}, {}),
            ({}, {"auction_date": "2025-10-21"}),
        ]
        for announcement, event in cases:
            with (
                self.subTest(announcement=announcement, event=event),
                patch.object(
                    amounts,
                    "Decimal",
                    side_effect=AssertionError("financial decode preceded identity check"),
                ) as decode,
                self.assertRaises(ValueError),
            ):
                parse(xml({"PrimaryDealerAccepted": "1e999999"}, announcement), **event)
            decode.assert_not_called()

    def test_missing_required_leaves_fail_but_explicit_zeros_are_preserved(self):
        for key in AMOUNTS:
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse(xml({key: None}))
        zeros = {
            "NonCompetitiveAccepted": "0",
            "FIMAAccepted": "0",
            "SOMAAccepted": "0",
            "SOMATendered": "0",
            "TotalAccepted": "400",
            "TotalTendered": "1000",
            "BidToCoverRatio": "2.50",
            "HighYield": "0",
        }
        result = parse(xml(zeros))
        for key in (
            "NonCompetitiveAccepted",
            "FIMAAccepted",
            "SOMAAccepted",
            "SOMATendered",
            "HighYield",
        ):
            self.assertEqual(result[key], Decimal(0))

    def test_par_amounts_require_unsigned_plain_integral_decimals(self):
        for key in PAR_KEYS:
            for token in (
                "1,000",
                "+200",
                "-200",
                "2e2",
                "NaN",
                "Infinity",
                "",
                " ",
                "200.1",
                ".0",
                "200 ",
            ):
                with self.subTest(key=key, token=token), self.assertRaises(ValueError):
                    parse(xml({key: token}))
        integral_fractions = {
            key: value + ".000" for key, value in AMOUNTS.items() if key in PAR_KEYS
        }
        self.assertEqual(parse(xml(integral_fractions)), parse())

    def test_ratio_precision_and_nonnegative_plain_yield_units(self):
        for token in ("2", "2.4", "2.450", "+2.45", "2.45e0", "NaN", "-2.45", "2.45 "):
            with self.subTest(ratio=token), self.assertRaises(ValueError):
                parse(xml({"BidToCoverRatio": token}))
        for token in ("-0.01", "+3.75", "3.75e0", "NaN", "Infinity", "", "3,750", "3.75 "):
            with self.subTest(yield_token=token), self.assertRaises(ValueError):
                parse(xml({"HighYield": token}))

    def test_exactly_one_results_and_direct_unique_unaliased_amount_leaves(self):
        original = xml()
        leaf = b"<PrimaryDealerAccepted>200</PrimaryDealerAccepted>"
        cases = [
            original[: original.index(b"<AuctionResults>")] + b"</t:AuctionData>",
            original.replace(b"</t:AuctionData>", b"<AuctionResults/></t:AuctionData>"),
            original.replace(b"</AuctionResults>", leaf + b"</AuctionResults>"),
            original.replace(leaf, b"<Wrapper>" + leaf + b"</Wrapper>"),
            original.replace(
                b"</AuctionResults>", b"<Wrapper>" + leaf + b"</Wrapper></AuctionResults>"
            ),
            original.replace(leaf, b"<t:PrimaryDealerAccepted>200</t:PrimaryDealerAccepted>"),
            original.replace(
                leaf, b"<PrimaryDealerAccepted><Inner>200</Inner></PrimaryDealerAccepted>"
            ),
            original.replace(
                b"</AuctionResults>",
                b"<t:PrimaryDealerAccepted>200</t:PrimaryDealerAccepted></AuctionResults>",
            ),
        ]
        for number, raw in enumerate(cases):
            with self.subTest(number=number), self.assertRaises(ValueError):
                parse(raw)

    def test_category_accepted_and_tendered_sums_must_reconcile(self):
        for changes in (
            {"PrimaryDealerAccepted": "201"},
            {"DirectBidderAccepted": "79"},
            {"IndirectBidderAccepted": "121"},
            {"PrimaryDealerTendered": "601"},
            {"DirectBidderTendered": "199"},
            {"IndirectBidderTendered": "201"},
            {"CompetitiveAccepted": "401", "TotalAccepted": "436"},
            {"CompetitiveTendered": "1001", "TotalTendered": "1036"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse(xml(changes))

    def test_each_category_and_soma_accepted_cannot_exceed_tendered(self):
        for changes in (
            {"PrimaryDealerTendered": "199", "DirectBidderTendered": "601"},
            {"DirectBidderTendered": "79", "PrimaryDealerTendered": "721"},
            {"IndirectBidderTendered": "119", "PrimaryDealerTendered": "681"},
            {"SOMATendered": "19", "TotalTendered": "1034"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse(xml(changes))

    def test_total_reconciliation_includes_noncompetitive_fima_and_soma(self):
        for changes in (
            {"TotalAccepted": "400"},
            {"TotalAccepted": "415"},
            {"TotalAccepted": "430"},
            {"TotalTendered": "1000"},
            {"TotalTendered": "1015"},
            {"TotalTendered": "1030"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse(xml(changes))
        more_soma_tendered = parse(xml({"SOMATendered": "30", "TotalTendered": "1045"}))
        self.assertEqual(more_soma_tendered["SOMATendered"], Decimal("30"))
        self.assertEqual(more_soma_tendered["BidToCoverRatio"], Decimal("2.45"))

    def test_bid_to_cover_uses_public_totals_not_competitive_or_soma_inclusive_totals(self):
        for token in ("2.50", "2.38"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                parse(xml({"BidToCoverRatio": token}))
        self.assertEqual(
            parse(
                xml(
                    {
                        "SOMAAccepted": "0",
                        "SOMATendered": "0",
                        "TotalAccepted": "415",
                        "TotalTendered": "1015",
                    }
                )
            )["BidToCoverRatio"],
            Decimal("2.45"),
        )

    def test_bid_to_cover_half_cent_cross_product_tolerance_is_inclusive(self):
        base = {
            "NonCompetitiveAccepted": "0",
            "FIMAAccepted": "0",
            "SOMAAccepted": "0",
            "SOMATendered": "0",
            "TotalAccepted": "400",
        }
        for tendered, accepted in (
            ("978", True),
            ("982", True),
            ("977", False),
            ("983", False),
        ):
            changes = base | {
                "CompetitiveTendered": tendered,
                "TotalTendered": tendered,
                "PrimaryDealerTendered": str(int(tendered) - 400),
            }
            with self.subTest(tendered=tendered):
                if accepted:
                    self.assertEqual(parse(xml(changes))["BidToCoverRatio"], Decimal("2.45"))
                else:
                    with self.assertRaises(ValueError):
                        parse(xml(changes))

    def test_zero_competitive_acceptance_is_not_an_admissible_public_auction(self):
        changes = {
            "PrimaryDealerAccepted": "0",
            "DirectBidderAccepted": "0",
            "IndirectBidderAccepted": "0",
            "CompetitiveAccepted": "0",
            "TotalAccepted": "35",
            "BidToCoverRatio": "67.67",
        }
        with self.assertRaises(ValueError):
            parse(xml(changes))
        changes.update(
            {"NonCompetitiveAccepted": "0", "FIMAAccepted": "0", "TotalAccepted": "20"}
        )
        with self.assertRaises(ValueError):
            parse(xml(changes))


if __name__ == "__main__":
    unittest.main()
