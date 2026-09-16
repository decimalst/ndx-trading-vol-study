"""Prewritten synthetic final/announced identity and frozen-accounting regressions."""

import unittest
from decimal import Decimal, localcontext
from unittest.mock import patch

from src import treasury_auction_amounts as frozen
from src import treasury_history_amounts as amounts
from tests.test_treasury_auction_amounts import AMOUNTS, EVENT, PAR_KEYS, xml

FINAL = "XY7654321"
_DEFAULT = object()


def source(values=None, *, final=FINAL, announced=EVENT["cusip"], original=None, fields=None):
    data = {"CUSIP": final, "AnnouncedCUSIP": announced}
    data.update(fields or {})
    raw = xml(values, data)
    if original is not None:
        raw = raw.replace(
            b"</AuctionResults>",
            f"<OriginalCUSIP>{original}</OriginalCUSIP></AuctionResults>".encode(),
        )
    return raw


def parse(raw=_DEFAULT, **expected):
    args = EVENT | {"cusip": FINAL, "announced_cusip": EVENT["cusip"]} | expected
    return amounts.parse_history_result_amounts(source() if raw is _DEFAULT else raw, **args)


class TreasuryHistoryAmountsTests(unittest.TestCase):
    def test_substitution_keeps_exact_sixteen_decimal_units_and_frozen_rejection(self):
        raw = source(original=EVENT["cusip"])
        self.assertEqual(parse(raw), {key: Decimal(value) for key, value in AMOUNTS.items()})
        self.assertEqual(tuple(parse(raw)), tuple(AMOUNTS))
        self.assertTrue(all(type(v) is Decimal for v in parse(raw).values()))
        with self.assertRaises(ValueError):
            frozen.parse_result_amounts(raw, **(EVENT | {"cusip": FINAL}))

    def test_regular_identity_absent_empty_and_explicit_announced_fields(self):
        expected = frozen.parse_result_amounts(xml(), **EVENT)
        for announced in (None, "", EVENT["cusip"]):
            for original in (None, "", EVENT["cusip"]):
                raw = source(final=EVENT["cusip"], announced=announced, original=original)
                with self.subTest(announced=announced, original=original):
                    self.assertEqual(parse(raw, cusip=EVENT["cusip"]), expected)

    def test_contradictory_or_missing_identity_fails_before_any_decimal(self):
        cases = [
            (source(final="ZZ7654321"), {}),
            (source(announced=None), {}),
            (source(announced=""), {}),
            (source(announced=FINAL), {}),
            (source(announced="ZZ7654321"), {}),
            (source(original="ZZ7654321"), {}),
            (source(final=EVENT["cusip"], announced=FINAL), {"cusip": EVENT["cusip"]}),
            (source(), {"cusip": "xy7654321"}),
            (source(), {"announced_cusip": None}),
            (source(), {"announced_cusip": ""}),
            (source(), {"announced_cusip": True}),
            (source(), {"announced_cusip": "ab1234567"}),
        ]
        for raw, kwargs in cases:
            with (
                self.subTest(kwargs=kwargs, raw=raw),
                patch.object(
                    amounts,
                    "Decimal",
                    side_effect=AssertionError("identity must precede decoding"),
                ) as decode,
            ):
                with self.assertRaises(ValueError):
                    parse(raw, **kwargs)
                decode.assert_not_called()
        with self.assertRaises(TypeError):
            amounts.parse_history_result_amounts(source(), **EVENT)

    def test_original_cusip_is_validated_without_dropping_or_rewriting_raw_xml(self):
        for original in (None, "", EVENT["cusip"], FINAL):
            raw = source(original=original)
            with patch.object(
                amounts, "read_xml_identity", wraps=amounts.read_xml_identity
            ) as reader:
                self.assertEqual(parse(raw)["CompetitiveAccepted"], Decimal("400"))
            reader.assert_called_once_with(raw)
            if original is not None:
                self.assertIn(f"<OriginalCUSIP>{original}</OriginalCUSIP>".encode(), raw)
        raw = source(original=EVENT["cusip"])
        leaf = f"<OriginalCUSIP>{EVENT['cusip']}</OriginalCUSIP>".encode()
        for malformed in (
            raw.replace(leaf, leaf + leaf),
            raw.replace(leaf, b"<Wrapper>" + leaf + b"</Wrapper>"),
            raw.replace(leaf, b"<OriginalCUSIP><Value>AB1234567</Value></OriginalCUSIP>"),
            raw.replace(leaf, b"<t:OriginalCUSIP>AB1234567</t:OriginalCUSIP>"),
        ):
            with self.subTest(raw=malformed), self.assertRaises(ValueError):
                parse(malformed)

    def test_dates_bounds_chronology_and_nominal_class_precede_decoding(self):
        cases = [
            ({"AuctionDate": "2010-01-06"}, {}),
            ({"AnnouncementDate": "2010-01-03"}, {}),
            ({"AnnouncementDate": "2010-01-06"}, {"announcement_date": "2010-01-06"}),
            ({"AuctionDate": "2010-02-30"}, {}),
            ({"AuctionDate": "2025-10-21"}, {"auction_date": "2025-10-21"}),
            ({"AuctionDate": "2009-12-31"}, {"auction_date": "2009-12-31"}),
            ({"AuctionDate": None}, {}),
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
        for fields, expected in cases:
            with (
                self.subTest(fields=fields),
                patch.object(
                    amounts,
                    "Decimal",
                    side_effect=AssertionError("premature financial decoding"),
                ) as decode,
            ):
                with self.assertRaises(ValueError):
                    parse(source(fields=fields), **expected)
                decode.assert_not_called()
        for security in ("NOTE", "BOND"):
            for floating in (None, "N"):
                self.assertEqual(
                    parse(source(fields={"SecurityType": security, "FloatingRate": floating}))[
                        "HighYield"
                    ],
                    Decimal("3.750"),
                )
        self.assertEqual(
            parse(
                source(
                    fields={
                        "AuctionDate": "2010-01-05T00:00:00",
                        "AnnouncementDate": "2010-01-04T00:00:00",
                    }
                )
            )["CompetitiveAccepted"],
            Decimal("400"),
        )

    def test_required_amounts_are_direct_unique_unaliased_unattributed_leaves(self):
        raw = source()
        for key in AMOUNTS:
            leaf = f"<{key}>{AMOUNTS[key]}</{key}>".encode()
            for bad in (
                raw.replace(leaf, b""),
                raw.replace(leaf, leaf + leaf),
                raw.replace(leaf, b"<Wrapper>" + leaf + b"</Wrapper>"),
                raw.replace(leaf, f'<{key} unit="unknown">{AMOUNTS[key]}</{key}>'.encode()),
                raw.replace(leaf, f"<t:{key}>{AMOUNTS[key]}</t:{key}>".encode()),
                raw.replace(leaf, f"<{key}><Value>{AMOUNTS[key]}</Value></{key}>".encode()),
            ):
                with (
                    self.subTest(key=key),
                    patch.object(
                        amounts, "Decimal", side_effect=AssertionError("invalid leaf decoded")
                    ) as decode,
                ):
                    with self.assertRaises(ValueError):
                        parse(bad)
                    decode.assert_not_called()
        for bad in (
            raw[: raw.index(b"<AuctionResults>")] + b"</t:AuctionData>",
            raw.replace(b"</t:AuctionData>", b"<AuctionResults/></t:AuctionData>"),
        ):
            with self.assertRaises(ValueError):
                parse(bad)

    def test_source_precision_nonfinite_and_malformed_numeric_tokens_fail(self):
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
                    parse(source({key: token}))
        for token in ("2", "2.4", "2.450", "+2.45", "2.45e0", "NaN", "-2.45", "2.45 "):
            with self.subTest(ratio=token), self.assertRaises(ValueError):
                parse(source({"BidToCoverRatio": token}))
        for token in ("-0.01", "+3.75", "3.75e0", "NaN", "Infinity", "", "3,750", "3.75 "):
            with self.subTest(yield_token=token), self.assertRaises(ValueError):
                parse(source({"HighYield": token}))
        self.assertEqual(
            parse(source({k: v + ".000" for k, v in AMOUNTS.items() if k in PAR_KEYS})),
            parse(),
        )

    def test_competitive_category_accounting_and_each_acceptance_ceiling(self):
        cases = [
            {"PrimaryDealerAccepted": "201"},
            {"DirectBidderAccepted": "79"},
            {"IndirectBidderAccepted": "121"},
            {"PrimaryDealerTendered": "601"},
            {"DirectBidderTendered": "199"},
            {"IndirectBidderTendered": "201"},
            {"CompetitiveAccepted": "401", "TotalAccepted": "436"},
            {"CompetitiveTendered": "1001", "TotalTendered": "1036"},
            {"PrimaryDealerTendered": "199", "DirectBidderTendered": "601"},
            {"DirectBidderTendered": "79", "PrimaryDealerTendered": "721"},
            {"IndirectBidderTendered": "119", "PrimaryDealerTendered": "681"},
            {"SOMATendered": "19", "TotalTendered": "1034"},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse(source(changes))

    def test_totals_keep_noncompetitive_fima_and_soma_accounting(self):
        for key, values in (
            ("TotalAccepted", ("400", "415", "430")),
            ("TotalTendered", ("1000", "1015", "1030")),
        ):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    parse(source({key: value}))
        self.assertEqual(
            parse(source({"SOMATendered": "30", "TotalTendered": "1045"}))["BidToCoverRatio"],
            Decimal("2.45"),
        )
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
        self.assertEqual(parse(source(zeros))["HighYield"], Decimal(0))
        no_comp = {
            "PrimaryDealerAccepted": "0",
            "DirectBidderAccepted": "0",
            "IndirectBidderAccepted": "0",
            "CompetitiveAccepted": "0",
            "TotalAccepted": "35",
            "BidToCoverRatio": "67.67",
        }
        with self.assertRaises(ValueError):
            parse(source(no_comp))

    def test_public_ratio_and_exact_inclusive_half_cent_boundaries(self):
        for value in ("2.50", "2.38"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(source({"BidToCoverRatio": value}))
        base = {
            "NonCompetitiveAccepted": "0",
            "FIMAAccepted": "0",
            "SOMAAccepted": "0",
            "SOMATendered": "0",
            "TotalAccepted": "400",
        }
        for tendered, valid in ((978, True), (982, True), (977, False), (983, False)):
            changes = base | {
                "CompetitiveTendered": str(tendered),
                "TotalTendered": str(tendered),
                "PrimaryDealerTendered": str(tendered - 400),
            }
            with self.subTest(tendered=tendered):
                if valid:
                    self.assertEqual(
                        parse(source(changes))["BidToCoverRatio"], Decimal("2.45")
                    )
                else:
                    with self.assertRaises(ValueError):
                        parse(source(changes))

    def test_large_par_values_ignore_decimal_precision_context(self):
        factor = 10**25
        scaled = {k: str(int(v) * factor) for k, v in AMOUNTS.items() if k in PAR_KEYS}
        with localcontext() as context:
            context.prec = 2
            result = parse(source(scaled))
        self.assertEqual(result["CompetitiveAccepted"], Decimal(scaled["CompetitiveAccepted"]))
        self.assertEqual(result["BidToCoverRatio"], Decimal("2.45"))

    def test_bounded_raw_xml_decoder_and_identity_leaf_rules_are_retained(self):
        raw = source()
        leaf = b"<AnnouncedCUSIP>AB1234567</AnnouncedCUSIP>"
        bads = [
            None,
            raw.decode(),
            b"\xff",
            raw * 10000,
            b'<!DOCTYPE x><!ENTITY q "x">' + raw,
            raw.replace(leaf, leaf + leaf),
            raw.replace(leaf, b"<Wrapper>" + leaf + b"</Wrapper>"),
            raw.replace(leaf, b"<t:AnnouncedCUSIP>AB1234567</t:AnnouncedCUSIP>"),
        ]
        for bad in bads:
            with self.subTest(kind=type(bad).__name__), self.assertRaises(ValueError):
                parse(bad)


if __name__ == "__main__":
    unittest.main()
