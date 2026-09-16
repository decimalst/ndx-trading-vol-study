"""Prewritten generated source-date conventions; no financial observations."""

import copy
import unittest
from datetime import date, datetime

from src.treasury_history_lineage import validate_lineage


def ordinary(**changes):
    args = {
        "dated": date(2018, 1, 31),
        "issue": date(2018, 2, 1),
        "maturity": date(2023, 1, 31),
        "original_dated": None,
        "original_issue": None,
        "pdf_original_issue": None,
        "reopening": False,
    }
    return args | changes


def substituted(**changes):
    return {
        "dated": date(2016, 2, 29),
        "issue": date(2016, 2, 29),
        "maturity": date(2018, 2, 28),
        "original_dated": date(2013, 2, 28),
        "original_issue": date(2013, 2, 28),
        "pdf_original_issue": date(2013, 2, 28),
        "reopening": True,
        "allow_distinct_original_dates": True,
    } | changes


class TreasuryHistoryLineageTests(unittest.TestCase):
    def test_new_issue_output_is_exact_and_preserves_source_dates(self):
        args = ordinary()
        before = copy.deepcopy(args)
        self.assertEqual(
            validate_lineage(**args),
            {
                "original_tenor_years": 5,
                "dated_date": "2018-01-31",
                "issue_date": "2018-02-01",
                "maturity_date": "2023-01-31",
                "original_issue_date": "2018-02-01",
                "original_dated_date": "2018-01-31",
                "reopening": False,
            },
        )
        self.assertEqual(args, before)

    def test_all_six_supported_tenors_accept_explicit_month_end_anniversaries(self):
        for start, tenor in (
            (2010, 2),
            (2017, 3),
            (2011, 5),
            (2013, 7),
            (2010, 10),
            (2010, 30),
        ):
            with self.subTest(tenor=tenor):
                result = validate_lineage(
                    **ordinary(
                        dated=date(start, 2, 28),
                        issue=date(start, 3, 1),
                        maturity=date(start + tenor, 2, 29),
                    )
                )
                self.assertEqual(result["original_tenor_years"], tenor)
                self.assertEqual(result["dated_date"], f"{start}-02-28")
                self.assertEqual(result["maturity_date"], f"{start + tenor}-02-29")

    def test_reverse_leap_day_and_ordinary_non_month_end_anniversaries(self):
        for dated, maturity in (
            (date(2012, 2, 29), date(2017, 2, 28)),
            (date(2011, 2, 28), date(2016, 2, 28)),
            (date(2018, 4, 15), date(2023, 4, 15)),
            (date(2018, 4, 30), date(2023, 4, 30)),
        ):
            with self.subTest(dated=dated):
                result = validate_lineage(
                    **ordinary(dated=dated, issue=dated, maturity=maturity)
                )
                self.assertEqual(result["original_tenor_years"], 5)
                self.assertEqual(result["dated_date"], dated.isoformat())
                self.assertEqual(result["maturity_date"], maturity.isoformat())

    def test_no_general_day_tolerance_month_change_or_unsupported_tenor(self):
        cases = (
            (date(2011, 2, 27), date(2016, 2, 29)),
            (date(2018, 4, 30), date(2023, 5, 31)),
            (date(2018, 1, 30), date(2023, 1, 31)),
            (date(2018, 1, 31), date(2022, 1, 31)),
        )
        for dated, maturity in cases:
            with self.subTest(dated=dated, maturity=maturity), self.assertRaises(ValueError):
                validate_lineage(**ordinary(dated=dated, issue=dated, maturity=maturity))

    def test_new_issue_optional_original_fields_must_agree(self):
        base = ordinary(
            original_dated=date(2018, 1, 31),
            original_issue=date(2018, 2, 1),
            pdf_original_issue=date(2018, 2, 1),
        )
        self.assertEqual(validate_lineage(**base), validate_lineage(**ordinary()))
        for key, value in (
            ("original_dated", date(2017, 1, 31)),
            ("original_issue", date(2018, 1, 31)),
            ("pdf_original_issue", date(2018, 2, 2)),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_lineage(**(base | {key: value}))

    def test_ordinary_reopening_uses_original_tenor_without_special_flag(self):
        result = validate_lineage(
            **ordinary(
                dated=date(2018, 1, 31),
                issue=date(2020, 1, 31),
                maturity=date(2028, 1, 31),
                original_dated=date(2018, 1, 31),
                original_issue=date(2018, 2, 1),
                pdf_original_issue=date(2018, 2, 1),
                reopening=True,
            )
        )
        self.assertEqual(result["original_tenor_years"], 10)
        self.assertEqual(result["original_issue_date"], "2018-02-01")
        self.assertEqual(result["original_dated_date"], "2018-01-31")

    def test_observed_eom_and_confirmed_substitution_overlap_preserves_both_anchors(self):
        announced = validate_lineage(
            **ordinary(
                dated=date(2016, 2, 29),
                issue=date(2016, 2, 29),
                maturity=date(2018, 2, 28),
            )
        )
        with self.assertRaises(ValueError):
            validate_lineage(**substituted(allow_distinct_original_dates=False))
        realized = validate_lineage(**substituted())
        self.assertEqual(announced["original_tenor_years"], 2)
        self.assertEqual(realized["original_tenor_years"], 5)
        self.assertEqual(realized["dated_date"], "2016-02-29")
        self.assertEqual(realized["original_dated_date"], "2013-02-28")
        self.assertEqual(realized["maturity_date"], "2018-02-28")

    def test_substitution_original_anchor_can_itself_cross_leap_day(self):
        result = validate_lineage(
            **substituted(
                dated=date(2015, 2, 28),
                issue=date(2015, 3, 2),
                maturity=date(2017, 2, 28),
                original_dated=date(2012, 2, 29),
                original_issue=date(2012, 2, 29),
                pdf_original_issue=date(2012, 2, 29),
            )
        )
        self.assertEqual(result["original_tenor_years"], 5)
        self.assertEqual(result["original_dated_date"], "2012-02-29")

    def test_reopening_requires_all_original_fields_and_pdf_agreement(self):
        for change in (
            {"original_dated": None},
            {"original_issue": None},
            {"pdf_original_issue": None},
            {"pdf_original_issue": date(2013, 3, 1)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_lineage(**substituted(**change))

    def test_chronology_and_distinct_original_date_direction_are_strict(self):
        cases = [
            ordinary(issue=date(2018, 1, 30)),
            ordinary(issue=date(2023, 1, 31)),
            ordinary(maturity=date(2018, 1, 31)),
            substituted(
                original_issue=date(2013, 2, 27), pdf_original_issue=date(2013, 2, 27)
            ),
            substituted(
                original_issue=date(2016, 2, 29), pdf_original_issue=date(2016, 2, 29)
            ),
            substituted(dated=date(2013, 2, 27)),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_lineage(**args)

    def test_calendar_and_boolean_types_are_not_coerced(self):
        cases = [
            {"dated": "2018-01-31"},
            {"dated": datetime(2018, 1, 31)},
            {"issue": None},
            {"maturity": False},
            {"original_dated": "2018-01-31"},
            {"original_issue": 0},
            {"pdf_original_issue": datetime(2018, 2, 1)},
            {"reopening": 0},
            {"allow_distinct_original_dates": 1},
        ]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_lineage(**ordinary(**change))


if __name__ == "__main__":
    unittest.main()
