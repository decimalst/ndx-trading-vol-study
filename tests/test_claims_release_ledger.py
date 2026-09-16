"""Generated first-report source contracts; no historical files or model inputs."""

import copy
import unittest
from datetime import date, timedelta

from src import claims_release_ledger as ledger

STATUS = "FIRST_REPORT_LEDGER_BUILT_NOT_INDEPENDENTLY_VERIFIED"
TAIL = ("2025-09-27", "2025-10-04", "2025-10-11", "2025-10-18")


def report(week="2018-11-17", released="2018-11-22", value=123456):
    day = date.fromisoformat(released)
    return {
        "statistic": "advance_seasonally_adjusted_initial_claims",
        "observation_date": week,
        "actual_release_date": released,
        "value": value,
        "source_url": f"https://oui.doleta.gov/press/{day.year}/{day:%m%d%y}.pdf",
        "release_text_sha256": "a" * 64,
        "body_release_dates": [released],
        "body_release_date_verified": True,
    }


def alfred(week="2018-11-17", released="2018-11-22", value=123456, line=2):
    return {
        "observation_date": week,
        "alfred_realtime_start_date": released,
        "value": value,
        "status": "observed" if value is not None else "missing",
        "source_line": line,
    }


def generated(start, end):
    reports, records = [], []
    day, last = date.fromisoformat(start), date.fromisoformat(end)
    while day <= last:
        week = day.isoformat()
        released = (day + timedelta(days=5)).isoformat()
        value = 100001 + len(reports)
        if week not in TAIL:
            reports.append(report(week, released, value))
            records.append(alfred(week, released, value, len(records) + 2))
            if week == "2017-03-18":
                reports[-1]["value"], records[-1]["value"] = 261000, 258000
            if week == "2018-03-17":
                reports[-1]["value"], records[-1]["value"] = 229000, 227000
                records[-1]["alfred_realtime_start_date"] = "2018-03-29"
        day += timedelta(days=7)
    return reports, records


class FirstReportLedgerTests(unittest.TestCase):
    def build(self, reports=None, records=None, start="2018-11-17", end=None):
        return ledger.build_first_report_ledger(
            [report()] if reports is None else reports,
            [alfred()] if records is None else records,
            start=start,
            end=start if end is None else end,
        )

    def test_exact_ordinary_schema_and_separate_comparison_clock(self):
        self.assertEqual(
            self.build(),
            {
                "status": STATUS,
                "start": "2018-11-17",
                "end": "2018-11-17",
                "counts": {
                    "observed": 1,
                    "unresolved_correction": 0,
                    "no_admitted_release": 0,
                },
                "rows": [
                    {
                        "reference_week": "2018-11-17",
                        "release_date": "2018-11-22",
                        "first_report_value": 123456,
                        "status": "observed",
                        "source_url": report()["source_url"],
                        "release_text_sha256": "a" * 64,
                        "source_comparison": {
                            "kind": "exact_agreement",
                            "dol_value": 123456,
                            "alfred_value": 123456,
                            "alfred_realtime_start_date": "2018-11-22",
                            "alfred_source_line": 2,
                            "value_agrees": True,
                            "release_date_agrees": True,
                        },
                    }
                ],
            },
        )

    def test_complete_default_grid_has_literal_dispositions_and_no_compression(self):
        reports, records = generated("2009-05-30", "2025-10-18")
        result = ledger.build_first_report_ledger(reports, records)
        self.assertEqual(result["status"], STATUS)
        self.assertEqual(len(result["rows"]), 856)
        self.assertEqual(
            result["counts"],
            {
                "observed": 851,
                "unresolved_correction": 1,
                "no_admitted_release": 4,
            },
        )
        weeks = [date.fromisoformat(r["reference_week"]) for r in result["rows"]]
        self.assertEqual(weeks[0], date(2009, 5, 30))
        self.assertEqual(weeks[-1], date(2025, 10, 18))
        self.assertTrue(all(b - a == timedelta(days=7) for a, b in zip(weeks, weeks[1:])))

    def test_input_order_does_not_change_sorted_grid(self):
        reports, records = generated("2018-11-03", "2018-11-24")
        a = self.build(reports, records, "2018-11-03", "2018-11-24")
        b = self.build(reports[::-1], records[1:] + records[:1], "2018-11-03", "2018-11-24")
        self.assertEqual(a, b)

    def test_2018_original_value_and_date_are_retained_with_both_disagreements(self):
        reports, records = generated("2018-03-17", "2018-03-17")
        row = self.build(reports, records, "2018-03-17")["rows"][0]
        self.assertEqual(
            (row["first_report_value"], row["release_date"], row["status"]),
            (229000, "2018-03-22", "observed"),
        )
        self.assertEqual(
            row["source_comparison"],
            {
                "kind": "documented_original_dol_disagreement",
                "dol_value": 229000,
                "alfred_value": 227000,
                "alfred_realtime_start_date": "2018-03-29",
                "alfred_source_line": 2,
                "value_agrees": False,
                "release_date_agrees": False,
            },
        )
        self.assertEqual(row["source_url"], reports[0]["source_url"])
        self.assertEqual(row["release_text_sha256"], reports[0]["release_text_sha256"])

    def test_2017_model_value_is_unknown_but_release_date_and_audit_survive(self):
        reports, records = generated("2017-03-18", "2017-03-18")
        row = self.build(reports, records, "2017-03-18")["rows"][0]
        self.assertIsNone(row["first_report_value"])
        self.assertEqual(row["release_date"], "2017-03-23")
        self.assertEqual(row["status"], "unresolved_correction")
        self.assertEqual(
            row["source_comparison"],
            {
                "kind": "unresolved_correction",
                "dol_value": 261000,
                "alfred_value": 258000,
                "alfred_realtime_start_date": "2017-03-23",
                "alfred_source_line": 2,
                "value_agrees": False,
                "release_date_agrees": True,
            },
        )
        self.assertEqual(row["source_url"], reports[0]["source_url"])
        self.assertEqual(row["release_text_sha256"], reports[0]["release_text_sha256"])

    def test_each_exception_requires_all_literal_counts_and_dates(self):
        for week in ("2017-03-18", "2018-03-17"):
            original_reports, original_records = generated(week, week)
            for side, field, replacement in (
                ("dol", "value", 123456),
                (
                    "dol",
                    "actual_release_date",
                    (date.fromisoformat(week) + timedelta(days=6)).isoformat(),
                ),
                ("alfred", "value", original_reports[0]["value"]),
                (
                    "alfred",
                    "alfred_realtime_start_date",
                    (date.fromisoformat(week) + timedelta(days=6)).isoformat(),
                ),
                ("alfred", "value", None),
            ):
                reports, records = copy.deepcopy((original_reports, original_records))
                row = reports[0] if side == "dol" else records[0]
                row[field] = replacement
                if side == "dol" and field == "actual_release_date":
                    reports[0] = report(week, replacement, row["value"])
                if side == "alfred" and replacement is None:
                    row["status"] = "missing"
                with (
                    self.subTest(week=week, side=side, field=field),
                    self.assertRaises(ValueError),
                ):
                    self.build(reports, records, week)

    def test_documented_counts_do_not_create_exceptions_on_other_weeks(self):
        for dol, other in ((229000, 227000), (261000, 258000)):
            with self.subTest(dol=dol), self.assertRaises(ValueError):
                self.build([report(value=dol)], [alfred(value=other)])

    def test_2018_delayed_alfred_clock_can_equal_next_weeks_clock(self):
        reports, records = generated("2018-03-17", "2018-03-24")
        self.assertEqual(
            records[0]["alfred_realtime_start_date"], records[1]["alfred_realtime_start_date"]
        )
        result = self.build(reports, records, "2018-03-17", "2018-03-24")
        self.assertEqual(result["counts"]["observed"], 2)
        self.assertEqual(
            [r["release_date"] for r in result["rows"]], ["2018-03-22", "2018-03-29"]
        )

    def test_four_declared_tail_weeks_preserve_absence_without_a_release_date(self):
        result = self.build([], [], TAIL[0], TAIL[-1])
        self.assertEqual([r["reference_week"] for r in result["rows"]], list(TAIL))
        self.assertEqual(result["counts"]["no_admitted_release"], 4)
        for row in result["rows"]:
            self.assertEqual(row["status"], "no_admitted_release")
            for field in (
                "release_date",
                "first_report_value",
                "source_url",
                "release_text_sha256",
            ):
                self.assertIsNone(row[field])
            self.assertEqual(row["source_comparison"]["kind"], "no_admitted_release")
            self.assertTrue(
                all(v is None for k, v in row["source_comparison"].items() if k != "kind")
            )

    def test_tail_cannot_acquire_a_report_or_alfred_value_or_missing_record(self):
        for week in TAIL[:3]:
            released = (date.fromisoformat(week) + timedelta(days=5)).isoformat()
            r, a = report(week, released), alfred(week, released)
            for reports, records in (
                ([r], []),
                ([], [a]),
                ([r], [a]),
                ([], [alfred(week, released, None)]),
            ):
                with (
                    self.subTest(week=week, reports=bool(reports), records=bool(records)),
                    self.assertRaises(ValueError),
                ):
                    self.build(reports, records, week)

    def test_interior_and_boundary_gaps_are_rejected_not_dropped(self):
        original_reports, original_records = generated("2018-11-03", "2018-11-17")
        for position in range(3):
            for side in ("both", "dol", "alfred"):
                reports, records = copy.deepcopy((original_reports, original_records))
                if side in ("both", "dol"):
                    del reports[position]
                if side in ("both", "alfred"):
                    del records[position]
                with self.subTest(position=position, side=side), self.assertRaises(ValueError):
                    self.build(reports, records, "2018-11-03", "2018-11-17")

    def test_missing_exception_sources_are_not_treated_as_documented_absence(self):
        for week in ("2017-03-18", "2018-03-17"):
            reports, records = generated(week, week)
            for rs, ars in (([], records), (reports, []), ([], [])):
                with (
                    self.subTest(week=week, sides=(bool(rs), bool(ars))),
                    self.assertRaises(ValueError),
                ):
                    self.build(rs, ars, week)

    def test_ordinary_alfred_week_value_date_and_missingness_must_agree(self):
        for row in (
            alfred(value=123457),
            alfred(released="2018-11-23"),
            alfred(value=None),
            alfred(week="2018-11-10"),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.build(records=[row])

    def test_out_of_scope_rows_are_rejected_on_each_side(self):
        for week, released in (("2018-11-10", "2018-11-15"), ("2018-11-24", "2018-11-29")):
            for rs, ars in (
                ([report(), report(week, released)], [alfred()]),
                ([report()], [alfred(), alfred(week, released)]),
            ):
                with (
                    self.subTest(week=week, lengths=(len(rs), len(ars))),
                    self.assertRaises(ValueError),
                ):
                    self.build(rs, ars)

    def test_duplicate_reference_weeks_reject_identical_and_conflicting_sources(self):
        for conflicting in (False, True):
            extra_report, extra_record = report(), alfred()
            if conflicting:
                extra_report["value"] += 1
                extra_record["value"] += 1
            for rs, ars in (
                ([report(), extra_report], [alfred()]),
                ([report()], [alfred(), extra_record]),
            ):
                with (
                    self.subTest(conflicting=conflicting, sides=(len(rs), len(ars))),
                    self.assertRaises(ValueError),
                ):
                    self.build(rs, ars)

    def test_duplicate_dol_release_date_is_ambiguous(self):
        reports = [report("2018-11-10"), report()]
        records = [alfred("2018-11-10"), alfred()]
        with self.assertRaises(ValueError):
            self.build(reports, records, "2018-11-10", "2018-11-17")

    def test_every_report_requires_an_explicit_matching_body_date(self):
        for dates, verified in (
            ([], False),
            ([], True),
            (["2018-11-23"], True),
            (["2018-11-22"], False),
            (["2018-11-22"] * 2, True),
            (["2018-11-22"], 1),
        ):
            bad = report()
            bad["body_release_dates"], bad["body_release_date_verified"] = dates, verified
            with self.subTest(dates=dates, verified=verified), self.assertRaises(ValueError):
                self.build([bad])

    def test_dated_url_identity_and_text_digest_are_required(self):
        for field, value in (
            ("source_url", "https://example.com/20181122.pdf"),
            ("source_url", report()["source_url"].replace("112218", "112118")),
            ("release_text_sha256", "a" * 63),
            ("release_text_sha256", "G" * 64),
            ("statistic", "continued_claims"),
        ):
            bad = report()
            bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.build([bad])

    def test_noncalendar_dol_identity_remains_valid_with_matching_body_date(self):
        bad_filename = report("2018-12-29", "2019-01-03")
        bad_filename["source_url"] = "https://oui.doleta.gov/press/2019/010318.pdf"
        opa = report("2019-10-12", "2019-10-17")
        opa["source_url"] = (
            "https://www.dol.gov/sites/dolgov/files/OPA/newsreleases/ui-claims/20191832.pdf"
        )
        for r in (bad_filename, opa):
            week, released = r["observation_date"], r["actual_release_date"]
            with self.subTest(week=week):
                result = self.build([r], [alfred(week, released)], week)
                self.assertEqual(result["rows"][0]["source_url"], r["source_url"])

    def test_missing_extra_schema_keys_and_nonlist_inputs_are_rejected(self):
        for side in ("dol", "alfred"):
            for change in ("missing", "extra"):
                r, a = report(), alfred()
                row = r if side == "dol" else a
                if change == "missing":
                    del row["value"]
                else:
                    row["invented_metadata"] = True
                with self.subTest(side=side, change=change), self.assertRaises(ValueError):
                    self.build([r], [a])
        for rs, ars in (((), []), ([], ()), ({}, []), ([], {})):
            with self.subTest(types=(type(rs), type(ars))), self.assertRaises(ValueError):
                self.build(rs, ars)

    def test_nonpositive_nonintegral_boolean_and_nonfinite_counts_are_invalid(self):
        for value in (True, False, 0, -1, 123456.0, float("nan"), float("inf"), "123456"):
            for side in ("dol", "alfred"):
                r, a = report(), alfred()
                (r if side == "dol" else a)["value"] = value
                with self.subTest(side=side, value=value), self.assertRaises(ValueError):
                    self.build([r], [a])

    def test_bounds_and_all_source_clocks_are_strict_iso_saturdays_and_ceiling(self):
        for start, end in (
            ("2018-11-18", "2018-11-24"),
            ("2018-11-17", "2018-11-23"),
            ("2018-11-24", "2018-11-17"),
            ("2009-05-23", "2009-05-30"),
            ("2025-10-18", "2025-10-25"),
            (date(2018, 11, 17), "2018-11-17"),
            ("2018-1-17", "2018-11-17"),
        ):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.build([], [], start, end)
        for side, field, value in (
            ("dol", "observation_date", "2018-11-18"),
            ("dol", "actual_release_date", "2025-10-21"),
            ("alfred", "alfred_realtime_start_date", "2025-10-21"),
            ("alfred", "alfred_realtime_start_date", "2018-11-16"),
            ("alfred", "source_line", True),
            ("alfred", "source_line", 1),
        ):
            r, a = report(), alfred()
            (r if side == "dol" else a)[field] = value
            with self.subTest(side=side, field=field), self.assertRaises(ValueError):
                self.build([r], [a])

    def test_no_input_mutation_or_output_aliases_on_success_and_failure(self):
        reports, records = generated("2018-03-17", "2018-03-24")
        before = copy.deepcopy((reports, records))
        output = self.build(reports, records, "2018-03-17", "2018-03-24")
        self.assertEqual((reports, records), before)
        output["rows"][0]["source_comparison"]["alfred_value"] = 1
        output["rows"][0]["first_report_value"] = 1
        self.assertEqual((reports, records), before)
        records[-1]["value"] += 1
        before = copy.deepcopy((reports, records))
        with self.assertRaises(ValueError):
            self.build(reports, records, "2018-03-17", "2018-03-24")
        self.assertEqual((reports, records), before)


if __name__ == "__main__":
    unittest.main()
