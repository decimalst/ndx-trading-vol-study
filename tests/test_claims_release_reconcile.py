"""Invented release text only; expected values do not use a source value parser."""

import copy
import hashlib
import unittest
from datetime import date

from src import claims_release_reconcile as reconcile


def url(released):
    day = date.fromisoformat(released)
    return f"https://oui.doleta.gov/press/{day.year}/{day:%m%d%y}.pdf"


def release_text(value="246,000", week="November 17", header=True):
    prefix = (
        "For release at 8:30 a.m. (Eastern) Wednesday, November 21, 2018\n" if header else ""
    )
    return (
        prefix + "UNEMPLOYMENT INSURANCE WEEKLY CLAIMS\nSEASONALLY ADJUSTED DATA\n"
        f"In the week ending {week}, the advance figure for seasonally adjusted "
        f"initial claims was {value}, a decrease of 7,000 from the previous week's "
        "revised level. The previous week's level was revised from 249,000 to 253,000.\n"
        "The advance number for seasonally adjusted insured unemployment during "
        "the week ending November 10 was 1,765,000.\n"
        "UNADJUSTED DATA\nThe advance number of actual initial claims was 231,000."
    )


def report_fixture(week="2018-11-17", released="2018-11-21", value=246000):
    return {
        "statistic": "advance_seasonally_adjusted_initial_claims",
        "observation_date": week,
        "actual_release_date": released,
        "value": value,
        "source_url": url(released),
        "release_text_sha256": "a" * 64,
        "body_release_dates": [released],
        "body_release_date_verified": True,
    }


def record_fixture(week="2018-11-17", vintage="2018-11-21", value=246000, line=2):
    return {
        "observation_date": week,
        "alfred_realtime_start_date": vintage,
        "value": value,
        "status": "missing" if value is None else "observed",
        "source_line": line,
    }


class ReleaseTextTests(unittest.TestCase):
    def parse(self, text=None, released="2018-11-21", source_url=None):
        return reconcile.parse_release_text(
            release_text() if text is None else text,
            released,
            url(released) if source_url is None else source_url,
        )

    def test_exact_initial_headline_and_separate_body_date_evidence(self):
        text = release_text()
        expected = report_fixture()
        expected["release_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        self.assertEqual(self.parse(text), expected)
        self.assertNotIn("alfred_realtime_start_date", self.parse(text))
        self.assertNotIn("ingestion_date", self.parse(text))

    def test_extracted_whitespace_case_and_thousands_separators(self):
        variants = [
            release_text().upper(),
            release_text().replace("seasonally adjusted", "seasonally\nadjusted"),
            release_text("246,\n000"),
            release_text("246 000"),
            release_text("246\u00a0000"),
            release_text("246\u202f000"),
            release_text("246000"),
        ]
        for text in variants:
            with self.subTest(text=text[:50]):
                self.assertEqual(self.parse(text)["value"], 246000)

    def test_prior_revised_continued_and_unadjusted_values_are_not_substituted(self):
        text = release_text("731,000")
        self.assertEqual(self.parse(text)["value"], 731000)
        for words in (
            "the revised figure for seasonally adjusted initial claims",
            "the advance figure for not seasonally adjusted initial claims",
            "the advance figure for seasonally adjusted continued claims",
            "the previous week's figure for seasonally adjusted initial claims",
        ):
            changed = text.replace(
                "the advance figure for seasonally adjusted initial claims", words
            )
            with self.subTest(words=words), self.assertRaises(ValueError):
                self.parse(changed)
        with self.assertRaises(ValueError):
            self.parse(text.replace("In the week ending", "In the previous week ending"))
        with self.assertRaises(ValueError):
            self.parse(text.replace("was 731,000", "was revised to 731,000"))

    def test_duplicate_or_conflicting_initial_headlines_are_ambiguous(self):
        for second in [
            release_text(),
            release_text("733,000"),
            release_text(week="November 10"),
        ]:
            with self.subTest(second=second[-30:]), self.assertRaises(ValueError):
                self.parse(release_text() + "\n" + second)

    def test_malformed_nonfinite_nonintegral_and_nonpositive_counts_fail(self):
        for token in [
            "24,60",
            "246,00",
            "2,46,000",
            "246 00",
            "2 46 000",
            "246000.5",
            "246000.0",
            "1e3",
            "NaN",
            "Infinity",
            "-Infinity",
            "0",
            "0,000",
            "-1",
            "+246000",
            ".",
            "",
            "246,000garbage",
        ]:
            with self.subTest(token=token), self.assertRaises(ValueError):
                self.parse(release_text(token))

    def test_explicit_week_year_and_year_rollover(self):
        text = release_text(week="December 29, 2018", header=False)
        self.assertEqual(self.parse(text, "2019-01-03")["observation_date"], "2018-12-29")
        text = release_text(week="December 29", header=False)
        self.assertEqual(self.parse(text, "2019-01-03")["observation_date"], "2018-12-29")
        self.assertEqual(
            self.parse(release_text(week="November 17, 2018"))["observation_date"],
            "2018-11-17",
        )

    def test_missing_year_does_not_search_older_years_for_a_saturday(self):
        # December29,2017 was Friday; the old Saturday in2012 must not be chosen.
        with self.assertRaises(ValueError):
            self.parse(release_text(week="December 29", header=False), "2018-01-04")

    def test_week_dates_must_be_real_saturdays_strictly_before_release(self):
        for week in [
            "November 18",
            "November 31",
            "February 30",
            "November 24, 2018",
            "November 17, 2019",
        ]:
            with self.subTest(week=week), self.assertRaises(ValueError):
                self.parse(release_text(week=week))
        with self.assertRaises(ValueError):
            self.parse(release_text(week="November 17", header=False), "2018-11-17")

    def test_wednesday_release_allowed_but_wrong_body_date_or_weekday_rejected(self):
        self.assertEqual(self.parse()["actual_release_date"], "2018-11-21")
        for old, new in [
            ("November 21, 2018", "November 22, 2018"),
            ("Wednesday, November 21", "Thursday, November 21"),
        ]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.parse(release_text().replace(old, new))
        conflicting = release_text() + "\nRelease Date: November 22, 2018\n"
        with self.assertRaises(ValueError):
            self.parse(conflicting)

    def test_absent_body_header_is_disclosed_and_other_header_forms_checked(self):
        result = self.parse(release_text(header=False))
        self.assertEqual(result["body_release_dates"], [])
        self.assertIs(result["body_release_date_verified"], False)
        for header in [
            "Release Date: November 21, 2018\n",
            "Transmission of material is embargoed until 8:30 A.M. (Eastern) Wednesday, November 21, 2018\n",
        ]:
            with self.subTest(header=header):
                self.assertEqual(
                    self.parse(header + release_text(header=False))["body_release_dates"],
                    ["2018-11-21"],
                )

    def test_url_must_be_official_dated_identity_matching_release(self):
        valid = url("2018-11-21")
        self.assertEqual(self.parse(source_url=valid[:-3] + "asp")["value"], 246000)
        for bad in [
            valid.replace("https:", "http:"),
            valid.replace("oui.doleta.gov", "evil.test"),
            valid.replace("/2018/", "/2017/"),
            valid.replace("112118", "112218"),
            valid + "?x=1",
            valid + "#x",
            valid.replace("gov/", "gov:443/"),
            valid.replace("https://", "https://user@"),
            "https://www.dol.gov/ui/data.pdf",
        ]:
            with self.subTest(url=bad), self.assertRaises(ValueError):
                self.parse(source_url=bad)

    def test_release_ceiling_and_strict_input_types(self):
        for released in ["2025-10-21", "2018-11-21T08:30:00", "11/21/2018", "2018-2-1"]:
            with self.subTest(released=released), self.assertRaises(ValueError):
                reconcile.parse_release_text(release_text(), released, url("2018-11-21"))
        for text, released, link in [
            (None, "2018-11-21", url("2018-11-21")),
            (b"text", "2018-11-21", url("2018-11-21")),
            ("", "2018-11-21", url("2018-11-21")),
            (release_text(), date(2018, 11, 21), url("2018-11-21")),
            (release_text(), "2018-11-21", None),
        ]:
            with (
                self.subTest(text=type(text), released=released),
                self.assertRaises(ValueError),
            ):
                reconcile.parse_release_text(text, released, link)


class ReconciliationTests(unittest.TestCase):
    def test_all_records_exactly_reconciled_with_distinct_clock_fields(self):
        records = [
            record_fixture("2018-11-10", "2018-11-15", 250000, 2),
            record_fixture(line=3),
        ]
        reports = [report_fixture("2018-11-10", "2018-11-15", 250000), report_fixture()]
        result = reconcile.reconcile_reports(records[::-1], reports)
        self.assertEqual(result["status"], "RECONCILED")
        self.assertEqual(
            (result["records_count"], result["reports_count"], result["matched_count"]),
            (2, 2, 2),
        )
        self.assertEqual(result["issues"], [])
        self.assertEqual(
            [r["observation_date"] for r in result["matched"]], ["2018-11-10", "2018-11-17"]
        )
        self.assertEqual(result["matched"][1]["actual_release_date"], "2018-11-21")
        self.assertEqual(result["matched"][1]["alfred_realtime_start_date"], "2018-11-21")
        self.assertEqual(reconcile.reconcile_reports(records, reports[::-1]), result)

    def test_date_and_value_mismatches_are_explicit_and_never_silently_replaced(self):
        record = record_fixture(vintage="2018-11-22", value=253000)
        before = copy.deepcopy(record)
        result = reconcile.reconcile_reports([record], [report_fixture()])
        self.assertEqual(result["status"], "RECONCILIATION_MISMATCH")
        self.assertEqual(result["matched_count"], 0)
        self.assertEqual(
            {x["kind"] for x in result["issues"]}, {"release_date_mismatch", "value_mismatch"}
        )
        self.assertEqual(record, before)

    def test_both_sided_coverage_and_explicit_missing_values(self):
        record = record_fixture(value=None)
        result = reconcile.reconcile_reports([record], [report_fixture()])
        self.assertEqual(result["status"], "RECONCILIATION_MISMATCH")
        self.assertEqual([x["kind"] for x in result["issues"]], ["alfred_value_unavailable"])
        self.assertEqual(result["matched"], [])
        for records, reports, kind in [
            ([record_fixture()], [], "missing_dol_report"),
            ([], [report_fixture()], "extra_dol_report"),
        ]:
            with self.subTest(kind=kind):
                result = reconcile.reconcile_reports(records, reports)
                self.assertEqual(result["status"], "RECONCILIATION_MISMATCH")
                self.assertIn(kind, [x["kind"] for x in result["issues"]])

    def test_interior_week_omitted_on_both_sides_is_not_complete(self):
        records = [
            record_fixture("2018-11-03", "2018-11-08", 111000, 2),
            record_fixture(line=3),
        ]
        reports = [report_fixture("2018-11-03", "2018-11-08", 111000), report_fixture()]
        result = reconcile.reconcile_reports(records, reports)
        self.assertEqual(result["status"], "RECONCILIATION_MISMATCH")
        self.assertEqual(result["matched_count"], 2)
        self.assertIn(
            {"kind": "missing_reference_week", "observation_date": "2018-11-10"},
            result["issues"],
        )

    def test_duplicate_weeks_and_release_dates_are_invalid(self):
        for records, reports in [
            ([record_fixture()] * 2, [report_fixture()]),
            ([record_fixture()], [report_fixture()] * 2),
            ([record_fixture()], [report_fixture(), report_fixture("2018-11-10")]),
            ([record_fixture(), record_fixture("2018-11-10", line=3)], [report_fixture()]),
        ]:
            with (
                self.subTest(records=len(records), reports=len(reports)),
                self.assertRaises(ValueError),
            ):
                reconcile.reconcile_reports(records, reports)

    def test_malformed_records_cannot_be_dropped(self):
        for field, value in [
            ("value", True),
            ("value", 246000.0),
            ("value", float("inf")),
            ("value", 0),
            ("status", "revised"),
            ("source_line", True),
            ("observation_date", "2018-11-18"),
            ("alfred_realtime_start_date", "2025-10-21"),
        ]:
            bad = record_fixture()
            bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                reconcile.reconcile_reports([bad], [report_fixture()])

    def test_malformed_reports_cannot_be_used_as_independent_confirmation(self):
        for field, value in [
            ("statistic", "continued_claims"),
            ("value", True),
            ("value", 246000.0),
            ("release_text_sha256", "bad"),
            ("actual_release_date", "2018-11-22"),
            ("body_release_date_verified", "true"),
            ("body_release_dates", ["2018-11-22"]),
        ]:
            bad = report_fixture()
            bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                reconcile.reconcile_reports([record_fixture()], [bad])

    def test_empty_scope_is_explicitly_incomplete(self):
        result = reconcile.reconcile_reports([], [])
        self.assertEqual(result["status"], "RECONCILIATION_MISMATCH")
        self.assertEqual(result["issues"], [{"kind": "empty_scope"}])


if __name__ == "__main__":
    unittest.main()
