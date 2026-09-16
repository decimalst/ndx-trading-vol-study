"""Prewritten URL/body identity regressions; all headline values are invented."""

import copy
import json
import unittest

from src import claims_release_reconcile as reconcile

OPA = "https://www.dol.gov/sites/dolgov/files/OPA/newsreleases/ui-claims/20199999.pdf"
EXCEPTION = "https://oui.doleta.gov/press/2019/010318.pdf"


def text(header=True):
    return (
        ("For release at 8:30 A.M. (Eastern) Thursday, January 3, 2019\n" if header else "")
        + "UNEMPLOYMENT INSURANCE WEEKLY CLAIMS\nSEASONALLY ADJUSTED DATA\n"
        + "In the week ending December 29, the advance figure for seasonally adjusted "
        + "initial claims was 345,000, an increase of 2,000 from the previous week's revised level."
    )


def source_record():
    return {
        "observation_date": "2018-12-29",
        "alfred_realtime_start_date": "2019-01-03",
        "value": 345000,
        "status": "observed",
        "source_line": 2,
    }


def report_record(source_url):
    return {
        "statistic": "advance_seasonally_adjusted_initial_claims",
        "observation_date": "2018-12-29",
        "actual_release_date": "2019-01-03",
        "value": 345000,
        "source_url": source_url,
        "release_text_sha256": "b" * 64,
        "body_release_dates": ["2019-01-03"],
        "body_release_date_verified": True,
    }


class ClaimsReleaseIdentityTests(unittest.TestCase):
    def test_opa_numeric_identity_requires_body_date_and_uses_release_year(self):
        parsed = reconcile.parse_release_text(text(), "2019-01-03", OPA)
        self.assertEqual(parsed["source_url"], OPA)
        self.assertEqual(parsed["value"], 345000)
        self.assertEqual(parsed["observation_date"], "2018-12-29")
        self.assertEqual(parsed["actual_release_date"], "2019-01-03")
        self.assertEqual(parsed["body_release_dates"], ["2019-01-03"])
        self.assertIs(parsed["body_release_date_verified"], True)
        transported = json.loads(json.dumps(parsed))
        self.assertEqual(
            reconcile.reconcile_reports([source_record()], [transported])["status"],
            "RECONCILED",
        )

    def test_exact_observed_2019_filename_exception_is_retained_without_rewriting(self):
        parsed = reconcile.parse_release_text(text(), "2019-01-03", EXCEPTION)
        self.assertEqual(parsed["source_url"], EXCEPTION)
        self.assertEqual(parsed["observation_date"], "2018-12-29")
        self.assertEqual(parsed["value"], 345000)
        self.assertEqual(parsed["body_release_dates"], ["2019-01-03"])
        result = reconcile.reconcile_reports([source_record()], [parsed])
        self.assertEqual(result["status"], "RECONCILED")
        self.assertEqual(result["matched"][0]["source_url"], EXCEPTION)

    def test_both_noncalendar_identity_classes_reject_absent_body_dates(self):
        for source_url in (OPA, EXCEPTION):
            with self.subTest(source_url=source_url), self.assertRaises(ValueError):
                reconcile.parse_release_text(text(header=False), "2019-01-03", source_url)

    def test_both_identity_classes_reject_conflicting_or_wrong_body_dates(self):
        for source_url in (OPA, EXCEPTION):
            for body in (
                text().replace("Thursday, January 3", "Friday, January 4"),
                text() + "\nRelease Date: January 4, 2019\n",
            ):
                with (
                    self.subTest(source_url=source_url, body=body[:35]),
                    self.assertRaises(ValueError),
                ):
                    reconcile.parse_release_text(body, "2019-01-03", source_url)

    def test_serialized_noncalendar_reports_cannot_drop_body_date_evidence(self):
        for source_url in (OPA, EXCEPTION):
            valid = report_record(source_url)
            self.assertEqual(
                reconcile.reconcile_reports([source_record()], [valid])["status"],
                "RECONCILED",
            )
            for dates, flag in (
                ([], False),
                ([], True),
                (["2019-01-04"], True),
                (["2019-01-03"], False),
                (["2019-01-03"], 1),
            ):
                bad = copy.deepcopy(valid)
                bad["body_release_dates"], bad["body_release_date_verified"] = dates, flag
                with (
                    self.subTest(source_url=source_url, dates=dates, flag=flag),
                    self.assertRaises(ValueError),
                ):
                    reconcile.reconcile_reports([source_record()], [bad])

    def test_opa_host_path_numeric_year_and_ceiling_remain_strict(self):
        for wrong in (
            OPA.replace("https:", "http:"),
            OPA.replace("www.dol.gov", "dol.gov"),
            OPA.replace("www.dol.gov", "evil.example"),
            OPA.replace("https://", "https://user@"),
            OPA.replace("gov/", "gov:443/"),
            OPA + "?download=1",
            OPA + "#page=1",
            OPA.replace("/OPA/", "/opa/"),
            OPA.replace("/ui-claims/", "/other/"),
            OPA.replace("20199999", "20189999"),
            OPA.replace("20199999", "2019"),
            OPA.replace("20199999", "2019x999"),
            OPA.replace(".pdf", ".asp"),
        ):
            with self.subTest(url=wrong), self.assertRaises(ValueError):
                reconcile.parse_release_text(text(), "2019-01-03", wrong)
        with self.assertRaises(ValueError):
            reconcile.parse_release_text(
                text(), "2025-10-21", OPA.replace("20199999", "20259999")
            )

    def test_no_other_mistyped_filename_or_exception_release_date_is_allowed(self):
        for wrong in (
            EXCEPTION.replace("010318", "010418"),
            EXCEPTION.replace("010318", "010218"),
            EXCEPTION.replace("/2019/", "/2018/"),
            EXCEPTION.replace(".pdf", ".asp"),
            EXCEPTION + "?x=1",
            EXCEPTION + "#x",
        ):
            with self.subTest(url=wrong), self.assertRaises(ValueError):
                reconcile.parse_release_text(text(), "2019-01-03", wrong)
        with self.assertRaises(ValueError):
            reconcile.parse_release_text(
                text().replace("Thursday, January 3", "Friday, January 4"),
                "2019-01-04",
                EXCEPTION,
            )

    def test_ordinary_calendar_url_keeps_existing_optional_body_date_contract(self):
        ordinary = "https://oui.doleta.gov/press/2019/011019.pdf"
        parsed = reconcile.parse_release_text(
            text(header=False).replace("December 29", "January 5"), "2019-01-10", ordinary
        )
        self.assertEqual(parsed["body_release_dates"], [])
        self.assertIs(parsed["body_release_date_verified"], False)
        self.assertEqual(parsed["source_url"], ordinary)


if __name__ == "__main__":
    unittest.main()
