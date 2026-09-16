"""Prewritten generated regressions for three observed extraction-format classes."""

import hashlib
import unittest

from src import claims_release_reconcile as reconcile

RELEASE = "2024-06-20"
URL = "https://oui.doleta.gov/press/2024/062024.pdf"
OPA = "https://www.dol.gov/sites/dolgov/files/OPA/newsreleases/ui-claims/20249999.pdf"


def fixture(reference="June 15,", advance="advance", header="June 20, 2024"):
    return (
        f"For release at 8:30 A.M. Thursday, {header}\n"
        "UNEMPLOYMENT INSURANCE WEEKLY CLAIMS\nSEASONALLY ADJUSTED DATA\n"
        f"In the week ending {reference} the {advance} figure for seasonally adjusted "
        "initial claims was 456,789, a decrease of 11,111 from the previous week's "
        "revised level of 467,900. The advance seasonally adjusted insured unemployment "
        "rate was 1.7 percent."
    )


class ClaimsReleaseExtractionFormatTests(unittest.TestCase):
    def assert_literal_report(self, text, source_url=URL):
        report = reconcile.parse_release_text(text, RELEASE, source_url)
        self.assertEqual(report["observation_date"], "2024-06-15")
        self.assertEqual(report["actual_release_date"], RELEASE)
        self.assertEqual(report["value"], 456789)
        self.assertEqual(report["body_release_dates"], [RELEASE])
        self.assertIs(report["body_release_date_verified"], True)
        self.assertEqual(
            report["release_text_sha256"], hashlib.sha256(text.encode()).hexdigest()
        )
        self.assertEqual(report["source_url"], source_url)

    def test_reference_day_comma_is_optional_without_erasing_boundaries(self):
        for reference in ("June 15,", "June 15", "June 15\n", "June 15, 2024,"):
            with self.subTest(reference=reference):
                self.assert_literal_report(fixture(reference=reference))

    def test_only_whitespace_may_split_the_exact_headline_word_advance(self):
        for advance in (
            "advance",
            "a dvance",
            "adv ance",
            "a\ndvance",
            "adv\tance",
            "a\u00a0dvance",
            "a d v a n c e",
        ):
            with self.subTest(advance=advance):
                self.assert_literal_report(fixture(advance=advance))

    def test_body_date_space_before_comma_preserves_explicit_evidence(self):
        for header in ("June 20 , 2024", "Jun. 20\n, 2024", "June 20\t,\n2024"):
            for source_url in (URL, OPA):
                with self.subTest(header=header, source_url=source_url):
                    self.assert_literal_report(fixture(header=header), source_url)

    def test_formats_compose_without_changing_first_count_or_source_text_hash(self):
        self.assert_literal_report(
            fixture(reference="June 15", advance="adv\n ance", header="June 20 , 2024"),
            OPA,
        )

    def test_newly_recognized_wrong_or_conflicting_body_dates_fail(self):
        for source_url in (URL, OPA):
            for text in (
                fixture(header="June 21 , 2024"),
                fixture() + "\nRelease Date: June 21 , 2024",
                fixture(header="June 20 , 2023"),
            ):
                with (
                    self.subTest(source_url=source_url, text=text[:75]),
                    self.assertRaises(ValueError),
                ):
                    reconcile.parse_release_text(text, RELEASE, source_url)

    def test_recognized_body_weekday_still_must_match_spaced_date(self):
        text = fixture(header="June 20 , 2024").replace("Thursday", "Friday")
        with self.assertRaises(ValueError):
            reconcile.parse_release_text(text, RELEASE, URL)

    def test_new_format_headlines_are_counted_for_ambiguity(self):
        first = fixture()
        for second in (
            fixture(reference="June 15"),
            fixture(advance="a dvance"),
            fixture(reference="June 15", advance="adv ance").replace("456,789", "555,444"),
        ):
            with self.subTest(second=second[:75]), self.assertRaises(ValueError):
                reconcile.parse_release_text(first + "\n" + second, RELEASE, URL)

    def test_letter_punctuation_and_other_word_repairs_are_not_allowed(self):
        for text in (
            fixture(advance="a-dvance"),
            fixture(advance="adv.ance"),
            fixture(advance="advice"),
            fixture(advance="advanced"),
            fixture(advance="a dv an ceX"),
            fixture().replace("seasonally", "season ally"),
            fixture().replace("initial", "ini tial"),
        ):
            with self.subTest(text=text[-180:]), self.assertRaises(ValueError):
                reconcile.parse_release_text(text, RELEASE, URL)

    def test_revised_continued_and_unadjusted_substitutions_still_fail(self):
        base = fixture(reference="June 15", advance="a dvance")
        for text in (
            base.replace("a dvance figure", "revised figure"),
            base.replace("seasonally adjusted initial", "seasonally adjusted continued"),
            base.replace("seasonally adjusted initial", "unadjusted initial"),
            base.replace("In the week ending", "In the previous week ending"),
        ):
            with self.subTest(text=text[150:240]), self.assertRaises(ValueError):
                reconcile.parse_release_text(text, RELEASE, URL)

    def test_format_support_does_not_repair_invalid_dates_or_delimiters(self):
        for reference in (
            "June 14",
            "June 32",
            "June 15:",
            "June 15;",
            "June 15,,",
            "June 15 2024,",
            "June 15, 2024",
        ):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                reconcile.parse_release_text(fixture(reference=reference), RELEASE, URL)
        with self.assertRaises(ValueError):
            reconcile.parse_release_text(
                fixture(reference="June 15").replace("15 the", "15the"), RELEASE, URL
            )

    def test_format_support_does_not_relax_number_validation(self):
        for count in ("0", "-12", "45,67", "45.5", "NaN", "Infinity", "4e5"):
            text = fixture(reference="June 15", advance="adv ance").replace("456,789", count)
            with self.subTest(count=count), self.assertRaises(ValueError):
                reconcile.parse_release_text(text, RELEASE, URL)

    def test_dummy_file_has_no_headline_and_cannot_become_a_report(self):
        with self.assertRaises(ValueError):
            reconcile.parse_release_text(
                "Dummy file: June 20, 2024\nFile name: 062024.pdf", RELEASE, URL
            )


if __name__ == "__main__":
    unittest.main()
