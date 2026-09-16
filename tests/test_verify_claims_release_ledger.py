"""Prewritten independent-ledger regressions; ordinary numbers are invented."""

import copy
import unittest

from src.verify_claims_release_ledger import verify_ledger


def report(week="2024-06-15", released="2024-06-20", value=456789, digest="a" * 64):
    return {
        "statistic": "advance_seasonally_adjusted_initial_claims",
        "observation_date": week,
        "actual_release_date": released,
        "value": value,
        "source_url": "https://oui.doleta.gov/press/"
        + released[:4]
        + "/"
        + released[5:7]
        + released[8:10]
        + released[2:4]
        + ".pdf",
        "release_text_sha256": digest,
        "body_release_dates": [released],
        "body_release_date_verified": True,
    }


def alfred(week="2024-06-15", released="2024-06-20", value=456789, line=2):
    return {
        "observation_date": week,
        "alfred_realtime_start_date": released,
        "value": value,
        "status": "observed",
        "source_line": line,
    }


def ordinary():
    ledger = {
        "status": "FIRST_REPORT_LEDGER_BUILT_NOT_INDEPENDENTLY_VERIFIED",
        "start": "2024-06-15",
        "end": "2024-06-22",
        "rows": [
            {
                "reference_week": "2024-06-15",
                "release_date": "2024-06-20",
                "first_report_value": 456789,
                "status": "observed",
                "source_url": "https://oui.doleta.gov/press/2024/062024.pdf",
                "release_text_sha256": "a" * 64,
                "source_comparison": {
                    "kind": "exact_agreement",
                    "dol_value": 456789,
                    "alfred_value": 456789,
                    "alfred_realtime_start_date": "2024-06-20",
                    "alfred_source_line": 2,
                    "value_agrees": True,
                    "release_date_agrees": True,
                },
            },
            {
                "reference_week": "2024-06-22",
                "release_date": "2024-06-27",
                "first_report_value": 345678,
                "status": "observed",
                "source_url": "https://oui.doleta.gov/press/2024/062724.pdf",
                "release_text_sha256": "b" * 64,
                "source_comparison": {
                    "kind": "exact_agreement",
                    "dol_value": 345678,
                    "alfred_value": 345678,
                    "alfred_realtime_start_date": "2024-06-27",
                    "alfred_source_line": 3,
                    "value_agrees": True,
                    "release_date_agrees": True,
                },
            },
        ],
        "counts": {"observed": 2, "unresolved_correction": 0, "no_admitted_release": 0},
    }
    reports = [report(), report("2024-06-22", "2024-06-27", 345678, "b" * 64)]
    records = [alfred(), alfred("2024-06-22", "2024-06-27", 345678, 3)]
    return ledger, reports, records


def exception(year):
    ledger, _, _ = ordinary()
    row = ledger["rows"][0]
    if year == 2017:
        week, dol_date, alfred_date = "2017-03-18", "2017-03-23", "2017-03-23"
        dol_value, alfred_value, first_value = 261000, 258000, None
        status = kind = "unresolved_correction"
    else:
        week, dol_date, alfred_date = "2018-03-17", "2018-03-22", "2018-03-29"
        dol_value, alfred_value, first_value = 229000, 227000, 229000
        status, kind = "observed", "documented_original_dol_disagreement"
    source = report(week, dol_date, dol_value)
    row.update(
        reference_week=week,
        release_date=dol_date,
        first_report_value=first_value,
        status=status,
        source_url=source["source_url"],
    )
    row["source_comparison"] = {
        "kind": kind,
        "dol_value": dol_value,
        "alfred_value": alfred_value,
        "alfred_realtime_start_date": alfred_date,
        "alfred_source_line": 17,
        "value_agrees": False,
        "release_date_agrees": year == 2017,
    }
    ledger.update(start=week, end=week, rows=[row])
    ledger["counts"] = {
        "observed": int(year == 2018),
        "unresolved_correction": int(year == 2017),
        "no_admitted_release": 0,
    }
    return ledger, [source], [alfred(week, alfred_date, alfred_value, 17)]


def gaps():
    return (
        {
            "status": "FIRST_REPORT_LEDGER_BUILT_NOT_INDEPENDENTLY_VERIFIED",
            "start": "2025-09-27",
            "end": "2025-10-18",
            "rows": [
                {
                    "reference_week": week,
                    "release_date": None,
                    "first_report_value": None,
                    "status": "no_admitted_release",
                    "source_url": None,
                    "release_text_sha256": None,
                    "source_comparison": {
                        "kind": "no_admitted_release",
                        "dol_value": None,
                        "alfred_value": None,
                        "alfred_realtime_start_date": None,
                        "alfred_source_line": None,
                        "value_agrees": None,
                        "release_date_agrees": None,
                    },
                }
                for week in ("2025-09-27", "2025-10-04", "2025-10-11", "2025-10-18")
            ],
            "counts": {"observed": 0, "unresolved_correction": 0, "no_admitted_release": 4},
        },
        [],
        [],
    )


def check(parts):
    ledger, reports, records = parts
    return verify_ledger(ledger, reports, records, ledger["start"], ledger["end"])


class VerifyClaimsReleaseLedgerTests(unittest.TestCase):
    def test_literal_ordinary_reconstruction_is_order_independent_and_nonmutating(self):
        parts = ordinary()
        parts[1].reverse()
        parts[2].reverse()
        before = copy.deepcopy(parts)
        actual = check(parts)
        self.assertEqual(actual["status"], "VERIFIED")
        self.assertEqual(actual["reference_weeks"], 2)
        self.assertEqual(
            actual["counts"],
            {"observed": 2, "unresolved_correction": 0, "no_admitted_release": 0},
        )
        self.assertEqual(actual["reports_checked"], 2)
        self.assertEqual(actual["alfred_records_checked"], 2)
        self.assertEqual(parts, before)

    def test_documented_correction_and_original_recovery_are_distinct(self):
        for year in (2017, 2018):
            with self.subTest(year=year):
                actual = check(exception(year))
                self.assertEqual(actual["status"], "VERIFIED")
                self.assertEqual(actual["reference_weeks"], 1)
                self.assertEqual(actual["counts"]["observed"], int(year == 2018))

    def test_all_four_literal_tail_gaps_are_retained(self):
        actual = check(gaps())
        self.assertEqual(actual["reference_weeks"], 4)
        self.assertEqual(actual["counts"]["no_admitted_release"], 4)
        self.assertEqual(actual["reports_checked"], 0)

    def test_value_date_and_each_provenance_field_are_independently_bound(self):
        for field, value in (
            ("first_report_value", 456788),
            ("release_date", "2024-06-21"),
            ("source_url", "https://oui.doleta.gov/press/2024/062124.pdf"),
            ("release_text_sha256", "f" * 64),
            ("status", "unresolved_correction"),
        ):
            parts = ordinary()
            parts[0]["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(parts)

    def test_each_comparison_field_is_reconstructed(self):
        for field, value in (
            ("kind", "documented_original_dol_disagreement"),
            ("dol_value", 1),
            ("alfred_value", 1),
            ("alfred_realtime_start_date", "2024-06-21"),
            ("alfred_source_line", 3),
            ("value_agrees", False),
            ("release_date_agrees", False),
        ):
            parts = ordinary()
            parts[0]["rows"][0]["source_comparison"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(parts)

    def test_no_hidden_alfred_count_or_corrected_dol_count_substitution(self):
        for year, value in ((2017, 258000), (2017, 261000), (2018, 227000)):
            parts = exception(year)
            parts[0]["rows"][0]["first_report_value"] = value
            with self.subTest(year=year, value=value), self.assertRaises(ValueError):
                check(parts)

    def test_fixed_exception_source_values_and_dates_cannot_drift(self):
        for year in (2017, 2018):
            for side, field, value in (
                (1, "value", 777777),
                (2, "value", 888888),
                (2, "alfred_realtime_start_date", f"{year}-03-30"),
            ):
                parts = exception(year)
                parts[side][0][field] = value
                with (
                    self.subTest(year=year, side=side, field=field),
                    self.assertRaises(ValueError),
                ):
                    check(parts)

    def test_unapproved_value_or_date_disagreement_is_not_a_new_exception(self):
        for field, value in (("value", 123456), ("alfred_realtime_start_date", "2024-06-21")):
            parts = ordinary()
            parts[2][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(parts)

    def test_missing_ordinary_source_on_either_or_both_sides_fails(self):
        for sides in ((1,), (2,), (1, 2)):
            parts = ordinary()
            for side in sides:
                parts[side].pop(0)
            with self.subTest(sides=sides), self.assertRaises(ValueError):
                check(parts)

    def test_tail_gap_cannot_hide_any_report_or_alfred_observation(self):
        for side in (1, 2):
            parts = gaps()
            parts[side].append(
                report("2025-09-27", "2025-10-02")
                if side == 1
                else alfred("2025-09-27", "2025-10-02")
            )
            with self.subTest(side=side), self.assertRaises(ValueError):
                check(parts)

    def test_null_gap_payload_and_comparison_are_exact(self):
        for field, value in (
            ("release_date", "2025-10-02"),
            ("first_report_value", 0),
            ("source_url", ""),
            ("release_text_sha256", "a" * 64),
        ):
            parts = gaps()
            parts[0]["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(parts)
        parts = gaps()
        parts[0]["rows"][0]["source_comparison"]["value_agrees"] = False
        with self.assertRaises(ValueError):
            check(parts)

    def test_row_grid_cannot_be_reordered_compressed_or_duplicated(self):
        for operation in ("reverse", "drop", "duplicate", "weekday"):
            parts = ordinary()
            rows = parts[0]["rows"]
            if operation == "reverse":
                rows.reverse()
            elif operation == "drop":
                rows.pop(0)
            elif operation == "duplicate":
                rows.append(copy.deepcopy(rows[0]))
            else:
                rows[0]["reference_week"] = "2024-06-16"
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                check(parts)

    def test_duplicate_and_out_of_bounds_source_rows_fail_without_filtering(self):
        for side in (1, 2):
            for extra in (
                copy.deepcopy(ordinary()[side][0]),
                report("2024-06-08", "2024-06-13")
                if side == 1
                else alfred("2024-06-08", "2024-06-13"),
            ):
                parts = ordinary()
                parts[side].append(extra)
                with (
                    self.subTest(side=side, week=extra["observation_date"]),
                    self.assertRaises(ValueError),
                ):
                    check(parts)

    def test_2018_documented_duplicate_alfred_date_remains_valid(self):
        ledger, reports, records = exception(2018)
        other = ordinary()[0]["rows"][1]
        other.update(
            reference_week="2018-03-24",
            release_date="2018-03-29",
            first_report_value=345678,
            source_url="https://oui.doleta.gov/press/2018/032918.pdf",
        )
        other["source_comparison"].update(
            alfred_realtime_start_date="2018-03-29", alfred_source_line=18
        )
        ledger["rows"].append(other)
        ledger["end"] = "2018-03-24"
        ledger["counts"]["observed"] = 2
        reports.append(report("2018-03-24", "2018-03-29", 345678, "b" * 64))
        records.append(alfred("2018-03-24", "2018-03-29", 345678, 18))
        self.assertEqual(check((ledger, reports, records))["status"], "VERIFIED")

    def test_input_statistic_body_date_identity_and_hash_are_checked(self):
        for field, value in (
            ("statistic", "continued_claims"),
            ("body_release_dates", []),
            ("body_release_dates", ["2024-06-21"]),
            ("body_release_date_verified", False),
            ("body_release_date_verified", 1),
            ("release_text_sha256", "A" * 64),
            ("release_text_sha256", "g" * 64),
            ("source_url", "https://evil.example/press/2024/062024.pdf"),
            ("source_url", "https://oui.doleta.gov/press/2024/062124.pdf"),
            ("source_url", "https://oui.doleta.gov/press/2024/062024.pdf?x=1"),
        ):
            parts = ordinary()
            parts[1][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                check(parts)

    def test_input_numeric_status_and_line_types_are_strict(self):
        for side, field, value in (
            (1, "value", True),
            (1, "value", 456789.0),
            (1, "value", 0),
            (2, "value", None),
            (2, "status", "missing"),
            (2, "source_line", True),
            (2, "source_line", 1),
        ):
            parts = ordinary()
            parts[side][0][field] = value
            with (
                self.subTest(side=side, field=field, value=value),
                self.assertRaises(ValueError),
            ):
                check(parts)

    def test_exact_schemas_reject_unknown_missing_and_wrong_types(self):
        for level in ("top", "row", "comparison", "counts", "report", "alfred"):
            for mutation in ("extra", "missing"):
                parts = ordinary()
                target = {
                    "top": parts[0],
                    "row": parts[0]["rows"][0],
                    "comparison": parts[0]["rows"][0]["source_comparison"],
                    "counts": parts[0]["counts"],
                    "report": parts[1][0],
                    "alfred": parts[2][0],
                }[level]
                if mutation == "extra":
                    target["unexpected"] = None
                else:
                    target.pop(next(iter(target)))
                with (
                    self.subTest(level=level, mutation=mutation),
                    self.assertRaises(ValueError),
                ):
                    verify_ledger(*parts, "2024-06-15", "2024-06-22")
        parts = ordinary()
        parts[0]["rows"][0]["source_comparison"]["value_agrees"] = 1
        with self.assertRaises(ValueError):
            check(parts)
        parts = ordinary()
        parts[0]["rows"][0]["first_report_value"] = 456789.0
        with self.assertRaises(ValueError):
            check(parts)

    def test_status_counts_and_caller_bounds_cannot_be_relabelled(self):
        for key, value in (
            ("observed", 3),
            ("unresolved_correction", False),
            ("no_admitted_release", 0.0),
        ):
            parts = ordinary()
            parts[0]["counts"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check(parts)
        parts = ordinary()
        parts[0]["status"] = "VERIFIED"
        with self.assertRaises(ValueError):
            check(parts)
        parts = ordinary()
        with self.assertRaises(ValueError):
            verify_ledger(*parts, "2024-06-15", "2024-06-15")

    def test_calendar_bounds_and_source_dates_remain_literal_and_fenced(self):
        for start, end in (
            ("2024-06-16", "2024-06-22"),
            ("2009-05-23", "2024-06-22"),
            ("2024-06-15", "2025-10-25"),
            ("2024-06-22", "2024-06-15"),
            ("2024-6-15", "2024-06-22"),
        ):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                verify_ledger(*ordinary(), start, end)
        for side, field, value in (
            (1, "actual_release_date", "2025-10-21"),
            (1, "observation_date", "2024-06-16"),
            (2, "alfred_realtime_start_date", "2024-06-14"),
        ):
            parts = ordinary()
            parts[side][0][field] = value
            with self.subTest(side=side, field=field), self.assertRaises(ValueError):
                check(parts)


if __name__ == "__main__":
    unittest.main()
