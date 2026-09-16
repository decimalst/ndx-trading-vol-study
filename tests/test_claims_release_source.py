"""Generated source contracts; no historical observation files are read."""

import copy
import hashlib
import io
import json
import stat
import struct
import unittest
import warnings
import zipfile
from unittest.mock import patch

from src import claims_release_source as source

DATA_NAME = "obs.,_initial_release_only.csv"
HEADER = "period_start_date,ICSA,realtime_start_date\n"
DEFAULT_ROWS = [
    ("2018-11-03", "111", "2018-11-08"),
    ("2018-11-10", ".", "2018-11-15"),
    ("2018-11-17", "123.0", "2018-11-21"),
]


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def request_fixture(**changes):
    request = {
        "requested_url": "https://alfred.stlouisfed.org/series/downloaddata?seid=ICSA",
        "method": "POST",
        "created_utc": "2026-09-08T20:28:13+00:00",
        "mode": "first_release_only",
        "series_id": "ICSA",
        "units": "lin",
        "file_type": "4",
        "file_format": "csv",
        "observation_start": "2018-11-03",
        "observation_end": "2018-11-17",
        "vintage_dates": ["2018-11-08", "2018-11-15", "2018-11-21"],
        "source_ceiling": "2025-10-20",
        "purpose": "Generated fixture; contains invented claims values only.",
    }
    request.update(changes)
    return request


def readme_fixture(vintages):
    # The official README layout was inspected without opening the CSV body.
    text = (
        "-------------------------------------------------------------------\n"
        "Series ID: ICSA\n"
        "Archival Federal Reserve Economic Data, Federal Reserve Bank of St.\n"
        "Louis\n"
        "Link: https://alfred.stlouisfed.org/series?seid=ICSA\n"
        "Help: https://alfred.stlouisfed.org/help\n"
        "This data may be copyrighted. Please refer to the Terms of Use:\n"
        "https://fred.stlouisfed.org/legal#fred-terms-faq\n"
        "Output Format: Observations, Initial Release Only\n"
        "File Created: 2026-09-08 3:28 PM CDT\n"
        "-------------------------------------------------------------------\n\n"
        "---------------------------------------------------------------------  ---------------  -------------\n"
        "Title                                                                  Real-Time Start  Real-Time End\n"
        "Initial Claims                                                         2009-05-28       Current\n"
        "Source\n"
        "U.S. Employment and Training Administration                            2009-05-28       Current\n"
        "Release\n"
        "Unemployment Insurance Weekly Claims Report                            2009-05-28       Current\n"
        "Units\n"
        "Number                                                                 2009-05-28       Current\n"
        "Frequency\n"
        "Weekly, Ending Saturday                                                2009-05-28       Current\n"
        "Seasonal Adjustment\n"
        "Seasonally Adjusted                                                    2009-05-28       Current\n"
        "Notes\n"
        "Generated fixture; no historical observations.                         2009-05-28       Current\n"
        "---------------------------------------------------------------------  ---------------  -------------\n\n"
        "Vintage Dates Specified:\n----------\n"
    )
    return (text + "\n".join(vintages) + "\n----------\n").encode()


def archive_fixture(
    request=None, rows=None, readme=None, data=None, compression=zipfile.ZIP_DEFLATED
):
    request = request_fixture() if request is None else request
    rows = DEFAULT_ROWS if rows is None else rows
    readme = readme_fixture(request["vintage_dates"]) if readme is None else readme
    data = (
        (HEADER + "".join(",".join(row) + "\n" for row in rows)).encode()
        if data is None
        else data
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        archive.writestr("README.txt", readme)
        archive.writestr(DATA_NAME, data)
    return buffer.getvalue(), readme, data


class ClaimsReleaseSourceTests(unittest.TestCase):
    def parse(self, request=None, **fixture_options):
        request = request_fixture() if request is None else request
        payload, readme, data = archive_fixture(request=request, **fixture_options)
        return source.parse_export(payload, request, digest(payload)), payload, readme, data

    def assert_preflight_rejects(self, request=None, **fixture_options):
        request = request_fixture() if request is None else request
        payload, _, _ = archive_fixture(request=request, **fixture_options)
        with patch.object(
            source,
            "_parse_value",
            side_effect=AssertionError("numeric conversion before preflight"),
        ) as convert:
            with self.assertRaises(ValueError):
                source.parse_export(payload, request, digest(payload))
            convert.assert_not_called()

    def test_valid_export_preserves_missing_and_independent_clocks_with_hashes(self):
        request = request_fixture()
        result, payload, readme, data = self.parse(request=request)
        self.assertEqual(result["status"], "STRUCTURALLY_VERIFIED_NOT_RELEASE_RECONCILED")
        expected = [
            {
                "observation_date": "2018-11-03",
                "alfred_realtime_start_date": "2018-11-08",
                "value": 111,
                "status": "observed",
                "source_line": 2,
            },
            {
                "observation_date": "2018-11-10",
                "alfred_realtime_start_date": "2018-11-15",
                "value": None,
                "status": "missing",
                "source_line": 3,
            },
            {
                "observation_date": "2018-11-17",
                "alfred_realtime_start_date": "2018-11-21",
                "value": 123,
                "status": "observed",
                "source_line": 4,
            },
        ]
        self.assertEqual(result["records"], expected)
        self.assertEqual(result["excluded_initial_snapshot"], [])
        self.assertEqual(result["missing_reference_weeks"], [])
        self.assertEqual(result["source_sha256"], digest(payload))
        self.assertEqual(result["readme_sha256"], digest(readme))
        self.assertEqual(result["data_sha256"], digest(data))
        canonical = json.dumps(
            request, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        self.assertEqual(result["request_sha256"], digest(canonical))
        self.assertNotIn("model_ready", result)

    def test_late_post_ceiling_row_prevents_conversion_of_earlier_rows(self):
        rows = DEFAULT_ROWS + [("2025-10-18", "DO_NOT_CONVERT_THIS_TOKEN", "2025-10-21")]
        self.assert_preflight_rejects(rows=rows)

    def test_initial_snapshot_excluded_before_value_conversion(self):
        request = request_fixture(
            observation_start="2009-05-23",
            observation_end="2009-05-30",
            vintage_dates=["2009-05-28", "2009-06-04"],
        )
        rows = [
            ("2009-05-23", "EXCLUDED_VALUE_MUST_NOT_CONVERT", "2009-05-28"),
            ("2009-05-30", "17", "2009-06-04"),
        ]
        with patch.object(source, "_parse_value", wraps=source._parse_value) as convert:
            result, _, _, _ = self.parse(request=request, rows=rows)
        self.assertEqual([call.args[0] for call in convert.call_args_list], ["17"])
        self.assertEqual([row["value"] for row in result["records"]], [17])
        self.assertEqual(
            result["excluded_initial_snapshot"],
            [
                {
                    "observation_date": "2009-05-23",
                    "alfred_realtime_start_date": "2009-05-28",
                    "source_line": 2,
                    "reason": "initial_2009_snapshot",
                }
            ],
        )
        self.assertEqual(result["missing_reference_weeks"], [])

    def test_missing_reference_weeks_include_edges_but_not_explicit_missing_rows(self):
        request = request_fixture(observation_start="2018-10-27", observation_end="2018-12-01")
        result, _, _, _ = self.parse(request=request)
        self.assertEqual(
            result["missing_reference_weeks"], ["2018-10-27", "2018-11-24", "2018-12-01"]
        )
        self.assertEqual(result["records"][1]["status"], "missing")

    def test_unused_requested_revision_vintage_does_not_invent_a_release(self):
        request = request_fixture(
            vintage_dates=["2018-11-08", "2018-11-15", "2018-11-16", "2018-11-21"]
        )
        result, _, _, _ = self.parse(request=request)
        self.assertEqual(len(result["records"]), 3)
        self.assertEqual(result["missing_reference_weeks"], [])

    def test_request_semantics_cannot_be_relaxed_by_matching_readme(self):
        changes = {
            "requested_url": "https://example.invalid/series/downloaddata?seid=ICSA",
            "method": "GET",
            "mode": "latest_revised",
            "series_id": "ICNSA",
            "units": "pc1",
            "file_type": "1",
            "file_format": "xlsx",
            "source_ceiling": "2025-10-21",
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                self.assert_preflight_rejects(request=request_fixture(**{key: value}))
        for key in changes:
            with self.subTest(missing=key):
                request = request_fixture()
                del request[key]
                self.assert_preflight_rejects(request=request)

    def test_readme_series_mode_and_all_six_attributes_are_binding(self):
        changes = [
            ("Series ID: ICSA", "Series ID: ICNSA"),
            (
                "Output Format: Observations, Initial Release Only",
                "Output Format: Observations by Real-Time Period",
            ),
            ("Initial Claims", "Continued Claims"),
            ("U.S. Employment and Training Administration", "Unverified uploader"),
            ("Unemployment Insurance Weekly Claims Report", "Unemployment Rate"),
            (
                "Number                                                                 ",
                "Thousands                                                              ",
            ),
            ("Weekly, Ending Saturday", "Weekly, Ending Friday"),
            (
                "Seasonally Adjusted                                                    ",
                "Not Seasonally Adjusted                                                ",
            ),
            ("2009-05-28", "2010-01-01"),
        ]
        original = readme_fixture(request_fixture()["vintage_dates"])
        for before, after in changes:
            with self.subTest(field=before):
                readme = original.replace(before.encode(), after.encode(), 1)
                self.assertNotEqual(readme, original)
                self.assert_preflight_rejects(readme=readme)

    def test_readme_duplicate_identity_or_attribute_is_ambiguous(self):
        readme = readme_fixture(request_fixture()["vintage_dates"])
        for extra in (
            b"Series ID: ICSA\n",
            b"Output Format: Observations, Initial Release Only\n",
            b"Units\nNumber                                                                 2009-05-28       Current\n",
        ):
            with self.subTest(extra=extra):
                self.assert_preflight_rejects(readme=extra + readme)

    def test_readme_vintage_list_must_equal_sorted_unique_request(self):
        original = request_fixture()["vintage_dates"]
        for vintages in (
            original[:-1],
            original[::-1],
            original + [original[-1]],
            original + ["2025-10-21"],
        ):
            with self.subTest(vintages=vintages):
                self.assert_preflight_rejects(readme=readme_fixture(vintages))
        for vintages in (
            [],
            original[::-1],
            original + [original[-1]],
            original + ["2025-10-21"],
            ["2018/11/08"],
        ):
            with self.subTest(request_vintages=vintages):
                self.assert_preflight_rejects(request=request_fixture(vintage_dates=vintages))

    def test_missing_vintage_section_or_extra_unparseable_vintage_fails(self):
        readme = readme_fixture(request_fixture()["vintage_dates"])
        cases = [
            readme.split(b"Vintage Dates Specified:")[0],
            readme.replace(b"2018-11-15\n", b"not-a-date\n"),
        ]
        for modified in cases:
            with self.subTest(readme=modified[-60:]):
                self.assert_preflight_rejects(readme=modified)

    def test_invalid_request_clocks_fail_before_numeric_conversion(self):
        changes = [
            {"created_utc": "2010-01-01T00:00:00+00:00"},
            {"created_utc": "2026-09-08T20:28:13"},
            {"created_utc": "not-a-timestamp"},
            {"observation_start": "2018-11-18"},
            {"observation_start": "2018/11/03"},
            {"observation_end": "2025-10-21"},
            {"observation_end": "2018-02-30"},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assert_preflight_rejects(request=request_fixture(**change))

    def test_late_row_structural_clock_errors_prevent_all_numeric_conversion(self):
        rows_to_reject = [
            ("2018-11-24", "DO_NOT_CONVERT", "2018-11-29"),
            ("2018/11/17", "DO_NOT_CONVERT", "2018-11-21"),
            ("2018-11-18", "DO_NOT_CONVERT", "2018-11-21"),
            ("2018-11-17", "DO_NOT_CONVERT", "2018-11-16"),
            ("2018-11-17", "DO_NOT_CONVERT", "2018-11-20"),
            ("2018-11-17", "DO_NOT_CONVERT", "2018-02-30"),
            ("2025-10-25", "DO_NOT_CONVERT", "2025-10-30"),
        ]
        for row in rows_to_reject:
            with self.subTest(row_dates=(row[0], row[2])):
                self.assert_preflight_rejects(rows=DEFAULT_ROWS[:2] + [row])

    def test_ceiling_release_is_allowed_without_thursday_assumption(self):
        request = request_fixture(
            observation_start="2025-10-18",
            observation_end="2025-10-18",
            vintage_dates=["2025-10-20"],
        )
        result, _, _, _ = self.parse(
            request=request, rows=[("2025-10-18", "37", "2025-10-20")]
        )
        self.assertEqual(result["records"][0]["value"], 37)
        self.assertEqual(result["records"][0]["alfred_realtime_start_date"], "2025-10-20")

    def test_duplicate_observation_or_conflicting_first_release_is_rejected(self):
        duplicates = [
            DEFAULT_ROWS[0],
            ("2018-11-03", "999", "2018-11-08"),
            ("2018-11-03", "111", "2018-11-15"),
        ]
        for duplicate in duplicates:
            with self.subTest(duplicate=duplicate):
                self.assert_preflight_rejects(rows=DEFAULT_ROWS + [duplicate])

    def test_wrong_header_and_late_malformed_csv_row_prevent_conversion(self):
        data = (HEADER + "".join(",".join(row) + "\n" for row in DEFAULT_ROWS)).encode()
        headers = [
            "period_start_date,ICNSA,realtime_start_date\n",
            "ICSA,period_start_date,realtime_start_date\n",
            "period_start_date,ICSA,realtime_start_date,extra\n",
        ]
        cases = [data.replace(HEADER.encode(), header.encode(), 1) for header in headers]
        cases.extend(
            [
                data + b"2018-11-17,5\n",
                data + b"2018-11-17,5,2018-11-21,extra\n",
                data + b'2018-11-17,"unterminated\n',
            ]
        )
        for case in cases:
            with self.subTest(data=case[-70:]):
                self.assert_preflight_rejects(data=case)

    def test_numeric_tokens_are_positive_integral_or_explicitly_missing(self):
        for token, expected in [("1", 1), ("123.0", 123), (".", None)]:
            with self.subTest(token=token):
                result = source._parse_value(token)
                self.assertEqual(result, expected)
                if expected is not None:
                    self.assertIs(type(result), int)
        for token in (
            "",
            "0",
            "0.0",
            "-1",
            "NaN",
            "nan",
            "Inf",
            "-Infinity",
            "1.5",
            "1,000",
            "1e309",
            "abc",
            "9" * 10000,
        ):
            with self.subTest(token=token[:20]), self.assertRaises(ValueError):
                source._parse_value(token)

    def test_invalid_numeric_row_fails_without_partial_panel(self):
        for token in ("", "0", "-2", "NaN", "Infinity", "15.2", "malformed"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                self.parse(rows=DEFAULT_ROWS[:2] + [("2018-11-17", token, "2018-11-21")])

    def test_initial_snapshot_old_backfill_never_becomes_first_report(self):
        request = request_fixture(
            observation_start="2008-12-27",
            observation_end="2009-05-30",
            vintage_dates=["2009-05-28", "2009-06-04"],
        )
        rows = [
            ("2008-12-27", "BACKFILLED_VALUE", "2009-05-28"),
            ("2009-05-30", "19", "2009-06-04"),
        ]
        result, _, _, _ = self.parse(request=request, rows=rows)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(
            result["excluded_initial_snapshot"][0]["observation_date"], "2008-12-27"
        )
        self.assertNotIn("value", result["excluded_initial_snapshot"][0])
        self.assertNotIn("BACKFILLED_VALUE", json.dumps(result))

    def test_wrong_hash_rejects_before_zip_or_numeric_processing(self):
        payload, _, _ = archive_fixture()
        with (
            patch.object(
                zipfile.ZipFile,
                "open",
                side_effect=AssertionError("member read before hash check"),
            ) as opened,
            patch.object(source, "_parse_value") as convert,
        ):
            with self.assertRaises(ValueError):
                source.parse_export(payload, request_fixture(), "0" * 64)
            opened.assert_not_called()
            convert.assert_not_called()

    def test_provenance_request_hash_is_canonical_and_covers_extra_fields(self):
        request = request_fixture()
        before = copy.deepcopy(request)
        result, _, _, _ = self.parse(request=request)
        self.assertEqual(request, before)
        reordered = dict(reversed(list(request.items())))
        reordered_result, _, _, _ = self.parse(request=reordered)
        self.assertEqual(result["request_sha256"], reordered_result["request_sha256"])
        request["purpose"] = "Different generated provenance description."
        changed, _, _, _ = self.parse(request=request)
        self.assertNotEqual(result["request_sha256"], changed["request_sha256"])
        self.assertEqual(result["records"], changed["records"])

    def test_plain_stored_archive_is_valid_and_nothing_is_extracted(self):
        with (
            patch.object(
                zipfile.ZipFile, "extract", side_effect=AssertionError("must not extract")
            ),
            patch.object(
                zipfile.ZipFile, "extractall", side_effect=AssertionError("must not extract")
            ),
        ):
            result, _, _, _ = self.parse(compression=zipfile.ZIP_STORED)
        self.assertEqual(len(result["records"]), 3)

    def test_duplicate_unexpected_and_unsafe_zip_members_are_rejected(self):
        request = request_fixture()
        readme = readme_fixture(request["vintage_dates"])
        data = (HEADER + "".join(",".join(row) + "\n" for row in DEFAULT_ROWS)).encode()
        cases = [
            [("README.txt", readme), (DATA_NAME, data), ("README.txt", readme)],
            [("README.txt", readme), (DATA_NAME, data), ("unexpected.txt", b"x")],
            [("../README.txt", readme), (DATA_NAME, data)],
            [("/README.txt", readme), (DATA_NAME, data)],
            [("nested/README.txt", readme), (DATA_NAME, data)],
            [("README.txt", readme)],
        ]
        for entries in cases:
            with self.subTest(members=[name for name, _ in entries]):
                buffer = io.BytesIO()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(
                        buffer, "w", compression=zipfile.ZIP_DEFLATED
                    ) as archive:
                        for name, body in entries:
                            archive.writestr(name, body)
                payload = buffer.getvalue()
                with patch.object(source, "_parse_value") as convert:
                    with self.assertRaises(ValueError):
                        source.parse_export(payload, request, digest(payload))
                    convert.assert_not_called()

    def test_zip_symlink_member_or_unsupported_compression_is_rejected(self):
        request = request_fixture()
        readme = readme_fixture(request["vintage_dates"])
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            info = zipfile.ZipInfo("README.txt")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, readme)
            archive.writestr(DATA_NAME, HEADER.encode())
        for payload in (buffer.getvalue(), archive_fixture(compression=zipfile.ZIP_BZIP2)[0]):
            with self.subTest(payload_hash=digest(payload)), self.assertRaises(ValueError):
                source.parse_export(payload, request, digest(payload))

    def test_archive_byte_and_expanded_limits_precede_member_reads(self):
        too_many_bytes = b"x" * (2 * 1024 * 1024 + 1)
        expanded, _, _ = archive_fixture(data=b"x" * (16 * 1024 * 1024 + 1))
        for payload in (too_many_bytes, expanded):
            with (
                self.subTest(compressed_size=len(payload)),
                patch.object(
                    zipfile.ZipFile,
                    "open",
                    side_effect=AssertionError("member read before size check"),
                ) as opened,
                patch.object(source, "_parse_value") as convert,
            ):
                with self.assertRaises(ValueError):
                    source.parse_export(payload, request_fixture(), digest(payload))
                opened.assert_not_called()
                convert.assert_not_called()

    def test_corrupt_truncated_or_encrypted_archive_is_rejected(self):
        payload, _, _ = archive_fixture(compression=zipfile.ZIP_STORED)
        corrupt = bytearray(payload)
        offset = corrupt.find(b"111")
        self.assertGreater(offset, 0)
        corrupt[offset] = ord("2")
        encrypted = bytearray(payload)
        position = 0
        while True:
            position = encrypted.find(b"PK\x01\x02", position)
            if position < 0:
                break
            flags = struct.unpack_from("<H", encrypted, position + 8)[0]
            struct.pack_into("<H", encrypted, position + 8, flags | 1)
            position += 4
        for modified in (b"not a ZIP", payload[:30], bytes(corrupt), bytes(encrypted)):
            with (
                self.subTest(payload_hash=digest(modified)),
                patch.object(source, "_parse_value") as convert,
            ):
                with self.assertRaises(ValueError):
                    source.parse_export(modified, request_fixture(), digest(modified))
                convert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
