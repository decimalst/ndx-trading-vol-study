"""Generated source contracts, written before the independent checker exists."""

import copy
import hashlib
import unittest
from unittest.mock import patch

from src import verify_commodity_implied_source as verify


def fixture(symbol="OVX"):
    payload = (
        f"DATE,{symbol}\r\n"
        "12/31/2008,BEFORE_WINDOW_MUST_NOT_CONVERT\r\n"
        "\r\n"
        "1/2/2009,12.5\r\n"
        "2009-01-05,.\r\n"
        "01/06/2009,\r\n"
        "2025-10-20,+2.5e1\r\n"
        "10/21/2025,PROTECTED_MUST_NOT_CONVERT\r\n"
    ).encode()
    parsed = {
        "status": "SOURCE_HISTORY_PARSED_NOT_INDEPENDENTLY_VERIFIED",
        "symbol": symbol,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "start": "2009-01-02",
        "end": "2025-10-20",
        "records": [
            {"date": "2009-01-02", "value": 12.5, "status": "observed", "source_line": 4},
            {"date": "2009-01-05", "value": None, "status": "missing", "source_line": 5},
            {"date": "2009-01-06", "value": None, "status": "missing", "source_line": 6},
            {"date": "2025-10-20", "value": 25.0, "status": "observed", "source_line": 7},
        ],
        "metadata": {
            "total_rows": 6,
            "before_start_rows": 1,
            "after_end_rows": 1,
            "retained_rows": 4,
            "observed_rows": 2,
            "missing_rows": 2,
            "first_retained_date": "2009-01-02",
            "last_retained_date": "2025-10-20",
        },
    }
    return payload, parsed


def one_row(token="1.25", date="2018-01-02", symbol="OVX"):
    payload = f"DATE,{symbol}\n{date},{token}\n".encode()
    parsed = {
        "status": "SOURCE_HISTORY_PARSED_NOT_INDEPENDENTLY_VERIFIED",
        "symbol": symbol,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "start": "2009-01-02",
        "end": "2025-10-20",
        "records": [
            {"date": "2018-01-02", "value": 1.25, "status": "observed", "source_line": 2}
        ],
        "metadata": {
            "total_rows": 1,
            "before_start_rows": 0,
            "after_end_rows": 0,
            "retained_rows": 1,
            "observed_rows": 1,
            "missing_rows": 0,
            "first_retained_date": "2018-01-02",
            "last_retained_date": "2018-01-02",
        },
    }
    return payload, parsed


def check(payload, parsed, **kwargs):
    return verify.verify_history(
        payload, parsed["symbol"], hashlib.sha256(payload).hexdigest(), parsed, **kwargs
    )


class SourceVerificationTests(unittest.TestCase):
    def test_complete_literal_fixture_for_both_symbols(self):
        for symbol in ("OVX", "GVZ"):
            with self.subTest(symbol=symbol):
                payload, parsed = fixture(symbol)
                self.assertEqual(
                    check(payload, parsed),
                    {
                        "status": "VERIFIED",
                        "symbol": symbol,
                        "retained_rows": 4,
                        "observed_rows": 2,
                        "missing_rows": 2,
                    },
                )

    def test_only_retained_value_tokens_are_converted(self):
        payload, parsed = fixture()
        with patch.object(verify, "_parse_value", wraps=verify._parse_value) as conversion:
            check(payload, parsed)
        self.assertEqual(
            [call.args[0] for call in conversion.call_args_list], ["12.5", ".", "", "+2.5e1"]
        )

    def test_all_dates_checked_before_any_value_conversion(self):
        base, parsed = one_row()
        for late in (
            b"2026-02-30,PROTECTED\n",
            b"2018-01-02,DUPLICATE\n",
            b"2017-12-29,REVERSED\n",
            b"2026-01-01,TOO,MANY\n",
            b"2026-01-01\n",
        ):
            with self.subTest(late=late), patch.object(verify, "_parse_value") as conversion:
                with self.assertRaises(ValueError):
                    check(base + late, parsed)
                conversion.assert_not_called()

    def test_future_duplicate_and_reverse_dates_rejected_before_values(self):
        base, parsed = one_row()
        for tail in (
            b"2026-01-02,X\n2026-01-02,Y\n",
            b"2026-01-02,X\n2026-01-01,Y\n",
        ):
            with self.subTest(tail=tail), patch.object(verify, "_parse_value") as conversion:
                with self.assertRaises(ValueError):
                    check(base + tail, parsed)
                conversion.assert_not_called()

    def test_hash_binds_payload_and_parsed_audit(self):
        payload, parsed = fixture()
        for digest in ("0" * 64, "x" * 64, "a" * 63, None, True):
            with (
                self.subTest(digest=digest),
                patch.object(verify, "_parse_value") as conversion,
            ):
                with self.assertRaises(ValueError):
                    verify.verify_history(payload, "OVX", digest, parsed)
                conversion.assert_not_called()
        altered = copy.deepcopy(parsed)
        altered["source_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            check(payload, altered)

    def test_exact_symbol_and_header_identity_before_conversion(self):
        payload, parsed = one_row()
        for header in (
            b"DATE,GVZ",
            b"date,OVX",
            b"DATE, OVX",
            b"DATE,OVX,OTHER",
            b"DATE",
            b"",
        ):
            altered = header + b"\n2018-01-02,1.25\n"
            with (
                self.subTest(header=header),
                patch.object(verify, "_parse_value") as conversion,
            ):
                with self.assertRaises(ValueError):
                    check(altered, parsed)
                conversion.assert_not_called()
        for symbol in ("ovx", "VIX", " OVX", None, 1):
            with self.subTest(symbol=symbol), self.assertRaises(ValueError):
                verify.verify_history(
                    payload, symbol, hashlib.sha256(payload).hexdigest(), parsed
                )

    def test_strict_utf8_with_optional_initial_bom(self):
        payload, parsed = one_row()
        for prefix in (b"", b"\xef\xbb\xbf"):
            altered = prefix + payload
            expected = copy.deepcopy(parsed)
            expected["source_sha256"] = hashlib.sha256(altered).hexdigest()
            self.assertEqual(check(altered, expected)["status"], "VERIFIED")
        for altered in (payload + b"\xff", b"\xef\xbb\xbf\xef\xbb\xbf" + payload):
            with (
                self.subTest(altered=altered),
                patch.object(verify, "_parse_value") as conversion,
            ):
                with self.assertRaises(ValueError):
                    check(altered, parsed)
                conversion.assert_not_called()

    def test_literal_date_formats_and_calendar_validity(self):
        for token in ("1/2/2018", "01/2/2018", "1/02/2018", "01/02/2018", "2018-01-02"):
            payload, parsed = one_row(date=token)
            self.assertEqual(check(payload, parsed)["status"], "VERIFIED")
        for token in (
            "2018-1-02",
            "2018/01/02",
            "01-02-2018",
            " 2018-01-02",
            "2018-01-02 ",
            "2/29/2018",
            "13/1/2018",
            "0/1/2018",
            "2018-01-02T00:00:00",
            "２０１８-01-02",
        ):
            payload, parsed = one_row(date=token)
            with self.subTest(token=token), patch.object(verify, "_parse_value") as conversion:
                with self.assertRaises(ValueError):
                    check(payload, parsed)
                conversion.assert_not_called()

    def test_duplicate_dates_in_different_supported_formats_rejected(self):
        payload, parsed = one_row()
        payload += b"1/2/2018,2\n"
        with patch.object(verify, "_parse_value") as conversion:
            with self.assertRaises(ValueError):
                check(payload, parsed)
            conversion.assert_not_called()

    def test_valid_decimal_lexemes_and_exact_float_values(self):
        for token, value in (
            ("1", 1.0),
            ("+1.25", 1.25),
            (".125", 0.125),
            ("1.", 1.0),
            ("01.25", 1.25),
            ("125e-2", 1.25),
            ("1.25E+2", 125.0),
        ):
            payload, parsed = one_row(token)
            parsed["records"][0]["value"] = value
            with self.subTest(token=token):
                self.assertEqual(check(payload, parsed)["status"], "VERIFIED")

    def test_invalid_nonpositive_and_nonfinite_values_rejected(self):
        for token in (
            "NaN",
            "nan",
            "Infinity",
            "inf",
            "+Inf",
            "True",
            "False",
            "0",
            "-0",
            "-1",
            "1e309",
            "1e-9999",
            "1_000",
            "0x10",
            "1,000",
            " 1.25",
            "1.25 ",
            "１.２５",
            "1e",
            "+",
            "-",
            "..",
            "1\x00",
        ):
            payload, parsed = one_row(token)
            with self.subTest(token=token), self.assertRaises(ValueError):
                check(payload, parsed)

    def test_only_empty_and_dot_are_missing(self):
        for token in ("", "."):
            payload, parsed = one_row(token)
            parsed["records"][0].update(value=None, status="missing")
            parsed["metadata"].update(observed_rows=0, missing_rows=1)
            self.assertEqual(check(payload, parsed)["missing_rows"], 1)
        for token in ("NA", "N/A", "null", "None", " .", ". ", " "):
            payload, parsed = one_row(token)
            with self.subTest(token=token), self.assertRaises(ValueError):
                check(payload, parsed)

    def test_narrower_scope_excludes_unconverted_values_and_counts_all_dates(self):
        payload = b"DATE,OVX\n2009-01-02,EARLY\n2018-01-02,1.25\n2018-01-03,LATE\n2026-01-01,PROTECTED\n"
        _, parsed = one_row()
        parsed.update(
            source_sha256=hashlib.sha256(payload).hexdigest(),
            start="2018-01-02",
            end="2018-01-02",
        )
        parsed["records"][0]["source_line"] = 3
        parsed["metadata"].update(total_rows=4, before_start_rows=1, after_end_rows=2)
        with patch.object(verify, "_parse_value", wraps=verify._parse_value) as conversion:
            check(payload, parsed, start="2018-01-02", end="2018-01-02")
        self.assertEqual([call.args[0] for call in conversion.call_args_list], ["1.25"])

    def test_bounds_cannot_expand_fixed_source_scope(self):
        payload, parsed = one_row()
        for start, end in (
            ("2009-01-01", "2025-10-20"),
            ("2009-01-02", "2025-10-21"),
            ("2018-02-01", "2018-01-01"),
            ("1/2/2009", "2025-10-20"),
            ("2009-01-02", "2025-02-30"),
            (None, "2025-10-20"),
        ):
            with (
                self.subTest(start=start, end=end),
                patch.object(verify, "_parse_value") as conversion,
            ):
                with self.assertRaises(ValueError):
                    check(payload, parsed, start=start, end=end)
                conversion.assert_not_called()

    def test_empty_retained_window_and_header_only_file(self):
        for payload, total, before, after in (
            (b"DATE,OVX\n", 0, 0, 0),
            (b"DATE,OVX\n2008-12-31,X\n2026-01-01,Y\n", 2, 1, 1),
        ):
            _, parsed = fixture()
            parsed.update(source_sha256=hashlib.sha256(payload).hexdigest(), records=[])
            parsed["metadata"] = dict(
                total_rows=total,
                before_start_rows=before,
                after_end_rows=after,
                retained_rows=0,
                observed_rows=0,
                missing_rows=0,
                first_retained_date=None,
                last_retained_date=None,
            )
            with patch.object(verify, "_parse_value") as conversion:
                self.assertEqual(check(payload, parsed)["retained_rows"], 0)
                conversion.assert_not_called()

    def test_physical_blank_lines_and_csv_reader_source_line(self):
        payload = b'DATE,OVX\n\n2018-01-02,"1.25"\n\n'
        _, parsed = one_row()
        parsed["source_sha256"] = hashlib.sha256(payload).hexdigest()
        parsed["records"][0]["source_line"] = 3
        self.assertEqual(check(payload, parsed)["status"], "VERIFIED")
        for row in (b" \n", b",\n", b'""\n'):
            with self.subTest(row=row), self.assertRaises(ValueError):
                check(payload + row, parsed)

    def test_quoted_multiline_outside_value_uses_reader_end_line(self):
        payload = b'DATE,OVX\n2008-12-31,"OUTSIDE\nDO NOT PARSE"\n2018-01-02,1.25\n'
        _, parsed = one_row()
        parsed["source_sha256"] = hashlib.sha256(payload).hexdigest()
        parsed["metadata"].update(total_rows=2, before_start_rows=1)
        parsed["records"][0]["source_line"] = 4
        self.assertEqual(check(payload, parsed)["status"], "VERIFIED")

    def test_malformed_csv_rejected_before_conversion(self):
        payload, parsed = one_row()
        for tail in (b'2026-01-01,"UNCLOSED\n', b'2026-01-01,"X"bad\n'):
            with self.subTest(tail=tail), patch.object(verify, "_parse_value") as conversion:
                with self.assertRaises(ValueError):
                    check(payload + tail, parsed)
                conversion.assert_not_called()

    def test_tampered_record_fields_are_rejected(self):
        payload, parsed = fixture()
        for field, value in (
            ("date", "2009-01-03"),
            ("value", 12.500000000000002),
            ("status", "missing"),
            ("source_line", 2),
            ("extra", "forbidden"),
        ):
            altered = copy.deepcopy(parsed)
            altered["records"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(payload, altered)
        for field in parsed["records"][0]:
            altered = copy.deepcopy(parsed)
            del altered["records"][0][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                check(payload, altered)

    def test_missing_record_cannot_be_zero_filled(self):
        payload, parsed = fixture()
        for value, status in ((0.0, "missing"), (1.0, "observed"), (None, "observed")):
            altered = copy.deepcopy(parsed)
            altered["records"][1].update(value=value, status=status)
            with self.subTest(value=value, status=status), self.assertRaises(ValueError):
                check(payload, altered)

    def test_record_omission_addition_duplicate_and_reordering_rejected(self):
        payload, parsed = fixture()
        rows = parsed["records"]
        for altered_rows in (
            rows[:-1],
            rows + [rows[0]],
            list(reversed(rows)),
            rows
            + [{"date": "2025-10-21", "value": 1.0, "status": "observed", "source_line": 8}],
        ):
            altered = copy.deepcopy(parsed)
            altered["records"] = copy.deepcopy(altered_rows)
            with self.subTest(altered_rows=altered_rows), self.assertRaises(ValueError):
                check(payload, altered)

    def test_all_metadata_fields_and_schema_are_verified(self):
        payload, parsed = fixture()
        for field, value in parsed["metadata"].items():
            altered = copy.deepcopy(parsed)
            altered["metadata"][field] = value + 1 if isinstance(value, int) else "2009-01-01"
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(payload, altered)
        for operation in ("extra", "missing"):
            altered = copy.deepcopy(parsed)
            if operation == "extra":
                altered["metadata"]["model_ready"] = True
            else:
                del altered["metadata"]["total_rows"]
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                check(payload, altered)

    def test_top_level_identity_status_bounds_and_unknown_keys(self):
        payload, parsed = fixture()
        for field, value in (
            ("status", "VERIFIED"),
            ("symbol", "GVZ"),
            ("start", "2010-01-01"),
            ("end", "2025-10-19"),
            ("model_ready", True),
        ):
            altered = copy.deepcopy(parsed)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.verify_history(
                    payload, "OVX", hashlib.sha256(payload).hexdigest(), altered
                )
        for field in parsed:
            altered = copy.deepcopy(parsed)
            del altered[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                verify.verify_history(
                    payload, "OVX", hashlib.sha256(payload).hexdigest(), altered
                )

    def test_exact_scalar_and_container_types_prevent_equality_aliases(self):
        payload, parsed = one_row(token="1")
        parsed["records"][0]["value"] = 1.0
        for location, field, value in (
            ("record", "value", True),
            ("record", "value", 1),
            ("record", "source_line", 2.0),
            ("metadata", "retained_rows", True),
            ("metadata", "observed_rows", 1.0),
        ):
            altered = copy.deepcopy(parsed)
            target = altered["records"][0] if location == "record" else altered["metadata"]
            target[field] = value
            with (
                self.subTest(location=location, field=field, value=value),
                self.assertRaises(ValueError),
            ):
                check(payload, altered)
        for altered in (
            None,
            [],
            dict(parsed, records=tuple(parsed["records"])),
            dict(parsed, metadata=[]),
        ):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                verify.verify_history(
                    payload, "OVX", hashlib.sha256(payload).hexdigest(), altered
                )

    def test_input_payload_type_and_no_mutation(self):
        payload, parsed = fixture()
        before = copy.deepcopy(parsed)
        check(payload, parsed)
        self.assertEqual(parsed, before)
        for altered in (payload.decode(), bytearray(payload), None):
            with self.subTest(kind=type(altered).__name__), self.assertRaises(ValueError):
                verify.verify_history(altered, "OVX", parsed["source_sha256"], parsed)


if __name__ == "__main__":
    unittest.main()
