"""Prospective synthetic-only contracts for Cboe commodity history admission."""

import hashlib
import unittest
from unittest import mock

from src import commodity_implied_source as source


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


class CommodityImpliedSourceTests(unittest.TestCase):
    def parse(self, payload, symbol="OVX", **kwargs):
        return source.parse_cboe_history(payload, symbol, digest(payload), **kwargs)

    def test_exact_records_missingness_and_all_date_accounting(self):
        payload = (
            b"\xef\xbb\xbfDATE,OVX\r\n12/31/2008,DO_NOT_CONVERT\r\n\r\n"
            b"1/2/2009,20\r\n2009-01-05,.\r\n01/06/2009,\r\n"
            b"10/20/2025,2.5e1\r\n10/21/2025,PROTECTED_SENTINEL\r\n"
        )
        result = self.parse(payload)
        self.assertEqual(
            result,
            {
                "status": "SOURCE_HISTORY_PARSED_NOT_INDEPENDENTLY_VERIFIED",
                "symbol": "OVX",
                "source_sha256": digest(payload),
                "start": "2009-01-02",
                "end": "2025-10-20",
                "records": [
                    {
                        "date": "2009-01-02",
                        "value": 20.0,
                        "status": "observed",
                        "source_line": 4,
                    },
                    {
                        "date": "2009-01-05",
                        "value": None,
                        "status": "missing",
                        "source_line": 5,
                    },
                    {
                        "date": "2009-01-06",
                        "value": None,
                        "status": "missing",
                        "source_line": 6,
                    },
                    {
                        "date": "2025-10-20",
                        "value": 25.0,
                        "status": "observed",
                        "source_line": 7,
                    },
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
            },
        )
        self.assertIs(type(result["records"][0]["value"]), float)
        self.assertNotIn("SENTINEL", repr(result))

    def test_all_date_preflight_precedes_any_value_conversion(self):
        tails = [
            b"2026-13-01,ignored\n",
            b"2025-11-03,x\n2025-11-03,y\n",
            b"2025-11-04,x\n2025-11-03,y\n",
            b"2026-01-01,x,extra\n",
            b'2026-01-01,"unterminated\n',
            b",\n",
            b" \n",
        ]
        for tail in tails:
            with self.subTest(tail=tail), mock.patch.object(source, "_parse_value") as convert:
                with self.assertRaises(ValueError):
                    self.parse(b"DATE,OVX\n2009-01-02,20\n" + tail)
                convert.assert_not_called()

    def test_hash_authentication_precedes_decoding_and_conversion(self):
        payload = b"\xff"
        with mock.patch.object(source, "_parse_value") as convert:
            with self.assertRaisesRegex(ValueError, "(?i)hash|sha|digest"):
                source.parse_cboe_history(payload, "OVX", "0" * 64)
            convert.assert_not_called()

    def test_byte_hash_binds_exact_payload(self):
        payload = b"DATE,OVX\n2009-01-02,20\n"
        with self.assertRaises(ValueError):
            source.parse_cboe_history(payload + b"\n", "OVX", digest(payload))
        for bad_hash in [None, 3, "x" * 64, digest(payload).upper(), " " + digest(payload)]:
            with self.subTest(hash=bad_hash), self.assertRaises(ValueError):
                source.parse_cboe_history(payload, "OVX", bad_hash)

    def test_only_selected_bounds_are_converted(self):
        payload = b"DATE,GVZ\n1/2/2009,EARLY\n1/5/2009,12\n1/6/2009,LATE\n2026-01-01,NEVER\n"
        with mock.patch.object(source, "_parse_value", wraps=source._parse_value) as convert:
            result = self.parse(payload, "GVZ", start="2009-01-05", end="2009-01-05")
        self.assertEqual(convert.call_args_list, [mock.call("12")])
        self.assertEqual(
            result["records"],
            [{"date": "2009-01-05", "value": 12.0, "status": "observed", "source_line": 3}],
        )
        self.assertEqual(result["metadata"]["before_start_rows"], 1)
        self.assertEqual(result["metadata"]["after_end_rows"], 2)

    def test_exact_symbol_header_width_and_utf8(self):
        for payload in [
            b"",
            b"\nDATE,OVX\n",
            b"date,OVX\n",
            b"DATE,GVZ\n",
            b"DATE,OVX,extra\n",
            b"DATE,OVX \n",
            b"DATE,OVX\n2009-01-02,20,extra\n",
            b"DATE,OVX\n2009-01-02\n",
            b"DATE,OVX\n2009-01-02,\xff\n",
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.parse(payload)
        for symbol in ["ovx", "VIX", "OVX ", None, 1]:
            with self.subTest(symbol=symbol), self.assertRaises(ValueError):
                self.parse(b"DATE,OVX\n", symbol)
        with self.assertRaises(ValueError):
            source.parse_cboe_history("DATE,OVX\n", "OVX", "0" * 64)

    def test_bounds_are_strict_dates_inside_fixed_source_scope(self):
        for start, end in [
            ("2008-12-31", "2025-10-20"),
            ("2009-01-02", "2025-10-21"),
            ("2010-01-02", "2009-01-02"),
            ("2009-1-2", "2025-10-20"),
            (None, "2025-10-20"),
            ("2009-01-02", "2025-02-30"),
        ]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.parse(b"DATE,OVX\n", start=start, end=end)

    def test_invalid_in_window_numbers_fail_without_imputation(self):
        for token in [
            "0",
            "-0",
            "-1",
            "nan",
            "NaN",
            "inf",
            "Infinity",
            "1e999",
            "1e-999",
            " 20",
            "20 ",
            " ",
            "1_000",
            "N/A",
            "+",
            "0x20",
        ]:
            with self.subTest(token=token), self.assertRaises(ValueError):
                self.parse(f"DATE,OVX\n2009-01-02,{token}\n".encode())

    def test_permitted_finite_positive_decimal_forms(self):
        for token, value in [
            ("20", 20.0),
            ("+20.0", 20.0),
            (".5", 0.5),
            ("2.", 2.0),
            ("1e-2", 0.01),
            ("3E+2", 300.0),
        ]:
            with self.subTest(token=token):
                result = self.parse(f"DATE,OVX\n2009-01-02,{token}\n".encode())
                self.assertEqual(result["records"][0]["value"], value)

    def test_invalid_or_unsorted_dates_in_any_scope_fail(self):
        for token in [
            "2009-1-02",
            "09-01-02",
            "1/2/09",
            "2009-01-02 ",
            " 1/2/2009",
            "2009-02-29",
            "13/2/2009",
            "2026-00-01",
        ]:
            with self.subTest(token=token), self.assertRaises(ValueError):
                self.parse(f"DATE,OVX\n{token},20\n".encode())
        for rows in [
            "2009-01-05,20\n2009-01-02,30",
            "1/2/2009,20\n2009-01-02,30",
            "2008-01-01,x\n2008-01-01,y",
        ]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.parse(f"DATE,OVX\n{rows}\n".encode())

    def test_header_only_and_zero_retained_are_structurally_valid(self):
        for payload, count in [
            (b"DATE,OVX\n", 0),
            (b"DATE,OVX\n2008-01-01,ignored\n2026-01-01,ignored\n", 2),
        ]:
            result = self.parse(payload)
            self.assertEqual(result["records"], [])
            self.assertEqual(result["metadata"]["total_rows"], count)
            self.assertIsNone(result["metadata"]["first_retained_date"])
            self.assertIsNone(result["metadata"]["last_retained_date"])
            self.assertEqual(result["metadata"]["observed_rows"], 0)

    def test_csv_quoted_physical_line_accounting_and_blank_rows(self):
        result = self.parse(b'"DATE","OVX"\n\n2008-12-31,"outside\ntext"\n"2009-01-02","20"\n')
        self.assertEqual(result["records"][0]["source_line"], 5)
        self.assertEqual(result["metadata"]["total_rows"], 2)


if __name__ == "__main__":
    unittest.main()
