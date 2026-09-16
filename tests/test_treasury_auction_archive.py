"""Generated metadata-only contracts, written before the archive projector."""

import copy
import json
import unittest
from unittest.mock import patch

from src import treasury_auction_archive as archive

START = "2010-01-01"
END = "2025-10-20"
PDF = "https://www.treasurydirect.gov/instit/annceresult/press/preanre/"
XML = "https://www.treasurydirect.gov/xml/"
DOCUMENTS = (
    "announcement_pdf",
    "competitive_pdf",
    "noncompetitive_pdf",
    "special_pdf",
    "announcement_xml",
    "competitive_xml",
)


def row(**changes):
    result = {
        "a": "912ABC123",
        "d": "13-Week",
        "z3a": "Bill",
        "t3a": "13-Week",
        "h": "2017-12-28",
        "i": "2018-01-02",
    }
    result.update(changes)
    return result


def encoded(value):
    return json.dumps(value, allow_nan=False).encode("utf-8")


def project(value, **bounds):
    return archive.project_archive(
        encoded(value),
        start_date=bounds.get("start_date", START),
        end_date=bounds.get("end_date", END),
    )


def expected(record, **documents):
    return {
        "cusip": record["a"],
        "security_term": record["d"],
        "security_type": record["z3a"],
        "term_bucket": record["t3a"],
        "announcement_date": record["h"][:10],
        "auction_date": record["i"][:10],
        "documents": {key: documents.get(key, []) for key in DOCUMENTS},
    }


class TreasuryAuctionArchiveTests(unittest.TestCase):
    def test_whitespace_only_identity_metadata_is_not_present(self):
        for key in ("d", "z3a", "t3a"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                project([row(**{key: " \t"})])

    def test_unicode_casefold_filename_does_not_pass_ascii_grammar(self):
        for filename in ("\u212a.pdf", "\u0130.xml"):
            key = "f3" if filename.endswith("pdf") else "e4"
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                project([row(**{key: filename})])

    def test_exact_metadata_schema_and_all_document_mappings(self):
        original = row(
            e3="Ann_1.PDF",
            f3="Result-1.pdf",
            f31="Noncomp_1.pdf",
            f32="Special_A.pdf, Special_B.PDF",
            e4="Ann_1.XML",
            ia1="Result-1.xml",
            ignored_yield="DO NOT PROJECT",
        )
        before = copy.deepcopy(original)
        self.assertEqual(
            project([original]),
            [
                expected(
                    original,
                    announcement_pdf=[PDF + "2017/Ann_1.PDF"],
                    competitive_pdf=[PDF + "2018/Result-1.pdf"],
                    noncompetitive_pdf=[PDF + "2018/Noncomp_1.pdf"],
                    special_pdf=[
                        PDF + "2018/Special_A.pdf",
                        PDF + "2018/Special_B.PDF",
                    ],
                    announcement_xml=[XML + "Ann_1.XML"],
                    competitive_xml=[XML + "Result-1.xml"],
                )
            ],
        )
        self.assertEqual(original, before)

    def test_sole_security_list_wrapper_matches_array(self):
        records = [row()]
        self.assertEqual(project({"securityList": records}), project(records))

    def test_sorting_and_same_cusip_on_distinct_auction_dates(self):
        records = [
            row(a="912ABC124"),
            row(i="2018-01-03"),
            row(),
        ]
        self.assertEqual(
            project(records),
            [expected(records[2]), expected(records[0]), expected(records[1])],
        )

    def test_midnight_dates_normalize_and_leap_day_is_valid(self):
        record = row(h="2020-02-28T00:00:00", i="2020-02-29T00:00:00")
        self.assertEqual(project([record]), [expected(record)])

    def test_inclusive_auction_bounds_and_prior_announcement_year(self):
        records = [
            row(h="2009-12-30", i=START, e3="Ann.pdf"),
            row(h=END, i=END),
        ]
        self.assertEqual(
            project(records),
            [
                expected(records[0], announcement_pdf=[PDF + "2009/Ann.pdf"]),
                expected(records[1]),
            ],
        )

    def test_absent_null_and_empty_document_fields_are_empty_lists(self):
        for optional in (
            {},
            dict.fromkeys(("e3", "f3", "f31", "f32", "e4", "ia1")),
            dict.fromkeys(("e3", "f3", "f31", "f32", "e4", "ia1"), ""),
        ):
            with self.subTest(optional=optional):
                original = row(**optional)
                self.assertEqual(project([original]), [expected(original)])

    def test_unknown_numerical_tokens_are_text_decoded_and_never_projected(self):
        base = encoded(row())[:-1]
        raw = (
            b"["
            + base
            + b',"ignored_integer":'
            + b"9" * 10000
            + b',"ignored_float":1e999999999999999999999999'
            + b',"ignored_nested":{"amount":-1e-999999999999999999999999}'
            + b"}]"
        )
        real_loads = json.loads
        calls = []

        def inspect_loads(*args, **kwargs):
            for key, token in (
                ("parse_int", "9" * 10000),
                ("parse_float", "1e999999999999999999999999"),
            ):
                self.assertTrue(callable(kwargs.get(key)), key)
                decoded = kwargs[key](token)
                self.assertIsInstance(decoded, str)
                self.assertNotIsInstance(decoded, (int, float))
                self.assertEqual(decoded, token)
            calls.append(True)
            return real_loads(*args, **kwargs)

        with patch.object(archive.json, "loads", side_effect=inspect_loads):
            actual = archive.project_archive(raw, start_date=START, end_date=END)
        self.assertTrue(calls)
        self.assertEqual(actual, [expected(row())])

    def test_actual_json_numbers_cannot_masquerade_as_core_strings(self):
        for key in ("a", "d", "z3a", "t3a", "h", "i"):
            for value in (123456789, 1.5, True, None, [], {}):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    project([row(**{key: value})])

    def test_each_core_metadata_key_is_required(self):
        for key in ("a", "d", "z3a", "t3a", "h", "i"):
            original = row()
            del original[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                project([original])

    def test_duplicate_json_keys_rejected_even_in_ignored_objects(self):
        base = encoded(row())[:-1]
        fixtures = [
            b'{"securityList":[],"securityList":' + encoded([row()]) + b"}",
            b"[" + base + b',"a":"912ABC123"}]',
            b"[" + base + b',"ignored":{"x":1,"x":2}}]',
        ]
        for raw in fixtures:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                archive.project_archive(raw, start_date=START, end_date=END)

    def test_nonstandard_json_numeric_constants_rejected_in_unknown_fields(self):
        for token in (b"NaN", b"Infinity", b"-Infinity"):
            raw = b"[" + encoded(row())[:-1] + b',"ignored":' + token + b"}]"
            with self.subTest(token=token), self.assertRaises(ValueError):
                archive.project_archive(raw, start_date=START, end_date=END)

    def test_bytes_only_valid_json_and_no_jsonp(self):
        for raw in (
            encoded([row()]).decode(),
            bytearray(encoded([row()])),
            None,
            b"",
            b"\xff",
            b"callback(" + encoded([row()]) + b");",
            b"[",
        ):
            with (
                self.subTest(kind=type(raw), raw=repr(raw)[:80]),
                self.assertRaises((TypeError, ValueError)),
            ):
                archive.project_archive(raw, start_date=START, end_date=END)

    def test_nonempty_rows_and_exact_wrapper_shape(self):
        for value in (
            [],
            {"securityList": []},
            {},
            {"rows": [row()]},
            {"securityList": [row()], "extra": None},
            {"securityList": row()},
            [None],
            [[]],
            ["row"],
            None,
            3,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                project(value)

    def test_cusip_requires_nine_uppercase_ascii_alphanumeric_characters(self):
        for value in (
            "912abc123",
            "912ABC12",
            "912ABC1234",
            "912-BC123",
            "912ABC12 ",
            "９12ABC123",
            "",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                project([row(a=value)])

    def test_real_exact_date_lexemes_required_for_both_date_fields(self):
        for key in ("h", "i"):
            for value in (
                "2018-02-30",
                "2019-02-29",
                "2018-1-02",
                "01/02/2018",
                "2018-01-02 ",
                "2018-01-02T00:00:00Z",
                "2018-01-02T01:00:00",
                "2018-01-02T00:00:00.000",
                "2018-01-02 00:00:00",
                "",
            ):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    project([row(**{key: value})])

    def test_announcement_cannot_follow_auction(self):
        with self.assertRaises(ValueError):
            project([row(h="2018-01-03")])

    def test_auction_outside_requested_window_rejects_whole_archive(self):
        for date in ("2017-12-31", "2018-01-04"):
            with self.subTest(date=date), self.assertRaises(ValueError):
                project(
                    [row(), row(h="2017-12-28", i=date)],
                    start_date="2018-01-01",
                    end_date="2018-01-03",
                )

    def test_bounds_cannot_relax_fixed_scope_or_reverse_calendar(self):
        for start, end in (
            ("2009-12-31", END),
            (START, "2025-10-21"),
            ("2018-01-03", "2018-01-02"),
            ("2018-02-30", END),
            (START, "2025-02-29"),
            (None, END),
            (START, 20251020),
        ):
            with (
                self.subTest(start=start, end=end),
                self.assertRaises((TypeError, ValueError)),
            ):
                project([row()], start_date=start, end_date=end)

    def test_all_row_dates_checked_before_any_filename_projection(self):
        for bad_date in ("2025-10-21", "2018-02-30"):
            with (
                self.subTest(bad_date=bad_date),
                self.assertRaisesRegex(ValueError, "(?i)(date|auction|scope|bound)"),
            ):
                project([row(f3="../unsafe.pdf"), row(i=bad_date)])

    def test_duplicate_auction_cusip_rejected_after_date_normalization(self):
        for duplicate in (
            row(),
            row(i="2018-01-02T00:00:00"),
            row(d="26-Week", f3="different.pdf"),
        ):
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                project([row(), duplicate])

    def test_document_fields_require_strings_or_explicit_empty_states(self):
        for key in ("e3", "f3", "f31", "f32", "e4", "ia1"):
            for value in (123, 1.5, False, [], {}):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    project([row(**{key: value})])

    def test_pdf_names_reject_path_url_query_and_wrong_extension(self):
        for filename in (
            "../x.pdf",
            "/x.pdf",
            "//example.com/x.pdf",
            "https://example.com/x.pdf",
            "folder/x.pdf",
            "folder\\x.pdf",
            "x.pdf?download=1",
            "x.pdf#fragment",
            " x.pdf",
            "x.pdf ",
            "x y.pdf",
            "x.xml",
            "x.pdf.exe",
            "x%2epdf",
        ):
            for key in ("e3", "f3", "f31", "f32"):
                with self.subTest(key=key, filename=filename), self.assertRaises(ValueError):
                    project([row(**{key: filename})])

    def test_xml_fields_require_safe_xml_names(self):
        for key in ("e4", "ia1"):
            for filename in ("x.pdf", "../x.xml", "x.xml?x=1", "x.xml, y.xml"):
                with self.subTest(key=key, filename=filename), self.assertRaises(ValueError):
                    project([row(**{key: filename})])

    def test_only_special_pdf_accepts_exact_comma_space_lists(self):
        for key, value in (
            ("f32", "a.pdf,b.pdf"),
            ("f32", "a.pdf,  b.pdf"),
            ("f32", "a.pdf, "),
            ("f32", ", a.pdf"),
            ("f32", "a.pdf; b.pdf"),
            ("f3", "a.pdf, b.pdf"),
            ("e3", "a.pdf, b.pdf"),
            ("f31", "a.pdf, b.pdf"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                project([row(**{key: value})])


if __name__ == "__main__":
    unittest.main()
