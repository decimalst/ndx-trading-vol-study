"""Generated metadata-only contracts, written before verifier implementation."""

import copy
import json
import unittest

from src.verify_treasury_auction_archive import verify_archive

PDF = "https://www.treasurydirect.gov/instit/annceresult/press/preanre/"
XML = "https://www.treasurydirect.gov/xml/"
DOCS = (
    "announcement_pdf",
    "competitive_pdf",
    "noncompetitive_pdf",
    "special_pdf",
    "announcement_xml",
    "competitive_xml",
)


def source(**changes):
    row = {
        "a": "AB1234567",
        "d": "9-Year 11-Month",
        "z3a": "Note",
        "t3a": "10-Year",
        "h": "2009-12-31T00:00:00",
        "i": "2010-01-05T00:00:00",
        "e3": "A_release.PDF",
        "f3": "R_release.pdf",
        "f31": "N_release.pdf",
        "f32": "S_first.pdf, S_second.pdf",
        "e4": "A_release.xml",
        "ia1": "R_release.XML",
    }
    row.update(changes)
    return row


def expected():
    return {
        "cusip": "AB1234567",
        "security_term": "9-Year 11-Month",
        "security_type": "Note",
        "term_bucket": "10-Year",
        "announcement_date": "2009-12-31",
        "auction_date": "2010-01-05",
        "documents": {
            "announcement_pdf": [PDF + "2009/A_release.PDF"],
            "competitive_pdf": [PDF + "2010/R_release.pdf"],
            "noncompetitive_pdf": [PDF + "2010/N_release.pdf"],
            "special_pdf": [PDF + "2010/S_first.pdf", PDF + "2010/S_second.pdf"],
            "announcement_xml": [XML + "A_release.xml"],
            "competitive_xml": [XML + "R_release.XML"],
        },
    }


def call(raw, projected, start="2010-01-01", end="2010-12-31"):
    payload = raw if isinstance(raw, bytes) else json.dumps(raw).encode()
    return verify_archive(payload, projected, start_date=start, end_date=end)


class VerifyTreasuryAuctionArchiveTests(unittest.TestCase):
    def test_all_documents_prior_year_and_exact_wrapper(self):
        for raw in ([source()], {"securityList": [source()]}):
            with self.subTest(wrapper=isinstance(raw, dict)):
                self.assertEqual(call(raw, [expected()]), {"status": "VERIFIED", "records": 1})

    def test_empty_response_rejected_and_missing_links_remain_six_empty_lists(self):
        for raw in ([], {"securityList": []}):
            with self.subTest(empty=raw), self.assertRaises(ValueError):
                call(raw, [])
        row = source()
        for key in ("e3", "f3", "f31", "f32", "e4", "ia1"):
            row.pop(key)
        wanted = expected()
        wanted["documents"] = {key: [] for key in DOCS}
        for replacement in ({}, {"e3": None, "f3": "", "f32": None}):
            with self.subTest(replacement=replacement):
                self.assertEqual(call([row | replacement], [wanted])["records"], 1)

    def test_numeric_tokens_are_ignored_without_numeric_conversion(self):
        ordinary = json.dumps(source())
        large_integer = "9" * 7000
        payload = (
            "["
            + ordinary[:-1]
            + ',"ignored":{"integer":'
            + large_integer
            + ',"overflow":1e999999999,"underflow":-1e-999999999,'
            + '"array":[-0,2.12345678901234567890]}}]'
        ).encode()
        self.assertEqual(call(payload, [expected()])["records"], 1)

    def test_duplicate_keys_constants_bad_utf8_and_envelopes_fail(self):
        valid = json.dumps([source()])
        bad = (
            valid.replace('"a": "AB1234567"', '"a":"AB1234567","a":"AB1234567"'),
            valid[:-2] + ',"ignored":{"x":1,"x":2}}]',
            valid[:-2] + ',"ignored":NaN}]',
            valid[:-2] + ',"ignored":Infinity}]',
            valid[:-2] + ',"ignored":-Infinity}]',
            "callback(" + valid + ")",
            json.dumps({"securityList": [source()], "count": 1}),
            json.dumps({"data": [source()]}),
            "null",
        )
        for payload in (*[s.encode() for s in bad], b"\xff"):
            with self.subTest(payload=payload[:65]), self.assertRaises(ValueError):
                call(payload, [expected()])

    def test_actual_metadata_strings_and_valid_cusip_are_required(self):
        for key in ("a", "d", "z3a", "t3a", "h", "i"):
            for value in (None, 123456789, True, [], {}, "", "   "):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    call([source(**{key: value})], [expected()])
        for cusip in ("ab1234567", "AB123456", "AB12345678", "AB12/4567"):
            with self.subTest(cusip=cusip), self.assertRaises(ValueError):
                call([source(a=cusip)], [expected()])
        with self.assertRaises(ValueError):
            call([source(), []], [expected()])

    def test_date_bounds_precision_calendar_and_whole_response_preflight(self):
        bad_dates = (
            "2010-02-30",
            "2010-01-05T01:00:00",
            "2010-01-05T00:00:00Z",
            "2010-01-05T00:00:00.000",
            "2010-1-5",
            "2011-01-01",
            "2009-12-31",
        )
        for date in bad_dates:
            with self.subTest(date=date), self.assertRaises(ValueError):
                call([source(i=date)], [expected()])
        with self.assertRaises(ValueError):
            call([source(h="2010-01-06")], [expected()])
        for start, end in (
            ("2009-12-31", "2010-12-31"),
            ("2010-01-01", "2025-10-21"),
            ("2010-12-31", "2010-01-01"),
            ("2010-01-01T00:00:00", "2010-12-31"),
        ):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                call([], [], start, end)
        with self.assertRaisesRegex(ValueError, "[Dd]ate|[Ii]nterval|[Bb]ound"):
            call(
                [source(e3="../bad.pdf"), source(a="BB1234567", i="2011-01-01")],
                [],
            )

    def test_duplicate_identity_fails_but_reopening_and_tie_sort_are_valid(self):
        with self.assertRaises(ValueError):
            call([source(), source(i="2010-01-05")], [expected(), expected()])
        later = source(i="2010-02-05", h="2010-02-01")
        another = source(a="AA1234567")
        wanted_later = expected()
        wanted_later.update(auction_date="2010-02-05", announcement_date="2010-02-01")
        wanted_later["documents"]["announcement_pdf"] = [PDF + "2010/A_release.PDF"]
        wanted_another = expected() | {"cusip": "AA1234567"}
        ordered = [wanted_another, expected(), wanted_later]
        self.assertEqual(call([later, source(), another], ordered)["records"], 3)
        with self.assertRaises(ValueError):
            call([later, source(), another], list(reversed(ordered)))

    def test_document_types_paths_extensions_and_separator_are_literal(self):
        unsafe = (
            "../a.pdf",
            "/a.pdf",
            "folder/a.pdf",
            "https://evil.example/a.pdf",
            "//evil.example/a.pdf",
            "a.pdf?x=1",
            "a.pdf#fragment",
            "a%2Fname.pdf",
            "a\\b.pdf",
            "a.b.pdf",
            " a.pdf",
            "a.pdf ",
            " ",
            "a.xml",
            "a.pdf, b.pdf",
            [],
            42,
        )
        for value in unsafe:
            with self.subTest(value=value), self.assertRaises(ValueError):
                call([source(e3=value)], [expected()])
        for value in ("one.pdf,two.pdf", "one.pdf,  two.pdf", "one.pdf, ", "one.pdf; two.pdf"):
            with self.subTest(special=value), self.assertRaises(ValueError):
                call([source(f32=value)], [expected()])
        with self.assertRaises(ValueError):
            call([source(e4="a.pdf")], [expected()])

    def test_projected_metadata_missing_extra_and_url_tampering_fail(self):
        mutations = []
        for key in expected():
            item = expected()
            item.pop(key)
            mutations.append([item])
        mutations.extend(([], [expected(), expected()], [expected() | {"extra": 1}]))
        for key in (
            "cusip",
            "security_term",
            "security_type",
            "term_bucket",
            "announcement_date",
            "auction_date",
        ):
            mutations.append([expected() | {key: "tampered"}])
        for key in DOCS:
            item = expected()
            item["documents"][key] = []
            mutations.append([item])
        item = expected()
        item["documents"]["competitive_pdf"] = ["https://evil.example/R_release.pdf"]
        mutations.append([item])
        item = expected()
        item["documents"]["announcement_pdf"] = [PDF + "2010/A_release.PDF"]
        mutations.append([item])
        item = expected()
        item["documents"]["extra"] = []
        mutations.append([item])
        for projected in mutations:
            with self.subTest(projected=projected), self.assertRaises(ValueError):
                call([source()], projected)

    def test_inclusive_ceiling_and_verbatim_core_values(self):
        row = source(i="2025-10-20", h="2025-10-19", d="  10-Year  ")
        wanted = expected()
        wanted.update(
            auction_date="2025-10-20",
            announcement_date="2025-10-19",
            security_term="  10-Year  ",
        )
        for key in (
            "announcement_pdf",
            "competitive_pdf",
            "noncompetitive_pdf",
            "special_pdf",
        ):
            wanted["documents"][key] = [
                url.replace("/2009/", "/2025/").replace("/2010/", "/2025/")
                for url in wanted["documents"][key]
            ]
        before = copy.deepcopy(wanted)
        self.assertEqual(call([row], [wanted], "2025-01-01", "2025-10-20")["records"], 1)
        self.assertEqual(wanted, before)


if __name__ == "__main__":
    unittest.main()
