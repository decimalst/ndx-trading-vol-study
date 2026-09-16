"""Prewritten generated contracts for blind Treasury XML schema inspection."""

import unittest
from unittest.mock import patch

from src import treasury_auction_document_probe as probe


class TreasuryAuctionDocumentProbeTests(unittest.TestCase):
    def test_exact_paths_counts_and_attribute_name_unions(self):
        raw = (
            b'<Auction release="DO NOT EXPORT"><Security code="SECRET" z="1">'
            b'<Result unit="PERCENT">123.456</Result>PRIVATE TAIL</Security>'
            b'<Security a="OTHER SECRET" code="ANOTHER SECRET">'
            b'<Result>999</Result><Result unit="DOLLARS">888</Result>'
            b"</Security></Auction>"
        )
        self.assertEqual(
            probe.inspect_xml_schema(raw),
            {
                "root_tag": "Auction",
                "tag_counts": {
                    "Auction": 1,
                    "Auction/Security": 2,
                    "Auction/Security/Result": 3,
                },
                "attribute_names": {
                    "Auction": ["release"],
                    "Auction/Security": ["a", "code", "z"],
                    "Auction/Security/Result": ["unit"],
                },
            },
        )

    def test_root_and_empty_leaves_remain_in_schema(self):
        self.assertEqual(
            probe.inspect_xml_schema(b"<Root><Empty/><Child><Empty/></Child></Root>"),
            {
                "root_tag": "Root",
                "tag_counts": {
                    "Root": 1,
                    "Root/Empty": 1,
                    "Root/Child": 1,
                    "Root/Child/Empty": 1,
                },
                "attribute_names": {
                    "Root": [],
                    "Root/Empty": [],
                    "Root/Child": [],
                    "Root/Child/Empty": [],
                },
            },
        )

    def test_case_and_expanded_namespace_names_preserved_literally(self):
        ns = "{https://example.test/treasury/schema}"
        attr_ns = "{https://example.test/attributes}"
        raw = (
            b'<Auction xmlns="https://example.test/treasury/schema" '
            b'xmlns:a="https://example.test/attributes" a:Id="opaque">'
            b'<Result a:Unit="secret" Unit="secret"/>'
            b'<result/><Result xmlns=""/></Auction>'
        )
        self.assertEqual(
            probe.inspect_xml_schema(raw),
            {
                "root_tag": ns + "Auction",
                "tag_counts": {
                    ns + "Auction": 1,
                    ns + "Auction/" + ns + "Result": 1,
                    ns + "Auction/" + ns + "result": 1,
                    ns + "Auction/Result": 1,
                },
                "attribute_names": {
                    ns + "Auction": [attr_ns + "Id"],
                    ns + "Auction/" + ns + "Result": ["Unit", attr_ns + "Unit"],
                    ns + "Auction/" + ns + "result": [],
                    ns + "Auction/Result": [],
                },
            },
        )

    def test_huge_numeric_and_future_date_text_are_opaque_and_omitted(self):
        expected = {
            "root_tag": "Root",
            "tag_counts": {"Root": 1, "Root/Value": 1, "Root/Date": 1},
            "attribute_names": {"Root": [], "Root/Value": [], "Root/Date": []},
        }
        for token in (b"9" * 10000, b"1e999999999999999999999", b"NaN", b"not a number"):
            with self.subTest(token_prefix=token[:24]):
                raw = b"<Root><Value>" + token + b"</Value><Date>2099-12-31</Date></Root>"
                self.assertEqual(probe.inspect_xml_schema(raw), expected)

    def test_values_text_tails_comments_and_processing_instructions_not_exported(self):
        baseline = b'<Root flag="a"><Child key="b"/></Root>'
        private = (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<?private SECRET?><Root flag="SECRET ATTRIBUTE">SECRET TEXT'
            b'<!-- PRIVATE COMMENT --><Child key="987654321">SECRET LEAF</Child>'
            b"PRIVATE TAIL</Root>"
        )
        self.assertEqual(probe.inspect_xml_schema(private), probe.inspect_xml_schema(baseline))

    def test_nonbytes_empty_and_invalid_utf8_rejected(self):
        for raw in (
            None,
            "<Root/>",
            bytearray(b"<Root/>"),
            b"",
            b"  \n",
            b"<Root>\xff</Root>",
        ):
            with (
                self.subTest(kind=type(raw), raw=repr(raw)),
                self.assertRaises((TypeError, ValueError)),
            ):
                probe.inspect_xml_schema(raw)

    def test_malformed_xml_rejected(self):
        for raw in (
            b"<Root>",
            b"<Root/><Other/>",
            b"<Root><Child></Root>",
            b'<Root duplicate="a" duplicate="b"/>',
            b"<Root>&undefined;</Root>",
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                probe.inspect_xml_schema(raw)

    def test_dtd_and_entity_declarations_rejected_case_insensitively(self):
        for raw in (
            b'<!DOCTYPE Root [<!ENTITY secret "SECRET">]><Root>&secret;</Root>',
            b'<!DOCTYPE Root SYSTEM "https://example.test/secret.dtd"><Root/>',
            b'<!DOCTYPE Root [<!ENTITY secret SYSTEM "file:///secret">]><Root/>',
            b"<!doctype Root><Root/>",
            b'<!EnTiTy secret "SECRET"><Root/>',
            b"<Root><!-- <!DOCTYPE hidden> --></Root>",
            b'<Root><![CDATA[<!ENTITY hidden "SECRET">]]></Root>',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                probe.inspect_xml_schema(raw)

    def test_two_mebibyte_limit_applies_to_raw_bytes_inclusively(self):
        limit = 2 * 1024 * 1024
        prefix, suffix = b"<Root>", b"</Root>"
        raw = prefix + b"9" * (limit - len(prefix) - len(suffix)) + suffix
        self.assertEqual(len(raw), limit)
        self.assertEqual(
            probe.inspect_xml_schema(raw),
            {"root_tag": "Root", "tag_counts": {"Root": 1}, "attribute_names": {"Root": []}},
        )
        with self.assertRaises(ValueError):
            probe.inspect_xml_schema(raw + b" ")

    def test_reference_attributes_do_not_open_files_or_network(self):
        xi = "{http://www.w3.org/2001/XInclude}include"
        raw = (
            b'<Root xmlns:xi="http://www.w3.org/2001/XInclude">'
            b'<xi:include href="file:///not-a-real-source" parse="text"/>'
            b'<xi:include href="https://example.test/not-a-real-source"/>'
            b"</Root>"
        )
        with (
            patch("builtins.open", side_effect=AssertionError("file access forbidden")),
            patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")),
        ):
            result = probe.inspect_xml_schema(raw)
        self.assertEqual(
            result,
            {
                "root_tag": "Root",
                "tag_counts": {"Root": 1, "Root/" + xi: 2},
                "attribute_names": {"Root": [], "Root/" + xi: ["href", "parse"]},
            },
        )


if __name__ == "__main__":
    unittest.main()
