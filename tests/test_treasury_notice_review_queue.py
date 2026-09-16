"""Invented text contracts for review grouping, never document admission."""

import decimal
import hashlib
import unittest
from unittest.mock import patch

from src.treasury_notice_review_queue import template_fingerprint


class TreasuryNoticeReviewQueueTests(unittest.TestCase):
    def test_date_amount_and_cusip_variants_collapse_with_typed_markers(self):
        first = (
            "FOR RELEASE May 7, 2021; auction 05/12/21. "
            "CUSIP 912ABC123; amount $1,234.50; yield 2.375%."
        )
        second = (
            "FOR RELEASE OCTOBER 28, 2024; auction 10/31/24. "
            "CUSIP 987XYZ456; amount $8,765.90; yield 6.125%."
        )
        expected = (
            "FOR RELEASE ⟦DATE_MONTH⟧ ⟦DATE_DAY⟧, ⟦DATE_YEAR⟧; "
            "auction ⟦DATE_MONTH⟧/⟦DATE_DAY⟧/⟦DATE_YEAR⟧. "
            "CUSIP ⟦CUSIP⟧; amount $⟦NUMBER⟧,⟦NUMBER⟧.⟦NUMBER⟧; "
            "yield ⟦NUMBER⟧.⟦NUMBER⟧%."
        )
        self.assertEqual(template_fingerprint(first), template_fingerprint(second))
        self.assertEqual(
            template_fingerprint(first),
            {
                "normalized_text": expected,
                "template_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
            },
        )

    def test_negation_addendum_correction_test_and_conditional_words_survive(self):
        texts = [
            "Auction results changed.",
            "Auction results not changed.",
            "Auction results addendum.",
            "Auction results correction.",
            "Auction results test.",
            "If auction results changed.",
            "Auction results unchanged.",
        ]
        results = [template_fingerprint(text) for text in texts]
        self.assertEqual([result["normalized_text"] for result in results], texts)
        self.assertEqual(len({result["template_sha256"] for result in results}), len(texts))

    def test_same_cusip_separate_events_remain_separate_caller_records(self):
        records = [
            {"event": "2021-05-12", "text": "Auction May 12, 2021 CUSIP 912ABC123."},
            {"event": "2021-06-16", "text": "Auction June 16, 2021 CUSIP 912ABC123."},
        ]
        queue = [(record["event"], template_fingerprint(record["text"])) for record in records]
        self.assertEqual([item[0] for item in queue], ["2021-05-12", "2021-06-16"])
        self.assertEqual(queue[0][1], queue[1][1])
        self.assertEqual(len(records), 2)
        self.assertEqual(set(queue[0][1]), {"template_sha256", "normalized_text"})

    def test_order_context_punctuation_headers_and_parentheticals_are_retained(self):
        texts = [
            "HEADER CONTACT Desk. If result changes (see addendum), retain original.",
            "HEADER CONTACT Desk. Retain original, if result changes (see addendum).",
            "HEADER CONTACT Other. If result changes (see addendum), retain original.",
            "HEADER CONTACT Desk. If result changes, retain original.",
            "HEADER CONTACT Desk. If result changes (see addendum); retain original.",
        ]
        self.assertEqual([template_fingerprint(t)["normalized_text"] for t in texts], texts)
        self.assertEqual(len({template_fingerprint(t)["template_sha256"] for t in texts}), 5)
        comma = template_fingerprint("May 12, 2021")
        no_comma = template_fingerprint("May 12 2021")
        self.assertNotEqual(comma, no_comma)
        self.assertNotEqual(
            template_fingerprint("CUSIP 912abc123"), template_fingerprint("CUSIP 912ABC123")
        )
        self.assertEqual(
            template_fingerprint("ABCDEFGHI X912ABC123Z _912ABC123_")["normalized_text"],
            "ABCDEFGHI X⟦NUMBER⟧ABC⟦NUMBER⟧Z _⟦NUMBER⟧ABC⟦NUMBER⟧_",
        )

    def test_month_words_require_adjacent_numeric_day_and_year(self):
        text = "May change. March onward. May 12 bids. May be revised 2021. December auction."
        self.assertEqual(
            template_fingerprint(text)["normalized_text"],
            "May change. March onward. May ⟦NUMBER⟧ bids. May be revised ⟦NUMBER⟧. December auction.",
        )
        self.assertEqual(
            template_fingerprint("May 12th, 2021")["normalized_text"],
            "May ⟦NUMBER⟧th, ⟦NUMBER⟧",
        )

    def test_nfkc_whitespace_hash_and_repeatability(self):
        unicode_text = (
            "\n Ｒｅｌｅａｓｅ\u00a0Ｍａｙ\t７,\n２０２１ \r\nＣＵＳＩＰ ９１２ＡＢＣ１２３.\f"
        )
        plain = "Release May 7, 2021 CUSIP 912ABC123."
        first = template_fingerprint(unicode_text)
        self.assertEqual(first, template_fingerprint(plain))
        self.assertEqual(first, template_fingerprint(unicode_text))
        self.assertEqual(
            first["template_sha256"],
            hashlib.sha256(first["normalized_text"].encode("utf-8")).hexdigest(),
        )

    def test_lexical_only_and_strict_text_limits(self):
        text = "Amount $" + "7" * 12000 + "; yield -9.7e99999%; retain NaN wording."
        with (
            patch("builtins.float", side_effect=AssertionError("no float conversion")),
            patch.object(
                decimal, "Decimal", side_effect=AssertionError("no decimal conversion")
            ),
        ):
            result = template_fingerprint(text)
        self.assertEqual(
            result["normalized_text"],
            "Amount $⟦NUMBER⟧; yield -⟦NUMBER⟧.⟦NUMBER⟧e⟦NUMBER⟧%; retain NaN wording.",
        )
        self.assertEqual(
            template_fingerprint("x" * (2 * 1024 * 1024))["normalized_text"],
            "x" * (2 * 1024 * 1024),
        )
        for value in [
            None,
            b"text",
            123,
            True,
            {},
            "",
            " \n\t",
            "bad\x00text",
            "bad\ud800text",
            "bad\x1btext",
            "⟦CUSIP⟧",
            "x" * (2 * 1024 * 1024 + 1),
            "é" * (1024 * 1024 + 1),
        ]:
            with (
                self.subTest(
                    kind=type(value).__name__,
                    length=len(value) if isinstance(value, (str, bytes)) else None,
                ),
                self.assertRaises(ValueError),
            ):
                template_fingerprint(value)


if __name__ == "__main__":
    unittest.main()
