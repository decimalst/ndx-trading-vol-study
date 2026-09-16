"""Generated prospective term-token contracts; no historical term application."""

import unittest
from unittest.mock import patch

from src import treasury_history_terms as terms


class TreasuryHistoryTermsTests(unittest.TestCase):
    def test_ordinary_terms_have_exact_named_output_for_both_roles(self):
        for text, years, months, kind in (
            ("1-Year Note", 1, 0, "NOTE"),
            ("30-Year Bond", 30, 0, "BOND"),
            ("8-Year 11-Month Note", 8, 11, "NOTE"),
            ("12-Year 0-Month Bond", 12, 0, "BOND"),
        ):
            for announcement in (False, True):
                with self.subTest(text=text, announcement=announcement):
                    self.assertEqual(
                        terms.parse_offered_term(text, announcement=announcement),
                        {
                            "years": years,
                            "months": months,
                            "kind": kind,
                            "canonical_term": text,
                            "coupon_token_present": False,
                            "split_month_token": False,
                        },
                    )

    def test_announcement_coupon_forms_preserve_descriptor_and_remove_only_token(self):
        for coupon in (
            "0%",
            "7%",
            "12%",
            "0-1/8%",
            "6-1/2%",
            "6-1/4%",
            "6-3/4%",
            "6-1/8%",
            "6-3/8%",
            "6-5/8%",
            "6-7/8%",
        ):
            for descriptor, kind in (("8-Year 7-Month", "Note"), ("20-Year", "Bond")):
                text = f"{descriptor} {coupon} {kind}"
                with self.subTest(text=text):
                    parsed = terms.parse_offered_term(text, announcement=True)
                    self.assertEqual(parsed["canonical_term"], f"{descriptor} {kind}")
                    self.assertTrue(parsed["coupon_token_present"])
                    self.assertFalse(parsed["split_month_token"])
                    self.assertEqual(parsed["kind"], kind.upper())
                    with self.assertRaises(ValueError):
                        terms.parse_offered_term(text, announcement=False)

    def test_exact_split_month_token_is_local_and_announcement_only(self):
        text = "8-Year 7-Mont h 6-3/8% Note"
        self.assertEqual(
            terms.parse_offered_term(text, announcement=True),
            {
                "years": 8,
                "months": 7,
                "kind": "NOTE",
                "canonical_term": "8-Year 7-Month Note",
                "coupon_token_present": True,
                "split_month_token": True,
            },
        )
        self.assertEqual(text, "8-Year 7-Mont h 6-3/8% Note")
        for bad in (
            "8-Year 7-Mon th 6-3/8% Note",
            "8-Year 7-Mont  h 6-3/8% Note",
            "8-Year 7-Mont\th 6-3/8% Note",
            "8-Year 7-Mont\nh 6-3/8% Note",
            "8-Y ear 7-Month 6-3/8% Note",
            "8-Year 7-Month 6-3/8% No te",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                terms.parse_offered_term(bad, announcement=True)
        with self.assertRaises(ValueError):
            terms.parse_offered_term("8-Year 7-Mont h Note", announcement=False)

    def test_year_month_bounds_and_ambiguous_integers_fail(self):
        for text in (
            "0-Year Note",
            "31-Year Bond",
            "100-Year Bond",
            "-1-Year Note",
            "+1-Year Note",
            "01-Year Note",
            "8-Year 12-Month Note",
            "8-Year 100-Month Note",
            "8-Year -1-Month Note",
            "8-Year +1-Month Note",
            "8-Year 07-Month Note",
            "8-Year 00-Month Note",
            "8.0-Year Note",
            "8-Year 7.0-Month Note",
            "8-Year 7e0-Month Note",
            "８-Year Note",
        ):
            for announcement in (False, True):
                with (
                    self.subTest(text=text, announcement=announcement),
                    self.assertRaises(ValueError),
                ):
                    terms.parse_offered_term(text, announcement=announcement)

    def test_unsupported_coupon_tokens_cannot_be_silently_ignored(self):
        for coupon in (
            "07%",
            "00%",
            "06-1/2%",
            "6-01/2%",
            "6-1/02%",
            "6-1/0%",
            "6-0/8%",
            "6-8/8%",
            "6-9/8%",
            "6-1/16%",
            "6-2/4%",
            "6/8%",
            "6.5%",
            "+6%",
            "-6%",
            "6e0%",
            "NaN%",
            "Infinity%",
            "6-1/2",
            "6 %",
            "6-1 /2%",
            "6%-7%",
            "6% 7%",
            "coupon",
        ):
            with self.subTest(coupon=coupon), self.assertRaises(ValueError):
                terms.parse_offered_term(f"8-Year 7-Month {coupon} Note", announcement=True)

    def test_exact_case_spacing_kind_position_and_extra_text_are_strict(self):
        for text in (
            "8-year Note",
            "8-Year NOTE",
            "8-Year note",
            "8-Year Bill",
            "8-Year TIPS",
            "8-Year FRN",
            "8-Year Notes",
            "8-Year Bond Note",
            " 8-Year Note",
            "8-Year Note ",
            "8-Year  Note",
            "8-Year\tNote",
            "8-Year Note\n",
            "8-Year Note\x00",
            "8-Year\u00a0Note",
            "8-Year 6% 7-Month Note",
            "6% 8-Year Note",
            "8-Year Note 6%",
            "8-Year Note (Reopening)",
            "8-Year 7-Month 6% Note extra",
            "Term and Type of Security 8-Year Note",
            "8-Year Note\n8-Year Note",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                terms.parse_offered_term(text, announcement=True)

    def test_strict_input_types_and_explicit_role(self):
        for text in (None, b"8-Year Note", 8, True, [], {}, ""):
            with self.subTest(text=text), self.assertRaises(ValueError):
                terms.parse_offered_term(text, announcement=True)
        for role in (None, 0, 1, "announcement", [], {}):
            with self.subTest(role=role), self.assertRaises(ValueError):
                terms.parse_offered_term("8-Year Note", announcement=role)
        with self.assertRaises(TypeError):
            terms.parse_offered_term("8-Year Note")
        with self.assertRaises(TypeError):
            terms.parse_offered_term("8-Year Note", True)

    def test_coupon_is_not_numerically_decoded(self):
        calls = []

        def descriptor_int(token):
            calls.append(token)
            self.assertIn(token, ("8", "7"))
            return int(token)

        with (
            patch.object(terms, "int", side_effect=descriptor_int, create=True),
            patch("builtins.float", side_effect=AssertionError("No coupon float decoding")),
        ):
            result = terms.parse_offered_term("8-Year 7-Month 6-3/8% Note", announcement=True)
        self.assertEqual(calls, ["8", "7"])
        self.assertEqual(result["canonical_term"], "8-Year 7-Month Note")


if __name__ == "__main__":
    unittest.main()
