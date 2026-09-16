"""Generated tests for preserving an observed empty archive display bucket."""

import json
import unittest

from src.treasury_auction_inventory import project_archive


def record(**changes):
    row = dict(
        a="912ABC123",
        d="56-Day",
        z3a="CMB",
        t3a="",
        h="2010-01-04",
        i="2010-01-05",
        f3="result.pdf",
    )
    row.update(changes)
    return row


def run(rows):
    return project_archive(
        json.dumps(rows).encode(), start_date="2010-01-01", end_date="2010-12-31"
    )


class TreasuryInventoryTests(unittest.TestCase):
    def test_empty_display_bucket_is_retained_without_inferred_term(self):
        rows = [record(), record(a="912ABC124", d="2-Year", z3a="Note", t3a="2-Year")]
        projected = run(rows)
        self.assertEqual(len(projected), 2)
        self.assertEqual(projected[0]["term_bucket"], "")
        self.assertEqual(projected[0]["security_term"], "56-Day")
        self.assertEqual(projected[1]["term_bucket"], "2-Year")
        self.assertEqual(
            projected[0]["documents"]["competitive_pdf"],
            [
                "https://www.treasurydirect.gov/instit/annceresult/press/preanre/2010/result.pdf"
            ],
        )

    def test_missing_null_whitespace_numeric_bucket_still_fail(self):
        for value in [None, " ", 2, False, [], {}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                run([record(t3a=value)])
        missing = record()
        del missing["t3a"]
        with self.assertRaises(ValueError):
            run([missing])

    def test_other_core_fields_cannot_be_empty(self):
        for key in ["a", "d", "z3a", "h", "i"]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                run([record(**{key: ""})])

    def test_unknown_bucket_does_not_bypass_dates_or_filenames(self):
        for changed in [
            dict(i="2011-01-01"),
            dict(h="2010-01-06"),
            dict(f3="../x.pdf"),
            dict(f3="\u212a.pdf"),
        ]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                run([record(**changed)])
        with self.assertRaises(ValueError):
            run([])
        with self.assertRaises(ValueError):
            run([record(), record()])

    def test_numeric_fields_stay_quarantined(self):
        raw = (
            b"["
            + json.dumps(record()).encode()[:-1]
            + b',"unadmitted":'
            + b"9" * 10000
            + b"}]"
        )
        result = project_archive(raw, start_date="2010-01-01", end_date="2010-12-31")
        self.assertEqual(result, run([record()]))

    def test_strict_v1_still_rejects_blank_bucket(self):
        from src.treasury_auction_archive import project_archive as original

        with self.assertRaises(ValueError):
            original(
                json.dumps([record()]).encode(), start_date="2010-01-01", end_date="2010-12-31"
            )


if __name__ == "__main__":
    unittest.main()
