"""Prewritten exact test-selection contracts; never run historical fixtures."""

import unittest

from src.treasury_dealer_test_gate import QUARANTINED, select_tests


class FakeCase:
    def __init__(self, name):
        self.name = name

    def id(self):
        return self.name

    def __call__(self, *args, **kwargs):
        raise AssertionError("Selection must not execute a test")


class TreasuryTestGateTests(unittest.TestCase):
    def test_exact_quarantine_and_all_other_tests_remain_ordered(self):
        names = ["generated.first", *sorted(QUARANTINED), "generated.last"]
        suite, record = select_tests(unittest.TestSuite(FakeCase(n) for n in names))
        self.assertEqual([t.id() for t in suite], ["generated.first", "generated.last"])
        self.assertEqual(record["quarantined_test_ids"], sorted(QUARANTINED))
        self.assertEqual(record["discovered_count"], 8)
        self.assertEqual(record["selected_count"], 2)

    def test_missing_quarantine_identity_or_duplicate_test_fails_closed(self):
        names = [*sorted(QUARANTINED), "generated.safe"]
        for bad in (names[1:], names + ["generated.safe"]):
            with self.assertRaises(ValueError):
                select_tests(unittest.TestSuite(FakeCase(n) for n in bad))


if __name__ == "__main__":
    unittest.main()
