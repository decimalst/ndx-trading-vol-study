"""The environment fence.

`reports/FROZEN_REPORT_HASHES.json` pins the frozen reports byte-for-byte, but a
digest over an output says nothing about the code path that produced it. If the
numerical stack drifts, the artifacts stop reproducing and the failure surfaces
as "metrics do not recompute" — which reads as a corrupted artifact and sends
the reader after the science instead of after their `pip list`.

That happened: a reviewer on scikit-learn 1.8, outside the `>=1.7,<1.8` pin in
requirements.txt, got three GBM artifact failures with no indication that the
environment was the cause.

So the pin is now asserted, and the artifact verifiers append the diagnosis.
This test is informational when it fails — it does not mean the science is
wrong, it means the environment cannot check the science.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import envcheck


class TestNumericalStackMatchesRequirements(unittest.TestCase):

    def test_load_bearing_packages_are_inside_their_pins(self):
        bad = envcheck.violations()
        self.assertEqual(
            bad, [],
            "\n\n" + envcheck.pin_advice()
            + "\n\nThis is an ENVIRONMENT failure, not a result failure. The "
              "frozen artifacts were produced inside requirements.txt; "
              "reproduce inside those pins before concluding anything about "
              "the artifacts themselves.\n")

    def test_every_load_bearing_package_is_actually_pinned(self):
        """A package listed as load-bearing but left unpinned is a fence that
        does nothing — the failure mode this whole file exists to close."""
        pins = envcheck._parse_requirements()
        for name in envcheck.NUMERICALLY_LOAD_BEARING:
            self.assertIn(name, pins, f"{name} is not in requirements.txt")
            self.assertTrue(pins[name].strip(),
                            f"{name} appears in requirements.txt with no version "
                            f"constraint, so pinning it is not enforced")


class TestPinAdviceIsActionable(unittest.TestCase):

    def test_advice_is_empty_when_the_environment_is_clean(self):
        if envcheck.violations():
            self.skipTest("environment is already outside the pins")
        self.assertEqual(envcheck.pin_advice(), "")

    def test_version_comparison_handles_the_case_that_bit_us(self):
        # scikit-learn 1.8.0 against ">=1.7,<1.8" must be a violation.
        self.assertFalse(envcheck._satisfies("1.8.0", ">=1.7,<1.8"))
        self.assertTrue(envcheck._satisfies("1.7.2", ">=1.7,<1.8"))
        self.assertFalse(envcheck._satisfies("1.6.9", ">=1.7,<1.8"))

    def test_comparison_is_not_string_lexicographic(self):
        """'1.10' > '1.9' numerically but not as a string."""
        self.assertTrue(envcheck._satisfies("1.10.0", ">=1.9"))


if __name__ == "__main__":
    unittest.main()
