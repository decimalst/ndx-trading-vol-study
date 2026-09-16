"""Prewritten regressions for ambiguous ALFRED README attribute spans."""

import unittest
from unittest.mock import patch

from src import claims_release_source as source
from tests.test_claims_release_source import (
    archive_fixture,
    digest,
    readme_fixture,
    request_fixture,
)


class ClaimsReleaseAttributeSpanTests(unittest.TestCase):
    def test_duplicate_or_contradictory_dated_attribute_rows_fail_before_values(self):
        fields = [
            (
                "Units",
                "Number                                                                 2009-05-28       Current",
                "Percent Change                                                         2018-01-01       Current",
            ),
            (
                "Frequency",
                "Weekly, Ending Saturday                                                2009-05-28       Current",
                "Weekly, Ending Friday                                                  2018-01-01       Current",
            ),
            (
                "Seasonal Adjustment",
                "Seasonally Adjusted                                                    2009-05-28       Current",
                "Not Seasonally Adjusted                                                2018-01-01       Current",
            ),
        ]
        request = request_fixture()
        original = readme_fixture(request["vintage_dates"])
        for label, valid, contradictory in fields:
            for case, extra in (("duplicate", valid), ("contradictory", contradictory)):
                with self.subTest(field=label, case=case):
                    section = f"{label}\n{valid}\n".encode()
                    changed = original.replace(section, section + extra.encode() + b"\n")
                    self.assertNotEqual(changed, original)
                    payload, _, _ = archive_fixture(request=request, readme=changed)
                    with patch.object(
                        source, "_parse_value", wraps=source._parse_value
                    ) as convert:
                        with self.assertRaises(ValueError):
                            source.parse_export(payload, request, digest(payload))
                        convert.assert_not_called()

    def test_documented_notes_continuation_remains_valid(self):
        request = request_fixture()
        original = readme_fixture(request["vintage_dates"])
        note = b"Generated fixture; no historical observations.                         2009-05-28       Current\n"
        changed = original.replace(note, note + b"A continuation of the documentary note.\n")
        self.assertNotEqual(changed, original)
        payload, _, _ = archive_fixture(request=request, readme=changed)
        result = source.parse_export(payload, request, digest(payload))
        self.assertEqual(result["status"], "STRUCTURALLY_VERIFIED_NOT_RELEASE_RECONCILED")
        self.assertEqual(len(result["records"]), 3)


if __name__ == "__main__":
    unittest.main()
