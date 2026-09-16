"""Prospective OriginalCUSIP rules plus the full adapted identity regressions."""

import copy
from unittest.mock import patch

from src.treasury_history_confirmed_identity import reconcile_confirmed_history_identity
from tests import test_treasury_history_identity as adapted_tests
from tests.test_treasury_realized_identity import ACTUAL, ANNOUNCED, fixture


def apply(f):
    return reconcile_confirmed_history_identity(*f["args"], **f["kwargs"])


def original_fixture(value=ANNOUNCED):
    f = fixture(True)
    f["args"][2] = f["args"][2].replace(
        b"<OriginalCUSIP></OriginalCUSIP>", f"<OriginalCUSIP>{value}</OriginalCUSIP>".encode()
    )
    return f


class ConfirmedHistoryIdentityTests(adapted_tests.HistoryIdentityTests):
    def setUp(self):
        self.enterContext(
            patch(
                "tests.test_treasury_history_identity.reconcile_history_identity",
                reconcile_confirmed_history_identity,
            )
        )

    def test_original_cusip_retains_announced_security_on_confirmed_substitution(self):
        f = original_fixture()
        before = copy.deepcopy(f)
        out = apply(f)
        self.assertEqual(out["original_cusip_lexeme"], ANNOUNCED)
        self.assertEqual(
            out["original_cusip_interpretation"],
            "ANNOUNCED_SECURITY_FOR_CONFIRMED_SUBSTITUTION",
        )
        self.assertEqual(out["actual_cusip"], ACTUAL)
        self.assertEqual(f, before)

    def test_original_cusip_cannot_be_unknown_or_final_alias(self):
        for value in ("ZZ1234567", ACTUAL, " " + ANNOUNCED, ""):
            f = original_fixture(value)
            if value:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    apply(f)
            else:
                self.assertEqual(apply(f)["original_cusip_lexeme"], "")

    def test_original_cusip_never_substitutes_for_final_confirmation(self):
        f = original_fixture()
        f["kwargs"]["confirmation"] = None
        with self.assertRaises(ValueError):
            apply(f)
        f = original_fixture()
        f["kwargs"]["confirmation"]["series"] = "WRONG"
        with self.assertRaises(ValueError):
            apply(f)

    def test_original_cusip_does_not_override_missing_announced_cusip(self):
        f = original_fixture()
        f["args"][2] = f["args"][2].replace(b"<AnnouncedCUSIP>AB1234567", b"<AnnouncedCUSIP>")
        with self.assertRaises(ValueError):
            apply(f)

    def test_unchanged_identity_cannot_claim_original_substitution(self):
        f = fixture()
        f["args"][2] = f["args"][2].replace(
            b"<OriginalCUSIP></OriginalCUSIP>",
            f"<OriginalCUSIP>{ANNOUNCED}</OriginalCUSIP>".encode(),
        )
        with self.assertRaises(ValueError):
            apply(f)
