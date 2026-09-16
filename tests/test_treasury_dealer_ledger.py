"""Prewritten generated source-ledger assembly contracts; no historical values."""

import copy
import unittest
from decimal import localcontext

from src.treasury_dealer_ledger import build_ledger

PAIRED = "PAIRED_METADATA_RECONCILED"
COMPLETE = "PAIRED_SOURCE_AMOUNTS_AND_OFFERING_MATCH"
MISSING = "NOT_APPLIED_PRIOR_IDENTITY_DISPOSITION_RETAINED"
HASH = "a" * 64
FIELDS = {
    "PrimaryDealerAccepted": "20.00",
    "DirectBidderAccepted": "30",
    "IndirectBidderAccepted": "50",
    "CompetitiveAccepted": "100.0",
    "CompetitiveTendered": "200",
    "PrimaryDealerTendered": "40",
    "DirectBidderTendered": "60",
    "IndirectBidderTendered": "100",
    "NonCompetitiveAccepted": "1",
    "FIMAAccepted": "0",
    "SOMAAccepted": "2",
    "SOMATendered": "2",
    "TotalAccepted": "103",
    "TotalTendered": "203",
    "BidToCoverRatio": "1.97",
    "HighYield": "3.125",
}
VALUE_FIELDS = (
    "primary_dealer_accepted",
    "competitive_accepted",
    "offering_amount_usd",
    "high_yield",
    "bid_to_cover",
)


def event_id(row):
    return row["auction_date"] + "_" + row["cusip"]


def identity(day="2021-04-21", cusip="AB1234567"):
    return {
        "auction_date": day,
        "announcement_date": "2021-04-15",
        "cusip": cusip,
        "status": PAIRED,
        "known_notice_dispositions": [],
        "unavailable_notice_urls": [],
        "identity": {
            "status": "RECONCILED_METADATA_ONLY",
            "auction_date": day,
            "actual_cusip": cusip,
            "announced_cusip": cusip,
            "identity_available_date": day,
            "result_release_date": day,
            "final_lineage": {"original_tenor_years": 5, "reopening": False},
            "input_hashes": {"result_xml_sha256": HASH},
            "amendment": None,
            "confirmation": None,
        },
        "amendment_bound_to_review": False,
        "confirmation_bound_to_review": False,
    }


def offering():
    return {
        "status": "VERIFIED_STATED_OFFERING_AGREEMENT",
        "unit": "USD",
        "offering_amount_usd": 100,
        "sources": {
            role: {"offering_amount_usd": 100, "source_unit": unit, "source_token": token}
            for role, unit, token in (
                ("announcement_pdf", "USD", "$100"),
                ("announcement_xml", "USD_BILLIONS", "0.0000001"),
                ("result_xml", "USD_BILLIONS", "0.0000001"),
            )
        },
    }


def original(row):
    account = {k: row[k] for k in ("auction_date", "announcement_date", "cusip")}
    account.update(
        status=COMPLETE,
        prior_identity_status=row["status"],
        known_notice_dispositions=copy.deepcopy(row["known_notice_dispositions"]),
        unavailable_notice_urls=copy.deepcopy(row["unavailable_notice_urls"]),
        stage_statuses=dict.fromkeys(("result_xml", "result_pdf", "offering"), "PARSED"),
        mismatched_fields=[],
        paired_field_comparisons=16,
        errors={},
    )
    payload = {
        "event": {
            k: account[k]
            for k in ("auction_date", "announcement_date", "cusip", "prior_identity_status")
        },
        "identity_source_hashes": copy.deepcopy(row["identity"]["input_hashes"]),
        "own_document_identities": {},
        "stages": {
            "result_xml": {"status": "PARSED", "result": copy.deepcopy(FIELDS)},
            "result_pdf": {"status": "PARSED", "result": copy.deepcopy(FIELDS)},
            "offering": {"status": "PARSED", "result": offering()},
        },
        "comparison": {"status": COMPLETE, "mismatched_fields": []},
    }
    return account, payload


def known_inputs():
    row = identity()
    account, payload = original(row)
    return [row], [account], {event_id(row): payload}, {}, {}


def unresolved(day="2021-04-22", cusip="CD2345678"):
    row = identity(day, cusip)
    row.pop("identity")
    row["status"] = "IDENTITY_REQUIRES_REVIEW"
    account = {k: row[k] for k in ("auction_date", "announcement_date", "cusip")}
    account.update(
        status=MISSING,
        prior_identity_status=row["status"],
        known_notice_dispositions=[],
        unavailable_notice_urls=[],
    )
    return row, account


def recovered(row):
    return {
        "auction_date": row["auction_date"],
        "cusip": row["cusip"],
        "original_status": row["status"],
        "status": COMPLETE,
        "identity": identity(row["auction_date"], row["cusip"])["identity"],
        "result_xml": copy.deepcopy(FIELDS),
        "result_pdf": copy.deepcopy(FIELDS),
        "offering": offering(),
        "known_notice_closures": [],
        "archived_result": {"body_sha256": HASH},
        "recovered_notice_sha256": None,
    }


def clock(*, exact=None, upper=None):
    return {
        "availability_after_date": exact,
        "clock_upper_bound_date": upper,
        "evidence_class": "INVENTED_DOCUMENTARY_CLOCK",
        "source_evidence": {"body_sha256": HASH},
    }


class TreasuryDealerLedgerTests(unittest.TestCase):
    def test_exact_known_mapping_and_schema_without_fraction(self):
        args = known_inputs()
        with localcontext() as ctx:
            ctx.prec = 2
            out = build_ledger(*args)
        self.assertEqual(set(out), {"events", "evidence"})
        self.assertEqual(
            out["events"],
            [
                {
                    "event_id": "2021-04-21_AB1234567",
                    "auction_date": "2021-04-21",
                    "availability_after_date": "2021-04-21",
                    "clock_upper_bound_date": None,
                    "status": "KNOWN",
                    "original_tenor_years": 5,
                    "reopening": False,
                    "primary_dealer_accepted": 20,
                    "competitive_accepted": 100,
                    "offering_amount_usd": 100,
                    "high_yield": "3.125",
                    "bid_to_cover": "1.97",
                }
            ],
        )
        self.assertEqual(out["evidence"][0]["original_identity_status"], PAIRED)
        self.assertEqual(out["evidence"][0]["original_accounting_status"], COMPLETE)
        self.assertFalse(out["evidence"][0]["recovered"])
        # The frozen accepted plain source token 0.0000000 serializes through
        # str(Decimal(...)) as 0E-7; it remains exactly zero, never missing.
        args = known_inputs()
        for role in ("result_xml", "result_pdf"):
            values = next(iter(args[2].values()))["stages"][role]["result"]
            values["FIMAAccepted"] = "0E-7"
            values["HighYield"] = "0E-7"
        zero = build_ledger(*args)["events"][0]
        self.assertEqual(zero["high_yield"], "0.0000000")
        self.assertEqual(zero["primary_dealer_accepted"], 20)

    def test_additive_recovery_preserves_original_disposition(self):
        args = list(known_inputs())
        row, account = unresolved()
        args[0].append(row)
        args[1].append(account)
        args[3][event_id(row)] = recovered(row)
        out = build_ledger(*args)
        self.assertEqual([r["status"] for r in out["events"]], ["KNOWN", "KNOWN"])
        evidence = out["evidence"][1]
        self.assertTrue(evidence["recovered"])
        self.assertEqual(evidence["original_identity_status"], "IDENTITY_REQUIRES_REVIEW")
        self.assertEqual(evidence["original_accounting_status"], MISSING)
        for key, value in (
            ("original_status", PAIRED),
            ("cusip", "ZZ9999999"),
            ("status", "FAILED"),
        ):
            bad = copy.deepcopy(args)
            bad[3][event_id(row)][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_ledger(*bad)

    def test_exact_order_membership_no_ignored_extra_or_duplicate(self):
        for defect in (
            "extra_identity",
            "extra_account",
            "extra_payload",
            "missing_payload",
            "extra_recovery",
            "duplicate",
            "wrong_key",
            "wrong_order",
        ):
            args = list(known_inputs())
            row, account = unresolved()
            if defect == "extra_identity":
                args[0].append(row)
            elif defect == "extra_account":
                args[1].append(account)
            elif defect == "extra_payload":
                args[2][event_id(row)] = copy.deepcopy(next(iter(args[2].values())))
            elif defect == "missing_payload":
                args[2].clear()
            elif defect == "extra_recovery":
                args[3][event_id(row)] = recovered(row)
            elif defect == "duplicate":
                args[0].append(copy.deepcopy(args[0][0]))
                args[1].append(copy.deepcopy(args[1][0]))
            elif defect == "wrong_key":
                args[1][0]["cusip"] = "ZZ9999999"
            else:
                args[0].insert(0, row)
                args[1].insert(0, account)
                args[4][event_id(row)] = clock()
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                build_ledger(*args)

    def test_recovery_cannot_override_old_known_or_excluded(self):
        args = list(known_inputs())
        args[3][event_id(args[0][0])] = recovered(args[0][0])
        with self.assertRaises(ValueError):
            build_ledger(*args)
        row, account = unresolved()
        row["status"] = "EXCLUDED_DOCUMENTED_CONTINGENCY_TEST"
        account["prior_identity_status"] = row["status"]
        notice = {
            "url": "https://example.test/notice.pdf",
            "disposition": "EXCLUDE_JUNE21_TEST_FROM_REGULAR_AUCTION_UNIVERSE",
        }
        row["known_notice_dispositions"] = [notice]
        account["known_notice_dispositions"] = [notice]
        with self.assertRaises(ValueError):
            build_ledger([row], [account], {}, {event_id(row): recovered(row)}, {})

    def test_all_sixteen_fields_and_offering_sources_checked(self):
        for defect in (
            "mismatch",
            "missing",
            "extra",
            "nonintegral",
            "nonfinite",
            "wrong_offering",
            "fractional_offering",
            "bad_stage",
            "bad_unit",
            "wrong_identity",
        ):
            args = list(known_inputs())
            payload = next(iter(args[2].values()))
            x = payload["stages"]["result_xml"]["result"]
            if defect == "mismatch":
                x["TotalTendered"] = "204"
            elif defect == "missing":
                x.pop("SOMAAccepted")
            elif defect == "extra":
                x["Alias"] = "3"
            elif defect in ("nonintegral", "nonfinite"):
                for role in ("result_xml", "result_pdf"):
                    payload["stages"][role]["result"]["PrimaryDealerAccepted"] = (
                        "20.5" if defect == "nonintegral" else "NaN"
                    )
            elif defect == "wrong_offering":
                payload["stages"]["offering"]["result"]["sources"]["result_xml"][
                    "offering_amount_usd"
                ] = 101
            elif defect == "fractional_offering":
                payload["stages"]["offering"]["result"]["offering_amount_usd"] = 100.5
            elif defect == "bad_stage":
                payload["stages"]["result_pdf"]["status"] = "REQUIRES_REVIEW"
            elif defect == "bad_unit":
                payload["stages"]["offering"]["result"]["unit"] = "USD_BILLIONS"
            else:
                payload["event"]["cusip"] = "ZZ9999999"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                build_ledger(*args)

    def test_zero_dealer_is_known_but_unknown_is_never_zeroed(self):
        args = list(known_inputs())
        payload = next(iter(args[2].values()))
        for role in ("result_xml", "result_pdf"):
            payload["stages"][role]["result"]["PrimaryDealerAccepted"] = "0"
        row, account = unresolved()
        args[0].append(row)
        args[1].append(account)
        args[4][event_id(row)] = clock()
        out = build_ledger(*args)
        self.assertEqual(out["events"][0]["primary_dealer_accepted"], 0)
        unknown = out["events"][1]
        self.assertEqual(unknown["status"], "UNKNOWN")
        self.assertTrue(all(unknown[k] is None for k in VALUE_FIELDS))
        self.assertIsNone(unknown["original_tenor_years"])
        self.assertIsNone(unknown["reopening"])

    def test_two_unknown_clocks_preserve_exact_and_upper_bound_distinction(self):
        row1, account1 = unresolved()
        row2, account2 = unresolved("2021-04-23", "EF3456789")
        clocks = {
            event_id(row1): clock(exact="2021-04-26"),
            event_id(row2): clock(upper="2021-04-28"),
        }
        out = build_ledger([row1, row2], [account1, account2], {}, {}, clocks)
        self.assertEqual(out["events"][0]["availability_after_date"], "2021-04-26")
        self.assertIsNone(out["events"][0]["clock_upper_bound_date"])
        self.assertIsNone(out["events"][1]["availability_after_date"])
        self.assertEqual(out["events"][1]["clock_upper_bound_date"], "2021-04-28")
        self.assertEqual(out["evidence"][1]["clock_review"], clocks[event_id(row2)])

    def test_clock_scope_dates_and_evidence_never_default(self):
        row, account = unresolved()
        for review in (
            None,
            {},
            clock(exact="2021-04-20"),
            clock(upper="2025-10-21"),
            clock(exact="2021-04-27", upper="2021-04-26"),
            clock() | {"evidence_class": ""},
            clock() | {"source_evidence": {}},
        ):
            clocks = {} if review is None else {event_id(row): review}
            with self.subTest(review=review), self.assertRaises(ValueError):
                build_ledger([row], [account], {}, {}, clocks)
        args = list(known_inputs())
        args[4][event_id(args[0][0])] = clock()
        with self.assertRaises(ValueError):
            build_ledger(*args)
        for mutation in ({"auction_date": "2009-12-31"}, {"cusip": "lowercase"}):
            r = row | mutation
            a = account | mutation
            with self.assertRaises(ValueError):
                build_ledger([r], [a], {}, {}, {event_id(r): clock()})

    def test_missing_notice_requires_exact_additive_closure_and_exclusion_evidence(self):
        row = identity()
        notice = {
            "url": "https://example.test/notice.pdf",
            "disposition": "UNKNOWN_NOTICE_CONTENT_AND_RELEASE_CLOCK",
        }
        row["known_notice_dispositions"] = [notice]
        row["unavailable_notice_urls"] = [notice["url"]]
        account, _ = original(row)
        account["status"] = "NOT_APPLIED_UNAVAILABLE_NOTICE_RETAINED"
        fix = recovered(row)
        closure = {
            "url": notice["url"],
            "prior_disposition": notice["disposition"],
            "status": "CONDITIONAL_ALTERNATIVE_NOT_REALIZED_FINAL_NEW_ISSUE_CONFIRMED",
            "release_date": "2021-04-15",
            "final_cusip": row["cusip"],
            "final_original_tenor": 5,
            "final_reopening": False,
            "body_sha256": HASH,
            "prior_review_sha256": HASH,
        }
        fix["known_notice_closures"] = [closure]
        self.assertEqual(
            build_ledger([row], [account], {}, {event_id(row): fix}, {})["events"][0][
                "status"
            ],
            "KNOWN",
        )
        for key, value in (
            ("status", "UNRESOLVED"),
            ("release_date", "2025-10-21"),
            ("final_cusip", "ZZ9999999"),
            ("body_sha256", ""),
        ):
            bad = copy.deepcopy(fix)
            bad["known_notice_closures"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_ledger([row], [account], {}, {event_id(row): bad}, {})
        bad = copy.deepcopy(fix)
        bad["known_notice_closures"] = []
        with self.assertRaises(ValueError):
            build_ledger([row], [account], {}, {event_id(row): bad}, {})
        row, account = unresolved()
        row["status"] = "EXCLUDED_DOCUMENTED_CONTINGENCY_TEST"
        account["prior_identity_status"] = row["status"]
        with self.assertRaises(ValueError):
            build_ledger([row], [account], {}, {}, {})
        notice = {
            "url": "https://example.test/test.pdf",
            "disposition": "EXCLUDE_JUNE21_TEST_FROM_REGULAR_AUCTION_UNIVERSE",
        }
        row["known_notice_dispositions"] = [notice]
        account["known_notice_dispositions"] = [notice]
        self.assertEqual(
            build_ledger([row], [account], {}, {}, {})["events"][0]["status"], "EXCLUDED"
        )

    def test_source_inputs_and_returned_provenance_are_independent_copies(self):
        args = list(known_inputs())
        before = copy.deepcopy(args)
        out = build_ledger(*args)
        self.assertEqual(args, before)
        out["evidence"][0]["identity_provenance"]["input_hashes"]["result_xml_sha256"] = (
            "changed"
        )
        self.assertEqual(args, before)


if __name__ == "__main__":
    unittest.main()
