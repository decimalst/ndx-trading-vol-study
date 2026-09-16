"""Measurement-only contracts written before the HARNet numeric audit implementation."""

import hashlib
import io
import unittest

import numpy as np
import pandas as pd

from src.audit_harnet_extension import (
    FIELDS,
    SYMBOLS,
    audit_frames,
    boundary_summary,
    compare_field,
    parse_csv,
    quality_summary,
    verify_hash,
)


def csv_fixture(dates=("2018-06-26", "2018-06-27"), symbols=SYMBOLS):
    rows = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            row = dict.fromkeys(FIELDS, "0.0001")
            row.update(date=date + " 00:00:00+01:00", Symbol=symbol,
                       rv5=str((i + 1) / 1000), rsv="0.0001", nobs="250",
                       open_price="100", close_price="101")
            rows.append(row)
    return pd.DataFrame(rows, columns=["date", "Symbol", *FIELDS]).to_csv(index=False).encode()


class SourceIdentityContracts(unittest.TestCase):
    def test_hash_checked_without_mutation(self):
        raw = csv_fixture()
        verify_hash(raw, hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            verify_hash(raw, "0" * 64)

    def test_local_date_label_is_preserved_and_old_unnamed_date_accepted(self):
        raw = csv_fixture()
        new = parse_csv(raw)
        old = parse_csv(raw.replace(b"date,", b",", 1))
        self.assertEqual(new.tokens.index.tolist(), old.tokens.index.tolist())
        self.assertEqual(new.tokens.index[0], (SYMBOLS[0], "2018-06-26"))

    def test_duplicate_keys_are_rejected_even_if_values_match(self):
        frame = pd.read_csv(io.BytesIO(csv_fixture()), dtype=str)
        raw = pd.concat([frame, frame.iloc[:1]]).to_csv(index=False).encode()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_csv(raw)

    def test_registered_market_and_shared_field_cannot_disappear(self):
        with self.assertRaisesRegex(ValueError, "market"):
            parse_csv(csv_fixture(symbols=SYMBOLS[:-1]))
        frame = pd.read_csv(io.BytesIO(csv_fixture())).drop(columns="rv10")
        with self.assertRaisesRegex(ValueError, "schema"):
            parse_csv(frame.to_csv(index=False).encode())

    def test_invalid_date_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_csv(csv_fixture().replace(b"2018-06-26", b"2018-02-30"))


class NumericComparisonContracts(unittest.TestCase):
    def test_decimal_equivalence_is_separate_from_text_equality(self):
        result = compare_field(["0.001", "1.0", ""], ["1e-3", "1", "NaN"], "rv5")
        self.assertEqual(result["decimal_equal"], 2)
        self.assertEqual(result["literal_equal"], 0)
        self.assertEqual(result["both_missing"], 1)
        self.assertEqual(result["status"], "EXACT_NUMERIC_WITH_MATCHED_MISSINGNESS")

    def test_rounding_equivalence_does_not_claim_decimal_identity(self):
        result = compare_field(["0.001"], ["0.0010000000001"], "rv5")
        self.assertEqual(result["decimal_equal"], 0)
        self.assertEqual(result["within_tolerance"], 1)
        self.assertEqual(result["status"], "TOLERANCE_EQUIVALENT")

    def test_nobs_is_exact_even_when_float_tolerance_would_accept(self):
        result = compare_field(["250"], ["250.00000000001"], "nobs")
        self.assertEqual(result["material_difference"], 1)

    def test_missingness_changes_and_malformed_values_fail(self):
        result = compare_field(["", "1", "bad", "inf"], ["1", "NaN", "bad", "inf"], "rv5")
        self.assertEqual(result["old_missing_only"], 1)
        self.assertEqual(result["new_missing_only"], 1)
        self.assertEqual(result["old_invalid_or_nonfinite"], 2)
        self.assertEqual(result["status"], "MATERIAL_REVISION_OR_INVALID")

    def test_ratio_diagnostic_identifies_scale_without_repair(self):
        old = [str(x / 1000) for x in range(1, 26)]
        new = [str(x / 10) for x in range(1, 26)]
        result = compare_field(old, new, "rv5")
        self.assertEqual(result["material_difference"], 25)
        self.assertAlmostEqual(result["constant_scale_candidate"], 100)
        self.assertAlmostEqual(result["ratio_new_over_old"]["p50"], 100)

    def test_zero_denominator_never_creates_infinite_ratio(self):
        result = compare_field(["0", "0", "2"], ["0", "1", "4"], "rsv")
        self.assertEqual(result["ratio_new_over_old"]["count"], 1)
        self.assertEqual(result["ratio_new_over_old"]["p50"], 2)

    def test_empty_comparison_is_not_an_equivalence_result(self):
        result = compare_field([], [], "rv5")
        self.assertEqual(result["paired_rows"], 0)
        self.assertEqual(result["status"], "MATERIAL_REVISION_OR_INVALID")

    def test_default_parser_equality_reported_separately(self):
        result = compare_field(["0.001"], ["0.001"], "rv5",
                               old_parsed=[.001], new_parsed=[np.nextafter(.001, 1)])
        self.assertEqual(result["decimal_equal"], 1)
        self.assertEqual(result["frozen_parser_equal"], 0)


class WholeAuditContracts(unittest.TestCase):
    def test_overlap_is_keyed_and_extensions_do_not_change_old_comparison(self):
        old = parse_csv(csv_fixture())
        new_raw = pd.read_csv(io.BytesIO(csv_fixture(dates=("2018-06-26", "2018-06-27", "2018-06-28"))))
        new = parse_csv(new_raw.iloc[::-1].to_csv(index=False).encode())
        result = audit_frames(old, new)
        self.assertEqual(result["overlap_rows"], 14)
        self.assertEqual(result["extension_rows"], 7)
        self.assertEqual(result["fields"]["rv5"]["decimal_equal"], 14)
        self.assertEqual(result["rv5_rsv_reproduction"], "EXACT_FROZEN_PARSER_INPUTS")

    def test_missing_old_date_is_not_silently_lost_in_inner_join(self):
        old = parse_csv(csv_fixture())
        new = parse_csv(csv_fixture(dates=("2018-06-27", "2018-06-28")))
        result = audit_frames(old, new)
        self.assertEqual(result["missing_old_keys"], 7)
        self.assertEqual(result["rv5_rsv_reproduction"], "DOES_NOT_REPRODUCE")

    def test_extra_old_period_date_prevents_whole_panel_reproduction(self):
        old = parse_csv(csv_fixture())
        new = parse_csv(csv_fixture(dates=("2018-06-25", "2018-06-26", "2018-06-27")))
        result = audit_frames(old, new)
        self.assertEqual(result["added_old_period_keys"], 7)
        self.assertEqual(result["rv5_rsv_reproduction"], "DOES_NOT_REPRODUCE")

    def test_audit_does_not_mutate_source_frames(self):
        old, new = parse_csv(csv_fixture()), parse_csv(csv_fixture())
        old_tokens, old_values = old.tokens.copy(deep=True), old.parsed.copy(deep=True)
        new_tokens, new_values = new.tokens.copy(deep=True), new.parsed.copy(deep=True)
        audit_frames(old, new)
        pd.testing.assert_frame_equal(old.tokens, old_tokens)
        pd.testing.assert_frame_equal(old.parsed, old_values)
        pd.testing.assert_frame_equal(new.tokens, new_tokens)
        pd.testing.assert_frame_equal(new.parsed, new_values)

    def test_boundary_uses_archived_rows_and_retains_missing_observations(self):
        old = pd.Series([1., np.nan, 2.], index=["2018-06-25", "2018-06-26", "2018-06-27"])
        new = pd.Series([np.nan, 4., 5.], index=["2018-06-28", "2018-06-29", "2018-07-02"])
        result = boundary_summary(old, new, window=2)
        self.assertEqual(result["before"]["rows"], 2)
        self.assertEqual(result["before"]["missing_or_nonfinite"], 1)
        self.assertEqual(result["after"]["rows"], 2)
        self.assertEqual(result["after"]["missing_or_nonfinite"], 1)
        self.assertIsNone(result["first_over_last"])
        self.assertEqual(result["after"]["last_date"], "2018-06-29")

    def test_quality_records_nonpositive_rv_and_semivariance_violations(self):
        frame = parse_csv(csv_fixture()).parsed.copy()
        frame.loc[frame.index[0], "rv5"] = 0
        frame.loc[frame.index[1], "rsv"] = -1
        frame.loc[frame.index[2], "nobs"] = 2.5
        result = quality_summary(frame)
        self.assertEqual(result["rv5_nonpositive"], 1)
        self.assertEqual(result["rsv_negative"], 1)
        self.assertEqual(result["rsv_above_rv5"], 1)
        self.assertEqual(result["nobs_not_positive_integer"], 1)


if __name__ == "__main__":
    unittest.main()
