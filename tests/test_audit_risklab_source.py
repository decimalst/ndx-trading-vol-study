"""Prewritten contracts for source measurement inspection, without forecast fits."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit_risklab_source import audit_table, parse_equity_response, read_calendar


def response(rows, symbol="SPY", identifier="84398", count=None):
    return "\n".join(
        [symbol, identifier, "Synthetic ETF", str(len(rows) if count is None else count),
         rows[0].split()[1], rows[-1].split()[1], *rows]
    ).encode()


def row(date, values="0.2 1 0.01 0.3 0.4 0.5 2 0.02 0.6 0.7"):
    return f"84398 {date} {values}"


class TestRiskLabAudit(unittest.TestCase):
    def test_parquet_calendar_uses_timestamp_cutoff_and_only_returns_labels(self):
        frame = pd.DataFrame({"price": [1., 999.]}, index=pd.DatetimeIndex(
            ["2025-10-20", "2025-11-03"], name="date"
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calendar.parquet"
            frame.to_parquet(path)
            calendar = read_calendar(path)
        self.assertEqual(calendar.tolist(), [pd.Timestamp("2025-10-20")])

    def test_provider_column_mapping_preserves_volatility_units(self):
        table, metadata = parse_equity_response(response([row("20200102")]))
        self.assertEqual(metadata["source_rows"], 1)
        self.assertEqual(table.index[0], pd.Timestamp("2020-01-02"))
        self.assertEqual(table.iloc[0]["qmle_trade"], 0.2)
        self.assertEqual(table.iloc[0]["rv5_trade"], 0.3)
        self.assertEqual(table.iloc[0]["rv15_trade"], 0.4)
        self.assertEqual(table.iloc[0]["qmle_quote"], 0.5)
        self.assertEqual(table.iloc[0]["rv5_quote"], 0.6)
        self.assertEqual(table.iloc[0]["rv15_quote"], 0.7)

    def test_zero_negative_and_large_values_are_not_chart_filtered(self):
        table, _ = parse_equity_response(response([row("20200102", "0 1 0 -1 4 5 2 0 6 7")]))
        self.assertEqual(table.iloc[0]["qmle_trade"], 0)
        self.assertEqual(table.iloc[0]["rv5_trade"], -1)
        self.assertEqual(table.iloc[0]["rv15_trade"], 4)
        result = audit_table(table, pd.DatetimeIndex(["2020-01-02"]), "2020-01-02", "2020-01-02")
        self.assertEqual(result["fields"]["qmle_trade"]["zero"], 1)
        self.assertEqual(result["fields"]["rv5_trade"]["negative"], 1)
        self.assertEqual(result["fields"]["rv15_trade"]["above_chart_cap"], 1)

    def test_nonfinite_values_remain_auditable(self):
        table, _ = parse_equity_response(response([row("20200102", "nan 1 0 inf 1 -inf 2 0 1 1")]))
        self.assertTrue(np.isnan(table.iloc[0]["qmle_trade"]))
        result = audit_table(table, pd.DatetimeIndex(["2020-01-02"]), "2020-01-02", "2020-01-02")
        self.assertEqual(result["fields"]["qmle_trade"]["nan"], 1)
        self.assertEqual(result["fields"]["rv5_trade"]["infinite"], 1)

    def test_source_gaps_and_invalid_estimates_are_distinct(self):
        table, _ = parse_equity_response(response([row("20200102"), row("20200106", "nan 1 0 1 1 1 2 0 1 1")]))
        result = audit_table(table, pd.bdate_range("2020-01-02", "2020-01-06"), "2020-01-02", "2020-01-06")
        self.assertEqual(result["missing_source_dates"], ["2020-01-03"])
        self.assertEqual(result["invalid_primary_estimate_dates"], ["2020-01-06"])
        self.assertEqual(result["primary_complete_5_sessions"], 0)

    def test_full_calendar_windows_never_roll_across_missing_session(self):
        dates = pd.bdate_range("2020-01-01", periods=8)
        rows = [row(date.strftime("%Y%m%d")) for date in dates if date != dates[2]]
        table, _ = parse_equity_response(response(rows))
        result = audit_table(table, dates, str(dates[0].date()), str(dates[-1].date()))
        self.assertEqual(result["primary_complete_5_sessions"], 1)

    def test_post_cutoff_estimates_are_never_parsed(self):
        first = row("20251020")
        table, meta = parse_equity_response(response([first, row("20251103", "DO NOT PARSE THESE FUTURE NUMERIC FIELDS AT ALL NOW")]))
        self.assertEqual(len(table), 1)
        self.assertEqual(meta["after_cutoff_rows"], 1)
        self.assertEqual(table.index.max(), pd.Timestamp("2025-10-20"))

    def test_duplicate_dates_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_equity_response(response([row("20200102"), row("20200102")]))

    def test_wrong_identity_and_declared_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            parse_equity_response(response([row("20200102")], symbol="QQQ"))
        with self.assertRaisesRegex(ValueError, "count"):
            parse_equity_response(response([row("20200102")], count=2))

    def test_malformed_historical_rows_and_dates_rejected(self):
        with self.assertRaisesRegex(ValueError, "fields"):
            parse_equity_response(response(["84398 20200102 1 2 3"]))
        with self.assertRaises(ValueError):
            parse_equity_response(response([row("20200231")]))
        with self.assertRaises(ValueError):
            parse_equity_response(response([row("20200102", "oops 1 0 1 1 1 2 0 1 1")]))

    def test_unknown_calendar_membership_is_visible(self):
        table, _ = parse_equity_response(response([row("20200104")]))
        result = audit_table(table, pd.DatetimeIndex(["2020-01-03"]), "2020-01-03", "2020-01-04")
        self.assertEqual(result["outside_reference_calendar"], ["2020-01-04"])


if __name__ == "__main__":
    unittest.main()
