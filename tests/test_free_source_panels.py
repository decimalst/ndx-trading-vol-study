"""Tests-first contracts for compact auxiliary free-source panels."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import free_source_panels


class CboePanelTests(unittest.TestCase):
    def test_valid_close_is_retained_but_inconsistent_ohlc_is_quarantined(self):
        text = (
            "DATE,OPEN,HIGH,LOW,CLOSE\n"
            "01/02/2024,20,19,17,18\n"
        )
        frame, audit = free_source_panels.parse_cboe_close_with_audit(text, "ohlc")
        self.assertEqual(frame["close"].tolist(), [18.0])
        self.assertEqual(audit["status"], "close_validated_ohlc_quarantined")
        self.assertEqual(audit["invalid_ohlc_rows"], 1)

    def test_valid_close_is_retained_when_auxiliary_open_is_nonpositive(self):
        text = (
            "DATE,OPEN,HIGH,LOW,CLOSE\n"
            "01/02/2024,0,19,17,18\n"
        )

        frame, audit = free_source_panels.parse_cboe_close_with_audit(text, "ohlc")

        self.assertEqual(frame["close"].tolist(), [18.0])
        self.assertEqual(audit["status"], "close_validated_ohlc_quarantined")
        self.assertEqual(audit["invalid_ohlc_rows"], 1)

    def test_close_is_available_only_on_next_declared_session(self):
        observations = {
            "VIX": pd.DataFrame(
                {"close": [20.0, 21.0]},
                index=pd.DatetimeIndex(["2024-01-05", "2024-01-08"], name="date"),
            )
        }
        sessions = pd.DatetimeIndex(
            ["2024-01-05", "2024-01-08", "2024-01-09"]
        )

        got = free_source_panels.build_cboe_close_panel(observations, sessions)

        self.assertEqual(
            got["available_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-08", "2024-01-09"],
        )
        self.assertTrue((got["observation_date"] < got["available_date"]).all())

    def test_observation_without_a_following_locked_session_is_withheld(self):
        observations = {
            "VIX": pd.DataFrame(
                {"close": [20.0, 21.0]},
                index=pd.DatetimeIndex(["2024-01-05", "2024-01-09"], name="date"),
            )
        }
        sessions = pd.DatetimeIndex(["2024-01-05", "2024-01-08"])

        got = free_source_panels.build_cboe_close_panel(observations, sessions)

        self.assertEqual(got["observation_date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-05"])

    def test_date_outside_locked_calendar_is_withheld_not_remapped(self):
        observations = {
            "VIX": pd.DataFrame(
                {"close": [20.0, 99.0]},
                index=pd.DatetimeIndex(["2024-01-05", "2024-01-06"], name="date"),
            )
        }
        sessions = pd.DatetimeIndex(["2024-01-05", "2024-01-08"])

        got = free_source_panels.build_cboe_close_panel(observations, sessions)

        self.assertEqual(len(got), 1)
        self.assertEqual(got.iloc[0]["close"], 20.0)

    def test_empty_locked_calendar_fails_closed(self):
        observations = {
            "VIX": pd.DataFrame(
                {"close": [20.0]},
                index=pd.DatetimeIndex(["2024-01-05"], name="date"),
            )
        }

        with self.assertRaisesRegex(ValueError, "locked sessions"):
            free_source_panels.build_cboe_close_panel(
                observations, pd.DatetimeIndex([])
            )


class QuarantineTests(unittest.TestCase):
    def test_hf_spx_zero_open_is_quarantined_not_imputed(self):
        text = (
            "Date,Open,High,Low,Close,Adj Close,Volume\n"
            "2024-01-02,0,101,99,100,100,1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spx.csv"
            path.write_text(text)
            audit = free_source_panels.audit_hf_spx(
                path,
                expected_bytes=path.stat().st_size,
                expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                declared_rows=1,
            )

        self.assertEqual(audit["status"], "quarantined_strict_ohlc_failure")
        self.assertEqual(audit["zero_open_rows"], 1)
        self.assertTrue(audit["pinned_bytes_match"])
        self.assertTrue(audit["pinned_sha256_match"])
        self.assertTrue(audit["declared_rows_match"])
        self.assertNotIn("imputed_open_rows", audit)


class CftcPanelTests(unittest.TestCase):
    def test_compact_panel_is_lagged_and_rejects_missing_positions(self):
        header = (
            "report_date_as_yyyy_mm_dd,cftc_contract_market_code,"
            "open_interest_all,lev_money_positions_long,lev_money_positions_short\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cftc.csv"
            path.write_text(header + "2024-01-02,209742,100,40,60\n")
            panel, audit = free_source_panels.build_cftc_209742(path)

            self.assertEqual(
                panel.columns.tolist(),
                [
                    "report_date", "contract_code", "open_interest", "lev_long",
                    "lev_short", "available_date_generic",
                ],
            )
            self.assertEqual(panel.loc[0, "available_date_generic"], pd.Timestamp("2024-01-09"))
            self.assertEqual(audit["rows"], 1)

            path.write_text(header + "2024-01-02,209742,100,,60\n")
            with self.assertRaisesRegex(ValueError, "missing|position"):
                free_source_panels.build_cftc_209742(path)


class ZenodoAuditTests(unittest.TestCase):
    def test_compact_audit_recognizes_bid_offer_and_open_interest(self):
        columns = free_source_panels.ZENODO_COLUMNS
        row = {
            "date": "2020-01-02", "exdate": "2020-01-17", "cp_flag": 1,
            "strike_price": 100.0, "best_bid": 2.0, "best_offer": 2.2,
            "volume": 10, "open_interest": 20, "impl_volatility": 0.3,
            "delta": -0.2, "gamma": 0.01, "vega": 1.0, "theta": -0.1,
            "current_price": 110.0, "time_to_maturity": 0.04,
            "historical_volatility": 0.2, "BS_price": 2.1,
            "avg_bid_offer": 2.1, "bspricediff": 0.0, "real_iv": 0.31,
            "r": 0.01, "SP/CP": 1.1, "predicted_iv": 0.29,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tsla.csv"
            pd.DataFrame([row], columns=columns).to_csv(path, index=False)
            expected_md5 = hashlib.md5(path.read_bytes()).hexdigest()
            audit = free_source_panels.audit_zenodo_tsla(
                path, expected_md5=expected_md5, chunksize=1
            )

        self.assertEqual(audit["status"], "validated_private_research_only")
        self.assertEqual(audit["rows"], 1)
        self.assertTrue(audit["has_bid_offer"])
        self.assertTrue(audit["has_open_interest"])
        self.assertTrue(audit["md5_pin_verified"])
        self.assertEqual(audit["schema_columns"], columns)

    def test_duplicate_option_keys_fail_closed(self):
        columns = free_source_panels.ZENODO_COLUMNS
        base = {column: 1 for column in columns}
        base.update({"date": "2020-01-02", "exdate": "2020-01-17", "best_bid": 1.0, "best_offer": 2.0})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tsla.csv"
            pd.DataFrame([base, base], columns=columns).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                free_source_panels.audit_zenodo_tsla(path, expected_md5=None, chunksize=1)

    def test_hash_collision_between_distinct_keys_is_not_a_false_duplicate(self):
        columns = free_source_panels.ZENODO_COLUMNS
        base = {column: 1 for column in columns}
        base.update({
            "date": "2020-01-02", "exdate": "2020-01-17",
            "best_bid": 1.0, "best_offer": 2.0,
        })
        other = dict(base, strike_price=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tsla.csv"
            pd.DataFrame([base, other], columns=columns).to_csv(path, index=False)
            collision = pd.Series([7], dtype="uint64")
            with mock.patch.object(pd.util, "hash_pandas_object", return_value=collision):
                audit = free_source_panels.audit_zenodo_tsla(
                    path, expected_md5=None, chunksize=1
                )

        self.assertEqual(audit["rows"], 2)


if __name__ == "__main__":
    unittest.main()
