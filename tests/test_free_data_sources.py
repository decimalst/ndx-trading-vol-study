"""Pre-acquisition contracts for free external research sources."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import free_data_sources as fds


class ProtocolContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = fds.load_protocol()

    def test_exact_source_identities_and_pins(self):
        fds.validate_protocol(self.protocol)
        sources = self.protocol["sources"]
        self.assertEqual(
            sources["hf_equities_5m_options"]["revision"],
            "99d9d32f99e955ca1f5b7fa4e08606da72707fb0",
        )
        self.assertEqual(sources["hf_equities_5m_options"]["repository_bytes"], 60594909400)
        self.assertEqual(sources["hf_spx"]["file_bytes"], 1743930)
        self.assertEqual(sources["hf_spx"]["rows"], 24167)
        self.assertEqual(sources["zenodo_tsla_options"]["file_bytes"], 1035799239)
        self.assertEqual(
            sources["zenodo_tsla_options"]["validated_window"],
            ["2018-08-06", "2023-08-31"],
        )
        self.assertEqual(sources["zenodo_tsla_options"]["rows"], 4584740)
        self.assertEqual(sources["kaggle_spy_options"]["version"], 2)
        self.assertEqual(sources["kaggle_qqq_options"]["archive_bytes"], 132165529)

    def test_license_and_provenance_classifications_are_fail_closed(self):
        for name, source in self.protocol["sources"].items():
            self.assertTrue(source["license_class"], name)
            self.assertTrue(source["provenance_class"], name)
            self.assertTrue(source["redistribution"], name)
            self.assertIn("raw_local_only", source["commit_policy"], name)
        self.assertEqual(
            self.protocol["sources"]["cftc_nq_tff"]["provenance_class"],
            "official_primary",
        )
        self.assertIn(
            "pending", self.protocol["sources"]["zenodo_tsla_options"]["redistribution"]
        )
        self.assertFalse(
            self.protocol["sources"]["hf_equities_5m_stockprices"]["acquisition_enabled"]
        )

    def test_raw_and_processed_roots_are_separate(self):
        storage = self.protocol["storage"]
        self.assertEqual(storage["raw_root"], "data/free_sources/raw")
        self.assertEqual(storage["processed_root"], "data/free_sources/processed")
        self.assertNotEqual(storage["raw_root"], storage["processed_root"])

    def test_plan_contains_only_pinned_or_official_commands(self):
        plan = fds.build_acquisition_plan(self.protocol)
        names = {item["source"] for item in plan}
        self.assertIn("cftc_nq_tff", names)
        self.assertIn("cboe_vxn", names)
        self.assertIn("hf_spx", names)
        self.assertIn("hf_equities_5m_options", names)
        self.assertIn("zenodo_tsla_options", names)
        self.assertIn("kaggle_nq_1m", names)
        self.assertNotIn("hf_equities_5m_stockprices", names)
        for item in plan:
            self.assertTrue(item["output"].startswith("data/free_sources/raw/"))
            self.assertNotIn("--unzip", item["command"])
        hf = next(item for item in plan if item["source"] == "hf_spx")
        self.assertIn("498783fd0df5ad1f13c98e42329c60b8f8f9c6c1", hf["command"])

    def test_every_enabled_download_has_a_disk_preflight_budget(self):
        storage = self.protocol["storage"]
        for item in fds.build_acquisition_plan(self.protocol):
            self.assertGreater(fds.required_free_bytes(item, storage), 0, item["source"])


class DiskSafetyContract(unittest.TestCase):
    def test_disk_preflight_includes_expansion_and_reserve(self):
        protocol = fds.load_protocol()
        spec = protocol["sources"]["zenodo_tsla_options"]["download"]
        required = fds.required_free_bytes(spec, protocol["storage"])
        self.assertEqual(required, 7440307598)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            with self.assertRaisesRegex(OSError, "free space"):
                fds.preflight_download(
                    raw / "tsla.csv", raw, spec, protocol["storage"], free_bytes=required - 1
                )
            got = fds.preflight_download(
                raw / "tsla.csv", raw, spec, protocol["storage"], free_bytes=required
            )
            self.assertEqual(got["required_free_bytes"], required)

    def test_path_escape_and_unmanifested_overwrite_are_rejected(self):
        protocol = fds.load_protocol()
        spec = protocol["sources"]["hf_spx"]["download"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "raw root"):
                fds.preflight_download(
                    Path(tmp) / "outside.csv", root, spec, protocol["storage"], free_bytes=10**12
                )
            existing = root / "spx.csv"
            existing.write_text("untracked")
            with self.assertRaisesRegex(FileExistsError, "manifest"):
                fds.preflight_download(
                    existing, root, spec, protocol["storage"], free_bytes=10**12
                )


class ManifestContract(unittest.TestCase):
    def test_manifest_is_hash_size_and_identity_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            path.write_bytes(b"abc")
            manifest = fds.build_raw_manifest(
                path,
                source_id="test:source",
                source_version="v1",
                retrieved_at_utc="2026-08-12T20:00:00Z",
            )
            self.assertEqual(manifest["sha256"], hashlib.sha256(b"abc").hexdigest())
            self.assertEqual(manifest["bytes"], 3)
            fds.verify_raw_manifest(path, manifest, "test:source", "v1")
            path.write_bytes(b"abcd")
            with self.assertRaisesRegex(ValueError, "hash|size"):
                fds.verify_raw_manifest(path, manifest, "test:source", "v1")

    def test_manifest_file_fails_closed_on_wrong_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            path.write_bytes(b"abc")
            manifest_path = Path(str(path) + ".manifest.json")
            manifest = fds.build_raw_manifest(path, "source:a", "r1", "2026-08-12T20:00:00Z")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source identity"):
                fds.load_and_verify_manifest(path, manifest_path, "source:b", "r1")


class AvailabilityContract(unittest.TestCase):
    def test_cftc_uses_seven_calendar_day_lag(self):
        frame = pd.DataFrame({"report_date": ["2024-01-02", "2024-01-09"]})
        got = fds.apply_cftc_availability(frame, "report_date", days=7)
        self.assertEqual(got["available_date"].dt.strftime("%Y-%m-%d").tolist(), [
            "2024-01-09", "2024-01-16"
        ])

    def test_cboe_uses_next_declared_session_not_weekday_guess(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-07-03", "2024-07-05"])})
        sessions = pd.DatetimeIndex(["2024-07-03", "2024-07-05", "2024-07-08"])
        got = fds.apply_cboe_availability(frame, "date", sessions)
        self.assertEqual(got["available_date"].dt.strftime("%Y-%m-%d").tolist(), [
            "2024-07-05", "2024-07-08"
        ])
        with self.assertRaisesRegex(ValueError, "following session"):
            fds.apply_cboe_availability(frame.iloc[1:], "date", sessions[:2])


class ParserContract(unittest.TestCase):
    def test_cboe_ohlc_and_close_only_schemas(self):
        ohlc = "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2024,10,12,9,11\n01/03/2024,11,13,10,12\n"
        got = fds.parse_cboe_csv(ohlc, "ohlc")
        self.assertEqual(got.columns.tolist(), ["open", "high", "low", "close"])
        close = fds.parse_cboe_csv("DATE,SKEW\n01/02/2024,125.5\n", "close_only")
        self.assertEqual(close.iloc[0]["close"], 125.5)
        with self.assertRaisesRegex(ValueError, "OHLC"):
            fds.parse_cboe_csv("DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2024,10,8,9,11\n", "ohlc")

    def test_hf_spx_parser_rejects_duplicate_or_bad_dates(self):
        text = (
            "Date,Open,High,Low,Close,Adj Close,Volume\n"
            "2024-01-02,10,12,9,11,11,100\n"
            "2024-01-03,11,13,10,12,12,110\n"
        )
        got = fds.parse_hf_spx_csv(text)
        self.assertTrue(got.index.is_monotonic_increasing)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fds.parse_hf_spx_csv(text + "2024-01-03,11,13,10,12,12,110\n")

    def test_hf_option_jsonl_checks_utc_and_unix_timestamp(self):
        row = {
            "option_symbol": "SPY240119C00470000", "underlying_symbol": "SPY",
            "option_type": "call", "strike_price": 470.0, "expiration_date": "2024-01-19",
            "datetime": "2024-01-18T14:30:00Z", "date": "2024-01-18",
            "unix_timestamp": 1705588200, "open": 1.0, "high": 1.2, "low": 0.9,
            "close": 1.1, "volume": 10, "trade_count": 2, "vwap": 1.05,
        }
        got = fds.parse_hf_options_jsonl(json.dumps(row) + "\n")
        self.assertEqual(str(got.loc[0, "datetime"].tz), "UTC")
        row["unix_timestamp"] += 1
        with self.assertRaisesRegex(ValueError, "unix"):
            fds.parse_hf_options_jsonl(json.dumps(row) + "\n")

    def test_optionsdx_parser_normalizes_bom_brackets_and_validates_clock(self):
        columns = fds.load_protocol()["schemas"]["optionsdx"]["columns"]
        values = [
            "1704229200", "2024-01-02 16:00", "2024-01-02", "16.0", "100",
            "2024-01-19", "1705622400", "17", ".5", ".01", ".1", "-.1", ".01",
            ".2", "10", "1", "1x2", ".9", "1.1", "100", ".8", "1.0", "1x2",
            ".9", "-.5", ".01", ".1", "-.1", "-.01", ".2", "11", "0", "0",
        ]
        header = ",".join("[" + c + "]" for c in columns)
        got = fds.parse_optionsdx_csv("\ufeff" + header + "\n" + ",".join(values) + "\n")
        self.assertEqual(got.loc[0, "QUOTE_DATE"], pd.Timestamp("2024-01-02"))
        self.assertNotIn("OPEN_INTEREST", got.columns)
        values[2] = "2024-01-03"
        with self.assertRaisesRegex(ValueError, "QUOTE_DATE"):
            fds.parse_optionsdx_csv(header + "\n" + ",".join(values) + "\n")

    def test_nq_parser_localizes_et_and_rejects_impossible_ohlc(self):
        text = (
            "timestamp ET,open,high,low,close,volume,Vwap_RTH,Vwap_ETH\n"
            "12/26/2022 18:01,100,102,99,101,5,,101\n"
        )
        got = fds.parse_nq_1m_csv(text)
        self.assertEqual(str(got.index.tz), "America/New_York")
        with self.assertRaisesRegex(ValueError, "OHLC"):
            fds.parse_nq_1m_csv(text.replace("102,99", "98,99"))

    def test_zenodo_tsla_parser_requires_disclosed_schema(self):
        columns = fds.load_protocol()["schemas"]["zenodo_tsla"]["columns"]
        row = ["2018-08-07", "2019-01-18", "1", "10", ".05", ".06", "354", "4623"]
        row += ["1.9", "-.1", ".01", ".2", "-.3", "379", ".4", ".5", ".1", ".055"]
        row += [".01", "1.96", ".045", ".02", "1.8"]
        got = fds.parse_zenodo_tsla_csv(",".join(columns) + "\n" + ",".join(row) + "\n")
        self.assertEqual(int(got.loc[0, "open_interest"]), 4623)
        with self.assertRaisesRegex(ValueError, "schema"):
            fds.parse_zenodo_tsla_csv("date,exdate\n2018-01-01,2018-02-01\n")

    def test_cftc_parser_rejects_wrong_contract_and_duplicate_date(self):
        text = (
            "Report_Date_as_YYYY_MM_DD,CFTC_Contract_Market_Code,Market_and_Exchange_Names,Open_Interest_All\n"
            "2024-01-02,20974+,NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE,100\n"
        )
        got = fds.parse_cftc_tff_csv(text, {"20974+"})
        self.assertEqual(got.loc[0, "report_date"], pd.Timestamp("2024-01-02"))
        with self.assertRaisesRegex(ValueError, "contract"):
            fds.parse_cftc_tff_csv(text, {"209742"})


if __name__ == "__main__":
    unittest.main()
