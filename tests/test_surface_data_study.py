"""Pre-written contracts for the frozen private surface-data study."""
from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.surface_data_study import (
    DEFAULT_PROTOCOL,
    OPTION_COLUMNS,
    add_full_session_lag,
    aapl_split_audit,
    build_daily_surface,
    build_forecast_design,
    hash_file,
    load_protocol,
    normalize_headers,
    normalize_option_frame,
    protocol_sha256,
    qlike,
    score_forecasts,
    validate_protocol,
    write_source_manifest,
)
from src.verify_surface_data_study import verify_artifacts


ROOT = pathlib.Path(__file__).resolve().parents[1]


def option_rows(day: str, hour: float = 16.0, underlying: float = 100.0) -> pd.DataFrame:
    """A tiny complete chain with exact 9D/30D delta matches."""
    rows = []
    for dte, expiry, niv in [(9.0, "2020-01-13", 0.22), (30.0, "2020-02-03", 0.20)]:
        for strike, cd, pd_, civ, piv in [
            (95.0, 0.50, -0.25, niv, niv + 0.04),
            (100.0, 0.50, -0.50, niv, niv + 0.02),
            (105.0, 0.25, -0.50, niv - 0.01, niv + 0.02),
        ]:
            rows.append({
                "[QUOTE_UNIXTIME]": 1,
                " [QUOTE_READTIME]": f"{day} 16:00",
                " [QUOTE_DATE]": day,
                " [QUOTE_TIME_HOURS]": hour,
                " [UNDERLYING_LAST]": underlying,
                " [EXPIRE_DATE]": expiry,
                " [EXPIRE_UNIX]": 2,
                " [DTE]": dte,
                " [C_DELTA]": cd,
                " [C_GAMMA]": 0.02,
                " [C_IV]": civ,
                " [C_VOLUME]": 10.0,
                " [C_BID]": 1.0,
                " [C_ASK]": 1.2,
                " [STRIKE]": strike,
                " [P_BID]": 1.1,
                " [P_ASK]": 1.3,
                " [P_DELTA]": pd_,
                " [P_GAMMA]": 0.03,
                " [P_IV]": piv,
                " [P_VOLUME]": 20.0,
            })
    return pd.DataFrame(rows)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_protocol()

    def test_protocol_was_frozen_before_results_and_is_private(self):
        self.assertEqual(self.spec["status"], "frozen_before_first_empirical_run")
        self.assertEqual(self.spec["evidence_class"], "private_diagnostic_only")
        self.assertTrue(self.spec["no_tuning_after_results"])
        self.assertEqual(len(protocol_sha256()), 64)
        validate_protocol(self.spec)

    def test_actual_qqq_coverage_overrides_title(self):
        self.assertEqual(self.spec["sources"]["qqq"]["validated_coverage"],
                         ["2021-01-04", "2022-12-30"])
        self.assertEqual(self.spec["sources"]["qqq"]["expected_rows"], 1775749)

    def test_no_open_interest_or_gex_claim_is_possible(self):
        shape = self.spec["shape_construction"]
        self.assertFalse(shape["open_interest_available"])
        self.assertTrue(shape["gamma_weighted_volume_is_not_dealer_gex"])
        text = DEFAULT_PROTOCOL.read_text().lower()
        self.assertNotIn("dealer_gex: true", text)

    def test_spy_split_and_roles_are_fixed(self):
        spy = self.spec["study"]["spy"]
        self.assertLess(pd.Timestamp(spy["early"][1]),
                        pd.Timestamp(spy["late_confirmation"][0]))
        self.assertEqual(spy["governing_split"], "late_confirmation")
        self.assertEqual(self.spec["study"]["qqq"]["role"],
                         "mechanism_diagnostic_only")
        self.assertEqual(self.spec["study"]["aapl"]["role"],
                         "mechanism_diagnostic_only")

    def test_earnings_is_omitted_for_lack_of_point_in_time_labels(self):
        aapl = self.spec["study"]["aapl"]
        self.assertEqual(aapl["earnings_overlay"], "omitted")
        self.assertIn("point-in-time", aapl["earnings_omission_reason"])

    def test_raw_paths_are_separate_from_compact_outputs(self):
        for source in ("qqq", "spy", "aapl"):
            self.assertTrue(self.spec["sources"][source]["raw_dir"].startswith(
                "data/free_sources/raw/kaggle/"))
        for key, value in self.spec["outputs"].items():
            if key != "report":
                self.assertTrue(value.startswith("data/free_sources/processed"))


class ParserAndShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_protocol()
        cls.sessions = pd.DatetimeIndex([
            "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"
        ])

    def test_header_normalization_handles_bom_brackets_and_spaces(self):
        got = normalize_headers(["\ufeff[QUOTE_DATE]", " [C_BID]", "STRIKE"])
        self.assertEqual(got, ["QUOTE_DATE", "C_BID", "STRIKE"])
        self.assertTrue(set(OPTION_COLUMNS).issubset(
            set(normalize_headers(option_rows("2020-01-02").columns))))

    def test_strict_time_and_exact_session_calendar_reject_bad_rows(self):
        good = option_rows("2020-01-02")
        bad_hour = option_rows("2020-01-03", hour=15.99)
        holiday = option_rows("2020-01-04")
        normalized, audit = normalize_option_frame(
            pd.concat([good, bad_hour, holiday], ignore_index=True),
            "SPY", self.sessions, self.spec,
        )
        self.assertEqual(set(normalized["quote_date"]), {pd.Timestamp("2020-01-02")})
        self.assertEqual(audit["rejected_wrong_time"], len(bad_hour))
        self.assertEqual(audit["rejected_non_session"], len(holiday))

    def test_crossed_and_nonfinite_quotes_are_rejected(self):
        frame = option_rows("2020-01-02")
        frame.loc[0, " [C_BID]"] = 2.0
        frame.loc[0, " [C_ASK]"] = 1.0
        frame.loc[1, " [P_BID]"] = np.nan
        normalized, audit = normalize_option_frame(
            frame, "SPY", self.sessions, self.spec,
        )
        self.assertEqual(len(normalized), len(frame) - 2)
        self.assertEqual(audit["rejected_invalid_quotes"], 2)

    def test_daily_shape_uses_frozen_delta_and_maturity_rules(self):
        normalized, _ = normalize_option_frame(
            option_rows("2020-01-02"), "SPY", self.sessions, self.spec,
        )
        daily = build_daily_surface(normalized, "SPY", self.spec)
        row = daily.iloc[0]
        # 30D call=.20, put=.22 at delta +/- .50.
        self.assertAlmostEqual(row["atm_iv_30"], 0.21)
        # 25-delta put=.24 at K95; call=.19 at K105.
        self.assertAlmostEqual(row["skew_25d"], 0.05)
        # 9D ATM mean=.23; 30D=.21.
        self.assertAlmostEqual(row["term_9d_30d"], 0.02)
        # Six rows, call .02*10 plus put .03*20 per row.
        self.assertAlmostEqual(row["gamma_weighted_volume"], 4.8)
        self.assertEqual(row["gamma_measure_label"],
                         "gamma_weighted_volume_not_dealer_gex")

    def test_delta_tolerance_makes_skew_missing_instead_of_extrapolating(self):
        frame = option_rows("2020-01-02")
        frame[" [C_DELTA]"] = 0.50
        frame[" [P_DELTA]"] = -0.50
        normalized, _ = normalize_option_frame(frame, "SPY", self.sessions, self.spec)
        row = build_daily_surface(normalized, "SPY", self.spec).iloc[0]
        self.assertTrue(np.isnan(row["skew_25d"]))

    def test_surface_is_lagged_one_full_session(self):
        daily = pd.DataFrame({
            "symbol": ["SPY", "SPY"],
            "quote_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "atm_iv_30": [0.20, 0.21],
        })
        got = add_full_session_lag(daily, self.sessions)
        self.assertEqual(got.loc[0, "measurement_date"], pd.Timestamp("2020-01-02"))
        self.assertEqual(got.loc[0, "origin"], pd.Timestamp("2020-01-03"))
        self.assertEqual(got.loc[0, "target_date"], pd.Timestamp("2020-01-06"))
        self.assertTrue((got["measurement_date"] < got["origin"]).all())
        self.assertTrue((got["origin"] < got["target_date"]).all())

    def test_aapl_split_audit_records_but_does_not_rewrite(self):
        daily = pd.DataFrame({
            "quote_date": pd.to_datetime(["2020-08-28", "2020-08-31"]),
            "underlying_last": [499.23, 129.04],
        })
        original = daily.copy(deep=True)
        audit = aapl_split_audit(daily, self.spec)
        self.assertTrue(audit["within_expected_range"])
        self.assertAlmostEqual(audit["pre_over_post_ratio"], 499.23 / 129.04)
        pd.testing.assert_frame_equal(daily, original)


class ForecastAndArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_protocol()

    def test_forecast_design_has_strict_surface_and_target_timing(self):
        idx = pd.bdate_range("2018-01-02", periods=40)
        close = pd.Series(np.exp(np.linspace(4.0, 4.2, len(idx))), index=idx)
        surface = pd.DataFrame({
            "symbol": "SPY",
            "measurement_date": idx[:-2],
            "origin": idx[1:-1],
            "target_date": idx[2:],
            "underlying_last": close.iloc[:-2].to_numpy(),
            "atm_iv_30": 0.20,
            "skew_25d": 0.04,
            "term_9d_30d": 0.01,
            "gamma_weighted_volume": 100.0,
        })
        design = build_forecast_design(surface, self.spec)
        self.assertTrue((design["measurement_date"] < design.index).all())
        self.assertTrue((design.index < design["target_date"]).all())
        self.assertTrue((design["surface_lag_sessions"] == 1).all())
        self.assertTrue((design["actual_var"] >= 0).all())

    def test_qlike_and_tail_scores_have_fixed_meaning(self):
        actual = np.array([1.0, 2.0, 4.0, 8.0])
        exact = qlike(actual, actual)
        np.testing.assert_allclose(exact, 0.0)
        frame = pd.DataFrame({
            "actual_var": actual,
            "baseline_var": [1, 1, 1, 1],
            "augmented_var": actual,
            "tail_event": [0, 0, 0, 1],
        })
        scores = score_forecasts(frame, self.spec)
        self.assertEqual(scores["augmented"]["mean_qlike"], 0.0)
        self.assertGreater(scores["augmented"]["auc"],
                           scores["baseline"]["auc"])
        self.assertEqual(scores["augmented"]["top_decile_event_rate"], 1.0)

    def test_manifest_hashes_private_sources_without_copying_them(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            raw = root / "raw.csv"
            raw.write_bytes(b"private raw bytes")
            out = root / "manifest.json"
            manifest = write_source_manifest(
                {"spy": [raw]}, out, self.spec, extra={"status": "test"}
            )
            self.assertEqual(manifest["sources"]["spy"][0]["sha256"],
                             hashlib.sha256(raw.read_bytes()).hexdigest())
            self.assertEqual(hash_file(raw), hashlib.sha256(raw.read_bytes()).hexdigest())
            self.assertNotIn(str(raw), out.read_text())

    def test_independent_verifier_recomputes_saved_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            processed = root / "data/free_sources/processed"
            processed.mkdir(parents=True)
            forecasts = pd.DataFrame({
                "symbol": ["SPY"] * 6,
                "split": ["late_confirmation"] * 6,
                "actual_var": [1., 2., 3., 4., 5., 10.],
                "baseline_var": [2.] * 6,
                "augmented_var": [1., 2., 3., 4., 5., 10.],
                "tail_event": [0, 0, 0, 0, 0, 1],
                "measurement_date": pd.bdate_range("2020-01-01", periods=6),
                "origin": pd.bdate_range("2020-01-02", periods=6),
                "target_date": pd.bdate_range("2020-01-03", periods=6),
            }).set_index("origin")
            fpath = processed / "surface_study_forecasts.parquet"
            forecasts.to_parquet(fpath)
            scores = score_forecasts(forecasts, self.spec)
            metrics = {
                "protocol_sha256": protocol_sha256(),
                "spy": {"late_confirmation": scores},
            }
            (processed / "surface_study_metrics.json").write_text(
                json.dumps(metrics, sort_keys=True)
            )
            result = verify_artifacts(DEFAULT_PROTOCOL, root=root)
            self.assertEqual(result["status"], "PASS")
            self.assertGreaterEqual(result["checks"], 8)


if __name__ == "__main__":
    unittest.main()
