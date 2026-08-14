"""Safety and estimand contracts for the frozen five-path extension."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.research_paths import (
    align_exact_dates,
    assign_top25_asof,
    build_absorption_design,
    build_quarterly_top25,
    build_single_name_design,
    build_top25_earnings_events,
    dm_test_hac,
    event_target_flags,
    fit_absorption_group,
    forward_calendar_realized_vol,
    forward_mean_variance,
    gk_plus_overnight,
    moving_block_bootstrap,
    parse_cboe_history,
    positive_front_slope,
    qlike_loss,
    training_rows_completed_by,
    validate_protocol,
    wald_attenuation,
    walk_forward_log_ols,
)

ROOT = Path(__file__).resolve().parents[1]


class ProtocolFenceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = yaml.safe_load((ROOT / "research_paths.yaml").read_text())

    def test_frozen_protocol_validates(self):
        validate_protocol(self.spec)
        self.assertEqual(self.spec["absorption_map"]["hac_maxlags"], 10)

    def test_ndx_studies_end_before_clean_window(self):
        clean = pd.Timestamp(self.spec["fences"]["sealed_ndx_clean_start"])
        for key in ("absorption_map", "horizon_curve"):
            self.assertLess(pd.Timestamp(self.spec[key]["end"]), clean)

    def test_spx_holdout_predates_ndx_term_discovery(self):
        score_end = pd.Timestamp(self.spec["spx_term_slope_replication"]["score_end"])
        ndx_start = pd.Timestamp(self.spec["fences"]["ndx_diagnostic_start"])
        self.assertLess(score_end, ndx_start)
        self.assertEqual(self.spec["spx_term_slope_replication"]
                         ["cboe_close_delay_sessions"], 1)

    def test_no_model_family_selection_is_allowed(self):
        self.assertTrue(self.spec["fences"]["no_candidate_selection"])
        self.assertEqual(list(self.spec["single_name_earnings"]["matched_iv_family"]),
                         ["AAPL", "AMZN", "GOOG", "IBM"])


class QuarterlyTop25Contract(unittest.TestCase):
    @staticmethod
    def sample_holdings() -> pd.DataFrame:
        rows = []
        accepted = pd.to_datetime(["2020-02-28T20:30:00Z", "2020-05-29T21:30:00Z"], utc=True)
        for snap, when in zip(("old", "new"), accepted):
            for i in range(27):
                issuer = f"{i:06d}"
                rows.append({
                    "accession": snap, "report_date": pd.Timestamp("2019-12-31")
                    if snap == "old" else pd.Timestamp("2020-03-31"),
                    "accepted_at": when, "name": f"Issuer {i}",
                    "cusip": issuer + "10", "pct_value": float(100 - i),
                    "value_usd": float(1000 - i), "asset_category": "EC",
                })
        # A second share class must be folded into issuer 000024, pushing it up.
        rows.append({
            "accession": "old", "report_date": pd.Timestamp("2019-12-31"),
            "accepted_at": accepted[0], "name": "Issuer 24 class B",
            "cusip": "00002430", "pct_value": 100.0, "value_usd": 1000.0,
            "asset_category": "EC",
        })
        # A derivative can never enter the ranking even with a huge value.
        rows.append({
            "accession": "old", "report_date": pd.Timestamp("2019-12-31"),
            "accepted_at": accepted[0], "name": "Future", "cusip": "99999999",
            "pct_value": 1000.0, "value_usd": 10000.0, "asset_category": "DE",
        })
        return pd.DataFrame(rows)

    def test_rank_is_issuer_level_and_exactly_25_per_snapshot(self):
        got = build_quarterly_top25(self.sample_holdings(), keep=25)
        self.assertTrue((got.groupby("accession").size() == 25).all())
        old = got[got["accession"] == "old"]
        self.assertEqual((old["issuer_id"] == "000024").sum(), 1)
        self.assertEqual(old.loc[old["issuer_id"] == "000024", "rank"].iloc[0], 1)
        self.assertNotIn("999999", set(old["issuer_id"]))

    def test_after_close_acceptance_waits_until_next_origin(self):
        top = build_quarterly_top25(self.sample_holdings(), keep=25)
        origins = pd.to_datetime(["2020-05-29", "2020-06-01"])
        got = assign_top25_asof(top, origins, origin_hour_et=16)
        self.assertEqual(got.loc[got.origin == origins[0], "accession"].unique().tolist(), ["old"])
        self.assertEqual(got.loc[got.origin == origins[1], "accession"].unique().tolist(), ["new"])

    def test_future_snapshot_cannot_change_past_universe(self):
        top = build_quarterly_top25(self.sample_holdings(), keep=25)
        origins = pd.to_datetime(["2020-03-02"])
        base = assign_top25_asof(top[top.accession == "old"], origins)
        future = assign_top25_asof(top, origins)
        pd.testing.assert_frame_equal(base.reset_index(drop=True), future.reset_index(drop=True))

    def test_pre_first_acceptance_has_no_current_membership_fallback(self):
        top = build_quarterly_top25(self.sample_holdings(), keep=25)
        got = assign_top25_asof(top, pd.to_datetime(["2020-01-02"]))
        self.assertTrue(got.empty)

    def test_frozen_symbol_map_covers_every_observed_top25_issuer(self):
        holdings = pd.read_parquet(ROOT / "data/raw/qqq_nport_holdings.parquet")
        top = build_quarterly_top25(holdings, keep=25)
        mapping = pd.read_csv(ROOT / "calendars/top25_symbol_map.csv", dtype=str,
                              comment="#")
        self.assertFalse(mapping["issuer_id"].duplicated().any())
        missing = set(top["issuer_id"]) - set(mapping["issuer_id"])
        self.assertEqual(missing, set())

    def test_event_panel_uses_historical_rank_not_present_membership(self):
        top = build_quarterly_top25(self.sample_holdings(), keep=25)
        sessions = pd.date_range("2020-03-02", periods=5, freq="B")
        events = pd.DataFrame({
            "date": [sessions[1], sessions[1]],
            "ticker": ["IN25", "OUT26"],
            "session": ["bmo", "bmo"],
        })
        mapping = pd.DataFrame({
            "issuer_id": ["000024", "000026"],
            "ticker": ["IN25", "OUT26"],
        })
        got = build_top25_earnings_events(events, top, sessions, mapping)
        self.assertEqual(got["ticker"].tolist(), ["IN25"])
        self.assertEqual(got["rank"].tolist(), [1])


class AvailabilityContract(unittest.TestCase):
    def test_exact_date_alignment_never_forward_fills(self):
        source = pd.Series([10.0, 30.0], index=pd.to_datetime(["2020-01-02", "2020-01-06"]))
        target = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        got = align_exact_dates(source, target)
        self.assertEqual(got.loc["2020-01-02"], 10.0)
        self.assertTrue(np.isnan(got.loc["2020-01-03"]))
        self.assertEqual(got.loc["2020-01-06"], 30.0)

    def test_overlapping_target_uses_only_strictly_future_sessions(self):
        x = pd.Series([1., 2., 3., 4., 5.], index=pd.date_range("2020-01-01", periods=5))
        got = forward_mean_variance(x, 2)
        self.assertEqual(got.iloc[0], 2.5)
        self.assertEqual(got.iloc[1], 3.5)
        self.assertTrue(np.isnan(got.iloc[-1]))

    def test_training_labels_must_be_complete_at_origin(self):
        idx = pd.date_range("2020-01-01", periods=8)
        mask = training_rows_completed_by(idx, origin=idx[5], horizon=3)
        self.assertEqual(idx[mask].tolist(), idx[:3].tolist())

    def test_future_rows_cannot_change_an_already_complete_target(self):
        idx = pd.date_range("2020-01-01", periods=6)
        base = pd.Series(np.arange(1., 7.), index=idx)
        extended = pd.concat([base, pd.Series([999.], index=[idx[-1] + pd.Timedelta(days=1)])])
        self.assertEqual(forward_mean_variance(base, 2).iloc[1],
                         forward_mean_variance(extended, 2).iloc[1])


class SourceAndEstimatorContract(unittest.TestCase):
    def test_cboe_parser_normalizes_columns_and_rejects_duplicates(self):
        text = "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2020,10,11,9,10.5\n01/03/2020,11,12,10,11.5\n"
        got = parse_cboe_history(text, "VIX9D")
        self.assertEqual(got.name, "vix9d")
        self.assertEqual(got.index[0], pd.Timestamp("2020-01-02"))
        bad = text + "01/03/2020,11,12,10,11.5\n"
        with self.assertRaises(ValueError):
            parse_cboe_history(bad, "VIX9D")

    def test_gk_and_overnight_reconcile_exactly(self):
        d = pd.DataFrame({
            "open": [100., 104., 103.], "high": [102., 106., 105.],
            "low": [99., 102., 101.], "close": [101., 103., 104.],
        }, index=pd.date_range("2020-01-01", periods=3))
        got = gk_plus_overnight(d)
        np.testing.assert_allclose(got["rv_total"],
                                   got["rv_intraday"] + got["var_overnight"])
        self.assertTrue((got[["rv_intraday", "var_overnight", "rv_total"]]
                         .dropna() >= 0).all().all())

    def test_frozen_surface_sources_match_their_spx_horizons(self):
        spec = yaml.safe_load((ROOT / "research_paths.yaml").read_text())
        vrp = spec["vrp_term_structure"]
        self.assertEqual(vrp["horizons_calendar_days"],
                         {"vix9d": 9, "vix": 30, "vix3m": 93})
        for symbol, url in vrp["implied_sources"].items():
            self.assertIn("cdn.cboe.com/api/global/us_indices/daily_prices/", url)
            self.assertIn(symbol.upper(), url.upper())


class StudyDefinitionContract(unittest.TestCase):
    def test_front_dislocation_is_positive_part_only(self):
        v9 = pd.Series([15., 20., 30.])
        v30 = pd.Series([20., 20., 20.])
        got = positive_front_slope(v9, v30)
        np.testing.assert_allclose(got, [0., 0., np.log(1.5)])

    def test_earnings_labels_are_mapped_after_the_event_session(self):
        sessions = pd.date_range("2020-01-02", periods=4, freq="B")
        events = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "session": ["bmo", "amc"],
        })
        got = event_target_flags(sessions, events)
        self.assertEqual(got.loc["2020-01-02"], 1)
        self.assertEqual(got.loc["2020-01-06"], 1)
        self.assertEqual(got.loc["2020-01-03"], 0)

    def test_single_name_forecast_design_cannot_accept_event_labels(self):
        idx = pd.date_range("2020-01-01", periods=30)
        frame = pd.DataFrame({
            "rv_total": np.linspace(.001, .002, 30),
            "log_rv": np.log(np.linspace(.001, .002, 30)),
            "own_iv": np.linspace(20., 25., 30),
        }, index=idx)
        design = build_single_name_design(frame)
        self.assertEqual(list(design.columns),
                         ["const", "lrv_d", "lrv_w", "lrv_m", "liv"])

    def test_wald_attenuation_is_unbounded_and_signed(self):
        self.assertAlmostEqual(wald_attenuation(100., 60.), .4)
        self.assertAlmostEqual(wald_attenuation(100., 120.), -.2)
        with self.assertRaises(ValueError):
            wald_attenuation(0., 1.)

    def test_absorption_fit_uses_one_common_sample(self):
        rng = np.random.default_rng(3)
        n = 400
        index = pd.date_range("2010-01-01", periods=n, freq="B")
        group = rng.normal(size=n)
        market = .5 * group + rng.normal(size=n)
        y = 2 * group + market + rng.normal(scale=.5, size=n)
        design = pd.DataFrame({
            "const": 1.0, "lrv_d": rng.normal(size=n),
            "lrv_w": rng.normal(size=n), "lrv_m": rng.normal(size=n),
            "liv": market, "signal": group, "y_next": y,
        }, index=index)
        design.loc[index[10], "liv"] = np.nan
        got = fit_absorption_group(design, ["signal"], index.min(), index.max(), 2)
        self.assertEqual(got["n_without"], got["n_with"])
        self.assertGreater(got["wald_without"], 0)
        self.assertTrue(np.isfinite(got["attenuation"]))

    def test_qlike_and_dm_sign_match_model_ordering(self):
        actual = np.array([1., 2., 3., 4.] * 20)
        good = actual * 1.01
        bad = actual * (1.8 + .2 * np.sin(np.arange(len(actual))))
        loss_good = qlike_loss(actual, good)
        loss_bad = qlike_loss(actual, bad)
        self.assertLess(loss_good.mean(), loss_bad.mean())
        result = dm_test_hac(loss_good, loss_bad, horizon=1)
        self.assertLess(result["dm"], 0)

    def test_absorption_events_and_weekdays_belong_to_target_session(self):
        idx = pd.date_range("2020-01-06", periods=30, freq="B")
        rv = np.linspace(.001, .003, len(idx))
        master = pd.DataFrame({
            "rv_total": rv, "log_rv": np.log(rv), "ret_cc": 0.0,
            "vxn": 20.0, "dow": idx.dayofweek,
            "is_fomc": 0.0, "is_cpi": 0.0, "is_nfp": 0.0,
        }, index=idx)
        master.loc[idx[5], "is_fomc"] = 1.0
        earnings = pd.Series(0.0, index=idx)
        earnings.loc[idx[8]] = 2.5
        got = build_absorption_design(master, earnings)
        self.assertEqual(got.loc[idx[4], "target_fomc"], 1.0)
        self.assertEqual(got.loc[idx[5], "target_post_fomc"], 1.0)
        self.assertEqual(got.loc[idx[7], "target_top25_earnings_weight"], 2.5)
        self.assertEqual(got.loc[idx[0], "target_tuesday"],
                         float(idx[1].dayofweek == 1))

    def test_forward_calendar_vol_uses_open_right_calendar_window(self):
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
        returns = pd.Series([np.nan, .01, .02, .03], index=idx)
        got = forward_calendar_realized_vol(returns, 3)
        expected = 100 * np.sqrt((365 / 3) * (.01**2 + .02**2))
        self.assertAlmostEqual(got.loc["2020-01-01"], expected)
        self.assertTrue(np.isnan(got.loc["2020-01-06"]))

    def test_moving_block_bootstrap_is_seed_deterministic(self):
        values = np.arange(30.0)
        a = moving_block_bootstrap(values, block=5, draws=20, seed=7,
                                   statistic=np.mean)
        b = moving_block_bootstrap(values, block=5, draws=20, seed=7,
                                   statistic=np.mean)
        np.testing.assert_array_equal(a, b)

    def test_walk_forward_ignores_uncompleted_and_future_labels(self):
        idx = pd.date_range("2010-01-01", periods=80, freq="B")
        x = pd.DataFrame({"const": 1.0, "x": np.linspace(-1, 1, 80)}, index=idx)
        y = pd.Series(1.0 + .2 * x["x"], index=idx)
        y_var = np.exp(y)
        origin = idx[60]
        base = walk_forward_log_ols(x, y, y_var, [origin], label_horizon=5,
                                    min_train_rows=20)
        altered = y.copy()
        altered.loc[idx[56]:] = 10.0
        changed = walk_forward_log_ols(x, altered, np.exp(altered), [origin],
                                       label_horizon=5, min_train_rows=20)
        self.assertAlmostEqual(base.loc[origin, "mean_var"],
                               changed.loc[origin, "mean_var"])


if __name__ == "__main__":
    unittest.main()
