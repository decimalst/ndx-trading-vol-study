"""Prewritten synthetic contracts for the archival SPX experiment."""
import hashlib
import io
import unittest
import zipfile

import numpy as np
import pandas as pd

from src.iterative_hf import (
    ALL_FEATURES,
    BASE,
    BLOCKS,
    build_features,
    fit_predict,
    forecast_panel,
    make_targets,
    merge_daily,
    parse_oxford_archive,
    training_mask,
)


def fixture(n=1200):
    rng = np.random.default_rng(143)
    dates = pd.bdate_range("2000-01-03", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .009, n)))
    opening = close * np.exp(rng.normal(0, .004, n))
    daily = pd.DataFrame({"open": opening, "close": close,
                          "high": np.maximum(opening, close) * np.exp(rng.uniform(.001, .02, n)),
                          "low": np.minimum(opening, close) * np.exp(-rng.uniform(.001, .02, n)),
                          "volume": rng.integers(1000, 100000, n)}, index=dates)
    rv = np.exp(rng.normal(-9.0, .7, n))
    hf = pd.DataFrame({"rv5": rv, "rsv": rv * rng.uniform(.05, .95, n),
                       "rk_parzen": rv * np.exp(rng.normal(0, .15, n))}, index=dates)
    vix = pd.Series(np.exp(rng.normal(3, .2, n)), index=dates)
    return daily, hf, vix


class ArchiveContracts(unittest.TestCase):
    def archive(self, dates=None, symbols=None):
        f = pd.DataFrame({"date": dates or ["2017-06-01 00:00:00+01:00", "2017-06-02 00:00:00+01:00"],
                          "Symbol": symbols or [".SPX", ".SPX"], "rv5": [.001, .002],
                          "rsv": [.0003, .0008], "rk_parzen": [.0011, .0018]})
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as z:
            z.writestr("archive.csv", f.to_csv(index=False))
        return content.getvalue()

    def test_local_trading_date_is_preserved_across_summer_offset(self):
        result = parse_oxford_archive(self.archive())
        self.assertEqual(result.index[0], pd.Timestamp("2017-06-01"))
        self.assertEqual(result.index[1], pd.Timestamp("2017-06-02"))

    def test_only_spx_is_selected_and_nasdaq_is_not_aliased(self):
        result = parse_oxford_archive(self.archive(symbols=[".SPX", ".IXIC"]))
        self.assertEqual(len(result), 1)
        with self.assertRaises(ValueError):
            parse_oxford_archive(self.archive(symbols=[".IXIC", ".IXIC"]))

    def test_source_hash_is_enforced(self):
        raw = self.archive()
        parse_oxford_archive(raw, hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            parse_oxford_archive(raw, "0" * 64)

    def test_source_end_fence_removes_later_archive_sessions(self):
        result = parse_oxford_archive(self.archive(), end="2017-06-01")
        self.assertEqual(result.index.tolist(), [pd.Timestamp("2017-06-01")])

    def test_duplicate_and_invalid_dates_are_rejected(self):
        for dates in [["2017-06-01", "2017-06-01"], ["invalid", "2017-06-02"]]:
            with self.assertRaises(ValueError):
                parse_oxford_archive(self.archive(dates=dates))


class FeatureContracts(unittest.TestCase):
    def test_gk_total_har_and_negative_returns_match_direct_definitions(self):
        d, h, v = fixture()
        f = build_features(d, h, v)
        gk = np.maximum(.5 * np.log(d.high / d.low) ** 2
                        - (2 * np.log(2) - 1) * np.log(d.close / d.open) ** 2, 1e-10)
        total = gk + np.log(d.open / d.close.shift()) ** 2
        ret = np.log(d.close).diff()
        for suffix, window in [("d", 1), ("w", 5), ("m", 22)]:
            np.testing.assert_allclose(f["gk_" + suffix], np.log(total.rolling(window).mean()), equal_nan=True)
            np.testing.assert_allclose(f["lev_" + suffix], ret.rolling(window).mean().clip(upper=0), equal_nan=True)

    def test_hf_har_share_and_kernel_use_matching_rolling_sums_and_lag(self):
        d, h, v = fixture()
        f = build_features(d, h, v)
        for suffix, window in [("d", 1), ("w", 5), ("m", 22)]:
            denom = h.rv5.rolling(window).sum()
            np.testing.assert_allclose(f["hf_" + suffix], np.log(h.rv5.rolling(window).mean()).shift(), equal_nan=True)
            np.testing.assert_allclose(f["share_" + suffix], (h.rsv.rolling(window).sum() / denom).shift(), equal_nan=True)
            np.testing.assert_allclose(f["kernel_" + suffix], np.log(h.rk_parzen.rolling(window).sum() / denom).shift(), equal_nan=True)
        np.testing.assert_allclose(f.liv, np.log(v).shift(), equal_nan=True)

    def test_missing_hf_session_is_not_compressed_or_filled(self):
        d, h, v = fixture()
        omitted = h.index[300]
        f = build_features(d, h.drop(omitted), v)
        self.assertTrue(np.isnan(f.loc[d.index[301], "hf_d"]))
        self.assertTrue(np.isnan(f.loc[d.index[305], "share_w"]))
        self.assertTrue(np.isfinite(f.loc[d.index[306], "share_w"]))

    def test_vix_lag_follows_spx_calendar_and_missing_is_not_filled(self):
        d, h, v = fixture()
        missing = v.index[300]
        v = v.drop(missing)
        f = build_features(d, h, v)
        self.assertTrue(np.isnan(f.loc[d.index[301], "liv"]))
        self.assertAlmostEqual(f.loc[d.index[302], "liv"], np.log(v.loc[d.index[301]]))

    def test_future_and_same_day_hf_changes_do_not_change_available_features(self):
        d, h, v = fixture()
        first = build_features(d, h, v)
        t = d.index[500]
        h.loc[t:, ["rv5", "rsv", "rk_parzen"]] *= 3
        v.loc[t:] *= 2
        second = build_features(d, h, v)
        pd.testing.assert_series_equal(first.loc[t, ALL_FEATURES], second.loc[t, ALL_FEATURES])

    def test_future_daily_changes_do_not_change_features_at_origin(self):
        d, h, v = fixture()
        first = build_features(d, h, v)
        d.iloc[501:, d.columns.get_indexer(["open", "high", "low", "close"])] *= 1.2
        second = build_features(d, h, v)
        pd.testing.assert_frame_equal(first.loc[:d.index[500]], second.loc[:d.index[500]])

    def test_invalid_price_and_measurement_identity_fail(self):
        for kind in ["price", "share", "kernel"]:
            d, h, v = fixture()
            if kind == "price":
                d.loc[d.index[40], "low"] = d.loc[d.index[40], "high"] * 2
            elif kind == "share":
                h.loc[h.index[40], "rsv"] = 2 * h.loc[h.index[40], "rv5"]
            else:
                h.loc[h.index[40], "rk_parzen"] = -1
            with self.assertRaises(ValueError):
                build_features(d, h, v)

    def test_zero_semivariance_is_valid_when_total_variance_is_positive(self):
        d, h, v = fixture()
        h.loc[h.index[40], "rsv"] = 0
        f = build_features(d, h, v)
        self.assertEqual(f.loc[h.index[41], "share_d"], 0)


class TargetContracts(unittest.TestCase):
    def test_future_mean_end_and_delayed_availability_follow_full_calendar(self):
        dates = pd.to_datetime(["2017-12-21", "2017-12-22", "2017-12-26", "2017-12-27", "2017-12-28", "2017-12-29"])
        rv = pd.Series(np.arange(1, 7) * .001, index=dates)
        t = make_targets(rv, dates, 3)
        self.assertAlmostEqual(t.loc[dates[0], "y"], .003)
        self.assertEqual(t.loc[dates[0], "target_end"], dates[3])
        self.assertEqual(t.loc[dates[0], "available_date"], dates[4])
        self.assertTrue(pd.isna(t.iloc[-3].available_date))

    def test_missing_zero_and_nonfinite_target_windows_are_not_averaged_around(self):
        dates = pd.bdate_range("2001-01-01", periods=10)
        for value in [0., np.nan, np.inf]:
            rv = pd.Series(.001, index=dates)
            rv.iloc[2] = value
            t = make_targets(rv, dates, 3)
            self.assertTrue(pd.isna(t.iloc[0].y))

    def test_training_waits_one_more_session_after_target_completion(self):
        d, h, v = fixture()
        f = build_features(d, h, v)
        targets = make_targets(h.rv5, d.index, 5)
        origin = d.index[1000]
        mask = training_mask(f, targets, origin, 750)
        self.assertFalse(mask.loc[d.index[995]])
        self.assertTrue(mask.loc[d.index[994]])
        self.assertTrue((targets.loc[mask, "available_date"] <= origin).all())

    def test_training_uses_complete_common_candidate_rows_and_minimum(self):
        d, h, v = fixture()
        f = build_features(d, h, v)
        t = make_targets(h.rv5, d.index, 1)
        f.loc[d.index[200], "share_d"] = np.nan
        mask = training_mask(f, t, d.index[1000], 750)
        self.assertFalse(mask.loc[d.index[200]])
        with self.assertRaises(ValueError):
            training_mask(f, t, d.index[400], 750)


class FitAndPanelContracts(unittest.TestCase):
    def test_scaled_ols_exact_smearing_matches_direct_unscaled_solve(self):
        d, h, v = fixture()
        f = build_features(d, h, v).dropna(subset=ALL_FEATURES)
        x, apply = f.iloc[:900], f.iloc[900:910]
        y = h.loc[x.index, "rv5"].to_numpy()
        result = fit_predict(x, y, apply)
        for model, extra in [("baseline", ()), *BLOCKS.items()]:
            columns = list(BASE + extra)
            a = x[columns].to_numpy()
            beta = np.linalg.lstsq(a, np.log(y), rcond=None)[0]
            expected = np.exp(apply[columns].to_numpy() @ beta) * np.mean(np.exp(np.log(y) - a @ beta))
            np.testing.assert_allclose(result["predictions"][model], expected, rtol=1e-10)
            audit = result["audit"][model]
            self.assertEqual(audit["rank"], len(columns))
            self.assertAlmostEqual(audit["smear"], np.mean(np.exp(np.log(y) - a @ beta)), places=10)
            np.testing.assert_allclose(audit["means"][1:], x[columns[1:]].to_numpy().mean(axis=0))
            np.testing.assert_allclose(audit["scales"][1:], x[columns[1:]].to_numpy().std(axis=0, ddof=0))

    def test_apply_rows_cannot_change_fit_transform(self):
        d, h, v = fixture()
        f = build_features(d, h, v).dropna(subset=ALL_FEATURES)
        x = f.iloc[:900]
        y = h.loc[x.index, "rv5"].to_numpy()
        apply = f.iloc[900:905].copy()
        one = fit_predict(x, y, apply)
        apply.loc[:, "gk_d"] += .2
        two = fit_predict(x, y, apply)
        self.assertEqual(one["audit"], two["audit"])

    def test_redundant_or_zero_scale_candidate_and_nonpositive_target_fail(self):
        d, h, v = fixture()
        f = build_features(d, h, v).dropna(subset=ALL_FEATURES)
        for column in [0., f.gk_d]:
            x = f.iloc[:900].copy()
            x.loc[:, "share_d"] = column if np.isscalar(column) else column.reindex(x.index)
            with self.assertRaises(ValueError):
                fit_predict(x, h.loc[x.index, "rv5"], f.iloc[900:905])
        y = h.loc[f.iloc[:900].index, "rv5"].copy()
        y.iloc[0] = 0
        with self.assertRaises(ValueError):
            fit_predict(f.iloc[:900], y, f.iloc[900:905])

    def test_panel_has_common_origins_all_horizons_and_purges_development_boundary(self):
        d, h, v = fixture(1400)
        f = build_features(d, h, v)
        targets = {k: make_targets(h.rv5, d.index, k) for k in [1, 5, 21]}
        cfg = {"development_start": str(d.index[900].date()),
               "development_end": str(d.index[1000].date()),
               "evaluation_start": str(d.index[1001].date()),
               "evaluation_end": str(d.index[1100].date()),
               "latest_target": str(d.index[1100].date()),
               "horizons": [1, 5, 21], "min_train": 750}
        forecasts, audits = forecast_panel(f, targets, cfg)
        counts = forecasts.groupby(["model", "horizon"]).size()
        self.assertEqual(len(counts), 9)
        self.assertEqual(len(set(counts)), 1)
        self.assertTrue((forecasts.target_end <= pd.Timestamp(cfg["latest_target"])).all())
        dev = forecasts[forecasts.phase == "development"]
        self.assertTrue((dev.available_date <= pd.Timestamp(cfg["development_end"])).all())
        self.assertTrue((forecasts.train_last_available <= forecasts.fit_origin).all())
        self.assertTrue(audits)
        self.assertEqual(forecasts.groupby(["horizon", forecasts.origin.dt.to_period("M")]).fit_origin.nunique().max(), 1)

        nested = {"development": [cfg["development_start"], cfg["development_end"]],
                  "evaluation": [cfg["evaluation_start"], cfg["evaluation_end"]],
                  "minimum_train": 750, "latest_target": cfg["latest_target"],
                  "horizons": [1, 5, 21], "baseline": list(BASE)}
        other, _ = forecast_panel(f, targets, nested)
        pd.testing.assert_frame_equal(forecasts, other)


class HistoricalMergeContracts(unittest.TestCase):
    def test_overlap_is_verified_and_existing_prices_preserved(self):
        d, _, _ = fixture(500)
        cutoff = d.index[200]
        merged, audit = merge_daily(d.iloc[:350], d.iloc[200:], cutoff, min_overlap=100)
        pd.testing.assert_frame_equal(merged, d)
        self.assertEqual(audit["overlap_rows"], 150)

    def test_material_price_disagreement_or_missing_overlap_session_fails(self):
        d, _, _ = fixture(500)
        bad = d.iloc[:350].copy()
        bad.loc[d.index[250], "close"] *= 1.01
        with self.assertRaises(ValueError):
            merge_daily(bad, d.iloc[200:], d.index[200], min_overlap=100)
        with self.assertRaises(ValueError):
            merge_daily(d.iloc[:350].drop(d.index[250]), d.iloc[200:], d.index[200], min_overlap=100)
        with self.assertRaises(ValueError):
            merge_daily(d.iloc[:350].drop(d.index[200]), d.iloc[200:], d.index[200], min_overlap=100)


if __name__ == "__main__":
    unittest.main()
