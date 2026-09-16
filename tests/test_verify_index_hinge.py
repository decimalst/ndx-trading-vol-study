"""Independent index-hinge contracts fixed before new empirical fits."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import verify_index_hinge as verify


def fixture():
    rng = np.random.default_rng(20260912)
    dates = pd.bdate_range("2005-01-03", periods=900)
    close = 100 * np.exp(rng.normal(.0001, .01, len(dates)).cumsum())
    opening = close * np.exp(rng.normal(0, .005, len(dates)))
    daily = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * 1.005,
                          "low": np.minimum(opening, close) / 1.005, "close": close}, index=dates)
    iv = pd.DataFrame({name: np.exp(rng.normal(3., .2, len(dates))) for name in ("vix", "vix9d", "vvix")}, index=dates)
    return daily, iv


class IndependentHingeFeatureTests(unittest.TestCase):
    def test_malformed_prices_and_invalid_calendar_rejected(self):
        daily, iv = fixture()
        for column, value in (("open", 0.), ("close", np.inf), ("high", .1), ("low", 10000.)):
            bad = daily.copy()
            bad.loc[daily.index[150], column] = value
            with self.assertRaises(ValueError):
                verify.raw_features(bad, iv)
        bad_iv = iv.copy()
        bad_iv.loc[daily.index[150], "vix"] = -1.
        with self.assertRaises(ValueError):
            verify.raw_features(daily, bad_iv)
        with self.assertRaises(ValueError):
            verify.raw_features(daily.iloc[::-1], iv)

    def test_exact_annualized_variance_levels_and_prior_market_lag(self):
        daily, iv = fixture()
        f = verify.raw_features(daily, iv)
        gk = np.maximum(.5 * np.log(daily.high / daily.low)**2
                        - (2 * np.log(2) - 1) * np.log(daily.close / daily.open)**2, 1e-10)
        variance = gk + np.log(daily.open / daily.close.shift())**2
        self.assertAlmostEqual(f.I.iloc[200], np.log((iv.vix.iloc[199] / 100)**2), places=14)
        self.assertAlmostEqual(f.R.iloc[200], np.log(252 * variance.iloc[178:200].mean()), places=14)
        self.assertAlmostEqual(f.lr_w.iloc[200], np.log(252 * variance.iloc[195:200].mean()), places=14)
        self.assertEqual(f.feature_cutoff_date.iloc[200], daily.index[199])

    def test_four_raw_return_controls_use_means_not_cumulative_returns(self):
        daily, iv = fixture()
        f = verify.raw_features(daily, iv)
        returns = np.log(daily.close).diff()
        for label, width in (("d", 1), ("w", 5), ("m", 22), ("q", 63)):
            self.assertAlmostEqual(f[f"ret_{label}"].iloc[200], returns.iloc[200-width:200].mean(), places=14)

    def test_missing_prior_observations_are_not_dropped_before_rolling(self):
        daily, iv = fixture()
        daily.loc[daily.index[140], ["open", "high", "low", "close"]] = np.nan
        f = verify.raw_features(daily, iv)
        self.assertTrue(np.isnan(f.lr_d.iloc[141]))
        self.assertTrue(np.isnan(f.R.iloc[163]))
        self.assertTrue(np.isfinite(f.R.iloc[164]))
        self.assertTrue(np.isnan(f.ret_q.iloc[204]))
        self.assertTrue(np.isfinite(f.ret_q.iloc[205]))

    def test_entry_and_future_price_mutations_do_not_change_current_features(self):
        daily, iv = fixture()
        before = verify.raw_features(daily, iv)
        later, later_iv = daily.copy(), iv.copy()
        later.iloc[200:] *= 1.3
        later_iv.iloc[200:] *= 2.
        after = verify.raw_features(later, later_iv)
        pd.testing.assert_series_equal(before.iloc[200], after.iloc[200])
        limited = verify.raw_features(daily.iloc[:201], iv.iloc[:201])
        pd.testing.assert_series_equal(before.iloc[200], limited.iloc[-1])

    def test_cboe_missing_date_is_not_forward_filled(self):
        daily, iv = fixture()
        iv = iv.drop(daily.index[140])
        f = verify.raw_features(daily, iv)
        self.assertTrue(f.loc[daily.index[141], ["I", "term", "lvvix"]].isna().all())
        self.assertTrue(np.isfinite(f.I.iloc[142]))

    def test_centered_squares_and_hinge_use_only_exact_training_rows(self):
        daily, iv = fixture()
        raw = verify.raw_features(daily, iv).loc[:, verify.RAW]
        train, application = raw.iloc[100:300], raw.iloc[301:305]
        a, b, audit = verify.derived_design(train, application)
        np.testing.assert_allclose(a.I_square, (train.I - train.I.mean())**2)
        np.testing.assert_allclose(b.R_square, (application.R - train.R.mean())**2)
        np.testing.assert_allclose(b.hinge, np.maximum(application.I - application.R - (train.I - train.R).mean(), 0))
        self.assertEqual(audit["train_n"], 200)
        application = application.copy()
        application[["I", "R"]] *= 10
        changed, _, after = verify.derived_design(train, application)
        pd.testing.assert_frame_equal(a, changed)
        self.assertEqual(audit, after)

    def test_hinge_threshold_is_training_gap_mean_not_mean_of_positive_gap(self):
        daily, iv = fixture()
        raw = verify.raw_features(daily, iv).loc[:, verify.RAW].iloc[100:200].copy()
        raw["I"] = np.linspace(-3, 2, len(raw))
        raw["R"] = np.linspace(2, -1, len(raw))
        transformed, _, audit = verify.derived_design(raw, raw.iloc[:1])
        self.assertAlmostEqual(audit["gap_mean"], (raw.I - raw.R).mean())
        np.testing.assert_allclose(transformed.hinge, np.maximum(raw.I - raw.R - audit["gap_mean"], 0))

    def test_derived_zero_scale_rejects_entire_design(self):
        daily, iv = fixture()
        raw = verify.raw_features(daily, iv).loc[:, verify.RAW].iloc[100:300].copy()
        raw["I"] = 3.
        with self.assertRaisesRegex(ValueError, "scale|degenerate"):
            verify.derived_design(raw, raw.iloc[:2])

    def test_future_returns_zero_and_negative_valid_and_long_tail_unavailable(self):
        dates = pd.bdate_range("2018-01-02", periods=70)
        close = pd.Series(np.repeat(100., len(dates)), index=dates)
        close.iloc[22] = 90.
        target = verify.future_returns(close, 21)
        self.assertEqual(target.y.iloc[0], 0.)
        self.assertAlmostEqual(target.y.iloc[1], np.log(.9))
        self.assertEqual(target.target_end.iloc[0], dates[21])
        self.assertEqual(target.available_date.iloc[0], dates[21])
        self.assertTrue(target.y.iloc[-21:].isna().all())
        self.assertTrue(target.available_date.iloc[-21:].isna().all())

    def test_training_maturity_requires_target_close_before_fit_entry(self):
        daily, _ = fixture()
        target = verify.future_returns(daily.close, 63)
        complete = pd.Series(True, index=daily.index)
        selected = verify.training_mask(complete, target, daily.index[300], daily.index)
        self.assertTrue(selected.iloc[236])
        self.assertFalse(selected.iloc[237])
        self.assertFalse(selected.iloc[300])

    def test_future_label_presence_cannot_select_monthly_refit(self):
        daily, iv = fixture()
        raw = verify.raw_features(daily, iv)
        target = verify.future_returns(daily.close, 21)
        section = {"origin_start": "2006-01-01", "origin_end": "2006-06-30", "latest_target": "2006-12-31",
                   "development": ["2006-01-01", "2006-03-31"], "evaluation": ["2006-04-01", "2006-06-30"],
                   "development_target_available_by": "2006-03-31"}
        first, scored, _ = verify.eligible_entries(raw, target, section)
        target.loc[first[0], "y"] = np.nan
        second, updated, _ = verify.eligible_entries(raw, target, section)
        self.assertTrue(first.equals(second))
        self.assertEqual(len(scored) - len(updated), 1)
        self.assertTrue((target.loc[scored[scored <= "2006-03-31"], "available_date"] <= "2006-03-31").all())


class IndependentHingeModelTests(unittest.TestCase):
    def test_augmented_least_squares_matches_mean_loss_ridge_and_intercept(self):
        rng = np.random.default_rng(10019)
        x = np.c_[np.ones(100), rng.normal(size=(100, 3))]
        y = rng.normal(.01, .05, 100)
        prediction, audit = verify.ridge_prediction(x, y, x[-3:])
        z = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)
        beta = np.linalg.solve(z.T @ z / len(y) + .01 * np.eye(3), z.T @ (y - y.mean()) / len(y))
        np.testing.assert_allclose(audit["beta"], np.r_[y.mean(), beta], atol=1e-12)
        np.testing.assert_allclose(prediction, y.mean() + z[-3:] @ beta, atol=1e-12)
        self.assertLessEqual(audit["gradient_max_abs"], 1e-10)

    def test_repeated_rows_preserve_mean_loss_penalty(self):
        rng = np.random.default_rng(819)
        x = np.c_[np.ones(80), rng.normal(size=(80, 3))]
        y = rng.normal(0, .1, 80)
        first, audit = verify.ridge_prediction(x, y, x[-5:])
        repeated, repeated_audit = verify.ridge_prediction(np.tile(x, (4, 1)), np.tile(y, 4), x[-5:])
        np.testing.assert_allclose(first, repeated, atol=1e-12)
        np.testing.assert_allclose(audit["beta"], repeated_audit["beta"], atol=1e-12)

    def test_application_extremes_never_change_fit_or_training_scaling(self):
        rng = np.random.default_rng(323)
        x = np.c_[np.ones(100), rng.normal(size=(100, 3))]
        y = rng.normal(0, .1, 100)
        _, a = verify.ridge_prediction(x, y, x[-3:])
        _, b = verify.ridge_prediction(x, y, np.array([[1., 300., -300., 200.]]))
        np.testing.assert_array_equal(a["means"], b["means"])
        np.testing.assert_array_equal(a["beta"], b["beta"])

    def test_intercept_only_mean_and_zero_signed_returns(self):
        y = np.array([0., -.1, .1])
        prediction, audit = verify.ridge_prediction(np.ones((3, 1)), y, np.ones((4, 1)))
        np.testing.assert_array_equal(prediction, np.zeros(4))
        self.assertEqual(audit["alpha"], 0.)

    def test_nonoverlap_offsets_use_full_calendar_positions_and_require_all_offsets(self):
        calendar = pd.bdate_range("2019-01-01", periods=100)
        origins = calendar[[1, 2, 5, 7, 8, 10, 12, 13, 15]]
        difference = -np.arange(1, len(origins) + 1, dtype=float)
        rows = verify.nonoverlap_rows(origins, difference, calendar, 3)
        for offset, row in enumerate(rows):
            chosen = calendar.get_indexer(origins) % 3 == offset
            self.assertEqual(row["n"], int(chosen.sum()))
            self.assertAlmostEqual(row["delta"], difference[chosen].mean())
        phases = [{"gain_relative": .0025, "nonoverlap_phases": rows, "stability": []},
                  {"gain_relative": .003, "nonoverlap_phases": rows, "stability": [{"delta": -.1}, {"delta": -.2}]}]
        self.assertTrue(verify.effect_passes(phases, 3))
        rows[1]["delta"] = 0.
        self.assertFalse(verify.effect_passes(phases, 3))

    def test_long_horizon_hac_and_seed_contract(self):
        n = 510
        dates = pd.bdate_range("2016-01-04", periods=n)
        values = np.linspace(-.1, .2, n)
        panel = pd.concat([pd.DataFrame({"origin": dates, "available_date": dates, "horizon": 21,
                                         "model": name, "y": values, "prediction": prediction})
                           for name, prediction in (("hinge", .02), ("baseline", .0))])
        protocol = {"index": {"development": ["2016-01-01", "2018-12-31"],
                               "development_target_available_by": "2018-12-31"},
                    "inference": {"blocks": [126, 252, 504], "seed": 20260912, "bootstrap_draws": 3}}
        with patch.object(verify, "explicit_bootstrap_means", return_value=np.zeros((3, 1))) as boot, \
                patch.object(verify, "independent_hac", return_value={"se": .1, "p": .8, "ci95": [-.1, .1], "mde80_nominal": .2}) as hac:
            got = verify.phase_statistics(panel, "baseline", 21, "development", 0, protocol, dates)
        self.assertEqual(hac.call_args.kwargs, {"maxlags": 504})
        self.assertEqual([call.args[-1] for call in boot.call_args_list], [20260912 + 21 * 1000000 + block for block in (126, 252, 504)])
        self.assertEqual(len(got["nonoverlap_phases"]), 21)
        self.assertGreaterEqual(got["p_conservative"], .8)

    def test_exactly_504_rows_cannot_claim_504_hac_lags(self):
        dates = pd.bdate_range("2016-01-04", periods=504)
        panel = pd.concat([pd.DataFrame({"origin": dates, "available_date": dates, "horizon": 21,
                                         "model": name, "y": .1, "prediction": prediction})
                           for name, prediction in (("hinge", .02), ("baseline", .0))])
        protocol = {"index": {"development": ["2016-01-01", "2018-12-31"],
                               "development_target_available_by": "2018-12-31"},
                    "inference": {"blocks": [126, 252, 504], "seed": 20260912, "bootstrap_draws": 3}}
        with self.assertRaisesRegex(AssertionError, "Insufficient"):
            verify.phase_statistics(panel, "baseline", 21, "development", 0, protocol, dates)

    def test_inherited_alternative_measure_identity_and_source_position_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"study": "prior", "candidate": "quality", "control": "mean", "horizon": 1,
                     "measure": "rv5" if index % 2 else "qmle", "p_conservative": 1.} for index in range(99)]
            (root / "prior.json").write_text(json.dumps({"rows": rows}))
            inherited = verify.inherited_rows(root, {"comparisons": {"inherited_sources": ["prior.json"]}})
            self.assertEqual([row["source_row_index"] for row in inherited], list(range(99)))
            self.assertEqual([row["measure"] for row in inherited], [row["measure"] for row in rows])

    def test_all_horizon_fits_replay_and_changed_center_is_rejected(self):
        daily, iv = fixture()
        features = verify.raw_features(daily, iv)
        targets = {h: verify.future_returns(daily.close, h) for h in verify.HORIZONS}
        section = {"origin_start": "2006-06-01", "origin_end": "2006-07-31", "latest_target": "2007-12-31",
                   "development": ["2006-06-01", "2006-06-30"], "evaluation": ["2006-07-01", "2006-07-31"],
                   "development_target_available_by": "2006-06-30", "minimum_train": 50}
        # Put both months in development-free scoring intervals so both long
        # horizons have rows, while the first month's training remains causal.
        section["development"] = ["2006-06-01", "2006-12-31"]
        section["development_target_available_by"] = "2006-12-31"
        section["evaluation"] = ["2007-01-01", "2007-12-31"]
        records, pieces = [], []
        for horizon, target in targets.items():
            entries, scored, complete = verify.eligible_entries(features, target, section)
            for origin in entries[~entries.to_period("M").duplicated()]:
                mask = verify.training_mask(complete, target, origin, features.index)
                train_dates = features.index[mask]
                application = entries[entries.to_period("M") == origin.to_period("M")]
                query = scored[scored.to_period("M") == origin.to_period("M")]
                training, apply, centers = verify.derived_design(features.loc[mask], features.loc[application])
                cutoff = features.loc[origin, "feature_cutoff_date"]
                last = target.loc[mask, "target_end"].max()
                record = {"horizon": horizon, "fit_origin": str(origin.date()), "fit_cutoff_date": str(cutoff.date()),
                          "train_n": int(mask.sum()), "train_first_origin": str(train_dates[0].date()),
                          "train_last_origin": str(train_dates[-1].date()), "train_last_target": str(last.date()),
                          "train_last_available": str(last.date()), "application_n": len(application),
                          "transform_audit": centers, "model_audit": {}}
                for model in verify.MODELS:
                    columns = ("const",) if model == "mean" else verify.BASE if model == "baseline" else verify.ALL_FEATURES
                    prediction, audit = verify.ridge_prediction(training.loc[:, columns], target.loc[mask, "y"], apply.loc[query, columns])
                    audit["columns"] = columns
                    record["model_audit"][model] = audit
                    rows = target.loc[query].copy()
                    rows["origin"], rows["model"], rows["horizon"], rows["prediction"] = query, model, horizon, prediction
                    rows["feature_cutoff_date"] = features.loc[query, "feature_cutoff_date"]
                    rows["fit_origin"], rows["fit_cutoff_date"] = origin, cutoff
                    rows["train_last_target"], rows["train_last_available"] = last, last
                    rows["train_n"], rows["phase"] = int(mask.sum()), "development"
                    pieces.append(rows.reset_index(drop=True))
                records.append(record)
        panel = pd.concat(pieces, ignore_index=True)
        audit = verify.verify_forecasts(features, targets, panel, records, {"index": section})
        self.assertEqual(audit["monthly_fits_verified"], 4)
        self.assertEqual(audit["forecasts_verified"], len(panel))
        records[0]["transform_audit"]["gap_mean"] += .01
        with self.assertRaises(AssertionError):
            verify.verify_forecasts(features, targets, panel, records, {"index": section})


if __name__ == "__main__":
    unittest.main()
