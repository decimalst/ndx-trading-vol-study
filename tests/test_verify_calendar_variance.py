"""Independent calendar-risk contracts, written before verifier implementation."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import verify_calendar_variance as verify


def market():
    rng = np.random.default_rng(92614)
    dates = pd.bdate_range("2014-01-02", periods=320, name="date")
    close = 100 * np.exp(rng.normal(0, .008, len(dates)).cumsum())
    opening = close * np.exp(rng.normal(0, .004, len(dates)))
    daily = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * 1.003,
                          "low": np.minimum(opening, close) / 1.003, "close": close}, index=dates)
    iv = pd.DataFrame({name: np.exp(rng.normal(3, .2, len(dates))) for name in ("vix", "vix9d", "vvix")}, index=dates)
    return daily, iv


def plan(event, announced="2017-03-10 08:30", planned="2017-04-14 08:30"):
    return {"event_type": event, "announced_at": pd.Timestamp(announced, tz="America/New_York").isoformat(),
            "planned_at": pd.Timestamp(planned, tz="America/New_York").isoformat(),
            "source_id": event + announced, "source_sha256": "a" * 64}


def annual():
    return [{"year": 2017, "announced_date": "2016-06-28", "source_id": "annual2017", "source_sha256": "b" * 64,
             "final_dates": ["2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
                             "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13"]}]


def section():
    return {"origin_start": "2014-12-01", "origin_end": "2015-01-30", "latest_target": "2015-02-02",
            "development": ["2014-12-01", "2014-12-31"], "evaluation": ["2015-01-01", "2015-01-30"],
            "development_target_available_by": "2014-12-31", "evaluation_stability": [["2015-01-01", "2015-01-15"], ["2015-01-16", "2015-01-30"]]}


class RawRiskTests(unittest.TestCase):
    def test_full_session_target_is_next_intraday_gk_plus_raw_overnight(self):
        daily, iv = market()
        features, targets = verify.market_tables(daily, iv)
        p = 201
        d = daily.iloc[p]
        gk = max(.5 * np.log(d.high / d.low) ** 2 - (2 * np.log(2) - 1) * np.log(d.close / d.open) ** 2, 1e-10)
        self.assertAlmostEqual(targets.y.iloc[p-1], gk + np.log(d.open / daily.close.iloc[p-1]) ** 2, places=15)
        self.assertEqual(targets.target_end.iloc[p-1], daily.index[p])
        self.assertEqual(targets.available_date.iloc[p-1], daily.index[p])
        self.assertEqual(features.feature_cutoff_date.iloc[p], daily.index[p-1])
        self.assertTrue(np.isnan(targets.y.iloc[-1]))

    def test_gk_floor_is_historical_measurement_rule_and_flat_target_is_positive(self):
        daily, iv = market()
        daily.loc[:, :] = 100.
        _, target = verify.market_tables(daily, iv)
        np.testing.assert_allclose(target.y.iloc[:-1], 1e-10, rtol=0, atol=0)

    def test_risk_and_negative_return_windows_end_at_previous_close(self):
        daily, iv = market()
        features, _ = verify.market_tables(daily, iv)
        var = (.5 * np.log(daily.high / daily.low) ** 2 - (2 * np.log(2) - 1) * np.log(daily.close / daily.open) ** 2).clip(lower=1e-10)
        var += np.log(daily.open / daily.close.shift()) ** 2
        negative = np.maximum(-np.log(daily.close / daily.close.shift()), 0)
        for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
            self.assertAlmostEqual(features[f"lrv_{suffix}"].iloc[200], np.log(var.iloc[200-width:200].mean()), places=13)
            self.assertAlmostEqual(features[f"neg_{suffix}"].iloc[200], negative.iloc[200-width:200].mean(), places=13)
        self.assertAlmostEqual(features.term.iloc[200], np.log(iv.vix9d.iloc[199] / iv.vix.iloc[199]), places=13)

    def test_market_inputs_do_not_observe_entry_or_future_prices(self):
        daily, iv = market()
        before, _ = verify.market_tables(daily, iv)
        altered, updated = daily.copy(), iv.copy()
        altered.iloc[200:] *= 2.
        updated.iloc[200:] *= 3.
        after, _ = verify.market_tables(altered, updated)
        pd.testing.assert_series_equal(before.iloc[200], after.iloc[200])

    def test_missing_market_date_or_value_does_not_forward_fill(self):
        daily, iv = market()
        iv = iv.drop(iv.index[199])
        daily.loc[daily.index[190], "high"] = np.nan
        features, target = verify.market_tables(daily, iv)
        self.assertTrue(features.loc[daily.index[200], ["lvix", "term", "lvvix"]].isna().all())
        self.assertTrue(np.isnan(features.lrv_m.iloc[200]))
        self.assertTrue(np.isnan(target.y.iloc[189]))

    def test_malformed_ohlc_is_rejected(self):
        daily, iv = market()
        daily.loc[daily.index[199], "high"] = daily.low.iloc[199] / 2
        with self.assertRaises(ValueError):
            verify.market_tables(daily, iv)


class OriginalPlanTests(unittest.TestCase):
    def test_full_session_window_ends_at_next_weekday_sixteen_with_dst(self):
        for date, end, hours in (("2017-04-13", "2017-04-14T16:00:00-04:00", 24),
                                 ("2017-03-10", "2017-03-13T16:00:00-04:00", 71),
                                 ("2017-11-03", "2017-11-06T16:00:00-05:00", 73),
                                 ("2017-01-13", "2017-01-16T16:00:00-05:00", 72)):
            start, ending, elapsed = verify.nominal_window(date)
            self.assertEqual(start.hour, 16)
            self.assertEqual(ending.isoformat(), end)
            self.assertEqual(elapsed, hours)

    def test_endpoint_inclusion_and_open_left_boundary(self):
        origins = pd.DatetimeIndex(["2017-04-13"])
        cutoff = pd.Series(pd.Timestamp("2017-04-12"), index=origins)
        for date, expected in (("2017-04-13 16:00", 0.), ("2017-04-14 16:00", 1.), ("2017-04-14 16:01", 0.)):
            records = [plan("cpi", planned=date), plan("nfp")]
            self.assertEqual(verify.calendar_features(origins, cutoff, records, annual()).cpi_plan.iloc[0], expected)

    def test_good_friday_plan_remains_despite_future_nontrading_day(self):
        origins = pd.DatetimeIndex(["2017-04-13"])
        cutoff = pd.Series(pd.Timestamp("2017-04-12"), index=origins)
        result = verify.calendar_features(origins, cutoff, [plan("cpi"), plan("nfp")], annual())
        self.assertEqual(result.cpi_plan.iloc[0], 1.)
        self.assertEqual(result.nfp_plan.iloc[0], 1.)
        self.assertEqual(result.nominal_hours.iloc[0], 24.)

    def test_publication_on_cutoff_date_is_unknown_and_cancellation_does_not_change_plan(self):
        origins = pd.DatetimeIndex(["2017-04-13"])
        cutoff = pd.Series(pd.Timestamp("2017-04-12"), index=origins)
        record = plan("cpi", announced="2017-04-12 08:30")
        output = verify.calendar_features(origins, cutoff, [record, plan("nfp")], annual())
        self.assertTrue(np.isnan(output.cpi_plan.iloc[0]))
        records = [plan("cpi"), plan("nfp")]
        records[0]["canceled_after_publication"] = True
        self.assertEqual(verify.calendar_features(origins, cutoff, records, annual()).cpi_plan.iloc[0], 1.)

    def test_month_boundary_requires_both_months_even_when_release_is_not_in_window(self):
        origins = pd.DatetimeIndex(["2017-04-28"])
        cutoff = pd.Series(pd.Timestamp("2017-04-27"), index=origins)
        records = [plan("cpi"), plan("nfp")]
        missing = verify.calendar_features(origins, cutoff, records, annual())
        self.assertTrue(missing[["cpi_plan", "nfp_plan"]].isna().all(axis=None))
        records += [plan(event, announced="2017-04-14 08:30", planned="2017-05-12 08:30") for event in ("cpi", "nfp")]
        self.assertTrue(verify.calendar_features(origins, cutoff, records, annual())[["cpi_plan", "nfp_plan"]].eq(0).all(axis=None))

    def test_fomc_requires_exact_original_eight_and_uses_date_without_decision_clock(self):
        origins = pd.DatetimeIndex(["2017-03-14"])
        cutoff = pd.Series(pd.Timestamp("2017-03-13"), index=origins)
        records = [plan(event, announced="2017-02-10 08:30", planned="2017-03-10 08:30") for event in ("cpi", "nfp")]
        self.assertEqual(verify.calendar_features(origins, cutoff, records, annual()).fomc_plan.iloc[0], 1.)
        broken = annual()
        broken[0]["final_dates"] = broken[0]["final_dates"][:-1]
        with self.assertRaises(ValueError):
            verify.calendar_features(origins, cutoff, records, broken)


class FitAndScoreTests(unittest.TestCase):
    def test_complete_synthetic_two_month_forecast_replay(self):
        from src.calendar_variance_models import forecast_panel
        daily, iv = market()
        features, targets = verify.market_tables(daily, iv)
        rng = np.random.default_rng(681)
        features["nominal_hours"] = [verify.nominal_window(date)[2] for date in features.index]
        for name in verify.ADDITIONS:
            features[name] = rng.binomial(1, .18, len(features)).astype(float)
        config = {**section(), "minimum_train": 150, "models": list(verify.MODELS), "baseline": list(verify.BASE)}
        forecasts, fits = forecast_panel(features, targets, config)
        result = verify.verify_forecasts(features, targets, forecasts, fits, {"index": config})
        self.assertEqual(result["monthly_fits_verified"], 2)
        self.assertEqual(result["models_verified"], 3)
        broken = forecasts.copy()
        broken.loc[0, "train_last_available"] += pd.Timedelta(days=1)
        with self.assertRaises(AssertionError):
            verify.verify_forecasts(features, targets, broken, fits, {"index": config})

    def test_missing_first_query_label_does_not_move_monthly_application_origin(self):
        daily, iv = market()
        features, targets = verify.market_tables(daily, iv)
        features[["nominal_hours", "cpi_plan", "nfp_plan", "fomc_plan"]] = 1.
        before, scored, _ = verify.eligible_entries(features, targets, section())
        targets.loc[before[0], "y"] = np.nan
        after, changed, _ = verify.eligible_entries(features, targets, section())
        self.assertTrue(before.equals(after))
        self.assertEqual(len(changed), len(scored)-1)

    def test_training_maturity_stops_at_previous_observed_session(self):
        daily, iv = market()
        _, targets = verify.market_tables(daily, iv)
        complete = pd.Series(True, index=daily.index)
        selected = verify.training_mask(complete, targets, daily.index[200], daily.index)
        self.assertTrue(selected.iloc[198])
        self.assertFalse(selected.iloc[199])
        self.assertFalse(selected.iloc[200])

    def test_development_label_cannot_cross_development_end(self):
        daily, iv = market()
        features, target = verify.market_tables(daily, iv)
        features[["nominal_hours", "cpi_plan", "nfp_plan", "fomc_plan"]] = 1.
        application, scored, _ = verify.eligible_entries(features, target, section())
        self.assertIn(pd.Timestamp("2014-12-31"), application)
        self.assertNotIn(pd.Timestamp("2014-12-31"), scored)

    def test_independent_bfgs_preserves_meanloss_penalty_under_duplicate_rows(self):
        rng = np.random.default_rng(19)
        x = np.c_[np.ones(350), rng.normal(size=(350, 3))]
        y = np.exp(.4 * x[:, 1] + rng.normal(size=350))
        a = x[:5].copy()
        first, audit = verify.second_moment_prediction(x, y, a)
        second, repeated = verify.second_moment_prediction(np.tile(x, (2, 1)), np.tile(y, 2), a)
        np.testing.assert_allclose(first, second, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(audit["beta"], repeated["beta"], atol=1e-8)
        a[0, 1] = 999.
        _, changed = verify.second_moment_prediction(x, y, a[1:])
        np.testing.assert_array_equal(audit["means"], changed["means"])

    def test_all_positive_model_audits_replay_and_bad_coefficient_is_rejected(self):
        from src.macro_second_moment import fit_second_moment
        rng = np.random.default_rng(173)
        training = pd.DataFrame(rng.normal(size=(280, 3)), columns=["a", "b", "c"])
        y = np.exp(.3 * training.a.to_numpy() + rng.normal(size=280))
        application = training.iloc[:8].copy()
        audit = fit_second_moment(training, y, application)
        audit["columns"] = ["const", *training.columns]
        prediction, _ = verify.verify_model(np.c_[np.ones(len(training)), training], y,
                                            np.c_[np.ones(len(application)), application], audit, "baseline")
        self.assertTrue(np.isfinite(prediction).all())
        broken = deepcopy(audit)
        broken["beta"][1] += .01
        with self.assertRaises(AssertionError):
            verify.verify_model(np.c_[np.ones(len(training)), training], y,
                                np.c_[np.ones(len(application)), application], broken, "baseline")

    def test_proper_score_and_fixed_absolute_effect_gate(self):
        np.testing.assert_allclose(verify.proper_score([0., 2.], [1., 2.]), [0., np.log(2.) + 1.])
        phases = [{"delta": -.005, "stability": []}, {"delta": -.006, "stability": [{"delta": -.001}, {"delta": -.002}]}]
        self.assertTrue(verify.effect_passes(phases))
        phases[0]["delta"] = -.004999
        self.assertFalse(verify.effect_passes(phases))

    def test_joint_calendar_model_and_two_hypotheses_have_exact_family(self):
        self.assertEqual(verify.MODELS, ("mean", "baseline", "calendar"))
        self.assertEqual(len(verify.BASE), 15)
        self.assertEqual(len(verify.ALL_FEATURES), 18)
        self.assertEqual(verify.COMPARISONS, (("calendar", "baseline"), ("calendar", "mean")))
        self.assertEqual(verify.WAVE_ALPHA, .05 / (8 * 9))

    def test_fixed_phase_inference_seed_rule_and_absolute_qlike(self):
        dates = pd.bdate_range("2017-01-02", periods=150)
        rng = np.random.default_rng(893)
        target = np.exp(rng.normal(size=len(dates)))
        panel = pd.concat([pd.DataFrame({"origin": dates, "model": model, "y": target,
                                        "prediction": np.exp(rng.normal(0, .2, len(dates))),
                                        "available_date": dates + pd.offsets.BDay()}) for model in verify.MODELS])
        protocol = {"index": {"development": ["2017-01-01", "2017-12-31"], "development_target_available_by": "2017-12-31"},
                    "inference": {"blocks": [21, 63, 126], "seed": 20260914, "bootstrap_draws": 99}}
        result = verify.phase_statistics(panel, "baseline", "development", 0, protocol)
        with patch.object(verify, "explicit_bootstrap_means", wraps=verify.explicit_bootstrap_means) as sampled:
            again = verify.phase_statistics(panel, "baseline", "development", 0, protocol)
        self.assertEqual(result, again)
        self.assertEqual([call.args[3] for call in sampled.call_args_list], [20260935, 20260977, 20261040])
        self.assertEqual(result["n"], 150)
        self.assertEqual(result["p_conservative"], max(result["hac126"]["p"], *(value["p"] for value in result["block_inference"].values())))


class FrozenPublicationTests(unittest.TestCase):
    def test_protocol_numerical_and_protected_fences_are_explicitly_validated(self):
        protocol = yaml.safe_load((verify.ROOT / "calendar_variance.yaml").read_text())
        verify.validate_protocol(protocol)
        for section_name, key, value in (("index", "minimum_train", 999), ("index", "source_end", "2025-11-03"),
                                         ("inference", "bootstrap_draws", 9999), ("verification", "forecast_relative_tolerance", 1e-4)):
            changed = deepcopy(protocol)
            changed[section_name][key] = value
            with self.assertRaises(AssertionError):
                verify.validate_protocol(changed)

    def test_failure_guard_retains_both_rows_and_invalidates_published_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "reports/calendar_variance"
            report.mkdir(parents=True)
            original = {"leads": ["calendar"], "rows": [{"candidate": "calendar", "p_conservative": .00001}], "protocol_sha256": "a" * 64}
            (report / "metrics.json").write_text(json.dumps(original))
            (report / "results.md").write_text("Passing candidates: calendar")
            (report / "trial_ledger.jsonl").write_text('{"event":"registered"}\n')
            with (patch.object(verify, "verify", side_effect=AssertionError("synthetic forecast mismatch")),
                  self.assertRaisesRegex(AssertionError, "synthetic forecast mismatch")):
                verify.verify_with_failure_guard(root)
            metrics = json.loads((report / "metrics.json").read_text())
            self.assertEqual(metrics["status"], "UNEVALUABLE")
            self.assertEqual(metrics["leads"], [])
            self.assertEqual(len(metrics["rows"]), 2)
            for row in metrics["rows"]:
                self.assertEqual(row["status"], "INVALID_RUN")
                self.assertEqual([row[name] for name in ("p_conservative", "p_holm_wave", "p_holm_cumulative")], [1., 1., 1.])
            self.assertEqual(json.loads((report / "failure.json").read_text()), metrics)
            self.assertEqual(json.loads((report / "verification.json").read_text())["status"], "FAILED")
            saved = json.loads((report / "unpublished_scored_metrics.json").read_text())
            self.assertEqual(saved["scored_metrics"], original)
            self.assertEqual(saved["status"], "UNPUBLISHED_DIAGNOSTIC_ONLY")
            self.assertIn("UNEVALUABLE", (report / "results.md").read_text())
            ledger = [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["event"] == "verification_failed" for row in ledger), 2)

    def test_unreadable_canonical_metrics_still_invalidates_human_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "reports/calendar_variance"
            report.mkdir(parents=True)
            (report / "metrics.json").write_text("{truncated metrics")
            (report / "results.md").write_text("Passing candidates: calendar")
            with (patch.object(verify, "verify", side_effect=AssertionError("invalid metric artifact")),
                  self.assertRaisesRegex(AssertionError, "invalid metric artifact")):
                verify.verify_with_failure_guard(root)
            self.assertEqual(json.loads((report / "metrics.json").read_text())["status"], "UNEVALUABLE")
            self.assertIn("UNEVALUABLE", (report / "results.md").read_text())
            self.assertEqual((report / "unpublished_invalid_metrics.txt").read_text(), "{truncated metrics")

    def test_manifest_cannot_silently_omit_prior_python_or_captured_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reports/tail_shape").mkdir(parents=True)
            (root / "data/source_discovery/macro_plans").mkdir(parents=True)
            (root / "reports/tail_shape/manifest.json").write_text(json.dumps({"code": {"src/frozen.py": "f" * 64}}))
            (root / "data/source_discovery/macro_plans/capture.txt").write_text("exact saved extraction")
            protocol = {"sources": {"daily": "daily.parquet", "vix": "vix.csv", "vix9d": "vix9d.csv", "vvix": "vvix.csv"}}
            with self.assertRaisesRegex(AssertionError, "prior|Python|corpus|manifest"):
                verify.verify_manifest_coverage(root, protocol, {"code": {}, "inputs": {}})


if __name__ == "__main__":
    unittest.main()
