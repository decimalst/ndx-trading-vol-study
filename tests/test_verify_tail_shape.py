"""Independent signed-tail contracts written before empirical fitting."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import t

from src import verify_tail_shape as verify


def fixture():
    rng = np.random.default_rng(20260913)
    dates = pd.bdate_range("2005-01-03", periods=500)
    close = 100 * np.exp(rng.normal(.0001, .01, len(dates)).cumsum())
    opening = close * np.exp(rng.normal(0, .005, len(dates)))
    daily = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * 1.004,
                          "low": np.minimum(opening, close) / 1.004, "close": close}, index=dates)
    iv = pd.DataFrame({name: np.exp(rng.normal(3., .2, len(dates))) for name in ("vix", "vix9d", "vvix")}, index=dates)
    iv["skew"] = 100 + rng.uniform(0, 50, len(dates))
    return daily, iv


class IndependentTailSourceTests(unittest.TestCase):
    def test_source_normalization_is_prior22_and_not_annualized(self):
        daily, iv = fixture()
        f, targets = verify.feature_target_tables(daily, iv)
        r = np.log(daily.close / daily.close.shift())
        gk = np.maximum(.5 * np.log(daily.high / daily.low)**2
                        - (2 * np.log(2) - 1) * np.log(daily.close / daily.open)**2, 1e-10)
        variance = gk + np.log(daily.open / daily.close.shift())**2
        mean = r.iloc[178:200].mean()
        scale = np.sqrt(variance.iloc[178:200].mean())
        self.assertAlmostEqual(f.normalization_mean.iloc[200], mean, places=14)
        self.assertAlmostEqual(f.normalization_scale.iloc[200], scale, places=14)
        self.assertAlmostEqual(targets.y.iloc[200], (np.log(daily.close.iloc[201] / daily.close.iloc[200]) - mean) / scale, places=13)
        self.assertEqual(targets.available_date.iloc[200], daily.index[201])

    def test_all_controls_and_negative_returns_are_lagged_one_session(self):
        daily, iv = fixture()
        f, _ = verify.feature_target_tables(daily, iv)
        negatives = np.maximum(-np.log(daily.close / daily.close.shift()), 0)
        for label, width in (("d", 1), ("w", 5), ("m", 22)):
            self.assertAlmostEqual(f[f"neg_{label}"].iloc[200], negatives.iloc[200-width:200].mean(), places=14)
        self.assertEqual(f["skew"].iloc[200], iv["skew"].iloc[199])
        old = f.iloc[200].copy()
        daily.iloc[200:] *= 1.2
        iv.iloc[200:] *= 2
        changed, _ = verify.feature_target_tables(daily, iv)
        pd.testing.assert_series_equal(old, changed.iloc[200])

    def test_event_threshold_is_strict_and_missing_target_not_a_nonevent(self):
        values = pd.Series([-2., -1.5, -1., 0., np.nan])
        event = verify.event_indicator(values)
        np.testing.assert_array_equal(event.iloc[:4], [1., 0., 0., 0.])
        self.assertTrue(np.isnan(event.iloc[-1]))

    def test_missing_price_and_skew_dates_remain_in_complete_calendar(self):
        daily, iv = fixture()
        daily.loc[daily.index[140], :] = np.nan
        iv = iv.drop(daily.index[150])
        f, target = verify.feature_target_tables(daily, iv)
        self.assertTrue(np.isnan(f.normalization_scale.iloc[162]))
        self.assertTrue(np.isnan(f["skew"].iloc[151]))
        self.assertTrue(np.isnan(target.event.iloc[139]))
        self.assertTrue(np.isnan(target.event.iloc[-1]))

    def test_three_curvatures_centered_on_train_and_skew_enters_shared_moments(self):
        daily, iv = fixture()
        f, _ = verify.feature_target_tables(daily, iv)
        train, application = f.iloc[100:300], f.iloc[300:310]
        x, a, audit = verify.derived_design(train, application)
        self.assertEqual(tuple(x), verify.BASE)
        self.assertEqual(len(verify.BASE), 22)
        for field in ("I", "R", "skew"):
            np.testing.assert_allclose(x[field + "_square"], (train[field] - train[field].mean())**2)
            np.testing.assert_allclose(a[field + "_square"], (application[field] - train[field].mean())**2)
        changed = application.copy()
        changed["skew"] *= 20
        _, _, revised = verify.derived_design(train, changed)
        self.assertEqual(audit, revised)

    def test_exact_historical_source_anchor_and_date_first_parsing(self):
        raw = b"DATE,SKEW\n08/13/2018,159.03\n10/20/2025,140.2\n10/21/2025,NOT_A_NUMBER\n"
        s = verify.parse_skew(raw)
        self.assertEqual(len(s), 2)
        self.assertEqual(s.iloc[0], 159.03)
        with self.assertRaises(ValueError):
            verify.parse_skew(raw.replace(b"159.03", b"149.03"))
        with self.assertRaises(ValueError):
            verify.parse_skew(raw.replace(b"DATE,SKEW", b"DATE,SKEW,CLOSE"))

    def test_source_order_is_sorted_and_missing_skew_value_preserved(self):
        raw = b"DATE,SKEW\n10/20/2025,NaN\n08/13/2018,159.03\n10/21/2025,NOT_NUMERIC\n"
        result = verify.parse_skew(raw)
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertTrue(np.isnan(result.iloc[-1]))

    def test_maturity_and_future_query_labels_do_not_move_monthly_entries(self):
        daily, iv = fixture()
        f, target = verify.feature_target_tables(daily, iv)
        complete = pd.Series(True, index=daily.index)
        mask = verify.training_mask(complete, target, daily.index[250], daily.index)
        self.assertTrue(mask.iloc[248])
        self.assertFalse(mask.iloc[249])
        section = {"origin_start": "2006-01-01", "origin_end": "2006-06-30", "latest_target": "2006-12-31",
                   "development": ["2006-01-01", "2006-03-31"], "evaluation": ["2006-04-01", "2006-06-30"],
                   "development_target_available_by": "2006-03-31"}
        entries, scored, _ = verify.eligible_entries(f, target, section)
        target.loc[entries[0], ["y", "event"]] = np.nan
        after, updated, _ = verify.eligible_entries(f, target, section)
        self.assertTrue(entries.equals(after))
        self.assertEqual(len(scored) - len(updated), 1)


class IndependentTailDensityTests(unittest.TestCase):
    def test_symmetric_limit_is_unit_variance_student_not_ordinary_student(self):
        values = np.array([-8., -1.5, 0., 1., 10.])
        expected = t(df=8, scale=np.sqrt(6 / 8))
        np.testing.assert_allclose(verify.skew_logpdf(values, 0), expected.logpdf(values), atol=1e-13)
        np.testing.assert_allclose(verify.skew_cdf(values, 0), expected.cdf(values), atol=1e-13)

    def test_density_integrates_to_one_and_has_zero_mean_unit_variance(self):
        for lam in (-.949, -.6, 0., .6, .949):
            a, b = verify.skew_constants(lam)
            join = -a / b
            for moment, expected in ((0, 1.), (1, 0.), (2, 1.)):
                fn = lambda z, moment=moment, lam=lam: z**moment * np.exp(verify.skew_logpdf(z, lam))
                actual = quad(fn, -np.inf, join, epsabs=1e-10)[0] + quad(fn, join, np.inf, epsabs=1e-10)[0]
                self.assertAlmostEqual(actual, expected, places=8)

    def test_cdf_agrees_with_quadrature_on_both_sides_and_near_extreme_shape(self):
        for lam in (-.94, -.4, .4, .94):
            a, b = verify.skew_constants(lam)
            join = -a / b
            for z in (-4., -1.5, 0., 3.):
                fn = lambda x, lam=lam: np.exp(verify.skew_logpdf(x, lam))
                actual = quad(fn, -np.inf, min(z, join), epsabs=1e-10)[0]
                if z > join:
                    actual += quad(fn, join, z, epsabs=1e-10)[0]
                self.assertAlmostEqual(verify.skew_cdf(z, lam), actual, places=8)

    def test_cdf_continuity_and_density_derivative_at_piecewise_join(self):
        for lam in (-.8, .8):
            a, b = verify.skew_constants(lam)
            join, step = -a / b, 1e-6
            self.assertAlmostEqual(verify.skew_cdf(join, lam), (1 - lam) / 2, places=13)
            numerical = (verify.skew_cdf(join + step, lam) - verify.skew_cdf(join - step, lam)) / (2 * step)
            self.assertAlmostEqual(numerical, np.exp(verify.skew_logpdf(join, lam)), places=7)

    def test_tail_probability_uses_shared_location_and_variance(self):
        mu, variance, lam = np.array([.2, -.3]), np.array([.5, 2.]), np.array([-.6, .6])
        probability = verify.tail_probability(mu, variance, lam)
        np.testing.assert_allclose(probability, verify.skew_cdf((-1.5 - mu) / np.sqrt(variance), lam), atol=1e-14)
        self.assertFalse(np.allclose(probability, verify.skew_cdf(-1.5, lam)))

    def test_shape_objective_has_only_slope_penalty_and_sample_mean_scaling(self):
        z, s = np.linspace(-2, 2, 100), np.linspace(-1, 1, 100)
        theta = np.array([.2, -.4])
        value, gradient = verify.shape_objective(theta, z, s)
        expected = -verify.skew_logpdf(z, .95 * np.tanh(theta[0] + theta[1] * s)).mean() + .01 * theta[1]**2
        self.assertAlmostEqual(value, expected, places=13)
        repeated, repeated_gradient = verify.shape_objective(theta, np.tile(z, 3), np.tile(s, 3))
        self.assertAlmostEqual(value, repeated, places=13)
        np.testing.assert_allclose(gradient, repeated_gradient, atol=1e-9)
        for index in range(2):
            step = np.eye(2)[index] * 1e-5
            numerical = (verify.shape_objective(theta + step, z, s)[0] - verify.shape_objective(theta - step, z, s)[0]) / 2e-5
            self.assertAlmostEqual(gradient[index], numerical, places=7)

    def test_projected_kkt_respects_both_bounds_and_not_just_unconstrained_gradient(self):
        gradient = np.array([.2, -.3])
        np.testing.assert_array_equal(verify.projected_gradient(np.array([-3., 3.]), gradient), [0., 0.])
        np.testing.assert_array_equal(verify.projected_gradient(np.array([3., -3.]), gradient), gradient)

    def test_independent_shape_fit_fixed_starts_and_finite_probabilities(self):
        rng = np.random.default_rng(1314)
        z, s = rng.standard_t(8, 600) * np.sqrt(6/8), rng.normal(size=600)
        control = verify.fit_shape(z, None)
        candidate = verify.fit_shape(z, s, constant=control["theta"][0])
        np.testing.assert_array_equal(control["start"], [0.])
        np.testing.assert_array_equal(candidate["start"], [control["theta"][0], 0.])
        self.assertLessEqual(candidate["projected_gradient_max_abs"], 2e-7)
        probability = verify.tail_probability(np.zeros(5), np.ones(5), .95*np.tanh(candidate["theta"][0]+candidate["theta"][1]*s[:5]))
        self.assertTrue(((probability > 0) & (probability < 1)).all())

    def test_event_support_is_required_before_models_or_phase_comparisons(self):
        self.assertEqual(verify.event_support(np.r_[np.ones(50), np.zeros(50)], 50), {"events": 50, "nonevents": 50})
        for values in (np.r_[np.ones(49), np.zeros(100)], np.r_[np.ones(100), np.zeros(49)], np.array([0, 1, np.nan])):
            with self.assertRaises(ValueError):
                verify.event_support(values, 50)

    def test_zero_residual_second_moments_remain_valid_and_shared_variance_positive(self):
        rng = np.random.default_rng(919)
        x = np.c_[np.ones(80), rng.normal(size=(80, 2))]
        y = .1 + .3*x[:, 1] + rng.normal(0, .1, 80)
        mean, _ = verify.ridge_prediction(x, y, x)
        residuals = (y - mean)**2
        residuals[0] = 0.
        variance, audit = verify.second_moment_prediction(x, residuals, x[-4:])
        self.assertTrue((variance > 0).all())
        self.assertLessEqual(audit["gradient_max_abs"], 2e-8)

    def test_saved_shared_moments_replay_and_omitted_skew_curvature_is_rejected(self):
        daily, iv = fixture()
        f, _ = verify.feature_target_tables(daily, iv)
        train, apply, _ = verify.derived_design(f.iloc[100:300], f.iloc[300:304])
        rng = np.random.default_rng(578)
        y = rng.standard_t(8, len(train)) * np.sqrt(6/8)
        train_mu, mean = verify.ridge_prediction(train, y, train)
        _, variance = verify.second_moment_prediction(train, (y-train_mu)**2, pd.concat([train, apply]))
        mean["columns"] = list(verify.BASE)
        variance.update({"columns": list(verify.BASE), "alpha": .01, "iterations": 1, "backtracks": 0})
        rebuilt_mu, rebuilt_h, _ = verify.verify_moments(train, y, apply, {"mean": mean, "variance": variance})
        np.testing.assert_allclose(rebuilt_mu[:len(train)], train_mu, atol=1e-12)
        self.assertTrue((rebuilt_h > 0).all())
        variance["columns"] = list(verify.BASE[:-1])
        with self.assertRaises(AssertionError):
            verify.verify_moments(train, y, apply, {"mean": mean, "variance": variance})

    def test_saved_shapes_replay_and_invented_start_or_nonconvergence_rejected(self):
        rng = np.random.default_rng(721)
        z, s = rng.standard_t(8, 400) * np.sqrt(6/8), rng.normal(size=400)
        baseline = verify.fit_shape(z, None)
        candidate = verify.fit_shape(z, s, constant=baseline["theta"][0])
        audits = {"constant_shape": baseline, "skew_shape": candidate}
        for model, audit in audits.items():
            audit.update({"nu": 8., "penalty": 0. if model == "constant_shape" else .01,
                          "bounds": [[-3., 3.]] * len(audit["theta"])})
        _, maximum = verify.verify_shapes(z, s, audits)
        self.assertLessEqual(maximum, 1.1e-7)
        candidate["success"] = False
        with self.assertRaises(AssertionError):
            verify.verify_shapes(z, s, audits)
        candidate["success"] = True
        candidate["start"] = [0., 1.]
        with self.assertRaises(AssertionError):
            verify.verify_shapes(z, s, audits)


class IndependentTailInferenceTests(unittest.TestCase):
    def test_complete_monthly_tail_forecast_replay_and_changed_shared_mean_rejected(self):
        daily, iv = fixture()
        f, targets = verify.feature_target_tables(daily, iv)
        rng = np.random.default_rng(2245)
        targets["y"] = np.where(np.arange(len(f)) % 3 == 0, -2.2, rng.normal(.3, .3, len(f)))
        targets["event"] = verify.event_indicator(targets.y)
        targets["raw_return"] = targets.y * f.normalization_scale + f.normalization_mean
        section = {"origin_start": "2006-01-02", "origin_end": "2006-02-28", "latest_target": "2006-03-31",
                   "development": ["2006-01-02", "2006-01-31"], "evaluation": ["2006-02-01", "2006-02-28"],
                   "development_target_available_by": "2006-01-31", "minimum_train": 100}
        entries, scored, complete = verify.eligible_entries(f, targets, section)
        records, pieces = [], []
        for origin in entries[~entries.to_period("M").duplicated()]:
            mask = verify.training_mask(complete, targets, origin, f.index)
            training_dates = f.index[mask]
            application = entries[entries.to_period("M") == origin.to_period("M")]
            query = scored[scored.to_period("M") == origin.to_period("M")]
            tr, ap, centers = verify.derived_design(f.loc[mask], f.loc[application])
            u, n = targets.loc[mask, "y"].to_numpy(), int(mask.sum())
            mu, mean_audit = verify.ridge_prediction(tr, u, pd.concat([tr, ap]))
            h, variance_audit = verify.second_moment_prediction(tr, (u-mu[:n])**2, pd.concat([tr, ap]))
            mean_audit["columns"] = list(verify.BASE)
            variance_audit.update({"columns": list(verify.BASE), "alpha": .01, "iterations": 1, "backtracks": 0})
            skew_mean, skew_scale = tr["skew"].mean(), tr["skew"].std(ddof=0)
            z, s = (u-mu[:n])/np.sqrt(h[:n]), ((tr["skew"]-skew_mean)/skew_scale).to_numpy()
            constant = verify.fit_shape(z, None)
            candidate = verify.fit_shape(z, s, constant=constant["theta"][0])
            shapes = {"constant_shape": constant, "skew_shape": candidate}
            for model, audit in shapes.items():
                audit.update({"nu": 8., "penalty": 0. if model == "constant_shape" else .01,
                              "bounds": [[-3., 3.]] * len(audit["theta"])})
            cutoff = f.loc[origin, "feature_cutoff_date"]
            last = targets.loc[mask, "target_end"].max()
            events = int(targets.loc[mask, "event"].sum())
            record = {"fit_origin": str(origin.date()), "fit_cutoff_date": str(cutoff.date()), "train_n": n,
                      "train_event_count": events, "train_nonevent_count": n-events,
                      "train_first_origin": str(training_dates[0].date()), "train_last_origin": str(training_dates[-1].date()),
                      "train_last_target": str(last.date()), "train_last_available": str(last.date()),
                      "application_n": len(application), "transform_audit": centers, "frequency": events/n,
                      "skew_mean": skew_mean, "skew_scale": skew_scale, "shape_audit": shapes,
                      "moment_audit": {"mean": mean_audit, "variance": variance_audit}}
            for model in verify.MODELS:
                rows = targets.loc[query].copy()
                rows["origin"], rows["model"], rows["horizon"] = query, model, 1
                for field in ("feature_cutoff_date", "normalization_mean", "normalization_scale"):
                    rows[field] = f.loc[query, field]
                rows["fit_origin"], rows["fit_cutoff_date"] = origin, cutoff
                rows["train_last_target"], rows["train_last_available"] = last, last
                rows["train_n"], rows["train_event_count"] = n, events
                rows["phase"] = np.where(query <= section["development"][1], "development", "evaluation")
                if model == "frequency":
                    rows["prediction"] = events/n
                    for field in ("mu", "variance", "lambda", "log_density"):
                        rows[field] = np.nan
                else:
                    positions = application.get_indexer(query)
                    rows["mu"], rows["variance"] = mu[n:][positions], h[n:][positions]
                    theta = shapes[model]["theta"]
                    linear = theta[0] if model == "constant_shape" else theta[0] + theta[1]*(ap.loc[query, "skew"]-skew_mean)/skew_scale
                    rows["lambda"] = .95*np.tanh(linear)
                    rows["prediction"] = verify.tail_probability(rows.mu, rows.variance, rows["lambda"])
                    rows["log_density"] = verify.skew_logpdf((rows.y-rows.mu)/np.sqrt(rows.variance), rows["lambda"])-.5*np.log(rows.variance)
                pieces.append(rows.reset_index(drop=True))
            records.append(record)
        forecasts = pd.concat(pieces, ignore_index=True)
        got = verify.verify_forecasts(f, targets, forecasts, records, {"index": section})
        self.assertEqual(got["forecasts_verified"], len(forecasts))
        self.assertEqual(got["shape_optimizations_verified"], 4)
        forecasts.loc[forecasts.model == "skew_shape", "mu"] += .001
        with self.assertRaisesRegex(AssertionError, "identical"):
            verify.verify_forecasts(f, targets, forecasts, records, {"index": section})

    def test_calibration_bins_include_zero_one_and_keep_empty_bins(self):
        output = verify.calibration(np.array([0., .1, .9, 1.]), np.array([0., 1., 0., 1.]))
        self.assertEqual(len(output["bins"]), 10)
        self.assertEqual([row["n"] for row in output["bins"]], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertIsNone(output["bins"][2]["event_rate"])
        self.assertAlmostEqual(output["brier"], (0 + .9**2 + .9**2 + 0) / 4)

    def test_brier_and_full_density_use_distinct_losses_without_refitting(self):
        dates = pd.bdate_range("2016-01-04", periods=180)
        event = np.r_[np.ones(40), np.zeros(140)]
        panel = pd.concat([pd.DataFrame({"origin": dates, "available_date": dates, "event": event,
                                         "y": np.where(event, -2., .1), "model": model, "prediction": prob,
                                         "log_density": density})
                           for model, prob, density in (("frequency", .2, np.nan), ("constant_shape", .1, -.5), ("skew_shape", .3, -.4))])
        p = {"index": {"development": ["2016-01-01", "2016-12-31"], "development_target_available_by": "2016-12-31"},
             "inference": {"blocks": [21, 63, 126], "bootstrap_draws": 3, "seed": 20260913}}
        with patch.object(verify, "explicit_bootstrap_means", return_value=np.zeros((3, 1))):
            brier = verify.phase_statistics(panel, "constant_shape", "brier", "development", 0, p)
            nll = verify.phase_statistics(panel, "constant_shape", "nll", "development", 0, p)
        self.assertAlmostEqual(brier["delta"], np.mean((event - .3)**2 - (event - .1)**2))
        self.assertAlmostEqual(nll["delta"], -.1)
        self.assertEqual(brier["event_support"]["events"], 40)
        self.assertEqual(brier["event_support"]["nonevents"], 140)

    def test_score_specific_absolute_gates_include_both_stability_slices(self):
        phases = [{"name": "development", "delta": -.0005, "stability": []},
                  {"name": "evaluation", "delta": -.0006, "stability": [{"delta": -.01}, {"delta": -.01}]}]
        self.assertTrue(verify.effect_passes(phases, "brier"))
        self.assertFalse(verify.effect_passes(phases, "nll"))
        phases[1]["stability"][0]["delta"] = 0.
        self.assertFalse(verify.effect_passes(phases, "brier"))


if __name__ == "__main__":
    unittest.main()
