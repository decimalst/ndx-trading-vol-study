"""Independent measurement-memory contracts, written before empirical fitting."""
from __future__ import annotations

import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import verify_measurement_memory as verify


def fixture():
    rng = np.random.default_rng(20260911)
    dates = pd.bdate_range("2010-01-04", periods=300)
    daily = pd.DataFrame({"adj close": 100 * np.exp(rng.normal(0, .01, len(dates)).cumsum())}, index=dates)
    measurement = pd.DataFrame({"qmle": np.exp(rng.normal(-1., .2, len(dates))),
                                "rv5": np.exp(rng.normal(-1., .2, len(dates))),
                                "rv15": np.exp(rng.normal(-1., .2, len(dates))),
                                "ci_width": rng.uniform(0, .03, len(dates))}, index=dates)
    iv = pd.DataFrame({name: np.exp(rng.normal(3., .2, len(dates))) for name in ("vix", "vix9d", "vvix")}, index=dates)
    return daily, measurement, iv


def source(rows):
    return ("SPY\n84398\nSPDR\n" + str(len(rows)) + "\n" + rows[0].split()[1] + "\n"
            + rows[-1].split()[1] + "\n" + "\n".join(rows) + "\n").encode()


class IndependentMeasurementSourceTests(unittest.TestCase):
    def test_exact_field_mapping_and_raw_hash(self):
        raw = source(["x 20251020 .2 3 .01 .3 .4 .5 2 .02 .6 .7"])
        frame, audit = verify.parse_source(raw, expected_sha256=hashlib.sha256(raw).hexdigest())
        np.testing.assert_array_equal(frame.iloc[0], [.2, .3, .4, .01])
        self.assertEqual(audit["source_sha256"], hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError, "hash"):
            verify.parse_source(raw, expected_sha256="f" * 64)

    def test_date_bound_precedes_any_future_numeric_parsing(self):
        raw = source(["x 20251020 .2 3 .01 .3 .4 .5 2 .02 .6 .7", "x 20251021 NOT_NUMERIC"])
        frame, audit = verify.parse_source(raw)
        self.assertEqual(len(frame), 1)
        self.assertEqual(audit["after_cutoff_rows"], 1)
        self.assertFalse(audit["numeric_post_cutoff_values_parsed"])

    def test_identity_order_and_malformed_bounded_source_rejected(self):
        valid = "x 20251020 .2 3 .01 .3 .4 .5 2 .02 .6 .7"
        for raw in (source([valid]).replace(b"SPY", b"QQQ"), source([valid, valid]),
                    source([valid.replace(".2", "broken", 1)])):
            with self.assertRaises(ValueError):
                verify.parse_source(raw)

    def test_raw_zero_nonfinite_and_extremes_are_not_chart_filtered(self):
        raw = source(["x 20251020 4 3 0 NaN inf .5 2 .02 .6 .7"])
        frame, _ = verify.parse_source(raw)
        self.assertEqual(frame.qmle.iloc[0], 4.)
        self.assertEqual(frame.ci_width.iloc[0], 0.)
        self.assertTrue(np.isnan(frame.rv5.iloc[0]))
        self.assertTrue(np.isinf(frame.rv15.iloc[0]))


class IndependentMeasurementTimingTests(unittest.TestCase):
    def test_variance_windows_and_relative_width_interaction(self):
        daily, m, iv = fixture()
        f = verify.features_from_tables(daily, m, iv)
        pos = 150
        for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
            expected = np.log(np.mean(m.qmle.iloc[pos-1-width:pos-1] ** 2))
            self.assertAlmostEqual(f[f"lq_{suffix}"].iloc[pos], expected, places=13)
        u = np.log1p(m.ci_width.iloc[pos-2] / m.qmle.iloc[pos-2])
        self.assertAlmostEqual(f.width.iloc[pos], u, places=14)
        self.assertAlmostEqual(f.quality_memory.iloc[pos], u * (f.lq_d.iloc[pos] - f.lq_m.iloc[pos]), places=14)

    def test_market_controls_are_prior_session_and_exact_negative_return(self):
        daily, m, iv = fixture()
        f = verify.features_from_tables(daily, m, iv)
        neg = np.maximum(-np.log(daily["adj close"]).diff(), 0)
        for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
            self.assertAlmostEqual(f[f"neg_{suffix}"].iloc[150], neg.iloc[150-width:150].mean(), places=14)
        self.assertAlmostEqual(f.livshape.iloc[150], np.log(iv.vix9d.iloc[149] / iv.vix.iloc[149]), places=14)
        self.assertEqual(f.market_cutoff_date.iloc[150], daily.index[149])
        self.assertEqual(f.measurement_cutoff_date.iloc[150], daily.index[148])

    def test_mutating_recent_measurements_and_current_market_cannot_change_features(self):
        daily, m, iv = fixture()
        before = verify.features_from_tables(daily, m, iv)
        m.iloc[149:] *= 10
        daily.iloc[150:] *= 2
        iv.iloc[150:] *= 2
        after = verify.features_from_tables(daily, m, iv)
        np.testing.assert_array_equal(before.loc[daily.index[150], verify.ALL_FEATURES], after.loc[daily.index[150], verify.ALL_FEATURES])

    def test_absent_dates_are_preserved_inside_windows(self):
        daily, m, iv = fixture()
        m = m.drop(daily.index[140])
        f = verify.features_from_tables(daily, m, iv)
        self.assertTrue(np.isnan(f.lq_d.iloc[142]))
        self.assertTrue(np.isnan(f.lq_w.iloc[146]))
        self.assertTrue(np.isnan(f.lq_m.iloc[163]))
        self.assertTrue(np.isfinite(f.lq_m.iloc[164]))

    def test_zero_width_valid_and_negative_width_unknown(self):
        daily, m, iv = fixture()
        m.loc[daily.index[140], "ci_width"] = 0
        m.loc[daily.index[141], "ci_width"] = -1
        f = verify.features_from_tables(daily, m, iv)
        self.assertEqual(f.width.iloc[142], 0.)
        self.assertEqual(f.quality_memory.iloc[142], 0.)
        self.assertTrue(np.isnan(f.width.iloc[143]))

    def test_target_date_and_two_later_reference_sessions_maturity(self):
        daily, m, _ = fixture()
        t = verify.targets_from_table(m, daily.index)
        for name in verify.MEASURES:
            self.assertAlmostEqual(t[f"y_{name}"].iloc[140], m[name].iloc[141] ** 2)
        self.assertEqual(t.target_end.iloc[140], daily.index[141])
        self.assertEqual(t.available_date.iloc[140], daily.index[143])
        self.assertTrue(pd.isna(t.available_date.iloc[-3]))

    def test_unrepresentable_squares_and_nonpositive_native_measurements_are_unknown(self):
        daily, m, iv = fixture()
        for position, value in enumerate([1e200, 1e-200, 0., -1.], 140):
            m.loc[daily.index[position], "qmle"] = value
        target = verify.targets_from_table(m, daily.index)
        f = verify.features_from_tables(daily, m, iv)
        self.assertTrue(target.y_qmle.iloc[139:143].isna().all())
        self.assertTrue(f.lq_d.iloc[142:146].isna().all())

    def test_training_only_complete_primary_and_alternative_labels_mature_by_prior_close(self):
        daily, m, _ = fixture()
        t = verify.targets_from_table(m, daily.index)
        complete = pd.Series(True, index=daily.index)
        t.loc[daily.index[140], "y_rv5"] = np.nan
        mask = verify.training_mask(complete, t, daily.index[150], daily.index)
        self.assertFalse(mask.iloc[140])
        self.assertTrue(mask.iloc[146])
        self.assertFalse(mask.iloc[147])

    def test_future_label_mutations_never_move_monthly_fit(self):
        daily, m, iv = fixture()
        f = verify.features_from_tables(daily, m, iv)
        t = verify.targets_from_table(m, daily.index)
        section = {"origin_start": "2010-03-01", "origin_end": "2011-01-31", "latest_target": "2011-02-28",
                   "development": ["2010-03-01", "2010-12-31"], "evaluation": ["2011-01-01", "2011-01-31"],
                   "development_target_available_by": "2010-12-31"}
        entries, scored, _ = verify.eligible_entries(f, t, section)
        t.loc[entries[0], "y_rv15"] = np.nan
        updated, scored_updated, _ = verify.eligible_entries(f, t, section)
        self.assertTrue(entries.equals(updated))
        self.assertEqual(len(scored) - len(scored_updated), 1)
        self.assertFalse((t.loc[scored[scored.year == 2010], "available_date"] > "2010-12-31").any())

    def test_common_units_scale_targets_but_not_interaction_or_paired_losses(self):
        daily, m, iv = fixture()
        first = verify.features_from_tables(daily, m, iv)
        second = verify.features_from_tables(daily, m * 100, iv)
        np.testing.assert_allclose(first.quality_memory, second.quality_memory, rtol=1e-11, atol=1e-12)
        np.testing.assert_array_equal(second.lq_w.isna(), first.lq_w.isna())
        np.testing.assert_allclose((second.lq_w - first.lq_w).dropna(), np.log(10000), rtol=1e-12)
        q, p, control = np.array([0, .3, .5]), np.array([.2, .3, .4]), np.array([.4, .5, .6])
        np.testing.assert_allclose(verify.proper_score(q, p) - verify.proper_score(q, control),
                                   verify.proper_score(q * 10000, p * 10000) - verify.proper_score(q * 10000, control * 10000))


class IndependentMeasurementModelTests(unittest.TestCase):
    def test_independent_bfgs_mean_optimum_and_zero_target(self):
        y = np.array([0., .2, .4, .6])
        predictions, audit = verify.second_moment_prediction(np.ones((4, 1)), y, np.ones((2, 1)))
        np.testing.assert_allclose(predictions, y.mean(), atol=1e-12)
        self.assertLessEqual(audit["gradient_max_abs"], 2e-8)

    def test_mean_loss_penalty_and_gradient_are_independent_of_sample_count(self):
        x = np.array([[1., -1.], [1., 0.], [1., 1.]])
        y, beta = np.array([.1, .3, 1.]), np.array([.2, .4])
        value, gradient = verify.penalized_objective(beta, x, y)
        expected = np.mean(x @ beta + y * np.exp(-x @ beta)) + .01 * .4**2
        self.assertAlmostEqual(value, expected)
        repeated = verify.penalized_objective(beta, np.tile(x, (4, 1)), np.tile(y, 4))
        np.testing.assert_allclose(gradient, repeated[1])
        for k in range(2):
            perturb = np.eye(2)[k] * 1e-6
            finite = (verify.penalized_objective(beta + perturb, x, y)[0] - verify.penalized_objective(beta - perturb, x, y)[0]) / 2e-6
            self.assertAlmostEqual(gradient[k], finite, places=8)

    def test_primary_fit_is_unchanged_by_alternative_label_values(self):
        rng = np.random.default_rng(94)
        x = np.c_[np.ones(80), rng.normal(size=(80, 2))]
        y = pd.DataFrame({"y_qmle": np.exp(x[:, 1]), "y_rv5": 2., "y_rv15": 3.})
        first = verify.fit_primary(x, y, x[-2:])[0]
        y[["y_rv5", "y_rv15"]] *= 100
        second = verify.fit_primary(x, y, x[-2:])[0]
        np.testing.assert_array_equal(first, second)

    def test_application_values_cannot_change_training_scaling(self):
        rng = np.random.default_rng(48)
        x = np.c_[np.ones(100), rng.normal(size=(100, 2))]
        y = np.exp(.1 * x[:, 1])
        _, before = verify.second_moment_prediction(x, y, x[-2:])
        _, after = verify.second_moment_prediction(x, y, np.array([[1., 9., -9.]]))
        np.testing.assert_array_equal(before["means"], after["means"])
        np.testing.assert_array_equal(before["beta"], after["beta"])

    def test_15_contrasts_and_complete_candidate_gates(self):
        self.assertEqual(len(verify.COMPARISONS), 15)
        self.assertEqual(len(set(verify.COMPARISONS)), 15)
        passing = [{"name": "development", "delta": -.005, "stability": []},
                   {"name": "evaluation", "delta": -.006, "stability": [{"delta": -.001}, {"delta": -.002}]}]
        self.assertTrue(verify.primary_effect(passing))
        passing[0]["delta"] = -.00499
        self.assertFalse(verify.primary_effect(passing))
        self.assertTrue(verify.alternative_effect(passing))
        passing[1]["delta"] = 0.
        self.assertFalse(verify.alternative_effect(passing))

    def test_bootstrap_resolution_at_15_way_wave_allocation(self):
        self.assertLess(1 / (99999 + 1), .1 * (1 / 600) / 15)
        self.assertGreater(1 / (49999 + 1), .1 * (1 / 600) / 15)

    def test_full_saved_fit_replay_rejects_changed_alternative_or_fit_date(self):
        daily, measurement, iv = fixture()
        f = verify.features_from_tables(daily, measurement, iv)
        t = verify.targets_from_table(measurement, daily.index)
        section = {"origin_start": "2010-06-01", "origin_end": "2010-07-30", "latest_target": "2010-08-10",
                   "development": ["2010-06-01", "2010-06-30"], "evaluation": ["2010-07-01", "2010-07-30"],
                   "development_target_available_by": "2010-06-30", "minimum_train": 50}
        entries, scored, complete = verify.eligible_entries(f, t, section)
        fits, rows = [], []
        for origin in entries[~entries.to_period("M").duplicated()]:
            selected = verify.training_mask(complete, t, origin, f.index)
            train_dates = f.index[selected]
            query = scored[scored.to_period("M") == origin.to_period("M")]
            record = {"fit_origin": str(origin.date()), "fit_cutoff_date": str(f.loc[origin, "market_cutoff_date"].date()),
                      "train_n": int(selected.sum()), "train_first_origin": str(train_dates[0].date()),
                      "train_last_origin": str(train_dates[-1].date()),
                      "train_last_target": str(t.loc[selected, "target_end"].max().date()),
                      "train_last_available": str(t.loc[selected, "available_date"].max().date()),
                      "application_n": int((entries.to_period("M") == origin.to_period("M")).sum()), "model_audit": {}}
            for model in verify.MODELS:
                columns = (("const",) if model == "mean" else verify.BASE if model == "baseline"
                           else verify.BASE + ("width",) if model == "width" else verify.ALL_FEATURES)
                prediction, audit = verify.fit_primary(f.loc[selected, columns], t.loc[selected], f.loc[query, columns])
                audit.update({"columns": columns, "alpha": .01, "iterations": 1, "backtracks": 0})
                if model == "mean":
                    audit["gradient_max_abs"] = 0.
                record["model_audit"][model] = audit
                data = t.loc[query].copy()
                data["origin"], data["model"], data["horizon"], data["prediction"] = query, model, 1, prediction
                for column in ("market_cutoff_date", "measurement_cutoff_date"):
                    data[column] = f.loc[query, column]
                for column in ("fit_origin", "fit_cutoff_date", "train_last_target", "train_last_available"):
                    data[column] = pd.Timestamp(record[column])
                data["train_n"] = int(selected.sum())
                data["phase"] = np.where(query <= section["development"][1], "development", "evaluation")
                rows.append(data.reset_index(drop=True))
            fits.append(record)
        panel = pd.concat(rows, ignore_index=True)
        got = verify.verify_forecasts(f, t, panel, fits, {"index": section})
        self.assertEqual(got["forecasts_verified"], len(panel))
        self.assertEqual(got["monthly_fits_verified"], 2)
        bad = panel.copy()
        bad.loc[0, "y_rv15"] *= 2
        with self.assertRaises(AssertionError):
            verify.verify_forecasts(f, t, bad, fits, {"index": section})
        bad = panel.copy()
        bad.loc[0, "fit_origin"] += pd.Timedelta(days=1)
        with self.assertRaises(AssertionError):
            verify.verify_forecasts(f, t, bad, fits, {"index": section})

    def test_alternative_scoring_reuses_predictions_and_centered_null_plus_one(self):
        dates = pd.bdate_range("2016-01-04", periods=150)
        q = np.linspace(.1, .4, len(dates))
        panel = pd.concat([pd.DataFrame({"origin": dates, "available_date": dates, "model": name,
                                         "prediction": value, "y_qmle": q, "y_rv5": q * 2, "y_rv15": q * .5})
                           for name, value in (("quality", .2), ("baseline", .3))])
        original = panel.copy(deep=True)
        protocol = {"index": {"development": ["2016-01-01", "2016-12-31"],
                               "development_target_available_by": "2016-12-31"},
                    "inference": {"blocks": [21], "seed": 20260911, "bootstrap_draws": 3}}
        difference = verify.proper_score(q * 2, np.repeat(.2, len(q))) - verify.proper_score(q * 2, np.repeat(.3, len(q)))
        delta = difference.mean()
        with patch.object(verify, "fit_primary", side_effect=AssertionError("Refitting prohibited")), \
                patch.object(verify, "explicit_bootstrap_means", return_value=np.array([[delta], [delta], [delta + 2 * abs(delta)]])):
            got = verify.phase_statistics(panel, "quality", "baseline", "rv5", "development", 0, protocol)
        self.assertAlmostEqual(got["delta"], delta)
        self.assertEqual(got["block_inference"]["21"]["p"], .5)
        pd.testing.assert_frame_equal(panel, original)


if __name__ == "__main__":
    unittest.main()
