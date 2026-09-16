"""Independent macro-wave synthetic contracts written before empirical fits."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import verify_macro_overnight as verify


def market_fixture():
    rng = np.random.default_rng(91270)
    dates = pd.bdate_range("2010-01-04", periods=350)
    close = 100 * np.exp(rng.normal(0, .008, len(dates)).cumsum())
    opening = close * np.exp(rng.normal(0, .004, len(dates)))
    daily = pd.DataFrame({"open": opening, "close": close,
                          "high": np.maximum(opening, close) * 1.002,
                          "low": np.minimum(opening, close) / 1.002,
                          "adj close": close * .8}, index=dates)
    iv = pd.DataFrame({name: np.exp(rng.normal(3., .2, len(dates)))
                       for name in ("vxn", "vix")}, index=dates)
    return daily, iv


def plan(event, announced="2017-03-10 08:30", planned="2017-04-14 08:30"):
    return {"event_type": event, "announced_at": pd.Timestamp(announced, tz="America/New_York").isoformat(),
            "planned_at": pd.Timestamp(planned, tz="America/New_York").isoformat(),
            "source_id": event + announced, "source_sha256": "a" * 64}


def annual():
    return [{"year": 2017, "announced_date": "2016-06-28",
             "final_dates": ["2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
                             "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13"],
             "source_id": "annual2017", "source_sha256": "b" * 64}]


class IndependentMarketTimingTests(unittest.TestCase):
    def test_squared_target_matches_adjusted_overnight_factor_identity(self):
        daily, _ = market_fixture()
        daily.loc[daily.index[201]:, "adj close"] *= 1.01
        target = verify.second_moment_targets(daily)
        expected_log = np.log(daily.open.iloc[201] / daily.close.iloc[200]) + np.log(1.01)
        self.assertAlmostEqual(target.y.iloc[200], expected_log ** 2, places=13)
        self.assertAlmostEqual(target.adjustment_jump.iloc[200], np.log(1.01), places=13)
        self.assertEqual(target.available_date.iloc[200], daily.index[201])

    def test_zero_target_is_valid_and_final_label_missing(self):
        dates = pd.bdate_range("2017-01-03", periods=3)
        daily = pd.DataFrame({"open": [2., 2., 2.], "close": [2., 2., 2.],
                              "adj close": [1., 1., 1.]}, index=dates)
        target = verify.second_moment_targets(daily)
        np.testing.assert_array_equal(target.y.iloc[:2], [0., 0.])
        self.assertTrue(np.isnan(target.y.iloc[-1]))
        self.assertTrue(pd.isna(target.available_date.iloc[-1]))

    def test_rms_levels_are_root_mean_squares_through_prior_close(self):
        daily, iv = market_fixture()
        features = verify.market_features(daily, iv)
        pos = 200
        daytime = np.log(daily.close / daily.open)
        overnight = np.log(daily["adj close"]).diff() - daytime
        for label, width in (("d", 1), ("w", 5), ("m", 22)):
            self.assertAlmostEqual(features[f"on_rms_{label}"].iloc[pos],
                                   np.sqrt(np.mean(overnight.iloc[pos-width:pos] ** 2)), places=13)
            self.assertAlmostEqual(features[f"day_rms_{label}"].iloc[pos],
                                   np.sqrt(np.mean(daytime.iloc[pos-width:pos] ** 2)), places=13)

    def test_all_market_features_ignore_entry_and_future_values(self):
        daily, iv = market_fixture()
        before = verify.market_features(daily, iv)
        entry = daily.index[200]
        later, future_iv = daily.copy(), iv.copy()
        later.loc[entry:] *= 1.2
        future_iv.loc[entry:] *= 2.
        after = verify.market_features(later, future_iv)
        np.testing.assert_allclose(before.loc[entry], after.loc[entry], rtol=0, atol=0)
        truncated = verify.market_features(daily.loc[:entry], iv.loc[:entry])
        np.testing.assert_array_equal(before.loc[entry], truncated.loc[entry])

    def test_training_requires_label_available_by_previous_session(self):
        daily, _ = market_fixture()
        complete = pd.Series(True, index=daily.index)
        targets = verify.second_moment_targets(daily)
        selected = verify.training_mask(complete, targets, daily.index[200], daily.index)
        self.assertTrue(selected.iloc[198])
        self.assertFalse(selected.iloc[199])
        self.assertFalse(selected.iloc[200])

    def test_monthly_fit_origin_uses_features_even_when_first_label_is_missing(self):
        dates = pd.to_datetime(["2017-01-03", "2017-01-04", "2017-02-01", "2017-02-02"])
        complete = pd.Series([True, True, False, True], index=dates)
        selected = verify.monthly_fit_origins(complete, "2017-01-01", "2017-02-28")
        self.assertEqual(list(selected), [dates[0], dates[3]])

    def test_query_label_mutation_changes_score_rows_but_never_monthly_fit_origins(self):
        dates = pd.to_datetime(["2017-01-03", "2017-01-04", "2017-02-01", "2017-02-02"])
        features = pd.DataFrame(1., index=dates, columns=verify.ALL_FEATURES)
        targets = pd.DataFrame({"y": 0., "adjustment_jump": 0., "available_date": dates + pd.Timedelta(days=1)}, index=dates)
        section = {"origin_start": "2017-01-01", "origin_end": "2017-02-28", "latest_target": "2017-03-01",
                   "development": ["2017-01-01", "2017-01-31"], "evaluation": ["2017-02-01", "2017-02-28"],
                   "development_target_available_by": "2017-01-31"}
        original, scored, _ = verify.eligible_entries(features, targets, section)
        targets.loc[dates[0], "y"] = np.nan
        revised, changed_scores, _ = verify.eligible_entries(features, targets, section)
        self.assertTrue(original.equals(revised))
        self.assertEqual(len(scored) - len(changed_scores), 1)
        self.assertEqual(revised[0], dates[0])

    def test_action_sensitivity_is_inclusive_and_not_a_training_filter(self):
        np.testing.assert_array_equal(verify.measurement_mask([0, 1e-5, -1e-5, 1.1e-5, np.nan]),
                                      [True, True, True, False, False])


class IndependentPlanTimingTests(unittest.TestCase):
    def test_document_timestamp_parser_requires_explicit_year_and_preserves_printed_zone(self):
        header = "Transmission embargoed until 8:30 a.m. (EST) Friday, January 8, 2010 USDL-09-1583"
        planned = "The Employment Situation for January is scheduled to be released on Friday, February 5, 2010, at 8:30 a.m. (EST)."
        self.assertEqual(verify.document_timestamp(header).isoformat(), "2010-01-08T08:30:00-05:00")
        self.assertEqual(verify.document_timestamp(planned).isoformat(), "2010-02-05T08:30:00-05:00")
        self.assertEqual(verify.document_timestamp("Friday, December 7, 2012 at 8:30 a.m. (EDT)").isoformat(),
                         "2012-12-07T08:30:00-04:00")
        for bad in ("released on Friday, February 5, at 8:30 a.m. (EST).",
                    "released on Friday, February 5, 2010, at 8:30 a.m."):
            with self.assertRaises(ValueError):
                verify.document_timestamp(bad)

    def test_next_plan_clause_does_not_inherit_date_from_following_revision_sentence(self):
        evidence = ("The Employment Situation for October is scheduled to be released on Friday, November 8, 2013, "
                    "at 8:30 a.m. (EST). This release was originally scheduled for Friday, November 1, 2013.")
        self.assertEqual(verify.document_timestamp(evidence).isoformat(), "2013-11-08T08:30:00-05:00")

    def test_pdf_tool_line_and_page_markers_are_removed_before_timestamp_parsing(self):
        evidence = "L110@P3: released on Thursday, February L111@P3-4: 17, 2011, at 8:30 a.m. (EST)."
        self.assertEqual(verify.document_timestamp(evidence).isoformat(), "2011-02-17T08:30:00-05:00")

    def test_reissued_source_is_retained_but_never_admitted_as_an_original_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.bls.gov/news.release/archives/cpi_02192010.htm"
            text = f"Release ({url})\nL1: This release was reissued on Friday, July16,2010.\n"
            (root / "source.txt").write_text(text)
            row = {"parse_status": "ORIGINAL_VINTAGE_UNCERTAIN", "source_url": url,
                   "snapshot_path": "source.txt", "snapshot_sha256": verify.digest(root / "source.txt"),
                   "historical_plan_admitted": False, "source_revision_evidence": "This release was reissued on Friday, July16,2010.",
                   "original_plan_timestamp": "2010-03-18T08:30:00-04:00", "raw_provider_bytes_sha256": None}
            (root / "ledger.json").write_text(json.dumps({"records": [row]}))
            output, audit = verify.verify_bls_sources(root, "ledger.json", "cpi",
                {"source_publication_start": "2010-01-01", "source_publication_end": "2025-10-20"})
            self.assertEqual(output, [])
            self.assertEqual(audit["statuses"], {"ORIGINAL_VINTAGE_UNCERTAIN": 1})

    def test_explicit_plan_with_reissue_notice_is_rejected_even_if_metadata_admits_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = "https://www.bls.gov/news.release/archives/empsit_02062015.htm"
            header = "Embargoed until 8:30 a.m. (EST) Friday, February 6, 2015 USDL-15-0152"
            planned = "The Employment Situation for February is scheduled to be released on Friday, March 6, 2015, at 8:30 a.m. (EST)."
            row = {"parse_status": "EXPLICIT_ORIGINAL_PLAN", "source_url": url, "snapshot_path": "source.txt",
                   "historical_plan_admitted": True, "source_document_id": "USDL-15-0152",
                   "source_publication_evidence": header, "original_plan_evidence": planned,
                   "source_publication_timestamp": "2015-02-06T08:30:00-05:00",
                   "original_plan_timestamp": "2015-03-06T08:30:00-05:00", "planned_calendar_month": "2015-03"}
            for notice in ("Prior-month estimates were revised.",
                           "NOTE: This news release was reissued on February 6, 2015, to correct data.",
                           "NOTE: BLS reissued this news release on June 3, 2025."):
                (root / "source.txt").write_text(f"Release ({url})\nL1: {header}\nL2: {notice}\nL3: {planned}\n")
                row["snapshot_sha256"] = verify.digest(root / "source.txt")
                (root / "ledger.json").write_text(json.dumps({"records": [row]}))
                section = {"source_publication_start": "2010-01-01", "source_publication_end": "2025-10-20"}
                if "reissued" in notice:
                    with self.assertRaisesRegex(AssertionError, "reissu|vintage"):
                        verify.verify_bls_sources(root, "ledger.json", "nfp", section)
                else:
                    self.assertEqual(len(verify.verify_bls_sources(root, "ledger.json", "nfp", section)[0]), 1)

    def test_same_release_notice_detection_covers_both_voices_and_preserves_routine_revisions(self):
        for text in ("This news release was reissued on February6.",
                     "This version of the release was reissued to replace a table.",
                     "BLS reissued this news release on September23.",
                     "The release has been re-issued."):
            self.assertTrue(verify.same_release_reissue(text))
        for text in ("The January and February news releases have been reissued with corrected data.",
                     "At that time, the format text of the News Release was updated in August2009.",
                     "Prior-month estimates were revised.",
                     "The Technical Note section of this release has been updated to cover new concepts."):
            self.assertFalse(verify.same_release_reissue(text))

    def test_combined_tool_capture_is_split_by_exact_source_document(self):
        a = "https://www.bls.gov/news.release/archives/cpi_01152010.htm"
        b = "https://www.bls.gov/news.release/archives/cpi_02192010.htm"
        text = f"First release ({a})\nL1: FIRST\nSecond release ({b})\nL1: SECOND\n"
        self.assertIn("FIRST", verify.document_section(text, a))
        self.assertNotIn("SECOND", verify.document_section(text, a))
        with self.assertRaises(ValueError):
            verify.document_section(text, "https://missing.example")
        repeated = text + f"First release ({a})\nL20: SAME_DOCUMENT_LATER_SECTION\n"
        self.assertIn("SAME_DOCUMENT_LATER_SECTION", verify.document_section(repeated, a))
        self.assertNotIn("SECOND", verify.document_section(repeated, a))

    def test_nominal_window_uses_utc_elapsed_hours_across_both_dst_changes(self):
        for origin, expected in (("2016-03-11", 64.5), ("2016-11-04", 66.5), ("2016-11-11", 65.5)):
            _, _, hours = verify.nominal_window(origin)
            self.assertEqual(hours, expected)

    def test_good_friday_and_future_closures_do_not_move_nominal_end(self):
        self.assertEqual(verify.nominal_window("2017-04-13")[1].date().isoformat(), "2017-04-14")
        self.assertEqual(verify.nominal_window("2012-10-26")[1].date().isoformat(), "2012-10-29")
        self.assertEqual(verify.nominal_window("2025-01-17")[1].date().isoformat(), "2025-01-20")

    def test_calendar_sources_are_strictly_earlier_than_cutoff_date(self):
        origins = pd.DatetimeIndex(["2017-04-13"])
        cutoff = pd.Series(pd.to_datetime(["2017-04-12"]), index=origins)
        data = [plan(x, announced="2017-04-12 08:30") for x in ("cpi", "nfp")]
        got = verify.calendar_features(origins, cutoff, data, annual())
        self.assertTrue(np.isnan(got.cpi_plan.iloc[0]))

    def test_missing_month_and_cross_month_coverage_remain_unknown(self):
        origins = pd.DatetimeIndex(["2017-03-31", "2017-05-01"])
        cutoff = pd.Series(pd.to_datetime(["2017-03-30", "2017-04-28"]), index=origins)
        got = verify.calendar_features(origins, cutoff, [plan(x) for x in ("cpi", "nfp")], annual())
        self.assertTrue(got.cpi_plan.isna().all())

    def test_original_plan_unchanged_by_later_cancellation_or_next_actual_session(self):
        origins = pd.DatetimeIndex(["2017-04-13"])
        cutoff = pd.Series(pd.to_datetime(["2017-04-12"]), index=origins)
        data = [dict(plan(x), canceled=True, actual_date="2017-04-21") for x in ("cpi", "nfp")]
        got = verify.calendar_features(origins, cutoff, data, annual())
        self.assertEqual(got.cpi_plan.iloc[0], 1.)
        self.assertEqual(got.nfp_plan.iloc[0], 1.)

    def test_incomplete_annual_fomc_schedule_is_rejected(self):
        bad = annual()
        bad[0]["final_dates"] = bad[0]["final_dates"][:1]
        with self.assertRaisesRegex(ValueError, "eight|8|complete"):
            verify.calendar_features(pd.DatetimeIndex([]), pd.Series(dtype="datetime64[ns]"), [], bad)

    def test_fomc_uses_original_final_date_without_statement_clock(self):
        dates = pd.DatetimeIndex(["2017-03-14"])
        cutoff = pd.Series(pd.to_datetime(["2017-03-13"]), index=dates)
        got = verify.calendar_features(dates, cutoff, [], annual())
        self.assertEqual(got.fomc_plan.iloc[0], 1.)


class IndependentEstimationAndInferenceTests(unittest.TestCase):
    def test_zero_compatible_score_and_paired_units_invariance(self):
        y, first, second = np.array([0., .1, 2.]), np.array([.2, .3, .8]), np.array([.3, .4, .9])
        gap = verify.proper_score(y, first) - verify.proper_score(y, second)
        np.testing.assert_allclose(gap, verify.proper_score(y * 100, first * 100) - verify.proper_score(y * 100, second * 100))
        self.assertEqual(verify.proper_score([0.], [1.])[0], 0.)
        for ybad, hbad in (([-1.], [1.]), ([0.], [0.]), ([np.nan], [1.])):
            with self.assertRaises(ValueError):
                verify.proper_score(ybad, hbad)

    def test_penalized_objective_gradient_and_intercept_penalty(self):
        z = np.c_[np.ones(5), np.arange(5.) - 2]
        q, beta = np.array([0., .2, 2., 1., 4.]), np.array([.1, .2])
        value, gradient = verify.penalized_objective(beta, z, q, .01)
        unpenalized, other_gradient = verify.penalized_objective(beta, z, q, 0.)
        self.assertAlmostEqual(value - unpenalized, .01 * .2 ** 2)
        np.testing.assert_allclose(gradient - other_gradient, [0., .004], atol=1e-14)
        for j in range(2):
            delta = np.eye(2)[j] * 1e-5
            numeric = (verify.penalized_objective(beta + delta, z, q, .01)[0]
                       - verify.penalized_objective(beta - delta, z, q, .01)[0]) / 2e-5
            self.assertAlmostEqual(gradient[j], numeric, places=8)

    def test_independent_mean_model_accepts_zero_targets(self):
        prediction, audit = verify.second_moment_prediction(np.ones((4, 1)), np.array([0., 0., 1., 3.]), np.ones((2, 1)))
        np.testing.assert_allclose(prediction, [1., 1.], atol=1e-12)
        self.assertLessEqual(audit["gradient_max_abs"], verify.INDEPENDENT_GRADIENT_TOLERANCE)

    def test_independent_fit_is_train_only_and_mean_penalty_is_replication_invariant(self):
        x = np.c_[np.ones(30), np.arange(30.), np.sin(np.arange(30.))]
        y = np.exp(np.arange(30.) / 30)
        y[::6] = 0.
        first, audit = verify.second_moment_prediction(x, y, x)
        second, repeated = verify.second_moment_prediction(np.repeat(x, 3, axis=0), np.repeat(y, 3), x)
        _, altered = verify.second_moment_prediction(x, y, x[:2] * [1., 2., 1.])
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-12)
        np.testing.assert_array_equal(audit["means"], altered["means"])
        np.testing.assert_array_equal(audit["scales"], altered["scales"])
        np.testing.assert_allclose(audit["beta"], repeated["beta"], atol=1e-6)
        with self.assertRaises(ValueError):
            verify.second_moment_prediction(x, np.zeros(len(x)), x)

    def test_absolute_effect_gate_is_not_percentage_of_negative_score(self):
        phase = {"delta": -.005, "action_sensitivity": {"delta": -.001}}
        self.assertTrue(verify.qualifies_effect([phase, dict(phase, stability=[{"delta": -.001}, {"delta": -.002}])]))
        self.assertFalse(verify.qualifies_effect([dict(phase, delta=-.004999), dict(phase, stability=[{"delta": -.001}])]))

    def test_all84_trials_and_six_wave_hypotheses_are_retained(self):
        cumulative = verify.holm([.00001] + [1.] * 83)
        self.assertEqual(len(cumulative), 84)
        self.assertAlmostEqual(cumulative[0], .00084)
        self.assertEqual(len(verify.COMPARISONS), 6)
        self.assertEqual(verify.WAVE_ALPHA, .0025)


if __name__ == "__main__":
    unittest.main()
