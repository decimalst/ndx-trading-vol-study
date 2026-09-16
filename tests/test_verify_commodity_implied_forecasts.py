"""Prewritten invented-data reconstruction contracts; no empirical file reads."""

import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import verify_commodity_implied_forecasts as verify

MARKET = (
    "const",
    "lrv_d",
    "lrv_w",
    "lrv_m",
    "lev_d",
    "lev_w",
    "lev_m",
    "liv",
    "lvix",
    "term",
    "xasset_stress",
    "market_stress",
)
HISTORY = (
    "uso_ret",
    "uso_r2",
    "uso_lrv5",
    "uso_lrv22",
    "gld_ret",
    "gld_r2",
    "gld_lrv5",
    "gld_lrv22",
)
ALL = MARKET + HISTORY + ("lovx", "lgvz")


def inputs():
    index = pd.bdate_range("2012-01-03", "2014-04-10").as_unit("ns")
    rng = np.random.default_rng(790821)
    n = len(index)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    opened = close * np.exp(rng.normal(0, 0.008, n))
    daily = pd.DataFrame(
        {
            "open": opened,
            "high": np.maximum(opened, close) * np.exp(rng.uniform(0.002, 0.016, n)),
            "low": np.minimum(opened, close) * np.exp(-rng.uniform(0.002, 0.016, n)),
            "close": close,
            "adj close": close * np.exp(np.cumsum(rng.normal(0, 0.0001, n))),
            "volume": np.exp(rng.normal(14, 0.4, n)),
        },
        index=index,
    )
    cross = pd.DataFrame(
        80 * np.exp(np.cumsum(rng.normal(0, 0.012, (n, 5)), axis=0)),
        index=index,
        columns=("hyg", "tlt", "gld", "uso", "uup"),
    )
    iv = pd.DataFrame(
        np.exp(rng.normal(3, 0.2, (n, 3))), index=index, columns=("vxn", "vix", "vix9d")
    )
    commodity = pd.DataFrame(
        np.exp(rng.normal(3.2, 0.3, (n, 2))), index=index, columns=("OVX", "GVZ")
    )
    config = {
        "origin_start": "2014-02-01",
        "origin_end": "2014-04-10",
        "development": ["2014-02-01", "2014-02-20"],
        "evaluation": ["2014-03-05", "2014-04-10"],
        "source_end": "2014-04-10",
        "minimum_train": 250,
    }
    return daily, cross, iv, commodity, config


def produce(daily, cross, iv, commodity, config):
    # Only generated agreement/corruption tests call producers. The verifier may
    # not import or call any of these routines to establish expected outputs.
    from src.claims_release_market import build_market_features, make_five_session_targets
    from src.commodity_implied_features import build_commodity_features
    from src.commodity_implied_pipeline import build_panel

    market = build_market_features(daily, cross, iv)
    extra = build_commodity_features(daily.index, cross, commodity)
    features = pd.concat([market.loc[:, MARKET], extra], axis=1)
    targets = make_five_session_targets(market.rv_total)
    return build_panel(features, targets, config)


def feature_inputs():
    index = pd.bdate_range("2008-12-29", periods=70)
    x = np.arange(len(index), dtype=float)
    cross = pd.DataFrame(
        {
            "hyg": 80.0,
            "tlt": 90.0,
            "gld": 70 * np.exp(0.003 * x + 0.03 * np.sin(x / 3)),
            "uso": 50 * np.exp(-0.001 * x + 0.04 * np.cos(x / 4)),
            "uup": 20.0,
        },
        index=index,
    )
    commodity = pd.DataFrame({"OVX": 20 + x / 5, "GVZ": 15 + x / 9}, index=index)
    return index, cross, commodity


def model_inputs():
    rng = np.random.default_rng(196039)
    index = pd.bdate_range("2010-01-04", periods=180)
    train = pd.DataFrame(rng.normal(size=(180, 22)), index=index, columns=ALL)
    query = pd.DataFrame(rng.normal(size=(7, 22)), columns=ALL)
    train["const"] = query["const"] = 1.0
    for frame in (train, query):
        frame["uso_r2"] = frame.uso_ret**2
        frame["gld_r2"] = frame.gld_ret**2
    y = pd.Series(np.exp(-6 + 0.2 * train.lovx + rng.normal(0, 0.2, len(train))), index=index)
    return train, y, query


class IndependentFeatureModelTests(unittest.TestCase):
    def test_scalar_feature_oracle_and_literal_order(self):
        index, cross, commodity = feature_inputs()
        result = verify._commodity_features(index, cross, commodity)
        self.assertEqual(
            tuple(result.columns), HISTORY + ("lovx", "lgvz", "commodity_cutoff_date")
        )
        i = 45
        for asset in ("uso", "gld"):
            ret = [
                math.log(cross[asset].iloc[j]) - math.log(cross[asset].iloc[j - 1])
                for j in range(i - 22, i)
            ]
            self.assertAlmostEqual(result.iloc[i][asset + "_ret"], ret[-1], places=14)
            self.assertAlmostEqual(result.iloc[i][asset + "_r2"], ret[-1] ** 2, places=14)
            for window in (5, 22):
                self.assertAlmostEqual(
                    result.iloc[i][asset + f"_lrv{window}"],
                    math.log(math.fsum(v * v for v in ret[-window:]) / window),
                    places=12,
                )
        for symbol, name in (("OVX", "lovx"), ("GVZ", "lgvz")):
            self.assertAlmostEqual(
                result.iloc[i][name],
                2 * (math.log(commodity[symbol].iloc[i - 1]) - math.log(100)) - math.log(252),
                places=13,
            )
        self.assertEqual(result.commodity_cutoff_date.iloc[i], index[i - 1])

    def test_floor_before_arithmetic_and_no_pre_floor_predecessor(self):
        index, cross, commodity = feature_inputs()
        expected = verify._commodity_features(index, cross, commodity)
        cross.loc[index < "2009-01-02", ["uso", "gld"]] = -np.inf
        commodity.loc[index < "2009-01-02"] = -1.0
        actual = verify._commodity_features(index, cross, commodity)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertTrue(actual.loc[:"2009-01-02"].iloc[:, :10].isna().all().all())
        self.assertTrue(pd.isna(actual.loc["2009-01-05", "uso_ret"]))
        self.assertTrue(np.isfinite(actual.loc["2009-01-05", "lovx"]))
        self.assertTrue(np.isfinite(actual.loc["2009-01-06", "uso_ret"]))

    def test_align_before_lag_and_strict_windows_preserve_missing_dates(self):
        index, cross, commodity = feature_inputs()
        actual = verify._commodity_features(
            index, cross.drop(index[30]), commodity.drop(index[35])
        )
        self.assertTrue(actual.uso_ret.iloc[31:33].isna().all())
        self.assertTrue(actual.uso_lrv5.iloc[31:37].isna().all())
        self.assertTrue(np.isfinite(actual.uso_lrv5.iloc[37]))
        self.assertTrue(pd.isna(actual.lovx.iloc[36]))
        self.assertEqual(actual.commodity_cutoff_date.iloc[36], index[35])

    def test_future_mutations_do_not_change_past_features(self):
        index, cross, commodity = feature_inputs()
        expected = verify._commodity_features(index, cross, commodity)
        cross.loc[index[40] :, "uso"] *= 1.3
        commodity.loc[index[40] :, "OVX"] *= 2
        actual = verify._commodity_features(index, cross, commodity)
        pd.testing.assert_frame_equal(actual.iloc[:41], expected.iloc[:41])
        self.assertNotEqual(actual.lovx.iloc[41], expected.lovx.iloc[41])

    def test_zero_windows_and_extreme_positive_iv_log_arithmetic(self):
        index, cross, commodity = feature_inputs()
        cross[["uso", "gld"]] = 50.0
        commodity.loc[index[40], "OVX"] = np.nextafter(0.0, 1.0)
        commodity.loc[index[40], "GVZ"] = np.finfo(float).max
        result = verify._commodity_features(index, cross, commodity)
        self.assertEqual(result.uso_r2.iloc[-1], 0.0)
        self.assertTrue(
            result[["uso_lrv5", "uso_lrv22", "gld_lrv5", "gld_lrv22"]].isna().all().all()
        )
        self.assertTrue(
            np.isfinite(result.loc[index[41], ["lovx", "lgvz"]].to_numpy(float)).all()
        )

    def test_nonzero_square_underflow_or_overflow_aborts(self):
        for number in (1e-200, 1e200):
            with self.subTest(number=number), self.assertRaises(ValueError):
                verify._square(number)
        self.assertEqual(verify._square(0.0), 0.0)
        self.assertEqual(verify._square(-2.0), 4.0)

    def test_separate_arms_match_independent_unscaled_qr_and_smearing(self):
        train, y, query = model_inputs()
        predictions, audits = verify._fit_expected(train, y, query)
        for arm, names in (
            ("market", MARKET),
            ("matched", MARKET + HISTORY),
            ("candidate", ALL),
        ):
            x, a = train.loc[:, names].to_numpy(), query.loc[:, names].to_numpy()
            q, r = np.linalg.qr(x, mode="reduced")
            beta = np.linalg.solve(r, q.T @ np.log(y.to_numpy()))
            residual = np.log(y.to_numpy()) - x @ beta
            np.testing.assert_allclose(
                predictions[arm],
                np.exp(a @ beta) * np.exp(residual).mean(),
                rtol=1e-11,
                atol=1e-14,
            )
            audit = audits[arm]
            self.assertEqual(audit["feature_names"], list(names))
            self.assertEqual(audit["rank"], len(names))
            self.assertEqual(audit["n_train"], len(train))
            self.assertAlmostEqual(
                audit["log_smearing"], math.log(np.exp(residual).mean()), places=12
            )
            np.testing.assert_allclose(
                list(audit["feature_scales"].values()),
                x[:, 1:].std(axis=0, ddof=0),
                rtol=1e-13,
            )
        self.assertEqual(len({a["log_smearing"] for a in audits.values()}), 3)

    def test_common22_rank_scale_target_and_application_failures(self):
        for defect in (
            "missing",
            "nan",
            "constant",
            "tiny",
            "dependent",
            "target",
            "alignment",
            "few",
            "query",
        ):
            train, y, query = model_inputs()
            if defect == "missing":
                train = train.drop(columns="lgvz")
            elif defect == "nan":
                train.iloc[0, -1] = np.nan
            elif defect == "constant":
                train["lgvz"] = 1.0
            elif defect == "tiny":
                train["lgvz"] *= 1e-14
            elif defect == "dependent":
                train["lgvz"] = train.lovx + train.uso_ret
            elif defect == "target":
                y.iloc[0] = 0.0
            elif defect == "alignment":
                y = y.iloc[::-1]
            elif defect == "few":
                train, y = train.iloc[:21], y.iloc[:21]
            else:
                query.iloc[0, -1] = np.nan
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                verify._fit_expected(train, y, query)

    def test_application_mutation_cannot_change_fit_or_other_predictions(self):
        train, y, query = model_inputs()
        first, audit = verify._fit_expected(train, y, query)
        query.iloc[-1, 1:] += 0.1
        second, other = verify._fit_expected(train, y, query)
        self.assertEqual(audit, other)
        for arm in first:
            np.testing.assert_array_equal(first[arm][:-1], second[arm][:-1])
        empty, _ = verify._fit_expected(train, y, query.iloc[:0])
        self.assertTrue(all(len(p) == 0 for p in empty.values()))


class ForecastVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = inputs()
        cls.produced = produce(*cls.source)

    def test_complete_generated_pipeline_and_json_fits_roundtrip(self):
        output = copy.deepcopy(self.produced)
        output["fits"] = json.loads(json.dumps(output["fits"], allow_nan=False))
        result = verify.verify_forecasts(*self.source, output)
        self.assertEqual(
            result,
            {
                "status": "VERIFIED",
                "application_origins": len(output["applications"]),
                "application_forecasts_verified": 3 * len(output["applications"]),
                "scored_origins": len(output["panel"]) // 3,
                "forecasts_verified": len(output["panel"]),
                "monthly_fits": len(output["fits"]),
                "model_fits": 3 * len(output["fits"]),
                "coverage_origins": len(output["coverage"]),
                "calendar_rows": len(self.source[0]),
            },
        )

    def test_native_ms_us_ns_calendar_transport(self):
        for unit in ("ms", "us", "ns"):
            source = copy.deepcopy(self.source)
            for frame in source[:4]:
                frame.index = frame.index.as_unit(unit)
            with self.subTest(unit=unit):
                self.assertEqual(
                    verify.verify_forecasts(*source, self.produced)["status"], "VERIFIED"
                )

    def test_all_input_calendar_envelopes_preflight_before_any_numeric(self):
        for which in range(4):
            source = copy.deepcopy(self.source)
            frame = source[which]
            frame.loc[pd.Timestamp("2025-10-21")] = -1.0
            with (
                self.subTest(which=which),
                patch.object(verify, "_market_targets") as market,
                patch.object(verify, "_commodity_features") as commodity,
            ):
                with self.assertRaises(ValueError):
                    verify.verify_forecasts(*source, self.produced)
                market.assert_not_called()
                commodity.assert_not_called()

    def test_malformed_calendar_or_config_rejected(self):
        for defect in (
            "duplicate",
            "reverse",
            "timezone",
            "midday",
            "nat",
            "narrow_source",
            "wide_source",
            "bool_floor",
            "extra",
            "phase",
            "early",
        ):
            source = list(copy.deepcopy(self.source))
            if defect == "duplicate":
                source[3] = pd.concat([source[3], source[3].iloc[-1:]])
            elif defect == "reverse":
                source[1] = source[1].iloc[::-1]
            elif defect == "timezone":
                source[2].index = source[2].index.tz_localize("UTC")
            elif defect == "midday":
                source[0].index += pd.Timedelta(hours=1)
            elif defect == "nat":
                source[3].index = source[3].index[:-1].append(pd.DatetimeIndex([pd.NaT]))
            elif defect == "narrow_source":
                source[4]["source_end"] = "2014-04-09"
            elif defect == "wide_source":
                source[4]["source_end"] = "2025-10-21"
            elif defect == "bool_floor":
                source[4]["minimum_train"] = True
            elif defect == "extra":
                source[4]["minimum_train_releases"] = 1
            elif defect == "phase":
                source[4]["evaluation"][0] = "2014-02-20"
            else:
                source[4]["origin_start"] = "2008-12-31"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                verify.verify_forecasts(*source, self.produced)

    def test_each_source_invalid_dtype_price_and_geometry_rejected(self):
        for which, column, value in (
            (0, "close", 0.0),
            (0, "volume", -1.0),
            (1, "uso", np.inf),
            (2, "vxn", -1.0),
            (3, "OVX", 0.0),
            (3, "GVZ", "3"),
            (3, "OVX", True),
            (0, "high", 0.1),
        ):
            source = copy.deepcopy(self.source)
            if type(value) in (str, bool):
                source[which][column] = value
            else:
                source[which].loc[source[which].index[300], column] = value
            with (
                self.subTest(which=which, column=column, value=value),
                self.assertRaises(ValueError),
            ):
                verify.verify_forecasts(*source, self.produced)

    def test_every_scored_and_unscored_prediction_and_loss_are_checked(self):
        for table, column, row in (
            ("applications", "pred_candidate", -1),
            ("applications", "pred_market", 0),
            ("applications", "pred_matched", 0),
            ("panel", "prediction", 0),
            ("panel", "y", 0),
            ("panel", "loss", 0),
        ):
            output = copy.deepcopy(self.produced)
            position = output[table].index[row]
            output[table].loc[position, column] += 0.01
            with (
                self.subTest(table=table, column=column, row=row),
                self.assertRaises(ValueError),
            ):
                verify.verify_forecasts(*self.source, output)

    def test_every_model_audit_field_and_training_identity_are_checked(self):
        for field in (
            "coefficients",
            "feature_means",
            "feature_scales",
            "singular_values",
            "rank",
            "log_smearing",
            "normal_equation_max_abs",
            "n_train",
            "n_application",
            "rank_relative_cutoff",
            "feature_names",
        ):
            output = copy.deepcopy(self.produced)
            audit = output["fits"][0]["model_audits"]["candidate"]
            if type(audit[field]) is dict:
                audit[field][next(iter(audit[field]))] += 0.01
            elif type(audit[field]) is list:
                if field == "feature_names":
                    audit[field].reverse()
                else:
                    audit[field][0] += 0.01
            else:
                audit[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.verify_forecasts(*self.source, output)
        for field in (
            "train_origins",
            "application_origins",
            "train_n",
            "fit_origin",
            "training_cutoff",
        ):
            output = copy.deepcopy(self.produced)
            fit = output["fits"][0]
            if type(fit[field]) is list:
                fit[field] = fit[field][1:]
            elif type(fit[field]) is int:
                fit[field] += 1
            else:
                fit[field] = "2014-02-06"
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.verify_forecasts(*self.source, output)

    def test_all_five_outputs_reject_omitted_duplicate_reordered_and_extra_rows(self):
        for name in self.produced:
            for change in ("omit", "duplicate", "reverse", "unknown"):
                output = copy.deepcopy(self.produced)
                if name == "fits":
                    if change == "omit":
                        output[name] = output[name][1:]
                    elif change == "duplicate":
                        output[name].append(copy.deepcopy(output[name][0]))
                    elif change == "reverse":
                        output[name].reverse()
                    else:
                        output[name][0]["extra"] = True
                elif change == "omit":
                    output[name] = output[name].iloc[1:].reset_index(drop=True)
                elif change == "duplicate":
                    output[name] = pd.concat(
                        [output[name], output[name].iloc[:1]], ignore_index=True
                    )
                elif change == "reverse":
                    output[name] = output[name].iloc[::-1].reset_index(drop=True)
                else:
                    output[name]["extra"] = True
                with self.subTest(name=name, change=change), self.assertRaises(ValueError):
                    verify.verify_forecasts(*self.source, output)

    def test_coverage_and_schedule_metadata_cannot_be_forged(self):
        changes = (
            ("coverage", "commodity_cutoff_date", pd.Timestamp("2014-01-29")),
            ("coverage", "offset", 99),
            ("coverage", "target_end", pd.Timestamp("2014-02-04")),
            ("coverage", "target_observed", False),
            ("coverage", "scored", False),
            ("coverage", "missing_features", "lgvz"),
            ("coverage", "status", "outside_phase"),
            ("schedules", "train_n", 1),
            ("schedules", "application_n", 0),
            ("schedules", "training_cutoff", pd.Timestamp("2014-01-29")),
        )
        for table, column, value in changes:
            output = copy.deepcopy(self.produced)
            output[table].loc[0, column] = value
            with self.subTest(table=table, column=column), self.assertRaises(ValueError):
                verify.verify_forecasts(*self.source, output)

    def test_query_is_feature_complete_and_label_blind_with_missing_future_target(self):
        source = copy.deepcopy(self.source)
        daily, _, _, commodity, config = source
        origins = daily.index[daily.index >= config["origin_start"]]
        first = origins[0]
        commodity.loc[daily.index[daily.index.get_loc(first) - 1], "OVX"] = np.nan
        query = origins[1]
        daily.loc[origins[3], "high"] = np.nan
        output = produce(*source)
        self.assertEqual(output["fits"][0]["fit_origin"], query.strftime("%Y-%m-%d"))
        self.assertEqual(
            output["coverage"].set_index("origin").loc[query, "status"], "missing_target"
        )
        self.assertIn(query, set(output["applications"].origin))
        self.assertEqual(verify.verify_forecasts(*source, output)["status"], "VERIFIED")

    def test_common22_rows_strict_previous_session_maturity_and_calendar_offsets(self):
        source = copy.deepcopy(self.source)
        daily, cross, _, commodity, _config = source
        hole = daily.index[300]
        commodity.loc[hole, "GVZ"] = np.nan
        source = list(source)
        source[1] = cross.drop(daily.index[350])
        output = produce(*source)
        fit = output["fits"][0]
        query_position = daily.index.get_loc(pd.Timestamp(fit["fit_origin"]))
        self.assertIn(
            daily.index[query_position - 6].strftime("%Y-%m-%d"), fit["train_origins"]
        )
        self.assertNotIn(
            daily.index[query_position - 5].strftime("%Y-%m-%d"), fit["train_origins"]
        )
        self.assertNotIn(daily.index[301].strftime("%Y-%m-%d"), fit["train_origins"])
        self.assertNotIn(daily.index[351].strftime("%Y-%m-%d"), fit["train_origins"])
        for row in output["coverage"].itertuples():
            self.assertEqual(row.offset, daily.index.get_loc(row.origin) % 5)
        self.assertEqual(verify.verify_forecasts(*source, output)["status"], "VERIFIED")

    def test_all_censoring_states_and_unscored_applications_survive(self):
        output = self.produced
        coverage = output["coverage"].set_index("origin")
        self.assertEqual(coverage.loc["2014-02-19", "status"], "target_after_phase_cutoff")
        self.assertEqual(coverage.loc["2014-02-25", "status"], "outside_phase")
        self.assertEqual(coverage.loc["2014-04-10", "status"], "target_not_mature")
        for day in ("2014-02-19", "2014-02-25", "2014-04-10"):
            self.assertIn(pd.Timestamp(day), set(output["applications"].origin))
        self.assertEqual(verify.verify_forecasts(*self.source, output)["status"], "VERIFIED")

    def test_empty_month_and_all_incomplete_month_remain_in_schedule(self):
        source = list(copy.deepcopy(self.source))
        source[0] = source[0].loc[
            ~((source[0].index >= "2014-03-01") & (source[0].index < "2014-04-01"))
        ]
        source[3].loc["2014-01-31":"2014-02-28", "OVX"] = np.nan
        output = produce(*source)
        schedule = output["schedules"].set_index("month")
        self.assertEqual(schedule.loc["2014-02", "status"], "no_complete_origin")
        self.assertEqual(schedule.loc["2014-03", "status"], "no_requested_origins")
        self.assertEqual(verify.verify_forecasts(*source, output)["status"], "VERIFIED")

    def test_insufficient_training_and_degenerate_design_abort_verification(self):
        source = copy.deepcopy(self.source)
        source[4]["minimum_train"] = 1000
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            verify.verify_forecasts(*source, self.produced)
        source = copy.deepcopy(self.source)
        source[3]["GVZ"] = 20.0
        with self.assertRaises(ValueError):
            verify.verify_forecasts(*source, self.produced)

    def test_output_payload_schema_and_inputs_are_preserved(self):
        before = copy.deepcopy(self.source)
        original = copy.deepcopy(self.produced)
        verify.verify_forecasts(*self.source, self.produced)
        for now, old in zip(self.source[:4], before[:4]):
            pd.testing.assert_frame_equal(now, old)
        self.assertEqual(self.source[4], before[4])
        for key in ("applications", "panel", "coverage", "schedules"):
            pd.testing.assert_frame_equal(self.produced[key], original[key])
        self.assertEqual(self.produced["fits"], original["fits"])
        for output in (
            None,
            dict(self.produced, extra=True),
            {k: v for k, v in self.produced.items() if k != "fits"},
        ):
            with self.assertRaises(ValueError):
                verify.verify_forecasts(*self.source, output)


if __name__ == "__main__":
    unittest.main()
