"""Prewritten generated oracles and corruption tests; no producer imports."""

import copy
import math
import unittest

import numpy as np
import pandas as pd

from src.verify_claims_release_forecasts import (
    _claim_frame,
    _fit_expected,
    _market_targets,
    verify_forecasts,
)

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
EXTRA = ("claim_m4", "claim_age", "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
ALL = MARKET + EXTRA + ("claim_x",)
ARMS = {"market": MARKET, "matched": MARKET + EXTRA, "candidate": ALL}


def oracle_market(daily, cross, iv):
    idx = daily.index
    gk = (
        0.5 * np.log(daily["high"] / daily["low"]) ** 2
        - (2 * np.log(2) - 1) * np.log(daily["close"] / daily["open"]) ** 2
    )
    overnight = np.log(daily["open"] / daily["close"].shift()) ** 2
    rv = gk.clip(lower=1e-10) + overnight
    returns = np.log(daily["adj close"] / daily["adj close"].shift())
    x = pd.DataFrame(index=idx)
    x["const"] = 1.0
    for suffix, window in (("d", 1), ("w", 5), ("m", 22)):
        x["lrv_" + suffix] = np.log(rv.rolling(window).mean())
        x["lev_" + suffix] = returns.rolling(window).mean().clip(upper=0)
    implied = iv.reindex(idx).shift()
    x["liv"] = np.log(implied.vxn)
    x["lvix"] = np.log(implied.vix)
    x["term"] = np.log(implied.vix9d / implied.vix)

    def z(s):
        past = s.shift()
        return (s - past.rolling(252, min_periods=126).mean()) / past.rolling(
            252, min_periods=126
        ).std(ddof=1)

    prices = cross.reindex(idx)
    zcross = np.log(prices / prices.shift()).apply(z)
    x["xasset_stress"] = np.sqrt((zcross**2).sum(axis=1, skipna=False) / 5)
    zv = z(np.log(daily.volume.replace(0, np.nan))).clip(lower=0)
    zo = z((overnight / rv).clip(0, 1)).clip(lower=0)
    x["market_stress"] = np.sqrt((zv**2 + zo**2) / 2)
    x = x.loc[:, MARKET]
    x["rv_total"] = rv
    y = pd.concat([rv.shift(-k) for k in range(1, 6)], axis=1).mean(axis=1, skipna=False)
    target = pd.DataFrame(
        {"y": y, "target_end": pd.Series(idx, index=idx).shift(-5)}, index=idx
    )
    return x, target


def oracle_claims(index, ledger):
    by_week = {pd.Timestamp(row["reference_week"]): row for row in ledger}
    dated = sorted(
        (pd.Timestamp(row["release_date"]), pd.Timestamp(row["reference_week"]), row)
        for row in ledger
        if row["release_date"] is not None
    )
    records = []
    for origin in index:
        row = {
            "claim_m4": np.nan,
            "claim_age": np.nan,
            **{f"entry_dow_{day}": float(origin.dayofweek == day) for day in range(1, 5)},
            "claim_x": np.nan,
            "claim_reference_week": pd.NaT,
            "claim_release_date": pd.NaT,
            "claim_age_days": np.nan,
            "claim_status": "no_prior_release",
        }
        eligible = [item for item in dated if item[0] < origin]
        if eligible:
            released, week, source = eligible[-1]
            age = (origin - released).days
            row.update(
                claim_reference_week=week,
                claim_release_date=released,
                claim_age_days=float(age),
            )
            if age > 7:
                row["claim_status"] = "stale_release"
            elif source["status"] != "observed":
                row["claim_status"] = "latest_" + source["status"]
            else:
                previous = []
                reason = "available"
                for lag in range(1, 5):
                    prior = by_week.get(week - pd.Timedelta(days=7 * lag))
                    if prior is None:
                        reason = "missing_prior_week"
                    elif prior["release_date"] is None:
                        reason = "prior_" + prior["status"]
                    elif pd.Timestamp(prior["release_date"]) >= origin:
                        reason = "prior_not_yet_released"
                    elif prior["status"] != "observed":
                        reason = "prior_" + prior["status"]
                    else:
                        previous.append(prior["first_report_value"])
                        continue
                    break
                row["claim_status"] = reason
                if reason == "available":
                    level = sum(math.log(v) for v in previous) / 4
                    row.update(
                        claim_m4=level,
                        claim_age=age / 7,
                        claim_x=math.log(source["first_report_value"]) - level,
                    )
        records.append(row)
    return pd.DataFrame(records, index=index)


def oracle_fit(train, y, application):
    result, audits = {}, {}
    for arm, columns in ARMS.items():
        raw = train.loc[:, columns].to_numpy(float)
        query = application.loc[:, columns].to_numpy(float)
        mean, scale = raw[:, 1:].mean(0), raw[:, 1:].std(0, ddof=0)
        design = np.column_stack((np.ones(len(raw)), (raw[:, 1:] - mean) / scale))
        query_design = np.column_stack((np.ones(len(query)), (query[:, 1:] - mean) / scale))
        beta, _, rank, singular = np.linalg.lstsq(design, np.log(y), rcond=1e-12)
        residual = np.log(y) - design @ beta
        top = residual.max()
        log_smear = float(top + np.log(np.exp(residual - top).mean()))
        result[arm] = np.exp(query_design @ beta + log_smear)
        audits[arm] = {
            "feature_names": list(columns),
            "coefficients": dict(zip(columns, beta.tolist())),
            "feature_means": dict(zip(columns[1:], mean.tolist())),
            "feature_scales": dict(zip(columns[1:], scale.tolist())),
            "singular_values": singular.tolist(),
            "rank": int(rank),
            "rank_relative_cutoff": 1e-12,
            "log_smearing": log_smear,
            "normal_equation_max_abs": float(np.abs(design.T @ residual / len(raw)).max()),
            "n_train": len(train),
            "n_application": len(application),
        }
    return result, audits


def synthetic_inputs():
    rng = np.random.default_rng(63721)
    index = pd.bdate_range("2010-01-04", "2011-08-12")
    n = len(index)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    opened = close * np.exp(rng.normal(0, 0.005, n))
    daily = pd.DataFrame(
        {
            "open": opened,
            "high": np.maximum(opened, close) * np.exp(rng.uniform(0.001, 0.024, n)),
            "low": np.minimum(opened, close) * np.exp(-rng.uniform(0.001, 0.024, n)),
            "close": close,
            "adj close": close * np.exp(0.02 * np.sin(np.arange(n) / 19)),
            "volume": np.exp(14 + rng.normal(0, 0.5, n)),
        },
        index=index,
    )
    cross = pd.DataFrame(
        30 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 5)), axis=0)),
        index=index,
        columns=["hyg", "tlt", "gld", "uso", "uup"],
    )
    iv = pd.DataFrame(
        np.exp(rng.normal(3, 0.12, (n, 3))), index=index, columns=["vxn", "vix", "vix9d"]
    )
    weeks = pd.date_range("2009-12-05", "2011-08-06", freq="W-SAT")
    ledger = [
        {
            "reference_week": week.strftime("%Y-%m-%d"),
            "release_date": (week + pd.Timedelta(days=4 if i % 7 == 0 else 5)).strftime(
                "%Y-%m-%d"
            ),
            "first_report_value": int(value),
            "status": "observed",
            "source_url": "opaque-generated",
            "release_text_sha256": "a" * 64,
            "source_comparison": {"value": "never use this"},
        }
        for i, (week, value) in enumerate(zip(weeks, rng.integers(80000, 300000, len(weeks))))
    ]
    config = {
        "origin_start": "2011-06-01",
        "origin_end": "2011-08-10",
        "development": ["2011-06-01", "2011-06-30"],
        "evaluation": ["2011-07-05", "2011-08-10"],
        "source_end": "2011-08-12",
        "minimum_train": 40,
        "minimum_train_releases": 5,
    }
    return daily, cross, iv, ledger, config


def oracle_produced(daily, cross, iv, ledger, config):
    market, targets = oracle_market(daily, cross, iv)
    claim = oracle_claims(daily.index, ledger)
    features = market.loc[:, MARKET].join(claim)
    complete = np.isfinite(features.loc[:, ALL].to_numpy(float)).all(1)
    index = daily.index
    requested = index[(index >= config["origin_start"]) & (index <= config["origin_end"])]

    def phase(t):
        if (
            pd.Timestamp(config["development"][0])
            <= t
            <= pd.Timestamp(config["development"][1])
        ):
            return "development"
        if pd.Timestamp(config["evaluation"][0]) <= t <= pd.Timestamp(config["evaluation"][1]):
            return "evaluation"
        return "outside_phase"

    applications, schedules, fits = [], [], []
    for period in pd.period_range(config["origin_start"], config["origin_end"], freq="M"):
        origins = requested[requested.to_period("M") == period]
        apps = origins[complete[index.get_indexer(origins)]]
        schedule = {
            "month": str(period),
            "status": "no_requested_origins" if not len(origins) else "no_complete_origin",
            "fit_origin": pd.NaT,
            "training_cutoff": pd.NaT,
            "requested_n": len(origins),
            "application_n": len(apps),
            "train_n": None,
            "train_releases": None,
        }
        if len(apps):
            fit = apps[0]
            cutoff = index[index.get_loc(fit) - 1]
            train = index[
                complete
                & (index < fit)
                & (targets.target_end <= cutoff)
                & np.isfinite(targets.y)
            ]
            count = features.loc[train, "claim_release_date"].nunique()
            pred, audit = oracle_fit(
                features.loc[train], targets.loc[train, "y"], features.loc[apps]
            )
            fits.append(
                {
                    "month": str(period),
                    "fit_origin": fit.strftime("%Y-%m-%d"),
                    "training_cutoff": cutoff.strftime("%Y-%m-%d"),
                    "train_origins": train.strftime("%Y-%m-%d").tolist(),
                    "application_origins": apps.strftime("%Y-%m-%d").tolist(),
                    "train_n": len(train),
                    "train_releases": count,
                    "model_audits": audit,
                }
            )
            schedule.update(
                status="fitted",
                fit_origin=fit,
                training_cutoff=cutoff,
                train_n=len(train),
                train_releases=count,
            )
            for j, origin in enumerate(apps):
                applications.append(
                    {
                        "origin": origin,
                        "fit_origin": fit,
                        "training_cutoff": cutoff,
                        "train_n": len(train),
                        "train_releases": count,
                        "claim_reference_week": claim.loc[origin, "claim_reference_week"],
                        "claim_release_date": claim.loc[origin, "claim_release_date"],
                        "offset": index.get_loc(origin) % 5,
                        "phase": phase(origin),
                        **{"pred_" + arm: pred[arm][j] for arm in ARMS},
                    }
                )
        schedules.append(schedule)
    app_frame = pd.DataFrame(applications)
    by_origin = {row["origin"]: row for row in applications}
    coverage, panel = [], []
    for origin in requested:
        i = index.get_loc(origin)
        assigned = phase(origin)
        endpoint = targets.loc[origin, "target_end"]
        value = targets.loc[origin, "y"]
        observed = bool(np.isfinite(value) and value > 0)
        mature = bool(pd.notna(endpoint) and endpoint <= pd.Timestamp(config["source_end"]))
        within = bool(
            assigned != "outside_phase"
            and pd.notna(endpoint)
            and endpoint
            <= pd.Timestamp(
                config["development"][1] if assigned == "development" else config["source_end"]
            )
        )
        if not complete[i]:
            status = "incomplete_features"
        elif assigned == "outside_phase":
            status = "outside_phase"
        elif not mature:
            status = "target_not_mature"
        elif not observed:
            status = "missing_target"
        elif not within:
            status = "target_after_phase_cutoff"
        else:
            status = "scored"
        app = by_origin.get(origin)
        coverage.append(
            {
                "origin": origin,
                "phase": assigned,
                "feature_complete": bool(complete[i]),
                "missing_features": "|".join(
                    name for name in ALL if not np.isfinite(features.loc[origin, name])
                ),
                "claim_status": claim.loc[origin, "claim_status"],
                "claim_reference_week": claim.loc[origin, "claim_reference_week"],
                "claim_release_date": claim.loc[origin, "claim_release_date"],
                "offset": i % 5,
                "target_end": endpoint,
                "target_observed": observed,
                "target_within_phase": within,
                "scored": status == "scored",
                "status": status,
                "fit_origin": pd.NaT if app is None else app["fit_origin"],
                "training_cutoff": pd.NaT if app is None else app["training_cutoff"],
            }
        )
        if status == "scored":
            for arm in ARMS:
                prediction = app["pred_" + arm]
                ratio = value / prediction
                panel.append(
                    {
                        "origin": origin,
                        "model": arm,
                        "prediction": prediction,
                        "y": value,
                        "loss": ratio - np.log(ratio) - 1,
                        "target_end": endpoint,
                        "phase": assigned,
                        "claim_reference_week": app["claim_reference_week"],
                        "claim_release_date": app["claim_release_date"],
                        "offset": i % 5,
                        "fit_origin": app["fit_origin"],
                        "training_cutoff": app["training_cutoff"],
                        "train_n": app["train_n"],
                        "train_releases": app["train_releases"],
                    }
                )
    return {
        "applications": app_frame,
        "panel": pd.DataFrame(panel),
        "coverage": pd.DataFrame(coverage),
        "schedules": pd.DataFrame(schedules),
        "fits": fits,
    }


class VerifyClaimsForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic_inputs()
        cls.produced = oracle_produced(*cls.inputs)

    def test_independently_assembled_payload_verifies_including_unscored_predictions(self):
        proof = verify_forecasts(*self.inputs, self.produced)
        self.assertEqual(proof["status"], "VERIFIED")
        self.assertEqual(proof["application_origins"], len(self.produced["applications"]))
        self.assertEqual(proof["scored_origins"], self.produced["panel"].origin.nunique())
        self.assertGreater(proof["application_origins"], proof["scored_origins"])

    def test_independent_market_and_target_arrays_match_vectorized_oracle(self):
        daily, cross, iv, _, _ = self.inputs
        actual, targets = _market_targets(daily, cross, iv)
        expected, expected_targets = oracle_market(daily, cross, iv)
        np.testing.assert_allclose(
            actual.to_numpy(float),
            expected.to_numpy(float),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            targets.y, expected_targets.y, rtol=1e-9, atol=1e-12, equal_nan=True
        )
        pd.testing.assert_series_equal(
            targets.target_end,
            expected_targets.target_end.astype("datetime64[ns]"),
            check_dtype=False,
        )
        i = 140
        self.assertAlmostEqual(
            targets.y.iloc[i], math.fsum(actual.rv_total.iloc[i + 1 : i + 6]) / 5
        )
        self.assertEqual(targets.target_end.iloc[i], daily.index[i + 5])

    def test_iv_delay_missing_constituents_and_prior_stress_scaling_are_not_filled(self):
        daily, cross, iv, _, _ = copy.deepcopy(self.inputs)
        cross = cross.drop(cross.index[200])
        iv = iv.drop(iv.index[220])
        daily.loc[daily.index[230], "volume"] = 0
        actual, target = _market_targets(daily, cross, iv)
        expected, ey = oracle_market(daily, cross, iv)
        np.testing.assert_allclose(
            actual.to_numpy(float),
            expected.to_numpy(float),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )
        self.assertTrue(np.isnan(actual.liv.iloc[221]))
        self.assertTrue(np.isnan(actual.xasset_stress.iloc[200]))
        self.assertTrue(np.isnan(actual.market_stress.iloc[230]))
        np.testing.assert_allclose(target.y, ey.y, rtol=1e-9, atol=1e-12, equal_nan=True)

    def test_claims_release_day_age_seven_eight_and_missing_chain_literals(self):
        weeks = pd.date_range("2023-12-30", periods=7, freq="W-SAT")
        ledger = [
            {
                "reference_week": w.strftime("%Y-%m-%d"),
                "release_date": (w + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                "first_report_value": 90 + 10 * i,
                "status": "observed",
                "source_comparison": {"value": "poison"},
            }
            for i, w in enumerate(weeks)
        ]
        dates = pd.DatetimeIndex(["2024-02-08", "2024-02-09", "2024-02-15", "2024-02-16"])
        actual = _claim_frame(dates, ledger)
        self.assertEqual(
            actual.loc[dates[0], "claim_release_date"], pd.Timestamp("2024-02-01")
        )
        self.assertEqual(actual.loc[dates[0], "claim_age"], 1.0)
        self.assertAlmostEqual(
            actual.loc[dates[1], "claim_x"],
            math.log(140) - sum(map(math.log, [100, 110, 120, 130])) / 4,
        )
        no_new = _claim_frame(dates, ledger[:-1])
        self.assertEqual(no_new.loc[dates[2], "claim_status"], "available")
        self.assertEqual(no_new.loc[dates[3], "claim_status"], "stale_release")
        missing = _claim_frame(dates, ledger[:2] + ledger[3:])
        self.assertEqual(missing.loc[dates[1], "claim_status"], "missing_prior_week")
        self.assertTrue(np.isnan(missing.loc[dates[1], "claim_x"]))

    def test_unresolved_2017_first_value_never_falls_back_to_audit_counts(self):
        weeks = pd.date_range("2017-02-11", periods=9, freq="W-SAT")
        ledger = [
            {
                "reference_week": w.strftime("%Y-%m-%d"),
                "release_date": (w + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                "first_report_value": None
                if w == pd.Timestamp("2017-03-18")
                else 111111 + i * 1234,
                "status": "unresolved_correction"
                if w == pd.Timestamp("2017-03-18")
                else "observed",
                "source_comparison": {"dol_value": 261000, "alfred_value": 258000},
            }
            for i, w in enumerate(weeks)
        ]
        dates = pd.DatetimeIndex(["2017-03-23", "2017-03-24", "2017-03-31", "2017-04-28"])
        output = _claim_frame(dates, ledger)
        self.assertEqual(output.claim_status.iloc[1], "latest_unresolved_correction")
        self.assertEqual(output.claim_status.iloc[2], "prior_unresolved_correction")
        self.assertTrue(np.isnan(output.claim_x.iloc[1]))
        altered = copy.deepcopy(ledger)
        for row in altered:
            row["source_comparison"] = {"value": object()}
        pd.testing.assert_frame_equal(output, _claim_frame(dates, altered))

    def test_qr_fit_matches_independent_lstsq_and_separate_smearing(self):
        rng = np.random.default_rng(481)
        train = pd.DataFrame(rng.normal(size=(140, 19)), columns=ALL)
        query = pd.DataFrame(rng.normal(size=(7, 19)), columns=ALL)
        train["const"] = query["const"] = 1.0
        y = np.exp(-8 + 0.3 * train.claim_x + rng.normal(0, 0.2, len(train)))
        pred, audit = _fit_expected(train, y, query)
        expected, ea = oracle_fit(train, y, query)
        for arm in ARMS:
            np.testing.assert_allclose(pred[arm], expected[arm], rtol=1e-7, atol=1e-12)
            np.testing.assert_allclose(
                list(audit[arm]["coefficients"].values()),
                list(ea[arm]["coefficients"].values()),
                rtol=1e-7,
                atol=1e-7,
            )
        self.assertNotAlmostEqual(
            audit["market"]["log_smearing"], audit["candidate"]["log_smearing"], places=6
        )
        bad = train.copy()
        bad["claim_x"] = bad["claim_m4"]
        with self.assertRaises(ValueError):
            _fit_expected(bad, y, query)
        bad["claim_x"] = 1.0
        with self.assertRaises(ValueError):
            _fit_expected(bad, y, query)

    def test_saved_coefficients_scaling_smearing_and_fit_counts_are_checked(self):
        for key in (
            "coefficients",
            "feature_means",
            "feature_scales",
            "log_smearing",
            "normal_equation_max_abs",
            "n_train",
            "rank",
        ):
            bad = copy.deepcopy(self.produced)
            audit = bad["fits"][0]["model_audits"]["candidate"]
            if isinstance(audit[key], dict):
                name = next(iter(audit[key]))
                audit[key][name] += 0.1
            else:
                audit[key] += 0.1
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_forecasts(*self.inputs, bad)

    def test_monthly_training_maturity_and_exact_origin_identities_are_checked(self):
        for field in (
            "training_cutoff",
            "train_origins",
            "application_origins",
            "fit_origin",
            "train_releases",
        ):
            bad = copy.deepcopy(self.produced)
            fit = bad["fits"][0]
            if field == "training_cutoff":
                fit[field] = fit["fit_origin"]
            elif field == "train_origins":
                fit[field][-1] = fit["fit_origin"]
            elif field == "application_origins":
                fit[field].pop()
            elif field == "fit_origin":
                fit[field] = fit["application_origins"][1]
            else:
                fit[field] -= 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_forecasts(*self.inputs, bad)

    def test_application_forecast_and_unscored_omission_are_checked(self):
        bad = copy.deepcopy(self.produced)
        bad["applications"].loc[len(bad["applications"]) - 1, "pred_candidate"] *= 1.01
        with self.assertRaises(ValueError):
            verify_forecasts(*self.inputs, bad)
        bad = copy.deepcopy(self.produced)
        bad["applications"] = bad["applications"].iloc[:-1].copy()
        with self.assertRaises(ValueError):
            verify_forecasts(*self.inputs, bad)

    def test_target_endpoint_value_and_qlike_are_independent(self):
        for field in ("target_end", "y", "loss"):
            bad = copy.deepcopy(self.produced)
            if field == "target_end":
                bad["panel"].loc[0, field] += pd.Timedelta(days=1)
            else:
                bad["panel"].loc[0, field] += 0.01
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_forecasts(*self.inputs, bad)

    def test_same_day_claim_timing_full_calendar_offset_and_coverage_are_checked(self):
        for field, value in (
            ("claim_release_date", pd.Timestamp("2011-06-01")),
            ("offset", 99),
            ("claim_status", "stale_release"),
            ("missing_features", "claim_x"),
            ("feature_complete", False),
            ("status", "missing_target"),
        ):
            bad = copy.deepcopy(self.produced)
            bad["coverage"].loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_forecasts(*self.inputs, bad)

    def test_schedule_no_complete_month_and_no_requested_month_are_retained(self):
        daily, cross, iv, ledger, config = copy.deepcopy(self.inputs)
        iv.loc["2011-06-30":"2011-07-29"] = np.nan
        produced = oracle_produced(daily, cross, iv, ledger, config)
        self.assertEqual(produced["schedules"].iloc[1].status, "no_complete_origin")
        self.assertEqual(
            verify_forecasts(daily, cross, iv, ledger, config, produced)["status"], "VERIFIED"
        )
        daily = daily.loc[daily.index.month != 7]
        produced = oracle_produced(daily, cross, iv, ledger, config)
        self.assertEqual(produced["schedules"].iloc[1].status, "no_requested_origins")
        self.assertEqual(
            verify_forecasts(daily, cross, iv, ledger, config, produced)["status"], "VERIFIED"
        )

    def test_support_floor_is_enforced_before_accepting_saved_fit(self):
        values = list(copy.deepcopy(self.inputs))
        values[-1]["minimum_train"] = 10000
        with self.assertRaises(ValueError):
            verify_forecasts(*values, self.produced)
        values = list(copy.deepcopy(self.inputs))
        values[-1]["minimum_train_releases"] = 10000
        with self.assertRaises(ValueError):
            verify_forecasts(*values, self.produced)

    def test_calendar_numeric_and_payload_schema_corruptions_fail(self):
        daily, cross, iv, ledger, config = copy.deepcopy(self.inputs)
        daily.index = daily.index + pd.Timedelta(hours=1)
        with self.assertRaises(ValueError):
            verify_forecasts(daily, cross, iv, ledger, config, self.produced)
        daily = self.inputs[0].copy()
        daily.iloc[0, daily.columns.get_loc("high")] = 0
        with self.assertRaises(ValueError):
            verify_forecasts(daily, cross, iv, ledger, config, self.produced)
        bad = copy.deepcopy(self.produced)
        bad["unused_predictions"] = []
        with self.assertRaises(ValueError):
            verify_forecasts(*self.inputs, bad)


if __name__ == "__main__":
    unittest.main()
