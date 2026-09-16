"""Independent original-plan full-session risk replication verifier.

No new producer imports. Sources, features, nominal windows and all forecasts
are independently reconstructed. Fixed pre-fit tolerances: producer KKT1e-8;
separate BFGS KKT2e-8; coefficients absolute1e-6; forecasts relative1e-6 and
absolute1e-12. Shared frozen independent source/inference helpers are reused.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from .verify_index_hinge import validate_dates
from .verify_international_volatility import same_tree
from .verify_iterative_signal_search import digest, holm, same
from .verify_macro_overnight import (
    penalized_objective,
    proper_score,
    scan_source_admissions,
    second_moment_prediction,
    training_mask,
    verify_bls_sources,
    verify_fomc_sources,
)
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = Path(__file__).resolve().parents[1]
ZONE = ZoneInfo("America/New_York")
BASE = ("const", "lrv_d", "lrv_w", "lrv_m", "neg_d", "neg_w", "neg_m", "lvix", "term", "lvvix",
        "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4", "nominal_hours")
ADDITIONS = ("cpi_plan", "nfp_plan", "fomc_plan")
ALL_FEATURES = BASE + ADDITIONS
MODELS = ("mean", "baseline", "calendar")
COMPARISONS = (("calendar", "baseline"), ("calendar", "mean"))
WAVE_ALPHA = .05 / (8 * 9)


def validate_protocol(protocol):
    expected_index = {
        "asset": "SPX", "source_end": "2025-10-20", "sealed_start": "2025-11-03", "origin_start": "2016-01-04",
        "origin_end": "2025-10-17", "latest_target": "2025-10-20", "development": ["2016-01-04", "2019-12-31"],
        "development_target_available_by": "2019-12-31", "evaluation": ["2020-01-02", "2025-10-17"],
        "evaluation_stability": [["2020-01-02", "2022-12-31"], ["2023-01-01", "2025-10-17"]],
        "horizons": [1], "models": list(MODELS), "baseline": list(BASE), "all_features": list(ALL_FEATURES),
        "market_lag": 1, "minimum_train": 1000, "penalty": .01, "optimizer_max_iter": 200,
        "gradient_tolerance": 1e-8, "armijo": 1e-4, "maximum_backtracks": 60, "effect_threshold_absolute": .005}
    expected_inference = {"blocks": [21, 63, 126], "hac_lags": 126, "minimum_phase_observations": 127,
                          "bootstrap_draws": 99999, "seed": 20260914}
    expected_tolerances = {"producer_gradient_tolerance": 1e-8, "independent_gradient_tolerance": 2e-8,
                           "coefficient_absolute_tolerance": 1e-6, "forecast_relative_tolerance": 1e-6,
                           "forecast_absolute_tolerance": 1e-12}
    expected_comparisons = {"new_hypotheses": 2, "inherited_hypotheses": 106, "cumulative_hypotheses": 108,
                            "controls": ["baseline", "mean"], "contrasts": [[a, b, "qlike"] for a, b in COMPARISONS]}
    for section, expected in (("index", expected_index), ("inference", expected_inference),
                              ("verification", expected_tolerances), ("comparisons", expected_comparisons)):
        if any(protocol[section].get(key) != value for key, value in expected.items()):
            raise AssertionError("Fixed independent protocol contract differs in " + section)
    if (protocol["wave"] != 8 or protocol["wave_alpha"] != WAVE_ALPHA or protocol["measurement"]["gk_floor"] != 1e-10
            or protocol["calendar"]["source_publication_start"] != "2010-01-01"
            or protocol["calendar"]["source_publication_end"] != "2025-10-20"):
        raise AssertionError("Fixed wave, risk measurement or source fence differs")


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    prior = json.loads((root / "reports/tail_shape/manifest.json").read_text())
    required_code = set(prior["code"])
    required_code.update("src/" + name + ".py" for name in ("calendar_variance_features", "calendar_variance_models", "calendar_variance_search", "verify_calendar_variance"))
    required_code.update("tests/test_" + name + ".py" for name in ("calendar_variance_features", "calendar_variance_models", "calendar_variance_search", "calendar_variance_publication", "verify_calendar_variance"))
    if not required_code.issubset(manifest["code"]):
        raise AssertionError("Full prior Python source/test corpus and new study modules must be in manifest")
    required_inputs = set(protocol["sources"].values())
    required_inputs.update(protocol["sources"][name] + ".manifest.json" for name in ("vix", "vix9d", "vvix"))
    required_inputs.add("data/research_paths/source_manifest.json")
    corpus = root / "data/source_discovery/macro_plans"
    required_inputs.update(str(file.relative_to(root)) for file in corpus.rglob("*") if file.is_file())
    original = json.loads((root / "reports/macro_overnight/manifest.json").read_text())
    required_inputs.update(path for path in original["inputs"] if path.startswith("data/source_discovery/macro_plans/"))
    if not required_inputs.issubset(manifest["inputs"]):
        raise AssertionError("Complete original captured source corpus and market manifests must be frozen")
    required_preserved = set(protocol["comparisons"]["inherited_sources"]) | {"reports/tail_shape/manifest.json", "reports/macro_overnight/manifest.json"}
    if not required_preserved.issubset(manifest["preserved"]):
        raise AssertionError("Inherited result and prior registration provenance omitted from manifest")


def market_tables(daily, iv):
    validate_dates(daily.index)
    prices = daily.loc[:, ["open", "high", "low", "close"]]
    if ((prices.fillna(1) <= 0).any(axis=None) or np.isinf(prices.to_numpy()).any()
            or (prices.high < prices[["open", "close", "low"]].max(axis=1)).any()
            or (prices.low > prices[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Finite positive or missing, internally consistent raw OHLC required")
    close_return = np.log(prices.close).diff()
    day_return = np.log(prices.close / prices.open)
    gk = (.5 * np.log(prices.high / prices.low).pow(2) - (2 * np.log(2) - 1) * day_return.pow(2)).clip(lower=1e-10)
    risk = gk + np.log(prices.open / prices.close.shift()).pow(2)
    output = pd.DataFrame({"const": 1.}, index=daily.index)
    negative = close_return.mul(-1).clip(lower=0)
    for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
        output["lrv_" + suffix] = np.log(risk.rolling(width, min_periods=width).mean()).shift()
        output["neg_" + suffix] = negative.rolling(width, min_periods=width).mean().shift()
    implied = iv.reindex(daily.index)
    output["lvix"] = np.log(implied.vix.where(implied.vix > 0)).shift()
    output["term"] = np.log(implied.vix9d.where(implied.vix9d > 0) / implied.vix.where(implied.vix > 0)).shift()
    output["lvvix"] = np.log(implied.vvix.where(implied.vvix > 0)).shift()
    for weekday in range(1, 5):
        output[f"entry_dow_{weekday}"] = (daily.index.weekday == weekday).astype(float)
    output = output.loc[:, BASE[:-1]].replace([np.inf, -np.inf], np.nan)
    output["feature_cutoff_date"] = pd.Series(daily.index, index=daily.index).shift()
    maturity = pd.Series(daily.index, index=daily.index).shift(-1)
    targets = pd.DataFrame({"y": risk.shift(-1).where(np.isfinite(risk.shift(-1))),
                            "target_end": maturity, "available_date": maturity}, index=daily.index)
    return output, targets


def nominal_window(origin):
    stamp = pd.Timestamp(origin)
    if pd.isna(stamp) or stamp.tz is not None or stamp != stamp.normalize() or stamp.weekday() > 4:
        raise ValueError("Normalized weekday civil entry required")
    day = stamp.date()
    ending = day + timedelta(days=1)
    while ending.weekday() > 4:
        ending += timedelta(days=1)
    start = datetime.combine(day, time(16), ZONE)
    end = datetime.combine(ending, time(16), ZONE)
    elapsed = (end.astimezone(UTC) - start.astimezone(UTC)).total_seconds() / 3600
    return start, end, elapsed


def calendar_features(origins, cutoffs, bls_records, annual_records):
    monthly, annual, ids = {}, {}, set()
    for record in (*bls_records, *annual_records):
        if not record.get("source_id") or re.fullmatch("[0-9a-f]{64}", record.get("source_sha256", "")) is None:
            raise ValueError("Identified and hash-pinned original source required")
    for record in bls_records:
        event = record["event_type"]
        publication, planned = pd.Timestamp(record["announced_at"]), pd.Timestamp(record["planned_at"])
        if (event not in {"cpi", "nfp"} or publication.tz is None or planned.tz is None
                or pd.isna(publication) or pd.isna(planned) or publication >= planned):
            raise ValueError("Explicit ordered original plan timestamps required")
        publication, planned = publication.tz_convert(ZONE), planned.tz_convert(ZONE)
        key = event, planned.strftime("%Y-%m")
        if key in monthly or record["source_id"] in ids:
            raise ValueError("Duplicate monthly source or document identity")
        ids.add(record["source_id"])
        monthly[key] = publication.date(), planned.to_pydatetime()
    for record in annual_records:
        year = int(record["year"])
        publication = pd.Timestamp(record["announced_date"])
        dates = tuple(pd.Timestamp(day) for day in record["final_dates"])
        if (year in annual or len(dates) != 8 or len(set(dates)) != 8 or publication.tz is not None
                or publication >= pd.Timestamp(f"{year}-01-01")
                or any(day.tz is not None or day != day.normalize() or day.year != year for day in dates)):
            raise ValueError("Exactly eight original meeting dates from one preceding annual source required")
        annual[year] = publication.date(), {day.date() for day in dates}
    if not cutoffs.index.equals(origins):
        raise ValueError("Calendar publication cutoffs must align with origins")
    rows = []
    for origin, cutoff in zip(origins, cutoffs, strict=True):
        start, end, hours = nominal_window(origin)
        cutoff = None if pd.isna(cutoff) else pd.Timestamp(cutoff).date()
        if cutoff is not None and cutoff >= origin.date():
            raise ValueError("Calendar cutoff must precede entry")
        months = {str(month) for month in pd.period_range(start.date(), end.date(), freq="M")}
        row = {"nominal_hours": hours}
        for event in ("cpi", "nfp"):
            selected = [monthly.get((event, month)) for month in sorted(months)]
            known = cutoff is not None and all(item is not None and item[0] < cutoff for item in selected)
            row[event + "_plan"] = float(any(start < item[1] <= end for item in selected)) if known else np.nan
        schedule = annual.get(end.year)
        known = schedule is not None and cutoff is not None and schedule[0] < cutoff
        row["fomc_plan"] = float(end.date() in schedule[1]) if known else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=origins, columns=("nominal_hours", *ADDITIONS))


def availability_audit(origins, cutoffs, bls, annual):
    evidence = []
    for origin, cutoff in zip(origins, cutoffs, strict=True):
        start, end, _ = nominal_window(origin)
        row = {"origin": str(origin.date()), "nominal_start": start.isoformat(), "nominal_end": end.isoformat(),
               "source_rule": "publication_date strictly before preceding observed market-session date"}
        months = {str(month) for month in pd.period_range(start.date(), end.date(), freq="M")}
        for event in ("cpi", "nfp"):
            selected = [record for record in bls if record["event_type"] == event and not pd.isna(cutoff)
                        and pd.Timestamp(record["announced_at"]).tz_convert(ZONE).date() < cutoff.date()
                        and pd.Timestamp(record["planned_at"]).tz_convert(ZONE).strftime("%Y-%m") in months]
            covered = {pd.Timestamp(record["planned_at"]).tz_convert(ZONE).strftime("%Y-%m") for record in selected}
            row[event + "_missing_months"] = sorted(months - covered)
            row[event + "_source_ids"] = sorted(record["source_id"] for record in selected)
            row[event + "_source_hashes"] = sorted(record["source_sha256"] for record in selected)
        schedule = next((record for record in annual if record["year"] == end.year), None)
        known = schedule is not None and not pd.isna(cutoff) and pd.Timestamp(schedule["announced_date"]) < cutoff
        row["fomc_source_id"] = schedule["source_id"] if known else None
        row["fomc_source_sha256"] = schedule["source_sha256"] if known else None
        evidence.append(row)
    return evidence


def load_source_tables(root, protocol):
    root = Path(root)
    paths, section = protocol["sources"], protocol["index"]
    limit = pd.Timestamp(section["source_end"])
    if limit > pd.Timestamp("2025-10-20") or limit >= pd.Timestamp(section["sealed_start"]):
        raise AssertionError("Protected source fence crossed")
    daily = pd.read_parquet(root / paths["daily"], columns=["open", "high", "low", "close"], filters=[("date", "<=", limit)])
    validate_dates(daily.index)
    def source_audit(path, frame):
        return {"source_path": str(path), "source_sha256": digest(path), "bounded_rows": len(frame),
                "first_date": str(frame.index.min().date()), "last_date": str(frame.index.max().date()),
                "missing_values": {name: int(frame[name].isna().sum()) for name in frame},
                "missing_reference_dates": [str(day.date()) for day in daily.index.difference(frame.index)],
                "outside_reference_dates": [str(day.date()) for day in frame.index.difference(daily.index)]}
    sources = {"daily": source_audit(root / paths["daily"], daily)}
    series = {}
    for name in ("vix", "vix9d", "vvix"):
        field = "VVIX" if name == "vvix" else "CLOSE"
        raw = pd.read_csv(root / paths[name], usecols=["DATE", field], dtype=str)
        dates = pd.to_datetime(raw.DATE, format="%m/%d/%Y")
        keep = dates <= limit
        one = pd.Series(pd.to_numeric(raw.loc[keep, field]).to_numpy(float), index=pd.DatetimeIndex(dates.loc[keep], name="date"), name=name)
        validate_dates(one.index)
        if np.isinf(one).any() or (one.fillna(1) <= 0).any():
            raise ValueError("Positive finite or missing bounded implied-volatility values required")
        series[name] = one
        sources[name] = {**source_audit(root / paths[name], one.to_frame()), "provider_date_field": "DATE", "provider_value_field": field}
    market = {"sources": sources, "source_end": section["source_end"], "sealed_start": section["sealed_start"],
              "reference_calendar": {"definition": "bounded observed SPX raw-OHLC sessions", "sessions": len(daily),
                                     "first_date": str(daily.index.min().date()), "last_date": str(daily.index.max().date())},
              "raw_columns": list(BASE[:-1]), "numeric_post_cutoff_values_parsed": False, "historical_vintage_certified": False,
              "timing_assumption": "all market predictors through prior observed SPX session close; current entry weekday only",
              "vix9d_caveat": "prelaunch January2011-October2013 values are back-calculated archival training inputs",
              "target_interpretation": "next observed full-session SPX daily OHLC risk proxy in native squared-log-return units; not high-frequency integrated variance or causal announcement contribution"}
    scan = scan_source_admissions(root, protocol)
    if scan["mismatches"]:
        raise AssertionError("Original-source admission mismatches: " + json.dumps(scan["mismatches"]))
    bls, plans = [], {}
    for event in ("cpi", "nfp"):
        records, checked = verify_bls_sources(root, paths[event], event, protocol["calendar"])
        bls.extend(records)
        plans[event] = {"records": checked["record_count"], "admitted": checked["explicit_plans_verified"], "states": checked["statuses"]}
    annual = verify_fomc_sources(root, paths)
    plans["fomc"] = {"annual_documents": len(annual), "planned_meetings": sum(len(record["final_dates"]) for record in annual)}
    return daily, pd.DataFrame(series).sort_index(), bls, annual, market, plans


def reconstruct(root, protocol):
    daily, iv, bls, annual, market, plans = load_source_tables(root, protocol)
    features, targets = market_tables(daily, iv)
    cutoffs = features.feature_cutoff_date
    calendar = calendar_features(daily.index, cutoffs, bls, annual)
    features = features.join(calendar).loc[:, (*ALL_FEATURES, "feature_cutoff_date")]
    audit = {"market": market, "plans": plans, "plan_availability": availability_audit(daily.index, cutoffs, bls, annual),
             "plan_counts_full_bounded_calendar": {name: int(calendar[name].sum()) for name in ADDITIONS},
             "unknown_rows": int(calendar.isna().any(axis=1).sum())}
    return features, targets, audit


def eligible_entries(features, targets, section):
    complete = pd.Series(np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1), index=features.index)
    phase_mask = pd.Series(False, index=features.index)
    for phase in ("development", "evaluation"):
        first, last = section[phase]
        phase_mask |= (features.index >= first) & (features.index <= last)
    mask = complete & phase_mask & (features.index >= section["origin_start"]) & (features.index <= section["origin_end"])
    applications = features.index[mask]
    valid = (np.isfinite(targets.y) & (targets.y > 0) & targets.available_date.notna()
             & (targets.available_date <= pd.Timestamp(section["latest_target"])))
    valid &= ((features.index > pd.Timestamp(section["development"][1]))
              | (targets.available_date <= pd.Timestamp(section["development_target_available_by"])))
    return applications, applications[valid.loc[applications]], complete


def verify_model(training, y, application, audit, model):
    prediction, rebuilt = second_moment_prediction(training, y, application)
    same(audit["train_mean"], rebuilt["train_mean"], "Independent target scaling", rtol=1e-9, atol=1e-14)
    same(audit["beta"], rebuilt["beta"], "Independent BFGS coefficients", rtol=0, atol=1e-6)
    if model == "mean":
        same(audit["gradient_max_abs"], 0., "Historical-mean optimum")
        replay, maximum = np.repeat(np.mean(y), len(application)), 0.
    else:
        if audit["alpha"] != .01 or not 0 <= audit["iterations"] < 200 or audit["backtracks"] < 0:
            raise AssertionError("Fixed regularization or optimization budget differs")
        same(audit["means"], rebuilt["means"], "Train-only population centers")
        same(audit["scales"], rebuilt["scales"], "Train-only population scales")
        design = np.c_[np.ones(len(training)), (training[:, 1:] - rebuilt["means"]) / rebuilt["scales"]]
        beta = np.asarray(audit["beta"], float).copy()
        beta[0] -= np.log(np.mean(y))
        same(audit["scaled_beta"], beta, "Scaled target intercept identity", rtol=1e-9, atol=1e-12)
        value, gradient = penalized_objective(beta, design, y / np.mean(y))
        maximum = float(np.max(np.abs(gradient)))
        if maximum > 1e-8 + 1e-12:
            raise AssertionError("Saved coefficients fail independent KKT equations")
        same(audit["gradient_max_abs"], maximum, "Saved first-order residual", rtol=1e-4, atol=1e-12)
        same(audit["objective"], value + np.log(np.mean(y)), "Mean proper loss plus standardized slope penalty", rtol=1e-9, atol=1e-12)
        az = np.c_[np.ones(len(application)), (application[:, 1:] - rebuilt["means"]) / rebuilt["scales"]]
        replay = np.exp(np.log(np.mean(y)) + az @ np.asarray(audit["scaled_beta"]))
    same(replay, prediction, "Independent BFGS and saved-coefficient forecasts", rtol=1e-6, atol=1e-12)
    return replay, {"producer_kkt": maximum, "independent_kkt": rebuilt["gradient_max_abs"]}


def verify_forecasts(features, targets, forecasts, fits, protocol):
    section = protocol["index"]
    required = {"origin", "model", "horizon", "feature_cutoff_date", "target_end", "available_date", "y", "prediction",
                "fit_origin", "fit_cutoff_date", "train_n", "train_last_target", "train_last_available", "phase"}
    if not required.issubset(forecasts) or set(forecasts.model) != set(MODELS) or set(forecasts.horizon) != {1}:
        raise AssertionError("All three models and explicit forecast fields required")
    if forecasts.duplicated(["origin", "model", "horizon"]).any():
        raise AssertionError("Duplicated calendar forecasts")
    entries, scored, complete = eligible_entries(features, targets, section)
    cutoffs = pd.Series(features.index, index=features.index).shift()
    for model in MODELS:
        rows = forecasts.loc[forecasts.model == model].sort_values("origin")
        if not pd.DatetimeIndex(rows.origin).equals(scored):
            raise AssertionError("All models require exact independently reconstructed common rows")
        same(rows.y, targets.loc[scored, "y"], "Independent full-session risk targets", rtol=1e-10, atol=1e-14)
        for name in ("target_end", "available_date"):
            if not np.array_equal(rows[name].to_numpy(), targets.loc[scored, name].to_numpy()):
                raise AssertionError("Independent target maturity differs")
        if not np.array_equal(rows.feature_cutoff_date.to_numpy(), cutoffs.loc[scored].to_numpy()):
            raise AssertionError("Forecast market cutoff differs from prior session")
        phase = np.where(scored <= section["development"][1], "development", "evaluation")
        if not np.array_equal(rows.phase, phase):
            raise AssertionError("Development/evaluation membership differs")
    lookup = {pd.Timestamp(record["fit_origin"]): record for record in fits}
    expected_fits = entries[~entries.to_period("M").duplicated()]
    if len(lookup) != len(fits) or set(lookup) != set(expected_fits):
        raise AssertionError("Monthly refit date must use first complete features, never future query labels")
    count, producer_max, independent_max = 0, 0., 0.
    for entry in expected_fits:
        application = entries[entries.to_period("M") == entry.to_period("M")]
        query = scored[scored.to_period("M") == entry.to_period("M")]
        record = lookup[entry]
        selected = training_mask(complete, targets, entry, features.index) & (targets.y > 0)
        origins, n, cutoff = features.index[selected], int(selected.sum()), cutoffs.loc[entry]
        if n < section["minimum_train"] or record["train_n"] != n or record["application_n"] != len(application):
            raise AssertionError("Common training/application counts differ")
        dates = {"fit_cutoff_date": cutoff, "train_first_origin": origins[0], "train_last_origin": origins[-1],
                 "train_last_target": targets.loc[selected, "target_end"].max(), "train_last_available": targets.loc[selected, "available_date"].max()}
        if any(pd.Timestamp(record[name]) != value for name, value in dates.items()) or dates["train_last_available"] > cutoff:
            raise AssertionError("Causal training-label dates differ")
        if set(record["model_audit"]) != set(MODELS):
            raise AssertionError("Registered control or candidate omitted from fit")
        if (features.loc[selected, ALL_FEATURES[1:]].std(ddof=0) <= 1e-12).any():
            raise AssertionError("Zero-scale component cannot be dropped after inspection")
        y = targets.loc[selected, "y"].to_numpy(float)
        for model in MODELS:
            columns = ("const",) if model == "mean" else BASE if model == "baseline" else ALL_FEATURES
            audit = record["model_audit"][model]
            if tuple(audit["columns"]) != columns or audit["train_n"] != n:
                raise AssertionError("Registered model design or training count differs")
            replay, numerical = verify_model(features.loc[selected, columns].to_numpy(float), y,
                                             features.loc[query, columns].to_numpy(float), audit, model)
            rows = forecasts.loc[(forecasts.model == model) & forecasts.origin.isin(query)].sort_values("origin")
            if not rows.fit_origin.eq(entry).all() or not rows.fit_cutoff_date.eq(cutoff).all() or not rows.train_n.eq(n).all():
                raise AssertionError("Forecast fit provenance differs")
            for name in ("train_last_target", "train_last_available"):
                if not rows[name].eq(dates[name]).all():
                    raise AssertionError("Forecast label-maturity provenance differs")
            same(rows.prediction, replay, "Independent saved-parameter forecast replay", rtol=1e-9, atol=1e-12)
            producer_max = max(producer_max, numerical["producer_kkt"])
            independent_max = max(independent_max, numerical["independent_kkt"])
            count += len(rows)
    return {"forecasts_verified": count, "common_scored_origins": len(scored), "feature_complete_applications": len(entries),
            "monthly_fits_verified": len(expected_fits), "models_verified": len(MODELS),
            "producer_kkt_max": producer_max, "independent_bfgs_kkt_max": independent_max}


def effect_passes(phases):
    return (len(phases) == 2 and all(phase["delta"] <= -.005 for phase in phases)
            and len(phases[1]["stability"]) == 2 and all(item["delta"] < 0 for item in phases[1]["stability"]))


def phase_statistics(panel, control, phase, code, protocol):
    section = protocol["index"]
    first, last = section[phase]
    selected = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
    if phase == "development":
        selected = selected.loc[selected.available_date <= pd.Timestamp(section["development_target_available_by"])]
    wide = selected.pivot(index="origin", columns="model", values="prediction").sort_index()
    actual = selected.loc[selected.model == control].set_index("origin").reindex(wide.index)
    candidate_loss, control_loss = proper_score(actual.y, wide.calendar), proper_score(actual.y, wide[control])
    difference = candidate_loss - control_loss
    if len(difference) < 127:
        raise AssertionError("Literal fixed HAC126 requires at least127 phase observations")
    delta = float(difference.mean())
    hac, blocks = independent_hac(difference), {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + code * 10000 + width
        samples = explicit_bootstrap_means(difference, width, protocol["inference"]["bootstrap_draws"], seed)[:, 0]
        p = float((1 + np.count_nonzero(np.abs(samples - delta) >= abs(delta))) / (len(samples) + 1))
        blocks[str(width)] = {"p": p, "ci95": np.quantile(samples, [.025, .975]).tolist()}
    intervals = [hac["ci95"], *(value["ci95"] for value in blocks.values())]
    output = {"name": phase, "first_origin": str(wide.index[0].date()), "last_origin": str(wide.index[-1].date()),
              "n": len(wide), "delta": delta, "candidate_loss": float(candidate_loss.mean()), "control_loss": float(control_loss.mean()),
              "block_inference": blocks, "hac126": hac,
              "ci95_envelope": [min(item[0] for item in intervals), max(item[1] for item in intervals)],
              "p_conservative": max(hac["p"], *(value["p"] for value in blocks.values())),
              "annual": [], "stability": [], "nonoverlap_phases": [{"phase": 0, "n": len(wide), "delta": delta}]}
    for year in sorted(set(wide.index.year)):
        mask = wide.index.year == year
        output["annual"].append({"year": int(year), "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    if phase == "evaluation":
        for start, end in section["evaluation_stability"]:
            mask = (wide.index >= start) & (wide.index <= end)
            if not mask.any():
                raise AssertionError("Fixed evaluation slice is empty")
            output["stability"].append({"start": start, "end": end, "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    return output


def inherited_rows(root, protocol):
    output = []
    for source in protocol["comparisons"]["inherited_sources"]:
        for number, row in enumerate(json.loads((root / source).read_text())["rows"]):
            output.append({"study": row.get("study", Path(source).parent.name or Path(source).stem), "candidate": row["candidate"],
                           "control": row.get("control", "baseline"), "horizon": row["horizon"], "p_conservative": row["p_conservative"],
                           "source": source, "source_sha256": digest(root / source), "source_row_index": number,
                           **{name: row[name] for name in ("measure", "score") if name in row}})
    if len(output) != 106:
        raise AssertionError("All106 inherited comparisons must remain identified")
    return output


def verify_metrics(root, panel, protocol, metrics):
    rows = metrics["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Both registered joint-calendar contrasts required in fixed order")
    probabilities, effects = [], []
    for row in rows:
        if row["study"] != "calendar_variance" or row["horizon"] != 1 or row["score"] != "qlike":
            raise AssertionError("Full-session calendar comparison identity differs")
        phases = [phase_statistics(panel, row["control"], phase, code, protocol)
                  for code, phase in enumerate(("development", "evaluation"))]
        same_tree(row["phases"], phases, "Independent full-session phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        same(row["p_conservative"], probability, "Two-phase conjunction probability")
        probabilities.append(probability)
        effects.append(effect_passes(phases))
    prior = inherited_rows(root, protocol)
    same_tree(metrics["inherited_rows"], prior, "Complete cumulative hypothesis identities")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-2:]
    passed = []
    for number, row in enumerate(rows):
        same(row["p_holm_wave"], wave[number], "Two-comparison wave Holm")
        same(row["p_holm_cumulative"], cumulative[number], "108-comparison cumulative Holm")
        eligible = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < .05)
        if row["verdict"] != ("COMPARISON_GATE_PASS" if eligible else "DOES_NOT_QUALIFY"):
            raise AssertionError("Registered joint-calendar comparison gate differs")
        passed.append(eligible)
    leads = ["calendar"] if all(passed) else []
    if metrics["leads"] != leads or metrics["hypothesis_count"] != 2 or metrics["cumulative_hypothesis_count"] != 108:
        raise AssertionError("Joint calendar candidate must pass both controls")
    return {"new_hypotheses_verified": 2, "cumulative_hypotheses_verified": 108, "phase_comparisons_verified": 4,
            "bootstrap_runs_verified": 12, "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"], "leads": leads}


def verify(root=ROOT):
    root = Path(root)
    output, report = root / "data/calendar_variance", root / "reports/calendar_variance"
    protocol_path = root / "calendar_variance.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    validate_protocol(protocol)
    manifest = json.loads((report / "manifest.json").read_text())
    verify_manifest_coverage(root, protocol, manifest)
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Frozen protocol changed")
    for group in ("code", "inputs", "preserved"):
        for path, expected in manifest[group].items():
            if digest(root / path) != expected:
                raise AssertionError("Frozen source, code or earlier artifact changed: " + path)
    features, targets, source = reconstruct(root, protocol)
    for filename, expected, numeric in (("features", features, ALL_FEATURES), ("targets", targets, ("y",))):
        actual = pd.read_parquet(output / (filename + ".parquet"))
        if not actual.index.equals(expected.index) or tuple(actual.columns) != tuple(expected.columns):
            raise AssertionError("Independent source table schema or dates differ")
        same(actual.loc[:, numeric], expected.loc[:, numeric], "Independent raw market/calendar cells", rtol=1e-10, atol=1e-12)
        for column in set(expected) - set(numeric):
            if not actual[column].equals(expected[column]):
                raise AssertionError("Independent source table maturity differs")
    same_tree(json.loads((output / "source_audit.json").read_text()), source, "Independent original sources and complete calendar joins")
    forecasts = pd.read_parquet(output / "forecasts.parquet")
    fits = json.loads((output / "fits.json").read_text())
    result = {"status": "VERIFIED", "protocol_sha256": digest(protocol_path), "verifier_sha256": digest(Path(__file__)),
              "raw_feature_rows_verified": len(features), "raw_feature_columns_verified": len(ALL_FEATURES),
              "source_availability_rows_verified": len(source["plan_availability"]), "sources": source["plans"],
              "forecast_reconstruction": verify_forecasts(features, targets, forecasts, fits, protocol)}
    metrics = json.loads((report / "metrics.json").read_text())
    if metrics["protocol_sha256"] != digest(protocol_path) or metrics["evidence_class"] != protocol["evidence_class"]:
        raise AssertionError("Metric provenance differs")
    result["inference"] = verify_metrics(root, forecasts, protocol, metrics)
    ledger = [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines() if line]
    for event, expected in (("inherited", metrics["inherited_rows"]), ("evaluated", metrics["rows"])):
        actual = [{key: value for key, value in row.items() if key != "event"} for row in ledger if row["event"] == event]
        same_tree(actual, expected, "Independent " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (len(ledger) != 110 or len(registered) != 2 or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
            or any(row["protocol_sha256"] != digest(protocol_path) or row["horizon"] != 1 or row["score"] != "qlike" for row in registered)):
        raise AssertionError("Complete106inherited+2registered+2evaluated ledger required")
    result["ledger_events_verified"] = {"inherited": 106, "registered": 2, "evaluated": 2}
    result["limitations"] = ["Original-plan source and timing replication of a previously tested broad calendar mechanism.",
                            "Nominal civil windows ignore holidays and early closes; target maturity alone uses future observed sessions.",
                            "Historical reuse, archival revisions, incomplete original-plan coverage and back-calculated VIX9D remain exploratory limitations."]
    (report / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def invalidate_publication(root, error):
    """A failed independent check cannot leave canonical success or p-values."""
    root = Path(root)
    report = root / "reports/calendar_variance"
    report.mkdir(parents=True, exist_ok=True)
    path = report / "metrics.json"
    original_bytes = path.read_bytes() if path.exists() else None
    previous, invalid_bytes = None, None
    if original_bytes is not None:
        try:
            decoded = json.loads(original_bytes)
            if isinstance(decoded, dict):
                previous = decoded
            else:
                invalid_bytes = original_bytes
        except (UnicodeDecodeError, json.JSONDecodeError):
            invalid_bytes = original_bytes
    protocol_hash = previous.get("protocol_sha256") if previous else None
    message = "Independent verification failed: " + type(error).__name__ + ": " + str(error)
    failure = {"status": "UNEVALUABLE", "whole_wave_aborted": True, "leads": [],
               "hypothesis_count": 2, "cumulative_hypothesis_count": 108, "protocol_sha256": protocol_hash,
               "rows": [{"study": "calendar_variance", "candidate": candidate, "control": control, "score": "qlike", "horizon": 1,
                         "phases": [], "p_conservative": 1., "p_holm_wave": 1., "p_holm_cumulative": 1.,
                         "status": "INVALID_RUN", "verdict": "UNEVALUABLE", "error": message} for candidate, control in COMPARISONS]}
    serialized = json.dumps(failure, indent=2, allow_nan=False) + "\n"
    # Clear canonical success first even if a later diagnostic-output write fails.
    path.write_text(serialized)
    (report / "failure.json").write_text(serialized)
    (report / "results.md").write_text("# Original-plan SPX daily-risk replication\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n" + message + "\n")
    (report / "verification.json").write_text(json.dumps({"status": "FAILED", "error": message, "protocol_sha256": protocol_hash}, indent=2) + "\n")
    backup = report / "unpublished_scored_metrics.json"
    if previous is not None and previous.get("status") != "UNEVALUABLE" and not backup.exists():
        backup.write_text(json.dumps({"status": "UNPUBLISHED_DIAGNOSTIC_ONLY", "not_for_inherited_inference_or_promotion": True,
                                     "scored_metrics": previous}, indent=2, allow_nan=False) + "\n")
    invalid_backup = report / "unpublished_invalid_metrics.txt"
    if invalid_bytes is not None and not invalid_backup.exists():
        invalid_backup.write_bytes(invalid_bytes)
    with (report / "trial_ledger.jsonl").open("a") as stream:
        for row in failure["rows"]:
            stream.write(json.dumps({"event": "verification_failed", **row}, sort_keys=True, allow_nan=False) + "\n")
    return failure


def verify_with_failure_guard(root=ROOT):
    try:
        return verify(root)
    except Exception as error:
        invalidate_publication(root, error)
        raise


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False))
