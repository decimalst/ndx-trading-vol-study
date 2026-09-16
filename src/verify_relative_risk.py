"""Independent relative intraday-risk source, measurement, fit and score checks.

Pre-fit tolerances are inherited from the independent wave6 ridge verifier:
coefficients rtol1e-7/atol1e-12; forecasts rtol1e-8/atol1e-12; full MSE KKT
at most1e-10. No relative-risk producer implementation is imported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .verify_index_hinge import ridge_prediction, training_mask, validate_dates
from .verify_international_volatility import same_tree
from .verify_iterative_signal_search import digest, holm, same
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = Path(__file__).resolve().parents[1]
BASE = ("const", *(f"{asset}_{measure}_{suffix}" for asset in ("qqq", "spx") for measure in ("lg", "lt", "neg") for suffix in ("d", "w", "m")),
        "lvxn", "lvix", "term", "lvvix", "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
ALL_FEATURES = BASE + ("corr22",)
MODELS = ("mean", "baseline", "correlation")
COMPARISONS = (("correlation", "baseline"), ("correlation", "mean"))
WAVE_ALPHA = .05 / (9 * 10)
FLOOR = 1e-10


def validate_protocol(protocol):
    expected_index = {
        "asset": "QQQ_ETF_relative_to_SPX_price_index", "source_end": "2025-10-20", "sealed_start": "2025-11-03",
        "origin_start": "2016-01-04", "origin_end": "2025-10-17", "latest_target": "2025-10-20",
        "development": ["2016-01-04", "2019-12-31"], "development_target_available_by": "2019-12-31",
        "evaluation": ["2020-01-02", "2025-10-17"], "evaluation_stability": [["2020-01-02", "2022-12-31"], ["2023-01-01", "2025-10-17"]],
        "horizons": [1], "models": list(MODELS), "baseline": list(BASE), "all_features": list(ALL_FEATURES),
        "market_lag": 1, "minimum_train": 1000, "penalty": .01, "effect_threshold_relative": .0025}
    expected_inference = {"blocks": [21, 63, 126], "hac_lags": 126, "minimum_phase_observations": 127, "bootstrap_draws": 99999, "seed": 20260915}
    expected_tolerances = {"coefficient_relative_tolerance": 1e-7, "coefficient_absolute_tolerance": 1e-12,
                           "forecast_relative_tolerance": 1e-8, "forecast_absolute_tolerance": 1e-12, "gradient_tolerance": 1e-10}
    expected_comparisons = {"new_hypotheses": 2, "inherited_hypotheses": 108, "cumulative_hypotheses": 110,
                            "controls": ["baseline", "mean"], "contrasts": [[a, b, "mse"] for a, b in COMPARISONS]}
    for section, expected in (("index", expected_index), ("inference", expected_inference), ("verification", expected_tolerances), ("comparisons", expected_comparisons)):
        if any(protocol[section].get(key) != value for key, value in expected.items()):
            raise AssertionError("Fixed relative-risk protocol contract differs in " + section)
    if (protocol["wave"] != 9 or protocol["wave_alpha"] != WAVE_ALPHA or protocol["measurement"]["gk_floor"] != FLOOR
            or protocol["correlation"]["window"] != 22 or protocol["correlation"]["roundoff_tolerance"] != 1e-12):
        raise AssertionError("Fixed wave, no-floor or strict-correlation contract differs")


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    prior = json.loads((root / "reports/calendar_variance/manifest.json").read_text())
    required_code = set(prior["code"])
    required_code.update("src/" + name + ".py" for name in ("relative_risk_features", "relative_risk_models", "relative_risk_search", "verify_relative_risk", "plot_relative_risk"))
    required_code.update("tests/test_" + name + ".py" for name in ("relative_risk_features", "relative_risk_models", "relative_risk_search", "relative_risk_publication", "verify_relative_risk"))
    if not required_code.issubset(manifest["code"]):
        raise AssertionError("Entire prior Python source/test corpus and new modules must remain frozen")
    required_inputs = set(protocol["sources"].values())
    required_inputs.update(protocol["sources"][name] + ".manifest.json" for name in ("vxn", "vix", "vix9d", "vvix"))
    required_inputs.update(["data/research_paths/source_manifest.json", "data/history_extension/source_manifest.json"])
    if not required_inputs.issubset(manifest["inputs"]):
        raise AssertionError("All paired market files and documentary source manifests required")
    required_preserved = set(protocol["comparisons"]["inherited_sources"]) | {"reports/calendar_variance/manifest.json"}
    if not required_preserved.issubset(manifest["preserved"]):
        raise AssertionError("Inherited result and prior registration provenance omitted")


def load_source_tables(root, protocol):
    root = Path(root)
    paths, section = protocol["sources"], protocol["index"]
    limit = pd.Timestamp(section["source_end"])
    if limit > pd.Timestamp("2025-10-20") or limit >= pd.Timestamp(section["sealed_start"]):
        raise AssertionError("Protected source fence crossed")
    frames = {name: pd.read_parquet(root / paths[name], columns=["open", "high", "low", "close"], filters=[("date", "<=", limit)]) for name in ("daily", "qqq")}
    for frame in frames.values():
        validate_ohlc(frame)
        if frame.empty or frame.index.max() > limit:
            raise AssertionError("Nonempty bounded raw OHLC source required")
    qqq, spx = frames["qqq"], frames["daily"]
    def source_audit(path, frame):
        return {"source_path": str(path), "source_sha256": digest(path), "bounded_rows": len(frame),
                "first_date": str(frame.index.min().date()), "last_date": str(frame.index.max().date()),
                "missing_values": {name: int(frame[name].isna().sum()) for name in frame},
                "missing_reference_dates": [str(day.date()) for day in spx.index.difference(frame.index)],
                "outside_reference_dates": [str(day.date()) for day in frame.index.difference(spx.index)]}
    sources = {name: source_audit(root / paths[name], frame) for name, frame in frames.items()}
    series = {}
    for name in ("vxn", "vix", "vix9d", "vvix"):
        field = "VVIX" if name == "vvix" else "CLOSE"
        csv = pd.read_csv(root / paths[name], usecols=["DATE", field], dtype=str)
        dates = pd.to_datetime(csv.DATE, format="%m/%d/%Y")
        keep = dates <= limit
        one = pd.Series(pd.to_numeric(csv.loc[keep, field]).to_numpy(float), index=pd.DatetimeIndex(dates.loc[keep], name="date"), name=name)
        validate_dates(one.index)
        if one.empty or np.isinf(one).any() or (one.fillna(1) <= 0).any():
            raise ValueError("Nonempty positive finite or missing IV source required")
        series[name] = one
        sources[name] = {**source_audit(root / paths[name], one.to_frame()), "provider_date_field": "DATE", "provider_value_field": field}
    audit = {"sources": sources, "source_end": section["source_end"], "sealed_start": section["sealed_start"],
             "reference_calendar": {"definition": "bounded observed SPX raw-OHLC sessions", "sessions": len(spx),
                                    "first_date": str(spx.index.min().date()), "last_date": str(spx.index.max().date())},
             "raw_columns": list(ALL_FEATURES), "numeric_post_cutoff_values_parsed": False, "historical_vintage_certified": False,
             "timing_assumption": "market predictors through prior observed SPX session; each raw previous-close calculation requires its source predecessor to equal the reference predecessor",
             "vix9d_caveat": "prelaunch January2011-October2013 values are back-calculated archival training inputs",
             "target_interpretation": "next observed session log raw intraday GK of QQQ minus log raw intraday GK of SPX; ETF/index relative proxy, not portfolio variance",
             "asset_identity": {"qqq": "QQQ ETF vendor raw OHLC", "spx": "Yahoo ^GSPC price-index raw OHLC"},
             "measurement_gate": "all finite raw GK on full SPX reference calendar strictly greater than 1e-10 before any feature/label mask"}
    return qqq, spx, pd.DataFrame(series).sort_index(), audit


def raw_gk(frame):
    validate_ohlc(frame)
    price = frame.loc[:, ["open", "high", "low", "close"]]
    result = .5 * np.log(price.high / price.low).pow(2) - (2 * np.log(2) - 1) * np.log(price.close / price.open).pow(2)
    complete = price.notna().all(axis=1)
    if not np.isfinite(result.loc[complete]).all():
        raise ValueError("Nonfinite GK from observed complete prices cannot become missing data")
    return result.where(complete)


def validate_ohlc(frame):
    validate_dates(frame.index)
    price = frame.loc[:, ["open", "high", "low", "close"]]
    complete = price.loc[price.notna().all(axis=1)]
    if (np.isinf(price.to_numpy()).any() or (price.fillna(1) <= 0).any(axis=None)
            or (complete.high < complete[["open", "close", "low"]].max(axis=1)).any()
            or (complete.low > complete[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Positive finite or missing and internally consistent OHLC required")


def measurement_audit(qqq, spx):
    validate_ohlc(qqq)
    validate_ohlc(spx)
    reference = spx.index
    assets = {}
    for asset, frame in (("qqq", qqq), ("spx", spx)):
        observed = frame.reindex(reference)
        values = raw_gk(observed)
        finite = np.isfinite(values)
        assets[asset] = {"observed_complete_rows": int(finite.sum()), "missing_ohlc_rows": int((~finite).sum()),
                         "finite_gk_rows": int(finite.sum()), "floor_hit_rows": int((values.loc[finite] <= FLOOR).sum()),
                         "nonpositive_gk_rows": int((values.loc[finite] <= 0).sum()),
                         "small_positive_gk_rows": int(((values.loc[finite] > 0) & (values.loc[finite] <= FLOOR)).sum()),
                         "minimum_raw_gk": float(values.loc[finite].min()) if finite.any() else None}
    status = "PASS" if all(item["floor_hit_rows"] == 0 and item["observed_complete_rows"] > 0 for item in assets.values()) else "INSUFFICIENT_MEASUREMENT"
    return {"status": status, "reference_rows": len(reference), "gk_floor": FLOOR, "per_asset": assets}


def require_measurement(audit):
    if (audit["status"] != "PASS" or audit["gk_floor"] != FLOOR
            or any(one["floor_hit_rows"] != 0 or one["finite_gk_rows"] <= 0 or one["minimum_raw_gk"] is None
                   or not np.isfinite(one["minimum_raw_gk"]) or one["minimum_raw_gk"] <= FLOOR for one in audit["per_asset"].values())
            or set(audit["per_asset"]) != {"qqq", "spx"}):
        raise ValueError("INSUFFICIENT_MEASUREMENT: log-risk ratio cannot depend on a bound GK floor")


def checked_correlation(value):
    if not np.isfinite(value) or abs(value) > 1 + 1e-12:
        raise ValueError("Invalid Pearson correlation or excessive numerical overshoot")
    return float(value)


def correlation_value(left, right):
    x, y = np.asarray(left, float), np.asarray(right, float)
    if x.shape != (22,) or y.shape != (22,):
        raise ValueError("Exactly22 reference-session observations required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.nan
    # Exact constants can leave nonzero centered roundoff (e.g. repeated0.1).
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denominator == 0:
        return np.nan
    return checked_correlation(np.dot(x, y) / denominator)


def feature_target_tables(qqq, spx, iv):
    require_measurement(measurement_audit(qqq, spx))
    validate_dates(iv.index)
    implied = iv.loc[:, ["vxn", "vix", "vix9d", "vvix"]]
    if np.isinf(implied.to_numpy()).any() or (implied.fillna(1) <= 0).any(axis=None):
        raise ValueError("Positive finite or missing implied-volatility observations required")
    reference = spx.index
    previous = pd.Series(reference, index=reference).shift()
    output = pd.DataFrame({"const": 1.}, index=reference)
    intraday, risks = {}, {}
    for asset, source in (("qqq", qqq), ("spx", spx)):
        predecessor = pd.Series(source.index, index=source.index).shift().reindex(reference)
        matched = predecessor.eq(previous)
        frame = source.reindex(reference)
        intraday[asset] = np.log(frame.close / frame.open)
        risk = raw_gk(frame)
        overnight = np.log(frame.open / frame.close.shift()).where(matched)
        close_return = np.log(frame.close / frame.close.shift()).where(matched)
        histories = {"lg": risk, "lt": risk + overnight.pow(2), "neg": close_return.mul(-1).clip(lower=0)}
        for measure, values in histories.items():
            for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
                average = values.rolling(width, min_periods=width).mean()
                output[f"{asset}_{measure}_{suffix}"] = (np.log(average) if measure != "neg" else average).shift()
        risks[asset] = risk
    aligned = iv.reindex(reference)
    output["lvxn"] = np.log(aligned.vxn.where(aligned.vxn > 0)).shift()
    output["lvix"] = np.log(aligned.vix.where(aligned.vix > 0)).shift()
    output["term"] = np.log(aligned.vix9d.where(aligned.vix9d > 0) / aligned.vix.where(aligned.vix > 0)).shift()
    output["lvvix"] = np.log(aligned.vvix.where(aligned.vvix > 0)).shift()
    for weekday in range(1, 5):
        output[f"entry_dow_{weekday}"] = (reference.weekday == weekday).astype(float)
    correlation = np.full(len(reference), np.nan)
    qday, sday = intraday["qqq"].to_numpy(), intraday["spx"].to_numpy()
    for position in range(22, len(reference)):
        correlation[position] = correlation_value(qday[position-22:position], sday[position-22:position])
    output["corr22"] = correlation
    output = output.loc[:, ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
    output["feature_cutoff_date"] = previous
    maturity = pd.Series(reference, index=reference).shift(-1)
    target = (np.log(risks["qqq"]) - np.log(risks["spx"])).shift(-1)
    targets = pd.DataFrame({"y": target.where(np.isfinite(target)), "target_end": maturity, "available_date": maturity}, index=reference)
    return output, targets


def eligible_entries(features, targets, section):
    complete = pd.Series(np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1), index=features.index)
    phases = pd.Series(False, index=features.index)
    for name in ("development", "evaluation"):
        first, last = section[name]
        phases |= (features.index >= first) & (features.index <= last)
    applications = features.index[complete & phases & (features.index >= section["origin_start"]) & (features.index <= section["origin_end"])]
    valid = (np.isfinite(targets.y) & targets.available_date.notna() & (targets.available_date <= pd.Timestamp(section["latest_target"])))
    valid &= ((features.index > pd.Timestamp(section["development"][1]))
              | (targets.available_date <= pd.Timestamp(section["development_target_available_by"])))
    return applications, applications[valid.loc[applications]], complete


def effect_passes(phases):
    return (len(phases) == 2 and all(phase["gain_relative"] >= .0025 and phase["delta"] < 0 for phase in phases)
            and len(phases[1]["stability"]) == 2 and all(item["delta"] < 0 for item in phases[1]["stability"]))


def verify_model(training, y, application, audit):
    prediction, rebuilt = ridge_prediction(training, y, application)
    for key in ("means", "scales"):
        same(audit[key], rebuilt[key], "Independent train-only population " + key, rtol=1e-10, atol=1e-12)
    same(audit["beta"], rebuilt["beta"], "Independent augmented least-squares coefficients", rtol=1e-7, atol=1e-12)
    if audit["alpha"] != rebuilt["alpha"] or audit["train_n"] != len(y):
        raise AssertionError("Meanloss ridge penalty or same-row sample differs")
    z = (training - np.asarray(audit["means"])) / np.asarray(audit["scales"])
    az = (application - np.asarray(audit["means"])) / np.asarray(audit["scales"])
    beta = np.asarray(audit["beta"])
    gradient = 2 * z.T @ (z @ beta - y) / len(y)
    gradient[1:] += 2 * audit["alpha"] * beta[1:]
    maximum = float(np.max(np.abs(gradient)))
    if maximum > 1e-10:
        raise AssertionError("Saved full MSE gradient exceeds fixed KKT tolerance")
    same(audit["gradient_max_abs"], maximum, "Independent full MSE gradient including intercept", rtol=1e-4, atol=1e-12)
    replay = az @ beta
    same(replay, prediction, "Independent ridge forecast replay", rtol=1e-8, atol=1e-12)
    return replay, maximum


def verify_forecasts(features, targets, forecasts, fits, protocol):
    section = protocol["index"]
    required = {"origin", "model", "horizon", "feature_cutoff_date", "target_end", "available_date", "y", "prediction",
                "fit_origin", "fit_cutoff_date", "train_n", "train_last_target", "train_last_available", "phase"}
    if not required.issubset(forecasts) or set(forecasts.model) != set(MODELS) or set(forecasts.horizon) != {1}:
        raise AssertionError("All three relative-risk models and explicit fields required")
    if forecasts.duplicated(["origin", "model", "horizon"]).any():
        raise AssertionError("Duplicated relative-risk forecasts")
    entries, scored, complete = eligible_entries(features, targets, section)
    cutoffs = pd.Series(features.index, index=features.index).shift()
    for model in MODELS:
        rows = forecasts.loc[forecasts.model == model].sort_values("origin")
        if not pd.DatetimeIndex(rows.origin).equals(scored):
            raise AssertionError("Each arm must use exact independently reconstructed common rows")
        same(rows.y, targets.loc[scored, "y"], "Independent signed log-risk-ratio targets", rtol=1e-10, atol=1e-12)
        for name in ("target_end", "available_date"):
            if not np.array_equal(rows[name].to_numpy(), targets.loc[scored, name].to_numpy()):
                raise AssertionError("Independent same-session target maturity differs")
        if not np.array_equal(rows.feature_cutoff_date.to_numpy(), cutoffs.loc[scored].to_numpy()):
            raise AssertionError("Prior-session market cutoff differs")
        phase = np.where(scored <= section["development"][1], "development", "evaluation")
        if not np.array_equal(rows.phase, phase):
            raise AssertionError("Fixed phase membership differs")
    lookup = {pd.Timestamp(record["fit_origin"]): record for record in fits}
    expected_fits = entries[~entries.to_period("M").duplicated()]
    if len(lookup) != len(fits) or set(lookup) != set(expected_fits):
        raise AssertionError("Monthly refits must use first complete features regardless of future labels")
    count, maximum = 0, 0.
    for entry in expected_fits:
        application = entries[entries.to_period("M") == entry.to_period("M")]
        query = scored[scored.to_period("M") == entry.to_period("M")]
        record = lookup[entry]
        selected = training_mask(complete, targets, entry, features.index)
        origins, n, cutoff = features.index[selected], int(selected.sum()), cutoffs.loc[entry]
        if n < section["minimum_train"] or record["train_n"] != n or record["application_n"] != len(application):
            raise AssertionError("Common training or application counts differ")
        dates = {"fit_cutoff_date": cutoff, "train_first_origin": origins[0], "train_last_origin": origins[-1],
                 "train_last_target": targets.loc[selected, "target_end"].max(), "train_last_available": targets.loc[selected, "available_date"].max()}
        if any(pd.Timestamp(record[name]) != value for name, value in dates.items()) or dates["train_last_available"] > cutoff:
            raise AssertionError("Causal paired-label fitting dates differ")
        if set(record["model_audit"]) != set(MODELS):
            raise AssertionError("Registered control or candidate missing from fit")
        scale = features.loc[selected, ALL_FEATURES[1:]].std(ddof=0)
        if (scale <= 1e-12).any() or not np.isfinite(scale).all():
            raise AssertionError("Degenerate common input cannot be dropped")
        y = targets.loc[selected, "y"].to_numpy(float)
        for model in MODELS:
            columns = ("const",) if model == "mean" else BASE if model == "baseline" else ALL_FEATURES
            audit = record["model_audit"][model]
            if tuple(audit["columns"]) != columns or audit["train_n"] != n:
                raise AssertionError("Registered model design differs")
            replay, residual = verify_model(features.loc[selected, columns].to_numpy(float), y,
                                             features.loc[query, columns].to_numpy(float), audit)
            rows = forecasts.loc[(forecasts.model == model) & forecasts.origin.isin(query)].sort_values("origin")
            if not rows.fit_origin.eq(entry).all() or not rows.fit_cutoff_date.eq(cutoff).all() or not rows.train_n.eq(n).all():
                raise AssertionError("Relative-risk forecast fit provenance differs")
            for name in ("train_last_target", "train_last_available"):
                if not rows[name].eq(dates[name]).all():
                    raise AssertionError("Forecast label maturity differs")
            same(rows.prediction, replay, "Saved-parameter forecast reconstruction", rtol=1e-10, atol=1e-12)
            maximum = max(maximum, residual)
            count += len(rows)
    return {"forecasts_verified": count, "common_scored_origins": len(scored), "feature_complete_applications": len(entries),
            "monthly_fits_verified": len(expected_fits), "models_verified": len(MODELS), "full_mse_kkt_max": maximum}


def phase_statistics(panel, control, phase, code, protocol):
    section = protocol["index"]
    first, last = section[phase]
    selected = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
    if phase == "development":
        selected = selected.loc[selected.available_date <= pd.Timestamp(section["development_target_available_by"])]
    wide = selected.pivot(index="origin", columns="model", values="prediction").sort_index()
    actual = selected.loc[selected.model == control].set_index("origin").reindex(wide.index)
    candidate_loss, control_loss = np.square(actual.y - wide.correlation), np.square(actual.y - wide[control])
    difference = candidate_loss - control_loss
    if len(difference) < 127 or control_loss.mean() <= 0:
        raise AssertionError("At least127 phase rows and positive control MSE required")
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
              "gain_relative": float(1 - candidate_loss.mean() / control_loss.mean()), "block_inference": blocks, "hac126": hac,
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
                raise AssertionError("Fixed evaluation slice empty")
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
    if len(output) != 108:
        raise AssertionError("All108 inherited comparisons must remain identified")
    return output


def verify_metrics(root, panel, protocol, metrics):
    rows = metrics["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Both registered relative-risk comparisons required in fixed order")
    probabilities, effects = [], []
    for row in rows:
        if row["study"] != "relative_risk" or row["horizon"] != 1 or row["score"] != "mse":
            raise AssertionError("Relative intraday-risk comparison identity differs")
        phases = [phase_statistics(panel, row["control"], phase, code, protocol) for code, phase in enumerate(("development", "evaluation"))]
        same_tree(row["phases"], phases, "Independent relative-risk phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        same(row["p_conservative"], probability, "Both-phase conjunction probability")
        probabilities.append(probability)
        effects.append(effect_passes(phases))
    prior = inherited_rows(root, protocol)
    same_tree(metrics["inherited_rows"], prior, "Complete cumulative trial identities")
    wave, cumulative = holm(probabilities), holm([row["p_conservative"] for row in prior] + probabilities)[-2:]
    passed = []
    for number, row in enumerate(rows):
        same(row["p_holm_wave"], wave[number], "Two-comparison wave Holm")
        same(row["p_holm_cumulative"], cumulative[number], "110-comparison cumulative Holm")
        eligible = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < .05)
        if row["verdict"] != ("COMPARISON_GATE_PASS" if eligible else "DOES_NOT_QUALIFY"):
            raise AssertionError("Relative-risk statistical or fixed effect gate differs")
        passed.append(eligible)
    leads = ["correlation"] if all(passed) else []
    if metrics["leads"] != leads or metrics["hypothesis_count"] != 2 or metrics["cumulative_hypothesis_count"] != 110:
        raise AssertionError("One joint-return increment must pass both controls")
    return {"new_hypotheses_verified": 2, "cumulative_hypotheses_verified": 110, "phase_comparisons_verified": 4,
            "bootstrap_runs_verified": 12, "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"], "leads": leads}


def verify_ledger(root, metrics, prior, final_event):
    report = Path(root) / "reports/relative_risk"
    ledger = [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines() if line]
    for event, expected in (("inherited", prior), (final_event, metrics["rows"])):
        actual = [{key: value for key, value in row.items() if key != "event"} for row in ledger if row["event"] == event]
        same_tree(actual, expected, "Independent " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (len(ledger) != 112 or len(registered) != 2 or len(prior) != 108
            or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
            or any(row["protocol_sha256"] != metrics["protocol_sha256"] or row["horizon"] != 1 or row["score"] != "mse" for row in registered)):
        raise AssertionError("Complete108inherited+2registered+2terminal ledger required")
    return {"inherited": 108, "registered": 2, final_event: 2}


def verify_terminal_measurement(root, measurement, metrics, prior):
    root = Path(root)
    output, report = root / "data/relative_risk", root / "reports/relative_risk"
    if (measurement["status"] != "INSUFFICIENT_MEASUREMENT" or metrics.get("status") != "UNEVALUABLE"
            or metrics.get("whole_wave_aborted") is not True or metrics.get("leads") != []
            or metrics.get("hypothesis_count") != 2 or metrics.get("cumulative_hypothesis_count") != 110):
        raise AssertionError("Confirmed measurement failure must retain both unevaluable hypotheses")
    if any((output / name).exists() for name in ("features.parquet", "targets.parquet", "forecasts.parquet", "fits.json")):
        raise AssertionError("No feature, target or model artifact may follow failed measurement gate")
    if [(row["candidate"], row["control"]) for row in metrics["rows"]] != list(COMPARISONS):
        raise AssertionError("Both measurement-gated hypotheses must remain")
    for row in metrics["rows"]:
        if (row["study"] != "relative_risk" or row["horizon"] != 1 or row["score"] != "mse" or row["phases"] != []
                or row["status"] not in {"INSUFFICIENT_MEASUREMENT", "INSUFFICIENT_DATA"}
                or any(row[key] != 1. for key in ("p_conservative", "p_holm_wave", "p_holm_cumulative"))):
            raise AssertionError("Measurement failure cannot retain fitted results or informative p-values")
    same_tree(json.loads((report / "failure.json").read_text()), metrics, "Canonical measurement failure publication")
    return {"status": "VERIFIED_INSUFFICIENT_MEASUREMENT", "measurement_audit": measurement,
            "new_hypotheses_verified": 2, "cumulative_hypotheses_verified": 110, "leads": [],
            "ledger_events_verified": verify_ledger(root, metrics, prior, "unevaluable")}


def verify(root=ROOT):
    root = Path(root)
    output, report = root / "data/relative_risk", root / "reports/relative_risk"
    protocol_path = root / "relative_risk.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    validate_protocol(protocol)
    manifest = json.loads((report / "manifest.json").read_text())
    verify_manifest_coverage(root, protocol, manifest)
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Frozen relative-risk protocol changed")
    for group in ("code", "inputs", "preserved"):
        for path, expected in manifest[group].items():
            if digest(root / path) != expected:
                raise AssertionError("Frozen source, code or prior artifact changed: " + path)
    qqq, spx, iv, source_audit = load_source_tables(root, protocol)
    same_tree(json.loads((output / "source_audit.json").read_text()), source_audit, "Independent paired raw source provenance")
    measured = measurement_audit(qqq, spx)
    same_tree(json.loads((output / "measurement_audit.json").read_text()), measured, "Full-reference measurement gate before any sample mask")
    metrics = json.loads((report / "metrics.json").read_text())
    if metrics["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Relative-risk metric protocol identity differs")
    if measured["status"] == "INSUFFICIENT_MEASUREMENT":
        result = verify_terminal_measurement(root, measured, metrics, inherited_rows(root, protocol))
    else:
        require_measurement(measured)
        if metrics.get("status") == "UNEVALUABLE":
            raise AssertionError("Measurement gate passed but completed fit/scoring outcome is absent")
        features, targets = feature_target_tables(qqq, spx, iv)
        for filename, expected, numeric in (("features", features, ALL_FEATURES), ("targets", targets, ("y",))):
            actual = pd.read_parquet(output / (filename + ".parquet"))
            if not actual.index.equals(expected.index) or tuple(actual.columns) != tuple(expected.columns):
                raise AssertionError("Independent relative-risk table schema or dates differ")
            same(actual.loc[:, numeric], expected.loc[:, numeric], "Independent raw marginal/joint input and target cells", rtol=1e-10, atol=1e-12)
            for column in set(expected) - set(numeric):
                if not actual[column].equals(expected[column]):
                    raise AssertionError("Independent relative-risk table maturity differs")
        forecasts = pd.read_parquet(output / "forecasts.parquet")
        fits = json.loads((output / "fits.json").read_text())
        if metrics["evidence_class"] != protocol["evidence_class"]:
            raise AssertionError("Exploratory ETF/index evidence classification differs")
        result = {"status": "VERIFIED", "raw_feature_rows_verified": len(features), "raw_feature_columns_verified": len(ALL_FEATURES),
                  "measurement_audit": measured, "forecast_reconstruction": verify_forecasts(features, targets, forecasts, fits, protocol),
                  "inference": verify_metrics(root, forecasts, protocol, metrics),
                  "ledger_events_verified": verify_ledger(root, metrics, metrics["inherited_rows"], "evaluated")}
    result.update(protocol_sha256=digest(protocol_path), verifier_sha256=digest(Path(__file__)))
    result["limitations"] = ["QQQ ETF versus SPX price-index archival intraday OHLC risk; no exact Nasdaq100-index or portfolio-variance claim.",
                            "Passing the measurement gate removes floor dependence, not proxy bias, ETF/index differences or source revisions.",
                            "Reused history and back-calculated early VIX9D inputs remain exploratory limitations."]
    (report / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def invalidate_publication(root, error):
    report = Path(root) / "reports/relative_risk"
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
    failure = {"status": "UNEVALUABLE", "whole_wave_aborted": True, "leads": [], "hypothesis_count": 2,
               "cumulative_hypothesis_count": 110, "protocol_sha256": protocol_hash,
               "rows": [{"study": "relative_risk", "candidate": candidate, "control": control, "score": "mse", "horizon": 1,
                         "phases": [], "p_conservative": 1., "p_holm_wave": 1., "p_holm_cumulative": 1.,
                         "status": "INVALID_RUN", "verdict": "UNEVALUABLE", "error": message} for candidate, control in COMPARISONS]}
    serialized = json.dumps(failure, indent=2, allow_nan=False) + "\n"
    path.write_text(serialized)
    (report / "failure.json").write_text(serialized)
    (report / "results.md").write_text("# QQQ–SPX relative intraday risk\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n" + message + "\n")
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
