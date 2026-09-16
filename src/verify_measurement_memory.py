"""Independent reconstruction of the frozen measurement-memory study.

New producer functions are never imported. Frozen independent BFGS, scoring,
HAC and explicit circular-bootstrap primitives are reused with unchanged
predeclared tolerances: producer KKT1e-8, independent KKT2e-8, coefficient
absolute1e-6 and forecast relative1e-6/absolute1e-12.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .verify_international_volatility import same_tree
from .verify_iterative_signal_search import digest, holm, same
from .verify_macro_overnight import (
    COEFFICIENT_ATOL,
    FORECAST_RTOL,
    PRODUCER_GRADIENT_TOLERANCE,
    penalized_objective,
    proper_score,
    second_moment_prediction,
)
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = Path(__file__).resolve().parents[1]
BASE = ("const", "lq_d", "lq_w", "lq_m", "lvix", "livshape", "lvvix", "neg_d", "neg_w", "neg_m",
        "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
ALL_FEATURES = BASE + ("width", "quality_memory")
MODELS = ("mean", "baseline", "width", "quality")
MEASURES = ("qmle", "rv5", "rv15")
TARGET_COLUMNS = tuple(f"y_{name}" for name in MEASURES)
CONTROLS = {"width": ("baseline", "mean"), "quality": ("baseline", "mean", "width")}
COMPARISONS = tuple((candidate, control, measure) for candidate in CONTROLS
                    for control in CONTROLS[candidate] for measure in MEASURES)
ALPHA = .01
WAVE_ALPHA = 1 / 600


def parse_source(raw, cutoff="2025-10-20", expected_sha256=None):
    """Read date metadata globally; parse numerical tokens only inside the fence."""
    actual_hash = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError("Risk Lab raw source hash mismatch")
    lines = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(lines) < 7 or lines[0:2] != ["SPY", "84398"]:
        raise ValueError("Risk Lab identity/header mismatch")
    if int(lines[3]) != len(lines) - 6:
        raise ValueError("Risk Lab declared source count differs")
    def date_token(token):
        if len(token) != 8 or not token.isdecimal():
            raise ValueError("Invalid Risk Lab date token")
        return pd.Timestamp(year=int(token[:4]), month=int(token[4:6]), day=int(token[6:]))
    dates, records, bounded_dates = [], [], []
    limit = pd.Timestamp(cutoff)
    for line in lines[6:]:
        tokens = line.split()
        if len(tokens) < 2:
            raise ValueError("Missing Risk Lab dated observation")
        day = date_token(tokens[1])
        dates.append(day)
        if day > limit:
            continue
        if len(tokens) != 12:
            raise ValueError("Bounded Risk Lab row must have12 fields")
        # Validate every bounded numerical measurement, even unused quote fields.
        numeric = [float(token) for token in tokens[2:]]
        records.append([numeric[0], numeric[3], numeric[4], numeric[2]])
        bounded_dates.append(day)
    dates = pd.DatetimeIndex(dates)
    if (dates.has_duplicates or not dates.is_monotonic_increasing
            or dates[0] != date_token(lines[4]) or dates[-1] != date_token(lines[5])):
        raise ValueError("Risk Lab date order/extent mismatch")
    frame = pd.DataFrame(records, columns=[*MEASURES, "ci_width"], index=pd.DatetimeIndex(bounded_dates, name="date"))
    return frame, {"symbol": "SPY", "identifier": "84398", "description": lines[2], "cutoff": cutoff,
                   "source_sha256": actual_hash, "source_bytes": len(raw), "source_rows": len(dates),
                   "bounded_rows": len(frame), "after_cutoff_rows": len(dates) - len(frame),
                   "source_first_date": str(dates[0].date()), "source_last_date": str(dates[-1].date()),
                   "numeric_post_cutoff_values_parsed": False}


def positive(values):
    return values.where(np.isfinite(values) & (values > 0))


def squared_native(values):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return positive(positive(values) ** 2)


def features_from_tables(daily, measurement, iv):
    dates = pd.DatetimeIndex(daily.index)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Unique ascending reference-session labels required")
    m, iv = measurement.reindex(dates), iv.reindex(dates)
    q = squared_native(m.qmle)
    f = pd.DataFrame(index=dates)
    f["const"] = 1.
    for label, width in (("d", 1), ("w", 5), ("m", 22)):
        f[f"lq_{label}"] = np.log(q.rolling(width, min_periods=width).mean()).shift(2)
    f["lvix"] = np.log(positive(iv.vix)).shift()
    f["livshape"] = np.log(positive(iv.vix9d) / positive(iv.vix)).shift()
    f["lvvix"] = np.log(positive(iv.vvix)).shift()
    negative = (-np.log(positive(daily["adj close"])).diff()).clip(lower=0)
    for label, width in (("d", 1), ("w", 5), ("m", 22)):
        f[f"neg_{label}"] = negative.rolling(width, min_periods=width).mean().shift()
    for day in range(1, 5):
        f[f"entry_dow_{day}"] = (dates.dayofweek == day).astype(float)
    width = m.ci_width.where(np.isfinite(m.ci_width) & (m.ci_width >= 0))
    f["width"] = np.log1p(width / positive(m.qmle)).shift(2)
    f["quality_memory"] = f.width * (f.lq_d - f.lq_m)
    f = f.replace([np.inf, -np.inf], np.nan)
    f["market_cutoff_date"] = pd.Series(dates, index=dates).shift()
    f["measurement_cutoff_date"] = pd.Series(dates, index=dates).shift(2)
    return f


def targets_from_table(measurement, dates):
    dates = pd.DatetimeIndex(dates)
    m = measurement.reindex(dates)
    target = pd.DataFrame({f"y_{name}": squared_native(m[name]).shift(-1) for name in MEASURES}, index=dates)
    target["target_end"] = pd.Series(dates, index=dates).shift(-1)
    target["available_date"] = pd.Series(dates, index=dates).shift(-3)
    return target


def valid_labels(targets):
    values = targets.loc[:, TARGET_COLUMNS].to_numpy(float)
    return pd.Series(np.isfinite(values).all(axis=1) & (values > 0).all(axis=1), index=targets.index)


def training_mask(complete, targets, origin, dates):
    cutoff = pd.Series(dates, index=dates).shift().loc[origin]
    return (complete & valid_labels(targets) & (targets.index < pd.Timestamp(origin))
            & targets.available_date.notna() & (targets.available_date <= cutoff))


def eligible_entries(features, targets, section):
    complete = pd.Series(np.isfinite(features.loc[:, ALL_FEATURES].to_numpy(float)).all(axis=1), index=features.index)
    in_phase = pd.Series(False, index=features.index)
    for phase in ("development", "evaluation"):
        first, last = section[phase]
        in_phase |= (features.index >= first) & (features.index <= last)
    application = features.index[complete & in_phase & (features.index >= section["origin_start"])
                                 & (features.index <= section["origin_end"])]
    valid = valid_labels(targets) & targets.available_date.notna() & (targets.available_date <= section["latest_target"])
    valid &= (features.index > section["development"][1]) | (targets.available_date <= section["development_target_available_by"])
    return application, application[valid.loc[application].to_numpy()], complete


def fit_primary(training, targets, application):
    return second_moment_prediction(training, targets["y_qmle"].to_numpy(float), application)


def primary_effect(phases):
    return (len(phases) == 2 and all(phase["delta"] <= -.005 for phase in phases)
            and len(phases[1]["stability"]) == 2 and all(row["delta"] < 0 for row in phases[1]["stability"]))


def alternative_effect(phases):
    return len(phases) == 2 and all(phase["delta"] < 0 for phase in phases)


def reconstruct(root, protocol):
    sources, section = protocol["sources"], protocol["index"]
    limit = pd.Timestamp(section["source_end"])
    if limit >= pd.Timestamp(section["sealed_start"]):
        raise AssertionError("Protected source boundary crossed")
    raw = (root / sources["risklab"]).read_bytes()
    m, source = parse_source(raw, section["source_end"], protocol["source_contract"]["risklab_sha256"])
    daily = pd.read_parquet(root / sources["daily"], columns=["adj close"], filters=[("date", "<=", limit)])
    if daily.index.max() > limit:
        raise AssertionError("Post-fence daily values decoded")
    def audit_market(path, index):
        return {"source_path": str(path), "source_sha256": digest(path), "bounded_rows": len(index),
                "first_date": str(index[0].date()) if len(index) else None,
                "last_date": str(index[-1].date()) if len(index) else None}
    implied, markets = {}, {"daily": audit_market(root / sources["daily"], daily.index)}
    for name in ("vix", "vix9d", "vvix"):
        column = "VVIX" if name == "vvix" else "CLOSE"
        csv = pd.read_csv(root / sources[name], dtype=str, usecols=["DATE", column])
        dates = pd.to_datetime(csv.DATE, format="%m/%d/%Y")
        within = dates <= limit
        index = pd.DatetimeIndex(dates.loc[within])
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise AssertionError("Ordered unique Cboe source dates required")
        implied[name] = pd.Series(pd.to_numeric(csv.loc[within, column]).to_numpy(), index=index)
        markets[name] = audit_market(root / sources[name], index)
    features = features_from_tables(daily, m, pd.DataFrame(implied))
    source.update({"source_path": str(root / sources["risklab"]),
                   "selected_field_map": {"qmle": 2, "rv5": 5, "rv15": 6, "ci_width": 4},
                   "invalid_selected_fields": {name: int((~(np.isfinite(m[name]) &
                       ((m[name] >= 0) if name == "ci_width" else (m[name] > 0)))).sum()) for name in m},
                   "units": "provider-native annualized volatility and reported interval half-width",
                   "normalization": "nonpositive/nonfinite volatility or negative/nonfinite width becomes missing; no clipping or filling",
                   "historical_publication_latency_verified": False})
    audit = {"risklab": source, "market_sources": markets, "numerical_source_end": section["source_end"],
             "calendar": {"definition": "bounded observed QQQ reference sessions; no certified historic SPY calendar",
                          "sessions": len(daily), "first_date": str(daily.index[0].date()), "last_date": str(daily.index[-1].date())},
             "measurement_calendar": {"missing_source_dates": [str(date.date()) for date in daily.index.difference(m.index)],
                                      "outside_reference_dates": [str(date.date()) for date in m.index.difference(daily.index)]},
             "timing_assumption": "Risk Lab inputs lag2; market inputs lag1; target next session; target maturity two additional reference sessions",
             "historical_publication_latency_verified": False}
    return features, targets_from_table(m, daily.index), audit, m


def verify_forecasts(features, targets, forecasts, fits, protocol):
    section = protocol["index"]
    required = {"origin", "model", "horizon", *TARGET_COLUMNS, "prediction", "target_end", "available_date",
                "market_cutoff_date", "measurement_cutoff_date", "fit_origin", "fit_cutoff_date", "train_n",
                "train_last_target", "train_last_available", "phase"}
    if not required.issubset(forecasts) or set(forecasts.model) != set(MODELS) or set(forecasts.horizon) != {1}:
        raise AssertionError("All four models and all three measured labels required")
    if forecasts.duplicated(["origin", "model", "horizon"]).any():
        raise AssertionError("Duplicated measurement forecasts")
    entries, scored, complete = eligible_entries(features, targets, section)
    for model in MODELS:
        rows = forecasts.loc[forecasts.model == model].sort_values("origin")
        if not pd.DatetimeIndex(rows.origin).equals(scored):
            raise AssertionError("Models must share independently reconstructed complete labels and features")
        same(rows.loc[:, TARGET_COLUMNS], targets.loc[scored, TARGET_COLUMNS], "All three measured squared targets", rtol=1e-10, atol=1e-14)
        for name in ("target_end", "available_date"):
            if not np.array_equal(rows[name].to_numpy(), targets.loc[scored, name].to_numpy()):
                raise AssertionError("Measurement target timing differs")
        for name in ("market_cutoff_date", "measurement_cutoff_date"):
            if not np.array_equal(rows[name].to_numpy(), features.loc[scored, name].to_numpy()):
                raise AssertionError("Independent market/measurement delays differ")
        phases = np.where(scored <= section["development"][1], "development", "evaluation")
        if not np.array_equal(rows.phase.to_numpy(), phases):
            raise AssertionError("Measurement phase assignment differs")
    expected_fits = entries[~entries.to_period("M").duplicated()]
    lookup = {pd.Timestamp(row["fit_origin"]): row for row in fits}
    if len(lookup) != len(fits) or set(lookup) != set(expected_fits):
        raise AssertionError("Refit origin must be first feature-complete observation regardless of its label")
    count, producer_max, independent_max = 0, 0., 0.
    for origin in expected_fits:
        record = lookup[origin]
        selected = training_mask(complete, targets, origin, features.index)
        training_origins = features.index[selected]
        n = int(selected.sum())
        application = entries[entries.to_period("M") == origin.to_period("M")]
        query = scored[scored.to_period("M") == origin.to_period("M")]
        cutoff = features.loc[origin, "market_cutoff_date"]
        if n < section["minimum_train"] or record["train_n"] != n or record["application_n"] != len(application):
            raise AssertionError("Common matured training/application counts differ")
        dates = {"fit_cutoff_date": cutoff, "train_first_origin": training_origins[0], "train_last_origin": training_origins[-1],
                 "train_last_target": targets.loc[selected, "target_end"].max(),
                 "train_last_available": targets.loc[selected, "available_date"].max()}
        if any(pd.Timestamp(record[key]) != value for key, value in dates.items()) or dates["train_last_available"] > cutoff:
            raise AssertionError("Measurement fit provenance violates declared label maturity")
        if set(record["model_audit"]) != set(MODELS):
            raise AssertionError("A registered model is missing")
        if (features.loc[selected, ALL_FEATURES[1:]].std(ddof=0) <= 1e-12).any():
            raise AssertionError("A common zero-scale feature cannot be removed")
        y = targets.loc[selected, "y_qmle"].to_numpy(float)
        for model in MODELS:
            columns = (("const",) if model == "mean" else BASE if model == "baseline"
                       else BASE + ("width",) if model == "width" else ALL_FEATURES)
            audit = record["model_audit"][model]
            if tuple(audit["columns"]) != columns or audit["train_n"] != n:
                raise AssertionError("Measured-memory model design differs")
            training = features.loc[selected, columns].to_numpy(float)
            apply = features.loc[query, columns].to_numpy(float)
            prediction, rebuilt = fit_primary(training, targets.loc[selected], apply)
            same(audit["train_mean"], rebuilt["train_mean"], "Only primary QMLE target scaling", rtol=1e-10, atol=1e-14)
            same(audit["beta"], rebuilt["beta"], "Independent BFGS measurement coefficients", rtol=0, atol=COEFFICIENT_ATOL)
            if model == "mean":
                same(audit["gradient_max_abs"], 0., "Historical mean optimum")
                replay = np.repeat(y.mean(), len(query))
            else:
                if audit["alpha"] != ALPHA or not 0 <= audit["iterations"] < 200 or audit["backtracks"] < 0:
                    raise AssertionError("Frozen penalty or optimizer budget differs")
                same(audit["means"], rebuilt["means"], "Training-only feature centers")
                same(audit["scales"], rebuilt["scales"], "Training-only population scales")
                z = np.c_[np.ones(n), (training[:, 1:] - rebuilt["means"]) / rebuilt["scales"]]
                beta = np.asarray(audit["beta"], float).copy()
                beta[0] -= np.log(y.mean())
                same(audit["scaled_beta"], beta, "Scaled intercept restores primary target units", rtol=1e-9, atol=1e-12)
                value, gradient = penalized_objective(beta, z, y / y.mean())
                maximum = float(np.max(np.abs(gradient)))
                if maximum > PRODUCER_GRADIENT_TOLERANCE + 1e-12:
                    raise AssertionError("Saved measurement coefficients fail independent KKT evaluation")
                same(audit["gradient_max_abs"], maximum, "Recorded first-order residual", rtol=1e-4, atol=1e-12)
                same(audit["objective"], value + np.log(y.mean()), "Saved penalized mean proper-score objective", rtol=1e-9, atol=1e-12)
                az = np.c_[np.ones(len(query)), (apply[:, 1:] - rebuilt["means"]) / rebuilt["scales"]]
                replay = np.exp(np.log(y.mean()) + az @ np.asarray(audit["scaled_beta"]))
                producer_max = max(producer_max, maximum)
            independent_max = max(independent_max, rebuilt["gradient_max_abs"])
            rows = forecasts.loc[(forecasts.model == model) & forecasts.origin.isin(query)].sort_values("origin")
            if not rows.fit_origin.eq(origin).all() or not rows.fit_cutoff_date.eq(cutoff).all() or not rows.train_n.eq(n).all():
                raise AssertionError("Recorded forecast fit provenance differs")
            for name in ("train_last_target", "train_last_available"):
                if not rows[name].eq(dates[name]).all():
                    raise AssertionError("Recorded forecast training maturity differs")
            same(rows.prediction, replay, "Exact frozen measurement prediction replay", rtol=1e-9, atol=1e-12)
            same(rows.prediction, prediction, "Independent primary-only BFGS forecasts", rtol=FORECAST_RTOL, atol=1e-12)
            count += len(rows)
    return {"forecasts_verified": count, "common_scored_origins": len(scored), "feature_complete_applications": len(entries),
            "monthly_fits_verified": len(expected_fits), "models_verified": len(MODELS), "fit_target": "qmle",
            "producer_kkt_max": producer_max, "independent_bfgs_kkt_max": independent_max}


def phase_statistics(panel, candidate, control, measure, phase, phase_code, protocol):
    section = protocol["index"]
    first, last = section[phase]
    selected = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
    if phase == "development":
        selected = selected.loc[selected.available_date <= section["development_target_available_by"]]
    wide = selected.pivot(index="origin", columns="model", values="prediction").sort_index()
    actual = selected.loc[selected.model == control].set_index("origin").reindex(wide.index)
    candidate_loss = proper_score(actual[f"y_{measure}"], wide[candidate])
    control_loss = proper_score(actual[f"y_{measure}"], wide[control])
    difference = candidate_loss - control_loss
    if len(difference) < max(protocol["inference"]["blocks"]):
        raise AssertionError("Insufficient retained phase observations for fixed block lengths")
    delta, hac = float(difference.mean()), independent_hac(difference)
    blocks = {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + phase_code * 10000 + width
        samples = explicit_bootstrap_means(difference, width, protocol["inference"]["bootstrap_draws"], seed)[:, 0]
        probability = float((1 + np.count_nonzero(np.abs(samples - delta) >= abs(delta))) / (len(samples) + 1))
        blocks[str(width)] = {"p": probability, "ci95": np.quantile(samples, [.025, .975]).tolist()}
    intervals = [hac["ci95"], *(item["ci95"] for item in blocks.values())]
    output = {"name": phase, "first_origin": str(wide.index[0].date()), "last_origin": str(wide.index[-1].date()),
              "n": len(wide), "delta": delta, "candidate_loss": float(candidate_loss.mean()), "control_loss": float(control_loss.mean()),
              "block_inference": blocks, "hac126": hac,
              "ci95_envelope": [min(value[0] for value in intervals), max(value[1] for value in intervals)],
              "p_conservative": max(hac["p"], *(item["p"] for item in blocks.values())), "annual": [], "stability": [],
              "nonoverlap_phases": [{"phase": 0, "n": len(wide), "delta": delta}]}
    for year in sorted(set(wide.index.year)):
        mask = wide.index.year == year
        output["annual"].append({"year": int(year), "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    if phase == "evaluation":
        for start, end in section["evaluation_stability"]:
            mask = (wide.index >= start) & (wide.index <= end)
            if not mask.any():
                raise AssertionError("Missing fixed evaluation stability slice")
            output["stability"].append({"start": start, "end": end, "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    return output


def inherited_rows(root, protocol):
    output = []
    for source in protocol["comparisons"]["inherited_sources"]:
        for row in json.loads((root / source).read_text())["rows"]:
            output.append({"study": row.get("study", source.split("/")[1]), "candidate": row["candidate"], "horizon": row["horizon"],
                           "control": row.get("control", "baseline"), "p_conservative": row["p_conservative"],
                           "source": source, "source_sha256": digest(root / source)})
    if len(output) != 84:
        raise AssertionError("All84 inherited hypotheses must remain")
    return output


def verify_metrics(root, panel, protocol, metrics):
    rows = metrics["rows"]
    if [(row["candidate"], row["control"], row["measure"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("All15 measured-memory hypotheses required in registered order")
    probabilities, effects = [], []
    for row in rows:
        if row["study"] != "measurement_memory" or row["horizon"] != 1:
            raise AssertionError("Measurement hypothesis identity differs")
        phases = [phase_statistics(panel, row["candidate"], row["control"], row["measure"], phase, code, protocol)
                  for code, phase in enumerate(("development", "evaluation"))]
        same_tree(row["phases"], phases, "Independent measurement paired inference")
        probability = max(phase["p_conservative"] for phase in phases)
        same(row["p_conservative"], probability, "Both-phase conservative p value")
        probabilities.append(probability)
        effects.append(primary_effect(phases) if row["measure"] == "qmle" else alternative_effect(phases))
    prior = inherited_rows(root, protocol)
    same_tree(metrics["inherited_rows"], prior, "All inherited hypotheses preserved")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-15:]
    passed = {}
    for number, row in enumerate(rows):
        same(row["p_holm_wave"], wave[number], "Fifteen-way wave Holm")
        same(row["p_holm_cumulative"], cumulative[number], "99-way cumulative Holm")
        primary = row["measure"] == "qmle"
        eligible = bool(effects[number] and (not primary or (wave[number] < WAVE_ALPHA and cumulative[number] < .05)))
        expected = ("PRIMARY_GATE_PASS" if primary else "MEASUREMENT_SIGN_PASS") if eligible else "DOES_NOT_QUALIFY"
        if row["verdict"] != expected:
            raise AssertionError("Primary statistical/effect gate or alternative sign gate differs")
        passed[row["candidate"], row["control"], row["measure"]] = eligible
    leads = [candidate for candidate, controls in CONTROLS.items()
             if all(passed[candidate, control, measure] for control in controls for measure in MEASURES)]
    if metrics["leads"] != leads or metrics["hypothesis_count"] != 15 or metrics["cumulative_hypothesis_count"] != 99:
        raise AssertionError("Candidate must pass every required primary control and alternative measurement")
    return {"new_hypotheses_verified": 15, "cumulative_hypotheses_verified": 99, "phase_comparisons_verified": 30,
            "bootstrap_runs_verified": 90, "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"], "leads": leads}


def verify(root=ROOT):
    root = Path(root)
    report, output = root / "reports/measurement_memory", root / "data/measurement_memory"
    protocol_file = root / "measurement_memory.yaml"
    protocol = yaml.safe_load(protocol_file.read_text())
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol_file):
        raise AssertionError("Registered measurement protocol changed")
    for group in ("code", "inputs", "preserved"):
        for path, expected in manifest[group].items():
            if digest(root / path) != expected:
                raise AssertionError("Frozen input, source or prior artifact changed: " + path)
    features, targets, audit, _ = reconstruct(root, protocol)
    for filename, expected, numeric in (("features", features, ALL_FEATURES), ("targets", targets, TARGET_COLUMNS)):
        saved = pd.read_parquet(output / f"{filename}.parquet")
        if not saved.index.equals(expected.index) or tuple(saved.columns) != tuple(expected.columns):
            raise AssertionError("Measurement table dates or schema differ: " + filename)
        same(saved.loc[:, numeric], expected.loc[:, numeric], "Independent " + filename, rtol=1e-10, atol=1e-12)
        for name in set(expected) - set(numeric):
            if not saved[name].equals(expected[name]):
                raise AssertionError("Independent cutoff/maturity dates differ: " + name)
    same_tree(json.loads((output / "source_audit.json").read_text()), audit, "Independent source and calendar audit")
    result = {"status": "VERIFIED", "protocol_sha256": digest(protocol_file), "verifier_sha256": digest(Path(__file__)),
              "feature_rows_verified": len(features), "feature_columns_verified": len(ALL_FEATURES),
              "source_reconstruction": {"source_rows": audit["risklab"]["source_rows"],
                                        "bounded_rows": audit["risklab"]["bounded_rows"],
                                        "post_fence_numeric_rows_parsed": 0,
                                        "missing_reference_dates": len(audit["measurement_calendar"]["missing_source_dates"])}}
    forecasts = pd.read_parquet(output / "forecasts.parquet")
    fits = json.loads((output / "fits.json").read_text())
    result["forecast_reconstruction"] = verify_forecasts(features, targets, forecasts, fits, protocol)
    metrics = json.loads((report / "metrics.json").read_text())
    if metrics["protocol_sha256"] != digest(protocol_file) or metrics["evidence_class"] != protocol["evidence_class"]:
        raise AssertionError("Measurement metric provenance differs")
    result["inference"] = verify_metrics(root, forecasts, protocol, metrics)
    ledger = [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines() if line]
    for event, expected in (("inherited", metrics["inherited_rows"]), ("evaluated", metrics["rows"])):
        recorded = [{key: value for key, value in row.items() if key != "event"} for row in ledger if row["event"] == event]
        same_tree(recorded, expected, "Measurement " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (len(ledger) != 114 or len(registered) != 15
            or [(row["candidate"], row["control"], row["measure"]) for row in registered] != list(COMPARISONS)
            or any(row["protocol_sha256"] != digest(protocol_file) or row["horizon"] != 1 for row in registered)):
        raise AssertionError("Complete84inherited+15registered+15evaluated ledger required")
    result["ledger_events_verified"] = {"inherited": 84, "registered": 15, "evaluated": 15}
    result["limitations"] = ["Retrospective source delays are assumptions; historical publication latency and revisions remain unverified.",
                            "Native squared annualized volatility measurements are not certified integrated daily variance.",
                            "Same-provider alternative measurements are not independent-source replication.",
                            "Adaptive historical reuse and missing stress-week observations remain limitations."]
    (report / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
