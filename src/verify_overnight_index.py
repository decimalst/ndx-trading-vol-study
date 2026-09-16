"""Independent raw reconstruction and inference audit for the overnight wave."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .verify_international_volatility import same_tree
from .verify_iterative_signal_search import digest, holm, ridge_prediction, same
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = Path(__file__).resolve().parents[1]
BASE = ("const", "on_d", "on_w", "on_m", "day_d", "day_w", "day_m", "lrv_d", "lrv_w", "lrv_m",
        "liv", "lvix", "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
BLOCKS = {"cross_signed": ("x_hyg", "x_tlt", "x_gld", "x_uso", "x_uup"),
          "volume_pressure": ("pressure_d", "pressure_w"), "iv_shape": ("term", "lvvix")}
ALL_FEATURES = BASE + tuple(column for columns in BLOCKS.values() for column in columns)
MODELS = ("mean", "baseline", *BLOCKS)


def overnight_targets(daily):
    factor = daily["adj close"] / daily.close
    adjusted_total = np.log(daily["adj close"]).diff()
    daytime = np.log(daily.close / daily.open)
    # The target is measured with next-session adjusted close and is therefore
    # treated as available at that close, even though its economic leg ends open.
    y = np.expm1((adjusted_total - daytime).shift(-1))
    jump = np.log(factor.shift(-1) / factor)
    date = pd.Series(daily.index, index=daily.index).shift(-1)
    return pd.DataFrame({"y": y, "target_end": date, "available_date": date,
                         "adjustment_jump": jump}, index=daily.index)


def prior_volume_z(volume):
    values = np.log(volume.where(np.isfinite(volume) & (volume > 0)))
    previous = values.shift()
    average = previous.rolling(252, min_periods=126).mean()
    scale = previous.rolling(252, min_periods=126).std(ddof=1)
    return (values - average) / scale.where(scale > 0)


def overnight_features(daily, cross, iv):
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise ValueError("Ordered unique QQQ sessions required")
    prices = daily[["open", "high", "low", "close", "adj close"]]
    if not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError("Finite positive market prices required")
    day = np.log(daily.close / daily.open)
    overnight = np.log(daily["adj close"]).diff() - day
    gk = (.5 * np.log(daily.high / daily.low) ** 2 - (2 * np.log(2) - 1) * day ** 2).clip(lower=1e-10)
    variance = gk + np.log(daily.open / daily.close.shift()) ** 2
    result = pd.DataFrame({"const": 1.}, index=daily.index)
    for label, span in (("d", 1), ("w", 5), ("m", 22)):
        result[f"on_{label}"] = overnight.rolling(span).mean().shift()
        result[f"day_{label}"] = day.rolling(span).mean().shift()
        result[f"lrv_{label}"] = np.log(variance.rolling(span).mean()).shift()
    aligned = iv.reindex(daily.index).where(lambda value: np.isfinite(value) & (value > 0))
    result["liv"] = np.log(aligned.vxn).shift()
    result["lvix"] = np.log(aligned.vix).shift()
    result["term"] = np.log(aligned.vix9d / aligned.vix).shift()
    result["lvvix"] = np.log(aligned.vvix).shift()
    for asset in ("hyg", "tlt", "gld", "uso", "uup"):
        values = cross[asset].reindex(daily.index)
        result[f"x_{asset}"] = np.log(values.where(np.isfinite(values) & (values > 0))).diff().shift()
    pressure = day * prior_volume_z(daily.volume)
    result["pressure_d"] = pressure.shift()
    result["pressure_w"] = pressure.rolling(5).mean().shift()
    for weekday in range(1, 5):
        result[f"entry_dow_{weekday}"] = (daily.index.dayofweek == weekday).astype(float)
    return result.loc[:, ALL_FEATURES].replace([np.inf, -np.inf], np.nan)


def training_mask(complete, targets, fit_entry, sessions):
    sessions = pd.DatetimeIndex(sessions)
    position = sessions.get_indexer([pd.Timestamp(fit_entry)])[0]
    if position < 1:
        raise ValueError("Fit entry requires a previous actual session")
    cutoff = sessions[position - 1]
    return (complete & np.isfinite(targets.y) & (targets.index < pd.Timestamp(fit_entry))
            & targets.available_date.notna() & (targets.available_date <= cutoff))


def measurement_mask(jumps):
    jumps = np.asarray(jumps, float)
    return np.isfinite(jumps) & (np.abs(jumps) <= 1e-5)


def reconstruct(root, protocol):
    cutoff = pd.Timestamp(protocol["index"]["source_end"])
    paths = protocol["sources"]
    daily = pd.read_parquet(root / paths["daily"], filters=[("date", "<=", cutoff)])
    cross = pd.read_parquet(root / paths["cross"], filters=[("date", "<=", cutoff)])
    inputs = {}
    for name in ("vxn", "vix", "vix9d", "vvix"):
        column = "VVIX" if name == "vvix" else "CLOSE"
        raw = pd.read_csv(root / paths[name], usecols=["DATE", column])
        dates = pd.to_datetime(raw.DATE, format="%m/%d/%Y")
        mask = dates <= cutoff
        series = pd.Series(raw.loc[mask, column].to_numpy(float), index=pd.DatetimeIndex(dates.loc[mask]))
        if series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise AssertionError("Cboe dates must be ordered and unique")
        inputs[name] = series
    features = overnight_features(daily, cross, pd.DataFrame(inputs))
    return features, overnight_targets(daily)


def eligible_origins(features, targets, section):
    complete = pd.Series(np.isfinite(features.to_numpy(float)).all(axis=1), index=features.index)
    valid = (complete & np.isfinite(targets.y) & targets.available_date.notna()
             & (targets.target_end <= pd.Timestamp(section["latest_target"]))
             & (features.index >= pd.Timestamp(section["origin_start"]))
             & (features.index <= pd.Timestamp(section["origin_end"])))
    before_evaluation = features.index < pd.Timestamp(section["evaluation"][0])
    valid &= ~before_evaluation | (targets.available_date <= pd.Timestamp(section["development_target_available_by"]))
    return pd.DatetimeIndex(features.index[valid]), complete


def verify_forecasts(features, targets, forecasts, fits, protocol):
    section = protocol["index"]
    required = {"origin", "horizon", "model", "feature_cutoff_date", "target_end", "available_date", "y", "prediction",
                "adjustment_event", "adjustment_log_change", "fit_origin", "fit_cutoff_date", "train_n",
                "train_last_target", "train_last_available"}
    if not required.issubset(forecasts) or set(forecasts.model) != set(MODELS) or set(forecasts.horizon) != {1}:
        raise AssertionError("Explicit overnight forecast schema differs")
    if forecasts.duplicated(["origin", "model", "horizon"]).any():
        raise AssertionError("Duplicated overnight forecasts")
    origins, complete = eligible_origins(features, targets, section)
    cutoff_dates = pd.Series(features.index, index=features.index).shift()
    for model in MODELS:
        rows = forecasts.loc[forecasts.model == model].sort_values("origin")
        if not pd.DatetimeIndex(rows.origin).equals(origins):
            raise AssertionError("All overnight models require identical complete origins")
        expected = targets.loc[origins]
        same(rows.y, expected.y, "Overnight proxy targets", rtol=1e-10, atol=1e-13)
        same(rows.adjustment_log_change, expected.adjustment_jump, "Retrospective adjustment-factor changes", rtol=1e-10, atol=1e-13)
        if not np.array_equal(rows.adjustment_event.to_numpy(), (~measurement_mask(expected.adjustment_jump)).astype(bool)):
            raise AssertionError("Retrospective adjustment-event flag differs")
        for key in ("target_end", "available_date"):
            if not np.array_equal(rows[key].to_numpy(), expected[key].to_numpy()):
                raise AssertionError("Overnight label availability dates differ")
        if not np.array_equal(rows.feature_cutoff_date.to_numpy(), cutoff_dates.loc[origins].to_numpy()):
            raise AssertionError("Entry predictors did not stop at the previous close")
    mapped = {pd.Timestamp(row["fit_origin"]): row for row in fits}
    used, count = set(), 0
    for month in origins.to_period("M").unique():
        query = origins[origins.to_period("M") == month]
        entry, used_cutoff = query[0], cutoff_dates.loc[query[0]]
        used.add(entry)
        if entry not in mapped:
            raise AssertionError("Missing monthly overnight fit")
        record = mapped[entry]
        mask = training_mask(complete, targets, entry, features.index)
        training_origins = features.index[mask]
        n = int(mask.sum())
        if n < section["minimum_train"] or record["train_n"] != n:
            raise AssertionError("Overnight training sample differs")
        dates = {"fit_cutoff_date": used_cutoff, "train_first_origin": training_origins[0],
                 "train_last_origin": training_origins[-1], "train_last_target": targets.loc[mask, "target_end"].max(),
                 "train_last_available": targets.loc[mask, "available_date"].max()}
        if any(pd.Timestamp(record[name]) != value for name, value in dates.items()):
            raise AssertionError("Overnight fit information cutoff or training dates differ")
        if dates["train_last_available"] > used_cutoff:
            raise AssertionError("Overnight training uses unavailable adjusted labels")
        y = targets.loc[mask, "y"].to_numpy(float)
        if set(record["model_audit"]) != set(MODELS):
            raise AssertionError("Overnight fit omitted a model")
        for model in MODELS:
            recorded = record["model_audit"][model]
            columns = ("const",) if model == "mean" else BASE + BLOCKS.get(model, ())
            if tuple(recorded["columns"]) != columns or recorded["train_n"] != n:
                raise AssertionError("Overnight ridge model design or sample differs")
            expected_alpha = 0. if model == "mean" else section["ridge_alpha"]
            if recorded["alpha"] != expected_alpha or not np.isfinite(recorded["gradient_max_abs"]) or recorded["gradient_max_abs"] > 1e-10:
                raise AssertionError("Overnight saved ridge objective or first-order residual differs")
            if model == "mean":
                prediction = np.repeat(y.mean(), len(query))
                means, scales, beta = np.array([0.]), np.array([1.]), np.array([y.mean()])
            else:
                training = features.loc[mask, columns].to_numpy(float)
                apply = features.loc[query, columns].to_numpy(float)
                prediction, audit = ridge_prediction(training, y, apply, alpha=section["ridge_alpha"])
                means, scales = np.r_[0., audit["means"]], np.r_[1., audit["scales"]]
                beta = np.r_[audit["intercept"], audit["beta"]]
                standardized = (training - means) / scales
                gradient = 2 * standardized.T @ (standardized @ beta - y) / n
                gradient[1:] += 2 * section["ridge_alpha"] * beta[1:]
                if np.max(np.abs(gradient)) > 1e-10:
                    raise AssertionError("Independent overnight ridge fit violates mean-loss objective")
            same(recorded["means"], means, "Overnight training-only centers")
            same(recorded["scales"], scales, "Overnight training-only scales")
            same(recorded["beta"], beta, "Overnight ridge coefficients", rtol=1e-7, atol=1e-12)
            rows = forecasts.loc[(forecasts.model == model) & forecasts.origin.isin(query)].sort_values("origin")
            if not rows.fit_origin.eq(entry).all() or not rows.fit_cutoff_date.eq(used_cutoff).all() or not rows.train_n.eq(n).all():
                raise AssertionError("Overnight row-level fit provenance differs")
            for key in ("train_last_target", "train_last_available"):
                if not rows[key].eq(dates[key]).all():
                    raise AssertionError("Overnight row-level label maturity differs")
            same(rows.prediction, prediction, "Every independently reconstructed overnight forecast", rtol=1e-8, atol=1e-12)
            count += len(rows)
    if set(mapped) != used or len(mapped) != len(fits):
        raise AssertionError("Unexpected overnight monthly fits")
    return {"forecasts_verified": count, "monthly_fits_verified": len(used), "common_origins": len(origins),
            "models_verified": len(MODELS), "adjustment_flagged_origins": int((~measurement_mask(targets.loc[origins, "adjustment_jump"])).sum())}


def phase_statistics(panel, candidate, control, phase, phase_code, protocol):
    section = protocol["index"]
    start, end = section[phase]
    block = panel.loc[(panel.origin >= start) & (panel.origin <= end)].copy()
    if phase == "development":
        block = block.loc[block.available_date <= pd.Timestamp(section["development_target_available_by"])]
    wide = block.pivot(index="origin", columns="model", values="prediction").sort_index()
    base = block.loc[block.model == control].set_index("origin").reindex(wide.index)
    actual = base.y.to_numpy(float)
    control_loss = (actual - wide[control].to_numpy()) ** 2
    candidate_loss = (actual - wide[candidate].to_numpy()) ** 2
    difference = candidate_loss - control_loss
    if len(wide) < max(protocol["inference"]["blocks"]):
        raise AssertionError("Too few overnight phase observations")
    mean = float(difference.mean())
    hac = independent_hac(difference)
    blocks = {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + phase_code * 10000 + width
        samples = explicit_bootstrap_means(difference, width, protocol["inference"]["bootstrap_draws"], seed)[:, 0]
        p = float((1 + np.count_nonzero(np.abs(samples - mean) >= abs(mean))) / (len(samples) + 1))
        blocks[str(width)] = {"p": p, "ci95": np.quantile(samples, [.025, .975]).tolist()}
    intervals = [hac["ci95"], *(item["ci95"] for item in blocks.values())]
    mask = measurement_mask(base.adjustment_log_change.to_numpy(float))
    if not mask.any():
        raise AssertionError("Empty unflagged overnight measurement subset")
    sensitivity = {"n": int(mask.sum()), "delta": float(difference[mask].mean()),
                   "control_loss": float(control_loss[mask].mean()), "candidate_loss": float(candidate_loss[mask].mean()),
                   "improvement_pct": float(-100 * difference[mask].mean() / control_loss[mask].mean())}
    result = {"name": phase, "first_origin": str(wide.index[0].date()), "last_origin": str(wide.index[-1].date()),
              "n": len(wide), "delta": mean, "control_loss": float(control_loss.mean()),
              "candidate_loss": float(candidate_loss.mean()), "improvement_pct": float(-100 * mean / control_loss.mean()),
              "block_inference": blocks, "hac126": hac,
              "ci95_envelope": [min(interval[0] for interval in intervals), max(interval[1] for interval in intervals)],
              "p_conservative": max(hac["p"], *(item["p"] for item in blocks.values())),
              "annual": [], "stability": [], "nonoverlap_phases": [{"phase": 0, "n": len(wide), "delta": mean}],
              "action_sensitivity": sensitivity}
    for year in sorted(set(wide.index.year)):
        selected = wide.index.year == year
        result["annual"].append({"year": int(year), "n": int(selected.sum()), "delta": float(difference[selected].mean())})
    if phase == "evaluation":
        for first, last in section["evaluation_stability"]:
            selected = (wide.index >= first) & (wide.index <= last)
            result["stability"].append({"start": first, "end": last, "n": int(selected.sum()), "delta": float(difference[selected].mean())})
    return result


def verify_metrics(root, forecasts, protocol, recorded):
    rows = recorded["rows"]
    keys = [(row["candidate"], row["control"]) for row in rows]
    expected_keys = {(candidate, control) for candidate in BLOCKS for control in ("baseline", "mean")}
    if len(keys) != 6 or len(set(keys)) != 6 or set(keys) != expected_keys:
        raise AssertionError("All six overnight comparisons must remain in the family")
    p_new, qualify = [], []
    for row in rows:
        if row["study"] != "overnight" or row["horizon"] != 1:
            raise AssertionError("Overnight hypothesis identity differs")
        expected = [phase_statistics(forecasts, row["candidate"], row["control"], phase, code, protocol)
                    for code, phase in enumerate(("development", "evaluation"))]
        same_tree(row["phases"], expected, "Overnight independent phase diagnostics and inference")
        value = max(phase["p_conservative"] for phase in expected)
        same(row["p_conservative"], value, "Overnight two-phase conjunction p")
        p_new.append(value)
        qualify.append(all(phase["improvement_pct"] >= .25 and phase["action_sensitivity"]["delta"] < 0 for phase in expected)
                       and all(part["delta"] < 0 for part in expected[1]["stability"]))
    inherited = []
    for path in protocol["comparisons"]["inherited_sources"]:
        for row in json.loads((root / path).read_text())["rows"]:
            inherited.append({"study": row.get("study", path.split("/")[1]), "horizon": row["horizon"],
                              "candidate": row["candidate"], "control": row.get("control", "baseline"),
                              "p_conservative": row["p_conservative"], "source": path, "source_sha256": digest(root / path)})
    if len(inherited) != 72:
        raise AssertionError("All 72 earlier comparisons must be retained")
    same_tree(recorded["inherited_rows"], inherited, "Complete inherited overnight trial family")
    wave = holm(p_new)
    cumulative = holm([row["p_conservative"] for row in inherited] + p_new)[-6:]
    passed = {}
    for index, row in enumerate(rows):
        same(row["p_holm_wave"], wave[index], "Six-way overnight Holm correction")
        same(row["p_holm_cumulative"], cumulative[index], "78-way cumulative Holm correction")
        passed[(row["candidate"], row["control"])] = bool(qualify[index] and wave[index] < .05 / 12 and cumulative[index] < .05)
        if row["verdict"] != ("EXPLORATORY_LEAD" if passed[(row["candidate"], row["control"])] else "DOES_NOT_QUALIFY"):
            raise AssertionError("Overnight comparison gates differ")
    leads = [{"candidate": candidate, "horizon": 1} for candidate in BLOCKS
             if all(passed[(candidate, control)] for control in ("baseline", "mean"))]
    if recorded["leads"] != leads:
        raise AssertionError("Overnight signals must pass both controls")
    return {"phase_comparisons_verified": 12, "bootstrap_runs_verified": 36,
            "measurement_sensitivities_verified": 12, "new_hypotheses_verified": 6,
            "cumulative_hypotheses_verified": 78, "leads": leads}


def verify(root=ROOT):
    root = Path(root)
    report, output = root / "reports/overnight_index", root / "data/overnight_index"
    protocol_path = root / "overnight_index.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Overnight protocol changed after registration")
    for group in ("code", "inputs", "preserved"):
        for path, expected in manifest[group].items():
            if digest(root / path) != expected:
                raise AssertionError(f"Frozen overnight input or earlier artifact changed: {path}")
    features, targets = reconstruct(root, protocol)
    produced = pd.read_parquet(output / "features.parquet")
    if not produced.index.equals(features.index) or tuple(produced.columns) != ALL_FEATURES + ("feature_cutoff_date",):
        raise AssertionError("Overnight feature schema or dates differ")
    same(produced.loc[:, ALL_FEATURES], features, "Every overnight input feature", rtol=1e-10, atol=1e-12)
    expected_cutoffs = pd.Series(features.index, index=features.index).shift()
    if not np.array_equal(produced.feature_cutoff_date.to_numpy(), expected_cutoffs.to_numpy(), equal_nan=True):
        raise AssertionError("Overnight feature cutoff audit differs")
    forecasts = pd.read_parquet(output / "forecasts.parquet")
    fits = json.loads((output / "fits.json").read_text())
    result = {"status": "VERIFIED", "protocol_sha256": digest(protocol_path), "verifier_sha256": digest(Path(__file__)),
              "feature_rows_verified": len(features), "feature_columns_verified": len(ALL_FEATURES),
              "forecast_reconstruction": verify_forecasts(features, targets, forecasts, fits, protocol)}
    metrics = json.loads((report / "metrics.json").read_text())
    if metrics["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Overnight metrics protocol differs")
    result["inference"] = verify_metrics(root, forecasts, protocol, metrics)
    ledger = [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines() if line]
    for event, expected in (("inherited", metrics["inherited_rows"]), ("evaluated", metrics["rows"])):
        actual = [{key: value for key, value in row.items() if key != "event"} for row in ledger if row["event"] == event]
        same_tree(actual, expected, f"Overnight {event} trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if len(ledger) != 84 or len(registered) != 6:
        raise AssertionError("Overnight trial ledger is incomplete")
    identities = {(row["candidate"], row["control"], row["horizon"]) for row in registered}
    if identities != {(row["candidate"], row["control"], row["horizon"]) for row in metrics["rows"]}:
        raise AssertionError("Overnight registered hypotheses differ")
    if any(row["protocol_sha256"] != digest(protocol_path) for row in registered):
        raise AssertionError("Overnight ledger protocol differs")
    result["ledger_events_verified"] = {"inherited": 72, "registered": 6, "evaluated": 6}
    result["limitation"] = "Historical reuse and unknown vintages remain exploratory. The adjusted overnight proxy is not verified cash profit or execution PnL."
    (report / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
