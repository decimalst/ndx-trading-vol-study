"""Independent lagged-index, training-centered hinge, ridge and inference audit.

No imports of wave-six producer functions. Ridge is independently fitted by
augmented least squares, with pre-fit tolerances: saved KKT1e-10, coefficients
relative1e-7/absolute1e-12, independent forecasts relative1e-8/absolute1e-12.
Frozen independent explicit bootstrap and HAC primitives are reused read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import lstsq

from .verify_international_volatility import same_tree
from .verify_iterative_signal_search import digest, holm, same
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = Path(__file__).resolve().parents[1]
RAW = ("const", "I", "R", "ret_d", "ret_w", "ret_m", "ret_q", "lr_d", "lr_w", "term", "lvvix",
       "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
BASE = RAW + ("I_square", "R_square")
ALL_FEATURES = BASE + ("hinge",)
MODELS = ("mean", "baseline", "hinge")
HORIZONS = (21, 63)
COMPARISONS = tuple(("hinge", control, horizon) for horizon in HORIZONS for control in ("baseline", "mean"))
ALPHA = .01
WAVE_ALPHA = .05 / (6 * 7)


def validate_dates(index):
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates
            or not index.is_monotonic_increasing or index.tz is not None or not index.equals(index.normalize())):
        raise ValueError("Unique ascending normalized timezone-naive source dates required")


def raw_features(daily, iv):
    dates = pd.DatetimeIndex(daily.index)
    validate_dates(dates)
    validate_dates(iv.index)
    for values in (daily[["open", "high", "low", "close"]], iv[["vix", "vix9d", "vvix"]]):
        array = values.to_numpy(float)
        if np.isinf(array).any() or ((array <= 0) & np.isfinite(array)).any():
            raise ValueError("Observed market values must be finite and positive; missing values stay missing")
    if ((daily.high < daily.low) | (daily.high < daily.open) | (daily.high < daily.close)
            | (daily.low > daily.open) | (daily.low > daily.close)).any():
        raise ValueError("Malformed OHLC range")
    day = np.log(daily.close / daily.open)
    gk = np.maximum(.5 * np.log(daily.high / daily.low)**2 - (2 * np.log(2) - 1) * day**2, 1e-10)
    variance = gk + np.log(daily.open / daily.close.shift())**2
    returns = np.log(daily.close).diff()
    aligned = iv.reindex(dates)
    result = pd.DataFrame(index=dates)
    result["const"] = 1.
    result["I"] = np.log((aligned.vix / 100)**2).shift()
    result["R"] = np.log(252 * variance.rolling(22, min_periods=22).mean()).shift()
    for label, width in (("d", 1), ("w", 5), ("m", 22), ("q", 63)):
        result[f"ret_{label}"] = returns.rolling(width, min_periods=width).mean().shift()
    for label, width in (("d", 1), ("w", 5)):
        result[f"lr_{label}"] = np.log(252 * variance.rolling(width, min_periods=width).mean()).shift()
    result["term"] = np.log(aligned.vix9d / aligned.vix).shift()
    result["lvvix"] = np.log(aligned.vvix).shift()
    for day in range(1, 5):
        result[f"entry_dow_{day}"] = (dates.dayofweek == day).astype(float)
    result = result.replace([np.inf, -np.inf], np.nan)
    result["feature_cutoff_date"] = pd.Series(dates, index=dates).shift()
    return result


def future_returns(close, horizon):
    if horizon not in HORIZONS:
        raise ValueError("Registered horizon must be21 or63 sessions")
    values = np.log(close.shift(-horizon) / close)
    dates = pd.Series(close.index, index=close.index).shift(-horizon)
    return pd.DataFrame({"y": values.where(np.isfinite(values)), "target_end": dates, "available_date": dates}, index=close.index)


def derived_design(training, application):
    train, apply = training.loc[:, RAW].copy(), application.loc[:, RAW].copy()
    if len(train) < 2 or not np.isfinite(train).all().all() or not np.isfinite(apply).all().all():
        raise ValueError("Finite complete common raw design required")
    if not train.const.eq(1).all() or not apply.const.eq(1).all():
        raise ValueError("Constant intercept required")
    means = {"I_mean": float(train.I.mean()), "R_mean": float(train.R.mean()),
             "gap_mean": float((train.I - train.R).mean())}
    for frame in (train, apply):
        frame["I_square"] = (frame.I - means["I_mean"])**2
        frame["R_square"] = (frame.R - means["R_mean"])**2
        frame["hinge"] = np.maximum(frame.I - frame.R - means["gap_mean"], 0)
    scale = train.loc[:, ALL_FEATURES[1:]].std(ddof=0)
    if (not np.isfinite(train).all().all() or not np.isfinite(apply).all().all()
            or not np.isfinite(scale).all() or (scale <= 1e-12).any()):
        raise ValueError("Nonfinite or zero-scale derived design cannot be dropped")
    return train, apply, {"train_n": len(train), **means}


def ridge_prediction(training, y, application):
    x, q, a = np.asarray(training, float), np.asarray(y, float), np.asarray(application, float)
    if (x.ndim != 2 or a.ndim != 2 or x.shape[1] != a.shape[1] or len(x) < 2 or q.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(q).all() or not np.isfinite(a).all()
            or not (x[:, 0] == 1).all() or not (a[:, 0] == 1).all()):
        raise ValueError("Finite aligned signed targets and complete intercept designs required")
    mean, scale = np.r_[0., x[:, 1:].mean(axis=0)], np.r_[1., x[:, 1:].std(axis=0, ddof=0)]
    if not np.isfinite(scale).all() or (scale <= 1e-12).any():
        raise ValueError("Zero-scale ridge input cannot be removed")
    centered, query = (x - mean) / scale, (a - mean) / scale
    intercept, slopes = float(q.mean()), x.shape[1] - 1
    if slopes:
        # This augmented system implements mean MSE +alpha||slope||² without
        # forming the producer's Gram matrix or using its normal-equation solve.
        matrix = np.vstack([centered[:, 1:], np.sqrt(len(q) * ALPHA) * np.eye(slopes)])
        values = np.r_[q - intercept, np.zeros(slopes)]
        beta_slopes = lstsq(matrix, values, lapack_driver="gelsd")[0]
        beta = np.r_[intercept, beta_slopes]
        alpha = ALPHA
    else:
        beta, alpha = np.array([intercept]), 0.
    gradient = centered.T @ (centered @ beta - q) / len(q)
    gradient[1:] += alpha * beta[1:]
    maximum = float(np.max(np.abs(gradient)))
    if maximum > 1e-10:
        raise AssertionError("Independent augmented least-squares KKT exceeds fixed tolerance")
    return query @ beta, {"means": mean, "scales": scale, "beta": beta, "alpha": alpha,
                           "train_n": len(q), "gradient_max_abs": maximum}


def training_mask(complete, targets, origin, dates):
    dates = pd.DatetimeIndex(dates)
    position = dates.get_indexer([pd.Timestamp(origin)])[0]
    if position < 1:
        raise ValueError("Monthly fit requires prior observed session")
    return (complete & np.isfinite(targets.y) & (targets.index < pd.Timestamp(origin))
            & targets.available_date.notna() & (targets.available_date <= dates[position - 1]))


def eligible_entries(features, targets, section):
    complete = pd.Series(np.isfinite(features.loc[:, RAW].to_numpy(float)).all(axis=1), index=features.index)
    in_phase = pd.Series(False, index=features.index)
    for phase in ("development", "evaluation"):
        first, last = section[phase]
        in_phase |= (features.index >= first) & (features.index <= last)
    entries = features.index[complete & in_phase & (features.index >= section["origin_start"])
                              & (features.index <= section["origin_end"])]
    valid = np.isfinite(targets.y) & targets.available_date.notna() & (targets.available_date <= section["latest_target"])
    valid &= (features.index > section["development"][1]) | (targets.available_date <= section["development_target_available_by"])
    return entries, entries[valid.loc[entries].to_numpy()], complete


def nonoverlap_rows(origins, difference, calendar, horizon):
    positions = pd.DatetimeIndex(calendar).get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("Every scored origin must belong to full source calendar")
    values = np.asarray(difference, float)
    return [{"phase": offset, "n": int((positions % horizon == offset).sum()),
             "delta": float(values[positions % horizon == offset].mean()) if (positions % horizon == offset).any() else None}
            for offset in range(horizon)]


def effect_passes(phases, horizon):
    return (len(phases) == 2 and all(phase["gain_relative"] >= .0025 for phase in phases)
            and len(phases[1]["stability"]) == 2 and all(part["delta"] < 0 for part in phases[1]["stability"])
            and all(len(phase["nonoverlap_phases"]) == horizon
                    and {part["phase"] for part in phase["nonoverlap_phases"]} == set(range(horizon))
                    and all(part["n"] > 0 and part["delta"] < 0 for part in phase["nonoverlap_phases"]) for phase in phases))


def phase_statistics(panel, control, horizon, phase, code, protocol, calendar):
    section = protocol["index"]
    first, last = section[phase]
    selected = panel.loc[(panel.horizon == horizon) & (panel.origin >= first) & (panel.origin <= last)]
    if phase == "development":
        selected = selected.loc[selected.available_date <= section["development_target_available_by"]]
    wide = selected.pivot(index="origin", columns="model", values="prediction").sort_index()
    actual = selected.loc[selected.model == control].set_index("origin").reindex(wide.index).y
    candidate_loss, control_loss = (actual - wide.hinge).to_numpy()**2, (actual - wide[control]).to_numpy()**2
    difference = candidate_loss - control_loss
    if len(difference) <= max(protocol["inference"]["blocks"]) or not control_loss.mean() > 0:
        raise AssertionError("Insufficient inference observations or zero control loss")
    delta, hac = float(difference.mean()), independent_hac(difference, maxlags=504)
    blocks = {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + horizon * 1000000 + code * 10000 + width
        samples = explicit_bootstrap_means(difference, width, protocol["inference"]["bootstrap_draws"], seed)[:, 0]
        p = float((1 + np.count_nonzero(np.abs(samples - delta) >= abs(delta))) / (len(samples) + 1))
        blocks[str(width)] = {"p": p, "ci95": np.quantile(samples, [.025, .975]).tolist()}
    intervals = [hac["ci95"], *(part["ci95"] for part in blocks.values())]
    result = {"name": phase, "first_origin": str(wide.index[0].date()), "last_origin": str(wide.index[-1].date()),
              "n": len(wide), "delta": delta, "candidate_loss": float(candidate_loss.mean()), "control_loss": float(control_loss.mean()),
              "gain_relative": float(1 - candidate_loss.mean() / control_loss.mean()), "block_inference": blocks, "hac504": hac,
              "ci95_envelope": [min(item[0] for item in intervals), max(item[1] for item in intervals)],
              "p_conservative": max(hac["p"], *(part["p"] for part in blocks.values())), "annual": [], "stability": [],
              "nonoverlap_phases": nonoverlap_rows(wide.index, difference, calendar, horizon)}
    for year in sorted(set(wide.index.year)):
        mask = wide.index.year == year
        result["annual"].append({"year": int(year), "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    if phase == "evaluation":
        for start, end in section["evaluation_stability"]:
            mask = (wide.index >= start) & (wide.index <= end)
            if not mask.any():
                raise AssertionError("Fixed stability slice is empty")
            result["stability"].append({"start": start, "end": end, "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    return result


def reconstruct(root, protocol):
    limit = pd.Timestamp(protocol["index"]["source_end"])
    if limit >= pd.Timestamp(protocol["index"]["sealed_start"]):
        raise AssertionError("Protected source fence crossed")
    sources = protocol["sources"]
    daily = pd.read_parquet(root / sources["daily"], columns=["open", "high", "low", "close"],
                            filters=[("date", "<=", limit)])
    validate_dates(daily.index)
    if daily.index.max() > limit:
        raise AssertionError("Post-fence price data decoded")
    implied, market_audits = {}, {}
    for name in ("daily", "vix", "vix9d", "vvix"):
        path = root / sources[name]
        if name == "daily":
            index = daily.index
            source_frame = daily
        else:
            column = "VVIX" if name == "vvix" else "CLOSE"
            raw = pd.read_csv(path, dtype=str, usecols=["DATE", column])
            dates = pd.to_datetime(raw.DATE, format="%m/%d/%Y")
            selected = dates <= limit
            index = pd.DatetimeIndex(dates.loc[selected])
            validate_dates(index)
            implied[name] = pd.Series(pd.to_numeric(raw.loc[selected, column]).to_numpy(float), index=index)
            source_frame = implied[name].to_frame(name=name)
        market_audits[name] = {"source_path": str(path), "source_sha256": digest(path), "bounded_rows": len(index),
                              "first_date": str(index[0].date()) if len(index) else None,
                              "last_date": str(index[-1].date()) if len(index) else None,
                              "missing_values": {column: int(source_frame[column].isna().sum()) for column in source_frame},
                              "missing_reference_dates": [str(date.date()) for date in daily.index.difference(index)],
                              "outside_reference_dates": [str(date.date()) for date in index.difference(daily.index)]}
        if name != "daily":
            market_audits[name].update({"provider_date_field": "DATE", "provider_value_field": column})
    features = raw_features(daily, pd.DataFrame(implied))
    targets = {h: future_returns(daily.close, h) for h in HORIZONS}
    audit = {"sources": market_audits, "source_end": str(limit.date()), "sealed_start": protocol["index"]["sealed_start"],
             "reference_calendar": {"definition": "bounded observed SPX raw-OHLC sessions", "sessions": len(daily),
                                    "first_date": str(daily.index[0].date()), "last_date": str(daily.index[-1].date())},
             "raw_columns": list(RAW), "numeric_post_cutoff_values_parsed": False, "historical_vintage_certified": False,
             "timing_assumption": "all market predictors through prior observed SPX session close; current entry weekday only",
             "vix9d_caveat": "prelaunch January2011-October2013 values are back-calculated archival training inputs",
             "target_interpretation": "SPX cumulative raw-close log price return; excludes reinvested dividends and risk-free subtraction",
             "gap_interpretation": "log implied-versus-trailing-OHLC-proxy gap; not a measured variance risk premium"}
    return features, targets, audit


def verify_forecasts(features, targets_by_horizon, forecasts, fits, protocol):
    required = {"origin", "model", "horizon", "y", "prediction", "target_end", "available_date", "feature_cutoff_date",
                "fit_origin", "fit_cutoff_date", "train_n", "train_last_target", "train_last_available", "phase"}
    if not required.issubset(forecasts) or set(forecasts.model) != set(MODELS) or set(forecasts.horizon) != set(HORIZONS):
        raise AssertionError("All registered horizon/model forecast fields required")
    if forecasts.duplicated(["origin", "model", "horizon"]).any():
        raise AssertionError("Duplicate index-hinge forecasts")
    lookup = {(record["horizon"], pd.Timestamp(record["fit_origin"])): record for record in fits}
    if len(lookup) != len(fits):
        raise AssertionError("Duplicate horizon-specific monthly fit")
    section, used = protocol["index"], set()
    count, fits_count, transforms, maximum = 0, 0, 0, 0.
    by_horizon = {}
    for horizon in HORIZONS:
        target = targets_by_horizon[horizon]
        entries, scored, complete = eligible_entries(features, target, section)
        expected_fits = entries[~entries.to_period("M").duplicated()]
        by_horizon[str(horizon)] = {"common_scored_origins": len(scored), "feature_complete_applications": len(entries),
                                    "monthly_fits": len(expected_fits)}
        for model in MODELS:
            rows = forecasts.loc[(forecasts.horizon == horizon) & (forecasts.model == model)].sort_values("origin")
            if not pd.DatetimeIndex(rows.origin).equals(scored):
                raise AssertionError("Complete common score rows differ within a registered horizon")
            same(rows.y, target.loc[scored, "y"], "Independent raw price-index return targets", rtol=1e-10, atol=1e-13)
            for name in ("target_end", "available_date"):
                if not np.array_equal(rows[name].to_numpy(), target.loc[scored, name].to_numpy()):
                    raise AssertionError("Horizon-specific label dates differ")
            if not np.array_equal(rows.feature_cutoff_date.to_numpy(), features.loc[scored, "feature_cutoff_date"].to_numpy()):
                raise AssertionError("All market inputs must end at previous SPX session")
            expected_phases = np.where(scored <= section["development"][1], "development", "evaluation")
            if not np.array_equal(rows.phase.to_numpy(), expected_phases):
                raise AssertionError("Recorded phase boundaries differ")
        for origin in expected_fits:
            key = horizon, origin
            used.add(key)
            if key not in lookup:
                raise AssertionError("Missing raw-feature-selected monthly fit")
            record = lookup[key]
            application = entries[entries.to_period("M") == origin.to_period("M")]
            query = scored[scored.to_period("M") == origin.to_period("M")]
            mask = training_mask(complete, target, origin, features.index)
            train_dates, n = features.index[mask], int(mask.sum())
            cutoff = features.loc[origin, "feature_cutoff_date"]
            if n < section["minimum_train"] or record["train_n"] != n or record["application_n"] != len(application):
                raise AssertionError("Horizon-specific common train/application counts differ")
            dates = {"fit_cutoff_date": cutoff, "train_first_origin": train_dates[0], "train_last_origin": train_dates[-1],
                     "train_last_target": target.loc[mask, "target_end"].max(),
                     "train_last_available": target.loc[mask, "available_date"].max()}
            if any(pd.Timestamp(record[name]) != expected for name, expected in dates.items()) or dates["train_last_target"] > cutoff:
                raise AssertionError("Training information maturity or dates differ")
            transformed, apply_all, centers = derived_design(features.loc[mask], features.loc[application])
            for name, expected in centers.items():
                same(record["transform_audit"][name], expected, "Training-only horizon-specific transform " + name, rtol=1e-12, atol=1e-14)
            transforms += 1
            y = target.loc[mask, "y"].to_numpy(float)
            if set(record["model_audit"]) != set(MODELS):
                raise AssertionError("A registered model was removed from a horizon fit")
            for model in MODELS:
                columns = ("const",) if model == "mean" else BASE if model == "baseline" else ALL_FEATURES
                audit = record["model_audit"][model]
                if tuple(audit["columns"]) != columns or audit["train_n"] != n:
                    raise AssertionError("Training-centered model design or row count differs")
                training, apply = transformed.loc[:, columns], apply_all.loc[query, columns]
                prediction, rebuilt = ridge_prediction(training, y, apply)
                expected_alpha = 0. if model == "mean" else ALPHA
                if audit["alpha"] != expected_alpha:
                    raise AssertionError("Fixed mean-loss ridge penalty differs")
                for name in ("means", "scales"):
                    same(audit[name], rebuilt[name], "Independent population scaling " + name, rtol=1e-10, atol=1e-13)
                same(audit["beta"], rebuilt["beta"], "Independent augmented least-squares coefficients", rtol=1e-7, atol=1e-12)
                z = (training.to_numpy(float) - np.asarray(audit["means"])) / np.asarray(audit["scales"])
                beta = np.asarray(audit["beta"], float)
                gradient = z.T @ (z @ beta - y) / n
                gradient[1:] += expected_alpha * beta[1:]
                kkt = float(np.max(np.abs(gradient)))
                if kkt > 1e-10 or not np.isfinite(audit["gradient_max_abs"]) or audit["gradient_max_abs"] > 1e-10:
                    raise AssertionError("Saved coefficients fail independent mean-loss ridge KKT")
                maximum = max(maximum, kkt)
                replay = ((apply.to_numpy(float) - np.asarray(audit["means"])) / np.asarray(audit["scales"])) @ beta
                rows = forecasts.loc[(forecasts.horizon == horizon) & (forecasts.model == model)
                                     & forecasts.origin.isin(query)].sort_values("origin")
                if not rows.fit_origin.eq(origin).all() or not rows.fit_cutoff_date.eq(cutoff).all() or not rows.train_n.eq(n).all():
                    raise AssertionError("Forecast fit provenance differs")
                for name in ("train_last_target", "train_last_available"):
                    if not rows[name].eq(dates[name]).all():
                        raise AssertionError("Forecast training-label audit differs")
                same(rows.prediction, replay, "Saved parameter prediction replay", rtol=1e-10, atol=1e-12)
                same(rows.prediction, prediction, "Every independently refitted index-hinge forecast", rtol=1e-8, atol=1e-12)
                count += len(rows)
            fits_count += 1
    if set(lookup) != used:
        raise AssertionError("Unregistered horizon or monthly fit appeared")
    return {"forecasts_verified": count, "monthly_fits_verified": fits_count, "training_transform_fits_verified": transforms,
            "models_verified": len(MODELS), "horizons": by_horizon, "producer_kkt_max": maximum}


def inherited_rows(root, protocol):
    rows = []
    for source in protocol["comparisons"]["inherited_sources"]:
        for source_row_index, record in enumerate(json.loads((root / source).read_text())["rows"]):
            rows.append({"study": record.get("study", Path(source).parent.name or Path(source).stem), "candidate": record["candidate"],
                         "control": record.get("control", "baseline"), "horizon": record["horizon"],
                         "p_conservative": record["p_conservative"], "source": source, "source_sha256": digest(root / source),
                         "source_row_index": source_row_index, **({"measure": record["measure"]} if "measure" in record else {})})
    if len(rows) != 99:
        raise AssertionError("All99 inherited trials must remain")
    return rows


def verify_metrics(root, forecasts, protocol, recorded, calendar):
    rows = recorded["rows"]
    if [(row["candidate"], row["control"], row["horizon"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Complete four-comparison family required in registered order")
    probabilities, effects = [], []
    for row in rows:
        if row["study"] != "index_hinge":
            raise AssertionError("Index-hinge study identity differs")
        phases = [phase_statistics(forecasts, row["control"], row["horizon"], name, code, protocol, calendar)
                  for code, name in enumerate(("development", "evaluation"))]
        same_tree(row["phases"], phases, "Independent long-horizon MSE inference and all calendar offsets")
        probability = max(phase["p_conservative"] for phase in phases)
        same(row["p_conservative"], probability, "Conjunction of both historical phases")
        probabilities.append(probability)
        effects.append(effect_passes(phases, row["horizon"]))
    prior = inherited_rows(root, protocol)
    same_tree(recorded["inherited_rows"], prior, "Complete inherited comparison family")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-4:]
    passed = {}
    for number, row in enumerate(rows):
        same(row["p_holm_wave"], wave[number], "Four-way wave Holm")
        same(row["p_holm_cumulative"], cumulative[number], "103-way cumulative Holm")
        eligible = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < .05)
        if row["verdict"] != ("COMPARISON_GATE_PASS" if eligible else "DOES_NOT_QUALIFY"):
            raise AssertionError("Fixed return, stability, offset or statistical gate differs")
        passed[row["horizon"], row["control"]] = eligible
    leads = [h for h in HORIZONS if all(passed[h, control] for control in ("baseline", "mean"))]
    if recorded["leads"] != leads or recorded["hypothesis_count"] != 4 or recorded["cumulative_hypothesis_count"] != 103:
        raise AssertionError("Each horizon must pass both controls and retain all103 cumulative comparisons")
    return {"new_hypotheses_verified": 4, "cumulative_hypotheses_verified": 103, "phase_comparisons_verified": 8,
            "bootstrap_runs_verified": 24, "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
            "hac_maxlags": 504, "nonoverlap_offsets_verified": 2 * 2 * sum(HORIZONS), "leads": leads}


def verify(root=ROOT):
    root = Path(root)
    report, output = root / "reports/index_hinge", root / "data/index_hinge"
    protocol_path = root / "index_hinge.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Registered index-hinge protocol changed")
    for group in ("code", "inputs", "preserved"):
        for path, expected in manifest[group].items():
            if digest(root / path) != expected:
                raise AssertionError("Frozen source, input or earlier result changed: " + path)
    features, targets, source_audits = reconstruct(root, protocol)
    saved = pd.read_parquet(output / "features.parquet")
    if not saved.index.equals(features.index) or tuple(saved.columns) != tuple(features.columns):
        raise AssertionError("Raw index feature dates or schema differ")
    same(saved.loc[:, RAW], features.loc[:, RAW], "Every independent raw index feature", rtol=1e-10, atol=1e-12)
    if not saved.feature_cutoff_date.equals(features.feature_cutoff_date):
        raise AssertionError("Independent lagged feature date audit differs")
    saved_targets = pd.read_parquet(output / "targets.parquet")
    if set(saved_targets.horizon) != set(HORIZONS) or tuple(saved_targets.columns) != ("y", "target_end", "available_date", "horizon"):
        raise AssertionError("Both fixed target panels required")
    for horizon, expected in targets.items():
        actual = saved_targets.loc[saved_targets.horizon == horizon]
        if not actual.index.equals(expected.index):
            raise AssertionError("Horizon target reference calendar differs")
        same(actual.y, expected.y, "All independently constructed finite/zero return targets", rtol=1e-10, atol=1e-13)
        for name in ("target_end", "available_date"):
            if not actual[name].equals(expected[name]):
                raise AssertionError("Target session count or maturity differs")
    recorded_sources = json.loads((output / "source_audit.json").read_text())
    same_tree(recorded_sources, source_audits, "Source hashes, bounded numeric rows and reference-calendar audit")
    forecasts = pd.read_parquet(output / "forecasts.parquet")
    fits = json.loads((output / "fits.json").read_text())
    result = {"status": "VERIFIED", "protocol_sha256": digest(protocol_path), "verifier_sha256": digest(Path(__file__)),
              "feature_rows_verified": len(features), "raw_feature_columns_verified": len(RAW),
              "forecast_reconstruction": verify_forecasts(features, targets, forecasts, fits, protocol)}
    metrics = json.loads((report / "metrics.json").read_text())
    if metrics["protocol_sha256"] != digest(protocol_path) or metrics["evidence_class"] != protocol["evidence_class"]:
        raise AssertionError("Index-hinge metric provenance differs")
    result["inference"] = verify_metrics(root, forecasts, protocol, metrics, features.index)
    ledger = [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines() if line]
    for event, expected in (("inherited", metrics["inherited_rows"]), ("evaluated", metrics["rows"])):
        actual = [{key: value for key, value in row.items() if key != "event"} for row in ledger if row["event"] == event]
        same_tree(actual, expected, "Index-hinge " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (len(ledger) != 107 or len(registered) != 4
            or [(row["candidate"], row["control"], row["horizon"]) for row in registered] != list(COMPARISONS)
            or any(row["protocol_sha256"] != digest(protocol_path) for row in registered)):
        raise AssertionError("Complete99inherited+4registered+4evaluated trial ledger required")
    result["ledger_events_verified"] = {"inherited": 99, "registered": 4, "evaluated": 4}
    result["limitations"] = ["Archival revisions and VIX9D back-calculation remain retrospective source limitations.",
                            "The gap combines different variance conventions and is not a measured variance risk premium.",
                            "Overlapping price-index returns and historical reuse remain exploratory after all gates."]
    (report / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
