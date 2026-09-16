"""Independent reconstruction for the iterative index/HF signal search.

No producer modules are imported. The block-bootstrap/HAC primitives come from
the earlier independently tested verifier; construction and fitting below are
separate implementations of this wave's contracts.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import zipfile

import numpy as np
import pandas as pd
import yaml

from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = pathlib.Path(__file__).resolve().parents[1]
HF_BASE = ("const", "gk_d", "gk_w", "gk_m", "lev_d", "lev_w", "lev_m", "liv", "hf_d", "hf_w", "hf_m")
HF_BLOCKS = {"semivariance": ("share_d", "share_w", "share_m"),
             "kernel": ("kernel_d", "kernel_w", "kernel_m")}
INDEX_BASE = ("const", "cc_d", "cc_w", "cc_m", "lrv_d", "lrv_w", "lrv_m", "liv", "lvix")
INDEX_BLOCKS = {"session_split": ("split_d", "split_w", "split_m"),
                "cross_signed": ("x_hyg", "x_tlt", "x_gld", "x_uso", "x_uup"),
                "calendar": ("target_dow_1", "target_dow_2", "target_dow_3", "target_dow_4", "target_first3")}


def local_archive_dates(values):
    """Oxford labels identify local session dates, not UTC instants."""
    return pd.DatetimeIndex([pd.Timestamp(str(value)[:10]) for value in values])


def _daily_variance(daily):
    price = daily[["open", "high", "low", "close"]]
    if (price <= 0).any().any():
        raise ValueError("Nonpositive market price")
    inside = (.5 * np.log(daily.high / daily.low) ** 2
              - (2 * np.log(2) - 1) * np.log(daily.close / daily.open) ** 2)
    return inside.clip(lower=1e-10) + np.log(daily.open / daily.close.shift()) ** 2


def hf_features(daily, hf, vix):
    raw = hf.reindex(daily.index)
    rv = raw.rv5.where(raw.rv5 > 0)
    semi = raw.rsv.where(raw.rsv >= 0)
    kernel = raw.rk_parzen.where(raw.rk_parzen > 0)
    daily_var = _daily_variance(daily)
    returns = np.log(daily.close).diff()
    out = pd.DataFrame({"const": 1.}, index=daily.index)
    for label, span in (("d", 1), ("w", 5), ("m", 22)):
        out[f"gk_{label}"] = np.log(daily_var.rolling(span).mean())
        out[f"lev_{label}"] = returns.rolling(span).mean().clip(upper=0)
        out[f"hf_{label}"] = np.log(rv.rolling(span).mean()).shift()
        out[f"share_{label}"] = (semi.rolling(span).sum() / rv.rolling(span).sum()).shift()
        out[f"kernel_{label}"] = np.log(kernel.rolling(span).sum() / rv.rolling(span).sum()).shift()
    out["liv"] = np.log(vix.reindex(daily.index).where(lambda series: series > 0)).shift()
    columns = HF_BASE + HF_BLOCKS["semivariance"] + HF_BLOCKS["kernel"]
    return out.loc[:, columns].replace([np.inf, -np.inf], np.nan)


def hf_target(rv5, sessions, horizon):
    dates = pd.DatetimeIndex(sessions)
    rv = rv5.reindex(dates)
    result = pd.DataFrame(index=dates, columns=["y", "target_end", "available_date"])
    result["y"] = np.nan
    result["target_end"] = pd.NaT
    result["available_date"] = pd.NaT
    for position in range(len(dates)):
        end = position + horizon
        if end < len(dates):
            result.loc[dates[position], "target_end"] = dates[end]
            values = rv.iloc[position + 1:end + 1].to_numpy(float)
            if np.isfinite(values).all() and (values > 0).all():
                result.loc[dates[position], "y"] = values.mean()
        if end + 1 < len(dates):
            result.loc[dates[position], "available_date"] = dates[end + 1]
    return result


def calendar_features(sessions):
    """Calendar is finalized pre-open; market-price features end at prior close."""
    dates = pd.DatetimeIndex(sessions)
    columns = list(INDEX_BLOCKS["calendar"])
    out = pd.DataFrame(np.nan, index=dates, columns=columns)
    seen_in_month = {}
    for position, date in enumerate(dates[:-1]):
        key = (date.year, date.month)
        seen_in_month[key] = seen_in_month.get(key, 0) + 1
        next_date = dates[position + 1]
        next_key = (next_date.year, next_date.month)
        out.loc[date, "target_first3"] = float(seen_in_month.get(next_key, 0) + 1 <= 3)
        for day in range(1, 5):
            out.loc[date, f"target_dow_{day}"] = float(next_date.dayofweek == day)
    return out


def index_features(daily, cross, iv):
    variance = _daily_variance(daily)
    total = np.log(daily["adj close"]).diff()
    daytime = np.log(daily.close / daily.open)
    out = pd.DataFrame({"const": 1.}, index=daily.index)
    for label, span in (("d", 1), ("w", 5), ("m", 22)):
        out[f"cc_{label}"] = total.rolling(span).mean()
        out[f"lrv_{label}"] = np.log(variance.rolling(span).mean())
        out[f"split_{label}"] = (total - 2 * daytime).rolling(span).mean()
    for name, raw in (("liv", "vxn"), ("lvix", "vix")):
        out[name] = np.log(iv[raw].reindex(daily.index).where(lambda values: values > 0)).shift()
    for asset in ("hyg", "tlt", "gld", "uso", "uup"):
        out[f"x_{asset}"] = np.log(cross[asset].reindex(daily.index).where(lambda values: values > 0)).diff()
    out = pd.concat([out, calendar_features(daily.index)], axis=1)
    target = pd.DataFrame({"y": (daily.close / daily.open - 1).shift(-1),
                           "target_end": pd.Series(daily.index, index=daily.index).shift(-1)}, index=daily.index)
    return out.replace([np.inf, -np.inf], np.nan), target


def training_eligibility(complete, target, fit_origin, *, availability):
    mask = complete & np.isfinite(target.y) & (complete.index < pd.Timestamp(fit_origin))
    if availability:
        return mask & (target.y > 0) & (target.available_date <= pd.Timestamp(fit_origin))
    return mask & (target.target_end <= pd.Timestamp(fit_origin))


def _design(train, query):
    train, query = np.asarray(train, float), np.asarray(query, float)
    if (train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]
            or not np.isfinite(train).all() or not np.isfinite(query).all()
            or not (train[:, 0] == 1).all() or not (query[:, 0] == 1).all()):
        raise ValueError("Finite aligned design with leading constant required")
    center, scale = train[:, 1:].mean(axis=0), train[:, 1:].std(axis=0)
    if (scale <= 1e-12).any():
        raise ValueError("Zero-scale regressor")
    training = (train[:, 1:] - center) / scale
    apply = (query[:, 1:] - center) / scale
    return training, apply, center, scale


def ridge_prediction(train, y, query, alpha=.01):
    training, apply, mean, scale = _design(train, query)
    y = np.asarray(y, float)
    if y.shape != (len(training),) or not np.isfinite(y).all():
        raise ValueError("Finite signed or zero return targets required")
    intercept = float(y.mean())
    n, width = training.shape
    beta = np.linalg.solve(training.T @ training + n * alpha * np.eye(width), training.T @ (y - intercept))
    return intercept + apply @ beta, {"means": mean, "scales": scale, "beta": beta, "intercept": intercept}


def ols_prediction(train, y, query):
    training, apply, mean, scale = _design(train, query)
    y = np.asarray(y, float)
    if y.shape != (len(training),) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Positive finite variance targets required")
    x = np.column_stack([np.ones(len(training)), training])
    q = np.column_stack([np.ones(len(apply)), apply])
    coefficient, _, rank, _ = np.linalg.lstsq(x, np.log(y), rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Rank-deficient OLS")
    smear = np.mean(np.exp(np.log(y) - x @ coefficient))
    return np.exp(q @ coefficient) * smear, {"means": mean, "scales": scale, "beta": coefficient, "smear": smear, "rank": rank}


def execution_returns(predicted, actual, cost_per_side, threshold=.0004):
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    if predicted.shape != actual.shape or not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Finite aligned trading inputs required")
    position = (predicted > threshold).astype(float)
    return {"position": position, "net": position * (actual - 2 * cost_per_side)}


def holm(p_values):
    values = np.asarray(p_values, float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid probability")
    ordering = np.argsort(values, kind="stable")
    result = np.empty(len(values))
    running = 0.
    for rank, index in enumerate(ordering):
        running = max(running, (len(values) - rank) * values[index])
        result[index] = min(1., running)
    return result


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def same(actual, expected, label, rtol=2e-8, atol=1e-12):
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as error:
        raise AssertionError(f"{label}: {error}") from error


def reconstruct_inputs(root, protocol):
    output = root / "data/iterative_signal_search"
    section = protocol["hf"]
    cutoff = pd.Timestamp(section["source_end"])
    archive_path = root / section["source_archive"]
    if digest(archive_path) != section["archive_sha256"]:
        raise AssertionError("Registered Oxford source hash differs")
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise AssertionError("Ambiguous archive CSV")
        with archive.open(members[0]) as source:
            archived = pd.read_csv(source)
    spx = archived.loc[archived.Symbol.eq(".SPX")].copy()
    spx.index = local_archive_dates(spx.iloc[:, 0])
    spx.index.name = "date"
    hf = spx.loc[:cutoff, ["rv5", "rsv", "rk_parzen"]].astype(float)
    if (hf.index.has_duplicates or not hf.index.is_monotonic_increasing
            or not np.isfinite(hf).all().all() or (hf <= 0).any().any()
            or (hf.rsv > hf.rv5 * (1 + 1e-12)).any()):
        raise AssertionError("Invalid archived SPX measures")
    historical = pd.read_parquet(output / "spx_daily_pre2009_download.parquet",
                                  filters=[("date", "<=", cutoff)])
    current = pd.read_parquet(root / section["source_price_existing"], filters=[("date", "<=", cutoff)])
    overlap = historical.index.intersection(current.index)
    if len(overlap) < 250:
        raise AssertionError("Insufficient independent price-source overlap")
    if not historical.loc[overlap.min():overlap.max()].index.equals(current.loc[overlap.min():overlap.max()].index):
        raise AssertionError("Historical extension's overlap calendar differs")
    ohlc = ["open", "high", "low", "close"]
    same(historical.loc[overlap, ohlc], current.loc[overlap, ohlc], "Historical SPX overlap", rtol=1e-6, atol=1e-4)
    daily = pd.concat([historical.loc[historical.index < "2009-01-01", current.columns], current])
    daily.index.name = current.index.name
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise AssertionError("Merged daily source order failure")
    recorded_daily = pd.read_parquet(output / "spx_daily_merged.parquet")
    pd.testing.assert_frame_equal(recorded_daily, daily)
    pd.testing.assert_frame_equal(pd.read_parquet(output / "hf_source.parquet"), hf)
    vix = pd.read_parquet(root / section["source_cboe"], filters=[("date", "<=", cutoff)])["vix"]
    features_hf = hf_features(daily, hf, vix)
    produced_features = pd.read_parquet(output / "hf_features.parquet")
    pd.testing.assert_index_equal(produced_features.index, features_hf.index)
    if tuple(produced_features.columns) != tuple(features_hf.columns):
        raise AssertionError("HF feature identity differs")
    same(produced_features, features_hf, "All HF features", rtol=1e-11, atol=1e-13)
    targets_hf = {h: hf_target(hf.rv5, daily.index, h) for h in (1, 5, 21)}

    section = protocol["index"]
    cutoff = pd.Timestamp(section["source_end"])
    qqq = pd.read_parquet(root / section["source_price"], filters=[("date", "<=", cutoff)])
    cross = pd.read_parquet(root / section["source_cross"], filters=[("date", "<=", cutoff)])
    inputs = section["source_cboe"]
    if set(inputs) != {"vxn", "vix"}:
        raise AssertionError("Index implied-volatility source mapping differs")
    series = {}
    for name, path in inputs.items():
        frame = pd.read_csv(root / path, usecols=["DATE", "CLOSE"])
        dates = pd.to_datetime(frame.DATE, format="%m/%d/%Y")
        selected = dates <= cutoff
        series[name] = pd.Series(frame.loc[selected, "CLOSE"].to_numpy(float), index=pd.DatetimeIndex(dates.loc[selected]))
    iv = pd.DataFrame(series)
    features_index, targets_index = index_features(qqq, cross, iv)
    return {"hf": features_hf, "index": features_index}, {"hf": targets_hf, "index": {1: targets_index}}


def expected_origins(study, features, target_frames, section):
    complete = pd.Series(np.isfinite(features).all(axis=1), index=features.index)
    valid = complete.copy()
    for target in target_frames.values():
        valid &= np.isfinite(target.y) & (target.target_end <= pd.Timestamp(section["latest_target"]))
        if study == "hf":
            valid &= target.y > 0
    dates = features.index
    development = (dates >= section["development"][0]) & (dates <= section["development"][1])
    for target in target_frames.values():
        end = target.available_date if study == "hf" else target.target_end
        development &= (end <= pd.Timestamp(section["development_target_available_by"])).to_numpy()
    evaluation = (dates >= section["evaluation"][0]) & (dates <= section["evaluation"][1])
    return dates[valid & (development | evaluation)], complete


def verify_forecasts(study, features, target_frames, section, forecast, recorded_fits):
    forecast = forecast.copy()
    if "horizon" not in forecast:
        raise AssertionError("Explicit forecast horizon is required")
    base = HF_BASE if study == "hf" else INDEX_BASE
    blocks = HF_BLOCKS if study == "hf" else INDEX_BLOCKS
    model_names = ["baseline", *blocks] if study == "hf" else ["mean", "baseline", *blocks]
    origins, complete = expected_origins(study, features, target_frames, section)
    if set(forecast.model) != set(model_names) or forecast.duplicated(["origin", "horizon", "model"]).any():
        raise AssertionError("Forecast arms or uniqueness differs")
    fits = recorded_fits if isinstance(recorded_fits, list) else recorded_fits["fits"]
    fit_map = {(row.get("horizon", 1), pd.Timestamp(row["fit_origin"])): row for row in fits}
    expected_fit_keys = set()
    count = 0
    for horizon, targets in target_frames.items():
        for model in model_names:
            observed = forecast.loc[(forecast.horizon == horizon) & (forecast.model == model)].sort_values("origin")
            if not pd.DatetimeIndex(observed.origin).equals(origins):
                raise AssertionError(f"{study}/{horizon}/{model}: common scored origins differ")
            same(observed.y, targets.loc[origins, "y"], "Exact shared targets", rtol=1e-12, atol=0)
            if not np.array_equal(observed.target_end.to_numpy(), targets.loc[origins, "target_end"].to_numpy()):
                raise AssertionError("Target endpoint dates differ")
            if (study == "hf"
                    and not np.array_equal(observed.available_date.to_numpy(), targets.loc[origins, "available_date"].to_numpy())):
                raise AssertionError("HF target publication dates differ")
        for month in origins.to_period("M").unique():
            query = origins[origins.to_period("M") == month]
            fit_origin = query[0]
            key = (horizon, fit_origin)
            expected_fit_keys.add(key)
            if key not in fit_map:
                raise AssertionError("Missing monthly fit")
            fit = fit_map[key]
            mask = training_eligibility(complete, targets, fit_origin, availability=(study == "hf"))
            n = int(mask.sum())
            if n < section["minimum_train"] or fit["train_n"] != n:
                raise AssertionError("Training sample count or minimum differs")
            last_target = targets.loc[mask, "target_end"].max()
            if pd.Timestamp(fit["train_last_target"]) != last_target:
                raise AssertionError("Training target maturity audit differs")
            if study == "hf":
                last_available = targets.loc[mask, "available_date"].max()
                if pd.Timestamp(fit["train_last_available"]) != last_available or last_available > fit_origin:
                    raise AssertionError("HF target admitted before publication")
            audit_models = fit["model_audit"] if study == "hf" else fit["models"]
            for model in model_names:
                block = forecast.loc[(forecast.horizon == horizon) & (forecast.model == model)
                                     & forecast.origin.isin(query)].sort_values("origin")
                if (not block.fit_origin.eq(fit_origin).all() or not block.train_n.eq(n).all()
                        or not block.train_last_target.eq(last_target).all()):
                    raise AssertionError("Per-forecast fit metadata differs")
                y = targets.loc[mask, "y"].to_numpy(float)
                audit = audit_models[model]
                if model == "mean":
                    prediction = np.full(len(query), y.mean())
                    same(audit["mean"], y.mean(), "Historical mean control")
                else:
                    columns = base + blocks.get(model, ())
                    training = features.loc[mask, columns].to_numpy(float)
                    apply = features.loc[query, columns].to_numpy(float)
                    if study == "hf":
                        prediction, independent = ols_prediction(training, y, apply)
                        same(audit["means"], np.r_[0., independent["means"]], "HF train mean")
                        same(audit["scales"], np.r_[1., independent["scales"]], "HF train scale")
                        same(audit["beta"], independent["beta"], "HF OLS coefficients")
                        same(audit["smear"], independent["smear"], "HF exact smear")
                        if audit["rank"] != independent["rank"]:
                            raise AssertionError("HF design rank differs")
                    else:
                        prediction, independent = ridge_prediction(training, y, apply, section["ridge_alpha"])
                        same(audit["mean"], independent["means"], "Ridge train mean")
                        same(audit["scale"], independent["scales"], "Ridge train scale")
                        same(audit["beta"], np.r_[independent["intercept"], independent["beta"]], "Ridge coefficients")
                        if audit["alpha"] != section["ridge_alpha"]:
                            raise AssertionError("Ridge penalty differs")
                    if tuple(audit["columns"]) != columns:
                        raise AssertionError("Model feature subset differs")
                same(block.prediction, prediction, f"{study}/{model}/{horizon} independent predictions", rtol=1e-8, atol=1e-12)
                count += len(block)
    if set(fit_map) != expected_fit_keys:
        raise AssertionError("Unexpected or duplicated monthly fit entries")
    return {"forecasts_verified": count, "monthly_fits_verified": len(expected_fit_keys),
            "common_origins": len(origins), "features_reconstructed": len(features.columns)}


def _phase_panel(panel, section, phase, study):
    start, finish = map(pd.Timestamp, section[phase])
    selected = panel.loc[(panel.origin >= start) & (panel.origin <= finish)].copy()
    if phase == "development":
        availability = selected.available_date if study == "hf" else selected.target_end
        selected = selected.loc[availability <= pd.Timestamp(section["development_target_available_by"])]
    return selected


def verify_metrics(root, panels, features, protocol, recorded):
    expected = [("hf", h, name, "baseline") for h in (1, 5, 21) for name in HF_BLOCKS]
    expected += [("index", 1, name, control) for name in INDEX_BLOCKS for control in ("baseline", "mean")]
    rows = recorded["rows"]
    keys = [(r["study"], r["horizon"], r["candidate"], r["control"]) for r in rows]
    if len(rows) != 12 or set(keys) != set(expected) or len(set(keys)) != 12:
        raise AssertionError("Missing or duplicate registered forecasting contrast")
    settings = protocol["inference"]
    new_p, positive, stable = [], [], []
    for row in rows:
        study, horizon, candidate, control = row["study"], row["horizon"], row["candidate"], row["control"]
        section = protocol[study]
        panel = panels[study].loc[panels[study].horizon == horizon]
        phase_ps, phase_effects, stability_gaps = [], [], []
        if [phase["name"] for phase in row["phases"]] != ["development", "evaluation"]:
            raise AssertionError("Both registered phases required")
        for phase_code, phase in enumerate(row["phases"]):
            phase_name = phase["name"]
            selected = _phase_panel(panel, section, phase_name, study)
            wide = selected.pivot(index="origin", columns="model", values="prediction").sort_index()
            target = selected.loc[selected.model == control].set_index("origin").reindex(wide.index).y.to_numpy(float)
            if study == "hf":
                ratio_c, ratio_b = target / wide[candidate].to_numpy(), target / wide[control].to_numpy()
                candidate_loss = ratio_c - np.log(ratio_c) - 1
                control_loss = ratio_b - np.log(ratio_b) - 1
            else:
                candidate_loss = np.square(target - wide[candidate].to_numpy())
                control_loss = np.square(target - wide[control].to_numpy())
            difference = candidate_loss - control_loss
            delta = float(difference.mean())
            gain = float(-100 * delta / control_loss.mean())
            phase_effects.append(gain)
            if phase["n"] != len(wide) or phase["first_origin"] != str(wide.index[0].date()) or phase["last_origin"] != str(wide.index[-1].date()):
                raise AssertionError("Phase dates or counts differ")
            for name, value in (("delta", delta), ("control_loss", control_loss.mean()),
                                ("candidate_loss", candidate_loss.mean()), ("improvement_pct", gain)):
                same(phase[name], value, f"{study}/{horizon}/{candidate}/{phase_name}/{name}", rtol=1e-10, atol=1e-16)
            hac = independent_hac(difference)
            for name, value in hac.items():
                same(phase["hac126"][name], value, f"Independent HAC {name}", rtol=2e-8, atol=1e-15)
            p_values, low, high = [hac["p"]], [hac["ci95"][0]], [hac["ci95"][1]]
            if set(phase["block_inference"]) != {str(block) for block in settings["blocks"]}:
                raise AssertionError("Bootstrap block family differs")
            for block in settings["blocks"]:
                seed = settings["seed"] + (1 if study == "hf" else 2) * 100000 + horizon * 1000 + phase_code * 10000 + block
                sampled = explicit_bootstrap_means(difference, block, settings["bootstrap_draws"], seed)[:, 0]
                p_value = (1 + np.count_nonzero(np.abs(sampled - delta) >= abs(delta))) / (len(sampled) + 1)
                interval = np.quantile(sampled, [.025, .975])
                saved = phase["block_inference"][str(block)]
                same(saved["p"], p_value, "Independent bootstrap p", rtol=0, atol=1e-14)
                same(saved["ci95"], interval, "Independent bootstrap interval", rtol=2e-8, atol=1e-15)
                p_values.append(float(p_value))
                low.append(float(interval[0]))
                high.append(float(interval[1]))
            phase_p = max(p_values)
            same(phase["p_conservative"], phase_p, "Phase conservative p", rtol=2e-8, atol=1e-15)
            same(phase["ci95_envelope"], [min(low), max(high)], "Nominal interval envelope", rtol=2e-8, atol=1e-15)
            phase_ps.append(phase_p)
            years = sorted(set(wide.index.year))
            if [item["year"] for item in phase["annual"]] != years:
                raise AssertionError("Annual diagnostic membership differs")
            for annual, year in zip(phase["annual"], years, strict=True):
                mask = wide.index.year == year
                if annual["n"] != int(mask.sum()):
                    raise AssertionError("Annual count differs")
                same(annual["delta"], difference[mask].mean(), "Annual difference", rtol=1e-10, atol=1e-16)
            position = features[study].index.get_indexer(wide.index)
            if (position < 0).any() or len(phase["nonoverlap_phases"]) != horizon:
                raise AssertionError("Nonoverlap phases must follow actual session positions")
            for i, saved in enumerate(phase["nonoverlap_phases"]):
                mask = position % horizon == i
                if saved["phase"] != i or saved["n"] != int(mask.sum()):
                    raise AssertionError("Nonoverlap phase membership differs")
                same(saved["delta"], difference[mask].mean(), "Nonoverlap phase difference", rtol=1e-10, atol=1e-16)
            if phase_name == "evaluation":
                if len(phase["stability"]) != len(section["evaluation_stability"]):
                    raise AssertionError("Stability slice family differs")
                for saved, (start, finish) in zip(phase["stability"], section["evaluation_stability"], strict=True):
                    mask = (wide.index >= start) & (wide.index <= finish)
                    if saved["start"] != start or saved["end"] != finish or saved["n"] != int(mask.sum()):
                        raise AssertionError("Stability dates/counts differ")
                    gap = float(difference[mask].mean())
                    same(saved["delta"], gap, "Stability difference", rtol=1e-10, atol=1e-16)
                    stability_gaps.append(gap)
            elif phase["stability"]:
                raise AssertionError("Unexpected development stability requirement")
        hypothesis_p = max(phase_ps)
        same(row["p_conservative"], hypothesis_p, "Conjunction of phase p-values", rtol=2e-8, atol=1e-15)
        new_p.append(hypothesis_p)
        positive.append(all(effect >= section["effect_threshold_pct"] for effect in phase_effects))
        stable.append(len(stability_gaps) == 2 and all(gap < 0 for gap in stability_gaps))
    inherited = []
    inherited_keys = []
    for path in protocol["comparisons"]["inherited_sources"]:
        old_rows = json.loads((root / path).read_text())["rows"]
        inherited.extend(row["p_conservative"] for row in old_rows)
        inherited_keys.extend((path, row["candidate"], row.get("control", "baseline"), row["horizon"]) for row in old_rows)
    if len(inherited) != 54 or len(recorded["inherited_rows"]) != 54:
        raise AssertionError("All 54 inherited hypotheses must remain")
    for saved, identity, p_value in zip(recorded["inherited_rows"], inherited_keys, inherited, strict=True):
        if (saved["source"], saved["candidate"], saved["control"], saved["horizon"]) != identity:
            raise AssertionError("Inherited trial identity differs")
        if saved["source_sha256"] != digest(root / saved["source"]):
            raise AssertionError("Inherited results changed")
        same(saved["p_conservative"], p_value, "Inherited raw p-value", rtol=0, atol=0)
    wave = holm(new_p)
    cumulative = holm(inherited + new_p)[-len(new_p):]
    pass_flags = []
    for index, row in enumerate(rows):
        same(row["p_holm_wave"], wave[index], "All-12 Holm", rtol=2e-8, atol=1e-14)
        same(row["p_holm_cumulative"], cumulative[index], "All-66 Holm", rtol=2e-8, atol=1e-14)
        passed = bool(wave[index] < .025 and cumulative[index] < .05 and positive[index] and stable[index])
        pass_flags.append(passed)
        if row["verdict"] != ("EXPLORATORY_LEAD" if passed else "DOES_NOT_QUALIFY"):
            raise AssertionError("Declared inferential/effect/stability gate differs")
    expected_leads = []
    for row, passed in zip(rows, pass_flags, strict=True):
        if not passed:
            continue
        if row["study"] == "index":
            siblings = [flag for other, flag in zip(rows, pass_flags, strict=True)
                        if other["study"] == "index" and other["candidate"] == row["candidate"]]
            if len(siblings) != 2 or not all(siblings):
                continue
        lead = {"study": row["study"], "candidate": row["candidate"], "horizon": row["horizon"]}
        if lead not in expected_leads:
            expected_leads.append(lead)
    if recorded["leads"] != expected_leads:
        raise AssertionError("A lead was promoted without both required controls")
    return {"new_hypotheses_verified": 12, "inherited_hypotheses_verified": 54,
            "phase_comparisons_verified": 24, "bootstrap_runs_verified": 72, "leads": expected_leads}


def verify_execution(forecast, protocol, execution):
    rows = execution["rows"]
    section = protocol["index"]
    models = [*section["models"], "always_daylong"]
    costs = section["descriptive_execution"]["sensitivity_costs_per_side"]
    expected_keys = {(phase, model, cost) for phase in ("development", "evaluation") for model in models for cost in costs}
    keys = [(row["phase"], row["model"], row["cost_per_side"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise AssertionError("Descriptive execution scenario set differs")
    for row in rows:
        selected = _phase_panel(forecast, section, row["phase"], "index")
        model = row["model"]
        block = selected.loc[selected.model == ("baseline" if model == "always_daylong" else model)].sort_values("origin")
        position = np.ones(len(block)) if model == "always_daylong" else (block.prediction.to_numpy() > .0004).astype(float)
        net = position * block.y.to_numpy() - 2 * row["cost_per_side"] * position
        capital = 1.
        peak = 1.
        drawdown = 0.
        for value in net:
            capital *= 1 + value
            peak = max(peak, capital)
            drawdown = min(drawdown, capital / peak - 1)
        if row["n"] != len(net) or row["held_days"] != int(position.sum()):
            raise AssertionError("Descriptive execution exposure counts differ")
        for name, value in (("exposure", position.mean()), ("mean_daily_net", net.mean()),
                            ("total_net_return", capital - 1), ("max_drawdown", drawdown)):
            same(row[name], value, f"Execution {name}", rtol=1e-11, atol=1e-14)
        if net.std(ddof=1) == 0:
            if row["annualized_sharpe"] is not None:
                raise AssertionError("Undefined Sharpe ratio must remain missing")
        else:
            same(row["annualized_sharpe"], np.sqrt(252) * net.mean() / net.std(ddof=1), "Descriptive Sharpe")
    return {"execution_scenarios_verified": len(rows), "two_legs_charged_each_held_day": True}


def verify(root=ROOT):
    root = pathlib.Path(root)
    report = root / "reports/iterative_signal_search"
    output = root / "data/iterative_signal_search"
    protocol_path = root / "iterative_signal_search.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Protocol changed after registration")
    for section in ("code", "inputs", "preserved"):
        for path, expected in manifest[section].items():
            if digest(root / path) != expected:
                raise AssertionError(f"Frozen code or earlier artifact changed: {path}")
    features, targets = reconstruct_inputs(root, protocol)
    panels = {study: pd.read_parquet(output / f"{study}_forecasts.parquet") for study in ("hf", "index")}
    fits = {study: json.loads((output / f"{study}_fits.json").read_text()) for study in ("hf", "index")}
    result = {"status": "VERIFIED", "protocol_sha256": digest(protocol_path),
              "verifier_sha256": digest(pathlib.Path(__file__)), "methods": {}}
    for study in panels:
        if "horizon" not in panels[study]:
            raise AssertionError("Explicit forecast horizon is required")
        result["methods"][study] = verify_forecasts(study, features[study], targets[study], protocol[study], panels[study], fits[study])
    index_audit = fits["index"]
    for path, expected in index_audit["inputs"].items():
        if digest(root / path) != expected:
            raise AssertionError("Index input changed after fit")
    if index_audit["producer_sha256"] != digest(root / "src/iterative_index.py"):
        raise AssertionError("Index producer changed after fitting")
    metrics = json.loads((report / "metrics.json").read_text())
    if metrics["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Metrics reference a different protocol")
    result["inference"] = verify_metrics(root, panels, features, protocol, metrics)
    execution = json.loads((report / "execution.json").read_text())
    result["execution"] = verify_execution(panels["index"], protocol, execution)
    ledger = [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines() if line]
    expected_counts = {"inherited": 54, "registered": 12, "evaluated": 12}
    actual_counts = {event: sum(row["event"] == event for row in ledger) for event in expected_counts}
    if actual_counts != expected_counts or len(ledger) != 78:
        raise AssertionError("Trial ledger omitted, duplicated, or added trials")
    metric_keys = {(row["study"], row["horizon"], row["candidate"], row["control"]) for row in metrics["rows"]}
    registered = [row for row in ledger if row["event"] == "registered"]
    if {(row["study"], row["horizon"], row["candidate"], row["control"]) for row in registered} != metric_keys:
        raise AssertionError("Registered ledger trial identities differ")
    if any(row["protocol_sha256"] != digest(protocol_path) for row in registered):
        raise AssertionError("Ledger trial registration protocol changed")
    for event, expected in (("inherited", metrics["inherited_rows"]), ("evaluated", metrics["rows"])):
        actual = [{key: value for key, value in row.items() if key != "event"} for row in ledger if row["event"] == event]
        if actual != expected:
            raise AssertionError(f"{event} trial ledger contents differ from retained results")
    result["ledger_events_verified"] = actual_counts
    result["limitations"] = ["Reused historical data remains exploratory.",
                              "Cumulative correction covers the 66 enumerated conversation contrasts, not every older unlogged search."]
    (report / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
