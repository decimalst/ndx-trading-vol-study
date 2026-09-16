"""Independent reconstruction for the exploratory model and memory study.

This verifier imports neither the producer nor any of its model, transformation,
retrieval or scoring helpers. Source reconstruction, optimization, neighbor
selection and dependent-loss inference are calculated independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import linalg, optimize, stats
from sklearn.preprocessing import SplineTransformer
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    "const", "lrv_d", "lrv_w", "lrv_m", "lev_d", "lev_w", "lev_m",
    "liv", "lvix", "term", "xasset_stress", "market_stress",
)
COMPONENTS = ("baseline", "gamma", "adaptive_ols", "adaptive_gamma", "ridge_gamma", "spline_gamma")
OTHER_INPUTS = ("vvix", "rv_dispersion", "stock_bond_corr", "close_pressure")


def independently_fit_gamma(design, y, weights, alpha):
    """Fit the convex Gamma objective with independent analytic derivatives."""
    x = np.column_stack([np.ones(len(design)), design])
    weights = np.asarray(weights, float) / np.sum(weights)
    y = np.asarray(y, float)
    scale = np.median(y)
    normalized_y = y / scale
    penalty = np.r_[0.0, np.repeat(float(alpha), design.shape[1])]

    def objective(beta):
        linear = x @ beta
        return float(np.dot(weights, normalized_y * np.exp(-linear) + linear) + 0.5 * np.dot(penalty, beta**2))

    def gradient(beta):
        ratio = normalized_y * np.exp(-x @ beta)
        return x.T @ (weights * (1 - ratio)) + penalty * beta

    def hessian(beta):
        ratio = normalized_y * np.exp(-x @ beta)
        return x.T @ (x * (weights * ratio)[:, None]) + np.diag(penalty)

    initial = np.r_[np.log(np.dot(weights, normalized_y)), np.zeros(design.shape[1])]
    result = optimize.minimize(objective, initial, method="trust-exact", jac=gradient, hess=hessian,
                               options={"gtol": 1e-10, "maxiter": 300})
    if np.max(np.abs(gradient(result.x))) > 2e-7:
        raise AssertionError("independent Gamma optimizer did not reach stationarity")
    beta = result.x.copy()
    beta[0] += np.log(scale)
    return beta


def verify_model_fit(train, y, query, ages, audits, protocol, refit_gamma=False):
    """Rebuild all maps; solve OLS and check every convex Gamma optimum."""
    raw = train.loc[:, list(BASELINE[1:])].to_numpy(float)
    apply = query.loc[:, list(BASELINE[1:])].to_numpy(float)
    center, scale = raw.mean(axis=0), raw.std(axis=0)
    if (scale <= 1e-12).any():
        raise AssertionError("zero-scale training input")
    z, zquery = (raw - center) / scale, (apply - center) / scale
    variables = protocol["estimators"]["spline"]["variables"]
    indices = [list(BASELINE[1:]).index(name) for name in variables]
    remaining = [i for i in range(raw.shape[1]) if i not in indices]
    spline = SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False, extrapolation="linear")
    basis = np.column_stack([spline.fit_transform(z[:, indices]), z[:, remaining]])
    basis_query = np.column_stack([spline.transform(zquery[:, indices]), zquery[:, remaining]])
    basis_center, basis_scale = basis.mean(axis=0), basis.std(axis=0)
    if (basis_scale <= 1e-12).any():
        raise AssertionError("zero-scale independent spline basis")
    outputs, max_gradient = {}, 0.0
    for name in COMPONENTS:
        audit = audits[name]
        same(audit["input_center"], center, f"{name} scaler mean", rtol=1e-11)
        same(audit["input_scale"], scale, f"{name} scaler scale", rtol=1e-11)
        if audit["input_columns"] != list(BASELINE[1:]) or not audit["converged"] or audit["n_train"] != len(train):
            raise AssertionError(f"{name}: estimator metadata differs")
        current, current_query = z, zquery
        if name == "spline_gamma":
            same(audit["spline_knots"], np.column_stack([s.t for s in spline.bsplines_]), "spline knots", rtol=1e-10)
            same(audit["design_center"], basis_center, "spline mean", rtol=1e-10)
            same(audit["design_scale"], basis_scale, "spline scale", rtol=1e-10)
            current, current_query = (basis - basis_center) / basis_scale, (basis_query - basis_center) / basis_scale
        x, xquery = np.column_stack([np.ones(len(train)), current]), np.column_stack([np.ones(len(query)), current_query])
        weights = np.exp2(-np.asarray(ages) / protocol["estimators"]["adaptive_half_life_sessions"]) if name.startswith("adaptive") else np.ones(len(train))
        same(audit["weight_sum"], weights.sum(), f"{name} weight sum")
        same(audit["effective_sample_size"], weights.sum()**2 / (weights**2).sum(), f"{name} effective sample")
        recorded_beta = np.r_[audit["intercept"], audit["coefficients"]]
        if name in ("baseline", "adaptive_ols"):
            beta = linalg.lstsq(x * np.sqrt(weights[:, None]), np.log(y) * np.sqrt(weights), lapack_driver="gelsy")[0]
            same(recorded_beta, beta, f"{name} independent weighted OLS", rtol=2e-7, atol=1e-10)
            smear = np.average(np.exp(np.log(y) - x @ beta), weights=weights)
            same(audit["smearing_factor"], smear, f"{name} smearing")
            outputs[name] = np.exp(xquery @ beta) * smear
        else:
            alpha = protocol["estimators"]["ridge_alpha"] if name in ("ridge_gamma", "spline_gamma") else 0.0
            same(audit["alpha"], alpha, f"{name} alpha")
            same(audit["target_scale"], np.median(y), f"{name} target unit normalization")
            prediction_train = np.exp(x @ recorded_beta)
            gradient = x.T @ (weights * (1 - y / prediction_train) / weights.sum())
            gradient[1:] += alpha * recorded_beta[1:]
            norm = float(np.max(np.abs(gradient)))
            same(audit["gradient_inf_norm"], norm, f"{name} independent gradient", atol=2e-12)
            if norm > 2e-7 or audit["n_iter"] >= protocol["estimators"]["gamma_max_iter"]:
                raise AssertionError(f"{name}: convex objective stationarity/convergence failed")
            max_gradient = max(max_gradient, norm)
            outputs[name] = np.exp(xquery @ recorded_beta)
            if refit_gamma:
                beta = independently_fit_gamma(current, y, weights, alpha)
                same(np.exp(xquery @ beta), outputs[name], f"{name} independent scipy refit", rtol=5e-6)
    return outputs, max_gradient


def verify_components(features, latents, components, fits, protocol):
    sample = protocol["sample"]
    saved = components.copy()
    date_columns = ["origin", "target_end", "fit_origin", "train_last_target"]
    for column in date_columns:
        saved[column] = pd.to_datetime(saved[column])
    if saved.duplicated(["origin", "horizon", "model"]).any() or set(saved["model"]) != set(COMPONENTS):
        raise AssertionError("component key/model contract violated")
    if set(saved["horizon"]) != set(sample["horizons"]):
        raise AssertionError("component horizon set differs")
    if not np.isfinite(saved[["y", "prediction"]]).all().all() or (saved[["y", "prediction"]] <= 0).any().any():
        raise AssertionError("component positive finite contract violated")
    if (saved["target_end"] > pd.Timestamp(sample["latest_target"])).any() or (saved["target_end"] >= pd.Timestamp(sample["sealed_start"])).any():
        raise AssertionError("component target fence violated")
    aligned = latents.reindex(features.index)
    complete = np.isfinite(features[list(BASELINE) + list(OTHER_INPUTS)]).all(axis=1) & np.isfinite(aligned).all(axis=1)
    keyed_fits = {(int(item["horizon"]), pd.Timestamp(item["fit_origin"])): item for item in fits}
    if len(keyed_fits) != len(fits):
        raise AssertionError("duplicate estimator fit audit")
    used_fits, checked, max_gradient, independent_refits = set(), 0, 0.0, 0
    for horizon in sample["horizons"]:
        target = target_table(features, horizon)
        possible = features.index[complete & target["y"].notna() & (target["y"] > 0)
                                  & (features.index >= pd.Timestamp(sample["historical_forecast_start"]))
                                  & (features.index <= pd.Timestamp(sample["score_end"]))]
        expected_origins = []
        for _, month in pd.Series(possible, index=possible).groupby(possible.to_period("M")):
            origins = pd.DatetimeIndex(month.to_numpy())
            first = origins[0]
            train_index = features.index[complete & (features.index < first) & (target["target_end"] <= first) & (target["y"] > 0)]
            if len(train_index) < sample["minimum_train"]:
                continue
            expected_origins.extend(origins)
            fit = keyed_fits[(horizon, first)]
            used_fits.add((horizon, first))
            last = target.loc[train_index, "target_end"].max()
            if fit["train_n"] != len(train_index) or pd.Timestamp(fit["train_last_target"]) != last:
                raise AssertionError("monthly fit eligibility metadata differs")
            ages = features.index.get_indexer([first])[0] - features.index.get_indexer(train_index)
            refit = first.month == 1 or len(used_fits) == 1
            predictions, gradient = verify_model_fit(features.loc[train_index], target.loc[train_index, "y"].to_numpy(),
                                                     features.loc[origins], ages, fit["model_audit"], protocol, refit_gamma=refit)
            max_gradient = max(max_gradient, gradient)
            independent_refits += int(refit) * 4
            for model in COMPONENTS:
                block = saved.loc[(saved["horizon"] == horizon) & (saved["model"] == model)].set_index("origin").reindex(origins)
                if block["prediction"].isna().any() or not (block["fit_origin"] == first).all() or not (block["train_n"] == len(train_index)).all() or not (block["train_last_target"] == last).all():
                    raise AssertionError("component row missing or monthly provenance differs")
                if not (block["target_end"] == target.loc[origins, "target_end"]).all():
                    raise AssertionError("component target completion differs")
                same(block["y"], target.loc[origins, "y"], "component actual target", rtol=1e-10)
                same(block["prediction"], predictions[model], f"{horizon}/{first}/{model} forecast")
                checked += len(block)
        for model in COMPONENTS:
            observed = saved.loc[(saved["horizon"] == horizon) & (saved["model"] == model), "origin"].sort_values()
            if not pd.DatetimeIndex(observed).equals(pd.DatetimeIndex(expected_origins)):
                raise AssertionError("full historical component origin population differs")
    if used_fits != set(keyed_fits) or checked != len(saved):
        raise AssertionError("unexpected estimator audit or unchecked component prediction")
    return {"component_forecasts_checked": checked, "monthly_component_fits_checked": len(fits) * 6,
            "independent_scipy_gamma_refits": independent_refits, "maximum_gamma_gradient": max_gradient,
            "component_method": "Independent source/train/map reconstruction; all OLS refit, all convex Gamma stationarity checked, January Gamma fits independently refit with scipy trust-exact"}


def same(actual, expected, label: str, rtol=2e-7, atol=1e-12):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError(f"{label}: independent calculation disagrees")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reconstruct_features(root: Path, protocol: dict) -> pd.DataFrame:
    cutoff = pd.Timestamp(protocol["sample"]["latest_target"])

    def read(name):
        raw = pd.read_parquet(root / "data/raw" / name, filters=[("date", "<=", cutoff)])
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
        if raw.index.has_duplicates or not raw.index.is_monotonic_increasing or raw.index.max() > cutoff:
            raise AssertionError(f"{name}: source order/fence failure")
        return raw

    daily = read("daily_ohlc.parquet")
    assets = read("cross_asset_daily.parquet")
    vxn, short = read("vxn_daily.parquet"), read("short_dated_iv.parquet")
    out = pd.DataFrame(index=daily.index)
    logrange = np.log(daily["high"] / daily["low"])
    intraday = (0.5 * logrange**2 - (2 * np.log(2) - 1) * np.log(daily["close"] / daily["open"])**2).clip(lower=1e-10)
    overnight = np.log(daily["open"] / daily["close"].shift(1))**2
    variance = intraday + overnight
    returns = np.log(daily["adj close"] / daily["adj close"].shift(1))
    out["const"], out["rv_total"], out["log_rv"] = 1.0, variance, np.log(variance.clip(lower=1e-10))
    for name, span in (("lrv_d", 1), ("lrv_w", 5), ("lrv_m", 22)):
        out[name] = np.log(variance.rolling(span).mean())
    for name, span in (("lev_d", 1), ("lev_w", 5), ("lev_m", 22)):
        out[name] = returns.rolling(span).mean().clip(upper=0)
    out["liv"] = np.log(vxn["close"].where(vxn["close"] > 0).reindex(out.index)).shift(1)
    out["lvix"] = np.log(short["vix"].where(short["vix"] > 0).reindex(out.index)).shift(1)
    out["term"] = np.log(short["vix9d"].where(short["vix9d"] > 0) / short["vix"].where(short["vix"] > 0)).reindex(out.index).shift(1)

    def zscore(x):
        history = x.rolling(252, min_periods=126)
        return (x - history.mean().shift(1)) / history.std(ddof=1).shift(1).replace(0.0, np.nan)

    prices = assets.reindex(out.index)[["hyg", "tlt", "gld", "uso", "uup"]]
    z = zscore(np.log(prices.where(prices > 0)).diff())
    out["xasset_stress"] = np.sqrt(z.pow(2).mean(axis=1, skipna=False))
    volume = np.log(daily["volume"].where(daily["volume"] > 0))
    share = (overnight / variance.replace(0.0, np.nan)).clip(0, 1)
    out["market_stress"] = np.sqrt((zscore(volume).clip(lower=0)**2 + zscore(share).clip(lower=0)**2) / 2)
    vvix = pd.read_csv(root / "data/free_sources/raw/cboe/VVIX_History.csv")
    vvix["DATE"] = pd.to_datetime(vvix["DATE"], format="%m/%d/%Y")
    vvix = vvix.loc[vvix["DATE"] <= cutoff].set_index("DATE")
    out["vvix"] = np.log(vvix["VVIX"].where(vvix["VVIX"] > 0).reindex(out.index)).shift(1)
    out["rv_dispersion"] = out["log_rv"].rolling(22).std(ddof=1)
    out["stock_bond_corr"] = returns.rolling(22).corr(np.log(prices["tlt"]).diff())
    out["close_pressure"] = (2 * np.log(daily["close"] / daily["low"]) / logrange.replace(0.0, np.nan) - 1).rolling(5).mean()
    return out.replace([np.inf, -np.inf], np.nan)


def target_table(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    future = pd.concat([features["rv_total"].shift(-i) for i in range(1, horizon + 1)], axis=1)
    return pd.DataFrame({"y": future.mean(axis=1, skipna=False),
                         "target_end": pd.Series(features.index, index=features.index).shift(-horizon)})


def eligible_positions(calendar, origins, target_ends, query, lookback: int, gap: int):
    """Use calendar-session positions, with target completion then the extra gap."""
    calendar = pd.DatetimeIndex(calendar)
    origins, target_ends = pd.DatetimeIndex(origins), pd.DatetimeIndex(target_ends)
    if (len(origins) != len(target_ends) or calendar.has_duplicates or not calendar.is_monotonic_increasing
            or lookback < 1 or gap < 0):
        raise AssertionError("invalid retrieval calendar")
    current = calendar.get_indexer([pd.Timestamp(query)])[0]
    at = calendar.get_indexer(origins)
    ends = calendar.get_indexer(target_ends)
    if current < 0 or (at < 0).any() or (ends < 0).any() or (ends <= at).any():
        raise AssertionError("memory has unknown or inconsistent timestamps")
    return np.flatnonzero((at < current) & (at >= current - lookback) & (ends + gap <= current))


def ratio_correction(actual, forecast, shrinkage: float) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    if not 0 <= shrinkage <= 1 or not len(actual) or actual.shape != forecast.shape:
        raise AssertionError("invalid ratio memory inputs")
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all() or (actual <= 0).any() or (forecast <= 0).any():
        raise AssertionError("ratio memory requires finite positive values")
    return float(1 + shrinkage * (np.mean(actual / forecast) - 1))


def nearest_indices(keys, query, neighbors: int):
    keys, query = np.asarray(keys, float), np.asarray(query, float)
    if keys.ndim != 2 or query.shape != (keys.shape[1],) or not 1 <= neighbors <= len(keys):
        raise AssertionError("invalid neighbor geometry")
    if not np.isfinite(keys).all() or not np.isfinite(query).all():
        raise AssertionError("neighbor geometry contains missing values")
    squared = np.sum((keys - query)**2, axis=1)
    # Chronological input rows resolve exactly equal distances by oldest origin.
    selected = np.lexsort((np.arange(len(keys)), squared))[:neighbors]
    return selected, squared[selected]


def qlike(actual, prediction):
    ratio = np.asarray(actual, float) / np.asarray(prediction, float)
    if not np.isfinite(ratio).all() or (ratio <= 0).any():
        raise AssertionError("QLIKE requires positive finite actuals and predictions")
    return ratio - np.log(ratio) - 1


def expert_weights(losses, ages, half_life=63.0, learning_rate=5.0, floor_mass=0.1):
    losses, ages = np.asarray(losses, float), np.asarray(ages, float)
    if losses.ndim != 2 or ages.shape != (len(losses),) or not len(losses):
        raise AssertionError("invalid dynamic ensemble history")
    if not np.isfinite(losses).all() or not np.isfinite(ages).all() or (ages < 0).any():
        raise AssertionError("dynamic ensemble includes invalid or future errors")
    weighted = np.average(losses, axis=0, weights=np.exp2(-ages / half_life))
    logits = -learning_rate * weighted
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    return floor_mass / losses.shape[1] + (1 - floor_mass) * probabilities


def explicit_bootstrap_means(values, block, draws, seed):
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    if not 1 <= block <= n or draws < 1 or not np.isfinite(values).all():
        raise AssertionError("invalid independent bootstrap inputs")
    count, remainder = divmod(n, block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, count))
    tails = rng.integers(0, n, size=draws) if remainder else None
    output = np.empty((draws, values.shape[1]))
    for lower in range(0, draws, 64):
        upper = min(draws, lower + 64)
        indices = (starts[lower:upper, :, None] + np.arange(block)).reshape(upper - lower, count * block) % n
        if remainder:
            indices = np.concatenate([indices, (tails[lower:upper, None] + np.arange(remainder)) % n], axis=1)
        output[lower:upper] = values[indices].mean(axis=1)
    return output


def independent_hac(values, maxlags=126):
    values = np.asarray(values, float)
    result = sm.OLS(values, np.ones((len(values), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(values) - 1), "use_correction": False}, use_t=False)
    mean, se = float(values.mean()), float(result.bse[0])
    return {"se": se, "p": float(result.pvalues[0]) if se > 0 else float(mean == 0),
            "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "mde80_nominal": float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se)}


def fit_memory_maps(observable, latent, dimensions):
    """Direct SVD of centered records, independent of covariance eigendecomposition."""
    observable, latent = np.asarray(observable, float), np.asarray(latent, float)
    center, scale = observable.mean(axis=0), observable.std(axis=0)
    latent_center = latent.mean(axis=0)
    centered = latent - latent_center
    _, singular, rotation = linalg.svd(centered, full_matrices=False, lapack_driver="gesdd")
    values, vectors = singular[:dimensions]**2 / (len(latent) - 1), rotation[:dimensions].T
    if (scale <= 1e-12).any() or values[-1] <= 1e-12 or not np.isfinite(values).all():
        raise AssertionError("memory training geometry has zero scale/rank")
    return center, scale, latent_center, vectors, float(np.sqrt(values.sum()))


def verify_layers(features, latents, components, forecasts, audit, neighbors, protocol):
    sample, memory, ensemble = protocol["sample"], protocol["memory"], protocol["ensemble"]
    saved, component = forecasts.copy(), components.copy()
    audit, neighbors = audit.copy(), neighbors.copy()
    for frame in (saved, component):
        for column in ("origin", "target_end", "fit_origin", "train_last_target"):
            frame[column] = pd.to_datetime(frame[column])
    for column in ("origin", "key_fit_origin", "max_memory_target", "max_ensemble_target"):
        audit[column] = pd.to_datetime(audit[column])
    for column in ("origin", "neighbor_origin"):
        neighbors[column] = pd.to_datetime(neighbors[column])
    if saved.duplicated(["origin", "horizon", "model"]).any() or set(saved["model"]) != set(protocol["models"]):
        raise AssertionError("final forecast key/model set differs")
    if audit.duplicated(["origin", "horizon"]).any() or neighbors.duplicated(["origin", "horizon", "kind", "rank"]).any():
        raise AssertionError("duplicate retrieval audit rows")
    if not np.isfinite(saved[["prediction", "y"]]).all().all() or (saved[["prediction", "y"]] <= 0).any().any():
        raise AssertionError("invalid final forecast positivity")
    audit = audit.set_index(["horizon", "origin"])
    keyed_neighbors = {key: part.sort_values("rank") for key, part in neighbors.groupby(["horizon", "origin", "kind"])}
    checked, checked_neighbors, map_fits = 0, 0, 0
    expected_audits, expected_neighbors = set(), set()
    largest_error = 0.0
    for horizon in sample["horizons"]:
        historical = component.loc[component["horizon"] == horizon]
        wide = historical.pivot(index="origin", columns="model", values="prediction").sort_index().loc[:, list(COMPONENTS)]
        base_rows = historical.loc[historical["model"] == "baseline"].set_index("origin").reindex(wide.index)
        actual, endings = base_rows["y"].to_numpy(), pd.DatetimeIndex(base_rows["target_end"])
        score_dates = wide.index[(wide.index >= pd.Timestamp(sample["score_start"])) & (wide.index <= pd.Timestamp(sample["score_end"]))]
        recorded_wide = saved.loc[saved["horizon"] == horizon].pivot(index="origin", columns="model", values="prediction").sort_index()
        if not recorded_wide.index.equals(score_dates) or recorded_wide.isna().any().any():
            raise AssertionError("full common final forecast population differs")
        losses = qlike(actual[:, None], wide.to_numpy())
        observable_all = features.loc[wide.index, list(BASELINE[1:])].to_numpy(float)
        latent_all = latents.loc[wide.index].to_numpy(float)
        calendar_positions = features.index.get_indexer(wide.index)
        for _, month in pd.Series(score_dates, index=score_dates).groupby(score_dates.to_period("M")):
            dates = pd.DatetimeIndex(month.to_numpy())
            fit_origin = dates[0]
            fitting_pool = eligible_positions(features.index, wide.index, endings, fit_origin,
                                                memory["lookback_sessions"], memory["additional_gap_after_target_sessions"])
            if len(fitting_pool) < memory["min_records"]:
                raise AssertionError("fixed monthly memory warmup insufficient")
            center, scale, latent_center, vectors, latent_scale = fit_memory_maps(
                observable_all[fitting_pool], latent_all[fitting_pool], memory["pca_components"])
            observable_keys = (observable_all - center) / scale
            latent_keys = (latent_all - latent_center) @ vectors / latent_scale
            map_fits += 1
            for origin in dates:
                position = wide.index.get_indexer([origin])[0]
                current = calendar_positions[position]
                pool = eligible_positions(features.index, wide.index, endings, origin,
                                           memory["lookback_sessions"], memory["additional_gap_after_target_sessions"])
                expert_pool = eligible_positions(features.index, wide.index, endings, origin,
                                                  ensemble["lookback_sessions"], 0)
                if len(pool) < memory["min_records"] or len(expert_pool) < ensemble["min_records"]:
                    raise AssertionError("fixed memory or ensemble warmup insufficient")
                row = audit.loc[(horizon, origin)]
                expected_audits.add((horizon, origin))
                if (row["pool_n"] != len(pool) or row["ensemble_n"] != len(expert_pool)
                        or row["key_fit_origin"] != fit_origin or row["max_memory_target"] != endings[pool].max()
                        or row["max_ensemble_target"] != endings[expert_pool].max()):
                    raise AssertionError("memory timing/pool metadata differs")
                predictions = {name: float(wide.iloc[position][name]) for name in COMPONENTS}
                multiplier = ratio_correction(actual[pool], wide["baseline"].to_numpy()[pool], memory["shrinkage"])
                same(row["global_multiplier"], multiplier, "global historical OOS ratio correction")
                predictions["global_calibration"] = predictions["baseline"] * multiplier
                for kind, keys in (("observable", observable_keys), ("latent", latent_keys)):
                    selected, distance = nearest_indices(keys[pool], keys[position], memory["neighbors"])
                    selected = pool[selected]
                    key = (horizon, origin, kind)
                    expected_neighbors.add(key)
                    recorded = keyed_neighbors[key]
                    if len(recorded) != memory["neighbors"] or not pd.DatetimeIndex(recorded["neighbor_origin"]).equals(wide.index[selected]):
                        raise AssertionError(f"{kind}: independent neighbor selection differs")
                    if not np.array_equal(recorded["rank"].to_numpy(), np.arange(memory["neighbors"])):
                        raise AssertionError(f"{kind}: neighbor rank metadata differs")
                    same(recorded["distance"], distance, f"{kind} neighbor distances", rtol=2e-6, atol=1e-10)
                    same(recorded["ratio"], actual[selected] / wide["baseline"].to_numpy()[selected], f"{kind} historical error ratios")
                    multiplier = ratio_correction(actual[selected], wide["baseline"].to_numpy()[selected], memory["shrinkage"])
                    same(row[f"{kind}_multiplier"], multiplier, f"{kind} ratio correction")
                    predictions[f"{kind}_memory"] = predictions["baseline"] * multiplier
                    checked_neighbors += len(recorded)
                weights = expert_weights(losses[expert_pool], current - calendar_positions[expert_pool],
                                         ensemble["half_life_sessions"], ensemble["temperature"], ensemble["equal_weight_floor"])
                same([row[f"weight_{name}"] for name in COMPONENTS], weights, "dynamic expert weights")
                predictions["equal_ensemble"] = float(wide.iloc[position].mean())
                predictions["dynamic_ensemble"] = float(np.dot(weights, wide.iloc[position]))
                for model, estimate in predictions.items():
                    observed = recorded_wide.loc[origin, model]
                    same(observed, estimate, f"{horizon}/{origin}/{model} final forecast")
                    largest_error = max(largest_error, float(abs(observed - estimate)))
                    checked += 1
                target_rows = saved.loc[(saved["horizon"] == horizon) & (saved["origin"] == origin)]
                same(target_rows["y"], actual[position], "common final target")
                if not (target_rows["target_end"] == endings[position]).all():
                    raise AssertionError("final target completion dates differ")
    if checked != len(saved) or checked_neighbors != len(neighbors) or expected_audits != set(audit.index) or expected_neighbors != set(keyed_neighbors):
        raise AssertionError("unchecked or unexpected final forecasts/memory audit")
    return {"final_forecasts_checked": checked, "neighbor_records_checked": checked_neighbors,
            "monthly_memory_maps_checked": map_fits, "maximum_layer_prediction_error": largest_error,
            "retrieval_method": "All eligible OOS ratios, map fitting pools, direct-SVD PCA distances, stable nearest neighbors and ensemble weights independently reconstructed"}


def verify_metrics(forecasts, features, recorded, protocol, comparisons=None, hypothesis_count=None):
    if comparisons is None:
        comparisons = [(name, "baseline", "primary") for name in protocol["models"] if name != "baseline"]
        comparisons += [(a, b, "mechanism") for a, b in protocol["comparisons"]["mechanisms"]]
    if hypothesis_count is None:
        hypothesis_count = protocol["comparisons"]["total_hypotheses"]
    rows = recorded["rows"]
    keyed = {(int(row["horizon"]), row["candidate"], row["control"]): row for row in rows}
    expected = {(int(h), a, b) for h in protocol["sample"]["horizons"] for a, b, _ in comparisons}
    if len(rows) != hypothesis_count or set(keyed) != expected:
        raise AssertionError(f"inference does not include exactly the fixed {hypothesis_count} contrasts")
    inferred, improvements, stable, summaries = {}, {}, {}, []
    settings = protocol["inference"]
    for horizon in protocol["sample"]["horizons"]:
        panel = forecasts.loc[forecasts["horizon"] == horizon].copy()
        panel["origin"] = pd.to_datetime(panel["origin"])
        wide = panel.pivot(index="origin", columns="model", values="prediction").sort_index()
        actual = panel.drop_duplicates("origin").set_index("origin")["y"].reindex(wide.index).to_numpy()
        losses = {name: qlike(actual, wide[name].to_numpy()) for name in wide.columns}
        differences = np.column_stack([losses[a] - losses[b] for a, b, _ in comparisons])
        means = differences.mean(axis=0)
        bootstrap = {}
        for block in settings["blocks"]:
            draws = explicit_bootstrap_means(differences, block, settings["bootstrap_draws"], settings["seed"] + int(horizon) * 1000 + block)
            bootstrap[str(block)] = {"ci95": np.quantile(draws, [0.025, 0.975], axis=0),
                                     "p": (1 + (np.abs(draws - means) >= np.abs(means)).sum(axis=0)) / (len(draws) + 1)}
        for column, (candidate, control, role) in enumerate(comparisons):
            key = (horizon, candidate, control)
            row = keyed[key]
            difference = differences[:, column]
            base, challenger = losses[control], losses[candidate]
            improvement = float(-100 * difference.mean() / base.mean())
            improvements[key] = improvement
            if row["n"] != len(wide) or row["role"] != role:
                raise AssertionError("contrast role/count differs")
            for name, value in (("baseline_qlike", base.mean()), ("candidate_qlike", challenger.mean()),
                                ("delta", difference.mean()), ("improvement_pct", improvement)):
                same(row[name], value, f"{key}/{name}", rtol=1e-10)
            if "best_component_qlike" in row:
                best = min(losses[name].mean() for name in COMPONENTS)
                same(row["best_component_qlike"], best, f"{key}/best component point")
                if row["beats_best_component_point"] != bool(challenger.mean() < best):
                    raise AssertionError("ensemble best-component point check differs")
            if "win_rate" in row:
                same(row["win_rate"], np.mean(difference < 0), f"{key}/win rate")
            hac = independent_hac(difference)
            for name, value in hac.items():
                same(row["hac126"][name], value, f"{key}/HAC/{name}", rtol=1e-9)
            ps, lower, upper = [hac["p"]], [hac["ci95"][0]], [hac["ci95"][1]]
            if set(row["block_inference"]) != set(bootstrap):
                raise AssertionError("registered bootstrap blocks differ")
            for block, result in bootstrap.items():
                interval, pvalue = result["ci95"][:, column], result["p"][column]
                same(row["block_inference"][block]["ci95"], interval, f"{key}/bootstrap {block} interval", rtol=1e-9)
                same(row["block_inference"][block]["p"], pvalue, f"{key}/bootstrap {block} p", rtol=0, atol=1e-14)
                ps.append(pvalue)
                lower.append(interval[0])
                upper.append(interval[1])
            inferred[key] = max(ps)
            same(row["p_conservative"], max(ps), f"{key}/conservative p", rtol=1e-9)
            same(row["ci95_envelope"], [min(lower), max(upper)], f"{key}/interval envelope")
            period_gaps = []
            if len(row["periods"]) != len(settings["stability_periods"]):
                raise AssertionError("fixed stability period set differs")
            for saved, (begin, end) in zip(row["periods"], settings["stability_periods"], strict=True):
                selected = (wide.index >= pd.Timestamp(begin)) & (wide.index <= pd.Timestamp(end))
                if saved["start"] != begin or saved["end"] != end or saved["n"] != int(selected.sum()):
                    raise AssertionError("stability dates/counts differ")
                delta = float(difference[selected].mean())
                same(saved["delta"], delta, f"{key}/period delta")
                period_gaps.append(delta)
            stable[key] = all(delta < 0 for delta in period_gaps)
            annual = {int(item["year"]): item for item in row["annual"]}
            if set(annual) != set(wide.index.year) or len(annual) != len(row["annual"]):
                raise AssertionError("annual slices differ")
            for year, saved in annual.items():
                selected = wide.index.year == year
                if saved["n"] != int(selected.sum()):
                    raise AssertionError("annual count differs")
                same(saved["delta"], difference[selected].mean(), f"{key}/annual delta")
            phases = row.get("phases", row.get("nonoverlap_phases"))
            if len(phases) != horizon:
                raise AssertionError("phase set differs")
            membership = features.index.get_indexer(wide.index) % horizon
            for phase, saved in enumerate(phases):
                selected = membership == phase
                if saved["phase"] != phase or saved["n"] != int(selected.sum()):
                    raise AssertionError("nonoverlap phase membership differs")
                same(saved["delta"], difference[selected].mean(), f"{key}/phase delta")
    keys = list(keyed)
    adjusted = multipletests([inferred[key] for key in keys], method="holm")[1]
    for key, pvalue in zip(keys, adjusted, strict=True):
        row = keyed[key]
        same(row["p_holm"], pvalue, f"{key}/Holm", rtol=1e-9)
        verdict = "EXPLORATORY_SHORTLIST" if pvalue < 0.05 and improvements[key] >= 1 and stable[key] else "INCONCLUSIVE"
        if row["verdict"] != verdict:
            raise AssertionError("fixed joint inference gate differs")
        summaries.append({"horizon": key[0], "candidate": key[1], "control": key[2], "p_holm": float(pvalue), "verdict": verdict})
    return {"contrasts_checked": len(rows), "independent_inference_verification": "PASS",
            "verified_verdicts": summaries, "inference_method": f"Explicit circular bootstrap row indices for all draws; statsmodels Bartlett HAC and Holm across {hypothesis_count} contrasts"}


def verify(root=ROOT):
    protocol_path = root / "model_memory_study.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    report_dir, data_dir = root / protocol["outputs"]["reports"], root / protocol["outputs"]["data"]
    manifest = json.loads((report_dir / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("registered protocol identity changed")
    for category in ("inputs", "code", "existing_artifacts_sha256"):
        for path, expected in manifest[category].items():
            if digest(root / path) != expected:
                raise AssertionError(f"{category} identity changed: {path}")
    if tuple(protocol["baseline_columns"]) != BASELINE or tuple(protocol["component_models"]) != COMPONENTS:
        raise AssertionError("registered feature/model schema differs")
    features = reconstruct_features(root, protocol)
    cached = pd.read_parquet(root / protocol["inputs"]["features"]).loc[:protocol["sample"]["latest_target"]]
    if not cached.index.equals(features.index):
        raise AssertionError("bounded cached feature date population differs from raw sources")
    for column in features.columns:
        same(cached[column].to_numpy(), features[column].to_numpy(), f"source feature {column}", rtol=1e-10)
    latents = pd.read_parquet(root / protocol["inputs"]["latents"]).loc[:protocol["sample"]["score_end"]]
    if (list(latents.columns) != [f"z{i:03d}" for i in range(512)]
            or latents.index.has_duplicates or not latents.index.is_monotonic_increasing):
        raise AssertionError("latent source date/coordinate schema differs")
    if not np.isfinite(latents).all().all():
        raise AssertionError("latent cache contains missing values")
    components = pd.read_parquet(data_dir / "components.parquet")
    fits = json.loads((data_dir / "estimator_fits.json").read_text())
    result = verify_components(features, latents, components, fits, protocol)
    forecasts = pd.read_parquet(data_dir / "forecasts.parquet")
    memory_audit = pd.read_parquet(data_dir / "memory_audit.parquet")
    neighbors = pd.read_parquet(data_dir / "neighbors.parquet")
    result.update(verify_layers(features, latents, components, forecasts, memory_audit, neighbors, protocol))
    metrics = json.loads((report_dir / "metrics.json").read_text())
    if metrics["protocol_sha256"] != manifest["protocol_sha256"]:
        raise AssertionError("metric identity differs")
    result.update(verify_metrics(forecasts, features, metrics, protocol))
    result.update({"status": "PASS", "independent_feature_verification": "PASS",
                   "feature_rows": len(features), "preserved_artifact_count": len(manifest["existing_artifacts_sha256"]),
                   "protocol_sha256": digest(protocol_path), "verifier_sha256": digest(Path(__file__)),
                   "limitations": "Cached TiRex representations are hash-bound but checkpoint pretraining cleanliness is unknown. Parent AR1 calibration is not independently rerun. Historical evidence remains exploratory."})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify(args.root)
    protocol = yaml.safe_load((args.root / "model_memory_study.yaml").read_text())
    path = args.root / protocol["outputs"]["reports"] / "verification.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
