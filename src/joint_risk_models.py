"""Shared conditional means/second moments with one bounded dependence increment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import joint_risk_density as density
from .index_hinge import _dates
from .joint_risk_features import ALL_FEATURES, MODELS
from .macro_second_moment import fit_second_moment

TARGETS = ["y_qqq", "y_spx"]
MARGINAL_COLUMNS = (*ALL_FEATURES, "corr22_centered_sq")
ALPHA = 0.01
MIN_CORRELATION_EIGENVALUE = 1e-6


def _alignment(features, targets):
    _dates(features.index)
    if not features.index.equals(targets.index):
        raise ValueError("Features and labels need identical entry calendars")
    dates = pd.Series(features.index, index=features.index)
    for frame, name, shift in [
        (features, "feature_cutoff_date", 1),
        (targets, "target_end", -1),
        (targets, "available_date", -1),
    ]:
        if not frame[name].equals(dates.shift(shift).rename(name)):
            raise ValueError(
                "Exact previous-session features and next-session labels required"
            )
    for name in TARGETS:
        known = targets[name].notna()
        if (
            not np.isfinite(targets.loc[known, name]).all()
            or targets.loc[known, "target_end"].isna().any()
        ):
            raise ValueError("Finite signed paired next-session return labels required")


def training_mask(features, targets, fit_entry, min_train=1000):
    _alignment(features, targets)
    fit_entry = pd.Timestamp(fit_entry)
    if fit_entry not in features.index or min_train < 2:
        raise ValueError("Invalid fit origin or minimum sample")
    cutoff = features.loc[fit_entry, "feature_cutoff_date"]
    if pd.isna(cutoff):
        raise ValueError("INSUFFICIENT_DATA: no previous session")
    mask = (
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
        & np.isfinite(targets[TARGETS]).all(axis=1)
        & (features.index < fit_entry)
        & (targets.available_date <= cutoff)
    )
    if int(mask.sum()) < min_train:
        raise ValueError(
            f"INSUFFICIENT_DATA: {int(mask.sum())} completed common training labels"
        )
    return mask


def transform(train, apply):
    tr, ap = train.loc[:, ALL_FEATURES].copy(), apply.loc[:, ALL_FEATURES].copy()
    if (
        len(tr) < 2
        or not np.isfinite(tr).all().all()
        or not np.isfinite(ap).all().all()
        or not tr.const.eq(1).all()
        or not ap.const.eq(1).all()
    ):
        raise ValueError("Finite aligned common input designs and unit intercept required")
    center = float(tr.corr22.mean())
    for frame in [tr, ap]:
        frame["corr22_centered_sq"] = (frame.corr22 - center) ** 2
    scales = tr.iloc[:, 1:].std(ddof=0).to_numpy()
    if not np.isfinite(scales).all() or (scales <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale common feature; no fallback")
    return tr, ap, {"corr22_mean": center}


def _serializable(audit):
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in audit.items()
    }


def _mean_fit(x, y, query):
    means, scales = x.mean(axis=0), x.std(axis=0, ddof=0)
    z, app = (x - means) / scales, (query - means) / scales
    center = float(y.mean())
    slope = np.linalg.solve(
        z.T @ z / len(z) + ALPHA * np.eye(z.shape[1]), z.T @ (y - center) / len(z)
    )
    design, beta = np.c_[np.ones(len(z)), z], np.r_[center, slope]
    fitted, prediction = design @ beta, center + app @ slope
    gradient = 2 * design.T @ (fitted - y) / len(z)
    gradient[1:] += 2 * ALPHA * slope
    if (
        not np.isfinite([*fitted, *prediction, *gradient]).all()
        or np.max(np.abs(gradient)) > 1e-10
    ):
        raise ValueError("Invalid shared mean fit or full MSE gradient; no fallback")
    return (
        fitted,
        prediction,
        {
            "columns": list(MARGINAL_COLUMNS),
            "means": [0.0, *means.tolist()],
            "scales": [1.0, *scales.tolist()],
            "beta": beta.tolist(),
            "alpha": ALPHA,
            "train_n": len(y),
            "gradient_max_abs": float(np.max(np.abs(gradient))),
        },
    )


def fit_predict(train, targets, apply):
    if not train.index.equals(targets.index):
        raise ValueError("Training label row ordering differs")
    y = targets[TARGETS].to_numpy(float)
    if y.shape != (len(train), 2) or not np.isfinite(y).all():
        raise ValueError("Finite signed paired return targets required")
    tr, ap, transform_audit = transform(train, apply)
    x, query = tr.iloc[:, 1:].to_numpy(float), ap.iloc[:, 1:].to_numpy(float)
    residual, innovations, mu, h = [], [], [], []
    moment_audit = {}
    for j, asset in enumerate(["qqq", "spx"]):
        fitted, prediction, mean_audit = _mean_fit(x, y[:, j], query)
        e = y[:, j] - fitted
        variance_audit = fit_second_moment(x, e**2, np.concatenate([x, query]))
        all_h = variance_audit.pop("prediction")
        residual.append(e)
        innovations.append(e / np.sqrt(all_h[: len(x)]))
        mu.append(prediction)
        h.append(all_h[len(x) :])
        moment_audit[asset] = {
            "mean": mean_audit,
            "variance": {"columns": list(MARGINAL_COLUMNS), **_serializable(variance_audit)},
        }
    residual, innovations = np.column_stack(residual), np.column_stack(innovations)
    mu, h = np.column_stack(mu), np.column_stack(h)
    matrix = residual.T @ residual / len(residual)
    diag = np.diag(matrix)
    if not np.isfinite(matrix).all() or (diag <= 0).any():
        raise ValueError("INSUFFICIENT_DATA: nonpositive constant residual second moment")
    rho_matrix = float(matrix[0, 1] / (np.sqrt(diag[0]) * np.sqrt(diag[1])))
    if not np.isfinite(rho_matrix) or abs(rho_matrix) > 1 - MIN_CORRELATION_EIGENVALUE:
        raise ValueError(
            "INSUFFICIENT_DATA: ill-conditioned constant residual correlation; no jitter"
        )
    corr_scale = float(tr.corr22.std(ddof=0))
    z = (tr.corr22.to_numpy() - transform_audit["corr22_mean"]) / corr_scale
    app_z = (ap.corr22.to_numpy() - transform_audit["corr22_mean"]) / corr_scale
    constant = density.fit_dependence(innovations)
    dynamic = density.fit_dependence(innovations, z, constant=constant["parameter"])
    rho0 = np.full(len(ap), 0.995 * np.tanh(constant["parameter"]))
    rho1 = 0.995 * np.tanh(constant["parameter"] + dynamic["parameter"] * app_z)
    forecasts = {
        "constant_matrix": {
            "mu": mu.copy(),
            "h": np.tile(diag, (len(ap), 1)),
            "rho": np.full(len(ap), rho_matrix),
        },
        "constant_correlation": {"mu": mu.copy(), "h": h.copy(), "rho": rho0},
        "dynamic_correlation": {"mu": mu.copy(), "h": h.copy(), "rho": rho1},
    }
    for p in forecasts.values():
        density.matrix_score(np.zeros_like(p["mu"]), p["h"], p["rho"])
    return {
        "forecasts": forecasts,
        "audit": {
            "moments": moment_audit,
            "transform": transform_audit,
            "residual_staging": "current_fit_training_residuals",
            "constant_matrix": {
                "matrix": matrix.tolist(),
                "rho": rho_matrix,
                "minimum_correlation_eigenvalue": 1 - abs(rho_matrix),
                "train_n": len(train),
            },
            "dependence": {"constant_correlation": constant, "dynamic_correlation": dynamic},
            "corr22_scale": corr_scale,
        },
    }


def forecast_panel(features, targets, config):
    _alignment(features, targets)
    if tuple(config["models"]) != MODELS or tuple(config["all_features"]) != ALL_FEATURES:
        raise ValueError("Fixed joint-risk feature/model family differs")
    start, end, latest = map(
        pd.Timestamp, [config["origin_start"], config["origin_end"], config["latest_target"]]
    )
    dev_start, dev_end = map(pd.Timestamp, config["development"])
    ev_start, ev_end = map(pd.Timestamp, config["evaluation"])
    if (
        not start <= dev_start <= dev_end < ev_start <= ev_end <= end < latest
        or config["development_target_available_by"] != config["development"][1]
    ):
        raise ValueError("Invalid fixed historical phases")
    dates = features.index
    phases = ((dates >= dev_start) & (dates <= dev_end)) | (
        (dates >= ev_start) & (dates <= ev_end)
    )
    ready = np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
    entries = dates[ready & phases & (dates >= start) & (dates <= end)]
    if not len(entries):
        raise ValueError("INSUFFICIENT_DATA: no common complete features")
    labels = (
        np.isfinite(targets[TARGETS]).all(axis=1)
        & (targets.available_date <= latest)
        & ((dates > dev_end) | (targets.available_date <= dev_end))
    )
    rows, fits = [], []
    for month in entries.to_period("M").unique():
        application_dates = entries[entries.to_period("M") == month]
        fit_entry = application_dates[0]
        mask = training_mask(features, targets, fit_entry, config["minimum_train"])
        result = fit_predict(
            features.loc[mask], targets.loc[mask], features.loc[application_dates]
        )
        cutoff = features.loc[fit_entry, "feature_cutoff_date"]
        last_target = targets.loc[mask, "target_end"].max()
        fits.append(
            {
                "fit_origin": str(fit_entry.date()),
                "fit_cutoff_date": str(cutoff.date()),
                "train_n": int(mask.sum()),
                "train_first_origin": str(features.index[mask][0].date()),
                "train_last_origin": str(features.index[mask][-1].date()),
                "train_last_target": str(last_target.date()),
                "train_last_available": str(last_target.date()),
                "application_n": len(application_dates),
                "model_audit": result["audit"],
            }
        )
        keep = labels.loc[application_dates].to_numpy()
        scored = application_dates[keep]
        if not len(scored):
            continue
        actual = targets.loc[scored, TARGETS].to_numpy()
        for name in MODELS:
            p = result["forecasts"][name]
            mu, h, rho = p["mu"][keep], p["h"][keep], p["rho"][keep]
            rows.append(
                pd.DataFrame(
                    {
                        "origin": scored,
                        "model": name,
                        "horizon": 1,
                        "feature_cutoff_date": features.loc[
                            scored, "feature_cutoff_date"
                        ].to_numpy(),
                        "target_end": targets.loc[scored, "target_end"].to_numpy(),
                        "available_date": targets.loc[scored, "available_date"].to_numpy(),
                        "y_qqq": actual[:, 0],
                        "y_spx": actual[:, 1],
                        "mu_qqq": mu[:, 0],
                        "mu_spx": mu[:, 1],
                        "h_qqq": h[:, 0],
                        "h_spx": h[:, 1],
                        "rho": rho,
                        "loss": density.matrix_score(actual - mu, h, rho),
                        "fit_origin": fit_entry,
                        "fit_cutoff_date": cutoff,
                        "train_n": int(mask.sum()),
                        "train_last_target": last_target,
                        "train_last_available": last_target,
                        "phase": np.where(scored <= dev_end, "development", "evaluation"),
                    }
                )
            )
    if not rows:
        raise ValueError("INSUFFICIENT_DATA: no scoreable common labels")
    panel = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    if panel.empty:
        raise ValueError("INSUFFICIENT_DATA: no scoreable common labels")
    return panel, fits
