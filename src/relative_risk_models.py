"""Fixed normalized ridge models for relative QQQ-SPX intraday log risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .index_hinge import _dates
from .relative_risk_features import ALL_FEATURES, BASE, MODELS


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
                "Exact previous-session features and next-session label dates required"
            )
    known = targets.y.notna()
    if (
        not np.isfinite(targets.loc[known, "y"]).all()
        or targets.loc[known, "target_end"].isna().any()
    ):
        raise ValueError("Finite signed next-session log-risk labels required")


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
        & np.isfinite(targets.y)
        & (features.index < fit_entry)
        & (targets.available_date <= cutoff)
    )
    if int(mask.sum()) < min_train:
        raise ValueError(
            f"INSUFFICIENT_DATA: {int(mask.sum())} completed common training labels"
        )
    return mask


def fit_predict(features, y, application):
    """One declared normalized ridge penalty with the same complete rows for all arms."""
    if len(features) < 2:
        raise ValueError("At least two training rows required")
    if isinstance(y, pd.Series) and not y.index.equals(features.index):
        raise ValueError("Training target row ordering differs")
    target = np.asarray(y, float)
    train = features.loc[:, ALL_FEATURES]
    query = application.loc[:, ALL_FEATURES]
    if (
        target.shape != (len(train),)
        or not np.isfinite(target).all()
        or not np.isfinite(train).all().all()
        or not np.isfinite(query).all().all()
        or not train.const.eq(1).all()
        or not query.const.eq(1).all()
    ):
        raise ValueError("Finite aligned common designs and signed targets required")
    scales = train.loc[:, ALL_FEATURES[1:]].std(ddof=0).to_numpy()
    if not np.isfinite(scales).all() or (scales <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale common feature; no fallback")
    mean = float(target.mean())
    predictions = {"mean": np.full(len(query), mean)}
    audits = {
        "mean": {
            "columns": ["const"],
            "means": [0.0],
            "scales": [1.0],
            "beta": [mean],
            "alpha": 0.0,
            "train_n": len(train),
            "gradient_max_abs": float(abs(2 * np.mean(mean - target))),
        }
    }
    for name in MODELS[1:]:
        columns = BASE[1:] if name == "baseline" else ALL_FEATURES[1:]
        x, q = train.loc[:, columns].to_numpy(float), query.loc[:, columns].to_numpy(float)
        means, scales = x.mean(axis=0), x.std(axis=0, ddof=0)
        z, app = (x - means) / scales, (q - means) / scales
        centered = target - mean
        slopes = np.linalg.solve(
            z.T @ z / len(z) + 0.01 * np.eye(len(columns)), z.T @ centered / len(z)
        )
        prediction = mean + app @ slopes
        design, beta = np.c_[np.ones(len(z)), z], np.r_[mean, slopes]
        gradient = 2 * design.T @ (design @ beta - target) / len(z)
        gradient[1:] += 0.02 * slopes
        maximum = float(np.max(np.abs(gradient)))
        if (
            not np.isfinite(prediction).all()
            or not np.isfinite(gradient).all()
            or maximum > 1e-10
        ):
            raise ValueError("Invalid ridge forecast or full mean-loss gradient; no fallback")
        predictions[name] = prediction
        audits[name] = {
            "columns": ["const", *columns],
            "means": [0.0, *means.tolist()],
            "scales": [1.0, *scales.tolist()],
            "beta": beta.tolist(),
            "alpha": 0.01,
            "train_n": len(train),
            "gradient_max_abs": maximum,
        }
    return predictions, audits


def forecast_panel(features, targets, config):
    """Monthly refit schedule depends on features; query labels only choose score rows."""
    _alignment(features, targets)
    if tuple(config["models"]) != MODELS or tuple(config["baseline"]) != BASE:
        raise ValueError("Fixed relative-risk feature/model family differs")
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
    feature_ready = np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
    entries = dates[feature_ready & phases & (dates >= start) & (dates <= end)]
    if not len(entries):
        raise ValueError("INSUFFICIENT_DATA: no common complete entry features")
    label_ready = (
        np.isfinite(targets.y)
        & (targets.available_date <= latest)
        & ((dates > dev_end) | (targets.available_date <= dev_end))
    )
    rows, fits = [], []
    for month in entries.to_period("M").unique():
        application_dates = entries[entries.to_period("M") == month]
        fit_entry = application_dates[0]
        mask = training_mask(features, targets, fit_entry, config["minimum_train"])
        predictions, audit = fit_predict(
            features.loc[mask], targets.loc[mask, "y"], features.loc[application_dates]
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
                "model_audit": audit,
            }
        )
        keep = label_ready.loc[application_dates].to_numpy()
        scored = application_dates[keep]
        for model in MODELS:
            rows.append(
                pd.DataFrame(
                    {
                        "origin": scored,
                        "model": model,
                        "horizon": 1,
                        "feature_cutoff_date": features.loc[
                            scored, "feature_cutoff_date"
                        ].to_numpy(),
                        "target_end": targets.loc[scored, "target_end"].to_numpy(),
                        "available_date": targets.loc[scored, "available_date"].to_numpy(),
                        "y": targets.loc[scored, "y"].to_numpy(),
                        "prediction": predictions[model][keep],
                        "fit_origin": fit_entry,
                        "fit_cutoff_date": cutoff,
                        "train_n": int(mask.sum()),
                        "train_last_target": last_target,
                        "train_last_available": last_target,
                        "phase": np.where(scored <= dev_end, "development", "evaluation"),
                    }
                )
            )
    panel = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    if panel.empty:
        raise ValueError("INSUFFICIENT_DATA: no scoreable common labels")
    return panel, fits
