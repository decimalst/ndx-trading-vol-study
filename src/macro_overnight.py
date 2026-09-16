"""Source-lagged QQQ overnight second moments and fixed monthly forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .macro_second_moment import fit_second_moment
from .overnight_index import _dates, _prices

BASE = ("const", "on_rms_d", "on_rms_w", "on_rms_m", "day_rms_d", "day_rms_w", "day_rms_m",
        "lrv_d", "lrv_w", "lrv_m", "liv", "lvix", "entry_dow_1", "entry_dow_2", "entry_dow_3",
        "entry_dow_4", "nominal_hours")
ADDITIONS = {"cpi": "cpi_plan", "nfp": "nfp_plan", "fomc": "fomc_plan"}
ALL_FEATURES = BASE+tuple(ADDITIONS.values())
MODELS = ("mean", "baseline", *ADDITIONS)


def build_features(daily, iv, plans):
    for frame in (daily, iv, plans):
        _dates(frame.index)
    _prices(daily, ["open", "high", "low", "close", "adj close"])
    _prices(iv, ["vxn", "vix"])
    d = daily
    complete = d[["open", "high", "low", "close"]].dropna()
    if ((complete.high < complete[["open", "low", "close"]].max(axis=1)).any()
            or (complete.low > complete[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLC ranges")
    calendar = plans.reindex(d.index)
    for column in ADDITIONS.values():
        values = calendar[column].dropna()
        if not values.isin([0., 1.]).all():
            raise ValueError("Calendar plans must be zero, one, or unknown")
    if (calendar.nominal_hours.dropna() <= 0).any():
        raise ValueError("Positive nominal duration required")
    day = np.log(d.close/d.open)
    overnight = np.log(d["adj close"]).diff()-day
    gk = (.5*np.log(d.high/d.low)**2-(2*np.log(2)-1)*day**2).clip(lower=1e-10)
    variance = gk+np.log(d.open/d.close.shift(1))**2
    features = pd.DataFrame(index=d.index)
    for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
        features["on_rms_"+suffix] = np.sqrt(overnight.pow(2).rolling(width, min_periods=width).mean())
        features["day_rms_"+suffix] = np.sqrt(day.pow(2).rolling(width, min_periods=width).mean())
        features["lrv_"+suffix] = np.log(variance.rolling(width, min_periods=width).mean())
    aligned_iv = iv.reindex(d.index)
    features["liv"], features["lvix"] = np.log(aligned_iv.vxn), np.log(aligned_iv.vix)
    features = features.shift(1)
    features["const"] = 1.
    for weekday in range(1, 5):
        features[f"entry_dow_{weekday}"] = (d.index.weekday == weekday).astype(float)
    for column in ("nominal_hours", *ADDITIONS.values()):
        features[column] = calendar[column]
    features = features.loc[:, ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
    dates = pd.Series(d.index, index=d.index)
    features["feature_cutoff_date"] = dates.shift(1)
    factor = d["adj close"]/d.close
    adjustment = np.log(factor.shift(-1)/factor)
    event = pd.Series(pd.NA, index=d.index, dtype="boolean")
    finite_adjustment = np.isfinite(adjustment)
    event.loc[finite_adjustment] = adjustment.loc[finite_adjustment].abs() > 1e-5
    target_log = overnight.shift(-1)
    targets = pd.DataFrame({"y": target_log.pow(2).where(np.isfinite(target_log)),
                            "target_end": dates.shift(-1), "available_date": dates.shift(-1),
                            "adjustment_event": event, "adjustment_log_change": adjustment}, index=d.index)
    return features, targets


def _alignment(features, targets):
    _dates(features.index)
    if not features.index.equals(targets.index):
        raise ValueError("Features and labels need identical entry calendars")
    if not targets.available_date.equals(targets.target_end.rename("available_date")):
        raise ValueError("Proxy labels must mature at the actual next session close")
    known = features.feature_cutoff_date.notna()
    if (features.loc[known, "feature_cutoff_date"] >= features.index[known]).any():
        raise ValueError("Feature cutoff must precede entry")
    observed = targets.target_end.notna()
    if (targets.loc[observed, "target_end"] <= features.index[observed]).any() or (targets.y.dropna() < 0).any():
        raise ValueError("Future nonnegative second-moment labels required")


def training_mask(features, targets, fit_entry, min_train=1000):
    _alignment(features, targets)
    fit_entry = pd.Timestamp(fit_entry)
    if fit_entry not in features.index or min_train < 2:
        raise ValueError("Invalid fit origin or minimum sample")
    cutoff = features.loc[fit_entry, "feature_cutoff_date"]
    if pd.isna(cutoff):
        raise ValueError("INSUFFICIENT_DATA: no previous session")
    mask = (np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
            & np.isfinite(targets.y) & (features.index < fit_entry)
            & (targets.available_date <= cutoff))
    if int(mask.sum()) < min_train:
        raise ValueError(f"INSUFFICIENT_DATA: {int(mask.sum())} completed common training labels")
    return mask


def fit_predict(features, y, application):
    target = np.asarray(y, float)
    if isinstance(y, pd.Series) and not y.index.equals(features.index):
        raise ValueError("Training target row ordering differs")
    if (target.shape != (len(features),) or not np.isfinite(target).all() or (target < 0).any()
            or not np.isfinite(features.loc[:, ALL_FEATURES]).all().all()
            or not np.isfinite(application.loc[:, ALL_FEATURES]).all().all()
            or not (features.const == 1).all() or not (application.const == 1).all()
            or target.mean() <= 0):
        raise ValueError("Finite common designs and positive target mean required")
    if (features.loc[:, ALL_FEATURES[1:]].std(ddof=0) <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale input; no fallback")
    result = {"mean": np.full(len(application), target.mean())}
    audits = {"mean": {"columns": ["const"], "beta": [float(np.log(target.mean()))],
                       "train_mean": float(target.mean()), "train_n": len(target), "gradient_max_abs": 0.}}
    for name in MODELS[1:]:
        columns = BASE[1:]+(() if name == "baseline" else (ADDITIONS[name],))
        fitted = fit_second_moment(features.loc[:, columns], target, application.loc[:, columns])
        result[name] = fitted.pop("prediction")
        audits[name] = {"columns": ["const", *columns], **{
            key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in fitted.items()
        }}
    return result, audits


def forecast_panel(features, targets, config):
    """Monthly refit schedule depends on features; query labels only choose score rows."""
    _alignment(features, targets)
    if tuple(config["models"]) != MODELS or tuple(config["baseline"]) != BASE:
        raise ValueError("Fixed macro feature/model family differs")
    start, end, latest = map(pd.Timestamp, [config["origin_start"], config["origin_end"], config["latest_target"]])
    dev_start, dev_end = map(pd.Timestamp, config["development"])
    ev_start, ev_end = map(pd.Timestamp, config["evaluation"])
    if (not start <= dev_start <= dev_end < ev_start <= ev_end <= end < latest
            or config["development_target_available_by"] != config["development"][1]):
        raise ValueError("Invalid fixed historical phases")
    dates = features.index
    phases = ((dates >= dev_start) & (dates <= dev_end)) | ((dates >= ev_start) & (dates <= ev_end))
    feature_ready = np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
    entries = dates[feature_ready & phases & (dates >= start) & (dates <= end)]
    if not len(entries):
        raise ValueError("INSUFFICIENT_DATA: no common complete entry features")
    label_ready = (np.isfinite(targets.y) & np.isfinite(targets.adjustment_log_change)
                   & targets.adjustment_event.notna() & (targets.available_date <= latest)
                   & ((dates > dev_end) | (targets.available_date <= dev_end)))
    rows, fits = [], []
    for month in entries.to_period("M").unique():
        application_dates = entries[entries.to_period("M") == month]
        fit_entry = application_dates[0]
        mask = training_mask(features, targets, fit_entry, config["minimum_train"])
        predictions, audit = fit_predict(features.loc[mask], targets.loc[mask, "y"], features.loc[application_dates])
        cutoff = features.loc[fit_entry, "feature_cutoff_date"]
        last_target = targets.loc[mask, "target_end"].max()
        fits.append({"fit_origin": str(fit_entry.date()), "fit_cutoff_date": str(cutoff.date()),
                     "train_n": int(mask.sum()), "train_first_origin": str(features.index[mask][0].date()),
                     "train_last_origin": str(features.index[mask][-1].date()),
                     "train_last_target": str(last_target.date()), "train_last_available": str(last_target.date()),
                     "application_n": len(application_dates), "model_audit": audit})
        keep = label_ready.loc[application_dates].to_numpy()
        scored = application_dates[keep]
        for model in MODELS:
            rows.append(pd.DataFrame({"origin": scored, "model": model, "horizon": 1,
                                     "feature_cutoff_date": features.loc[scored, "feature_cutoff_date"].to_numpy(),
                                     "target_end": targets.loc[scored, "target_end"].to_numpy(),
                                     "available_date": targets.loc[scored, "available_date"].to_numpy(),
                                     "y": targets.loc[scored, "y"].to_numpy(), "prediction": predictions[model][keep],
                                     "adjustment_event": targets.loc[scored, "adjustment_event"].to_numpy(dtype=bool),
                                     "adjustment_log_change": targets.loc[scored, "adjustment_log_change"].to_numpy(),
                                     "fit_origin": fit_entry, "fit_cutoff_date": cutoff, "train_n": int(mask.sum()),
                                     "train_last_target": last_target, "train_last_available": last_target,
                                     "phase": np.where(scored <= dev_end, "development", "evaluation")}))
    panel = pd.concat(rows, ignore_index=True).sort_values(["origin", "model"]).reset_index(drop=True)
    if panel.empty:
        raise ValueError("INSUFFICIENT_DATA: no scoreable common labels")
    return panel, fits
