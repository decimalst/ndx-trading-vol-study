"""Entry-compatible historical overnight return proxy, with fixed causal inputs.

Every market feature ends at the prior session close. The target's adjusted
price convention is deliberately separate from dividend cash-flow accounting.
Empirical execution requires the caller's frozen protocol and prewritten tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ("const", "on_d", "on_w", "on_m", "day_d", "day_w", "day_m",
        "lrv_d", "lrv_w", "lrv_m", "liv", "lvix",
        "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
BLOCKS = {"cross_signed": ("x_hyg", "x_tlt", "x_gld", "x_uso", "x_uup"),
          "volume_pressure": ("pressure_d", "pressure_w"), "iv_shape": ("term", "lvvix")}
ALL_FEATURES = BASE + sum(BLOCKS.values(), ())
MODELS = ("mean", "baseline", *BLOCKS)
ALPHA = .01
MIN_TRAIN = 1000
EVENT_THRESHOLD = 1e-5


def _dates(index):
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates
            or not index.is_monotonic_increasing or index.tz is not None
            or not index.equals(index.normalize())):
        raise ValueError("Unique sorted normalized timezone-naive sessions required")


def _prices(frame, columns):
    values = frame.loc[:, columns].to_numpy(float)
    if np.isinf(values).any() or ((values <= 0) & np.isfinite(values)).any():
        raise ValueError("Positive prices required where observed, without infinity")


def build_features(daily, cross, iv):
    """Return prior-close market predictors and next-opening proxy labels.

    Invalid volume becomes missing. Missing rows remain in the calendar and
    strict rolling windows rather than being filled or removed before rolling.
    """
    for frame in (daily, cross, iv):
        _dates(frame.index)
    _prices(daily, ["open", "high", "low", "close", "adj close"])
    _prices(cross, ["hyg", "tlt", "gld", "uso", "uup"])
    _prices(iv, ["vxn", "vix", "vix9d", "vvix"])
    observed = daily[["open", "high", "low", "close"]].notna().all(axis=1)
    complete = daily.loc[observed]
    if ((complete.high < complete[["open", "close", "low"]].max(axis=1)).any()
            or (complete.low > complete[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLC ranges")
    d = daily
    day = np.log(d.close/d.open)
    overnight = np.log(d["adj close"]).diff()-day
    gk = (.5*np.log(d.high/d.low)**2 - (2*np.log(2)-1)*day**2).clip(lower=1e-10)
    variance = gk + np.log(d.open/d.close.shift(1))**2
    f = pd.DataFrame(index=d.index)
    for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
        f["on_"+suffix] = overnight.rolling(width, min_periods=width).mean()
        f["day_"+suffix] = day.rolling(width, min_periods=width).mean()
        f["lrv_"+suffix] = np.log(variance.rolling(width, min_periods=width).mean())
    aligned_iv = iv.reindex(d.index)
    for output, source in [("liv", "vxn"), ("lvix", "vix"), ("lvvix", "vvix")]:
        f[output] = np.log(aligned_iv[source])
    f["term"] = np.log(aligned_iv.vix9d/aligned_iv.vix)
    cross_returns = np.log(cross.reindex(d.index)).diff()
    for name in ["hyg", "tlt", "gld", "uso", "uup"]:
        f["x_"+name] = cross_returns[name]
    volume = d.volume.where(np.isfinite(d.volume) & (d.volume > 0))
    log_volume = np.log(volume)
    reference = log_volume.rolling(252, min_periods=126)
    prior_mean = reference.mean().shift(1)
    prior_scale = reference.std(ddof=1).shift(1).replace(0., np.nan)
    pressure = day*(log_volume-prior_mean)/prior_scale
    f["pressure_d"], f["pressure_w"] = pressure, pressure.rolling(5, min_periods=5).mean()
    # An order at this close may use only the preceding completed session.
    f = f.shift(1)
    f["const"] = 1.
    for weekday in range(1, 5):
        f[f"entry_dow_{weekday}"] = (d.index.weekday == weekday).astype(float)
    f = f.loc[:, ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
    dates = pd.Series(d.index, index=d.index)
    f["feature_cutoff_date"] = dates.shift(1)
    factor = d["adj close"]/d.close
    adjustment = np.log(factor.shift(-1)/factor)
    event = pd.Series(pd.NA, index=d.index, dtype="boolean")
    finite_adjustment = np.isfinite(adjustment)
    event.loc[finite_adjustment] = adjustment.loc[finite_adjustment].abs() > EVENT_THRESHOLD
    targets = pd.DataFrame({"y": np.expm1(overnight.shift(-1)),
                            "target_end": dates.shift(-1), "available_date": dates.shift(-1),
                            "adjustment_event": event, "adjustment_log_change": adjustment}, index=d.index)
    targets["y"] = targets.y.where(np.isfinite(targets.y))
    return f, targets


def _alignment(features, target):
    _dates(features.index)
    if not features.index.equals(target.index):
        raise ValueError("Features and labels must have identical entry calendars")
    cutoff = features.feature_cutoff_date
    valid_cutoff = cutoff.notna()
    if (cutoff.loc[valid_cutoff] >= features.index[valid_cutoff]).any():
        raise ValueError("Market feature cutoff must precede entry")
    observed = target.target_end.notna()
    if (not target.available_date.equals(target.target_end.rename("available_date"))
            or (target.loc[observed, "target_end"] <= features.index[observed]).any()):
        raise ValueError("Proxy label is available at the next session close")


def training_mask(features, target, fit_entry, min_train=MIN_TRAIN):
    _alignment(features, target)
    fit_entry = pd.Timestamp(fit_entry)
    if fit_entry not in features.index or min_train < 2:
        raise ValueError("Fit entry and minimum sample are invalid")
    cutoff = features.loc[fit_entry, "feature_cutoff_date"]
    if pd.isna(cutoff):
        raise ValueError("INSUFFICIENT_DATA: no preceding session at fit entry")
    mask = (np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
            & np.isfinite(target.y) & (features.index < fit_entry)
            & (target.available_date <= cutoff))
    if int(mask.sum()) < min_train:
        raise ValueError(f"INSUFFICIENT_DATA: {int(mask.sum())} completed common training rows")
    return mask


def fit_predict(train_features, y, apply_features, alpha=ALPHA):
    """Normalized ridge with unpenalized mean and exact common training rows."""
    if not np.isfinite(alpha) or alpha <= 0 or len(train_features) < 2:
        raise ValueError("Positive normalized ridge penalty and training sample required")
    if isinstance(y, pd.Series) and not y.index.equals(train_features.index):
        raise ValueError("Training target ordering must match design rows")
    target = np.asarray(y, float)
    tr, ap = train_features.loc[:, ALL_FEATURES], apply_features.loc[:, ALL_FEATURES]
    if (target.ndim != 1 or len(target) != len(tr) or not np.isfinite(target).all()
            or not np.isfinite(tr.to_numpy(float)).all() or not np.isfinite(ap.to_numpy(float)).all()
            or not (tr.const == 1).all() or not (ap.const == 1).all()):
        raise ValueError("Finite common designs and aligned targets required")
    if (tr.loc[:, ALL_FEATURES[1:]].std(ddof=0) <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale feature; no fallback")
    target_mean = float(target.mean())
    predictions = {"mean": np.full(len(ap), target_mean)}
    audits = {"mean": {"columns": ["const"], "means": [0.], "scales": [1.],
                        "beta": [target_mean], "alpha": 0., "train_n": len(tr), "gradient_max_abs": 0.}}
    for model, extra in [("baseline", ())] + list(BLOCKS.items()):
        columns = BASE[1:]+extra
        a, b = tr.loc[:, columns].to_numpy(float), ap.loc[:, columns].to_numpy(float)
        means, scales = a.mean(axis=0), a.std(axis=0, ddof=0)
        z, query = (a-means)/scales, (b-means)/scales
        centered = target-target_mean
        beta = np.linalg.solve(z.T@z/len(z) + alpha*np.eye(len(columns)), z.T@centered/len(z))
        prediction = target_mean + query@beta
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite forecast; no clipping fallback")
        predictions[model] = prediction
        audits[model] = {"columns": ["const", *columns], "means": [0., *means.tolist()],
                         "scales": [1., *scales.tolist()], "beta": [target_mean, *beta.tolist()],
                         "alpha": float(alpha), "train_n": len(z),
                         "gradient_max_abs": float(np.max(np.abs(z.T@(z@beta-centered)/len(z)+alpha*beta)))}
    return {"predictions": predictions, "model_audit": audits}


def forecast_panel(features, targets, config):
    _alignment(features, targets)
    if (tuple(config["baseline"]) != BASE or tuple(config["models"]) != MODELS
            or config["horizons"] != [1] or config["ridge_alpha"] != ALPHA):
        raise ValueError("Fixed overnight feature, model or ridge family differs")
    origin_start, origin_end, latest = map(pd.Timestamp, [config["origin_start"], config["origin_end"], config["latest_target"]])
    dev_start, dev_end = map(pd.Timestamp, config["development"])
    eval_start, eval_end = map(pd.Timestamp, config["evaluation"])
    dev_available = pd.Timestamp(config["development_target_available_by"])
    if (not origin_start <= dev_start <= dev_end < eval_start <= eval_end <= origin_end <= latest
            or dev_available != dev_end):
        raise ValueError("Invalid historical development/evaluation fences")
    complete = (np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
                & np.isfinite(targets.y) & np.isfinite(targets.adjustment_log_change)
                & targets.adjustment_event.notna() & (targets.available_date <= latest))
    dates = features.index
    development = ((dates >= dev_start) & (dates <= dev_end)
                   & (targets.available_date <= dev_available).to_numpy())
    evaluation = (dates >= eval_start) & (dates <= eval_end)
    origins = dates[complete & (development | evaluation)
                    & (dates >= origin_start) & (dates <= origin_end)]
    if len(origins) == 0:
        raise ValueError("INSUFFICIENT_DATA: no complete common overnight origins")
    rows, fits = [], []
    for month in origins.to_period("M").unique():
        apply_dates = origins[origins.to_period("M") == month]
        fit_entry = apply_dates[0]
        fit_cutoff = features.loc[fit_entry, "feature_cutoff_date"]
        mask = training_mask(features, targets, fit_entry, config["minimum_train"])
        fit = fit_predict(features.loc[mask], targets.loc[mask, "y"], features.loc[apply_dates], config["ridge_alpha"])
        last_target = targets.loc[mask, "target_end"].max()
        last_available = targets.loc[mask, "available_date"].max()
        fits.append({"fit_origin": str(fit_entry.date()), "fit_cutoff_date": str(fit_cutoff.date()),
                     "train_n": int(mask.sum()), "train_first_origin": str(features.index[mask][0].date()),
                     "train_last_origin": str(features.index[mask][-1].date()),
                     "train_last_target": str(last_target.date()), "train_last_available": str(last_available.date()),
                     "model_audit": fit["model_audit"]})
        for model in MODELS:
            rows.append(pd.DataFrame({
                "origin": apply_dates, "feature_cutoff_date": features.loc[apply_dates, "feature_cutoff_date"].to_numpy(),
                "horizon": 1, "model": model, "target_end": targets.loc[apply_dates, "target_end"].to_numpy(),
                "available_date": targets.loc[apply_dates, "available_date"].to_numpy(),
                "y": targets.loc[apply_dates, "y"].to_numpy(), "prediction": fit["predictions"][model],
                "adjustment_event": targets.loc[apply_dates, "adjustment_event"].to_numpy(dtype=bool),
                "adjustment_log_change": targets.loc[apply_dates, "adjustment_log_change"].to_numpy(),
                "fit_origin": fit_entry, "fit_cutoff_date": fit_cutoff, "train_n": int(mask.sum()),
                "train_last_target": last_target, "train_last_available": last_available,
                "phase": np.where(apply_dates <= dev_end, "development", "evaluation"),
            }))
    return pd.concat(rows, ignore_index=True).sort_values(["origin", "model"]).reset_index(drop=True), fits


def load_inputs(protocol, root=ROOT):
    root = Path(root)
    sources, config = protocol["sources"], protocol["index"]
    cutoff, sealed = pd.Timestamp(config["source_end"]), pd.Timestamp(config["sealed_start"])
    if cutoff >= sealed:
        raise ValueError("Source bound crosses the protected phase")
    daily = pd.read_parquet(root/sources["daily"], filters=[("date", "<=", cutoff)]).loc[:cutoff]
    cross = pd.read_parquet(root/sources["cross"], filters=[("date", "<=", cutoff)]).loc[:cutoff]
    iv = {}
    for name in ["vxn", "vix", "vix9d", "vvix"]:
        raw = pd.read_csv(root/sources[name])
        dates = pd.to_datetime(raw.DATE, format="%m/%d/%Y")
        keep = dates <= cutoff
        column = "VVIX" if name == "vvix" else "CLOSE"
        iv[name] = pd.Series(raw.loc[keep, column].to_numpy(float),
                             index=pd.DatetimeIndex(dates.loc[keep]), name=name)
    panel = pd.concat(iv.values(), axis=1).sort_index()
    for frame in [daily, cross, panel]:
        if frame.empty or frame.index.max() >= sealed:
            raise ValueError("Invalid bounded source data")
    return daily, cross, panel


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_overnight(protocol, root=ROOT):
    """Run only after caller's manifest freeze and synthetic-test gate."""
    root = Path(root)
    daily, cross, iv = load_inputs(protocol, root)
    features, targets = build_features(daily, cross, iv)
    forecasts, fits = forecast_panel(features, targets, protocol["index"])
    output = root/"data/overnight_index"
    output.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output/"features.parquet")
    forecasts.to_parquet(output/"forecasts.parquet", index=False)
    (output/"fits.json").write_text(json.dumps(fits, indent=2, allow_nan=False)+"\n")
    audit = {
        "inputs": {path: _digest(root/path) for path in protocol["sources"].values()},
        "producer_sha256": _digest(__file__), "tests_sha256": _digest(root/"tests/test_overnight_index.py"),
        "source_end": protocol["index"]["source_end"], "sealed_start": protocol["index"]["sealed_start"],
        "feature_columns": list(ALL_FEATURES), "baseline": list(BASE),
        "candidate_blocks": {name: list(block) for name, block in BLOCKS.items()},
        "market_feature_cutoff": "previous actual session close",
        "label_availability": "next actual session close; proxy label uses its completed daily adjustment factor",
        "target_interpretation": "vendor-adjusted overnight proxy, not dividend cash profit",
        "adjustment_event_threshold": EVENT_THRESHOLD,
        "adjustment_flag_use": "retrospective measurement sensitivity only; never an input or training exclusion",
        "source_rows": {"daily": len(daily), "cross": len(cross), "iv": len(iv)},
        "daily_first_date": str(daily.index.min().date()), "daily_last_date": str(daily.index.max().date()),
        "feature_rows": len(features), "forecast_rows": len(forecasts), "fit_count": len(fits),
        "output_hashes": {name: _digest(output/name) for name in ["features.parquet", "forecasts.parquet", "fits.json"]},
    }
    (output/"source_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
    return {"forecasts": str(output/"forecasts.parquet"), "fits": str(output/"fits.json"),
            "features": str(output/"features.parquet"), "source_audit": str(output/"source_audit.json")}
