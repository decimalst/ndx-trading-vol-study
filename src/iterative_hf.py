"""Fixed archival SPX feature experiment; empirical fits require root protocol.

Oxford rows are source estimates, available one complete SPX session later.
The implementation does not import any earlier study's producer or estimator.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import zipfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ("const", "gk_d", "gk_w", "gk_m", "lev_d", "lev_w", "lev_m",
        "liv", "hf_d", "hf_w", "hf_m")
BLOCKS = {"semivariance": ("share_d", "share_w", "share_m"),
          "kernel": ("kernel_d", "kernel_w", "kernel_m")}
ALL_FEATURES = BASE + BLOCKS["semivariance"] + BLOCKS["kernel"]
ARCHIVE_SHA256 = "e0dd80edc0c2cedac5ed3f72250ee4460e963b4efd458d68525a61bcc5c27ea2"
SOURCE_END = "2018-01-03"


def _dates(index):
    dates = pd.DatetimeIndex(index)
    if (dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or dates.tz is not None):
        raise ValueError("Unique sorted timezone-naive trading dates required")
    return dates


def parse_oxford_archive(content, expected_sha256=None, end=None):
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Oxford archive source hash differs")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError("Exactly one Oxford source CSV required")
        frame = pd.read_csv(archive.open(csv_names[0]))
    required = ["rv5", "rsv", "rk_parzen"]
    if not {"Symbol", *required}.issubset(frame.columns):
        raise ValueError("Oxford source schema differs")
    frame = frame.loc[frame.Symbol == ".SPX"].copy()
    if frame.empty:
        raise ValueError("Registered .SPX asset is absent")
    # The text's calendar day is the session label. UTC conversion moves BST
    # midnight into the preceding day and is deliberately not performed.
    dates = pd.to_datetime(frame.iloc[:, 0].astype(str).str[:10],
                           format="%Y-%m-%d", errors="raise")
    frame.index = pd.DatetimeIndex(dates)
    frame.index.name = "date"
    frame = frame.sort_index()
    _dates(frame.index)
    if end is not None:
        frame = frame.loc[:pd.Timestamp(end)]
    result = frame.loc[:, required].astype(float)
    _validate_hf(result)
    return result


def _validate_hf(hf):
    _dates(hf.index)
    values = hf.loc[:, ["rv5", "rsv", "rk_parzen"]].to_numpy(float)
    if (not np.isfinite(values).all() or (hf[["rv5", "rk_parzen"]] <= 0).any().any()
            or (hf.rsv < 0).any()
            or (hf.rsv > hf.rv5 * (1 + 1e-12)).any()):
        raise ValueError("Invalid Oxford measures or semivariance identity")


def _validate_daily(daily):
    _dates(daily.index)
    prices = daily.loc[:, ["open", "high", "low", "close"]]
    if (not np.isfinite(prices).all().all() or (prices <= 0).any().any()
            or (daily.high < daily[["open", "close", "low"]].max(axis=1)).any()
            or (daily.low > daily[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid daily OHLC prices")


def merge_daily(historical, current, cutoff="2009-01-01", min_overlap=250,
                rtol=1e-6, atol=1e-4):
    """Check overlap before appending historical rows preceding the fixed cutoff."""
    _validate_daily(historical)
    _validate_daily(current)
    cutoff = pd.Timestamp(cutoff)
    required = ["open", "high", "low", "close"]
    overlap = historical.index.intersection(current.index)
    if len(overlap) < min_overlap:
        raise ValueError("Insufficient historical price overlap")
    lower = max(historical.index.min(), current.index.min())
    upper = min(historical.index.max(), current.index.max())
    if not historical.loc[lower:upper].index.equals(current.loc[lower:upper].index):
        raise ValueError("Historical and current overlap calendars differ")
    a, b = historical.loc[overlap, required], current.loc[overlap, required]
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise ValueError("Historical OHLC overlap differs beyond fixed tolerance")
    columns = list(current.columns)
    if not set(columns).issubset(historical.columns):
        raise ValueError("Historical daily schema missing current columns")
    merged = pd.concat([historical.loc[historical.index < cutoff, columns], current])
    merged.index.name = current.index.name
    _validate_daily(merged)
    audit = {"overlap_rows": len(overlap), "overlap_start": str(lower.date()),
             "overlap_end": str(upper.date()), "rtol": rtol, "atol": atol,
             "maximum_price_absolute_difference": float(np.abs(a - b).to_numpy().max()),
             "maximum_price_relative_difference": float(np.abs((a - b) / b).to_numpy().max()),
             "historical_appended_rows": int((historical.index < cutoff).sum()),
             "cutoff": str(cutoff.date()), "existing_rows_unchanged": True}
    return merged, audit


def build_features(daily, hf, vix):
    _validate_daily(daily)
    _validate_hf(hf)
    _dates(vix.index)
    sessions = daily.index
    # Reindex before any shift or rolling aggregation. Missing trading sessions
    # remain missing, rather than turning the horizon into available-source rows.
    h = hf.reindex(sessions)
    f = pd.DataFrame(index=sessions)
    gk = (.5 * np.log(daily.high / daily.low) ** 2
          - (2 * np.log(2) - 1) * np.log(daily.close / daily.open) ** 2).clip(lower=1e-10)
    total = gk + np.log(daily.open / daily.close.shift(1)) ** 2
    returns = np.log(daily.close).diff()
    f["const"] = 1.0
    for suffix, window in [("d", 1), ("w", 5), ("m", 22)]:
        f["gk_" + suffix] = np.log(total.rolling(window).mean())
        f["lev_" + suffix] = returns.rolling(window).mean().clip(upper=0)
        f["hf_" + suffix] = np.log(h.rv5.rolling(window).mean()).shift(1)
        denominator = h.rv5.rolling(window).sum()
        f["share_" + suffix] = (h.rsv.rolling(window).sum() / denominator).shift(1)
        f["kernel_" + suffix] = np.log(h.rk_parzen.rolling(window).sum() / denominator).shift(1)
    f["liv"] = np.log(vix.reindex(sessions).where(lambda x: x > 0)).shift(1)
    f = f.loc[:, ALL_FEATURES]
    return f.replace([np.inf, -np.inf], np.nan)


def make_targets(rv5, sessions, horizon):
    sessions = _dates(sessions)
    _dates(rv5.index)
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("Positive integer horizon required")
    raw = rv5.reindex(sessions)
    future = pd.concat([raw.shift(-i) for i in range(1, horizon + 1)], axis=1)
    valid = np.isfinite(future).all(axis=1) & (future > 0).all(axis=1)
    y = future.mean(axis=1, skipna=False).where(valid)
    dates = pd.Series(sessions, index=sessions)
    return pd.DataFrame({"y": y, "target_end": dates.shift(-horizon),
                         "available_date": dates.shift(-horizon - 1)}, index=sessions)


def training_mask(features, targets, origin, min_train=750):
    _dates(features.index)
    if not features.index.equals(targets.index):
        raise ValueError("Feature and target calendar mismatch")
    origin = pd.Timestamp(origin)
    mask = (np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
            & np.isfinite(targets.y) & (targets.y > 0)
            & (targets.available_date <= origin) & (targets.index < origin))
    if int(mask.sum()) < min_train:
        raise ValueError(f"INSUFFICIENT_DATA: {int(mask.sum())} completed common training rows")
    if (targets.loc[mask, "target_end"] >= targets.loc[mask, "available_date"]).any():
        raise ValueError("Target availability must follow completion")
    return mask


def fit_predict(train_features, train_y, apply_features):
    y = np.asarray(train_y, dtype=float)
    if (y.ndim != 1 or len(y) != len(train_features) or not np.isfinite(y).all()
            or (y <= 0).any()):
        raise ValueError("Positive finite matching training target required")
    if (not np.isfinite(train_features.loc[:, ALL_FEATURES]).all().all()
            or not np.isfinite(apply_features.loc[:, ALL_FEATURES]).all().all()):
        raise ValueError("Every arm requires common complete input rows")
    if not (train_features["const"].eq(1).all() and apply_features["const"].eq(1).all()):
        raise ValueError("Constant column must be exactly one")
    means = train_features.loc[:, ALL_FEATURES].mean()
    scales = train_features.loc[:, ALL_FEATURES].std(ddof=0)
    means.loc["const"], scales.loc["const"] = 0.0, 1.0
    if (scales <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale feature")
    log_y = np.log(y)
    result = {"predictions": {}, "audit": {}}
    for name, extra in [("baseline", ()), *BLOCKS.items()]:
        columns = list(BASE + extra)
        a = ((train_features[columns] - means[columns]) / scales[columns]).to_numpy(float)
        b = ((apply_features[columns] - means[columns]) / scales[columns]).to_numpy(float)
        beta, _, rank, singular = np.linalg.lstsq(a, log_y, rcond=None)
        if rank != len(columns):
            raise ValueError("INSUFFICIENT_DATA: rank-deficient feature design")
        smear = float(np.exp(log_y - a @ beta).mean())
        prediction = np.exp(b @ beta) * smear
        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise ValueError("Nonpositive or nonfinite prediction; no fallback")
        result["predictions"][name] = prediction
        result["audit"][name] = {"columns": columns, "means": means[columns].tolist(),
                                  "scales": scales[columns].tolist(), "beta": beta.tolist(),
                                  "smear": smear, "rank": int(rank), "train_n": len(y),
                                  "singular_values": singular.tolist()}
    return result


def forecast_panel(features, target_frames, config):
    if "development" in config:
        config = {**config, "development_start": config["development"][0],
                  "development_end": config["development"][1],
                  "evaluation_start": config["evaluation"][0],
                  "evaluation_end": config["evaluation"][1],
                  "min_train": config["minimum_train"]}
    if "baseline" in config and tuple(config["baseline"]) != BASE:
        raise ValueError("Protocol baseline differs from implemented feature names")
    horizons = config["horizons"]
    if horizons != [1, 5, 21] or set(target_frames) != set(horizons):
        raise ValueError("The fixed three HF horizons are required")
    start, dev_end, eval_start, end, latest = [pd.Timestamp(config[k]) for k in
        ["development_start", "development_end", "evaluation_start", "evaluation_end", "latest_target"]]
    if not start <= dev_end < eval_start <= end <= latest:
        raise ValueError("Invalid chronological study boundaries")
    eligible = pd.Series(np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1), index=features.index)
    for h in horizons:
        t = target_frames[h]
        if not features.index.equals(t.index):
            raise ValueError("Feature and target calendar mismatch")
        eligible &= np.isfinite(t.y) & (t.y > 0) & (t.target_end <= latest)
    dates = features.index
    dev = (dates >= start) & (dates <= dev_end)
    for h in horizons:
        dev &= (target_frames[h].available_date <= dev_end).to_numpy()
    evaluation = (dates >= eval_start) & (dates <= end)
    origins = dates[eligible & (dev | evaluation)]
    if len(origins) == 0:
        raise ValueError("INSUFFICIENT_DATA: no common forecast origins")
    rows, fits = [], []
    for horizon in horizons:
        targets = target_frames[horizon]
        for _, month in pd.Series(origins, index=origins).groupby(origins.to_period("M")):
            apply_dates = pd.DatetimeIndex(month.to_numpy())
            origin = apply_dates[0]
            mask = training_mask(features, targets, origin, config["min_train"])
            train_last_target = targets.loc[mask, "target_end"].max()
            train_last_available = targets.loc[mask, "available_date"].max()
            fit = fit_predict(features.loc[mask], targets.loc[mask, "y"].to_numpy(), features.loc[apply_dates])
            fits.append({"horizon": horizon, "fit_origin": str(origin.date()),
                         "train_n": int(mask.sum()), "train_last_target": str(train_last_target.date()),
                         "train_last_available": str(train_last_available.date()),
                         "model_audit": fit["audit"]})
            for name, prediction in fit["predictions"].items():
                for i, t in enumerate(apply_dates):
                    rows.append({"origin": t, "horizon": horizon, "model": name,
                                 "target_end": targets.loc[t, "target_end"],
                                 "available_date": targets.loc[t, "available_date"],
                                 "y": targets.loc[t, "y"], "prediction": float(prediction[i]),
                                 "fit_origin": origin, "train_n": int(mask.sum()),
                                 "train_last_target": train_last_target,
                                 "train_last_available": train_last_available,
                                 "phase": "development" if t <= dev_end else "evaluation"})
    return pd.DataFrame(rows).sort_values(["origin", "horizon", "model"]).reset_index(drop=True), fits


def _dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def prepare_inputs(root=ROOT):
    root = pathlib.Path(root)
    output = root / "data/iterative_signal_search"
    historical = pd.read_parquet(output / "spx_daily_pre2009_download.parquet")
    # The filtered parquet read does not load later protected observations.
    current = pd.read_parquet(root / "data/research_paths/spx_daily.parquet",
                              filters=[("date", "<=", pd.Timestamp(SOURCE_END))])
    daily, overlap = merge_daily(historical, current)
    daily = daily.loc[:SOURCE_END]
    hf = parse_oxford_archive((root / "data/raw/oxford_man_realized.zip").read_bytes(),
                              ARCHIVE_SHA256, SOURCE_END)
    vix = pd.read_parquet(root / "data/research_paths/cboe_indices.parquet",
                          filters=[("date", "<=", pd.Timestamp(SOURCE_END))])["vix"]
    features = build_features(daily, hf, vix)
    targets = {h: make_targets(hf.rv5, daily.index, h) for h in [1, 5, 21]}
    daily.to_parquet(output / "spx_daily_merged.parquet")
    hf.to_parquet(output / "hf_source.parquet")
    features.to_parquet(output / "hf_features.parquet")
    _dump(output / "hf_source_audit.json", {"archive_sha256": ARCHIVE_SHA256,
        "source_end": SOURCE_END, "hf_delay_sessions": 1, "target_availability_delay_sessions": 1,
        "hf_rows": len(hf), "daily_rows": len(daily), "overlap": overlap})
    return features, targets


def run_hf(config, root=ROOT):
    """Fit the already frozen family. Caller must run synthetic tests first."""
    features, targets = prepare_inputs(root)
    forecasts, fits = forecast_panel(features, targets, config)
    output = pathlib.Path(root) / "data/iterative_signal_search"
    forecast_path, fit_path = output / "hf_forecasts.parquet", output / "hf_fits.json"
    forecasts.to_parquet(forecast_path, index=False)
    _dump(fit_path, fits)
    return {"forecasts": str(forecast_path), "fits": str(fit_path)}
