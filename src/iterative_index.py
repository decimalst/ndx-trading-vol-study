"""Fixed next-session index-return screen; empirical execution is root-gated.

Price information ends at the previous close. Calendar information is finalized
before the actual next opening, including any announced unscheduled closure.
No sealed-window values are admitted by the empirical input loader.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/iterative_signal_search"
ORIGIN_START = pd.Timestamp("2016-01-04")
ORIGIN_END = pd.Timestamp("2025-10-17")
LATEST_TARGET = pd.Timestamp("2025-10-20")
SEALED_START = pd.Timestamp("2025-11-03")
DEVELOPMENT_END = pd.Timestamp("2019-12-31")
EVALUATION_START = pd.Timestamp("2020-01-02")
ALPHA = .01
MIN_TRAIN = 1000
BASE = ("const", "cc_d", "cc_w", "cc_m", "lrv_d", "lrv_w", "lrv_m", "liv", "lvix")
BLOCKS = {
    "session_split": ("split_d", "split_w", "split_m"),
    "cross_signed": ("x_hyg", "x_tlt", "x_gld", "x_uso", "x_uup"),
    "calendar": ("target_dow_1", "target_dow_2", "target_dow_3", "target_dow_4", "target_first3"),
}
ALL_FEATURES = BASE + sum(BLOCKS.values(), ())
MODELS = ("mean", "baseline", *BLOCKS)
SOURCE_PATHS = {
    "daily": "data/raw/daily_ohlc.parquet",
    "cross": "data/raw/cross_asset_daily.parquet",
    "vxn": "data/free_sources/raw/cboe/VXN_History.csv",
    "vix": "data/free_sources/raw/cboe/VIX_History.csv",
}


def _valid_index(index):
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
            or index.has_duplicates or not index.is_monotonic_increasing
            or index.tz is not None or not index.equals(index.normalize())):
        raise ValueError("Unique sorted normalized timezone-naive sessions required")


def _validate_prices(frame, columns):
    values = frame.loc[:, columns].to_numpy(float)
    if np.isinf(values).any() or ((values <= 0) & np.isfinite(values)).any():
        raise ValueError("Prices must be positive where observed, without infinity")


def calendar_features(sessions, origin, target_date):
    """Calendar covariates finalized before target_date's opening auction.

    Future dates in sessions are deliberately ignored: month-session counts use
    only actual sessions at or before the origin, plus the target opening date.
    """
    sessions = pd.DatetimeIndex(sessions)
    _valid_index(sessions)
    origin, target_date = pd.Timestamp(origin), pd.Timestamp(target_date)
    if (pd.isna(origin) or pd.isna(target_date) or target_date <= origin
            or target_date.weekday() > 4):
        raise ValueError("Target weekday must occur after forecast price origin")
    prior = sessions[sessions <= origin]
    count = int(((prior.year == target_date.year) & (prior.month == target_date.month)).sum()) + 1
    result = {f"target_dow_{i}": float(target_date.weekday() == i) for i in range(1, 5)}
    result["target_first3"] = float(count <= 3)
    return result


def build_features(daily, cross, iv):
    """Return all trailing information and next-session daytime targets.

    Missing observations propagate. Splits and dividend adjustments enter the
    signed overnight predictor through the adjusted close-to-close identity;
    the fixed variance control retains the prior studies' raw overnight proxy.
    """
    for frame in (daily, cross, iv):
        _valid_index(frame.index)
    _validate_prices(daily, ["open", "high", "low", "close", "adj close"])
    _validate_prices(cross, ["hyg", "tlt", "gld", "uso", "uup"])
    _validate_prices(iv, ["vxn", "vix"])
    observed = daily[["open", "high", "low", "close"]].notna().all(axis=1)
    d = daily.loc[observed]
    if ((d.high < d[["open", "close", "low"]].max(axis=1)).any()
            or (d.low > d[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLC range")
    d = daily
    cc = np.log(d["adj close"]).diff()
    daytime = np.log(d.close / d.open)
    gk = (.5 * np.log(d.high / d.low)**2
          - (2 * np.log(2) - 1) * daytime**2).clip(lower=1e-10)
    rv = gk + np.log(d.open / d.close.shift(1))**2
    f = pd.DataFrame(index=d.index)
    f["const"] = 1.
    for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
        f[f"cc_{suffix}"] = cc.rolling(width, min_periods=width).mean()
        f[f"lrv_{suffix}"] = np.log(rv.rolling(width, min_periods=width).mean())
        f[f"split_{suffix}"] = (cc - 2*daytime).rolling(width, min_periods=width).mean()
    delayed_iv = iv.reindex(d.index).shift(1)
    f["liv"], f["lvix"] = np.log(delayed_iv.vxn), np.log(delayed_iv.vix)
    asset_returns = np.log(cross.reindex(d.index)).diff()
    for name in ["hyg", "tlt", "gld", "uso", "uup"]:
        f[f"x_{name}"] = asset_returns[name]
    targets = pd.DataFrame({
        "y": (d.close / d.open - 1).shift(-1),
        "target_end": pd.Series(d.index, index=d.index).shift(-1),
    }, index=d.index)
    calendar_rows = [calendar_features(d.index, origin, target)
                     if pd.notna(target) else dict.fromkeys(BLOCKS["calendar"], np.nan)
                     for origin, target in targets.target_end.items()]
    f = f.join(pd.DataFrame(calendar_rows, index=d.index))
    f = f.loc[:, ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
    return f, targets


def _validate_target_alignment(features, targets):
    _valid_index(features.index)
    if not features.index.equals(targets.index):
        raise ValueError("Feature and target indices must match exactly")
    end = targets.target_end
    observed = end.notna()
    if (end.loc[observed] <= features.index[observed]).any():
        raise ValueError("Targets must complete after their historical origins")


def training_mask(features, targets, origin, min_train=MIN_TRAIN):
    _validate_target_alignment(features, targets)
    origin = pd.Timestamp(origin)
    if min_train < 2:
        raise ValueError("At least two training rows required")
    mask = (np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
            & np.isfinite(targets.y) & (features.index < origin)
            & (targets.target_end <= origin))
    count = int(mask.sum())
    if count < min_train:
        raise ValueError(f"INSUFFICIENT_DATA: {count} completed common training rows")
    return mask


def fit_predict(train_features, train_y, apply_features, alpha=ALPHA):
    """Solve mean squared error plus alpha times squared nonintercept weights.

    Each model independently standardizes its included columns using the same
    training rows. The target is centered but is not scaled. Intercept is
    unpenalized; mean-loss normalization makes alpha invariant to row replication.
    """
    if (not np.isfinite(alpha) or alpha <= 0 or len(train_features) < 2
            or len(train_features) != len(train_y)):
        raise ValueError("Valid training sample and positive penalty required")
    if isinstance(train_y, pd.Series) and not train_y.index.equals(train_features.index):
        raise ValueError("Training labels must align with feature rows")
    scalar = isinstance(apply_features, pd.Series)
    if scalar:
        apply_features = apply_features.to_frame().T
    target = np.asarray(train_y, float)
    tr = train_features.loc[:, ALL_FEATURES]
    ap = apply_features.loc[:, ALL_FEATURES]
    if (target.ndim != 1 or not np.isfinite(target).all()
            or not np.isfinite(tr.to_numpy(float)).all()
            or not np.isfinite(ap.to_numpy(float)).all()
            or not (tr.const == 1).all() or not (ap.const == 1).all()):
        raise ValueError("Finite aligned designs with unit intercept required")
    if (tr.loc[:, ALL_FEATURES[1:]].std(ddof=0) <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale candidate or baseline")
    target_mean = float(target.mean())
    predictions = {"mean": np.full(len(ap), target_mean)}
    audit = {"mean": {"mean": target_mean, "train_n": len(tr)}}
    for model, block in [("baseline", ())] + list(BLOCKS.items()):
        columns = BASE[1:] + block
        x, z = tr.loc[:, columns].to_numpy(float), ap.loc[:, columns].to_numpy(float)
        mu, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
        x, z = (x-mu)/sd, (z-mu)/sd
        beta = np.linalg.solve(x.T@x/len(x) + alpha*np.eye(len(columns)),
                               x.T@(target-target_mean)/len(x))
        pred = target_mean + z@beta
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite forecast; no clipping fallback")
        predictions[model] = pred
        audit[model] = {
            "columns": ["const", *columns], "mean": mu.tolist(), "scale": sd.tolist(),
            "beta": [target_mean, *beta.tolist()], "alpha": float(alpha), "train_n": len(x),
            "objective": "mean((y - prediction)^2) + alpha * sum(beta_nonintercept^2)",
            "gradient_max_abs": float(np.max(np.abs(x.T@(x@beta-(target-target_mean))/len(x) + alpha*beta))),
        }
    if scalar:
        predictions = {model: float(values[0]) for model, values in predictions.items()}
    return {"forecasts": predictions, "audit": audit}


def forecast_panel(features, targets, *, origin_start=ORIGIN_START, origin_end=ORIGIN_END,
                   latest_target=LATEST_TARGET, min_train=MIN_TRAIN, alpha=ALPHA):
    """Generate common-origin predictions and monthly-fit audit, without scores."""
    _validate_target_alignment(features, targets)
    origin_start, origin_end, latest_target = map(pd.Timestamp, [origin_start, origin_end, latest_target])
    if origin_start > origin_end or latest_target <= origin_start:
        raise ValueError("Invalid scoring fences")
    eligible = (np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
                & np.isfinite(targets.y) & (targets.target_end <= latest_target)
                & (features.index >= origin_start) & (features.index <= origin_end)
                & ((features.index >= EVALUATION_START) | (targets.target_end <= DEVELOPMENT_END)))
    origins = features.index[eligible]
    if len(origins) == 0:
        raise ValueError("INSUFFICIENT_DATA: no common forecast origins")
    pieces, fits = [], []
    for month in origins.to_period("M").unique():
        apply_origins = origins[origins.to_period("M") == month]
        fit_origin = apply_origins[0]
        train_mask = training_mask(features, targets, fit_origin, min_train=min_train)
        result = fit_predict(features.loc[train_mask], targets.loc[train_mask, "y"],
                             features.loc[apply_origins], alpha=alpha)
        train_last = targets.loc[train_mask, "target_end"].max()
        fits.append({"fit_origin": fit_origin.date().isoformat(),
                     "train_n": int(train_mask.sum()),
                     "train_first_origin": features.index[train_mask][0].date().isoformat(),
                     "train_last_origin": features.index[train_mask][-1].date().isoformat(),
                     "train_last_target": train_last.date().isoformat(),
                     "models": result["audit"]})
        for model in MODELS:
            pieces.append(pd.DataFrame({
                "origin": apply_origins, "target_end": targets.loc[apply_origins, "target_end"].to_numpy(),
                "horizon": 1, "model": model, "y": targets.loc[apply_origins, "y"].to_numpy(),
                "prediction": result["forecasts"][model], "fit_origin": fit_origin,
                "train_n": int(train_mask.sum()), "train_last_target": train_last,
            }))
    frame = pd.concat(pieces, ignore_index=True).sort_values(["origin", "model"]).reset_index(drop=True)
    return frame, fits


def trade_screen(forecasts, costs=(0., .0002, .0005), threshold=.0004):
    """Descriptive flat-overnight long/flat trades, paying both legs each day.

    The decision threshold stays fixed across cost sensitivities. These rows are
    not claims of attainable historical spreads, auction fills, or investment alpha.
    """
    required = ["origin", "target_end", "model", "prediction", "y"]
    frame = forecasts.loc[:, required].copy()
    costs = np.asarray(costs, float)
    if (frame.empty or frame.duplicated(["origin", "model"]).any()
            or costs.ndim != 1 or len(costs) == 0 or len(np.unique(costs)) != len(costs)
            or not np.isfinite(costs).all() or (costs < 0).any()
            or not np.isfinite(threshold)
            or not np.isfinite(frame[["prediction", "y"]].to_numpy(float)).all()
            or (frame.target_end <= frame.origin).any()
            or (frame.groupby("origin").y.nunique() > 1).any()
            or (frame.groupby("origin").target_end.nunique() > 1).any()
            or "always_day_long" in set(frame.model)):
        raise ValueError("Invalid paired forecast panel or cost assumptions")
    frame["position"] = (frame.prediction > threshold).astype(float)
    always = frame.drop_duplicates("origin").copy()
    always["model"], always["position"] = "always_day_long", 1.
    always["prediction"] = np.nan  # This benchmark is a fixed holding rule, not a forecast.
    frame = pd.concat([frame, always], ignore_index=True)
    result = []
    for cost in costs:
        rows = frame.copy()
        rows["cost_per_side"] = float(cost)
        rows["gross_return"] = rows.position*rows.y
        rows["net_return"] = rows.gross_return - 2*cost*rows.position
        result.append(rows)
    return pd.concat(result, ignore_index=True)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs():
    """Mask market dates before any feature computation or observed-value audit."""
    daily = pd.read_parquet(ROOT/SOURCE_PATHS["daily"]).loc[:LATEST_TARGET]
    cross = pd.read_parquet(ROOT/SOURCE_PATHS["cross"]).loc[:LATEST_TARGET]
    series = {}
    for name in ("vxn", "vix"):
        frame = pd.read_csv(ROOT/SOURCE_PATHS[name])
        dates = pd.to_datetime(frame["DATE"], format="%m/%d/%Y")
        frame = frame.loc[dates <= LATEST_TARGET]
        dates = dates.loc[dates <= LATEST_TARGET]
        series[name] = pd.Series(frame["CLOSE"].to_numpy(float), index=pd.DatetimeIndex(dates), name=name)
    iv = pd.concat(series.values(), axis=1).sort_index()
    if any(len(frame) == 0 or frame.index.max() >= SEALED_START for frame in [daily, cross, iv]):
        raise ValueError("Input fence failure")
    return daily, cross, iv


def run(output_dir=OUT):
    """Produce local forecasts/audit only; caller owns pretests and inference gates."""
    daily, cross, iv = load_inputs()
    features, targets = build_features(daily, cross, iv)
    forecasts, fits = forecast_panel(features, targets)
    audit = {
        "evidence_class": "exploratory_reused_history",
        "inputs": {path: _sha256(ROOT/path) for path in SOURCE_PATHS.values()},
        "producer_sha256": _sha256(__file__),
        "tests_sha256": _sha256(ROOT/"tests/test_iterative_index.py"),
        "origin_start": str(ORIGIN_START.date()), "origin_end": str(ORIGIN_END.date()),
        "latest_target": str(LATEST_TARGET.date()), "sealed_start": str(SEALED_START.date()),
        "alpha": ALPHA, "min_train": MIN_TRAIN,
        "target": "next_session_raw_close / next_session_raw_open - 1",
        "calendar_available": "before actual target-session opening; price information through previous close",
        "baseline": list(BASE), "candidate_blocks": {k: list(v) for k, v in BLOCKS.items()},
        "models": list(MODELS), "fit_count": len(fits), "forecast_rows": len(forecasts),
        "fits": fits,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        forecasts.to_parquet(output_dir/"index_forecasts.parquet", index=False)
        (output_dir/"index_fits.json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
    return forecasts, audit
