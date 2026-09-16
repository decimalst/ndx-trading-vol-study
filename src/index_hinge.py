"""Causal archival SPX return inputs and fixed training-centered ridge models.

No empirical runner lives here. The caller owns horizon-specific completed-label
admission, monthly fit origins, evaluation fences, and multiplicity accounting.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ("const", "I", "R", "ret_d", "ret_w", "ret_m", "ret_q", "lr_d", "lr_w",
       "term", "lvvix", "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
BASE = RAW+("I_square", "R_square")
ALL_FEATURES = BASE+("hinge",)
MODELS = ("mean", "baseline", "hinge")
HORIZONS = (21, 63)
ALPHA = .01
SOURCE_END = "2025-10-20"
SEALED_START = "2025-11-03"
OHLC = ("open", "high", "low", "close")
IV_FIELDS = {"vix": "CLOSE", "vix9d": "CLOSE", "vvix": "VVIX"}


def _dates(index):
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates
            or not index.is_monotonic_increasing or index.tz is not None
            or not index.equals(index.normalize())):
        raise ValueError("Unique sorted normalized timezone-naive sessions required")


def _positive_market(frame, columns):
    values = frame.loc[:, columns].to_numpy(float)
    if np.isinf(values).any() or ((values <= 0) & np.isfinite(values)).any():
        raise ValueError("Observed market prices must be positive and finite")


def _daily_contract(daily):
    _dates(daily.index)
    _positive_market(daily, OHLC)
    complete = daily.loc[daily.loc[:, OHLC].notna().all(axis=1)]
    if ((complete.high < complete[["open", "low", "close"]].max(axis=1)).any()
            or (complete.low > complete[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError("Invalid observed OHLC ranges")


def build_features(daily, iv):
    """Keep the full SPX session calendar, lag market inputs, and label 21/63 closes.

    All numerical predictors end at the preceding observed session. Weekday
    indicators use the current entry date only. Missing source observations
    remain inside strict rolling windows; no adjusted-close field is consumed.
    """
    _daily_contract(daily)
    _dates(iv.index)
    _positive_market(iv, tuple(IV_FIELDS))
    aligned = iv.reindex(daily.index)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        returns = np.log(daily.close/daily.close.shift(1))
        day = np.log(daily.close/daily.open)
        gk = (.5*np.log(daily.high/daily.low)**2-(2*np.log(2)-1)*day**2).clip(lower=1e-10)
        variance = gk+np.log(daily.open/daily.close.shift(1))**2
        features = pd.DataFrame(index=daily.index)
        features["I"] = np.log((aligned.vix/100)**2)
        features["R"] = np.log(252*variance.rolling(22, min_periods=22).mean())
        for suffix, width in [("d", 1), ("w", 5), ("m", 22), ("q", 63)]:
            features["ret_"+suffix] = returns.rolling(width, min_periods=width).mean()
        for suffix, width in [("d", 1), ("w", 5)]:
            features["lr_"+suffix] = np.log(252*variance.rolling(width, min_periods=width).mean())
        features["term"] = np.log(aligned.vix9d/aligned.vix)
        features["lvvix"] = np.log(aligned.vvix)
    features = features.shift(1)
    features["const"] = 1.
    for weekday in range(1, 5):
        features[f"entry_dow_{weekday}"] = (daily.index.weekday == weekday).astype(float)
    features = features.loc[:, RAW].replace([np.inf, -np.inf], np.nan)
    dates = pd.Series(daily.index, index=daily.index)
    features["feature_cutoff_date"] = dates.shift(1)
    targets = {}
    for horizon in HORIZONS:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            y = np.log(daily.close.shift(-horizon)/daily.close)
        targets[horizon] = pd.DataFrame({
            "y": y.where(np.isfinite(y)), "target_end": dates.shift(-horizon),
            "available_date": dates.shift(-horizon),
        }, index=daily.index)
    return features, targets


def transform(train, apply):
    """Fit the two centers and gap center on precisely the supplied training rows."""
    tr, ap = train.loc[:, RAW].astype(float).copy(), apply.loc[:, RAW].astype(float).copy()
    if (len(tr) < 2 or not np.isfinite(tr.to_numpy()).all() or not np.isfinite(ap.to_numpy()).all()
            or not tr.const.eq(1).all() or not ap.const.eq(1).all()):
        raise ValueError("Finite common designs with unit intercept and at least two training rows required")
    with np.errstate(over="ignore", invalid="ignore"):
        i_mean, r_mean = float(tr.I.mean()), float(tr.R.mean())
        gap_mean = float((tr.I-tr.R).mean())
        for frame in (tr, ap):
            frame["I_square"] = (frame.I-i_mean)**2
            frame["R_square"] = (frame.R-r_mean)**2
            frame["hinge"] = (frame.I-frame.R-gap_mean).clip(lower=0)
    if (not np.isfinite([i_mean, r_mean, gap_mean]).all()
            or not np.isfinite(tr.to_numpy()).all() or not np.isfinite(ap.to_numpy()).all()):
        raise ValueError("Nonfinite training-centered transform; no clipping fallback")
    audit = {"train_n": len(tr), "I_mean": i_mean, "R_mean": r_mean, "gap_mean": gap_mean,
             "definition": "I_square=(I-I_mean)^2; R_square=(R-R_mean)^2; hinge=max(I-R-gap_mean,0); all means training-only"}
    return tr.loc[:, ALL_FEATURES], ap.loc[:, ALL_FEATURES], audit


def fit_predict(train_features, y, apply_features):
    """Fit fixed normalized ridge; every model shares the supplied complete rows.

    All 17 non-intercept columns, including the hinge, must have population scale
    above 1e-12. The caller must establish label maturity before invoking this
    function. The intercept is the training target mean and is never penalized.
    """
    if isinstance(y, pd.Series) and not y.index.equals(train_features.index):
        raise ValueError("Training target ordering must match design rows")
    target = np.asarray(y, float)
    if target.ndim != 1 or len(target) != len(train_features) or not np.isfinite(target).all():
        raise ValueError("Finite aligned one-dimensional training target required")
    tr, ap, transform_audit = transform(train_features, apply_features)
    all_scales = tr.loc[:, ALL_FEATURES[1:]].std(ddof=0).to_numpy(float)
    if not np.isfinite(all_scales).all() or (all_scales <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero or nonfinite common feature scale; no fallback")
    target_mean = float(target.mean())
    if not np.isfinite(target_mean):
        raise ValueError("Nonfinite training target mean")
    predictions = {"mean": np.full(len(ap), target_mean)}
    audits = {"mean": {"columns": ["const"], "means": [0.], "scales": [1.],
                        "beta": [target_mean], "alpha": 0., "train_n": len(tr), "gradient_max_abs": 0.}}
    for name, columns in [("baseline", BASE[1:]), ("hinge", ALL_FEATURES[1:])]:
        a, b = tr.loc[:, columns].to_numpy(float), ap.loc[:, columns].to_numpy(float)
        means, scales = a.mean(axis=0), a.std(axis=0, ddof=0)
        z, query = (a-means)/scales, (b-means)/scales
        centered = target-target_mean
        beta = np.linalg.solve(z.T@z/len(z)+ALPHA*np.eye(len(columns)), z.T@centered/len(z))
        prediction = target_mean+query@beta
        gradient = z.T@(z@beta-centered)/len(z)+ALPHA*beta
        if not np.isfinite(prediction).all() or not np.isfinite(beta).all() or not np.isfinite(gradient).all():
            raise ValueError("Nonfinite ridge result; no clipping fallback")
        predictions[name] = prediction
        audits[name] = {"columns": ["const", *columns], "means": [0., *means.tolist()],
                        "scales": [1., *scales.tolist()], "beta": [target_mean, *beta.tolist()],
                        "alpha": ALPHA, "train_n": len(z),
                        "gradient_max_abs": float(np.max(np.abs(gradient)))}
    return {"predictions": predictions, "model_audit": audits, "transform_audit": transform_audit}


def _source_audit(path, frame, reference):
    return {
        "source_path": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bounded_rows": len(frame), "first_date": str(frame.index.min().date()),
        "last_date": str(frame.index.max().date()),
        "missing_values": {name: int(frame[name].isna().sum()) for name in frame},
        "missing_reference_dates": [str(date.date()) for date in reference.difference(frame.index)],
        "outside_reference_dates": [str(date.date()) for date in frame.index.difference(reference)],
    }


def load_sources(protocol, root=ROOT):
    """Read four declared archival sources, bounding dates before numeric parsing."""
    root = Path(root)
    end = pd.Timestamp(protocol["index"]["source_end"])
    sealed = pd.Timestamp(protocol["index"]["sealed_start"])
    if (pd.isna(end) or end.tz is not None or end != end.normalize()
            or end > pd.Timestamp(SOURCE_END) or sealed != pd.Timestamp(SEALED_START)
            or end >= sealed):
        raise ValueError("Invalid source cutoff or protected-period fence")
    paths = {name: root/protocol["sources"][name] for name in ["daily", *IV_FIELDS]}
    daily = pd.read_parquet(paths["daily"], columns=list(OHLC), filters=[("date", "<=", end)]).loc[:end]
    _daily_contract(daily)
    if daily.empty:
        raise ValueError("INSUFFICIENT_DATA: no bounded SPX sessions")
    sources = {"daily": _source_audit(paths["daily"], daily, daily.index)}
    series = []
    for name, column in IV_FIELDS.items():
        raw = pd.read_csv(paths[name], usecols=["DATE", column], dtype=str)
        dates = pd.to_datetime(raw.DATE, format="%m/%d/%Y", errors="raise")
        keep = dates <= end
        values = pd.to_numeric(raw.loc[keep, column], errors="raise").to_numpy(float)
        one = pd.Series(values, index=pd.DatetimeIndex(dates.loc[keep], name="date"), name=name)
        _dates(one.index)
        if one.empty:
            raise ValueError(f"INSUFFICIENT_DATA: no bounded {name} observations")
        frame = one.to_frame()
        _positive_market(frame, [name])
        sources[name] = {**_source_audit(paths[name], frame, daily.index),
                         "provider_date_field": "DATE", "provider_value_field": column}
        series.append(one)
    iv = pd.concat(series, axis=1).sort_index()
    audit = {
        "sources": sources, "source_end": str(end.date()), "sealed_start": str(sealed.date()),
        "reference_calendar": {"definition": "bounded observed SPX raw-OHLC sessions",
                               "sessions": len(daily), "first_date": str(daily.index.min().date()),
                               "last_date": str(daily.index.max().date())},
        "raw_columns": list(RAW), "numeric_post_cutoff_values_parsed": False,
        "historical_vintage_certified": False,
        "timing_assumption": "all market predictors through prior observed SPX session close; current entry weekday only",
        "vix9d_caveat": "prelaunch January2011-October2013 values are back-calculated archival training inputs",
        "target_interpretation": "SPX cumulative raw-close log price return; excludes reinvested dividends and risk-free subtraction",
        "gap_interpretation": "log implied-versus-trailing-OHLC-proxy gap; not a measured variance risk premium",
    }
    return daily, iv, audit
