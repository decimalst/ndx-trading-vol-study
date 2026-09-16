"""Fixed international-input study with delayed, auditable foreign observations."""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import zipfile

import numpy as np
import pandas as pd

from . import iterative_hf

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYMBOLS = (".N225", ".HSI", ".KS11", ".FTSE", ".GDAXI", ".FCHI")
SUFFIXES = ("d", "w", "m")
WINDOWS = (1, 5, 22)
BASE = iterative_hf.BASE
FOREIGN_COLUMNS = tuple(f"{s[1:].lower()}_{k}" for s in SYMBOLS for k in SUFFIXES)
REGIONAL_COLUMNS = tuple(f"{region}_{k}" for region in ("asia", "europe") for k in SUFFIXES)
ALL_FEATURES = BASE + FOREIGN_COLUMNS
PCA_COLUMNS = ("pc1", "pc2", "pc3")
EIGENGAP_RELATIVE = 1e-8
SOURCE_END = "2018-01-03"


def _dates(index):
    dates = pd.DatetimeIndex(index)
    if (dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or dates.tz is not None):
        raise ValueError("Unique sorted timezone-naive date labels required")
    return dates


def parse_international_archive(content, expected_sha256=None, end=None):
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("International archive source hash differs")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Exactly one source CSV required")
        frame = pd.read_csv(archive.open(names[0]))
    if not {"Symbol", "rv5"}.issubset(frame.columns):
        raise ValueError("International source schema differs")
    frame = frame.loc[frame.Symbol.isin(SYMBOLS)].copy()
    frame.index = pd.to_datetime(frame.iloc[:, 0].astype(str).str[:10],
                                  format="%Y-%m-%d", errors="raise")
    frame.index.name = "date"
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
    result = {}
    for symbol in SYMBOLS:
        rows = frame.loc[frame.Symbol == symbol].sort_index()
        if rows.empty:
            raise ValueError(f"Registered market is missing: {symbol}")
        _dates(rows.index)
        value = pd.to_numeric(rows.rv5, errors="coerce").astype(float)
        # Retain bad-value rows as missing observations. Removing them would
        # turn a 22-observation rolling window into 22 selectively valid values.
        result[symbol] = value.where(np.isfinite(value) & (value > 0))
    return result


def build_foreign_features(foreign, sessions, max_age=3):
    sessions = _dates(sessions)
    if set(foreign) != set(SYMBOLS) or max_age != 3:
        raise ValueError("The fixed six markets and three-session age limit are required")
    cutoff = pd.Series(sessions, index=sessions).shift(1)
    cutoff_values = cutoff.to_numpy(dtype="datetime64[ns]")
    us_values = sessions.to_numpy(dtype="datetime64[ns]")
    features = pd.DataFrame(index=sessions)
    audit = []
    for symbol in SYMBOLS:
        raw = foreign[symbol].astype(float)
        source_dates = _dates(raw.index)
        value = raw.where(np.isfinite(raw) & (raw > 0))
        local = pd.DataFrame({suffix: np.log(value.rolling(window, min_periods=window).mean())
                              for suffix, window in zip(SUFFIXES, WINDOWS, strict=True)})
        locations = source_dates.searchsorted(cutoff_values, side="right") - 1
        found = cutoff.notna().to_numpy() & (locations >= 0)
        source = np.full(len(sessions), np.datetime64("NaT"), dtype="datetime64[ns]")
        source[found] = source_dates.to_numpy(dtype="datetime64[ns]")[locations[found]]
        age = np.full(len(sessions), np.nan)
        age[found] = (np.searchsorted(us_values, cutoff_values[found], side="right")
                      - np.searchsorted(us_values, source[found], side="right"))
        fresh = found & (age <= max_age)
        selected = np.full((len(sessions), len(SUFFIXES)), np.nan)
        selected[found] = local.to_numpy()[locations[found]]
        complete = np.isfinite(selected).all(axis=1)
        admitted = selected.copy()
        admitted[~fresh] = np.nan
        for i, suffix in enumerate(SUFFIXES):
            features[f"{symbol[1:].lower()}_{suffix}"] = admitted[:, i]
        audit.append(pd.DataFrame({"origin": sessions, "symbol": symbol,
                                   "cutoff_date": cutoff_values, "source_date": source,
                                   "extra_us_sessions": age, "fresh": fresh,
                                   "complete_window": complete}))
    features = features.loc[:, FOREIGN_COLUMNS]
    features.index.name = sessions.name
    return features, pd.concat(audit, ignore_index=True)


def regional_features(features):
    regional = pd.DataFrame(index=features.index)
    for region, symbols in [("asia", SYMBOLS[:3]), ("europe", SYMBOLS[3:])]:
        for suffix in SUFFIXES:
            columns = [f"{s[1:].lower()}_{suffix}" for s in symbols]
            regional[f"{region}_{suffix}"] = features.loc[:, columns].mean(axis=1, skipna=False)
    return regional


def fit_pca(train, apply):
    if tuple(train.columns) != FOREIGN_COLUMNS or tuple(apply.columns) != FOREIGN_COLUMNS:
        raise ValueError("PCA requires the fixed eighteen-column order")
    a, b = train.to_numpy(float), apply.to_numpy(float)
    if len(a) < len(FOREIGN_COLUMNS) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Complete finite training and application matrices required")
    means, scales = a.mean(axis=0), a.std(axis=0, ddof=0)
    if (scales <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale international feature")
    z = (a - means) / scales
    _, singular, right = np.linalg.svd(z, full_matrices=False)
    gap = float(singular[2] ** 2 - singular[3] ** 2)
    threshold = float(EIGENGAP_RELATIVE * singular[0] ** 2)
    if not gap > threshold:
        raise ValueError("INSUFFICIENT_DATA: unstable retained PCA boundary")
    components = right[:3].copy()
    for component in components:
        anchor = int(np.argmax(np.abs(component)))
        if component[anchor] < 0:
            component *= -1
    train_scores = z @ components.T
    apply_scores = ((b - means) / scales) @ components.T
    if np.linalg.matrix_rank(train_scores) != 3:
        raise ValueError("INSUFFICIENT_DATA: deficient PCA score rank")
    audit = {"columns": list(FOREIGN_COLUMNS), "means": means.tolist(), "scales": scales.tolist(),
             "components": components.tolist(), "singular_values": singular.tolist(),
             "eigen_gap": gap, "gap_threshold": threshold}
    return {"train_scores": train_scores, "apply_scores": apply_scores, "audit": audit}


def fit_log_ols(train, y, apply):
    columns = list(train.columns)
    if list(apply.columns) != columns or not columns or columns[0] != "const":
        raise ValueError("Matching OLS column order with leading constant required")
    a, b, y = train.to_numpy(float), apply.to_numpy(float), np.asarray(y, float)
    if (y.ndim != 1 or len(y) != len(a) or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(y).all() or (y <= 0).any()
            or not (a[:, 0] == 1).all() or not (b[:, 0] == 1).all()):
        raise ValueError("Finite matched designs and positive targets required")
    means, scales = a.mean(axis=0), a.std(axis=0, ddof=0)
    means[0], scales[0] = 0.0, 1.0
    if (scales <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale OLS feature")
    x, query = (a - means) / scales, (b - means) / scales
    log_y = np.log(y)
    beta, _, rank, singular = np.linalg.lstsq(x, log_y, rcond=None)
    if rank != len(columns):
        raise ValueError("INSUFFICIENT_DATA: deficient augmented OLS design")
    smear = float(np.exp(log_y - x @ beta).mean())
    prediction = np.exp(query @ beta) * smear
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Invalid OLS forecast; no clipping fallback")
    audit = {"columns": columns, "means": means.tolist(), "scales": scales.tolist(),
             "beta": beta.tolist(), "smear": smear, "rank": int(rank),
             "train_n": len(y), "singular_values": singular.tolist()}
    return prediction, audit


def fit_predict(train_features, y, apply_features):
    if (not np.isfinite(train_features.loc[:, ALL_FEATURES]).all().all()
            or not np.isfinite(apply_features.loc[:, ALL_FEATURES]).all().all()):
        raise ValueError("All three arms require the complete common information set")
    pca = fit_pca(train_features.loc[:, FOREIGN_COLUMNS], apply_features.loc[:, FOREIGN_COLUMNS])
    base_train, base_apply = train_features.loc[:, BASE], apply_features.loc[:, BASE]
    pc_train = pd.DataFrame(pca["train_scores"], index=train_features.index, columns=PCA_COLUMNS)
    pc_apply = pd.DataFrame(pca["apply_scores"], index=apply_features.index, columns=PCA_COLUMNS)
    models = {
        "baseline": (base_train, base_apply),
        "regional": (pd.concat([base_train, regional_features(train_features)], axis=1),
                     pd.concat([base_apply, regional_features(apply_features)], axis=1)),
        "latent": (pd.concat([base_train, pc_train], axis=1), pd.concat([base_apply, pc_apply], axis=1)),
    }
    predictions, model_audit = {}, {}
    for name, (a, b) in models.items():
        predictions[name], model_audit[name] = fit_log_ols(a, y, b)
    return {"predictions": predictions, "model_audit": model_audit, "pca_audit": pca["audit"]}


def training_mask(features, target, origin, min_train=750):
    _dates(features.index)
    if not features.index.equals(target.index):
        raise ValueError("Target and feature calendars differ")
    origin = pd.Timestamp(origin)
    mask = (np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
            & np.isfinite(target.y) & (target.y > 0)
            & (target.available_date <= origin) & (target.index < origin))
    if int(mask.sum()) < min_train:
        raise ValueError(f"INSUFFICIENT_DATA: {int(mask.sum())} completed common training rows")
    if (target.loc[mask, "target_end"] >= target.loc[mask, "available_date"]).any():
        raise ValueError("Target admitted before source publication")
    return mask


def forecast_panel(features, target_frames, config):
    if config["horizons"] != [1, 5, 21] or set(target_frames) != {1, 5, 21}:
        raise ValueError("Fixed horizon family required")
    if tuple(config["baseline"]) != BASE:
        raise ValueError("Strong SPX baseline changed")
    start, dev_end = map(pd.Timestamp, config["development"])
    eval_start, end = map(pd.Timestamp, config["evaluation"])
    latest = pd.Timestamp(config["latest_target"])
    dev_available = pd.Timestamp(config["development_target_available_by"])
    if not start <= dev_end < eval_start <= end <= latest or dev_available != dev_end:
        raise ValueError("Invalid chronological fences")
    complete = np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
    dates = features.index
    development = (dates >= start) & (dates <= dev_end)
    for target in target_frames.values():
        if not dates.equals(target.index):
            raise ValueError("Target and feature calendars differ")
        complete &= np.isfinite(target.y) & (target.y > 0) & (target.target_end <= latest)
        development &= (target.available_date <= dev_available).to_numpy()
    evaluation = (dates >= eval_start) & (dates <= end)
    origins = dates[complete & (development | evaluation)]
    if len(origins) == 0:
        raise ValueError("INSUFFICIENT_DATA: no common origins")
    rows, fits = [], []
    for horizon in config["horizons"]:
        target = target_frames[horizon]
        for _, month in pd.Series(origins, index=origins).groupby(origins.to_period("M")):
            apply_dates = pd.DatetimeIndex(month.to_numpy())
            origin = apply_dates[0]
            mask = training_mask(features, target, origin, config["minimum_train"])
            last_target = target.loc[mask, "target_end"].max()
            last_available = target.loc[mask, "available_date"].max()
            fit = fit_predict(features.loc[mask], target.loc[mask, "y"], features.loc[apply_dates])
            fits.append({"horizon": horizon, "fit_origin": str(origin.date()), "train_n": int(mask.sum()),
                         "train_last_target": str(last_target.date()),
                         "train_last_available": str(last_available.date()),
                         "pca_audit": fit["pca_audit"], "model_audit": fit["model_audit"]})
            for model, prediction in fit["predictions"].items():
                for i, t in enumerate(apply_dates):
                    rows.append({"origin": t, "horizon": horizon, "model": model,
                                 "target_end": target.loc[t, "target_end"],
                                 "available_date": target.loc[t, "available_date"], "y": target.loc[t, "y"],
                                 "prediction": float(prediction[i]), "fit_origin": origin,
                                 "train_n": int(mask.sum()), "train_last_target": last_target,
                                 "train_last_available": last_available,
                                 "phase": "development" if t <= dev_end else "evaluation"})
    frame = pd.DataFrame(rows).sort_values(["origin", "horizon", "model"]).reset_index(drop=True)
    return frame, fits


def _dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def prepare_inputs(protocol, root=ROOT):
    """Read raw inputs and write only this wave's new data directory; no fits."""
    root = pathlib.Path(root)
    source = protocol["sources"]
    international = protocol["international"]
    if (tuple(international["asia"] + international["europe"]) != SYMBOLS
            or international["windows"] != list(WINDOWS)
            or international["max_extra_us_sessions"] != 3
            or source["source_end"] != SOURCE_END):
        raise ValueError("Fixed international input contract differs")
    output = root / "data/international_volatility"
    output.mkdir(parents=True, exist_ok=True)
    historical = pd.read_parquet(root / source["spx_extension"])
    current = pd.read_parquet(root / source["spx_existing"],
                              filters=[("date", "<=", pd.Timestamp(SOURCE_END))])
    daily, overlap = iterative_hf.merge_daily(historical, current)
    daily = daily.loc[:SOURCE_END]
    content = (root / source["archive"]).read_bytes()
    hf = iterative_hf.parse_oxford_archive(content, source["archive_sha256"], SOURCE_END)
    vix = pd.read_parquet(root / source["cboe"],
                          filters=[("date", "<=", pd.Timestamp(SOURCE_END))])["vix"]
    baseline = iterative_hf.build_features(daily, hf, vix).loc[:, BASE]
    foreign = parse_international_archive(content, source["archive_sha256"], SOURCE_END)
    foreign_features, availability = build_foreign_features(foreign, daily.index)
    features = pd.concat([baseline, foreign_features], axis=1).loc[:, ALL_FEATURES]
    targets = {h: iterative_hf.make_targets(hf.rv5, daily.index, h) for h in [1, 5, 21]}
    features.to_parquet(output / "features.parquet")
    availability_path = output / "foreign_availability.parquet"
    availability.to_parquet(availability_path, index=False)
    records = []
    for symbol, values in foreign.items():
        records.append({"symbol": symbol, "observed_rows": len(values),
                        "first_date": str(values.index.min().date()),
                        "last_date": str(values.index.max().date()),
                        "invalid_value_rows_retained": int(values.isna().sum())})
    audit = {"archive_sha256": hashlib.sha256(content).hexdigest(),
             "source_end": SOURCE_END, "daily_rows": len(daily),
             "feature_columns": list(ALL_FEATURES), "foreign_markets": records,
             "availability_file": str(availability_path.relative_to(root)),
             "availability_sha256": hashlib.sha256(availability_path.read_bytes()).hexdigest(),
             "fresh_meaning": "source date exists and additional US-session age is at most three",
             "complete_window_meaning": "all three selected local windows finite, independently of freshness",
             "overlap": overlap, "wave1_outputs_untouched": True}
    _dump(output / "source_audit.json", audit)
    return features, targets


def run_international(protocol, root=ROOT):
    """Called only after the caller's protocol freeze and prewritten test gate."""
    features, targets = prepare_inputs(protocol, root)
    forecasts, fits = forecast_panel(features, targets, protocol["hf"])
    output = pathlib.Path(root) / "data/international_volatility"
    forecast_path, fit_path = output / "forecasts.parquet", output / "fits.json"
    forecasts.to_parquet(forecast_path, index=False)
    _dump(fit_path, fits)
    return {"forecasts": str(forecast_path), "fits": str(fit_path)}
