"""Frozen SPX jump-versus-continuous target study.

The source and scoring contract are in ``target_regime.yaml``.  This study uses
SPX because the Oxford-Man archive's ``IXIC`` identity is not safe to relabel as
Nasdaq-100.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import requests
import yaml
from scipy.special import expit

from . import config

PROTOCOL_PATH = config.ROOT / "target_regime.yaml"
RAW_ARCHIVE = config.ROOT / "data" / "raw" / "oxford_man_realized.zip"
RAW_PARQUET = config.ROOT / "data" / "raw" / "oxford_man_spx.parquet"
SOURCE_META = config.ROOT / "data" / "raw" / "oxford_man_spx_source.json"
OUTPUT_DIR = config.ROOT / "data" / "target_regime"
REPORT_PATH = config.ROOT / "reports" / "jump_target.md"


def load_protocol(path=PROTOCOL_PATH) -> dict:
    with open(path) as source:
        return yaml.safe_load(source)["jump_target"]


def validate_protocol(protocol: dict) -> None:
    windows = protocol["windows"]
    training = pd.Timestamp(windows["training_start"])
    confirmation = pd.Timestamp(windows["confirmation_start"])
    end = pd.Timestamp(windows["confirmation_end"])
    if not training < confirmation <= end < pd.Timestamp("2022-01-01"):
        raise ValueError("jump study must use ordered windows ending before 2022")
    if not windows.get("forbid_2022_and_later", False):
        raise ValueError("Oxford-Man confirmation must forbid 2022 and later")
    source = protocol["source"]
    if source.get("asset") != "SPX":
        raise ValueError("only SPX is registered for the Oxford-Man study")
    if set(source.get("accepted_symbols", ())) & set(source.get("forbidden_symbols", ())):
        raise ValueError("accepted and forbidden source symbols overlap")
    if not source.get("immutable_hash_required", False):
        raise ValueError("the static source must be content-hashed")
    if not protocol["inputs"].get("no_forward_fill", False):
        raise ValueError("missing Cboe inputs must not be forward-filled")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        str(column).strip().lower().replace(".", "_").replace(" ", "_")
        for column in out.columns
    ]
    aliases = {
        "realized_variance_5_min": "rv5",
        "realized_variance_5min": "rv5",
        "bipower_variation_5_min": "bv",
        "bipower_variation_5min": "bv",
    }
    return out.rename(columns={key: value for key, value in aliases.items() if key in out})


def _parse_dates(values: pd.Series) -> pd.DatetimeIndex:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    if (~compact).any():
        # Oxford-Man labels the trading date at local midnight and changes its
        # UTC offset with daylight saving time. Converting those timestamps to
        # UTC would move summer rows to the previous calendar date. Preserve
        # the explicitly stated YYYY-MM-DD portion instead.
        wall_date = text.loc[~compact].str.slice(0, 10)
        parsed.loc[~compact] = pd.to_datetime(wall_date, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError("Oxford-Man archive contains invalid dates")
    return pd.DatetimeIndex(parsed).normalize()


def parse_oxford_archive(content: bytes, protocol: dict) -> pd.DataFrame:
    """Extract the registered SPX five-minute RV/BV series from a ZIP archive."""
    validate_protocol(protocol)
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError("Oxford-Man response is not a ZIP archive") from exc
    with archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Oxford-Man archive must contain one CSV, found {len(csv_names)}")
        frame = pd.read_csv(archive.open(csv_names[0]))
    frame = _normalize_columns(frame)
    symbol_col = next((c for c in ("symbol", "index", "ticker") if c in frame), None)
    if symbol_col is None:
        raise ValueError("Oxford-Man archive has no Symbol column")
    symbols = frame[symbol_col].astype(str).str.strip().str.upper()
    accepted = [str(value).upper() for value in protocol["source"]["accepted_symbols"]]
    selected_symbol = next((value for value in accepted if (symbols == value).any()), None)
    if selected_symbol is None:
        raise ValueError("Oxford-Man archive contains no registered SPX symbol")
    selected = frame.loc[symbols == selected_symbol].copy()
    required = set(protocol["source"]["required_columns"])
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Oxford-Man SPX series missing columns: {sorted(missing)}")
    date_col = next(
        (c for c in ("date", "dateid", "dataid", "x", "unnamed:_0") if c in selected),
        selected.columns[0],
    )
    selected.index = _parse_dates(selected[date_col])
    out = selected.loc[:, sorted(required)].apply(pd.to_numeric, errors="raise")
    out = out.rename_axis("date").sort_index()
    if out.index.duplicated().any():
        raise ValueError("Oxford-Man SPX series contains duplicate dates")
    validate_oxford_frame(out, protocol)
    out.attrs["source_symbol"] = selected_symbol
    return out


def validate_oxford_frame(
    frame: pd.DataFrame, protocol: dict, *, enforce_coverage: bool = False
) -> None:
    required = set(protocol["source"]["required_columns"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Oxford-Man frame missing columns: {sorted(missing)}")
    values = frame.loc[:, sorted(required)].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Oxford-Man RV/BV values must be finite and positive")
    if enforce_coverage:
        if frame.index.min() > pd.Timestamp(protocol["windows"]["training_start"]):
            raise ValueError("Oxford-Man SPX history starts after registered training")
        if frame.index.max() < pd.Timestamp(protocol["windows"]["confirmation_end"]):
            raise ValueError("Oxford-Man SPX history ends before confirmation")


def decompose_jump(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exactly reconciling continuous and non-negative jump components."""
    if {"rv5", "bv"} - set(frame.columns):
        raise ValueError("jump decomposition requires rv5 and bv")
    out = frame.copy()
    out["continuous"] = np.minimum(out["rv5"], out["bv"])
    out["jump"] = np.maximum(out["rv5"] - out["bv"], 0.0)
    out["jump_share"] = out["jump"] / out["rv5"]
    if not np.allclose(out["continuous"] + out["jump"], out["rv5"], rtol=1e-12, atol=1e-15):
        raise RuntimeError("jump components do not reconcile to realized variance")
    return out


def align_cboe_close(
    values: pd.Series, sessions: pd.DatetimeIndex, *, delay_sessions: int = 1
) -> pd.Series:
    if delay_sessions != 1:
        raise ValueError("registered Cboe close delay is exactly one session")
    series = values.copy()
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    if series.index.duplicated().any():
        raise ValueError("Cboe series contains duplicate dates")
    return series.sort_index().reindex(pd.DatetimeIndex(sessions)).shift(delay_sessions)


def completed_training_origins(
    index: pd.DatetimeIndex, *, cutoff: pd.Timestamp, horizon: int
) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index).sort_values()
    cutoff_pos = int(index.searchsorted(pd.Timestamp(cutoff), side="right") - 1)
    last_origin = cutoff_pos - int(horizon)
    if last_origin < 0:
        return pd.DatetimeIndex([])
    return index[: last_origin + 1]


def build_fold_targets(
    jump_share: pd.Series,
    *,
    cutoff: pd.Timestamp,
    origins: pd.DatetimeIndex,
    horizon: int,
    quantile: float,
) -> pd.DataFrame:
    series = jump_share.sort_index()
    training = series.loc[:pd.Timestamp(cutoff)].dropna()
    if training.empty:
        raise ValueError("jump fold has no threshold history")
    threshold = float(training.quantile(float(quantile)))
    positions = pd.Series(np.arange(len(series)), index=series.index)
    rows = []
    for origin in pd.DatetimeIndex(origins):
        if origin not in positions:
            continue
        position = int(positions.loc[origin])
        future = series.iloc[position + 1 : position + 1 + int(horizon)]
        if len(future) != int(horizon) or future.isna().any():
            continue
        rows.append(
            {
                "origin": origin,
                "target_end": future.index[-1],
                "threshold": threshold,
                "event": bool((future > threshold).any()),
            }
        )
    return pd.DataFrame(rows).set_index("origin") if rows else pd.DataFrame()


def _fit_logistic(X: pd.DataFrame, y: pd.Series, ridge: float) -> dict:
    values = X.to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (values - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    for _ in range(100):
        prob = expit(design @ beta)
        weight = np.clip(prob * (1.0 - prob), 1e-6, None)
        hessian = design.T @ (design * weight[:, None]) + penalty
        gradient = design.T @ (target - prob) - penalty @ beta
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return {"beta": beta, "mean": mean, "scale": scale, "columns": list(X.columns)}


def _predict_logistic(model: dict, X: pd.DataFrame) -> np.ndarray:
    values = X.loc[:, model["columns"]].to_numpy(dtype=float)
    z = (values - model["mean"]) / model["scale"]
    return np.clip(expit(np.column_stack([np.ones(len(z)), z]) @ model["beta"]), 1e-6, 1 - 1e-6)


def _losses(frame: pd.DataFrame, probability: str) -> tuple[float, float]:
    y = frame["event"].to_numpy(dtype=float)
    p = np.clip(frame[probability].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    return float(np.mean((p - y) ** 2)), float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def evaluate_phases(frame: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    rows = []
    for phase in range(step):
        sample = frame.iloc[phase::step]
        if sample.empty:
            continue
        row = {"phase": phase, "n": len(sample), "event_rate": float(sample["event"].mean())}
        for model in ("history", "atm", "surface"):
            brier, logloss = _losses(sample, f"p_{model}")
            row[f"{model}_brier"] = brier
            row[f"{model}_logloss"] = logloss
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_oxford(protocol: dict | None = None) -> pd.DataFrame:
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    response = requests.get(protocol["source"]["url"], timeout=60)
    response.raise_for_status()
    frame = parse_oxford_archive(response.content, protocol)
    validate_oxford_frame(frame, protocol, enforce_coverage=True)
    RAW_ARCHIVE.write_bytes(response.content)
    frame.to_parquet(RAW_PARQUET)
    metadata = {
        "url": protocol["source"]["url"],
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "rows": len(frame),
        "first_date": str(frame.index.min().date()),
        "last_date": str(frame.index.max().date()),
        "source_symbol": frame.attrs.get("source_symbol"),
    }
    SOURCE_META.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {RAW_PARQUET} ({len(frame)} rows, {frame.index.min().date()} .. {frame.index.max().date()})")
    print(f"source sha256 {metadata['sha256']}")
    return frame


def run_study(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    raw = pd.read_parquet(RAW_PARQUET)
    validate_oxford_frame(raw, protocol, enforce_coverage=True)
    data = decompose_jump(raw)
    data["jump_share_d"] = data["jump_share"]
    data["jump_share_w"] = data["jump_share"].rolling(5).mean()
    data["jump_share_m"] = data["jump_share"].rolling(22).mean()

    main_cfg = config.load()
    implied = pd.read_parquet(main_cfg["paths"]["raw"] / "short_dated_iv.parquet")
    skew = pd.read_parquet(main_cfg["paths"]["raw"] / "skew_daily.parquet")
    data["vix_lagged"] = align_cboe_close(implied["vix"], data.index)
    data["skew_lagged"] = align_cboe_close(skew["close"], data.index)

    window = protocol["windows"]
    start = pd.Timestamp(window["confirmation_start"])
    end = pd.Timestamp(window["confirmation_end"])
    horizon = int(protocol["target"]["horizon_sessions"])
    quantile = float(protocol["target"]["material_jump_quantile"])
    ridge = float(protocol["fitting"]["logistic_ridge"])
    min_train = int(protocol["fitting"]["min_train_observations"])
    feature_sets = {
        "history": ["jump_share_d", "jump_share_w", "jump_share_m"],
        "atm": ["jump_share_d", "jump_share_w", "jump_share_m", "vix_lagged"],
        "surface": ["jump_share_d", "jump_share_w", "jump_share_m", "vix_lagged", "skew_lagged"],
    }
    rows = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(f"{year}-01-01"))
        year_end = min(end, pd.Timestamp(f"{year}-12-31"))
        cutoff_candidates = data.index[data.index < year_start]
        if not len(cutoff_candidates):
            continue
        cutoff = cutoff_candidates[-1]
        train_origins = completed_training_origins(data.index, cutoff=cutoff, horizon=horizon)
        train_origins = train_origins[train_origins >= pd.Timestamp(window["training_start"])]
        test_origins = data.index[(data.index >= year_start) & (data.index <= year_end)]
        train_targets = build_fold_targets(
            data["jump_share"], cutoff=cutoff, origins=train_origins,
            horizon=horizon, quantile=quantile,
        )
        test_targets = build_fold_targets(
            data["jump_share"], cutoff=cutoff, origins=test_origins,
            horizon=horizon, quantile=quantile,
        )
        if train_targets.empty or test_targets.empty:
            continue
        test_targets = test_targets.loc[test_targets["target_end"] <= end]
        models = {}
        for name, columns in feature_sets.items():
            train = data.loc[train_targets.index, columns].join(train_targets["event"]).dropna()
            if len(train) < min_train:
                raise RuntimeError(f"{year} {name} has only {len(train)} training rows")
            models[name] = _fit_logistic(train[columns], train["event"], ridge)
        common = test_targets.index
        for columns in feature_sets.values():
            common = common.intersection(data.loc[common, columns].dropna().index)
        fold = test_targets.loc[common].copy()
        fold["fold_year"] = year
        fold["cutoff"] = cutoff
        stored_features = sorted({column for columns in feature_sets.values() for column in columns})
        for column in stored_features:
            fold[column] = data.loc[common, column]
        for name, columns in feature_sets.items():
            fold[f"p_{name}"] = _predict_logistic(models[name], data.loc[common, columns])
        rows.append(fold)
    if not rows:
        raise RuntimeError("jump-target study produced no confirmation forecasts")
    scored = pd.concat(rows).sort_index()
    if scored.index.min() < start or scored.index.max() > end:
        raise RuntimeError("jump-target origins escaped the registered confirmation window")
    phases = evaluate_phases(scored, step=horizon)
    avg = phases.mean(numeric_only=True)
    checks = {
        "surface_brier_below_atm": bool(avg["surface_brier"] < avg["atm_brier"]),
        "surface_logloss_below_atm": bool(avg["surface_logloss"] < avg["atm_logloss"]),
    }
    metrics = {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "origins": len(scored),
        "event_rate": float(scored["event"].mean()),
        "bv_above_rv_fraction": float((raw["bv"] > raw["rv5"]).mean()),
        "phase_average": {key: float(value) for key, value in avg.items()},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUTPUT_DIR / "jump_forecasts.parquet")
    phases.to_parquet(OUTPUT_DIR / "jump_phase_metrics.parquet", index=False)
    (OUTPUT_DIR / "jump_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    _write_report(metrics, phases, raw)
    print(f"jump-target verdict: {'PASS' if metrics['pass'] else 'FAIL'}")
    return metrics


def _write_report(metrics: dict, phases: pd.DataFrame, raw: pd.DataFrame) -> None:
    avg = metrics["phase_average"]
    lines = [
        "# SPX jump-target diagnostic",
        "",
        "The Oxford-Man SPX target was not used anywhere else in this repository. "
        "The surface hypothesis was motivated by prior NDX work, so this is an "
        "external mechanism confirmation rather than a pristine strategy test.",
        "",
        "The source columns are five-minute realized variance and five-minute "
        "bipower variation computed by Oxford-Man. The repository's hourly QQQ "
        "bars do not enter this study.",
        "",
        f"- Source coverage: {raw.index.min().date()} through {raw.index.max().date()}.",
        f"- Scored confirmation origins: {metrics['origins']}.",
        f"- Five-session material-jump event rate: {metrics['event_rate']:.1%}.",
        f"- BPV exceeded RV on {metrics['bv_above_rv_fraction']:.1%} of raw days; "
        "continuous variation was conservatively truncated to RV on those days.",
        "",
        "| model | phase-average Brier | phase-average log loss |",
        "|---|---:|---:|",
    ]
    for name in ("history", "atm", "surface"):
        lines.append(f"| {name} | {avg[f'{name}_brier']:.6f} | {avg[f'{name}_logloss']:.6f} |")
    lines.extend([
        "",
        f"- Surface Brier below ATM: **{metrics['checks']['surface_brier_below_atm']}**",
        f"- Surface log loss below ATM: **{metrics['checks']['surface_logloss_below_atm']}**",
        "",
        f"Frozen verdict: **{'PASS' if metrics['pass'] else 'FAIL'}**.",
        "",
        "A pass says surface shape transfers better than the ATM level to a jump "
        "target in SPX. It does not establish an NDX dispersion trade, and it does "
        "not repair the source's lack of a formal BNS significance statistic.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "run"))
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch_oxford()
    else:
        run_study()


if __name__ == "__main__":
    main()
