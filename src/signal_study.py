"""Leakage-controlled, diagnostic-only orthogonal-signal study.

The protocol in signal_study.yaml was committed before this module existed.
Discovery selects at most one candidate on 2016-2021. Confirmation spends the
2022-2025-10-17 holdout once, and only for the protocol-locked winner. The clean
window is outside both stages by construction.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from . import config, metrics, models


PROTOCOL_PATH = config.ROOT / "signal_study.yaml"
BASELINE = "safe_har_iv_lev"
SIGNAL_COLUMNS = ("iv_term_slope", "xasset_stress")
DEFAULT_ASSETS = ("hyg", "tlt", "gld", "uso", "uup")


def load_protocol(path: str | pathlib.Path | None = None) -> dict:
    with open(pathlib.Path(path) if path else PROTOCOL_PATH) as f:
        return yaml.safe_load(f)


def validate_protocol(protocol: dict, main_cfg: dict) -> None:
    """Reject overlap, reordered windows, and drift from the main clean fence."""
    w = protocol["windows"]
    ds = pd.Timestamp(w["discovery_start"])
    de = pd.Timestamp(w["discovery_end"])
    cs = pd.Timestamp(w["confirmation_start"])
    ce = pd.Timestamp(w["confirmation_end"])
    clean = pd.Timestamp(main_cfg["clean_start"])
    fence = pd.Timestamp(protocol["fences"]["clean_start"])
    diag_end = pd.Timestamp(main_cfg["diagnostic_end"])
    if not ds <= de < cs <= ce:
        raise ValueError("study windows must be ordered and disjoint")
    if ce > diag_end or ce >= clean or ce >= fence:
        raise ValueError("confirmation window crosses the diagnostic/clean fence")
    if clean != fence:
        raise ValueError("signal-study clean fence disagrees with config.yaml")
    if not protocol["fences"].get("forbid_clean_origins", False):
        raise ValueError("clean origins must be forbidden")
    candidates = list(protocol["models"]["candidates"])
    if protocol["models"]["baseline"] != BASELINE:
        raise ValueError(f"baseline must remain {BASELINE}")
    if candidates != ["term_slope", "cross_asset", "combined"]:
        raise ValueError("candidate set or order changed from the frozen protocol")


def stage_origins(index: pd.DatetimeIndex, protocol: dict, stage: str) -> pd.DatetimeIndex:
    if stage not in ("discovery", "confirmation"):
        raise ValueError(f"unknown stage: {stage}")
    w = protocol["windows"]
    lo = pd.Timestamp(w[f"{stage}_start"])
    hi = pd.Timestamp(w[f"{stage}_end"])
    clean = pd.Timestamp(protocol["fences"]["clean_start"])
    out = pd.DatetimeIndex(index[(index >= lo) & (index <= hi)])
    if len(out) and (out >= clean).any():
        raise ValueError("clean origin escaped the protocol fence")
    return out


def align_daily_observations(values: pd.Series, trading_index: pd.DatetimeIndex,
                             delay_sessions: int) -> pd.Series:
    """Align exact-date observations, then apply an explicit session delay.

    There is deliberately no as-of join and no forward-fill: a source missing
    on an origin date remains missing rather than silently becoming stale data.
    """
    if delay_sessions < 0:
        raise ValueError("delay_sessions cannot be negative")
    s = values.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    out = s.reindex(pd.DatetimeIndex(trading_index))
    if delay_sessions:
        out = out.shift(delay_sessions)
    return out


def build_signal_features(trading_index: pd.DatetimeIndex,
                          cross_asset_close: pd.DataFrame,
                          short_iv: pd.DataFrame,
                          protocol: dict) -> pd.DataFrame:
    """Build the two fixed candidate features under their timing contracts."""
    idx = pd.DatetimeIndex(trading_index)
    close = cross_asset_close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.columns = [str(c).lower() for c in close.columns]
    assets = tuple(protocol.get("cross_asset", {}).get("prices", {}).keys()) or DEFAULT_ASSETS
    missing = [c for c in assets if c not in close.columns]
    if missing:
        raise ValueError(f"cross-asset input missing columns: {missing}")

    returns = np.log(close.loc[:, assets].where(close.loc[:, assets] > 0)).diff()
    window = int(protocol["cross_asset"]["scale_window"])
    min_obs = int(protocol["cross_asset"]["min_scale_observations"])
    # The shift is the important line: the current return is divided by a scale
    # estimated only through the prior asset session.
    mu = returns.rolling(window, min_periods=min_obs).mean().shift(1)
    sd = returns.rolling(window, min_periods=min_obs).std(ddof=1).shift(1)
    z = (returns - mu) / sd.replace(0.0, np.nan)
    z_aligned = pd.DataFrame(index=idx)
    delay = int(protocol["information_set"]["qqq_and_cross_asset_close_delay_sessions"])
    for col in assets:
        z_aligned[col] = align_daily_observations(z[col], idx, delay)

    out = pd.DataFrame(index=idx)
    out["xasset_stress"] = np.sqrt(z_aligned.pow(2).mean(axis=1, skipna=False))

    iv = short_iv.copy()
    iv.index = pd.to_datetime(iv.index).tz_localize(None).normalize()
    iv.columns = [str(c).lower() for c in iv.columns]
    if not {"vix9d", "vix"}.issubset(iv.columns):
        raise ValueError("short-IV input requires vix9d and vix")
    slope = np.log(iv["vix9d"].where(iv["vix9d"] > 0) /
                   iv["vix"].where(iv["vix"] > 0))
    cboe_delay = int(protocol["information_set"]["cboe_daily_close_delay_sessions"])
    out["iv_term_slope"] = align_daily_observations(slope, idx, cboe_delay)
    out.index.name = "date"
    return out.loc[:, SIGNAL_COLUMNS]


def _design(master: pd.DataFrame, signals: pd.DataFrame, candidate: str) -> pd.DataFrame:
    allowed = {BASELINE, "term_slope", "cross_asset", "combined"}
    if candidate not in allowed:
        raise ValueError(f"unknown candidate: {candidate}")
    d = pd.DataFrame(index=master.index)
    d["const"] = 1.0
    d["lrv_d"] = master["log_rv"]
    d["lrv_w"] = np.log(master["rv_total"].rolling(5).mean())
    d["lrv_m"] = np.log(master["rv_total"].rolling(22).mean())
    ret = master["ret_cc"]
    rw = ret.rolling(5).mean()
    rm = ret.rolling(22).mean()
    d["lev_d"] = ret.where(ret < 0, 0.0)
    d["lev_w"] = rw.where(rw < 0, 0.0)
    d["lev_m"] = rm.where(rm < 0, 0.0)
    # Cboe daily index values can contain 16:00-16:15 information. The complete
    # one-session delay makes the feature safe for a 16:00 QQQ forecast origin.
    d["liv_safe"] = np.log(master["vxn"].where(master["vxn"] > 0)).shift(1)
    if candidate in ("term_slope", "combined"):
        d["iv_term_slope"] = signals["iv_term_slope"].reindex(d.index)
    if candidate in ("cross_asset", "combined"):
        d["xasset_stress"] = signals["xasset_stress"].reindex(d.index)
    d["y_next"] = master["log_rv"].shift(-1)
    return d


def run_walk_forward(master: pd.DataFrame, signals: pd.DataFrame,
                     origins: pd.DatetimeIndex, main_cfg: dict,
                     candidate: str) -> pd.DataFrame:
    """Expanding OLS whose row at origin t never trains on y(t+1)."""
    taus = np.asarray(main_cfg["quantiles"], dtype=float)
    design = _design(master, signals, candidate)
    feat_cols = [c for c in design.columns if c != "y_next"]
    rows: list[dict] = []
    for t in pd.DatetimeIndex(origins):
        if t not in design.index:
            continue
        train = design.loc[:t].dropna()
        train = train[train.index < t]
        if len(train) < int(main_cfg["min_train_days"]):
            continue
        x_t = design.loc[t, feat_cols].to_numpy(dtype=float)
        if np.any(~np.isfinite(x_t)):
            continue
        X = train[feat_cols].to_numpy(dtype=float)
        y = train["y_next"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        mu = float(x_t @ beta)
        q = mu + np.quantile(resid, taus)
        rows.append({
            "origin": t,
            **{f"q{tau:.2f}": float(v) for tau, v in zip(taus, q)},
            "mean_var": models.trunc_mean_var(taus, q),
        })
    if not rows:
        cols = [f"q{t:.2f}" for t in taus] + ["mean_var"]
        out = pd.DataFrame(columns=cols)
        out.index = pd.DatetimeIndex([], name="origin")
        return out
    return pd.DataFrame(rows).set_index("origin")


def score_forecasts(master: pd.DataFrame, forecasts: pd.DataFrame,
                    main_cfg: dict) -> pd.DataFrame:
    taus = np.asarray(main_cfg["quantiles"], dtype=float)
    y_log = master["log_rv"].shift(-1).rename("y_log")
    y_var = master["rv_total"].shift(-1).rename("y_var")
    joined = forecasts.join(y_log).join(y_var).dropna(subset=["y_log", "y_var", "mean_var"])
    out = pd.DataFrame(index=joined.index)
    out["qlike"] = metrics.qlike(joined["y_var"].to_numpy(),
                                  joined["mean_var"].to_numpy())
    qcols = [f"q{t:.2f}" for t in taus]
    if all(c in joined for c in qcols):
        out["y_log"] = joined["y_log"]
        out["lo"] = joined[qcols[0]]
        out["hi"] = joined[qcols[-1]]
    return out


def select_discovery_winner(losses: dict[str, pd.Series],
                            protocol: dict) -> tuple[str | None, dict]:
    """Select strictly on one common-origin set; ties use the frozen order."""
    names = [protocol["models"]["baseline"], *protocol["models"]["candidates"]]
    missing = [n for n in names if n not in losses]
    if missing:
        raise ValueError(f"missing discovery loss series: {missing}")
    common = losses[names[0]].dropna().index
    for name in names[1:]:
        common = common.intersection(losses[name].dropna().index)
    common = common.sort_values()
    if not len(common):
        raise ValueError("no common discovery origins")
    means = {name: float(losses[name].reindex(common).mean()) for name in names}
    base = means[BASELINE]
    order = list(protocol["selection"]["tie_break_order"])
    rank = {name: i for i, name in enumerate(order)}
    best = min(order, key=lambda name: (means[name], rank[name]))
    winner = best if means[best] < base else None
    improvement = {name: 100.0 * (base - means[name]) / base for name in order}
    return winner, {
        "n_common": int(len(common)),
        "first_origin": str(common.min().date()),
        "last_origin": str(common.max().date()),
        "mean_qlike": means,
        "improvement_pct": improvement,
    }


def _protocol_sha256(protocol: dict) -> str:
    raw = json.dumps(protocol, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_discovery_lock(path: pathlib.Path, protocol: dict, winner: str | None,
                         discovery_scores: dict) -> None:
    if winner is not None and winner not in protocol["models"]["candidates"]:
        raise ValueError("winner is not a registered candidate")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_sha256": _protocol_sha256(protocol),
        "winner": winner,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovery_scores": discovery_scores,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_discovery_lock(path: pathlib.Path, protocol: dict) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("protocol_sha256") != _protocol_sha256(protocol):
        raise ValueError("protocol changed after discovery lock was written")
    winner = payload.get("winner")
    if winner is not None and winner not in protocol["models"]["candidates"]:
        raise ValueError("lock names an unregistered winner")
    return payload


def _paths(protocol: dict) -> tuple[pathlib.Path, pathlib.Path]:
    out = config.ROOT / protocol["fences"]["outputs_dir"]
    reports = config.ROOT / protocol["fences"]["reports_dir"]
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return out, reports


def _load_inputs(main_cfg: dict, protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = main_cfg["paths"]["raw"]
    master = pd.read_parquet(main_cfg["paths"]["processed"] / "master_daily.parquet")
    cross_path = raw / "cross_asset_daily.parquet"
    if not cross_path.exists():
        raise FileNotFoundError(f"{cross_path} missing; run make fetch-signal-inputs")
    short_path = raw / "short_dated_iv.parquet"
    signals = build_signal_features(master.index, pd.read_parquet(cross_path),
                                    pd.read_parquet(short_path), protocol)
    return master, signals


def _paired_details(a: pd.Series, b: pd.Series) -> dict:
    j = pd.concat([a.rename("candidate"), b.rename("baseline")], axis=1).dropna()
    dm = metrics.dm_test(j["candidate"].to_numpy(), j["baseline"].to_numpy(), h=1)
    wins = int((j["candidate"] < j["baseline"]).sum())
    return {
        "n": int(len(j)),
        "candidate_mean": float(j["candidate"].mean()),
        "baseline_mean": float(j["baseline"].mean()),
        "improvement_pct": float(100 * (j["baseline"].mean() - j["candidate"].mean()) /
                                 j["baseline"].mean()),
        "wins": wins,
        "win_rate": float(wins / len(j)),
        "dm": float(dm["dm"]),
        "dm_p": float(dm["p"]),
    }


def cmd_discover() -> None:
    main_cfg = config.load()
    protocol = load_protocol()
    validate_protocol(protocol, main_cfg)
    master, signals = _load_inputs(main_cfg, protocol)
    origins = stage_origins(master.index, protocol, "discovery")
    names = [BASELINE, *protocol["models"]["candidates"]]
    out_dir, report_dir = _paths(protocol)
    forecasts: dict[str, pd.DataFrame] = {}
    scores: dict[str, pd.DataFrame] = {}
    for name in names:
        fc = run_walk_forward(master, signals, origins, main_cfg, name)
        if fc.empty:
            raise RuntimeError(f"{name} emitted no discovery forecasts")
        forecasts[name] = fc
        scores[name] = score_forecasts(master, fc, main_cfg)
        fc.to_parquet(out_dir / f"discovery_{name}.parquet")
    winner, summary = select_discovery_winner(
        {name: scores[name]["qlike"] for name in names}, protocol
    )
    lock_path = out_dir / "discovery_lock.json"
    write_discovery_lock(lock_path, protocol, winner, summary)

    lines = ["# Orthogonal-signal discovery", "",
             "Selection window: 2016-01-04 through 2021-12-31. "
             "All models use the same origins shown below.", "",
             "| model | n | mean QLIKE | improvement vs safe baseline | DM p vs baseline | win rate |",
             "|---|---:|---:|---:|---:|---:|"]
    base = scores[BASELINE]["qlike"]
    for name in names:
        if name == BASELINE:
            lines.append(f"| {name} | {summary['n_common']} | "
                         f"{summary['mean_qlike'][name]:.6f} | — | — | — |")
            continue
        d = _paired_details(scores[name]["qlike"], base)
        lines.append(f"| {name} | {d['n']} | {d['candidate_mean']:.6f} | "
                     f"{d['improvement_pct']:+.3f}% | {d['dm_p']:.4f} | "
                     f"{d['win_rate']:.3f} |")
    lines += ["", f"Locked winner: **{winner or 'none'}**.", "",
              "The DM and win-rate columns are descriptive in discovery; only mean "
              "QLIKE selects the winner under the frozen rule."]
    (report_dir / "discovery.md").write_text("\n".join(lines) + "\n")
    print(f"discovery lock: {winner or 'none'} -> {lock_path}")


def cmd_confirm() -> None:
    main_cfg = config.load()
    protocol = load_protocol()
    validate_protocol(protocol, main_cfg)
    out_dir, report_dir = _paths(protocol)
    lock = read_discovery_lock(out_dir / "discovery_lock.json", protocol)
    winner = lock["winner"]
    if winner is None:
        text = ("# Orthogonal-signal confirmation\n\nNot run: discovery locked no "
                "winner, so the confirmation window was not spent.\n")
        (report_dir / "confirmation.md").write_text(text)
        print("confirmation not run: discovery locked no winner")
        return

    master, signals = _load_inputs(main_cfg, protocol)
    origins = stage_origins(master.index, protocol, "confirmation")
    forecasts = {}
    scores = {}
    for name in (BASELINE, winner):
        fc = run_walk_forward(master, signals, origins, main_cfg, name)
        if fc.empty:
            raise RuntimeError(f"{name} emitted no confirmation forecasts")
        forecasts[name] = fc
        scores[name] = score_forecasts(master, fc, main_cfg)
        fc.to_parquet(out_dir / f"confirmation_{name}.parquet")
    detail = _paired_details(scores[winner]["qlike"], scores[BASELINE]["qlike"])

    s = scores[winner].dropna(subset=["y_log", "lo", "hi"])
    nominal = float(main_cfg["quantiles"][-1] - main_cfg["quantiles"][0])
    cov = metrics.coverage_tests(s["y_log"].to_numpy(), s["lo"].to_numpy(),
                                 s["hi"].to_numpy(), nominal=nominal)
    success = (detail["candidate_mean"] < detail["baseline_mean"] and
               detail["dm_p"] < float(protocol["confirmation"]["alpha"]) and
               detail["win_rate"] > 0.5 and cov["p_ind"] > 0.05)
    payload = {**detail, "winner": winner, "coverage": float(cov["coverage"]),
               "p_uc": float(cov["p_uc"]), "p_ind": float(cov["p_ind"]),
               "success": bool(success)}
    (out_dir / "confirmation_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = ["# Orthogonal-signal confirmation", "",
             f"Locked candidate: **{winner}**", "",
             "| n | candidate QLIKE | baseline QLIKE | improvement | DM | DM p | wins | win rate |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|",
             f"| {detail['n']} | {detail['candidate_mean']:.6f} | "
             f"{detail['baseline_mean']:.6f} | {detail['improvement_pct']:+.3f}% | "
             f"{detail['dm']:.3f} | {detail['dm_p']:.4f} | {detail['wins']} | "
             f"{detail['win_rate']:.3f} |", "",
             f"Interval coverage: {cov['coverage']:.3f}; p_uc={cov['p_uc']:.4f}; "
             f"p_ind={cov['p_ind']:.4f}.", "",
             f"Pre-registered verdict: **{'PASS' if success else 'FAIL'}**."]
    (report_dir / "confirmation.md").write_text("\n".join(lines) + "\n")
    print(f"confirmation {winner}: {'PASS' if success else 'FAIL'}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("discover", "confirm"):
        raise SystemExit("usage: python -m src.signal_study discover|confirm")
    if sys.argv[1] == "discover":
        cmd_discover()
    else:
        cmd_confirm()


if __name__ == "__main__":
    main()

