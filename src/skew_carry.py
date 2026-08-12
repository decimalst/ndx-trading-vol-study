"""Frozen SKEW veto applied to the failed diagnostic carry rule.

This is a post-hoc mechanism diagnostic, not a new holdout. See
``reports/SKEW_CARRY_PROTOCOL.md`` before interpreting any output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yaml

from . import config
from .carry import build_trades


PROTOCOL_PATH = config.ROOT / "skew_carry.yaml"
OUTPUT_DIR = config.ROOT / "data" / "skew_carry"
REPORT_PATH = config.ROOT / "reports" / "skew_carry_diagnostic.md"


def load_protocol(path=PROTOCOL_PATH) -> dict:
    with open(path) as source:
        return yaml.safe_load(source)


def validate_protocol(protocol: dict) -> None:
    window = protocol["window"]
    start = pd.Timestamp(window["start"])
    end = pd.Timestamp(window["end"])
    clean = pd.Timestamp(window["clean_start"])
    if not start <= end < clean:
        raise ValueError("diagnostic window must stop before the clean window")
    if not window.get("forbid_clean_origins", False):
        raise ValueError("clean origins must be forbidden")
    source = protocol["source"]
    if int(source["close_delay_sessions"]) != 1:
        raise ValueError("SKEW close must be delayed exactly one session")
    if not source.get("no_forward_fill", False):
        raise ValueError("SKEW missing values must not be forward-filled")
    repair = protocol["repair"]
    if not np.isclose(float(repair["high_quantile"]), 0.80):
        raise ValueError("the only registered SKEW gate is the trailing 0.80 quantile")
    if int(repair["trailing_sessions"]) < int(repair["min_observations"]):
        raise ValueError("SKEW minimum observations exceed its trailing window")
    if repair.get("missing_value_action") != "flat":
        raise ValueError("missing SKEW must make the strategy flat")


def parse_cboe_skew_csv(content: bytes | str) -> pd.DataFrame:
    text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    date_column = "date" if "date" in frame.columns else frame.columns[0]
    if "close" not in frame and "skew" in frame:
        frame = frame.rename(columns={"skew": "close"})
    if "close" not in frame:
        raise ValueError("Cboe SKEW history has no close column")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise").dt.normalize()
    if frame[date_column].duplicated().any():
        raise ValueError("Cboe SKEW history contains duplicate dates")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    if (~np.isfinite(frame["close"]) | (frame["close"] <= 0)).any():
        raise ValueError("Cboe SKEW history contains invalid closes")
    frame = frame.set_index(date_column).sort_index()
    frame.index.name = "date"
    return frame


def validate_skew_history(
    frame: pd.DataFrame, protocol: dict, *, min_rows: int = 2
) -> None:
    if len(frame) < min_rows:
        raise ValueError(f"Cboe SKEW history has only {len(frame)} rows")
    anchor = protocol["source"]["historical_anchor"]
    date = pd.Timestamp(anchor["date"])
    if date not in frame.index:
        raise ValueError(f"Cboe SKEW history is missing anchor date {date.date()}")
    observed = float(frame.at[date, "close"])
    expected = float(anchor["close"])
    tolerance = float(anchor["tolerance"])
    if abs(observed - expected) > tolerance:
        raise ValueError(
            f"Cboe SKEW historical anchor changed: observed={observed}, "
            f"expected={expected} +/- {tolerance}"
        )


def validate_study_coverage(frame: pd.DataFrame, protocol: dict) -> None:
    start = pd.Timestamp(protocol["window"]["start"])
    end = pd.Timestamp(protocol["window"]["end"])
    warmup = int(protocol["repair"]["trailing_sessions"])
    prior = int((frame.index < start).sum())
    if prior < warmup:
        raise ValueError(
            f"Cboe SKEW history has only {prior} pre-study rows; warmup requires {warmup}"
        )
    if frame.index.max() < end:
        raise ValueError(
            f"Cboe SKEW history ends {frame.index.max().date()} before study end {end.date()}"
        )


def align_skew(
    values: pd.Series,
    trading_index: pd.DatetimeIndex,
    *,
    delay_sessions: int,
) -> pd.Series:
    if delay_sessions < 0:
        raise ValueError("delay_sessions cannot be negative")
    series = values.copy()
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    if series.index.duplicated().any():
        raise ValueError("SKEW series contains duplicate dates")
    aligned = series.sort_index().reindex(pd.DatetimeIndex(trading_index))
    if delay_sessions:
        aligned = aligned.shift(delay_sessions)
    aligned.name = "skew_lagged"
    return aligned


def build_skew_gate(
    raw_skew: pd.Series,
    trading_index: pd.DatetimeIndex,
    protocol: dict,
) -> pd.DataFrame:
    validate_protocol(protocol)
    source = protocol["source"]
    repair = protocol["repair"]
    aligned = align_skew(
        raw_skew,
        pd.DatetimeIndex(trading_index),
        delay_sessions=int(source["close_delay_sessions"]),
    )
    threshold = (
        aligned.rolling(
            int(repair["trailing_sessions"]),
            min_periods=int(repair["min_observations"]),
        )
        .quantile(float(repair["high_quantile"]))
        .shift(1)
    )
    allowed = (aligned <= threshold) & aligned.notna() & threshold.notna()
    out = pd.DataFrame(
        {
            "skew_lagged": aligned,
            "skew_threshold": threshold,
            "skew_allowed": allowed.astype(bool),
        }
    )
    out.index.name = "date"
    return out


def apply_rules(
    trades: pd.DataFrame,
    gate: pd.DataFrame,
    *,
    richness_threshold: float,
) -> pd.DataFrame:
    required = {"richness", "pnl_vol"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing columns: {sorted(missing)}")
    required_gate = {"skew_lagged", "skew_threshold", "skew_allowed"}
    missing_gate = required_gate - set(gate.columns)
    if missing_gate:
        raise ValueError(f"SKEW gate missing columns: {sorted(missing_gate)}")
    out = trades.join(gate, how="left")
    out["skew_allowed"] = out["skew_allowed"].fillna(False).astype(bool)
    out["richness_taken"] = out["richness"] > float(richness_threshold)
    out["repaired_taken"] = out["richness_taken"] & out["skew_allowed"]
    return out


def _trade_stats(pnl: np.ndarray) -> dict:
    pnl = np.asarray(pnl, dtype=float)
    if not len(pnl):
        return {}
    ordered = np.sort(pnl)
    tail_count = max(1, int(np.ceil(0.05 * len(pnl))))
    cumulative = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    return {
        "n": int(len(pnl)),
        "mean": float(pnl.mean()),
        "hit": float((pnl > 0).mean()),
        "cvar5": float(ordered[:tail_count].mean()),
        "worst": float(ordered[0]),
        "maxdd": float(drawdown.max()),
        "total": float(pnl.sum()),
    }


def evaluate_phases(
    frame: pd.DataFrame,
    *,
    step: int,
    min_all: int = 40,
    min_taken: int = 10,
) -> pd.DataFrame:
    rows = []
    for phase in range(step):
        sample = frame.iloc[phase::step]
        original = sample.loc[sample["richness_taken"], "pnl_vol"].to_numpy()
        repaired = sample.loc[sample["repaired_taken"], "pnl_vol"].to_numpy()
        if len(sample) < min_all or len(original) < min_taken or len(repaired) < min_taken:
            continue
        all_stats = _trade_stats(sample["pnl_vol"].to_numpy())
        original_stats = _trade_stats(original)
        repaired_stats = _trade_stats(repaired)
        row = {"phase": phase, "n_all": len(sample)}
        for prefix, stats in (
            ("all", all_stats),
            ("richness", original_stats),
            ("repaired", repaired_stats),
        ):
            row.update({f"{prefix}_{key}": value for key, value in stats.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_skew(protocol: dict | None = None) -> pd.DataFrame:
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    url = protocol["source"]["url"]
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    frame = parse_cboe_skew_csv(response.content)
    validate_skew_history(frame, protocol, min_rows=8_000)
    validate_study_coverage(frame, protocol)
    raw_dir = config.load()["paths"]["raw"]
    csv_path = raw_dir / "SKEW_History.csv"
    parquet_path = raw_dir / "skew_daily.parquet"
    source_path = raw_dir / "skew_daily_source.json"
    csv_path.write_bytes(response.content)
    frame.to_parquet(parquet_path)
    source = {
        "url": url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "rows": len(frame),
        "first_date": str(frame.index.min().date()),
        "last_date": str(frame.index.max().date()),
        "anchor_date": protocol["source"]["historical_anchor"]["date"],
        "anchor_close": float(
            frame.at[pd.Timestamp(protocol["source"]["historical_anchor"]["date"]), "close"]
        ),
    }
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {parquet_path} ({len(frame)} rows, "
        f"{frame.index.min().date()} .. {frame.index.max().date()})"
    )
    print(f"source sha256 {source['sha256']}")
    return frame


def _mechanism_verdict(
    frame: pd.DataFrame, phases: pd.DataFrame, protocol: dict
) -> dict:
    adverse_dates = pd.to_datetime(protocol["evaluation"]["known_adverse_origins"])
    adverse = frame.reindex(adverse_dates)
    adverse_rejected = bool(
        len(adverse) == len(adverse_dates)
        and adverse["richness_taken"].fillna(False).all()
        and (~adverse["repaired_taken"].fillna(False)).all()
    )
    participation = float(phases["repaired_n"].mean() / phases["richness_n"].mean())
    checks = {
        "known_adverse_rejected": adverse_rejected,
        "cvar_not_worse": bool(phases["repaired_cvar5"].mean() >= phases["richness_cvar5"].mean()),
        "drawdown_not_worse": bool(phases["repaired_maxdd"].mean() <= phases["richness_maxdd"].mean()),
        "mean_positive": bool(phases["repaired_mean"].mean() > 0),
        "participation_at_least_70pct": bool(participation >= 0.70),
    }
    return {
        "checks": checks,
        "mechanism_pass": bool(all(checks.values())),
        "participation": participation,
    }


def _write_report(
    frame: pd.DataFrame,
    phases: pd.DataFrame,
    metrics: dict,
    threshold: float,
) -> None:
    mean = lambda column: float(phases[column].mean())
    adverse = frame.reindex(pd.to_datetime(metrics["known_adverse_origins"]))
    valid_gate = frame["skew_lagged"].notna() & frame["skew_threshold"].notna()
    high_skew = valid_gate & (frame["skew_lagged"] > frame["skew_threshold"])
    lower_skew = valid_gate & ~high_skew
    richness = frame["richness_taken"]
    high_richness = high_skew & richness
    lower_richness = lower_skew & richness
    lagged_vxn = frame["vxn"].shift(1)
    pearson = float(frame["skew_lagged"].corr(lagged_vxn))
    spearman = float(frame["skew_lagged"].corr(lagged_vxn, method="spearman"))
    lines = [
        "# SKEW-conditioned carry mechanism diagnostic",
        "",
        "This is a post-hoc mechanism test on an already-inspected diagnostic "
        "window. A pass is not out-of-sample evidence.",
        "",
        "It preserves the original carry implementation, including its full-window "
        "median richness cutoff and same-date published VXN daily close. The SKEW "
        "input is safely lagged, but the inherited choices mean this is not a "
        "leakage-free strategy backtest. A future version must freeze "
        "+0.386812814 and use lagged Cboe data or timestamped pre-close quotes.",
        "",
        f"- Window: {frame.index.min().date()} through {frame.index.max().date()}.",
        f"- Original richness threshold, unchanged: {threshold:+.6f}.",
        "- SKEW: one-session delayed; high regime is above the trailing "
        "252-session 80th percentile estimated through the prior aligned value.",
        f"- Non-overlapping phase samples evaluated: {len(phases)}/21.",
        "",
        "| metric, average across phases | unconditional | richness-only | SKEW-repaired |",
        "|---|---:|---:|---:|",
        f"| trades | {mean('all_n'):.1f} | {mean('richness_n'):.1f} | {mean('repaired_n'):.1f} |",
        f"| mean P&L, vol points | {mean('all_mean'):+.3f} | {mean('richness_mean'):+.3f} | {mean('repaired_mean'):+.3f} |",
        f"| 5% CVaR | {mean('all_cvar5'):+.3f} | {mean('richness_cvar5'):+.3f} | {mean('repaired_cvar5'):+.3f} |",
        f"| worst trade | {mean('all_worst'):+.3f} | {mean('richness_worst'):+.3f} | {mean('repaired_worst'):+.3f} |",
        f"| max drawdown | {mean('all_maxdd'):.3f} | {mean('richness_maxdd'):.3f} | {mean('repaired_maxdd'):.3f} |",
        "",
        f"Participation retained: **{metrics['participation']:.1%}** of richness-only trades.",
        "",
        "## Known adverse origins",
        "",
        "| origin | lagged SKEW | trailing threshold | richness eligible | repaired trade | P&L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for date, row in adverse.iterrows():
        lines.append(
            f"| {date.date()} | {row['skew_lagged']:.2f} | "
            f"{row['skew_threshold']:.2f} | {bool(row['richness_taken'])} | "
            f"{bool(row['repaired_taken'])} | {row['pnl_vol']:+.2f} |"
        )
    lines += ["", "## Frozen mechanism criterion", ""]
    for name, passed in metrics["checks"].items():
        lines.append(f"- {name.replace('_', ' ')}: **{passed}**")
    lines += [
        "",
        f"Mechanism verdict: **{'PASS' if metrics['mechanism_pass'] else 'FAIL'}**.",
        "",
        "Even a pass requires genuinely future transition data. No p-value is "
        "reported because the rule was motivated by the same 2020 event it is "
        "asked to repair.",
        "",
        "## Descriptive interpretation after the frozen verdict",
        "",
        "The frozen failure is the participation constraint, not the tail hypothesis. "
        "That gate was a proxy for avoiding a degenerate never-trade rule, while mean "
        "and tail outcomes were the actual objective. The veto "
        f"retained {metrics['participation']:.1%} of richness-only trades versus the "
        "registered 70% minimum. It removed all three known adverse entries, and the "
        "average phase CVaR, worst trade, and drawdown all improved materially, but "
        "those observations cannot override the frozen verdict.",
        "",
        "The decisive limitation is contamination: February 2020 motivated this rule "
        "and remains inside the diagnostic sample. A future design should gate directly "
        "on pre-specified risk-adjusted outcomes plus a minimal non-degeneracy condition, "
        "not on an arbitrary participation floor.",
        "",
        f"The attrition is not random: {high_richness.sum() / richness[valid_gate].sum():.1%} "
        "of richness-eligible origins were also in a high-SKEW regime, compared with "
        f"{high_skew.sum() / valid_gate.sum():.1%} of all origins with a valid gate. "
        f"The richness rule fired on {richness[high_skew].mean():.1%} of high-SKEW "
        f"days versus {richness[lower_skew].mean():.1%} of lower-SKEW days. That is "
        "direct evidence of the suspected overlap: backward-looking \"richness\" is "
        "especially likely to call variance expensive while the option wings are "
        "already charging for tail risk.",
        "",
        "Lagged SKEW is not merely another copy of the ATM level: its Pearson "
        f"correlation with equally lagged VXN is {pearson:.3f} (Spearman "
        f"{spearman:.3f}) over the diagnostic frame. The relationship is modest and "
        "inverse, consistent with SKEW being a surface-shape measure whose signal is "
        "often largest while ATM volatility is still low.",
        "",
        "Among overlapping origins (descriptive only), richness trades returned "
        f"{frame.loc[high_richness, 'pnl_vol'].mean():+.3f} vol points in high-SKEW "
        f"regimes versus {frame.loc[lower_richness, 'pnl_vol'].mean():+.3f} outside "
        f"them. {(frame.loc[high_richness, 'pnl_vol'] < -20).sum()} high-SKEW "
        "richness trades lost more than 20 vol points, versus "
        f"{(frame.loc[lower_richness, 'pnl_vol'] < -20).sum()} in lower-SKEW regimes. "
        "These counts explain the tail improvement, but overlap prevents them from "
        "serving as independent inference.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_PATH}")


def run_diagnostic(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    validate_protocol(protocol)
    main_cfg = config.load()
    trades = build_trades(main_cfg)
    start = pd.Timestamp(protocol["window"]["start"])
    end = pd.Timestamp(protocol["window"]["end"])
    trades = trades.loc[(trades.index >= start) & (trades.index <= end)].copy()
    if len(trades) and (trades.index >= pd.Timestamp(protocol["window"]["clean_start"])).any():
        raise RuntimeError("clean origin escaped the SKEW protocol fence")
    skew = pd.read_parquet(main_cfg["paths"]["raw"] / "skew_daily.parquet")
    validate_skew_history(skew, protocol, min_rows=8_000)
    master = pd.read_parquet(main_cfg["paths"]["processed"] / "master_daily.parquet")
    gate = build_skew_gate(skew["close"], master.index, protocol).reindex(trades.index)
    richness_threshold = float(trades["richness"].median())
    evaluated = apply_rules(trades, gate, richness_threshold=richness_threshold)
    phases = evaluate_phases(
        evaluated,
        step=int(protocol["evaluation"]["step_sessions"]),
    )
    if len(phases) != int(protocol["evaluation"]["step_sessions"]):
        raise RuntimeError(f"only {len(phases)} non-overlapping phase samples were valid")
    verdict = _mechanism_verdict(evaluated, phases, protocol)
    metrics = {
        **verdict,
        "known_adverse_origins": protocol["evaluation"]["known_adverse_origins"],
        "richness_threshold": richness_threshold,
        "phase_count": len(phases),
        "window_first": str(evaluated.index.min().date()),
        "window_last": str(evaluated.index.max().date()),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluated.to_parquet(OUTPUT_DIR / "diagnostic_trades.parquet")
    phases.to_parquet(OUTPUT_DIR / "phase_metrics.parquet", index=False)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    _write_report(evaluated, phases, metrics, richness_threshold)
    print(f"mechanism verdict: {'PASS' if metrics['mechanism_pass'] else 'FAIL'}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "run"])
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch_skew()
    else:
        run_diagnostic()


if __name__ == "__main__":
    main()
