"""Realized variance, calendar covariates, and master dataset assembly.

Units: daily *variance* of log returns (not annualized). log_rv = ln(rv_total).
rv_total = intraday RV (or Garman-Klass fallback) + squared overnight gap, so it
is comparable to close-to-close total variance and to VXN's calendar-time quote.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

from . import config

LN2 = np.log(2.0)


# ---------------------------------------------------------------- realized vol
def rv_from_bars(cfg: dict) -> pd.Series:
    """Sum of squared 5-min log returns within RTH, per day."""
    files = sorted(glob.glob(str(cfg["paths"]["raw"] / f"bars_{cfg['ticker']}_*.parquet")))
    if not files:
        raise FileNotFoundError("no bar files — run the polygon fetcher or use --source daily")
    bars = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    bars["ts"] = pd.to_datetime(bars["ts"])
    if bars["ts"].dt.tz is None:
        bars["ts"] = bars["ts"].dt.tz_localize("UTC").dt.tz_convert(cfg["tz"])
    bars = bars.set_index("ts").sort_index()
    bars = bars.between_time(cfg["rth_start"], cfg["rth_end"])
    r = np.log(bars["close"]).groupby(bars.index.date).diff()
    rv = (r**2).groupby(bars.index.date).sum()
    n = r.groupby(bars.index.date).count()
    rv = rv[n >= cfg["min_bars_per_day"]]
    rv.index = pd.to_datetime(rv.index)
    rv.index.name = "date"
    return rv.rename("rv_intraday")


def rv_garman_klass(daily: pd.DataFrame) -> pd.Series:
    """Range-based daily variance from OHLC (Garman-Klass, RTH portion)."""
    hl = np.log(daily["high"] / daily["low"])
    co = np.log(daily["close"] / daily["open"])
    gk = 0.5 * hl**2 - (2 * LN2 - 1) * co**2
    return gk.clip(lower=1e-10).rename("rv_intraday")


def overnight_var(daily: pd.DataFrame) -> pd.Series:
    """Squared close(t-1) -> open(t) log return. Unadjusted prices; QQQ's
    quarterly ex-div gaps (~0.15%) are negligible once squared vs ~1% daily vol."""
    r_on = np.log(daily["open"] / daily["close"].shift(1))
    return (r_on**2).rename("var_overnight"), r_on.rename("r_overnight")


def downside_share(cfg: dict) -> pd.Series:
    """Fraction of each session's intraday variance from negative returns.

    Patton-Sheppard semivariance needs intraday returns; the free ceiling is
    yfinance 1-hour bars (~7 returns/session, ~730 sessions). Coarse for an RV
    estimator but adequate for a *share*, which is all this is used for.

    Returning a share rather than a level is deliberate: it lets the signed
    split be applied to the pre-registered rv_total without redefining the
    target, so HAR-SV stays comparable to every model already scored.
    """
    p = cfg["paths"]["raw"] / "hourly_bars.parquet"
    if not p.exists():
        return pd.Series(dtype=float, name="dn_share")
    bars = pd.read_parquet(p)
    bars.index = pd.to_datetime(bars.index)
    r = np.log(bars["close"]).groupby(bars.index.date).diff()
    sq = r**2
    neg = (sq * (r < 0)).groupby(bars.index.date).sum()
    tot = sq.groupby(bars.index.date).sum()
    n = r.groupby(bars.index.date).count()
    share = (neg / tot.replace(0.0, np.nan))[n >= 4]  # need a usable session
    share.index = pd.to_datetime(share.index)
    share.index.name = "date"
    return share.rename("dn_share")


def add_semivariance(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Split rv_total into signed halves that sum back to it exactly.

    Intraday variance is split by the hourly-return downside share; the
    overnight gap is assigned whole to the sign of the overnight return, since
    it is a single return and has an unambiguous sign.
    """
    share = downside_share(cfg).reindex(df.index)
    on_neg = (df["r_overnight"] < 0).astype(float)
    df["dn_share"] = share
    df["rv_neg"] = share * df["rv_intraday"] + on_neg * df["var_overnight"]
    df["rv_pos"] = (1 - share) * df["rv_intraday"] + (1 - on_neg) * df["var_overnight"]
    return df


def build_rv(cfg: dict, source: str = "daily") -> pd.DataFrame:
    daily = pd.read_parquet(cfg["paths"]["raw"] / "daily_ohlc.parquet")
    on, r_on = overnight_var(daily)
    if source == "bars":
        intraday = rv_from_bars(cfg)
    elif source == "daily":
        intraday = rv_garman_klass(daily)
    else:
        raise ValueError(source)
    df = pd.concat([intraday, on, r_on], axis=1).dropna(subset=["rv_intraday",
                                                                "var_overnight"])
    df["rv_total"] = df["rv_intraday"] + df["var_overnight"]
    df["log_rv"] = np.log(df["rv_total"].clip(lower=1e-10))
    # daily close-to-close log return (adjusted) for the EWMA baseline
    adj = daily["adj close"] if "adj close" in daily.columns else daily["close"]
    df["ret_cc"] = np.log(adj / adj.shift(1)).reindex(df.index)
    df = add_semivariance(cfg, df)
    return df


# ----------------------------------------------------------------- covariates
def _load_calendar(path, date_col="date") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, comment="#")
    except FileNotFoundError:
        print(f"warning: {path} missing — features default to 0", file=sys.stderr)
        return pd.DataFrame(columns=[date_col])
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def build_covariates(cfg: dict, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    cal_dir = cfg["paths"]["calendars"]
    cap = cfg["days_to_event_cap"]
    out = pd.DataFrame(index=trading_dates)
    idx = pd.Series(np.arange(len(trading_dates)), index=trading_dates)

    for name in ("fomc", "cpi", "nfp"):
        dates = _load_calendar(cal_dir / f"{name}.csv")["date"]
        dates = pd.DatetimeIndex(sorted(d for d in dates if d in trading_dates))
        out[f"is_{name}"] = out.index.isin(dates).astype(float)
        if len(dates):
            pos_next = np.searchsorted(trading_dates, dates, side="left")
            days_to = np.full(len(trading_dates), cap, dtype=float)
            for p in pos_next:  # trading-day countdown to the next event
                lo = max(0, p - cap)
                days_to[lo : p + 1] = np.minimum(days_to[lo : p + 1], np.arange(p - lo, -1, -1))
            out[f"days_to_{name}"] = days_to
        else:
            out[f"days_to_{name}"] = float(cap)

    # earnings weight loading: BMO on day d plus AMC on the prior trading day
    earn = _load_calendar(cal_dir / "earnings_top.csv")
    out["earnings_wt"] = 0.0
    if len(earn) and {"session", "weight_pct"}.issubset(earn.columns):
        for _, row in earn.iterrows():
            d = row["date"]
            if row["session"] == "bmo":
                eff = d
            else:  # amc or unknown -> impact lands next trading day
                later = trading_dates[trading_dates > d]
                if not len(later):
                    continue
                eff = later[0]
            if eff in out.index:
                out.loc[eff, "earnings_wt"] += float(row["weight_pct"])

    out["dow"] = out.index.dayofweek.astype(float)
    return out


# -------------------------------------------------------------------- assembly
def forward_realized_var(rv_total: pd.Series, cal_days: int) -> pd.Series:
    """Sum of rv_total over trading dates in (t, t + cal_days]. Eval-only."""
    idx = rv_total.index
    vals = rv_total.values
    fwd = np.full(len(idx), np.nan)
    j = 0
    for i, t in enumerate(idx):
        end = t + pd.Timedelta(days=cal_days)
        j = max(j, i + 1)
        while j < len(idx) and idx[j] <= end:
            j += 1
        if j < len(idx) or (len(idx) and idx[-1] >= end):
            fwd[i] = vals[i + 1 : j].sum()
        j = i + 1  # reset scan start for next origin
    return pd.Series(fwd, index=idx, name="fwd_var_30c")


def assemble(cfg: dict, source: str = "daily") -> pd.DataFrame:
    rv = build_rv(cfg, source=source)
    cov = build_covariates(cfg, rv.index)
    df = rv.join(cov)
    vxn_path = cfg["paths"]["raw"] / "vxn_daily.parquet"
    if vxn_path.exists():
        vxn = pd.read_parquet(vxn_path)
        close_col = "close" if "close" in vxn.columns else vxn.columns[-1]
        df["vxn"] = vxn[close_col].reindex(df.index)
        # market's expected 30-calendar-day variance in daily-log-return units
        df["vxn_exp_var_30c"] = (df["vxn"] / 100.0) ** 2 * (30.0 / 365.0)
    # Short-dated implied vol (SPX proxy — no NDX short-dated index exists) and
    # CBOE implied correlation / average constituent vol. Joined here so they
    # are available to models; each is opt-in per model, not automatic.
    for fname, cols in (("short_dated_iv.parquet", ("vix9d", "vix1d", "vix")),
                        ("implied_corr.parquet", ("cor1m", "cor3m", "cor6m", "vixeq"))):
        p = cfg["paths"]["raw"] / fname
        if not p.exists():
            continue
        extra = pd.read_parquet(p)
        for c in cols:
            if c in extra.columns:
                df[c] = extra[c].reindex(df.index)
    df["fwd_var_30c"] = forward_realized_var(df["rv_total"], cfg["cum_horizon_cal_days"])
    out = cfg["paths"]["processed"] / f"master_{source}.parquet"
    df.to_parquet(out)
    print(f"wrote {out} ({len(df)} rows, {df.index.min().date()} .. {df.index.max().date()})")
    return df


if __name__ == "__main__":
    cfg = config.load()
    src = sys.argv[1] if len(sys.argv) > 1 else "daily"
    assemble(cfg, source=src)
