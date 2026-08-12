"""Data acquisition.

Free path:   fetch_daily_ohlc (yfinance) + fetch_vxn (CBOE) -> Garman-Klass RV.
Paid path:   fetch_polygon_bars (Polygon.io, POLYGON_API_KEY) -> 5-min RV.
Earnings:    fetch_yf_earnings (yfinance, no API key — the default) or
             fetch_fmp_earnings (Financial Modeling Prep, FMP_API_KEY). Either
             way, verify sessions by hand before the clean run.

All fetchers write parquet/CSV under data/raw and are idempotent (re-run to
refresh). Respect each provider's terms of service; raw data is for your own
research and must not be redistributed.
"""
from __future__ import annotations

import io
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import requests

from . import config

CBOE_VXN_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv"
CBOE_BASE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
# Short-dated implied vol, for the term-structure-slope extension. Checked
# 2026-08-11: CBOE publishes VIX9D and VIX1D free, but there is NO VXN9D,
# VXN1D or VXNC — the NDX complex has no short-dated index. These SPX series
# are therefore a cross-index PROXY for NDX surface slope; the horizon-matched
# NDX equivalent needs paid options data (ThetaData / ORATS / CBOE DataShop).
SHORT_DATED = ("VIX9D", "VIX1D", "VIX")
POLYGON_AGGS = "https://api.polygon.io/v2/aggs/ticker/{t}/range/{m}/minute/{s}/{e}"
FMP_EARNINGS = "https://financialmodelingprep.com/api/v3/historical/earning_calendar/{t}"


def fetch_daily_ohlc(cfg: dict) -> pd.DataFrame:
    """Daily unadjusted OHLC + adjusted close for the ETF, via yfinance."""
    import yfinance as yf

    tk = yf.Ticker(cfg["ticker"])
    df = tk.history(start="1999-03-10", auto_adjust=False)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "adj close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    out = cfg["paths"]["raw"] / "daily_ohlc.parquet"
    df.to_parquet(out)
    print(f"wrote {out} ({len(df)} rows, {df.index.min().date()} .. {df.index.max().date()})")
    return df


def fetch_vxn(cfg: dict) -> pd.DataFrame:
    """CBOE VXN (Nasdaq-100 30-day implied vol index) daily history, free.

    If the URL 404s, CBOE has moved the file: find the current CSV link on the
    VXN page at cboe.com and update CBOE_VXN_URL.
    """
    r = requests.get(CBOE_VXN_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"
    out = cfg["paths"]["raw"] / "vxn_daily.parquet"
    df.to_parquet(out)
    print(f"wrote {out} ({len(df)} rows)")
    return df


def fetch_short_dated_iv(cfg: dict) -> pd.DataFrame:
    """VIX9D / VIX1D / VIX daily history from CBOE (free).

    Groundwork for the term-structure extension: VXN gives the 30-day implied
    *level*, which is what HAR-IV already uses. Whether the surface's *slope*
    adds anything is untested, and needs a short-dated measure. No NDX
    short-dated index exists, so these SPX series stand in as a proxy.
    Fetched, not yet used by any model.
    """
    out = {}
    for sym in SHORT_DATED:
        r = requests.get(CBOE_BASE.format(sym=sym), timeout=30)
        if r.status_code != 200:
            print(f"  {sym}: HTTP {r.status_code} — skipped", file=sys.stderr)
            continue
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = [c.strip().lower() for c in d.columns]
        dc = "date" if "date" in d.columns else d.columns[0]
        d[dc] = pd.to_datetime(d[dc])
        out[sym.lower()] = d.set_index(dc).sort_index()["close"]
    if not out:
        sys.exit("no short-dated IV series fetched")
    df = pd.DataFrame(out)
    df.index.name = "date"
    p = cfg["paths"]["raw"] / "short_dated_iv.parquet"
    df.to_parquet(p)
    print(f"wrote {p} ({len(df)} rows, cols={list(df.columns)})")
    for c in df.columns:
        s = df[c].dropna()
        print(f"  {c}: {s.index.min().date()} .. {s.index.max().date()} ({len(s)})")
    return df


def fetch_hourly_bars(cfg: dict) -> pd.DataFrame:
    """QQQ 1-hour bars via yfinance — the free intraday source, ~730 days deep.

    Used only to split each day's variance into signed halves (Patton-Sheppard
    semivariance). yfinance caps 5-minute history at 60 days, which is far short
    of the clean window; 1-hour reaches back ~2 years and gives ~7 intraday
    returns per session. Coarse for an RV estimator, adequate for a *share*.

    Deliberately NOT used to redefine rv_total: the target is pre-registered on
    the Garman-Klass path, and swapping the estimator mid-experiment would break
    comparability with every forecast already scored.
    """
    import yfinance as yf

    h = yf.Ticker(cfg["ticker"]).history(period="730d", interval="1h")
    if not len(h):
        sys.exit("no hourly bars returned")
    h.index = pd.to_datetime(h.index).tz_convert(cfg["tz"])
    h = h.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    h = h.between_time(cfg["rth_start"], cfg["rth_end"])
    out = cfg["paths"]["raw"] / "hourly_bars.parquet"
    h.to_parquet(out)
    days = h.index.normalize().nunique()
    print(f"wrote {out} ({len(h)} bars, {days} sessions, "
          f"{h.index.min().date()} .. {h.index.max().date()}, "
          f"{len(h) / days:.1f} bars/session)")
    return h


def fetch_implied_correlation(cfg: dict) -> pd.DataFrame:
    """CBOE implied-correlation indices + average constituent vol (free).

    COR1M/COR3M: market-expected average pairwise correlation among the largest
    SPX names, backed out of index vs component option implied vols.
    VIXEQ: average implied vol of those constituents.

    This makes the index-variance decomposition observable:
        index variance ~ (avg constituent variance) x (avg correlation)
    which is the direct test of the earnings mechanism — a heavy single-name
    earnings day should show constituent vol UP with implied correlation DOWN.

    SPX-constructed, so for NDX this is a proxy for the correlation *regime*,
    not a measurement of NDX's own. No NDX equivalent is published.
    """
    out = {}
    for sym in ("COR1M", "COR3M", "COR6M", "VIXEQ"):
        r = requests.get(CBOE_BASE.format(sym=sym), timeout=30)
        if r.status_code != 200:
            print(f"  {sym}: HTTP {r.status_code} — skipped", file=sys.stderr)
            continue
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = [c.strip().lower() for c in d.columns]
        dc = "date" if "date" in d.columns else d.columns[0]
        d[dc] = pd.to_datetime(d[dc])
        col = "close" if "close" in d.columns else d.columns[-1]
        out[sym.lower()] = d.set_index(dc).sort_index()[col]
    if not out:
        sys.exit("no correlation indices fetched")
    df = pd.DataFrame(out)
    df.index.name = "date"
    p = cfg["paths"]["raw"] / "implied_corr.parquet"
    df.to_parquet(p)
    print(f"wrote {p} ({len(df)} rows, cols={list(df.columns)})")
    for c in df.columns:
        s = df[c].dropna()
        print(f"  {c}: {s.index.min().date()} .. {s.index.max().date()} ({len(s)})")
    return df


def fetch_polygon_bars(cfg: dict, start: str, end: str) -> pd.DataFrame:
    """5-minute bars from Polygon.io aggregates API, paginated, rate-limit aware.

    Requires POLYGON_API_KEY in the environment and a plan whose history depth
    covers [start, end]. Writes one parquet per calendar year.
    """
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        sys.exit("POLYGON_API_KEY not set")
    frames = []
    url = POLYGON_AGGS.format(t=cfg["ticker"], m=cfg["bar_minutes"], s=start, e=end)
    params = {"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": key}
    while url:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(15)
            continue
        r.raise_for_status()
        js = r.json()
        rows = js.get("results", [])
        if rows:
            frames.append(pd.DataFrame(rows))
        nxt = js.get("next_url")
        url = nxt
        params = {"apiKey": key}  # next_url carries the cursor
        time.sleep(0.2)
    if not frames:
        sys.exit("no bars returned — check plan history depth and date range")
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(cfg["tz"])
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["ts", "open", "high", "low", "close", "volume"]].sort_values("ts")
    for year, grp in df.groupby(df["ts"].dt.year):
        out = cfg["paths"]["raw"] / f"bars_{cfg['ticker']}_{year}.parquet"
        grp.to_parquet(out, index=False)
        print(f"wrote {out} ({len(grp)} bars)")
    return df


# Share classes of one issuer report earnings once; fold the secondary class's
# index weight into the primary so earnings_wt counts the announcement's full
# index impact instead of half of it.
DUAL_CLASS = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}


def load_index_weights(cfg: dict, path: str | None = None,
                       min_weight: float = 2.0) -> dict[str, float]:
    """Ticker -> NDX weight (%) from an Invesco QQQ daily holdings CSV.

    Invesco's download endpoint rejects scripted requests (HTTP 406), so the
    CSV is saved by hand into data/raw/qqq_holdings_YYYY-MM-DD.csv; the newest
    one is used unless `path` is given. Only names at or above `min_weight` are
    kept — below ~1-2% an announcement's index-level effect washes into the
    correlation term and just adds noise to the covariate.
    """
    if path:
        p = pathlib.Path(path)
    else:
        cands = sorted(cfg["paths"]["raw"].glob("qqq_holdings_*.csv"))
        if not cands:
            sys.exit("no data/raw/qqq_holdings_*.csv — download QQQ 'Complete "
                     "Holdings' from invesco.com and save it there")
        p = cands[-1]
    df = pd.read_csv(p, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["w"] = pd.to_numeric(df["% TNA"].astype(str).str.rstrip("%"), errors="coerce")
    df = df.dropna(subset=["w", "Ticker"])
    weights: dict[str, float] = {}
    for tk, w in zip(df["Ticker"].astype(str).str.strip(), df["w"]):
        weights[DUAL_CLASS.get(tk, tk)] = weights.get(DUAL_CLASS.get(tk, tk), 0.0) + float(w)
    kept = {k: round(v, 2) for k, v in weights.items() if v >= min_weight}
    print(f"{p.name}: {len(kept)} names at >= {min_weight}% "
          f"({sum(kept.values()):.1f}% of index)")
    return dict(sorted(kept.items(), key=lambda kv: -kv[1]))


def _parse_ticker_weights(spec: list[str]) -> dict[str, float]:
    """Parse ['NVDA:12.7', 'AAPL:10.7'] -> {'NVDA': 12.7, ...}.

    A bare ticker gets weight 0.0, which makes it contribute nothing to
    earnings_wt — pass real index weights or the covariate is inert.
    """
    out = {}
    for item in spec:
        tk, _, w = item.partition(":")
        out[tk.strip().upper()] = float(w) if w else 0.0
    return out


def fetch_yf_earnings(cfg: dict, spec: list[str] | dict[str, float]) -> pd.DataFrame:
    """Historical + scheduled earnings dates per ticker via yfinance (no API key).

    Session is inferred from the announcement time, which yfinance reports in
    US/Eastern: before 09:30 -> bmo, at/after 16:00 -> amc, anything else
    'unknown' (treated as amc downstream). This inference is right for the
    mega-caps in practice but is still a guess — VERIFY against company IR
    pages before the clean run, since a wrong session shifts the vol-relevant
    day by one.

    Usage: make fetch-earnings TICKERS=NVDA:12.7,AAPL:10.7,MSFT:9.0
    """
    import yfinance as yf

    weights = spec if isinstance(spec, dict) else _parse_ticker_weights(spec)
    rows = []
    for tk, wt in weights.items():
        try:
            ed = yf.Ticker(tk).get_earnings_dates(limit=60)
        except Exception as e:  # noqa: BLE001 — one bad ticker must not kill the run
            print(f"  {tk}: earnings lookup failed ({type(e).__name__}: {e})",
                  file=sys.stderr)
            continue
        if ed is None or not len(ed):
            print(f"  {tk}: no earnings dates returned", file=sys.stderr)
            continue
        for ts in ed.index:
            ts = pd.Timestamp(ts)
            hhmm = ts.hour * 60 + ts.minute
            if hhmm and hhmm < 9 * 60 + 30:
                session = "bmo"
            elif hhmm >= 16 * 60:
                session = "amc"
            else:
                session = "unknown"
            rows.append({"date": ts.tz_localize(None).normalize()
                         if ts.tzinfo else ts.normalize(),
                         "ticker": tk, "session": session, "weight_pct": wt})
        time.sleep(0.3)
    if not rows:
        sys.exit("no earnings dates fetched")
    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "ticker"])
    df = df.sort_values(["date", "ticker"])
    out = config.ROOT / "calendars" / "earnings_fetched.csv"
    df.to_csv(out, index=False)
    n_unk = int((df["session"] == "unknown").sum())
    print(f"wrote {out} ({len(df)} rows, {n_unk} unknown-session)")
    print("  VERIFY sessions, then merge into calendars/earnings_top.csv")
    return df


def build_pit_weights(cfg: dict, tickers: list[str], anchor: dict[str, float],
                      start: str = "2001-01-01") -> pd.DataFrame:
    """Daily point-in-time index weight per ticker, from market capitalisation.

    A single Invesco snapshot applied to twenty years of announcements is a
    look-ahead: it gives NVDA its 2026 weight on its 2016 prints, and it leaks
    into exactly the heavy-earnings slice the experiment is watching.

    Construction, per date t:
        cap_i(t)    = close_i(t) * shares_i(t)      (shares step-changed, ffilled)
        share_i(t)  = cap_i(t) / sum_j cap_j(t)     (within the basket)
        weight_i(t) = share_i(t) * K,  K = sum_j anchor_j

    K is the basket's aggregate index weight on the snapshot date. Two
    approximations remain and are deliberate:
      - K is held constant, so the *rise in mega-cap concentration* is not
        captured; only the split between names is. Relative weights are the
        first-order term and this removes the bulk of the look-ahead.
      - Basket membership is today's. MU and AMD are here because they are
        large now, so pre-2020 weights carry survivorship. This matters for the
        diagnostic phase, not for the clean window.
    Neither uses any information after t except through K.
    """
    import yfinance as yf

    K = float(sum(anchor.values()))
    caps = {}
    for tk in tickers:
        try:
            t_obj = yf.Ticker(tk)
            h = t_obj.history(start=start, auto_adjust=False)
            if not len(h):
                print(f"  {tk}: no price history — SKIPPED", file=sys.stderr)
                continue
            px = h["Close"]
            px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
            sh = t_obj.get_shares_full(start=start)
            if sh is None or not len(sh):
                print(f"  {tk}: no shares history — SKIPPED (a price-only proxy "
                      f"would be off by orders of magnitude)", file=sys.stderr)
                continue
            sh = pd.Series(np.asarray(sh.values, dtype="float64"),
                           index=pd.to_datetime(sh.index).tz_localize(None).normalize()
                           ).sort_index()
            sh = sh[~sh.index.duplicated(keep="last")]
            sh = sh.reindex(px.index.union(sh.index)).ffill().reindex(px.index)
            # yfinance Close is SPLIT-ADJUSTED to today's basis, while
            # get_shares_full reports the point-in-time count. Multiplying them
            # directly understates old caps by the cumulative split factor —
            # NVDA's 2016 weight came out 40x too small before this correction.
            # Market cap is split-invariant, so restate shares onto the price's
            # basis: shares_adj(t) = shares(t) * prod(splits after t).
            splits = t_obj.splits
            factor = pd.Series(1.0, index=px.index)
            if splits is not None and len(splits):
                sp = pd.Series(splits.values,
                               index=pd.to_datetime(splits.index).tz_localize(None)
                               .normalize()).sort_index()
                sp = sp[sp > 0]
                # cumulative product of every split strictly after each date
                total = float(sp.prod())
                applied = sp.reindex(px.index.union(sp.index)).fillna(1.0).cumprod() \
                            .reindex(px.index).ffill().fillna(1.0)
                factor = total / applied
            caps[tk] = px * sh * factor
        except Exception as e:  # noqa: BLE001
            print(f"  {tk}: {type(e).__name__}: {e}", file=sys.stderr)
        time.sleep(0.2)
    if not caps:
        sys.exit("no market caps built")
    cap = pd.DataFrame(caps).sort_index().ffill()
    # Rescale each column by a constant so the snapshot date reproduces the
    # Invesco anchor (absorbing float adjustment and the index's capping rules),
    # then renormalise. The constant is time-invariant, so relative weight
    # history is still driven entirely by market cap.
    w = cap.div(cap.sum(axis=1), axis=0) * K
    snap = w.dropna(how="all").index[-1]
    adj = pd.Series({t: (anchor.get(t, np.nan) / w.loc[snap, t])
                     for t in w.columns if w.loc[snap, t] > 0})
    w = (cap * adj).div((cap * adj).sum(axis=1), axis=0) * K
    out = cfg["paths"]["raw"] / "pit_weights.parquet"
    w.to_parquet(out)
    print(f"wrote {out} ({len(w)} days, {w.shape[1]} tickers, "
          f"{w.index.min().date()} .. {w.index.max().date()})")
    print("  weight drift check (NVDA):")
    for d in ("2016-06-30", "2020-06-30", "2024-06-28", "2025-11-03", str(snap.date())):
        try:
            print(f"    {d}: {float(w['NVDA'].asof(pd.Timestamp(d))):.2f}%")
        except Exception:
            pass
    return w


def merge_earnings(cfg: dict) -> pd.DataFrame:
    """Promote calendars/earnings_fetched.csv to the file features.py reads.

    Kept as a separate step so the fetch output can be eyeballed (especially the
    session column) before it becomes an input to the experiment.
    """
    cal = config.ROOT / "calendars"
    src = cal / "earnings_fetched.csv"
    if not src.exists():
        sys.exit(f"{src} missing — run `make fetch-earnings` first")
    df = pd.read_csv(src, parse_dates=["date"]).sort_values(["date", "ticker"])
    n_unk = int((df["session"] == "unknown").sum())

    # Replace the constant snapshot weight with the weight as of the trading day
    # BEFORE the announcement — strictly ex ante, and no longer a look-ahead
    # into the heavy-earnings slice.
    pit_path = cfg["paths"]["raw"] / "pit_weights.parquet"
    if pit_path.exists():
        pit = pd.read_parquet(pit_path)
        def _w(row):
            tk = row["ticker"]
            if tk not in pit.columns:
                return row["weight_pct"]
            s = pit[tk].dropna()
            s = s[s.index < row["date"]]
            return float(s.iloc[-1]) if len(s) else np.nan
        df["weight_pct_snapshot"] = df["weight_pct"]
        df["weight_pct"] = df.apply(_w, axis=1)
        # Pre-IPO / pre-history announcements have no cap: fall back to snapshot.
        miss = int(df["weight_pct"].isna().sum())
        df["weight_pct"] = df["weight_pct"].fillna(df["weight_pct_snapshot"])
        wsrc = (f"point-in-time from data/raw/pit_weights.parquet (weight as of the\n"
                f"# trading day before each announcement; {miss} pre-history row(s) "
                f"fell back to the snapshot)")
    else:
        wsrc = ("CONSTANT snapshot weight — run `make pit-weights` to replace this\n"
                "# with point-in-time weights (the snapshot is a look-ahead)")

    header = (
        "# Earnings announcements for the top-weight NDX names.\n"
        "# GENERATED by `make merge-earnings` from calendars/earnings_fetched.csv\n"
        "# (yfinance announcement timestamps, US/Eastern).\n"
        f"# weight_pct: {wsrc}.\n"
        "# Basket membership is still today's top names, so pre-2020 rows carry\n"
        "# survivorship; that affects the diagnostic phase, not the clean window.\n"
        "# session: bmo = impact same day, amc/unknown = impact next trading day.\n"
        f"# {n_unk} row(s) have session=unknown and are treated as amc.\n"
        "# Re-verify sessions against company IR pages before trusting an\n"
        "# event-sliced result; a wrong session shifts the vol day by one.\n"
    )
    out = cal / "earnings_top.csv"
    with open(out, "w") as f:
        f.write(header)
        df.to_csv(f, index=False, date_format="%Y-%m-%d")
    print(f"wrote {out} ({len(df)} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} .. {df['date'].max().date()})")
    return df


def fetch_fmp_earnings(cfg: dict, tickers: list[str]) -> pd.DataFrame:
    """Historical earnings announcement dates per ticker from FMP.

    Output columns: date, ticker, session (bmo/amc/unknown). VERIFY the session
    field against company IR pages for the mega-caps — free-source BMO/AMC flags
    are unreliable, and a wrong session shifts the vol-relevant day by one.
    """
    key = os.environ.get("FMP_API_KEY")
    if not key:
        sys.exit("FMP_API_KEY not set (free tier at financialmodelingprep.com)")
    rows = []
    for t in tickers:
        r = requests.get(FMP_EARNINGS.format(t=t), params={"apiKey": key}, timeout=30)
        r.raise_for_status()
        for rec in r.json():
            session = str(rec.get("time", "")).lower()
            rows.append(
                {
                    "date": rec.get("date"),
                    "ticker": t,
                    "session": session if session in ("bmo", "amc") else "unknown",
                }
            )
        time.sleep(0.5)
    df = pd.DataFrame(rows).dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"])
    out = config.ROOT / "calendars" / "earnings_fetched.csv"
    df.sort_values(["date", "ticker"]).to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows) — merge/verify into calendars/earnings_top.csv")
    return df


if __name__ == "__main__":
    cfg = config.load()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "free"
    if cmd == "free":
        fetch_daily_ohlc(cfg)
        fetch_vxn(cfg)
        fetch_short_dated_iv(cfg)
        fetch_implied_correlation(cfg)
        fetch_hourly_bars(cfg)
    elif cmd == "polygon":
        fetch_polygon_bars(cfg, sys.argv[2], sys.argv[3])
    elif cmd == "earnings":
        # No TICKERS= given: derive the name list and weights from the newest
        # saved QQQ holdings file.
        if len(sys.argv) > 2 and sys.argv[2]:
            fetch_yf_earnings(cfg, sys.argv[2].split(","))
        else:
            fetch_yf_earnings(cfg, load_index_weights(cfg))
    elif cmd == "pit-weights":
        w = load_index_weights(cfg)
        build_pit_weights(cfg, list(w), w)
    elif cmd == "merge-earnings":
        merge_earnings(cfg)
    elif cmd == "earnings-fmp":
        fetch_fmp_earnings(cfg, sys.argv[2].split(","))
    else:
        sys.exit(f"unknown command {cmd}")
