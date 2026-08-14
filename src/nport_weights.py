"""Point-in-time QQQ holdings from public SEC Form N-PORT filings.

The report date is descriptive portfolio time, not information availability.
For a 16:00 America/New_York forecast origin, holdings are joined by the SEC
acceptance timestamp.  A filing accepted after that origin becomes available
only at the next supplied origin.
"""

from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable

import numpy as np
import pandas as pd
import requests

from . import config

SEC_CIK = "0001067839"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_ROOT = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_USER_AGENT = "ndx-vol-experiment/1.0 local-research"
HOLDINGS_COLUMNS = (
    "name",
    "title",
    "cusip",
    "isin",
    "balance",
    "units",
    "value_usd",
    "pct_value",
    "asset_category",
    "issuer_category",
    "payoff_profile",
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_text(parent: ET.Element, local_name: str) -> str | None:
    for elem in parent.iter():
        if _local_name(elem.tag) == local_name:
            text = (elem.text or "").strip()
            return text or None
    return None


def _number(text: str | None) -> float:
    if text is None or text == "":
        return np.nan
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"invalid numeric N-PORT value: {text!r}") from exc


def parse_nport_xml(content: bytes | str) -> tuple[dict, pd.DataFrame]:
    """Parse one N-PORT XML document without depending on its namespace URI."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"invalid N-PORT XML: {exc}") from exc

    report_text = _first_text(root, "repPdDate")
    if report_text is None:
        raise ValueError("N-PORT document has no portfolio report date")
    report_date = pd.to_datetime(report_text, errors="raise").normalize()
    meta = {
        "report_date": report_date,
        "registrant": _first_text(root, "regName"),
        "series_name": _first_text(root, "seriesName"),
        "series_id": _first_text(root, "seriesId"),
    }

    rows: list[dict] = []
    for holding in root.iter():
        if _local_name(holding.tag) != "invstOrSec":
            continue
        isin = None
        for elem in holding.iter():
            if _local_name(elem.tag) == "isin":
                isin = elem.attrib.get("value") or (elem.text or "").strip() or None
                break
        rows.append(
            {
                "name": _first_text(holding, "name"),
                "title": _first_text(holding, "title"),
                "cusip": _first_text(holding, "cusip"),
                "isin": isin,
                "balance": _number(_first_text(holding, "balance")),
                "units": _first_text(holding, "units"),
                "value_usd": _number(_first_text(holding, "valUSD")),
                "pct_value": _number(_first_text(holding, "pctVal")),
                "asset_category": _first_text(holding, "assetCat"),
                "issuer_category": _first_text(holding, "issuerCat"),
                "payoff_profile": _first_text(holding, "payoffProfile"),
            }
        )
    holdings = pd.DataFrame(rows, columns=HOLDINGS_COLUMNS)
    if holdings.empty:
        raise ValueError("N-PORT document has no investment records")
    return meta, holdings


def validate_snapshot(
    holdings: pd.DataFrame,
    *,
    min_total: float = 98.0,
    max_total: float = 102.0,
    min_holdings: int = 2,
) -> None:
    """Reject truncated or structurally implausible QQQ holdings snapshots."""
    missing = {"name", "value_usd", "pct_value"} - set(holdings.columns)
    if missing:
        raise ValueError(f"snapshot missing required columns: {sorted(missing)}")
    if len(holdings) < min_holdings:
        raise ValueError(f"snapshot has only {len(holdings)} holdings")
    pct = pd.to_numeric(holdings["pct_value"], errors="coerce")
    value = pd.to_numeric(holdings["value_usd"], errors="coerce")
    if pct.isna().any() or value.isna().any():
        raise ValueError("snapshot contains missing/non-numeric value fields")
    # N-PORT values can be negative for a disclosed derivative liability (QQQ
    # has a small E-mini Nasdaq-100 futures line in the 2026-03-31 snapshot).
    # Completeness is guarded by the net pctVal total, not a long-only rule.
    total = float(pct.sum())
    if not min_total <= total <= max_total:
        raise ValueError(
            f"snapshot percentage total {total:.6f} outside "
            f"[{min_total:.3f}, {max_total:.3f}]"
        )


def archive_url(cik: str, accession: str) -> str:
    """Return the primary XML URL; archive directory is the registrant CIK."""
    cik_dir = str(int(str(cik)))
    accession_dir = accession.replace("-", "")
    return f"{ARCHIVES_ROOT}/{cik_dir}/{accession_dir}/primary_doc.xml"


def assign_snapshot_asof(
    filings: pd.DataFrame,
    origins: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    origin_hour_et: int = 16,
) -> pd.DataFrame:
    """Assign the latest SEC-accepted snapshot known at each forecast origin."""
    required = {"accession", "report_date", "accepted_at"}
    missing = required - set(filings.columns)
    if missing:
        raise ValueError(f"filings missing required columns: {sorted(missing)}")
    left_dates = pd.DatetimeIndex(pd.to_datetime(list(origins))).tz_localize(None).normalize()
    left = pd.DataFrame({"origin": left_dates})
    left["origin_at"] = (
        left["origin"].dt.tz_localize("America/New_York")
        + pd.Timedelta(hours=origin_hour_et)
    ).dt.tz_convert("UTC")

    right = filings.copy()
    right["accepted_at"] = pd.to_datetime(right["accepted_at"], utc=True)
    right["report_date"] = pd.to_datetime(right["report_date"]).dt.tz_localize(None).dt.normalize()
    right = right.sort_values("accepted_at").reset_index(drop=True)
    if right["accepted_at"].duplicated().any():
        raise ValueError("filings contain duplicate SEC acceptance timestamps")

    out = pd.merge_asof(
        left.sort_values("origin_at"),
        right,
        left_on="origin_at",
        right_on="accepted_at",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("origin")
    out = out.drop(columns="origin_at").set_index("origin")
    out.index.name = "origin"
    return out


def summarize_equity_concentration(holdings: pd.DataFrame) -> pd.DataFrame:
    """Summarize disclosed positive equity weights for each N-PORT snapshot.

    Securities, not issuers, are the units (for example, the two Alphabet share
    classes remain separate, as they are in the fund holdings).  Derivatives and
    cash are excluded from the concentration denominator.
    """
    required = {
        "accession",
        "report_date",
        "accepted_at",
        "asset_category",
        "pct_value",
    }
    missing = required - set(holdings.columns)
    if missing:
        raise ValueError(
            f"holdings missing concentration columns: {sorted(missing)}"
        )
    rows = []
    keys = ["accession", "report_date", "accepted_at"]
    for key, frame in holdings.groupby(keys, sort=True, dropna=False):
        equity = frame.loc[
            (frame["asset_category"] == "EC")
            & (pd.to_numeric(frame["pct_value"], errors="coerce") > 0)
        ].copy()
        if equity.empty:
            raise ValueError(f"{key[0]}: no positive equity positions")
        weights = pd.to_numeric(equity["pct_value"], errors="raise")
        total = float(weights.sum())
        normalized = weights / total
        hhi = float(normalized.pow(2).sum())
        ordered = weights.sort_values(ascending=False)
        rows.append(
            {
                "accession": key[0],
                "report_date": key[1],
                "accepted_at": key[2],
                "equity_holdings": len(equity),
                "equity_pct_value": total,
                "top5_pct_value": float(ordered.head(5).sum()),
                "top10_pct_value": float(ordered.head(10).sum()),
                "hhi": hhi,
                "effective_holdings": 1.0 / hhi,
            }
        )
    return pd.DataFrame(rows).sort_values("report_date").reset_index(drop=True)


def _sec_session(user_agent: str | None = None) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent
            or os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT),
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
    )
    return session


def fetch_filing_index(
    *, cik: str = SEC_CIK, user_agent: str | None = None
) -> pd.DataFrame:
    """Read the QQQ submission index and return public NPORT-P filings."""
    session = _sec_session(user_agent)
    url = SUBMISSIONS_URL.format(cik=cik)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    recent = response.json()["filings"]["recent"]
    rows = []
    n = len(recent["form"])
    for i in range(n):
        if recent["form"][i] != "NPORT-P":
            continue
        accession = recent["accessionNumber"][i]
        rows.append(
            {
                "accession": accession,
                "filing_date": pd.Timestamp(recent["filingDate"][i]),
                "accepted_at": pd.Timestamp(recent["acceptanceDateTime"][i]),
                "report_date": pd.Timestamp(recent["reportDate"][i]),
                "source_url": archive_url(cik, accession),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("SEC submission index returned no public NPORT-P filings")
    out["accepted_at"] = pd.to_datetime(out["accepted_at"], utc=True)
    out = out.sort_values(["report_date", "accepted_at"]).reset_index(drop=True)
    if out["accession"].duplicated().any():
        raise RuntimeError("SEC submission index contains duplicate accessions")
    return out


def _download_xml(url: str, user_agent: str | None = None) -> bytes:
    # Archive host differs from data.sec.gov, so do not retain a fixed Host
    # header from the submission-index session.
    headers = {
        "User-Agent": user_agent
        or os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT),
        "Accept-Encoding": "gzip, deflate",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if b"<edgarSubmission" not in response.content[:1000] and "xml" not in content_type:
        raise RuntimeError(f"SEC archive did not return N-PORT XML: {url}")
    return response.content


def fetch_nport_history(
    *,
    cik: str = SEC_CIK,
    user_agent: str | None = None,
    pause_seconds: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download, validate, and persist every public QQQ NPORT-P snapshot."""
    cfg = config.load()
    filings = fetch_filing_index(cik=cik, user_agent=user_agent)
    holding_frames = []
    audit_rows = []
    for row in filings.itertuples(index=False):
        content = _download_xml(row.source_url, user_agent=user_agent)
        meta, holdings = parse_nport_xml(content)
        if meta["report_date"] != row.report_date:
            raise RuntimeError(
                f"{row.accession}: XML report date {meta['report_date'].date()} "
                f"!= submission index {row.report_date.date()}"
            )
        validate_snapshot(holdings, min_holdings=80)
        enriched = holdings.copy()
        enriched.insert(0, "accession", row.accession)
        enriched.insert(1, "report_date", row.report_date)
        enriched.insert(2, "accepted_at", row.accepted_at)
        enriched.insert(3, "filing_date", row.filing_date)
        enriched.insert(4, "source_url", row.source_url)
        holding_frames.append(enriched)
        audit_rows.append(
            {
                "accession": row.accession,
                "report_date": row.report_date,
                "accepted_at": row.accepted_at,
                "filing_date": row.filing_date,
                "source_url": row.source_url,
                "holding_count": len(holdings),
                "pct_value_sum": float(holdings["pct_value"].sum()),
                "value_usd_sum": float(holdings["value_usd"].sum()),
                "registrant": meta["registrant"],
                "series_name": meta["series_name"],
            }
        )
        time.sleep(pause_seconds)

    holdings_all = pd.concat(holding_frames, ignore_index=True)
    filings_audit = pd.DataFrame(audit_rows).sort_values("report_date").reset_index(drop=True)
    raw = cfg["paths"]["raw"]
    holdings_path = raw / "qqq_nport_holdings.parquet"
    filings_path = raw / "qqq_nport_filings.parquet"
    concentration_path = raw / "qqq_nport_concentration.parquet"
    concentration = summarize_equity_concentration(holdings_all)
    holdings_all.to_parquet(holdings_path, index=False)
    filings_audit.to_parquet(filings_path, index=False)
    concentration.to_parquet(concentration_path, index=False)
    _write_audit_report(filings_audit, concentration)
    print(
        f"wrote {holdings_path} ({len(holdings_all)} holding rows, "
        f"{len(filings_audit)} snapshots)"
    )
    print(f"wrote {filings_path}")
    print(f"wrote {concentration_path}")
    return filings_audit, holdings_all


def _write_audit_report(
    filings: pd.DataFrame, concentration: pd.DataFrame
) -> None:
    report_path = config.ROOT / "reports" / "qqq_nport_audit.md"
    lag_days = (filings["accepted_at"].dt.tz_convert(None).dt.normalize()
                - filings["report_date"]).dt.days
    lines = [
        "# Public QQQ N-PORT history audit",
        "",
        "Generated from SEC CIK 1067839. This is fund-holdings history, not "
        "licensed Nasdaq index-weight history.",
        "",
        f"- Snapshots: {len(filings)} ({filings['report_date'].min().date()} "
        f"through {filings['report_date'].max().date()}).",
        f"- Disclosure lag: {int(lag_days.min())}–{int(lag_days.max())} calendar "
        "days from report date to SEC acceptance.",
        f"- Holdings per snapshot: {int(filings['holding_count'].min())}–"
        f"{int(filings['holding_count'].max())}.",
        f"- Investment percentage totals: {filings['pct_value_sum'].min():.3f}%–"
        f"{filings['pct_value_sum'].max():.3f}%.",
        "- Point-in-time rule: join on `accepted_at` against the 16:00 ET "
        "forecast timestamp. Never join on `report_date`.",
        f"- Disclosed positive-equity top-10 weight: "
        f"{concentration.iloc[0]['top10_pct_value']:.2f}% in the first snapshot; "
        f"{concentration.iloc[-1]['top10_pct_value']:.2f}% in the latest.",
        f"- Security-level HHI: {concentration.iloc[0]['hhi']:.4f} in the first "
        f"snapshot; {concentration.iloc[-1]['hhi']:.4f} in the latest. These "
        "are disclosed QQQ portfolio values, available only after SEC acceptance.",
        "",
        "| report date | SEC accepted (UTC) | holdings | pct total | accession |",
        "|---|---|---:|---:|---|",
    ]
    for row in filings.itertuples(index=False):
        lines.append(
            f"| {row.report_date.date()} | {row.accepted_at.isoformat()} | "
            f"{row.holding_count} | {row.pct_value_sum:.3f}% | "
            f"[{row.accession}]({row.source_url}) |"
        )
    lines.extend(
        [
            "",
            "The earliest public snapshot is accepted late in 2019, so this "
            "source cannot support the 2016–2019 portion of discovery. Its "
            "quarterly, roughly two-month-delayed values are suitable only as "
            "a slow robustness/regime feature in a future protocol.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {report_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch"], nargs="?", default="fetch")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch_nport_history()


if __name__ == "__main__":
    main()
