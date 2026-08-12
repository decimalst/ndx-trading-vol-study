"""Public, point-in-time-safe QQQ holdings and Nasdaq-100 membership history.

This module joins two different kinds of evidence without pretending they are
interchangeable:

* audited QQQ annual-report holdings (N-30B-2) through 2018 and structured
  quarterly N-PORT holdings thereafter provide disclosed fund weights; and
* Nasdaq's public historical XLSX export provides official membership only.

The SEC report date describes the portfolio.  ``accepted_at`` is the earliest
time a simulated forecaster may use it.  The unauthenticated Nasdaq workbook
does not contain weights and this module never manufactures a weight column for
it.
"""

from __future__ import annotations

import argparse
import html as html_module
import io
import os
import re
import time
import zipfile
from pathlib import PurePosixPath
from typing import Iterable
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests
from lxml import html as lxml_html

from . import config
from .nport_weights import (
    DEFAULT_USER_AGENT,
    HOLDINGS_COLUMNS,
    SEC_CIK,
    summarize_equity_concentration,
    validate_snapshot,
)


SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSIONS_ROOT = "https://data.sec.gov/submissions"
SEC_ARCHIVES_ROOT = "https://www.sec.gov/Archives/edgar/data"
NASDAQ_EXPORT_ROOT = "https://indexes.nasdaqomx.com/Index/ExportWeightings/NDX"
LEGACY_FORM = "N-30B-2"
STRUCTURED_FORM = "NPORT-P"
COMBINED_COLUMNS = (
    "accession",
    "report_date",
    "accepted_at",
    "filing_date",
    "source_url",
    "source_form",
    *HOLDINGS_COLUMNS,
)

_NUMBER = re.compile(r"^\(?\$?\s*([0-9][0-9,]*(?:\.\d+)?)\s*\)?$")
_REPORT_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),\s+(\d{4})\b",
    re.IGNORECASE,
)
_PLAIN_HOLDING = re.compile(
    r"^(?P<name>.+?)\s+\.{2,}\s+(?P<shares>[0-9][0-9,]*)\s+"
    r"\$?\s*(?P<value>[0-9][0-9,]*)\s*$"
)
_PLAIN_HOLDING_SPACED = re.compile(
    r"^(?P<name>.*?\S)\s{2,}(?P<shares>[0-9][0-9,]*)\s+"
    r"\$?\s*(?P<value>[0-9][0-9,]*)\s*$"
)


def _source_text(content: bytes | str) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return content


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html_module.unescape(value).replace("\xa0", " ")).strip()


def _numeric_cell(value: str) -> float | None:
    match = _NUMBER.fullmatch(_clean_text(value))
    if not match:
        return None
    number = float(match.group(1).replace(",", ""))
    return -number if value.strip().startswith("(") else number


def _clean_security_name(value: str) -> str:
    value = _clean_text(value)
    value = re.sub(r"(?:\*+|\(\s*[a-z0-9]+\s*\)|\[[a-z0-9]+\])\s*$", "", value, flags=re.I)
    return value.strip()


def _extract_report_date(text: str) -> pd.Timestamp:
    searchable = (
        _clean_text(" ".join(lxml_html.fromstring(text).itertext()))
        if "<" in text and ">" in text
        else _clean_text(text)
    )
    matches = list(_REPORT_DATE.finditer(searchable))
    if not matches:
        raise ValueError("legacy annual report has no report date")
    september = [match for match in matches if match.group(1).lower() == "september"
                 and match.group(2) == "30"]
    selected = september[0] if september else matches[0]
    return pd.to_datetime(selected.group(0), errors="raise").normalize()


def _holding_row_from_cells(cells: list[str]) -> dict | None:
    clean = [_clean_text(cell) for cell in cells if _clean_text(cell) not in {"", "$"}]
    numeric = [(index, _numeric_cell(value)) for index, value in enumerate(clean)]
    numeric = [(index, value) for index, value in numeric if value is not None]
    if len(numeric) < 2:
        return None
    first_index, shares = numeric[0]
    last_index, value = numeric[-1]
    if first_index == last_index or shares is None or value is None or shares <= 0 or value <= 0:
        return None
    name_cells = [
        cell
        for index, cell in enumerate(clean)
        if index not in {item[0] for item in numeric}
        and re.search(r"[A-Za-z]", cell)
    ]
    if not name_cells:
        return None
    name = _clean_security_name(" ".join(name_cells))
    lowered = name.lower()
    if any(
        phrase in lowered
        for phrase in (
            "common stock",
            "common shares",
            "number of shares",
            "total investments",
            "net assets",
            "statement of",
        )
    ):
        return None
    return {"name": name, "balance": float(shares), "value_usd": float(value)}


def _parse_html_schedule(text: str) -> tuple[list[dict], float | None]:
    try:
        root = lxml_html.fromstring(text)
    except (ValueError, lxml_html.ParserError):
        return [], None
    started = False
    holdings: list[dict] = []
    disclosed_total = None
    for row in root.xpath("//tr"):
        cells = [_clean_text(cell.text_content()) for cell in row.xpath("./th|./td")]
        joined = " ".join(cells).lower()
        if not started:
            if "share" in joined and "value" in joined and (
                "common stock" in joined or "number of shares" in joined
            ):
                started = True
            continue
        if "total investments" in joined:
            values = [_numeric_cell(cell) for cell in cells]
            values = [value for value in values if value is not None]
            disclosed_total = float(values[-1]) if values else None
            break
        parsed = _holding_row_from_cells(cells)
        if parsed is not None:
            holdings.append(parsed)
    return holdings, disclosed_total


def _parse_plain_schedule(text: str) -> tuple[list[dict], float | None]:
    upper = text.upper()
    total_at = upper.find("TOTAL INVESTMENTS")
    if total_at < 0:
        return [], None
    starts = [
        match.start()
        for match in re.finditer("SCHEDULE OF INVESTMENTS", upper[:total_at])
        if "COMMON STOCK" in upper[match.start() : match.start() + 1500]
        and "SHARES" in upper[match.start() : match.start() + 1500]
        and "VALUE" in upper[match.start() : match.start() + 1500]
    ]
    if not starts:
        return [], None
    # Page headers repeat "Schedule of Investments (Continued)".  Begin at the
    # first header with the actual column labels so earlier pages are retained.
    start_at = starts[0]
    section = text[start_at:total_at]
    holdings = []
    for raw_line in section.splitlines():
        line = _clean_text(raw_line)
        match = _PLAIN_HOLDING.match(line) or _PLAIN_HOLDING_SPACED.match(raw_line.strip())
        if not match:
            continue
        holdings.append(
            {
                "name": _clean_security_name(match.group("name")),
                "balance": float(match.group("shares").replace(",", "")),
                "value_usd": float(match.group("value").replace(",", "")),
            }
        )
    total_line = text[total_at:text.find("\n", total_at) if "\n" in text[total_at:] else None]
    dollar_values = re.findall(r"\$\s*([0-9][0-9,]*)", total_line)
    if not dollar_values:
        numeric_values = re.findall(r"\b([0-9][0-9,]{3,})\b", total_line)
        dollar_values = numeric_values
    disclosed_total = float(dollar_values[-1].replace(",", "")) if dollar_values else None
    return holdings, disclosed_total


def parse_legacy_annual_report(content: bytes | str) -> tuple[dict, pd.DataFrame]:
    """Parse an audited QQQ N-30B-2 Schedule of Investments.

    Both name-first and shares-first HTML tables are supported, along with the
    pre-HTML fixed-width layout used by the 2004 filing.  Parsed positions must
    reconcile to the filing's disclosed total before weights are calculated.
    """
    text = _source_text(content)
    report_date = _extract_report_date(text)
    rows, disclosed_total = _parse_html_schedule(text)
    layout = "html"
    if not rows:
        rows, disclosed_total = _parse_plain_schedule(text)
        layout = "plain_text"
    if not rows:
        raise ValueError("legacy annual report has no investment rows")
    if disclosed_total is None or disclosed_total <= 0:
        raise ValueError("legacy annual report has no disclosed total investments")

    holdings_total = float(sum(row["value_usd"] for row in rows))
    tolerance = max(5.0, disclosed_total * 1e-8)
    if abs(holdings_total - disclosed_total) > tolerance:
        raise ValueError(
            "legacy holdings do not reconcile to disclosed total: "
            f"parsed={holdings_total:.2f}, disclosed={disclosed_total:.2f}"
        )
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "name": row["name"],
                "title": row["name"],
                "cusip": None,
                "isin": None,
                "balance": row["balance"],
                "units": "NS",
                "value_usd": row["value_usd"],
                "pct_value": 100.0 * row["value_usd"] / disclosed_total,
                "asset_category": "EC",
                "issuer_category": None,
                "payoff_profile": "Long",
            }
        )
    holdings = pd.DataFrame(normalized_rows, columns=HOLDINGS_COLUMNS)
    meta = {
        "report_date": report_date,
        "disclosed_total_usd": float(disclosed_total),
        "parsed_total_usd": holdings_total,
        "layout": layout,
    }
    return meta, holdings


def membership_export_url(trade_date: pd.Timestamp | str) -> str:
    date = pd.Timestamp(trade_date).tz_localize(None).normalize()
    query = urlencode(
        {"tradeDate": f"{date.date().isoformat()}T00:00:00.000", "timeOfDay": "EOD"}
    )
    return f"{NASDAQ_EXPORT_ROOT}?{query}"


def _xlsx_relationships(archive: zipfile.ZipFile) -> dict[str, str]:
    root = lxml_html.etree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    return {element.get("Id"): element.get("Target") for element in root}


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = lxml_html.etree.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(element.itertext()) for element in root]


def _xlsx_cell_text(cell, shared_strings: list[str]) -> str:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{namespace}is")
        return "" if inline is None else "".join(inline.itertext())
    value = cell.find(f"{namespace}v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text


def parse_nasdaq_membership_xlsx(content: bytes) -> pd.DataFrame:
    """Read names and symbols from Nasdaq's unauthenticated XLSX export."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError("Nasdaq membership response is not an XLSX workbook") from exc
    with archive:
        workbook = lxml_html.etree.fromstring(archive.read("xl/workbook.xml"))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        relationships = _xlsx_relationships(archive)
        selected = None
        for sheet in workbook.xpath("//m:sheet", namespaces=ns):
            if (sheet.get("name") or "").strip().lower() == "weightings":
                relation_id = sheet.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                )
                selected = relationships.get(relation_id)
                break
        if selected is None:
            raise ValueError("Nasdaq workbook has no Weightings sheet")
        path = PurePosixPath("xl") / selected.lstrip("/")
        path = PurePosixPath(*[part for part in path.parts if part not in {"."}])
        sheet_root = lxml_html.etree.fromstring(archive.read(str(path)))
        shared_strings = _xlsx_shared_strings(archive)
        rows = []
        for row in sheet_root.xpath("//m:sheetData/m:row", namespaces=ns):
            values: dict[int, str] = {}
            for cell in row.xpath("./m:c", namespaces=ns):
                reference = cell.get("r", "A1")
                column_letters = re.match(r"[A-Z]+", reference)
                if column_letters is None:
                    continue
                column = 0
                for letter in column_letters.group(0):
                    column = column * 26 + ord(letter) - 64
                values[column - 1] = _clean_text(_xlsx_cell_text(cell, shared_strings))
            width = max(values, default=-1) + 1
            rows.append([values.get(index, "") for index in range(width)])

    header_index = None
    name_column = None
    symbol_column = None
    for index, row in enumerate(rows):
        lowered = [value.lower() for value in row]
        if "company name" in lowered and "security symbol" in lowered:
            header_index = index
            name_column = lowered.index("company name")
            symbol_column = lowered.index("security symbol")
            break
    if header_index is None or name_column is None or symbol_column is None:
        raise ValueError("Nasdaq workbook is missing Company Name/Security Symbol headers")
    records = []
    for row in rows[header_index + 1 :]:
        name = row[name_column] if name_column < len(row) else ""
        symbol = row[symbol_column] if symbol_column < len(row) else ""
        if not name and not symbol:
            continue
        records.append({"company_name": name or None, "symbol": symbol})
    frame = pd.DataFrame(records, columns=["company_name", "symbol"])
    if frame.empty:
        raise ValueError("Nasdaq workbook has no membership records")
    return frame


def validate_membership_snapshot(
    membership: pd.DataFrame, *, min_members: int = 90, max_members: int = 120
) -> None:
    missing = {"company_name", "symbol"} - set(membership.columns)
    if missing:
        raise ValueError(f"membership missing required columns: {sorted(missing)}")
    if not min_members <= len(membership) <= max_members:
        raise ValueError(
            f"membership has {len(membership)} securities; expected {min_members}-{max_members}"
        )
    symbols = membership["symbol"].fillna("").astype(str).str.strip()
    if (symbols == "").any():
        raise ValueError("membership contains blank security symbols")
    if symbols.duplicated().any():
        duplicates = sorted(symbols[symbols.duplicated(keep=False)].unique())
        raise ValueError(f"membership contains duplicate security symbols: {duplicates}")


def select_membership_trade_dates(
    report_dates: Iterable[pd.Timestamp], trading_dates: Iterable[pd.Timestamp]
) -> pd.DatetimeIndex:
    reports = pd.DatetimeIndex(pd.to_datetime(list(report_dates))).tz_localize(None).normalize()
    sessions = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).tz_localize(None).normalize()
    sessions = sessions.sort_values().unique()
    if len(sessions) == 0:
        raise ValueError("trading_dates is empty")
    positions = sessions.searchsorted(reports, side="right") - 1
    if (positions < 0).any():
        raise ValueError("a report date precedes the first observed trading session")
    return pd.DatetimeIndex(sessions[positions])


def combine_disclosed_holdings(
    legacy: pd.DataFrame, nport: pd.DataFrame
) -> pd.DataFrame:
    """Combine SEC histories, preferring structured N-PORT on overlaps."""
    legacy = legacy.copy()
    nport = nport.copy()
    if "source_form" not in legacy:
        legacy["source_form"] = LEGACY_FORM
    if "source_form" not in nport:
        nport["source_form"] = STRUCTURED_FORM
    legacy["report_date"] = pd.to_datetime(legacy["report_date"]).dt.normalize()
    nport["report_date"] = pd.to_datetime(nport["report_date"]).dt.normalize()
    overlap = set(nport["report_date"].dropna())
    legacy = legacy.loc[~legacy["report_date"].isin(overlap)]
    combined = pd.concat([legacy, nport], ignore_index=True, sort=False)
    for column in COMBINED_COLUMNS:
        if column not in combined:
            combined[column] = np.nan
    combined["accepted_at"] = pd.to_datetime(combined["accepted_at"], utc=True)
    return combined.loc[:, COMBINED_COLUMNS].sort_values(
        ["report_date", "source_form", "name"], kind="stable"
    ).reset_index(drop=True)


def _sec_headers(user_agent: str | None = None) -> dict[str, str]:
    return {
        "User-Agent": user_agent or os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT),
        "Accept-Encoding": "gzip, deflate",
    }


def _filing_rows(payload: dict) -> list[dict]:
    arrays = payload.get("filings", {}).get("recent", payload)
    forms = arrays.get("form", [])
    return [
        {key: values[index] for key, values in arrays.items() if isinstance(values, list)}
        for index in range(len(forms))
    ]


def fetch_legacy_filing_index(
    *, cik: str = SEC_CIK, user_agent: str | None = None
) -> pd.DataFrame:
    """Return audited QQQ annual reports covering 2004 through 2018."""
    headers = _sec_headers(user_agent)
    main_url = SEC_SUBMISSIONS_URL.format(cik=cik)
    response = requests.get(main_url, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = _filing_rows(payload)
    for older in payload.get("filings", {}).get("files", []):
        url = f"{SEC_SUBMISSIONS_ROOT}/{quote(older['name'])}"
        older_response = requests.get(url, headers=headers, timeout=30)
        older_response.raise_for_status()
        rows.extend(_filing_rows(older_response.json()))

    selected = []
    cik_dir = str(int(cik))
    for row in rows:
        if row.get("form") != LEGACY_FORM or not row.get("reportDate"):
            continue
        report_date = pd.Timestamp(row["reportDate"]).normalize()
        if not (2004 <= report_date.year <= 2018 and report_date.month == 9
                and report_date.day == 30):
            continue
        accession = row["accessionNumber"]
        document = row.get("primaryDocument")
        if not document:
            continue
        selected.append(
            {
                "accession": accession,
                "report_date": report_date,
                "accepted_at": pd.to_datetime(row["acceptanceDateTime"], utc=True),
                "filing_date": pd.Timestamp(row["filingDate"]).normalize(),
                "source_url": (
                    f"{SEC_ARCHIVES_ROOT}/{cik_dir}/{accession.replace('-', '')}/"
                    f"{quote(document)}"
                ),
            }
        )
    frame = pd.DataFrame(selected)
    if frame.empty:
        raise RuntimeError("SEC submissions returned no 2004-2018 QQQ annual reports")
    frame = frame.sort_values(["report_date", "accepted_at"]).reset_index(drop=True)
    if frame["report_date"].duplicated().any():
        duplicates = frame.loc[frame["report_date"].duplicated(False), "report_date"]
        raise RuntimeError(f"multiple annual reports for report dates: {duplicates.tolist()}")
    return frame


def _download(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    max_attempts: int = 4,
    retry_seconds: float = 1.0,
) -> bytes:
    """Download a public source with bounded backoff for transient throttling."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    response = None
    for attempt in range(max_attempts):
        response = requests.get(url, headers=headers, timeout=45)
        if response.status_code not in {403, 429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response.content
        if attempt + 1 < max_attempts:
            time.sleep(retry_seconds * (2**attempt))
    assert response is not None
    response.raise_for_status()
    raise RuntimeError(f"unreachable download state: {url}")


def fetch_legacy_history(
    *, cik: str = SEC_CIK, user_agent: str | None = None, pause_seconds: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filings = fetch_legacy_filing_index(cik=cik, user_agent=user_agent)
    holdings_frames = []
    audit_rows = []
    for row in filings.itertuples(index=False):
        content = _download(row.source_url, headers=_sec_headers(user_agent))
        try:
            meta, holdings = parse_legacy_annual_report(content)
        except ValueError as exc:
            raise ValueError(
                f"{row.accession} ({row.report_date.date()}): {exc}"
            ) from exc
        if meta["report_date"] != row.report_date:
            raise RuntimeError(
                f"{row.accession}: parsed report date {meta['report_date'].date()} "
                f"!= SEC index {row.report_date.date()}"
            )
        validate_snapshot(holdings, min_total=99.99, max_total=100.01, min_holdings=80)
        enriched = holdings.copy()
        enriched.insert(0, "accession", row.accession)
        enriched.insert(1, "report_date", row.report_date)
        enriched.insert(2, "accepted_at", row.accepted_at)
        enriched.insert(3, "filing_date", row.filing_date)
        enriched.insert(4, "source_url", row.source_url)
        enriched.insert(5, "source_form", LEGACY_FORM)
        holdings_frames.append(enriched)
        audit_rows.append(
            {
                **row._asdict(),
                "source_form": LEGACY_FORM,
                "holding_count": len(holdings),
                "pct_value_sum": float(holdings["pct_value"].sum()),
                "value_usd_sum": float(holdings["value_usd"].sum()),
                "disclosed_total_usd": meta["disclosed_total_usd"],
                "layout": meta["layout"],
            }
        )
        time.sleep(pause_seconds)
    return pd.DataFrame(audit_rows), pd.concat(holdings_frames, ignore_index=True)


def fetch_membership_snapshots(
    trade_dates: Iterable[pd.Timestamp], *, pause_seconds: float = 0.12
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    audit_rows = []
    for trade_date in pd.DatetimeIndex(pd.to_datetime(list(trade_dates))).unique().sort_values():
        url = membership_export_url(trade_date)
        content = _download(url, headers={"User-Agent": DEFAULT_USER_AGENT})
        membership = parse_nasdaq_membership_xlsx(content)
        validate_membership_snapshot(membership)
        enriched = membership.copy()
        enriched.insert(0, "membership_date", pd.Timestamp(trade_date).normalize())
        enriched.insert(1, "source_url", url)
        frames.append(enriched)
        valid_names = enriched["company_name"].fillna("").astype(str).str.contains(
            r"[A-Za-z]", regex=True
        )
        audit_rows.append(
            {
                "membership_date": pd.Timestamp(trade_date).normalize(),
                "source_url": url,
                "member_count": len(enriched),
                "valid_name_fraction": float(valid_names.mean()),
            }
        )
        time.sleep(pause_seconds)
    return pd.DataFrame(audit_rows), pd.concat(frames, ignore_index=True)


def _filing_audit_from_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "accession",
        "report_date",
        "accepted_at",
        "filing_date",
        "source_url",
        "source_form",
    ]
    return (
        holdings.groupby(keys, as_index=False, dropna=False)
        .agg(
            holding_count=("name", "size"),
            pct_value_sum=("pct_value", "sum"),
            value_usd_sum=("value_usd", "sum"),
        )
        .sort_values("report_date")
        .reset_index(drop=True)
    )


def _write_combined_audit(
    filings: pd.DataFrame,
    concentration: pd.DataFrame,
    membership_audit: pd.DataFrame,
) -> None:
    report_path = config.ROOT / "reports" / "historical_weights_audit.md"
    filing_lag = (
        filings["accepted_at"].dt.tz_convert(None).dt.normalize() - filings["report_date"]
    ).dt.days
    sources = filings.groupby("source_form").size().to_dict()
    lines = [
        "# Public QQQ holdings and Nasdaq membership audit",
        "",
        "This dataset is a delayed QQQ fund-weight proxy plus official Nasdaq-100 "
        "membership. It is not licensed daily NDX weight history.",
        "",
        f"- SEC holding snapshots: {len(filings)} "
        f"({filings['report_date'].min().date()} through {filings['report_date'].max().date()}).",
        f"- Source mix: {sources.get(LEGACY_FORM, 0)} audited annual reports; "
        f"{sources.get(STRUCTURED_FORM, 0)} structured quarterly N-PORT filings.",
        f"- SEC disclosure lag: {int(filing_lag.min())}-{int(filing_lag.max())} calendar days.",
        f"- Official Nasdaq membership snapshots: {len(membership_audit)}; "
        f"{int(membership_audit['member_count'].min())}-"
        f"{int(membership_audit['member_count'].max())} securities.",
        "- Point-in-time rule: weights become usable at `accepted_at`, never at "
        "`report_date`. Nasdaq membership files contain symbols/names but no "
        "unauthenticated weight column.",
        "",
        "| report date | accepted UTC | form | holdings | pct total | top 10 | HHI | accession |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    merged = filings.merge(
        concentration[
            ["accession", "top10_pct_value", "hhi"]
        ],
        on="accession",
        how="left",
        validate="one_to_one",
    )
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {row.report_date.date()} | {row.accepted_at.isoformat()} | "
            f"{row.source_form} | {row.holding_count} | {row.pct_value_sum:.3f}% | "
            f"{row.top10_pct_value:.2f}% | {row.hhi:.4f} | "
            f"[{row.accession}]({row.source_url}) |"
        )
    lines.extend(
        [
            "",
            "The annual-to-quarterly frequency change is explicit. No value is "
            "interpolated in these raw files. Any future daily drift series must use "
            "only returns observed after the latest available SEC acceptance time.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {report_path}")


def fetch_combined_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download, reconcile, and persist the public combined history."""
    cfg = config.load()
    raw = cfg["paths"]["raw"]
    legacy_filings, legacy_holdings = fetch_legacy_history()
    nport_holdings = pd.read_parquet(raw / "qqq_nport_holdings.parquet")
    combined = combine_disclosed_holdings(legacy_holdings, nport_holdings)
    filings = _filing_audit_from_holdings(combined)
    concentration = summarize_equity_concentration(combined)

    daily = pd.read_parquet(raw / "daily_ohlc.parquet")
    trading_dates = daily.index if isinstance(daily.index, pd.DatetimeIndex) else daily["date"]
    mapped_dates = select_membership_trade_dates(filings["report_date"], trading_dates)
    membership_audit, membership = fetch_membership_snapshots(mapped_dates.unique())

    legacy_filings.to_parquet(raw / "qqq_legacy_filings.parquet", index=False)
    legacy_holdings.to_parquet(raw / "qqq_legacy_holdings.parquet", index=False)
    combined.to_parquet(raw / "qqq_disclosed_holdings.parquet", index=False)
    filings.to_parquet(raw / "qqq_disclosed_filings.parquet", index=False)
    concentration.to_parquet(raw / "qqq_disclosed_concentration.parquet", index=False)
    membership.to_parquet(raw / "ndx_membership_snapshots.parquet", index=False)
    membership_audit.to_parquet(raw / "ndx_membership_audit.parquet", index=False)
    _write_combined_audit(filings, concentration, membership_audit)
    print(
        f"wrote combined public history: {len(combined)} holdings, "
        f"{len(filings)} SEC snapshots, {len(membership_audit)} membership snapshots"
    )
    return filings, combined, membership


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch"], nargs="?", default="fetch")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch_combined_history()


if __name__ == "__main__":
    main()
