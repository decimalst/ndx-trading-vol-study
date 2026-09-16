"""Inspect provider-native equity volatility fields; never fit or score forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VOL_COLUMNS = {
    "qmle_trade": 2,
    "rv5_trade": 5,
    "rv15_trade": 6,
    "qmle_quote": 7,
    "rv5_quote": 10,
    "rv15_quote": 11,
}
AUX_COLUMNS = {"order_trade": 3, "ci_width_trade": 4, "order_quote": 8, "ci_width_quote": 9}


def _date(value: str) -> pd.Timestamp:
    if len(value) != 8 or not value.isdigit():
        raise ValueError("invalid provider date")
    return pd.to_datetime(value, format="%Y%m%d", errors="raise")


def parse_equity_response(
    raw: bytes, cutoff: str = "2025-10-20", symbol: str = "SPY", identifier: str = "84398"
) -> tuple[pd.DataFrame, dict]:
    """Bound before numerical parsing; preserve zeros, infinities and missing fields."""
    lines = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(lines) < 7:
        raise ValueError("response has no equity records")
    if lines[:2] != [symbol, identifier]:
        raise ValueError("source identity mismatch")
    if int(lines[3]) != len(lines) - 6:
        raise ValueError("declared row count mismatch")
    columns = {**VOL_COLUMNS, **AUX_COLUMNS}
    timestamps, records, all_dates = [], [], []
    end = pd.Timestamp(cutoff)
    for line in lines[6:]:
        fields = line.split()
        if len(fields) < 2:
            raise ValueError("missing date fields")
        date = _date(fields[1])
        all_dates.append(date)
        if date > end:
            continue
        if len(fields) != 12:
            raise ValueError("equity record must have 12 fields")
        records.append({name: float(fields[position]) for name, position in columns.items()})
        timestamps.append(date)
    dates = pd.DatetimeIndex(all_dates)
    if dates.has_duplicates:
        raise ValueError("duplicate source dates")
    if not dates.is_monotonic_increasing:
        raise ValueError("source dates are not ordered")
    if _date(lines[4]) != dates.min() or _date(lines[5]) != dates.max():
        raise ValueError("declared date range mismatch")
    table = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps, name="date"), columns=columns)
    metadata = {
        "symbol": symbol, "identifier": identifier, "description": lines[2],
        "source_rows": len(dates), "bounded_rows": len(table),
        "after_cutoff_rows": int((dates > end).sum()), "cutoff": cutoff,
        "source_first_date": str(dates.min().date()), "source_last_date": str(dates.max().date()),
        "numeric_post_cutoff_values_parsed": False,
        "source_sha256": hashlib.sha256(raw).hexdigest(), "source_bytes": len(raw),
    }
    return table, metadata


def _dates(index: pd.DatetimeIndex) -> list[str]:
    return [str(date.date()) for date in index]


def audit_table(table: pd.DataFrame, calendar: pd.DatetimeIndex, start: str, end: str) -> dict:
    """Count measurement validity on a full reference calendar without selecting a model."""
    if table.index.has_duplicates or calendar.has_duplicates:
        raise ValueError("duplicate audit calendar")
    calendar = calendar[(calendar >= start) & (calendar <= end)].sort_values()
    table = table.loc[(table.index >= start) & (table.index <= end)]
    fields = {}
    for name in VOL_COLUMNS:
        values = table[name].to_numpy(float)
        valid = np.isfinite(values) & (values > 0)
        positive = values[valid]
        fields[name] = {
            "rows": len(values), "finite_positive": int(valid.sum()),
            "zero": int((values == 0).sum()), "negative": int((values < 0).sum()),
            "nan": int(np.isnan(values).sum()), "infinite": int(np.isinf(values).sum()),
            "above_chart_cap": int((np.isfinite(values) & (values > 3)).sum()),
            "positive_quantiles": (
                dict(zip(["min", "p01", "median", "p99", "max"],
                         np.quantile(positive, [0, .01, .5, .99, 1]).tolist(), strict=True))
                if len(positive) else None
            ),
        }
    primary = table["qmle_trade"]
    valid = primary.gt(0) & np.isfinite(primary)
    regular_valid = valid.reindex(calendar, fill_value=False)
    aux = {}
    for name in AUX_COLUMNS:
        values = table[name].to_numpy(float)
        ok = np.isfinite(values) & (values >= 0)
        if name.startswith("order"):
            ok &= values == np.floor(values)
        aux[name] = {"valid": int(ok.sum()), "invalid": int((~ok).sum())}
    return {
        "start": start, "end": end, "source_rows": len(table), "reference_sessions": len(calendar),
        "missing_source_dates": _dates(calendar.difference(table.index)),
        "outside_reference_calendar": _dates(table.index.difference(calendar)),
        "invalid_primary_estimate_dates": _dates(table.index[~valid]),
        "fields": fields, "auxiliary_fields": aux,
        **{f"primary_complete_{window}_sessions": int(
            regular_valid.astype(int).rolling(window, min_periods=window).sum().eq(window).sum()
        ) for window in (1, 5, 22)},
    }


def read_calendar(path: Path) -> pd.DatetimeIndex:
    """Read only index labels using the parquet timestamp's comparison type."""
    calendar = pd.DatetimeIndex(pd.read_parquet(
        path, columns=[], filters=[("date", "<=", pd.Timestamp("2025-10-20"))]
    ).index).normalize()
    if calendar.tz is not None:
        calendar = calendar.tz_localize(None)
    return calendar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calendar-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table, source = parse_equity_response(args.source.read_bytes())
    calendar = read_calendar(args.calendar_parquet)
    periods = {"pre2016": ("2000-01-01", "2015-12-31"),
               "development": ("2016-01-04", "2019-12-31"),
               "evaluation": ("2020-01-02", "2025-10-20")}
    result = {
        "status": "MEASUREMENT_AUDIT_ONLY_NOT_A_FORECAST_OR_ADMISSION", "source": source,
        "units": "native annualized volatility; no square root, squaring, percent or annualization conversion",
        "primary_estimate_fixed_before_inspection": "qmle_trade",
        "calendar": "QQQ observed sessions as an equity calendar diagnostic, not a historical announced calendar",
        "periods": {name: audit_table(table, calendar, *dates) for name, dates in periods.items()},
        "yearly": {str(year): audit_table(table, calendar, f"{year}-01-01", min(f"{year}-12-31", "2025-10-20"))
                   for year in range(2010, 2026)},
        "forecast_fits": 0, "predictive_comparisons": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "bounded_rows": source["bounded_rows"],
                      "period_counts": {name: {"source_rows": item["source_rows"],
                                                "missing_dates": len(item["missing_source_dates"]),
                                                "invalid_qmle": len(item["invalid_primary_estimate_dates"])}
                                        for name, item in result["periods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
