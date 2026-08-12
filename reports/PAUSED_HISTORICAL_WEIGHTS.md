# Paused public historical-weight reconstruction

Paused 2026-08-12 at the user's direction. This work is preserved as source and
tests, but it is not a completed dataset and is not connected to any feature,
forecast, signal-selection, or carry code.

## Intended construction

The experimental pipeline in `src/historical_weights.py` keeps two public data
types separate:

1. QQQ's audited annual N-30B-2 schedules supply security-level shares, values,
   and value-derived fund weights for September 30 snapshots from 2004 through
   2018.
2. The existing structured N-PORT history supplies quarterly QQQ snapshots from
   2019 onward.
3. Nasdaq's unauthenticated historical XLSX export supplies official
   point-in-time NDX names and symbols. The inspected public workbooks do **not**
   contain a weight column, so the parser never creates one.

The SEC `report_date` describes the portfolio. A simulated forecaster may use a
snapshot only at its EDGAR `accepted_at` timestamp. Structured N-PORT supersedes
an annual filing if both report the same date. No daily interpolation or return
drift has been implemented.

## Test-first work completed

`tests/test_historical_weights.py` was written before the acquisition run. Its
10 contracts cover:

- name-first HTML, shares-first HTML, and fixed-width SEC schedules;
- repeated `Schedule of Investments (continued)` page headers;
- exact reconciliation of parsed values to disclosed total investments;
- rejection of truncated totals and duplicate Nasdaq symbols;
- XLSX parsing without inferring unavailable weights;
- explicit EOD Nasdaq URLs and weekend report-date rollback;
- N-PORT precedence on a same-date overlap; and
- bounded retry on transient source throttling.

The pre-run gate also reran the nine existing N-PORT contracts, the 14
signal-safety contracts, and the synthetic end-to-end smoke test. All passed.

## Acquisition attempts and exact stop

The first run stopped on the 2004 fixed-width schedule because only its final
page was initially parsed. The parser was corrected to retain repeated pages,
and the disclosed total then reconciled. The second layout issue was the 2005
filing's space-delimited rows without dot leaders; a real-layout regression was
added and the parser again passed the exact total check.

The next run reached the 2006 report and repeatedly received HTTP 403 from the
SEC archive URL despite bounded exponential backoff and a 0.5-second request
cadence. Work stopped there rather than weakening validation or bypassing SEC
access controls.

No combined output was promoted. In particular, these files should not be
assumed to exist or be valid until a future full run succeeds and passes a
post-fetch audit:

- `data/raw/qqq_legacy_holdings.parquet`
- `data/raw/qqq_disclosed_holdings.parquet`
- `data/raw/qqq_disclosed_concentration.parquet`
- `data/raw/ndx_membership_snapshots.parquet`

## Safe restart conditions

If resumed, use a compliant SEC identity in `SEC_USER_AGENT`, retain the total
reconciliation and acceptance-time rules unchanged, and run
`make fetch-historical-weights`. A successful fetch still needs source coverage,
name/symbol matching, snapshot totals, and membership-difference audits before
any feature can be registered. The already-spent 2022-2025 signal confirmation
window must not be reused to promote a concentration feature.
