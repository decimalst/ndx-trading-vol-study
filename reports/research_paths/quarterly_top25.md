# Point-in-time quarterly QQQ top-25 universe

This dataset replaces present-day name selection in the new earnings studies.
It is built from the 27 already-audited QQQ N-PORT filings in
`data/raw/qqq_nport_holdings.parquet`.

## Construction

1. Keep positive equity (`asset_category == EC`) positions.
2. Aggregate share classes by the six-character CUSIP issuer identifier before
   ranking. Alphabet therefore occupies one issuer rank, not two security ranks.
3. Rank issuers by disclosed `pct_value` inside each filing, with issuer ID as a
   deterministic tie-break, and retain exactly 25.
4. Make a snapshot usable only at its SEC `accepted_at` timestamp. Carry the
   latest accepted snapshot between filings; never substitute current members
   before history begins.

The resulting parquet has 675 rows, 27 snapshots from 2019-09-30 through
2026-03-31, and 44 distinct issuers. The top 25 represent 68.46% to 76.67% of
disclosed fund value. A quarter introduces 1.69 new issuers on average (maximum
five), so a fixed current-name basket is materially different from the actual
historical large-weight universe.

## The two concentration examples

| issuer | first disclosed top-25 observation | most recent observation | implication |
|---|---|---|---|
| NVIDIA | rank 17, 1.27% at 2019-09-30 (accepted 2019-11-29) | rank 1, 8.68% at 2026-03-31 (accepted 2026-05-28) | Its later concentration is not projected into earlier events. |
| Micron | rank 25, 0.80% at 2021-03-31; next appears rank 24, 0.97% at 2024-06-30 | rank 12, 2.15% at 2026-03-31 | It contributes only in quarters where the filing actually puts it in the top 25. |

For the research window ending 2025-10-17, the 2025-09-30 filing was not yet
public—it was accepted 2025-11-19—so neither its NVIDIA nor Micron rank can leak
into the study.

## Earnings coverage and limitations

The realized-event audit covers all 43 active mapped symbols. After
acceptance-time membership filtering, it retains 578 events across 39 issuers,
starting 2019-11-29. Sessions are inferred from Yahoo timestamps and are attached
only after forecasts are fixed.

This is more accurate and much less survivorship-biased than the current
13-name basket, but it is still a delayed quarterly QQQ fund proxy. It is not
exact daily Nasdaq-100 index weight history, and it does not fill pre-2019 with
the paused, incomplete annual-report reconstruction.

Artifacts:

- `data/research_paths/qqq_top25_quarterly.parquet`
- `data/research_paths/top25_earnings_raw.parquet`
- `data/research_paths/top25_earnings_events.parquet`
- `data/research_paths/top25_earnings_daily.parquet`
- `data/research_paths/top25_earnings_audit.json`
