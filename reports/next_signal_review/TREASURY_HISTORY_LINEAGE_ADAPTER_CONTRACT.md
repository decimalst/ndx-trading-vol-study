# Treasury calendar and lineage adapter contract

This new date-only helper implements the bounded source conventions identified in `TREASURY_HISTORY_LINEAGE_REVIEW.json`. It preserves all frozen helpers, initial failures and source bytes. No original documents, financial observations, market arrays or models are read by this helper or its synthetic tests.

```python
validate_lineage(
    *, dated: date, issue: date, maturity: date,
    original_dated: date | None, original_issue: date | None,
    pdf_original_issue: date | None, reopening: bool,
    allow_distinct_original_dates: bool = False,
) -> dict
```

Inputs must be actual `datetime.date` objects, with `None` allowed only for the three optional original-date descriptors. Datetime objects, strings and other coercible types are rejected. Both flags must be actual Booleans. The current source chronology must satisfy `dated <= issue < maturity`.

For a new issue, optional original descriptors may be absent or must match the respective current dated/issue dates. A supplied PDF original issue must equal the current issue date. The tenor anchor is the current dated date; the returned original issue and original dated dates resolve to the current issue and dated dates. The caller retains the raw optional descriptors separately, as in the existing identity metadata output.

A reopening requires explicit original dated and issue dates and matching PDF original issue. Chronology is `original_dated <= original_issue < issue`. The original dated date is always the tenor anchor. If it differs from the current dated date, it must precede that date and the explicit permission flag must be true. Root will set this flag only for a separately confirmed substitution result; the helper itself cannot authenticate a confirmation or document role. Ordinary reopenings with equal current/original dated dates remain supported without the flag.

Tenor is the stated maturity year minus the anchor year, restricted to 2, 3, 5, 7, 10 or 30. The dates must have exactly the same month/day, or share the same month and both be the actual last calendar day of that month. This admits February 28/29 end-of-month pairs in either direction without a general day tolerance, business-day calendar or date repair. Exact ordinary anniversaries remain allowed even when only one endpoint is month-end.

The result has exactly seven keys: `original_tenor_years`, `dated_date`, `issue_date`, `maturity_date`, `original_issue_date`, `original_dated_date`, and `reopening`. Date fields are ISO strings. No supplied source date is shifted or replaced; `original_dated_date` records the resolved tenor anchor. Pairing PDF/XML date labels, matching explicit offered tenor for a new issue, bounding the auction/release dates and authenticating a realized confirmation remain the caller's existing responsibilities.

Eleven generated tests were written before implementation. `TREASURY_HISTORY_LINEAGE_ADAPTER_RED.log` records the genuine missing-module import failure. The tests cover all six tenors, both leap-day directions, ordinary anniversaries, non-month-end/day/month contradictions, original-field agreement, current/original chronology, strict input types and explicit permission for distinct dates. The observed 2016-02-23 overlap is represented by two generated calls: a two-year February end-of-month announcement and a confirmed-substitution result with a five-year original anchor. An additional generated case makes the original reopening anchor itself cross a leap day.

All eleven tests passed on the first implementation and after formatting (0.001 seconds). The final result is saved in `TREASURY_HISTORY_LINEAGE_ADAPTER_GREEN.log`; scoped Ruff checks and formatting passed. No historical source application or model run was performed. Root owns separate integration and any later versioned historical application.
