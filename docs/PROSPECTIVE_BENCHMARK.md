# Prospective forecast and spread-risk collection

Prepared 2026-09-13. The additive ledger is implemented and synthetic-tested;
collection, live model issuance and a spread execution engine are **not running**.
No market data after the historical study ceiling was opened for this work. No
subscription, credential setup, scheduled task or order was created.

The next useful evidence is a sequence of forecasts saved before their outcomes,
paired with actual quotes for the options whose risk is being assessed. Start
with one instrument family and one frozen put-spread, call-spread or iron-condor
selection rule. Daily underlying returns can test terminal liability under
explicit assumptions; they cannot establish collected premium, early-exit
performance or net spread profit.

## What the code does

`src/prospective_benchmark.py` provides `ProspectiveLedger(directory)` and a CLI.
It writes `events.jsonl` plus content-addressed `blobs/<sha256>`. Each event has
schema version, sequence, local UTC receipt time, previous-event hash and its own
SHA256. A process lock serializes writers; each append is flushed to disk.
Source and model bytes are retained exactly. Identical imports are idempotent;
changed source bytes or normalized observations append a revision linked to the
previous receipt. Existing issued forecasts cannot be replaced.

| API | Contract |
| --- | --- |
| `ingest_source(source_key, raw, metadata)` | Record original bytes, origin/availability metadata and local first receipt; retain source revisions. |
| `freeze_model(model_id, artifact, protocol)` | Bind model artifact bytes to code/plan hashes, training cutoff, input schema and target definition. Returns the model freeze identifier. |
| `schedule_target(target, freeze_ids)` | Declare the complete paired cohort before the target starts. Missing forecasts remain visible. |
| `issue_forecast(freeze_id, target, prediction, input_receipts)` | Record the forecast and exact prior input receipts. A receipt at or after target start is retained but never scored as prospective. |
| `record_observation(target, value, source_receipts)` | Accept a normalized outcome only after target end, with matching instrument/provider/feed/adjustment evidence. |
| `mature_scores(as_of=None)` | Append normalized log score/PIT for normal or Student-t return densities, or QLIKE for a positive variance mean. Idempotent. |
| `coverage(as_of=None)` | Report every scheduled model/target, including missing or late forecasts and unscored outcomes. |
| `paired_benchmark(baseline_freeze, candidate_freeze)` | Report matched score differences and all missing coverage; output is explicitly descriptive. |
| `verify(expected_head=None)` | Check the local chain and all referenced raw/model hashes; optionally compare an independently retained exact head. |

All dates require explicit offsets and are normalized to UTC. Targets include
exact start/end instants and exchange session identity; midnight date labels
are insufficient. For overnight-return targets, the target starts at the prior
close, so a forecast produced after that close is already late. Choose an
actually forecastable interval before collection and never move its start to
accommodate a late run. Calendar adapters must handle DST, holidays, early closes,
maintenance intervals, settlement versus last trade, and contract expiry.

The primary outcome policy is **first received**. Corrections remain visible
but do not replace already issued scores, even if they arrive before the scoring
command first runs. This fixed policy prevents selecting a favorable vintage;
it does not guarantee the first vintage is correct. A corrected-vintage report
would need a separately declared evaluation version. Historical `as_of` views
filter by actual local receipt, including the score's own materialization time.
No reissued forecast or revised model inherits the old model's prospective clock.

Normal and Student-t forecasts use explicit location and **scale**, not variance.
The density includes the scale Jacobian. The Student-t distribution is not
asserted to have finite variance. Variance targets and predicted means must be
strictly positive; zero or invalid outcomes fail explicitly without clipping.
QLIKE is `y / mean - log(y / mean) - 1`. This initial scorer does not accept
multivariate copulas, arbitrary calibrated densities, spread payoffs or Greeks.

## Data to retain for selling spreads

For every decision, save an immutable JSON decision bundle as a source receipt
and require its dataset in the frozen input specification. It should contain
the model freeze, selected legs, quantity, quote receipt IDs, planned execution
rule and the following fields. The actual option/sizing adapter must validate
these fields before a trade benchmark can begin; this ledger stores their bytes
without pretending to implement that adapter.

| Data | Required content |
| --- | --- |
| Option contracts | Exact contract identifier, call/put, strike, expiry timestamp, multiplier, currency, exercise style, settlement method, AM/PM settlement, deliverable and corporate-action version. |
| Decision quotes | Bid/ask, bid/ask size, exchange/OPRA timestamps, local receipt, quote age, underlying price and timestamp, source/feed, condition codes, IV/Greeks with methodology/version. |
| Leg selection | Short and long legs, widths, expiration, put/call/condor rule, any delta filters, quantity and collateral rule. Freeze before outcomes; retain rejected/no-trade decisions. |
| Execution and exits | Sell-at-bid/buy-at-ask reference or another predeclared fill rule, quote synchronization/staleness rule, size/depth limits, commissions, exchange fees, actual broker fills if later authorized. Intraday quotes for every stop/early-exit rule and marking time. |
| Carry and account constraints | Decision-time interest/dividend assumptions, broker margin/house requirements, buying-power treatment, financing and fee schedules with effective dates and original receipts. |
| Outcomes | All exit-leg quotes/fills, actual settlement value, assignment/exercise events, expired/adjusted contract handling, and instrument-specific realized cash flow. |

Do not treat a historical midpoint as an executable spread fill, add incompatible
leg quotes, infer an intraday stop from a final close, or replace SPX settlement
with a convenient SPY close. A risk-neutral quote delta is not an empirically
validated physical tail probability. Compare predicted breach probability,
expected loss, and realized risk separately from premium and net cash flow.

For an eventual NQ/ES options or futures extension, retain actual dated contracts,
quotes, trades, instrument definitions, tick value, expiry/roll decisions, session
calendar and margin/fee vintages. MNQ/MES and NQ/ES are separate products. A
continuous or adjusted series cannot establish executable roll transactions.

## Provider choices and constraints

Primary documentation checked 2026-09-13; no data purchases or market API requests
were made. Choose a provider only after checking the specific account entitlement.

- **QQQ/SPX options:** [Cboe Option Quote Intervals](https://datashop.cboe.com/option-quote-intervals)
  supplies OPRA option NBBO/size and optional IV/Greeks. It excludes options on
  futures. Its intraday files arrive with a 15-minute delay; daily files arrive
  overnight or next morning. That is useful for outcome collection, but the
  historical quote timestamp cannot justify a contemporaneous forecast or fill.
  SPX underlying values can require a separate CGIF license. Its documented
  June 2026 quote-size convention change also belongs in source provenance.
- **Per-contract option quotes:** [Massive's options quotes API](https://massive.com/docs/rest/options/trades-quotes/quotes)
  is an alternative for contract-level bid/ask records. Verify quote coverage,
  timestamps, pagination and real-time entitlement; do not infer Greeks or
  settlement fields from a quote-only response.
- **Options on NQ/ES and actual futures:** [CME DataMine](https://www.cmegroup.com/datamine/datamine-api.html)
  provides API access to purchased historical datasets; [CME's data catalog](https://www.cmegroup.com/market-data/browse-data/catalog/futures-and-options-data.html)
  distinguishes products and granularity. This is a separate route from OPRA
  equity/index options. A licensed real-time feed or broker feed is still needed
  for contemporaneous decision inputs.
- **QQQ/SPY underlying bars:** [Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars)
  exposes explicit `feed`, `adjustment`, timestamp bounds and pagination. SIP
  covers all US exchanges; IEX is a distinct venue feed. Retain raw OHLC and
  corporate-action receipts separately; never silently change feeds or back-adjust
  stored observations. [Massive aggregates](https://massive.com/docs/rest/stocks/aggregates/custom-bars)
  is another documented source. Confirm regular-hours aggregation before using
  daily OHLC for the existing Garman–Klass estimator.
- **VXN and SPX model inputs:** the repo downloads Cboe's VXN history and uses
  separately sourced SPX history. Save each retrieval's bytes and first-receipt
  time. [Cboe's VXN dashboard](https://www.cboe.com/us/indices/dashboard/VXN/)
  identifies the series, but a daily index value is not an executable quote or
  a guarantee of publication at equity close. SPY is an ETF proxy, not SPX;
  substitutions require a new instrument/target definition and benchmark cohort.

No advertised provider price is assumed. Options quotes, index values and
real-time access can have separate entitlements. Keep API secrets out of stored
URLs and metadata; record sanitized endpoint paths and non-secret query fields.

## Operating sequence

1. Freeze one candidate and its benchmark, feature parser/code, outcome recipe,
   data vintages, required inputs, session calendar, decision deadline, accrual
   endpoint and evaluation plan. For variance forecasts consuming VXN, use HAR-IV
   as the relevant control. Version all subsequent calibration/model changes.
2. Before each target starts, schedule both model freezes for that exact target.
   Collect provider responses into the new ledger only, validate feed/instrument,
   freshness, pagination and session completeness, then compute features from
   those receipt IDs. New training data can only enter under the frozen update
   schedule and must already be mature.
3. Run the live forecast adapter and append both forecasts before the target
   starts. Persist leg decisions and all input receipts. A missed deadline stays
   missed; a late catch-up forecast is retained without prospective eligibility.
4. After target end and receipt of complete source evidence, use a separately
   tested parser to calculate the frozen outcome and append it. Append later
   revisions separately. Run score maturation and inspect complete coverage.
5. Back up raw blobs and the ledger, and retain its current hash in an independent
   location. Optional external timestamping would strengthen receipt evidence.
   The local host clock and hash chain are not external time attestation and
   cannot defeat a privileged operator rewriting the whole archive.
6. Check collection integrity routinely. Reserve model comparison, calibration
   retuning and inferential decisions for the prespecified endpoint; do not stop
   when a favorable score appears. The ledger supplies matched observations, not
   a sequential testing rule or automatic promotion gate.

Plan an initial **500 new scheduled daily origins**, approximately two trading
years, to assess coverage and broad calibration. A nominal 2.5% tail then has only
12.5 expected events; 2,000 origins give 50. Neither arithmetic establishes power,
and dependence or overlapping option positions reduces information further.
Determine final sample size from a declared economically relevant loss difference
and dependence-aware design before opening prospective comparative scores.

## Using the local interface

The CLI reads JSON keyword arguments matching the API table. Source/model bytes
are separate files. It always stamps the current local UTC time; there is no
backdated receipt argument. For example:

```sh
.venv/bin/python -m src.prospective_benchmark --ledger /private/path/prospective ingest --request source-request.json --payload response.json
.venv/bin/python -m src.prospective_benchmark --ledger /private/path/prospective freeze --request model-request.json --payload model.bin
.venv/bin/python -m src.prospective_benchmark --ledger /private/path/prospective schedule --request next-target.json
.venv/bin/python -m src.prospective_benchmark --ledger /private/path/prospective issue --request forecast.json
.venv/bin/python -m src.prospective_benchmark --ledger /private/path/prospective observe --request observation.json
.venv/bin/python -m src.prospective_benchmark --ledger /private/path/prospective score
.venv/bin/python -m src.prospective_benchmark --ledger /private/path/prospective coverage
.venv/bin/python -m src.prospective_benchmark --ledger /private/path/prospective verify
```

The fully working synthetic fixtures in `tests/test_prospective_benchmark.py`
show exact request objects, including all target and protocol fields. They use
an injected synthetic clock and never access market data. Run them with:

```sh
.venv/bin/python -m unittest tests.test_prospective_benchmark -v
```

The remaining integration work is explicit: provider fetch/parsing adapters,
feature-freshness and model-fit validation, calendar-derived target scheduling,
an actual live forecast entry point, a versioned scorer for the newly calibrated
densities, and an independently checked option decision/execution evaluator.
The existing `src/fetch.py` overwrites historical panels; the existing study
runners generate retrospective forecasts. Neither should be relabeled as a
prospective collector. No current forecast has been issued by this subsystem.
