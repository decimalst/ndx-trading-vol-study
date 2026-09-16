# Scheduled macro releases and overnight variance: feasibility

2026-09-06. Prospective design audit only. No models were fitted, no forecast
scores or event-return comparisons were inspected, and no frozen study file
was changed. Price observations were bounded at 2025-10-20; the protected
2025-11-03-and-later market data were not used.

**CPI and payroll release schedules have a more direct timing rationale for
overnight squared returns than for signed returns. The existing calendars do
not yet support a verified historical information set.** An announcement can
cause a move whose direction depends on its unknown surprise; a known release
time can nevertheless identify a period of greater uncertainty. This is a
mechanism to test, not evidence of incremental forecasting value. Andersen,
Bollerslev, Diebold and Vega document price responses to real-time macro news
and state-dependent equity responses; their results do not establish that a
calendar dummy beats overnight history and implied volatility here.
[Author paper at the Federal Reserve](https://www.federalreserve.gov/pubs/ifdp/2006/871/ifdp871.pdf)

## What the repository currently contains

The three checked-in calendars have a single `date` column. Comments say they
were manually checked on 2026-08-11 against BLS and Federal Reserve pages.
There are no per-event publication timestamps, schedule vintages, cancellation
timestamps, source snapshots, or machine-readable scheduled/emergency labels.
The files begin in 2016; absent earlier rows mean unknown coverage, not zero
events. The inspected git history first records these files in the August
2026 baseline, followed by the research-path commit.

| File | Rows dated through 2025-10-20 | First / last bounded date | Specific problem |
| --- | ---: | --- | --- |
| `calendars/cpi.csv` | 117 | 2016-01-20 / 2025-09-11 | Retrospective actual/revised release calendar; includes two releases on QQQ holidays |
| `calendars/nfp.csv` | 117 | 2016-01-08 / 2025-09-05 | Same vintage problem; includes two releases on QQQ holidays |
| `calendars/fomc.csv` | 79 | 2016-01-27 / 2025-09-17 | Includes two emergency decisions and shifts a Sunday announcement to Monday |

There are no duplicate dates. CPI and payroll each contain 12 dates per year
from 2016 through 2024 and nine bounded 2025 dates. FOMC contains 77 scheduled
actual decision dates after removing the two explicitly identified emergencies;
that count is not a reconstruction of schedules known at each origin.

`src/features.py::build_covariates` matches events only to realized trading
dates and drops other dates. That is inappropriate for this overnight target:
the stored CPI releases on Good Friday 2017-04-14 and 2020-04-10, and payroll
releases on Good Friday 2021-04-02 and 2023-04-07, occur between Thursday's close
and Monday's opening. They belong to that planned holding interval. Nasdaq's
dated 2017 holiday reminder, published April 10, confirms that April 14 closure
was known beforehand. The existing missing-file fallback to all-zero flags and
full-calendar countdown construction must also not be reused.
[Nasdaq advance closure notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2017-78)

The FOMC comments explicitly include 2020-03-03 and 2020-03-16. The Fed identifies
March 2 and March 15 as unscheduled meetings and March 17–18 as cancelled. The
second emergency announcement was Sunday March 15 at 17:00; replacing that
timestamp with Monday March 16 would put the surprise itself into an allegedly
preannounced feature. Exclude emergency events from every scheduled predictor,
while retaining their ordinary target observations in training and scoring.
[Fed 2020 record](https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm),
[March 15 announcement timing](https://www.federalreserve.gov/monetarypolicy/fomcpresconf20200315.htm)

The original studies already tested calendar covariates for broader volatility
targets. This proposal isolates a different measurement interval and information
cutoff. It is a new, adaptively proposed question on reused history, not a
reclassification of the completed signed-return experiment.

## A viable source route, with unresolved work

BLS states that its principal release schedule is ordinarily published in the
fall for the following calendar year, with CPI and Employment Situation at
08:30 Eastern. This general policy supports advance availability but is not a
substitute for historical snapshots.
[BLS dissemination policy](https://www.bls.gov/about-bls/dissemination.htm)

Concrete dated primary documents preserve next-release preannouncements:

| Public document | Future schedule stated in that document |
| --- | --- |
| [CPI, published 2015-12-15](https://www.bls.gov/news.release/archives/cpi_12152015.htm) | 2016-01-20, 08:30 EST |
| [Employment Situation, published 2015-12-04](https://www.bls.gov/news.release/archives/empsit_12042015.htm) | 2016-01-08, 08:30 EST |
| [CPI, published 2025-09-11](https://www.bls.gov/news.release/archives/cpi_09112025.htm) | 2025-10-15, 08:30 ET |
| [Employment Situation, published 2025-09-05](https://www.bls.gov/news.release/archives/empsit_09052025.htm) | 2025-10-03, 08:30 ET |

The last two dates are absent from the current bounded CSVs because those
releases were delayed. Thus, retrospective actual dates demonstrably differ
from what an earlier forecaster was told. Neither a cancelled flag applied
before the cancellation announcement nor a revised date applied before its
announcement is admissible. Shutdown disruptions also affect earlier training:
BLS records that the 2013 payroll release originally scheduled for October 4
was moved to October 22. The current revised page alone does not timestamp
each schedule update.
[BLS 2013 rescheduling record](https://www.bls.gov/bls/updated_release_schedule.htm)

For FOMC, dated public schedule announcements are stronger evidence than a
current historical meeting list. The Fed's May 11, 2015 announcement gives the
2016 schedule; its May 17, 2019 announcement gives the 2020 schedule, including
the subsequently cancelled March meeting. Minutes and transcripts must use
their public release date, not their meeting date; some materials were
confidential for years.
[2016 schedule announcement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20150511a.htm),
[2020 schedule announcement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20190517a.htm)

Before fitting, construct a separate append-only ledger with event type,
reference period, planned timestamp in `America/New_York`, announcement
timestamp, document policy, public source URL, retrieved timestamp, and source
SHA-256. A dated publisher archive provides documentary evidence of a
preannouncement; a currently served archive is still not an independently
captured historical byte vintage. Record that distinction and inspect errata.
Complete coverage must justify zero-event indicators; missing evidence stays
unknown. Archive source documents, not just parsed dates.

The preferred acquisition policy is explicitly **the next-release date printed
in the preceding monthly release**. This is an originally announced plan,
although an annual calendar might have announced it even earlier. Once a
monthly report supplies that plan, retain it permanently for this predictor.
Later cancellations and reschedulings do not erase, move, or duplicate it. Thus
the September 2025 documents continue to mark the October 3 payroll plan and
October 15 CPI plan. The feature asks whether an originally planned release
date predicts overnight uncertainty; it does not assert that a release will
actually occur or that traders have received no subsequent information.

This fixed policy is causal without reconstructing every revision. It can use
only documents whose public release timestamps precede the feature cutoff.
Do not select a preceding document by looking backward from the eventual
actual event date: ingest documents in publication order and extract the plan
stated in each. A missing or unparseable next-plan statement remains unknown
under a prewritten coverage rule; do not carry an unrelated stale plan forward.
For a valid stored plan that has already passed, the original-plan indicator
is zero on other dates, even if the eventual report remains delayed. Annual
schedules or later reports must not be silently substituted for failed monthly
extractions. Any competing latest-revised-calendar model would need its own
complete timestamped revision ledger and separate hypothesis allocation.

The scheduled FOMC timing control should likewise use a fixed **advance annual
schedule announcement** policy and retain planned meeting dates even if later
cancelled. It never adds an emergency meeting. This keeps the interpretation
consistent as a planned-calendar comparison, with source-policy differences
explicitly identified.

### Concrete acquisition path and four verified examples

The official indexes are [CPI archived releases](https://www.bls.gov/bls/news-release/cpi.htm)
and [Employment Situation archived releases](https://www.bls.gov/bls/news-release/empsit.htm).
Both currently list 2009–2015 HTML/PDF links as well as later releases, so a
pre-2016 training extension is discoverable. Index headings describe the
**reference month**, not the document's publication month. Follow actual links
rather than generating presumed release dates. Verified HTML URL patterns are
`https://www.bls.gov/news.release/archives/cpi_MMDDYYYY.htm` and
`https://www.bls.gov/news.release/archives/empsit_MMDDYYYY.htm`, where the digits
are publication date. The index also links PDF versions; follow those links
rather than assuming every extension exists.

Extract only the embargo/publication timestamp, release identifier, reference
period, and next-release sentence. Do not extract macro values or surprises.
Require planned time after announcement time, a recognized timezone, a unique
source document, and a reviewed discrepancy record. For model development,
freeze input documents published no later than the permitted source cutoff;
links to later documents in a current index must not be followed into features.

The following four records were independently checked against the publication
header and next-release sentence via the browser on 2026-09-06. Times below
include the appropriate UTC offset. Direct downloads were attempted only for
these four pages: sandbox DNS failed, and the approved system-network retry
received HTTP 403 from BLS. Consequently **raw source byte hashes are not
available in this audit**; the `record_sha256` values are reproducible hashes of
the canonical field records shown below, not hashes of downloaded HTML and not
evidence of historical byte vintages. `raw_source_sha256` is explicitly null
for all four. A successful browser export or permitted source acquisition must
capture those bytes and fill that field before the acquisition manifest can
claim source-hash verification.

| Event | Announcement timestamp | Originally planned timestamp | Source URL |
| --- | --- | --- | --- |
| cpi | 2015-12-15T08:30:00-05:00 | 2016-01-20T08:30:00-05:00 | [cpi_12152015.htm](https://www.bls.gov/news.release/archives/cpi_12152015.htm) |
| nfp | 2015-12-04T08:30:00-05:00 | 2016-01-08T08:30:00-05:00 | [empsit_12042015.htm](https://www.bls.gov/news.release/archives/empsit_12042015.htm) |
| cpi | 2025-09-11T08:30:00-04:00 | 2025-10-15T08:30:00-04:00 | [cpi_09112025.htm](https://www.bls.gov/news.release/archives/cpi_09112025.htm) |
| nfp | 2025-09-05T08:30:00-04:00 | 2025-10-03T08:30:00-04:00 | [empsit_09052025.htm](https://www.bls.gov/news.release/archives/empsit_09052025.htm) |

For each row, `record_sha256` is SHA-256 of UTF-8
`event|announcement_timestamp|planned_timestamp|full_source_url` followed by
one LF newline, with no spaces or other fields. In table order:

```
5d177c12c70b4063bb43fbfb20341836d9ae1624123a13d766192a369b379ab0
b1f12c6a326b6a08f3db95bd448d8899af0dcc4b59d90ca3b1e207d6084022e4
9b6c06095386b661998a9fc0be9315d184c8b5c1979cffc09fae649cffa2a544
22908761bc9e096ccf8e153e35e8b957a0450d67ce05058fffec9f1a3918a7cf
```

## Entry-compatible event construction

Let entry `t` be a known QQQ session. All market measurement dates end at the
preceding completed session, and eligible advance-plan documents must be public
by its QQQ close `c(t-1)`. An order is submitted before the entry-session
closing-auction deadline. The prior-session Cboe daily close can contain
calculations after the 16:00 QQQ close, so retaining wave three's prior-session
VXN/VIX convention means a prior-session end-of-day state assembled before the
entry order, not literally a complete 16:00 snapshot. A protocol that instead
requires every observation available at exactly 16:00 on the prior day must lag
Cboe an additional session. Freeze and name that timestamp convention before
implementation; neither interpretation permits entry-day market data. Use an
exchange session schedule that was publicly known by that cutoff to determine
the **planned** next opening `o_planned(t)` and the entry close, including
announced holidays and early closes. Use timezone-aware timestamps so 08:30
ET is handled correctly across daylight-saving changes.

`cpi_preopen(t)` equals one if an original CPI plan timestamp known by the cutoff
falls in `(close_planned(t), o_planned(t)]`; define payroll identically. This
includes holiday releases without consulting the next observed price row.
Record planned duration as a baseline control, because holiday/weekend length
can otherwise masquerade as event information. Event dates on entry morning
are already outside this target interval and receive zero.

The FOMC timing-control flag identifies an originally scheduled announcement **after** the
planned next opening and on that planned opening session. It must remain
separate from the preopen flags. Scheduled announcements generally occur at
14:00 in the scored years; earlier training must preserve historically correct
scheduled times, which changed. FOMC is a timing comparison, not a guaranteed
statistical null: anticipatory repricing is possible before an afternoon
announcement. Do not require its coefficient to be zero to recognize a CPI or
payroll result.
[Federal Reserve discussion of announcement timing](https://www.federalreserve.gov/econres/notes/feds-notes/how-do-principal-trading-firms-and-dealers-trade-around-fomc-statement-releases-20201231.html)

Never use `next observed QQQ date` to construct a predictor at this cutoff.
An unanticipated closure can change the realized exit. Retain the original
forecast and record planned-versus-realized exit as a later measurement audit;
do not move event flags or remove affected returns using hindsight. The actual
next session remains valid for target construction and label maturity only.

## Target and the permitted zero-count audit

Use one primary target, fixed before modeling:

`r_ON(t) = log(A_next/A_t) - log(C_next/O_next)`

`q(t) = r_ON(t)^2`.

This is the squared vendor-adjusted overnight log-return proxy. Its conditional
expectation is a **second moment**; it equals conditional variance only under
a zero conditional mean assumption. It is not integrated overnight variance
observed from intraday transactions, and it is not a cash-dividend trading
profit. Do not import a fitted signed-return correction from the failed wave
or claim the measurement is noiseless. Keep the existing adjustment-factor
jump flag as a retrospective sensitivity diagnostic, never as a predictor or
training exclusion.

The following counts use only raw QQQ data through 2025-10-20. They apply no
feature-completeness mask, no event conditioning, no rounding, and no positive
floor. Arithmetic overnight squares were counted only to check whether zero
handling depends on that convention; they are not a second proposed target.

| Label window | Finite labels | Exact zero adjusted log squares | Exact zero adjusted arithmetic squares | Exact zero raw unadjusted log squares |
| --- | ---: | ---: | ---: | ---: |
| All bounded completed labels | 6,695 | 2 | 2 | 81 |
| Before first 2016 entry, available by 2015-12-31 | 4,231 | 2 | 2 | 70 |
| Entries 2016-01-04–2019-12-31, available by 2019-12-31 | 1,005 | 0 | 0 | 9 |
| Entries 2020-01-02–2025-10-17, available by 2025-10-20 | 1,457 | 0 | 0 | 2 |

Counts are exact floating-point zeros for the declared expression. Tiny
nonzero vendor-adjustment differences can turn a nominally unchanged raw
opening into a nonzero adjusted return; they must not motivate a data-dependent
tolerance or a new target after inspection. Historical formula evaluations can
also differ at floating-point precision. Zero-compatible scoring is required
even if a particular evaluation segment has no exact zeros.

## Fixed small experiment proposed for registration

Use monthly expanding fits with at least 1,000 completed common labels.
Historical entry must precede fit entry, and the label becomes available at
the actual next session **close**, not its open. Admit it only when
`available_date <= feature_cutoff_date` of the first scored entry in the month.
All arms share the same finite feature, verified-schedule, and mature-label
sample. Calendar training coverage must be extended to support the initial
fits; the current 2016 start cannot support a 2016 evaluation with 1,000
calendar-complete training labels. Do not fill pre-2016 event history with zero
or silently reduce the minimum.

Proposed fixed dates remain development 2016-01-04–2019-12-31 with labels
available by 2019-12-31, evaluation 2020-01-02–2025-10-17, target/source end
2025-10-20, and evaluation stability slices 2020–2022 and 2023–2025. If source
reconstruction cannot support these dates, report the design unevaluable or
register a separate later-start design before any scores. These are disjoint
historical phases, not untouched data after this repo's previous research.

The baseline should contain an intercept; square roots of strict mean own
adjusted overnight squares over 1/5/22 sessions; square roots of mean daytime
log-return squares over 1/5/22 sessions; log mean GK-plus-raw-overnight variance
over 1/5/22 sessions; prior-session log VXN and log VIX; entry weekday Tuesday
through Friday; and planned hours from entry close to next opening. The square
root features accept zero without inventing a log-target floor. All rolling
market features end at the prior session. Retain a historical-mean `q` forecast
on exactly the same training rows as a second control.

Register three singleton augmentations: baseline plus CPI-preopen; baseline
plus payroll-preopen; baseline plus scheduled FOMC-after-open as the timing
control. No interactions, alternative windows, release-surprise fields, or
indicator combinations in this family. Compare each augmentation with baseline
and historical mean, retaining all six contrasts in multiplicity bookkeeping.
CPI and payroll are the two substantive candidates; the FOMC arm addresses
timing interpretation and cannot be relabelled as proof of preopen news risk.

A zero-compatible estimator is a log-link quasi-likelihood model,
`h = exp(b0 + z' b)`, minimizing
`mean(log(h) + q/h) + 0.01 * ||b||^2`, with training-only population scaling and
an unpenalized intercept. This directly estimates the positive conditional
second moment without fitting `log(q)` or using a smearing correction. The
historical-mean control equals the intercept-only optimum. Require a positive
training mean, finite strictly positive forecasts and objective values, and
verified optimizer convergence/KKT conditions; report failures rather than
clipping predictions or replacing zero targets. Fixed penalty, optimizer,
tolerance, and iteration budget must be frozen before fits.

## Proper score and an absolute effect gate

For `q >= 0` and forecast `h > 0`, score

`L(q,h) = log(h) + q/h`.

At `q=0` this is finite `log(h)`. Its conditional expected derivative is
`(h - E[q|information])/h^2`, so it is minimized at the conditional second
moment when that moment is positive. Patton explicitly presents this QLIKE
form and explains the assumptions needed for noisy variance-proxy comparisons.
The alternative expression `q/h - log(q/h) - 1` is undefined at zero and must
not be used here. Raw scores can be negative and shift with return units;
paired score differences are invariant to a common rescaling of `q` and `h`.
[Patton, equation 6](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)

Propose an absolute, predeclared gate:
`mean(L_candidate - L_control) <= -0.005` natural-log score units in **both**
phases and against **both** controls. This is a proposed design choice to
freeze now, not a threshold derived from any event result or a percentage of a
possibly negative baseline. Report the absolute gap and uncertainty; no
percentage-QLIKE improvement column. As an analytical scale reference only,
forecasting 1.1 times the true constant second moment has expected excess loss
`log(1.1)+1/1.1-1`, about 0.0044. That illustration is not a trading-value claim.

Retain the existing paired circular bootstrap blocks 21/63/126 and HAC126
checks, max-p across methods and phases, negative gaps in both evaluation
stability slices, and negative gaps on unflagged adjustment-event origins in
both phases using the same frozen forecasts. If this is wave four, the
pre-score ledger would contain 78 inherited plus six new contrasts: Holm over
six at `0.05/(4*5)`, and independently Holm over all 84 at 0.05. Failure keeps
all planned hypotheses as unevaluable with p=1. Effective sample size and
uncertainty must reflect serial dependence and the small number of release
occurrences; a large daily-row count is not a large event count. Do not use
event slices as replacement primary tests.

## Minimum prewritten contracts and readiness

Before any empirical fit, independently test: zero-target scores and gradients;
unit-rescaling invariance of paired loss; historical-mean optimum; training-only
scaling and mean-loss penalty normalization; score invariance to irrelevant
common constants; positive-forecast/convergence failure handling; prior-close
source mutation fences; label maturity at next close; later schedule revisions
and cancellations leaving original-plan flags unchanged; plan documents used
only after publication; emergency FOMC exclusion; holiday release assignment;
early-close and DST timestamps; planned-calendar immunity to a future realized
closure; unknown calendar coverage remaining missing; identical arms and phase
fences; and all six failures retained in both correction families. Carry over
the split/dividend and target-close-cancellation measurement tests.

**Current readiness: mechanism and zero-compatible measurement/scoring are
specified; historical schedule provenance is incomplete.** The next useful work
is assembling and independently verifying the fixed original-plan and
exchange-session ledgers, including enough pre-2016 coverage. The existing CSVs
alone justify only a clearly retrospective diagnostic, not a verified
point-in-time predictive claim. No variance experiment was run in this audit.

Calendar byte hashes at this audit:

```
cpi.csv  0eef8d151ef95a266278b13dc21999f62d5c1f5fc67b73b5e13241cb75a201f2
nfp.csv  8c58050ee3444b6f262af377e778cfa7c1f9a67f8a268d6191f939dd03e01760
fomc.csv 795a9257bd3ac25318d2e4a72726b4bdadf0e55e9c7578db46664205378f7e1c
```
