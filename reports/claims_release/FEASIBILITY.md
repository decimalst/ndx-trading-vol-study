# Prospective source decision: first-release unemployment claims

Reviewed 2026-09-08. **Advance one bounded source-admission audit of first-release U.S. weekly unemployment insurance initial claims. No source is admitted and no new experiment is registered by this note.** This is an administrative labor-market information source, materially different from another transformation of equity closing prices, realized variance or implied volatility.

The proposed mechanism is that recent deterioration in first-reported claims contains information about subsequent equity variance after market prices have incorporated the release. This is a hypothesis, with a demanding incremental comparison; no reviewed evidence establishes that it works. The suggested statistic and models below remain prospective until the source audit is complete.

## What previous work did and did not exhaust

The completed program tested many configurations of daily prices, realized-risk measurements, Cboe indexes, international realized volatility, cross-asset prices, return shape, calendar structure and forecast calibration. Those failures describe the registered comparisons, not every possible external source. In particular:

| Avenue | Recorded status and implication |
| --- | --- |
| Macro announcement information | The [macro design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/macro_overnight/DESIGN.md) tested previously announced CPI/payroll/FOMC plans against overnight variance controls. Its [verified summary](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/macro_overnight/SUMMARY.md) found no qualifying increment. It did not test released unemployment-claims values or first-release labor deterioration. |
| CFTC positioning | The official Nasdaq futures panel was acquired and one frozen positioning augmentation failed its gate. This is an empirical result for that feature, not exhaustion of all positioning mechanisms. The [source ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/FREE_DATA_SOURCES.md) distinguishes this from unavailable sources. |
| Option activity and surfaces | The frozen activity composite was undefined and produced `INSUFFICIENT_DATA`; NQ intraday history produced `VERIFIED_NO_EVALUABLE_FOLDS`. Neither is a predictive null. QQQ/SPY/AAPL surface studies were private, inconclusive diagnostics with unresolved upstream provenance. TSLA option fields were inspected without a frozen predictive study. The [source ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/FREE_DATA_SOURCES.md) records these distinctions. |
| Richer option quotes and early-session index data | These avenues remain blocked by specific source, timing, vintage, coverage or permission requirements, as documented in the [quote review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/civil_quarter_replay/QUOTE_SOURCE_DOCUMENTATION.md) and [early-session feasibility report](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/early_session_feasibility/SUMMARY.md). Their non-admission is not evidence against their predictive mechanisms. |
| Other administrative liquidity data | Funding/liquidity appears in the [prospective backlog](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/HISTORICAL_WEIGHTS_AND_SIGNAL_BACKLOG.md). The bounded text and manifest-name inventory found no recorded claims/ALFRED, dealer-fails or Treasury-auction trial. This is a bounded inventory finding, not proof that every repository artifact was searched. |

Root separately checked current NAAIM documentation and found a paid route to comprehensive history; no free historical route is established. This note does not propose another source comparison or substitute an unverified mirror.

## Concrete new source and public timing evidence

The originating agency is the U.S. Department of Labor's Employment and Training Administration. Its [official claims archive](https://oui.doleta.gov/unemploy/archive.asp) provides earlier releases and documents a normal Thursday morning publication time with holiday exceptions. The constant latest-release PDF is overwritten and cannot serve as the historical identity. Actual dated releases must determine release timing.

The Federal Reserve Bank of St. Louis provides a specific, currently accessible [ALFRED ICSA download form](https://alfred.stlouisfed.org/series/downloaddata?seid=ICSA), which supports historical vintage selection. Its [download documentation](https://alfred.stlouisfed.org/help/downloaddata) defines an initial-release-only output with initial public-release dates. The [API observation documentation](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) names this `output_type=4`. The Fed's own [ALFRED explanation](https://www.stlouisfed.org/open-vault/2021/august/using-the-alfred-database) explicitly uses initial unemployment claims as a vintage-data example.

This establishes a concrete free-access route, rather than an inference from a current unrevised-history download. [FRED's current terms](https://fred.stlouisfed.org/legal/) describe no-cost access for permitted personal/noncommercial uses. The API requires a user's registered key; the download form is a separate route. No key was sought, account created or download submitted here. [API key requirements](https://fred.stlouisfed.org/docs/api/api_key.html).

The series metadata exposed in the official search result begins its real-time attribute history on **2009-05-28**. Treat pre-2009 data as ineligible for this proposal. Do not interpret a long observation history in the first archived snapshot as genuine original releases: an initial archive backfill may contain previously revised observations. Exact per-release coverage after that boundary remains unverified; the metadata date alone does not prove a complete weekly vintage series.

ALFRED's [source-date policy](https://alfred.stlouisfed.org/help) uses source release dates where available, otherwise provider dates, otherwise first FRED availability. Updates are typically added within one business day. Therefore release date, observation week, archival vintage date and retrieval date must remain separate. A current retrieval cannot itself authenticate an intraday historical feed.

## One proposed test, subject to admission

Let `c[k]` be the seasonally adjusted claims number first publicly reported for reference week `k`, preserved at its first genuine release. The single candidate would be:

`x[k] = log(c[k]) - mean(log(c[k-1]), ..., log(c[k-4]))`.

This is signed deterioration relative to the four preceding first reports. It is **not a consensus surprise**; no archived economist forecast survey has been established. Require consecutive reference weeks and complete positive first reports. Never substitute today's revised history or quietly skip a missing week.

The proposed target is the **next five observed QQQ sessions' arithmetic mean GK-plus-overnight variance**, using the already frozen measurement definition from the earlier volatility experiments. Five sessions matches the weekly information-update cycle and directly returns to the user's volatility question. It is the only suggested horizon; neither signed returns nor a horizon grid is proposed.

The incremental control should retain the established HAR realized-variance history, leverage, delayed VXN/VIX/term inputs and cross-asset/market-stress controls. Give the control the same release age/calendar state and the four-prior-first-report log mean. The candidate model adds only `x[k]`. Thus the primary contrast asks whether the latest first report contributes beyond prices, release timing and the preexisting labor level. A historical-mean forecast may remain a descriptive reference; the substantive claim requires the matched strong-control comparison. Exact estimator, training gates and cumulative-family accounting need a separately frozen protocol, not a decision based on later numerical support or scores.

For an after-close origin, make each release eligible only on the first observed market session strictly after the later of its authenticated release date and ALFRED first-availability date. This conservative date rule avoids assuming an intraday vintage timestamp. Persist a known released state only until the next scheduled update is due; missing or ambiguous new-release coverage then makes the state unknown. Freeze that release-ledger rule before extraction. Do not borrow later revisions, later-discovered release calendars or a future realized label to decide availability.

Daily forecasts may share one weekly claim report; repeated days do not create independent macro observations. Support and uncertainty must explicitly account for distinct releases and serial dependence. Seasonal-adjustment changes and administrative coverage changes require documented treatment fixed before inspection. Do not remove pandemic or other inconvenient episodes after outcomes are known.

## Exact next admission checks and stopping condition

1. Pin official ICSA series identity, units/seasonal treatment, source terms, vintage-date metadata and dated DOL archive identities through the existing 2025-10-20 boundary. Establish a usable acquisition route without fetching later values.
2. Establish the first genuine weekly releases after the initial archive snapshot. Reject backfilled prehistory. Reconcile week-ending dates, first-release dates and values against original dated DOL notices under a prewritten bounded audit; record exceptions and unresolved revisions explicitly.
3. Freeze missing-release, holiday, delayed-release, seasonal-method-change and initial-snapshot rules. Audit documentary coverage before constructing any market-aligned cohort. If authentic first reports cannot be distinguished from revised backfills or sufficient historical releases cannot be authenticated, stop as source-ineligible/insufficient; do not change the phase boundaries or use revised FRED data as a rescue.
4. Only after admission, freeze the single feature/control/target design, meaningful synthetic timing tests, independent reconstruction and family accounting before any numerical feasibility counts or fits. Keep frozen studies unchanged. Reused historical phases remain exploratory; a new information source does not replenish the already spent historical holdout.

The present blocker is **absence of a pinned, independently reconciled first-release ledger and authenticated usable vintage coverage**, not a demonstrated paywall or an empirical failure. The documentary route is credible enough to justify that one next audit; it is not enough to declare the source ready for modeling.

## Scope and incidental exposure

Only existing report prose, source-manifest filenames and primary documentation were reviewed. No dataset was downloaded, no market observations or model outputs were newly parsed, and no features, support cohorts, models or scores were computed. A read-only attempt to inspect date-valued fields on the official download form failed at local DNS resolution and produced no source data.

An official-domain metadata search returned the ICSA landing-page snippet containing current post-cutoff claims observations. Those incidental values were not requested as an observation table, copied into this note or used in selecting a transformation or assessing results. They are nevertheless an exposure, so this review does **not** claim that zero post-cutoff source values were displayed. No protected post-cutoff equity outcome was accessed. Root was notified immediately. Only this temporary prospective note was written; all frozen repository files remain unchanged.

## Root handoff after the completed peak-age study

Saved to the repository on 2026-09-08 as a prospective source-admission decision, outside the now immutable peak-age report. Peak-age publication audit `45e60e48cbb4a3aa65f3ce7909f4b9d21c4621061e7706979054fc88e7641730` certifies no lead and 140 cumulative hypotheses. This note registers no additional comparison.

Root independently confirmed the ALFRED initial-release-only documentation and also encountered current claims values in metadata-search snippets; no such values entered any file of numerical inputs, feature, cohort, fit or score. An unrelated Treasury-auction documentation search displayed a dated 2024 example release. These incidental documentary observations must not be described as a sealed, entirely unseen external-data universe. No protected equity outcomes were inspected.

The next concrete step is a tested source loader and dated-release reconciliation audit. No claim-series forecast, numerical cohort analysis or model tuning has occurred. The proposed estimator, model-comparison family and precise origin clock remain to be specified prospectively before numerical experimentation.
