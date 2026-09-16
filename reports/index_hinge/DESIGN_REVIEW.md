# Independent pre-fit review: medium-horizon index hinge

2026-09-07. **Source/design preflight and five owned synthetic tests pass after
the pre-fit repairs below.** This review fitted no empirical model, computed no
predictive relationship, and read no numerical market values after 2025-10-20.
Only this new report and `tests/test_index_hinge_publication.py` were edited.
All earlier Python, reports, protocols and source files remain untouched.

## Source and measurement contract

The fixed target is raw SPX cumulative log price return over 21 or 63 observed
SPX sessions, `log(C[t+h]/C[t])`. It excludes reinvested dividends and risk-free
interest; it is neither an excess return nor an executable fund return. Using
SPX with VIX avoids an underlying-index mismatch, but does not align the VIX
30-calendar-day options construction with trailing 22-session OHLC variation.
The factor 252 is a convention. The feature is a log implied-versus-trailing-
proxy gap, not an observed variance risk premium.

Read-only hashing confirms the inputs still match their existing manifests:

| Input | SHA-256 |
|---|---|
| `data/research_paths/spx_daily.parquet` | `3958fbb1eb36689df1596c26b9f0e02e3f1d4fa3b0032381e5287a2a03697bf0` |
| Raw VIX CSV | `a34aabce269632f30904cf482986dd50b6d4cf51f2203dc51a0a9f460f3c90b2` |
| Raw VIX9D CSV | `0d6f600ee71bf6ffb5069d0c583cbe0b1ec97df6da4e4e439d5f0f22f5616abe` |
| Raw VVIX CSV | `f6bc726455fa3859c662875a005e97ba656b9536ae58fa6977ad24c6228e4f6a` |

The SPX manifest identifies Yahoo `^GSPC`, `auto_adjust=False`, and retrieval
on 2026-08-12. It contains 4,227 rows, including a 2025-10-21 row outside this
experiment. Only bounded date labels were loaded here: the usable reference
calendar has 4,226 dates from 2009-01-02 through 2025-10-20. The new loader
requests bounded OHLC columns and converts Cboe values only after filtering
`DATE`. Raw VIX/VIX9D use `CLOSE`; VVIX uses `VVIX`. No processed or filled IV
series is substituted. VIX9D's prelaunch 2011–October 2013 history remains an
archival back-calculation. None of these caches establishes historical revisions,
publication latency or immutable point-in-time vintages.

## Timing and identifiable model question

All numerical predictors end at the preceding observed session. The entry
weekday is known from the current date. Labels start at the current close and
end at the full horizon; their end also determines training availability.
Monthly refits select the first complete feature origin independently of future
label presence. Historical labels must end by that fit's preceding session.
The implementation preserves missing positions and complete trailing windows.

Date-only checks support 1,234 potential mature training origins for horizon 21
and 1,192 for horizon 63 at the 2016-01-04 fit, above the fixed minimum of 1,000.
Both begin 2011-01-05; their final eligible origins are 2015-12-01 and
2015-10-01, respectively, with last targets ending 2015-12-31. These are source-
date feasibility counts, not fitted samples or numerical-completeness claims.

The baseline keeps one representation each of `I` and `R`; it does not duplicate
them with log VIX or a monthly realized-log feature. Separate curvature and the
single gap hinge use the exact horizon-specific admitted training rows for
centering, then training-only population scaling. The fixed ridge objective and
same-row historical mean are consistent across arms. Zero-scale transforms
abort the family rather than selecting a replacement. The new hinge can add a
joint nonlinear basis beyond component levels and separate curvature. It adds
no independent source information and does not identify a causal risk-premium
mechanism. A redundant linear gap would not establish incremental information.

## Dependence, inference and publication

The YAML fixes four contrasts and all 103 accumulated hypotheses, wave alpha
`1/840`, both-phase relative MSE improvement of at least 0.25%, both evaluation
slices, and every nonoverlapping offset in both phases. Offsets use the full
bounded SPX calendar position modulo horizon, anchored at 2009-01-02, before
complete-sample filtering. Every one of the 21 or 63 offsets must be present
and improve; an omitted or empty offset cannot pass. Models share rows within
each horizon; horizons may have different boundary exclusions.

Blocks 126/252/504 exceed both target horizons. HAC504 and all block results
enter the conservative phase p-value, followed by the maximum across phases.
The 99,999 draws give minimum raw bootstrap p-value `1/100000`. The first
wave-Holm cutoff is `1/3360`; its one-tenth resolution bound is `1/33600`.
The 504-session block occupies a large
part of the development window, and quarterly targets supply few independent
periods. Conservative gates and every-offset signs do not remove that power,
dependence, historical-reuse or source-vintage limitation.

The review identified and the new runner repaired two issues before any fit:

- Prior wave-five measurement labels were dropped when importing hypotheses.
  Optional `measure` and explicit `source_row_index` now preserve distinct
  identities alongside source hashes. The synthetic standalone prior-file
  fixture also prompted a robust pathname fallback.
- A 504-row phase would silently reduce the reused HAC helper to 503 lags.
  The new protocol and runner require at least 505 phase observations, with
  synthetic rejection checks at 503 and 504. No historical phase was changed.

The four publication-fault tests were written before the runner existed. They
now pass for report failure, partially written evaluated ledger, partially
written registration ledger, and a changed protocol hash. Each case leaves
canonical `UNEVALUABLE` metrics and `failure.json` with all four p-values equal
to one and no passing horizon. Available source, fit and row artifacts remain;
completed scores are retained only as explicitly unpublished diagnostics. The
fifth test confirms the distinct identities of all 15 prior measurement
contrasts. Scoped lint also passes. The guarded initial ledger writes and
protocol recheck introduced before this wave remain effective.

Source hashes, runtime/backend metadata and all Python dependencies are captured
by the new runner manifest. Independent numerical reconstruction and inference
verification remain required after the authorized run. This review is not an
empirical validation or permission to relax a failed gate.
