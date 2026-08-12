# Orthogonal-signal protocol — frozen before implementation

Written 2026-08-12 before any signal code, data fetch, discovery score, or
confirmation score. The machine-readable contract is `signal_study.yaml`.

## Why this is a separate study

The original clean window has already been inspected and is frozen until its
pre-committed gate. This study cannot read or score a clean origin. It uses an
older diagnostic period split once into discovery (2016–2021) and confirmation
(2022–2025-10-17). Discovery may select at most one candidate. Confirmation is
spent once on that locked candidate, or not at all if discovery finds no
improvement.

## Information timing

The forecast origin is the 16:00 ET QQQ close. Same-session QQQ and cross-asset
ETF closes are available at the origin. Cboe volatility indexes continue to be
calculated through 16:15 ET; their published daily closes are therefore delayed
one full trading session. Missing values are never forward-filled.

This timing rule also tightens the baseline: the study uses VXN(t-1), not the
same-date daily close used by the original harness. The original reports remain
immutable; the new baseline and candidates are compared only within this study.

## Fixed candidates

1. `term_slope`: safe HAR-IV-LEV plus lagged `log(VIX9D/VIX)`.
2. `cross_asset`: safe HAR-IV-LEV plus one cross-asset stress composite built
   from HYG, TLT, GLD, USO, and UUP close-to-close returns.
3. `combined`: both additions.

The cross-asset composite is a single feature, not five opportunities to tune.
Each return is standardized with trailing data whose scale estimate ends at
t-1; the feature is the root mean square across assets.

## Deferred ideas

- Dealer gamma requires historical strike/expiry-level open interest. It will
  not be approximated from the underlying or implied-vol level because neither
  identifies position sign or inventory size.
- Exact historical NDX weights are a licensed Nasdaq data product. Public QQQ
  Form N-PORT filings can support a delayed quarterly proxy from late 2019,
  provided each snapshot becomes usable only on its filing date. Weight data
  are audited separately and do not enter this holdout.
- Historical observed weather is available, but it is revision-prone, its
  same-day availability is inconsistent by station, and the causal prior for
  next-day NDX variance is weak. It does not spend this first holdout.

## Test-before-run rule

Both `make signals-discover` and `make signals-confirm` depend on
`make test-signal-safety`. Tests cover the clean-window fence, source lags,
future-target invariance, missing-data behavior, and discovery-lock integrity.

