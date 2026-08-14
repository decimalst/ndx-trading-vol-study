# Diagnostic-only orthogonal-signal study

*Split out of `README.md` on 2026-08-13 to keep the top-level document short.
Nothing here changed in the move; it is the same text.*

## Diagnostic-only orthogonal-signal study

The independently frozen protocol is in
[`reports/ORTHOGONAL_SIGNALS_PROTOCOL.md`](../reports/ORTHOGONAL_SIGNALS_PROTOCOL.md).
It never reads the original clean window. Safety tests run before both empirical
stages:

```bash
make fetch-signal-inputs
make signals-discover            # tests first; locks at most one candidate
make signals-confirm             # tests first; spends confirmation once
make verify-signals              # independent metric recomputation
```

Seven combinations of term structure, cross-asset stress, and QQQ market state
were evaluated in discovery. Lagged `log(VIX9D/VIX)` won discovery, improved
sealed confirmation QLIKE by 2.92%, but **failed confirmation** (DM p=0.1016).
Its improvement decayed from 5.3%/7.0% in 2022-2023 to 0.2%/0.9% in 2024-2025,
making regime-conditional front-end dislocation a prospective-only hypothesis.
See [`reports/signal_study/verification.md`](../reports/signal_study/verification.md)
and the [weight/source audit](../reports/HISTORICAL_WEIGHTS_AND_SIGNAL_BACKLOG.md).

Historical QQQ holdings can be refreshed independently with
`make fetch-nport-weights`. The target runs nine parser/integrity/as-of tests
first, then writes 27 SEC filing snapshots plus a quarterly concentration
summary. These are not exact daily Nasdaq-100 weights and are not fed into the
completed signal holdout. Top-10 weight fell 55.13% to 46.85% and HHI fell
0.0459 to 0.0312, with a sharp step after the 2023 anti-concentration rebalance.
That adverse direction, flat earnings win rates, and the 2025 low effectively
close the earnings-concentration defence on existing project data.

An experimental 2004-2018 annual-report backfill plus official Nasdaq
membership parser was started test-first and then paused before a complete
dataset was produced. It remains disconnected from every model. See
[`reports/PAUSED_HISTORICAL_WEIGHTS.md`](../reports/PAUSED_HISTORICAL_WEIGHTS.md)
for the exact passing contracts, parser repairs, SEC 403 stop, and safe restart
conditions.

The failed carry rule also has one frozen post-hoc mechanism diagnostic:
`make skew-carry`, documented in
[`reports/SKEW_CARRY_PROTOCOL.md`](../reports/SKEW_CARRY_PROTOCOL.md). It tests a
single lagged Cboe SKEW veto and mechanically excludes clean origins. Its strict
verdict is FAIL because it retained 63.3% of eligible trades versus a registered
70% floor, despite rejecting the three known pre-COVID adverse entries and
improving the descriptive tail aggregates. This is not a validated strategy.
The participation gate was a poor proxy for the risk-adjusted outcome; the
decisive limitation is that February 2020 motivated the rule and remains in its
sample.

The resulting ranked research program, verified public cross-asset/surface
coverage, and pre-run safety gate are recorded in
[`reports/NEXT_RESEARCH_PROGRAM.md`](../reports/NEXT_RESEARCH_PROGRAM.md).

Two target/model-frame diagnostics are frozen in `target_regime.yaml`:

```bash
make fetch-jump-target       # tests first; commit-pinned Oxford-Man SPX source
make jump-target             # 2014-2017 jump-event comparison
make regime-transition       # forward-filtered two-state QQQ diagnostic
make regime-repair           # Platt calibration + incremental-state holdout
make verify-target-regime    # targets/timing/metrics re-derived from source;
                             # forecasts NOT independently refit (see banner)
```

The SPX jump surface comparison failed. It used Oxford-Man five-minute RV and
bipower variation—not the local hourly bars—but still lacks the formal BNS
quarticity statistic. The original HMM comparison was calibration-confounded:
Platt scaling improved both holdout scores. The fair incremental test still
failed; adding the calibrated state to the same-row supervised benchmark made
both losses slightly worse. A correctly matched correlation-premium study is
specified for SPX and remains data-blocked. See
[`reports/TARGET_REGIME_FINDINGS.md`](../reports/TARGET_REGIME_FINDINGS.md) and the
[`SPX data contract`](../reports/SPX_DISPERSION_DATA_CONTRACT.md).

