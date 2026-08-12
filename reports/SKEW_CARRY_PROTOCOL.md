# SKEW-conditioned carry repair — frozen mechanism diagnostic

Written before downloading or scoring SKEW. The machine-readable contract is
`skew_carry.yaml`.

## Why this is allowed, and what it cannot establish

The original carry rule failed for a specific mechanism: implied variance
looked rich relative to a backward-looking HAR forecast immediately before the
COVID transition, so the rule selected three of the five worst overlapping
trades in the decade. SKEW is not being tried as another generic predictor of
realized variance. It is tested only as a crash-risk veto on that already-fixed
richness rule.

This is necessarily post hoc on 2016-2025 because the observed February 2020
failure motivated the repair. The clean window has also already been inspected
and contains no comparable transition. Therefore a pass is a mechanism result,
not out-of-sample evidence, and no p-value can promote it.

It also inherits two limitations from `src.carry`: the richness cutoff is the
median over the full diagnostic window, and the carry frame uses the same-date
published VXN daily close even though the repository's later, stricter timing
policy delays Cboe daily closes one session at a 16:00 origin. Those choices are
preserved here so the SKEW veto is compared with the rule that actually failed,
but they prevent this from being called a leakage-free strategy backtest.

## One fixed rule

Preserve the original carry signal and its diagnostic-window median threshold.
Take an otherwise eligible trade only if the latest available Cboe SKEW close
is no higher than its trailing 252-session 80th percentile. The percentile is
estimated from at least 126 observations and is shifted one observation behind
the SKEW value being judged.

The forecast origin is the 16:00 ET QQQ close. Cboe's published daily close is
delayed one complete trading session, matching the conservative timing rule in
the completed orthogonal-signal study. Missing observations mean flat; they are
never forward-filled.

SKEW is constructed from SPX options and is only a market-regime proxy for NDX.
It measures 30-day option-implied statistical skewness, not a simple put/call
volatility ratio. Cboe consulted on replacing the methodology in 2025 and said
a modified history could be recalculated. The fetch therefore requires the
downloaded file to retain Cboe's published 2018-08-13 close of 159.03 and records
the source hash.

## Fixed diagnostic criterion

Use non-overlapping 21-session trades at every phase offset. A mechanism pass
requires all of the following:

- reject the three previously selected adverse origins: 2020-02-18,
  2020-02-19, and 2020-02-21;
- no worse average 5% CVaR than richness-only;
- no worse average maximum drawdown than richness-only;
- positive average P&L; and
- retain at least 70% of richness-only trades.

There is no threshold search, alternative SKEW transform, VVIX runner-up, or
combination sweep. If SKEW fails, the repair fails. VVIX remains a separate
future hypothesis rather than a second chance on the same data.

Any forward version must freeze the historical richness cutoff at +0.386812814
without refitting and use either one-session-lagged Cboe values or timestamped
pre-close quotes available before entry. It must not use the current official
daily close at a 16:00 decision time.
