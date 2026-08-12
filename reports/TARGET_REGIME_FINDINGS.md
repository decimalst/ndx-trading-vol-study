# Changing the target and modeling transitions

Both frozen diagnostics failed their primary criteria. They still answer the
feedback in a useful way: a different target is not automatically predictable,
and a latent-state model is much better at describing regimes than anticipating
their arrival.

## SPX jump target: different object, no surface win

The static Oxford-Man archive supplied 4,641 SPX sessions from 2000-01-03
through 2018-06-27. The study used 2004-2013 for initial training and scored
1,001 untouched-for-this-target origins in 2014-2017. Every Cboe close was
delayed one session.

The target was a five-session material-jump event derived from
`jump_share = max(rv5 - bv, 0) / rv5`, with the event threshold estimated only
in each prior annual training fold.

| model | phase-average Brier | phase-average log loss |
|---|---:|---:|
| jump history | 0.214213 | 0.616525 |
| history + ATM VIX | **0.213344** | **0.614335** |
| history + VIX + SKEW | 0.217646 | 0.625756 |

Strict verdict: **FAIL**. SKEW worsened both registered losses versus ATM. ATM
itself improved Brier only 0.4% versus jump history, and its probabilities were
correlated 0.9997 with the history-only model. Changing the target removed most
of the ATM level's dominance, but did not uncover a useful free surface signal.

The target proxy is also unstable. Bipower variation exceeded realized variance
on 16.9% of raw days, requiring the pre-specified non-negative truncation. The
annual-training threshold entering 2017 was 0.450, while 2017's realized
jump-share 90th percentile fell to 0.246; only 2.0% of 2017 origins realized a
five-session event. A formal BNS jump indicator with realized quarticity is a
better future target than treating every positive `RV - BV` estimate as a jump.

The source is commit-pinned and content-hashed at
`e0dd80edc0c2cedac5ed3f72250ee4460e963b4efd458d68525a61bcc5c27ea2`.
The original branch URL returned 404 before any bytes or outcomes were accepted;
the source-only amendment is recorded in `TARGET_REGIME_PROTOCOL.md`.

## Two-state QQQ model: strong ranking, worse probabilities

The HMM and benchmark used the same QQQ log-RV history. Parameters were refit
annually on prior data. HMM probabilities were forward-filtered only; no
full-sample state labels or smoothed probabilities entered a forecast. The
target was entry from a calm origin into the annual-training 80th-percentile
stress region during the next five sessions.

| model | phase-average Brier | phase-average log loss |
|---|---:|---:|
| supervised HAR-state logistic | **0.126650** | **0.420496** |
| two-state Gaussian HMM | 0.137616 | 0.457171 |

Strict verdict: **FAIL**. The HMM lost both registered probability scores. It
did rank risk: its top probability quintile realized a 61.7% event rate versus
2.9% in the bottom quintile. But ranking is not enough when the probabilities
are systematically less calibrated than a direct target model.

Most importantly, the HMM did not rescue the transition years. It beat the
logistic benchmark on Brier only in 2016; it lost in 2020 and in 2022, when
stress-entry rates were 45.4% and 80.8%. The estimated latent states are highly
persistent, so the filter recognizes stress rapidly after RV changes but does
not create advance information about the switch. Regime switching is therefore
a useful descriptive frame here, not an orthogonal predictive signal.

Calibrating the HMM after observing this result or adding covariate-dependent
transition probabilities would be a new supervised model search. Neither is a
valid repair on this window.

## Why the correlation-premium target is not scored yet

The repository's COR1M series is not an NDX implied-correlation measure. Cboe
defines its implied-correlation family from the top 50 SPX stocks and SPX and
component option volatilities. Subtracting a QQQ realized-correlation estimate
from COR1M would mix different universes and would not be a correlation premium.

Cboe's DSPX is also forward-looking implied dispersion, not realized
dispersion. It can be an implied input, but it does not supply the missing
realized target.

An honest 30-day SPX correlation-premium target requires, point in time:

1. the exact eligible top-50 SPX basket and its rebalance dates;
2. origin-date weights and corporate-action-safe constituent identifiers;
3. constituent returns for the full forward 30-calendar-day window; and
4. a realized weighted average correlation constructed on the same basket and
   convention as the implied index.

The local `pit_weights.parquet` is a survivorship-biased 13-name QQQ proxy. The
N-PORT archive starts in late 2019, is delayed, lacks a complete ticker-return
panel, and describes QQQ rather than COR1M's SPX basket. Those sources cannot be
used to manufacture this target.

Official context: [Cboe describes implied correlation as the expected average
correlation of the top 50 SPX stocks](https://www.cboe.com/us/indices/implied/),
while [DSPX is explicitly a forward-looking implied measure rather than
realized dispersion](https://www.cboe.com/us/indices/dispersion/).

## Decision

- Do not tune the HMM or SKEW jump model on these windows.
- The best next target experiment is QQQ five-minute RV/BPV/BNS using a stable,
  full-tape source with realized quarticity. The free HF Data Library exposes
  those measures, but requires a free API key and has a documented March-2022
  full-tape-to-IEX structural break; a protocol must stay entirely on one side
  of that boundary. [Source and cleaning caveats](https://hfdatalibrary.com/pages/docs).
- The correlation premium is economically cleaner than the current jump proxy,
  but only after acquiring the matching point-in-time SPX basket and return
  panel. Until then it remains data-blocked, not model-blocked.
- If a state model is revisited, validate it on a new asset or future period and
  compare against a directly supervised transition benchmark. Do not score it
  only against HAR's conditional-mean variance forecast.
