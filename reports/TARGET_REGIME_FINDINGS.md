# Changing the target and modeling transitions

Both original frozen diagnostics failed their primary criteria. A subsequent
frozen repair qualifies the HMM result: calibration succeeds, but calibrated
state probability still adds nothing to a supervised model using the same
training rows. The jump study remains a five-minute proxy test rather than a
formal BNS jump-statistic test.

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

This result did **not** use the repository's seven-bars-per-session yfinance
history. Oxford-Man's `rv5` is realized variance from five-minute returns and
its `bv` is five-minute bipower variation; `hourly_bars.parquet` is never read
by the jump module. The resolution objection therefore does not invalidate this
run. The result is nevertheless proxy-specific: bipower variation exceeded
realized variance on 16.9% of raw days, requiring the pre-specified non-negative
truncation. The
annual-training threshold entering 2017 was 0.450, while 2017's realized
jump-share 90th percentile fell to 0.246; only 2.0% of 2017 origins realized a
five-session event. A formal BNS jump indicator with realized quarticity is a
better future target than treating every positive `RV - BV` estimate as a jump.

The field definitions are documented as [five-minute RV and five-minute
bipower variation](https://search.r-project.org/CRAN/refmans/bvhar/html/oxfordman.html),
and the underlying BNS result establishes `RV - BV` as a jump-component
estimator under its assumptions ([Oxford research record](https://ora.ox.ac.uk/objects/uuid%3A2eac6b4f-bda9-4195-b119-2e7098ab4aeb)).

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

The direct HMM-versus-logistic contest was not the right incremental question,
and its proper-score loss mixed ranking with calibration. A repair was therefore
frozen before inspecting this target after 2025-11-03. The project has used
those dates elsewhere, so this is target-specific—not pristine project-wide—
confirmation. Platt, benchmark, and augmented-model parameters used only annual
out-of-fold rows through 2024; HMM parameters and the stress threshold were
also fixed through 2024 with no holdout refit.

| model | phase-average Brier | phase-average log loss |
|---|---:|---:|
| raw HMM | 0.218189 | 0.673834 |
| Platt-calibrated HMM | **0.205309** | **0.604679** |
| supervised benchmark | **0.195725** | **0.578219** |
| benchmark + calibrated HMM | 0.196617 | 0.579151 |

Calibration verdict: **PASS**. The original HMM loss was partly a probability-
scaling problem. The calibrated HMM retained useful ranking in the 166-origin
holdout: 54.5% events in its top quintile versus 18.2% in its bottom.

Incremental-state verdict: **FAIL**. Adding the calibrated HMM probability made
Brier worse by 0.000892 and log loss worse by 0.000933. The differences are
small, so this is evidence of no detectable increment—not evidence that the
latent state contains literally zero information. The supervised RV features
already absorb its useful ranking. This is the fair negative: the HMM is a
helpful state summary but did not add advance transition information.

## Move the correlation-premium target to SPX

The asset correction is accepted: the dispersion study should be SPX, not QQQ.
The repository's COR1M series is not an NDX implied-correlation measure. Cboe
defines its implied-correlation family from the top 50 SPX stocks and SPX and
component option volatilities. COR1M, DSPX, and VIXEQ therefore belong in an SPX
study, serving both a correctly specified target and cross-sectional replication.

Cboe's DSPX is also forward-looking implied dispersion, not realized
dispersion. It can be an implied input, but it does not supply the missing
realized target.

An honest 30-day SPX correlation-premium target requires, point in time:

1. the exact eligible top-50 SPX basket and its rebalance dates;
2. origin-date weights and corporate-action-safe constituent identifiers;
3. constituent returns for the full forward 30-calendar-day window; and
4. a realized weighted average correlation constructed on the same basket and
   convention as the implied index.

The implied side is already partly local: COR1M begins in 2006 and VIXEQ is
available for 3,054 sessions. The missing object is the point-in-time top-50 SPX
tracking basket and corporate-action-safe return panel. The local
`pit_weights.parquet` is a survivorship-biased 13-name QQQ proxy. The
N-PORT archive starts in late 2019, is delayed, lacks a complete ticker-return
panel, and describes QQQ rather than COR1M's SPX basket. Those sources cannot be
used to manufacture this target.

Official context: [Cboe describes implied correlation as the expected average
correlation of the top 50 SPX stocks](https://www.cboe.com/us/indices/implied/),
while [DSPX is explicitly a forward-looking implied measure rather than
realized dispersion](https://www.cboe.com/us/indices/dispersion/).

## Decision

- Do not tune the HMM or SKEW jump model on these windows.
- The jump experiment already uses five-minute Oxford-Man estimators. Its next
  version needs the underlying returns or a supplied realized-quarticity series
  for a formal BNS statistic. NQ GLBX data is the clean paid path; do not use the
  seven hourly QQQ bars for jump inference.
- The correctly matched SPX correlation premium is specified in
  `SPX_DISPERSION_DATA_CONTRACT.md`. It remains data-blocked pending the exact
  historical tracking basket and constituent return panel, not conceptually
  blocked by the NDX mismatch.
- Do not tune the repaired state feature. Platt calibration worked; incremental
  state information did not. A future state model needs a new asset or future
  target-specific window.
