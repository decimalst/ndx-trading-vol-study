# Spread-selling safety: prospective payoff and account engine

This additive component implements the user's spread-selling question. It is
not a historical run, option-price model or amendment to a frozen study. The
earlier futures sizing draft was superseded before implementation or execution;
its design and unrun test text remain under `reports/copula_risk_backtest/`.
No orphan draft test remains in unittest discovery. All existing frozen sources
and reports remain unchanged.

## Question and instruments

Can an issued joint-return model identify sessions when a defined-risk spread
portfolio has sufficiently low predicted short-strike breach probability to
sell? Fixed marginal forecasts make single-asset terminal breach probabilities
the same within a copula pair. Dependence can change the chance that either or
both assets breach and the portfolio liability distribution. This motivates an
explicit two-asset test instead of attributing a univariate probability to the
copula.

The saved outcomes are **raw log(close/open)** for QQQ ETF and SPX price index
on the next actual SPX session. They are not SPY, NQ/ES fills or option prices.
The engine uses the open normalized to one and terminal intrinsic value at the
next-session close. This is an expiration-liability proxy. It does not model
actual QQQ or SPX contract selection, synchronized settlement, exercise style,
early assignment, bid/ask spread, entry quotes, marks, exchange fees, margin,
contract multipliers or integer lots. No claim about executable spread profit,
intraday stop performance or live margin survival follows from this engine.

Features stop at the preceding reference session of the forecast origin; the
target is the next reference session after that origin. The extra inherited lag
is retained. Both origin and target must be in the declared phase. Numerical
outcomes end by 2025-10-20; later protected arrays are not needed. The root
runner authenticates the saved archive and fixes the historical protocol before
any empirical application.

## Bounded terminal payoff

`terminal_debit(log_returns, structure, distance=.02, width=.01)` accepts any
nonempty numeric array shape, including a scalar array, and returns the same
shape of per-asset intrinsic debit divided by wing width. Structure is exactly
`put`, `call`, or `condor`.

With normalized terminal price `S=exp(y)`, the put shorts `1-distance` and buys
`1-distance-width`; the call shorts `1+distance` and buys
`1+distance+width`. Each vertical's liability is the difference of its two
intrinsic values. The condor combines mutually exclusive put and call wings,
so its maximum terminal liability is **one** wing width, not two. Distances
.01/.02/.03 and width .01 are fixed in the account engine; .02 is primary and
the others are sensitivities.

The helper compares finite log returns to log strikes first. It returns exact
zero or one outside the interior interval and evaluates `expm1` ratios only
between the two nearby strikes. Thus even finite log returns near the float64
extremes do not cause exponential overflow. There is no distribution-tail
clipping or numerical conversion of extreme outcomes to an arbitrary finite
price. Short-strike equality has zero intrinsic loss and is not a breach;
long-strike equality is a full-width loss. The helper rejects malformed
numbers, nonpositive width and nonpositive put long strikes.

This bounded payoff also avoids interpreting exponential moments of an
unbounded Student-t or transformed log-return distribution as finite. Such
models need not have finite arithmetic-return means or variances. Their
bounded spread liability, expected debit and ES remain meaningful; no Kelly
calculation is introduced.

## Inputs and fixed safety policies

`run_backtest(risks, realized, calendar, config=None)` returns exactly four
DataFrames: `positions`, `outcomes`, `path`, and `summary`.

The risk frame's exact columns are `origin`, `target_end`,
`feature_cutoff_date`, `phase`, `model`, `structure`, `distance`, `width`,
`p_any_breach`, `mean_debit`, `es97_5`. The last two values describe the mean
and 97.5% ES of the equally weighted, width-normalized portfolio debit; neither
is an additional trading gate. Probabilities and liabilities lie in [0,1],
and ES cannot be smaller than mean debit apart from a 1e-12 arithmetic
tolerance. The root producer specifies deterministic common random sampling
and independently verifies its forecasts before these inputs are admitted.

The exact six cells are `orig_gaussian`, `orig_t8`, `cal_gaussian`, `cal_t8`,
`shape_gaussian`, `shape_t8`. Every origin must contain all six models in all
nine structure/distance cases, with the same target, cutoff and phase. No
cell, sensitivity, or case is silently discarded. The seventh strategy is
`always_sell`, synthesized as a control without a predictive-risk input.

The realized frame is exactly `origin`, `target_end`, `y_qqq`, `y_spx`, in
unique increasing origin order. It must match all risk origins and target
dates exactly. Missing outcomes are not treated as zero loss, cash returns
or successful trades. There is no instrument alias from SPX to SPY.

Each model sells if and only if **p_any_breach <= .10**. Probability equality
passes. The aggregate gross-width liability reserve is 2% of current equity,
split equally between the two assets, whether a put spread, call spread or
condor is sold. If blocked, exposure, credit, liability and costs are zero.
Mean debit and ES are diagnostics, not sizing inputs or secondary thresholds.
The planned positions are constructed before the realized-outcome frame is
read, remain independent of it, and are identical across premium scenarios.

## Hypothetical credit accounting

For each case and policy, report all three predetermined credits .05/.10/.20
of wing width. The round-trip fee is .02 of width. A condor's stated credit is
the aggregate credit for both wings, not a credit for each wing separately.
These are explicitly hypothetical sensitivity assumptions, not observed
premiums, fees, break-even quotes or evidence of option mispricing. The
reserve is based on gross width before credit, so a different assumed premium
does not increase position size.

Let `b=.02` be aggregate reserve, `q` credit fraction, `f=.02` fee fraction,
and `D=(debit_qqq+debit_spx)/2`. When sold, equity return is
`b*(q-f-D)`; dollar credit, liability and fee use that session's starting
equity. When blocked the return is zero. The proxy account is closed at the
session end, with no overnight position and cash return zero. Each phase,
structure, distance, policy and credit scenario is a separate counterfactual
account starting at one; the nine cases are not simultaneously traded with
an aggregate 18% reserve. Fixed-case comparisons share the same outcomes.

The path includes planned and executed reserve, credit/liability/fee returns
and P&L, compounded equity, initial-inclusive running peak and drawdown.
Nonpositive equity would be retained, flagged as ruin and made absorbing,
with all subsequent rows visible and execution inactive; no clipped loss or
artificial recovery occurs. Default reserve and premium/fee assumptions
cannot produce one-session bankruptcy because liability is bounded.

Avoided liability, missed credit and saved fee are per-session return
contributions relative to always selling the same reserve. Their identity is
`avoided_liability - missed_credit + saved_fee = strategy_net_return -
always_sell_counterfactual_return`. This is not an additive decomposition of
differently compounded terminal wealth, nor a claim that skipped contracts
would have been available at the hypothetical credit.

## Output interpretation and validation

`positions` is one row per origin/case/policy, before premium assumptions or
outcome use. `outcomes` contains the shared per-asset/portfolio debit and the
distinct any/both short-breach and any/both full-loss flags. `path` expands
positions over all credits with account accounting and attribution.
`summary` includes sold coverage, sold-session mean debit and joint event
rates, capital results, costs and attribution for all policies/scenarios.
Undefined sold-session means for a policy that sells nothing remain missing,
not favorable zeroes. First/last origin and actual target dates remain visible.
No annualization uses a compressed count of retained observations.

Default configuration keys are `breach_threshold`, `max_liability_budget`,
`credit_fractions`, `fee_fraction`, `development`, `evaluation`, `source_end`.
Known-key overrides permit generated fixtures; the actual protocol must pin
the values above and development2016-01-04..2019-12-31 /
evaluation2020-01-01..2025-10-20. Wider numerical ceilings are rejected.

The ten tests were written before implementation, and `RED.log` records the
actual missing-module failure. All ten passed the first implementation in
4.178s. They use invented values only and check an independent dollar
intrinsic oracle, extreme log returns, boundaries, Decimal credit accounting,
inclusive filtering, no ES/mean sizing, future-outcome invariance, input
immutability, compound equity/drawdown, exact attribution, joint event flags,
complete models/cases, source clocks and malformed input rejection.
`GREEN.log` records the final scoped checks and code hashes.

This is an exploratory application of reused history and current-vintage
forecasts. Filtering can reduce observed losses merely by selling less, so
coverage, the always-sell control and foregone credit are essential. Joint
tail calibration and model comparisons require independent, dependence-aware
analysis specified by the caller. No empirical execution, new source fetch,
options-profit ranking, statistical promotion, or frozen-file modification
is authorized by these synthetic checks alone.
