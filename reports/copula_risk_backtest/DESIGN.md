# Leveraged intraday risk-sizing proxy: prospective engine contract

This additive component tests risk sizing on existing forecast information. It
does not change a frozen study or authorize historical execution. The caller must
first freeze the actual protocol, source identities, forecast construction and
independent verification. The repository was on `main`, tracking `origin/main`,
with existing `.gitignore` and `Makefile` changes; those changes are preserved.
Implementation and generated checks use Python 3.11.7 in `.venv`.

## Instrument and clock

The saved model predicts next actual SPX-session **raw log(close/open)** returns
for QQQ ETF and SPX price index. It does not predict SPY ETF or actual NQ/ES futures
contract returns. This engine therefore reports a 50/50 QQQ/SPX leveraged proxy.
It must not be labeled an executable futures, options, or SPY backtest. The
existing SPY source-discovery material is quarantined, not an admitted substitute.
No new data acquisition or protected later numerical window is required.

For each origin, the feature cutoff is the preceding full reference session and
the traded target is the following full reference session. The engine retains
that extra lag; it does not use the origin's close to improve a decision. Both
origin and target must be inside their declared phase, and numerical outcomes
end no later than 2025-10-20. Native, unique, increasing calendar dates are
required. The caller authenticates the entire issued forecast archive and
handles unscored/missing origins explicitly; this engine rejects missing or
extra realized rows rather than filling missing returns with cash or zero.

## Pure interfaces

`size_positions(risks, calendar, config=None)` accepts no realized outcomes and
returns a long positions frame. `run_backtest(risks, realized, calendar,
config=None)` returns `positions`, `path`, and `summary` DataFrames. All inputs
remain unchanged. No market data readers or production fitting functions are
imported.

Risk columns are exactly `origin`, `feature_cutoff_date`, `target_end`, `phase`,
`portfolio_vol`, and six ES estimates: `es_orig_gaussian`, `es_orig_t8`,
`es_cal_gaussian`, `es_cal_t8`, `es_shape_gaussian`, `es_shape_t8`.
Realized columns are exactly `origin`, `target_end`, `y_qqq`, `y_spx`.
Dates are native pandas dates; risk and return columns are real, nonboolean
numeric columns. The original origin order and exact target join are retained.

The conventional volatility input is the standard deviation of the basket's
preceding 252 mature realized **simple** returns, constructed upstream. The
six ES inputs are one-unit portfolio downside ES at 97.5%, with identical
marginals within each copula pair, built upstream with predetermined common
random samples. They measure the nonnegative loss
`max(1 - (exp(y_qqq)+exp(y_spx))/2, 0)`. The engine does not infer tail risk from
future outcomes or select a better-looking cell.

Unbounded Student-t log returns need not have positive exponential moments.
Consequently simulated arithmetic-return means or variances must not be
interpreted as finite population moments or used for Kelly sizing. The
long-only basket's nonnegative downside loss is bounded, so this downside ES
does not have that problem. QMC zero downside loss is not proof of zero true
tail risk; upstream sampling precision remains a separate validation issue.

## Fixed sizing and costs

The nine strategies are `fixed_1x`, `fixed_2x`, `vol_target`, and the six ES
column names above. Fixed strategies provide 1x and 2x equal-weight exposure.
Volatility sizing is `min(2, (.10/sqrt(252))/portfolio_vol)`. ES sizing is
`min(2, .01/ES)`. A finite exact zero estimate maps to the cap without epsilon
or division; negative or unknown estimates fail. No shorts, fitted mean bet,
Kelly sizing, constant-exposure fitting, or post-result parameter search occurs.

Configuration defaults are `annual_vol_target=.10`, `annualization=252`,
`es_budget=.01`, `gross_cap=2`, `cost_bps=[0,5,10]`, development
2016-01-04 through 2019-12-31, evaluation 2020-01-01 through 2025-10-20, and
`source_end=2025-10-20`. Partial known-key overrides support generated tests;
the root protocol must pin actual settings. The cap remains exactly two. The
5-bp one-way scenario is primary; 0 and 10 bp are fixed proxy sensitivities.

The account starts each phase with equity one and is flat overnight. If `g` is
gross exposure and `r=(expm1(y_qqq)+expm1(y_spx))/2`, gross return is `g*r`.
For one-way proportional cost `c`, entry cost is `c*g`, exit cost is
`c*g*(1+r)`, and net return is `g*r-c*g*(2+r)`. Exit notional is marked to
the realized close; this is not the `2*c*g` approximation. Constant exposure
still pays both legs every session. Positions are fractional proxy notionals;
cash yields zero. Equity compounds actual net returns and drawdown includes
the initial equity peak. These are not actual exchange fees or an assumption
that fees are charged this way on futures contracts.

If net equity becomes zero or negative, the uncapped loss and nonpositive
equity remain visible; the account is marked ruined and future executed
positions are zero. All later scheduled rows remain as `inactive_after_ruin`.
Planned target-blind positions are retained separately. The next phase starts
a new account. No clipping creates artificial solvency or recovery.

The engine does not model futures contract multipliers, integer lots, basis,
rolls, collateral or interest, intraday variation margin, historical margin
requirements, liquidation slippage, options premium/Greeks/expiry, or a stop
path. Daily open/close data cannot validate stop execution or intraday margin
survival. Actual target dates remain in outputs; no retained-row annualization
or compression of calendar gaps is performed.

## Saved accounting and limits

Each path row retains planned and executed exposure, starting and ending
equity, gross and net returns/P&L, both costs, peak/drawdown and ruin status.
Four nonnegative return contributions versus fixed 1x are missed upside,
saved loss, extra upside and extra loss. Together with signed cost savings,
`saved - missed + extra_upside - extra_loss + cost_saving` equals the
strategy's net return minus the fixed-1x same-session counterfactual. These
are per-session return decompositions, not an additive decomposition of
differently compounded terminal wealth. After ruin, attribution compares the
inactive account's zero session return with that counterfactual.

Summaries report retained session counts/dates, total return, maximum
drawdown, average planned/executed exposure, total costs/P&L, worst session
return and ruin. They do not claim a Sharpe ratio, statistical superiority or
an annually comparable return from compressed observed rows. All nine arms
and all cost scenarios remain visible. Dependence-aware comparisons, reused
window qualifications and multiplicity are caller-owned. Better downside
outcomes can result simply from less exposure, so fixed controls and upside
opportunity costs are essential to interpretation.

## Prewritten generated checks

The test file is written before the implementation. Its ten tests use invented
dates/returns only: closed-form sizing and zero-risk cap, Decimal cash-flow
oracle, correct log-to-simple conversion, daily round trips, cost monotonicity,
phase-reset compounding/drawdown, exact attribution, outcome-blind positions,
an underestimated-risk stress with a fixed cap, ruin without recovery,
strict calendar and matching outcomes, malformed numbers/identities, schema
and source-ceiling validation, and input immutability. Actual RED and GREEN
receipts are retained beside this design. These checks validate this numerical
engine, not the quality or live tradability of any historical signal.
