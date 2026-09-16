# Overnight index returns: feasibility and a fixed next-wave design

2026-09-06. Read-only feasibility work, apart from this new report. No overnight
features were scored, no forecasts were fitted, and no current-wave performance
was examined. Earlier protocols and code were searched for duplicate targets.

**A historical adjusted-overnight forecast test is feasible. An exact cash
overnight trading test is not yet supported by the cached fields alone.** The
existing QQQ file includes OHLC and adjusted close but omits corporate-action
amounts, declaration timestamps, payment dates, and historical quote/fill data.
Its current-vintage prices are not a point-in-time archive.

## What can be reconstructed

Let `C_t`, `O_t`, and `A_t` denote vendor close, open, and adjusted close, and
define `F_t=A_t/C_t`. The one-night adjusted-return proxy is

```
g_ON[t+1] = log(A[t+1]/A[t]) - log(C[t+1]/O[t+1])
         = log(O[t+1]/C[t]) + log(F[t+1]/F[t])
y_ON[t]  = exp(g_ON[t+1]) - 1
```

The corresponding day component is `log(C/O)`. The two log components exactly
sum to adjusted close-to-close return. An apparent dependence on the next day's
closing price cancels algebraically if the adjustment factor is held fixed.
This is the same decomposition convention used by [Lou, Polk, and Skouras
(2019)](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf), who explicitly
allocate corporate events to the overnight component and also inspect results
excluding dividend months. Their evidence concerns firm-level returns and
portfolios; applying it to QQQ is a new adaptation.

The local `yfinance.utils.auto_adjust` implementation likewise multiplies OHLC
by `Adj Close / Close`. Calling that routine would reproduce the algebra, not
independently validate corporate actions or open prices. `src/fetch.py` retains
only OHLC, adjusted close, and volume, so the action metadata cannot be audited
from this parquet alone.

[Yahoo's own adjustment description](https://in.help.yahoo.com/kb/adjusted-close-sln28256.html)
uses multiplicative historical dividend adjustments based on the previous close.
For a cash dividend `D` with no split, its convention implies adjusted overnight
gross return `O_next/(C-D)`. Actual price-plus-dividend-entitlement gross return
is `(O_next+D)/C`, before payment-delay treatment. They agree on a pure dividend
price drop but differ when there is also a market move. For example, `C=100`,
`D=1`, `O_next=100` gives a proxy return of 1.010101% and cash-entitlement return
of 1%. This is a synthetic arithmetic example, not an observed QQQ result.

Thus a target built from `A` is acceptable when explicitly named an adjusted
overnight proxy. It must not silently become cash P&L. Also, `auto_adjust=False`
does not prove that vendor OHLC reproduces the original share units on old
split dates; Yahoo's historical tables describe their close as split adjusted.
Consistent units, not the word "raw", must determine split treatment.

## Timing that would survive a trading interpretation

A feature observed at the closing price cannot justify purchasing at that same
closing price. Nasdaq's [2018 cutoff-change notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2018-61)
documents the earlier 15:50 ET market-on-close cutoff and the planned extension
to 15:55. Its [Closing Cross FAQ](https://www.nasdaqtrader.com/content/productsservices/Trading/ClosingCrossfaq.pdf)
also makes clear that orders enter before the closing auction price is known.
Use an order decision sufficiently before the applicable historical cutoff,
including early-close sessions, with all market features fixed through the
previous completed session `t-1`. Buy at close `t`, exit at the next actual open.
The closing/opening prices remain execution proxies.

This costs a full session of freshness compared with an idealized after-close
statistical forecast, but makes the information set compatible with the entry.
Do not choose between these timing schemes after seeing their scores. The
recommended primary experiment is the stricter prior-session scheme.

Calendar features must describe the known entry-session date `t`. In contrast
to the daytime study, the next actual opening date is not always known before
close `t`: an unscheduled closure can intervene overnight. Never use that
future realized date to define a pre-close calendar predictor. The realized
next opening may still label the outcome and holding duration.

The cached adjusted label needs the next session's daily adjustment factor.
Without an independently timestamped action feed, conservatively treat that
label as available only after the next session closes, even though its price
move ends at the open. A pre-close refit at `t` therefore admits only labels
whose target-session close is at or before `t-1`'s close. No intraday refit may
use `A_t` merely because the raw overnight move already finished at `O_t`.

## Recommended three-candidate family

One horizon: close `t` to the next open. All variables below use completed
information through `t-1`; previous-session Cboe closes are available before
the next day's pre-close decision. Do not add an unnecessary second lag by
reusing the earlier after-close feature function unchanged.

The strong baseline includes intercept; separate adjusted-overnight and
daytime log-return averages over 1, 5, and 22 sessions; log mean total variance
over 1, 5, and 22 sessions; log VXN and log VIX; and entry-session weekday
indicators. Keep a historical-mean overnight control on exactly the same
training rows. This controls persistence, reversal, volatility, implied risk,
and an ordinary weekday-dependent overnight premium.

Adding overnight-minus-daytime averages to that baseline would be redundant:
the baseline already spans them. Do not repeat the daytime session-split arm
against a weaker total-return-only benchmark and call it orthogonal.

| Singleton augmentation | Exact proposed information | New question |
|---|---|---|
| Signed cross-asset returns | HYG, TLT, GLD, USO, UUP one-session adjusted log returns through `t-1`, individually | Does information outside QQQ predict its next overnight outcome beyond QQQ's own overnight/daytime history? |
| Signed volume pressure | `day_log_return * prior_z(log(volume))`, latest value and strict five-session mean, where the z-score uses the preceding 252 sessions, minimum 126, with sample standard deviation | Does an unusually active signed session carry information about subsequent overnight repricing? This is a price/volume proxy, not measured order imbalance. |
| Implied uncertainty shape | `log(VIX9D/VIX)` and `log(VVIX)`, each observed on `t-1` | Does information beyond implied-volatility levels matter for the overnight return rather than next-day variance? |

These are candidates fixed by mechanism before scoring, not literature claims
that these particular regressors work. Existing local sources contain these
inputs; any new download should be kept additive with its own hash and source
record. Use the same fixed normalized ridge objective as the daytime study:
mean squared error plus `0.01` times squared standardized slopes, unpenalized
intercept, training-only population scaling. At least 1,000 complete training
rows and monthly expanding refits; identical complete rows for every arm.

Entry origins: 2016-01-04 through 2025-10-17. Latest opening outcome and its
conservatively assigned label-close availability: 2025-10-20. Development:
2016–2019, excluding any origin whose label is available after 2019-12-31.
Evaluation: 2020-01-02–2025-10-17; stability slices 2020–2022 and 2023–2025.
No field from the protected phase starting 2025-11-03 enters. These dates
reuse already inspected history and cannot provide pristine confirmation.

Six primary comparisons: three candidates against both strong baseline and
historical mean. Freeze a new wave allocation and extend the cumulative trial
ledger before scores. Keep the previously declared index forecast-loss effect
threshold, block-length sensitivities, and both-control requirement; do not
relax them because a different component is harder to predict. Model selection
and trading thresholds stay fixed across phases and sensitivities.

## Controls against mechanical successes

1. **Corporate-action invariance tests before runs.** A synthetic split with
   consistent share units must not manufacture a return. A pure dividend price
   drop must have zero adjusted overnight return. Apply a common later
   adjustment multiplier to both dates and require unchanged local returns and
   features. Never use adjustment-factor levels as predictors: those levels
   can encode future corporate actions.
2. **Target-close cancellation test.** Change `C_next` and `A_next` by the same
   factor while holding the next open and action factor fixed. The overnight
   target must remain unchanged. Future prices must not change any pre-entry
   feature, scaler, fitted coefficient, or already issued forecast.
3. **Outcome-measurement sensitivity, fixed in advance.** Flag a potential
   action boundary when `abs(log(F_next/F_t)) > 1e-5`. This threshold concerns
   adjustment changes, not prediction performance. Keep the primary full sample
   and also report paired loss on the unflagged subset with the *same forecasts*.
   The flag is known only after the fact in this cache: it cannot filter trades,
   train an ex-ante signal, or retroactively define an investable universe.
4. **Independent action audit before a cash claim.** Reconstruct corporate
   actions from a separately stored issuer/source record, including split
   units and ex/payment dates. [Invesco's distribution table](https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=investors&productId=QQQ&ticker=QQQ)
   distinguishes ex-date, record date, pay date, and per-share amounts. A
   [2020 QQQ prospectus](https://www.sec.gov/Archives/edgar/data/1067839/000119312520018563/d835635d485bpos.htm)
   describes its then-current quarterly distribution schedule; it is not proof
   of when each historical amount was announced. A scheduled ex-date is not a
   known cash amount. A distribution receivable is not cash available to spend
   at the opening exit.
5. **No silent price repair.** The [yfinance repair documentation](https://ranaroussi.github.io/yfinance/advanced/price_repair.html)
   discusses missing adjustments, wrong ex-dates, and repairs inferred from
   subsequent price moves. Freeze the source version. Any repaired variant
   needs a separate declared measurement sensitivity and source audit; it must
   not replace the input after inspecting the fit.
6. **Economic controls.** Always-long overnight and historical-mean rules,
   identical two-leg costs, exposure and turnover, and financing/cash-dividend
   accounting are required. The ordinary positive overnight mean and reduced
   daytime risk are not new forecasting signals. Adjusted-price cost screens
   remain proxy screens until exact corporate-action accounting is available.

## Duplicate search

No completed overnight **signed-return forecast** was identified in repository
protocols or source modules. Existing overnight quantities are squared gaps,
variance shares, leverage/semivariance assignments, or contextual features.
`src/features.py::overnight_var`, `src/history_extension.py`, and the earlier
signal protocols all use overnight information for volatility. The current
`iterative_signal_search.yaml` index target is next-session open-to-close and
explicitly stays flat overnight. Existing carry work targets variance/option
carry, not QQQ closing-auction to opening-auction return. This is consequently
a distinct target, though it reuses the same assets, history, and much of the
same information.
