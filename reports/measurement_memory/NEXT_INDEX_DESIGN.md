# Prospective medium-horizon index-return design

**Status: prospective and unregistered.** This note proposes a bounded next question. No new forecasts, fits, predictor–target comparisons, or economic results were produced for it. Earlier frozen studies remain unchanged.

## What the repository has already tested

No implemented 21- or 63-session **signed index-return forecast using an implied-versus-trailing-realized risk gap** was found in the inspected implementations and protocols. This is narrower than proving that no historical notebook ever explored the idea.

| Existing work | Actual question |
| --- | --- |
| `src/research_paths.py::run_horizon_curve` | Forecast future variance at horizons including 21 and 63 sessions. |
| `src/research_paths.py::run_vrp_term_structure` | Describe contemporaneous implied volatility minus subsequently realized volatility at calendar-day horizons. Its future realized term is unavailable as a forecast input. |
| `src/carry.py` | Describe a variance-swap proxy payoff. Its 21-session spacing is not a 21-session index-return target; its whole-window median richness threshold must not be reused as a causal forecasting rule. |
| `src/iterative_index.py` and `src/overnight_index.py` | Predict next-session daytime or overnight returns. |

The slower-horizon route was already identified, but not implemented, in `reports/iterative_signal_search/INDEX_OPPORTUNITIES.md`. Bollerslev, Tauchen, and Zhou motivate studying aggregate returns at monthly/quarterly horizons using implied and accurately measured realized variation. Their construction does not establish that this repository's daily OHLC proxy measures the same premium. [Original paper](https://public.econ.duke.edu/~get/wpapers/btz.pdf).

## A small, identifiable question

Use existing **SPX price-index returns with VIX**, avoiding an underlying mismatch. Fix both horizons, 21 and 63 observed US sessions, and predict cumulative log price return `log(C[t+h]/C[t])`. Forecasts submitted before close `t` use price and Cboe observations only through the completed session `t−1`.

At that cutoff, define `I = log((VIX/100)^2)` and `R = log(252 × trailing_22_mean(daily_variance_proxy))`. The proxy is the existing Garman–Klass daily variance plus squared raw overnight log return, with its existing documented validity convention. The factor 252 is an explicit conventional scale, not evidence that the two sources measure identical variance contracts.

**Do not add `I−R` to a linear model already containing `I` and `R`.** It adds no predictor span. Under ridge, any change can arise solely from altered penalty geometry; that is not incremental information.

Instead, the sole proposed candidate is the fixed joint-state hinge

`k = max((I−R) − mean_training(I−R), 0)`.

Its center must come from the exact admitted training rows at each monthly fit. No threshold search or whole-sample centering is allowed. A strong component baseline must retain `I` and `R`, their separately centered univariate squares, own return and realized-risk histories, and the existing available IV-level/shape controls. The candidate adds only `k`. This asks whether an asymmetric *joint relationship* helps beyond component levels and separate curvature. It supplies no new source information and is not a replication of the paper's linear specification.

Compare this candidate with both the component baseline and the historical mean fitted on identical training labels: one candidate × two controls × two horizons gives four declared contrasts. Keep the prior return study's fixed ridge objective, training-only population scaling, and unpenalized intercept; retain the same-row mean rather than assuming zero expected return.

## Timing, evaluation, and claims

Use monthly expanding fits with at least 1,000 complete matured labels. Admit a historical label only when its target end is at or before the previous-session feature cutoff of the monthly fit, and its origin precedes that fit. Match all models' training and scoring rows within each horizon.

Keep development origins in 2016–2019 with target ends no later than 2019-12-31, and evaluation origins from 2020-01-02 with target ends no later than 2025-10-20. Do not shorten boundary targets or read the sealed period. Future session dates may identify outcome maturity, never predictors. Freeze the exact baseline, row policy, effect gate, inference, and cumulative multiplicity allocation before implementation or scoring; require improvement over both controls without relaxing prior gates.

Daily 63-session targets overlap by 62 sessions. Four development years contain only about 16 nonoverlapping quarters; six evaluation years contain about 24 before boundary exclusions. Dependence-aware uncertainty must use blocks exceeding the longest horizon, with longer-block sensitivity fixed in advance. Nonoverlapping-offset checks must report every predetermined offset, not select the best. Low power remains a material limitation even with thousands of daily origins.

Yahoo/Cboe extracts are current archival vintages, not certified historical vintages. VIX represents a 30-calendar-day options construction; the trailing OHLC proxy uses trading sessions and different measurement inputs. Call the feature a **log implied-versus-trailing-proxy gap**, not measured variance risk premium. SPX price returns exclude reinvested dividends and are neither excess returns nor executable fund returns. Any subsequent trading claim needs a separate instrument, execution, overlapping-position capital, and cost contract.

**Recommendation:** the nonlinear four-contrast question is worth considering as a distinct return-target experiment. A raw-gap augmentation with component controls is redundant and should not consume another experiment.
