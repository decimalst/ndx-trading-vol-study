# Unregistered next question: range location and a volatility alert

Test whether the previous session's **closing-location extremity within its reported high–low range** improves the probability of next-session SPX risk exceeding twice its prior 22-session mean, beyond strong conditional controls and a recent event-rate forecast. This is a new transformation of already admitted OHLC information and a volatility-magnitude event target. It is not a new provider, an order-flow measurement, or another calendar experiment.

This proposal is independent of wave 17's result. Only three completed research summaries, existing protocols and a prior source-review record were consulted. No wave 17 metrics, forecasts, new raw values, derived quantities, class counts, associations or fits were inspected. A protocol text check found no registered closing-location/range-position or risk-doubling arm. That establishes novelty within the checked local contracts, not a claim about the wider literature.

## Why this is a different question

The [model-memory study](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/model_memory_study/SUMMARY.md) changed architectures, objectives, retrieval and weighting while largely retaining existing inputs. This proposal asks whether the OHLC compression itself discarded predictive information. Garman–Klass uses the log high–low width and open–close change, but does not uniquely determine where the close lies inside the range.

For a purely algebraic example, let `O=C=P`, `H=P*exp(a)` and `L=P*exp(a-r)`, with `0<a<r`. Holding `r`, `P` and the previous close fixed holds GK, intraday return and overnight return fixed, while the location statistic below changes with `a`. Thus the proposed input contains a degree of information not algebraically recoverable from those summaries. This is not evidence of statistical orthogonality or forecast improvement.

The [target-aligned study](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/SUMMARY.md) documents the noisy daily-product target and dependence of its squared loss on cross fourth moments. A bounded binary score avoids that particular moment problem while returning to volatility magnitude. The economic question is whether how a session finishes within its range helps anticipate an unusually large following-session risk proxy. It does not establish why any association occurs or whether it is tradable.

## One input, one event, two comparisons

For a valid positive OHLC bar with `H>L`, define

`Z = (log(C/L) - log(H/C)) / log(H/L)` and `E = Z²`.

Use only `E[t-1]`, with no threshold, lag, averaging-window or directional variant search. Valid OHLC implies `Z` lies in `[-1,1]`; numerical violations, invalid bars or a zero range require explicit prospective arithmetic handling, never clipping or replacement by zero. Missing bars remain unknown. The observable is the vendor-recorded position of the close, not a fresh quote, auction imbalance, intraday path or executable price.

Let `V_s = max(GK_s,1e-10) + log(O_s/C_(s-1))²`, retaining the existing daily-proxy definition and observed SPX calendar. Define the binary target

`A_t = 1{ V[t+1] > 2 * mean(V[t-22:t-1]) }`.

The trailing reference contains exactly 22 observed sessions ending at `t-1`. Equality is a nonevent; missing target or reference is unknown. The factor two is a prospective reference, not a fitted quantile, universal tail probability or economic payoff. The label matures at the next observed SPX close. This forecasts a reported full-session OHLC-risk event, not measured high-frequency integrated variance or a signed crash.

Use all 22 baseline columns from the [SPX tail-shape protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/tail_shape.yaml), including its training-centered curvature, and add lagged raw intraday return, its square, raw overnight return and its square as nuisance controls. These hold more of the bar's direction and magnitude information fixed. Fit the resulting conditional logistic baseline with the existing `.01` normalized slope penalty and unpenalized intercept. All nuisance transformations use only the common mature training rows.

Freeze that baseline's logit function, then fit one scalar correction `b*(E - training_mean(E))`, with fixed scale one and penalty `.01*b²`. Exact constant `E` has canonical zero correction with fold retention. This is a staged probability model, not a joint refit or proof of a structural mechanism.

The second control is a 63-session-half-life recent event frequency, using the [causal-pool initialization and full-calendar maturity rules](/Users/byrons/code/trading-vol/ndx-vol-experiment/causal_pool.yaml). It follows the newly defined event labels, includes all eligible label arrivals, never double-feeds its initial history and never resets between phases. Register exactly two hypotheses: the location candidate versus the conditional baseline and versus recent frequency. Both must pass. This prevents a changing unconditional event rate from being promoted as conditional location information.

## Source, chronology and falsification gates

The required existing inputs are SPX raw OHLC plus VIX, VIX9D, VVIX and the legacy SKEW inputs already admitted in the [source review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/tail_shape/SOURCE_REVIEW.md). No option archive, license exception or new acquisition is needed. Re-admit the exact original files and complete source closure before construction. Archival revisions, unverified historical delivery latency, early back-calculated VIX9D and an observed rather than independently certified exchange calendar remain limitations; “already admitted” does not establish original-vintage availability.

Retain the numerical fence of October 20, 2025 and protected period beginning November 3. All market inputs end at the prior observed session. Refit monthly at the first feature-complete origin, before testing future label presence, with at least 1,000 common mature observations and 50 of each class. Labels used for fitting must be available by that origin's previous-session cutoff. Retain unscored application months. Use development 2016–2019, evaluation 2020–October 2025 and the existing 2020–2022/2023-onward slices, with the existing phase-boundary maturity exclusions.

Propose the existing binary support gates: at least 30 of each class per phase and 15 per evaluation slice. Require at least `.0005` lower mean Brier score against both controls in both phases and negative paired gaps in both evaluation slices. Use the existing conservative phase/block/HAC inference, next-wave alpha allocation, and cumulative Holm family including every prior failed or unevaluable hypothesis. All sources, class-support, optimizer, verification and publication failures retain both new hypotheses at p=1. No favorable calibration bin, event subset or alternative score may substitute for these gates.

Prewrite tests for the algebraic same-GK/different-location example, unit invariance, ties and zero ranges, strict 22-session windows, target-threshold timing, future-value mutation, complete monthly schedules, both controls' exact label maturity, scalar convexity, and complete failure accounting. Fix arithmetic and optimizer contracts before reading event counts. If these tests or support gates fail, report that limitation without changing the threshold or deleting difficult rows.

## Overlap limits

Relative-risk work modeled a log ratio; joint-risk and target-aligned work modeled paired second moments and residual products. Sign-memory and causal pooling modeled cross-asset directional agreement; the latter demonstrated why a strong recent-frequency control matters. Those trials neither measured this within-bar location information nor scored this one-asset event relative to its trailing risk mean. The proposed logistic and frequency machinery deliberately reuses tested methods. Its novelty is the specific information increment and prediction functional, not greater model complexity. This is an adaptive exploratory proposal on reused history, not an untouched-confirmation claim or a revision of any existing verdict.
