# Prospective next question: directional agreement memory

**Unregistered proposal, separate from wave 12.** No new binary outcomes, event counts, products, relationships, losses, or fits were computed for this note. No source, protocol, code, or earlier report was changed.

One bounded next test is whether recent **strict agreement of QQQ and SPX intraday return signs** improves a direct probability forecast beyond lagged marginal risk, signed-return history, continuous correlation, and marginal sign frequencies. This is a different forecast functional using the same archival information. It is neither a new data channel nor independent historical confirmation.

## What is and is not new

The inspected [joint-risk protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/joint_risk.yaml) targets paired intraday residual second moments with matrix loss. [Wave 11](/Users/byrons/code/trading-vol/ndx-vol-experiment/cross_moment.yaml) scores issued residual-product means; [wave 12](/Users/byrons/code/trading-vol/ndx-vol-experiment/target_aligned.yaml) directly fits those means. These are magnitude-sensitive functionals, including signed products, rather than probabilities of directional agreement.

Binary classification itself is already present. [Tail shape](/Users/byrons/code/trading-vol/ndx-vol-experiment/tail_shape.yaml) forecasts a standardized negative SPX return event with Brier scoring; [representation study](/Users/byrons/code/trading-vol/ndx-vol-experiment/representation_study.yaml), [jump/regime work](/Users/byrons/code/trading-vol/ndx-vol-experiment/target_regime.yaml), and the [NQ intraday diagnostic](/Users/byrons/code/trading-vol/ndx-vol-experiment/nq_intraday_study.yaml) concern volatility or jump events. A source-wide search of Python implementations and these target contracts found no paired raw QQQ–SPX sign-agreement experiment. That is a bounded novelty finding, not a claim that every historical notebook or possible sign model has been audited.

## Observable target and timing

Use the existing full observed SPX calendar. At origin session `t`, forecast the next actual SPX session's paired raw intraday log returns `r_Q=log(C_Q/O_Q)` and `r_S=log(C_S/O_S)`. Define

`Y_t = 1[(r_Q>0 and r_S>0) or (r_Q<0 and r_S<0)]`.

An exact zero in either return gives `Y=0`, including two zeros. Thus the named event is **strict same-direction movement**; its complement includes opposite directions and ties. Do not call that complement exclusively “opposite signs.” Keep all valid zero observations, use exact comparisons without an epsilon, and compute signs directly rather than multiplying tiny returns. Missing or invalid paired returns remain unknown under the existing measurement rules; they do not become class zero.

Raw signs avoid a retrospective mean convention. Residual-sign probabilities would answer a different question and require explicit replay of the mean issued for every training label; they are excluded from this proposal. Strict raw agreement is observable and unchanged by a positive rescaling of either return, but it also reflects marginal directional biases. It does not isolate a copula or true correlation parameter.

Every market and sign-history input ends at the preceding observed SPX session `t−1`; entry weekday is known at `t`. Target and availability dates are the actual next SPX close, used only as label metadata. No next observed date or future opening information enters a predictor. Preserve the full calendar, strict windows, source predecessor rules for inherited overnight/close-return features, and the existing whole-source measurement gate. Do not align to the next common asset observation or fill missing sessions.

Reuse the exact bounded [joint-risk sources and loader contract](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/joint_risk_features.py): QQQ ETF raw OHLC, SPX price-index OHLC, VXN, VIX, VIX9D, and VVIX. QQQ is not the NDX index, SPX is not a tradable ETF, and their reported opens need not represent identical executable auctions. Raw intraday prices do not measure total returns or hedge profit. The Yahoo/Cboe snapshots remain archival rather than certified historical vintages, and early VIX9D is back-calculated. No new acquisition or sealed-period source access is needed.

## One candidate beyond a strong direct probability baseline

For each session with both finite raw intraday returns, record its strict agreement and each asset's positive/negative indicators. Over the strict 22-session window ending at `t−1`, calculate `J22`, the agreement fraction, and four marginal fractions `P_Q+`, `P_Q−`, `P_S+`, `P_S−`. All five summaries require the same complete paired window; a missing observation invalidates the window without compressing dates. Define

`A22 = P_Q+*P_S+ + P_Q−*P_S−`, `M22 = J22−A22`.

`A22` is the agreement probability under the window's empirical marginal independence construction, not an assumption about future returns. It explicitly includes the effect of ties through the positive/negative marginal fractions. `M22` is a simple excess-agreement memory statistic; it is not a formal unbiased dependence estimator. Its information is not determined by marginal sign frequencies alone. Continuous Pearson correlation and the other return moments remain competing controls.

Fit three probability models on one exact common training sample:

1. **Frequency:** the historical mean of `Y` on those training rows.
2. **Baseline:** direct ridge logistic regression using the existing 34 joint-risk raw features, their training-centered corr22 square, the four marginal sign fractions, and `A22`. This supplies the full existing lagged risk/correlation/return controls and both direct and joint marginal-direction controls before adding memory.
3. **Memory:** freeze the baseline's fitted logit function `eta(X)` and add one slope: `p=expit(eta(X)+b*m)`, where `m=M22−training_mean(M22)`. Use no extra intercept, alternate lag, regime split, or run-length search.

Propose normalized Bernoulli negative log likelihood plus `.01` times squared slope norm for the baseline, with unpenalized intercept, and mean conditional Bernoulli negative log likelihood plus `.01*b²` for the single memory slope. The baseline's existing 34 nonintercept continuous/calendar terms retain the declared training population scaling and strict scale checks. The five newly bounded frequency controls and `M22` use a **predeclared fixed scale of one**, with training centering, rather than inverse empirical dispersion. Exact constant bounded columns are centered to exact zero by a predeclared equality rule and retained; this is not a post-inspection zero-scale repair. Their penalized coefficients then have canonical zero optima.

For the memory stage, gradient and curvature are `mean((p−Y)*m)+.02*b` and `mean(p*(1−p)*m²)+.02`. The positive penalty curvature makes this a unique convex scalar fit, including `b=0` when `m` is identically zero. The two stages are not a jointly optimized full augmented model. Freeze exact stable log-likelihood arithmetic, finite-probability validation, solver limits, objective/gradient tolerances, and independent optimizer checks before fitting. There must be no clipping of probabilities to repair a failed fit or deletion of failed folds. No elliptical conversion of corr22 into a sign probability is assumed.

## Score, family, and prospective gates

Use Brier loss `(Y−p)²` for both comparisons. If the conditional event probability is `pi`, expected loss is `pi*(1−pi)+(p−pi)²`; hence the score directly elicits that probability. Loss lies in `[0,1]` and does not require return fourth moments. This removes large-return magnitude dominance from the target, while serial dependence, noisy finite samples, and imperfect controls remain.

Register exactly **two** primary hypotheses: memory versus baseline, and memory versus same-row frequency. No log-score, AUC, accuracy, trading-return, or subset-selection promotion arm is included. If this is wave 13 immediately after the present 117 comparisons, cumulative family size becomes **119**, with Holm2 at `.05/(13*14)` and cumulative Holm119 at `.05`. Require both controls to pass.

Propose the already used absolute Brier improvement threshold **0.0005** in both development and evaluation. It is a fixed squared-probability-error reference, approximately `(0.0224)²`, not a percentage of control loss or profit calibration. Require negative differences in both fixed evaluation slices, 2020–2022 and 2023–2025. Preserve the 21/63/126 block bootstrap, HAC126, 99,999 draws, and predeclare a new seed. The minimum bootstrap p of `1/100000` is below one tenth of the strictest two-comparison wave-13 raw cutoff; this establishes resolution, not power. Report nominal MDE and calibration descriptively, with no post-score power gate, threshold change, or equivalence claim.

Keep monthly expanding fits, at least 1,000 mature complete training rows, and prospectively require at least 50 events and 50 nonevents in every fit, 30 of each per phase, and 15 of each per fixed evaluation slice. Fit dates come from first feature-complete monthly origins before future-label eligibility. Use origins 2016-01-04 through 2025-10-17, development labels available by 2019-12-31, evaluation from 2020-01-02, and all source/label dates no later than 2025-10-20; 2025-11-03 onward remains sealed. Any insufficient support, numerical failure, or invalid verification makes the whole two-hypothesis wave unevaluable with `p=1`, preserving diagnostics and prior artifacts. New support counts have not been inspected.

## Immediate decision and prewritten tests

This is eligible for a small new contract, not yet registered or established as feasible. Before any new sign counts or fits, freeze the source/target rule, complete feature order, fixed versus empirical scaling, finite arithmetic and solver contract, all class-support gates, threshold, seed, and family ledger. Prewrite sign/tie/missingness truth-table tests; positive-price-unit and paired-sign symmetry tests; strict-window and prior-cutoff mutation tests; target maturity and development-boundary checks; training-only centers; common frequency/control samples; convex derivatives and exact zero-memory nesting; stable extreme-logit probability/loss checks; literal paired Brier algebra and bounded-loss tests; and complete failure-publication/independent reconstruction checks.

A pass would support one prediction of raw directional agreement beyond this declared baseline on reused history. It would not establish better volatility magnitude, covariance, tail severity, standalone index-direction forecasting, or trading profit. A failure would leave those earlier questions and their frozen verdicts unchanged.
