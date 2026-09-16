# Preregistered directional-agreement memory experiment

This design precedes any new binary outcome, class-count, feature association, probability fit or Brier-score calculation on the archival data. It follows the unaltered wave-12 prospectus. All previous source, test, protocol, report and trial identities remain frozen. This is a new functional on reused history, not independent confirmation or a new information source.

## Target and controls

At an origin on the full observed SPX calendar, predict strict same-direction movement of next-session raw QQQ ETF and SPX price-index intraday returns. Both positive or both negative gives one. Any exact zero gives zero, including two zeros. Missing paired observations stay unknown. Signs are compared directly without multiplication or an epsilon. Target and availability dates are the actual next SPX close; those dates do not enter predictors.

The memory is the preceding 22-session strict-agreement fraction minus the independence reference from the same window's four marginal positive/negative fractions. All fractions require one complete paired window ending at the session before origin. This controls marginal directional bias descriptively; it does not identify a copula or a pure correlation parameter.

All models use identical mature, complete training rows and scored origins. Frequency estimates the same-row training event rate. The strong logistic baseline uses the existing 34 raw joint-risk columns including its intercept, the training-centered correlation square, four marginal sign fractions, and their independence reference. The original 33 nonintercept columns plus the square retain training population scaling with every scale above 1e-12. The five bounded additions use fixed scale one and training centering. A prospectively declared exact-constant equality rule centers such a column to exactly zero, retaining it with its canonical penalized zero coefficient.

The candidate freezes the baseline logit and fits only one additional coefficient multiplying centered excess agreement. Its fixed scale is one. An exactly constant memory gives canonical coefficient zero with the model and fold retained. No alternate lag, state split, model search, calibration fit, or new intercept is included.

## Estimation and numerical contract

The baseline minimizes mean Bernoulli negative log likelihood plus .01 times squared slopes, with an unpenalized intercept. The memory stage minimizes the conditional likelihood with the frozen offset plus .01 times its squared coefficient. Each stage is convex; the two stages are not jointly optimized. Both-class support and positive slope penalty supply a finite unique optimum under finite arithmetic.

The producer uses deterministic damped Newton, 200 updates, at most 60 backtracks, Armijo 1e-4, and full gradient infinity norm at most 1e-8. Baseline initialization is the logit of training frequency with slopes zero; the memory starts at zero. Stable signed-logit likelihood, signed sigmoid residuals and sigmoid-product curvature avoid avoidable cancellation. Finite sigmoid endpoints are permitted. There is no probability clipping, solver restart or fallback.

Independent verification uses an independently implemented objective and a different baseline solver, plus a guaranteed bracket for the strictly increasing scalar memory derivative. The exact budgets, derivative checks, coefficient tolerances and strict saved-coefficient probability replay tolerances are fixed in the protocol before fitting. Primary scores are reconstructed through strict saved-coefficient replay; the looser independent-optimizer comparison does not set inference tolerance. Floating-point checks are numerical evidence, not an exact-arithmetic certificate.

## Timing, sources and feasibility

Retain the bounded original QQQ/SPX OHLC and VXN/VIX/VIX9D/VVIX sources, the full SPX calendar and predecessor checks. Copy hash-checked source bytes to isolated temporary paths before decoding; the frozen loader filters dates before parsing numeric values and source audits identify the original paths. All complete observed OHLC rows must pass the inherited GK measurement gate before feature selection. Source and label dates end October 20, 2025; November 3 onward stays sealed. Archival revision and synchronized auction availability are not certified.

Use expanding monthly fits at the first feature-complete origin before query-label filtering. All labels used for fitting must be available by the preceding SPX session. Retain every fit and application, including an entirely unscorable application month. Require at least 1,000 complete mature training rows and 50 examples of each class in every fit. Development origins start January 4, 2016 and their labels must be available by December 31, 2019. Evaluation runs January 2, 2020 through October 17, 2025. Require 30 examples of each class per phase and 15 per fixed evaluation slice (2020–2022 and 2023–2025), with at least 127 phase observations for literal inference bandwidth. These are prospective support requirements; actual counts have not been inspected.

## Scores, accounting and decision

Register exactly two Brier comparisons: memory versus baseline and memory versus frequency. Brier loss directly elicits the strict-agreement probability and lies in [0,1]. Its expectation is pi(1-pi)+(p-pi)^2, so this target avoids return-fourth-moment dependence while retaining sampling uncertainty and serial dependence. Compute stable factored paired score gaps after validating both individual losses. Exact-zero errors are valid; nonzero square/product underflow and nonfinite arithmetic fail.

Require an absolute decrease of at least .0005 in both development and evaluation against both controls and a negative gap in both fixed evaluation slices. This threshold is a squared-probability-error reference, not profit or a literal probability-point improvement. Use Bartlett HAC126 and circular bootstrap blocks 21/63/126 with 99,999 draws and seed 20260919 plus the fixed phase/block offsets. Take the maximum two-sided p across methods and then phases. Holm2 must be below .05/(13*14), and cumulative Holm119 below .05, retaining all 117 previous hypotheses. Report nominal HAC80% minimum detectable effect descriptively; no retrospective power gate or effect adjustment is allowed.

Report descriptive calibration in the large for each phase/model: observed event frequency, mean issued probability, their difference, and Brier score. This uses no bins, additional fitting, hypothesis or promotion gate and does not establish conditional calibration.

Prewrite source/tie/window/maturity/scaling/derivative/extreme-logit/nesting tests, paired-score identities, support-before-inference checks, complete family gates, immutable source checks, and partial-publication failure tests. Run the full existing suite, focused checks, and independent reconstruction before interpreting results. A whole-wave failure keeps both registered hypotheses unevaluable with p=1, saved diagnostics and all earlier artifacts intact.

A qualifying result would support this specific raw agreement-probability increment beyond the declared controls on reused history. It would not alone establish volatility magnitude, covariance, standalone index direction, executable hedges, trading profit, or fresh-holdout confirmation.
