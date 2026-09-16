# Range location and next-session SPX risk alerts

This prospective wave tests one information increment: the previous close's location within the reported daily high–low range. The outcome is whether next-session daily risk exceeds twice the mean of the 22 sessions ending at the previous-session cutoff. It does not mean doubling relative to the immediately preceding day.

The prior quarter-end study is complete and independently verified. Its review found no improvement. Its protocol, code, failures inherited from earlier attempts, and published evidence remain unchanged. The previous turn completed the publication audit, so this continuation starts from substantive progress.

## Fixed question and comparisons

Let Z=(log(C/L)−log(H/C))/log(H/L) and E=Z² for a valid positive bar with nonzero range. E distinguishes bars with identical range, open-to-close return and Garman–Klass risk. That algebra motivates a predictive test; it does not establish statistical orthogonality, order flow, a causal mechanism, or profitability.

The candidate fits a single penalized coefficient on centered E, conditional on a frozen fitted logistic baseline. The baseline retains all 22 controls from the tail-shape study and adds raw intraday and overnight returns and their squares. All 25 baseline slopes use training-only population scales. Duplicate or linearly dependent columns are retained: the slope penalty handles dependence. An exactly constant E has an exact zero correction and retains its fold. There is no feature selection by observed association or rank.

A second comparison requires beating a recent event-frequency forecast with a fixed 63-session half-life. It starts from the first training event mean, then consumes all eligible labels on the full reference calendar, including labels attached to feature-incomplete or unscored origins. It never feeds pre-initialization labels again or resets between study periods.

The registered family will contain exactly two new comparisons, 127 inherited comparisons, and 129 cumulative hypotheses. Both controls must improve by at least 0.0005 absolute mean Brier score in development and evaluation, improve in both fixed later slices, and pass the wave and cumulative Holm gates. No selected date range, calibration bin, event threshold, alternative lag or auxiliary score can replace these conditions.

## Time, support and arithmetic

Market inputs end at the previous observed SPX session. Every label becomes usable at the next observed SPX close. The 22-session target reference is calculated with a checked compensated sum divided by 22; equality with exactly twice that mean is a nonevent. This fixes the arithmetic before event counts are inspected, avoiding a binary label disagreement from different rolling-sum implementations. Missing values remain unknown, and valid zero-range bars can still contribute known risk labels.

The observed SPX calendar is retained before rolling windows or shifts. Monthly fitting begins at the first feature-complete application, before checking whether its future query label exists. All monthly training, transformation, and support gates must pass before any model is fitted. Every application, including an unscored one, must receive valid predictions and a state audit. At least 1,000 mature training rows and 50 examples of each class are required; each phase needs 127 observations and 30 of each class, and each fixed evaluation slice needs 15 of each class.

The same source fence, October 20, 2025, and protected period beginning November 3 remain. The nominal study periods are 2016–2019 and 2020–October 2025. Final reporting must identify actual scored date coverage rather than infer coverage from those nominal windows.

The producer uses the existing normalized ridge-logistic Newton solver. Independent verification solves the gradient equations from zero with a different numerical method and solves the scalar correction by a fixed bracketed root. Their exact budgets and gradient, coefficient and prediction checks are specified in the protocol and synthetic tests. There is no empirical retry, fallback, clipping, dropping of a difficult fold, or tolerance relaxation.

Bootstrap inference uses blocks 21, 63 and 126 and 199,999 draws. This draw count was set before data construction so its minimum attainable p-value is below one tenth of the strictest two-comparison wave cutoff. HAC uses the unchanged lag of 126. Nominal uncertainty and minimum detectable effects are descriptive, not equivalence tests or substitute acceptance gates.

## Verification and interpretation

Before registration, the entire repository test suite and independent synthetic reconstruction tests must pass. A freeze record binds the protocol, new and prior code, designs, and test evidence. Source files are decoded from hash-checked byte snapshots; the original provenance records and complete prior artifact closures are rechecked. The new independent verifier reconstructs sources, features, labels, training cohorts, fits, application predictions, frequency states, scores, inference and the trial ledger. Prior model fits are represented by their unchanged successful proof records, without rerunning every historical experiment.

Any source, arithmetic, support, optimizer, forecast, inference, verification or publication failure leaves both new comparisons unevaluable with p=1. Diagnostics and failure records are preserved. The prior two failed quarter-end attempts retain their original status and are not reclassified as statistical null results.

These are archival vendor data with unresolved original release latency and revision history; early VIX9D values are back-calculated. Daily OHLC risk is a proxy. Repeated adaptive research on this history supplies exploratory evidence, not untouched confirmation. A favorable result would need that distinction preserved before any claim of practical predictive usefulness.

Trial-ledger appends are atomic. On a recoverable producer publication failure, the previous ledger bytes are retained as a diagnostic and a terminal ledger is reconstructed from the two registrations, the validated inherited snapshot if available, and two unevaluable outcomes. If inherited metadata cannot be reconstructed, that limitation and a zero reconstructed count are explicit; missing historical records are never fabricated. Complete success has 131 events. Independent verification saves the exact hashes of the outputs it checked, and figure generation requires the metrics bytes to match that saved snapshot.
