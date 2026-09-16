# Registered signed-return downside-tail experiment

Wave 7 asks whether lagged SKEW can improve a signed next-session SPX downside probability and the full return density after the same source is already allowed to affect conditional mean and variance. It follows the independently verified negative wave-6 medium-horizon result. This is a distinct target from the repository's previous log-variance quantiles, unsigned jump-share events, volatility stress classifiers, and variance-carry diagnostics.

The new candidate introduces one SKEW-dependent asymmetry slope. The source is not new, and distribution forecasting itself is not new to the repository. The source/target pairing and comparison with identical predicted moments are the specific new question.

## Fixed target and timing

At entry close t, market information ends at the preceding observed SPX session. Normalize the next raw-close log price return by subtracting the preceding strict 22-session mean return and dividing by the square root of the preceding strict 22-session mean daily GK-plus-overnight variance. The GK floor remains 1e-10. There is no annualization in this target normalization. The event is normalized return strictly below −1.5, fixed before constructing or counting empirical events. It is not a fitted quantile or a universal 5% VaR event.

Monthly fits occur at the first complete feature origin, independently of subsequent label readiness. Training labels must end at or before the preceding session of that fit. All three models use the same complete observations. Development spans 2016–2019 with outcomes completed within 2019; evaluation starts 2020-01-02 with complete outcomes by 2025-10-20. The protected period beginning 2025-11-03 stays excluded.

Each training fit requires at least 1,000 common matured labels, including 50 events and 50 nonevents. Each phase requires 30 of each, and each of the two fixed evaluation slices requires 15 of each. Insufficient support aborts the registered family; the threshold and support requirements will not be adjusted after counting outcomes.

## Shared moments and distinct shape

Both density forecasts use the same 22-column conditional mean and variance basis: the prior implied/realized-risk and return histories, IV shape/VVIX, weekdays, three negative-return histories, raw SKEW, and training-centered squares of implied risk, realized risk, and SKEW. Population scaling and every square center use exactly the admitted training rows. Any of the 21 non-intercept scales at or below 1e-12 aborts the wave.

First fit the shared mean by normalized ridge MSE with slope penalty 0.01 and an unpenalized intercept. Then fit the shared positive conditional variance by the frozen log-link second-moment solver with slope penalty 0.01, using squared residuals from the current training mean fit. These are training residuals, not predictions made out of sample. The fitted mean and variance functions remain identical for both density forecasts.

Hansen's moment-standardized skewed Student distribution changes asymmetry while keeping innovation mean zero and variance one. Its internal centering and scaling constants are recomputed at each shape value. [Hansen (1994), Autoregressive Conditional Density Estimation](https://www.ssc.wisc.edu/~bhansen/papers/ier_94.pdf); [official implementation reference](https://arch.readthedocs.io/en/latest/_modules/arch/univariate/distribution.html#SkewStudent).

Tail heaviness is fixed at eight degrees of freedom before fitting, with no search. The constant-shape baseline uses lambda=0.95*tanh(a); the candidate uses lambda=0.95*tanh(a+b*S), where S is training-standardized, one-session-old raw SKEW. The coefficient box is [-3,3]; only b has a 0.01 squared penalty. One deterministic fit starts at a=0 for the constant model, then at that fitted a and b=0 for the candidate. The fixed bounded optimizer must report success and satisfy the predeclared projected-gradient tolerance. There is no fallback, retry, or multistart selection. A stationary fixed-start solution is not a proof of a global optimum.

Event probabilities evaluate the standardized density CDF at (-1.5−shared_mean)/sqrt(shared_variance). Full normalized-return log density includes the scale Jacobian. A frequency control uses the exact common training event rate and makes no density or conditional-moment claim.

## Three comparisons, no selected successes

The candidate must improve Brier score against both the constant-shape model and historical frequency, and improve full-density negative log score against constant shape. Each comparison must improve in both phases and both evaluation slices. The fixed absolute improvements are 0.0005 for each Brier comparison and 0.005 for the density comparison. All three must pass wave Holm at 0.05/(7*8) and cumulative Holm at 0.05 across 106 enumerated comparisons. This tracked family is not an exhaustive count of all repository experimentation.

Paired uncertainty takes the maximum two-sided probability over circular bootstrap blocks 21/63/126 and Bartlett HAC126, then the maximum across phases. The 99,999 draws resolve below one tenth of the strictest wave cutoff. At least 127 observations preserve the literal HAC bandwidth. Calibration bins are fixed deciles of predicted probability; empty bins remain explicitly empty. No AUC, event cutoff, year, or subgroup is selected for promotion.

## Validation and limitations

Synthetic tests preceded implementation. They cover raw source pins/anchor/derived equality, date-first source bounds, missing windows, target normalization, event definition, future mutations, matched moments, population scaling, distribution mass and first two moments, CDF and scale Jacobian, analytic derivative checks, optimizer rejection, event support, full synthetic walk-forward scoring, and publication faults. The independent verifier reconstructs raw inputs and outcomes, fits the mean by augmented least squares and variance by a separate optimizer, reconstructs density formulas and shape gradients independently, and checks all forecasts, support, scores, uncertainty, corrections and ledger events.

Implementation checks caught a pandas attribute-name collision for the SKEW column and a one-bit floating representation mismatch for the rational wave allocation; these were corrected before registration or empirical feature construction. No statistical threshold changed. Numerical verification tolerances are declared in the verifier before the empirical run and cannot be relaxed afterward.

The original SKEW raw/derived archive and historical anchor passed source admission; five missing SPX SKEW dates remain gaps. Historical revisions and precise release latency are unverified, and early VIX9D is back-calculated. SPX returns exclude reinvested dividends and interest. SKEW's option-implied 30-day construction differs from a physical one-day tail probability. Even with identical modeled moments, remaining mean, variance, or fixed-tail-heaviness misspecification can resemble shape predictability. Any passing result remains exploratory, with no executable trading claim.

The runner freezes all Python sources/tests, exact inputs and earlier protocols/reports before the first empirical construction or count. A registered source, support, optimizer, scoring, validation or publication failure retains all three hypotheses as unevaluable with probability one. Earlier frozen work remains unchanged.
