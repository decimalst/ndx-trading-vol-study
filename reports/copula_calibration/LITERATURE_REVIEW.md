# Literature check before the wave28 freeze

Reviewed primary texts on 2026-09-09, before new market fits. The review supports
the four-cell experiment as a **joint-score sensitivity diagnostic**. It does
not turn that experiment into a formal decomposition of the true mechanism.
The seven registered scalar comparisons and complete density scoring remain
the proposed numerical design.

## Forecast comparison and attribution

[Fissler and Hoga, How to Compare Copula Forecasts?](https://arxiv.org/pdf/2410.04165)
prove general non-elicitability of the copula functional (Proposition 4).
Proposition 5 and Example 6, PDF page 8, require correctly specified common
marginals for the restricted-class guarantee. Merely keeping incorrect
marginal forecasts identical does not guarantee identifying the true copula.
This reinforces the existing qualification of our historical result.

Their Theorem 10 orders a vector of marginal and copula scores
lexicographically, giving marginal accuracy priority. Remark 11 retains the
misspecification caveat. Section 4's sequential testing procedure uses a joint
long-run covariance and calibrated critical values; our Holm-adjusted seven
scalar tests are not that attribution procedure. Within each row of our crossed
design, the marginal-score difference is identically zero, so scalar comparison
of the copula-score component is appropriate for comparing those complete
forecasting systems. Across rows, full normalized density scores answer a
different question from lexicographic ranking.

The results report will therefore also show the summed marginal loss and
copula loss separately for every cell and phase. This is the algebraic identity
`joint loss = summed marginal loss + copula loss`, not seven additional tests
or a causal attribution claim.

## What ranks do and do not remove

The cited [Chen and Fan time-series paper](https://www.accessecon.com/pubs/VUECON/vu02-w26R.pdf)
studies a stationary first-order univariate Markov model. Its Section 3.1 uses
empirical marginal probabilities based on ranks divided by `n+1`.

The more directly relevant [Chen and Fan multivariate dynamic-model paper](https://www.accessecon.com/pubs/VUECON/vu04-w19.pdf)
allows copula misspecification while assuming specified conditional mean and
variance models and i.i.d. standardized innovations. It does not establish
robustness to arbitrary misspecified time-varying volatility forecasts.
[Patton's review, Section 3.3](https://public.econ.duke.edu/~ap172/Patton_JMVA_survey_2012.pdf)
distinguishes these two 2006 papers and discusses the estimation assumptions.

Ranks are exactly invariant to a fixed strictly increasing transformation of
each coordinate. Constant positive rescaling is one example. Our calibration,
holding a month's fitted parameters fixed, is another. Consequently, rank
pseudo-observations of original and calibrated training coordinates are
identical; rank refitting those two versions would duplicate the same
pseudo-likelihood. Four generated checks now verify this invariance, ties and
both fitted copula objectives. A negative control confirms that time-varying
scaling can change ranks. These are synthetic implementation checks, not a
new historical forecasting comparison; see `prefit/RANK_INVARIANCE.log`.

A comparison of t8 and Gaussian fits on ranks can still describe dependence.
Its translation into a forecast test needs care: ranking the entire evaluation
sample uses future outcomes, while fitting on ranks and evaluating the original
uncalibrated PITs leaves marginal error in the evaluation. A raw empirical CDF
has jumps and no suitable positive continuous return density for the proposed
log score. A future rank-based forecast experiment should specify a training-only
continuous marginal estimator, tails, ties and density factors before running.
No ad hoc rank pseudo-score is appended to this four-cell density test.

## Dependence shape and financial evidence

[Chicheportiche and Bouchaud](https://arxiv.org/html/1009.1100v3)
find Student copulas approximate strongly correlated stock pairs better than
weakly correlated pairs, where a single common volatility mode is inadequate.
That is relevant context for QQQ/SPX, not confirmation of this pair's
conditional forecasts. Their daily-stock cross-sectional results do not
identify our fitted correlation with their population correlation measure.

[Deng, Smith and Maneesoonthorn, Section 6](https://arxiv.org/pdf/2308.05564)
study 93 equities using 15-minute returns and report better portfolio-return
density forecasts for an Azzalini–Capitanio skew-t factor copula than their
Gaussian and symmetric-t benchmarks. This motivates a future asymmetric
dependence arm with an explicitly chosen skew-t family. It is not direct
evidence for a two-index next-session model. Crucially, the observed 4.1% versus
1.9% individual PIT-tail frequencies concern marginal calibration; a copula
change cannot alter either marginal distribution.

[Demarta and McNeil, Sections 4–5](https://www.planchet.net/EXT/ISFA/1226.nsf/0/303eb11b4d617b79c1257b0800744575/$FILE/t%20copula%20demarta%20mcneil.pdf)
describe maximum-likelihood estimation of both correlation and degrees of
freedom, including estimating degrees of freedom with correlation held fixed.
The paper does not justify eight degrees of freedom by claiming no estimation
method exists. Keeping eight here is a frozen modeling choice that limits
post-result tuning. Section 5 explains the symmetric-t copula's radial symmetry
and motivates skewed and grouped extensions.

[Frattarolo, Sections 2–3](https://arxiv.org/html/2501.00634v2)
distinguishes copula central symmetry from marginal symmetry and proposes a
dependence-aware empirical-copula test. Such a diagnostic would address joint
asymmetry more directly than individual tail counts, but requires its own
stationarity assumptions and bootstrap specification. It has not been added
after the fact as a significance test for this experiment.

## Execution decision

Proceed with the normalized four-cell comparison and its original seven
prespecified contrasts. Interpret any changed gap alongside held-out marginal
scores and calibration descriptions. Consider a properly specified rank-based
marginal system and a skew-t dependence arm as distinct subsequent experiments,
with all new choices and comparisons recorded before their numerical runs.
