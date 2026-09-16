# Validation record — 2026-09-06

The first empirical run used protocol SHA-256
`2c081c8e9dcc9a407cd19d2d49c5e8b76462befeabcebc9a3539dbf69d516c93`
and producer SHA-256
`9e8b9f47c2c7f9ef1c8fceb9cda6f910e5f95cfb9dbf1e110161e65440800779`.
Neither changed after scores were observed. No new market files were fetched.

- The initial 15 synthetic producer tests failed at import before the producer
  existed. Before scoring, a test expectation was aligned with the written
  leverage definition: take the negative part **after** averaging returns.
  A floating-point equality assertion was changed to tight numeric tolerance.
  Tests also detected an infinite-value training-row defect, which was fixed.
- Nine additional verifier tests check direct-OLS reconstruction and deliberately
  corrupt predictions, targets, model rows, and training metadata.
- Five inference tests check exact bootstrap sample length, shared paired draws,
  deterministic resampling, HAC against statsmodels, and eight-way Holm.
- All 29 new tests passed before empirical scoring. Together with the relevant
  existing safety and methodology tests, the automatic pre-run gate passed
  81 tests. The earlier baseline check also passed 89 existing tests.
- The pre-run AR(1) null check covered zero in 187/200 trials (93.5%), above
  the fixed 90% minimum. This is a limited simulation check, not a claim of
  exact 95% coverage in financial data.
- The full repository suite passed **418 tests** in 22.957 seconds. All source
  and test files passed the repository linter. Numerically load-bearing package
  versions matched `requirements.txt`.
- Independent reconstruction verified **6,696 feature rows**, **24,610
  forecasts**, and **1,180 monthly model fits**. Maximum relative forecast
  difference was `6.07e-14`. The verifier uses direct augmented regressions,
  independently generated circular bootstrap indices, statsmodels HAC, and
  statsmodels Holm adjustment. Every point estimate and verdict matched.
- Source identities and **85 existing protocol/report artifacts** matched their
  recorded hashes. The protected evaluation phase was not used by this study.
- A subsequent verifier-only lint repair bound a local helper argument; its
  nine tests and complete numerical verification were repeated successfully.
  The empirical producer and its outputs were unchanged.

The independent verifier hash and complete audit are in `verification.json`.
It hash-checks the preliminary simulation but does not independently rerun it.
The new raw-derived feature and forecast caches are local and ignored by Git;
compact metrics, protocols, tests, reports, and verification are retained here.

## Interpretation and next research choice

None of the eight tests qualifies for the fixed exploratory shortlist. All
eight adjusted p-values equal 1.0; every uncertainty envelope includes zero.
Five-session realized-variance dispersion has the only favorable full-sample
point estimate (+0.227% QLIKE improvement), but it is smaller than the fixed
1% minimum, reverses in 2016–2019, and reverses in one of the five sampling
phases. It should not be promoted or have its windows retuned.

The useful next information upgrade would be a consistent intraday history:
test a fixed downside-semivariance or time-of-day variance-concentration
feature against the same strong control, and measure the target using actual
intraday variance. The repository already documents limitations in the hourly
proxy and gaps in the NQ sample. First resolve coverage, session conventions,
availability, and access terms; then fix the test before acquisition/scoring.
Any historical replication stays exploratory. Genuine confirmation needs new,
prospectively recorded forecasts and does not reopen the existing sealed gate.
