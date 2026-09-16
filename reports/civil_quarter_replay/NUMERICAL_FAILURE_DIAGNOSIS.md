# Training-only diagnosis of the failed independent baseline check

Wave 16 remains **UNEVALUABLE**, with both new comparisons retained at p=1 and no lead. This diagnostic explains a numerical stopping failure; it does not validate the generated forecasts or revive unpublished inference.

The original traceback omitted the monthly fit index. One diagnostic pass therefore replayed only the unchanged independent baseline optimizations, in saved chronological order, stopping at the first failed stationarity check. It used each fit's original mature training mask, training transform, positive-risk normalization, analytic objective/gradient/Hessian, zero initialization, and SciPy `trust-exact` settings. It produced no quarter fits, application predictions, loss comparisons, alternative solver attempts, or parameter search. The complete record, including all attempted baseline terminations and the failing full gradient, is in [numerical_failure_diagnosis.json](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/civil_quarter_replay/numerical_failure_diagnosis.json).

| Check | Diagnostic result |
|---|---:|
| First failing fit | 99th monthly fit, index 98; origin 2025-08-01 |
| Training rows | 3,055 |
| Training origin range | 2011-01-05 through 2025-07-30 |
| Latest training availability / fit cutoff | 2025-07-31 |
| Independent solver target / maximum iterations | Gradient tolerance 1e-10 / 500 |
| Frozen independent acceptance limit | Full gradient infinity norm ≤ 1.0001e-8 |
| Returned gradient infinity norm | 1.817252736312225e-8, largest component `neg_d` |
| Termination | Status 2 after 20 iterations; failure to predict improvement |
| Saved producer gradient, independently recomputed | 6.312225048210607e-16; exact saved-vector replay |
| Producer's frozen acceptance limit | ≤ 1e-8; passes |
| Producer iterations / backtracks / invalid trial rejections | 6 / 0 / 0 |

The preceding 98 independent baseline replays passed their frozen stationarity gate. The diagnostic stopped immediately at fit 99; it did not examine the remaining two baseline optimizations. Runtime versions were SciPy 1.17.1 and NumPy 2.4.6, with the same thread limits as the original verification.

The normalized penalized training objective is 0.6442049925062784 at both the returned independent coefficients and the saved producer coefficients, equal at the stored floating-point precision. Their largest coefficient difference is 1.454385639004796e-8. The independent solution's Hessian is positive definite: minimum eigenvalue 0.020006135145103285 and condition number 275.5833381886003. The transformed baseline has rank 31 and the civil design rank 15; the quarter column's relative residual norm against the baseline is 0.7870590157785744. The original 17 empirical scales range from 0.002120055298335537 to 19.225666421925837, all above the frozen 1e-12 gate. These checks do not indicate a rank or zero-scale failure.

The exact installed SciPy source assigns status 2 when its computed predicted objective reduction is nonpositive. It forms that reduction by subtracting the quadratic model value from the current objective. This agrees with the [primary SciPy implementation](https://github.com/scipy/scipy/blob/main/scipy/optimize/_trustregion.py); the installed version was inspected directly at [.venv/lib/python3.11/site-packages/scipy/optimize/_trustregion.py:262](/Users/byrons/code/trading-vol/ndx-vol-experiment/.venv/lib/python3.11/site-packages/scipy/optimize/_trustregion.py:262). At the failed returned point, the local quadratic Newton-decrease estimate is 1.6891892410344813e-16, approximately 1.52 spacings of the objective value. This supports a finite-precision reduction/stopping explanation. The original trust-radius history was not saved, so the exact failing trial's arithmetic is not proven. A positive-definite Hessian and a very small coefficient difference do not override the registered full-gradient requirement.

A prospective numerical remedy can preserve the objective and tolerance while avoiding objective-value subtraction as the sole route to convergence. In a new registered implementation, analytically profile the unpenalized intercept and solve the remaining gradient equations with a fixed, independently implemented safeguarded procedure. For normalized positive labels q, slope design U, slopes s, and penalty α=.01, the exact intercept is c(s)=log(mean(q·exp(−Us))). The profiled objective is mean(Us)+c(s)+1+α‖s‖². Its Hessian is the weighted covariance of U plus .02I, so the slope problem has a unique optimum and a global .02 curvature lower bound. Stable log-sum-exp evaluation and direct gradient/Hessian checks can supply predeclared convergence evidence; the original full gradient must still pass after reconstructing the intercept.

Before any new historical run, synthetic tests should establish exact profiling equivalence, derivative agreement, full-gradient stationarity, invariance to target units, and convergence when objective changes fall below a floating-point spacing. They should also reject nonfinite arithmetic and exhausted iteration budgets. No such remedy was tried on this training sample. Any later implementation requires a new frozen registration; the current failed comparison remains in the cumulative family.

All watched protocol, verifier, manifest, freeze, canonical failure/metrics/verification, and ledger hashes remained unchanged during the diagnostic. The source feature, target, and fit bytes were checked before use and afterward. The diagnosis JSON SHA256 is `13be3e97d9edfd71c3ba0a887c22ba5b0ecdbc0802255f7409ae886c9ec6d9e6`.
