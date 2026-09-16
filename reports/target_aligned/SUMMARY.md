# Directly fitted shared-risk models: verified, no qualifying signal

The new target-aligned dynamic model fails all three registered comparisons. Its mean product-MSE is worse than the existing QLIKE dynamic model in both development and evaluation. Against the new constant-correlation model trained for the same target, it worsens development, improves evaluation on average, and worsens the first fixed evaluation slice. It improves average scores against the weaker constant matrix, but those gains remain statistically uncertain.

This is a completed, independently verified negative experiment, not an unevaluable or numerically degenerate run. All 118 original monthly fits produced both new scalar stages; **none had a flat slope objective**. The original 7,386 forecast rows were preserved exactly, and **4,924 new forecasts** were added.

## Experiment

At each original monthly cutoff, apply the saved mean and variance coefficients to the exact mature training rows. Define current-fit residual product `Y=e_QQQ*e_SPX` and geometric marginal moment `D=sqrt(h_QQQ)*sqrt(h_SPX)`. A single positive training unit normalizes both; compensated sums preserve the intended unweighted squared-error objectives. Fit a bounded constant correlation, then one bounded adjustment driven by the original standardized lagged 22-session correlation. Hold the constant coefficient fixed while fitting the adjustment.

Both scalar stages have global one-dimensional constrained least-squares solutions. They are not jointly optimized in two parameters. The correlation bound is 0.995, and exact flat objectives have a predeclared canonical zero slope with fold/model retention. Each new application copies the exact previously issued means and marginal moments; only the correlation and its resulting product forecast change. The model class also differs from the earlier tanh-index model, so this experiment cannot attribute a win or loss solely to changing the training loss.

The primary score is `(Y-q)^2`, where scored Y uses the common means actually issued at that origin and `q=rho*sqrt(h_QQQ)*sqrt(h_SPX)`. Negative paired score differences mean improvement. Values below are fourth-power decimal raw log-return units. The fixed useful-effect reference is an absolute decrease of `1e-10` in both phases against all three controls, with improvements in both fixed later slices. This reference is not a measured economic benefit or a literal RMSE reduction.

Use the original 2,462 scored origins: development January 4, 2016–December 30, 2019 (1,005 observations), and evaluation January 2, 2020–October 17, 2025 (1,457). The prediction target is the next observed SPX session’s paired intraday QQQ ETF/SPX price-index return product about the shared issued means. The original one-session predictor lag, source fence of October 20, 2025, and sealed period from November 3, 2025 onward remain in force.

## All registered results

| Control | Period | Candidate MSE | Control MSE | Paired difference | Nominal 95% interval envelope | Conservative phase p |
|---|---|---:|---:|---:|---|---:|
| Target-aligned constant correlation | development | 2.415774646e-08 | 2.383490853e-08 | +3.228379314e-10 | [-6.125853845e-11, +7.563684921e-10] | 0.11431000 |
| Target-aligned constant correlation | evaluation | 1.309206991e-07 | 1.317306476e-07 | -8.099484953e-10 | [-2.745545835e-09, +8.830508838e-10] | 0.49944000 |
| Original constant matrix | development | 2.415774646e-08 | 2.595995611e-08 | -1.802209653e-09 | [-4.242189898e-09, +4.205289120e-11] | 0.07003000 |
| Original constant matrix | evaluation | 1.309206991e-07 | 1.483864184e-07 | -1.746571928e-08 | [-3.861416726e-08, +5.370498329e-10] | 0.05723260 |
| Original QLIKE dynamic correlation | development | 2.415774646e-08 | 2.304236922e-08 | +1.115377244e-09 | [-3.113919978e-10, +2.818925403e-09] | 0.15580000 |
| Original QLIKE dynamic correlation | evaluation | 1.309206991e-07 | 1.302356334e-07 | +6.850657112e-10 | [-6.161902963e-09, +7.253569825e-09] | 0.83466658 |

Every nominal interval envelope contains zero. Each envelope combines percentile circular-block bootstrap intervals at blocks 21/63/126 and the Bartlett HAC126 interval. Each phase p-value is the maximum of their two-sided tests, with centered-null bootstrap tests. Percentile intervals and centered-null tests need not be exact inversions; the frozen decision uses the conservative p-values. There is no alternative loss, percentage conversion, or selected favorable method.

| Control | Both-phase p | Holm3 p | Holm117 p | Result |
|---|---:|---:|---:|---|
| Target-aligned constant correlation | 0.49944000 | 0.99888000 | 1.00000000 | Does not qualify |
| Original constant matrix | 0.07003000 | 0.21009000 | 1.00000000 | Does not qualify |
| Original QLIKE dynamic correlation | 0.83466658 | 0.99888000 | 1.00000000 | Does not qualify |

Both phases must pass the fixed effect requirement and the model must pass all three controls. Holm3 must be below `0.05/(12*13) = 0.0003205128205`; separately, Holm117 must be below `0.05`. The 114 inherited hypotheses and all three new hypotheses remain counted. None of these comparisons qualifies even before the multiplicity requirement is considered alongside the full effect/stability requirements.

## Stability and uncertainty

| Control | 2020–2022 gap | 2023–October 2025 gap | Development nominal MDE / effect reference | Evaluation nominal MDE / effect reference |
|---|---:|---:|---:|---:|
| Target-aligned constant correlation | +1.948247514e-10 | -1.893555592e-09 | 5.490 | 24.199 |
| Original constant matrix | -1.876257563e-08 | -1.606711244e-08 | 26.362 | 257.328 |
| Original QLIKE dynamic correlation | -2.974739058e-09 | +4.632016361e-09 | 20.394 | 91.954 |

The nominal HAC-based 80% minimum detectable effect is a diagnostic at the usual two-sided 5% level, not a power claim at the stricter wave/family thresholds. Evaluation uncertainty is large: the nominal detectable effect is about 24.20 times the reference against the matched constant model and 91.95 times against the existing QLIKE dynamic model. Failure to qualify does not establish equivalence or absence of predictability. All annual diagnostics remain recorded; no period was selected for promotion.

## Implication and limits

This specific direct-fitting approach does not improve the existing dynamic forecast on average in either phase. The matched constant control also shows that adding the chosen state adjustment is not a stable increment. The current evidence therefore does not support promoting either this adjustment or the complete new specification as an orthogonal predictive signal. It does not establish that every target-aligned model would fail.

Training residuals are about the frozen current monthly fit’s mean estimates, so they are causal at the fit cutoff but in sample for those marginal models. Application products use the exact issued means. The conditional expectation of Y equals conditional covariance plus the product of the two conditional mean errors; a daily Y is a noisy observation. Unweighted product loss depends on cross fourth moments. Frozen marginal forecast error, changed model class, archival ETF/index data, and repeated historical reuse limit interpretation. No measured high-frequency covariance, fresh-holdout confirmation, exact Nasdaq-100 index, trading execution, or profit claim is made.

## Validation and accounting

- **1,366 repository tests passed** in 41.578 seconds. The runner then passed **82 prewritten focused checks** in 3.403 seconds before admission/fitting. Scoped lint passed for all eleven new source/test files. The unrelated existing whole-repository lint issue was not changed.
- The first independent verification reconstructed **118 monthly fits and 236 scalar stages**, **1,170,384 training moment cells**, all **12,310 forecast rows**, **36,930 primitive product-score cells**, **7,386 paired gaps**, six phase analyses, and **18 bootstrap runs of 99,999 draws**. All 2,463 original application origins and 2,462 scored origins were retained. There were no repairs, retries, or tolerance changes after fitting.
- All **634 pinned manifest entries** remain unchanged: 211 source/test files, 97 input/upstream artifacts, and 326 preserved records. The 200 previously frozen source/test files were retained, and all eleven previous wave manifests were checked before the new run. Prior admission and verification records were reconstructed through read-only functions.
- The complete ledger contains **120 events**: 114 inherited hypotheses, three registrations, and three evaluated outcomes. All three unsuccessful trials remain in the cumulative 117-hypothesis family.
- Across the twelve recent waves, 63 comparisons have been added to the earlier 54 enumerated hypotheses. The generated-forecast total is now **126,203 = 121,279 + 4,924**. The 7,386 reused rows in this wave are not counted again as newly generated forecasts. These totals are not repository-lifetime counts.

Protocol SHA256: `ca11ec1c90f8d83867b6dce4efaaac4f0e5ed11e23143e2ab1f28d78a9f136b3`. Independent verifier SHA256: `93e8ac7f88624fec0933600a85062397c9b0b2306afe87d0b4e3ee4af0f751d2`.

## Prospective next question

The next proposed experiment asks whether recent strict same-direction QQQ–SPX movements improve a probability forecast beyond lagged risk, continuous correlation, return history, and marginal sign frequencies. Its bounded Brier score avoids dependence on return fourth moments, but it answers a different question about directional agreement. The proposal is unregistered, and no new sign counts, associations, or fits have been calculated. It would not by itself demonstrate better volatility magnitude or standalone index-direction forecasts.

## Files

- [Frozen protocol](/Users/byrons/code/trading-vol/ndx-vol-experiment/target_aligned.yaml)
- [Prefit design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/DESIGN.md)
- [Independent design and implementation review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/DESIGN_REVIEW.md)
- [Pre-run freeze record](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/freeze_record.json)
- [Full repository checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/full_pre_fit_checks.txt)
- [Runner prefit checks](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/pre_run_checks.txt)
- [Run manifest](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/manifest.json)
- [Complete metrics and annual diagnostics](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/metrics.json)
- [Independent verification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/verification.json)
- [Complete trial ledger](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/trial_ledger.jsonl)
- [Comparison figure (PNG)](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/comparison_intervals.png)
- [Comparison figure (PDF)](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/comparison_intervals.pdf)
- [Unchanged previous direct-score study](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/cross_moment/SUMMARY.md)
- [Prospective directional-agreement memory design](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/target_aligned/NEXT_SIGN_DEPENDENCE_DESIGN.md)
