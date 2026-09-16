# Goal-completion review: a qualified joint-index research finding

**Assessment:** wave27 meets a meaningful research stopping milestone under the broadened objective, “keep iterating till we find something, either volatility or indexes.” It finds an incrementally better joint QQQ/SPX predictive distribution against both declared controls. It does not complete a narrower claim of finding a new orthogonal volatility input, better stand-alone variance forecasts, index-return alpha, or an operational trading strategy. Those distinctions must remain explicit in the completion message.

The substantive research search may conclude after the current result's preservation audit and readable publication are finished. Continued historical model searching is not required merely because future confirmation and practical application remain unproved. Those are subsequent validation objectives, rather than achievements supplied by this result. This review does not mark or change the goal. At review time the terminal and computational proofs were complete, and SCIENTIFIC_RESULT_REVIEW.md was available; the separate closure audit and final readable publication remained assigned work.

## What the saved result establishes

[metrics.json](metrics.json) records COMPLETED, the sole lead joint_copula, and two COMPARISON_GATE_PASS verdicts. All three arms share the same conditional mean, variance and fixed t8 marginal distributions; only the dependence model changes. The relevant endpoint is negative joint log density of the next SPX session's QQQ ETF/SPX price-index intraday return pair, not a variance proxy or return-direction score.

| Control | Development mean difference | Evaluation mean difference | Interpretation |
|---|---:|---:|---|
| Separately fitted Gaussian copula | -0.030569 | -0.031017 | Incremental evidence beyond a fitted dependence control |
| Independence | -0.606797 | -0.746935 | Dependence beats the product of the same marginal forecasts |

Differences are candidate minus control in nats per pair; lower is better. They must not be presented as percentage volatility accuracy or expected return. There are 1005 development and 1457 evaluation scored origins, totaling 2462. Both comparisons meet the frozen effect, support, subgroup-sign and multiplicity gates. Each saved Holm151 probability is 0.0003775 and Holm-wave probability is 0.000005. The block probabilities reached their simulation resolution floor of 0.0000025; smaller exact probabilities are not established. The [scientific result review](SCIENTIFIC_RESULT_REVIEW.md) checks the full phase, uncertainty and subgroup evidence.

[verification.json](verification.json) records VERIFIED, including 7389 issued application forecasts, 7386 scored forecasts, 118 monthly fits, 472 shared marginal fits and 236 dependence fits. The score proof verifies both comparisons and all 149 inherited comparisons. The [test gate](TEST_GATE_RESULT.json) records 3400 executed tests, zero failures/errors/skips, and six disclosed quarantines for historical-data access. I checked the current metrics, both separate proofs, combined proof and test-gate hashes against [terminal.json](terminal.json), plus the proof's protocol hash against the actual YAML. I did not rerun tests, forecasts, fitting, density calculations, inference or full source-preservation checks.

The 151-comparison adjustment is the maintained numerical family, not a complete census of every historical repository search. [PRIOR_SEARCH_INVENTORY.md](../../iterative_signal_search/PRIOR_SEARCH_INVENTORY.md) explicitly preserves that limitation. Reused development/evaluation history and exploratory interpretation remain material even after passing these gates.

## Coverage of the original requests

| Request | Saved work and honest completion status |
|---|---|
| Review methodology and test orthogonal volatility signals | [METHODOLOGY_REVIEW.md](../../orthogonal_round2/METHODOLOGY_REVIEW.md) covers timing, exact smearing, dependence-aware inference, target consistency, proxy limits and matched controls. The original eight comparisons were run and retained. Finding a new successful external volatility input remains unachieved; the review and empirical tests themselves are complete. |
| Try altered objectives, layered models and latent memory | [model-memory SUMMARY.md](../../model_memory_study/SUMMARY.md) documents objectives, recency, regularization, nonlinear functions, causal calibration, observable and latent error retrieval, and equal/dynamic ensembles. The final study contains 15 models including the benchmark and 46 comparisons. None of its 14 alternatives qualified as improving the established volatility benchmark. The experiment does not establish that every possible memory method must fail. |
| Consider the supplied library and four references | All supplied references were reviewed in [REFERENCE_REVIEW.md](../../model_memory_study/REFERENCE_REVIEW.md) and [LIBRARY_AND_LARGE_MODEL_REVIEW.md](../../model_memory_study/LIBRARY_AND_LARGE_MODEL_REVIEW.md). The specific execution distinctions below remain essential. |
| Continue until something useful is found in volatility or indexes | The new result is a qualifying exploratory index joint-density lead. It supplies model-level predictive structure using existing inputs. It is not a new exogenous information source or a demonstrated improvement to either individual marginal forecast. |

The supplied arXiv2511.11698 (Moirai2.0) led to a pinned official small-checkpoint run with causal calibration. The supplied NeurIPS xLSTM-Mixer paper led to a disclosed small CPU adaptation with matched NLinear, not an exact paper reproduction. [reference_metrics.json](../../model_memory_study/reference_metrics.json) and [reference_verification.json](../../model_memory_study/reference_verification.json) retain those results and verification; the combined46-comparison family remains authoritative for its final multiplicity interpretation. Moirai's improvement from adding the established controls to its univariate arm does not demonstrate incremental Moirai signal over the strong controls.

Time-Series-Library was inspected as an implementation source; no exhaustive architecture sweep was run. TimeCopilot (arXiv2509.00616) was reviewed as orchestration, with no package/LLM-selection forecasting arm. Timer-S1 (arXiv2603.04791) was reviewed and deferred under the resource observations recorded at that time; no checkpoint forecast was produced. Those hardware observations have not been remeasured by this review. The user's later links asked for references to look at; they do not explicitly require reproducing every paper, purchasing infrastructure or running every library architecture. These omissions should be disclosed, not falsely claimed as completed experiments, but they do not by themselves keep this research milestone unfinished. The earlier “all of the above” instruction predates those supplied links and should not silently be expanded into an infinite model sweep.

## Claims still unmet and the next needed validation

The strongest defensible immediate application is a candidate model for joint return scenarios and dependence. No portfolio loss, VaR/expected-shortfall calibration, hedging improvement, transaction-cost result, economic utility or deployment test was part of the endpoint. A deployment claim would require those additional task-specific checks. A pure tail-dependence mechanism is also not identified: the entire copula changes, and the shared marginal models may be misspecified.

The original request for useful volatility information has therefore produced a different successful forecast functional after scope broadened to indexes. It would be misleading to label that as fulfillment of a narrower volatility-alpha objective. The broader request to find useful modeling structure can be fulfilled as an exploratory research finding; nothing in the supplied task requires silently upgrading that to proven profitability or guaranteed future performance.

The next evidential step should freeze this model, the fitted Gaussian control and independence, the input/issuance clock and a primary joint-log-score comparison before accessing subsequently untouched outcomes. Preserve all forecasts, gaps and failed issuances. Predeclare any marginal-calibration or joint-event/portfolio-risk diagnostics as separate endpoints. If a concrete use such as portfolio risk management is chosen, select its decision rule and transaction/instrument assumptions prospectively, then assess practical value against the same controls. This report starts no new experiment or recurring run.

The period beginning2025-11-03 cannot be relabeled untouched: the [historical access correction](../../treasury_dealer/predictive_prefit/HISTORICAL_TEST_ACCESS_CORRECTION.md) documents prior regression-test access. Current-vintage price/IV snapshots, back-calculated VIX9D training history and unverified synchronized auctions/publication latency also remain limits. New confirmation needs genuinely uninspected outcomes and an explicit source-availability record.

## Suggested completion message

“We found a qualifying result: a heavy-tailed dependence model predicts the joint QQQ/SPX return distribution better than both fitted Gaussian dependence and independence on the reused history, and it passed the predeclared checks. This is a joint-risk modeling lead; it does not improve the individual volatility forecasts or establish trading profit. The earlier memory and reference-model tests remain negative against the volatility benchmark. The next validation is to freeze this result and test it on genuinely untouched outcomes before using it operationally.”

## Evidence hashes

- reports/joint_copula/predictive/metrics.json: `d1de728c8fb6909a4ce8ecd6edb055d8b3652693550de140dfca3ff688593442`
- reports/joint_copula/predictive/terminal.json: `9dfbe34029b5b8095fd68ff0c3e1e624bd854de287449a1045b6700906742b2f`
- reports/joint_copula/predictive/verification.json: `6abe3595a307ba61860093b12d891696d95ca41451f68397ad470f8840d907d4`
- reports/joint_copula/predictive/SCIENTIFIC_RESULT_REVIEW.md: `64281d03d8ebe0a7fac38a704def3edd88b446099ccfbb31f16b715d22f1016f`
- joint_copula.yaml: `cf43700c6c37ba860e9ef8ae4adb4c70158030efeedd3bca9b073f0691cd5d50`
- reports/orthogonal_round2/METHODOLOGY_REVIEW.md: `3311e5a35b55c88572da92765c5ff5631dcd3b9b9e94d5121211be4fd9869344`
- reports/model_memory_study/SUMMARY.md: `f3c34ffd6a6cf3ad52d4afac942f7b40db14957314ce9134df31e57349538935`
- reports/model_memory_study/REFERENCE_REVIEW.md: `41fba634808d0199442a3b3fcacc1c823c556544ff6005554aaa1458ddef95b9`
- reports/model_memory_study/LIBRARY_AND_LARGE_MODEL_REVIEW.md: `7037b2e1eb68652b1ec679ba632b16d08dff25ac51959e1179d35699b901fcbf`
- reports/model_memory_study/reference_metrics.json: `6e319b6e0d3804f822a63dfad6f5e09ea410cb119fd027845c02d74d50d95b32`
- reports/model_memory_study/combined_metrics.json: `03917199569cfe00efb53eb1fa0393b745225866df8f24cfd9046d8cdc430e7c`
- reports/model_memory_study/reference_verification.json: `d770b0851a77b1330d4080cfede5ab868a3283ed268eed5653f5a23cb025e2ed`
- reports/iterative_signal_search/PRIOR_SEARCH_INVENTORY.md: `2ddac8944bdfaab2da2a31e001a20208e6cc4d21920836981f7b13ddd430a9a3`
- reports/treasury_dealer/predictive_prefit/HISTORICAL_TEST_ACCESS_CORRECTION.md: `325915e178d0ebc58ac05ced3d281a8c8ec840ed86fd9fec348e5ae0969ab007`
