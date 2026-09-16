# Prospective QQQ–SPX relative intraday risk question

**Blinded, unregistered design review, 2026-09-07.** This note did not inspect wave 8 results, construct new empirical features or targets, count joint observations, fit models, acquire data, or change an earlier artifact.

**A relative-risk target was not found in the inspected implementations, but the available pair is QQQ ETF versus the SPX price index—not Nasdaq-100 index versus SPX.** A small relative **intraday OHLC-risk** study is a plausible next question, conditional on a joint source/calendar audit. It would change the target and test one joint return statistic; it would not introduce new raw information or establish that relative risk is easier to predict.

## What is already covered

| Existing implementation/specification | Overlap and distinction |
| --- | --- |
| [Orthogonal round two](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/orthogonal_round2.py:23) | Absolute QQQ total-risk targets already have both VXN and VIX controls. Its correlation candidate is trailing QQQ–TLT return correlation, not a QQQ–SPX relative target or their intraday correlation. Cross-asset stress uses HYG/TLT/GLD/USO/UUP. |
| [Wave 1 specification](/Users/byrons/code/trading-vol/ndx-vol-experiment/iterative_signal_search.yaml:9) and [overnight source/target contract](/Users/byrons/code/trading-vol/ndx-vol-experiment/overnight_index.yaml) | SPX high-frequency variance and QQQ daytime/overnight signed returns were forecast separately. Session decomposition and signed cross-ETF returns are already tested information channels; this is not a reason to repeat those blocks under a different title. |
| [International specification](/Users/byrons/code/trading-vol/ndx-vol-experiment/international_volatility.yaml:16) | Foreign-market RV summaries and train-only PCA predict absolute SPX RV. No relative QQQ–SPX response is defined. A new regional or latent arm would overlap this work. |
| [Model/memory specification](/Users/byrons/code/trading-vol/ndx-vol-experiment/model_memory_study.yaml:6) | Changed losses, splines, observable/latent memory, and ensembles target absolute QQQ variance. A relative target would not make those estimators new. |
| [Research-path specification](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/RESEARCH_PATHS_PROTOCOL.md:58), [wave 6 inputs](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/index_hinge.py:13), and [wave 7 target](/Users/byrons/code/trading-vol/ndx-vol-experiment/tail_shape.yaml:25) | Separate NDX-proxy horizon analysis, SPX implied-versus-future-risk measurement, SPX signed returns, and standardized SPX downside shape are different responses. The SPX constituent [dispersion contract](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/SPX_DISPERSION_DATA_CONTRACT.md:23) requires a missing tracking basket and is not a two-index correlation forecast. |

These inspected target definitions support a narrow “not found” statement. They do not certify that every historical exploratory notebook or unrecorded comparison is absent. Earlier absolute-risk studies already use both IV levels and cross-market information, so a passing result here could only establish target-specific incremental value under the new fixed design.

## Source identity determines the question

The existing [configuration](/Users/byrons/code/trading-vol/ndx-vol-experiment/config.yaml:6) selects `ticker: QQQ`; its `index_symbol: ^NDX` is described as a cross-check. [The actual downloader](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/fetch.py:46) stores QQQ `open,high,low,close,adj close,volume` in `data/raw/daily_ohlc.parquet`, using `auto_adjust=False` and discarding corporate-action event columns. That configuration comment is not evidence of an available, audited Nasdaq-100 OHLC source. [The SPX acquisition](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/research_paths.py:1327) stores Yahoo `^GSPC` raw OHLC in `data/research_paths/spx_daily.parquet`.

The prior [QQQ source review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/overnight_index/SOURCE_FEASIBILITY.md:13) and [SPX source review](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/index_hinge/SOURCE_REVIEW.md:19) establish archival field validity separately. They do not establish a joint observation calendar or interchangeable ETF/index measurements. Raw QQQ close/overnight changes can contain distributions and fund-specific effects; using adjusted QQQ returns against raw SPX returns instead mixes return conventions. There is no reason to relabel either as matched index excess returns or a variance-swap spread.

The bounded proposal below uses only each session's raw OHLC ratios. This avoids the direct previous-close ex-distribution adjustment in the **target**, and any common multiplicative price adjustment within a session cancels. It does not remove ETF tracking, intraday pricing, source-revision, or index-construction differences. Those limits belong in the name **QQQ–SPX relative intraday OHLC risk**.

Use the existing VXN/VIX/VIX9D/VVIX CSVs only as lagged controls. VXN is a Nasdaq-100 measure and therefore a proxy for QQQ; VIX and the short SPX term structure describe the SPX side. Their already documented units and historical-vintage limits carry over, including VIX9D's back-calculated early history. No implied-volatility ratio becomes a measured relative variance risk premium.

## One candidate against both assets' marginal histories

For asset `a` and session `s`, define the same fixed intraday risk proxy used elsewhere:

`g[a,s] = max(0.5·log(H/L)^2 − (2·log(2)−1)·log(C/O)^2, 1e−10)`.

For origin `t`, let `s` be the next observed SPX session, used **only to label the outcome**. The proposed single-horizon response is

`y[t] = log(g[QQQ,s]) − log(g[SPX,s])`.

This is a forecast of the conditional mean of a **log risk ratio** under squared error. It is not a mean-variance forecast, a ratio of conditional expected variances, or the variance of a hedged QQQ-minus-SPX portfolio; portfolio variance also needs covariance. Exponentiation alone cannot change those estimands.

Use one scalar increment: **strict trailing 22-session Pearson correlation between the two raw open-to-close log returns**, ending at `t−1`. Its motivation is whether changing common intraday movement contains information about subsequent relative risk after controlling each asset's own risk. A zero correlation denominator is unknown; there is no filling, clipping to an interior bound, alternate window, sign search, or latent representation. This joint statistic is not determined by the separate marginal means/variances, but its forecast value remains unknown. It is the same general correlation technique previously applied to QQQ–TLT, applied to a distinct pair and response—not a newly invented model.

The common baseline should include **both** assets' strict 1/5/22 mean intraday GK risk and total GK-plus-raw-overnight risk (log after averaging), both assets' 1/5/22 mean negative raw close-return parts, lagged log VXN, log VIX, log(VIX9D/VIX), log VVIX, and entry weekday indicators. All market features end at `t−1`. The candidate adds only the declared correlation to this baseline. Fit the same fixed normalized ridge objective on the same mature rows, with train-only population scaling and an unpenalized intercept; keep an exactly same-row historical-mean control. The baseline intentionally retains the separate components of relative risk and IV.

**Do not add `log(g_QQQ)−log(g_SPX)` or `log(VXN)−log(VIX)` as new linear features when their components are already controls.** They lie in the existing linear span and can only change ridge geometry. Likewise, with identical rows, regressors, scaling and penalty, subtracting two linear ridge forecasts of component log risks equals fitting their log-risk difference directly. Synthetic verification should establish that identity; a repackaged pair forecast is not an additional information arm. The sole prospective addition here is the joint intraday-correlation statistic.

## Timing, support, and the next decision

Preserve the full bounded SPX observed calendar. Align QQQ by exact civil date without intersecting away missing rows before rolling. An absent QQQ session remains missing. When a baseline uses previous-close changes, require the source's preceding observed session to agree with the reference preceding session; do not manufacture a longer return across an unmatched date. The source audit must enumerate any mismatch and its consequences before registration. A label requires the target-date QQQ and SPX observations for the same session; do not substitute the next common date. Both label observations mature at that session's close. The monthly fit date must depend on current features, not whether a future paired label exists.

Maintain the existing source fence 2025-10-20 and sealed boundary 2025-11-03, previous-session market cutoff, minimum 1,000 common mature training rows, and fixed development/evaluation periods. The separately audited QQQ/SPX histories and IV availability make a 2016 start plausible; **their exact joint sample, correlation validity, floor-hit frequency, and first-fit feature scales have not been checked here**. Prior single-asset counts cannot be carried over as the new feasibility count. A future source/date check should establish calendar identity and source support first; only a subsequently frozen measurement contract should authorize numeric target-quality checks and the experiment.

If that audit passes, register only the correlation augmentation versus the strong baseline and versus the same-row mean: two primary contrasts with the continuing wave/cumulative accounting, a predeclared useful MSE effect, dependence-aware uncertainty, and fixed-period stability. No parallel relative-return, overnight, alternate-horizon, or model search is proposed. If source semantics or support fail, record that outcome and stop this design rather than switching assets, conventions, or targets after inspection.

**Recommended immediate action: a joint source/calendar and measurement-contract audit, not a forecast run.** Exact Nasdaq-100-versus-SPX index claims require a separately identified and audited Nasdaq-100 source; the current files support only the explicitly named ETF/index proxy question above.
