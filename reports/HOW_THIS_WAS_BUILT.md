# How this was built

This repository was produced with heavy AI assistance over a short calendar
window. That is visible in the commit timestamps, so it is stated here rather
than left for a reader to infer. This document covers the division of labour,
the guardrails, and — the part worth reading — the specific errors the
guardrails caught.

Twenty-two corrections are recorded below, in three rounds. The four in the
second group invalidate or downgrade headline claims this repository previously
made; those are in `reports/METHODOLOGY_FORK.md` in full. The third group is
from 2026-08-13 and includes the worst single failure in the project's history:
a set of corrections that were documented, tested and reported **without ever
having been implemented**.

## Division of labour

**Model** (Claude, via Claude Code and chat sessions): implementation, data
plumbing, parser work, test scaffolding, report prose, and first-draft
statistical specifications.

**Human**: hypothesis selection, the pre-registration discipline, choice of
control model, loss-function selection, leakage policy, and every decision about
whether a result counted. Every correction below originated from human review or
from a review pass explicitly commissioned to attack the work — none from the
model flagging its own output unprompted.

The summary is that the model was a fast, tireless, largely correct executor
that could not reliably tell when a test was incapable of delivering the verdict
it had been registered to deliver.

### What changed in the third round

The 2026-08-13 round was run differently, and the difference is the most
transferable thing in this document. Instead of one model reviewing the work, a
**fan-out of eight independent reviewers** was pointed at separate dimensions —
the restored code, the inference core, prose-versus-code consistency,
degenerate/unreachable gates, leakage and timing, verifier independence, and two
sweeps over the study modules. Each was given the ledger of past failures as a
description of the failure mode to hunt for. Every finding then went through
**two adversarial passes**: one instructed to refute it, one instructed to
reproduce it empirically, with `refuted` as the default. 39 raw findings, 20
verified, **14 survived, 6 refuted**.

The refuted six matter as much as the fourteen. A single reviewing model
produces plausible findings at roughly the same rate it produces plausible code,
and it cannot tell the difference from the inside. Requiring a finding to
survive an agent whose job is to kill it is the same structural move as
pre-registration: it removes the reviewer's discretion over whether their own
output counts.

The human role did not shrink. The fan-out was commissioned by a human asking
"are we sure?", the dimensions were chosen by a human, and every decision about
which surviving finding to act on, disclose, or defer was made by a human. What
the fan-out replaced was the *attention* budget, not the judgement.

## The guardrails

* `config.yaml` and the per-study protocol YAMLs are pre-registration
  artifacts, hashed and frozen before empirical runs.
* Every empirical command runs its safety tests first; a failing fence blocks
  the run rather than warning.
* Frozen verdicts are never rewritten. Post-hoc work is labelled post-result and
  reported alongside, never in place of, the frozen verdict.
* `reports/AMENDMENTS.md` records every change made after results were observed.
* Independent recomputation (`src/verify_*.py`) re-derives headline metrics from
  source rather than trusting the pipeline's own outputs. **Correction, 2026-08-13:**
  this is true of most verifiers and not all — see ledger item 21.
* Corrections are additive forks, not edits.
  `tests/test_methodology.py::TestFrozenReportsUnchanged` asserts that the
  default command line still reproduces the pre-registered reports
  byte-for-byte, so a correction cannot quietly rewrite the result it was meant
  to be compared against.
* Any model handed VXN is scored against HAR-IV, never plain HAR.

## Ledger: implementation and inference errors

1. **Look-ahead in index weights.** The original run applied a single
   2026-08-10 Invesco holdings snapshot across all history — a look-ahead
   sitting directly on the earnings slice. Removing it made `chronos_cov`
   worse, which is the tell: part of its already negligible edge was the leak.
2. **H3 withdrawn — HAC borrowing power from overlap.** The 30-day encompassing
   regression produced a significant coefficient under HAC with lags raised to
   32. Refitting on non-overlapping blocks (every 21st origin, ~117
   observations, ordinary errors) at all 21 phase offsets: `har_cum` reaches
   p<0.05 in 5% of phases, `persistence_cum` in 24%, median p 0.21 and 0.19,
   with the persistence coefficient sign unstable across phases. The HAC
   correction was not wrong; it was being asked to manufacture independent
   observations.
3. **A proposed fix that was algebraically inert.** Faced with collinearity
   between `har_cum` and `persistence_cum`, the suggested remedy was to
   orthogonalise. Frisch–Waugh–Lovell makes the orthogonalised coefficient and
   p-value identical to the original — verified. What the exercise did yield was
   the standardised effect c·sd(e): +0.0660 vs +0.0616, near-identical, showing
   the apparent ordering between them was a collinearity artefact. The fix did
   nothing; the diagnostic it incidentally produced was the finding.
4. **A pre-specified test that was unestimable by construction.** The
   concentration defence was registered before anyone noticed that the
   `pit_weights` build renormalizes the tracked basket to its snapshot total
   daily, making the regressor constant. A specification error discoverable ex
   ante. Recorded as unestimable and disclosed; post-hoc substitutes were run
   and labelled post-hoc, and do not support the defence.
5. **A results characterisation that contradicted the numbers.** The earnings
   covariate had been described as "pays off on a few large days." The
   full-sample paired count reverses that shape: on all 2,463 origins
   `har_iv_x` wins 61.3% of days (p=1.4e-29) with +0.0% mean improvement, the
   top ten days carrying 5,300% of the near-zero gap; `har_x` wins 60.7% with
   mean −0.6%. Many small wins funded by rare large losses — the same shape as
   the carry study's failure and the GBM result.
6. **A confirmation gate that could not be passed.** The latent-probe study
   registered a 95% evidence criterion against ten synthetic controls. Ten
   controls impose a minimum exact corrected p-value of 1/11 = 0.0909; the gate
   was unreachable by construction. No controls were silently added — the
   stored flag was renamed a descriptive heuristic, `formal_evidence` set false,
   and exact p=0.0909 shown for every rung. A separate frozen 99-control run
   followed: 0 of 99 reached the actual 0.8153 AUC, corrected exact p=0.01.
7. **An inert covariate channel reported as working.** TiRex-2's past-covariate
   channel was carrying VXN. An oracle probe — feeding it the answer — left RMSE
   unchanged, proving the channel was ignored. Results were rerouted through the
   future channel (`tirex_cov_ivf`).
8. **Two overstatements in the Eidos-derived noise report.** Review of the
   source paper found that Eidos Appendix A.1.2 re-normalizes corrupted inputs
   using noisy statistics, while this implementation passes raw corruption to
   each model's native preprocessing and does not renormalize HAR's origin state
   — so the HAR/foundation magnitude comparison is not apples-to-apples. The
   stored `crps` is also a trapezoidal approximation over quantiles 0.1–0.9, not
   full-tail CRPS. Both now stated explicitly; no forecasts or draws changed.

## Ledger: the four defects that changed conclusions

Documented in full in `reports/METHODOLOGY_FORK.md`, which adds a parallel
evaluation path rather than editing the frozen reports.

9. **Point forecast estimator.** `trunc_mean_var` discards quantile mass outside
   the grid and returns 0.87 of the true conditional mean, while EWMA — which
   emits a native variance forecast and is not understated at all — sat in the
   same DM table. Replaced with Duan smearing where residuals exist.

   This one was also overstated by the review that found it, and the walk-back
   matters more than the finding. Across the HAR family the frozen estimator's
   bias spans 0.866–0.873 — a spread of 0.70pp, not the 4.7pp originally
   claimed. The inflated figure came from using a lognormal as the reference,
   which is itself dispersion-biased. **And the walk-back was itself wrong in
   two ways — see items 15 and 16.**
10. **Failure to reject reported as a null.** `metrics.dm_test` returned a
    p-value and nothing else, so every non-significant comparison read as
    evidence of no difference. Every DM row now carries a minimum detectable
    effect, the origins needed to resolve the observed gap, a TOST equivalence
    test against a declared 3% margin, and a three-way verdict. A
    non-significant DM alone can now never produce `equivalent`.

    The consequence is that "foundation models add nothing beyond HAR-IV" was
    never established. Conversely the earnings result is refuted rather than
    unsupported — at n=2463 the equivalence test rejects non-equivalence, a
    positive finding of no effect.
11. **No specification dates.** `config.yaml` is frozen but records no date for
    when each model spec was written, so a model designed after seeing
    clean-window results could be scored on that window beside genuinely
    pre-registered models with unadjusted p-values. `spec_registry.yaml`
    generalises the existing hand-applied rule: a model is confirmatory only
    from `max(phase_start, specified_on, available_from)`, and undetermined
    dates are treated as exploratory.
12. **Overlapping-window inference reported n as n_eff.** The 30-day MZ and
    encompassing regressions run on daily origins whose targets share ~21
    trading days, and reported n=171. The effective sample is 8. HAC(32) on
    n=171 is a lag/n ratio of 19%. Replaced with a circular moving-block
    bootstrap plus refits on all 21 non-overlapping subsamples, reporting
    `n_eff` beside `n`. **The replacement is itself miscalibrated — item 17.**

## Ledger: 2026-08-13 — the round that found the machinery missing

13. **The corrected fork was documented, tested, reported — and never
    implemented.** `METHODOLOGY_FORK.md` described items 9–12, `src/methodology.py`
    held the statistics, `tests/test_methodology.py` asserted the behaviour, and
    the `_est`/`_inf`/`_v2` reports sat in `reports/`. The code that produced
    them did not exist. `models.py` had no smearing estimator, `config.py` no
    registry loader, `experiment.py` no `--estimator`/`--inference` flags, and
    `make scenarios` — the command the fork document instructs the reader to run
    — was not a target. **12 of 28 tests in the fork's own suite failed on a
    clean checkout**, one of them because the CLI rejected its own documented
    arguments.

    This is the single most instructive failure in the repository. Every
    guardrail worked as designed and none of them fired, because the guardrail
    protecting the *frozen* reports had no counterpart for the *corrected* ones.
    The corrected reports were unreproducible artifacts sitting in the
    repository looking exactly like reproducible ones. Nobody ran the fork's
    test suite between writing it and citing its output.

    The implementation was rebuilt from the documentation and pinned: the
    restored estimator reproduces the 20 surviving `*_sm.parquet` forecast files
    bit-identically, and reproduces the published reconstruction table exactly.
14. **The clean window was reopened for two quarantined models.** `har_lev` and
    `har_iv_lev` are barred from the clean window until the 500-origin gate "so
    testing them cannot become a peek". The guard was inside the h=1 table loop
    only; the corrected fork's replication panel called the scoring function
    directly and published clean-window win rates, mean gaps and a
    `replicates? yes` verdict for both over all 192 clean origins — and
    `METHODOLOGY_FORK.md` cited the result. Suppressing a row from a report is
    not a quarantine. **~38% of the gate draw is spent with direction known**,
    and no re-run un-spends it. The guard now lives in the scoring function and
    is driven by a registry field that had been computed and read by nothing.
15. **"The estimator fix changes no rankings" was false.** `har_sv vs har`, with
    both sides exactly smeared on identical 188 origins, moves from DM −1.615
    (p=0.1080) to −2.163 (**p=0.0318**) — across the pre-registered α on the
    estimator switch alone. The only `equivalent` verdict in the clean window
    also requires the estimator fix; under the inference fix alone, none of the
    three rows in that table earns a null. A section heading claiming to show
    "fix 2" was showing fixes 1+2.
16. **The cancellation argument was backwards, and the band was subset-scoped.**
    The claim that the estimator bias "largely cancels in paired comparisons" is
    wrong: QLIKE differentials are not scale-invariant, so a common
    multiplicative factor moves the DM statistic rather than cancelling — and
    reproduces most of the observed movement on its own. Separately, the
    "0.866–0.873 / 0.70pp near-common factor" holds only over the five models it
    was measured on; it is 0.812–0.883 (7.05pp) across every quantile model
    scored. The pinning test quantified over the narrow set while the prose
    quantified over all of them, so the test passed while the claim was false.
17. **Both replacement standard errors were misdescribed.** The figure published
    as the "honest standard error" was the median *within-subsample* SE — the SE
    of a coefficient fitted to ~8 points, algebraically ≈ √21 × the full-sample
    OLS SE. It is not a standard error of the reported coefficient, and the
    ratios formed against HAC from it (3×, then 2.9×) are withdrawn. Worse, the
    block bootstrap that replaced HAC is **anti-conservative**: measured
    nominal-95% coverage is 82.2% and the coefficient p-value has true size
    ~10–16%. The claim that a significant `c_model` was "a real rejection" is
    struck, and a Monte Carlo coverage check now runs in the test suite.
18. **Failures to reject were still being reported as established nulls**, in
    the standing findings document, after item 10 had supposedly fixed exactly
    that. H1 was headed "dead past arguing" and described as "a well-powered
    null … not an underpowered maybe". Measured: the minimum detectable effect
    is 43× the observed gap, **power against the observed gap is 0.050**, and
    resolving it would take 357,040 origins. The numbers quoted in that headline
    reproduced nowhere in the repository.
19. **The pre-committed earnings gate cannot fire at its own trigger.** The
    registered confirmatory test requires 40 heavy-earnings days. Over *every*
    rolling window of the length between the window opening and the registered
    trigger date, across the whole sample, the maximum attainable is **34**;
    40 accrues around 2028. The field was also read by no code. `config.yaml`
    is frozen, so this is disclosed rather than repaired — the evaluator now
    prints the reachability arithmetic beside the gate.
20. **The ranking scoreboard mostly measures the calendar.** The registered
    phase-mean AUC pools 24 annual folds whose event rates range 0.000 to 0.808,
    four of them containing no event at all, so most scored pairs straddle two
    different years. A fold-constant score with **zero within-year information**
    reaches 0.8317 — above two published headlines, including the 0.8153 latent
    result that cleared the 99-control gate. Every model's within-fold AUC is
    far lower (0.677–0.731) and the ordering reorders: the calibrated HMM is
    best within fold while ranking third pooled. Both the pooling and the
    control design are *registered*, so the verdicts stand and this is published
    additively in `reports/representation_study/pooling_diagnostic.md`.
21. **A verifier claimed independence it did not have.**
    `verify_target_regime.py` re-derives features, thresholds, targets, lags and
    fences from raw, but never re-derives a single forecast probability — it
    aggregates the pipeline's own stored columns. Demonstrated by injecting real
    look-ahead into the model (full-sample HMM fit, smoothed rather than
    filtered states) and watching the verifier still print `VERIFICATION PASSED
    … both frozen verdicts independently match`. Currently inert: independent
    re-derivation reproduces every stored probability exactly, and the leakage
    pushes toward PASS while both frozen verdicts are FAIL. The banner now
    states its actual scope.
22. **Open, and recorded rather than quietly dropped.** Two audit findings
    require protocol amendments rather than re-runs and are not fixed: the NQ
    study's "no evaluable folds" appears to be a rolling-window construction
    artifact rather than a shortage of history, and the noise-robustness HAR arm
    aggregates on the variance scale while its impulse is injected on the log
    scale, which reportedly flips the significance of all eight Gaussian
    intervals. A further 19 lower-severity findings were never adversarially
    verified and carry no claim.

## The pattern

Nearly every entry is the same failure mode: machinery that is individually
correct and collectively incapable of answering the registered question. The
model does not notice that a standard-error correction is being asked to
manufacture independent observations, that a remedy is algebraically inert, that
a regressor is constant, that a p-value has a floor above its own threshold,
that a reported n is twenty times the effective one, or that a ranking statistic
is mostly measuring which year it is. It notices none of these because each
component is right.

The third round adds two variants worth naming separately.

**Documentation can substitute for implementation without anyone noticing**
(item 13). Prose describing a correction, tests asserting it, and reports
displaying its output are three independent artifacts that all look like
evidence the correction exists. None of them is. Only running the thing is.

**A correction can inherit the defect it corrects** (items 15–18). The fix to
"failures to reject reported as nulls" was itself reported as a null. The fix to
"HAC understates the standard error" replaced it with a bootstrap that also
understates the standard error, and quoted a ratio built on a statistic that
measures something else. The fix to "the estimator is grid-dependent" shipped
with a pinning test scoped to exactly the models where the claim held. Reviewing
a correction as carefully as the original is not optional, and it is the step
most likely to be skipped, because the correction arrives wearing the
authority of having caught something.

Speed of implementation raises the volume of plausible-looking output faster
than it raises the rate of catching bad inference. So the catching has to be
structural — pre-registration, independent recomputation, additive forks with a
byte-for-byte guard on the frozen artifact, adversarial verification with
refutation as the default — rather than attentional.

The corollary is uncomfortable and worth stating: the four defects in the second
group survived eight prior rounds of review, including rounds that produced the
first group. The nine in the third group survived those, plus the round that
produced the second group. Each review found real errors and left these. The
previous version of this document ended "There is no reason to believe this
document is the last one." That was written on 2026-08-12 and was proven right
the following day, by a margin of nine.

## 8 → 12 → 22 is not a converging sequence

It is tempting to read three rounds of 8, 12 and 22 findings as a process
tightening up. It is not. Each round's findings had **survived every prior
round**, including rounds explicitly commissioned to attack the work. The count
went up, not down, and the third round found defects in the corrections produced
by the second. Nothing in that trajectory licenses the inference that the
remaining defect count is small.

Reviewing harder is not the same as converging, and a reader has no way to tell
the two apart from the outside.

**And the sequence is confounded, which the first version of this section missed.**
Each round also *added new studies* — the free-source acquisition, the
representation extensions, the residualized probe. So 8 → 12 → 22 partly counts
**new surface** rather than residual error in old surface. A count that rises
while the codebase grows says nothing about whether review is converging, and
citing it as if it did was the same category error as reading a rising p-value
as evidence. Corrected 2026-08-13 after external review.

The version of the test that can actually pass:

> **Freeze scope for one round and re-review only what existed before it.**
> Convergence means a scope-frozen round producing **zero conclusion-changing
> findings**. No scope-frozen round has been run. Until one is, nothing here
> has been shown to converge, and the three rounds to date are not evidence
> either way.

Every result in this repository remains provisional by the project's own
standard — not merely by a reader's suspicion.

Two things follow. First, "22 corrections caught" is not a claim of quality; it
is a claim about the *rate at which this method produces defects*, which is the
honest thing to take from it. Second, the open surface is enumerated rather than
gestured at: `docs/OPEN_FINDINGS.md` lists the confirmed-but-unfixed findings,
the guardrail holes, and the 19 leads that were never adversarially verified. A
sentence like "19 lower-severity findings remain" makes the backlog
unenumerable; a list makes it a backlog.

## What to distrust in this repository

* Every headline QQQ table uses a Garman–Klass daily estimate plus the squared
  overnight gap, not five-minute realized variance.
* Event-sliced results run on n=6–12 in the clean window. Descriptive, full stop.
* The clean window contains no volatility event and cannot settle tail questions.
* The ~16 studies are not statistically independent; their count is not a
  meta-test.
* Headline table rows written before the methodology fork should be read against
  `reports/METHODOLOGY_FORK.md`, which supersedes several of them.
* Any AUC from the annual-fold ranking scoreboard should be read against
  `reports/representation_study/pooling_diagnostic.md`; most of it is
  between-fold.
* `har_lev` and `har_iv_lev` no longer have a clean draw — 192 origins of it
  were spent and published before the quarantine was repaired.
* Earnings announcement *sessions* (before/after the bell) are inferred from
  timestamps, not verified against company IR pages. A wrong session shifts a
  vol day by one.
