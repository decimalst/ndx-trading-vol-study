# Full-calendar masked paired-mean inference core

Prospectively specified September8,2026. This is a numerical primitive for generated-data testing; it performs no source admission, market loading, registration, model fit or study-level support/multiple-comparison decision. Root review and independent verification/calibration remain required before empirical use.

## API and exact output

`masked_mean_inference(differences, mask, *, blocks=(21,63,126), hac_lags=126, draws=399999, seed=20261001)` takes a one-dimensional full observed-session array and a matching boolean mask. It returns exactly:

- `mean`: selected paired mean, Python float.
- `n`: number of selected observations; `full_calendar_n`: uncompressed calendar length, Python integers.
- `hac`: `se`, `p`, `ci95` and `mde80_nominal`.
- `block_inference`: each requested block's decimal-string key maps to `p` and `ci95`.
- `p_conservative`: maximum of the HAC and all requested block p-values.

Intervals are two-element lists of native floats. No partial result is returned after a failed block. Inputs are not mutated.

## Calendar and mean

Let T be the full calendar length, m[t] the boolean mask, n=sum(m), q=n/T, and μ=sum of selected differences divided by n. Gaps remain calendar positions. A missing difference may be NaN only where the mask is false. Finite placeholders, even large ones, at false-mask cells are ignored before arithmetic. Infinity is rejected anywhere.

For masked sums, allocate zeros and assign only selected values; never calculate NaN multiplied by zero. These internal zeros represent absent contributions, not observed zero losses or fabricated source values. Masked-out values never enter the mean, influence variance or bootstrap numerator.

The caller supplies the complete session order and chooses the mask. This API has no dates and cannot certify that an array represents the correct calendar, phase, source clock, matured target or common model cohort.

## Bartlett HAC

Use the full-calendar influence ψ[t]=m[t]*(d[t]−μ)/q, with explicit zero influence at false-mask positions. With L=min(hac_lags,T−1), compute γ[k]=sum from t=k throughT−1 of ψ[t]*ψ[t−k], divided byT.

The long-run variance is γ[0]+2*sum((1−k/(L+1))*γ[k], k=1..L. Set se=sqrt(long-run variance/T). Any negative computed long-run variance raises ValueError, even if finite. The theoretically nonnegative Bartlett variance is never replaced with zero and no untested roundoff tolerance is applied.

The HAC p-value is the two-sided standard-normal tail at |μ|/se. At se=0, p=1 if μ=0 and p=0 otherwise. The interval is μ±1.96*se. Nominal80% MDE is (normal_quantile(.975)+normal_quantile(.8))*se; it is not adjusted for a wave family and is not a power certification.

## Exact masked circular bootstrap

For each block length B, use an independent NumPy Generator seeded with `seed+B`. A caller retaining the prior phase/horizon scheme should pass its phase/horizon seed **without** another block addition; a calibration caller analogously passes the trial seed.

Let K=T//B and R=T%B. Generate the entire logical row-major array of full-block starts, shape(draws,K), uniformly from0..T−1. Only after every full-block start has been generated, draw the tail starts of length draws if R>0. Each draw uses K complete circular blocks of lengthB and one circular remainder of lengthR, totaling exactlyT calendar positions.

Sample the mask and masked difference numerator with the same positions. The replicate estimate is sampled numerator sum divided by sampled mask count; its observed-support count can differ from n. Any replicate with count zero raises ValueError for the whole call. It is never discarded, replaced or redrawn.

The two-sided centered-null plus-one p-value is (1+count(|replicate_mean−μ|≥|μ|))/(draws+1). The interval is the raw replicate mean's .025/.975 percentiles using NumPy's default linear quantile interpolation.

For efficiency, the implementation uses circular cumulative counts/value sums and draw chunks of1024. It retains per-draw totals and never allocates a draws-by-calendar tensor. Full-block chunks finish before any tail chunk, preserving the literal RNG sequence. Memory is proportional to the calendar, draw totals and one chunk of block starts.

## Domain and failure rules

Require at least two full-calendar positions and two selected observations. These are primitive domain rules, not the forthcoming study support floors. Real numerical array dtypes are accepted; boolean, complex, string and object differences are rejected. The mask must have boolean dtype. Incompatible dimensions/lengths fail.

Blocks must be a nonempty unique integer sequence with1≤B≤T. HAC lag is a nonnegative integer; draws is a positive integer; seed is a nonnegative integer. Boolean and nonintegral parameter values are rejected. Integer NumPy scalars are accepted. All evaluated means, cumulative sums, influence values, covariance results and intervals must be finite; overflowing arithmetic fails instead of repairing inputs.

There is no inference from compressed auction rows, independence assumption for same-day auctions, automatic missing-data fill, endpoint selection, effect-size gate or family correction in this module. Separate wrappers must apply the prospectively frozen masks, support floors and controls. Identical full calendars and seeds give identical resampling indices across paired comparisons.

## Verification evidence

Eleven tests were written before the producer existed. `INFERENCE_CORE_RED.log` preserves the actual missing-module import failure. The first implementation passed all11 tests in0.017 seconds, retained unchanged at the start of `INFERENCE_CORE_GREEN.log`. Before freeze, root review required rejecting a negative computed HAC variance instead of clamping it. A twelfth generated regression injects finite covariance sums to exercise that failure path; its observed assertion failure is appended to the RED log before the guard was changed. The final12-test result and scoped formatting/lint checks are appended to the GREEN log.

The independent oracle explicitly constructs every sampled calendar index, obtains selected draw values and uses a literal mean; it does not reuse the producer's prefix-sum shortcut. Contracts cover partial last blocks, more than one draw chunk, all-observed reduction, gap-dependent HAC with an unchanged selected sequence, false-mask NaNs and large placeholders, whole-call zero-support failure, zero-variance limits, no input mutation and invalid/overflowing domains. Only invented arrays were used. This evidence is not empirical calibration, market-data coverage certification or independent study verification.

## Pending prospective calibration

The unregistered `treasury_dealer.yaml` specifies an external calibration before market access:1500 full sessions, stationary Gaussian AR1 with rho.8 after200 burn-in, active residues0/1/7/8/14/15 modulo21, and a fixed missing interval600..725. It fixes200 repetitions,499 bootstrap draws and calibration seed20261002. For both daily and active-date masks, the combined HAC/block95% interval envelope must cover zero in at least.90 of repetitions; individual-method coverage and zero-support failures must also be recorded. The wrapper must use the declared same full-calendar resampling design. This calibration was read as a prospective contract but was not run by this task, and this primitive's passing tests do not assert its outcome.
