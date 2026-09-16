# HARNet Oxford extension: numeric source audit

## Rules fixed before numeric comparison

This is a measurement audit, not predictive validation. No model, forecast score,
directional signal, country selection, rescaling, repair, or imputation is part
of the audit. Existing sources, source manifests, and frozen studies are preserved.

The seven fixed symbols are `.SPX,.N225,.HSI,.KS11,.FTSE,.GDAXI,.FCHI`. All nine
shared numeric fields are compared: `open_to_close,rsv,medrv,rsv_ss,nobs,rv5,
close_price,rv10,open_price`. Join keys are exact `Symbol` and the first ten
characters of the source's local date label; no UTC conversion is performed.
Duplicates, malformed dates, absent registered markets or missing required fields
stop the audit. Missing/invalid numeric observations remain in the audit.

For every field, report literal equality, exact decimal numeric equality,
Float64 equality, and equality under the frozen studies' default pandas CSV
parser separately. Numeric equivalence for finite non-count values uses
`abs(new-old) <= 1e-12 + 1e-10*abs(old)`. `nobs` requires exact decimal equality.
Matched missing values are recorded explicitly, never counted as finite equality.
Any missingness change, nonfinite/malformed value, or beyond-tolerance difference
prevents equivalence for that field. Any lost or added old-period key prevents
whole-panel reproduction. Exact reproduction of `rv5/rsv` additionally requires
all old values to be finite and identical under the frozen parser. Tolerance-only
agreement must not be called exact reproduction.

Discrepancy distributions use fixed min/1%/5%/50%/95%/99%/max summaries for absolute
and signed differences, absolute relative differences, and new/old ratios where
the old value is nonzero. A possible constant unit change is flagged only with
at least 20 positive pairs, a median ratio differing from one by over 0.1%, and
maximum relative deviation from that median no greater than 1e-6. This diagnostic
does not authorize any conversion.

Continuity summaries compare each market's final 22 archived old observations
with its first 22 extension observations, preserving missing rows. They report
first/last dates, finite/missing counts, medians, and first-new/last-old and
new-window/old-window median ratios. Ratios outside `[0.01,100]` for the boundary
observation or `[0.1,10]` for the median trigger descriptive source-review flags;
they do not prove a unit change or distinguish it from genuine market variation.
No exchange calendar is inferred from the archived observation calendar.

Quality counts cover missing/nonfinite/malformed values, nonpositive `rv5`,
negative variance/semivariance fields, nonpositive prices, nonpositive or
nonintegral `nobs`, and `rsv > rv5*(1+1e-12)` (the existing HF input constraint).
`rsv_ss` is not compared with `rv5`, since the corresponding `rv5_ss` estimator
is absent from HARNet. No label direction is changed.

Source hashes, code/tests hashes and this pre-comparison contract are retained
in an ignored JSON audit record. The numeric source audit does not establish
point-in-time release vintages, redistribution permission, or predictive usefulness.

## Results

**Measurement agreement passes the fixed tolerance; exact reproduction fails.**
The recovered supplement preserves every old observation date for the seven
fixed markets and has no material numeric revision under the declared tolerance.
It does **not** reproduce the old `rv5/rsv` inputs exactly, either as decimal
numbers or under the frozen pandas parser. Small representation differences
remain and must not be concealed by replacing the frozen source.

The contract was recorded at `2026-09-07T03:59:57.995702+00:00`; comparison
completed at `2026-09-07T04:00:03.598716+00:00`. Before comparison, all **19
synthetic tests passed**, and scoped lint passed. A separate reconstruction using
raw CSV/Decimal comparisons and default pandas parsing, without importing the
audit producer, agrees with the key counts, decimal-equality counts, tolerance
classification, and frozen-parser equality counts. That reconstruction is an
additional implementation check, not an independent-agent review.

### Coverage and exact input reproduction

There are **32,289 matched old observations**, with zero missing old keys and
zero additional keys inside the old period. The new source has **35,057 selected
observations**, including **2,768 additional market dates**. Every market's old
last date is 2018-06-27, first added date is 2018-06-28, and new last date is
2020-01-31. These are archived observation dates, not an independently certified
exchange-session calendar.

| Symbol | Matched old rows | Additional rows | Exactly equal RV5 under frozen parser | Exactly equal RSV under frozen parser |
|---|---:|---:|---:|---:|
| `.SPX` | 4,641 | 397 | 418 | 365 |
| `.N225` | 4,501 | 384 | 414 | 341 |
| `.HSI` | 4,532 | 387 | 399 | 326 |
| `.KS11` | 4,550 | 389 | 369 | 386 |
| `.FTSE` | 4,660 | 404 | 367 | 328 |
| `.GDAXI` | 4,692 | 400 | 463 | 400 |
| `.FCHI` | 4,713 | 407 | 388 | 365 |
| **Total** | **32,289** | **2,768** | **2,818** | **2,511** |

### All shared fields

All **290,601 matched field pairs** are finite and within the predeclared
tolerance. There are no missingness changes, malformed numeric values, or
infinities in either selected source. `nobs` matches exactly as a numeric count
on every old date. Other fields have small numeric differences.

The discrepancy columns below use Float64 conversion of the raw decimal tokens;
they are distinct from the separately reported frozen-parser comparison.
The JSON retains literal/decimal/Float64/parser equality separately, all fixed
discrepancy quantiles, and each field's per-market results.

| Field | Exact decimal pairs | Exact frozen-parser pairs | 99th percentile absolute difference | Maximum absolute difference | Maximum absolute relative difference |
|---|---:|---:|---:|---:|---:|
| `open_to_close` | 79 | 3,284 | 5.55e-16 | 5.97e-16 | 4.67e-12 |
| `rsv` | 1,791 | 2,511 | 4.67e-16 | 5.01e-16 | 4.95e-12 |
| `medrv` | 1,987 | 2,595 | 4.51e-16 | 5.07e-16 | 4.90e-12 |
| `rsv_ss` | 1,815 | 2,549 | 4.67e-16 | 5.01e-16 | 4.95e-12 |
| `nobs` | 32,289 | 32,289 | 0 | 0 | 0 |
| `rv5` | 1,578 | 2,818 | 4.87e-16 | 5.01e-16 | 4.98e-12 |
| `close_price` | 32,275 | 32,276 | 0 | 9.09e-13 | 2.21e-16 |
| `rv10` | 1,693 | 3,021 | 4.84e-16 | 5.07e-16 | 4.93e-12 |
| `open_price` | 32,283 | 32,285 | 0 | 4.55e-13 | 2.21e-16 |

Under the actual frozen pandas parser, maximum absolute/relative discrepancies
are **6.00214e-16 / 5.61035e-12 for RV5** and **5.99997e-16 / 5.35919e-12 for
RSV**. Thus the exact-input verdict remains `TOLERANCE_ONLY_NOT_EXACT`.
Ratios of new/old RV5 and RSV are extremely close to one. No aggregate or
per-market field triggers the fixed constant-unit-change diagnostic. These
differences are consistent with numeric representation/serialization rounding;
their precise upstream cause is not established by this audit. The raw decimal
comparison and parser comparison both reject a claim of exact equivalence.

### Quality and extension boundary

Across all seven markets, old and added RV5 values are positive; RSV values are
nonnegative and never exceed `rv5*(1+1e-12)`. Prices are positive and `nobs` is a
positive integer. No shared variance or semivariance field is negative. All
24,912 added field values are finite. `medrv` has six zeros in the overlap
(FTSE three, GDAXI two, FCHI one) and two additional zeros in FTSE; zeros are
retained, and this nonnegative field is not silently log-transformed or repaired.

All 22-observation windows on both sides of the boundary are complete. The table
reports a new-side/old-side median ratio; the RV5 single-observation ratio is
first added RV5 divided by last old RV5.

| Symbol | RV5 first/last | RV5 22-row median ratio | RSV 22-row median ratio | nobs 22-row median ratio |
|---|---:|---:|---:|---:|
| `.SPX` | 0.780 | 0.875 | 0.801 | 1.000 |
| `.N225` | 2.034 | 1.369 | 1.255 | 1.000 |
| `.HSI` | 1.045 | 1.653 | 1.085 | 1.000 |
| `.KS11` | 1.655 | 1.286 | 1.039 | 1.00046 |
| `.FTSE` | 0.999 | 0.631 | 0.647 | 0.976 |
| `.GDAXI` | 0.931 | 0.686 | 0.592 | 0.971 |
| `.FCHI` | 0.580 | 0.775 | 0.661 | 0.99976 |

No variance, semivariance, price, or observation-count field triggers either
fixed boundary-ratio flag. The signed `open_to_close` field triggers the
single-observation flag for all seven markets and the median flag for N225,
KS11, GDAXI and FCHI. These flags are retained in the JSON. Ratios of signed
returns can change sign or become large around zero; the flags do not establish
a source break. No directional label or raw observation was changed. Boundary
summaries are descriptive source screens, not statistical tests of stationarity
or evidence of forecasting performance.

### Decision and preserved provenance

The supplement is a **numerically consistent dated extension under the fixed
tolerance**, with 397 additional SPX dates and the fixed foreign-market coverage
shown above. Exact reproduction of older experiments must continue to use the
original archive. A future registered input could preserve that old prefix and
append only the newly dated observations, but this audit does not implement a
splice, change a model input, or authorize a new empirical wave.

The [HARNet author supplement](https://github.com/mdsunivie/HARNet) remains
quarantined. Measurement agreement does not settle underlying provider
redistribution rights, original historical publication times, the complete
upstream processing chain, or the independent sign interpretation of `rsv`.
No sign interpretation was inferred from these values. The absent `rk_parzen`
field still prevents extending the old kernel arm unchanged. The source ends
before the main 2020 pandemic period and supplies no 2021–2022 measurements.

Local evidence:

- `src/audit_harnet_extension.py` and `tests/test_audit_harnet_extension.py`.
- `data/source_discovery/harnet_numeric_audit/contract.json`: pre-comparison
  rules, source identities, code/tests hashes and runtime versions.
- `data/source_discovery/harnet_numeric_audit/audit.json`: all aggregate and
  per-market comparisons, quality counts, and boundary summaries.
- `data/source_discovery/harnet_numeric_audit/separate_reconstruction.json`:
  additional raw-Decimal/default-parser reconstruction.

Original ZIP, new gzip, and discovery `source_manifest.json` hashes were verified
before and after the audit. They are unchanged. JSON evidence is ignored by Git;
the audit refuses to overwrite its contract/results. No predictive result,
forecast fit, performance comparison, imputation, or raw-data correction was made.
