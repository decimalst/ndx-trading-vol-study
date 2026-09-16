# Independent verification of the HARNet numeric source audit

2026-09-07. **The requested independent checks pass. The extension preserves
all old date keys and agrees with old measurements within the registered
tolerance; it does not exactly reproduce the old RV5/RSV inputs.** No producer
module was imported. No forecasts, predictive scores, source repairs,
rescaling, or raw-data modifications were performed.

The independent implementation is retained privately in ignored
`data/source_discovery/harnet_numeric_verification/verify_independent.py`.
It reads the old ZIP member `oxfordmanrealizedvolatilityindices.csv` and the
quarantined `MAN_data.csv.gz` directly, using standard-library CSV parsing and
Decimal arithmetic with precision 80. A separate read uses pandas' default CSV
parser and casts the selected fields to float, matching the frozen studies'
parser convention without calling their code. Runtime versions are pandas
3.0.5 and NumPy 2.4.6. Small synthetic assertions for local dates, exact versus
tolerance equality, integer counts, and invalid/missing tokens ran before any
numeric source comparison.

Keys are exact symbol plus the validated first ten local date characters;
there is no UTC conversion, forward fill, or inferred exchange calendar.
The seven selected series end in January 2020, before the protected phase.
Every old key is present in the new archive, with no added old-period key and
no duplicate selected key.

| Symbol | Old / matched rows | New rows | Additional rows |
| --- | ---: | ---: | ---: |
| `.SPX` | 4,641 | 5,038 | 397 |
| `.N225` | 4,501 | 4,885 | 384 |
| `.HSI` | 4,532 | 4,919 | 387 |
| `.KS11` | 4,550 | 4,939 | 389 |
| `.FTSE` | 4,660 | 5,064 | 404 |
| `.GDAXI` | 4,692 | 5,092 | 400 |
| `.FCHI` | 4,713 | 5,120 | 407 |
| Total | 32,289 | 35,057 | 2,768 |

Every selected old series ends 2018-06-27. Every extension begins 2018-06-28
and ends 2020-01-31. These are observed archive boundaries, not a claim that
every exchange session exists in the source.

The registered non-count tolerance is
`abs(new-old) <= 1e-12 + 1e-10*abs(old)`; observation counts require exact
Decimal equality. All 32,289 overlap pairs are finite in each of the nine
shared fields. No field has a beyond-tolerance difference. All `nobs` values
match exactly. The finer equality levels for the requested volatility inputs
are:

| Comparison | RV5 | RSV |
| --- | ---: | ---: |
| Literal token matches | 2 | 0 |
| Exact Decimal numeric matches | 1,578 | 1,791 |
| Exact direct Float64 matches | 1,578 | 1,791 |
| Exact frozen pandas-parser matches | 2,818 | 2,511 |
| Finite pairs within registered tolerance | 32,289 | 32,289 |
| Maximum absolute Decimal difference | 5.014e-16 | 5.0074e-16 |
| Maximum absolute frozen-parser difference | 6.002143226879753e-16 | 5.999974822534782e-16 |
| Maximum relative frozen-parser difference | 5.610350574795861e-12 | 5.359192043937937e-12 |

These are small, nonzero differences in the recorded measurements. Their
size and tolerance agreement do not establish their cause. Different numeric
serialization could explain them, but this audit does not prove roundoff,
recover an upstream export procedure, or show that one source is more accurate.
The correct classification is `TOLERANCE_ONLY_NOT_EXACT`. Substituting the new
archive's overlap for frozen old inputs would change those inputs.

All 2,768 extension rows have finite values across all nine shared fields, with
no malformed or missing numeric tokens. RV5 and RSV are strictly positive;
there are no violations of `rsv <= rv5*(1+1e-12)`. Prices and observation counts
are positive, and observation counts are integral. The checked variance and
semivariance fields have no negative values. There are two zero `medrv`
observations, so a statement that every variance field is strictly positive
would be incorrect. The RSV field retains its neutral archived-semivariance
interpretation; numerical agreement does not establish its direction.

The independently calculated file identities match the registered audit:

| Object | SHA-256 |
| --- | --- |
| Old ZIP | `e0dd80edc0c2cedac5ed3f72250ee4460e963b4efd458d68525a61bcc5c27ea2` |
| Old extracted CSV | `865c771a2c9d0887e6db3a20e6239f6190867ddc3605f2dcba398a06dba771bb` |
| New gzip | `30ef85340891ea9b8fdd69b8f15e459037c2586505096a0fa27c9ea546685143` |
| New extracted CSV | `c6be6c68280e9100ce734721a4c05bd557ec9009053ad7526fda20b6bd62851e` |
| Existing discovery manifest | `296f64574240e471b8430d20643a4a317c2e7e91a56313dd6de520f5f6c7d2b9` |

Source files were hashed again after reconstruction and remained unchanged.
The registered producer, test-file and pre-comparison contract-prefix hashes
also match. Comparison of the independently reconstructed keys, equality
counts, per-symbol extension quality and identities with the producer's saved
audit passed 1,041 bookkeeping assertions. This count is not a statistical
sample size or evidence of forecasting value.

The ignored outputs are
[`independent_verification.json`](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/harnet_numeric_verification/independent_verification.json)
and
[`agreement_checks.json`](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/harnet_numeric_verification/agreement_checks.json).
The latter pins the producer audit SHA-256
`dae463b925695908d91f54f5706706152a9f5cba915984c7e4250af54fc96119`
and independent reconstruction SHA-256
`2bb237654349d1c9dc41b260ec670bdb77738360a84efbbebd84faa1a0a25c4f`.

This verifies the requested measurement and integrity checks. It does not
establish historical publication vintages, the unavailable subsampled-field
contract, redistribution or commercial-use rights, a proven roundoff cause,
or predictive usefulness. No existing source, calendar, frozen result, or
earlier report was modified by this verification.
