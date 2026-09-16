# Wave five source and design preflight

2026-09-07. **No source-mapping or date-support blocker found for a retrospective
measurement-prediction experiment.** This review did not fit models, compute
predictive relationships, or parse any source estimate after 2025-10-20. Only
saved documentation, provenance, bounded reference dates, and source-byte hashes
were inspected. Earlier files and the frozen source corpus were not changed.

## Measurement mapping and interpretation

The pinned provider `volatility.js` uses zero-based fields 2/5/6 for trade QMLE,
five-minute trades and fifteen-minute trades. For trade QMLE, lines 365–369 read
the width from `index[0]+2`, hence **field 4**, and draw `vol-ciw, vol+ciw`.
Both the level and width are multiplied by 100 for percentage display. The
commented square-root expression is inactive. The chart's removal below 1e-6
and clipping above 3 are display operations; neither belongs in the model.

Consequently `u=log1p(c/v)` is a dimensionless recorded relative half-width, and
`q=v²` is an explicit new target in squared provider-native annualized-volatility
units. This evidence does not establish a confidence level, a standard error,
observation-error variance, exact annualization factor, or integrated daily
variance. The proposal correctly avoids a normal-quantile conversion, invented
quarticity, or a daily-variance conversion. A common constant rescaling of both
volatility and width leaves `u` and `H1-H22` unchanged; scaling the target and
forecasts together leaves paired proper-score gaps unchanged.

The interaction tests an incremental forecast specification. A width main effect
does **not** identify a measurement-error mechanism: `u` itself contains `v`, and
width may reflect volatility or liquidity states. Thus `u×(H1-H22)` can capture
ordinary nonlinear state dependence. Any favorable result must retain the narrow
recorded-measurement forecasting interpretation. Alternative estimates come
from the same provider and related underlying observations; agreement is a
measurement sensitivity check, not independent-source or latent-truth validation.

## Pinned provenance

All following current byte hashes match the prior source manifests:

| Source | SHA-256 |
|---|---|
| Quarantined SPY response, 737,911 bytes | `b7802ef21565b13831420c8fdd2b15176e57265fd5afe981be73693c8350e392` |
| Provider `volatility.js`, 21,343 bytes | `f7d04e08d9bd25eaf307d061da4fd885ee7615084b11c84640fe5c21460d9d3b` |
| QQQ daily parquet | `710290d8ad8569172559334b23e423b50ecf7cd10f0fd8ec24a9fc29608bca4d` |
| Raw VIX CSV | `a34aabce269632f30904cf482986dd50b6d4cf51f2203dc51a0a9f460f3c90b2` |
| Raw VIX9D CSV | `0d6f600ee71bf6ffb5069d0c583cbe0b1ec97df6da4e4e439d5f0f22f5616abe` |
| Raw VVIX CSV | `f6bc726455fa3859c662875a005e97ba656b9536ae58fa6977ad24c6228e4f6a` |

Risk Lab's acquisition URL is the fixed `data.php?ticker=84398` endpoint on
`dachxiu.chicagobooth.edu`; its acquisition manifest records TLS-verified
retrieval before 2026-09-07 04:00:18.955230 UTC. All three saved documentation
files also match their manifest. Both original and repaired audit-attempt code,
test and contract hashes match their corresponding preserved snapshots. This
preserves the disclosed typed-calendar repair; it does not mislabel the first
implementation as unchanged.

QQQ provenance remains the cached Yahoo/yfinance adjusted-close snapshot in
`data/history_extension/source_manifest.json`. Cboe adjacent manifests pin
bytes, retrieval time and `daily-live` source identity but omit the URL; the
official CDN download pattern is recorded in `src/fetch.py`. These records are
adequate to reproduce the chosen archival inputs, not to reconstruct historical
publication vintages. In particular, VIX9D's January 2011–October 2013 history
precedes its October 2013 launch and is back-calculated. Its use in training a
2016 model must not be described as contemporaneously observed 2011 input.
Risk Lab remains research-only and quarantined.

## Date-only feasibility and causal ordering

The CSV inspection loaded `DATE` only; the QQQ parquet inspection loaded only
bounded index labels. Risk Lab estimates were not numerically reparsed.

| Control | First archived date | Dates through 2025-10-20 | Dates during 2011-01-04–2015-12-31 |
|---|---|---:|---:|
| VIX | 1990-01-02 | 9,040 | 1,257 |
| VIX9D | 2011-01-04 | 3,721 | 1,257 |
| VVIX | 2006-03-06 | 4,878 | 1,256 |

The latter interval has one missing VVIX reference date, 2013-05-13; VIX and
VIX9D have none. With full reference-session windows, Risk Lab inputs through
`t−2`, other controls through `t−1`, and next-session labels maturing at `t+3`,
there are **1,252 date-supported potential training origins** for the 2016-01-04
fit. They span 2011-01-05–2015-12-28. This exceeds the fixed 1,000 requirement
at the date-support stage; it is not an actual numerical-completeness, fitted,
or scored sample count.

The ordering is internally causal under the declared archival lag assumption:
monthly training requires `label maturity <= fit's preceding reference session`.
The first refit must depend on current feature completeness, not future label
presence. Future observations can determine targets and maturity only. Two
sessions of lag cannot certify unknown provider publication latency or remove
revisions. Preserve the observed QQQ reference calendar as explicitly approximate,
all missing Risk Lab positions, and the omitted February 2018 stress week. No
claim extends to dates excluded by source gaps and their trailing windows.

## Freeze requirements

The parent has expanded the still-unrun proposal from nine to **15 contrasts**:
width versus mean/baseline across three measurements, and interaction versus
mean/baseline/width across three measurements. Freeze that family and all failure
slots in the new YAML; the earlier nine-contrast note remains a proposal. With
wave alpha `1/600`, Holm15's first cutoff is `1/9000`; its one-tenth resolution
bound is `1/90000`. The proposed 99,999 draws give minimum plus-one p-value
`1/100000`, satisfying that rule. The cumulative family is 99. No alternative
target may refit or recalibrate the forecasts trained to primary QMLE squared.

Freeze actual dependency versions and every reused numerical code dependency,
alongside source, protocol and test hashes. `requirements.txt` mostly provides
version ranges rather than a reproducible lock. The inspected runtime is Python
3.11.7, NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, PyArrow 25.0.1 and PyYAML 6.0.3;
the numerical backend/platform should also be recorded. Newly registered tests
and independent reconstruction must still verify feature geometry, maturity,
optimizer conditions, common samples and all 15 inference slots before a result
is interpreted. This preflight does not certify an implementation not yet frozen.

Evidence: [prospective design](../macro_overnight/NEXT_MEASUREMENT_DESIGN.md),
[fixed measurement contract](../overnight_index/RISKLAB_MEASUREMENT_CONTRACT.md),
[independent measurement audit](../overnight_index/RISKLAB_MEASUREMENT_VERIFICATION.md),
and [existing Cboe source audit](../overnight_index/SOURCE_FEASIBILITY.md).
