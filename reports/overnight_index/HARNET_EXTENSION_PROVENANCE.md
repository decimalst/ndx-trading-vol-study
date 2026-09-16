# HARNet Oxford supplement: documentary provenance review

Reviewed 2026-09-07 UTC. This review read source documentation, author code and
the existing discovery manifest. It did not inspect realized-measure values,
compare numerical source records, execute author code, or fit or score models.

**The supplement has attributable research provenance, but its exact measurement
contract is only partly documented.** Two reproducibility discrepancies matter:
the pinned loader requires a field absent from the distributed CSV, and the
paper describes a later end date. The direction of the retained `rsv` field and
the supplement's exact numerical scale remain unverified by documents alone.

The object under review is `data/MAN_data.csv` at author-repository revision
`3b2832f163ec52cab2435163e82f2e816ae348c9`, identified by the
[local discovery manifest](/Users/byrons/code/trading-vol/ndx-vol-experiment/data/source_discovery/quarantine/harnet_2020/source_manifest.json).
That record pins CSV SHA-256
`c6be6c68280e9100ce734721a4c05bd557ec9009053ad7526fda20b6bd62851e`
and records all seven selected indexes through 2020-01-31. Those are inherited
discovery findings, not newly repeated data checks. The
[pinned author README](https://raw.githubusercontent.com/mdsunivie/HARNet/3b2832f163ec52cab2435163e82f2e816ae348c9/README.md)
explicitly attributes the CSV to Oxford-Man and requests citation when using the
package in research. It supplies neither an upstream download date nor a
reproducible export/cleaning history for the CSV.

| Field or convention | Documentary finding | Consequence for this supplement |
|---|---|---|
| `rv5` | The conventional Oxford label denotes five-minute realized **variance**. HARNet defines daily RV as squared intraday log returns summed within a day, but its experiment and loader use the subsampled variant. | Intended dimension is daily log-return squared; do not relabel `rv5` as subsampled variance or annualized volatility. Exact stored scale still needs the separate measurement audit. |
| `rsv` | A five-minute realized-semivariance label. HARNet explicitly defines positive and negative components but directly loads `rsv_ss`, not `rsv`. | Direction of this retained non-subsampled field remains unresolved. Use a neutral archived-semivariance label. |
| `open_to_close` | Published Oxford field descriptions identify the within-day log return between first/opening and last/closing prices. HARNet does not read this field. | Expected dimension is a signed log return, not squared return or annualized volatility. Documentary evidence alone does not certify a multiplier or price identity in this particular CSV. |
| `rsv_ss` | The author loader subtracts this field from `rv5_ss` to obtain the positive component. | This establishes the authors' intended **negative** interpretation of `rsv_ss`; it does not prove a direction mapping for `rsv`. |

The formula and subsampled-experiment statements are in
[HARNet §§2, 5–6, equations 3 and 26–28](https://arxiv.org/html/2205.07719v1).
The literal column-name distinctions are also documented in an original study's
[Oxford field table and appendix](https://nul.repository.guildhe.ac.uk/id/eprint/2290/1/Brandi%20Multiscaling%20and%20rough%20volatility%20An%20empirical%20investigation.pdf).
That study corroborates names and the open-to-close log-return convention; it
does not authenticate HARNet's export or resolve the retained semivariance sign.
No percentage, annualization or variance-to-volatility conversion should be
inferred solely from a familiar field name.

The [pinned loader, `get_MAN_data`](https://raw.githubusercontent.com/mdsunivie/HARNet/3b2832f163ec52cab2435163e82f2e816ae348c9/src/harnet/util.py)
reads the CSV, selects one symbol and requests `rv5_ss`, optionally with
`rsv_ss`. It then derives positive semivariance and signed jumps. The discovery
manifest lists `rv5` but no `rv5_ss`: the unmodified loader therefore cannot
consume this exact file as specified. Renaming `rv5` to satisfy it would silently
change the estimator contract. The loader neither uses `open_to_close` nor
shows winsorization, row-quality filtering, interpolation, or an operation that
rewrites the source CSV. It converts timestamps to UTC; that can move local
session dates across a date boundary. Our local-date convention should remain
independent of that conversion.

The [pinned runner](https://raw.githubusercontent.com/mdsunivie/HARNet/3b2832f163ec52cab2435163e82f2e816ae348c9/src/harnet/main.py)
transforms the loaded series in memory and selects training/testing years. In
logarithmic semivariance experiments it first replaces signed jumps with a
positive relative-jump expression. Its scaler call precedes the year split,
but the [scaler implementation](https://raw.githubusercontent.com/mdsunivie/HARNet/3b2832f163ec52cab2435163e82f2e816ae348c9/src/harnet/model.py)
uses fixed configured endpoints: `fit_transform` performs no estimation of data
minima or maxima. The default endpoints are 0 and 0.001. Other variants apply a
logarithm or identity transformation. The model also clips predictions. These
are analysis-time operations; they are not evidence of edits to raw CSV
measurements, and the scaler call alone does not demonstrate future-fitted
normalization. No CSV-generation script was identified in the reviewed loader
and runner; upstream or pre-export changes remain undocumented.

HARNet §5 states a sample ending **2020-12-31**, whereas the pinned supplement's
manifest ends **2020-01-31** for the seven relevant symbols. The code does not
explain that discrepancy. A paper's stated coverage cannot extend an archived
file's actual dates, nor establish when individual measurements were published.
These findings preclude calling this file an exact reproduction of the paper's
subsampled dataset. They do not establish that its retained `rv5` values are
incorrect. [HARNet §5](https://arxiv.org/html/2205.07719v1#S5)

There is historical primary support for **academic research use with
attribution**: Shephard and Sheppard's discussion of OMI library **version 0.1**
states that commercial use requires written Oxford-Man permission. It also
explains that underlying Reuters ticks and their cleaned version cannot be
redistributed, and that market closures or low-quality observations produce
missingness. Those statements document an earlier library's terms and cleaning
context; they do not certify identical terms, filters, or session coverage in
this later author supplement.
[Original upstream discussion, §3 and footnote 9](https://www.nuffield.ox.ac.uk/Economics/papers/2009/w3/heavy080709.pdf)

The [author repository's MIT notice](https://github.com/mdsunivie/HARNet/blob/main/LICENSE.md)
names Reisenhofer and Bayer and grants rights in their software and associated
documentation. This review could read that current notice; the pinned license
URL did not resolve through the browser. Neither the notice nor the public
presence of a CSV demonstrates that the authors can relicense the underlying
provider's dataset. Historical research permission and the README's research
intent support a private, cited research assessment; **current supplement-specific
redistribution and commercial-use rights are not established**. The original
Oxford estimator and terms pages were unavailable during this review. No
contact was sent and no permission was obtained.

The documentary disposition is therefore: retain the quarantined original and
its fingerprints; continue the separately authorized numerical measurement
audit using the literal retained fields and local dates; do not introduce
subsampled-field aliases or a downside interpretation of `rsv`; and publish
provenance, methods and permissible summaries rather than the source bytes.
Any eventual admission should state the exact date extent, estimator variant,
scale verification, unknown historical publication vintages, and unresolved
rights scope. A successful numerical overlap check would establish measurement
consistency over that overlap, not paper replication, historical vintages, or
redistribution permission.
