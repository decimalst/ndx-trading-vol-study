# After the precision gate: one different prediction question

September 9, 2026. Design review only; no new registration, empirical array, feature, target, support count, fit or score. Wave26 is completed with no qualifying lead and 149 recorded comparisons. Its failure does not justify tuning another state, decay rate, memory window or mixing penalty.

**One bounded next hypothesis is worth specifying: a fixed Student-t dependence model improves the joint next-session QQQ/SPX intraday return density beyond both Gaussian dependence and independence, with identical conditional marginal distributions.** This is a probabilistic joint-risk question using already admitted daily data. It changes the forecast distribution being tested; it does not claim a new external signal, a novel statistical method, or improved expected returns.

## Novelty evidence

The [admitted-data alternative review](../treasury_dealer/ADMITTED_DATA_ALTERNATIVE_REVIEW.md) inventories the 54 inherited comparisons and waves1–24. The subsequent Treasury study and precision gate bring the tracked family to149. The [earlier inventory](../iterative_signal_search/PRIOR_SEARCH_INVENTORY.md) also identifies regime/jump, nonlinear, latent/residual, surface, futures and positioning work outside that tracked family. This is not a complete repository-lifetime trial census.

| Prior work | What it covers | Remaining distinction |
| --- | --- | --- |
| [Relative risk](../../relative_risk.yaml), wave9 | Relative QQQ/SPX intraday risk | A scalar risk difference is not a joint return density. |
| [Joint risk](../../joint_risk.yaml), wave10 | Conditional means and residual second-moment matrices, matrix QLIKE, dynamic versus constant correlation/matrix | Matrix QLIKE scores the second-moment forecast. It does not fit the dependence of probability-transformed marginal returns with shared heavy-tailed marginal distributions. |
| [Cross moment](../../cross_moment.yaml) and [target alignment](../../target_aligned.yaml), waves11–12 | Signed residual product prediction, including a model fitted directly for that product | Different distributions can have the same cross moment. A density comparison is not a rescore of those saved product forecasts. |
| [Sign memory](../../sign_memory.yaml) and [causal pool](../../causal_pool.yaml), waves13–14 | Strict same-direction probability and probability calibration | Their defined event discards return magnitude; the sign-memory design expressly disclaims identifying a copula parameter. |
| [Tail shape](../../tail_shape.yaml), wave7 | Single-index SPX signed downside shape and density, fixed degrees of freedom8, SKEW asymmetry | It does not model the dependence of the QQQ and SPX marginal return distributions. |
| [Precision gate](../../precision_gate.yaml), wave26 | State-dependent scalar variance combination | Neither its scalar QLIKE objective nor its latent reliability state tests a joint distribution. |

Scoped searches of source, tests and design reports for copula, joint-tail dependence, co-exceedance and bivariate-tail density found no such forecast implementation; the copula hits were exclusions in sign-memory documentation/code. That supports an untested-repository hypothesis, not a claim that all dependence ideas are absent. The [timing review](INDEX_TIMING_NOVELTY.md) rules out presenting session splits or civil-calendar variants as new information. The [early-session review](../early_session_feasibility/SUMMARY.md) remains source-blocked; daily OHLC cannot supply a new 09:35 observation. Neither alternative is substituted here.

## Exact candidate and controls

Use the previously admitted QQQ ETF and SPX price-index raw OHLC snapshots and Cboe controls from the joint-risk source envelope. The observation is the pair of next-session raw intraday log(close/open) returns on the unchanged SPX calendar. Missing either component makes that pair unknown. Zeros are valid. No high/low order, execution price, new intraday boundary or exact Nasdaq-100 index identity is inferred.

The exact six numerical source paths and saved SHA256 values below are transcribed from `reports/joint_risk/manifest.json` metadata; this review does not decode them or repeat full source admission. A future runner must authenticate these original pins and their transitive source/protocol/review closure before bounded decoding, including the old measurement gate. This is not a substitution of wave26's differently packaged IV inputs. Only raw open/high/low/close and the declared Cboe series are in scope; no new vendor vintage, volume, adjusted-close repair or Treasury quantity is added.

| Source | Existing saved SHA256 |
| --- | --- |
| `data/research_paths/spx_daily.parquet` | `3958fbb1eb36689df1596c26b9f0e02e3f1d4fa3b0032381e5287a2a03697bf0` |
| `data/raw/daily_ohlc.parquet` | `710290d8ad8569172559334b23e423b50ecf7cd10f0fd8ec24a9fc29608bca4d` |
| `data/free_sources/raw/cboe/VIX_History.csv` | `a34aabce269632f30904cf482986dd50b6d4cf51f2203dc51a0a9f460f3c90b2` |
| `data/free_sources/raw/cboe/VIX9D_History.csv` | `0d6f600ee71bf6ffb5069d0c583cbe0b1ec97df6da4e4e439d5f0f22f5616abe` |
| `data/free_sources/raw/cboe/VVIX_History.csv` | `f6bc726455fa3859c662875a005e97ba656b9536ae58fa6977ad24c6228e4f6a` |
| `data/free_sources/raw/cboe/VXN_History.csv` | `753b5a406a7a888a9e4ca1a3e37a71fbb415a3883bc4b02ff20dbf0c5bdc6c98` |

Regenerate causal shared conditional means `mu_Q,mu_S` and positive residual second moments `h_Q,h_S` using the existing joint-risk feature basis and its fixed mean/positive-moment estimators. Retain its lag and predecessor rules. Choose each monthly fit from feature completeness without future labels; train on common prior origins with outcomes mature by the preceding full reference session, minimum1000. Current-fit training residuals are permissible for this staged fit but must be disclosed as in-sample residuals, not historical issued errors. All three arms use exactly the same fitted means, scales, marginal distributions and rows.

Define each marginal distribution as a Student-t with **fixed8 degrees of freedom**, location `mu_j` and scale `sqrt(h_j)*sqrt(6/8)`, so its modeled variance is `h_j`. This borrows the already used fixed tail-heaviness convention; it is not selected from these new results. Let `u_j` be that marginal CDF evaluated at the realized return. The three joint densities are:

1. **Independence:** `f_Q(r_Q)*f_S(r_S)`.
2. **Gaussian dependence control:** `f_Q*f_S*c_G(u_Q,u_S;rho_G)`.
3. **Student-t candidate:** `f_Q*f_S*c_t8(u_Q,u_S;rho_t)`.

Here `c_G` is a bivariate Gaussian density divided by its two univariate Gaussian densities after the normal inverse-CDF transformation. `c_t8` is the analogous bivariate-t8 density divided by its t8 marginal densities after the t8 inverse-CDF transformation. These are dependence densities with uniform marginals; only their dependence structure differs. Fit one constant correlation parameter per dependence family on the same current monthly mature training pairs by that family's own negative joint log likelihood, with the same fixed domain `[-.995,.995]`. No dynamic slope, alternative degrees of freedom, window grid, additional marginal skew parameter or mixture is included. Both families receive target-aligned fitting, rather than borrowing the old matrix-QLIKE correlation optimum.

The model predicts the full pair distribution at issuance. It is not an algorithm for predicting the eventual CDF-transformed observations: those are realized scoring quantities. Shared marginal log densities cancel from paired score differences, permitting direct stable comparison of dependence log densities. This cancellation does not eliminate errors in the marginal forecasts: incorrect marginal scales or shapes can still masquerade as a dependence-model advantage. A positive finding therefore concerns the specified joint density with fixed modeled marginals, not proven pure tail dependence or improved covariance.

## Falsification criterion

Register exactly two comparisons: candidate versus Gaussian dependence and candidate versus independence, both using unscaled negative joint log density in nats per observed pair. Require **at least0.005 absolute mean score reduction in both development2016–2019 and evaluation2020–2025-10-20 against both controls**. Also require negative gaps in both fixed evaluation slices2020–2022 and2023–ceiling and every fixed full-calendar modulo-five diagnostic group. No tail-only subset or secondary Brier/portfolio result can rescue a failed primary family.

Use the current full-calendar dependence-aware inference convention: preserve missing positions inside each literal phase, paired shared circular blocks21/63/126, HAC126,399999 draws and maximum phase/method p. Fix the new seed before running. Proposed support is1000 common mature training pairs,505 scored pairs per phase,252 per evaluation slice and63 per modulo-five group. Support and numerical feasibility are uninspected. Both wave and cumulative Holm must pass; if this is the next registered wave27, the wave budget is `.05/(27*28)` and two new comparisons make151. The149 inherited entries, including failures, remain unchanged.

A completed failure of either comparison falsifies this bounded predictive claim on the reused historical test. A source, support, CDF/inverse-CDF, optimization, numerical or independent-verification failure makes the whole proposed family unevaluable with both p=1; it is not a statistical null. In particular, CDF endpoints caused by floating-point saturation cannot be clipped or converted to artificial tail observations. Stable log-tail arithmetic and an independently checked bounded one-dimensional optimum are implementation requirements to fix and test before empirical use, not grounds for trying alternative families afterward.

Before any implementation/run, prewrite generated tests for copula normalization/marginal preservation, independence and parameter boundaries, fixed marginal variance, exact shared-marginal score cancellation, finite tail arithmetic, independent optimum checks, target-blind maturity, common calendars and failure retention. Regenerate the new response and fits; do not relabel old matrix-score outputs as density predictions. This note selects no solver tolerances, does not constitute a frozen executable protocol, and starts no new experiment.

The rationale is a gap in the forecast functional, not evidence that a useful signal exists. A win would support a richer joint risk distribution for these two archival instruments; it would not establish tradable index direction, execution feasibility, synchronized auctions or exact historical source vintages. Previously evaluated history remains exploratory, including the later period accessed by legacy regression tests. Real confirmation requires subsequently untouched outcomes.

## Read-only evidence identities

- Admitted-data alternative review: `7413f007ac69b92d54607f02988a2b0b05393a7827bf4df0df4a306a8ffeed01`
- Earlier search inventory: `2ddac8944bdfaab2da2a31e001a20208e6cc4d21920836981f7b13ddd430a9a3`
- Index timing review: `f6cde75fe36f99ccd57c39862f612f28ee3968540b46f6d1cbd1d6b04cba8142`
- `joint_risk.yaml`: `358d863853d79c876476e67bcf91e4c7fc855f2f107c8df58ede074f19a809ef`
- `cross_moment.yaml`: `edeb72e2d7708c0a5227318d1ea54742050f9be0b761941ce598cdda28c26ed0`
- `target_aligned.yaml`: `ca11ec1c90f8d83867b6dce4efaaac4f0e5ed11e23143e2ab1f28d78a9f136b3`
- `sign_memory.yaml`: `a825704da953cd84302ee37430e446f8558d700859ccc7d2b44fbf5e2268f3d0`
- `causal_pool.yaml`: `1e329ee208612ed8db1fede873b92a938820d51a3fcf2ec583c813d37ac000ad`
- `tail_shape.yaml`: `0d7184c2d966f28d6ab1602ee6ff6bce4de54019ed0d601688ed7281011366ee`
- `precision_gate.yaml`: `c9ac4c6f238cc898e219b8e1b1e9c8f1bf84371e40bf363a7572ee63182f11a2`
- Completed precision-gate summary: `196028247ae9635ba822cf165b379adff59d2978bdaee449ef73c78d87bd1d73`
