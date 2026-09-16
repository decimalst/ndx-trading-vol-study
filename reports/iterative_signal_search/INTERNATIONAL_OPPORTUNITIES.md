# International volatility: source feasibility and fixed next experiment

Prepared before any international model fit or forecast score. This audit
examined source dates, missing observations and numeric validity only. The
existing Oxford-Man archive supplies information from foreign markets that is
absent from the SPX feature set; this tests a new information source rather
than another transformation of SPX prices.

## Fixed markets and observed coverage

Use Japan, Hong Kong and Korea as the Asia group, and the UK, Germany and
France as the Europe group. These baskets were chosen by geography before
relating their measurements to any forecasting outcome. All six have positive
finite `rv5`, unique stated dates, observations in every year 2000–2017, and at
least 227 observations in their shortest included year. Japan begins February
2000, so its initial year is partial.

| Archive symbol | Market | Rows through 2017 | First stated date | Last stated date | Missing exact prior-US-date matches in 2010–2017 |
|---|---|---:|---|---|---:|
| `.N225` | Nikkei 225, Japan | 4,382 | 2000-02-02 | 2017-12-29 | 117 |
| `.HSI` | Hang Seng, Hong Kong | 4,413 | 2000-01-03 | 2017-12-29 | 92 |
| `.KS11` | KOSPI, Korea | 4,432 | 2000-01-04 | 2017-12-28 | 98 |
| `.FTSE` | FTSE 100, UK | 4,537 | 2000-01-04 | 2017-12-29 | 37 |
| `.GDAXI` | DAX, Germany | 4,569 | 2000-01-03 | 2017-12-29 | 33 |
| `.FCHI` | CAC 40, France | 4,590 | 2000-01-03 | 2017-12-29 | 17 |

The final column counts source-date absence against 2,013 actual SPX sessions;
it does not distinguish local holidays from archive omissions, and it is not a
forecast count. The symbol meanings are documented in the
[official package's Oxford-Man dataset description](https://search.r-project.org/CRAN/refmans/bvhar/html/oxfordman.html).
Index-provider descriptions confirm the market identities:
[Nikkei](https://indexes.nikkei.co.jp/en/nkave/index/),
[Hang Seng](https://www.hsi.com.hk/en-hk/indexes/),
[KRX](https://global.krx.co.kr/contents/GLB/02/0201/0201010100/GLB0201010100.jsp),
[FTSE Russell](https://www.lseg.com/en/ftse-russell/indices/ftse100),
[STOXX](https://stoxx.com/index/daxk/?factsheet=true), and
[Euronext](https://www.euronext.com/en/news/cac-40-index-0).
These current descriptions establish identities, not historical constituent
lists, historical publication times or a reconstructed trading universe.

The same archive also has full 2000–2017 annual coverage for `.AEX`, `.AORD`,
`.BFX`, `.BSESN`, `.BVSP`, `.IBEX`, `.KSE`, `.MXX`, `.NSEI`, `.SSEC`, `.SSMI`
and `.STOXX50E`, with positive finite RV5. Karachi's shortest year has 198 rows.
They remain outside this fixed family; there will be no ranking of countries
or replacement of poorly performing members.

Other international series start later: `.OSEAX` on 2001-09-03, `.GSPTSE` on
2002-05-02, `.SMSI` on 2005-07-04, `.OMXC20`/`.OMXHPI`/`.OMXSPI` on
2005-10-03, `.FTMIB` on 2009-06-01, and `.BVLG` on 2012-10-15. `.STI` starts
in 2000 but has no observations in 2009–2014 and one nonpositive/nonfinite
RV5 observation in the inspected range. These are unsuitable substitutions
for a common long-history design without separately declared data rules.

## Calendar and availability contract

Use the same archive bytes and source hash as wave 1:
`data/raw/oxford_man_realized.zip`, SHA256
`e0dd80edc0c2cedac5ed3f72250ee4460e963b4efd458d68525a61bcc5c27ea2`.
Preserve the first ten characters of each stated trading date; never convert
the source's summer local-midnight timestamps to UTC dates. Read only source
dates within the new protocol fence. The original provider's publication
timezone convention remains unresolved, so same-day availability is not
assumed.

For each foreign index, compute log arithmetic-mean RV5 over its own previous
1, 5 and 22 observed local sessions, ending on source session `s`. Then, for
US origin `t`, select the latest available `s` whose date is no later than the
previous actual US session date. This enforces the specified one-US-session
delay even when the US was closed while a foreign market traded.

Retain `s` and its age. Additional age is the number of US sessions strictly
after `s` and through that previous-US-session cutoff. Accept only age at most
three; otherwise mark the entire market state unavailable. This is an explicit
use of the last known historical state, with bounded staleness, not fabricated
returns on a closed market. Do not interpolate, backfill from a future foreign
observation, use a same-US-date observation, or renormalize a region after a
member becomes unavailable.

This calendar choice matters numerically before any modeling: exact previous-US
date matching leaves 1,740 of 2,013 origins with all six markets present.
Incorrectly applying a 22-US-session rolling window after that intersection
leaves only 72 complete origins. Rolling on each local calendar first avoids
that artificial loss. The declared as-of age rule leaves 2,010 origins with
all six source dates usable. The three exceptions are US origins 2017-10-06,
2017-10-09 and 2017-10-10, when the latest Korean source date is 2017-09-29
and additional ages are four, five and six US sessions. Final forecast counts
will additionally reflect SPX features and target completion.

The stale-observation limit was chosen from source-availability considerations
before forecasts were fitted. It must not be extended after seeing results.

## Fixed candidate family for implementation

Keep wave 1's eleven SPX baseline inputs, 1/5/21-session future RV5 targets,
monthly expanding fits, 750-row minimum, log-OLS with exact Duan smearing,
publication delay, development/evaluation dates and target fences. The new
sample requires the same complete baseline and all eighteen foreign features
for every model and horizon. Refit the baseline on these identical rows.

| Model | Additional information |
|---|---|
| Baseline | None; current SPX daily range/overnight and leverage controls, lagged VIX, lagged SPX HF HAR |
| Regional | Six features: equal arithmetic means of the three Asia log-RV features at each local horizon, and the corresponding three Europe means |
| Latent | Three principal components of all eighteen foreign log-RV features, with standardization and PCA fitted only on eligible training observations |

The regions average log variances; equivalently, they summarize geometric
means of the constituent local-horizon arithmetic mean variances. They are
fixed information aggregates, not investable portfolios. Regional model
scaling is fitted only on training data. The latent model uses exactly three
components, fixed before scoring, with a deterministic component-sign rule.
Its geometry must be refitted only when the forecast model is refitted.

The initial primary family compares regional and latent against the matched
baseline at three horizons. Any claim that PCA itself helps over regional
aggregation would need its own declared comparison; improvement over SPX alone
would establish the usefulness of foreign inputs within that specification.
All new comparisons belong in the continuing trial ledger and cannot reset
the earlier multiplicity or reused-history warnings.

[Hamao, Masulis and Ng](https://academic.oup.com/rfs/article-abstract/3/2/281/1595767)
study directional volatility transmission across Tokyo, London and New York;
their early sample does not establish Japan-to-US predictive gains.
[Diebold and Yilmaz](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0297.2008.02208.x)
measure time-varying international return and volatility interdependence.
These papers motivate a test of different-market information, but neither
shows that this delayed six-market input beats the present SPX/VIX control.
The proposed models do not reproduce their ARCH or variance-decomposition
estimators, and forecast improvement would not itself identify causal
contagion.

Historical index measurements were acquired retrospectively. Conservative
date lags establish which raw market session a feature represents, but cannot
prove the original archive's real-time publication or absence of later
revisions. The resulting study remains exploratory. Its intentionally delayed
inputs also do not test whether precisely timed same-day foreign market
information could be useful at a different execution time.
