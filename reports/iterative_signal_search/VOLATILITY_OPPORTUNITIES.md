# Volatility opportunities after the model and memory round

Design and source audit, 2026-09-06. No new model was fitted or scored for this
note. Existing reports were inspected; new data checks were limited to schema,
coverage, timestamps, missingness and measurement identities. The protected
post-2025-11-03 observations were not loaded.

The most useful discovery is in a source file already on disk: the original
Oxford-Man ZIP contains 19 fields for 31 indices, while the processed SPX file
retains only `rv5` and `bv`. This makes a substantially different, inexpensive
experiment possible: add intraday semivariance and measurement-disagreement
features to a strong SPX variance forecast, using a genuine five-minute RV
target. This is more informative than another large model on the QQQ daily
OHLC proxy.

## Verified source inventory

- Archive: `data/raw/oxford_man_realized.zip`.
- Archive SHA256:
  `e0dd80edc0c2cedac5ed3f72250ee4460e963b4efd458d68525a61bcc5c27ea2`.
- Contained file: `oxfordmanrealizedvolatilityindices.csv`.
- CSV SHA256:
  `865c771a2c9d0887e6db3a20e6239f6190867ddc3605f2dcba398a06dba771bb`.
- The existing `data/raw/oxford_man_spx_source.json` pins the
  [mirror source](https://raw.githubusercontent.com/onnokleen/mfGARCH/308b795fa220a58dea6784fe8e2566bcf8dea334/data-raw/OxfordManRealizedVolatilityIndices.zip)
  at commit `308b795fa220a58dea6784fe8e2566bcf8dea334`, fetched 2026-08-12.
  Its recorded archive hash matches the current bytes.

The exact fields after the date label are:

```
Symbol, open_time, close_price, rv5_ss, bv, rk_parzen, rk_twoscale,
bv_ss, open_price, nobs, rk_th2, medrv, rv10_ss, rsv_ss,
open_to_close, rv10, rsv, rv5, close_time
```

| U.S. symbol | Rows | First stated session | Last stated session |
|---|---:|---|---|
| `.SPX` | 4,641 | 2000-01-03 | 2018-06-27 |
| `.DJI` | 4,635 | 2000-01-03 | 2018-06-27 |
| `.RUT` | 4,636 | 2000-01-03 | 2018-06-27 |
| `.IXIC` | 4,637 | 2000-01-03 | 2018-06-27 |

Do not label `.IXIC` as Nasdaq-100 without independently establishing the
instrument identity; external papers and wrappers disagree. Use `.SPX` for the
first experiment, matching the existing repository source contract.

All 31 symbols are `.AEX`, `.AORD`, `.BFX`, `.BSESN`, `.BVLG`, `.BVSP`, `.DJI`,
`.FCHI`, `.FTMIB`, `.FTSE`, `.GDAXI`, `.GSPTSE`, `.HSI`, `.IBEX`, `.IXIC`,
`.KS11`, `.KSE`, `.MXX`, `.N225`, `.NSEI`, `.OMXC20`, `.OMXHPI`, `.OMXSPI`,
`.OSEAX`, `.RUT`, `.SMSI`, `.SPX`, `.SSEC`, `.SSMI`, `.STI`, `.STOXX50E`.
Their observations end at the same 2018-06-27 archive frontier. They should not
be searched and ranked individually after inspecting scores.

For `.SPX`, all source fields are complete and dates are unique. `rv5`,
`rv5_ss`, `rsv`, `rsv_ss`, `bv`, `bv_ss`, `rk_parzen`, `rk_twoscale`, `rk_th2`,
`medrv` and `nobs` are finite and strictly positive. There are no violations of
`rsv <= rv5` or `rsv_ss <= rv5_ss` at relative tolerance 1e-12. These checks
establish usable numeric columns, not the validity of every measurement.

The SPX source supplies 2,505 rows through 2009, 1,006 during 2010–2013,
1,007 during 2014–2017, and 123 during 2018 through June 27. These are source
counts before lags, forward-target completion, missing external inputs or other
eligibility rules; they are not forecast counts.

## Timing and meaning that must be fixed before a run

The archive labels trading dates at local midnight, with `+00:00` and `+01:00`
offsets. Preserve the stated first ten date characters. Converting to UTC first
moves summer observations into the preceding calendar day. The existing
`src/jump_target.py::_parse_dates` already documents and handles this issue.

The raw SPX `open_time` range is 083000–133712 and `close_time` is
094639–170000; 1,634 rows have `close_time > 160000`. Their timezone and
publication convention were not established by this audit. At a 16:00 ET
forecast origin, delay every Oxford realized measure and archive return by one
complete SPX session. Delay Cboe daily closes by one complete session as well.
Do not infer as-of publication merely from a field named `close_time`.
`nobs` ranges from 66 to 23,403 and is an observation count, not a count of
five-minute bars; the QQQ bar-completeness threshold cannot be applied to it.

The official [bvhar package data documentation](https://search.r-project.org/CRAN/refmans/bvhar/html/oxfordman.html)
identifies `rsv` as five-minute realized semivariance, `rsv_ss` as its
subsampled counterpart, `rv5_ss` as subsampled five-minute RV, and
`rk_parzen` as non-flat Parzen realized kernel variance. That is primary
documentation for the package's imported data, not the original Oxford
provider's construction code. The original provider page and its archived
snapshot could not be retrieved in this audit. In particular, the directional
sign of `rsv` and the exact subsampling grid remain unverified.

An unconstrained feature named **archive semivariance share** can be tested
without claiming that its coefficient identifies downside risk. Using
`rsv/rv5` and a freely fitted coefficient is invariant in predictive capacity to
swapping the labels of the two complementary halves. Do not impose the
literature's negative-semivariance coefficient restrictions, or describe a
signed jump, until the sign convention is independently pinned. Use `rv5` and
`rsv`, keeping their non-subsampled definitions matched, for the first run.

No realized quarticity or intraday return sequence is present in the archive.
`nobs`, realized-kernel disagreement and `rv5-bv` cannot substitute for
quarticity in a formal BNS jump test or HARQ model. Keep source bytes private;
the earlier mirror/provenance and redistribution limitations remain in force.

## Proposed finite SPX family

This is a design recommendation; the executable experiment must separately
freeze its exact transformations, dates, losses, fitting rule and complete
comparison family before any score is computed.

Use next 1, 5 and 21-session arithmetic mean `rv5` as the three targets.
Forecasting trading-hours five-minute RV is explicitly a different target
from the existing daily GK-plus-overnight total. Keep the complete outcome
window inside the archive; do not shorten targets near June 2018.

The minimum strong control is log-HAR on lagged `rv5` at 1/5/22 scales,
one-session-lagged VIX, and daily/weekly/monthly negative-return leverage.
Use the same additional available stress/term inputs, training rows and
availability lags on both sides if the root protocol includes them. A weak
HAR-only baseline may be shown as context but cannot establish orthogonality.
Use the already verified exact log-OLS smearing estimator for every arm.

| Candidate block | Fixed raw construction | Question and required control |
|---|---|---|
| Semivariance share | For windows 1, 5 and 22, sum `rsv` divided by sum `rv5`; then one-session lag | Does intraday composition add information beyond total RV, VIX and return leverage? |
| Measurement disagreement | For windows 1, 5 and 22, log of sum `rk_parzen` divided by sum `rv5`; then one-session lag | Does a second estimator's discrepancy improve on the same `rv5` history? This is not HARQ. |

The bounded initial family is the two blocks individually against the same
strong baseline at the three horizons: **six primary comparisons**. A joint
block or additional target creates another comparison and must be declared
before the first score, rather than added because the individual arms fail.
Train-only residualization against the baseline can describe redundancy; it
does not itself turn a correlated input into an economically new signal.

A defensible staged design has initial training through 2009, development
forecasts in 2010–2013, and an unchanged fixed family evaluated in 2014–2017.
The 2018 partial year can be left unscored for a later algorithm check. Purge
development origins whose targets enter validation. Annual or monthly fits
must use completed targets only. No estimator, lag, feature or asset should
be selected by validation results. Existing SPX jump and term-slope work has
already examined 2014–2017 for other questions; describe this as reused-history
replication, not a pristine holdout.

Retain the repository's conservative paired block inference and multiplicity
correction across the whole new round, including any simultaneously registered
index-return tests. Report years and all horizon phase offsets, effect sizes,
uncertainty and nominal detectable effects. A shortlist requires the fixed
minimum improvement, corrected significance and period stability; no parameter
changes after inspecting outcomes. A retrospective pass warrants prospective
validation and a separate implementation-cost study, not immediate trading.

The rationale is substantive: [Patton and Sheppard's original paper](https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf)
finds different persistence for the positive and negative components of
high-frequency variation. It does not establish an increment over this repo's
VIX-plus-leverage control. [Liu, Patton and Sheppard](https://public.econ.duke.edu/~ap172/Liu_Patton_Sheppard_JoE_2015.pdf)
find five-minute RV difficult to improve on across many measures; accordingly,
the kernel-disagreement arm is a test of incremental information, not a claim
that a kernel target is automatically superior. [Patton's forecast-loss paper](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)
motivates loss functions robust to imperfect volatility proxies under its
assumptions; a chosen loss does not repair a biased target or mismatched asset.

## Other paths and existing limits

- **QQQ day/overnight decomposition remains feasible.** Raw OHLC provides 6,696
  sessions through 2025-10-20. Its overnight squared gap has 81 exact zeros.
  A future fixed experiment could model GK daytime and squared overnight
  targets separately and add their conditional means, with a direct total
  model using identical predictors and fitting objective. Zero overnight
  targets require an estimator and loss defined at zero; silently deleting
  them or flooring them after a failure is not acceptable. This is a different
  decomposition question, motivated by [coupled component volatility models](https://ifs.org.uk/sites/default/files/output_url_files/cwp051717.pdf)
  and [overnight GARCH-Itô](https://arxiv.org/abs/2102.13467). It does not
  reproduce those papers' estimators or high-frequency overnight data.
- **Long horizons alone are not unexplored.** `RESEARCH_PATHS_FINDINGS.md`
  already covers HAR versus HAR+VXN at 1, 5, 10, 21, 42 and 63 sessions.
  Running the new high-frequency feature family at a fixed 21-session horizon
  asks a different question; a broad scan for the most favorable horizon does
  not.
- **True semivariance is different from the old hourly proxy.** Existing QQQ
  signed variation uses about seven hourly bars per day over a short history.
  Leverage is already in the strong control. Do not advertise rediscovery of
  that mechanism as a new orthogonal result.
- **State conditioning has substantial negative evidence.** The two-state
  forward-filtered HMM, its calibration repair, observed-state memory, latent
  memory, recency weighting and dynamic ensembles have already been tested.
  Replacing state thresholds after those scores is not a fresh hypothesis.
- **Do not reopen the term-slope interaction on the same windows.** Its
  regime-dependent interpretation came from inspected results and the SPX
  repair failed. Likewise, do not retune the SKEW jump target or earnings
  concentration hypothesis on their spent windows.
- **HARQ and formal jump tests need another source.** The [HARQ paper](https://public.econ.duke.edu/~ap172/BPQ_Exploiting_Errors_JoE_2016.pdf)
  uses measurement-error information from high-frequency quarticity. Existing
  `nq_intraday_study.yaml` has a proper tripower-quarticity contract, but its
  former no-evaluable-fold result does not authorize lowering its data gate.
