# SPY high-frequency measurement audit

The downloaded SPY research series has positive finite trade-QMLE estimates on
every available date in the fixed periods. It supplies substantially more
recent measurements than the Oxford supplement, but has important calendar
gaps. This is a source audit: **zero forecasts and zero predictive comparisons**.
No source has yet been admitted to a new forecasting study.

The public response has 7,680 dates; only the 7,473 dates through 2025-10-20 were
numerically parsed. Header identity, count, date extent, ordering and uniqueness
passed. The downloaded 737,911 bytes have SHA-256
`b7802ef21565b13831420c8fdd2b15176e57265fd5afe981be73693c8350e392`,
matching the earlier metadata-only discovery. Full raw bytes remain quarantined
under ignored `data/source_discovery/quarantine/risklab_spy/`.

| Fixed measurement period | Reference equity sessions | Available source dates | Missing source dates | Invalid trade-QMLE values | Complete 22-session windows |
|---|---:|---:|---:|---:|---:|
| 2000–2015 | 4,025 | 4,025 | 0 | 0 | 4,004 |
| 2016–2019 | 1,006 | 999 | 7 | 0 | 927 |
| 2020–2025-10-20 | 1,458 | 1,438 | 20 | 0 | 1,056 |

These are measurement dates, not the earlier overnight experiment's entry and
mature-label samples. Window counts restart at each displayed period boundary;
they are diagnostics, not a prospective model's final sample. QQQ date labels
provide a common-equity-calendar check; they do not establish a historically
announced SPY calendar.

All six estimate fields are finite and positive within these fixed periods.
Their associated displayed confidence widths and MA orders meet the declared
validity rules. The source omits the entire 2018-02-05–2018-02-09 week. Treating
remaining rows as consecutive sessions would conceal that gap. A future study
must retain missing calendar positions, require complete forecast targets and
lookbacks, and report missingness as a representativeness limit.

Pre-2016 quote-based fields include values above the public chart's 300% cap:
4 QMLE, 7 five-minute and 6 fifteen-minute observations. They remain in the audit;
none was clipped or repaired. Trade-QMLE was fixed as the prospective primary
measure before numerical inspection. This choice was not changed after seeing
the quote estimates. Detailed quantiles and all annual/date diagnostics are
retained in the ignored measurement JSON; no target or predictor relationship
was evaluated.

The [provider](https://dachxiu.chicagobooth.edu/#risklab) describes daily annualized
volatility and restricts the data to research use. The
[public chart parser](https://dachxiu.chicagobooth.edu/volatility.js) confirms that
stored levels are displayed as volatility percentages without an active square
root transformation. The audit preserves native levels. The exact equity
session boundary, annualization factor and historical publication/revision
contract remain unresolved. A prospective study could explicitly target
**squared provider-native annualized volatility**, avoiding an invented daily
conversion; that would still require a fixed timing and missingness contract.
It would be a separate ETF research target, not a replacement for frozen SPX
realized variance or a claim about a tradable data license.

The [pre-inspection contract](RISKLAB_MEASUREMENT_CONTRACT.md) and ten synthetic
tests preceded the download. The first execution failed before producing audit
output because the calendar reader compared a parquet timestamp with a string.
The original source, tests, hashes and failure record are retained. A minimal
typed-cutoff correction and an additional synthetic parquet test preceded the
successful calculation; all 11 tests and scoped lint passed. No measurement,
period, field-selection or decision rule changed. This was a source-audit
implementation repair, not a rerun or amendment of any frozen forecasting study.

- [Independent reconstruction](RISKLAB_MEASUREMENT_VERIFICATION.md)
- [Broader source discovery](EXTENDED_HF_DATA_DISCOVERY.md)
- [Current signal-search result](SUMMARY.md)
