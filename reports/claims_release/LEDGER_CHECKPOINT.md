# Claims first-report ledger: independent source verification

The first-report ledger was built and independently verified on 2026-09-08. This advances the completed source audit; it is not a forecasting result. The original source-audit checkpoint and its reports remain unchanged.

The ledger retains every one of the 856 reference Saturdays from 2009-05-30 through 2025-10-18: 851 supported first-report values, one unresolved correction and four weeks without an admitted report. The checker independently reconstructs these rows from all 852 parsed original DOL reports and 852 ALFRED comparison records. It does not import the producer or its source-validation helpers.

The 2018 discrepancy retains the original DOL value and original release date. The 2017 correction retains its known release date with an unavailable model value; both conflicting counts remain comparison evidence only. The four absent tail weeks have no invented publication date. The original ALFRED reconciliation remains `SOURCE_RECONCILIATION_INCOMPLETE`.

All 129 source and ledger synthetic tests passed before real ledger construction, including the producer's 23 and independent checker's 19 prewritten contracts. Source inputs and component hashes were checked before and after construction. All 336 earlier frozen Python files and 90 completed peak-age report artifacts were unchanged.

The private ledger is `data/claims_release/first_report_ledger.json`, SHA256 `f504c475a65eabab242ef583c7494b773cae13e3026de8a62fb5390ddab65c52`. The separate [admission certificate](LEDGER_ADMISSION.json) binds this output to its inputs, implementations, test evidence and independent verification. The producer payload retains its original unverified status string; the separate certificate records verification without rewriting that payload.

The next stage is to finish and test the forecasting components, freeze the two-comparison protocol, then construct the market cohort and run it. No claims market cohort, numerical support counts, model fit or score has yet been calculated. Registered comparisons remain 140. Historical archive timing and reused-market-data limitations remain in force.
