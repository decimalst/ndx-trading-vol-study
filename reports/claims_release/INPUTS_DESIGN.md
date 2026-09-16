# Claims market snapshot reader

Seven synthetic tests were written and failed with the expected missing-module error before implementing `src/claims_release_inputs.py`. The reader is isolated from the source-ledger work and has not been used on real market data.

It accepts exactly the four existing raw-source paths and their prospectively frozen SHA256 values. All files are read into byte snapshots and authenticated before any Parquet decoding. All complete date columns are then checked before any numerical column is requested. Dates must be unique, ordered, native, timezone-naive and midnight; date-only metadata may extend beyond the numerical source ceiling.

Numerical decoding uses those same authenticated byte snapshots, exact necessary column lists and a fixed date filter through 2025-10-20. Bounded output dates must equal the independently inspected date subset. The reader omits unused source columns, preserves missing values and gaps, and combines the three required implied-volatility series without filling. The market-feature builder separately checks numerical validity.

Tests cover a mismatch in the last source hash before any decoding, disk mutation after authentication, malformed calendars in both first and last sources, exact column/date requests, missingness and source-inventory validation. Future/protected numerical sentinels occur only in generated fixtures and do not enter the returned frames. Source-vintage and vendor limitations of the original market files remain unchanged.

Production invocation is pending forecasting registration. This implementation and its synthetic tests do not establish market coverage or a predictive result.
