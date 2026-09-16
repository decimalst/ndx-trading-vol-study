# Bounded review of commodity implied source admission

The code and metadata reviewed support the completed bounded-source admission, with the declared limitation that these are current Cboe captures whose original historical vintages remain unverified. No blocking source-validity defect was identified. This review does not authorize a forecast, establish point-in-time immutability, or independently repeat the actual numerical admission.

The review read the new producer parser, one-time admission script, independent source checker, generated integration tests, source test log, admission certificates, and HTTPS receipt/header metadata. It did not open either raw CSV body or parsed-record file, run a parser on historical data, inspect index values, construct features or cohorts, or compute support, fits, or scores. The reviewer authored the separate checker before seeing the producer; this subsequent code review is not a new third numerical reconstruction.

## Scope and conversion boundary

`src/commodity_implied_source.py` and `src/verify_commodity_implied_source.py` agree on the frozen lexical and structural contract. Both authenticate the exact raw SHA-256, decode UTF-8 with an optional initial BOM, require the exact two-field symbol header, use strict CSV parsing, and validate every nonblank record's calendar date and strictly increasing unique order before any index-value conversion. Both handle source lines using the CSV reader's physical ending line. Neither strips date or value fields.

Both enforce an outer numerical window of 2009-01-02 through 2025-10-20, allowing narrower windows only. Outside-window tokens may pass through UTF-8/CSV tokenization, but neither parser sends them to numerical conversion or returns their values. Retained missing cells are exactly empty or `.`; observed cells must be finite positive ASCII decimals. The independent checker reconstructs and compares the complete result, including exact scalar types, metadata, record order and original lines. The generated tests explicitly instrument the conversion boundary and place invalid numeric sentinels outside the retained window.

The pinned combined log records 39 passing generated tests in 0.009 seconds, comprising producer contracts, independent-checker contracts, and two integration tests. The integration tests cover JSON roundtrip and deliberate record/value/missingness/line/count corruption. No test rerun was needed for this read-only review.

## Certificate and receipt evidence

The following current certificate byte hashes independently matched the supplied pins:

- `SOURCE_PRE_ADMISSION.json`: `802dd08edab6cd08e13f52777f10e8f5757849edac2f0216a9cf961bba2b35e2`.
- `SOURCE_ADMISSION.json`: `139f29534f2a6ea289d265d128c938f5ff898d35556b640457d2175f4b2c5385`.

The final certificate references the exact pre-admission hash. A further 20 non-numerical files matched the pre-admission component, evidence and acquisition pins: six source/test/script components, nine evidence files, and five acquisition ledger/header/receipt files. Raw bodies were deliberately excluded from this review's byte reads, so their reauthentication rests on the completed admission execution and its pinned evidence.

Both saved receipts agree with the acquisition ledger and final certificate. They show GET requests to the exact official `https://cdn.cboe.com/api/global/us_indices/daily_prices/OVX_History.csv` and `GVZ_History.csv` identities, matching effective URLs, HTTP 200, zero curl return code, TLS verification result zero, no redirects, and CSV content type. The acquisition timestamps precede the pre-admission and final certificate timestamps. The response headers' September 2026 `Last-Modified` values describe the captured files, not the first publication dates of their historical observations.

The script checks the prior claims freeze and publication hashes and rehashes their declared preservation closure before admission. It records 1,511 preserved artifacts checked and repeats the preservation/component/evidence/acquisition hash checks after the two source verifications. This review inspected that control and the pinned record, rather than rereading the old numerical artifacts. The script does not modify source bodies or prior artifacts. It refuses an existing attempt and creates its outputs exclusively. Filesystem metadata independently confirms both raw bodies and both parsed outputs are regular files with mode 0600. No source-admission failure marker is present.

## Limits to carry forward

The certificate reports 4,045 observed rows for each source, beginning 2009-09-18 and ending 2025-10-20, with 221 later rows counted by date and excluded from numerical conversion. These are source-file metadata, not a market-calendar coverage audit, matched cohort, or effective sample size. The absent period between the prospective lower bound and the reported first row must remain absent; the admission does not justify backfilling it or silently moving an experimental start date.

The completed script verifies both parser outputs and their serialized JSON representations before saving private parsed files, then binds each saved file's byte hash in the final certificate. A downstream reader must consume those exact bound outputs and require the completed certificate; an individual parsed file or the pre-admission checkpoint alone is not the completed source proof. The existing certificate's explicit no-forecast and cumulative-family-142 fields correctly avoid treating source admission as an additional predictive trial.

Original immutable historical vintages and exact historical CSV publication clocks remain unverified, as already documented in `SOURCE_FEASIBILITY.md`. A declared lag can govern prospective availability assumptions, but cannot turn this capture into an authenticated original-vintage history. Any subsequent experiment must retain that limitation. No change to pinned code or certificates is requested by this review.
