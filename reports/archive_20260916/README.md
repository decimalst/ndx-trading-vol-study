# Research archive verification — 2026-09-16

This is a documentation and archival commit of the accumulated research after
`40af1fc`. It adds the current README, a follow-up study index, protocols,
source, tests, reports, generated figures and audit records. It does not change
the verdict of a frozen study or start a new predictive experiment.

The commit excludes private market/forecast datasets, ignored source captures,
model weights and environments. Eight unignored Treasury forecasting artifacts
under `data/treasury_dealer/forecasting/` also remain local. Existing frozen
`.gitignore` and Makefile changes are included at their already-pinned bytes.

## Checks in this archival session

| Check | Result / evidence |
|---|---|
| Documentation review | Independent review checked the numerical claims, outcome labels, caveats and relative links against saved results. All local links in the README and new index resolve. |
| Latest research tests | **53 passed**, with no skips: spread replay, geometry, independent geometry verifier and prospective ledger. [Log](LATEST_TESTS.log). |
| Original fast target | **38 methodology tests passed**, then smoke passed. This target also accesses previously evaluated history; see the disclosure below. [Log](FAST_CHECK.log). |
| Environment | Required versions matched for scikit-learn, NumPy, SciPy, pandas and statsmodels. [Log](ENV_CHECK.log). |
| Selected repository suite | **3,571 passed**, no failures, errors or skips; exactly six historical-access tests excluded from 3,577 discovered. Completed in 587.2 seconds. [Result](TEST_GATE_RESULT.json), [selection](TEST_SELECTION.json), [log](SELECTED_SUITE.log). |
| Lint | **15 existing style/simplification findings remain** in frozen source/test files: import order, dictionary construction, context managers and the UTC alias. No automatic rewrite was performed. This is not a clean lint/CI result. [Log](LINT.log). |
| Staged whitespace check | Reports 2,037 existing whitespace findings across 39 preserved report files. The newly edited documentation has none. Historical logs and frozen artifacts were retained byte-for-byte rather than normalized. |
| Frozen hashes | **PASS:** 29 receipts, 61,851 hash references, 30,302 distinct files; zero missing files or mismatches. This reads bytes and manifest metadata, not numerical observations. [Receipt](FROZEN_HASH_AUDIT.json). |

## Historical-access disclosure

The initial fast target executed all 38 methodology tests, including the six
historical-access tests identified in the earlier
[correction](../treasury_dealer/predictive_prefit/HISTORICAL_TEST_ACCESS_CORRECTION.md).
The general discovery suite was also started, then interrupted when this scope
was identified; it is **not a completed suite result**. Its
[interrupted log](INTERRUPTED_FULL_SUITE.log) is retained.

The replacement run uses the existing `treasury_dealer_test_gate` selector,
which saves the complete discovered inventory and excludes exactly those six
previously disclosed identities. No test bytes or selection rules were changed
for this commit. The earlier fast run still accessed the historical period;
using the selected suite afterward does not reverse that access or make any
observations untouched. These checks support regression and preservation, not
fresh empirical confirmation.

The preserved formal shape study remains UNEVALUABLE. The later spread replay
and geometry search remain descriptive, with hypothetical premiums and
hindsight geometry selection. No actual options profitability is asserted.
