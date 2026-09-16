# Independent recovered-source application review

Status: **VERIFIED_INDEPENDENT_RECOVERED_SOURCE_CHECKS**. Final audit found no unresolved discrepancy in the bounded three-event application. All 28,215 old/new pins remained unchanged before and after the independent check. The sections below preserve the initial inspection and pre-application correction history; the final output audit completes their previously pending steps. No market arrays, feature panels, model fits, test reruns or predictive scores have been read or produced in this review.

## Delayed TLT and weekday controls

Read the complete market-context implementation, its nine prewritten tests, the frozen calendar/schema/arithmetic helpers it calls, and the common32 model contract. No concrete implementation discrepancy was found in the inspected version.

The module checks both complete date envelopes before examining selected numeric values. It requires the inherited exact five-column schema while examining only TLT prices, preserves the reference calendar, makes pre-2010 TLT prices unknown before any lag/return, and uses log P[t-1] minus log P[t-2]. Squared returns and strict five/22-session mean squares retain the same full-calendar gaps. A genuine zero return remains zero; an all-zero variance window has unknown logarithm. No backfill, skipped missing session, epsilon or same-day price enters the controls. Tuesday–Friday indicators use the entry date; the Treasury cutoff is the immediately preceding full reference session when that session is at least January 1, 2010. These eight numeric controls fit the declared 12/31/32 arm ordering.

The nine tests include independent loop arithmetic, current/future price poisoning, missing-session contamination, exact floor warmup, zero-return and weekday behavior, untouched unrelated assets, typed numeric/date guards, time-unit transport and input preservation. The saved 57-test source precheck reports success, including these nine; this review did not rerun those tests or turn their generated-data success into historical validation. Source snapshot authenticity and actual market-calendar provenance remain the caller's responsibility, as the module states.

## Recovered-source application boundary

The application contract restricts additions to YZ7 (2019-12-23), YY0 (2019-12-24) and AV3 (2020-12-09). Existing parsers must authenticate unchanged checked originals and exact PDF/XML amount agreement; recovered PDFs retain original failed captures as separate evidence. YY0 requires complete conditional-notice review plus unchanged final five-year identity, never presumed substitution. Z52 remains outside numerical recovery. Qualified printed agency release dates, observed archive upper bounds and current retrieval timestamps must remain distinct. No new forecast evidence follows from source reconciliation.

Script inspection and an independent standard-library XML/Decimal transcription check of the three private outputs will be recorded here only after the parent supplies the completed implementation and outputs. Current pending status is not a successful source-admission certificate.

## Inspected version pins

| File | SHA256 |
| --- | --- |
| `src/treasury_market_context.py` | `3f1ee35fc81171fa58d90f52670d199501c8ad341f95f6e79069b5a4bb9c4d9b` |
| `tests/test_treasury_market_context.py` | `e843d6a869a83cad6764a55bf3449748a9bf7c26e36bc42b07c188f122ef7d76` |
| `src/commodity_implied_features.py` | `84228da0ddc1b81b554b361ec8bdfeaa621d235b029f334708b25d9617411678` |
| `src/treasury_dealer_models.py` | `f45f000436e66f5134077af4a236e46b47959828222ee943d3ff56ec2788f54c` |
| `reports/treasury_dealer/MARKET_CONTEXT_CONTRACT.md` | `f85a0c454e893cbf18d1660edad514cae9873d4420c06c23a9289734d31c3314` |
| `reports/treasury_dealer/RECOVERED_SOURCE_APPLICATION_CONTRACT.md` | `3fa4da812b5d71cf019b93a74ac725d9087441a37b03b6f3553babae6d9a5ce0` |
| `reports/treasury_dealer/RECOVERED_SOURCE_PRECHECK.log` | `3da87653535c468be5d797624ffdb8d7eb6e75afc5a9f86e1fecbe17785ba5fd` |

## Initial application inspection and concrete requested correction

Read the complete first application script and the three events' prior identity/notice metadata. It preserves the 1,138-event original membership; checks every old/new pin before parsing and again before terminal publication; hashes the exact bytes it decodes; binds recovered result PDFs to exact original URLs, archive digests and successful verified-TLS receipts; and applies unchanged identity, sixteen-field result and three-source offering parsers. It refuses existing output destinations and leaves original failed captures untouched. YY0's final identity must exactly equal its prior paired identity. No unsupported substitution or date alias was identified in these paths.

A concrete audit-closure omission was reported before freeze: YZ7 has two linked conditional notices (_20191219_1 and _2) whose old dispositions still require final-result confirmation, while YY0 also has a separate schedule notice (_20191219_4). The initial script validates unchanged final identities but does not consume/save these per-notice dispositions. Requested correction: bind the exact prior notice memberships and classifications, then explicitly retain how the final unchanged identity closes each conditional warning; retain the schedule-only notice separately. AV3 has no linked special notices. This changes documentary closure, not the existing financial parsing or allocation definition. Correction inspection remains pending.

After the first market-context review, the parent formatted its new test file only; the new test hash below is additive version evidence. The previously recorded inspected hash remains historical evidence, not a claim that the bytes did not change. The market-context implementation itself was unchanged.

| Follow-up inspected file | SHA256 |
| --- | --- |
| `scripts/reconcile_treasury_recovered_sources.py` | `acf0ce6b39c03df5e85fdc000ed4ebbe97cf5e73e3750e214a6a56cf6e11c5ad` |
| `tests/test_treasury_market_context.py` | `c8fb6d18a8ab932c05f9edc46e789df19e280c29267556d4f51e9888ccffb6e8` |
| `reports/next_signal_review/TREASURY_HISTORY_IDENTITY_V3.json` | `5ef8caa91cf9a7418c73a625fad0929625bbb54b82889d8bf507d0f7875326a1` |
| `reports/next_signal_review/TREASURY_FULL_NOTICE_REVIEW.json` | `4bea660c12ae0ba31ef388f0f8559fa3b04daee6bdbb9690d7f60b062a2ce155` |

## Final pre-application read

The revised script resolves the reported omission before financial conversion. It checks exact four-notice memberships across the three events, preserves the old disposition, binds the reviewed body/text hashes, and saves the final actual CUSIP/original-tenor/reopening outcome for every notice. The two YZ7 conditionals and recovered YY0 conditional require unchanged new-issue outcomes; YY0's separate closing-time notice remains schedule-only. AV3's empty special-notice membership is explicit. Expected final original tenors/reopening are 2/false, 5/false and 10/true. The revised script also requires original role filename agreement and bounds each recovered result snapshot between its auction and October 20, 2025.

No further concrete discrepancy was found in the bounded final pre-application inspection. Source values and output checks remain pending actual application; this is a code/contract review only. The running independent review is intentionally outside input pins until its final output audit is appended.

Final pre-application script SHA256: `38bac53bfdc7d884e5f43f089ac606521a50fa92cee1d4c7d25ea11d655e5421`.

## Completed independent output and source-value audit

The application completed under checkpoint `abcd5ae5d0f120e4308e47ef522f5d0956f1219151f0bb60601248064642a7a2`. This audit independently authenticated that checkpoint, its original terminal anchor, and every 28,215 combined input pin before source decoding and after all comparisons: 28,136 original terminal pins plus 79 new pins. Every selected input was read through a hash-checked byte snapshot. All 1,138 original event identities remained present and matched the prior event membership; the exact three additive output identities and hashes matched the terminal application record. The original failure/disposition records and full prior output set remained unchanged.

The source-value check used Python's standard-library ElementTree and Decimal directly, with a literal independently declared sixteen-field list. It did not import or call the production XML/PDF amount parsers, offering parser or application runner. It independently decoded 48 direct XML result fields, checked their exact equality against 48 saved XML values and 48 saved PDF values, and reconstructed competitive category sums, accepted-versus-tendered ceilings, public/SOMA/FIMA total relationships and the published bid-to-cover rounding check. Financial figures are deliberately absent from this public review.

All nine offering-source tokens were checked independently: three literal USD-labelled announcement PDF rows, three announcement XML fields and three result XML fields. Exact Decimal scaling by the fixed 1e9 hypothesis agreed with each saved integral USD amount and the three-source total. This is the declared record-specific paired-unit check, not a claim that the XML scale is universally schema-authenticated.

The audit also checked source XML date/CUSIP leaves, unchanged announced/final identities, original lineage dates and Y/N reopening lexemes, exact source-input hashes, the recovered PDFs' archive digests and transport identities, complete one-page source text, original result headers, and YY0's byte-identical prior identity result. Recovered raw result PDFs were independently re-extracted for text identity; their sixteen financial table fields were compared against independent XML decoding via the saved PDF fields, **not** parsed a second time by a new independent PDF monetary-table implementation. The existing separately authored PDF parser's exact cross-format checks remain part of the application evidence.

| Event | Original tenor / reopening | XML fields independently decoded | Comparisons to saved formats | Offering source tokens | Explicit notice closures |
| --- | --- | ---: | ---: | ---: | ---: |
| 2019-12-23 / 912828YZ7 | 2 years / no | 16 | 32 | 3 | 2 |
| 2019-12-24 / 912828YY0 | 5 years / no | 16 | 32 | 3 | 2 |
| 2020-12-09 / 91282CAV3 | 10 years / yes | 16 | 32 | 3 | 0 |
| Total | Three exact scoped events | 48 | 96 | 9 | 4 |

All four closures retain the original notice review/disposition and source hashes while recording the final paired identity. YZ7's two alternatives were not realized; YY0's recovered conditional alternative was not realized and its separate closing schedule notice remains schedule-only. AV3 has no special notices in this fixed scope. No additional confirmation document is fabricated for an unchanged final identity.

Two initial audit-side checks were corrected without any source, output or producer-code change: the source uses exact `Y`/`N` reopening lexemes rather than the auditor's initial `Yes`/`No` assumption; AV3's English release header prints a leading-zero day, so the audit parses the exact header date instead of demanding an unpadded display string. The first correction preceded financial checks; the second was encountered after YZ7 and YY0 completed. The final full audit then passed all three records and repeated every preservation hash. These were audit-oracle assumptions, not concealed parser or source repairs.

## Final result pins and practical boundary

| Artifact | SHA256 |
| --- | --- |
| `reports/treasury_dealer/RECOVERED_SOURCE_CHECKPOINT.json` | `abcd5ae5d0f120e4308e47ef522f5d0956f1219151f0bb60601248064642a7a2` |
| `reports/treasury_dealer/RECOVERED_SOURCE_APPLICATION.json` | `f4097f11c9da4f7f395cd55643b5dcd12c437cdd3e92e97b24ddcc3af372901c` |
| `reports/treasury_dealer/RECOVERED_SOURCE_APPLICATION.log` | `de4d74f01f256fab864024758f42107050d3fb47b202157c67b734c0314dd5dd` |
| `scripts/reconcile_treasury_recovered_sources.py` | `38bac53bfdc7d884e5f43f089ac606521a50fa92cee1d4c7d25ea11d655e5421` |
| `data/source_discovery/treasury_auction/clock_recovery_v1/reconciled_events_v1/2019-12-23_912828YZ7.json` | `41c6df1ce4ec14392a5b4828a1b03a80a302c8922d5fe5bca48ba5cec3c7d20c` |
| `data/source_discovery/treasury_auction/clock_recovery_v1/reconciled_events_v1/2019-12-24_912828YY0.json` | `5f941445b2f0be66c36eb8f4f92e13cbdb0f762d05c6f24bef15615a150baa56` |
| `data/source_discovery/treasury_auction/clock_recovery_v1/reconciled_events_v1/2020-12-09_91282CAV3.json` | `b381e983e21c7d8c10d49f98c31455e65919b2709ff0ca74728130101a1aeffd` |

The original terminal records 1,132 complete source agreements; these three disjoint additions bring the potential reconciled source-event count to 1,135 of the unchanged 1,138-event scope. This does not imply 1,135 usable forecast origins, complete donor history or satisfied model support. January 2016's identity conflict, January 2020's missing original result PDF and the documented contingency test retain their separate dispositions. Qualified printed agency release dates remain distinct from archive observation upper bounds; no exact first-publication or historical-delivery claim is added. Full source-history admission remains false, new predictive comparisons remain zero, and the registered family remains 144.
