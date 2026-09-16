# Commodity implied-volatility source checkpoint

September 8, 2026. Both original Cboe history captures passed the bounded parser and independent reconstruction. This is a source-admission result, not a forecasting result. The cumulative family remains **142 comparisons**; no commodity hypothesis has been registered and no commodity feature panel, forecast cohort, fit or score has been computed.

| Source | Retained original rows | Explicit missing values in those rows | First retained date | Last retained date |
|---|---:|---:|---|---|
| OVX | 4,045 | 0 | 2009-09-18 | 2025-10-20 |
| GVZ | 4,045 | 0 | 2009-09-18 | 2025-10-20 |

These counts do not establish complete trading-calendar coverage. Neither downloaded file supplies an observation between the declared 2009-01-02 floor and its first retained date. The floor, model specification and evaluation dates remain unchanged; no earlier alternative source or filled history is substituted. Each original file also has 221 later rows: their dates and structure were validated, but their numerical cells were never converted or returned. All original bytes and bounded numerical outputs remain private.

Before numerical admission, all **39 synthetic source contracts** passed: 12 producer tests, 25 independent-checker tests and two end-to-end integrity tests. The producer and checker independently enforce exact source hashes, headers, date ordering, complete-file date preflight, fixed bounds, explicit missingness, valid positive decimal values and original physical line numbers. The first two components' missing-module failures were retained before implementation. The serialized output was independently checked again before saving. This check does not substitute for the later full repository preflight or independent forecasting reconstruction.

The acquisition receipts retain HTTPS verification, successful responses, exact official download identities, original response bytes and hashes. Current historical captures still do **not** authenticate immutable original daily vintages. The [source feasibility review](SOURCE_FEASIBILITY.md) records historical methodology changes and this limitation. A one-session lag does not eliminate revision risk.

The new feature/model components separately passed 18 generated tests before using real inputs. They fix one joint oil/gold option-implied block and three common-sample models: the established market model, that model plus recent oil/gold ETF return and variance controls, and the full model adding both option indexes. No single-index selection or model grid is proposed. Monthly causal scheduling and independent historical verification remain implementation work; full testing and a complete prospective freeze must precede any forecasting run.

Admission evidence: [source certificate](SOURCE_ADMISSION.json), [pre-admission checkpoint](SOURCE_PRE_ADMISSION.json), [parser contract](SOURCE_PARSER_DESIGN.md), [independent verification contract](SOURCE_VERIFICATION_DESIGN.md), [component tests](SOURCE_COMPONENTS_PRE_ADMISSION.log), and [prospective model review](PROSPECTIVE_DESIGN_REVIEW.md). The original plans and review notes describe their earlier inspection boundaries; this later checkpoint records actual bounded source admission without rewriting those notes.

The one-time admission script initially stopped before creating its checkpoint because prior publication-report paths were relative to their report directory. That orchestration path was corrected before any source value conversion; no parser rule, test expectation or model choice changed. All 370 previously frozen Python files and the prior claims freeze, input, preserved and report artifact hashes passed their checks before and after successful source admission. The completed claims negative result is unchanged.
