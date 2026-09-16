# Treasury realized-security identity contract

This new helper reconciles metadata from an original announcement PDF/XML pair and a final competitive-result PDF/XML pair. It does not admit Treasury history, decode financial fields, apply conditional yield thresholds, or alter frozen source bytes or helpers. Only generated fixtures were exercised.

```python
reconcile_realized_identity(
    announcement_xml, announcement_pdf_text,
    result_xml, result_pdf_text,
    *, auction_date, announcement_date, selected_final_cusip,
    confirmation=None, conditional_alternatives=(),
) -> dict
```

Selected dates are plain ISO strings; the auction must be within 2010-01-01 through 2025-10-20. The future-auction guard runs before decoding. XML dates retain their exact source lexemes and accept only plain ISO or midnight ISO. PDF text is supplied by the caller; original PDF transport and text-extraction bindings remain the caller's responsibility.

Each PDF must match its own XML CUSIP, dates, document role, fixed nominal Note/Bond type and offered term. An announcement typo cannot become an alias. The frozen XML whitelist/safety extractor and frozen PDF header checker remain unchanged. `DatedDate` is independently read as one exact, direct announcement leaf after the frozen XML safety check; financial leaves remain opaque.

The output retains both CUSIPs, exact extracted metadata, both dated-date lexemes, PDF headers, separate original-lineage descriptions, and hashes of the four supplied inputs. The offered term includes the security type, for example `2-Year Note` or `29-Year 10-Month Bond`. Original tenor uses the exact stated dated-to-maturity calendar anniversary within {2, 3, 5, 7, 10, 30}; unsupported non-anniversary cases fail. Settlement may follow the dated date. A reopening requires internally consistent original dates and matching PDF/XML original issue dates.

For an actual substitution, the announcement must describe a new issue and the final pair a reopening. Both pairs must agree on offered term, auction type, settlement and maturity context. The hypothetical announced security and realized security may have different original dated dates, issue dates, series and original tenors; those differences remain visible. The final actual identity must equal `selected_final_cusip` and have a dated, explicit realized-reopening confirmation. Conditional alternatives are retained only as non-exhaustive audit metadata: the final identity need not occur in that tuple. An unchanged identity needs no confirmation and must retain consistent lineage.

The canonical confirmation has exactly these fields:

```text
kind: REALIZED_REOPENING
url, body_sha256, text_sha256
release_date, auction_date, announcement_date
offered_term, actual_cusip
original_term_years, original_issue_date, series
```

The helper validates an official TreasuryDirect PDF URL, bounded dated identity, lowercase SHA-256 strings, and agreement of all identity/context/lineage facts with the paired documents. The caller must independently bind this record to the completed notice ledger, its exact field-extraction location, checkpoint, and PDF/text pins. Merely supplying plausible hashes does not establish provenance; the returned `confirmation_provenance_verified` is always false.

`identity_available_date` is the latest required actual document date, including a later final result or confirmation. It is a date-only evidence boundary, not proof of historical delivery time or a trading availability rule. Source release clocks are preserved in extracted metadata where present; no result-publication date is inferred from auction date or `ReleaseTime`.

Two known cases remain explicit `UnsupportedIdentity` exceptions carrying an `.audit` dictionary:

- `UNSUPPORTED_ANNOUNCEMENT_DATE_CONTEXT` retains the archive announcement date, exact source announcement-date lexeme, PDF release date and affected source role. Revised announcements, including the January 2015 weather cases and February 2016 technical reschedule, require later dated replacement reconciliation; source dates are never silently substituted.
- `UNSUPPORTED_ORIGINAL_CUSIP` retains any nonempty results `OriginalCUSIP` lexeme and the announced/actual CUSIPs. Its meaning is deliberately unresolved pending full-history metadata inspection.

Successful status is `RECONCILED_METADATA_ONLY`, with `source_admitted: false`. Unknown schemas or contradictory facts raise `ValueError`; source problems are not numerical nulls. Original records and frozen strict identity checks remain unchanged, including their rejection of a substituted CUSIP under the old single-identity contract.

## Prewritten evidence

The 12 generated test methods were written before the new source module. The initial invocation of `python -m unittest tests.test_treasury_realized_identity -v` failed at import with `ModuleNotFoundError: No module named 'src.treasury_realized_identity'` (one loader error, 0.000 seconds). This was the intended missing-module RED and preceded implementation.

After implementation and formatting, the same focused command passed all 12 methods in 0.036 seconds. Ruff checks passed for only the two new Python files. Tests cover unchanged and substituted identities, non-exhaustive warnings, required actual confirmation, paired identity and role conflicts, revised announcement dates, final lineage, confirmation facts/pins schema, opaque pathological financial text, future-source rejection before decoding, delayed evidence dates, settlement shifts and exact unsupported-lexeme retention. No actual historical documents were passed through the new helper, and no full suite or empirical calculation ran in this task.

## Announcement auction-date follow-up

Before any historical helper application or checkpoint, root identified that the announcement PDF's labeled `Auction Date` had not been compared with the selected/XML date. A thirteenth generated test first demonstrated this exact single-field defect: the correct paired fixture succeeded, then changing only `Auction Date January 13, 2017` to `Auction Date January 12, 2017` failed the expected rejection with `AssertionError: ValueError not raised` (one test, 0.002 seconds). This RED preceded the correction.

The announcement pair now requires one valid labeled PDF auction date matching the selected/XML event. Result PDFs retain the separate release-header rule and do not require an auction-date body label. After the correction, all 13 focused generated tests passed in 0.040 seconds and Ruff remained clean. No real source documents were used in this follow-up.
