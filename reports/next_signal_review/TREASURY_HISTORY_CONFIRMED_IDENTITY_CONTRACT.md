# Original announced identity on confirmed substitutions

The first adapted history pass completed successfully with 1,115 reconciled events, 18 explicit unsupported OriginalCUSIP contexts, four original unresolved sources and one excluded contingency test. The remaining 18 cases reached a previously masked check: their result OriginalCUSIP names the originally announced security, not the final substituted security. This pass preserves that result and its checkpoint.

The separate confirmed-identity adapter retains the complete tested history reconciliation. It recognizes a nonempty OriginalCUSIP only when it exactly equals the announcement's own CUSIP, the final CUSIP differs, and the dated final confirmation passes every original identity, offered-term, original-date, tenor and series check. The final announcement section must also explicitly name the same AnnouncedCUSIP. OriginalCUSIP cannot override a missing AnnouncedCUSIP, a missing or contradictory confirmation, an unchanged final identity, or an unknown third identifier. Source bytes are neither rewritten nor stripped of metadata. The returned record preserves the OriginalCUSIP lexeme and its narrow interpretation.

Eighteen prewritten synthetic tests cover the thirteen prior integrated invariants and five additional substitution cases. The exact eighteen-source review remains separate evidence and includes the existing canonical offered-term distribution: nine two-year, eight five-year and one three-year note. No canonical correction is required.

The new output is `TREASURY_HISTORY_IDENTITY_V3.json`, bound to `TREASURY_HISTORY_IDENTITY_CHECKPOINT_V4.json`. Every selected event remains in scope. No financial decoding, source admission or predictive comparison follows from identity reconciliation alone.
