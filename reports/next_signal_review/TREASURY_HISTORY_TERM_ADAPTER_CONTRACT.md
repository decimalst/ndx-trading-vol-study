# Prospective Treasury term-token adapter

The standalone `parse_offered_term(term: str, *, announcement: bool)` returns exactly `years`, `months`, `kind`, `canonical_term`, `coupon_token_present`, and `split_month_token`. It receives one caller-preserved raw field value, never a whole document. This component performs no identity reconciliation, source admission or financial calculation.

Recognize exact case and ASCII single spaces in `N-Year [M-Month] Note/Bond`, with positive years through 30 and optional months from 0 through 11. Leading-zero integers, extra whitespace, Unicode alternatives, signs and other number formats are invalid. Omitted months return zero; an explicitly present `0-Month` remains present in the canonical term. Kind is returned as `NOTE` or `BOND`.

Announcements additionally permit one coupon token in the sole position between the complete term descriptor and `Note`/`Bond`. Tokens are unsigned integer-percent or an unsigned whole part followed by one observed mixed-fraction suffix: `1/2`, `1/4`, `3/4`, `1/8`, `3/8`, `5/8`, or `7/8`, then `%`. Zero is permitted as the whole part; ambiguous leading zeroes are not. The observed finite suffix grammar rejects zero, improper, unreduced and unsupported denominators without converting coupon digits. No coupon value is returned or computed. Result terms reject all coupon tokens.

The exact observed `Mont h` unit spelling is recognized only inside an announcement's optional month descriptor. It maps to `Month` in the returned canonical string and sets `split_month_token`; no input string is rewritten or stripped. Other within-word whitespace remains invalid. Results reject the split spelling. The canonical output removes only the coupon token, if present; this derived descriptor does not overwrite source text.

Eight synthetic test methods are prewritten before implementation: ordinary output across roles; all permitted coupon shapes; local split-word handling; bounds and canonical integers; unsupported coupon grammar; case/spacing/kind/position and extra-token failures; strict input/role types; and an integer-decoder spy proving only year/month descriptors are converted. Examples are invented, and no historical term panel is loaded or reapplied. Genuine missing-module RED precedes implementation, followed by focused GREEN and lint evidence.

All frozen parsers, source documents, review findings and prior reconciliation failures remain unchanged. Root owns integration, raw-term retention and any later checkpoint/application. The limited coupon grammar is prospective; an unrecognized later shape requires separate evidence and tests rather than broad token removal.

The genuine missing-module RED exited 1. The first implementation passed all eight generated tests in 0.001 seconds, and scoped Ruff passed. No post-implementation expectation changes were needed. RED and GREEN outputs are retained in the corresponding new adapter logs.
