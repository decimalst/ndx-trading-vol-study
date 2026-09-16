# Registered design before empirical fitting

This wave asks whether known monthly release plans improve forecasts of the
size of the QQQ overnight move. CPI, payroll and the scheduled Fed meeting date
are separate additions. The Fed arm is a timing comparison: a useful effect
would not establish that a release occurred before the opening.

The target is the square of the vendor-adjusted overnight **log** return proxy.
It estimates a conditional second moment. Zero targets remain valid and retain
their observations. The score is `log(h) + q/h`, whose optimum is the conditional
mean of q. Its raw value can be negative, so the gate uses an absolute paired
score difference, never a percentage improvement. This is the QLIKE form
discussed in [Patton's loss-function analysis](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf).
No cash-profit or integrated-variance interpretation is asserted.

Two controls make each addition accountable: a historical-mean forecast on
identical training observations and a market-history model containing separate
overnight/daytime realized history, daily range variance, VXN/VIX, weekdays and
nominal elapsed hours. Every market input is one observed session old; every
training target must already be available at the preceding session's close.
The forecast is formed before the entry closing auction. Month-start refitting
depends on available features, not whether the first future label happens to
be present in the final dataset.

The calendar measures the **next plan printed in the preceding monthly
release**, not the subsequently realized release calendar. Original cancellations
and later Fed meeting expansions do not rewrite these features. Documents with
explicit later reissue uncertainty or missing timestamp information remain
unusable under the source contracts. Current official archive extracts are
hashed evidence, not historical web snapshots. The precise source limitations
are preserved in the separate source audits.

The final source review applies the same explicit reissue exclusion to both
BLS series. The first payroll ledger omitted this check. It is preserved
unchanged, and a separate corrected admission ledger excludes affected archives
before fitting. Unknown coverage clusters in parts of 2020; the shared scoring
sample therefore cannot represent every market episode in the declared date
range. Calendar completeness is a property of the surviving documentary
evidence, not a claim that missing observations are randomly distributed.

The feature window is deliberately a civil-time convention: entry date16:00
Eastern through the next Monday–Friday date09:30 Eastern, with weekends skipped
and elapsed hours calculated in UTC. It does not claim to reconstruct exchange
holiday or early-close announcements. A Friday before a Monday holiday still
points to Monday and may miss a Tuesday release inside the actual overnight
holding interval. This fixed definition was chosen before any event-outcome
comparison. Future observed opening dates appear only in targets and maturity.

Source eligibility is stricter than the stated header time: the document's
publication date must precede the previous observed market-session date.
Every civil month touched by the window needs an eligible explicit CPI/payroll
plan, and each Fed year needs all eight original annual meeting dates. Unknown
source coverage stays missing for every model; it is never converted into a
non-event. Bootstrap blocks count retained paired observations in time order;
they do not invent losses on missing-calendar dates.

The numerical family remains six comparisons against two controls, with a
0.005 absolute improvement gate in both development and evaluation, stability
in both evaluation subperiods, an adjustment-event sensitivity, conservative
bootstrap/HAC checks, Holm over six at0.0025 and separately over all84 enumerated
contrasts at0.05. There are49,999 bootstrap draws for each of blocks21/63/126.
Any model failure retains the whole six-comparison family as unevaluable.

Prewritten synthetic tests cover derivatives and optimizer convergence, target
unit invariance, zeros, matured labels, source publication timing, unknown
coverage, annual-calendar completeness, DST, source revisions, future-session
mutation, and the complete inference family. An independent implementation
checks both source extraction and resulting numerical forecasts before results
are interpreted. The protocol and implementation become immutable in the
pre-fit manifest. No result existed when this design was written.

- [Full specification](../../macro_overnight.yaml)
- [Payroll sources](NFP_SOURCE_AUDIT.md)
- [CPI sources](CPI_SOURCE_AUDIT.md)
- [Fed and civil-time source contracts](SESSION_AND_FOMC_SOURCES.md)
- [Independent source verification](SOURCE_INDEPENDENT_VERIFICATION.md)
