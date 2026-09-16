# Treasury matched market context

This isolated component builds four TLT controls and four weekday indicators,
plus the exact prior full-session Treasury cutoff date. It reads no historical
data itself. Nine generated contract tests were written and run against the
missing module before its implementation; the RED log is preserved separately.

Source and reference date envelopes must be unique, ascending, normalized naive
dates at or before October20,2025. The input cross-asset frame retains the exact
existing five-column schema. Only TLT numeric values are examined. Observations
before January1,2010 are unavailable for this new source history. On origin t,
the return is log P[t-1] minus log P[t-2]; its square and logs of strict five- and
22-session arithmetic means form the remaining three controls. Missing dates
stay on the full reference calendar. Exact zero returns are valid; an entirely
zero variance window has unknown log variance. No epsilon, filling, compressed
window or alternative asset substitutes for missing information.

Tuesday through Friday are entry-date indicators with Monday as reference.
The cutoff is the immediately preceding observed QQQ session when it is at or
after the source floor, otherwise unknown. The source snapshot and provenance
must be authenticated by the eventual runner. This code reuses the frozen
commodity module's calendar, schema, strict-square and strict-log-mean helpers;
it is not an independent implementation of those helpers.

The generated tests cover an independent loop oracle, one-session timing,
future-value poisoning, source-floor warmup, missing-date windows, zero returns,
weekday coding, opaque unused cross-asset values, source/calendar/numeric
validation, timestamp-unit transport, and input preservation. No Treasury
feature panel, market cohort, fit or predictive score is created by these tests.
