# Finer volatility measurements and index returns: completed first wave

No new candidate qualified. All 12 registered comparisons were evaluated and
independently verified. The larger search now contains 66 enumerated contrasts;
older repository exploration is inventoried separately and is not hidden by
that count.

The SPX experiment used five-minute archive measurements that earlier studies
had not used as predictors. Its baseline already includes current daily
volatility and return asymmetry, plus conservatively delayed implied volatility
and high-frequency volatility history. The archival semivariance partition
reduced one-session forecast loss by 0.56% in development and 0.86% in evaluation.
At 21 sessions it improved evaluation loss by 2.65% but worsened development
loss by 1.58%. None passed the fixed effect, stability and multiplicity gates.
The partition's direction is unresolved, so it is not described as downside
variance.

The QQQ experiment predicted the next session's open-to-close return. Splitting
overnight and daytime behavior helped development loss but worsened evaluation
loss by 0.43% versus the baseline. Signed cross-asset returns and calendar
features also failed the required comparisons against both the market model
and the historical mean. The accompanying cost scenarios are descriptive;
their positive individual returns do not establish incremental trading value.

The data work recovered pre-2009 SPX prices in a separate file. All 271 overlap
sessions matched the existing source exactly. Oxford-Man dates retain their
stated local trading date; archive features and target availability have an
explicit extra-session delay. Features, transformations and model training use
only eligible historical information. The experiment never reads the protected
later evaluation window for model inputs or targets.

The independent implementation reconstructed **30,040 forecasts across 400
monthly fits**, every phase score, 72 bootstrap calculations, all multiplicity
adjustments, all 36 descriptive cost scenarios, and all 78 trial-ledger events.
It passed on the first attempt without changing frozen code or tolerances.
The final repository suite passed **594 tests**. The old cosmetic import-order
lint finding in an already frozen test remains unchanged; the new experiment's
source and tests pass scoped lint.

- [Complete estimates and cost scenarios](results.md)
- [Scientific comparison figure](comparison.png) and [exportable PDF](comparison.pdf)
- [Independent verification](verification.json)
- [Fixed protocol](../../iterative_signal_search.yaml), [source/code manifest](manifest.json), and [append-only trial ledger](trial_ledger.jsonl)
- [Prior search inventory](PRIOR_SEARCH_INVENTORY.md)
- [Pre-run contract checks](pre_run_checks.txt) and [final repository tests](final_tests.txt)

Historical reuse remains exploratory. No feature was promoted, no failed
comparison was removed, and no threshold was relaxed after the scores. The next
fixed wave asks whether overseas market volatility supplies information missing
from the SPX-only state. It is a separate registered experiment.
