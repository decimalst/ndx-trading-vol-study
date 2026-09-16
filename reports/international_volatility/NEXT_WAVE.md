# Next question: can index overnight returns be predicted incrementally?

This is a research handoff, not a registered experiment or an empirical result.
No overnight signed-return forecasts have been run. The first two waves are
complete and must retain their protocol, source and result hashes.

The next distinct target is QQQ close-to-next-open movement. Daytime and
overnight returns need not have the same predictors. The detailed feasibility
review is [OVERNIGHT_OPPORTUNITIES.md](../iterative_signal_search/OVERNIGHT_OPPORTUNITIES.md).

Before registering a third wave:

1. Decide and label the return object precisely. Existing adjusted-close and
   raw open/close data identify an adjusted overnight-return proxy. They do
   not alone prove exact cash returns including dividends, splits and funding.
   Any execution claim needs separate corporate-action records and unit checks.
2. Make the entry feasible in time. For entry at close t, use market features
   through close t−1. Do not let the final close t price enter its own entry
   decision. Keep calendar inputs limited to information known before entry;
   an unscheduled next-session closure is not known merely because the dataset
   later records the next actual opening date.
3. Include a strong benchmark with separate own overnight and daytime trailing
   means, volatility history and delayed implied volatility, plus a historical
   mean control on identical eligible rows. A simple overnight/day split is
   already contained in that benchmark and cannot be relabeled an addition.
4. Fix a small candidate set before scores. Mechanism-based possibilities are
   signed cross-asset returns, signed abnormal volume, and short-term implied
   volatility shape. Inspect their coverage and provenance before selecting
   fixed lags and dates; do not choose variants from returns or forecast scores.
5. Write timing, corporate-action, target-maturity, common-sample and independent
   reconstruction tests before fitting. Register every arm, control and phase,
   with a third-wave error allocation of .05/(3*4), and retain all 72 earlier
   enumerated comparisons. Historical reuse remains exploratory.

Do not refit or retune the international models on their favorable later-period
subset and call that independent confirmation. Their 21-session latent estimate
is uncertain and did not meet the declared gate. A future revision would be a
new registered trial with its own controls and explicit historical exposure.
