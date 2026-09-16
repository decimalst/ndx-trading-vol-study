# International volatility: completed second wave

No international candidate passed the fixed criteria. The strongest later-period
estimate was the latent model at 21 sessions: **7.36% lower QLIKE loss** during
2014–2017. Its development gain was only **0.28%** during 2010–2013. The later
estimate itself remains uncertain: conservative unadjusted p=0.199, with a
nominal gain-interval envelope of approximately **−3.4% to +20.4%**. It therefore
does not establish a new predictive signal, even before cumulative correction.

| Candidate | Horizon | Development gain | Evaluation gain |
|---|---:|---:|---:|
| Regional summaries | 1 | +0.03% | +0.32% |
| Regional summaries | 5 | −0.14% | −0.05% |
| Regional summaries | 21 | −2.43% | −0.18% |
| Three latent factors | 1 | −0.23% | +0.63% |
| Three latent factors | 5 | −0.91% | +1.74% |
| Three latent factors | 21 | +0.28% | +7.36% |

The new inputs come from six fixed foreign indexes: Japan, Hong Kong and Korea;
the UK, Germany and France. Each market contributes its one-, five- and
22-observation log mean realized variance. Windows use the market's own archived
observation dates before alignment to the previous SPX session. A fixed
freshness limit rejects overly stale states; no missing member is dropped from
its region. Exact foreign dates, ages and window completeness are retained.

The regional model uses six equal-weight summaries. The latent model uses three
principal components fitted on eligible training rows at each monthly refit.
The baseline includes current SPX daily volatility and leverage, lagged SPX
high-frequency history and lagged VIX. All three models use the same training
rows within each fit and the same 1,967 scored origins across all horizons.
Targets mature before training; scaling, PCA and smearing are training-only.

The independent implementation reconstructed **17,703 forecasts, 282 monthly
fits and PCA subspaces, and all 27,312 foreign availability records**. It
recomputed every phase comparison, 36 bootstrap runs, all six new tests and
the full 72-comparison cumulative correction. Verification passed without
changing source, forecasts or numerical tolerances. The full pre-run repository
suite passed **625 tests**; all new experiment source and tests pass scoped lint.

Across these two new waves, **18 comparisons and 47,743 forecasts** were run
and independently reconstructed. Neither wave produced a qualifying signal.
The earlier paper-inspired modeling and memory study also found no alternative
that qualified as improving the established benchmark. These results describe
the declared configurations; they do not prove that every possible signal is
absent. Archived publication vintages and reused historical data remain explicit
limits on interpretation.

- [All six results](results.md), [full numerical evidence](metrics.json), [figure](comparison.png), and [PDF](comparison.pdf)
- [Independent reconstruction](verification.json), [pre-run tests](full_pre_run_tests.txt), [manifest](manifest.json), and [trial ledger](trial_ledger.jsonl)
- [Fixed second-wave protocol](../../international_volatility.yaml)
- [Completed first wave: finer volatility measures and QQQ daytime returns](../iterative_signal_search/SUMMARY.md)
- [Completed modeling, latent-memory and supplied-paper experiments](../model_memory_study/SUMMARY.md)
- [International source feasibility and methodology](../iterative_signal_search/INTERNATIONAL_OPPORTUNITIES.md)
- [Next research question and constraints](NEXT_WAVE.md)

No old threshold, window or result was changed to create a positive finding.
The broader search remains active; no candidate is promoted by this report.
