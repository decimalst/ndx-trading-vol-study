# SPX regime-conditional VIX9D/VIX replication

**Registered verdict: FAIL.** This is a different asset and a
2014-2015 score window that ends before the inspected 2016-2025 NDX study.
Both Cboe closes are delayed one full session at the 16:00 ET origin.

| model | n | QLIKE | improvement vs baseline | DM p | paired win rate |
|---|---:|---:|---:|---:|---:|
| unconditional_slope | 503 | 0.2955 | -0.16% | 0.8626 | 54.1% |
| dislocation_only | 503 | 0.2968 | -0.60% | 0.03253 | 48.1% |

| origin state | n | baseline QLIKE | dislocation QLIKE | improvement |
|---|---:|---:|---:|---:|
| inverted | 141 | 0.3034 | 0.3084 | -1.66% |
| not_inverted | 362 | 0.2918 | 0.2922 | -0.17% |

The primary rule required the dislocation-only form to beat baseline and
the unconditional form, DM p<0.05, and a paired win rate above 50%.
