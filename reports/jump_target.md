# SPX jump-target diagnostic

The Oxford-Man SPX target was not used anywhere else in this repository. The surface hypothesis was motivated by prior NDX work, so this is an external mechanism confirmation rather than a pristine strategy test.

- Source coverage: 2000-01-03 through 2018-06-27.
- Scored confirmation origins: 1001.
- Five-session material-jump event rate: 38.9%.
- BPV exceeded RV on 16.9% of raw days; continuous variation was conservatively truncated to RV on those days.

| model | phase-average Brier | phase-average log loss |
|---|---:|---:|
| history | 0.214213 | 0.616525 |
| atm | 0.213344 | 0.614335 |
| surface | 0.217646 | 0.625756 |

- Surface Brier below ATM: **False**
- Surface log loss below ATM: **False**

Frozen verdict: **FAIL**.

A pass says surface shape transfers better than the ATM level to a jump target in SPX. It does not establish an NDX dispersion trade, and it does not repair the source's lack of a formal BNS significance statistic.
