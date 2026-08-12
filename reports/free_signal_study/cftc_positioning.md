# CFTC Nasdaq positioning diagnostic

This is a post-program, target-specific diagnostic on already inspected QQQ history. It scores one origin per conservatively delayed weekly release and never repeats a weekly value across daily rows.

- Origins: 591 (2013-01-03 through 2025-10-10).
- Base event rate: 16.75% (99 positives).

| model | AUC | top-decile lift | top-decile event rate |
|---|---:|---:|---:|
| RV-history benchmark | 0.8308 | 3.482x | 58.33% |
| + leveraged-money net/OI | 0.8304 | 3.681x | 61.67% |

Registered two-metric gate: **FAIL**. Delta AUC -0.0004; delta lift +0.199x.

Known federal-shutdown and 2023 ION-backlog report dates are excluded. Ordinary reports are delayed ten calendar days and then mapped to the first QQQ session on or after that date, deliberately sacrificing timeliness to avoid treating Tuesday positions as public Tuesday data. The May 2023 e-micro inclusion is reported as a fixed sensitivity, not fitted away.
