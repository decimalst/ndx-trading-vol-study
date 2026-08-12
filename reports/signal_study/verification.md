# Independent signal-study verification

This recomputation imports neither `src.signal_study` nor `src.metrics`. All recorded confirmation metrics matched at 1e-12 absolute/relative tolerance.

- Protocol SHA-256 matched the discovery lock.
- Origins: 2022-01-03 through 2025-10-17 (n=952); last realized target: 2025-10-20.
- QLIKE: term slope 0.325970; safe baseline 0.335776; improvement +2.920%.
- Paired wins: 519/952 (0.545).
- DM: -1.639, p=0.1016.
- 90% interval: coverage 0.916, p_uc=0.0922, p_ind=0.9099.
- Post-hoc conservative loss-differential HAC checks: p=0.0893 at 5 lags; p=0.1427 at 22 lags. These do not alter the pre-registered verdict.

## Year-by-year description

| year | n | term QLIKE | baseline QLIKE | improvement | win rate |
|---:|---:|---:|---:|---:|---:|
| 2022 | 251 | 0.287217 | 0.303271 | +5.294% | 0.566 |
| 2023 | 250 | 0.231784 | 0.249201 | +6.989% | 0.520 |
| 2024 | 252 | 0.372707 | 0.373357 | +0.174% | 0.516 |
| 2025 | 199 | 0.433988 | 0.437944 | +0.903% | 0.588 |

Year rows are post-hoc stability descriptions, not additional tests.
