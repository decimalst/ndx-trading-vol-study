# SKEW-conditioned carry mechanism diagnostic

This is a post-hoc mechanism test on an already-inspected diagnostic window. A pass is not out-of-sample evidence.

It preserves the original carry implementation, including its full-window median richness cutoff and same-date published VXN daily close. The SKEW input is safely lagged, but the inherited choices mean this is not a leakage-free strategy backtest. A future version must freeze +0.386812814 and use lagged Cboe data or timestamped pre-close quotes.

- Window: 2016-01-04 through 2025-10-17.
- Original richness threshold, unchanged: +0.386813.
- SKEW: one-session delayed; high regime is above the trailing 252-session 80th percentile estimated through the prior aligned value.
- Non-overlapping phase samples evaluated: 21/21.

| metric, average across phases | unconditional | richness-only | SKEW-repaired |
|---|---:|---:|---:|
| trades | 117.3 | 58.6 | 37.1 |
| mean P&L, vol points | +1.714 | +2.381 | +2.800 |
| 5% CVaR | -33.299 | -22.711 | -10.897 |
| worst trade | -105.336 | -49.629 | -14.317 |
| max drawdown | 118.157 | 55.127 | 19.179 |

Participation retained: **63.3%** of richness-only trades.

## Known adverse origins

| origin | lagged SKEW | trailing threshold | richness eligible | repaired trade | P&L |
|---|---:|---:|---:|---:|---:|
| 2020-02-18 | 132.67 | 129.55 | True | False | -185.63 |
| 2020-02-19 | 137.66 | 129.64 | True | False | -196.69 |
| 2020-02-21 | 137.12 | 129.78 | True | False | -158.17 |

## Frozen mechanism criterion

- known adverse rejected: **True**
- cvar not worse: **True**
- drawdown not worse: **True**
- mean positive: **True**
- participation at least 70pct: **False**

Mechanism verdict: **FAIL**.

Even a pass requires genuinely future transition data. No p-value is reported because the rule was motivated by the same 2020 event it is asked to repair.

## Descriptive interpretation after the frozen verdict

The failure is the participation constraint, not the tail hypothesis. The veto retained 63.3% of richness-only trades versus the registered 70% minimum. It removed all three known adverse entries, and the average phase CVaR, worst trade, and drawdown all improved materially, but those observations cannot override the frozen verdict.

The attrition is not random: 36.7% of richness-eligible origins were also in a high-SKEW regime, compared with 29.7% of all origins with a valid gate. The richness rule fired on 61.8% of high-SKEW days versus 45.1% of lower-SKEW days. That is direct evidence of the suspected overlap: backward-looking "richness" is especially likely to call variance expensive while the option wings are already charging for tail risk.

Lagged SKEW is not merely another copy of the ATM level: its Pearson correlation with equally lagged VXN is -0.270 (Spearman -0.208) over the diagnostic frame. The relationship is modest and inverse, consistent with SKEW being a surface-shape measure whose signal is often largest while ATM volatility is still low.

Among overlapping origins (descriptive only), richness trades returned +1.663 vol points in high-SKEW regimes versus +2.818 outside them. 11 high-SKEW richness trades lost more than 20 vol points, versus 3 in lower-SKEW regimes. These counts explain the tail improvement, but overlap prevents them from serving as independent inference.
