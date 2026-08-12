# SPX volatility-risk-premium term structure

**Evidence class: descriptive SPX measurement.** This matches official Cboe
9-, 30-, and 93-calendar-day implied-volatility indices to subsequent SPX
close-to-close realized volatility on one common-origin sample.

| horizon | common n | implied vol | realized vol | premium (95% block CI) | premium after negative 5d return | otherwise | difference (95% block CI) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9d | 3656 | 17.58 | 13.98 | +3.61 [+3.05, +4.10] | +3.43 | +3.72 | -0.29 [-1.04, +0.41] |
| 30d | 3656 | 18.18 | 14.71 | +3.47 [+2.50, +4.29] | +3.63 | +3.36 | +0.27 [-0.57, +1.03] |
| 93d | 3656 | 20.04 | 15.32 | +4.72 [+3.53, +5.79] | +5.33 | +4.31 | +1.02 [+0.11, +2.10] |

The premium is an index-level implied-minus-realized volatility difference,
not an option P&L. It excludes skew, jumps, delta hedging, transaction costs,
margin, and the path dependence that determines whether it can be harvested.
