# What the 30-day implied level absorbs

**Evidence class: diagnostic measurement.** The original NDX clean window is not read.

The reported percentage is `1 - Wald with VXN / Wald without VXN`. It is a
descriptive attenuation of a joint Wald statistic—not a literal information share,
not bounded to [0, 100%], and not a forecast or trading result.

| regularity | common n | Wald without VXN | Wald with VXN | attenuation | partial R² without / with |
|---|---:|---:|---:|---:|---:|
| leverage | 2462 | 123.49 (p=1.37e-26) | 72.03 (p=1.57e-15) | +41.7% | 0.0640 / 0.0369 |
| weekday | 2462 | 11.49 (p=0.0216) | 10.56 (p=0.0319) | +8.0% | 0.0043 / 0.0040 |
| earnings | 1479 | 39.88 (p=2.7e-10) | 28.15 (p=1.12e-07) | +29.4% | 0.0156 / 0.0145 |
| scheduled_macro | 2462 | 39.94 (p=1.1e-08) | 37.35 (p=3.88e-08) | +6.5% | 0.0146 / 0.0140 |
| post_fomc | 2462 | 15.01 (p=0.000107) | 17.58 (p=2.76e-05) | -17.1% | 0.0056 / 0.0073 |

The earnings row starts only when an SEC-accepted quarterly top-25
snapshot exists. Realized announcement labels are used after the fact for
measurement; they are not claimed as a versioned ex-ante calendar.

All other rows use 2016-01-04 through 2025-10-17. Same-origin VXN is valid
because this extension fixes the decision time after the 16:15 ET Cboe close.
