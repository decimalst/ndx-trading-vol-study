# What the 30-day implied level absorbs

**Evidence class: diagnostic measurement.** The original NDX clean window is not read.

The reported percentage is `1 - Wald with VXN / Wald without VXN`. It is a
descriptive attenuation of a joint Wald statistic—not a literal information share,
not bounded to [0, 100%], and not a forecast or trading result.

| regularity | common n | Wald without VXN | Wald with VXN | attenuation | partial R² without / with |
|---|---:|---:|---:|---:|---:|
| leverage | 2463 | 103.19 (p=3.21e-22) | 62.63 (p=1.61e-13) | +39.3% | 0.0642 / 0.0371 |
| weekday | 2463 | 12.55 (p=0.0137) | 11.39 (p=0.0225) | +9.2% | 0.0043 / 0.0041 |
| earnings | 1479 | 39.68 (p=2.99e-10) | 26.42 (p=2.74e-07) | +33.4% | 0.0156 / 0.0145 |
| scheduled_macro | 2463 | 39.83 (p=1.16e-08) | 37.80 (p=3.11e-08) | +5.1% | 0.0147 / 0.0140 |
| post_fomc | 2463 | 14.99 (p=0.000108) | 17.76 (p=2.51e-05) | -18.4% | 0.0056 / 0.0073 |

The earnings row starts only when an SEC-accepted quarterly top-25
snapshot exists. Realized announcement labels are used after the fact for
measurement; they are not claimed as a versioned ex-ante calendar.

All other rows use 2016-01-04 through 2025-10-17. Same-origin VXN is valid
because this extension fixes the decision time after the 16:15 ET Cboe close.
