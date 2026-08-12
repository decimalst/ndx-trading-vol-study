# Where VXN contributes along the horizon curve

**Evidence class: diagnostic forecast-shape measurement.** Targets are built
only from returns through 2025-10-17; the sealed clean phase is never read.

| h sessions | n | HAR QLIKE | HAR+VXN QLIKE | improvement | DM p | win rate | partial R² |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2462 | 0.3701 | 0.3400 | +8.14% | 2.345e-07 | 54.2% | 0.0903 |
| 5 | 2458 | 0.2213 | 0.1960 | +11.45% | 0.0008762 | 54.6% | 0.1512 |
| 10 | 2453 | 0.2271 | 0.2051 | +9.67% | 0.02157 | 53.9% | 0.1493 |
| 21 | 2442 | 0.2758 | 0.2567 | +6.91% | 0.07991 | 52.2% | 0.1273 |
| 42 | 2421 | 0.3229 | 0.3144 | +2.64% | 0.3315 | 50.5% | 0.0828 |
| 63 | 2400 | 0.3485 | 0.3358 | +3.64% | 0.3032 | 51.6% | 0.0774 |

Every point is reported; no horizon is selected. OOS forecasts are expanding
direct regressions with exact Duan smearing. Training labels enter only after all
h future sessions are complete, and DM uses h-1 overlap lags.
