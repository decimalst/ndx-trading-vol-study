# Single-name earnings residual after own implied volatility

**Evidence class: cross-sectional mechanism diagnostic.** Forecasts are fixed
without earnings inputs. Realized announcement labels are attached afterward.
An issuer-session is retained only while the latest SEC-accepted quarterly QQQ
snapshot ranks that issuer in the top 25.

| asset / own IV | eligible n | event n | event − other log residual | variance ratio effect | 95% block CI (log) |
|---|---:|---:|---:|---:|---:|
| AAPL / VXAPL | 1478 | 23 | +1.715 | +455.7% | [+1.431, +2.033] |
| AMZN / VXAZN | 1478 | 23 | +2.139 | +748.9% | [+1.652, +2.597] |
| GOOG / VXGOG | 1478 | 23 | +2.295 | +892.2% | [+1.954, +2.628] |
| IBM / VXIBM | 0 | 0 | n/a | n/a | n/a |

Equal-asset pooled effect across AAPL, AMZN, GOOG: **+2.049 log variance (+676.4%)** with 95% block interval [+1.829, +2.272].

This asks whether earnings variance remains unusually large after each name's
own 30-day Cboe implied level. It does not claim an executable earnings-date
forecast: the historical announcement archive is not versioned as-of each origin.
