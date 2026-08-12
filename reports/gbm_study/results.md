# GBM functional-form study

**Evidence class: internally split diagnostic study; sealed NDX clean origins were not read.**

Timing qualification: both models use the same-session published Cboe VXN close, so the functional-form comparison is internally fair. However, that close may contain 16:00–16:15 ET information after the repository's standing 16:00 origin. Treat this frozen run as timing-ambiguous for a strict 16:00 forecast; the separately registered lagged-VXN sensitivity determines whether the substantive result survives a fully known-at-origin input.

Protocol SHA-256: `64700689a7682818cbcc7b446d432bf59da3fab7c908a8c3149f13117561ef05`.

## Same-information-set forecast comparison

| split | n | HAR-IV QLIKE | GBM QLIKE | improvement | DM p | block p | 95% block interval | win rate | equivalent within 3% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| all_diagnostic | 2463 | 0.340085 | 0.354742 | -4.310% | 0.00185 | 0.07678 | [+0.000937, +0.035246] | 51.8% | no |
| discovery | 1006 | 0.335742 | 0.348428 | -3.779% | 0.007086 | 0.003999 | [+0.003124, +0.020723] | 47.6% | no |
| confirmation | 1457 | 0.343083 | 0.359102 | -4.669% | 0.02748 | 0.2659 | [-0.003847, +0.049626] | 54.7% | no |

Frozen functional-form verdict: **INCONCLUSIVE**.

Negative mean differences favor the candidate. The block p-value and interval use paired 21-session moving blocks; DM is h=1 and is reported for continuity with the existing harness.

## Exact interaction fallback

SHAP was not installed before the protocol freeze. The registered fallback exactly double-centers each fitted two-feature partial-dependence surface on the discovery quantile grid.

| pair | interaction score | selected |
|---|---:|:---:|
| lrv_d × lrv_w | 0.006023 | no |
| lrv_d × lrv_m | 0.012531 | no |
| lrv_d × liv | 0.001700 | no |
| lrv_w × lrv_m | 0.045414 | yes |
| lrv_w × liv | 0.030968 | no |
| lrv_m × liv | 0.000589 | no |

Locked term: `max(+1·(lrv_w − -8.219338)/1.195818, 0) × max(+1·(lrv_m − -8.233019)/1.058726, 0)`.

The pair, location, directions, thresholds, and IQR scales were written to the interaction lock before any confirmation forecast was computed.

## Interpretable-term confirmation

| n | HAR-IV QLIKE | HAR-IV + locked term QLIKE | improvement | DM p | block p | 95% block interval | win rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1457 | 0.343083 | 0.343915 | -0.243% | 0.04923 | 0.2284 | [-0.000109, +0.002553] | 57.4% |

Frozen interpretable-term verdict: **DOES_NOT_ADD**.

## Scope

This closes only the fixed HAR-IV functional-form question on the already-open diagnostic history. It does not test additional signals, optimize a GBM, or constitute evidence from the sealed clean phase.
