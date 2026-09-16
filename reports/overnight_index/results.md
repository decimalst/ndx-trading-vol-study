# Overnight index returns: third fixed wave

The target is a vendor-adjusted overnight-return proxy. All market inputs precede the entry session.
Six new comparisons remain in the cumulative family of 78; each candidate must pass both controls.

| Candidate vs control | Development gain | Evaluation gain | Wave Holm p | Cumulative Holm p | Gate |
|---|---:|---:|---:|---:|---|
| cross_signed vs baseline | -0.358% | -0.194% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| cross_signed vs mean | -1.705% | -0.912% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| volume_pressure vs baseline | +0.014% | -0.188% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| volume_pressure vs mean | -1.328% | -0.907% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| iv_shape vs baseline | -0.097% | +0.049% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| iv_shape vs mean | -1.440% | -0.668% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |

Passing exploratory leads: []

## Measurement sensitivity

The following excludes flagged adjustment-factor changes after forecasts are fixed. It is a retrospective label audit, not a tradable filter.

| Candidate vs control | Development unflagged gain | Evaluation unflagged gain |
|---|---:|---:|
| cross_signed vs baseline | -0.393% (n=989) | -0.176% (n=1434) |
| cross_signed vs mean | -1.696% (n=989) | -0.906% (n=1434) |
| volume_pressure vs baseline | +0.011% (n=989) | -0.202% (n=1434) |
| volume_pressure vs mean | -1.287% (n=989) | -0.932% (n=1434) |
| iv_shape vs baseline | -0.095% (n=989) | +0.063% (n=1434) |
| iv_shape vs mean | -1.394% (n=989) | -0.665% (n=1434) |

All numerical intervals, MDE, years, stability slices and adjustment sensitivities are in metrics.json.
The adjusted target does not establish cash profit, corporate-action receivable accounting, or execution at the auction prices.
Historical reuse and retrospectively acquired sources remain exploratory; no frozen criterion is changed after scores.
