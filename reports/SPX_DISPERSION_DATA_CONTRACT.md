# SPX correlation-premium and dispersion data contract

Status: specified, not scored. This moves the asset to the instruments rather
than forcing SPX instruments onto QQQ.

## Matched universe

Cboe computes COR1M from SPX option implied variance and the option-implied
variances of a tracking portfolio containing the 50 highest-weight SPX
components. The selection for a business day uses prior-close float-adjusted
market capitalizations. DSPX and VIXEQ are also SPX constituent instruments.
They may be combined in one SPX study; none may be called an NDX measurement.

The local implied series are usable as inputs: COR1M has 5,183 observations
from 2006-01-03 through 2026-08-11, and VIXEQ has 3,054 non-missing observations.
DSPX has not yet been acquired. Cboe reports DSPX history beginning 2014-06-19.

Official methodology: [COR1M tracking-basket construction](https://cdn.cboe.com/resources/indices/documents/Implied_Correlation-WhitePaper-v1.0.5.pdf)
and [DSPX methodology](https://cdn.cboe.com/resources/indices/documents/methodology-the-dispersion-index.pdf).

## Required realized side

Before scoring, acquire and version:

1. the exact top-50 tracking portfolio and float-adjusted weights effective at
   every origin, including the five-name replacement pool and rebalance events;
2. stable security identifiers and corporate-action/delisting-safe adjusted
   returns for every selected constituent;
3. the SPX return series on the same close convention; and
4. a documented session calendar mapping each 30-calendar-day implied horizon
   to its completed realized-return window.

Do not substitute current constituents, all 500 constituents, QQQ holdings, or
renormalized weights unless the Cboe methodology calls for that transformation.

## Frozen target definitions before acquisition

At origin `t`, hold the effective tracking basket and weights fixed through the
forward 30-calendar-day measurement window. Following the Cboe formula, compute
realized average correlation from forward SPX variance, constituent variances,
and the weighted cross-volatility denominator using the exact published weight
convention. The primary target is:

`correlation_premium_t = COR1M_t / 100 - realized_correlation_(t,t+30d)`.

The DSPX companion target is implied dispersion minus realized dispersion using
the same origin basket, horizon, return convention, and methodology. VIXEQ is a
diagnostic decomposition term, not a substitute target.

## Mandatory pre-run tests

Before downloading returns or scoring:

- historical basket membership and weights must reproduce published Cboe
  snapshots on sampled dates;
- every origin must use information published no later than that origin;
- delisted/replaced securities must remain in their origin basket through the
  target calculation with documented corporate-action handling;
- weights and formula components must reconcile to the methodology;
- all 30-day targets must be complete, with no forward-filled returns; and
- COR1M/DSPX/VIXEQ dates and availability lags must be explicit.

Until those tests can be satisfied, this remains data-blocked. The economic
question is valid and the SPX universe mismatch is resolved by design.
