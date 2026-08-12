# Hugging Face option-flow diagnostic

**Evidence class: post-program exploratory diagnostic.** The archive is activity-only,
identifies its upstream bars only as various unnamed sources, and uses an `other` license.
It cannot measure NBBO, spreads, open interest, implied volatility, or dealer gamma.

- Status: **INSUFFICIENT_DATA**.
- Scored origins: 0.
- Frozen minimum common training origins: 126.
- Maximum available common training origins: 0.
- Flow and VXN close from session t first enter the model at origin t+1; the target is QQQ RV at t+2.
- Missing source sessions are absent, never fabricated as zero or carried forward.

No registered comparison is reported because no origin met every frozen availability,
common-row, and minimum-training gate. The gate was not relaxed after inspecting coverage.

- QQQ zero-scale registered components: near_expiry_volume_share_7d.
- SPY zero-scale registered components: near_expiry_volume_share_7d.
A zero-scale component has no training z-score. The frozen four-component
equal-weight composite is therefore undefined; the implementation does not
drop that component, assign it an invented zero score, or reweight the other three.

- QQQ: 406 observed activity days; 0 have a strictly-prior scaled composite.
- SPY: 439 observed activity days; 0 have a strictly-prior scaled composite.
