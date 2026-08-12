# Frozen protocol: QQQ history extension to 1999

Frozen 2026-08-12 before transformation. The executable specification is
[`history_extension.yaml`](../../history_extension.yaml).

## Purpose

Construct one provenance-locked, price-only QQQ panel that includes the dot-com
collapse and the global financial crisis. This step only changes the available
history. It does not define a tail event, train a model, extract an embedding, or
score a forecast.

## Source and timing

- Use the repository's existing QQQ daily OHLC acquisition, beginning at QQQ's
  first session on 1999-03-10. Unadjusted OHLC determines same-session
  Garman-Klass variance and the prior-close/current-open gap. Adjusted close is
  used only for the close-to-close return.

Invesco's product page independently reports QQQ's inception date as 1999-03-10:
https://www.invesco.com/us/financial-products/etfs/product-detail?productId=QQQ&ticker=QQQ
- At a 16:00 ET origin, those values and trailing 5- and 22-session means are
  known. No feature is centered, smoothed, backfilled, or forward-filled.
- Stop origins on 2025-10-17. The clean NDX window beginning 2025-11-03 remains
  sealed.
- Require exact SHA-256 matches for the existing QQQ and VXN raw files before
  building. A refresh is a new source version and must be re-audited.

The QQQ input is a frozen Yahoo Finance snapshot, not an exchange-maintained
point-in-time archive. Its hash makes this exact reconstruction repeatable; it
does not establish that the vendor never revised an older bar. This limitation
applies to all existing QQQ daily-GK results, not only the extension.

## VXN limitation

Cboe's current selected-index methodology lists January 1995 as VXN's first
value month, but the current free Cboe history file retained by the repository
starts on 2009-09-14. This protocol does not splice a third-party series or use
VIX as an NDX proxy. The 1999 extension therefore enables price-only studies;
VXN-fed studies remain bounded by the acquired VXN history.

## Output

`data/history_extension/qqq_price_only_daily.parquet` contains daily
GK-plus-overnight variance, log variance, adjusted close return, and trailing
HAR state. It contains no implied-volatility, macro, constituent, event, or
future target column.

Downstream tail and latent-space studies must separately freeze their target,
threshold-training rule, non-overlapping evaluation phases, AUC/lift metrics,
calibration comparison, and incremental test before scoring.
