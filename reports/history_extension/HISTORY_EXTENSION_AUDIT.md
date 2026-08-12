# QQQ 1999 price-only history extension audit

This is an enabling-data artifact, not a model result. No tail labels,
thresholds, HMMs, embeddings, boosted trees, or scorecards were fit.

## Usable history

- Price-only panel: **6,694 sessions**, 1999-03-11 through 2025-10-17.
- Rows before 2016: **4,231**; dot-com window: **958**; 2007-09 GFC window: **756**.
- Complete daily/weekly/monthly HAR state: **6,673 sessions** after the 22-session warmup.
- Inputs are QQQ OHLC known at the 16:00 close. All rolling features are trailing-only.
- The panel stops on 2025-10-17 and does not enter the sealed NDX clean window beginning 2025-11-03.
- QQQ is a hash-frozen Yahoo Finance snapshot, not an exchange point-in-time archive; vendor revisions before this snapshot cannot be ruled out.

This panel can enable price-only HMM, tail-ranking, or latent-probe studies
from 1999. It cannot extend HAR-IV or any VXN-fed model to 1999.

## VXN boundary

Cboe's current methodology lists January 1995 as VXN's first value month,
but the frozen free Cboe file available here starts **2009-09-14**.
The earlier values were not silently sourced from a vendor, proxy-spliced,
or reconstructed. A pre-2009 VXN-fed study remains blocked on an official
complete daily series and a separate methodology-change audit.

- Official methodology: https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf
- Free history endpoint: https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv

## Provenance

- QQQ inception reference: https://www.invesco.com/us/financial-products/etfs/product-detail?productId=QQQ&ticker=QQQ
- QQQ raw SHA-256: `710290d8ad8569172559334b23e423b50ecf7cd10f0fd8ec24a9fc29608bca4d` (6,898 rows).
- VXN raw SHA-256: `61c4dec804140831dadb05d0cc34ee38a3d662d31c5b6fc258eae6918b455731` (4,255 rows).
- Derived panel SHA-256: `bfb3b938a5ac41cabfb34f19b5cde073d8d3373faf244640d747c686ffabd156`.
- Frozen build protocol SHA-256: `1002363f53c88388e66c0bbb7ae7a47a12324d90f158a87b61b979a8072821b2`.
- Frozen transform implementation SHA-256: `09eb7ee7074dc06f121cda6b35ac6af25093ae587c3602f2910538b8b7d25094`.
- The exact hashes are frozen in `history_extension.yaml`; a refreshed raw file fails closed.

## Downstream fence

Freeze the tail definition, ranking metrics, time splits, and latent-probe
incremental comparison before reading any downstream scores. A downstream
origin may consume only panel rows at or before that origin. This audit
does not authorize re-opening or peeking at the existing clean window.
