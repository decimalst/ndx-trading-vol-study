# SPX intraday providers: documentary feasibility only

**No provider is admitted.** Cboe Main Channel Tick Data is the strongest documented technical lead; FirstRate Data is a second route with explicit private-research language but weaker timing evidence. Massive and ThetaData advertise index histories that start too late for the unchanged experiment. An additional bounded check of algoseek did not establish a product-specific SPX archive. This review covers exactly five providers.

All external pages below were accessed on **2026-09-07**, with the final documentation check at approximately **15:24 UTC**. These are current provider claims, not archived historical-vintage proofs or market-data completeness tests. Prices and terms require confirmation before any order. No account, purchase, inquiry message, sample download, market-value parsing, new feature/target, support count or fit was performed. No protected-period market observations were inspected; advertised end dates and documentation examples are not admitted observations. Only this new report was written.

The question comes from the frozen [prospective note](/Users/byrons/code/trading-vol/ndx-vol-experiment/reports/issued_calibration/NEXT_RESEARCH_DIRECTION.md): a value actually observable by 09:35 America/New_York, followed by a target using only the later interval. It needs pretraining history as well as 2016–2025-10-20. SPX means the cash-index level, not SPY, futures, an options premium or a special settlement value. The existing daily file cannot establish this temporal separation.

| Provider and exact product | Public historical claim | Access/cost found | Documentary disposition |
|---|---|---|---|
| Cboe DataShop, Main Channel Tick Data | Main-channel disseminated values, including SPX; active-symbol archive from January 2004. [Product](https://datashop.cboe.com/main-channel-tick-data) | Historical order or subscription; exact requested-history/internal-use price not visible without selection. Empty-cart $0 is not a quote. | Technical lead; separate index-derived-research rights and timing/vintage evidence remain unresolved. |
| FirstRate Data, SPX index bars | SPX/GSPC/INX, 1-minute through daily OHLC, 2008-01-02 through advertised 2026-09-04. [SPX product](https://firstratedata.com/i/index/SPX) | Upfront price not visible in the accessible product text; individual update subscription shown as $99.95/year after the included month. | Advertised dates could cover the required history; bars alone do not prove cutoff availability. |
| Massive, indices aggregates/values | Both current historical endpoint pages specify 2023-02-14 as the beginning. [Aggregates](https://massive.com/docs/rest/indices/aggregates), [values](https://massive.com/docs/flat-files/indices/values) | Individual Starter $49/month; Advanced $99/month. [Pricing](https://massive.com/indices) | Fails the unchanged history requirement; do not import the vendor's stock-history claims into indices. |
| ThetaData, index history | Index PRO begins 2017-01-01; STANDARD 2022-01-01; VALUE 2023-01-01. [Subscription coverage](https://docs.thetadata.us/Articles/Getting-Started/Subscriptions.html) | Terminal/API subscription. Commercial index page displays $400/month; billing choice and exact quote need confirmation. [Business pricing](https://www.thetadata.net/commercial-use) | Even the earliest advertised tier omits training and 2016 development. |
| algoseek | General archive and Cboe-index-feed references, but no SPX-specific start date/product schema established in this bounded check. [Official catalog/home](https://algoseek.com/) | Product-specific SPX access and price unconfirmed. | Unconfirmed lead only; broad equities/options or constituent-history coverage is insufficient. |

## 1. Cboe: promising tick semantics, material rights gate

The Main Channel product is explicitly an index-value archive, and its historical files are delivered on the next trading day. That delivery schedule describes the purchased historical service; it does not mean the underlying index value was unavailable when originally disseminated. Conversely, the archival delivery is not proof of what a historical subscriber had received by 09:35. The product supports only active symbols and warns that particular symbol/date selections can be unavailable. No SPX-specific completeness inventory was obtained. [Main Channel Tick Data](https://datashop.cboe.com/main-channel-tick-data)

The **v1.0, October 2024 specification** gives one zipped CSV per date, `main_tick_yyyymmdd.csv.zip`, with `symbol`, `datetime`, `price` and `total_option_volume`. It describes `price` as a spot index value and `datetime` as the UTC time of the disseminated value. It states that millisecond precision was added on **2019-01-14**, replacing the earlier second-resolution format. No receipt timestamp, correction sequence, historical as-of version or revision flag appears in this layout. The precision break needs an explicit cutoff-boundary policy; a second label must not silently acquire subsecond certainty. [Main Channel Tick Data specification](https://datashop.cboe.com/documents/Main_Channel_Tick_Data_v1.pdf)

The current license is a substantive unresolved gate. Its general internal-use grant is limited by **§I.1(e)(viii), PDF page 4**, which requires a separate executed agreement for specified indicator/strategy/benchmark uses and for derived data created from index-related data. Therefore an ordinary purchase or the phrase “internal use” does not itself establish permission for these SPX features, targets and models. The order must explicitly permit the intended private derived research, retained source snapshots and allowed publication of non-reconstructive results. This report does not determine the user's license classification. [DataShop License Agreement](https://datashop.cboe.com/documents/Cboe_LiveVol_DataShop_License_Agreement.pdf)

Supplier requirements can also govern use notwithstanding the generic agreement. Exact licensing and price remain unconfirmed; no academic eligibility or discount is assumed. [Historical Data Services policies](https://datashop.cboe.com/documents/DataShop_Policies_for_Historical_Data_Services.pdf)

**Alternative within the same provider:** Cboe Legacy Market Data Replay explicitly advertises `^SPX` from January 1990 and SFTP delivery. Its mixed quote/trade record layout needs instrument/record-type admission before treating any field as the cash-index path. The product documents a **2019-10-01** change from a field described as CST to EST with milliseconds, a sequence-field change, and historical values with more than two decimals before **2021-07-15**. Those changes are distinct from Main Channel's 2019-01-14 precision change. They must not be combined or “fixed” by assuming the two products share a clock. DST meaning, message origin, corrections and right-to-use scope remain open. No MDR price was obtained. [Cboe Legacy Market Data Replay](https://datashop.cboe.com/mdr-quotes-trades-data)

**Next admissible documentary request:** exact SPX date coverage/schema versions, dissemination versus collection/receipt semantics, revision/as-of policy, and a written quote/license covering this specific derived research. No request was sent.

## 2. FirstRate Data: explicit private-research terms, incomplete temporal proof

The SPX product is specifically the index, and its advertised 2008 start is early enough to investigate pretraining support without moving the historical phases. This is not proof that every required session/minute is present. The product provides zipped CSV files, and its update price is distinct from an initial historical purchase price, which this review could not read. The index bundle separately advertises updates at $59.95/month after its first month; that is not a quote for the SPX historical archive. [SPX product](https://firstratedata.com/i/index/SPX), [index bundle](https://firstratedata.com/b/8/us-index-historic-intraday)

The FAQ specifies US Eastern timestamps with EST/EDT seasonality, OHLC without volume for indices, and daily updates. It identifies direct exchange sourcing where possible and enterprise vendors otherwise, but does not identify the exact upstream SPX source across the entire archive. It also describes customer download links and a bundle API serving archives, not individual raw API rows. These facts do not establish original receipt time. [FAQ](https://firstratedata.com/about/faq)

The published license expressly allows private use, internal models and derived published research, subject to its restrictions and attribution requirement. Commercial third-party resale/redistribution is prohibited; limited disclosure of insubstantial amounts and certain service-provider uses are separately allowed by the terms. Multi-user scope requires the appropriate license. It also disclaims timeliness/completeness guarantees and permits corrections without notice. That is a plausible private-research rights route, but it provides no historical revision log or original-vintage guarantee. [License agreement](https://firstratedata.com/about/license)

FirstRate's transformation tool describes its *generated resampled bars* as beginning-labeled. That is not adequate evidence that every original SPX bar across 2008–2025 used the same convention. [Transformation documentation](https://tools.firstratedata.com/)

**Unresolved admission:** original index bar label/inclusion rules, tick versus sampled-value aggregation, upstream feed and format history, empty intervals, staleness and corrections. Even a confirmed 09:35 beginning-labeled bar cannot supply its later close/high/low to the 09:35 forecast. Its first observation might also occur after the cutoff. A preceding completed bar's final value needs a separately justified availability and staleness convention. One-minute bars may help isolate future extrema, but this review does not admit them as a sufficient clock for the intended cutoff price.

## 3. Massive: clear endpoint semantics, insufficient dates

The index aggregate endpoint derives OHLC from index values rather than trades, omits intervals with no updates, and labels bars by the Unix-millisecond **start** of their window. The documented history begins in 2023, so no combination of its listed personal plans supports the unchanged training/development period. Index-value flat files have the same stated first date. [Aggregate specification](https://massive.com/docs/rest/indices/aggregates), [index-value files](https://massive.com/docs/flat-files/indices/values)

The advertised paid plans distinguish delayed from real-time access and restrict personal plans to individual/nonprofessional use; business plans are separate. This review did not establish a business license, extra historical backfill, receipt-time archive or revision policy. Paying for a higher tier is not evidence of unavailable earlier index history. [Indices plans](https://massive.com/indices)

**Disposition:** excluded for this unchanged historical design, despite useful modern APIs. No API request or shorter-period replacement was attempted.

## 4. ThetaData: informative clock documentation, also too late

ThetaData explicitly distinguishes actual indices such as SPX from ETF substitutes. [Index product](https://www.thetadata.net/indices-data)

Its price-history documentation describes interval snapshots at the stated time and suppression of unchanged updates; the OHLC documentation defines a beginning-labeled half-open bar, `[timestamp, timestamp + interval)`. Those are different products and cannot be interchanged at 09:35. Neither page establishes a historical subscriber-receipt timestamp or a reversible revision history. Suppressed repeated values also must not be confused with an outage without an admitted status/completeness rule. [Price history](https://docs.thetadata.us/operations/index_history_price.html), [OHLC history](https://docs.thetadata.us/operations/index_history_ohlc.html)

The subscription table's earliest index access date is 2017, which cannot support the fixed first development year or its pretraining. The commercial page repeats that historical boundary. Personal pricing is explicitly for personal use without business use or redistribution; the accessible general pricing page initially rendered options prices, which were **not** treated as index prices. Exact individual index cost was therefore not established. [Subscriptions](https://docs.thetadata.us/Articles/Getting-Started/Subscriptions.html), [commercial index access](https://www.thetadata.net/commercial-use), [personal pricing/usage](https://www.thetadata.net/pricing)

**Disposition:** excluded for the unchanged phases, without inferring pre-2017 coverage from longer stock/options advertisements.

## 5. algoseek: bounded negative finding

The official catalog identifies Cboe Indices among upstream feeds and describes long general archives, but the inspected SPX mentions largely concern options. Its point-in-time index-components product is constituent membership, not a cash-index level series. This review did not find a specific SPX cash-index schema, date range or separately priced product. It does **not** establish that such a custom product is unavailable. [Official catalog](https://algoseek.com/)

The general package terms support internal research, controlled storage and retained non-reconstructive derived results; leased raw data must be deleted after the lease unless separately purchased. Those terms cannot be assigned to an unidentified SPX product. A permanent immutable-source requirement would need an appropriate retained-data license. [Multi-asset package terms](https://algoseek.com/multi-asset-package/)

**Disposition:** do not treat generic archive length or broad exchange-fee statements as SPX admission. Product confirmation would come before price, time-field and license checks.

## Decision and unsatisfied gates

The practical first documentary route is **Cboe Main Channel Tick Data**, conditional on explicit rights for index-derived research and a timing/vintage explanation. FirstRate remains a potentially simpler archival purchase only if its original SPX bars can support the exact as-of/cutoff measurement. Neither route is approved for acquisition or modeling by this report.

Before any source or trial is admitted, the following remain required:

1. Exact cash-index identity and required pretraining-to-2025-10-20 coverage, with gaps and historical schema versions. No sample shortening or proxy substitution.
2. Source-event/dissemination time distinguished from provider capture and subscriber availability; precision/DST rules and any defensible latency assumption written down. Timestamp formatting alone is insufficient.
3. An observable price by 09:35 and an independently separated later interval. No completed 09:35 bar may supply future fields; no later print may fill a missing cutoff value.
4. Correction/as-of provenance, stale-message/outage distinctions, regular and early closes, and permissible retention/research/publication rights. A current cleaned archive cannot silently become a historical live record.
5. Only after documentary admission, a separately authorized acquisition plan bounded before numerical parsing, immutable raw hashes, synthetic parser/clock tests and independent measurement validation. Advertised coverage has not established the minimum mature sample or class support.

The new post-cutoff history `K_t` also needs an adequately matched nuisance control before any gain can be attributed specifically to the current price shock. That model-identification decision remains prospective and is not solved by choosing a vendor.
