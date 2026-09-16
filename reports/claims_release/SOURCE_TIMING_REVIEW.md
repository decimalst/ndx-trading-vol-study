# Claims source timing review

Reviewed 2026-09-08. Scope: official source documentation, release-calendar HTML and dated archive-link metadata for a prospective source window ending **2025-10-20**. **No claims source is admitted by this note.** No observation dataset, seasonal-factor file or release PDF was downloaded. No numerical claims series was constructed or assessed.

## The clocks that must remain separate

| Clock or field | Documented meaning | Admission implication |
| --- | --- | --- |
| Observation date | The reference period for the measurement; ALFRED distinguishes this from the vintage date. | A reference week is not its publication date. |
| ALFRED vintage date | The historical date on which that version represents the information available. | Pin the requested vintage/output convention, not just the series identifier. |
| Initial-release `realtime_start_date` | ALFRED describes this as initial public availability; its initial-release-only output omits later revisions. | This is a candidate release-date field, subject to archive-boundary and source-date checks. |
| ALFRED date provenance | Preference order: originating source release date, provider release date, then first FRED availability. | A vintage start date is **not necessarily an independent ALFRED ingestion timestamp**. |
| Retrieval date | When this audit obtains the historical material. | It cannot establish when a historical user could have obtained it. |

Field definitions and export conventions: [ALFRED download help](https://alfred.stlouisfed.org/help/downloaddata). Date-source hierarchy: [ALFRED help](https://alfred.stlouisfed.org/help).

ALFRED reports that updates are usually entered within one business day of release; this is not a guaranteed maximum lag or a historical intraday log. The feasibility note's proposed “later of release date and ALFRED first-availability date” rule requires two independently meaningful dates. Applying it to a source-dated `realtime_start_date` twice would provide no additional protection. [ALFRED help](https://alfred.stlouisfed.org/help).

**Prospective timing rule:** authenticate the DOL public-release date, preserve the ALFRED field and its date provenance separately, and apply the declared conservative market-session lag. If the claim specifically requires availability through ALFRED, independently establish that clock or label it an unverified proxy; the documented typical delay cannot certify it. This is a recommendation, not a newly frozen rule.

## First release versus first archived snapshot

| Boundary or behavior | Evidence | Required treatment |
| --- | --- | --- |
| 2009-05-28 archive boundary | ALFRED's release listing gives this as the first archived date for the Unemployment Insurance Weekly Claims Report. The DOL 2009 index also contains a dated May 28 release link. | This establishes an archive boundary, not original-release authenticity for every older observation carried in that snapshot. |
| Older weeks assigned the initial archive date | No reviewed documentation guarantees these are unrevised original reports. | Quarantine initial-snapshot backfill until reconciled to original dated releases. Never equate “earliest available vintage” with “first published value.” |
| First boundary-week report | The existence of the May 28 DOL link makes direct reconciliation possible, but its contents were not inspected here. | Neither automatically admit nor silently discard it: establish the reference week and first report under a prewritten reconciliation check. |
| Missing ALFRED vintage date | ALFRED may omit a release date from vintage selection when values did not change. | Do not infer a missing DOL release from the vintage selector alone. |

Sources: [ALFRED release-date listing](https://alfred.stlouisfed.org/releases?ob=n&od=desc), [DOL archive selector](https://oui.doleta.gov/unemploy/claims_arch.asp), [ALFRED download help](https://alfred.stlouisfed.org/help/downloaddata). The backfill treatment is an inference about what these metadata do **not** authenticate. Original values and complete weekly vintage coverage remain unverified, including after May 2009.

## Public-release calendar and dated identities

DOL documents the normal release as Thursday at 8:30 a.m. Eastern, with exceptions. Its current page is not a complete historical holiday ledger. Use the historical date and, when separately inspected, that release's timestamp; do not hard-code Thursday or a year-round UTC−5 offset. [DOL archive instructions](https://oui.doleta.gov/unemploy/claims_arch.asp).

The archive form posts `report=press`, `year=YYYY`, `submit=Submit` to `https://oui.doleta.gov/unemploy/archive.asp`. This review inspected the returned HTML links only. Observed older links end in `.asp`; later ones end in `.pdf`. The exact format-transition year was not established. Enumerate actual links instead of inventing a uniform extension. The rolling `https://www.dol.gov/ui/data.pdf` link is not a dated historical identity. [Official archive form](https://oui.doleta.gov/unemploy/claims_arch.asp).

| Verified index metadata | Example or consequence |
| --- | --- |
| Observed 2009–2011 pattern | `https://oui.doleta.gov/press/YYYY/MMDDYY.asp`; example: [May 28, 2009](https://oui.doleta.gov/press/2009/052809.asp). |
| Observed 2020 and 2022–2025 pattern | `https://oui.doleta.gov/press/YYYY/MMDDYY.pdf`; example: [September 3, 2020](https://oui.doleta.gov/press/2020/090320.pdf). |
| Dated Wednesday releases | [December 23, 2020](https://oui.doleta.gov/press/2020/122320.pdf), [January 8, 2025](https://oui.doleta.gov/press/2025/010825.pdf), [June 18, 2025](https://oui.doleta.gov/press/2025/061825.pdf). These links demonstrate that a fixed Thursday rule fails. |
| 2025 links through the ceiling | The index contains 39 dated release links on or before 2025-10-20. Its last is [September 25](https://oui.doleta.gov/press/2025/092525.pdf); the next is [November 20](https://oui.doleta.gov/press/2025/112025.pdf), after the ceiling. This counts links, not admitted observations. |

The indexed DOL November 20 notice says releases scheduled for October 2 through November 13 were not published because of the October 1–November 12 appropriations lapse. This later notice corroborates the archive gap; it is **not information assumed known before the study ceiling**, and it does not establish when other official data channels supplied those weeks. [DOL dated notice, methodological/calendar excerpt only](https://www.dol.gov/sites/dolgov/files/OPA/newsreleases/ui-claims/20251509.pdf).

Do not fill September's state indefinitely across a missing scheduled update or backdate a later recovered record. A current archive calendar also cannot prove that every exception was announced in advance. Prospective scheduled-update and missing-release handling needs a separately fixed rule with contemporaneous notice evidence where required.

## Seasonal-adjustment dates relevant to this window

Seasonally adjusted history changes when annual factors and underlying reports are revised. BLS supplies seasonal factors; ETA publishes claims. Current BLS documentation is retrospective and was updated after the proposed source ceiling. Its historical method descriptions are documentary evidence, not permission to use later factors or revised values. [BLS seasonal-adjustment methods](https://www.bls.gov/lau/seasonal-adjustment-for-weekly-unemployment-insurance-claims.htm).

| Date of public release or notice | Documented change | Availability distinction |
| --- | --- | --- |
| 2009 onward; annual revision dates not exhaustively authenticated here | Annual estimation and historical revisions apply throughout the source window. | First-report extraction must preserve the contemporaneous adjustment; today's revised seasonally adjusted history is insufficient. [BLS methods](https://www.bls.gov/lau/seasonal-adjustment-for-weekly-unemployment-insurance-claims.htm). |
| 2020-08-27 announcement; **2020-09-03 effective release** | DOL announced a change from multiplicative to additive adjustment beginning September 3. | Do not relabel live March–August 2020 first reports using a later retrospective treatment. [Contemporaneous announcement excerpt](https://www.dol.gov/sites/dolgov/files/OPA/newsreleases/ui-claims/20201637.pdf), [effective-release excerpt](https://www.dol.gov/sites/dolgov/files/OPA/newsreleases/ui-claims/20201671.pdf). |
| **2022-04-07** | Multiplicative adjustment resumed outside the exceptional pandemic period; additive treatment was retained within that period in the revised history. | The release date of the revision differs from the observation periods revised. [DOL methodological excerpt](https://www.dol.gov/sites/dolgov/files/OPA/newsreleases/ui-claims/20220598.pdf). |
| **2023-04-06** | Changes to outlier sets produced larger historical revisions; the pandemic additive interval was excluded from that revision. | Preserve first-report vintages across this release. [DOL methodological excerpt](https://www.dol.gov/sites/dolgov/files/OPA/newsreleases/ui-claims/20230653.pdf). |
| **2024-03-14** | Structural time-series models replaced locally weighted regression. | BLS says the fixed additive interval, reference weeks 2020-03-21 through 2021-06-19, was not revised in this update. This is a retrospective interval definition. [BLS methods](https://www.bls.gov/lau/seasonal-adjustment-for-weekly-unemployment-insurance-claims.htm). |
| 2025-03-20 announcement; **2025-03-27 revision release** | DOL announced revised 2020–2024 historical series with the March 27 release and new/revised factor files available by noon Eastern that day. | The morning claims release and the factor-file deadline are separate availability claims; “by noon” is not an authenticated exact file-posting timestamp. [DOL announcement excerpt](https://www.dol.gov/newsroom/releases/eta/eta20250320). |

The dated archive index independently contains the corresponding September 3, 2020; April 7, 2022; April 6, 2023; March 14, 2024; and March 27, 2025 links. Their report bodies and numerical revisions were not retrieved. This table is not a complete annual revision calendar or a complete audit of administrative coverage changes.

## Conditions still unverified

1. Exact ALFRED export schema, file attributes, reproducible transport and date provenance for the intended export; parent and parser owner handle these separately.
2. Original-release value reconciliation, reference-week identity, archive backfill exclusions and all weekly gaps through 2025-10-20. Metadata establish none of these numerically.
3. Historical ALFRED ingestion times, full historical announcement/holiday evidence, all annual revision dates and administrative coverage changes. No assertion of complete point-in-time availability is warranted yet.
4. A frozen rule distinguishing missing release, unchanged values, delayed publication and late archival recovery, including October 2025. Do not resolve ambiguity with later observation values.

Only this new document was written. Official search excerpts incidentally displayed current and historical claims levels, including a post-ceiling November 2025 release, while locating methodological and calendar notices. Those values were not copied into an input file or used for any calculation; this review does not claim an entirely unseen source-value environment. No protected equity outcomes were accessed, and no frozen code, protocol or report was edited.
