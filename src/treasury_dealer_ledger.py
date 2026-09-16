"""Pure assembly of caller-authenticated source reports; no source admission."""

import copy
import re
from datetime import date
from decimal import Decimal, InvalidOperation

PAIRED = "PAIRED_METADATA_RECONCILED"
COMPLETE = "PAIRED_SOURCE_AMOUNTS_AND_OFFERING_MATCH"
MISSING = "NOT_APPLIED_PRIOR_IDENTITY_DISPOSITION_RETAINED"
NOTICE_MISSING = "NOT_APPLIED_UNAVAILABLE_NOTICE_RETAINED"
EXCLUDED = "EXCLUDED_DOCUMENTED_CONTINGENCY_TEST"
EXCLUSION_NOTICE = "EXCLUDE_JUNE21_TEST_FROM_REGULAR_AUCTION_UNIVERSE"
FIELDS = (
    "PrimaryDealerAccepted",
    "DirectBidderAccepted",
    "IndirectBidderAccepted",
    "CompetitiveAccepted",
    "CompetitiveTendered",
    "PrimaryDealerTendered",
    "DirectBidderTendered",
    "IndirectBidderTendered",
    "NonCompetitiveAccepted",
    "FIMAAccepted",
    "SOMAAccepted",
    "SOMATendered",
    "TotalAccepted",
    "TotalTendered",
    "BidToCoverRatio",
    "HighYield",
)
VALUE_FIELDS = (
    "primary_dealer_accepted",
    "competitive_accepted",
    "offering_amount_usd",
    "high_yield",
    "bid_to_cover",
)
STAGES = {"result_xml", "result_pdf", "offering"}
TENORS = {2, 3, 5, 7, 10, 30}
CLOSURES = {
    "CONDITIONAL_ALTERNATIVE_NOT_REALIZED_FINAL_NEW_ISSUE_CONFIRMED",
    "SCHEDULE_ONLY_NO_ALLOCATION_REVISION",
}


def _day(value):
    if type(value) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("Exact ISO source date required")
    parsed = date.fromisoformat(value)
    if not date(2010, 1, 1) <= parsed <= date(2025, 10, 20):
        raise ValueError("Source date outside the fixed envelope")
    return value


def _key(row):
    if type(row) is not dict or not {"auction_date", "cusip"} <= set(row):
        raise ValueError("Explicit event identity required")
    day = _day(row["auction_date"])
    cusip = row["cusip"]
    if (
        type(cusip) is not str
        or re.fullmatch(r"[A-Z0-9]{9}", cusip) is None
        or not any(c.isdigit() for c in cusip)
    ):
        raise ValueError("Exact nine-character source CUSIP required")
    return day, cusip


def _name(key):
    return "_".join(key)


def _rows(rows):
    if type(rows) is not list or not rows:
        raise ValueError("Nonempty ordered source event list required")
    keys = [_key(row) for row in rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("Source event rows must already be ordered and unique")
    for row in rows:
        if _day(row.get("announcement_date")) > row["auction_date"]:
            raise ValueError("Announcement date follows auction date")
    return keys


def _notices(row):
    notices = row.get("known_notice_dispositions")
    unavailable = row.get("unavailable_notice_urls")
    if type(notices) is not list or type(unavailable) is not list:
        raise ValueError("Explicit old notice associations and unavailable list required")
    mapped = {}
    for notice in notices:
        if type(notice) is not dict or set(notice) != {"url", "disposition"}:
            raise ValueError("Exact old notice association schema required")
        url, disposition = notice["url"], notice["disposition"]
        if (
            type(url) is not str
            or not url.strip()
            or url in mapped
            or type(disposition) is not str
            or not disposition.strip()
        ):
            raise ValueError("Unique nonempty notice URL and disposition required")
        mapped[url] = disposition
    if (
        any(type(url) is not str for url in unavailable)
        or len(unavailable) != len(set(unavailable))
        or not set(unavailable) <= set(mapped)
    ):
        raise ValueError("Unavailable notice must be an exact unique known association")
    return mapped


def _decimal(value, *, integral=False, positive=False):
    if (
        type(value) is not str
        or re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|0E-[0-9]+)", value) is None
    ):
        raise ValueError("Plain nonnegative decimal or serialized decimal zero required")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Invalid decimal result string") from error
    if not number.is_finite() or number < 0 or (positive and number <= 0):
        raise ValueError("Invalid finite source amount")
    if integral and number != number.to_integral_value():
        raise ValueError("Source par amount is not exactly integral")
    return number


def _dollars(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Exact positive integer USD offering required")
    return int(Decimal(value))


def _financial(xml, pdf, offering):
    if any(type(data) is not dict or set(data) != set(FIELDS) for data in (xml, pdf)):
        raise ValueError("Exactly sixteen named result fields required in both sources")
    decoded = []
    for data in (xml, pdf):
        decoded.append(
            {
                key: _decimal(
                    value,
                    integral=key not in {"HighYield", "BidToCoverRatio"},
                    positive=key == "BidToCoverRatio",
                )
                for key, value in data.items()
            }
        )
    if decoded[0] != decoded[1]:
        raise ValueError("Saved sixteen-field PDF/XML amounts disagree")
    values = decoded[0]
    dealer, total = int(values["PrimaryDealerAccepted"]), int(values["CompetitiveAccepted"])
    if total <= 0 or dealer > total:
        raise ValueError("Invalid dealer numerator or competitive denominator")
    if (
        type(offering) is not dict
        or offering.get("status") != "VERIFIED_STATED_OFFERING_AGREEMENT"
        or offering.get("unit") != "USD"
        or type(offering.get("sources")) is not dict
        or set(offering["sources"]) != {"announcement_pdf", "announcement_xml", "result_xml"}
    ):
        raise ValueError("Three-source USD offering agreement required")
    amount = _dollars(offering.get("offering_amount_usd"))
    for role, source in offering["sources"].items():
        if (
            type(source) is not dict
            or source.get("source_unit")
            != ("USD" if role == "announcement_pdf" else "USD_BILLIONS")
            or _dollars(source.get("offering_amount_usd")) != amount
        ):
            raise ValueError("Stated offering sources or declared units disagree")
    return {
        "primary_dealer_accepted": dealer,
        "competitive_accepted": total,
        "offering_amount_usd": amount,
        "high_yield": format(values["HighYield"], "f"),
        "bid_to_cover": format(values["BidToCoverRatio"], "f"),
    }


def _identity(identity, key):
    if (
        type(identity) is not dict
        or identity.get("status") != "RECONCILED_METADATA_ONLY"
        or identity.get("auction_date") != key[0]
        or identity.get("actual_cusip") != key[1]
    ):
        raise ValueError("Reconciled exact final identity required")
    release = _day(identity.get("result_release_date"))
    available = _day(identity.get("identity_available_date"))
    if not key[0] <= release <= available:
        raise ValueError("Invalid final release/identity availability chronology")
    lineage = identity.get("final_lineage")
    if (
        type(lineage) is not dict
        or type(lineage.get("original_tenor_years")) is not int
        or lineage["original_tenor_years"] not in TENORS
        or type(lineage.get("reopening")) is not bool
    ):
        raise ValueError("Explicit supported original tenor and reopening required")
    return {
        "availability_after_date": available,
        "original_tenor_years": lineage["original_tenor_years"],
        "reopening": lineage["reopening"],
    }


def _old_values(row, account, payload, key):
    identity = row.get("identity")
    fields = _identity(identity, key)
    if row["unavailable_notice_urls"]:
        raise ValueError("Unavailable notice cannot be admitted by old complete status")
    if type(payload) is not dict or _key(payload.get("event")) != key:
        raise ValueError("Original accounting payload identity mismatch")
    event = payload["event"]
    if (
        event.get("announcement_date") != row["announcement_date"]
        or event.get("prior_identity_status") != row["status"]
        or payload.get("identity_source_hashes") != identity.get("input_hashes")
        or not isinstance(identity.get("input_hashes"), dict)
        or not identity["input_hashes"]
    ):
        raise ValueError("Original payload does not match source identity/provenance")
    stages = payload.get("stages")
    if (
        type(stages) is not dict
        or set(stages) != STAGES
        or any(
            type(s) is not dict or s.get("status") != "PARSED" or "result" not in s
            for s in stages.values()
        )
        or account.get("stage_statuses") != dict.fromkeys(STAGES, "PARSED")
        or account.get("paired_field_comparisons") != 16
        or account.get("mismatched_fields") != []
        or account.get("errors") != {}
        or payload.get("comparison") != {"status": COMPLETE, "mismatched_fields": []}
    ):
        raise ValueError(
            "Exact three parsed stages and complete cross-source comparison required"
        )
    return fields | _financial(
        stages["result_xml"]["result"],
        stages["result_pdf"]["result"],
        stages["offering"]["result"],
    )


def _recovered_values(row, payload, key):
    if (
        type(payload) is not dict
        or _key(payload) != key
        or payload.get("status") != COMPLETE
        or payload.get("original_status") != row["status"]
    ):
        raise ValueError(
            "Additive recovery identity, original status or completed status mismatch"
        )
    fields = _identity(payload.get("identity"), key)
    mapped = _notices(row)
    closures = payload.get("known_notice_closures")
    if type(closures) is not list or len(closures) != len(mapped):
        raise ValueError("Every prior notice needs an exact recovery closure")
    seen = set()
    for closure in closures:
        if type(closure) is not dict:
            raise ValueError("Explicit recovery notice closure required")
        url = closure.get("url")
        if type(url) is not str or url not in mapped or url in seen:
            raise ValueError("Recovery closure notice membership changed")
        seen.add(url)
        if (
            closure.get("prior_disposition") != mapped[url]
            or closure.get("status") not in CLOSURES
            or closure.get("final_cusip") != key[1]
            or type(closure.get("final_original_tenor")) is not int
            or closure["final_original_tenor"] != fields["original_tenor_years"]
            or type(closure.get("final_reopening")) is not bool
            or closure["final_reopening"] != fields["reopening"]
            or _day(closure.get("release_date")) > fields["availability_after_date"]
            or any(
                type(closure.get(k)) is not str
                or re.fullmatch(r"[0-9a-f]{64}", closure[k]) is None
                for k in ("body_sha256", "prior_review_sha256")
            )
        ):
            raise ValueError(
                "Recovery notice status, identity, clock or provenance unresolved"
            )
    return fields | _financial(
        payload.get("result_xml"), payload.get("result_pdf"), payload.get("offering")
    )


def _clock(review, key):
    if (
        type(review) is not dict
        or set(review)
        != {
            "availability_after_date",
            "clock_upper_bound_date",
            "evidence_class",
            "source_evidence",
        }
        or type(review["evidence_class"]) is not str
        or not review["evidence_class"].strip()
        or type(review["source_evidence"]) is not dict
        or not review["source_evidence"]
    ):
        raise ValueError("Explicit reviewed unknown clock and provenance required")
    exact, upper = review["availability_after_date"], review["clock_upper_bound_date"]
    for value in (exact, upper):
        if value is not None and _day(value) < key[0]:
            raise ValueError("Unknown clock precedes auction")
    if exact is not None and upper is not None and upper < exact:
        raise ValueError("Unknown clock upper bound precedes exact clock")
    return {"availability_after_date": exact, "clock_upper_bound_date": upper}


def build_ledger(
    identity_rows, accounting_rows, accounting_payloads, recovered_rows, clock_reviews
):
    """Assemble all exact source events; caller must authenticate every input pin."""
    keys = _rows(identity_rows)
    if _rows(accounting_rows) != keys:
        raise ValueError("Identity/accounting event membership or order mismatch")
    if any(type(d) is not dict for d in (accounting_payloads, recovered_rows, clock_reviews)):
        raise ValueError("Event-ID-keyed source payload and clock dictionaries required")
    complete, unknown, excluded = set(), set(), set()
    for row, account, key in zip(identity_rows, accounting_rows, keys, strict=True):
        mapped = _notices(row)
        if (
            account.get("prior_identity_status") != row.get("status")
            or account["announcement_date"] != row["announcement_date"]
            or account.get("known_notice_dispositions") != row["known_notice_dispositions"]
            or account.get("unavailable_notice_urls") != row["unavailable_notice_urls"]
        ):
            raise ValueError("Original identity/accounting metadata or notices disagree")
        name = _name(key)
        if row.get("status") == EXCLUDED:
            if (
                account.get("status") != MISSING
                or EXCLUSION_NOTICE not in mapped.values()
                or row["unavailable_notice_urls"]
            ):
                raise ValueError("Exact documented contingency exclusion evidence required")
            excluded.add(name)
        elif row.get("status") == PAIRED and account.get("status") == COMPLETE:
            if EXCLUSION_NOTICE in mapped.values():
                raise ValueError("Excluded test notice contradicts known accounting")
            complete.add(name)
        elif (
            row.get("status") == "IDENTITY_REQUIRES_REVIEW"
            and account.get("status") == MISSING
        ) or (
            row.get("status") == PAIRED
            and account.get("status") == NOTICE_MISSING
            and row["unavailable_notice_urls"]
        ):
            unknown.add(name)
        else:
            raise ValueError("Unrecognized or inconsistent source disposition")
    if set(accounting_payloads) != complete:
        raise ValueError("Original payload keyset must equal all old completed events")
    if not set(recovered_rows) <= unknown:
        raise ValueError(
            "Recovery may only resolve old unknowns, never override known/excluded events"
        )
    if set(clock_reviews) != unknown - set(recovered_rows):
        raise ValueError("Clock reviews must cover exactly the final unknown event set")
    events, evidence = [], []
    for row, account, key in zip(identity_rows, accounting_rows, keys, strict=True):
        name = _name(key)
        recovered = name in recovered_rows
        event = {
            "event_id": name,
            "auction_date": key[0],
            "availability_after_date": None,
            "clock_upper_bound_date": None,
            "status": "UNKNOWN",
            "original_tenor_years": None,
            "reopening": None,
        } | dict.fromkeys(VALUE_FIELDS)
        identity = row.get("identity")
        recovery = recovered_rows.get(name)
        if recovered:
            event.update(_recovered_values(row, recovery, key), status="KNOWN")
            identity = recovery["identity"]
        elif name in complete:
            event.update(
                _old_values(row, account, accounting_payloads[name], key), status="KNOWN"
            )
        elif name in excluded:
            event["status"] = "EXCLUDED"
        else:
            event.update(_clock(clock_reviews[name], key))
        events.append(event)
        evidence.append(
            {
                "event_id": name,
                "original_identity_status": row["status"],
                "original_accounting_status": account["status"],
                "recovered": recovered,
                "final_status": event["status"],
                "known_notice_dispositions": copy.deepcopy(row["known_notice_dispositions"]),
                "unavailable_notice_urls": copy.deepcopy(row["unavailable_notice_urls"]),
                "identity_provenance": {
                    k: copy.deepcopy(identity.get(k))
                    for k in ("input_hashes", "amendment", "confirmation")
                }
                if identity is not None
                else None,
                "original_review_bindings": {
                    k: copy.deepcopy(row.get(k))
                    for k in ("amendment_bound_to_review", "confirmation_bound_to_review")
                },
                "recovery_provenance": {
                    k: copy.deepcopy(recovery.get(k))
                    for k in (
                        "archived_result",
                        "recovered_notice_sha256",
                        "known_notice_closures",
                    )
                }
                if recovered
                else None,
                "clock_review": copy.deepcopy(clock_reviews.get(name)),
            }
        )
    return {"events": events, "evidence": evidence}
