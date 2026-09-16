"""Parse documented term syntax without decoding coupon values or changing text."""

import re

_COUPON = r"(?:0|[1-9][0-9]*)(?:-(?:1/2|[13]/4|[1357]/8))?%"
_TERM = re.compile(
    r"(?P<years>[1-9][0-9]?)-Year"
    r"(?: (?P<months>0|[1-9][0-9]?)-(?P<unit>Month|Mont h))?"
    r"(?: (?P<coupon>" + _COUPON + r"))? (?P<kind>Note|Bond)"
)


def parse_offered_term(term: str, *, announcement: bool) -> dict:
    """Return a canonical descriptor conditional on the caller's document role.

    Coupon digits are matched only as lexical syntax. The caller retains the
    original field and authenticates its document identity independently.
    """
    if type(term) is not str or type(announcement) is not bool:
        raise ValueError("Exact string term and explicit boolean announcement role required")
    match = _TERM.fullmatch(term)
    if match is None:
        raise ValueError("Unsupported exact offered-term syntax")
    years = int(match["years"])
    months = int(match["months"]) if match["months"] is not None else 0
    if not 1 <= years <= 30 or not 0 <= months <= 11:
        raise ValueError("Offered-term year/month descriptor outside supported bounds")
    coupon_present = match["coupon"] is not None
    split_month = match["unit"] == "Mont h"
    if not announcement and (coupon_present or split_month):
        raise ValueError("Coupon and split-month layouts are announcement-only")
    canonical = f"{years}-Year"
    if match["months"] is not None:
        canonical += f" {months}-Month"
    canonical += f" {match['kind']}"
    return {
        "years": years,
        "months": months,
        "kind": match["kind"].upper(),
        "canonical_term": canonical,
        "coupon_token_present": coupon_present,
        "split_month_token": split_month,
    }
