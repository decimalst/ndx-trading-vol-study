"""Validate explicit Treasury calendar and original-security lineage dates."""

from calendar import monthrange
from datetime import date

TENORS = frozenset({2, 3, 5, 7, 10, 30})


def validate_lineage(
    *,
    dated: date,
    issue: date,
    maturity: date,
    original_dated: date | None,
    original_issue: date | None,
    pdf_original_issue: date | None,
    reopening: bool,
    allow_distinct_original_dates: bool = False,
) -> dict:
    """Return source dates unchanged; the caller binds document identity/evidence.

    The distinct-date flag is a permission supplied only by the caller's
    confirmed-substitution branch. It is not evidence of confirmation itself.
    """
    if any(type(value) is not date for value in (dated, issue, maturity)):
        raise ValueError("Actual calendar date objects are required")
    if any(
        value is not None and type(value) is not date
        for value in (original_dated, original_issue, pdf_original_issue)
    ):
        raise ValueError("Original dates must be actual dates or None")
    if type(reopening) is not bool or type(allow_distinct_original_dates) is not bool:
        raise ValueError("Lineage flags must be actual Booleans")
    if not dated <= issue < maturity:
        raise ValueError("Inconsistent current dated, issue or maturity chronology")

    if reopening:
        if original_dated is None or original_issue is None:
            raise ValueError("Reopening requires explicit original dated and issue dates")
        if not original_dated <= original_issue < issue:
            raise ValueError("Inconsistent original reopening chronology")
        if original_dated != dated and (
            not allow_distinct_original_dates or original_dated >= dated
        ):
            raise ValueError(
                "Distinct original dated date requires permission and must predate current dated date"
            )
        if pdf_original_issue != original_issue:
            raise ValueError("PDF original issue date disagrees with reopening lineage")
        anchor = original_dated
        resolved_original_issue = original_issue
    else:
        if (
            original_dated not in (None, dated)
            or original_issue not in (None, issue)
            or pdf_original_issue not in (None, issue)
        ):
            raise ValueError("New issue conflicts with original date descriptors")
        anchor = dated
        resolved_original_issue = issue

    years = maturity.year - anchor.year
    anniversary = (anchor.month, anchor.day) == (maturity.month, maturity.day)
    same_month_ends = (
        anchor.month == maturity.month
        and anchor.day == monthrange(anchor.year, anchor.month)[1]
        and maturity.day == monthrange(maturity.year, maturity.month)[1]
    )
    if years not in TENORS or not (anniversary or same_month_ends):
        raise ValueError("Unsupported original tenor from stated calendar dates")
    return {
        "original_tenor_years": years,
        "dated_date": dated.isoformat(),
        "issue_date": issue.isoformat(),
        "maturity_date": maturity.isoformat(),
        "original_issue_date": resolved_original_issue.isoformat(),
        "original_dated_date": anchor.isoformat(),
        "reopening": reopening,
    }
