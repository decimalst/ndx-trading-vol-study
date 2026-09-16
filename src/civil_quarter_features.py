"""Civil month/quarter endpoints on an unchanged original SPX feature table.

No source loading, future session lookup, target construction or numerical
market transformation occurs here. Support failures retain no partial sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_variance_features import ALL_FEATURES as OLD_FEATURES
from .index_hinge import _dates

MONTHS = tuple(f"month_{month}" for month in range(2, 13))
NUISANCE = MONTHS + ("month_end5", "year_end5")
MEMORY = "quarter_end5"
CIVIL_FEATURES = NUISANCE + (MEMORY,)
BASE = OLD_FEATURES + NUISANCE
ALL_FEATURES = BASE + (MEMORY,)
MODELS = ("mean", "baseline", "quarter")
RANK_RELATIVE_THRESHOLD = 1e-10


def nominal_date(origin):
    """Next Gregorian Mon-Fri civil date; holidays and early closes are ignored."""
    origin = pd.Timestamp(origin)
    if pd.isna(origin) or origin.tz is not None or origin != origin.normalize():
        raise ValueError("A normalized timezone-naive civil origin is required")
    candidate = origin + pd.Timedelta(days=1)
    while candidate.weekday() > 4:
        candidate += pd.Timedelta(days=1)
    return candidate


def augment_features(original):
    """Append raw civil indicators without modifying any old value or cutoff.

    The input is the full reference calendar, including unknown old features.
    Training-only centering/scaling belongs to the caller's fitting transform.
    """
    expected = OLD_FEATURES + ("feature_cutoff_date",)
    if (
        not isinstance(original, pd.DataFrame)
        or original.columns.has_duplicates
        or len(original.columns) != len(expected)
        or set(original.columns) != set(expected)
    ):
        raise ValueError("Exactly the original eighteen features and cutoff are required")
    _dates(original.index)
    previous = (
        pd.Series(original.index, index=original.index).shift(1).rename("feature_cutoff_date")
    )
    if not original.feature_cutoff_date.equals(previous):
        raise ValueError("The unchanged full-calendar preceding-session cutoff is required")
    result = original.loc[:, OLD_FEATURES].copy()
    nominal = pd.DatetimeIndex([nominal_date(date) for date in original.index])
    month_end = nominal.day >= nominal.days_in_month - 4
    for month, column in zip(range(2, 13), MONTHS, strict=True):
        result[column] = (nominal.month == month).astype(float)
    result["month_end5"] = month_end.astype(float)
    result["year_end5"] = (month_end & (nominal.month == 12)).astype(float)
    result[MEMORY] = (month_end & np.isin(nominal.month, [3, 6, 9])).astype(float)
    result["feature_cutoff_date"] = original.feature_cutoff_date
    return result.loc[:, ALL_FEATURES + ("feature_cutoff_date",)]


def _civil_values(features):
    if (
        not isinstance(features, pd.DataFrame)
        or features.columns.has_duplicates
        or not set(CIVIL_FEATURES).issubset(features.columns)
    ):
        raise ValueError("All distinct raw civil indicators are required")
    values = features.loc[:, CIVIL_FEATURES].to_numpy()
    if (
        np.iscomplexobj(values)
        or not np.issubdtype(values.dtype, np.number)
        or not np.isfinite(values).all()
        or not np.isin(values, [0.0, 1.0]).all()
    ):
        raise ValueError(
            "Every admitted civil indicator must be finite binary, with no missing-row removal"
        )
    return values.astype(float, copy=False)


def civil_support(features, scope="train"):
    """Check fixed marginal binary support on all supplied rows, without masking."""
    if scope not in ("train", "phase", "slice"):
        raise ValueError("Support scope must be train, phase or slice")
    values = _civil_values(features)
    month_minimum = 10 if scope == "slice" else 20
    quarter_minimum = {"train": 20, "phase": 30, "slice": 15}[scope]
    minimums = {
        **dict.fromkeys(MONTHS, month_minimum),
        "month_end5": 20,
        "year_end5": 5,
        MEMORY: quarter_minimum,
    }
    groups = {}
    for index, column in enumerate(CIVIL_FEATURES):
        ones = int(np.count_nonzero(values[:, index] == 1.0))
        zeros = len(values) - ones
        minimum = minimums[column]
        groups[column] = {"ones": ones, "zeros": zeros, "minimum_per_class": minimum}
        if min(ones, zeros) < minimum:
            raise ValueError(
                f"INSUFFICIENT_DATA: {scope} {column} requires {minimum} rows in each binary class"
            )
    return {"scope": scope, "n": len(values), "groups": groups}


def require_civil_rank(features):
    """Require rank15 of the raw intercept plus all fourteen civil indicators."""
    values = _civil_values(features)
    if "const" not in features or not np.array_equal(
        features.const.to_numpy(), np.ones(len(features))
    ):
        raise ValueError("A literal unit intercept is required for civil rank")
    design = np.column_stack([np.ones(len(values)), values])
    singular = np.linalg.svd(design, compute_uv=False)
    if not len(singular) or not np.isfinite(singular).all() or singular[0] <= 0:
        raise ValueError("INSUFFICIENT_DATA: finite nonempty civil design required")
    threshold = RANK_RELATIVE_THRESHOLD * float(singular[0])
    rank = int(np.count_nonzero(singular > threshold))
    if rank != 15:
        raise ValueError(
            "INSUFFICIENT_DATA: civil intercept/month/end-point design must have rank15"
        )
    return {
        "columns": ["const", *CIVIL_FEATURES],
        "n": len(features),
        "rank": rank,
        "relative_threshold": RANK_RELATIVE_THRESHOLD,
        "threshold": threshold,
        "singular_values": singular.tolist(),
    }
