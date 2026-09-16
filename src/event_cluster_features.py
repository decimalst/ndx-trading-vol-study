"""Fixed22 mature-label adjacency, count and recency summaries.

This module consumes an already admitted, full-reference target table. It does
not load sources, infer missing labels, center training features, fit models or
choose applications from their future labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

L = 22
SOURCE_END = "2025-10-20"
TARGET_COLUMNS = ("y", "target_end", "available_date")
NUISANCE = (
    "event_fraction22",
    "event_fraction22_sq",
    "last_event",
    "expected_adjacency22",
    "linear_recency22",
)
MEMORY = "adjacency_fraction22"
DIAGNOSTIC = "excess_adjacency22"
FEATURES = NUISANCE + (MEMORY,)
AUDIT_FIELDS = (
    "event_count22",
    "adjacent_pairs22",
    "expected_adjacency_numerator",
    "linear_recency_numerator",
    "excess_adjacency_numerator",
)
SUMMARY_COLUMNS = FEATURES + (DIAGNOSTIC,) + AUDIT_FIELDS
DATE_COLUMNS = (
    "feature_cutoff_date",
    "window_first_available",
    "window_last_available",
    "window_first_origin",
    "window_last_origin",
)
MEMORY_COLUMNS = SUMMARY_COLUMNS + DATE_COLUMNS


def _binary(values, *, unknown=False):
    """Reject coercible nonnumeric types; NaN is the only permitted unknown."""
    if isinstance(values, (tuple, list)) and any(
        isinstance(value, (bool, np.bool_)) for value in values
    ):
        raise ValueError("Binary observations must be real numeric, not boolean")
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "One-dimensional real numeric binary observations required"
        ) from error
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise ValueError("One-dimensional real numeric binary observations required")
    if np.isinf(array).any():
        raise ValueError("Observed binary labels must be finite")
    known = ~np.isnan(array)
    if not unknown and not known.all():
        raise ValueError("All22 binary observations must be known")
    if not np.all((array[known] == 0) | (array[known] == 1)):
        raise ValueError("Known labels must be exactly zero or one")
    return array


def cluster_summary(events):
    """Return six features, one diagnostic and five oldest-first integer audits.

    Exactly22 known numeric binary observations are required. Every numerator
    is formed using Python integers before one division. In particular, the
    squared fraction is N*N/484, fitted adjacency is C/21, and descriptive
    excess adjacency is (21*C-Gnum)/441. No feature is obtained by manipulating
    a previously rounded feature; the excess diagnostic never enters FEATURES.
    """
    values = _binary(events)
    if values.shape != (L,):
        raise ValueError("Exactly22 binary observations required")
    sequence = tuple(int(value) for value in values)
    count = sum(sequence)
    last = sequence[-1]
    pairs = sum(left * right for left, right in zip(sequence, sequence[1:]))
    expected_numerator = (count - last) * (count - 1)
    recency_numerator = sum((2 * i - 23) * event for i, event in enumerate(sequence, start=1))
    excess_numerator = 21 * pairs - expected_numerator
    return {
        "event_fraction22": count / 22,
        "event_fraction22_sq": (count * count) / 484,
        "last_event": float(last),
        "expected_adjacency22": expected_numerator / 441,
        "linear_recency22": recency_numerator / 462,
        "adjacency_fraction22": pairs / 21,
        "excess_adjacency22": excess_numerator / 441,
        "event_count22": count,
        "adjacent_pairs22": pairs,
        "expected_adjacency_numerator": expected_numerator,
        "linear_recency_numerator": recency_numerator,
        "excess_adjacency_numerator": excess_numerator,
    }


def _validate_targets(targets):
    if not isinstance(targets, pd.DataFrame):
        raise ValueError("Full-reference target DataFrame required")
    if tuple(targets.columns) != TARGET_COLUMNS or targets.columns.has_duplicates:
        raise ValueError("Exact y/target_end/available_date schema required")
    index = targets.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or not len(index)
        or index.hasnans
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or index.tz is not None
        or not index.equals(index.normalize())
        or index[-1] > pd.Timestamp(SOURCE_END)
    ):
        raise ValueError(
            "Nonempty unique ascending normalized bounded naive date index required"
        )
    next_dates = pd.Series(index, index=index).shift(-1)
    for column in TARGET_COLUMNS[1:]:
        actual = targets[column]
        if not pd.api.types.is_datetime64_dtype(actual.dtype) or not actual.equals(
            next_dates.rename(column)
        ):
            raise ValueError(
                "Exact next-reference-close target and availability dates required"
            )
    labels = _binary(targets["y"], unknown=True)
    if not np.isnan(labels[-1]):
        raise ValueError(
            "Terminal origin without a next reference close must have unknown label"
        )
    return labels


def build_memory(targets):
    """Keep the full index and use only literal22 arrivals before each origin.

    At position i>=23, y[i-23:i-1] has availability positions i-22..i-1.
    Earlier origins have unknown summaries and no complete structural window.
    A missing label leaves all summary/audit fields unknown for its entire22
    arrival windows, while the structural window dates remain recorded. The
    known preceding reference cutoff is retained even during warmup.
    """
    labels = _validate_targets(targets)
    index = targets.index
    result = pd.DataFrame(np.nan, index=index, columns=SUMMARY_COLUMNS)
    dates = pd.Series(index, index=index)
    result["feature_cutoff_date"] = dates.shift(1)
    for column in DATE_COLUMNS[1:]:
        result[column] = pd.Series(pd.NaT, index=index, dtype=dates.dtype)
    for i in range(L + 1, len(index)):
        origin = index[i]
        result.loc[origin, list(DATE_COLUMNS[1:])] = [
            index[i - L],
            index[i - 1],
            index[i - L - 1],
            index[i - 2],
        ]
        window = labels[i - L - 1 : i - 1]
        if np.isnan(window).any():
            continue
        summary = cluster_summary(window)
        result.loc[origin, list(SUMMARY_COLUMNS)] = list(summary.values())
    return result.loc[:, MEMORY_COLUMNS]
