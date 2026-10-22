"""Independent, paired summaries for a synthetic TiRex intervention pilot.

These summaries concern intervention sensitivity on designed synthetic pairs.
Exact binomial tests concern the probability of exceeding the declared effect
floor, not a population mean. Percentile bootstrap intervals are descriptive,
pointwise intervals and do not receive a multiplicity adjustment here.
"""

from numbers import Integral

import numpy as np
from scipy.stats import binomtest

_BOOTSTRAP_REPLICATES = 5000


def _forecast_array(value, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _crossings(values):
    """Count forecast vectors with any adjacent-quantile order violation."""
    return int(np.any(np.diff(values, axis=-1) < 0, axis=-1).sum())


def _mean_ci(values, sampled_indices):
    bootstrap_means = values[sampled_indices].mean(axis=1)
    return [float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])]


def analyze_candidate(base, swapped, controls, direction, floor=0.001, seed=0):
    """Summarize paired activation swaps and matched control interventions.

    Parameters
    ----------
    base, swapped : arrays, shape (N, 2, 3)
        Quantiles q10, q50, q90, with side 0 recipient and side 1 donor.
        Swapped side 0 receives donor activation; side 1 receives recipient.
    controls : array, shape (K, N, 2, 3)
        Forecasts from K control-unit swaps, aligned to the same N pairs.
    direction : +1 or -1
        Declared expected recipient-shift sign; donor shift should reverse it.
    floor : nonnegative finite float
        Strict success threshold, in the same units as the supplied forecasts.
    seed : nonnegative integer
        Deterministic bootstrap seed. Paired sample indices are shared across
        influence and control-advantage intervals.

    Quantile crossings are counted without sorting or otherwise repairing them.
    No pair is dropped, including a pair with identical native forecasts.
    """
    baseline = _forecast_array(base, "base")
    intervention = _forecast_array(swapped, "swapped")
    comparison = _forecast_array(controls, "controls")
    if baseline.ndim != 3 or baseline.shape[1:] != (2, 3) or baseline.shape[0] == 0:
        raise ValueError("base must have shape (N, 2, 3) with N positive")
    if intervention.shape != baseline.shape:
        raise ValueError("swapped must match base shape")
    if (
        comparison.ndim != 4
        or comparison.shape[0] == 0
        or comparison.shape[1:] != baseline.shape
    ):
        raise ValueError("controls must have shape (K, N, 2, 3) with K positive")
    if isinstance(direction, (bool, np.bool_)) or direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    try:
        threshold = float(floor)
    except (TypeError, ValueError) as exc:
        raise ValueError("floor must be nonnegative and finite") from exc
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("floor must be nonnegative and finite")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    pair_count = baseline.shape[0]
    median_delta = intervention[:, :, 1] - baseline[:, :, 1]
    control_median_delta = comparison[:, :, :, 1] - baseline[None, :, :, 1]
    width_delta = (
        intervention[:, :, 2] - intervention[:, :, 0]
        - baseline[:, :, 2] + baseline[:, :, 0]
    )
    control_width_delta = (
        comparison[:, :, :, 2] - comparison[:, :, :, 0]
        - baseline[None, :, :, 2] + baseline[None, :, :, 0]
    )
    influence = direction * (median_delta[:, 0] - median_delta[:, 1]) / 2.0
    candidate_magnitude = np.abs(median_delta).mean(axis=1)
    control_magnitude = np.abs(control_median_delta).mean(axis=(0, 2))
    advantage = candidate_magnitude - control_magnitude
    influence_successes = int(np.count_nonzero(influence > threshold))
    advantage_successes = int(np.count_nonzero(advantage > threshold))
    expected_recipient = direction * median_delta[:, 0] > 0
    expected_donor = -direction * median_delta[:, 1] > 0
    native_contrast = baseline[:, 1, 1] - baseline[:, 0, 1]
    rng = np.random.default_rng(int(seed))
    sampled_indices = rng.integers(0, pair_count, size=(_BOOTSTRAP_REPLICATES, pair_count))

    return {
        "n_pairs": int(pair_count),
        "n_controls": int(comparison.shape[0]),
        "direction": int(direction),
        "floor": threshold,
        "seed": int(seed),
        "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
        "influence_effects": influence.tolist(),
        "advantage_effects": advantage.tolist(),
        "influence_successes": influence_successes,
        "advantage_successes": advantage_successes,
        "influence_pvalue": float(
            binomtest(influence_successes, pair_count, p=0.5, alternative="greater").pvalue
        ),
        "advantage_pvalue": float(
            binomtest(advantage_successes, pair_count, p=0.5, alternative="greater").pvalue
        ),
        "mean_influence_effect": float(influence.mean()),
        "mean_advantage_effect": float(advantage.mean()),
        "influence_mean_ci95": _mean_ci(influence, sampled_indices),
        "advantage_mean_ci95": _mean_ci(advantage, sampled_indices),
        "mean_native_donor_recipient_contrast": float(native_contrast.mean()),
        "mean_recipient_median_shift": float(median_delta[:, 0].mean()),
        "mean_donor_median_shift": float(median_delta[:, 1].mean()),
        "mean_abs_median_shift": float(np.abs(median_delta).mean()),
        "mean_abs_width_shift": float(np.abs(width_delta).mean()),
        "mean_abs_control_median_shift": float(np.abs(control_median_delta).mean()),
        "mean_abs_control_width_shift": float(np.abs(control_width_delta).mean()),
        "both_directions_expected_sign_fraction": float(
            np.mean(expected_recipient & expected_donor)
        ),
        "crossing_count": _crossings(intervention),
        "base_crossing_count": _crossings(baseline),
        "control_crossing_count": _crossings(comparison),
    }


def holm_adjust(pvalues):
    """Holm step-down adjusted p values in original input order."""
    try:
        values = np.asarray(pvalues, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("pvalues must be a one-dimensional finite numeric sequence") from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("pvalues must be a one-dimensional finite numeric sequence")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("pvalues must lie between zero and one")
    if values.size == 0:
        return []
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.minimum(
        1.0, np.maximum.accumulate(values[order] * np.arange(values.size, 0, -1))
    )
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()
