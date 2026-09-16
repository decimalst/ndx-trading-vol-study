"""Paired-mean uncertainty on an uncompressed observed-session calendar."""

import math

import numpy as np
from scipy.stats import norm

_DRAW_CHUNK = 1024


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Nonfinite or overflowing {name}")
    return value


def _bootstrap(weighted, selected, mean, block, draws, seed):
    """Sum circular blocks without constructing sampled calendar tensors."""
    total = len(selected)
    complete, remainder = divmod(total, block)
    count_prefix = np.concatenate(([0], np.cumsum(np.tile(selected, 2), dtype=np.int64)))
    value_prefix = np.concatenate(([0.0], np.cumsum(np.tile(weighted, 2), dtype=np.float64)))
    _finite(value_prefix, "circular cumulative sums")
    counts = np.empty(draws, dtype=np.int64)
    sums = np.empty(draws, dtype=np.float64)
    rng = np.random.default_rng(seed + block)
    # Generating every full-block start before any tail start preserves the
    # literal (draws, complete) RNG sequence regardless of chunk boundaries.
    for lower in range(0, draws, _DRAW_CHUNK):
        upper = min(lower + _DRAW_CHUNK, draws)
        starts = rng.integers(0, total, size=(upper - lower, complete))
        counts[lower:upper] = np.sum(
            count_prefix[starts + block] - count_prefix[starts], axis=1
        )
        sums[lower:upper] = np.sum(value_prefix[starts + block] - value_prefix[starts], axis=1)
    if remainder:
        for lower in range(0, draws, _DRAW_CHUNK):
            upper = min(lower + _DRAW_CHUNK, draws)
            starts = rng.integers(0, total, size=upper - lower)
            counts[lower:upper] += count_prefix[starts + remainder] - count_prefix[starts]
            sums[lower:upper] += value_prefix[starts + remainder] - value_prefix[starts]
    if np.any(counts == 0):
        raise ValueError(f"zero-support bootstrap replicate for block {block}")
    means = _finite(sums / counts, "bootstrap means")
    centered = _finite(means - mean, "centered bootstrap differences")
    p = (1 + int(np.count_nonzero(np.abs(centered) >= abs(mean)))) / (draws + 1)
    interval = _finite(np.quantile(means, [0.025, 0.975]), "bootstrap interval")
    return {"p": float(p), "ci95": [float(x) for x in interval]}


def masked_mean_inference(
    differences,
    mask,
    *,
    blocks=(21, 63, 126),
    hac_lags=126,
    draws=399999,
    seed=20261001,
):
    """Infer the selected paired mean while preserving every calendar position.

    False-mask cells may be NaN or finite placeholders. They contribute zero
    to masked sums and influence values; they never become observed losses.
    Support, calendar, phase, model and multiple-comparison gates are external.
    """
    try:
        original = np.asarray(differences)
        selected = np.asarray(mask)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "One-dimensional numerical differences and boolean mask required"
        ) from exc
    if original.ndim != 1 or original.dtype.kind not in "fiu":
        raise ValueError("One-dimensional real numerical differences required")
    if selected.ndim != 1 or selected.dtype.kind != "b":
        raise ValueError("One-dimensional boolean mask required")
    total = len(original)
    if total < 2 or len(selected) != total:
        raise ValueError("Matching full calendar arrays with at least two positions required")
    values = original.astype(np.float64, copy=True)
    selected = selected.copy()
    if np.isinf(values).any() or not np.isfinite(values[selected]).all():
        raise ValueError("Finite selected differences required; infinity is never allowed")
    count = int(np.count_nonzero(selected))
    if count < 2:
        raise ValueError("At least two selected observations required")
    draws = _integer(draws, "draws", 1)
    seed = _integer(seed, "seed", 0)
    lag = min(_integer(hac_lags, "hac_lags", 0), total - 1)
    try:
        requested_blocks = tuple(_integer(x, "block", 1) for x in blocks)
    except TypeError as exc:
        raise ValueError("A nonempty sequence of integer blocks required") from exc
    if (
        not requested_blocks
        or len(set(requested_blocks)) != len(requested_blocks)
        or any(block > total for block in requested_blocks)
    ):
        raise ValueError("Unique nonempty blocks no larger than the full calendar required")

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = float(_finite(np.mean(values[selected]), "selected mean"))
            influence = np.zeros(total, dtype=np.float64)
            influence[selected] = (values[selected] - mean) / (count / total)
            _finite(influence, "influence")
            long_variance = float(np.dot(influence, influence) / total)
            for distance in range(1, lag + 1):
                covariance = float(np.dot(influence[distance:], influence[:-distance]) / total)
                long_variance += 2 * (1 - distance / (lag + 1)) * covariance
            _finite(long_variance, "HAC variance")
            if long_variance < 0:
                raise ValueError("Negative computed HAC variance")
            se = math.sqrt(long_variance / total)
            hac_p = float(2 * norm.sf(abs(mean) / se)) if se else float(mean == 0)
            hac_interval = _finite([mean - 1.96 * se, mean + 1.96 * se], "HAC interval")
            mde = float(_finite((norm.ppf(0.975) + norm.ppf(0.8)) * se, "nominal MDE"))
            weighted = np.zeros(total, dtype=np.float64)
            weighted[selected] = values[selected]
            block_results = {
                str(block): _bootstrap(weighted, selected, mean, block, draws, seed)
                for block in requested_blocks
            }
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("Nonfinite or overflowing inference arithmetic") from exc

    return {
        "mean": mean,
        "n": count,
        "full_calendar_n": total,
        "hac": {
            "se": se,
            "p": hac_p,
            "ci95": [float(x) for x in hac_interval],
            "mde80_nominal": mde,
        },
        "block_inference": block_results,
        "p_conservative": max(hac_p, *(entry["p"] for entry in block_results.values())),
    }
