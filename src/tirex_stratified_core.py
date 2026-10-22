"""Fresh balanced motif generator and descriptive activation-swap statistics.

Calibration and validation share the previous group's *validation* construction
law. They differ only in their immutable seed and identifiers. The old group
discovery/validation generator mismatch remains preserved in its original files.
"""

from numbers import Integral

import numpy as np
from scipy.stats import binomtest, norm, t

SEEDS = {"calibration": 837194, "validation": 529086}
FAMILIES = ("persistence", "recency", "recovery")
WINDOWS = {"short": 76, "long": 180}


def _integer(value, minimum, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def make_stratified_pairs(split, per_cell=320, length=2048):
    """Make exactly ``per_cell`` pairs in each family-by-motif-length cell.

    Each pair permutes one terminal motif while sharing its histogram, last
    observation, background and affine normalization. Motif width is always 76
    or 180 observations; context length must accommodate the longer motif.
    Amplitude and noise vary under the same law in both splits. Parameter
    metadata are saved to make that law auditable rather than inferred from
    normalized histories.
    """
    if split not in SEEDS:
        raise ValueError("split must be calibration or validation")
    _integer(per_cell, 1, "per_cell")
    _integer(length, 192, "length")
    rng = np.random.default_rng(SEEDS[split])
    result = {key: [] for key in ("recipient", "donor", "family", "variant", "pair_id",
                                  "window", "amplitude", "noise_sd")}
    for family in FAMILIES:
        for variant, window in WINDOWS.items():
            for index in range(per_cell):
                innovations = rng.normal(0, .3, size=length)
                background = np.zeros(length)
                for time in range(1, length):
                    background[time] = .92 * background[time - 1] + innovations[time]
                amplitude = float(rng.uniform(.8, 2.8))
                noise_sd = float(rng.uniform(.04, .3))
                recipient, donor = background.copy(), background.copy()
                start, end = length - 1 - window, length - 1
                if family == "persistence":
                    low = -amplitude + rng.normal(0, noise_sd, window // 2)
                    high = amplitude + rng.normal(0, noise_sd, window // 2)
                    recipient[start:end] = np.column_stack((low, high)).ravel()
                    donor[start:end] = np.concatenate((low, high))
                elif family == "recency":
                    burst = max(2, window // 4)
                    quiet = rng.normal(0, noise_sd, window - burst)
                    high = amplitude + rng.normal(0, noise_sd, burst)
                    tail = max(2, window // 6)
                    recipient[start:end] = np.concatenate((high, quiet))
                    donor[start:end] = np.concatenate((quiet[:-tail], high, quiet[-tail:]))
                else:
                    ramp = np.linspace(-amplitude, amplitude, window) + rng.normal(0, noise_sd, window)
                    recipient[start:end] = ramp
                    donor[start:end] = ramp[::-1]
                recipient[-1] = donor[-1] = 0.
                location, scale = recipient.mean(), recipient.std()
                recipient = -9. + .6 * (recipient - location) / scale
                donor = -9. + .6 * (donor - location) / scale
                for key, value in (("recipient", recipient), ("donor", donor), ("family", family),
                                   ("variant", variant), ("window", window), ("amplitude", amplitude),
                                   ("noise_sd", noise_sd),
                                   ("pair_id", f"stratified-{split}-{family}-{variant}-{index:04d}")):
                    result[key].append(value)
    return {**{key: np.asarray(value) for key, value in result.items()}, "split": split}


def _nonnegative(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _forecasts(values, name, minimum=2):
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric forecasts") from exc
    if values.ndim != 3 or values.shape[1:] != (2, 3) or len(values) < minimum:
        raise ValueError(f"{name} must have shape (N >= {minimum}, 2, 3)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite numeric forecasts")
    return values


def summarize_swaps(base, swapped, deadband, seed=0, n_bootstrap=5000):
    """Summarize one already selected family-by-motif cell, without pooling.

    Inputs have pair, side and quantile axes. Side zero is recipient; side one
    donor. Quantiles are q10/q50/q90. The raw paired effect e is half the recipient
    shift minus the donor shift, with no direction learned on discovery.

    The primary mean test targets a mean outside the calibrated numerical band
    [-deadband, deadband]. It is a two-sided, conservative t approximation, not
    a claim about semantic specificity, superiority over singletons or practical
    forecast value. A caller must apply the declared familywise adjustment.

    All bootstrap intervals are paired, pointwise and descriptive. Sign counts
    retain neutral pairs in the total; their descriptive exact sign test is
    conditional on a pair lying strictly outside the numerical deadband.
    """
    baseline = _forecasts(base, "base")
    intervention = _forecasts(swapped, "swapped")
    if baseline.shape != intervention.shape:
        raise ValueError("swapped must have the same shape as base")
    threshold = _nonnegative(deadband, "deadband")
    _integer(seed, 0, "seed")
    _integer(n_bootstrap, 1, "n_bootstrap")
    count = len(baseline)
    delta = intervention - baseline
    quantile_e = (delta[:, 0] - delta[:, 1]) / 2
    effects = quantile_e[:, 1]
    magnitude = np.abs(delta[:, :, 1]).mean(axis=1)
    width_delta = delta[:, :, 2] - delta[:, :, 0]
    width_e = (width_delta[:, 0] - width_delta[:, 1]) / 2
    mean, sd = float(effects.mean()), float(effects.std(ddof=1))
    se = sd / np.sqrt(count)
    if se == 0:
        primary_pvalue = float(abs(mean) <= threshold)
    else:
        primary_pvalue = float(min(1., 2 * min(t.sf((mean - threshold) / se, count - 1),
                                              t.sf((-mean - threshold) / se, count - 1))))
    positive = int(np.count_nonzero(effects > threshold))
    negative = int(np.count_nonzero(effects < -threshold))
    nonneutral = positive + negative
    samples = np.random.default_rng(seed).integers(0, count, size=(n_bootstrap, count))

    def interval(values):
        return np.quantile(values[samples].mean(axis=1), [.025, .975], axis=0).T.tolist()

    return {
        "n_pairs": count, "deadband": threshold, "seed": int(seed),
        "bootstrap_replicates": int(n_bootstrap), "primary_pvalue": primary_pvalue,
        "primary_test_scope": "raw paired mean outside calibrated numerical band; t approximation",
        "mean_e": mean, "mean_signed": mean, "mean_e_sd": sd, "mean_e_se": float(se),
        "mean_e_ci95": interval(effects), "e": effects.tolist(),
        "recipient_median_delta": delta[:, 0, 1].tolist(),
        "donor_median_delta": delta[:, 1, 1].tolist(),
        "mean_recipient_median_shift": float(delta[:, 0, 1].mean()),
        "recipient_median_ci95": interval(delta[:, 0, 1]),
        "mean_donor_median_shift": float(delta[:, 1, 1].mean()),
        "donor_median_ci95": interval(delta[:, 1, 1]),
        "mean_abs_median_shift": float(magnitude.mean()),
        "abs_median_ci95": interval(magnitude),
        "mean_native_donor_recipient_contrast": float((baseline[:, 1, 1] - baseline[:, 0, 1]).mean()),
        "mean_quantile_e": quantile_e.mean(axis=0).tolist(),
        "quantile_e_ci95": interval(quantile_e),
        "mean_recipient_quantile_shift": delta[:, 0].mean(axis=0).tolist(),
        "recipient_quantile_ci95": interval(delta[:, 0]),
        "mean_donor_quantile_shift": delta[:, 1].mean(axis=0).tolist(),
        "donor_quantile_ci95": interval(delta[:, 1]),
        "mean_width_e": float(width_e.mean()), "width_e_ci95": interval(width_e),
        "mean_recipient_width_shift": float(width_delta[:, 0].mean()),
        "recipient_width_ci95": interval(width_delta[:, 0]),
        "mean_donor_width_shift": float(width_delta[:, 1].mean()),
        "donor_width_ci95": interval(width_delta[:, 1]),
        "mean_abs_width_shift": float(np.abs(width_delta).mean()),
        "abs_width_ci95": interval(np.abs(width_delta).mean(axis=1)),
        "sign_counts": {"positive": positive, "negative": negative,
                        "neutral": count - nonneutral, "nonneutral": nonneutral, "total": count},
        "sign_pvalue": float(binomtest(positive, nonneutral, .5).pvalue) if nonneutral else 1.,
        "sign_test_scope": "descriptive; conditional on pair effect strictly outside deadband",
        "base_crossing_count": int(np.any(np.diff(baseline, axis=-1) < 0, axis=-1).sum()),
        "crossing_count": int(np.any(np.diff(intervention, axis=-1) < 0, axis=-1).sum()),
    }


def calibrate_deadband(baseline, *null_forecasts):
    """Freeze 10 times observed jitter or float32 spacing, whichever is larger.

    Null forecasts are aligned repeats, self-swaps, or alternative batch-layout
    evaluations of the same calibration histories. This estimates a numerical
    resolution floor, not practical relevance or uncertainty in forecasting.
    """
    baseline = _forecasts(baseline, "baseline", minimum=1)
    if not null_forecasts:
        raise ValueError("at least one aligned null forecast is required")
    maxima = []
    for index, values in enumerate(null_forecasts):
        values = _forecasts(values, f"null_forecasts[{index}]", minimum=1)
        if values.shape != baseline.shape:
            raise ValueError("all null forecasts must align exactly with baseline")
        maxima.append(float(np.max(np.abs(values - baseline))))
    with np.errstate(over="ignore", invalid="ignore"):
        spacing = float(np.max(np.abs(np.spacing(baseline.astype(np.float32)))))
    if not np.isfinite(spacing):
        raise ValueError("calibration forecasts must have finite float32 spacing")
    observed = max(maxima)
    return {"deadband": max(10 * observed, 10 * spacing), "multiplier": 10,
            "observed_max_jitter": observed, "max_float32_spacing": spacing,
            "null_maxima": maxima, "n_null_evaluations": len(maxima)}


def minimum_detectable_mean(calibration_sd, n=320, tau=0., claims=18, alpha=.05, power=.8):
    """Approximate two-sided mean sensitivity, assuming calibration SD transfers.

    This normal approximation uses Bonferroni alpha/claims as a conservative
    Holm planning threshold. It is not a guarantee or a retrospectively measured
    power estimate. The numerical deadband is added to the mean shift needed.
    """
    sd = _nonnegative(calibration_sd, "calibration_sd")
    threshold = _nonnegative(tau, "tau")
    _integer(n, 2, "n")
    _integer(claims, 1, "claims")
    alpha, power = _nonnegative(alpha, "alpha"), _nonnegative(power, "power")
    if not 0 < alpha < 1 or not .5 < power < 1:
        raise ValueError("alpha must be in (0,1) and power in (.5,1)")
    return float(threshold + (norm.ppf(1 - alpha / (2 * claims)) + norm.ppf(power)) * sd / np.sqrt(n))
