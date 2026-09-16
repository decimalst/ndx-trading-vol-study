"""Independent QLIKE, uncertainty, gate and calibration reconstruction.

No producer module, historical reader, model or publishing code is imported.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy.stats import norm

CONTROLS = ("matched", "market")
PHASES = ("development", "evaluation")
P_FIELDS = {"p", "p_conservative", "p_holm_wave", "p_holm_cumulative"}
REQUIRED_PANEL = {
    "origin",
    "model",
    "prediction",
    "y",
    "loss",
    "target_end",
    "phase",
    "claim_reference_week",
    "offset",
    "fit_origin",
    "training_cutoff",
    "train_n",
    "train_releases",
}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid integer {name}")
    return value


def _real(value, name, lower=None, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"Invalid real {name}")
    if (
        not np.isfinite(value)
        or (lower is not None and value < lower)
        or (upper is not None and value > upper)
    ):
        raise ValueError(f"Invalid real {name}")
    return float(value)


def _holm(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Finite probability vector required")
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def _bootstrap_means(values, block, draws, seed):
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Finite bootstrap observations required")
    _integer(block, "block")
    _integer(draws, "draws")
    _integer(seed, "seed", 0)
    n = len(x)
    if block > n:
        raise ValueError("INSUFFICIENT_DATA: block exceeds retained observations")
    rng = np.random.default_rng(seed)
    full, remainder = divmod(n, block)
    starts = rng.integers(0, n, size=(draws, full))
    tails = rng.integers(0, n, size=draws) if remainder else None
    extended = np.concatenate((x, x[:block]), axis=0)
    prefix = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(extended, axis=0)))
    block_sums = prefix[np.arange(n) + block] - prefix[np.arange(n)]
    result = np.empty((draws, x.shape[1]), dtype=float)
    # Keep the prescribed complete RNG draw order while bounding temporaries.
    for begin in range(0, draws, 4096):
        end = min(begin + 4096, draws)
        for column in range(x.shape[1]):
            result[begin:end, column] = block_sums[starts[begin:end], column].sum(axis=1)
        if remainder:
            result[begin:end] += (
                prefix[tails[begin:end] + remainder] - prefix[tails[begin:end]]
            )
    return result / n


def _bootstrap_p(means, observed):
    samples, center = np.asarray(means, dtype=float), np.asarray(observed, dtype=float)
    if samples.ndim != 2 or center.shape != (samples.shape[1],) or not len(samples):
        raise ValueError("Bootstrap means and observed contrast dimensions disagree")
    if not np.isfinite(samples).all() or not np.isfinite(center).all():
        raise ValueError("Finite bootstrap means required")
    return (1 + (np.abs(samples - center) >= np.abs(center)).sum(axis=0)) / (len(samples) + 1)


def _hac(values, lags):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite HAC vector required")
    _integer(lags, "HAC lags", 0)
    n = len(x)
    mean = float(x.mean())
    centered = x - mean
    lag = min(lags, n - 1)
    longvar = float(centered @ centered / n)
    for k in range(1, lag + 1):
        longvar += float(2 * (1 - k / (lag + 1)) * (centered[k:] @ centered[:-k]) / n)
    se = float(np.sqrt(max(longvar, 0) / n))
    return {
        "se": se,
        "p": float(2 * norm.sf(abs(mean) / se)) if se > 0 else float(mean == 0),
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "mde80_nominal": float((norm.ppf(0.975) + norm.ppf(0.8)) * se),
    }


def _calendar(calendar):
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or not len(calendar)
        or calendar.tz is not None
        or calendar.hasnans
        or calendar.has_duplicates
        or not calendar.is_monotonic_increasing
        or not calendar.equals(calendar.normalize())
    ):
        raise ValueError("Complete unique increasing naive midnight calendar required")


def _offsets(origins, calendar):
    _calendar(calendar)
    positions = calendar.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("Every origin must be in the full calendar")
    return positions % 5


def _dates(values):
    try:
        result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid panel date") from error
    if result.tz is not None or result.hasnans or not result.equals(result.normalize()):
        raise ValueError("Finite naive midnight panel dates required")
    return result


def _pair(values):
    if type(values) is not list or len(values) != 2 or any(type(x) is not str for x in values):
        raise ValueError("Two ISO phase bounds required")
    dates = _dates(values)
    if any(x.strftime("%Y-%m-%d") != y for x, y in zip(dates, values)) or dates[0] > dates[1]:
        raise ValueError("Ordered ISO phase bounds required")
    return dates[0], dates[1]


def _config(protocol):
    if type(protocol) is not dict or type(protocol.get("inference")) is not dict:
        raise ValueError("Inference protocol required")
    inf = protocol["inference"]
    blocks = inf.get("blocks")
    if type(blocks) is not list or not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("Distinct bootstrap block lengths required")
    for block in blocks:
        _integer(block, "block")
    for key in (
        "bootstrap_draws",
        "minimum_phase_observations",
        "minimum_phase_releases",
        "minimum_slice_observations",
        "minimum_slice_releases",
        "minimum_offset_observations",
    ):
        _integer(inf.get(key), key)
    _integer(inf.get("hac_lags"), "hac_lags", 0)
    _integer(inf.get("seed"), "seed", 0)
    if _real(inf.get("effect_threshold_absolute"), "effect threshold", 0) == 0:
        raise ValueError("Positive effect threshold required")
    for key in ("wave_alpha", "cumulative_alpha"):
        if _real(inf.get(key), key, 0, 1) == 0:
            raise ValueError("Positive significance thresholds required")
    forecast = protocol.get("forecast")
    if type(forecast) is not dict:
        raise ValueError("Forecast bounds required")
    bounds = {name: _pair(forecast.get(name)) for name in PHASES}
    source_end = _pair([forecast.get("source_end"), forecast.get("source_end")])[0]
    if bounds["development"][1] >= bounds["evaluation"][0] or any(
        end > source_end for _, end in bounds.values()
    ):
        raise ValueError("Distinct ordered phases within source ceiling required")
    slices = protocol.get("evaluation_stability")
    if type(slices) is not list or len(slices) != 2:
        raise ValueError("Two fixed evaluation stability slices required")
    slice_bounds = [_pair(s) for s in slices]
    if slice_bounds[0][1] >= slice_bounds[1][0]:
        raise ValueError("Nonoverlapping stability slices required")
    return inf, bounds, source_end, slice_bounds


def _panel(panel, calendar, bounds, source_end):
    _calendar(calendar)
    if (calendar > source_end).any():
        raise ValueError("Calendar exceeds source ceiling")
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.columns.has_duplicates
        or not set(panel.columns) >= REQUIRED_PANEL
    ):
        raise ValueError("Complete scored prediction panel required")
    if panel.empty:
        raise ValueError("INSUFFICIENT_DATA: empty scored panel")
    p = panel.copy(deep=True)
    date_fields = [
        "origin",
        "target_end",
        "claim_reference_week",
        "fit_origin",
        "training_cutoff",
    ]
    if "claim_release_date" in p:
        date_fields.append("claim_release_date")
    for field in date_fields:
        p[field] = _dates(p[field])
    if set(p.model) != {"candidate", *CONTROLS} or not set(p.phase) <= set(PHASES):
        raise ValueError("Only the three declared models and two phases are permitted")
    if p.duplicated(["origin", "model"]).any() or not (p.groupby("origin").size() == 3).all():
        raise ValueError("Exactly one prediction per arm and common origin required")
    for field in ("prediction", "y", "loss"):
        if not pd.api.types.is_numeric_dtype(p[field].dtype) or pd.api.types.is_bool_dtype(
            p[field].dtype
        ):
            raise ValueError("Finite numerical predictions, targets and losses required")
        if not np.isfinite(p[field].to_numpy(dtype=float)).all():
            raise ValueError("Finite numerical predictions, targets and losses required")
    if (p.prediction <= 0).any() or (p.y <= 0).any():
        raise ValueError("QLIKE requires positive target and prediction")
    shared = [
        "y",
        "target_end",
        "phase",
        "claim_reference_week",
        "offset",
        "fit_origin",
        "training_cutoff",
        "train_n",
        "train_releases",
    ]
    if "claim_release_date" in p:
        shared.append("claim_release_date")
    for field in shared:
        if (
            p[field].isna().any()
            or not (p.groupby("origin")[field].nunique(dropna=False) == 1).all()
        ):
            raise ValueError(f"Common-arm metadata mismatch: {field}")
    positions = calendar.get_indexer(p.origin)
    if (positions < 0).any() or (positions + 5 >= len(calendar)).any():
        raise ValueError("Scored origin lacks its complete five-session calendar target")
    if not np.array_equal(p.target_end.to_numpy(), calendar[positions + 5].to_numpy()):
        raise ValueError("Saved target endpoint differs from the fifth full-calendar session")
    if not pd.api.types.is_integer_dtype(p.offset.dtype) or pd.api.types.is_bool_dtype(
        p.offset.dtype
    ):
        raise ValueError("Integer full-calendar offsets required")
    if not np.array_equal(p.offset.to_numpy(), positions % 5):
        raise ValueError("Offsets must precede cohort filtering")
    for name, (start, end) in bounds.items():
        chosen = p.phase == name
        ceiling = end if name == "development" else source_end
        if ((p.loc[chosen, "origin"] < start) | (p.loc[chosen, "origin"] > end)).any() or (
            p.loc[chosen, "target_end"] > ceiling
        ).any():
            raise ValueError("Phase origin or target maturity fence violated")
    ratio = p.y.to_numpy(dtype=float) / p.prediction.to_numpy(dtype=float)
    loss = ratio - np.log(ratio) - 1
    if not np.isfinite(loss).all() or not np.allclose(p.loss, loss, atol=1e-10, rtol=1e-8):
        raise ValueError("Saved QLIKE loss does not match targets and predictions")
    p["loss"] = loss
    return p.sort_values(["origin", "model"])


def _support(n, releases, minimum_n, minimum_releases, label):
    if n < minimum_n or releases < minimum_releases:
        raise ValueError(f"INSUFFICIENT_DATA: {label}")


def _expected_scores(panel, calendar, prior, protocol):
    inf, bounds, source_end, slices = _config(protocol)
    p = _panel(panel, calendar, bounds, source_end)
    if type(prior) is not list or len(prior) != 140:
        raise ValueError("Exactly 140 unchanged inherited hypotheses required")
    old_p = []
    for row in prior:
        if type(row) is not dict or "p_conservative" not in row:
            raise ValueError("Inherited hypothesis probability required")
        old_p.append(_real(row["p_conservative"], "inherited probability", 0, 1))
    common = p[p.model == "candidate"].set_index("origin").sort_index()
    controls = {name: p[p.model == name].set_index("origin").sort_index() for name in CONTROLS}
    rows = [
        {
            "study": "claims_release",
            "candidate": "candidate",
            "control": name,
            "horizon": 5,
            "score": "qlike",
            "phases": [],
        }
        for name in CONTROLS
    ]
    for phase_code, phase_name in enumerate(PHASES):
        phase = common[common.phase == phase_name]
        n, releases = len(phase), int(phase.claim_reference_week.nunique())
        _support(
            n,
            releases,
            max(inf["minimum_phase_observations"], max(inf["blocks"])),
            inf["minimum_phase_releases"],
            phase_name,
        )
        origins = phase.index
        offset = _offsets(origins, calendar)
        for k in range(5):
            _support(
                int((offset == k).sum()),
                0,
                inf["minimum_offset_observations"],
                0,
                f"{phase_name} offset {k}",
            )
        if phase_name == "evaluation":
            for start, end in slices:
                selected = phase.loc[(origins >= start) & (origins <= end)]
                _support(
                    len(selected),
                    int(selected.claim_reference_week.nunique()),
                    inf["minimum_slice_observations"],
                    inf["minimum_slice_releases"],
                    "evaluation slice",
                )
        candidate_loss = phase.loss.to_numpy(dtype=float)
        control_losses = np.column_stack(
            [controls[name].loc[origins, "loss"].to_numpy(dtype=float) for name in CONTROLS]
        )
        delta = candidate_loss[:, None] - control_losses
        means = delta.mean(axis=0)
        bootstrap = {}
        for block in inf["blocks"]:
            samples = _bootstrap_means(
                delta,
                block,
                inf["bootstrap_draws"],
                inf["seed"] + 5_000_000 + phase_code * 10_000 + block,
            )
            bootstrap[str(block)] = (
                _bootstrap_p(samples, means),
                np.quantile(samples, [0.025, 0.975], axis=0),
            )
        for j, row in enumerate(rows):
            values = delta[:, j]
            hac = _hac(values, inf["hac_lags"])
            blocks = {
                key: {"p": float(probabilities[j]), "ci95": [float(ci[0, j]), float(ci[1, j])]}
                for key, (probabilities, ci) in bootstrap.items()
            }
            start, end = bounds[phase_name]
            bounded_calendar = calendar[(calendar >= start) & (calendar <= end)]
            annual = []
            for year in sorted(set(bounded_calendar.year)):
                selected = origins.year == year
                count = int(selected.sum())
                calendar_count = int((bounded_calendar.year == year).sum())
                annual.append(
                    {
                        "year": int(year),
                        "n": count,
                        "distinct_releases": int(
                            phase.loc[selected, "claim_reference_week"].nunique()
                        ),
                        "delta": float(values[selected].mean()) if count else None,
                        "calendar_origins": calendar_count,
                        "missing_origins": calendar_count - count,
                    }
                )
            stability = []
            if phase_name == "evaluation":
                for a, z in protocol["evaluation_stability"]:
                    selected = (origins >= pd.Timestamp(a)) & (origins <= pd.Timestamp(z))
                    stability.append(
                        {
                            "start": a,
                            "end": z,
                            "n": int(selected.sum()),
                            "distinct_releases": int(
                                phase.loc[selected, "claim_reference_week"].nunique()
                            ),
                            "delta": float(values[selected].mean()),
                        }
                    )
            release_means = (
                pd.Series(values, index=phase.claim_reference_week.to_numpy())
                .groupby(level=0)
                .mean()
            )
            row["phases"].append(
                {
                    "name": phase_name,
                    "n": n,
                    "distinct_releases": releases,
                    "delta": float(means[j]),
                    "candidate_loss": float(candidate_loss.mean()),
                    "control_loss": float(control_losses[:, j].mean()),
                    "first_origin": origins[0].strftime("%Y-%m-%d"),
                    "last_origin": origins[-1].strftime("%Y-%m-%d"),
                    "block_inference": blocks,
                    "hac": hac,
                    "p_conservative": max([hac["p"], *[b["p"] for b in blocks.values()]]),
                    "ci95_envelope": [
                        min([hac["ci95"][0], *[b["ci95"][0] for b in blocks.values()]]),
                        max([hac["ci95"][1], *[b["ci95"][1] for b in blocks.values()]]),
                    ],
                    "release_weighted_delta": float(release_means.mean()),
                    "annual": annual,
                    "offsets": [
                        {
                            "offset": k,
                            "n": int((offset == k).sum()),
                            "delta": float(values[offset == k].mean()),
                        }
                        for k in range(5)
                    ],
                    "stability": stability,
                }
            )
    probabilities = [max(phase["p_conservative"] for phase in row["phases"]) for row in rows]
    wave, cumulative = _holm(probabilities), _holm(old_p + probabilities)[-2:]
    for i, row in enumerate(rows):
        passes = wave[i] < inf["wave_alpha"] and cumulative[i] < inf["cumulative_alpha"]
        for phase in row["phases"]:
            passes &= (
                phase["delta"] <= -inf["effect_threshold_absolute"]
                and phase["release_weighted_delta"] < 0
            )
            passes &= all(item["delta"] < 0 for item in phase["offsets"] + phase["stability"])
        row.update(
            p_conservative=float(probabilities[i]),
            p_holm_wave=float(wave[i]),
            p_holm_cumulative=float(cumulative[i]),
            verdict="COMPARISON_GATE_PASS" if passes else "DOES_NOT_QUALIFY",
        )
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["claims_first_report"]
        if all(r["verdict"] == "COMPARISON_GATE_PASS" for r in rows)
        else [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 142,
        "common_scored_origins": len(common),
    }


def _compare(actual, expected, path="root", *, exact=False):
    if type(expected) is dict:
        if type(actual) is not dict or set(actual) != set(expected):
            raise ValueError(f"Verification dictionary schema mismatch: {path}")
        for key, value in expected.items():
            _compare(
                actual[key],
                value,
                f"{path}.{key}",
                exact=exact or key in P_FIELDS or key == "inherited_rows",
            )
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError(f"Verification list mismatch: {path}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            _compare(a, e, f"{path}[{i}]", exact=exact)
    elif isinstance(expected, (float, np.floating)):
        a = _real(actual, path)
        if (
            (a != float(expected))
            if exact
            else (not np.isclose(a, expected, atol=1e-10, rtol=1e-8))
        ):
            raise ValueError(f"Verification numerical mismatch: {path}")
    else:
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Verification exact mismatch: {path}")


def verify_scores(panel, calendar, metrics, prior, protocol):
    """Reconstruct every reported field; support failures retain whole-wave status."""
    expected = _expected_scores(panel, calendar, prior, protocol)
    _compare(metrics, expected)
    return {
        "status": "VERIFIED",
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 142,
        "common_scored_origins": expected["common_scored_origins"],
    }


def verify_calibration(protocol, result):
    """Independently repeat the declared serial-null experiment, without I/O."""
    inf, _, _, _ = _config(protocol)
    settings = protocol.get("calibration")
    keys = {
        "rho",
        "n",
        "burn_in",
        "replications",
        "bootstrap_draws",
        "minimum_envelope_coverage",
    }
    if type(settings) is not dict or set(settings) != keys:
        raise ValueError("Exact calibration design required")
    rho = _real(settings["rho"], "rho", -1, 1)
    if abs(rho) == 1:
        raise ValueError("Stationary calibration rho required")
    for name in ("n", "replications", "bootstrap_draws"):
        _integer(settings[name], name)
    _integer(settings["burn_in"], "burn_in", 0)
    threshold = _real(settings["minimum_envelope_coverage"], "minimum coverage", 0, 1)
    seed = inf["seed"]
    rng = np.random.default_rng(seed)
    total = 0
    block_hits = {str(block): 0 for block in inf["blocks"]}
    for trial in range(settings["replications"]):
        shocks = rng.normal(
            scale=np.sqrt(1 - rho**2), size=settings["n"] + settings["burn_in"]
        )
        state = rng.normal()
        values = []
        for shock in shocks:
            state = rho * state + shock
            values.append(state)
        retained = np.asarray(values[settings["burn_in"] :])
        lo, hi = _hac(retained, inf["hac_lags"])["ci95"]
        for block in inf["blocks"]:
            means = _bootstrap_means(
                retained, block, settings["bootstrap_draws"], seed + trial * 1000 + block
            )[:, 0]
            a, b = np.quantile(means, [0.025, 0.975])
            block_hits[str(block)] += int(a <= 0 <= b)
            lo, hi = min(lo, a), max(hi, b)
        total += int(lo <= 0 <= hi)
    coverage = total / settings["replications"]
    expected = {
        "status": "PASS" if coverage >= threshold else "FAIL",
        **settings,
        "seed": seed,
        "envelope_coverage": coverage,
        "individual_block_coverage": {
            key: hits / settings["replications"] for key, hits in block_hits.items()
        },
        "limitation": "One stationary synthetic null; not market-data coverage certification",
    }
    _compare(result, expected, exact=True)
    return {"status": "VERIFIED"}
