"""Independent wave28 score arithmetic and full-calendar inference audit.

The preceding forecast verifier establishes archive completeness, issued-fit
chronology, correlations and full row densities. This verifier independently
checks normalized marginal Jacobians, all seven contrast identities, the common
scored calendar, diagnostics and inference. It never imports the producer's
contrast builder, evaluator or prefix-sum bootstrap.
"""

import copy
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.verify_claims_release_scores import _holm, _integer, _real
from src.verify_joint_copula_scores import _normal_coordinates
from src.verify_treasury_dealer_scores import _indexed_inference

CELLS = ("orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8")
CONTRASTS = (
    "original_gap",
    "calibrated_gap",
    "interaction",
    "gaussian_calibration",
    "t8_calibration",
    "qqq_calibration",
    "spx_calibration",
)
PHASES = ("development", "evaluation")
ASSETS = ("qqq", "spx")


def _compare(actual, expected, path="metrics"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not set(expected) <= set(actual):
            raise ValueError("Missing independent verification fields: " + path)
        for key, value in expected.items():
            _compare(actual[key], value, path + "." + str(key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("Independent list length mismatch: " + path)
        for i, value in enumerate(expected):
            _compare(actual[i], value, path + "." + str(i))
    elif expected is None or isinstance(expected, str):
        if actual != expected:
            raise ValueError("Independent identity mismatch: " + path)
    elif isinstance(expected, bool):
        if not isinstance(actual, (bool, np.bool_)) or bool(actual) != expected:
            raise ValueError("Independent decision mismatch: " + path)
    elif isinstance(expected, int):
        if (
            isinstance(actual, (bool, np.bool_))
            or not isinstance(actual, (int, np.integer))
            or actual != expected
        ):
            raise ValueError("Independent count mismatch: " + path)
    else:
        observed = _real(actual, path)
        if ".block_inference." in path and path.endswith(".p"):
            if observed != expected:
                raise ValueError("Exact bootstrap count probability mismatch: " + path)
        else:
            key = path.rsplit(".", 1)[-1]
            atol = 1e-12 if key == "p" or key.startswith("p_") else 1e-10
            if not math.isclose(observed, float(expected), abs_tol=atol, rel_tol=1e-8):
                raise ValueError("Independent numerical mismatch: " + path)


def _dates(values, name):
    if not isinstance(values, pd.DatetimeIndex) and not pd.api.types.is_datetime64_any_dtype(
        values.dtype
    ):
        raise ValueError("Native dates required: " + name)
    result = pd.DatetimeIndex(values)
    if result.hasnans or result.tz is not None or not result.equals(result.normalize()):
        raise ValueError("Finite naive midnight dates required: " + name)
    return result


def _contract(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("Scientific protocol required")
    try:
        forecast, inference, support, comparisons = (
            protocol[k] for k in ("forecast", "inference", "support", "comparisons")
        )
        if not all(isinstance(x, dict) for x in (forecast, inference, support, comparisons)):
            raise ValueError("Scientific sections required")
        for key, value in {"wave": 28, "inherited": 151, "new": 7, "cumulative": 158}.items():
            if type(comparisons[key]) is not int or comparisons[key] != value:
                raise ValueError("Fixed seven-comparison family required")
        if comparisons["ordered_contrasts"] != list(CONTRASTS):
            raise ValueError("Fixed ordered contrasts required")
        for field in ("wave_alpha", "cumulative_alpha"):
            _real(comparisons[field], field, 0, 1)
        for field in ("phase_daily", "slice_daily", "offset_daily"):
            _integer(support[field], field)
        for phase in PHASES:
            bounds = forecast[phase]
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or pd.Timestamp(bounds[0]) > pd.Timestamp(bounds[1])
            ):
                raise ValueError("Ordered literal phase required")
        if pd.Timestamp(forecast["development"][1]) >= pd.Timestamp(forecast["evaluation"][0]):
            raise ValueError("Disjoint ordered phases required")
        if inference.get("phase_codes", {"development": 0, "evaluation": 1}) != {
            "development": 0,
            "evaluation": 1,
        }:
            raise ValueError("Fixed phase random-stream codes required")
        if (
            not pd.Timestamp(forecast["origin_start"])
            <= pd.Timestamp(forecast["origin_end"])
            <= pd.Timestamp(forecast["source_end"])
            <= pd.Timestamp("2025-10-20")
        ):
            raise ValueError("Bounded original issue domain required")
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete scientific protocol") from error
    return forecast, inference, support, comparisons


def _panel(panel, calendar, forecast):
    required = {"origin", "phase", "offset", "target_end", "available_date"}
    required |= {"loss_" + x for x in CELLS} | {"d_" + x for x in CONTRASTS}
    required |= {
        f"{kind}_{asset}"
        for asset in ASSETS
        for kind in (
            "y",
            "mu",
            "h",
            "a",
            "b",
            "normal_original",
            "normal_calibrated",
            "marginal_original",
            "marginal_calibrated",
            "pit_original",
            "pit_calibrated",
        )
    }
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.empty
        or panel.columns.has_duplicates
        or not required <= set(panel)
    ):
        raise ValueError("Complete scored four-cell panel required")
    calendar = _dates(calendar, "calendar")
    if (
        len(calendar) < 2
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
        or calendar[-1] > pd.Timestamp(forecast["source_end"])
    ):
        raise ValueError("Unchanged increasing bounded full calendar required")
    p = panel.copy(deep=True)
    origins = _dates(p.origin, "origin")
    if (
        not origins.is_unique
        or not origins.is_monotonic_increasing
        or (origins < forecast["origin_start"]).any()
        or (origins > forecast["origin_end"]).any()
    ):
        raise ValueError("Unique increasing bounded scored origins required")
    positions = calendar.get_indexer(origins)
    if (positions < 1).any() or (positions + 1 >= len(calendar)).any():
        raise ValueError("Every origin needs its original predecessor and endpoint")
    for name in ("target_end", "available_date"):
        if not _dates(p[name], name).equals(calendar[positions + 1]):
            raise ValueError("Actual next-calendar-session outcome required")
    if (
        p.offset.dtype.kind not in "iu"
        or p.offset.isna().any()
        or not np.array_equal(p.offset.to_numpy(), positions % 5)
    ):
        raise ValueError("Original global offsets required")
    if "horizon" in p and (p.horizon.dtype.kind not in "iu" or not p.horizon.eq(1).all()):
        raise ValueError("Next-session horizon required")
    admitted = np.zeros(len(p), dtype=bool)
    for phase in PHASES:
        start, end = forecast[phase]
        selected = (origins >= start) & (origins <= end)
        if (
            not p.loc[selected, "phase"].eq(phase).all()
            or (p.loc[selected, "target_end"] > pd.Timestamp(end)).any()
        ):
            raise ValueError("Literal phase labels and endpoints required")
        admitted |= selected
    if not admitted.all():
        raise ValueError("Out-of-phase panel row")
    for name in required - {"origin", "phase", "offset", "target_end", "available_date"}:
        if p[name].dtype.kind not in "fiu" or not np.isfinite(p[name].to_numpy(float)).all():
            raise ValueError("Finite real panel column required: " + name)
    return p, calendar, positions


def _marginals(panel):
    output = {}
    constant = math.lgamma(4.5) - math.lgamma(4) - 0.5 * math.log(8 * math.pi)
    for asset in ASSETS:
        h = panel["h_" + asset].to_numpy(float)
        a = panel["a_" + asset].to_numpy(float)
        b = panel["b_" + asset].to_numpy(float)
        if (h <= 0).any() or (b <= 1e-12).any():
            raise ValueError(
                "Positive original variance and admissible calibration scale required"
            )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            z = (
                panel["y_" + asset].to_numpy(float) - panel["mu_" + asset].to_numpy(float)
            ) / (np.sqrt(h) * math.sqrt(0.75))
            w = _normal_coordinates(z)
            v = (w - a) / b
            with np.errstate(divide="ignore"):
                shape = np.logaddexp(0.0, 2 * np.log(np.abs(z)) - math.log(8))
            original = constant - 4.5 * shape - 0.5 * (np.log(h) + math.log(0.75))
            calibrated = original + 0.5 * (w * w - v * v) - np.log(b)
        for system, normal, margin in (
            ("original", w, original),
            ("calibrated", v, calibrated),
        ):
            pit = ndtr(normal)
            for kind, value in (("normal", normal), ("marginal", margin), ("pit", pit)):
                observed = panel[f"{kind}_{system}_{asset}"].to_numpy(float)
                if not np.isfinite(value).all() or not np.allclose(
                    observed, value, atol=1e-10, rtol=1e-8
                ):
                    raise ValueError(
                        "Independent normalized marginal mismatch: "
                        + kind
                        + "_"
                        + system
                        + "_"
                        + asset
                    )
            output[(asset, system)] = {"normal": normal, "marginal": margin, "pit": pit}
    return output


def _differences(panel, margins):
    loss = {c: panel["loss_" + c].to_numpy(float) for c in CELLS}
    original = loss["orig_t8"] - loss["orig_gaussian"]
    calibrated = loss["cal_t8"] - loss["cal_gaussian"]
    d = dict(
        zip(
            CONTRASTS,
            (
                original,
                calibrated,
                calibrated - original,
                loss["cal_gaussian"] - loss["orig_gaussian"],
                loss["cal_t8"] - loss["orig_t8"],
                margins[("qqq", "original")]["marginal"]
                - margins[("qqq", "calibrated")]["marginal"],
                margins[("spx", "original")]["marginal"]
                - margins[("spx", "calibrated")]["marginal"],
            ),
            strict=True,
        )
    )
    for name, value in d.items():
        if not np.isfinite(value).all() or not np.allclose(
            panel["d_" + name], value, atol=1e-10, rtol=1e-8
        ):
            raise ValueError("Independent contrast identity mismatch: " + name)
    return d


def verify_scores(panel, calendar, metrics, prior, protocol):
    """Verify a completed four-cell diagnostic; failure receipts are not scored."""
    forecast, inf, support, comparisons = _contract(protocol)
    if not isinstance(prior, list) or len(prior) != 151:
        raise ValueError("Exactly151 inherited records required")
    prior_p = []
    for row in prior:
        if not isinstance(row, dict):
            raise ValueError("Inherited dictionaries required")
        prior_p.append(_real(row.get("p_conservative"), "inherited probability", 0, 1))
    if not isinstance(metrics, dict) or metrics.get("inherited_rows") != prior:
        raise ValueError("Exact preserved inherited records required")
    p, calendar, positions = _panel(panel, calendar, forecast)
    try:
        margins = _marginals(p)
        differences = _differences(p, margins)
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("Invalid independent density arithmetic") from error
    phase_masks = {phase: p.phase.to_numpy() == phase for phase in PHASES}
    for phase, selected in phase_masks.items():
        if selected.sum() < support["phase_daily"] or any(
            np.sum(selected & (positions % 5 == k)) < support["offset_daily"] for k in range(5)
        ):
            raise ValueError("INSUFFICIENT_DATA: phase or offset support")
    for start, end in forecast["stability"]:
        if p.origin.between(start, end).sum() < support["slice_daily"]:
            raise ValueError("INSUFFICIENT_DATA: stability support")
    rows = []
    for name in CONTRASTS:
        phases = []
        for code, phase in enumerate(PHASES):
            selected = phase_masks[phase]
            frame = p.loc[selected]
            diff = differences[name][selected]
            start, end = forecast[phase]
            full = calendar[(calendar >= start) & (calendar <= end)]
            values = (
                pd.Series(diff, index=pd.DatetimeIndex(frame.origin)).reindex(full).to_numpy()
            )
            result = _indexed_inference(
                values,
                full.isin(frame.origin),
                blocks=inf["blocks"],
                hac_lags=inf["hac_lags"],
                draws=inf["bootstrap_draws"],
                seed=inf["seed"] + code * 10000,
            )
            intervals = [
                result["hac"]["ci95"],
                *[r["ci95"] for r in result["block_inference"].values()],
            ]

            def group(mask, diff=diff):
                subset = diff[np.asarray(mask)]
                return {
                    "n": len(subset),
                    "mean": float(subset.mean()) if len(subset) else None,
                }

            result.update(
                name=phase,
                ci95_envelope=[min(x[0] for x in intervals), max(x[1] for x in intervals)],
                offsets=[{"offset": k, **group(frame.offset == k)} for k in range(5)],
                stability=[
                    {"start": a, "end": b, **group(frame.origin.between(a, b))}
                    for a, b in forecast["stability"]
                ]
                if phase == "evaluation"
                else [],
                annual=[
                    {"year": int(y), **group(frame.origin.dt.year == y)}
                    for y in sorted(set(full.year))
                ],
            )
            phases.append(result)
        rows.append(
            {
                "contrast": name,
                "phases": phases,
                "p_conservative": max(r["p_conservative"] for r in phases),
            }
        )
    wave = _holm([r["p_conservative"] for r in rows])
    cumulative = _holm(prior_p + [r["p_conservative"] for r in rows])[-7:]
    for row, pw, pc in zip(rows, wave, cumulative, strict=True):
        means = [x["mean"] for x in row["phases"]]
        same_sign = all(x > 0 for x in means) or all(x < 0 for x in means)
        row.update(
            p_holm_wave=float(pw),
            p_holm_cumulative=float(pc),
            adjusted_difference_detected=bool(
                same_sign
                and pw <= comparisons["wave_alpha"]
                and pc <= comparisons["cumulative_alpha"]
            ),
        )
    diagnostics, joint = {}, {}
    for phase, selected in phase_masks.items():
        joint[phase] = {c: float(p.loc[selected, "loss_" + c].mean()) for c in CELLS}
        diagnostics[phase] = {}
        for asset in ASSETS:
            diagnostics[phase][asset] = {}
            for system in ("original", "calibrated"):
                m = margins[(asset, system)]
                w = m["normal"][selected]
                pit = m["pit"][selected]
                mean = float(w.mean())
                variance = float(np.mean((w - mean) ** 2))
                if not np.isfinite(variance) or variance <= 0:
                    raise ValueError("Undefined held-out normal-score skewness")
                diagnostics[phase][asset][system] = {
                    "mean_loss": float(-m["marginal"][selected].mean()),
                    "pit_lower": float(np.mean(pit < 0.025)),
                    "pit_upper": float(np.mean(pit > 0.975)),
                    "normal_mean": mean,
                    "normal_variance": variance,
                    "normal_skew": float(np.mean((w - mean) ** 3) / variance**1.5),
                }
    expected = {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "hypothesis_count": 7,
        "cumulative_hypothesis_count": 158,
        "leads": [],
        "common_scored_origins": len(p),
        "calibration_diagnostics": diagnostics,
        "full_joint_losses": joint,
    }
    _compare(metrics, expected)
    return {
        "status": "VERIFIED",
        "contrasts_verified": 7,
        "phases_verified": 2,
        "common_scored_origins": len(p),
        "inherited_hypotheses_verified": 151,
        "density_scope": "independent normalized marginals and seven contrast identities; full joint fits covered by forecast verifier",
        "inference": "independent explicit circular-index tables and Bartlett HAC; shared phase streams; Holm7/Holm158",
    }
