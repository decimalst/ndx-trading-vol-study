"""Independent issued-archive calibration, chronology and joint-density checks.

The new producer is not imported. Frozen independent copula certificates and
normal-score formulas are reused; input source authentication remains external.
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import betaln, hyp2f1, log_ndtr, ndtr, stdtrit

from src.verify_joint_copula_forecasts import verify_dependence
from src.verify_joint_copula_scores import (
    _independent_logcopula,
    _log_one_plus_square,
    _normal_coordinates,
)

ASSETS = ("qqq", "spx")
CELLS = ("orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8")
APP_SOURCE = (
    "origin",
    "phase",
    "offset",
    "mu_qqq",
    "mu_spx",
    "h_qqq",
    "h_spx",
    "fit_origin",
    "training_cutoff",
)
DATES = ("origin", "available_date", "target_end", "fit_origin", "training_cutoff")
LIMIT = pd.Timestamp("2025-10-20")
T_CONSTANT = math.lgamma(4.5) - math.lgamma(4) - 0.5 * math.log(8 * math.pi)


def _array(value, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError("Finite nonboolean real values required")
    result = raw.astype(float)
    if shape is not None and result.shape != shape:
        raise ValueError("Exact numerical shape required")
    return result


def _pairs(value):
    result = _array(value)
    if result.ndim != 2 or result.shape[1] != 2 or not len(result):
        raise ValueError("Nonempty two-asset pairs required")
    return result


def _calibration(z):
    z = _pairs(z)
    if len(z) < 2:
        raise ValueError("At least two calibration pairs required")
    w = _normal_coordinates(z)
    a = np.mean(w, axis=0)
    b = np.sqrt(np.mean(np.square(w - a), axis=0))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (b <= 1e-12).any():
        raise ValueError("Independent calibration scale exceeds fixed floor")
    return {"location": a.tolist(), "scale": b.tolist(), "train_n": len(z)}


def _from_normal(values):
    """Independently invert log t8 tails, solving in log magnitude when needed."""
    v = _array(values)
    result = np.zeros_like(v)
    for index in np.ndindex(v.shape):
        a = abs(float(v[index]))
        if a == 0:
            continue
        if a <= 8:
            answer = -float(stdtrit(8, ndtr(-a)))
        else:
            wanted = float(log_ndtr(-a))
            if not math.isfinite(wanted):
                raise ValueError("Unrepresentable normal log tail")

            def residual(logt, wanted=wanted):
                logx = -float(np.logaddexp(0.0, 2 * logt - math.log(8)))
                with np.errstate(under="ignore"):
                    x = math.exp(logx)
                log_tail = (
                    -math.log(2)
                    + 4 * logx
                    - math.log(4)
                    - betaln(4, 0.5)
                    + math.log(float(hyp2f1(4, 0.5, 5, x)))
                )
                return log_tail - wanted

            upper = math.log(np.finfo(float).max)
            if residual(upper) > 0:
                raise ValueError("Calibrated t8 coordinate cannot be represented")
            log_answer = brentq(residual, 0.0, upper, xtol=1e-13, rtol=4 * np.finfo(float).eps)
            answer = math.exp(log_answer)
        result[index] = math.copysign(answer, float(v[index]))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite independent inverse coordinate")
    return result


def _row_scores(z, h, location, scale, rhos):
    z = _pairs(z)
    h = _array(h, z.shape)
    a, b = _array(location, (2,)), _array(scale, (2,))
    correlations = _array(rhos, (4,))
    if (h <= 0).any() or (b <= 1e-12).any() or (abs(correlations) > 0.995).any():
        raise ValueError("Positive variance/scales and fixed copula domain required")
    w = _normal_coordinates(z)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            v = (w - a) / b
            log_original = (
                T_CONSTANT - 4.5 * _log_one_plus_square(z) - 0.5 * (np.log(h) + math.log(0.75))
            )
            log_calibrated = log_original - 0.5 * (v * v - w * w) - np.log(b)
        except FloatingPointError as error:
            raise ValueError("Unrepresentable normalized calibration density") from error
    zcal = _from_normal(v)
    result = {}
    for j, asset in enumerate(ASSETS):
        result["marginal_original_" + asset] = log_original[:, j]
        result["marginal_calibrated_" + asset] = log_calibrated[:, j]
        result["pit_original_" + asset] = ndtr(w[:, j])
        result["pit_calibrated_" + asset] = ndtr(v[:, j])
        result["normal_original_" + asset] = w[:, j]
        result["normal_calibrated_" + asset] = v[:, j]
        result["d_" + asset + "_calibration"] = log_original[:, j] - log_calibrated[:, j]
    for name, r, values, marginal in zip(
        CELLS,
        correlations,
        (w, z, v, zcal),
        (log_original, log_original, log_calibrated, log_calibrated),
        strict=True,
    ):
        if name.endswith("gaussian"):
            det = (1 - r) * (1 + r)
            copula = (
                -0.5 * math.log(det)
                + (
                    r * values[:, 0] * values[:, 1]
                    - 0.5 * r * r * np.square(values).sum(axis=1)
                )
                / det
            )
        else:
            copula = _independent_logcopula(values, float(r), "t8")
        result["loss_" + name] = -marginal.sum(axis=1) - copula
    result["d_original_gap"] = result["loss_orig_t8"] - result["loss_orig_gaussian"]
    result["d_calibrated_gap"] = result["loss_cal_t8"] - result["loss_cal_gaussian"]
    result["d_interaction"] = result["d_calibrated_gap"] - result["d_original_gap"]
    result["d_gaussian_calibration"] = (
        result["loss_cal_gaussian"] - result["loss_orig_gaussian"]
    )
    result["d_t8_calibration"] = result["loss_cal_t8"] - result["loss_orig_t8"]
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError("Nonfinite complete density or contrast")
    return result


def _index(value, *, missing=False):
    index = pd.DatetimeIndex(value)
    if index.tz is not None or (index.hasnans and not missing):
        raise ValueError("Native timezone-free dates required")
    known = index[~index.isna()]
    if not known.equals(known.normalize()) or (known > LIMIT).any():
        raise ValueError("Midnight source dates must remain within ceiling")
    return index


def _archive(archive, calendar):
    required = set(APP_SOURCE) | {
        "issued",
        "available_date",
        "target_end",
        "eligible_scored",
        "y_qqq",
        "y_spx",
        "old_rho_t8",
        "old_rho_gaussian",
    }
    if (
        not isinstance(archive, pd.DataFrame)
        or not archive.columns.is_unique
        or not required <= set(archive)
        or archive.empty
    ):
        raise ValueError("Complete canonical issued archive required")
    a = archive.copy(deep=True).reset_index(drop=True)
    for name in DATES:
        if not pd.api.types.is_datetime64_any_dtype(a[name].dtype):
            raise ValueError("Native archive date columns required")
        a[name] = _index(a[name], missing=name in {"target_end", "available_date"})
    origins = _index(a.origin)
    if not origins.is_unique or not origins.is_monotonic_increasing:
        raise ValueError("Unique sorted original origins required")
    for name in ("issued", "eligible_scored"):
        if a[name].dtype.kind != "b" or a[name].isna().any():
            raise ValueError("Literal complete boolean archive flags required")
    if not a.issued.all():
        raise ValueError("Canonical archive contains only genuinely issued forecasts")
    for name in ("mu_qqq", "mu_spx", "h_qqq", "h_spx", "old_rho_t8", "old_rho_gaussian"):
        _array(a[name])
    if (a[["h_qqq", "h_spx"]].to_numpy() <= 0).any() or (
        abs(a[["old_rho_t8", "old_rho_gaussian"]].to_numpy()) > 0.995
    ).any():
        raise ValueError("Valid original marginal variances and correlations required")
    for name in ("y_qqq", "y_spx"):
        if a[name].dtype.kind not in "fiu" or np.isinf(a[name]).any():
            raise ValueError("Finite observed real outcomes or explicit missingness required")
    if (
        not pd.api.types.is_integer_dtype(a.offset.dtype)
        or a.offset.dtype.kind == "b"
        or a.offset.isna().any()
        or not a.offset.between(0, 4).all()
    ):
        raise ValueError("Original global offset must be an integer from zero through four")
    if (
        not a.available_date.equals(a.target_end)
        or (a.target_end.notna() & (a.target_end <= a.origin)).any()
    ):
        raise ValueError("Original target and availability must be identical future dates")
    if (a.training_cutoff >= a.fit_origin).any() or (a.fit_origin > a.origin).any():
        raise ValueError("Strictly prior original monthly cutoff required")
    if not np.array_equal(a.origin.dt.to_period("M"), a.fit_origin.dt.to_period("M")):
        raise ValueError("Original monthly fit must be in the query month")
    expected_phase = np.where(a.origin <= "2019-12-31", "development", "evaluation")
    if not np.array_equal(a.phase.to_numpy(), expected_phase):
        raise ValueError("Original phase identities differ")
    observed = a[["y_qqq", "y_spx"]].notna().all(axis=1)
    if (a.eligible_scored & (~observed | a.target_end.isna())).any():
        raise ValueError("A scored pair must have mature observed target metadata")
    for _, group in a.groupby(a.origin.dt.to_period("M"), sort=True):
        if (
            group.fit_origin.nunique() != 1
            or group.training_cutoff.nunique() != 1
            or group.origin.iloc[0] != group.fit_origin.iloc[0]
        ):
            raise ValueError("One original first-query fit and cutoff per month required")
    if calendar is not None:
        if not isinstance(calendar, pd.DatetimeIndex):
            raise ValueError("Full original reference calendar required")
        c = _index(calendar)
        if not c.is_unique or not c.is_monotonic_increasing:
            raise ValueError("Unique increasing original reference calendar required")
        p, f = c.get_indexer(a.origin), c.get_indexer(a.fit_origin)
        if (p < 0).any() or (f < 1).any():
            raise ValueError("Original origins and fit predecessors must exist")
        if not np.array_equal(a.offset.to_numpy(), p % 5) or not pd.DatetimeIndex(
            a.training_cutoff
        ).equals(c[f - 1]):
            raise ValueError("Original offsets or previous-session fit cutoff differ")
        following = pd.DatetimeIndex([c[i + 1] if i + 1 < len(c) else pd.NaT for i in p])
        if not pd.DatetimeIndex(a.target_end).equals(following):
            raise ValueError("Original target must be the next full-calendar session")
    return a


def _compare(actual, expected, label):
    if type(expected) is dict:
        if type(actual) is not dict or set(actual) != set(expected):
            raise ValueError("Exact dictionary schema differs: " + label)
        for key in expected:
            _compare(actual[key], expected[key], label + "." + key)
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError("Ordered audit list differs: " + label)
        for i, (value, wanted) in enumerate(zip(actual, expected, strict=True)):
            _compare(value, wanted, label + "." + str(i))
    elif isinstance(expected, float):
        value = _array(actual, ()).item()
        if not math.isclose(value, expected, abs_tol=1e-10, rel_tol=1e-8):
            raise ValueError("Independent audit number differs: " + label)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError("Exact audit identity differs: " + label)


def _table(actual, expected, label):
    if (
        not isinstance(actual, pd.DataFrame)
        or not actual.columns.is_unique
        or set(actual) != set(expected)
        or len(actual) != len(expected)
    ):
        raise ValueError("Complete output table differs: " + label)
    actual = actual.reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    exact = set(APP_SOURCE) | {
        "target_end",
        "available_date",
        "y_qqq",
        "y_spx",
        "status",
        "train_n",
    }
    for name in expected:
        a, b = actual[name], expected[name]
        if pd.api.types.is_datetime64_any_dtype(b.dtype):
            if not pd.api.types.is_datetime64_any_dtype(a.dtype) or not _index(
                a, missing=True
            ).equals(_index(b, missing=True)):
                raise ValueError("Exact output dates differ: " + label + "." + name)
        elif name in exact:
            if not np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=False):
                raise ValueError(
                    "Exact copied identities or base forecasts differ: " + label + "." + name
                )
        else:
            aa = _array(a)
            if name.startswith("b_") and (aa <= 1e-12).any():
                raise ValueError("Output calibration scale outside domain")
            if name.startswith("rho_") and (abs(aa) > 0.995).any():
                raise ValueError("Output correlation outside fixed domain")
            if name.startswith("pit_") and ((aa < 0).any() or (aa > 1).any()):
                raise ValueError("Output PIT outside probability domain")
            if not np.allclose(aa, b.to_numpy(float), atol=1e-10, rtol=1e-8):
                raise ValueError("Independent output number differs: " + label + "." + name)


def _standardized(frame):
    mu = frame[["mu_qqq", "mu_spx"]].to_numpy(float)
    h = frame[["h_qqq", "h_spx"]].to_numpy(float)
    y = frame[["y_qqq", "y_spx"]].to_numpy(float)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            z = (y - mu) / (np.sqrt(h) * math.sqrt(0.75))
        except FloatingPointError as error:
            raise ValueError("Unrepresentable original issued residuals") from error
    return _pairs(z), h


def verify(archive, produced, minimum_train=252, calendar=None):
    if type(minimum_train) is not int or minimum_train < 2:
        raise ValueError("Exact training floor of at least two required")
    a = _archive(archive, calendar)
    if (
        type(produced) is not dict
        or set(produced) != {"applications", "panel", "fits", "coverage"}
        or type(produced["fits"]) is not list
    ):
        raise ValueError("Exact four-output crossed result required")
    applications, panels, coverage = [], [], []
    fit_number = 0
    gap_max = 0.0
    for _, query in a.groupby(a.origin.dt.to_period("M"), sort=True):
        fit_origin, cutoff = query.fit_origin.iloc[0], query.training_cutoff.iloc[0]
        historical = a[
            (a.origin < fit_origin)
            & (a.available_date <= cutoff)
            & (a.target_end <= cutoff)
            & a.issued
            & a[["y_qqq", "y_spx"]].notna().all(axis=1)
        ]
        if len(historical) < minimum_train:
            coverage.extend({"origin": date, "status": "warmup"} for date in query.origin)
            continue
        if fit_number >= len(produced["fits"]):
            raise ValueError("Missing eligible original monthly fit")
        fit = produced["fits"][fit_number]
        expected_keys = {
            "fit_origin",
            "training_cutoff",
            "train_origins",
            "train_n",
            "calibration",
            "dependence",
        }
        if type(fit) is not dict or set(fit) != expected_keys:
            raise ValueError("Exact eligible fit schema required")
        metadata = {
            "fit_origin": str(fit_origin.date()),
            "training_cutoff": str(cutoff.date()),
            "train_origins": historical.origin.dt.strftime("%Y-%m-%d").tolist(),
            "train_n": len(historical),
        }
        _compare({k: fit[k] for k in metadata}, metadata, "fit")
        z, _ = _standardized(historical)
        calibration = _calibration(z)
        _compare(fit["calibration"], calibration, "calibration")
        w = _normal_coordinates(z)
        v = (w - np.asarray(calibration["location"])) / np.asarray(calibration["scale"])
        zcal = _from_normal(v)
        if type(fit["dependence"]) is not dict or set(fit["dependence"]) != set(CELLS):
            raise ValueError("All four refitted dependence cells required")
        rhos = []
        for cell in CELLS:
            item = fit["dependence"][cell]
            if type(item) is not dict or set(item) != {"rho", "audit"}:
                raise ValueError("Explicit fitted parameter and complete certificate required")
            rho = _array(item["rho"], ()).item()
            certificate = verify_dependence(
                zcal if cell.startswith("cal_") else z,
                "gaussian" if cell.endswith("gaussian") else "t8",
                rho,
                item["audit"],
            )
            gap_max = max(gap_max, certificate["global_value_gap"])
            rhos.append(rho)
        app = query.loc[:, list(APP_SOURCE)].copy()
        app["train_n"] = len(historical)
        for j, asset in enumerate(ASSETS):
            app["a_" + asset] = calibration["location"][j]
            app["b_" + asset] = calibration["scale"][j]
        for cell, rho in zip(CELLS, rhos, strict=True):
            app["rho_" + cell] = rho
        applications.append(app)
        scored = query.eligible_scored
        frame = app.loc[scored].copy()
        for name in ("target_end", "available_date", "y_qqq", "y_spx"):
            frame[name] = query.loc[scored, name]
        if len(frame):
            z_query, h_query = _standardized(frame)
            for name, values in _row_scores(
                z_query, h_query, calibration["location"], calibration["scale"], rhos
            ).items():
                frame[name] = values
            panels.append(frame)
        coverage.extend(
            {
                "origin": row.origin,
                "status": "scored" if row.eligible_scored else "issued_unscored",
            }
            for row in query.itertuples()
        )
        fit_number += 1
    if fit_number != len(produced["fits"]):
        raise ValueError("Unexpected or warmup fit record")
    if not applications or not panels:
        raise ValueError("INSUFFICIENT_DATA: no supported scored crossed outputs")
    app = pd.concat(applications, ignore_index=True)
    panel = pd.concat(panels, ignore_index=True)
    _table(produced["applications"], app, "applications")
    _table(produced["panel"], panel, "panel")
    _table(produced["coverage"], pd.DataFrame(coverage), "coverage")
    return {
        "status": "VERIFIED",
        "monthly_fits_verified": fit_number,
        "dependence_fits_verified": fit_number * 4,
        "calibration_parameters_verified": fit_number * 4,
        "applications_verified": len(app),
        "scored_origins": len(panel),
        "contrasts_verified": 7,
        "largest_independent_global_gap": gap_max,
        "clock_verification": "full_reference_calendar"
        if calendar is not None
        else "inherited_original_calendar_validation",
    }
