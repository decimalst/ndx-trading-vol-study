"""Independent peak-age reconstruction and augmented least-squares verification.

No producer feature, transform, ridge, scalar or scheduling functions are used.
Source files are supplied by the separately checked admission boundary.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
from scipy.linalg import lstsq

RAW = (
    "const",
    "I",
    "R",
    "ret_d",
    "ret_w",
    "ret_m",
    "ret_q",
    "lr_d",
    "lr_w",
    "term",
    "lvvix",
    "entry_dow_1",
    "entry_dow_2",
    "entry_dow_3",
    "entry_dow_4",
)
BASE = RAW + ("I_square", "R_square")
DEPTH = BASE + ("drawdown", "drawdown_sq", "window_return")
COMMON = RAW + ("peak_age", "drawdown", "drawdown_sq", "window_return")
MODELS = ("mean", "baseline", "depth", "peak_age")
STATE_COLUMNS = (
    "origin_position",
    "cutoff_position",
    "window_start_position",
    "window_end_position",
    "peak_position",
    "peak_age_sessions",
    "window_start_date",
    "window_end_date",
    "peak_date",
    "feature_cutoff_date",
    "window_complete",
)
ALPHA = 0.01


def array(values, name, *, missing=False, ndim=None):
    if isinstance(values, (list, tuple)):
        pending = list(values)
        while pending:
            value = pending.pop()
            if isinstance(value, (list, tuple)):
                pending.extend(value)
            elif isinstance(value, (bool, np.bool_)):
                raise ValueError("Boolean cannot substitute for numeric evidence: " + name)
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu" or (ndim is not None and raw.ndim != ndim):
        raise ValueError("Literal real numeric array required: " + name)
    result = raw.astype(float)
    if np.isinf(result).any() or (not missing and np.isnan(result).any()):
        raise ValueError("Finite numeric values required: " + name)
    return result


def dates(index):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or index.tz is not None
        or not index.equals(index.normalize())
    ):
        raise ValueError("Exact unique ordered naive midnight source calendar required")


def decimal_log_ratio(x, y):
    a = array([x], "positive ratio numerator")[0]
    b = array([y], "positive ratio denominator")[0]
    if min(a, b) <= 0:
        raise ValueError("Positive stored prices required")
    if a == b:
        return 0.0
    with localcontext() as context:
        context.prec = 100
        result = float(Decimal.from_float(float(a)).ln() - Decimal.from_float(float(b)).ln())
    if not math.isfinite(result) or result == 0 or ((result > 0) != (a > b)):
        raise ValueError("Nonzero finite correctly signed high-precision ratio required")
    return result


def product(a, b):
    with np.errstate(all="ignore"):
        value = float(a) * float(b)
    if not math.isfinite(value) or (a != 0 and b != 0 and value == 0):
        raise ValueError("Required product overflow or nonzero underflow")
    return value


def mean_sum(values):
    values = array(values, "compensated mean", ndim=1)
    if not len(values):
        raise ValueError("Nonempty mean required")
    numerator = math.fsum(map(float, values))
    answer = numerator / len(values)
    if not math.isfinite(answer) or (numerator != 0 and answer == 0):
        raise ValueError("Finite compensated mean without nonzero underflow required")
    return answer


def peak_summary(closes):
    x = array(closes, "literal252-close window", ndim=1)
    if x.shape != (252,) or (x <= 0).any():
        raise ValueError("Exactly252 finite positive stored closes required")
    maximum = max(map(float, x))
    peak = max(i for i, value in enumerate(x) if value == maximum)
    depth = decimal_log_ratio(maximum, x[-1])
    return {
        "peak_offset": peak,
        "peak_age_sessions": 251 - peak,
        "peak_age": (251 - peak) / 251,
        "drawdown": depth,
        "drawdown_sq": product(depth, depth),
        "window_return": decimal_log_ratio(x[-1], x[0]),
    }


def independent_features(daily, iv):
    dates(daily.index)
    dates(iv.index)
    for frame, columns in (
        (daily, ("open", "high", "low", "close")),
        (iv, ("vix", "vix9d", "vvix")),
    ):
        values = array(frame.loc[:, columns], "positive archival inputs", missing=True)
        if ((values <= 0) & np.isfinite(values)).any():
            raise ValueError("Observed source prices must be positive")
    complete = daily.loc[daily.loc[:, ["open", "high", "low", "close"]].notna().all(axis=1)]
    if (complete.high < complete[["open", "low", "close"]].max(axis=1)).any() or (
        complete.low > complete[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("Malformed observed OHLC range")
    index = daily.index
    aligned = iv.reindex(index)
    result = pd.DataFrame(index=index)
    with np.errstate(over="raise", divide="raise", invalid="ignore", under="raise"):
        body = np.log(daily.close / daily.open)
        span = np.log(daily.high / daily.low)
        variance = np.maximum(0.5 * span**2 - (2 * np.log(2) - 1) * body**2, 1e-10)
        variance += np.log(daily.open / daily.close.shift()) ** 2
        returns = np.log(daily.close).diff()
        result["const"] = 1.0
        result["I"] = np.log((aligned.vix / 100) ** 2).shift()
        result["R"] = np.log(252 * variance.rolling(22, min_periods=22).mean()).shift()
        for label, width in (("d", 1), ("w", 5), ("m", 22), ("q", 63)):
            result["ret_" + label] = returns.rolling(width, min_periods=width).mean().shift()
        for label, width in (("d", 1), ("w", 5)):
            result["lr_" + label] = np.log(
                252 * variance.rolling(width, min_periods=width).mean()
            ).shift()
        result["term"] = np.log(aligned.vix9d / aligned.vix).shift()
        result["lvvix"] = np.log(aligned.vvix).shift()
    for day in range(1, 5):
        result[f"entry_dow_{day}"] = (index.dayofweek == day).astype(float)
    cutoff = pd.Series(index, index=index).shift()
    state = pd.DataFrame(index=index)
    for column in STATE_COLUMNS[:6]:
        state[column] = pd.Series(pd.NA, index=index, dtype="Int64")
    for column in STATE_COLUMNS[6:10]:
        state[column] = pd.Series(pd.NaT, index=index, dtype=index.dtype)
    state["window_complete"] = False
    for name in COMMON[len(RAW) :]:
        result[name] = np.nan
    closes = daily.close.to_numpy(float)
    for i, origin in enumerate(index):
        state.loc[origin, "origin_position"] = i
        if i:
            state.loc[origin, "cutoff_position"] = i - 1
            state.loc[origin, "feature_cutoff_date"] = index[i - 1]
        if i < 252:
            continue
        state.loc[origin, "window_start_position"] = i - 252
        state.loc[origin, "window_end_position"] = i - 1
        state.loc[origin, "window_start_date"] = index[i - 252]
        state.loc[origin, "window_end_date"] = index[i - 1]
        window = closes[i - 252 : i]
        if not np.isfinite(window).all():
            continue
        one = peak_summary(window)
        peak = i - 252 + one["peak_offset"]
        state.loc[origin, "peak_position"] = peak
        state.loc[origin, "peak_age_sessions"] = one["peak_age_sessions"]
        state.loc[origin, "peak_date"] = index[peak]
        state.loc[origin, "window_complete"] = True
        for name in COMMON[len(RAW) :]:
            result.loc[origin, name] = one[name]
    result["feature_cutoff_date"] = cutoff
    ends = pd.Series(index, index=index).shift(-21)
    target = pd.DataFrame(
        {"y": np.nan, "target_end": ends, "available_date": ends}, index=index
    )
    for i in range(max(0, len(index) - 21)):
        if np.isfinite(closes[[i, i + 21]]).all():
            target.iloc[i, target.columns.get_loc("y")] = decimal_log_ratio(
                closes[i + 21], closes[i]
            )
    if np.isinf(result.loc[:, COMMON].to_numpy(float)).any():
        raise ValueError("Nonfinite independently reconstructed controls")
    return result.loc[:, (*COMMON, "feature_cutoff_date")], target, state.loc[:, STATE_COLUMNS]


def ridge_fit(training, y, application):
    x = array(training, "ridge training", ndim=2)
    target = array(y, "ridge labels", ndim=1)
    ap = array(application, "ridge applications", ndim=2)
    if (
        len(x) < 2
        or x.shape[1] < 1
        or ap.shape[1] != x.shape[1]
        or len(target) != len(x)
        or not (x[:, 0] == 1).all()
        or not (ap[:, 0] == 1).all()
    ):
        raise ValueError("Aligned complete intercept designs required")
    center = np.r_[0.0, x[:, 1:].mean(axis=0)]
    scale = np.r_[1.0, x[:, 1:].std(axis=0, ddof=0)]
    if not np.isfinite(scale).all() or (scale <= 1e-12).any():
        raise ValueError("No zero-scale nuisance deletion or fallback")
    z = (x - center) / scale
    query = (ap - center) / scale
    target_mean = float(target.mean())
    p = x.shape[1] - 1
    if p:
        matrix = np.vstack([z[:, 1:], np.sqrt(len(x) * ALPHA) * np.eye(p)])
        rhs = np.r_[target - target_mean, np.zeros(p)]
        coefficients = lstsq(matrix, rhs, lapack_driver="gelsd")[0]
        beta = np.r_[target_mean, coefficients]
    else:
        beta = np.array([target_mean])
    fitted = z @ beta
    residual = fitted - target
    gradient = z.T @ residual / len(x)
    gradient[1:] += ALPHA * beta[1:]
    prediction = query @ beta
    objective = float(np.mean(residual**2) + ALPHA * np.dot(beta[1:], beta[1:]))
    if (
        not np.isfinite(np.r_[beta, gradient, prediction, objective]).all()
        or np.max(abs(gradient)) > 1e-10
    ):
        raise ValueError("Independent augmented ridge finite/full-gradient check failed")
    return prediction, {
        "means": center.tolist(),
        "scales": scale.tolist(),
        "beta": beta.tolist(),
        "alpha": ALPHA if p else 0.0,
        "train_n": len(x),
        "objective": objective,
        "gradient": gradient.tolist(),
        "gradient_max_abs": float(max(abs(gradient))),
    }


def scalar_fit(ages, y, offset, query_ages, query_offset):
    age = array(ages, "scalar train age", ndim=1)
    target = array(y, "scalar target", ndim=1)
    base = array(offset, "saved validated depth offsets", ndim=1)
    qa = array(query_ages, "scalar application age", ndim=1)
    qb = array(query_offset, "scalar application offsets", ndim=1)
    if (
        len(age) < 2
        or age.shape != target.shape
        or age.shape != base.shape
        or qa.shape != qb.shape
        or (age < 0).any()
        or (age > 1).any()
        or (qa < 0).any()
        or (qa > 1).any()
    ):
        raise ValueError("Aligned bounded scalar ages and complete offsets required")
    constant = bool((age == age[0]).all())
    center = float(age[0]) if constant else mean_sum(age)
    a = age - center
    residual = target - base
    numerator = mean_sum([product(x, r) for x, r in zip(a, residual, strict=True)])
    second = mean_sum([product(x, x) for x in a])
    denominator = second + ALPHA
    if constant or numerator == 0:
        coefficient = 0.0
        status = "EXACT_CONSTANT_INPUT" if constant else "BALANCED_AT_ZERO"
    else:
        matrix = np.r_[a, math.sqrt(len(a) * ALPHA)].reshape(-1, 1)
        coefficient = float(lstsq(matrix, np.r_[residual, 0.0], lapack_driver="gelsd")[0][0])
        if not math.isfinite(coefficient) or coefficient == 0:
            raise ValueError("Nonzero scalar optimum became nonfinite or underflowed")
        status = "FITTED"
    corrections = np.array([product(coefficient, x) for x in a])
    error = base + corrections - target
    gradient = mean_sum([product(x, e) for x, e in zip(a, error, strict=True)]) + product(
        ALPHA, coefficient
    )
    objective = mean_sum([product(e, e) for e in error]) + product(
        ALPHA, product(coefficient, coefficient)
    )
    if not math.isfinite(gradient) or abs(gradient) > 1e-10 or not math.isfinite(objective):
        raise ValueError("Independent scalar objective/full-gradient check failed")
    correction = np.array([product(coefficient, x) for x in qa - center])
    prediction = qb.copy() if coefficient == 0 else qb + correction
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite independent scalar application")
    return prediction, {
        "coefficient": coefficient,
        "age_center": center,
        "age_scale": 1.0,
        "alpha": ALPHA,
        "status": status,
        "train_n": len(a),
        "numerator": numerator,
        "second_moment": second,
        "denominator": denominator,
        "objective": objective,
        "gradient": gradient,
        "gradient_max_abs": abs(gradient),
    }


def exact(actual, expected, name):
    if type(expected) is dict:
        if type(actual) is not dict or set(actual) != set(expected):
            raise ValueError("Exact mapping schema required: " + name)
        for key in expected:
            exact(actual[key], expected[key], name + "." + str(key))
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError("Exact list required: " + name)
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            exact(a, b, name + "." + str(i))
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError("Exact value/type required: " + name)


def close(actual, expected, name, *, rtol=1e-10, atol=1e-12):
    a = array(actual, name, missing=True)
    b = array(expected, name, missing=True)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ValueError("Exact numeric shapes/unknown masks required: " + name)
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        raise ValueError("Independent numerical comparison failed: " + name)


def date_view(values):
    if not (
        isinstance(values, pd.DatetimeIndex)
        or (
            isinstance(values, pd.Series)
            and pd.api.types.is_datetime64_any_dtype(values.dtype)
        )
    ):
        raise ValueError("Native date containers required; no parsing/coercion")
    original = pd.DatetimeIndex(values)
    unit = np.datetime_data(original.dtype)[0] if original.tz is None else None
    if unit not in ("ms", "us", "ns") or not original.equals(original.normalize()):
        raise ValueError("Native naive midnight ms/us/ns dates required")
    try:
        converted = original.as_unit("ns", round_ok=False)
        returned = converted.as_unit(unit, round_ok=False)
    except (OverflowError, ValueError, pd.errors.OutOfBoundsDatetime) as error:
        raise ValueError("Lossless date conversion required") from error
    if not returned.equals(original):
        raise ValueError("Date roundtrip changed instants or unknown masks")
    return converted


def compare_frame(
    actual, expected, *, numeric=(), date_columns=(), exact_numeric=(), rtol=1e-10, atol=1e-12
):
    if isinstance(expected.index, pd.DatetimeIndex):
        index_matches = isinstance(actual.index, pd.DatetimeIndex) and date_view(
            actual.index
        ).equals(date_view(expected.index))
    else:
        index_matches = type(actual.index) is type(expected.index) and actual.index.equals(
            expected.index
        )
    if (
        not isinstance(actual, pd.DataFrame)
        or list(actual.columns) != list(expected.columns)
        or actual.columns.name != expected.columns.name
        or actual.index.name != expected.index.name
        or not index_matches
    ):
        raise ValueError("Exact frame schema, ordering and calendar required")
    for name in expected:
        a, b = actual[name], expected[name]
        if name in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(a.dtype):
                raise ValueError("Native date column required: " + name)
            if not date_view(a).equals(date_view(b)):
                raise ValueError("Exact date instants and masks required: " + name)
        elif name in numeric:
            if a.dtype != b.dtype:
                raise ValueError("Exact numeric dtype required: " + name)
            close(
                a,
                b,
                name,
                rtol=0 if name in exact_numeric else rtol,
                atol=0 if name in exact_numeric else atol,
            )
        else:
            if a.dtype != b.dtype or not a.reset_index(drop=True).equals(
                b.reset_index(drop=True)
            ):
                raise ValueError("Exact nondate values/dtypes required: " + name)


def _verify_source_frames(expected, features, targets, states):
    f, t, s = expected
    compare_frame(
        features,
        f,
        numeric=COMMON,
        date_columns=("feature_cutoff_date",),
        exact_numeric=("peak_age",),
    )
    for name in ("drawdown", "drawdown_sq", "window_return"):
        mask = f[name].notna()
        if not np.array_equal(np.sign(features.loc[mask, name]), np.sign(f.loc[mask, name])):
            raise ValueError("Exact new-coordinate sign and zero semantics required: " + name)
    # The target uses its separately declared tighter absolute tolerance.
    if list(targets.columns) != list(t.columns) or targets.y.dtype != t.y.dtype:
        raise ValueError("Exact21-session target schema/dtype required")
    close(targets.y, t.y, "independent21-session target", rtol=1e-10, atol=1e-13)
    target_view = targets.copy()
    target_view["y"] = t.y.to_numpy()
    compare_frame(
        target_view, t, numeric=("y",), date_columns=("target_end", "available_date")
    )
    compare_frame(states, s, date_columns=STATE_COLUMNS[6:10])
    return {
        "source_calendar_rows_verified": len(f),
        "complete_peak_windows_verified": int(s.window_complete.sum()),
        "feature_state_rows_verified": len(s),
        "finite21_session_targets_verified": int(t.y.notna().sum()),
        "new_log_ratio_oracle": "Decimal100-digit logarithms of exact stored float values",
        "date_comparison": "lossless native naive midnight ms/us/ns instants and exact NaT masks",
    }


def verify_features(daily, iv, features, targets, states):
    return _verify_source_frames(independent_features(daily, iv), features, targets, states)


def derived_design(train, apply):
    for data in (train, apply):
        dates(data.index)
        if len(data) < 1 or not np.isfinite(array(data.loc[:, COMMON], "common design")).all():
            raise ValueError("Complete common feature design required")
        if not data.const.eq(1).all() or not data.peak_age.between(0, 1).all():
            raise ValueError("Literal intercept and bounded peak age required")
        if (data.drawdown < 0).any() or (data.drawdown_sq < 0).any():
            raise ValueError("Nonnegative drawdown controls required")
        close(data.drawdown_sq, data.drawdown**2, "declared squared depth")
    centers = {
        "train_n": len(train),
        "I_mean": float(train.I.mean()),
        "R_mean": float(train.R.mean()),
    }
    out = []
    for data in (train, apply):
        frame = data.loc[:, RAW].copy()
        with np.errstate(over="raise", under="raise", invalid="raise"):
            frame["I_square"] = (frame.I - centers["I_mean"]) ** 2
            frame["R_square"] = (frame.R - centers["R_mean"]) ** 2
        for name in DEPTH[len(BASE) :]:
            frame[name] = data[name]
        out.append(frame.loc[:, DEPTH])
    return *out, centers


def _numeric_audit(actual, expected, label):
    if type(actual) is not dict or set(actual) != set(expected):
        raise ValueError("Exact audit schema required: " + label)
    for name, value in expected.items():
        if name in ("alpha", "age_scale", "train_n", "status", "columns"):
            exact(actual[name], value, label + "." + name)
        else:
            close(
                actual[name],
                value,
                label + "." + name,
                rtol=1e-7 if name in ("coefficient", "beta") else 1e-10,
                atol=1e-12,
            )
    if "gradient_max_abs" in actual:
        g = array(actual["gradient"], label + " full gradient")
        maximum = float(np.max(abs(g)))
        recorded = array(actual["gradient_max_abs"], label + " gradient maximum")
        if recorded.ndim or not 0 <= float(recorded) <= 1e-10 or maximum > 1e-10:
            raise ValueError("Saved full gradient outside fixed tolerance")
        close(recorded, maximum, label + " gradient maximum identity")


def verify_models(train, y, apply, saved):
    if (
        len(train) < 1000
        or not isinstance(y, pd.Series)
        or not y.index.equals(train.index)
        or set(saved) != {"predictions", "model_audit", "transform_audit", "scalar_audit"}
        or set(saved["model_audit"]) != {"mean", "baseline", "depth"}
        or set(saved["predictions"]) != set(MODELS)
    ):
        raise ValueError("Exact common minimum1000 model inputs/audit schema required")
    target = array(y, "aligned common labels", ndim=1)
    tr, ap, transform = derived_design(train, apply)
    _numeric_audit(saved["transform_audit"], transform, "training-only curvature")
    rebuilt = {}
    predictions = {}
    depth_offsets = None
    for name, columns in (("mean", ("const",)), ("baseline", BASE), ("depth", DEPTH)):
        prediction, independent = ridge_fit(tr.loc[:, columns], target, ap.loc[:, columns])
        independent = {"columns": list(columns), **independent}
        audit = saved["model_audit"][name]
        _numeric_audit(audit, independent, "independent " + name + " optimum")
        # Replay the saved independently checked coefficients on their precise
        # retained input features, preserving the conditional scalar objective.
        mean = array(audit["means"], name + " centers", ndim=1)
        scale = array(audit["scales"], name + " scales", ndim=1)
        beta = array(audit["beta"], name + " saved coefficients", ndim=1)
        z = (tr.loc[:, columns].to_numpy(float) - mean) / scale
        query = (ap.loc[:, columns].to_numpy(float) - mean) / scale
        actual_train = beta[0] + z[:, 1:] @ beta[1:]
        actual_query = beta[0] + query[:, 1:] @ beta[1:]
        error = actual_train - target
        gradient = z.T @ error / len(train)
        gradient[1:] += audit["alpha"] * beta[1:]
        objective = float(np.mean(error**2) + audit["alpha"] * np.dot(beta[1:], beta[1:]))
        if (
            not np.isfinite(np.r_[gradient, objective, actual_query]).all()
            or max(abs(gradient)) > 1e-10
        ):
            raise ValueError("Saved " + name + " coefficients fail independent objective/KKT")
        close(audit["gradient"], gradient, name + " saved full gradient")
        close(audit["objective"], objective, name + " saved objective")
        close(saved["predictions"][name], actual_query, name + " retained query replay")
        close(
            saved["predictions"][name],
            prediction,
            name + " independent predictions",
            rtol=1e-8,
        )
        predictions[name] = actual_query
        rebuilt[name] = independent
        if name == "depth":
            depth_offsets = actual_train
    prediction, scalar = scalar_fit(
        train.peak_age, target, depth_offsets, apply.peak_age, predictions["depth"]
    )
    audit = saved["scalar_audit"]
    _numeric_audit(audit, scalar, "independent conditional scalar")
    if scalar["status"] != "FITTED":
        exact(audit["coefficient"], 0.0, "canonical zero coefficient")
        if scalar["status"] == "EXACT_CONSTANT_INPUT":
            exact(
                audit["age_center"], float(train.peak_age.iloc[0]), "exact constant age center"
            )
    a = train.peak_age.to_numpy(float) - audit["age_center"]
    coefficient = float(audit["coefficient"])
    error = depth_offsets + np.array([product(coefficient, x) for x in a]) - target
    gradient = mean_sum([product(x, e) for x, e in zip(a, error, strict=True)]) + product(
        ALPHA, coefficient
    )
    objective = mean_sum([product(e, e) for e in error]) + product(
        ALPHA, product(coefficient, coefficient)
    )
    if abs(gradient) > 1e-10:
        raise ValueError("Saved scalar coefficient fails independent half-gradient")
    close(audit["gradient"], gradient, "saved scalar half-gradient")
    close(audit["objective"], objective, "saved scalar objective")
    correction = np.array(
        [product(coefficient, x) for x in apply.peak_age.to_numpy(float) - audit["age_center"]]
    )
    replay = (
        predictions["depth"].copy() if coefficient == 0 else predictions["depth"] + correction
    )
    actual = array(saved["predictions"]["peak_age"], "retained candidate predictions", ndim=1)
    close(actual, replay, "candidate saved coefficient replay")
    close(actual, prediction, "independent candidate predictions", rtol=1e-8)
    parent = array(saved["predictions"]["depth"], "retained parent predictions", ndim=1)
    if coefficient == 0 and not np.array_equal(actual.view(np.uint64), parent.view(np.uint64)):
        raise ValueError("Zero scalar coefficient must preserve exact parent bits")
    predictions["peak_age"] = replay
    return predictions, {
        "nuisance_fits_verified": 2,
        "mean_fits_verified": 1,
        "scalar_fits_verified": 1,
        "train_n": len(train),
        "application_n": len(apply),
        "independent_model_audits": rebuilt,
        "independent_scalar_audit": scalar,
    }


APPLICATION_COLUMNS = (
    "origin",
    "fit_origin",
    "feature_cutoff_date",
    "train_n",
    "pred_mean",
    "pred_baseline",
    "pred_depth",
    "pred_peak_age",
)
PANEL_COLUMNS = (
    "origin",
    "fit_origin",
    "feature_cutoff_date",
    "target_end",
    "available_date",
    "horizon",
    "phase",
    "model",
    "prediction",
    "y",
    "loss",
    "train_n",
)


def verify_pipeline(
    daily,
    iv,
    protocol,
    features,
    targets,
    feature_states,
    panel,
    fits,
    application_states,
    support,
):
    section = protocol["index"]
    for key, value in {
        "models": list(MODELS),
        "horizons": [21],
        "raw": list(RAW),
        "baseline": list(BASE),
        "depth": list(DEPTH),
        "common": list(COMMON),
        "minimum_train": 1000,
        "market_lag": 1,
    }.items():
        exact(section[key], value, "fixed mathematical contract " + key)
    start, end = map(pd.Timestamp, (section["origin_start"], section["origin_end"]))
    ds, de = map(pd.Timestamp, section["development"])
    es, ee = map(pd.Timestamp, section["evaluation"])
    latest = pd.Timestamp(section["latest_target"])
    ceiling = pd.Timestamp(section["source_end"])
    if (
        not start
        <= ds
        <= de
        < es
        <= ee
        <= end
        <= latest
        <= ceiling
        < pd.Timestamp(section["sealed_start"])
        or section["development_target_available_by"] != section["development"][1]
        or daily.index.max() > ceiling
        or iv.index.max() > ceiling
    ):
        raise ValueError("Original calendar, phase, maturity and protected fences required")
    expected = independent_features(daily, iv)
    source_proof = _verify_source_frames(expected, features, targets, feature_states)
    index = features.index
    complete = (
        np.isfinite(features.loc[:, COMMON].to_numpy(float)).all(axis=1)
        & features.feature_cutoff_date.notna()
    )
    in_phase = ((index >= ds) & (index <= de)) | ((index >= es) & (index <= ee))
    entries = index[complete & in_phase & (index >= start) & (index <= end)]
    if not len(entries) or type(fits) is not list:
        raise ValueError("Complete original monthly applications and fit list required")
    groups = []
    for period in entries.to_period("M").unique():
        groups.append(entries[entries.to_period("M") == period])
    if len(fits) != len(groups):
        raise ValueError("Every original monthly fit required")
    if (
        list(application_states.columns) != list(APPLICATION_COLUMNS)
        or len(application_states) != len(entries)
        or not date_view(application_states.origin).equals(date_view(entries))
    ):
        raise ValueError("Complete ordered application state cohort required")
    ready = (
        np.isfinite(targets.y)
        & (targets.available_date <= latest)
        & ((index > de) | (targets.available_date <= de))
    )
    all_states = []
    all_rows = []
    model_proofs = []
    cursor = 0
    for application, fit in zip(groups, fits, strict=True):
        origin = application[0]
        position = index.get_loc(origin)
        cutoff = index[position - 1]
        mask = (
            complete
            & np.isfinite(targets.y)
            & (index < origin)
            & (targets.available_date <= cutoff)
        )
        train = index[mask]
        if len(train) < 1000:
            raise ValueError("INSUFFICIENT_DATA: original common mature cohort below1000")
        metadata = {
            "horizon": 21,
            "fit_origin": str(origin.date()),
            "feature_cutoff_date": str(cutoff.date()),
            "train_n": len(train),
            "train_origins": [str(x.date()) for x in train],
            "application_origins": [str(x.date()) for x in application],
        }
        if type(fit) is not dict or set(fit) != (
            set(metadata) | {"model_audit", "transform_audit", "scalar_audit"}
        ):
            raise ValueError("Exact monthly metadata and model-audit schema required")
        exact(
            {k: fit[k] for k in metadata},
            metadata,
            "complete mature training and application identities",
        )
        issued = application_states.iloc[cursor : cursor + len(application)]
        cursor += len(application)
        saved = {
            "predictions": {name: issued["pred_" + name].to_numpy() for name in MODELS},
            **{k: fit[k] for k in ("model_audit", "transform_audit", "scalar_audit")},
        }
        predictions, proof = verify_models(
            features.loc[train], targets.loc[train, "y"], features.loc[application], saved
        )
        model_proofs.append({"fit_origin": str(origin.date()), "audit": proof})
        state = pd.DataFrame(
            {
                "origin": application,
                "fit_origin": origin,
                "feature_cutoff_date": features.loc[
                    application, "feature_cutoff_date"
                ].to_numpy(),
                "train_n": len(train),
            }
        )
        chosen = ready.loc[application].to_numpy()
        scored = application[chosen]
        for name in MODELS:
            state["pred_" + name] = predictions[name]
            row = pd.DataFrame(
                {
                    "origin": scored,
                    "fit_origin": origin,
                    "feature_cutoff_date": features.loc[
                        scored, "feature_cutoff_date"
                    ].to_numpy(),
                    "target_end": targets.loc[scored, "target_end"].to_numpy(),
                    "available_date": targets.loc[scored, "available_date"].to_numpy(),
                    "horizon": 21,
                    "phase": np.where(scored <= de, "development", "evaluation"),
                    "model": name,
                    "prediction": predictions[name][chosen],
                    "y": targets.loc[scored, "y"].to_numpy(),
                    "train_n": len(train),
                }
            )
            residual = row.y.to_numpy() - row.prediction.to_numpy()
            row["loss"] = [product(x, x) for x in residual]
            all_rows.append(row.loc[:, PANEL_COLUMNS])
        all_states.append(state.loc[:, APPLICATION_COLUMNS])
    expected_states = (
        pd.concat(all_states, ignore_index=True).sort_values("origin").reset_index(drop=True)
    )
    expected_panel = (
        pd.concat(all_rows, ignore_index=True)
        .sort_values(["origin", "model"])
        .reset_index(drop=True)
    )
    if expected_panel.empty:
        raise ValueError("INSUFFICIENT_DATA: no complete mature scoring observations")
    compare_frame(
        application_states,
        expected_states,
        numeric=tuple("pred_" + n for n in MODELS),
        date_columns=APPLICATION_COLUMNS[:3],
        rtol=1e-8,
    )
    compare_frame(
        panel,
        expected_panel,
        numeric=("prediction", "y", "loss"),
        date_columns=PANEL_COLUMNS[:5],
        rtol=1e-8,
    )
    copied_targets = targets.loc[pd.DatetimeIndex(panel.origin), "y"].to_numpy(float)
    if not np.array_equal(
        panel.y.to_numpy(float).view(np.uint64), copied_targets.view(np.uint64)
    ):
        raise ValueError("Scored labels must exactly copy the checked target table")
    issued_by_origin = application_states.set_index("origin")
    for model in MODELS:
        part = panel.loc[panel.model.eq(model)]
        copied_predictions = issued_by_origin.loc[
            pd.DatetimeIndex(part.origin), "pred_" + model
        ].to_numpy(float)
        if not np.array_equal(
            part.prediction.to_numpy(float).view(np.uint64), copied_predictions.view(np.uint64)
        ):
            raise ValueError("Scored predictions must exactly copy full application states")
    # Independently check exact primitive scores against the retained prediction,
    # in addition to comparison with independently fitted predictions above.
    actual_residual = panel.y.to_numpy(float) - panel.prediction.to_numpy(float)
    exact_losses = np.array([product(x, x) for x in actual_residual])
    if not np.array_equal(panel.loss.to_numpy(float), exact_losses):
        raise ValueError("Exact primitive squared prediction losses required")
    unscored = expected_states.loc[
        ~expected_states.origin.isin(expected_panel.origin), "origin"
    ]
    expected_support = {
        "calendar_rows": len(index),
        "common_feature_rows": int(complete.sum()),
        "monthly_fits": len(fits),
        "application_origins": len(entries),
        "scored_origins": int(expected_panel.origin.nunique()),
        "unscored_origins": len(unscored),
        "unscored_origin_dates": [str(x.date()) for x in unscored],
        "phases": {},
    }
    for phase in ("development", "evaluation"):
        origins = expected_panel.loc[
            expected_panel.phase.eq(phase), "origin"
        ].drop_duplicates()
        expected_support["phases"][phase] = {
            "n": len(origins),
            "first_origin": str(origins.iloc[0].date()) if len(origins) else None,
            "last_origin": str(origins.iloc[-1].date()) if len(origins) else None,
        }
    exact(support, expected_support, "complete original support and exclusion audit")
    return {
        "source_reconstruction": source_proof,
        "forecasts_verified": len(panel),
        "primitive_squared_losses_verified": len(panel),
        "application_origins_verified": len(entries),
        "application_predictions_verified": 4 * len(entries),
        "common_scored_origins": expected_support["scored_origins"],
        "unscored_origins_verified": len(unscored),
        "monthly_fits_verified": len(fits),
        "new_nuisance_fits_verified": 2 * len(fits),
        "new_scalar_fits_verified": len(fits),
        "training_mean_fits_verified": len(fits),
        "monthly_model_audits": model_proofs,
        "support": expected_support,
    }
