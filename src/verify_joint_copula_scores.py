"""Independent t8-marginal/copula formulas and wave27 score reconstruction.

Producer density calls are subjects checked against independent rowwise values;
they never establish expected scores or the inference reconstruction.
"""

import copy
import json
import math
import re

import numpy as np
import pandas as pd
from scipy.special import betaln, erfinv, hyp2f1, ndtri_exp

from src.joint_copula_protocol import EVIDENCE_CLASS, EVIDENCE_LIMITATION
from src.verify_claims_release_scores import _calendar, _dates, _holm, _integer, _real
from src.verify_treasury_dealer_scores import _indexed_inference

CONTROLS = ("gaussian_copula", "independence")
PHASES = ("development", "evaluation")
FAMILIES = {"t8_copula": "t8", "gaussian_copula": "gaussian", "independence": "independence"}
_T8_C = math.lgamma(4.5) - math.lgamma(4) - 0.5 * math.log(8 * math.pi)


def _log_one_plus_square(z):
    with np.errstate(divide="ignore"):
        return np.logaddexp(0.0, 2 * np.log(np.abs(z)) - math.log(8))


def _normal_coordinates(z):
    a = np.abs(z)
    result = np.empty_like(a)
    central = a <= 1
    # Integrate the density from zero, retaining tiny signed central masses.
    mass = math.exp(_T8_C) * a[central] * hyp2f1(0.5, 4.5, 1.5, -(a[central] ** 2) / 8)
    result[central] = math.sqrt(2) * erfinv(2 * mass)
    tail = ~central
    if tail.any():
        logx = math.log(8) - np.logaddexp(math.log(8), 2 * np.log(a[tail]))
        # x may underflow, but its logarithm and therefore the log-tail do not.
        # The hypergeometric factor tends to1 at x=0; no CDF clipping occurs.
        with np.errstate(under="ignore"):
            x = np.exp(logx)
        logtail = (
            -math.log(2)
            + 4 * logx
            - math.log(4)
            - betaln(4, 0.5)
            + np.log(hyp2f1(4, 0.5, 5, x))
        )
        result[tail] = -ndtri_exp(logtail)
    result *= np.sign(z)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite independent normal coordinates")
    return result


def _independent_logcopula(z, rho, family):
    source = np.asarray(z)
    r = np.asarray(rho)
    if (
        source.ndim != 2
        or source.shape[1] != 2
        or len(source) == 0
        or source.dtype.kind not in "fiu"
        or r.dtype.kind not in "fiu"
    ):
        raise ValueError("Real pairs and real correlation parameters required")
    values = source.astype(float)
    if r.ndim == 0:
        r = np.full(len(values), float(r))
    elif r.ndim == 1 and len(r) == len(values):
        r = r.astype(float)
    else:
        raise ValueError("Scalar or matching rowwise correlation required")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(r).all()
        or (abs(r) > 0.995).any()
        or family not in {"t8", "gaussian", "independence"}
    ):
        raise ValueError("Finite standardized pairs and bounded declared family required")
    if family == "independence":
        if (r != 0).any():
            raise ValueError("Independence correlation must equal zero")
        return np.zeros(len(values))
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            det = 1 - r * r
            if family == "gaussian":
                w = _normal_coordinates(values)
                result = (
                    -0.5 * np.log1p(-r * r)
                    + (r * w[:, 0] * w[:, 1] - 0.5 * r * r * (w[:, 0] ** 2 + w[:, 1] ** 2))
                    / det
                )
            else:
                largest = np.max(abs(values), axis=1)
                normalized = np.divide(
                    values,
                    largest[:, None],
                    out=np.zeros_like(values),
                    where=largest[:, None] > 0,
                )
                a, b = normalized.T
                q = ((a - r * b) ** 2 + det * b * b) / det
                logq = np.full(len(values), -np.inf)
                nonzero = largest > 0
                logq[nonzero] = 2 * np.log(largest[nonzero]) + np.log(q[nonzero])
                joint = (
                    math.lgamma(5)
                    - math.lgamma(4)
                    - math.log(8 * math.pi)
                    - 0.5 * np.log1p(-r * r)
                    - 5 * np.logaddexp(0, logq - math.log(8))
                )
                marginal = _T8_C - 4.5 * _log_one_plus_square(values)
                result = joint - marginal.sum(axis=1)
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("Invalid independent copula arithmetic") from error
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite independent copula")
    return result


def _independent_components(frame):
    fields = {"model", "mu_qqq", "mu_spx", "h_qqq", "h_spx", "y_qqq", "y_spx", "rho"}
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.columns.has_duplicates
        or not set(frame) >= fields
        or frame.empty
        or not set(frame.model) <= set(FAMILIES)
    ):
        raise ValueError("Complete independent density input required")
    for name in fields - {"model"}:
        if (
            frame[name].dtype.kind not in "fiu"
            or not np.isfinite(frame[name].to_numpy(float)).all()
        ):
            raise ValueError("Finite real density inputs required")
    h = frame[["h_qqq", "h_spx"]].to_numpy(float)
    if (h <= 0).any():
        raise ValueError("Positive variance required")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            z = (
                frame[["y_qqq", "y_spx"]].to_numpy(float)
                - frame[["mu_qqq", "mu_spx"]].to_numpy(float)
            ) / (np.sqrt(h) * math.sqrt(0.75))
            if not np.isfinite(z).all():
                raise ValueError("Nonfinite independent standardized residual")
            marginal = (
                _T8_C - 4.5 * _log_one_plus_square(z) - 0.5 * (np.log(h) + math.log(0.75))
            )
            copula = np.empty(len(frame))
            for model, family in FAMILIES.items():
                selected = frame.model.to_numpy() == model
                if selected.any():
                    copula[selected] = _independent_logcopula(
                        z[selected], frame.rho.to_numpy(float)[selected], family
                    )
            loss = -marginal.sum(axis=1) - copula
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("Invalid independent marginal density arithmetic") from error
    if not np.isfinite(marginal).all() or not np.isfinite(loss).all():
        raise ValueError("Finite independent joint densities required")
    return {
        "log_copula": copula,
        "marginal_log_density": marginal,
        "loss": loss,
        "standardized": z,
    }


def _density_agreement(frame, expected):
    # These are observed producer values, never expected-value calculations.
    from src.joint_copula_density import log_copula, marginal_logpdf

    marginal = marginal_logpdf(
        expected["standardized"], frame[["h_qqq", "h_spx"]].to_numpy(float)
    )
    actual = np.empty(len(frame))
    for model, family in FAMILIES.items():
        mask = frame.model.to_numpy() == model
        if mask.any():
            actual[mask] = log_copula(
                expected["standardized"][mask], frame.rho.to_numpy(float)[mask], family
            )
    for a, b in (
        (marginal, expected["marginal_log_density"]),
        (actual, expected["log_copula"]),
        (-np.asarray(marginal).sum(axis=1) - actual, expected["loss"]),
    ):
        if not np.isfinite(a).all() or not np.allclose(a, b, atol=1e-10, rtol=1e-8):
            raise ValueError("Producer density exceeds independent per-row tolerance")


def _contract(protocol):
    fixed = {
        "forecast": {
            "origin_start": "2016-01-04",
            "origin_end": "2025-10-17",
            "source_end": "2025-10-20",
            "development": ["2016-01-04", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "stability": [["2020-01-01", "2022-12-31"], ["2023-01-01", "2025-10-20"]],
            "horizon": 1,
            "minimum_train": 1000,
        },
        "support": {"phase_daily": 505, "slice_daily": 252, "offset_daily": 63},
        "inference": {
            "blocks": [21, 63, 126],
            "hac_lags": 126,
            "bootstrap_draws": 399999,
            "seed": 20260909,
        },
        "comparisons": {
            "wave": 27,
            "inherited": 149,
            "new": 2,
            "cumulative": 151,
            "contrasts": [["t8_copula", "gaussian_copula"], ["t8_copula", "independence"]],
            "wave_alpha": 0.05 / (27 * 28),
            "cumulative_alpha": 0.05,
        },
    }
    if type(protocol) is not dict:
        raise ValueError("Declared scientific protocol required")
    for section, fields in fixed.items():
        if type(protocol.get(section)) is not dict:
            raise ValueError("Missing scientific section")
        for key, expected in fields.items():
            try:
                equal = json.dumps(protocol[section].get(key), allow_nan=False) == json.dumps(
                    expected
                )
            except (TypeError, OverflowError) as error:
                raise ValueError("Invalid protocol") from error
            if not equal:
                raise ValueError("Independent scientific protocol mismatch: " + key)


def _prior_p(prior):
    if type(prior) is not list or len(prior) != 149:
        raise ValueError("Exactly149 inherited hypotheses required")
    probabilities = []
    seen = set()
    hashes = {}
    for row in prior:
        if type(row) is not dict:
            raise ValueError("Inherited dictionary required")
        probabilities.append(_real(row.get("p_conservative"), "prior probability", 0, 1))
        for key in ("p_holm_wave", "p_holm_cumulative"):
            if key in row:
                _real(row[key], key, 0, 1)
        for key in ("study", "candidate", "control", "source"):
            if type(row.get(key)) is not str or not row[key]:
                raise ValueError("Inherited identity/provenance required")
        _integer(row.get("horizon"), "horizon")
        index = _integer(row.get("source_row_index"), "source row", 0)
        source = row["source"]
        digest = row.get("source_sha256")
        if type(digest) is not str or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Inherited SHA256 required")
        if (source, index) in seen or (source in hashes and hashes[source] != digest):
            raise ValueError("Conflicting prior source identities")
        seen.add((source, index))
        hashes[source] = digest
    return probabilities


def _panel(panel, calendar):
    _calendar(calendar)
    if (calendar > "2025-10-20").any():
        raise ValueError("Calendar exceeds fixed source ceiling")
    required = {
        "origin",
        "model",
        "horizon",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "y_qqq",
        "y_spx",
        "mu_qqq",
        "mu_spx",
        "h_qqq",
        "h_spx",
        "rho",
        "fit_origin",
        "training_cutoff",
        "train_n",
        "phase",
        "offset",
    }
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.columns.has_duplicates
        or not set(panel) >= required
        or panel.empty
    ):
        raise ValueError("Complete joint panel required")
    p = panel.copy(deep=True)
    for field in (
        "origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "fit_origin",
        "training_cutoff",
    ):
        if not pd.api.types.is_datetime64_any_dtype(p[field].dtype):
            raise ValueError("Native dates required")
        p[field] = _dates(p[field])
    if (
        set(p.model) != set(FAMILIES)
        or p.duplicated(["origin", "model"]).any()
        or not p.groupby("origin").size().eq(3).all()
    ):
        raise ValueError("Three distinct common model rows required")
    for field in ("horizon", "train_n", "offset"):
        if (
            not pd.api.types.is_integer_dtype(p[field].dtype)
            or pd.api.types.is_bool_dtype(p[field].dtype)
            or p[field].isna().any()
        ):
            raise ValueError("Integer horizon/train/offset required")
    if (
        not p.horizon.eq(1).all()
        or not p.train_n.ge(1000).all()
        or not p.origin.between("2016-01-04", "2025-10-17").all()
    ):
        raise ValueError("Fixed scored origin window and train/horizon required")
    for field in p.columns.difference(["model", "rho"]):
        if (
            p[field].isna().any()
            or not p.groupby("origin")[field].nunique(dropna=False).eq(1).all()
        ):
            raise ValueError("Shared marginal/target metadata differs: " + field)
    position = calendar.get_indexer(p.origin)
    fit = calendar.get_indexer(p.fit_origin)
    if (
        (position < 1).any()
        or (position + 1 >= len(calendar)).any()
        or (fit < 1).any()
        or (fit > position).any()
    ):
        raise ValueError("Valid prior/full-calendar origin/target/fit required")
    if not np.array_equal(p.offset.to_numpy(), position % 5):
        raise ValueError("Offsets must remain global origin positions")
    expected = {
        "feature_cutoff_date": calendar[position - 1],
        "target_end": calendar[position + 1],
        "available_date": calendar[position + 1],
        "training_cutoff": calendar[fit - 1],
    }
    for key, dates in expected.items():
        if not np.array_equal(p[key].to_numpy(), dates.to_numpy()):
            raise ValueError("Independent one-session clock mismatch: " + key)
    if not np.array_equal(
        pd.DatetimeIndex(p.origin).to_period("M"),
        pd.DatetimeIndex(p.fit_origin).to_period("M"),
    ):
        raise ValueError("Monthly fit differs from origin month")
    dev = p.origin.between("2016-01-04", "2019-12-31")
    evaluation = p.origin.between("2020-01-01", "2025-10-20")
    if (
        not (dev | evaluation).all()
        or not np.array_equal(p.phase, np.where(dev, "development", "evaluation"))
        or (p.loc[dev, "target_end"] > "2019-12-31").any()
    ):
        raise ValueError("Phase/target fences differ")
    return p.sort_values(["origin", "model"])


def _compare(actual, expected, path="metrics", exact=False):
    if type(expected) is dict:
        if type(actual) is not dict or set(actual) != set(expected):
            raise ValueError("Report dictionary mismatch: " + path)
        for key, value in expected.items():
            _compare(actual[key], value, path + "." + key, exact or key == "inherited_rows")
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError("Report list mismatch: " + path)
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            _compare(a, b, path + "." + str(i), exact)
    elif isinstance(expected, (float, np.floating)):
        observed = _real(actual, path)
        probability = path.rsplit(".", 1)[-1] in {
            "p",
            "p_conservative",
            "p_holm_wave",
            "p_holm_cumulative",
        }
        if probability and not 0 <= observed <= 1:
            raise ValueError("Invalid probability: " + path)
        discrete = ".block_inference." in path and path.endswith(".p")
        if exact or discrete:
            if observed != float(expected) or (exact and type(actual) is not type(expected)):
                raise ValueError("Exact report value mismatch: " + path)
        else:
            atol = 1e-12 if probability else 1e-10
            if not np.isclose(observed, expected, atol=atol, rtol=1e-8):
                raise ValueError("Numerical report mismatch: " + path)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError("Literal report mismatch: " + path)


def verify_scores(panel, calendar, metrics, prior, protocol):
    _contract(protocol)
    old = _prior_p(prior)
    qualification = {
        "evidence_class": EVIDENCE_CLASS,
        "evidence_limitation": EVIDENCE_LIMITATION,
    }
    if type(metrics) is not dict:
        raise ValueError("Metrics dictionary required")
    _compare({k: metrics[k] for k in qualification if k in metrics}, qualification, exact=True)
    p = _panel(panel, calendar)
    common = p[p.model == "t8_copula"].set_index("origin")
    for phase in PHASES:
        frame = common[common.phase == phase]
        if len(frame) < 505 or any(
            int((calendar.get_indexer(frame.index) % 5 == k).sum()) < 63 for k in range(5)
        ):
            raise ValueError("INSUFFICIENT_DATA: independent phase/offset support")
    for a, b in protocol["forecast"]["stability"]:
        if len(common.loc[a:b]) < 252:
            raise ValueError("INSUFFICIENT_DATA: independent slice support")
    independent = _independent_components(p)
    _density_agreement(p, independent)
    p["log_copula"] = independent["log_copula"]
    p["loss"] = independent["loss"]
    common = p[p.model == "t8_copula"].set_index("origin")
    rows = []
    for control in CONTROLS:
        reference = p[p.model == control].set_index("origin")
        phases = []
        for code, phase in enumerate(PHASES):
            frame = common[common.phase == phase]
            other = reference.loc[frame.index]
            difference = other.log_copula.to_numpy() - frame.log_copula.to_numpy()
            a, b = protocol["forecast"][phase]
            full = calendar[(calendar >= a) & (calendar <= b)]
            indices = full.get_indexer(frame.index)
            mask = np.zeros(len(full), bool)
            mask[indices] = True
            values = np.full(len(full), np.nan)
            values[indices] = difference
            stats = _indexed_inference(
                values,
                mask,
                blocks=[21, 63, 126],
                hac_lags=126,
                draws=399999,
                seed=20260909 + code * 10000,
            )
            intervals = [
                stats["hac"]["ci95"],
                *[g["ci95"] for g in stats["block_inference"].values()],
            ]

            def group(selected, difference=difference):
                n = int(selected.sum())
                return {"n": n, "mean": float(difference[selected].mean()) if n else None}

            offset = calendar.get_indexer(frame.index) % 5
            stats |= {
                "name": phase,
                "candidate_loss": float(frame.loss.to_numpy().mean()),
                "control_loss": float(other.loss.to_numpy().mean()),
                "first_origin": str(frame.index[0].date()),
                "last_origin": str(frame.index[-1].date()),
                "ci95_envelope": [min(x[0] for x in intervals), max(x[1] for x in intervals)],
                "offsets": [{"offset": k} | group(offset == k) for k in range(5)],
                "stability": [
                    {"start": start, "end": end}
                    | group((frame.index >= start) & (frame.index <= end))
                    for start, end in protocol["forecast"]["stability"]
                ]
                if phase == "evaluation"
                else [],
                "annual": [
                    {
                        "year": int(y),
                        "calendar_origins": int((full.year == y).sum()),
                        "missing_origins": int((full.year == y).sum())
                        - int((frame.index.year == y).sum()),
                    }
                    | group(frame.index.year == y)
                    for y in sorted(set(full.year))
                ],
            }
            phases.append(stats)
        rows.append(
            {
                "study": "joint_copula",
                "candidate": "t8_copula",
                "control": control,
                "horizon": 1,
                "score": "negative_joint_log_density",
                "phases": phases,
                "p_conservative": max(q["p_conservative"] for q in phases),
            }
        )
    new = [r["p_conservative"] for r in rows]
    wave, cumulative = _holm(new), _holm(old + new)[-2:]
    for row, pw, pc in zip(rows, wave, cumulative, strict=True):
        passed = (
            pw <= 0.05 / (27 * 28)
            and pc <= 0.05
            and all(
                phase["mean"] <= -0.005
                and all(x["mean"] < 0 for x in phase["offsets"] + phase["stability"])
                for phase in row["phases"]
            )
        )
        row |= {
            "p_holm_wave": float(pw),
            "p_holm_cumulative": float(pc),
            "verdict": "COMPARISON_GATE_PASS" if passed else "DOES_NOT_QUALIFY",
        }
    expected = {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["joint_copula"]
        if all(row["verdict"] == "COMPARISON_GATE_PASS" for row in rows)
        else [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 151,
        "common_scored_origins": len(common),
    } | qualification
    _compare(metrics, expected)
    return {
        "status": "VERIFIED",
        "comparisons_verified": 2,
        "endpoints_verified": 4,
        "bootstrap_runs_verified": 12,
        "hac_runs_verified": 4,
        "inherited_comparisons_verified": 149,
        "density_rows_verified": len(p),
        "common_scored_origins": len(common),
    }
