"""Two fixed joint-density comparisons with shared marginal cancellation."""

import copy
import json

import numpy as np
import pandas as pd

from src.commodity_implied_score import _identity, _probabilities, _signature
from src.commodity_implied_score import _prior as _old_prior
from src.joint_copula_density import log_copula, marginal_logpdf
from src.joint_copula_protocol import EVIDENCE_CLASS, EVIDENCE_LIMITATION
from src.orthogonal_round2 import holm_adjust
from src.treasury_dealer_inference import masked_mean_inference

CONTROLS = ("gaussian_copula", "independence")
FAMILIES = {"t8_copula": "t8", "gaussian_copula": "gaussian", "independence": "independence"}
PHASES = ("development", "evaluation")


def _config(protocol):
    expected = {
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
        raise ValueError("Joint-copula scientific protocol required")
    for section, fields in expected.items():
        if type(protocol.get(section)) is not dict:
            raise ValueError("Missing scientific section")
        for key, value in fields.items():
            try:
                same = json.dumps(protocol[section].get(key), allow_nan=False) == json.dumps(
                    value
                )
            except (TypeError, OverflowError) as error:
                raise ValueError("Invalid scientific protocol") from error
            if not same:
                raise ValueError("Fixed scientific protocol mismatch: " + section + "." + key)


def _prior(rows, count=149):
    _old_prior(rows, count)


def inherit_family(previous, signature):
    _signature(signature)
    if (
        type(previous) is not dict
        or previous.get("status") not in {"COMPLETED", "UNEVALUABLE"}
        or type(previous.get("hypothesis_count")) is not int
        or previous["hypothesis_count"] != 3
        or type(previous.get("cumulative_hypothesis_count")) is not int
        or previous["cumulative_hypothesis_count"] != 149
    ):
        raise ValueError("Complete previous3/149 comparison family required")
    _prior(previous.get("inherited_rows"), 146)
    rows = previous.get("rows")
    if type(rows) is not list or len(rows) != 3:
        raise ValueError("Three previous precision-gate rows required")
    inherited = copy.deepcopy(previous["inherited_rows"])
    for i, (row, control) in enumerate(
        zip(rows, ("base", "adaptive", "constant"), strict=True)
    ):
        _identity(row)
        _probabilities(row)
        if (
            row["study"],
            row["candidate"],
            row["control"],
            row["horizon"],
            row.get("score"),
        ) != ("precision_gate", "contextual", control, 5, "qlike"):
            raise ValueError("Wrong inherited precision-gate identity")
        if previous["status"] == "COMPLETED":
            if row.get("verdict") not in {"COMPARISON_GATE_PASS", "DOES_NOT_QUALIFY"}:
                raise ValueError("Completed prior verdict required")
        elif (
            row.get("verdict") != "UNEVALUABLE"
            or row.get("status") not in {"INSUFFICIENT_DATA", "INVALID_RUN"}
            or any(
                row.get(k) != 1.0
                for k in ("p_conservative", "p_holm_wave", "p_holm_cumulative")
            )
        ):
            raise ValueError("Unevaluable previous comparisons require p1")
        inherited.append(
            copy.deepcopy(row)
            | {
                "source": "reports/precision_gate/predictive/metrics.json",
                "source_sha256": signature,
                "source_row_index": i,
            }
        )
    _prior(inherited)
    return inherited


def _density_components(frame):
    required = {"model", "y_qqq", "y_spx", "mu_qqq", "mu_spx", "h_qqq", "h_spx", "rho"}
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.columns.has_duplicates
        or not set(frame) >= required
        or frame.empty
        or not set(frame.model) <= set(FAMILIES)
    ):
        raise ValueError("Known model and complete density inputs required")
    for column in required - {"model"}:
        if (
            frame[column].dtype.kind not in "fiu"
            or not np.isfinite(frame[column].to_numpy(float)).all()
        ):
            raise ValueError("Finite nonboolean real density inputs required")
    h = frame[["h_qqq", "h_spx"]].to_numpy(float)
    rho = frame.rho.to_numpy(float)
    if (
        (h <= 0).any()
        or (abs(rho) > 0.995).any()
        or not (rho[frame.model == "independence"] == 0).all()
    ):
        raise ValueError("Positive marginal variances and fixed valid correlations required")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            scale = np.sqrt(h) * np.sqrt(0.75)
            z = (
                frame[["y_qqq", "y_spx"]].to_numpy(float)
                - frame[["mu_qqq", "mu_spx"]].to_numpy(float)
            ) / scale
            if not np.isfinite(z).all():
                raise ValueError("Nonfinite standardized marginal residual")
            marginal = np.asarray(marginal_logpdf(z, h), float)
            copula = np.empty(len(frame))
            for model, family in FAMILIES.items():
                selected = frame.model.to_numpy() == model
                if selected.any():
                    copula[selected] = log_copula(z[selected], rho[selected], family)
            loss = -marginal.sum(axis=1) - copula
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("Invalid full-density arithmetic") from error
    if (
        marginal.shape != z.shape
        or not np.isfinite(marginal).all()
        or not np.isfinite(copula).all()
        or not np.isfinite(loss).all()
    ):
        raise ValueError("Finite complete joint density required")
    return {"log_copula": copula, "marginal_log_density": marginal, "loss": loss}


def _paired(panel, calendar, protocol):
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or calendar.tz is not None
        or calendar.hasnans
        or calendar.has_duplicates
        or not calendar.is_monotonic_increasing
        or not calendar.equals(calendar.normalize())
        or (calendar > "2025-10-20").any()
    ):
        raise ValueError("Unchanged full SPX calendar required")
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
        raise ValueError("INSUFFICIENT_DATA: complete three-arm joint-density panel required")
    p = panel.copy(deep=True)
    if set(p.model) != set(FAMILIES) or p.duplicated(["origin", "model"]).any():
        raise ValueError("Exact three-arm identity required")
    for name in (
        "origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "fit_origin",
        "training_cutoff",
    ):
        if not pd.api.types.is_datetime64_any_dtype(p[name].dtype):
            raise ValueError("Native date columns required")
        dates = pd.DatetimeIndex(p[name])
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("Finite naive-midnight dates required")
    for name in ("horizon", "offset", "train_n"):
        if p[name].dtype.kind not in "iu" or p[name].isna().any():
            raise ValueError("Exact integer horizon/offset/training count required")
    if (
        not p.horizon.eq(1).all()
        or not p.train_n.ge(1000).all()
        or not p.origin.between("2016-01-04", "2025-10-17").all()
    ):
        raise ValueError("Fixed origin window, horizon and mature training support required")
    position = calendar.get_indexer(p.origin)
    fit = calendar.get_indexer(p.fit_origin)
    if (
        (position < 1).any()
        or (position + 1 >= len(calendar)).any()
        or (fit < 1).any()
        or (fit > position).any()
    ):
        raise ValueError("Known origin/fit/prior and next target sessions required")
    for name, expected in (
        ("feature_cutoff_date", calendar[position - 1]),
        ("target_end", calendar[position + 1]),
        ("available_date", calendar[position + 1]),
        ("training_cutoff", calendar[fit - 1]),
    ):
        if not np.array_equal(p[name].to_numpy(), expected.to_numpy()):
            raise ValueError("Conservative full-calendar clock mismatch: " + name)
    if not np.array_equal(p.offset.to_numpy(), position % 5) or not pd.DatetimeIndex(
        p.origin
    ).to_period("M").equals(pd.DatetimeIndex(p.fit_origin).to_period("M")):
        raise ValueError("Origin-based global offsets or monthly fit mismatch")
    dev = p.origin.between(*protocol["forecast"]["development"])
    evaluation = p.origin.between(*protocol["forecast"]["evaluation"])
    if (
        not (dev | evaluation).all()
        or not np.array_equal(p.phase, np.where(dev, "development", "evaluation"))
        or (p.loc[dev, "target_end"] > "2019-12-31").any()
    ):
        raise ValueError("Fixed phase/mature-target boundaries violated")
    parts = {name: p[p.model == name].set_index("origin").sort_index() for name in FAMILIES}
    common = parts["t8_copula"]
    shared = [c for c in p if c not in {"origin", "model", "rho"}]
    for name in CONTROLS:
        other = parts[name]
        if not common.index.equals(other.index) or any(
            not common[c].equals(other[c]) for c in shared
        ):
            raise ValueError(
                "All joint models must share exact marginal forecasts/targets/metadata"
            )
    components = _density_components(p)
    p["log_copula"] = components["log_copula"]
    p["loss"] = components["loss"]
    return {name: p[p.model == name].set_index("origin").sort_index() for name in FAMILIES}


def _support(common, protocol):
    for phase in PHASES:
        frame = common[common.phase == phase]
        if len(frame) < 505 or any(int(frame.offset.eq(k).sum()) < 63 for k in range(5)):
            raise ValueError("INSUFFICIENT_DATA: phase or global-offset support")
    for start, end in protocol["forecast"]["stability"]:
        if len(common.loc[start:end]) < 252:
            raise ValueError("INSUFFICIENT_DATA: evaluation slice support")


def _diagnostics(frame, difference, calendar, phase, protocol):
    def group(selected):
        values = difference[selected]
        return {"n": len(values), "mean": float(values.mean()) if len(values) else None}

    offsets = [{"offset": k} | group(frame.offset.to_numpy() == k) for k in range(5)]
    slices = (
        [
            {"start": a, "end": b} | group((frame.index >= a) & (frame.index <= b))
            for a, b in protocol["forecast"]["stability"]
        ]
        if phase == "evaluation"
        else []
    )
    a, b = protocol["forecast"][phase]
    full = calendar[(calendar >= a) & (calendar <= b)]
    annual = [
        {
            "year": int(y),
            "calendar_origins": int((full.year == y).sum()),
            "missing_origins": int((full.year == y).sum())
            - int((frame.index.year == y).sum()),
        }
        | group(frame.index.year == y)
        for y in sorted(set(full.year))
    ]
    return {"offsets": offsets, "stability": slices, "annual": annual}


def _passes(row, protocol):
    return (
        row["p_holm_wave"] <= protocol["comparisons"]["wave_alpha"]
        and row["p_holm_cumulative"] <= protocol["comparisons"]["cumulative_alpha"]
        and all(
            p["mean"] <= -0.005 and all(g["mean"] < 0 for g in p["offsets"] + p["stability"])
            for p in row["phases"]
        )
    )


def evaluate(panel, calendar, prior, protocol):
    _config(protocol)
    _prior(prior)
    parts = _paired(panel, calendar, protocol)
    common = parts["t8_copula"]
    _support(common, protocol)
    rows = []
    inf = protocol["inference"]
    for control in CONTROLS:
        phases = []
        for code, phase in enumerate(PHASES):
            frame = common[common.phase == phase]
            other = parts[control].loc[frame.index]
            difference = other.log_copula.to_numpy() - frame.log_copula.to_numpy()
            a, b = protocol["forecast"][phase]
            full = calendar[(calendar >= a) & (calendar <= b)]
            values = pd.Series(difference, index=frame.index).reindex(full).to_numpy()
            result = masked_mean_inference(
                values,
                full.isin(frame.index),
                blocks=inf["blocks"],
                hac_lags=inf["hac_lags"],
                draws=inf["bootstrap_draws"],
                seed=inf["seed"] + code * 10000,
            )
            intervals = [
                result["hac"]["ci95"],
                *[r["ci95"] for r in result["block_inference"].values()],
            ]
            result |= {
                "name": phase,
                "candidate_loss": float(frame.loss.mean()),
                "control_loss": float(other.loss.mean()),
                "first_origin": str(frame.index[0].date()),
                "last_origin": str(frame.index[-1].date()),
                "ci95_envelope": [min(x[0] for x in intervals), max(x[1] for x in intervals)],
            }
            result |= _diagnostics(frame, difference, calendar, phase, protocol)
            phases.append(result)
        rows.append(
            {
                "study": "joint_copula",
                "candidate": "t8_copula",
                "control": control,
                "horizon": 1,
                "score": "negative_joint_log_density",
                "phases": phases,
                "p_conservative": max(p["p_conservative"] for p in phases),
            }
        )
    wave = holm_adjust([r["p_conservative"] for r in rows])
    cumulative = holm_adjust([r["p_conservative"] for r in prior + rows])[-2:]
    for row, pw, pc in zip(rows, wave, cumulative, strict=True):
        row |= {"p_holm_wave": float(pw), "p_holm_cumulative": float(pc)}
        row["verdict"] = (
            "COMPARISON_GATE_PASS" if _passes(row, protocol) else "DOES_NOT_QUALIFY"
        )
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["joint_copula"]
        if all(r["verdict"] == "COMPARISON_GATE_PASS" for r in rows)
        else [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 151,
        "common_scored_origins": len(common),
        "evidence_class": EVIDENCE_CLASS,
        "evidence_limitation": EVIDENCE_LIMITATION,
    }


def failure_metrics(error, signature, prior=None):
    _signature(signature)
    if prior is not None:
        _prior(prior)
    unsupported = "INSUFFICIENT_DATA" in str(error)
    result = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "error": str(error),
        "error_kind": "UNSUPPORTED" if unsupported else "FAILED",
        "protocol_sha256": signature,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 151,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_limitation": EVIDENCE_LIMITATION,
        "rows": [
            {
                "study": "joint_copula",
                "candidate": "t8_copula",
                "control": c,
                "horizon": 1,
                "score": "negative_joint_log_density",
                "status": "INSUFFICIENT_DATA" if unsupported else "INVALID_RUN",
                "verdict": "UNEVALUABLE",
                "phases": [],
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
            }
            for c in CONTROLS
        ],
    }
    if prior is not None:
        result["inherited_rows"] = copy.deepcopy(prior)
    return result
