"""Three fixed daily contextual-precision QLIKE comparisons."""

import copy
import json

import numpy as np
import pandas as pd

from src.commodity_implied_score import _identity, _probabilities, _signature
from src.commodity_implied_score import _prior as _old_prior
from src.orthogonal_round2 import holm_adjust
from src.precision_gate_protocol import EVIDENCE_CLASS, EVIDENCE_LIMITATION
from src.treasury_dealer_inference import masked_mean_inference

CONTROLS = ("base", "adaptive", "constant")
MODELS = (*CONTROLS, "contextual")
PHASES = ("development", "evaluation")


def _config(protocol):
    expected = {
        "forecast": {
            "issuance_start": "2010-01-04",
            "origin_start": "2016-01-01",
            "origin_end": "2025-10-20",
            "source_end": "2025-10-20",
            "development": ["2016-01-01", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "stability": [["2020-01-01", "2022-12-31"], ["2023-01-01", "2025-10-20"]],
            "horizon": 5,
            "minimum_train": 1000,
            "adaptive_half_life": 252,
            "gate_window": 1260,
            "gate_minimum_train": 252,
        },
        "support": {
            "phase_daily": 505,
            "slice_daily": 252,
            "offset_daily": 63,
            "phase_fitted_gate": 252,
            "slice_fitted_gate": 126,
        },
        "inference": {
            "blocks": [21, 63, 126],
            "hac_lags": 126,
            "bootstrap_draws": 399999,
            "seed": 20261003,
        },
        "comparisons": {
            "wave": 26,
            "inherited": 146,
            "new": 3,
            "cumulative": 149,
            "contrasts": [["contextual", c] for c in CONTROLS],
            "wave_alpha": 0.05 / (26 * 27),
            "cumulative_alpha": 0.05,
        },
    }
    if type(protocol) is not dict:
        raise ValueError("Precision gate protocol required")
    try:
        for branch, fields in expected.items():
            if type(protocol.get(branch)) is not dict:
                raise ValueError("Missing scientific protocol section")
            for key, value in fields.items():
                if json.dumps(protocol[branch].get(key), allow_nan=False) != json.dumps(value):
                    raise ValueError(
                        "Fixed scientific protocol mismatch: " + branch + "." + key
                    )
    except (TypeError, OverflowError) as error:
        raise ValueError("Invalid scientific protocol metadata") from error


def _prior(rows, count=146):
    _old_prior(rows, count)


def inherit_family(previous, signature):
    _signature(signature)
    if (
        type(previous) is not dict
        or previous.get("status") not in {"COMPLETED", "UNEVALUABLE"}
        or type(previous.get("hypothesis_count")) is not int
        or previous["hypothesis_count"] != 2
        or type(previous.get("cumulative_hypothesis_count")) is not int
        or previous["cumulative_hypothesis_count"] != 146
    ):
        raise ValueError("Complete2/146 prior Treasury comparison accounting required")
    _prior(previous.get("inherited_rows"), 144)
    rows = previous.get("rows")
    if type(rows) is not list or len(rows) != 2:
        raise ValueError("Exactly two prior Treasury rows required")
    inherited = copy.deepcopy(previous["inherited_rows"])
    for i, (row, control) in enumerate(zip(rows, ("matched", "market"), strict=True)):
        _identity(row)
        _probabilities(row)
        if (
            row["study"],
            row["candidate"],
            row["control"],
            row["horizon"],
            row.get("score"),
        ) != ("treasury_dealer", "candidate", control, 5, "qlike"):
            raise ValueError("Wrong prior Treasury comparison identity")
        if previous["status"] == "COMPLETED":
            if row.get("verdict") not in {"COMPARISON_GATE_PASS", "DOES_NOT_QUALIFY"}:
                raise ValueError("Completed prior comparison verdict required")
        elif (
            row.get("verdict") != "UNEVALUABLE"
            or row.get("status") not in {"INSUFFICIENT_DATA", "INVALID_RUN"}
            or any(
                row.get(k) != 1.0
                for k in ("p_conservative", "p_holm_wave", "p_holm_cumulative")
            )
        ):
            raise ValueError("Unevaluable prior comparisons must retain all p1")
        inherited.append(
            copy.deepcopy(row)
            | {
                "source": "reports/treasury_dealer/predictive/metrics.json",
                "source_sha256": signature,
                "source_row_index": i,
            }
        )
    _prior(inherited)
    return inherited


def _paired(panel, calendar, protocol):
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or calendar.hasnans
        or calendar.tz is not None
        or calendar.has_duplicates
        or not calendar.is_monotonic_increasing
        or not calendar.equals(calendar.normalize())
        or (calendar > "2025-10-20").any()
    ):
        raise ValueError("Unchanged finite ordered naive-midnight full calendar required")
    required = {
        "origin",
        "model",
        "prediction",
        "y",
        "loss",
        "target_end",
        "phase",
        "offset",
        "fit_origin",
        "training_cutoff",
        "train_n",
        "gate_n",
        "gate_status",
        "state",
    }
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.columns.has_duplicates
        or not set(panel) >= required
        or panel.empty
    ):
        raise ValueError("INSUFFICIENT_DATA: complete four-arm scored panel required")
    p = panel.copy(deep=True)
    if set(p.model) != set(MODELS) or p.duplicated(["origin", "model"]).any():
        raise ValueError("Unique declared four-arm origins required")
    for name in ("origin", "target_end", "fit_origin", "training_cutoff"):
        if not pd.api.types.is_datetime64_any_dtype(p[name].dtype):
            raise ValueError("Native scored date columns required")
        dates = pd.DatetimeIndex(p[name])
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("Finite naive-midnight dates required")
    for name in ("prediction", "y", "loss", "state"):
        if p[name].dtype.kind not in "fiu" or not np.isfinite(p[name].to_numpy(float)).all():
            raise ValueError("Finite nonboolean real scored values required")
    for name in ("offset", "train_n", "gate_n"):
        if p[name].dtype.kind not in "iu" or p[name].isna().any():
            raise ValueError("Exact integer scored counts/offsets required")
    if (
        (p.prediction <= 0).any()
        or (p.y <= 0).any()
        or (p.train_n < 1000).any()
        or not p.gate_n.between(0, 1260).all()
        or not np.array_equal(p.gate_status, np.where(p.gate_n >= 252, "fitted", "cold_start"))
    ):
        raise ValueError(
            "Positive forecasts, mature training and exact gate statuses required"
        )
    pos = calendar.get_indexer(p.origin)
    fitpos = calendar.get_indexer(p.fit_origin)
    if (
        (pos < 1).any()
        or (pos + 5 >= len(calendar)).any()
        or (fitpos < 1).any()
        or (fitpos > pos).any()
    ):
        raise ValueError("Known origin, prior cutoff, monthly fit and fifth target required")
    if (
        not np.array_equal(p.target_end.to_numpy(), calendar[pos + 5].to_numpy())
        or not np.array_equal(p.offset.to_numpy(), pos % 5)
        or not np.array_equal(p.training_cutoff.to_numpy(), calendar[fitpos - 1].to_numpy())
        or not pd.DatetimeIndex(p.origin)
        .to_period("M")
        .equals(pd.DatetimeIndex(p.fit_origin).to_period("M"))
    ):
        raise ValueError("Full-calendar endpoint/offset/monthly maturity mismatch")
    dev = p.origin.between(*protocol["forecast"]["development"])
    evaluation = p.origin.between(*protocol["forecast"]["evaluation"])
    if (
        not (dev | evaluation).all()
        or not np.array_equal(p.phase, np.where(dev, "development", "evaluation"))
        or (p.loc[dev, "target_end"] > "2019-12-31").any()
    ):
        raise ValueError("Fixed phase or label maturity fence mismatch")
    try:
        with np.errstate(over="raise", under="raise", divide="raise", invalid="raise"):
            ratio = p.y.to_numpy(float) / p.prediction.to_numpy(float)
            loss = ratio - np.log(ratio) - 1
    except FloatingPointError as error:
        raise ValueError("Invalid QLIKE arithmetic") from error
    if not np.isfinite(loss).all() or not np.array_equal(loss, p.loss.to_numpy()):
        raise ValueError("Scored QLIKE differs from direct reconstruction")
    parts = {m: p[p.model == m].set_index("origin").sort_index() for m in MODELS}
    common = parts["contextual"]
    shared = [c for c in p if c not in {"origin", "model", "prediction", "loss"}]
    for control in CONTROLS:
        other = parts[control]
        if not common.index.equals(other.index) or any(
            not common[c].equals(other[c]) for c in shared
        ):
            raise ValueError("All four arms require exact common origins/metadata")
    return parts


def _support(common, protocol):
    def check(frame, minimum, fitted_minimum):
        if len(frame) < minimum or int((frame.gate_status == "fitted").sum()) < fitted_minimum:
            raise ValueError("INSUFFICIENT_DATA: daily or fitted-gate support")

    for phase in PHASES:
        frame = common[common.phase == phase]
        check(frame, 505, 252)
        for offset in range(5):
            check(frame[frame.offset == offset], 63, 0)
    for start, end in protocol["forecast"]["stability"]:
        check(common.loc[start:end], 252, 126)


def _diagnostics(frame, difference, calendar, phase, protocol):
    def group(selected):
        d = difference[selected]
        fitted = int((frame.gate_status.to_numpy()[selected] == "fitted").sum())
        return {
            "n": len(d),
            "mean": float(d.mean()) if len(d) else None,
            "fitted_gate_n": fitted,
            "cold_start_n": len(d) - fitted,
        }

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
    dates = calendar[(calendar >= a) & (calendar <= b)]
    annual = [
        {
            "year": int(y),
            "calendar_origins": int((dates.year == y).sum()),
            "missing_origins": int((dates.year == y).sum())
            - int((frame.index.year == y).sum()),
        }
        | group(frame.index.year == y)
        for y in sorted(set(dates.year))
    ]
    return {"offsets": offsets, "stability": slices, "annual": annual}


def _passes(row, protocol):
    return (
        row["p_holm_wave"] <= protocol["comparisons"]["wave_alpha"]
        and row["p_holm_cumulative"] <= protocol["comparisons"]["cumulative_alpha"]
        and all(
            phase["mean"] <= -0.005
            and all(g["mean"] < 0 for g in phase["offsets"] + phase["stability"])
            for phase in row["phases"]
        )
    )


def evaluate(panel, calendar, prior, protocol):
    _config(protocol)
    _prior(prior)
    parts = _paired(panel, calendar, protocol)
    common = parts["contextual"]
    _support(common, protocol)
    rows, inf = [], protocol["inference"]
    for control in CONTROLS:
        phases = []
        for code, phase in enumerate(PHASES):
            frame = common[common.phase == phase]
            other = parts[control].loc[frame.index]
            difference = frame.loss.to_numpy() - other.loss.to_numpy()
            a, b = protocol["forecast"][phase]
            dates = calendar[(calendar >= a) & (calendar <= b)]
            values = pd.Series(difference, index=frame.index).reindex(dates).to_numpy()
            stats = masked_mean_inference(
                values,
                dates.isin(frame.index),
                blocks=inf["blocks"],
                hac_lags=inf["hac_lags"],
                draws=inf["bootstrap_draws"],
                seed=inf["seed"] + code * 10000,
            )
            intervals = [
                stats["hac"]["ci95"],
                *[x["ci95"] for x in stats["block_inference"].values()],
            ]
            fitted = int((frame.gate_status == "fitted").sum())
            stats |= {
                "name": phase,
                "fitted_gate_n": fitted,
                "cold_start_n": len(frame) - fitted,
                "candidate_loss": float(frame.loss.mean()),
                "control_loss": float(other.loss.mean()),
                "first_origin": str(frame.index[0].date()),
                "last_origin": str(frame.index[-1].date()),
                "ci95_envelope": [min(x[0] for x in intervals), max(x[1] for x in intervals)],
            }
            stats |= _diagnostics(frame, difference, calendar, phase, protocol)
            phases.append(stats)
        rows.append(
            {
                "study": "precision_gate",
                "candidate": "contextual",
                "control": control,
                "horizon": 5,
                "score": "qlike",
                "phases": phases,
                "p_conservative": max(p["p_conservative"] for p in phases),
            }
        )
    wave = holm_adjust([r["p_conservative"] for r in rows])
    cumulative = holm_adjust([r["p_conservative"] for r in prior + rows])[-3:]
    for row, pw, pc in zip(rows, wave, cumulative, strict=True):
        row |= {"p_holm_wave": float(pw), "p_holm_cumulative": float(pc)}
        row["verdict"] = (
            "COMPARISON_GATE_PASS" if _passes(row, protocol) else "DOES_NOT_QUALIFY"
        )
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["precision_gate"]
        if all(r["verdict"] == "COMPARISON_GATE_PASS" for r in rows)
        else [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 149,
        "common_scored_origins": len(common),
        "fitted_gate_scored_origins": int((common.gate_status == "fitted").sum()),
        "evidence_class": EVIDENCE_CLASS,
        "evidence_limitation": EVIDENCE_LIMITATION,
    }


def failure_metrics(error, signature, prior=None):
    _signature(signature)
    if prior is not None:
        _prior(prior)
    unsupported = "INSUFFICIENT_DATA" in str(error)
    out = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "error": str(error),
        "error_kind": "UNSUPPORTED" if unsupported else "FAILED",
        "protocol_sha256": signature,
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 149,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_limitation": EVIDENCE_LIMITATION,
        "rows": [
            {
                "study": "precision_gate",
                "candidate": "contextual",
                "control": c,
                "horizon": 5,
                "score": "qlike",
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
        out["inherited_rows"] = copy.deepcopy(prior)
    return out
