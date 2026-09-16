"""Two fixed Treasury QLIKE contrasts on uncompressed phase calendars."""

import copy

import numpy as np
import pandas as pd

from src.commodity_implied_score import (
    _identity,
    _paired,
    _probabilities,
    _signature,
)
from src.commodity_implied_score import (
    _prior as _old_prior,
)
from src.orthogonal_round2 import holm_adjust
from src.treasury_dealer_inference import masked_mean_inference
from src.treasury_dealer_protocol import validate

CONTROLS = ("matched", "market")
PHASES = ("development", "evaluation")
TENORS = ("2", "3", "5", "7", "10", "30")


def _prior(rows, count=144):
    _old_prior(rows, count)


def inherit_family(previous, signature):
    _signature(signature)
    if (
        type(previous) is not dict
        or previous.get("status") != "COMPLETED"
        or type(previous.get("hypothesis_count")) is not int
        or previous["hypothesis_count"] != 2
        or type(previous.get("cumulative_hypothesis_count")) is not int
        or previous["cumulative_hypothesis_count"] != 144
    ):
        raise ValueError("Completed commodity family with2/144 comparisons required")
    _prior(previous.get("inherited_rows"), 142)
    rows = previous.get("rows")
    if type(rows) is not list or len(rows) != 2:
        raise ValueError("Both prior commodity rows required")
    result = copy.deepcopy(previous["inherited_rows"])
    for i, (row, control) in enumerate(zip(rows, CONTROLS, strict=True)):
        _probabilities(row)
        _identity(row)
        if (
            row["study"],
            row["candidate"],
            row["control"],
            row["horizon"],
            row.get("score"),
        ) != ("commodity_implied", "candidate", control, 5, "qlike") or row.get(
            "verdict"
        ) not in {"COMPARISON_GATE_PASS", "DOES_NOT_QUALIFY"}:
            raise ValueError("Exact completed prior commodity comparison required")
        result.append(
            copy.deepcopy(row)
            | {
                "source": "reports/commodity_implied/predictive/metrics.json",
                "source_sha256": signature,
                "source_row_index": i,
            }
        )
    _prior(result)
    return result


def _activation(mask, calendar):
    if (
        not isinstance(mask, pd.Series)
        or not mask.index.equals(calendar)
        or mask.dtype != np.dtype(bool)
        or mask.isna().any()
    ):
        raise ValueError("Exact full-calendar-aligned bool activation Series required")


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("INSUFFICIENT_DATA: invalid count " + label)
    return value


def _tenors(row, prefix, floor):
    counts = row.get(prefix + "events_per_tenor")
    total = _integer(row.get(prefix + "event_n"), "event_n")
    if type(counts) is not dict or set(counts) != set(TENORS):
        raise ValueError("Exact six-tenor support counts required")
    if sum(_integer(counts[t], t, floor) for t in TENORS) != total:
        raise ValueError("Event count does not sum across tenors")
    return total


def _support_report(common, calendar, active, protocol, report):
    keys = {"status", "passed", "monthly", "phases", "slices", "offsets", "discrepancies"}
    if (
        type(report) is not dict
        or set(report) != keys
        or report["status"] != "SUPPORT_PASS"
        or report["passed"] is not True
        or report["discrepancies"] != []
    ):
        raise ValueError("INSUFFICIENT_DATA: complete passing source support report required")
    floors = protocol["support"]
    forecast = protocol["forecast"]
    if (
        type(report["phases"]) is not dict
        or set(report["phases"]) != set(PHASES)
        or type(report["offsets"]) is not dict
        or set(report["offsets"]) != set(PHASES)
        or type(report["slices"]) is not list
        or len(report["slices"]) != 2
    ):
        raise ValueError("Exact phase/slice/offset support scope required")
    basekeys = {
        "origin_start",
        "origin_end",
        "daily_n",
        "activation_dates",
        "event_n",
        "events_per_tenor",
        "passed",
        "failures",
    }

    def check(row, bounds, daily_floor, active_floor, tenor_floor, extra=None):
        extra = {} if extra is None else extra
        if (
            type(row) is not dict
            or set(row) != basekeys | set(extra)
            or row["origin_start"] != bounds[0]
            or row["origin_end"] != bounds[1]
            or any(type(row[k]) is not type(v) or row[k] != v for k, v in extra.items())
            or row["passed"] is not True
            or row["failures"] != []
        ):
            raise ValueError("Exact passing support row scope required")
        frame = common.loc[bounds[0] : bounds[1]]
        if "offset" in extra:
            frame = frame[frame.offset == extra["offset"]]
        n = len(frame)
        a = int(active.reindex(frame.index).sum())
        if (
            _integer(row["daily_n"], "daily_n", daily_floor) != n
            or _integer(row["activation_dates"], "activation_dates", active_floor) != a
            or a > n
            or _tenors(row, "", tenor_floor) < a
        ):
            raise ValueError(
                "INSUFFICIENT_DATA: support counts differ from common paired daily/active origins"
            )

    for phase in PHASES:
        bounds = forecast[phase]
        check(
            report["phases"][phase],
            bounds,
            floors["phase_daily"],
            floors["phase_activation_dates"],
            floors["phase_events_per_tenor"],
        )
        offsets = report["offsets"][phase]
        if type(offsets) is not list or len(offsets) != 5:
            raise ValueError("Five full-calendar offsets required")
        for k, row in enumerate(offsets):
            check(
                row,
                bounds,
                floors["offset_daily"],
                floors["phase_active_dates_per_offset"],
                0,
                {"offset": k},
            )
    for i, (row, bounds) in enumerate(
        zip(report["slices"], forecast["stability"], strict=True)
    ):
        check(
            row,
            bounds,
            floors["slice_daily"],
            floors["slice_activation_dates"],
            floors["slice_events_per_tenor"],
            {"slice_index": i},
        )
    months = list(
        pd.period_range(forecast["origin_start"], forecast["origin_end"], freq="M").astype(str)
    )
    monthly = report["monthly"]
    mkeys = {
        "month",
        "status",
        "fit_origin",
        "training_cutoff",
        "requested_n",
        "application_n",
        "train_n",
        "train_activation_dates",
        "train_event_n",
        "train_events_per_tenor",
        "passed",
        "failures",
    }
    if (
        type(monthly) is not list
        or [r.get("month") for r in monthly if type(r) is dict] != months
    ):
        raise ValueError("Exact ordered full monthly support scope required")
    for row, month in zip(monthly, months, strict=True):
        if set(row) != mkeys or row["passed"] is not True or row["failures"] != []:
            raise ValueError("Passing exact monthly support row required")
        requested = calendar[
            (calendar >= forecast["origin_start"])
            & (calendar <= forecast["origin_end"])
            & (calendar.to_period("M") == pd.Period(month))
        ]
        part = common[common.index.to_period("M") == pd.Period(month)]
        n = _integer(row["requested_n"], "requested_n")
        applications = _integer(row["application_n"], "application_n")
        if n != len(requested) or not len(part) <= applications <= n:
            raise ValueError("Monthly requested/application support mismatch")
        if row["status"] == "SUPPORTED":
            train = _integer(row["train_n"], "train_n", forecast["minimum_train"])
            activations = _integer(
                row["train_activation_dates"],
                "train_activation_dates",
                floors["training_activation_dates"],
            )
            if (
                activations > train
                or _tenors(row, "train_", floors["training_events_per_tenor"]) < activations
                or applications < 1
            ):
                raise ValueError("INSUFFICIENT_DATA: monthly training support")
            fit = pd.Timestamp(row["fit_origin"])
            cutoff = pd.Timestamp(row["training_cutoff"])
            position = calendar.get_indexer([fit])[0]
            if position < 1 or fit not in requested or calendar[position - 1] != cutoff:
                raise ValueError("Monthly first-query cutoff not on the full calendar")
            if any(
                not part[name].eq(value).all()
                for name, value in (
                    ("fit_origin", fit),
                    ("training_cutoff", cutoff),
                    ("train_n", train),
                )
            ):
                raise ValueError("Monthly support identity differs from scored fits")
        elif row["status"] in {"NO_COMPLETE_ORIGIN", "NO_REQUESTED_ORIGINS"}:
            if (
                applications
                or len(part)
                or any(
                    row[k] is not None
                    for k in (
                        "fit_origin",
                        "training_cutoff",
                        "train_n",
                        "train_activation_dates",
                        "train_event_n",
                        "train_events_per_tenor",
                    )
                )
                or (row["status"] == "NO_REQUESTED_ORIGINS") != (n == 0)
            ):
                raise ValueError("Invalid explicit no-fit month")
        else:
            raise ValueError("INSUFFICIENT_DATA: unsupported monthly fit")


def _deltas(frame, differences, active):
    a = active.reindex(frame.index).to_numpy(bool)
    d = np.asarray(differences, float)
    return {
        "daily_n": len(d),
        "active_n": int(a.sum()),
        "daily_delta": float(d.mean()) if len(d) else None,
        "active_delta": float(d[a].mean()) if a.any() else None,
    }


def _diagnostics(frame, difference, calendar, active, phase, protocol):
    offsets = []
    for k in range(5):
        selected = frame.offset.to_numpy() == k
        offsets.append({"offset": k} | _deltas(frame[selected], difference[selected], active))
    stability = []
    if phase == "evaluation":
        for start, end in protocol["forecast"]["stability"]:
            selected = (frame.index >= start) & (frame.index <= end)
            stability.append(
                {"start": start, "end": end}
                | _deltas(frame[selected], difference[selected], active)
            )
    start, end = protocol["forecast"][phase]
    full = calendar[(calendar >= start) & (calendar <= end)]
    annual = []
    for year in sorted(set(full.year)):
        selected = frame.index.year == year
        total = int((full.year == year).sum())
        annual.append(
            {
                "year": int(year),
                "calendar_origins": total,
                "missing_origins": total - int(selected.sum()),
            }
            | _deltas(frame[selected], difference[selected], active)
        )
    return {"offsets": offsets, "stability": stability, "annual": annual}


def _passes(row, protocol):
    return (
        row["p_holm_wave"] < protocol["comparisons"]["wave_alpha"]
        and row["p_holm_cumulative"] < protocol["comparisons"]["cumulative_alpha"]
        and all(
            phase["daily"]["mean"] <= -0.005
            and phase["active"]["mean"] < 0
            and all(
                x["daily_delta"] < 0 and x["active_delta"] < 0
                for x in phase["offsets"] + phase["stability"]
            )
            for phase in row["phases"]
        )
    )


def evaluate(panel, calendar, activation_mask, prior, protocol, support_report):
    validate(protocol)
    _prior(prior)
    _activation(activation_mask, calendar)
    parts = _paired(panel, calendar, protocol)
    common = parts["candidate"]
    _support_report(common, calendar, activation_mask, protocol, support_report)
    inf = protocol["inference"]
    rows = []
    for control in CONTROLS:
        row = {
            "study": "treasury_dealer",
            "candidate": "candidate",
            "control": control,
            "horizon": 5,
            "score": "qlike",
            "phases": [],
        }
        for phase_code, phase in enumerate(PHASES):
            frame = common[common.phase == phase]
            other = parts[control].loc[frame.index]
            diff = frame.loss.to_numpy() - other.loss.to_numpy()
            start, end = protocol["forecast"][phase]
            phase_calendar = calendar[(calendar >= start) & (calendar <= end)]
            values = pd.Series(diff, index=frame.index).reindex(phase_calendar).to_numpy()
            daily = phase_calendar.isin(frame.index)
            masks = (daily, daily & activation_mask.reindex(phase_calendar).to_numpy())
            item = {"name": phase}
            for endpoint_code, (name, mask) in enumerate(
                zip(("daily", "active"), masks, strict=True)
            ):
                stats = masked_mean_inference(
                    values,
                    mask,
                    blocks=inf["blocks"],
                    hac_lags=inf["hac_lags"],
                    draws=inf["bootstrap_draws"],
                    seed=inf["seed"] + phase_code * 10000 + endpoint_code * 1000000,
                )
                intervals = [
                    stats["hac"]["ci95"],
                    *(r["ci95"] for r in stats["block_inference"].values()),
                ]
                selected = frame.index.isin(phase_calendar[mask])
                stats.update(
                    ci95_envelope=[min(x[0] for x in intervals), max(x[1] for x in intervals)],
                    candidate_loss=float(frame.loss.to_numpy()[selected].mean()),
                    control_loss=float(other.loss.to_numpy()[selected].mean()),
                    first_origin=str(frame.index[selected][0].date()),
                    last_origin=str(frame.index[selected][-1].date()),
                )
                item[name] = stats
            item["p_conservative"] = max(
                item[name]["p_conservative"] for name in ("daily", "active")
            )
            item.update(_diagnostics(frame, diff, calendar, activation_mask, phase, protocol))
            row["phases"].append(item)
        row["p_conservative"] = max(x["p_conservative"] for x in row["phases"])
        rows.append(row)
    wave = holm_adjust([r["p_conservative"] for r in rows])
    cumulative = holm_adjust([r["p_conservative"] for r in prior + rows])[-2:]
    for row, pw, pc in zip(rows, wave, cumulative, strict=True):
        row.update(p_holm_wave=float(pw), p_holm_cumulative=float(pc))
        row["verdict"] = (
            "COMPARISON_GATE_PASS" if _passes(row, protocol) else "DOES_NOT_QUALIFY"
        )
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["treasury_dealer"]
        if all(r["verdict"] == "COMPARISON_GATE_PASS" for r in rows)
        else [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 146,
        "common_scored_origins": len(common),
        "common_active_origins": int(activation_mask.reindex(common.index).sum()),
        "support_report": copy.deepcopy(support_report),
    }


def failure_metrics(error, signature, prior=None):
    _signature(signature)
    if prior is not None:
        _prior(prior)
    unsupported = "INSUFFICIENT_DATA" in str(error)
    result = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "error_kind": "UNSUPPORTED" if unsupported else "FAILED",
        "error": str(error),
        "protocol_sha256": signature,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 146,
        "rows": [
            {
                "study": "treasury_dealer",
                "candidate": "candidate",
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
        result["inherited_rows"] = copy.deepcopy(prior)
    return result
