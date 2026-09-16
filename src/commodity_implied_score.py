"""Fixed daily paired QLIKE inference and complete commodity family accounting."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from src.orthogonal_round2 import bootstrap_means, hac_summary, holm_adjust

CONTROLS = ("matched", "market")
MODELS = ("candidate", "matched", "market")
PRIOR_SOURCE = "reports/claims_release/predictive/metrics.json"


def _signature(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Literal SHA256 required")


def _probabilities(row):
    if type(row) is not dict or "p_conservative" not in row:
        raise ValueError("Explicit comparison probability required")
    for name in ("p_conservative", "p_holm_wave", "p_holm_cumulative"):
        if name in row:
            value = row[name]
            if (
                type(value) not in (int, float)
                or not np.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Literal finite comparison probabilities required")


def _identity(row):
    if any(
        type(row.get(name)) is not str or not row[name]
        for name in ("study", "candidate", "control")
    ):
        raise ValueError("Complete named comparison identity required")
    if type(row.get("horizon")) is not int or row["horizon"] <= 0:
        raise ValueError("Positive exact horizon identity required")


def _prior(rows, count=142):
    if type(rows) is not list or len(rows) != count:
        raise ValueError(f"All {count} inherited comparisons required")
    identities, source_hashes = set(), {}
    for row in rows:
        _probabilities(row)
        _identity(row)
        source, position = row.get("source"), row.get("source_row_index")
        if type(source) is not str or not source or type(position) is not int or position < 0:
            raise ValueError("Explicit inherited source and row identity required")
        signature = row.get("source_sha256")
        _signature(signature)
        identity = (source, position)
        if identity in identities or source_hashes.get(source, signature) != signature:
            raise ValueError(
                "Unique inherited source rows and consistent source hashes required"
            )
        identities.add(identity)
        source_hashes[source] = signature


def inherit_family(previous, signature):
    """Preserve the completed claims family verbatim, with authenticated row pins."""
    _signature(signature)
    if (
        type(previous) is not dict
        or previous.get("status") != "COMPLETED"
        or type(previous.get("hypothesis_count")) is not int
        or previous["hypothesis_count"] != 2
        or type(previous.get("cumulative_hypothesis_count")) is not int
        or previous["cumulative_hypothesis_count"] != 142
    ):
        raise ValueError(
            "Completed prior two-comparison claims family with cumulative142 required"
        )
    _prior(previous.get("inherited_rows"), 140)
    rows = previous.get("rows")
    if type(rows) is not list or len(rows) != 2:
        raise ValueError("Both completed claims comparison rows required")
    result = copy.deepcopy(previous["inherited_rows"])
    for number, (row, control) in enumerate(zip(rows, CONTROLS)):
        _probabilities(row)
        _identity(row)
        if (
            row["study"],
            row["candidate"],
            row["control"],
            row["horizon"],
            row.get("score"),
        ) != ("claims_release", "candidate", control, 5, "qlike"):
            raise ValueError("Exact ordered prior claims comparison identities required")
        if row.get("verdict") not in ("COMPARISON_GATE_PASS", "DOES_NOT_QUALIFY"):
            raise ValueError("Completed prior comparison verdict required")
        result.append(
            {
                **copy.deepcopy(row),
                "source": PRIOR_SOURCE,
                "source_sha256": signature,
                "source_row_index": number,
            }
        )
    _prior(result)
    return result


def _paired(panel, calendar, protocol):
    required = {"origin", "model", "prediction", "y", "loss", "target_end", "phase", "offset"}
    if (
        not isinstance(panel, pd.DataFrame)
        or not panel.columns.is_unique
        or not required <= set(panel)
        or panel.duplicated(["origin", "model"]).any()
    ):
        raise ValueError("Unique complete paired panel required")
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.tz is not None
        or calendar.hasnans
        or calendar.has_duplicates
        or not len(calendar)
        or not calendar.is_monotonic_increasing
        or not calendar.equals(calendar.normalize())
    ):
        raise ValueError("Full ordered naive midnight reference calendar required")
    source_end = pd.Timestamp(protocol["forecast"]["source_end"])
    if calendar.max() > source_end or source_end > pd.Timestamp("2025-10-20"):
        raise ValueError("Fixed source ceiling required")
    if panel.empty:
        raise ValueError("INSUFFICIENT_DATA: no common scored origins")
    if set(panel.model) != set(MODELS):
        raise ValueError("Exactly three declared model arms required")
    for name in ("origin", "target_end"):
        if (
            not pd.api.types.is_datetime64_any_dtype(panel[name].dtype)
            or panel[name].isna().any()
            or getattr(panel[name].dtype, "tz", None) is not None
            or not panel[name].equals(panel[name].dt.normalize())
        ):
            raise ValueError("Known native naive midnight scored dates required")
    for name in ("prediction", "y", "loss"):
        dtype = panel[name].dtype
        if pd.api.types.is_bool_dtype(dtype) or not (
            pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)
        ):
            raise ValueError("Real nonboolean scored numbers required")
    if pd.api.types.is_bool_dtype(panel.offset.dtype) or not pd.api.types.is_integer_dtype(
        panel.offset.dtype
    ):
        raise ValueError("Exact integer full-calendar offsets required")
    values = panel[["prediction", "y", "loss"]].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or (values[:, :2] <= 0).any():
        raise ValueError("Finite positive targets and predictions required")
    try:
        with np.errstate(over="raise", under="raise", divide="raise", invalid="raise"):
            ratio = values[:, 1] / values[:, 0]
            expected_loss = ratio - np.log(ratio) - 1
    except FloatingPointError as error:
        raise ValueError("Invalid QLIKE arithmetic") from error
    if not np.array_equal(values[:, 2], expected_loss):
        raise ValueError("Saved QLIKE differs from targets and forecasts")
    positions = calendar.get_indexer(panel.origin)
    if (positions < 0).any() or (positions + 5 >= len(calendar)).any():
        raise ValueError("Scored origin/target must lie on full reference calendar")
    if not np.array_equal(panel.target_end.to_numpy(), calendar[positions + 5].to_numpy()):
        raise ValueError("Exact fifth subsequent observed target end required")
    if panel.offset.isna().any() or not np.array_equal(panel.offset.to_numpy(), positions % 5):
        raise ValueError("Offsets must use the full unfiltered reference calendar")
    dev = panel.origin.between(*protocol["forecast"]["development"])
    evaluation = panel.origin.between(*protocol["forecast"]["evaluation"])
    if (
        not (dev | evaluation).all()
        or (dev & evaluation).any()
        or not np.array_equal(panel.phase, np.where(dev, "development", "evaluation"))
        or not panel.target_end.le(source_end).all()
        or not panel.loc[dev, "target_end"]
        .le(pd.Timestamp(protocol["forecast"]["development"][1]))
        .all()
    ):
        raise ValueError("Fixed phase and target-maturity boundaries required")
    parts = {
        name: panel[panel.model == name].set_index("origin").sort_index() for name in MODELS
    }
    candidate = parts["candidate"]
    common = [name for name in panel if name not in ("origin", "model", "prediction", "loss")]
    for control in CONTROLS:
        other = parts[control]
        if not candidate.index.equals(other.index) or any(
            not candidate[name].equals(other[name]) for name in common
        ):
            raise ValueError("All arms must share exact targets, cohorts and common metadata")
    return parts


def _support(n, floor, label):
    if type(floor) is not int or floor <= 0:
        raise ValueError("Positive exact daily support floor required")
    if n < floor:
        raise ValueError(f"INSUFFICIENT_DATA: {label} has {n} origins")


def _diagnostics(frame, differences, calendar, phase_name, protocol):
    config = protocol["inference"]
    series = pd.Series(differences, index=frame.index)
    _support(len(frame), config["minimum_phase_observations"], phase_name)
    offsets = []
    for offset in range(5):
        chosen = series[frame.offset == offset]
        _support(
            len(chosen), config["minimum_offset_observations"], f"{phase_name} offset{offset}"
        )
        offsets.append({"offset": offset, "n": len(chosen), "delta": float(chosen.mean())})
    stability = []
    if phase_name == "evaluation":
        for start, end in protocol["evaluation_stability"]:
            chosen = frame.index.to_series().between(start, end)
            n = int(chosen.sum())
            _support(n, config["minimum_slice_observations"], "evaluation slice")
            stability.append(
                {"start": start, "end": end, "n": n, "delta": float(series[chosen].mean())}
            )
    start, end = map(pd.Timestamp, protocol["forecast"][phase_name])
    full_phase = calendar[(calendar >= start) & (calendar <= end)]
    annual = []
    for year in sorted(set(full_phase.year)):
        chosen = frame.index.year == year
        n, total = int(chosen.sum()), int((full_phase.year == year).sum())
        annual.append(
            {
                "year": int(year),
                "n": n,
                "delta": float(series[chosen].mean()) if n else None,
                "calendar_origins": total,
                "missing_origins": total - n,
            }
        )
    return {"annual": annual, "offsets": offsets, "stability": stability}


def _passes(row, config):
    return (
        row["p_holm_wave"] < config["wave_alpha"]
        and row["p_holm_cumulative"] < config["cumulative_alpha"]
        and all(
            phase["delta"] <= -config["effect_threshold_absolute"]
            and all(item["delta"] < 0 for item in phase["offsets"])
            and all(item["delta"] < 0 for item in phase["stability"])
            for phase in row["phases"]
        )
    )


def evaluate(panel, calendar, prior, protocol):
    """Evaluate both daily contrasts; unsupported phases invalidate the whole wave."""
    _prior(prior)
    parts = _paired(panel, calendar, protocol)
    config = protocol["inference"]
    rows = [
        {
            "study": "commodity_implied",
            "candidate": "candidate",
            "control": control,
            "horizon": 5,
            "score": "qlike",
            "phases": [],
        }
        for control in CONTROLS
    ]
    phases = []
    # Complete both phases' common-sample/support checks before any resampling.
    for phase_name in ("development", "evaluation"):
        candidate = parts["candidate"].loc[parts["candidate"].phase == phase_name]
        controls = [parts[name].loc[candidate.index] for name in CONTROLS]
        matrix = np.column_stack(
            [candidate.loss.to_numpy() - other.loss.to_numpy() for other in controls]
        )
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite paired differences")
        diagnostics = [
            _diagnostics(candidate, matrix[:, j], calendar, phase_name, protocol)
            for j in range(2)
        ]
        if any(block > len(candidate) for block in config["blocks"]):
            raise ValueError(
                "INSUFFICIENT_DATA: phase cannot support the declared bootstrap block"
            )
        phases.append((phase_name, candidate, controls, matrix, diagnostics))
    for phase_code, (phase_name, candidate, controls, matrix, diagnostics) in enumerate(
        phases
    ):
        means = matrix.mean(axis=0)
        blocks = [{}, {}]
        for block in config["blocks"]:
            sample = bootstrap_means(
                matrix,
                block,
                config["bootstrap_draws"],
                config["seed"] + 5 * 1000000 + phase_code * 10000 + block,
            )
            for column in range(2):
                probability = (
                    1
                    + int(
                        np.sum(np.abs(sample[:, column] - means[column]) >= abs(means[column]))
                    )
                ) / (config["bootstrap_draws"] + 1)
                blocks[column][str(block)] = {
                    "p": probability,
                    "ci95": np.quantile(sample[:, column], [0.025, 0.975]).tolist(),
                }
        for column, row in enumerate(rows):
            hac = hac_summary(matrix[:, column], config["hac_lags"])
            intervals = [hac["ci95"], *(value["ci95"] for value in blocks[column].values())]
            row["phases"].append(
                {
                    "name": phase_name,
                    "n": len(candidate),
                    "delta": float(means[column]),
                    "candidate_loss": float(candidate.loss.mean()),
                    "control_loss": float(controls[column].loss.mean()),
                    "first_origin": str(candidate.index[0].date()),
                    "last_origin": str(candidate.index[-1].date()),
                    "block_inference": blocks[column],
                    "hac": hac,
                    "p_conservative": max(
                        hac["p"], *(value["p"] for value in blocks[column].values())
                    ),
                    "ci95_envelope": [
                        min(item[0] for item in intervals),
                        max(item[1] for item in intervals),
                    ],
                    **diagnostics[column],
                }
            )
    for row in rows:
        row["p_conservative"] = max(phase["p_conservative"] for phase in row["phases"])
    wave = holm_adjust([row["p_conservative"] for row in rows])
    cumulative = holm_adjust([row["p_conservative"] for row in prior + rows])[-2:]
    for row, pw, pc in zip(rows, wave, cumulative):
        row.update(p_holm_wave=float(pw), p_holm_cumulative=float(pc))
        row["verdict"] = "COMPARISON_GATE_PASS" if _passes(row, config) else "DOES_NOT_QUALIFY"
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["commodity_implied"]
        if all(row["verdict"] == "COMPARISON_GATE_PASS" for row in rows)
        else [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 144,
        "common_scored_origins": len(parts["candidate"]),
    }


def failure_metrics(error, signature):
    _signature(signature)
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    return {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 144,
        "protocol_sha256": signature,
        "rows": [
            {
                "study": "commodity_implied",
                "candidate": "candidate",
                "control": control,
                "horizon": 5,
                "score": "qlike",
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "phases": [],
                "status": status,
                "error": str(error),
            }
            for control in CONTROLS
        ],
    }


def calibrate(protocol):
    """Fixed generated serial null; no source access or empirical calibration."""
    config, inf = protocol["calibration"], protocol["inference"]
    rng = np.random.default_rng(inf["seed"])
    hits = 0
    per_block = {str(block): 0 for block in inf["blocks"]}
    for trial in range(config["replications"]):
        shocks = rng.normal(
            scale=np.sqrt(1 - config["rho"] ** 2), size=config["n"] + config["burn_in"]
        )
        state, values = rng.normal(), []
        for shock in shocks:
            state = config["rho"] * state + shock
            values.append(state)
        differences = np.asarray(values[config["burn_in"] :])
        low, high = hac_summary(differences, inf["hac_lags"])["ci95"]
        for block in inf["blocks"]:
            sample = bootstrap_means(
                differences,
                block,
                config["bootstrap_draws"],
                inf["seed"] + trial * 1000 + block,
            )
            a, b = np.quantile(sample[:, 0], [0.025, 0.975])
            per_block[str(block)] += int(a <= 0 <= b)
            low, high = min(low, a), max(high, b)
        hits += int(low <= 0 <= high)
    coverage = hits / config["replications"]
    return {
        "status": "PASS" if coverage >= config["minimum_envelope_coverage"] else "FAIL",
        **config,
        "seed": inf["seed"],
        "envelope_coverage": coverage,
        "individual_block_coverage": {
            key: value / config["replications"] for key, value in per_block.items()
        },
        "limitation": "One stationary synthetic null; not market-data coverage certification",
    }
