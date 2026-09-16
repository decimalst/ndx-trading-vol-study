"""Fixed paired QLIKE inference and complete claims comparison accounting."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from src.orthogonal_round2 import bootstrap_means, hac_summary, holm_adjust

CONTROLS = ("matched", "market")
MODELS = ("candidate", "matched", "market")


def _prior(rows, count=140):
    if type(rows) is not list or len(rows) != count:
        raise ValueError(f"All {count} inherited comparisons required")
    for row in rows:
        value = row.get("p_conservative")
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Literal finite inherited probabilities required")


def inherit_family(previous, signature):
    if (
        previous.get("hypothesis_count") != 3
        or previous.get("cumulative_hypothesis_count") != 140
        or type(signature) is not str
        or re.fullmatch(r"[0-9a-f]{64}", signature) is None
    ):
        raise ValueError("Verified previous complete140 family required")
    _prior(previous["inherited_rows"], 137)
    _prior(previous["rows"], 3)
    result = copy.deepcopy(previous["inherited_rows"])
    for number, row in enumerate(previous["rows"]):
        result.append(
            {
                **copy.deepcopy(row),
                "source": "reports/peak_age/metrics.json",
                "source_sha256": signature,
                "source_row_index": number,
            }
        )
    return result


def _paired(panel, calendar, protocol):
    required = {
        "origin",
        "model",
        "prediction",
        "y",
        "loss",
        "target_end",
        "phase",
        "claim_reference_week",
        "offset",
    }
    if (
        not required <= set(panel)
        or panel.duplicated(["origin", "model"]).any()
    ):
        raise ValueError("Unique complete paired panel required")
    if panel.empty:
        raise ValueError("INSUFFICIENT_DATA: no common scored origins")
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.tz is not None
        or calendar.hasnans
        or calendar.has_duplicates
        or not calendar.is_monotonic_increasing
        or not calendar.equals(calendar.normalize())
        or set(panel.model) != set(MODELS)
    ):
        raise ValueError("Full ordered calendar and exactly three models required")
    source_end = pd.Timestamp(protocol["forecast"]["source_end"])
    if calendar.max() > source_end or source_end > pd.Timestamp("2025-10-20"):
        raise ValueError("Fixed source ceiling required")
    for name in ("origin", "target_end", "claim_reference_week"):
        if (
            not pd.api.types.is_datetime64_any_dtype(panel[name].dtype)
            or panel[name].isna().any()
        ):
            raise ValueError("Known native scored dates required")
        if getattr(panel[name].dtype, "tz", None) is not None:
            raise ValueError("Naive scored dates required")
    values = panel[["prediction", "y", "loss"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :2] <= 0).any():
        raise ValueError("Finite positive targets and predictions required")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        ratio = values[:, 1] / values[:, 0]
        expected_loss = ratio - np.log(ratio) - 1
    if not np.array_equal(values[:, 2], expected_loss):
        raise ValueError("Saved QLIKE differs from targets and forecasts")
    positions = calendar.get_indexer(panel.origin)
    if (positions < 0).any() or (positions + 5 >= len(calendar)).any():
        raise ValueError("Scored origin/target must lie on full reference calendar")
    if not np.array_equal(panel.target_end.to_numpy(), calendar[positions + 5].to_numpy()):
        raise ValueError("Exact fifth subsequent observed target end required")
    if not np.array_equal(panel.offset.to_numpy(), positions % 5):
        raise ValueError("Offsets must use the full unfiltered reference calendar")
    if not (panel.claim_reference_week < panel.origin).all():
        raise ValueError("Claims reference week must precede origin")
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
    for name in CONTROLS:
        other = parts[name]
        if not candidate.index.equals(other.index) or any(
            not candidate[col].equals(other[col]) for col in common
        ):
            raise ValueError("All arms must share exact targets, cohorts and metadata")
    return parts


def _support(n, releases, floor, release_floor, label):
    if n < floor or releases < release_floor:
        raise ValueError(
            f"INSUFFICIENT_DATA: {label} has {n} origins and {releases} distinct releases"
        )


def _diagnostics(frame, differences, calendar, phase_name, protocol):
    config = protocol["inference"]
    series = pd.Series(differences, index=frame.index)
    releases = frame.claim_reference_week
    count = int(releases.nunique())
    _support(
        len(frame),
        count,
        config["minimum_phase_observations"],
        config["minimum_phase_releases"],
        phase_name,
    )
    offsets = []
    for offset in range(5):
        chosen = series[frame.offset == offset]
        if len(chosen) < config["minimum_offset_observations"]:
            raise ValueError(
                f"INSUFFICIENT_DATA: {phase_name} offset{offset} has {len(chosen)} origins"
            )
        offsets.append({"offset": offset, "n": len(chosen), "delta": float(chosen.mean())})
    stability = []
    if phase_name == "evaluation":
        for start, end in protocol["evaluation_stability"]:
            chosen = frame.index.to_series().between(start, end)
            n, distinct = int(chosen.sum()), int(releases[chosen].nunique())
            _support(
                n,
                distinct,
                config["minimum_slice_observations"],
                config["minimum_slice_releases"],
                "evaluation slice",
            )
            stability.append(
                {
                    "start": start,
                    "end": end,
                    "n": n,
                    "distinct_releases": distinct,
                    "delta": float(series[chosen].mean()),
                }
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
                "distinct_releases": int(releases[chosen].nunique()),
                "delta": float(series[chosen].mean()) if n else None,
                "calendar_origins": total,
                "missing_origins": total - n,
            }
        )
    return {
        "distinct_releases": count,
        "release_weighted_delta": float(series.groupby(releases).mean().mean()),
        "annual": annual,
        "offsets": offsets,
        "stability": stability,
    }


def _passes(row, config):
    return (
        row["p_holm_wave"] < config["wave_alpha"]
        and row["p_holm_cumulative"] < config["cumulative_alpha"]
        and all(
            phase["delta"] <= -config["effect_threshold_absolute"]
            and phase["release_weighted_delta"] < 0
            and all(x["delta"] < 0 for x in phase["offsets"])
            and all(x["delta"] < 0 for x in phase["stability"])
            for phase in row["phases"]
        )
    )


def evaluate(panel, calendar, prior, protocol):
    """Evaluate exactly two complete contrasts; any unsupported phase aborts both."""
    _prior(prior)
    parts = _paired(panel, calendar, protocol)
    config = protocol["inference"]
    rows = [
        {
            "study": "claims_release",
            "candidate": "candidate",
            "control": control,
            "horizon": 5,
            "score": "qlike",
            "phases": [],
        }
        for control in CONTROLS
    ]
    for phase_code, phase_name in enumerate(("development", "evaluation")):
        a = parts["candidate"].loc[parts["candidate"].phase == phase_name]
        controls = [parts[name].loc[a.index] for name in CONTROLS]
        matrix = np.column_stack([a.loss.to_numpy() - b.loss.to_numpy() for b in controls])
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite paired differences")
        diagnostics = [
            _diagnostics(a, matrix[:, j], calendar, phase_name, protocol) for j in range(2)
        ]
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
                p = (
                    1
                    + int(
                        np.sum(np.abs(sample[:, column] - means[column]) >= abs(means[column]))
                    )
                ) / (config["bootstrap_draws"] + 1)
                blocks[column][str(block)] = {
                    "p": p,
                    "ci95": np.quantile(sample[:, column], [0.025, 0.975]).tolist(),
                }
        for column, row in enumerate(rows):
            hac = hac_summary(matrix[:, column], config["hac_lags"])
            intervals = [hac["ci95"], *(v["ci95"] for v in blocks[column].values())]
            row["phases"].append(
                {
                    "name": phase_name,
                    "n": len(a),
                    "delta": float(means[column]),
                    "candidate_loss": float(a.loss.mean()),
                    "control_loss": float(controls[column].loss.mean()),
                    "first_origin": str(a.index[0].date()),
                    "last_origin": str(a.index[-1].date()),
                    "block_inference": blocks[column],
                    "hac": hac,
                    "p_conservative": max(
                        hac["p"], *(v["p"] for v in blocks[column].values())
                    ),
                    "ci95_envelope": [
                        min(x[0] for x in intervals),
                        max(x[1] for x in intervals),
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
    leads = (
        ["claims_first_report"]
        if all(row["verdict"] == "COMPARISON_GATE_PASS" for row in rows)
        else []
    )
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": leads,
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 142,
        "common_scored_origins": len(parts["candidate"]),
    }


def failure_metrics(error, signature):
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    return {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 142,
        "protocol_sha256": signature,
        "rows": [
            {
                "study": "claims_release",
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
    """Fixed generated serial null; never reads or writes historical sources."""
    config, inf = protocol["calibration"], protocol["inference"]
    rng = np.random.default_rng(inf["seed"])
    hits = 0
    per_block = {str(b): 0 for b in inf["blocks"]}
    for trial in range(config["replications"]):
        shocks = rng.normal(
            scale=np.sqrt(1 - config["rho"] ** 2), size=config["n"] + config["burn_in"]
        )
        state, values = rng.normal(), []
        for shock in shocks:
            state = config["rho"] * state + shock
            values.append(state)
        d = np.asarray(values[config["burn_in"] :])
        lo, hi = hac_summary(d, inf["hac_lags"])["ci95"]
        for block in inf["blocks"]:
            sample = bootstrap_means(
                d, block, config["bootstrap_draws"], inf["seed"] + trial * 1000 + block
            )
            a, b = np.quantile(sample[:, 0], [0.025, 0.975])
            per_block[str(block)] += int(a <= 0 <= b)
            lo, hi = min(lo, a), max(hi, b)
        hits += int(lo <= 0 <= hi)
    coverage = hits / config["replications"]
    return {
        "status": "PASS" if coverage >= config["minimum_envelope_coverage"] else "FAIL",
        **config,
        "seed": inf["seed"],
        "envelope_coverage": coverage,
        "individual_block_coverage": {
            k: v / config["replications"] for k, v in per_block.items()
        },
        "limitation": "One stationary synthetic null; not market-data coverage certification",
    }
