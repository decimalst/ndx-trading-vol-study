"""Independent daily commodity inference, accounting, and calibration checks."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from src.verify_claims_release_scores import (
    _bootstrap_means,
    _bootstrap_p,
    _calendar,
    _compare,
    _dates,
    _hac,
    _holm,
    _integer,
    _offsets,
    _pair,
    _real,
)

CONTROLS = ("matched", "market")
PHASES = ("development", "evaluation")
REQUIRED_PANEL = {
    "origin",
    "model",
    "prediction",
    "y",
    "loss",
    "target_end",
    "phase",
    "commodity_cutoff_date",
    "offset",
    "fit_origin",
    "training_cutoff",
    "train_n",
}


def _config(protocol):
    if type(protocol) is not dict or type(protocol.get("inference")) is not dict:
        raise ValueError("Inference protocol required")
    inf = protocol["inference"]
    blocks = inf.get("blocks")
    if (
        type(blocks) is not list
        or not blocks
        or any(type(b) is not int for b in blocks)
        or len(set(blocks)) != len(blocks)
    ):
        raise ValueError("Distinct integer bootstrap block lengths required")
    for block in blocks:
        _integer(block, "block")
    for name in (
        "bootstrap_draws",
        "minimum_phase_observations",
        "minimum_slice_observations",
        "minimum_offset_observations",
    ):
        _integer(inf.get(name), name)
    _integer(inf.get("seed"), "seed", 0)
    _integer(inf.get("hac_lags"), "HAC lags", 0)
    for name in ("wave_alpha", "cumulative_alpha"):
        if _real(inf.get(name), name, 0, 1) == 0:
            raise ValueError("Positive significance threshold required")
    if _real(inf.get("effect_threshold_absolute"), "effect threshold", 0) == 0:
        raise ValueError("Positive effect threshold required")
    if any("release" in str(key) for key in inf):
        raise ValueError("Daily commodity inference has no release-specific support")
    forecast = protocol.get("forecast")
    if type(forecast) is not dict:
        raise ValueError("Forecast configuration required")
    _integer(forecast.get("minimum_train"), "training floor")
    bounds = {name: _pair(forecast.get(name)) for name in PHASES}
    source_end = _pair([forecast.get("source_end"), forecast.get("source_end")])[0]
    if (
        source_end > pd.Timestamp("2025-10-20")
        or bounds["development"][1] >= bounds["evaluation"][0]
        or any(end > source_end for _, end in bounds.values())
    ):
        raise ValueError("Ordered phases within fixed source ceiling required")
    slices = protocol.get("evaluation_stability")
    if type(slices) is not list or len(slices) != 2:
        raise ValueError("Two evaluation stability slices required")
    slice_bounds = [_pair(values) for values in slices]
    if slice_bounds[0][1] >= slice_bounds[1][0] or any(
        a < bounds["evaluation"][0] or z > bounds["evaluation"][1] for a, z in slice_bounds
    ):
        raise ValueError("Nonoverlapping slices inside evaluation required")
    comparisons = protocol.get("comparisons")
    expected = {
        "controls": list(CONTROLS),
        "new_hypotheses": 2,
        "inherited_hypotheses": 142,
        "cumulative_hypotheses": 144,
    }
    if type(comparisons) is not dict:
        raise ValueError("Declared commodity hypothesis family required")
    for key, value in expected.items():
        if type(comparisons.get(key)) is not type(value) or comparisons[key] != value:
            raise ValueError(
                "Exactly two fixed comparisons and inherited142/cumulative144 required"
            )
    return inf, bounds, source_end, slice_bounds


def _prior(prior):
    if type(prior) is not list or len(prior) != 142:
        raise ValueError("Exactly 142 unchanged inherited hypotheses required")
    probabilities, identities, signatures = [], set(), {}
    for row in prior:
        if type(row) is not dict:
            raise ValueError("Inherited hypothesis records required")
        probabilities.append(_real(row.get("p_conservative"), "inherited probability", 0, 1))
        for key in ("p_holm_wave", "p_holm_cumulative"):
            if key in row:
                _real(row[key], key, 0, 1)
        for key in ("study", "candidate", "control", "source"):
            if type(row.get(key)) is not str or not row[key]:
                raise ValueError("Nonempty inherited identity and provenance required")
        _integer(row.get("horizon"), "inherited horizon")
        position = _integer(row.get("source_row_index"), "source row index", 0)
        signature = row.get("source_sha256")
        if type(signature) is not str or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
            raise ValueError("Literal inherited source SHA256 required")
        source = row["source"]
        if (source, position) in identities or (
            source in signatures and signatures[source] != signature
        ):
            raise ValueError("Unique source rows and consistent source hashes required")
        identities.add((source, position))
        signatures[source] = signature
    return probabilities


def _panel(panel, calendar, bounds, source_end, minimum_train):
    _calendar(calendar)
    if (calendar > source_end).any():
        raise ValueError("Calendar exceeds source ceiling")
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.columns.has_duplicates
        or not set(panel.columns) >= REQUIRED_PANEL
    ):
        raise ValueError("Complete commodity scored panel required")
    if panel.empty:
        raise ValueError("INSUFFICIENT_DATA: empty paired panel")
    p = panel.copy(deep=True)
    for field in (
        "origin",
        "target_end",
        "commodity_cutoff_date",
        "fit_origin",
        "training_cutoff",
    ):
        p[field] = _dates(p[field])
    if set(p.model) != {"candidate", *CONTROLS} or not set(p.phase) <= set(PHASES):
        raise ValueError("Exactly three declared arms and two phase labels required")
    if p.duplicated(["origin", "model"]).any() or not (p.groupby("origin").size() == 3).all():
        raise ValueError("Exactly one row for every common origin and arm required")
    for field in ("prediction", "y", "loss"):
        dtype = p[field].dtype
        if pd.api.types.is_bool_dtype(dtype) or not (
            pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(dtype)
        ):
            raise ValueError("Nonboolean real scored values required")
        if not np.isfinite(p[field].to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError("Finite scored values required")
    if (p.prediction <= 0).any() or (p.y <= 0).any():
        raise ValueError("Positive targets and predictions required")
    shared = (
        "y",
        "target_end",
        "phase",
        "commodity_cutoff_date",
        "offset",
        "fit_origin",
        "training_cutoff",
        "train_n",
    )
    for field in shared:
        if (
            p[field].isna().any()
            or not (p.groupby("origin")[field].nunique(dropna=False) == 1).all()
        ):
            raise ValueError(f"Paired metadata mismatch: {field}")
    for field in ("offset", "train_n"):
        if pd.api.types.is_bool_dtype(p[field].dtype) or not pd.api.types.is_integer_dtype(
            p[field].dtype
        ):
            raise ValueError("Integer offsets and training counts required")
    if (p.train_n < minimum_train).any():
        raise ValueError("Saved training count violates the fixed floor")
    positions = calendar.get_indexer(p.origin)
    if (positions < 1).any() or (positions + 5 >= len(calendar)).any():
        raise ValueError("Scored origin lacks full-calendar prior or fifth target session")
    if not np.array_equal(p.target_end.to_numpy(), calendar[positions + 5].to_numpy()):
        raise ValueError("Target endpoint must be the fifth full-calendar session")
    if not np.array_equal(p.offset.to_numpy(), positions % 5):
        raise ValueError("Offsets must precede cohort filtering")
    if not np.array_equal(
        p.commodity_cutoff_date.to_numpy(), calendar[positions - 1].to_numpy()
    ):
        raise ValueError("Commodity cutoff must be the previous full-calendar session")
    if (p.commodity_cutoff_date < pd.Timestamp("2009-01-02")).any():
        raise ValueError("Commodity inputs precede the source floor")
    fit_positions = calendar.get_indexer(p.fit_origin)
    if (fit_positions < 1).any() or (fit_positions > positions).any():
        raise ValueError("Known nonfuture monthly fit origin required")
    if not np.array_equal(
        pd.DatetimeIndex(p.fit_origin).to_period("M"),
        pd.DatetimeIndex(p.origin).to_period("M"),
    ) or not np.array_equal(
        p.training_cutoff.to_numpy(), calendar[fit_positions - 1].to_numpy()
    ):
        raise ValueError("Monthly fit and previous-session training cutoff disagree")
    for name, (start, end) in bounds.items():
        selected = p.phase == name
        fence = end if name == "development" else source_end
        if (
            (p.loc[selected, "origin"] < start)
            | (p.loc[selected, "origin"] > end)
            | (p.loc[selected, "target_end"] > fence)
        ).any():
            raise ValueError("Phase origin or target maturity fence violated")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            ratio = p.y.to_numpy(dtype=float) / p.prediction.to_numpy(dtype=float)
            loss = ratio - np.log(ratio) - 1
    except FloatingPointError as error:
        raise ValueError("Invalid QLIKE arithmetic") from error
    if not np.isfinite(loss).all() or not np.allclose(p.loss, loss, atol=1e-10, rtol=1e-8):
        raise ValueError("Saved QLIKE differs from target/prediction reconstruction")
    p["loss"] = loss
    return p.sort_values(["origin", "model"])


def _support(n, minimum, label):
    if n < minimum:
        raise ValueError(f"INSUFFICIENT_DATA: {label}")


def _expected_scores(panel, calendar, prior, protocol):
    inf, bounds, source_end, slices = _config(protocol)
    p = _panel(panel, calendar, bounds, source_end, protocol["forecast"]["minimum_train"])
    old_p = _prior(prior)
    common = p[p.model == "candidate"].set_index("origin").sort_index()
    controls = {name: p[p.model == name].set_index("origin").sort_index() for name in CONTROLS}
    # Check all support before drawing any bootstrap samples.
    phases = {}
    for phase_name in PHASES:
        part = common[common.phase == phase_name]
        offsets = _offsets(part.index, calendar)
        _support(
            len(part), max(inf["minimum_phase_observations"], max(inf["blocks"])), phase_name
        )
        for k in range(5):
            _support(
                int((offsets == k).sum()),
                inf["minimum_offset_observations"],
                f"{phase_name} offset {k}",
            )
        if phase_name == "evaluation":
            for a, z in slices:
                _support(
                    int(((part.index >= a) & (part.index <= z)).sum()),
                    inf["minimum_slice_observations"],
                    "evaluation slice",
                )
        phases[phase_name] = (part, offsets)
    rows = [
        {
            "study": "commodity_implied",
            "candidate": "candidate",
            "control": name,
            "horizon": 5,
            "score": "qlike",
            "phases": [],
        }
        for name in CONTROLS
    ]
    for phase_code, phase_name in enumerate(PHASES):
        part, offsets = phases[phase_name]
        origins = part.index
        candidate_loss = part.loss.to_numpy(dtype=float)
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
        start, end = bounds[phase_name]
        bounded = calendar[(calendar >= start) & (calendar <= end)]
        for j, row in enumerate(rows):
            values = delta[:, j]
            hac = _hac(values, inf["hac_lags"])
            blocks = {
                key: {"p": float(probabilities[j]), "ci95": [float(ci[0, j]), float(ci[1, j])]}
                for key, (probabilities, ci) in bootstrap.items()
            }
            annual = []
            for year in sorted(set(bounded.year)):
                selected = origins.year == year
                count = int(selected.sum())
                calendar_n = int((bounded.year == year).sum())
                annual.append(
                    {
                        "year": int(year),
                        "n": count,
                        "delta": float(values[selected].mean()) if count else None,
                        "calendar_origins": calendar_n,
                        "missing_origins": calendar_n - count,
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
                            "delta": float(values[selected].mean()),
                        }
                    )
            row["phases"].append(
                {
                    "name": phase_name,
                    "n": len(part),
                    "delta": float(means[j]),
                    "candidate_loss": float(candidate_loss.mean()),
                    "control_loss": float(control_losses[:, j].mean()),
                    "first_origin": origins[0].strftime("%Y-%m-%d"),
                    "last_origin": origins[-1].strftime("%Y-%m-%d"),
                    "block_inference": blocks,
                    "hac": hac,
                    "p_conservative": max(
                        hac["p"], *[block["p"] for block in blocks.values()]
                    ),
                    "ci95_envelope": [
                        min(hac["ci95"][0], *[block["ci95"][0] for block in blocks.values()]),
                        max(hac["ci95"][1], *[block["ci95"][1] for block in blocks.values()]),
                    ],
                    "annual": annual,
                    "offsets": [
                        {
                            "offset": k,
                            "n": int((offsets == k).sum()),
                            "delta": float(values[offsets == k].mean()),
                        }
                        for k in range(5)
                    ],
                    "stability": stability,
                }
            )
    probabilities = [max(phase["p_conservative"] for phase in row["phases"]) for row in rows]
    wave, cumulative = _holm(probabilities), _holm(old_p + probabilities)[-2:]
    for j, row in enumerate(rows):
        passed = wave[j] < inf["wave_alpha"] and cumulative[j] < inf["cumulative_alpha"]
        for phase in row["phases"]:
            passed &= phase["delta"] <= -inf["effect_threshold_absolute"]
            passed &= all(item["delta"] < 0 for item in phase["offsets"] + phase["stability"])
        row.update(
            p_conservative=float(probabilities[j]),
            p_holm_wave=float(wave[j]),
            p_holm_cumulative=float(cumulative[j]),
            verdict="COMPARISON_GATE_PASS" if passed else "DOES_NOT_QUALIFY",
        )
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["commodity_implied"]
        if all(row["verdict"] == "COMPARISON_GATE_PASS" for row in rows)
        else [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 144,
        "common_scored_origins": len(common),
    }


def verify_scores(panel, calendar, metrics, prior, protocol):
    """Reconstruct both daily comparisons and every serialized result field."""
    expected = _expected_scores(panel, calendar, prior, protocol)
    _compare(metrics, expected)
    return {
        "status": "VERIFIED",
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 144,
        "common_scored_origins": expected["common_scored_origins"],
    }


def verify_calibration(protocol, result):
    """Reconstruct generated stationary-null coverage without producer imports."""
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
        raise ValueError("Exact calibration settings required")
    rho = _real(settings["rho"], "rho", -1, 1)
    if abs(rho) == 1:
        raise ValueError("Stationary null correlation required")
    for key in ("n", "replications", "bootstrap_draws"):
        _integer(settings[key], key)
    _integer(settings["burn_in"], "burn_in", 0)
    threshold = _real(settings["minimum_envelope_coverage"], "minimum coverage", 0, 1)
    seed = inf["seed"]
    rng = np.random.default_rng(seed)
    hits = 0
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
        lower, upper = _hac(retained, inf["hac_lags"])["ci95"]
        for block in inf["blocks"]:
            samples = _bootstrap_means(
                retained, block, settings["bootstrap_draws"], seed + trial * 1000 + block
            )[:, 0]
            a, z = np.quantile(samples, [0.025, 0.975])
            block_hits[str(block)] += int(a <= 0 <= z)
            lower, upper = min(lower, a), max(upper, z)
        hits += int(lower <= 0 <= upper)
    coverage = hits / settings["replications"]
    expected = {
        "status": "PASS" if coverage >= threshold else "FAIL",
        **settings,
        "seed": seed,
        "envelope_coverage": coverage,
        "individual_block_coverage": {
            key: count / settings["replications"] for key, count in block_hits.items()
        },
        "limitation": "One stationary synthetic null; not market-data coverage certification",
    }
    _compare(result, expected, exact=True)
    return {"status": "VERIFIED"}
