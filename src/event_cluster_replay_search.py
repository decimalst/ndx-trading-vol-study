"""Registered event-adjacency trial, preserved controls and dependence-aware inference."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import cross_moment_score as cs
from . import cross_moment_search as old_inference
from . import event_cluster_pipeline as sm
from . import event_cluster_replay_admission as admission
from . import event_cluster_replay_verification as replay
from . import orthogonal_round2 as inference
from .international_search import diagnostics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "event_cluster_replay.yaml"
REPORT = ROOT / "reports/event_cluster_replay"
OUT = ROOT / "data/event_cluster_replay"
CONTRASTS = tuple(
    ("cluster", c, "brier") for c in ("baseline", "nuisance", "recent_frequency")
)
WAVE_ALPHA = 0.05 / (21 * 22)
EFFECT = 0.0005
CONTRACT = {
    "study_id": "event_cluster_replay_wave21",
    "specified_on": "2026-09-07",
    "status": "specified_before_retained_output_reconstruction_or_inference",
    "wave": 21,
    "evidence_class": "exploratory_technical_replay_of_unverified_archival_SPX_event_clustering",
    "objective": "Complete independent verification of exact retained wave20 producer output under "
    "lossless application-state date representation",
    "original_index": {
        "asset": "SPX",
        "source_end": "2025-10-20",
        "sealed_start": "2025-11-03",
        "origin_start": "2016-01-04",
        "origin_end": "2025-10-17",
        "latest_target": "2025-10-20",
        "development": ["2016-01-04", "2019-12-31"],
        "development_target_available_by": "2019-12-31",
        "evaluation": ["2020-01-02", "2025-10-17"],
        "evaluation_stability": [["2020-01-02", "2022-12-31"], ["2023-01-01", "2025-10-17"]],
        "horizons": [1],
        "market_lag": 1,
        "minimum_train": 1000,
        "models": ["baseline", "recent_frequency", "location"],
        "minimum_train_per_class": 50,
        "minimum_phase_per_class": 30,
        "minimum_slice_per_class": 15,
        "minimum_phase_observations": 127,
        "baseline": [
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
            "neg_d",
            "neg_w",
            "neg_m",
            "skew",
            "I_square",
            "R_square",
            "skew_square",
            "intraday",
            "intraday_sq",
            "overnight",
            "overnight_sq",
        ],
        "all_features": [
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
            "neg_d",
            "neg_w",
            "neg_m",
            "skew",
            "intraday",
            "intraday_sq",
            "overnight",
            "overnight_sq",
            "range_extremity",
        ],
    },
    "history": {
        "window": 22,
        "ordered_by": "available_date on full original SPX calendar, oldest first, through "
        "previous-close cutoff",
        "origin_position_rule": "at i>=23 use y.iloc[i-23:i-1], whose available positions are "
        "i-22..i-1",
        "labels": "all known original target labels, including "
        "unissued/unscored/feature-incomplete origins; not historical forecast "
        "errors",
        "unknown": "every original training/application window must contain22 known labels; "
        "otherwise whole-trial failure before fitting; never fill, compress or "
        "drop original rows",
        "nuisance": [
            "event_fraction22",
            "event_fraction22_sq",
            "last_event",
            "expected_adjacency22",
            "linear_recency22",
        ],
        "nuisance_formula": [
            "N/22",
            "N*N/484",
            "last_event",
            "(N-last_event)*(N-1)/441",
            "sum((2*i-23)*e_i for i=1..22)/462",
        ],
        "candidate": "adjacency_fraction22",
        "candidate_formula": "C/21; C=sum(e_i*e_(i-1) for i=2..22)",
        "diagnostic": "excess_adjacency22=(21*C-(N-last_event)*(N-1))/441; never enters fit "
        "or promotion",
        "arithmetic": "exact Python integer numerators followed by single fixed division; "
        "N*N/484 is not square of rounded N/22",
        "centering": "training-only math.fsum/n; exactconstant uses firstvalue; fixedscale1 "
        "and no rank/column deletion",
    },
    "models": ["baseline", "recent_frequency", "nuisance", "cluster"],
    "model": {
        "baseline": "original wave18 saved26-coefficient monthly fit; no refitting; original "
        "issuedquery probabilities strictlyreplayed and retained",
        "training_offset": "apply that same savedmonthlyfit to its original maturetrainingrows; "
        "in-sample currentfit offsets, not historicalissuances",
        "nuisance": "mean Bernoulli NLL with frozenbaselineoffset +freeintercept +five "
        "centeredboundedslopes, .01sum(slope^2)",
        "candidate": "mean Bernoulli NLL with frozennuisanceoffset +one centeredC/21 slope, "
        ".01coefficient^2; no newintercept",
        "ridge": 0.01,
        "producer_nuisance": {
            "method": "frozen sign_memory_models._newton",
            "start": "six literalzeros",
            "maxiter": 200,
            "max_backtracks": 60,
            "armijo": 0.0001,
            "gradient_tolerance": 1e-08,
        },
        "producer_scalar": {
            "method": "brentq",
            "bracket": "+/- (mean(abs(centered_candidate))+1)/.02",
            "xtol": 1e-12,
            "rtol": 1e-14,
            "maxiter": 200,
            "gradient_tolerance": 1e-08,
        },
        "constant": "center exactconstants exactly; constantnuisances retainzerocoefficients; "
        "constantadjacency retainsscalarzero and nuisanceprobability",
        "probability": "copy checkedparent probability when addedquerycorrection exactlyzero; "
        "otherwise expit(parentlogit+correction); no clipping",
        "scalar_arithmetic": "finitefloat64; signedsoftplus and signedresidual remain nonzero "
        "for finite logits; checked nonzero products rejectunderflow; "
        "compensated scalar reductions",
        "nuisance_arithmetic": "frozen logistic_state numerical contract; finite objective, "
        "fullgradient and Hessian; no optional unusedcurvature admission "
        "gate",
        "fallback": "none; no empiricalrepair or tolerance change",
    },
    "comparisons": {
        "new_hypotheses": 3,
        "inherited_hypotheses": 134,
        "cumulative_hypotheses": 137,
        "inherited_sources": [
            "reports/orthogonal_round2/metrics.json",
            "reports/model_memory_study/combined_metrics.json",
            "reports/iterative_signal_search/metrics.json",
            "reports/international_volatility/metrics.json",
            "reports/overnight_index/metrics.json",
            "reports/macro_overnight/metrics.json",
            "reports/measurement_memory/metrics.json",
            "reports/index_hinge/metrics.json",
            "reports/tail_shape/metrics.json",
            "reports/calendar_variance/metrics.json",
            "reports/relative_risk/metrics.json",
            "reports/joint_risk/metrics.json",
            "reports/cross_moment/metrics.json",
            "reports/target_aligned/metrics.json",
            "reports/sign_memory/metrics.json",
            "reports/causal_pool/metrics.json",
            "reports/civil_quarter/metrics.json",
            "reports/civil_quarter_replay/metrics.json",
            "reports/profiled_quarter/metrics.json",
            "reports/range_alert/metrics.json",
            "reports/issued_calibration/metrics.json",
            "reports/event_cluster/metrics.json",
        ],
        "contrasts": [
            ["cluster", "baseline", "brier"],
            ["cluster", "nuisance", "brier"],
            ["cluster", "recent_frequency", "brier"],
        ],
        "gate": "allthreecontrols must improve in bothphases by>=.0005 Brier, "
        "bothfixedlateslices negative, bothmultiplicity gates; nootherpromotion "
        "contrast",
    },
    "inference": {
        "loss": "brier",
        "effect_absolute": 0.0005,
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "bootstrap_draws": 399999,
        "seed": 20260926,
        "seed_rule": "seed+phase_code*10000+block; development0,evaluation1; same "
        "drawdesign acrosscontrols",
        "wave_alpha": 0.00010822510822510823,
        "cumulative_alpha": 0.05,
        "p_phase": "maximum two-sided centered-null circularblock bootstrap and "
        "BartlettHAC126p",
        "p_hypothesis": "maximumdevelopment/evaluationp",
        "multiplicity": "Holm3 versus wave_alpha and cumulativeHolm137 versus.05",
        "resolution": "1/400000 below one tenth minimumHolm3rawwavecutoff1/27720",
        "support": {
            "minimum_train": 1000,
            "train_per_class": 50,
            "phase_observations": 127,
            "phase_per_class": 30,
            "slice_per_class": 15,
        },
        "failure": "any new source, representation, reconstruction, support, arithmetic, "
        "solver, forecast, inference or publication failure keeps all3 "
        "UNEVALUABLE p1; retain original wave20 failure and all earlier "
        "hypotheses",
        "calibration": "descriptivein-the-large meanprob,eventrate,gap,Brier "
        "perphase/model; no fittedcalibrationbins",
        "power": "nominal HAC80percentMDE/.0005 descriptive; not equivalence or "
        "post-hocselection",
    },
    "verification": {
        "nuisance": {
            "method": "root_hybr",
            "start": "six literalzeros",
            "xtol": 1e-10,
            "maxfev": 2000,
            "factor": 1.0,
            "analytic_jacobian": True,
            "full_gradient_tolerance": 1.0001e-08,
        },
        "scalar": {
            "method": "bisect",
            "bracket": "+/- (mean(abs(centered_candidate))+1)/.02",
            "xtol": 1e-12,
            "rtol": 1e-14,
            "maxiter": 200,
            "full_gradient_tolerance": 1.0001e-08,
        },
        "coefficient_rtol": 1e-07,
        "coefficient_atol": 1e-06,
        "replay_rtol": 1e-10,
        "replay_atol": 1e-12,
        "inference_rtol": 1e-09,
        "inference_atol": 1e-14,
        "conditional_scalar": "independentlysolve using saved independentlyvalidated "
        "nuisancebeta, preservingactualfrozenoffsetobjective",
        "scope": "fullnewhistory, originalcohortsandissuedcontrols, "
        "allnewmonthlystages/applicationpredictionsincludingunscored, "
        "scores,inference,ledger,sourceclosure",
    },
    "upstream": {
        "anchors": {
            "event_cluster.yaml": "824a9beda7cf293568dd2d16bbd3b76a8235f359a78c764147d75731c22b734c",
            "reports/event_cluster/publication_audit.json": "0d901a938c6a88ff4dbf1bb68d114640a50279e38826be29c97ea167ca56be15",
            "reports/event_cluster/failure_review.json": "ae554fd105b5a48b50739af4a6e64c3a8895c3e94bb14fbeb0230490d88d3408",
        },
        "admission": "complete exact failed wave20 preservation envelope and retained output "
        "hashes; full original verified wave19/wave18 closure; checked buffers "
        "before decoding",
    },
    "outputs": {
        "data": "data/event_cluster_replay",
        "reports": "reports/event_cluster_replay",
    },
    "replay": {
        "retained_producer_protocol_sha256": "824a9beda7cf293568dd2d16bbd3b76a8235f359a78c764147d75731c22b734c",
        "input_role": "exact pinned UNVERIFIED wave20 producer artifacts; original wave18/19 "
        "source/control proofs remain verified",
        "newly_generated_forecasts": 0,
        "new_producer_fits": 0,
        "no_producer_entrypoint_or_optimizer_calls": True,
        "sequence": "register, admit checked retained/source buffers, full independent "
        "history/cohort/stage/application reconstruction, then score and infer; "
        "separate guarded verifier repeats reconstruction/inference",
        "mathematics": "unchanged wave20 history, baseline replay, nuisance/scalar objectives, "
        "original coefficients, centers, solver budgets and numeric tolerances; "
        "verification optima never replace retained predictions",
        "artifact_policy": "no rewriting, coercion, replacement or reserialization of retained "
        "source files; ephemeral date comparison views only",
        "old_failure_policy": "keep canonical wave20 UNEVALUABLE/FAILED and all3p1 forever; "
        "its evaluated/unpublished scores are opaque preservation "
        "evidence only",
        "counting": "report retained candidate/control rows, retained monthly schedules and "
        "independently verified stage optima separately from zero new producer "
        "fits/forecasts",
    },
    "date_representation": {
        "scope": "only seven declared full-application-state datetime columns at "
        "final exact-frame comparison; all other frame/index comparisons "
        "unchanged",
        "fields": [
            "origin",
            "feature_cutoff_date",
            "source_fit_origin",
            "window_first_available",
            "window_last_available",
            "window_first_origin",
            "window_last_origin",
        ],
        "accepted_units": ["ms", "us", "ns"],
        "timezone": "naive only",
        "known_values": "normalized midnight calendar dates; unchanged calendar, "
        "source, phase and maturity fences",
        "comparison": "checked exact nanosecond integer representation with "
        "lossless roundtrip to original unit; identical unknown "
        "masks; reject overflow or any rounding",
        "unchanged_structure": "exact columns/order/index/row order/names; exact "
        "numeric boolean integer and nondate "
        "dtypes/values; no blanket dtype suppression",
        "rejected": "object/string/timezone/subday dates, changed instants, "
        "unknown masks, overflow, row/schema/numeric dtype changes",
        "audit": "record original units and successful lossless comparisons "
        "without modifying retained artifacts",
    },
}


def validate(p):
    if json.dumps(p, sort_keys=True, allow_nan=False) != json.dumps(
        CONTRACT, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Frozen event-cluster specification differs")


def paired_difference(candidate, control, actual):
    a, b, y = (cs._finite(v, "Brier input") for v in [candidate, control, actual])
    if any(v.ndim != 1 for v in [a, b, y]) or not a.shape == b.shape == y.shape:
        raise ValueError("Aligned one-dimensional Brier inputs required")
    if not np.isin(y, [0.0, 1.0]).all() or any(((v < 0) | (v > 1)).any() for v in [a, b]):
        raise ValueError("Binary labels and probabilities in[0,1] required")
    return cs.paired_difference(a, b, y)


def paired_inference(candidate, control, difference, p, seed):
    adapted = {**p, "inference": {**p["inference"], "inference_scale": 1.0}}
    result = old_inference.paired_inference(candidate, control, difference, adapted, seed)
    result["nominal_mde_effect_ratio"] = result["hac126"]["mde80_nominal"] / EFFECT
    return result


def inherited(p, pins=None):
    rows = []
    for source_name in p["comparisons"]["inherited_sources"]:
        payload = (ROOT / source_name).read_bytes()
        signature = hashlib.sha256(payload).hexdigest()
        if pins is not None and pins.get(source_name) != signature:
            raise ValueError(
                "Pinned inherited metrics changed before decoding: " + source_name
            )
        for number, row in enumerate(json.loads(payload)["rows"]):
            rows.append(
                {
                    "study": row.get(
                        "study", Path(source_name).parent.name or Path(source_name).stem
                    ),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": source_name,
                    "source_sha256": signature,
                    "source_row_index": number,
                    **{name: row[name] for name in ["measure", "score"] if name in row},
                }
            )
    if len(rows) != 134:
        raise ValueError("All134 inherited comparisons required")
    if any(
        isinstance(row["p_conservative"], bool)
        or not isinstance(row["p_conservative"], (int, float))
        or not np.isfinite(row["p_conservative"])
        or not 0 <= row["p_conservative"] <= 1
        for row in rows
    ):
        raise ValueError("Finite inherited probabilities in [0,1] required")
    return rows


def passes(row):
    phases = row["phases"]
    if len(phases) != 2 or {x["name"] for x in phases} != {"development", "evaluation"}:
        return False
    evaluation = next(x for x in phases if x["name"] == "evaluation")
    return (
        row["p_holm_wave"] < WAVE_ALPHA
        and row["p_holm_cumulative"] < 0.05
        and all(x["n"] > 0 and x["delta"] <= -0.0005 for x in phases)
        and len(evaluation["stability"]) == 2
        and all(x["delta"] < 0 for x in evaluation["stability"])
    )


def candidate_leads(rows):
    keys = [(x["candidate"], x["control"], x["score"]) for x in rows]
    if len(keys) != 3 or set(keys) != set(CONTRASTS):
        raise ValueError("Complete three-comparison family required")
    return ["cluster"] if all(passes(row) for row in rows) else []


def require_support(y, minimum, label):
    values = np.asarray(y)
    if values.ndim != 1 or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Binary support labels required")
    events = int((values == 1).sum())
    nonevents = int((values == 0).sum())
    if min(events, nonevents) < minimum:
        raise ValueError("INSUFFICIENT_DATA: binary class support " + label)
    return {"n": len(values), "events": events, "nonevents": nonevents}


def validate_support(panel, p):
    base = panel.loc[panel.model.eq("recent_frequency")].set_index("origin").sort_index()
    support = {}
    for name in ["development", "evaluation"]:
        first, last = p["original_index"][name]
        y = base.loc[first:last, "y"].to_numpy()
        support[name] = require_support(y, p["inference"]["support"]["phase_per_class"], name)
        if len(y) < p["inference"]["support"]["phase_observations"]:
            raise ValueError("INSUFFICIENT_DATA: literal inference bandwidth " + name)
    support["evaluation_slices"] = []
    for first, last in p["original_index"]["evaluation_stability"]:
        one = require_support(
            base.loc[first:last, "y"].to_numpy(),
            p["inference"]["support"]["slice_per_class"],
            first + ".." + last,
        )
        support["evaluation_slices"].append({"start": first, "end": last, **one})
    return support


def evaluate(panel, calendar, p, monthly_schedules, *, prior=None):
    if type(monthly_schedules) is not int or monthly_schedules <= 0:
        raise ValueError("A positive literal monthly fit count is required")
    sm.validate_panel(panel)
    dates = pd.DatetimeIndex(calendar)
    if dates.has_duplicates or dates.hasnans or not dates.is_monotonic_increasing:
        raise ValueError("Complete ordered reference calendar required")
    expected = pd.Series(dates, index=dates)
    if not panel.origin.isin(dates).all():
        raise ValueError("Every forecast origin must belong to the reference calendar")
    for column, shift in [("feature_cutoff_date", 1), ("target_end", -1)]:
        if not np.array_equal(
            panel[column].to_numpy(), expected.shift(shift).loc[panel.origin].to_numpy()
        ):
            raise ValueError("Exact source-calendar predecessor and target session required")
    development = panel.origin.between(*p["original_index"]["development"])
    evaluation = panel.origin.between(*p["original_index"]["evaluation"])
    if (
        not (development | evaluation).all()
        or not np.array_equal(panel.phase, np.where(development, "development", "evaluation"))
        or not panel.train_n.ge(p["original_index"]["minimum_train"]).all()
        or not panel.loc[development, "available_date"]
        .le(pd.Timestamp(p["original_index"]["development_target_available_by"]))
        .all()
    ):
        raise ValueError(
            "Declared training support and development/evaluation fences required"
        )
    support = validate_support(panel, p)
    calibration = {}
    for phase in ["development", "evaluation"]:
        calibration[phase] = {}
        for model in sm.MODELS:
            one = panel.loc[panel.phase.eq(phase) & panel.model.eq(model)]
            observed, probability = float(one.y.mean()), float(one.probability.mean())
            calibration[phase][model] = {
                "n": len(one),
                "observed_frequency": observed,
                "mean_probability": probability,
                "calibration_gap": probability - observed,
                "brier": float(one.loss.mean()),
            }
    rows = []
    for candidate, control, score in CONTRASTS:
        phases = []
        for code, name in enumerate(["development", "evaluation"]):
            first, last = p["original_index"][name]
            frame = panel.loc[panel.origin.between(first, last)]
            models = {
                m: frame.loc[frame.model.eq(m)].set_index("origin").sort_index()
                for m in sm.MODELS
            }
            a, b = models[candidate], models[control]
            d = paired_difference(
                a.probability.to_numpy(), b.probability.to_numpy(), a.y.to_numpy()
            )
            phase = paired_inference(
                a.loss.to_numpy(),
                b.loss.to_numpy(),
                d,
                p,
                p["inference"]["seed"] + code * 10000,
            )
            phase.update(
                name=name,
                first_origin=str(a.index[0].date()),
                last_origin=str(a.index[-1].date()),
                class_support=support[name],
            )
            phase.update(
                diagnostics(
                    a.index,
                    d,
                    calendar,
                    1,
                    p["original_index"]["evaluation_stability"]
                    if name == "evaluation"
                    else [],
                )
            )
            phases.append(phase)
        rows.append(
            {
                "study": "event_cluster_replay",
                "candidate": candidate,
                "control": control,
                "score": score,
                "horizon": 1,
                "phases": phases,
                "p_conservative": max(x["p_conservative"] for x in phases),
            }
        )
    prior = inherited(p) if prior is None else prior
    if len(prior) != 134:
        raise ValueError("All134 inherited comparisons required")
    wave = inference.holm_adjust([r["p_conservative"] for r in rows])
    family = inference.holm_adjust([r["p_conservative"] for r in prior + rows])[-len(rows) :]
    for row, pw, pc in zip(rows, wave, family, strict=True):
        row.update(p_holm_wave=float(pw), p_holm_cumulative=float(pc))
        row["verdict"] = "PASSES_ALL_GATES" if passes(row) else "DOES_NOT_QUALIFY"
    return {
        "rows": rows,
        "inherited_rows": prior,
        "leads": candidate_leads(rows),
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 137,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
        "newly_generated_forecasts": 0,
        "new_producer_fits": 0,
        "retained_monthly_schedules": monthly_schedules,
        "independent_stage_fits_verified": 2 * monthly_schedules,
        "retained_candidate_scored_forecasts": int(
            panel.model.isin(["nuisance", "cluster"]).sum()
        ),
        "original_control_forecasts": int(
            panel.model.isin(["baseline", "recent_frequency"]).sum()
        ),
        "combined_retained_forecasts": len(panel),
        "common_scored_origins": panel.origin.nunique(),
        "class_support": support,
        "calibration": calibration,
    }


def failure_metrics(error, protocol_hash):
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    return {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 137,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "event_cluster_replay",
                "candidate": a,
                "control": b,
                "score": s,
                "horizon": 1,
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "phases": [],
                "status": status,
                "error": str(error),
            }
            for a, b, s in CONTRASTS
        ],
    }


def report(metrics):
    lines = [
        "# SPX event-clustering verification replay",
        "",
        "Exact retained wave20 producer outputs; zero new producer fits or forecasts. The original failed family remains unchanged.",
        "",
        "Negative Brier difference means improvement; all three controls required.",
        "",
        "| Control | Development | Evaluation | Wave Holm p | Cumulative Holm p | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in metrics["rows"]:
        d, e = row["phases"]
        lines.append(
            f"| {row['control']} | {d['delta']:+.8f} | {e['delta']:+.8f} | {row['p_holm_wave']:.8f} | {row['p_holm_cumulative']:.8f} | {row['verdict']} |"
        )
    lines += [
        "",
        "Passing candidates: " + json.dumps(metrics["leads"]),
        "",
        "Reused archival daily OHLC risk proxy relative to its prior22-session mean; no untouched confirmation or trading-profit claim.",
        "",
    ]
    (REPORT / "results.md").write_text("\n".join(lines))


def _replace_ledger(rows):
    payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=REPORT, prefix=".trial-ledger-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(REPORT / "trial_ledger.jsonl")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ledger(rows):
    path = REPORT / "trial_ledger.jsonl"
    previous = (
        [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    )
    _replace_ledger(previous + rows)


def recover_failure_ledger(registrations, prior, failure):
    path = REPORT / "trial_ledger.jsonl"
    if path.exists():
        # Preserve any partial or successful writes before producing the canonical terminal record.
        (REPORT / "interrupted_trial_ledger.jsonl").write_bytes(path.read_bytes())
    rows = registrations + [{"event": "inherited", **row} for row in (prior or [])]
    rows += [{"event": "unevaluable", **row} for row in failure["rows"]]
    _replace_ledger(rows)


def reconstruct(loaded, retained):
    """Fully verify the retained producer output before any replay inference."""
    return replay.verify_pipeline(
        loaded["features"],
        loaded["targets"],
        loaded["forecasts"],
        loaded["fits"],
        loaded["states"],
        loaded["protocol"]["index"],
        retained["forecasts"],
        retained["fits"],
        retained["states"],
        retained["memory"],
        retained["support"],
    )


def validate_upstream(root=ROOT, registered_pins=None):
    return admission.admit_upstream(
        root, expected=CONTRACT["upstream"]["anchors"], registered_pins=registered_pins
    )


def input_paths(p):
    paths = set(admission.collect_input_pins(ROOT, expected=p["upstream"]["anchors"]))
    paths.update(p["comparisons"]["inherited_sources"])
    paths.add("reports/event_cluster_replay/freeze_record.json")
    freeze = json.loads((REPORT / "freeze_record.json").read_bytes())
    paths.update(freeze["prefit_design"])
    return paths


def validate_freeze(p, signature):
    freeze = json.loads((REPORT / "freeze_record.json").read_bytes())
    if freeze["protocol_sha256"] != signature or inference.digest(PROTOCOL) != signature:
        raise ValueError("Pre-run protocol freeze identity differs")
    actual = {
        str(f.relative_to(ROOT)) for d in ["src", "tests"] for f in (ROOT / d).rglob("*.py")
    }
    if actual != set(freeze["code"]):
        raise ValueError("Pre-run code inventory differs")
    for group in ["code", "prefit_design"]:
        for name, expected in freeze[group].items():
            if inference.digest(ROOT / name) != expected:
                raise ValueError("Pre-run frozen artifact changed: " + name)
    if (
        inference.digest(REPORT / "full_repository_tests.txt")
        != freeze["checks"]["full_log_sha256"]
    ):
        raise ValueError("Full pre-run test log changed")
    return freeze


def require_bound_failed_attempt(pins):
    """Recheck the identified failed attempt immediately before new publication."""
    for filename in ("failure.json", "metrics.json", "verification.json"):
        name = "reports/event_cluster/" + filename
        path = ROOT / name
        if (
            name not in pins
            or not path.is_file()
            or path.is_symlink()
            or inference.digest(path) != pins[name]
        ):
            raise ValueError("Required pinned failed-attempt record changed: " + name)


def run():
    payload = PROTOCOL.read_bytes()
    signature = hashlib.sha256(payload).hexdigest()
    p = yaml.safe_load(payload)
    validate(p)
    validate_freeze(p, signature)
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if any(
        (REPORT / name).exists()
        for name in ["manifest.json", "trial_ledger.jsonl", "failure.json"]
    ) or any(OUT.iterdir()):
        raise ValueError("Refusing to overwrite a registered event-cluster experiment")
    modules = [
        "tests.test_event_cluster_replay_admission",
        "tests.test_event_cluster_replay_verification",
        "tests.test_event_cluster_replay_search",
        "tests.test_verify_event_cluster_replay",
        "tests.test_event_cluster_replay_publication",
        "tests.test_plot_event_cluster_replay",
    ]
    checked = subprocess.run(
        [sys.executable, "-m", "unittest", *modules, "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (REPORT / "pre_run_checks.txt").write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise RuntimeError("Prewritten event-cluster tests failed; no registration or fit")
    validate_freeze(p, signature)
    metrics = None
    prior = None
    registrations = [
        {
            "event": "registered",
            "study": "event_cluster_replay",
            "candidate": a,
            "control": b,
            "score": score,
            "horizon": 1,
            "protocol_sha256": signature,
        }
        for a, b, score in CONTRASTS
    ]
    try:
        ledger(registrations)
        code = [f for d in ["src", "tests"] for f in (ROOT / d).rglob("*.py")]
        inputs = input_paths(p)
        preserved = [
            f
            for f in list(ROOT.glob("*.yaml")) + list((ROOT / "reports").rglob("*"))
            if f.is_file()
            and f != PROTOCOL
            and REPORT not in f.parents
            and str(f.relative_to(ROOT)) not in inputs
        ]
        backend = io.StringIO()
        with redirect_stdout(backend):
            np.show_config()
        manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": signature,
            "environment": {
                "python": sys.version,
                "packages": {
                    name: version(name)
                    for name in ["numpy", "pandas", "scipy", "pyarrow", "PyYAML"]
                },
                "numpy_backend": backend.getvalue(),
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in [
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "LOKY_MAX_CPU_COUNT",
                    ]
                },
            },
            "code": {str(f.relative_to(ROOT)): inference.digest(f) for f in sorted(code)},
            "inputs": {name: inference.digest(ROOT / name) for name in sorted(inputs)},
            "preserved": {
                str(f.relative_to(ROOT)): inference.digest(f) for f in sorted(preserved)
            },
        }
        inference.dump(REPORT / "manifest.json", manifest)
        prior = inherited(p, manifest["inputs"])
        ledger([{"event": "inherited", **row} for row in prior])
        audit, loaded, retained = validate_upstream(ROOT, manifest["inputs"])
        inference.dump(OUT / "upstream_admission.json", audit)
        reconstruction = reconstruct(loaded, retained)
        inference.dump(OUT / "reconstruction_audit.json", reconstruction)
        panel, states, fits = retained["forecasts"], retained["states"], retained["fits"]
        metrics = evaluate(panel, loaded["features"].index, p, len(fits), prior=prior)
        metrics["common_application_origins"] = len(states)
        inference.dump(REPORT / "metrics.json", metrics)
        report(metrics)
        validate_freeze(p, signature)
        for group in ["code", "inputs", "preserved"]:
            for name, expected in manifest[group].items():
                if inference.digest(ROOT / name) != expected:
                    raise ValueError("Frozen artifact changed: " + name)
        if inference.digest(PROTOCOL) != signature:
            raise ValueError("Frozen protocol changed during run")
        if any(
            (ROOT / f"reports/{name}/failure.json").exists()
            or (ROOT / f"reports/{name}/failure.json").is_symlink()
            for name in ("range_alert", "issued_calibration", "event_cluster_replay")
        ):
            raise ValueError("Canonical upstream failure blocks scored publication")
        require_bound_failed_attempt(manifest["inputs"])
        ledger([{"event": "evaluated", **row} for row in metrics["rows"]])
    except Exception as error:
        failure = failure_metrics(error, signature)
        failure["inherited_rows_available"] = prior is not None
        failure["inherited_rows_reconstructed"] = len(prior) if prior is not None else 0
        inference.dump(REPORT / "metrics.json", failure)
        inference.dump(REPORT / "failure.json", failure)
        (REPORT / "results.md").write_text(
            "# SPX event-cluster risk-alert forecasts\n\nUNEVALUABLE: all three comparisons retained with p=1; no lead.\n"
        )
        recover_failure_ledger(registrations, prior, failure)
        if metrics is not None:
            inference.dump(
                REPORT / "unpublished_scored_metrics.json",
                {
                    "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                    "not_for_inherited_inference_or_promotion": True,
                    "scored_metrics": metrics,
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": "REPLAY_SCORED_AWAITING_INDEPENDENT_PUBLICATION_VERIFICATION",
                "retained_forecasts": len(panel),
                "retained_monthly_schedules": len(fits),
                "new_producer_fits": 0,
                "newly_generated_forecasts": 0,
            }
        )
    )


if __name__ == "__main__":
    run()
