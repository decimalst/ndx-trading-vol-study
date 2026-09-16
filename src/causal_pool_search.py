"""Frozen probability pooling, causal event-rate memory and full trial accounting."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import causal_pool_models as cm
from . import cross_moment_score as cs
from . import cross_moment_search as old_inference
from . import orthogonal_round2 as inference
from . import sign_memory_features as sf
from .international_search import diagnostics
from .verify_causal_pool import validate_upstream

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "causal_pool.yaml"
REPORT = ROOT / "reports/causal_pool"
OUT = ROOT / "data/causal_pool"
CONTRASTS = (("pooled", "frozen_baseline", "brier"), ("pooled", "recent_frequency", "brier"))
WAVE_ALPHA = 0.05 / (14 * 15)
EFFECT = 0.0005


CONTRACT = {
    "study_id": "causal_pool_wave14",
    "specified_on": "2026-09-07",
    "status": "specified_before_new_filter_states_forecasts_or_scores",
    "wave": 14,
    "wave_alpha": 0.0002380952380952381,
    "objective": "Test one fixed probability pool against both its frozen conditional baseline "
    "and causal recent-frequency component",
    "evidence_class": "exploratory_reused_history_archival_QQQ_SPX_raw_sign_agreement",
    "index": {
        "asset": "QQQ_ETF_and_SPX_price_index",
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
        "models": ["frozen_baseline", "recent_frequency", "pooled"],
        "minimum_train_per_class": 50,
    },
    "upstream": {
        "protocol": "sign_memory.yaml",
        "reports": "reports/sign_memory",
        "data": "data/sign_memory",
        "features": "data/sign_memory/features.parquet",
        "targets": "data/sign_memory/targets.parquet",
        "forecasts": "data/sign_memory/forecasts.parquet",
        "fits": "data/sign_memory/fits.json",
        "protocol_sha256": "a825704da953cd84302ee37430e446f8558d700859ccc7d2b44fbf5e2268f3d0",
        "manifest_sha256": "47c6bc32556a30ae444a80380a2cd3a0076aa2df511d6814c5cec359cf1dcb0c",
        "verification_sha256": "6a9e1348e93b44e6d0e88aba9b977f85b29489d1913834bbec58e267bafe4039",
        "verifier_sha256": "44c0dc5ab288dfd95cc483b98ea7f198bd587830a24141baa210d259b13b5c27",
        "prospectus": "reports/sign_memory/NEXT_CAUSAL_CALIBRATION_DESIGN.md",
        "prospectus_sha256": "e74e8a8182a8af5e6da6551e4d0d1df3ada7c13e37b2553819a77d23f58a9f7f",
        "required_status": "VERIFIED",
        "admission": "Reconstruct complete original VERIFIED record through frozen "
        "read-only functions; anchor historical inventory by its exact "
        "original manifest hash, never by enumerating the expanded current "
        "source tree; pin all previous code, inputs, preserved and "
        "output/report artifacts",
    },
    "target": {
        "event": "Both next observed SPX-session raw log(close/open) returns strictly "
        "positive, or both strictly negative",
        "zero": "Either or both exactly zero gives class0; complement includes opposite "
        "directions and ties",
        "missing": "Any missing return gives unknown, never class0; reject infinity or "
        "invalid observed source",
        "sign_arithmetic": "Direct comparisons, never multiplication of returns or epsilon "
        "sign threshold",
        "positive_rescaling": "Within-session positive price unit changes preserve each "
        "raw return; no total-return or executable auction "
        "equivalence",
        "fit_schedule": "Monthly first feature-complete origin before future query-label "
        "filtering; expanding training labels mature by prior SPX close; "
        "retain unscored applications and folds",
    },
    "source_contract": {
        "reference_calendar": "Exact full bounded SPX reference calendar from "
        "frozen wave13 features; no compressed dates or new "
        "source acquisition",
        "source_and_measurement": "All frozen wave13 raw-source, "
        "whole-measurement and feature/target gates "
        "remain enforced by complete read-only "
        "independent replay",
        "immutable_read": "Decode each admitted parquet/json input from a single "
        "hash-checked byte snapshot; final rehash all pins; no "
        "previous file writes",
        "limitations": "Archival QQQ ETF versus SPX price index, vendor revisions "
        "and unsynchronized opens, unverified historical "
        "publication latency and back-calculated early VIX9D "
        "remain; no new untouched validation",
    },
    "pooling": {
        "half_life_sessions": 63,
        "decay": "2**(-1/63)",
        "baseline_weight": 0.5,
        "recent_weight": 0.5,
        "baseline": "Exact original issued baseline probability on every original scored "
        "origin; no refit, memory-arm substitution or reconstructed in-sample "
        "predictions",
        "seed": "At first original feature-complete monthly application origin "
        "preceding-session cutoff k0, S=f0,W=1; f0 exact original first fit "
        "frequency on its eligible mature training subset",
        "seed_provenance": "Record first fit train_last_available separately from state "
        "cutoff k0; excluded old labels intentionally absent; never "
        "refeed any target available<=k0",
        "update": "Every subsequent full reference session k: S=delta*S,W=delta*W, then "
        "for its unique finite binary target add (1-delta)*y to S and (1-delta) "
        "to W; advance only through previous-session cutoff of prediction",
        "labels": "Full immutable target table, including labels whose original origin "
        "was unscored or feature-incomplete; require exact "
        "next-reference-session target_end=available_date, duplicate or "
        "misaligned availability invalid",
        "missing": "Decay numerator and mass on every reference session; missing paired "
        "return adds nothing, never class0; zero raw return remains valid "
        "nonevent; no compressing observations",
        "continuity": "Single initialization and no resets across phases, feature gaps, "
        "all-unscored months or fixed evaluation slices; no burn-in or "
        "scored-origin deletion",
        "outputs": "For every original application save state cutoff, latest included "
        "label availability, cumulative post-seed observation update count, "
        "S,W,q and original fit identity. Only original scored origins receive "
        "three probability rows; unscored applications have state audit only",
        "arithmetic": "IEEE float64 in fixed operation order: delta*S and delta*W; "
        "(1-delta)*y; additions; q=S/W; separate 0.5*p and 0.5*q products "
        "then sum. All state finite,0<=S<=W,W>0,probabilities in[0,1]. "
        "Reject unsupported nonzero products/divisions underflowing tozero; "
        "allow nonzero subnormals and exactzero outcomes; no clipping, "
        "reset, omission or alternative memory",
    },
    "scoring": {
        "loss": "brier",
        "effect_threshold_absolute": 0.0005,
        "paired_difference": "(p_candidate-p_control)*((p_candidate-y)+(p_control-y)), "
        "after finite individual squared loss validation",
        "arithmetic": "Binary finite y and finite probabilities in[0,1]; loss in[0,1]; "
        "exactzero errors valid; nonzero squares/products underflowing "
        "tozero reject; coherence64epsilon/downwardULP no arbitrary "
        "absolute floor",
        "functional": "Conditional strict-agreement probability; expected Brier "
        "pi(1-pi)+(p-pi)^2; bounded loss needs no return fourth moments",
        "effect_reference": "Absolute squared-probability error decrease .0005 "
        "bothphases; fixed statistical reference, not profit or "
        "direct probability-point improvement",
    },
    "comparisons": {
        "new_hypotheses": 2,
        "inherited_hypotheses": 119,
        "cumulative_hypotheses": 121,
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
        ],
        "controls": ["frozen_baseline", "recent_frequency"],
        "contrasts": [
            ["pooled", "frozen_baseline", "brier"],
            ["pooled", "recent_frequency", "brier"],
        ],
        "candidate_gate": "Both registered contrasts must pass all fixed "
        "phase/effect/stability/multiplicity gates",
    },
    "inference": {
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "minimum_phase_observations": 127,
        "minimum_phase_per_class": 30,
        "minimum_slice_per_class": 15,
        "bootstrap_draws": 99999,
        "seed": 20260920,
        "inference_scale": 1.0,
        "seed_rule": "seed+phase_code*10000+block; development0,evaluation1",
        "p_phase": "Maximum two-sided centered-null circular-block bootstrap p and "
        "Bartlett HAC126 p",
        "p_hypothesis": "Maximum development/evaluation p",
        "multiplicity": "Holm2 at.05/(14*15), separately cumulativeHolm121 at.05",
        "gate": "Bothphase absolute Brier decrease>=.0005 against BOTHcontrols; "
        "negative differences in bothfixed evaluation slices; "
        "waveHolm<.05/210,cumulativeHolm<.05",
        "support": "Original fits retain at least50events/50nonevents;30/class "
        "perphase;15/class perfixed evaluationslice; any insufficient "
        "support aborts wholefamily; no droppedfolds",
        "power": "Nominal HAC80percent MDE diagnostic only, divided by fixed .0005 "
        "effect; no postscore power gate or equivalence claim",
        "resolution": "Minimum p1/100000 below one tenth strictest two-comparison raw "
        "wave cutoff1/8400",
        "failure_rule": "All2new hypotheses UNEVALUABLEp1 after any "
        "admission/measurement/support/fit/scoring/verification/publication "
        "failure; preserve all diagnostics and old artifacts; no "
        "outcome-dependent repair",
        "calibration": "Descriptive calibration in the large only: perphase/model n, "
        "observed binary frequency, mean issued probability, "
        "probability-minus-frequency gap and Brier. No bins, calibration "
        "fitting, hypothesis, selection or promotion gate; not a full "
        "conditional calibration claim.",
    },
    "verification": {
        "state_relative_roundoff_multiplier": 64,
        "state_roundoff": "At k full reference sessions after k0, "
        "abs(actual-explicit)<=64*(k+1)*max(eps*abs(explicit),abs(explicit)-nextafter(abs(explicit),0)); "
        "exactzero and sign masks, strict state/probability domain "
        "first",
        "independent_state": "Explicit compensated sum of seed*delta**k and observed "
        "(1-delta)*delta**age weights; denominator seed mass "
        "delta**k plus same weights; do not call producer "
        "recurrence",
        "saved_replay": "Exact float64 bits for q from saved S/W, pooled from two "
        "separate half products, preserved baseline and all Brier "
        "losses; no tolerance permitting domain violation",
        "inference_relative_tolerance": 1e-09,
        "inference_absolute_tolerance": 1e-14,
        "reconstruction": "Complete frozen upstream proof; independent application "
        "and scored cohorts, every state/probability/primitive "
        "Brier/difference, four phase analyses, six calibration "
        "rows, complete121family ledger, immutable output "
        "snapshots and final pin rehash",
    },
    "outputs": {
        "data": "data/causal_pool",
        "reports": "reports/causal_pool",
        "forecasts": "data/causal_pool/forecasts.parquet",
        "states": "data/causal_pool/states.parquet",
        "admission": "data/causal_pool/upstream_admission.json",
    },
}


def validate(p):
    if p != CONTRACT:
        raise ValueError("Fixed causal-pooling protocol differs")


def paired_difference(candidate, control, actual):
    a, b, y = (cs._finite(v, "Brier input") for v in [candidate, control, actual])
    if any(v.ndim != 1 for v in [a, b, y]) or not a.shape == b.shape == y.shape:
        raise ValueError("Aligned one-dimensional Brier inputs required")
    if not np.isin(y, [0.0, 1.0]).all() or any(((v < 0) | (v > 1)).any() for v in [a, b]):
        raise ValueError("Binary labels and probabilities in[0,1] required")
    return cs.paired_difference(a, b, y)


def paired_inference(candidate, control, difference, p, seed):
    result = old_inference.paired_inference(candidate, control, difference, p, seed)
    result["nominal_mde_effect_ratio"] = result["hac126"]["mde80_nominal"] / EFFECT
    return result


def inherited(p, pins=None):
    rows = []
    for source in p["comparisons"]["inherited_sources"]:
        payload = (ROOT / source).read_bytes()
        signature = hashlib.sha256(payload).hexdigest()
        if pins is not None and pins.get(source) != signature:
            raise ValueError("Pinned inherited metrics changed before decoding: " + source)
        for number, row in enumerate(json.loads(payload)["rows"]):
            rows.append(
                {
                    "study": row.get("study", Path(source).parent.name or Path(source).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": source,
                    "source_sha256": signature,
                    "source_row_index": number,
                    **{name: row[name] for name in ["measure", "score"] if name in row},
                }
            )
    if len(rows) != 119:
        raise ValueError("All119 inherited comparisons required")
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
    if len(keys) != 2 or set(keys) != set(CONTRASTS):
        raise ValueError("Complete two-comparison family required")
    return ["pooled"] if all(passes(row) for row in rows) else []


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
        first, last = p["index"][name]
        y = base.loc[first:last, "y"].to_numpy()
        support[name] = require_support(y, p["inference"]["minimum_phase_per_class"], name)
        if len(y) < p["inference"]["minimum_phase_observations"]:
            raise ValueError("INSUFFICIENT_DATA: literal inference bandwidth " + name)
    support["evaluation_slices"] = []
    for first, last in p["index"]["evaluation_stability"]:
        one = require_support(
            base.loc[first:last, "y"].to_numpy(),
            p["inference"]["minimum_slice_per_class"],
            first + ".." + last,
        )
        support["evaluation_slices"].append({"start": first, "end": last, **one})
    return support


def evaluate(panel, calendar, p, *, prior=None):
    cm.validate_panel(panel)
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
    development = panel.origin.between(*p["index"]["development"])
    evaluation = panel.origin.between(*p["index"]["evaluation"])
    if (
        not (development | evaluation).all()
        or not np.array_equal(panel.phase, np.where(development, "development", "evaluation"))
        or not panel.train_n.ge(p["index"]["minimum_train"]).all()
        or not panel.loc[development, "available_date"]
        .le(pd.Timestamp(p["index"]["development_target_available_by"]))
        .all()
    ):
        raise ValueError(
            "Declared training support and development/evaluation fences required"
        )
    support = validate_support(panel, p)
    calibration = {}
    for phase in ["development", "evaluation"]:
        calibration[phase] = {}
        for model in cm.MODELS:
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
            first, last = p["index"][name]
            frame = panel.loc[panel.origin.between(first, last)]
            models = {
                m: frame.loc[frame.model.eq(m)].set_index("origin").sort_index()
                for m in cm.MODELS
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
                    p["index"]["evaluation_stability"] if name == "evaluation" else [],
                )
            )
            phases.append(phase)
        rows.append(
            {
                "study": "causal_pool",
                "candidate": candidate,
                "control": control,
                "score": score,
                "horizon": 1,
                "phases": phases,
                "p_conservative": max(x["p_conservative"] for x in phases),
            }
        )
    prior = inherited(p) if prior is None else prior
    if len(prior) != 119:
        raise ValueError("Complete119 inherited family required")
    wave = inference.holm_adjust([r["p_conservative"] for r in rows])
    family = inference.holm_adjust([r["p_conservative"] for r in prior + rows])[-len(rows) :]
    for row, pw, pc in zip(rows, wave, family, strict=True):
        row.update(p_holm_wave=float(pw), p_holm_cumulative=float(pc))
        row["verdict"] = "PASSES_ALL_GATES" if passes(row) else "DOES_NOT_QUALIFY"
    return {
        "rows": rows,
        "inherited_rows": prior,
        "leads": candidate_leads(rows),
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 121,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
        "new_monthly_fits": 0,
        "new_forecasts": 2 * panel.origin.nunique(),
        "preserved_forecasts": panel.origin.nunique(),
        "combined_forecasts": len(panel),
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
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 121,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "causal_pool",
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
        "# QQQ-SPX causal probability pooling",
        "",
        "Negative Brier difference means improvement; both controls required.",
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
        "Reused archival history and raw strict sign-agreement probability; no volatility magnitude, true correlation, standalone direction or trading-profit claim.",
        "",
    ]
    (REPORT / "results.md").write_text("\n".join(lines))


def ledger(rows):
    with (REPORT / "trial_ledger.jsonl").open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def input_paths(p):
    payload = (ROOT / p["upstream"]["reports"] / "manifest.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != p["upstream"]["manifest_sha256"]:
        raise ValueError("Frozen upstream manifest changed before decoding")
    manifest = json.loads(payload)
    paths = (
        set(manifest["inputs"])
        | {p["upstream"]["protocol"]}
        | set(p["comparisons"]["inherited_sources"])
    )
    for directory in [p["upstream"]["reports"], p["upstream"]["data"]]:
        paths.update(
            str(f.relative_to(ROOT)) for f in (ROOT / directory).rglob("*") if f.is_file()
        )
    return paths


def load_pinned_inputs(p, manifest):
    names = {p["upstream"][key] for key in ("features", "targets", "forecasts", "fits")}
    snapshots = {}
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Relative registered input paths required")
        payload = (ROOT / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["inputs"][name]:
            raise ValueError("Pinned upstream input changed before decoding: " + name)
        snapshots[name] = payload
    frames = [
        pd.read_parquet(io.BytesIO(snapshots[p["upstream"][key]]))
        for key in ("features", "targets", "forecasts")
    ]
    fits = json.loads(snapshots[p["upstream"]["fits"]])
    return *frames, fits


def application_origins(features, targets, config):
    if not features.index.equals(targets.index):
        raise ValueError("Identical full reference calendars required")
    dates = features.index
    if dates.has_duplicates or dates.hasnans or not dates.is_monotonic_increasing:
        raise ValueError("Complete ordered reference calendar required")
    previous = pd.Series(dates, index=dates).shift(1).rename("feature_cutoff_date")
    if not features.feature_cutoff_date.equals(previous):
        raise ValueError("Exact previous-session feature cutoff required")
    ready = (
        np.isfinite(features.loc[:, sf.ALL_FEATURES]).all(axis=1)
        & features.feature_cutoff_date.notna()
    )
    phases = pd.Series(False, index=dates)
    for name in ("development", "evaluation"):
        first, last = config[name]
        phases |= (dates >= first) & (dates <= last)
    return dates[
        ready & phases & (dates >= config["origin_start"]) & (dates <= config["origin_end"])
    ]


def run():
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    p = yaml.safe_load(protocol_bytes)
    validate(p)
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if (REPORT / "manifest.json").exists():
        raise ValueError("Refusing to overwrite a registered causal-pooling experiment")
    modules = [
        "tests.test_causal_pool_models",
        "tests.test_causal_pool_integration",
        "tests.test_causal_pool_search",
        "tests.test_causal_pool_publication",
        "tests.test_verify_causal_pool",
        "tests.test_plot_causal_pool",
        "tests.test_round2_inference",
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
        raise RuntimeError("Prewritten causal-pooling tests failed; no registration or fit")
    if inference.digest(PROTOCOL) != protocol_hash:
        raise ValueError("Frozen protocol changed during pretests; no registration")
    code = list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    inputs = input_paths(p)
    preserved = [
        file
        for file in list(ROOT.glob("*.yaml")) + list((ROOT / "reports").rglob("*"))
        if file.is_file()
        and file != PROTOCOL
        and REPORT not in file.parents
        and str(file.relative_to(ROOT)) not in inputs
    ]
    backend = io.StringIO()
    with redirect_stdout(backend):
        np.show_config()
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_hash,
        "environment": {
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in ["numpy", "pandas", "scipy", "pyarrow", "PyYAML"]
            },
            "numpy_backend": backend.getvalue(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]
            },
        },
        "code": {str(file.relative_to(ROOT)): inference.digest(file) for file in sorted(code)},
        "inputs": {name: inference.digest(ROOT / name) for name in sorted(inputs)},
        "preserved": {
            str(file.relative_to(ROOT)): inference.digest(file) for file in sorted(preserved)
        },
    }
    inference.dump(REPORT / "manifest.json", manifest)
    metrics = None
    try:
        prior = inherited(p, manifest["inputs"])
        ledger([{"event": "inherited", **row} for row in prior])
        ledger(
            [
                {
                    "event": "registered",
                    "study": "causal_pool",
                    "candidate": a,
                    "control": b,
                    "score": s,
                    "horizon": 1,
                    "protocol_sha256": manifest["protocol_sha256"],
                }
                for a, b, s in CONTRASTS
            ]
        )
        proof = validate_upstream(ROOT)
        inference.dump(OUT / "upstream_admission.json", proof)
        features, targets, original, fits = load_pinned_inputs(p, manifest)
        applications = application_origins(features, targets, p["index"])
        forecasts, states = cm.pool_panel(
            original,
            targets,
            features.index,
            fits,
            p["index"],
            application_origins=applications,
        )
        forecasts.to_parquet(OUT / "forecasts.parquet")
        states.to_parquet(OUT / "states.parquet")
        metrics = evaluate(forecasts, features.index, p, prior=prior)
        metrics["common_application_origins"] = len(states)
        if inference.digest(PROTOCOL) != manifest["protocol_sha256"]:
            raise ValueError("Frozen protocol changed during causal-pooling run")
        for group in ["code", "inputs", "preserved"]:
            for name, expected in manifest[group].items():
                if inference.digest(ROOT / name) != expected:
                    raise ValueError("Frozen artifact changed: " + name)
        inference.dump(REPORT / "metrics.json", metrics)
        report(metrics)
        ledger([{"event": "evaluated", **row} for row in metrics["rows"]])
    except Exception as error:
        failure = failure_metrics(error, manifest["protocol_sha256"])
        inference.dump(REPORT / "metrics.json", failure)
        inference.dump(REPORT / "failure.json", failure)
        (REPORT / "results.md").write_text(
            "# Causal probability-pooling experiment\n\nUNEVALUABLE: all2comparisons retained with p=1; no lead.\n"
        )
        ledger([{"event": "unevaluable", **row} for row in failure["rows"]])
        if metrics is not None:
            diagnostic = {
                "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                "not_for_inherited_inference_or_promotion": True,
            }
            try:
                json.dumps(metrics, allow_nan=False)
                diagnostic["scored_metrics"] = metrics
            except (TypeError, ValueError):
                diagnostic["nonserializable_metrics_repr"] = repr(metrics)
            inference.dump(REPORT / "unpublished_scored_metrics.json", diagnostic)
        raise
    print(
        json.dumps(
            {
                "status": "SCORED_AWAITING_INDEPENDENT_VERIFICATION",
                "forecasts": len(forecasts),
                "new_monthly_fits": 0,
                "new_forecasts": metrics["new_forecasts"],
                "leads": metrics["leads"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
