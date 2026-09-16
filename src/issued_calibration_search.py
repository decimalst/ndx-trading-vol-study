"""Causal calibration of original issued SPX risk-alert probabilities."""

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
from . import issued_calibration_admission as admission
from . import issued_calibration_models as sm
from . import orthogonal_round2 as inference
from .international_search import diagnostics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "issued_calibration.yaml"
REPORT = ROOT / "reports/issued_calibration"
OUT = ROOT / "data/issued_calibration"
CONTRASTS = (("calibrated", "baseline", "brier"), ("calibrated", "recent_frequency", "brier"))
WAVE_ALPHA = 0.05 / (19 * 20)
EFFECT = 0.0005

CONTRACT = {
    "specified_on": "2026-09-07",
    "evidence_class": "exploratory_reused_history_archival_SPX_daily_risk_event",
    "source_contract": {
        "skew_sha256": "becbf3f7510de66a736495df29a84ba1911d362402f05f6c46dedd5e9b971492",
        "skew_anchor": {"date": "2018-08-13", "close": 159.03, "tolerance": 0.01},
        "skew_derived": "bounded raw and existing derived values must agree; raw CSV "
        "is the predictor source; no fetch or vintage substitution",
        "historical_availability": "archival current-vintage Yahoo and Cboe extracts; "
        "revisions and exact release latency unverified",
        "vix9d": "January2011-October2013 values back-calculated; no original-date "
        "historical availability claim",
        "missing": "preserve observed SPX reference calendar and strict windows; no "
        "filling; numerical CSV values parsed only after date cutoff",
        "interpretation": "SKEW is an option-implied30day construction; it is not a "
        "physical one-day crash probability",
    },
    "source_files": {
        "data/free_sources/raw/cboe/VIX9D_History.csv": "0d6f600ee71bf6ffb5069d0c583cbe0b1ec97df6da4e4e439d5f0f22f5616abe",
        "data/free_sources/raw/cboe/VIX9D_History.csv.manifest.json": "aee15817f60f0e3cde73a55fa99b807ca4dd10b6b045d8f9b87683df7050cab0",
        "data/free_sources/raw/cboe/VIX_History.csv": "a34aabce269632f30904cf482986dd50b6d4cf51f2203dc51a0a9f460f3c90b2",
        "data/free_sources/raw/cboe/VIX_History.csv.manifest.json": "51caa998f0dadc1a707182a0222f793626da93da535a31f49d66b79db0ab2917",
        "data/free_sources/raw/cboe/VVIX_History.csv": "f6bc726455fa3859c662875a005e97ba656b9536ae58fa6977ad24c6228e4f6a",
        "data/free_sources/raw/cboe/VVIX_History.csv.manifest.json": "f79ecf39b83642b0df23743a1f3aa0077bbf8fc8628fdc0b5de49f0a8ccd0a52",
        "data/raw/SKEW_History.csv": "becbf3f7510de66a736495df29a84ba1911d362402f05f6c46dedd5e9b971492",
        "data/raw/skew_daily.parquet": "159deae9944941c92a454e80ff32d33eddc94e6e4f2e79a64678bd0429ade5a7",
        "data/raw/skew_daily_source.json": "d45a9e3080116655b8f613b131903de33eb23543037893c8603624e84d4121d7",
        "data/research_paths/source_manifest.json": "457afd656244fc527535984874ee11313fd3156815f0d982d0c9f29ab85d244c",
        "data/research_paths/spx_daily.parquet": "3958fbb1eb36689df1596c26b9f0e02e3f1d4fa3b0032381e5287a2a03697bf0",
    },
    "index": {
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
        "models": ["baseline", "recent_frequency", "calibrated"],
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
    "target": {
        "risk": "V_s=max(0.5*log(H_s/L_s)^2-(2*log(2)-1)*log(C_s/O_s)^2,1e-10)+log(O_s/C_previous)^2",
        "event": "y_t=1{V[t+1]>2*mean(V[t-22:t-1])};exact22 observed sessions ending prior "
        "session",
        "threshold": 2.0,
        "reference_sessions": 22,
        "ties": "equality is0; missing risk or anyreference value unknown, never0",
        "maturity": "target_end=available_date=nextobservedSPXsessionclose; cutoff "
        "priorobservedclose",
        "measurement": "dailyOHLCriskproxy relative to trailingmean, not integratedvariance or "
        "doubling versus immediatepreviousday",
        "schedule": "first feature-complete monthly application before future querylabel mask; "
        "expanding mature commontraining>=1000 and>=50eachclass; everymonth "
        "preflight before anyfit; retain unscoredapplications and fits",
        "reference_arithmetic": "Checked math.fsum of exactly22 finite positive session-risk "
        "values dividedby22, then checked2*mean; no epsilon at "
        "equality. Legacy rolling baseline features unchanged.",
    },
    "scoring": {
        "loss": "brier",
        "effect_threshold_absolute": 0.0005,
        "paired_difference": "(p_candidate-p_control)*((p_candidate-y)+(p_control-y)), "
        "afterfiniteindividualsquaredlossvalidation",
        "arithmetic": "finitebinaryy "
        "andprobabilitiesin[0,1],boundedloss;rejectnonzerosquares/productsunderflowtozero; "
        "coherence64epsilon/downwardULP noabsolutefloor",
        "interpretation": "conditionalprobability ofdefinedevent; absolute Brierscore gain is "
        "notprofit or probabilitypointgain",
    },
    "comparisons": {
        "new_hypotheses": 2,
        "inherited_hypotheses": 129,
        "cumulative_hypotheses": 131,
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
        ],
        "controls": ["baseline", "recent_frequency"],
        "contrasts": [
            ["calibrated", "baseline", "brier"],
            ["calibrated", "recent_frequency", "brier"],
        ],
        "candidate_gate": "bothcontrols pass bothphases/effect/stability "
        "andbothmultiplicitygates; no "
        "selectiveevent/subperiod/calibrationbin promotion",
    },
    "inference": {
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "minimum_phase_observations": 127,
        "minimum_phase_per_class": 30,
        "minimum_slice_per_class": 15,
        "bootstrap_draws": 199999,
        "seed": 20260925,
        "inference_scale": 1.0,
        "seed_rule": "seed+phase_code*10000+block;development0,evaluation1;common "
        "drawdesign forbothcontrols",
        "p_phase": "maximum two-sided centered-null circularblock bootstrap "
        "andBartlettHAC126p",
        "p_hypothesis": "maximumdevelopment/evaluationp",
        "multiplicity": "Holm2 at.05/(19*20);cumulativeHolm131 at.05 includingallfailedprior",
        "gate": "bothphasesmeanBrierdecrease>=.0005 againstbothcontrols;bothlate "
        "slicesnegative;bothHolmgates",
        "power": "ordinary5percentHAC80percentMDE diagnostic "
        "dividedby.0005;notadjustedpower or equivalencegate",
        "failure_rule": "anysource,arithmetic,support,optimizer,forecast,score,verification,publicationfailure "
        "keeps bothnewhypothesesUNEVALUABLEp1; nooutcome-dependentrepair",
        "calibration": "descriptivein-the-large perphase/model "
        "n,eventfrequency,meanprobability,gap,Brier only;no fitting,bins "
        "orpromotion",
        "resolution": "Minimum p=1/200000 below one tenth strictest two-comparison raw "
        "wave19 cutoff .05/(19*20*2).",
    },
    "study_id": "issued_calibration_wave19",
    "status": "specified_before_new_calibration_states_fits_or_scores",
    "wave": 19,
    "wave_alpha": 0.00013157894736842105,
    "objective": "Test causal intercept calibration from originally issued baseline risk-alert "
    "forecasts and their later mature outcomes",
    "model": {
        "candidate": "calibrated",
        "controls": ["baseline", "recent_frequency"],
        "new_baseline_fits": 0,
        "half_life_sessions": 63,
        "decay": "2**(-1/63)",
        "ridge": 0.01,
        "objective": "sum_j w_j*logaddexp(0,(1-2*y_j)*(eta_j+a))+.01*a*a",
        "weight": "(1-delta)*delta**(cutoff_reference_position-label_available_reference_position); "
        "no renormalization",
        "seed": "empty at first original application prior-session cutoff; no "
        "fitted-probability seeding or labels available at or before seed",
        "logit": "replay original saved monthly transformations and coefficients on original "
        "application features; require saved application probability replay; never "
        "invert rounded probabilities",
        "history": "only genuinely issued baseline applications with known binary outcome "
        "available by cutoff; cache all original applications independent of future "
        "label mask",
        "clock": "full frozen reference calendar; feature gaps and unknown labels age weights; "
        "no phase resets or warmup deletion",
        "controls_preserved": "retain original recent-frequency broader known-label stream and "
        "all original baseline/frequency probabilities and training "
        "metadata",
        "derivative": "sum w*expit(eta+a) for y0 or -w*expit(-eta-a) for y1, plus .02*a; stable "
        "signed residual is not rounded saved p minus y",
        "producer": {
            "method": "brentq",
            "bracket": "plus/minus(sum(weights)+1)/.02",
            "xtol": 1e-12,
            "rtol": 1e-14,
            "maxiter": 200,
            "gradient_tolerance": 1e-08,
            "empty": "canonical zero",
            "balanced": "exact computed g(0)==0 may return zero",
            "fallback": "none",
        },
        "arithmetic": "finite float64 logits, weights, signed softplus losses, stable signed "
        "residuals, gradients, objective and probabilities; signed softplus and "
        "signed residual for finite logits must remain nonzero; reject nonzero "
        "multiplication or division underflow; expit forecast endpoints allowed; "
        "never silently delete tiny records; scalar reductions use math.fsum",
        "arrival_categories": "mutually exclusive precedence: no original issuance "
        "NO_ISSUED_FORECAST; issued missing label UNKNOWN_LABEL; "
        "otherwise ADMITTED",
        "probability": "If a==0 retain original saved baseline probability after mandatory "
        "strict expit(original_eta) replay; otherwise expit(replayed_eta+a). "
        "Fixed identity branch preserves exact nesting; no clipping or solver "
        "fallback.",
    },
    "verification": {
        "scalar": {
            "method": "bisect",
            "bracket": "plus/minus(sum(weights)+1)/.02",
            "xtol": 1e-12,
            "rtol": 1e-14,
            "maxiter": 200,
            "full_gradient_tolerance": 1.0001e-08,
            "fallback": "none",
            "empty": "canonical zero",
        },
        "coefficient_rtol": 1e-07,
        "coefficient_atol": 1e-06,
        "saved_replay_rtol": 1e-10,
        "saved_replay_atol": 1e-12,
        "prediction_rtol": 1e-10,
        "prediction_atol": 1e-12,
        "inference_relative_tolerance": 1e-09,
        "inference_absolute_tolerance": 1e-14,
        "weight_roundoff_multiplier": 64,
        "independent_state": "explicit ages on full reference calendar and compensated "
        "sums; strict domains and zero masks; no empirical "
        "arithmetic tolerance repairs",
        "reconstruction": "original monthly geometry and every original issued logit, "
        "mature calibration records, independent scalar optima, "
        "unchanged cohort/controls, scores/inference/ledger; no prior "
        "baseline refits",
    },
    "upstream": {
        "anchors": {
            "range_alert.yaml": "ff7b2a7fee497da11f1f88613a17314d3b92d683f506bc4cc51d5089d2bc16a3",
            "reports/range_alert/manifest.json": "3390dfe5a8cae1d18951625fafb7cdbd5ba6693a32a3158d6df182c2856af895",
            "reports/range_alert/freeze_record.json": "a0a00285aab3e3cb233ee34f62c944f693444ea0a830ca0812a65a25b295cb7f",
            "reports/range_alert/verification.json": "de5391c863c1008248b8ddfbc0b2c945126a5a8b36154cc6bfcdc980848bf873",
            "reports/range_alert/publication_audit.json": "d7d46dc66648c959d494c57a79feba9d8a7c4c1010ece89d4adde1fca449562e",
            "reports/range_alert/publication_review.json": "38c661b97184eff4aff9792ab4891d2d9f13a814ef287b58fe363067beed16bf",
            "reports/range_alert/NEXT_RESEARCH_DIRECTION.md": "8dc17f9701ebcdd05dfdaca219bb371d4c85fb432353c16d6fc777975e3d4106",
        },
        "admission": "Complete immutable wave18 proof, output, prior manifest and "
        "publication closure; checked byte snapshots before decoding; inherited "
        "historical fits admitted by successful verified records; no new source "
        "acquisition or protected data",
        "prospectus": "reports/range_alert/NEXT_RESEARCH_DIRECTION.md",
    },
    "outputs": {"data": "data/issued_calibration", "reports": "reports/issued_calibration"},
}


def validate(p):
    if json.dumps(p, sort_keys=True, allow_nan=False) != json.dumps(
        CONTRACT, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Frozen issued-calibration specification differs")


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
    if len(rows) != 129:
        raise ValueError("All129 inherited comparisons required")
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
    if len(keys) != 2 or set(keys) != set(CONTRASTS):
        raise ValueError("Complete two-comparison family required")
    return ["calibrated"] if all(passes(row) for row in rows) else []


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


def evaluate(panel, calendar, p, monthly_fits, *, prior=None):
    if isinstance(monthly_fits, bool) or monthly_fits != 0:
        raise ValueError("This registered calibration reuses all monthly baseline fits")
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
            first, last = p["index"][name]
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
                    p["index"]["evaluation_stability"] if name == "evaluation" else [],
                )
            )
            phases.append(phase)
        rows.append(
            {
                "study": "issued_calibration",
                "candidate": candidate,
                "control": control,
                "score": score,
                "horizon": 1,
                "phases": phases,
                "p_conservative": max(x["p_conservative"] for x in phases),
            }
        )
    prior = inherited(p) if prior is None else prior
    if len(prior) != 129:
        raise ValueError("All129 inherited comparisons required")
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
        "cumulative_hypothesis_count": 131,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
        "new_monthly_fits": monthly_fits,
        "new_forecasts": int(panel.model.eq("calibrated").sum()),
        "reused_control_forecasts": int(panel.model.ne("calibrated").sum()),
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
        "cumulative_hypothesis_count": 131,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "issued_calibration",
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
        "# SPX issued-error risk-alert calibration",
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


def validate_upstream(root=ROOT, registered_pins=None):
    return admission.admit_upstream(
        root, expected=CONTRACT["upstream"]["anchors"], registered_pins=registered_pins
    )


def input_paths(p):
    paths = set(admission.collect_input_pins(ROOT, expected=p["upstream"]["anchors"]))
    paths.update(p["comparisons"]["inherited_sources"])
    paths.add("reports/issued_calibration/freeze_record.json")
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
        raise ValueError("Refusing to overwrite a registered issued-calibration experiment")
    modules = [
        "tests.test_issued_calibration_admission",
        "tests.test_issued_calibration_models",
        "tests.test_issued_calibration_search",
        "tests.test_verify_issued_calibration",
        "tests.test_issued_calibration_integration",
        "tests.test_issued_calibration_publication",
        "tests.test_plot_issued_calibration",
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
        raise RuntimeError(
            "Prewritten issued-calibration tests failed; no registration or fit"
        )
    validate_freeze(p, signature)
    metrics = None
    prior = None
    registrations = [
        {
            "event": "registered",
            "study": "issued_calibration",
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
        audit, loaded = validate_upstream(ROOT, manifest["inputs"])
        inference.dump(OUT / "upstream_admission.json", audit)
        panel, states, calibration_audit = sm.forecast_panel(
            loaded["features"],
            loaded["targets"],
            loaded["forecasts"],
            loaded["fits"],
            loaded["states"],
            loaded["protocol"]["index"],
        )
        panel.to_parquet(OUT / "forecasts.parquet")
        states.to_parquet(OUT / "states.parquet")
        inference.dump(OUT / "calibration_audit.json", calibration_audit)
        metrics = evaluate(panel, loaded["features"].index, p, 0, prior=prior)
        metrics["common_application_origins"] = len(states)
        for group in ["code", "inputs", "preserved"]:
            for name, expected in manifest[group].items():
                if inference.digest(ROOT / name) != expected:
                    raise ValueError("Frozen artifact changed: " + name)
        if inference.digest(PROTOCOL) != signature:
            raise ValueError("Frozen protocol changed during run")
        if (ROOT / "reports/range_alert/failure.json").exists():
            raise ValueError("Canonical upstream failure blocks scored publication")
        inference.dump(REPORT / "metrics.json", metrics)
        report(metrics)
        ledger([{"event": "evaluated", **row} for row in metrics["rows"]])
    except Exception as error:
        failure = failure_metrics(error, signature)
        failure["inherited_rows_available"] = prior is not None
        failure["inherited_rows_reconstructed"] = len(prior) if prior is not None else 0
        inference.dump(REPORT / "metrics.json", failure)
        inference.dump(REPORT / "failure.json", failure)
        (REPORT / "results.md").write_text(
            "# SPX issued-error risk-alert calibration\n\nUNEVALUABLE: both comparisons retained with p=1; no lead.\n"
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
                "status": "SCORED_AWAITING_INDEPENDENT_VERIFICATION",
                "forecasts": len(panel),
                "monthly_fits": 0,
            }
        )
    )


if __name__ == "__main__":
    run()
