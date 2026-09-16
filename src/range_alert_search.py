"""Prospective within-range location increment for a daily SPX risk alert."""

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
from . import orthogonal_round2 as inference
from . import range_alert_features as sf
from . import range_alert_models as sm
from .international_search import diagnostics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "range_alert.yaml"
REPORT = ROOT / "reports/range_alert"
OUT = ROOT / "data/range_alert"
CONTRASTS = (("location", "baseline", "brier"), ("location", "recent_frequency", "brier"))
WAVE_ALPHA = 0.05 / (18 * 19)
EFFECT = 0.0005

CONTRACT = {
    "study_id": "range_alert_wave18",
    "specified_on": "2026-09-07",
    "status": "specified_before_new_features_counts_or_fits",
    "wave": 18,
    "wave_alpha": 0.00014619883040935673,
    "objective": "Test whether prior closing location within the high-low range adds conditional "
    "information about next-session SPX risk exceeding twice its prior22-session mean",
    "evidence_class": "exploratory_reused_history_archival_SPX_daily_risk_event",
    "sources": {
        "daily": "data/research_paths/spx_daily.parquet",
        "vix": "data/free_sources/raw/cboe/VIX_History.csv",
        "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
        "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
        "skew": "data/raw/SKEW_History.csv",
        "skew_source": "data/raw/skew_daily_source.json",
        "skew_derived": "data/raw/skew_daily.parquet",
    },
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
    "feature": {
        "raw": [
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
        "location": "Z=(log(C/L)-log(H/C))/log(H/L);E=Z*Z;use E[t-1] only",
        "zero_range": "unknown E shared common-sample mask; retain full reference calendar "
        "and independently known risk-event labels",
        "domain": "positive ordered OHLC, finite real arithmetic,abs(Z)<=1,0<=E<=1; reject "
        "violations; no clipping or zero-fill",
        "nuisance": "raw log(C/O),its square,raw log(O/C_previous),its square;all shifted one "
        "observed session",
        "curvature": "(I-training_mean_I)^2,(R-training_mean_R)^2,(skew-training_mean_skew)^2 "
        "on common mature rows",
        "scaling": "all25 baseline slopes use training population mean/std; scale>1e-12; no "
        "dropping duplicates or collinear columns; unit intercept remains",
        "location_scaling": "training-centered E, fixed scale1; exactconstant E "
        "center=firstvalue gives exactzero with foldretention; audit "
        "constant condition",
        "nonredundancy": "algebraic distinctness motivates one increment; no empirical "
        "rank-based feature selection; coefficient ridge handles linear "
        "dependencies and retained zero correction cannot improve scores",
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
    "model": {
        "alpha": 0.01,
        "objective": "mean(logaddexp(0,(1-2*y)*eta))+.01*sum(slope^2);unpenalizedbaselineintercept",
        "baseline": "26-column normalized ridge logistic, trainingfrequencylogit intercept "
        "start,zero slopes",
        "candidate": "holdbaselineeta fixed; one penalized scalar b*centeredE, start0; "
        "exactconstantE canonicalzero",
        "producer": {
            "method": "newton_armijo",
            "maximum_iterations": 200,
            "maximum_backtracks": 60,
            "armijo": 0.0001,
            "gradient_tolerance": 1e-08,
        },
        "probabilities": "finite expit(logit) in[0,1]; endpoints valid; no clipping; "
        "alloriginalgradients mandatory",
    },
    "frequency": {
        "half_life_sessions": 63,
        "decay": "2**(-1/63)",
        "seed": "firstfit eligible commontraining eventmean S=f0,W=1 at firstapplication "
        "prior-sessioncutoff k0; no labelavailable<=k0 is fed again",
        "update": "on every later fullreference session decay S,W bydelta, then "
        "add(1-delta)*y and(1-delta) ifoneknownlabelmatures; missing addsnothing; "
        "no resets acrossphase orfeaturegaps",
        "labels": "fulltargetcalendar includingfeatureincomplete or unscoredorigins; "
        "exactunique nextsessionavailability",
        "arithmetic": "fixedfloat64 order with checked nonzero multiplication/division "
        "underflow;0<=S<=W<=1,W>0; q=S/W in[0,1]; no clipping or reset",
        "outputs": "stateaudit everyapplication; baseline/location applicationprobabilities "
        "included in everyfit; scoredorigins get all3modelrows",
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
        "inherited_hypotheses": 127,
        "cumulative_hypotheses": 129,
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
        ],
        "controls": ["baseline", "recent_frequency"],
        "contrasts": [
            ["location", "baseline", "brier"],
            ["location", "recent_frequency", "brier"],
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
        "seed": 20260924,
        "inference_scale": 1.0,
        "seed_rule": "seed+phase_code*10000+block;development0,evaluation1;common "
        "drawdesign forbothcontrols",
        "p_phase": "maximum two-sided centered-null circularblock bootstrap "
        "andBartlettHAC126p",
        "p_hypothesis": "maximumdevelopment/evaluationp",
        "multiplicity": "Holm2 at.05/(18*19);cumulativeHolm129 at.05 includingallfailedprior",
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
        "wave18 cutoff .05/(18*19*2).",
    },
    "verification": {
        "baseline": {
            "method": "scipy_root_hybr",
            "start": "all26zero",
            "xtol": 1e-10,
            "maxfev": 2000,
            "factor": 1.0,
            "success_required": True,
            "full_gradient_tolerance": 1.0001e-08,
            "fallback": "none",
        },
        "scalar": {
            "method": "brentq",
            "bracket": "plus/minus(mean(abs(centeredE))+1)/.02",
            "xtol": 1e-12,
            "rtol": 1e-14,
            "maxiter": 200,
            "full_gradient_tolerance": 1.0001e-08,
            "constant": "exactzero canonicalcoefficient",
        },
        "coefficient_rtol": 1e-07,
        "coefficient_atol": 1e-06,
        "prediction_rtol": 1e-07,
        "prediction_atol": 1e-06,
        "saved_replay_rtol": 1e-10,
        "saved_replay_atol": 1e-12,
        "inference_relative_tolerance": 1e-09,
        "inference_absolute_tolerance": 1e-14,
        "state_roundoff_multiplier": 64,
        "independent_state": "explicitcompensatedsum seed*delta^k plus "
        "known(1-delta)*delta^age weights;domainandzero masks "
        "mandatory;64*(k+1)*max(eps*absexpected,downwardULP) "
        "tolerance",
        "reconstruction": "independentboundedsourcevalues,features/targets,allmonthlytraining/transform/fit/application/state/score/contrast/ledger;immutableinputsandoutputs "
        "rehashedatend",
    },
    "upstream": {
        "protocol": "profiled_quarter.yaml",
        "reports": "reports/profiled_quarter",
        "data": "data/profiled_quarter",
        "protocol_sha256": "6e7821d33c13524ffbfafa666e8c85c1fab7aa48cdd94563989dcd735d73af0c",
        "manifest_sha256": "051c96e0f04bb4f42553ba735a750ffe27d589730d15d734f038debf0a361d8d",
        "verification_sha256": "d76047352010750e117c928a9fbd41c3f361be6bc345db50fcddeaeb878d3b7f",
        "verifier_sha256": "dc7a3a04941594b7e52c2ec3d3da04c16aa9306c3f30e65130b950e2239b4f91",
        "required_status": "VERIFIED",
        "publication_audit_sha256": "84613cd856c92994d9b23e0e74f88fdc457a00ce07b5095601ea5906a51a6bda",
        "prospectus": "reports/profiled_quarter/NEXT_RESEARCH_DIRECTION.md",
        "prospectus_sha256": "c59a79477858e9d58862b26761c98d8c9c8dbead1ed8f9a9b7d74394e5e18560",
    },
    "source_anchor": {
        "protocol": "tail_shape.yaml",
        "reports": "reports/tail_shape",
        "data": "data/tail_shape",
        "protocol_sha256": "0d7184c2d966f28d6ab1602ee6ff6bce4de54019ed0d601688ed7281011366ee",
        "manifest_sha256": "136cfa954874231d305af552d3f89d7e77bee76d3638a9152f764b425509644d",
        "verification_sha256": "d4f63b1848a4d33170101c8717c50c5bf0ad07b254d262ffda83517735fa89ff",
        "verifier_sha256": "18916219840f3a2b01cfaf9b1ada77e7f3bd83219912e70ae4bf5c5e10aadd48",
        "required_status": "VERIFIED",
    },
    "admission": "Hashcheck complete frozen17manifest/code/input/preserved closure, originaltailshape "
    "proof andsource11pins, current17published artifacts; preserve "
    "allpreviousfailedattempts. Priorhistoricalfits/inference are anchored by their "
    "exact successfulrecords, not newlyrefit; new study independently reconstructs its "
    "own inputs andforecasts. Source numbers decoded onlyfrom singlehashchecked byte "
    "snapshots.",
    "outputs": {"data": "data/range_alert", "reports": "reports/range_alert"},
}


def validate(p):
    if json.dumps(p, sort_keys=True, allow_nan=False) != json.dumps(
        CONTRACT, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Frozen range-alert specification differs")


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
    if len(rows) != 127:
        raise ValueError("All127 inherited comparisons required")
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
    return ["location"] if all(passes(row) for row in rows) else []


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
        for model in sf.MODELS:
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
                for m in sf.MODELS
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
                "study": "range_alert",
                "candidate": candidate,
                "control": control,
                "score": score,
                "horizon": 1,
                "phases": phases,
                "p_conservative": max(x["p_conservative"] for x in phases),
            }
        )
    prior = inherited(p) if prior is None else prior
    if len(prior) != 127:
        raise ValueError("All127 inherited comparisons required")
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
        "cumulative_hypothesis_count": 129,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
        "new_monthly_fits": monthly_fits,
        "new_forecasts": len(panel),
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
        "cumulative_hypothesis_count": 129,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "range_alert",
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
        "# SPX range-location risk alert",
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


def _checked_json(root, name, expected):
    payload = (Path(root) / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Pinned record changed before decoding: " + name)
    return json.loads(payload)


def validate_upstream(root=ROOT):
    root = Path(root)
    p = yaml.safe_load((root / "range_alert.yaml").read_bytes())
    validate(p)
    anchors = {}
    for key in ["upstream", "source_anchor"]:
        a = p[key]
        directory = a["reports"]
        if inference.digest(root / a["protocol"]) != a["protocol_sha256"]:
            raise ValueError("Prior protocol identity changed")
        manifest = _checked_json(root, directory + "/manifest.json", a["manifest_sha256"])
        verified = _checked_json(
            root, directory + "/verification.json", a["verification_sha256"]
        )
        if (
            verified["status"] != "VERIFIED"
            or verified["protocol_sha256"] != a["protocol_sha256"]
            or verified["verifier_sha256"] != a["verifier_sha256"]
            or manifest["protocol_sha256"] != a["protocol_sha256"]
        ):
            raise ValueError("Prior successful proof identity required")
        if (
            inference.digest(root / ("src/verify_" + Path(a["protocol"]).stem + ".py"))
            != a["verifier_sha256"]
        ):
            raise ValueError("Prior verifier identity changed")
        for group in ["code", "inputs", "preserved"]:
            for name, expected in manifest[group].items():
                if inference.digest(root / name) != expected:
                    raise ValueError("Prior frozen artifact changed: " + name)
        anchors[key] = {
            "status": "VERIFIED_RECORD_PINNED",
            "protocol_sha256": a["protocol_sha256"],
            "manifest_sha256": a["manifest_sha256"],
            "verification_sha256": a["verification_sha256"],
            "entries_checked": {k: len(manifest[k]) for k in ["code", "inputs", "preserved"]},
        }
    a = p["upstream"]
    directory = a["reports"]
    publication = _checked_json(
        root, directory + "/publication_audit.json", a["publication_audit_sha256"]
    )
    if (
        publication["status"] != "VERIFIED_REPORT_AUDITED"
        or publication["verification_sha256"] != a["verification_sha256"]
    ):
        raise ValueError("Prior reviewed publication required")
    for name, expected in publication["report_artifact_hashes"].items():
        if inference.digest(root / directory / name) != expected:
            raise ValueError("Prior publication changed: " + name)
    if inference.digest(root / a["prospectus"]) != a["prospectus_sha256"]:
        raise ValueError("Prospectus changed")
    for name, expected in p["source_files"].items():
        if inference.digest(root / name) != expected:
            raise ValueError("Original source identity changed: " + name)
    return {
        "status": "PINNED_PRIOR_VERIFIED_PROOFS",
        "anchors": anchors,
        "source_files_checked": len(p["source_files"]),
        "prior_publication_sha256": a["publication_audit_sha256"],
        "prior_files_written": False,
        "scope": "Prior successful evidence and original input closure rehashed; prior fits and inference are not newly recomputed.",
    }


def input_paths(p):
    paths = set(p["source_files"]) | set(p["comparisons"]["inherited_sources"])
    for key in ["upstream", "source_anchor"]:
        a = p[key]
        manifest = _checked_json(ROOT, a["reports"] + "/manifest.json", a["manifest_sha256"])
        paths.update(manifest["inputs"])
        paths.add(a["protocol"])
        for directory in [a["reports"], a["data"]]:
            paths.update(
                str(f.relative_to(ROOT)) for f in (ROOT / directory).rglob("*") if f.is_file()
            )
    paths.add("reports/range_alert/freeze_record.json")
    freeze = json.loads((REPORT / "freeze_record.json").read_bytes())
    paths.update(freeze["prefit_design"])
    return paths


def load_pinned_sources(p, manifest):
    with tempfile.TemporaryDirectory(prefix="range-alert-sources-") as directory:
        staged = Path(directory)
        for name, expected in sorted(p["source_files"].items()):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Relative registered source paths required")
            payload = (ROOT / name).read_bytes()
            if (
                hashlib.sha256(payload).hexdigest() != expected
                or manifest["inputs"].get(name) != expected
            ):
                raise ValueError("Pinned raw source changed before decoding: " + name)
            dest = staged / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
        daily, iv, audit = sf.load_sources(p, staged)
        for key, one in audit["sources"].items():
            one["source_path"] = str(ROOT / p["sources"][key])
        return daily, iv, audit


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
    if (REPORT / "manifest.json").exists() or any(OUT.iterdir()):
        raise ValueError("Refusing to overwrite a registered range-alert experiment")
    modules = [
        "tests.test_range_alert_features",
        "tests.test_range_alert_models",
        "tests.test_range_alert_search",
        "tests.test_verify_range_alert",
        "tests.test_range_alert_integration",
        "tests.test_range_alert_publication",
        "tests.test_plot_range_alert",
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
        raise RuntimeError("Prewritten range-alert tests failed; no registration or fit")
    validate_freeze(p, signature)
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
                for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]
            },
        },
        "code": {str(f.relative_to(ROOT)): inference.digest(f) for f in sorted(code)},
        "inputs": {name: inference.digest(ROOT / name) for name in sorted(inputs)},
        "preserved": {
            str(f.relative_to(ROOT)): inference.digest(f) for f in sorted(preserved)
        },
    }
    inference.dump(REPORT / "manifest.json", manifest)
    metrics = None
    prior = None
    registrations = [
        {
            "event": "registered",
            "study": "range_alert",
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
        prior = inherited(p, manifest["inputs"])
        ledger([{"event": "inherited", **row} for row in prior])
        inference.dump(OUT / "upstream_admission.json", validate_upstream(ROOT))
        daily, iv, audit = load_pinned_sources(p, manifest)
        inference.dump(OUT / "source_audit.json", audit)
        features, targets = sf.build_features(daily, iv)
        features.to_parquet(OUT / "features.parquet")
        targets.to_parquet(OUT / "targets.parquet")
        inference.dump(OUT / "support_audit.json", sm.preflight(features, targets, p["index"]))
        panel, fits, states = sm.forecast_panel(features, targets, p["index"])
        panel.to_parquet(OUT / "forecasts.parquet")
        states.to_parquet(OUT / "states.parquet")
        inference.dump(OUT / "fits.json", fits)
        metrics = evaluate(panel, features.index, p, len(fits), prior=prior)
        metrics["common_application_origins"] = len(states)
        for group in ["code", "inputs", "preserved"]:
            for name, expected in manifest[group].items():
                if inference.digest(ROOT / name) != expected:
                    raise ValueError("Frozen artifact changed: " + name)
        if inference.digest(PROTOCOL) != signature:
            raise ValueError("Frozen protocol changed during run")
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
            "# SPX range-location risk alert\n\nUNEVALUABLE: both comparisons retained with p=1; no lead.\n"
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
                "monthly_fits": len(fits),
            }
        )
    )


if __name__ == "__main__":
    run()
