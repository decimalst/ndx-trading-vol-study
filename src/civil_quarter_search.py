"""Frozen civil quarter-end risk experiment with complete trial accounting."""

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

from . import civil_quarter_features as cf
from . import civil_quarter_models as cm
from . import civil_quarter_score as cs
from . import orthogonal_round2 as inference
from .international_search import diagnostics
from .verify_civil_quarter import validate_upstream

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "civil_quarter.yaml"
REPORT = ROOT / "reports/civil_quarter"
OUT = ROOT / "data/civil_quarter"
CONTRASTS = (
    ("quarter", "baseline", "proper_variance"),
    ("quarter", "mean", "proper_variance"),
)
WAVE_ALPHA = 0.05 / (15 * 16)
EFFECT = 0.005


CONTRACT = {
    "study_id": "civil_quarter_wave15",
    "specified_on": "2026-09-07",
    "status": "specified_before_new_civil_features_counts_or_fits",
    "wave": 15,
    "wave_alpha": 0.00020833333333333335,
    "objective": "Test one civil quarter-end SPX risk increment beyond market, original macro-plan, "
    "month-end, December year-end and additive annual calendar controls",
    "evidence_class": "exploratory_reused_history_archival_SPX_daily_risk_civil_calendar_interaction",
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
        "models": ["mean", "baseline", "quarter"],
        "baseline": [
            "const",
            "lrv_d",
            "lrv_w",
            "lrv_m",
            "neg_d",
            "neg_w",
            "neg_m",
            "lvix",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "nominal_hours",
            "cpi_plan",
            "nfp_plan",
            "fomc_plan",
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "month_end5",
            "year_end5",
        ],
        "all_features": [
            "const",
            "lrv_d",
            "lrv_w",
            "lrv_m",
            "neg_d",
            "neg_w",
            "neg_m",
            "lvix",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "nominal_hours",
            "cpi_plan",
            "nfp_plan",
            "fomc_plan",
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "month_end5",
            "year_end5",
            "quarter_end5",
        ],
    },
    "upstream": {
        "protocol": "causal_pool.yaml",
        "reports": "reports/causal_pool",
        "data": "data/causal_pool",
        "protocol_sha256": "1e329ee208612ed8db1fede873b92a938820d51a3fcf2ec583c813d37ac000ad",
        "manifest_sha256": "a775d4491b3dab27f2c36c037090d37a094dfad85c5ff67256acfa262d40f4c2",
        "verification_sha256": "0f2142d97e62e4ab1dbfc02c6a27bdf23dc02676bdd43bc41fa64e180a2d1010",
        "verifier_sha256": "cb34da06786f61e50b89f494e4071abb8d250539a52145d782cc1e122aeaa900",
        "required_status": "VERIFIED",
    },
    "feature_source": {
        "protocol": "calendar_variance.yaml",
        "reports": "reports/calendar_variance",
        "data": "data/calendar_variance",
        "protocol_sha256": "b3357641d874d22eb6bf0903049e8dcfd93df4e38b0748f717923447ff367ca3",
        "manifest_sha256": "521deb0c053c3eb724f2cd84c4644688863ec263ecce62d00c8ea129580bb4c8",
        "verification_sha256": "47f2a0a333df19d258ca95dfef9fbf403ee0ae6f6a675ec85903755b84943cb7",
        "verifier_sha256": "c213b47e91c25c7293a557064dbf8c62c583fc32bf58701cbf2f5afa61f78677",
        "required_status": "VERIFIED",
        "features": "data/calendar_variance/features.parquet",
        "targets": "data/calendar_variance/targets.parquet",
    },
    "prospectus": {
        "path": "reports/causal_pool/NEXT_CALENDAR_VARIANCE_DESIGN.md",
        "sha256": "0933f511c14df8c5bdb32382663b1472d5e09d7678158ca67f455edfb1e7254f",
    },
    "source_contract": {
        "admission": "Reconstruct complete original wave14 VERIFIED record and "
        "separately complete wave8 VERIFIED record through frozen "
        "read-only functions; historical inventories anchored by "
        "original manifest hashes, never expanded-current-tree old "
        "coverage",
        "immutable_read": "Read and hash each admitted parquet/JSON input once before "
        "decoding; inherited metrics and ledgers use same checked "
        "snapshots. For old wave8 raw reconstruction stage all its "
        "registered input bytes into a temporary same-relative-path "
        "tree, normalize only documentary source_path prefixes back "
        "to originalroot, and preserve old parser/gates",
        "sources": {
            "daily": "data/research_paths/spx_daily.parquet",
            "vix": "data/free_sources/raw/cboe/VIX_History.csv",
            "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
            "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
            "cpi": "data/source_discovery/macro_plans/cpi/ledger.json",
            "nfp": "data/source_discovery/macro_plans/nfp/ledger_verified.json",
            "fomc": "data/source_discovery/macro_plans/calendar/fomc_original_annual_plans.csv",
            "fomc_coverage": "data/source_discovery/macro_plans/calendar/fomc_annual_coverage.json",
        },
        "calendar": {
            "source_publication_start": "2010-01-01",
            "source_publication_end": "2025-10-20",
            "bls_policy": "next plan printed in preceding monthly release; "
            "original planned dates retained even if later "
            "canceled or changed",
            "bls_revision_policy": "current archives with explicit "
            "this-release reissue notices remain "
            "unadmitted for both CPI and payroll; "
            "retain original source ledger and "
            "separate pre-fit admission correction",
            "source_eligibility": "publication civil date strictly before "
            "preceding observed market-session date",
            "nominal_start": "entry civil date1600America/New_York",
            "nominal_end": "next Monday-through-Friday civil "
            "date1600America/New_York; skip weekends only",
            "duration": "nominal endpoint UTC difference in hours; includes "
            "DST elapsed-time change",
            "cpi": "eligible original CPI planned timestamp falls in "
            "open-left closed-right nominal window",
            "nfp": "eligible original payroll planned timestamp falls in "
            "open-left closed-right nominal window",
            "fomc": "eligible original annual meeting final DATE equals "
            "nominal ending date; date-only timing control, no "
            "invented statement time",
            "fomc_annual": "exactly8original planned meetings per year from "
            "selected full annual announcement published "
            "before covered year",
            "coverage": "each nominal-window calendar month requires its "
            "eligible explicit original BLS plan; full eligible "
            "annual FOMC plan required",
            "missing": "pending retrieval blocks execution; intrinsically "
            "absent or ambiguous source statements remain unknown "
            "and enter all-arm complete-sample mask; no date/year "
            "inference or zero fill",
            "limitations": "nominal window ignores holidays and early "
            "closes; neither actual nor historically planned "
            "exchange holding interval; no actual next market "
            "date enters a predictor",
            "provenance": "exact currently captured official-document tool "
            "text and extraction hashes; raw provider bytes "
            "and immutable historical web vintages unverified",
            "replication": "Legacy repository calendars already forecast "
            "next-session variance; this tests original-plan "
            "admission with stronger matched SPX controls, "
            "not a first calendar mechanism",
        },
        "measurement": "Exact frozen wave8 daily SPX target/market transformations; "
        "max(GK,1e-10) is part of that preexisting risk proxy, not the "
        "later strict paired-asset raw-GK gate",
        "missing": "Keep full bounded SPX reference calendar, strict old rolling "
        "history and exact original-plan known-month masks; no zero "
        "filling, removed macro controls, holiday repair or "
        "next-common-date labels",
        "limitations": "Reused Yahoo/Cboe archival values, unverified historical "
        "publication latency and revisions, original-plan documentary "
        "coverage, nominal civil windows and early back-calculated "
        "VIX9D persist; no source acquisition or institutional "
        "mechanism claim",
    },
    "target": {
        "formula": "max(Garman-Klass[next],1e-10)+log(raw_open[next]/raw_close[entry])^2",
        "availability": "Next actual observed SPX close, target_end=available_date; strictly "
        "positive finite known risk labels; missing stays unknown",
        "interpretation": "Native squared-log-return daily OHLC full-session risk proxy, not "
        "measured high-frequency integrated variance",
        "cohort": "All32complete predictors and exact common mature labels across everymodel; "
        "monthly firstfeaturecomplete application before futurequerylabelmask; no "
        "event-only evaluation or dropped unscored month",
    },
    "civil": {
        "month_columns": [
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
        ],
        "nuisance_columns": [
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "month_end5",
            "year_end5",
        ],
        "candidate": "quarter_end5",
        "nominal_date": "First Monday-through-Friday civil date strictly after origin; skip "
        "weekends only; no future observed prices/holidays/earlycloses",
        "month_end5": "Nominal date lies within final5 civil dates of its Gregorian month, "
        "inclusive",
        "year_end5": "month_end5 times nominal December indicator",
        "quarter_end5": "month_end5 times nominal month in March,June,September; December "
        "belongs to year-end control",
        "seasonality": "Eleven nominal-month dummies omitting January",
        "training": "Original18 retain old geometry; all13 new nuisance civil columns and "
        "candidate train-centered on exact admitted rows with fixedscale1; query "
        "means unchanged",
        "constant": "Exact all-equal new civil values center exactly zero in mathematical "
        "primitives; whole-wave scientific support and rank gates still reject "
        "unsupported empirical folds",
    },
    "support": {
        "minimum_train": 1000,
        "minimum_phase_observations": 127,
        "per_class": {
            "train": {"quarter_end5": 20, "month_end5": 20, "year_end5": 5, "month_dummy": 20},
            "phase": {"quarter_end5": 30, "month_end5": 20, "year_end5": 5, "month_dummy": 20},
            "slice": {"quarter_end5": 15, "month_end5": 20, "year_end5": 5, "month_dummy": 10},
        },
        "civil_rank": "Everytraining [const,11month,E5,Y5,G5] block must have rank15 by "
        "singularvalues>1e-10*largest; no dropping or recoding",
        "civil_rank_relative_tolerance": 1e-10,
        "novelty": "Regress centeredG5 on exact transformedBASE31 using lstsq rcond1e-12; "
        "normresidual/normcenteredG5 must exceed1e-8; fullmarket rank not required "
        "because slopes regularized",
        "novelty_lstsq_rcond": 1e-12,
        "minimum_relative_residual_norm": 1e-08,
        "timing": "Audit everymonthly training group/rank/novelty and allphase/slice groups "
        "on exact commoncohorts before any optimization; any failure aborts both "
        "hypotheses, no support-dependent deletion",
    },
    "fitting": {
        "penalty": 0.01,
        "old_scale_minimum": 1e-12,
        "new_civil_scale": 1.0,
        "baseline": "31-column normalized mean eta+exp(log(q/meanq)-eta) "
        "plus.01squaredslopes; unpenalizedintercept, old17populationmean/std, "
        "new13fixedscale1center",
        "normalization": "Trainmean=arithmetic mean of positivey; qscaled=y/trainmean; "
        "nativeforecast exp(log(trainmean)+design@scaled_beta)",
        "quarter": "Freeze baselineeta; z=G5-trainmeanG5; fitonly b via "
        "mean(eta0+bz+exp(logqscaled-eta0-bz))+.01b²; no extra intercept or "
        "jointrefit",
        "mean": "Exact arithmetic training mean on sameall32featurecomplete mature rows",
        "optimizer": "Deterministic Newton from allzero baselinecoefficients and quarterb0; "
        "atmost200states and60Armijo halvings perstep, armijo1e-4; accepted "
        "fullgradientmax<=1e-8",
        "maximum_iterations": 200,
        "maximum_backtracks": 60,
        "armijo": 0.0001,
        "gradient_tolerance": 1e-08,
        "scalar_bracket": "R=max(1,abs(initial_scalar_gradient)/.02); audit finite gradients "
        "at -R,+R enclosing zero; unique optimum follows curvature>=.02",
        "arithmetic": "Strictrealfinite designs/parameters/states/predictions; "
        "positivey/mean/scaledtargets/ratio/forecast; reject unsupported "
        "nonzeroproduct,quotient,exp underflow. Tentative invalid objectives "
        "may be rejected within the fixed Armijo schedule; accepted iterates "
        "and published forecasts cannot be repaired/clipped/restarted",
    },
    "scoring": {
        "loss": "proper_variance",
        "formula": "log(h)+y/h",
        "effect_threshold_absolute": 0.005,
        "paired_difference": "d=(hA-hB)/max(hA,hB); gap=logratio-d*(y/min(hA,hB)). For "
        "abs(d)<=.5 logratio=-log1p(-d) ifd>=0 else log1p(d); otherwise "
        "log(hA)-log(hB)",
        "arithmetic": "First validate every positive real input and finite individual score, "
        "reject unsupported quotient/product underflow. Coherence allowance64 "
        "times sum of unit(x) for loghA,y/hA,loghB,y/hB,stablegap,directgap; "
        "unit=max(eps*abs(x),abs(x)-nextafter(abs(x),0)), no unit floor",
        "effect_reference": "Absolute mean natural-log proper-score decrease.005 bothphases, "
        "not percentage of potentiallynegative rawscore or profit",
    },
    "comparisons": {
        "new_hypotheses": 2,
        "inherited_hypotheses": 121,
        "cumulative_hypotheses": 123,
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
        ],
        "controls": ["baseline", "mean"],
        "contrasts": [
            ["quarter", "baseline", "proper_variance"],
            ["quarter", "mean", "proper_variance"],
        ],
        "candidate_gate": "Both contrasts must pass "
        "everyfixedphase/effect/stability/multiplicity gate",
    },
    "inference": {
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "minimum_phase_observations": 127,
        "bootstrap_draws": 99999,
        "seed": 20260921,
        "inference_scale": 1.0,
        "seed_rule": "seed+phase_code*10000+block; development0,evaluation1",
        "p_phase": "Maximum two-sided centered-null circular-block bootstrap and "
        "BartlettHAC126 p",
        "p_hypothesis": "Maximum development/evaluation p",
        "multiplicity": "Holm2 at.05/(15*16), separately cumulativeHolm123 at.05",
        "gate": "Bothphase proper-score delta<=-.005 againstBOTHcontrols; bothfixed "
        "evaluationslice deltas<0; waveHolm<.05/240,cumulativeHolm<.05",
        "power": "Nominal HAC80percent minimumdetectableeffect dividedby.005; "
        "ordinary5percent diagnostic, noequivalence or adjustedpowerclaim",
        "failure_rule": "Any admission/support/fit/score/verification/publication failure "
        "leaves bothregistered hypotheses UNEVALUABLEp1; retain "
        "alloldresults and diagnosticfailures, no outcome-dependent repair",
    },
    "verification": {
        "baseline_method": "trust-exact",
        "baseline_initialization": "all_zero",
        "baseline_maximum_iterations": 500,
        "independent_gradient_target": 1e-10,
        "accepted_gradient_tolerance": 1.0001e-08,
        "quarter_method": "brentq",
        "scalar_absolute_tolerance": 1e-12,
        "scalar_relative_tolerance": 1e-14,
        "scalar_maximum_iterations": 200,
        "coefficient_relative_tolerance": 1e-07,
        "coefficient_absolute_tolerance": 1e-06,
        "independent_normalized_prediction_relative_tolerance": 1e-07,
        "independent_normalized_prediction_absolute_tolerance": 1e-06,
        "saved_normalized_prediction_relative_tolerance": 1e-10,
        "saved_normalized_prediction_absolute_tolerance": 1e-12,
        "feature_relative_tolerance": 1e-10,
        "feature_absolute_tolerance": 1e-12,
        "inference_relative_tolerance": 1e-09,
        "inference_absolute_tolerance": 1e-14,
        "reconstruction": "Independent "
        "originalupstreamproofs,civilcalendar,commonmaturity/support/rank/novelty,trainingtransforms,convexfits,strictsavedprediction "
        "replay "
        "normalizedbyexacttrainmean,primitiveproperlosses/stablepairedgaps,fourphaseinference "
        "and complete123family ledger; no newproducerimport",
    },
    "outputs": {
        "data": "data/civil_quarter",
        "reports": "reports/civil_quarter",
        "features": "data/civil_quarter/features.parquet",
        "targets": "data/civil_quarter/targets.parquet",
        "forecasts": "data/civil_quarter/forecasts.parquet",
        "fits": "data/civil_quarter/fits.json",
        "support_audit": "data/civil_quarter/support_audit.json",
        "admission": "data/civil_quarter/upstream_admission.json",
    },
}


def validate(p):
    if p != CONTRACT:
        raise ValueError("Fixed civil-quarter protocol differs")


paired_difference = cs.paired_difference


def paired_inference(candidate, control, difference, p, seed):
    if any(np.iscomplexobj(v) for v in [candidate, control, difference]):
        raise ValueError("Real finite proper variance scores and paired differences required")
    candidate, control, d = (np.asarray(x, float) for x in [candidate, control, difference])
    if (
        candidate.ndim != 1
        or candidate.shape != control.shape
        or candidate.shape != d.shape
        or not np.isfinite([candidate, control, d]).all()
    ):
        raise ValueError(
            "Aligned finite proper variance scores and finite paired gaps required"
        )
    if len(d) <= max(p["inference"]["hac_lags"], max(p["inference"]["blocks"])):
        raise ValueError("INSUFFICIENT_DATA: phase shorter than literal inference bandwidth")
    # The caller obtains d from the checked primitive proper-score functional.
    # Its component-level roundoff envelope is enforced before inference; raw
    # scores can be negative and near zero after changing physical risk units.
    scale = float(p["inference"]["inference_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive fixed inference units required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = d / scale
        center = float(normalized.mean())
        centered = normalized - center
        squared_norm = float(centered @ centered)
    if (
        not np.isfinite(normalized).all()
        or not np.isfinite([center, squared_norm]).all()
        or (not np.equal(normalized, normalized[0]).all() and squared_norm == 0)
    ):
        raise ValueError("Invalid normalized difference variance or arithmetic underflow")
    blocks = {}
    for block in p["inference"]["blocks"]:
        samples = inference.bootstrap_means(
            normalized, block, p["inference"]["bootstrap_draws"], seed + block
        )[:, 0]
        if not np.isfinite(samples).all():
            raise ValueError("Nonfinite bootstrap means")
        probability = (1 + int((np.abs(samples - center) >= abs(center)).sum())) / (
            len(samples) + 1
        )
        blocks[str(block)] = {
            "p": probability,
            "ci95": (np.quantile(samples, [0.025, 0.975]) * scale).tolist(),
        }
    hac = inference.hac_summary(normalized, lags=p["inference"]["hac_lags"])
    hac = {
        "se": hac["se"] * scale,
        "p": hac["p"],
        "ci95": (np.asarray(hac["ci95"]) * scale).tolist(),
        "mde80_nominal": hac["mde80_nominal"] * scale,
    }
    intervals = [hac["ci95"]] + [item["ci95"] for item in blocks.values()]
    result = {
        "n": len(d),
        "delta": center * scale,
        "candidate_loss": float(candidate.mean()),
        "control_loss": float(control.mean()),
        "block_inference": blocks,
        "hac126": hac,
        "nominal_mde_effect_ratio": hac["mde80_nominal"] / EFFECT,
        "ci95_envelope": [min(x[0] for x in intervals), max(x[1] for x in intervals)],
        "p_conservative": max(hac["p"], *(x["p"] for x in blocks.values())),
    }
    json.dumps(result, allow_nan=False)
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
    if len(rows) != 121:
        raise ValueError("All121 inherited comparisons required")
    return rows


def passes(row):
    phases = row["phases"]
    if len(phases) != 2 or {x["name"] for x in phases} != {"development", "evaluation"}:
        return False
    evaluation = next(x for x in phases if x["name"] == "evaluation")
    return (
        row["p_holm_wave"] < WAVE_ALPHA
        and row["p_holm_cumulative"] < 0.05
        and all(x["n"] > 0 and x["delta"] <= -EFFECT for x in phases)
        and len(evaluation["stability"]) == 2
        and all(x["delta"] < 0 for x in evaluation["stability"])
    )


def candidate_leads(rows):
    keys = [(x["candidate"], x["control"], x["score"]) for x in rows]
    if len(keys) != 2 or set(keys) != set(CONTRASTS):
        raise ValueError("Complete two-comparison family required")
    return ["quarter"] if all(passes(row) for row in rows) else []


def validate_support(panel, features, p):
    base = panel.loc[panel.model.eq("mean")].set_index("origin").sort_index()
    support = {}
    for name in ["development", "evaluation"]:
        first, last = p["index"][name]
        origins = base.loc[first:last].index
        support[name] = cf.civil_support(features.loc[origins], "phase")
        if len(origins) < p["inference"]["minimum_phase_observations"]:
            raise ValueError("INSUFFICIENT_DATA: literal inference bandwidth " + name)
    support["evaluation_slices"] = []
    for first, last in p["index"]["evaluation_stability"]:
        one = cf.civil_support(features.loc[base.loc[first:last].index], "slice")
        support["evaluation_slices"].append({"start": first, "end": last, **one})
    return support


def evaluate(panel, features, p, monthly_fits, *, prior=None):
    cm.validate_panel(panel)
    dates = pd.DatetimeIndex(features.index)
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
    values = features.loc[panel.origin.unique(), cf.ALL_FEATURES].to_numpy()
    if (
        np.iscomplexobj(values)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Every scored origin requires the complete finite real source-known feature basis"
        )
    support = validate_support(panel, features, p)
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
                a.prediction.to_numpy(), b.prediction.to_numpy(), a.y.to_numpy()
            )
            phase = paired_inference(
                cs.proper_score(a.y.to_numpy(), a.prediction.to_numpy()),
                cs.proper_score(b.y.to_numpy(), b.prediction.to_numpy()),
                d,
                p,
                p["inference"]["seed"] + code * 10000,
            )
            phase.update(
                name=name,
                first_origin=str(a.index[0].date()),
                last_origin=str(a.index[-1].date()),
                civil_support=support[name],
            )
            phase.update(
                diagnostics(
                    a.index,
                    d,
                    dates,
                    1,
                    p["index"]["evaluation_stability"] if name == "evaluation" else [],
                )
            )
            phases.append(phase)
        rows.append(
            {
                "study": "civil_quarter",
                "candidate": candidate,
                "control": control,
                "score": score,
                "horizon": 1,
                "phases": phases,
                "p_conservative": max(x["p_conservative"] for x in phases),
            }
        )
    prior = inherited(p) if prior is None else prior
    if len(prior) != 121:
        raise ValueError("Complete121 inherited family required")
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
        "cumulative_hypothesis_count": 123,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
        "new_monthly_fits": monthly_fits,
        "new_forecasts": len(panel),
        "common_scored_origins": panel.origin.nunique(),
        "civil_support": support,
    }


def failure_metrics(error, protocol_hash):
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    return {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 123,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "civil_quarter",
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
        "# SPX civil quarter-end risk experiment",
        "",
        "Negative proper variance score difference means improvement; both controls required.",
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
        "Reused archival SPX daily risk proxy and nominal civil calendar; no untouched confirmation or trading-profit claim.",
        "",
    ]
    (REPORT / "results.md").write_text("\n".join(lines))


def ledger(rows):
    with (REPORT / "trial_ledger.jsonl").open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def input_paths(p):
    paths = set(p["comparisons"]["inherited_sources"]) | {p["prospectus"]["path"]}
    for anchor in ("upstream", "feature_source"):
        info = p[anchor]
        payload = (ROOT / info["reports"] / "manifest.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != info["manifest_sha256"]:
            raise ValueError("Frozen upstream manifest changed before decoding")
        manifest = json.loads(payload)
        paths.update(manifest["inputs"])
        paths.add(info["protocol"])
        for directory in (info["reports"], info["data"]):
            paths.update(
                str(f.relative_to(ROOT)) for f in (ROOT / directory).rglob("*") if f.is_file()
            )
    return paths


def load_pinned_inputs(p, manifest):
    names = {p["feature_source"][key] for key in ("features", "targets")}
    snapshots = {}
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Relative registered input paths required")
        payload = (ROOT / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["inputs"][name]:
            raise ValueError("Pinned upstream input changed before decoding: " + name)
        snapshots[name] = payload
    return tuple(
        pd.read_parquet(io.BytesIO(snapshots[p["feature_source"][key]]))
        for key in ("features", "targets")
    )


def run():
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    p = yaml.safe_load(protocol_bytes)
    validate(p)
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if (REPORT / "manifest.json").exists():
        raise ValueError("Refusing to overwrite a registered civil-quarter experiment")
    modules = [
        "tests.test_civil_quarter_features",
        "tests.test_civil_quarter_models",
        "tests.test_civil_quarter_score",
        "tests.test_civil_quarter_integration",
        "tests.test_civil_quarter_search",
        "tests.test_civil_quarter_publication",
        "tests.test_verify_civil_quarter",
        "tests.test_plot_civil_quarter",
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
        raise RuntimeError("Prewritten civil-quarter tests failed; no registration or fit")
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
                    "study": "civil_quarter",
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
        original, targets = load_pinned_inputs(p, manifest)
        features = cf.augment_features(original)
        features.to_parquet(OUT / "features.parquet")
        targets.to_parquet(OUT / "targets.parquet")
        support = cm.preflight(features, targets, p["index"])
        inference.dump(OUT / "support_audit.json", support)
        forecasts, fits = cm.forecast_panel(features, targets, p["index"])
        forecasts.to_parquet(OUT / "forecasts.parquet")
        inference.dump(OUT / "fits.json", fits)
        metrics = evaluate(forecasts, features, p, len(fits), prior=prior)
        metrics["common_application_origins"] = sum(fit["application_n"] for fit in fits)
        if inference.digest(PROTOCOL) != manifest["protocol_sha256"]:
            raise ValueError("Frozen protocol changed during civil-quarter run")
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
            "# Civil quarter-end risk experiment\n\nUNEVALUABLE: all2comparisons retained with p=1; no lead.\n"
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
                "new_monthly_fits": len(fits),
                "new_forecasts": metrics["new_forecasts"],
                "leads": metrics["leads"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
