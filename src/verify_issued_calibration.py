"""Independent causal issued-logit intercept calibration; no old model refits."""

from __future__ import annotations

import io
import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import bisect
from scipy.special import expit

from . import verify_range_alert as previous
from .verify_cross_moment import compare_tree, digest, holm, same_tree

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("baseline", "recent_frequency", "calibrated")
COMPARISONS = (("calibrated", "baseline"), ("calibrated", "recent_frequency"))
PANEL_COLUMNS = previous.PANEL_COLUMNS
DELTA = 2 ** (-1 / 63)
ALPHA = 0.01
EFFECT = 0.0005
WAVE_ALPHA = 0.05 / (19 * 20)
read_snapshot = previous.read_snapshot
read_json_snapshot = previous.read_json_snapshot
close = previous.close
real = previous.real
support = previous.support
brier_loss = previous.brier_loss
paired_difference = previous.paired_difference
normalized_difference = previous.normalized_difference
explicit_bootstrap_means = previous.explicit_bootstrap_means
independent_hac = previous.independent_hac
_finite = previous._finite


def array(value):
    a = np.asarray(value)
    if a.dtype.kind not in "iuf":
        raise ValueError("Finite real numerical values required")
    a = a.astype(float)
    if not np.isfinite(a).all():
        raise ValueError("Finite numerical values required")
    return a


def multiply(a, b):
    a, b = np.broadcast_arrays(array(a), array(b))
    with np.errstate(all="ignore"):
        value = a * b
    if not np.isfinite(value).all() or ((a != 0) & (b != 0) & (value == 0)).any():
        raise ValueError("Nonfinite product or nonzero multiplication underflow")
    return value


def summation(value):
    try:
        result = math.fsum(map(float, array(value).ravel()))
    except OverflowError as error:
        raise ValueError("Compensated sum overflow") from error
    return _finite(result)


def weighted_inputs(eta, y, weights):
    eta, y, weights = map(array, (eta, y, weights))
    if (
        eta.ndim != 1
        or eta.shape != y.shape
        or eta.shape != weights.shape
        or not np.isin(y, [0.0, 1.0]).all()
    ):
        raise ValueError(
            "Exactly aligned finite issued logits, binary labels and weights required"
        )
    mass = summation(weights)
    if (weights <= 0).any() or not 0 <= mass <= 1:
        raise ValueError("Positive unnormalized weights with mass at most1 required")
    return eta, y, weights, mass


def weighted_objective(intercept, eta, y, weights):
    eta, y, w, _ = weighted_inputs(eta, y, weights)
    a = float(array(intercept))
    values = array(eta + a)
    with np.errstate(all="ignore"):
        losses = np.logaddexp(0.0, (1 - 2 * y) * values)
        score = np.where(y == 0, expit(values), -expit(-values))
    if (
        not np.isfinite(losses).all()
        or (losses <= 0).any()
        or not np.isfinite(score).all()
        or (score == 0).any()
    ):
        raise ValueError("Finite-logit softplus/residual cannot underflow to zero")
    objective = _finite(
        summation(multiply(w, losses)) + float(multiply(ALPHA, multiply(a, a)))
    )
    gradient = _finite(summation(multiply(w, score)) + float(multiply(0.02, a)))
    # The solver needs only objective and gradient. The third return value is
    # the analytic curvature lower bound, not a computed weighted Hessian.
    return objective, gradient, 0.02


def independent_calibration_fit(eta, y, weights):
    eta, y, w, mass = weighted_inputs(eta, y, weights)
    radius = _finite((mass + 1) / 0.02)
    brackets = [-radius, radius]
    endpoints = [weighted_objective(b, eta, y, w)[1] for b in brackets]
    if not endpoints[0] < 0 < endpoints[1]:
        raise ValueError("Strict independent derivative bracket required")
    g0 = weighted_objective(0.0, eta, y, w)[1]
    if not len(y):
        a = 0.0
        status = "EMPTY_HISTORY"
        iterations = calls = 0
    elif g0 == 0:
        a = 0.0
        status = "BALANCED_AT_ZERO"
        iterations = calls = 0
    else:
        a, result = bisect(
            lambda b: weighted_objective(b, eta, y, w)[1],
            *brackets,
            xtol=1e-12,
            rtol=1e-14,
            maxiter=200,
            full_output=True,
        )
        if not result.converged:
            raise ValueError("Independent fixed bisection did not converge")
        status = "FITTED"
        iterations = int(result.iterations)
        calls = int(result.function_calls)
    value, g, _ = weighted_objective(a, eta, y, w)
    if abs(g) > 1e-8 + 1e-12:
        raise ValueError("Independent original scalar full-gradient gate failed")
    return {
        "intercept": float(a),
        "alpha": ALPHA,
        "history_n": len(y),
        "weight_sum": mass,
        "objective": value,
        "gradient": g,
        "gradient_max_abs": abs(g),
        "gradient_at_zero": g0,
        "bracket": brackets,
        "bracket_gradients": endpoints,
        "iterations": iterations,
        "function_calls": calls,
        "converged": True,
        "status": status,
        "method": "bisect",
        "xtol": 1e-12,
        "rtol": 1e-14,
        "maxiter": 200,
        "curvature_lower": 0.02,
        "curvature_upper": 0.02 + 0.25 * mass,
    }


def explicit_weights(reference, arrivals, cutoff):
    reference = pd.DatetimeIndex(reference)
    arrivals = pd.DatetimeIndex(arrivals)
    cutoff = pd.Timestamp(cutoff)
    if (
        reference.hasnans
        or reference.has_duplicates
        or not reference.is_monotonic_increasing
        or cutoff not in reference
        or arrivals.hasnans
        or arrivals.has_duplicates
        or not arrivals.isin(reference).all()
        or (arrivals > cutoff).any()
    ):
        raise ValueError("Unique mature arrivals on exact reference calendar required")
    positions = reference.get_indexer(arrivals)
    age = reference.get_loc(cutoff) - positions
    with np.errstate(all="ignore"):
        powers = np.power(DELTA, age.astype(float))
    if not np.isfinite(powers).all() or (powers <= 0).any():
        raise ValueError("Nonzero age weights cannot underflow")
    weights = multiply(1 - DELTA, powers)
    if summation(weights) > 1:
        raise ValueError("Unnormalized total mass cannot exceed1")
    return weights


def replay_saved_logit(application, audit):
    geometry = audit["transform"]
    base = audit["baseline"]
    if base["columns"] != list(previous.BASE) or geometry["columns"] != list(previous.BASE):
        raise AssertionError("Original baseline column identity differs")
    means = array(base["means"])
    scales = array(base["scales"])
    beta = array(base["beta"])
    if (
        means.shape != (26,)
        or scales.shape != (26,)
        or beta.shape != (26,)
        or means[0] != 0
        or scales[0] != 1
        or (scales <= 1e-12).any()
    ):
        raise ValueError("Exact original26-column finite geometry required")
    raw = application.loc[:, previous.ALL_FEATURES].copy()
    array(raw)
    for name in ("I", "R", "skew"):
        offset = array(raw[name]) - float(array(geometry["curvature_means"][name]))
        raw[name + "_square"] = multiply(offset, offset)
    numerator = array(raw.loc[:, previous.BASE].to_numpy() - means)
    with np.errstate(all="ignore"):
        x = array(numerator / scales)
    if ((numerator != 0) & (x == 0)).any():
        raise ValueError("Nonzero saved-coordinate division underflow")
    multiply(x, beta)
    eta = array(x @ beta)
    probability = expit(eta)
    return eta, array(probability)


def arrival_history(targets, reference, issued, seed_cutoff, last_cutoff):
    reference = previous.clock.target_alignment(targets, reference)
    origins = pd.DatetimeIndex(issued.origin)
    if origins.has_duplicates or not origins.isin(reference).all():
        raise ValueError("Unique original issued application records required")
    logits = array(issued.baseline_logit)
    mapping = dict(zip(origins, logits, strict=True))
    seed_cutoff, last_cutoff = map(pd.Timestamp, (seed_cutoff, last_cutoff))
    if (
        seed_cutoff not in reference
        or last_cutoff not in reference
        or seed_cutoff > last_cutoff
    ):
        raise ValueError("Ordered observed seed and final cutoff required")
    arrivals = []
    records = []
    for pos in range(reference.get_loc(seed_cutoff) + 1, reference.get_loc(last_cutoff) + 1):
        origin, available = reference[pos - 1], reference[pos]
        if origin not in mapping:
            status = "NO_ISSUED_FORECAST"
        elif pd.isna(targets.loc[origin, "y"]):
            status = "UNKNOWN_LABEL"
        else:
            status = "ADMITTED"
            records.append(
                {
                    "origin": str(origin.date()),
                    "available_date": str(available.date()),
                    "baseline_logit": float(mapping[origin]),
                    "y": float(targets.loc[origin, "y"]),
                }
            )
        arrivals.append(
            {
                "origin": str(origin.date()),
                "available_date": str(available.date()),
                "status": status,
            }
        )
    return arrivals, records


def invalidate_publication(root, error):
    report = Path(root) / "reports/issued_calibration"
    report.mkdir(parents=True, exist_ok=True)
    path = report / "metrics.json"
    original_bytes = path.read_bytes() if path.exists() else None
    previous, invalid_bytes = None, None
    if original_bytes is not None:
        try:
            decoded = json.loads(original_bytes)
            if isinstance(decoded, dict):
                # JSON's permissive parser accepts NaN/Infinity. These cannot
                # enter either the canonical failure record or its JSON backup.
                json.dumps(decoded, allow_nan=False)
                previous = decoded
            else:
                invalid_bytes = original_bytes
        except (UnicodeDecodeError, ValueError):
            invalid_bytes = original_bytes
    protocol_hash = previous.get("protocol_sha256") if previous else None
    message = "Independent verification failed: " + type(error).__name__ + ": " + str(error)
    failure = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 131,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "issued_calibration",
                "candidate": candidate,
                "control": control,
                "score": "brier",
                "horizon": 1,
                "phases": [],
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "status": "INVALID_RUN",
                "verdict": "UNEVALUABLE",
                "error": message,
            }
            for candidate, control in COMPARISONS
        ],
    }
    serialized = json.dumps(failure, indent=2, allow_nan=False) + "\n"
    path.write_text(serialized)
    (report / "failure.json").write_text(serialized)
    (report / "results.md").write_text(
        "# SPX issued-probability intercept calibration\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n"
        + message
        + "\n"
    )
    (report / "verification.json").write_text(
        json.dumps(
            {"status": "FAILED", "error": message, "protocol_sha256": protocol_hash}, indent=2
        )
        + "\n"
    )
    with (report / "trial_ledger.jsonl").open("a") as stream:
        for row in failure["rows"]:
            stream.write(
                json.dumps(
                    {"event": "verification_failed", **row}, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
    backup = report / "unpublished_scored_metrics.json"
    if (
        previous is not None
        and previous.get("status") != "UNEVALUABLE"
        and not backup.exists()
    ):
        backup.write_text(
            json.dumps(
                {
                    "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                    "not_for_inherited_inference_or_promotion": True,
                    "scored_metrics": previous,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    invalid_backup = report / "unpublished_invalid_metrics.txt"
    if invalid_bytes is not None and not invalid_backup.exists():
        invalid_backup.write_bytes(invalid_bytes)
    return failure


def verify_with_failure_guard(root=ROOT):
    try:
        return verify(root)
    except Exception as error:
        invalidate_publication(root, error)
        raise


def validate_panel(panel):
    if (
        panel.empty
        or tuple(panel.columns) != PANEL_COLUMNS
        or panel.duplicated(["origin", "model"]).any()
        or set(panel.model) != set(MODELS)
    ):
        raise AssertionError("Three full model cohorts and exact schema required")
    shared = [
        column for column in PANEL_COLUMNS if column not in ("model", "probability", "loss")
    ]
    first = (
        panel.loc[panel.model == MODELS[0], shared]
        .sort_values("origin")
        .reset_index(drop=True)
    )
    for name in MODELS[1:]:
        if not first.equals(
            panel.loc[panel.model == name, shared].sort_values("origin").reset_index(drop=True)
        ):
            raise AssertionError("Exact cross-model metadata and labels required")
    for column in (
        "origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "fit_origin",
        "fit_cutoff_date",
        "train_last_target",
        "train_last_available",
    ):
        dates = pd.DatetimeIndex(panel[column])
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise AssertionError("Normalized finite dates required")
    if not np.array_equal(brier_loss(panel.probability, panel.y), real(panel.loss)):
        raise AssertionError("Exact issued Brier score required")


CONTRACT = {
    "specified_on": "2026-09-07",
    "evidence_class": "exploratory_reused_history_archival_SPX_daily_risk_event",
    "source_contract": {
        "skew_sha256": "becbf3f7510de66a736495df29a84ba1911d362402f05f6c46dedd5e9b971492",
        "skew_anchor": {"date": "2018-08-13", "close": 159.03, "tolerance": 0.01},
        "skew_derived": "bounded raw and existing derived values must agree; raw "
        "CSV is the predictor source; no fetch or vintage "
        "substitution",
        "historical_availability": "archival current-vintage Yahoo and Cboe "
        "extracts; revisions and exact release "
        "latency unverified",
        "vix9d": "January2011-October2013 values back-calculated; no "
        "original-date historical availability claim",
        "missing": "preserve observed SPX reference calendar and strict windows; "
        "no filling; numerical CSV values parsed only after date "
        "cutoff",
        "interpretation": "SKEW is an option-implied30day construction; it is "
        "not a physical one-day crash probability",
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
        "event": "y_t=1{V[t+1]>2*mean(V[t-22:t-1])};exact22 observed sessions ending "
        "prior session",
        "threshold": 2.0,
        "reference_sessions": 22,
        "ties": "equality is0; missing risk or anyreference value unknown, never0",
        "maturity": "target_end=available_date=nextobservedSPXsessionclose; cutoff "
        "priorobservedclose",
        "measurement": "dailyOHLCriskproxy relative to trailingmean, not "
        "integratedvariance or doubling versus immediatepreviousday",
        "schedule": "first feature-complete monthly application before future querylabel "
        "mask; expanding mature commontraining>=1000 and>=50eachclass; "
        "everymonth preflight before anyfit; retain unscoredapplications and "
        "fits",
        "reference_arithmetic": "Checked math.fsum of exactly22 finite positive "
        "session-risk values dividedby22, then checked2*mean; no "
        "epsilon at equality. Legacy rolling baseline features "
        "unchanged.",
    },
    "scoring": {
        "loss": "brier",
        "effect_threshold_absolute": 0.0005,
        "paired_difference": "(p_candidate-p_control)*((p_candidate-y)+(p_control-y)), "
        "afterfiniteindividualsquaredlossvalidation",
        "arithmetic": "finitebinaryy "
        "andprobabilitiesin[0,1],boundedloss;rejectnonzerosquares/productsunderflowtozero; "
        "coherence64epsilon/downwardULP noabsolutefloor",
        "interpretation": "conditionalprobability ofdefinedevent; absolute Brierscore "
        "gain is notprofit or probabilitypointgain",
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
        "keeps bothnewhypothesesUNEVALUABLEp1; "
        "nooutcome-dependentrepair",
        "calibration": "descriptivein-the-large perphase/model "
        "n,eventfrequency,meanprobability,gap,Brier only;no "
        "fitting,bins orpromotion",
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
        "logit": "replay original saved monthly transformations and coefficients on "
        "original application features; require saved application probability "
        "replay; never invert rounded probabilities",
        "history": "only genuinely issued baseline applications with known binary outcome "
        "available by cutoff; cache all original applications independent of "
        "future label mask",
        "clock": "full frozen reference calendar; feature gaps and unknown labels age "
        "weights; no phase resets or warmup deletion",
        "controls_preserved": "retain original recent-frequency broader known-label stream "
        "and all original baseline/frequency probabilities and "
        "training metadata",
        "derivative": "sum w*expit(eta+a) for y0 or -w*expit(-eta-a) for y1, plus .02*a; "
        "stable signed residual is not rounded saved p minus y",
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
        "arithmetic": "finite float64 logits, weights, signed softplus losses, stable "
        "signed residuals, gradients, objective and probabilities; signed "
        "softplus and signed residual for finite logits must remain nonzero; "
        "reject nonzero multiplication or division underflow; expit forecast "
        "endpoints allowed; never silently delete tiny records; scalar "
        "reductions use math.fsum",
        "arrival_categories": "mutually exclusive precedence: no original issuance "
        "NO_ISSUED_FORECAST; issued missing label UNKNOWN_LABEL; "
        "otherwise ADMITTED",
        "probability": "If a==0 retain original saved baseline probability after mandatory "
        "strict expit(original_eta) replay; otherwise "
        "expit(replayed_eta+a). Fixed identity branch preserves exact "
        "nesting; no clipping or solver fallback.",
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
        "independent_state": "explicit ages on full reference calendar and "
        "compensated sums; strict domains and zero masks; no "
        "empirical arithmetic tolerance repairs",
        "reconstruction": "original monthly geometry and every original issued "
        "logit, mature calibration records, independent scalar "
        "optima, unchanged cohort/controls, "
        "scores/inference/ledger; no prior baseline refits",
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
        "publication closure; checked byte snapshots before decoding; "
        "inherited historical fits admitted by successful verified "
        "records; no new source acquisition or protected data",
        "prospectus": "reports/range_alert/NEXT_RESEARCH_DIRECTION.md",
    },
    "outputs": {"data": "data/issued_calibration", "reports": "reports/issued_calibration"},
}


def validate_protocol(p):
    if json.dumps(p, sort_keys=True, allow_nan=False) != json.dumps(
        CONTRACT, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Frozen issued-calibration protocol differs")


STATE_COLUMNS = (
    "origin",
    "feature_cutoff_date",
    "source_fit_origin",
    "seed_cutoff_date",
    "elapsed_sessions",
    "history_n",
    "weight_sum",
    "latest_admitted_origin",
    "latest_admitted_available",
    "missing_label_arrivals",
    "no_forecast_arrivals",
    "intercept",
    "baseline_logit",
    "baseline_probability",
    "calibrated_probability",
    "scored",
    "status",
    "objective",
    "gradient",
    "gradient_max_abs",
)


def replay_issued(features, targets, upstream_panel, upstream_fits, upstream_states, section):
    original_support = previous.preflight(features, targets, section)
    apps, scored = previous.eligible_entries(features, targets, section)
    previous.validate_panel(upstream_panel)
    groups = {
        name: upstream_panel.loc[upstream_panel.model == name].set_index("origin").sort_index()
        for name in previous.MODELS
    }
    if any(not frame.index.equals(scored) for frame in groups.values()):
        raise AssertionError("Complete original scored cohort differs")
    if len(upstream_fits) != original_support["monthly_fits"]:
        raise AssertionError("Every original monthly fitting identity required")
    records = []
    for expected, fit in zip(original_support["fits"], upstream_fits, strict=True):
        audit = fit["model_audit"]
        for key, value in expected.items():
            actual = audit[key] if key in ("support", "transform") else fit[key]
            if key == "transform":
                same_tree(actual, value, "Original training-only geometry")
            elif actual != value:
                raise AssertionError("Original mature sample " + key + " differs")
        month = pd.Timestamp(fit["fit_origin"]).to_period("M")
        app = apps[apps.to_period("M") == month]
        if (
            fit["application_origins"] != [str(x.date()) for x in app]
            or audit["application_n"] != len(app)
            or audit["train_n"] != fit["train_n"]
        ):
            raise AssertionError("Every original application and fit metadata required")
        if (
            audit["baseline"]["means"] != audit["transform"]["means"]
            or audit["baseline"]["scales"] != audit["transform"]["scales"]
        ):
            raise AssertionError("Original saved transform and coefficient scales differ")
        eta, reconstructed = replay_saved_logit(features.loc[app], audit)
        saved = array(fit["application_probabilities"]["baseline"])
        if saved.shape != (len(app),) or ((saved < 0) | (saved > 1)).any():
            raise ValueError("Original application probability domain")
        close(saved, reconstructed, "Strict original application probability replay")
        chosen = app[app.isin(scored)]
        positions = app.get_indexer(chosen)
        for model, frame in groups.items():
            rows = frame.loc[chosen]
            fields = {
                "feature_cutoff_date": features.loc[chosen, "feature_cutoff_date"].to_numpy(),
                "target_end": targets.loc[chosen, "target_end"].to_numpy(),
                "available_date": targets.loc[chosen, "available_date"].to_numpy(),
                "y": targets.loc[chosen, "y"].to_numpy(),
                "horizon": np.ones(len(chosen), int),
                "train_n": np.full(len(chosen), fit["train_n"]),
                "phase": np.where(
                    chosen <= pd.Timestamp(section["development"][1]),
                    "development",
                    "evaluation",
                ),
            }
            for column in (
                "fit_origin",
                "fit_cutoff_date",
                "train_last_target",
                "train_last_available",
            ):
                fields[column] = np.full(
                    len(chosen), pd.Timestamp(fit[column]).to_datetime64()
                )
            for column, value in fields.items():
                if not np.array_equal(rows[column], value):
                    raise AssertionError("Original forecast " + column + " differs")
            if model == "baseline" and not np.array_equal(rows.probability, saved[positions]):
                raise AssertionError("Original scored/application probability bits differ")
        for date, logit, p in zip(app, eta, saved, strict=True):
            records.append(
                {
                    "origin": str(date.date()),
                    "feature_cutoff_date": str(
                        features.loc[date, "feature_cutoff_date"].date()
                    ),
                    "source_fit_origin": fit["fit_origin"],
                    "baseline_logit": float(logit),
                    "baseline_probability": float(p),
                    "scored": date in scored,
                }
            )
    if tuple(upstream_states.columns) != previous.STATE_COLUMNS or not pd.DatetimeIndex(
        upstream_states.origin
    ).equals(apps):
        raise AssertionError("Original full application frequency states required")
    first = upstream_fits[0]
    mask = previous.training_mask(features, targets, apps[0])
    seedp = float(targets.loc[mask, "y"].mean())
    if first["model_audit"]["frequency"]["probability"] != seedp:
        raise AssertionError("Original first mature frequency seed differs")
    full = previous.explicit_states(
        targets,
        features.index,
        first["fit_cutoff_date"],
        seedp,
        first["train_last_available"],
        features.loc[apps[-1], "feature_cutoff_date"],
    )
    for row in upstream_states.itertuples(index=False):
        expected = full.loc[row.feature_cutoff_date]
        if (
            row.origin not in apps
            or row.feature_cutoff_date != features.loc[row.origin, "feature_cutoff_date"]
        ):
            raise AssertionError("Original rate-state chronology differs")
        for name in ("S", "W", "recent_frequency"):
            previous.clock.state_equal(
                getattr(row, name),
                expected[name],
                int(expected.elapsed_sessions),
                "Original explicit " + name,
            )
        if row.origin in scored:
            previous.clock.exact_float(
                groups["recent_frequency"].loc[row.origin, "probability"],
                row.recent_frequency,
                "Unchanged original frequency",
            )
    return records, original_support


def record_comparison(actual, expected, float_fields, label):
    if len(actual) != len(expected):
        raise AssertionError(label + " row count differs")
    for left, right in zip(actual, expected, strict=True):
        if set(left) != set(right):
            raise AssertionError(label + " schema differs")
        for key in right:
            if key in float_fields:
                close(left[key], right[key], label + " " + key)
            elif left[key] != right[key]:
                raise AssertionError(label + " " + key + " differs")


def verify_forecasts(
    features,
    targets,
    upstream_panel,
    upstream_fits,
    upstream_states,
    panel,
    states,
    audit,
    protocol,
):
    section = protocol["index"]
    reference = features.index
    issued, old_support = replay_issued(
        features, targets, upstream_panel, upstream_fits, upstream_states, section
    )
    validate_panel(panel)
    apps = pd.DatetimeIndex([row["origin"] for row in issued])
    scored = apps[[row["scored"] for row in issued]]
    if tuple(states.columns) != STATE_COLUMNS or not pd.DatetimeIndex(states.origin).equals(
        apps
    ):
        raise AssertionError("Every original application calibration state required")
    for model in ("baseline", "recent_frequency"):
        old = (
            upstream_panel.loc[upstream_panel.model == model]
            .sort_values("origin")
            .reset_index(drop=True)
        )
        new = panel.loc[panel.model == model].sort_values("origin").reset_index(drop=True)
        if not old.equals(new):
            raise AssertionError("Original " + model + " control must be preserved exactly")
    groups = {
        name: panel.loc[panel.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    if any(not group.index.equals(scored) for group in groups.values()):
        raise AssertionError("Full unchanged common cohort required")
    shared = [c for c in PANEL_COLUMNS if c not in ("model", "probability", "loss")]
    if not groups["calibrated"][[c for c in shared if c != "origin"]].equals(
        groups["baseline"][[c for c in shared if c != "origin"]]
    ):
        raise AssertionError("Calibrated metadata must clone original baseline")
    record_comparison(
        audit["issued_applications"], issued, {"baseline_logit"}, "Original issued logit audit"
    )
    same_tree(audit["support"], old_support, "Original support preserved")
    expected_top = {
        "seed_origin": issued[0]["origin"],
        "seed_cutoff_date": issued[0]["feature_cutoff_date"],
        "common_application_origins": len(apps),
        "common_scored_origins": len(scored),
        "replayed_monthly_fits": len(upstream_fits),
        "new_monthly_fits": 0,
    }
    for key, value in expected_top.items():
        if audit[key] != value:
            raise AssertionError("Calibration " + key + " differs")
    # The original source geometry has been independently checked. Saved issued
    # finite logits specify the authoritative float64 objective/replay; they are
    # never recovered from rounded probabilities.
    admitted_issued = audit["issued_applications"]
    issued_frame = pd.DataFrame(admitted_issued)
    issued_frame["origin"] = pd.to_datetime(issued_frame.origin)
    seed = pd.Timestamp(issued[0]["feature_cutoff_date"])
    last = features.loc[apps[-1], "feature_cutoff_date"]
    arrivals, records = arrival_history(targets, reference, issued_frame, seed, last)
    record_comparison(
        audit["arrival_audit"], arrivals, set(), "Full session arrival categories"
    )
    record_comparison(
        audit["admitted_records"], records, set(), "Exact admitted original issued records"
    )
    if len(audit["calibrations"]) != len(apps):
        raise AssertionError("All application scalar audit records required")
    maxg = 0.0
    empty = balanced = 0
    rootaudits = []
    for ordinal, row in enumerate(states.itertuples(index=False)):
        date = pd.Timestamp(row.origin)
        cutoff = features.loc[date, "feature_cutoff_date"]
        elapsed = reference.get_loc(cutoff) - reference.get_loc(seed)
        selected = [
            record for record in records if pd.Timestamp(record["available_date"]) <= cutoff
        ]
        available = pd.DatetimeIndex([one["available_date"] for one in selected])
        w = explicit_weights(reference, available, cutoff)
        eta = array([one["baseline_logit"] for one in selected])
        y = array([one["y"] for one in selected])
        mass = summation(w)
        saved = audit["calibrations"][ordinal]
        expected_fields = {
            "feature_cutoff_date": cutoff,
            "source_fit_origin": pd.Timestamp(issued[ordinal]["source_fit_origin"]),
            "seed_cutoff_date": seed,
            "elapsed_sessions": elapsed,
            "history_n": len(selected),
            "scored": date in scored,
            "missing_label_arrivals": sum(
                one["status"] == "UNKNOWN_LABEL"
                and pd.Timestamp(one["available_date"]) <= cutoff
                for one in arrivals
            ),
            "no_forecast_arrivals": sum(
                one["status"] == "NO_ISSUED_FORECAST"
                and pd.Timestamp(one["available_date"]) <= cutoff
                for one in arrivals
            ),
        }
        for key, value in expected_fields.items():
            if getattr(row, key) != value:
                raise AssertionError("Causal state " + key + " differs")
        for key in ("origin", "available_date"):
            actual = getattr(
                row, "latest_admitted_" + ("available" if key == "available_date" else key)
            )
            expected = pd.Timestamp(selected[-1][key]) if selected else pd.NaT
            if not (pd.isna(actual) and pd.isna(expected)) and actual != expected:
                raise AssertionError("Latest admitted record differs")
        if saved["origin"] != str(date.date()) or saved["history_n"] != len(selected):
            raise AssertionError("Scalar application identity differs")
        weight = _finite(saved["weight_sum"])
        if not 0 <= weight <= 1:
            raise AssertionError("Strict saved weight mass domain")
        previous.clock.state_equal(weight, mass, elapsed, "Explicit unnormalized mass")
        if row.weight_sum != weight:
            raise AssertionError("State/scalar mass differs")
        a = _finite(saved["intercept"])
        if row.intercept != a:
            raise AssertionError("State/scalar intercept differs")
        bracket = [-(weight + 1) / 0.02, (weight + 1) / 0.02]
        if saved["bracket"] != bracket:
            raise AssertionError("Fixed mass-dependent bracket differs")
        if (
            saved["method"] != "brentq"
            or saved["alpha"] != 0.01
            or saved["xtol"] != 1e-12
            or saved["rtol"] != 1e-14
            or saved["maxiter"] != 200
            or saved["converged"] is not True
            or saved["curvature_lower"] != 0.02
            or saved["curvature_upper"] != 0.02 + 0.25 * weight
        ):
            raise AssertionError("Fixed producer scalar solver contract differs")
        if (
            not isinstance(saved["iterations"], int)
            or isinstance(saved["iterations"], bool)
            or not 0 <= saved["iterations"] <= 200
        ):
            raise AssertionError("Fixed scalar iteration budget")
        value, g, _ = weighted_objective(a, eta, y, w)
        g0 = weighted_objective(0.0, eta, y, w)[1]
        bg = [weighted_objective(b, eta, y, w)[1] for b in bracket]
        if not bg[0] < 0 < bg[1]:
            raise AssertionError("Original strict bracket signs differ")
        close(saved["bracket_gradients"], bg, "Independent bracket gradients")
        for key, expected in [
            ("objective", value),
            ("gradient", g),
            ("gradient_max_abs", abs(g)),
            ("gradient_at_zero", g0),
        ]:
            close(saved[key], expected, "Independent scalar " + key)
        if not 0 <= _finite(saved["gradient_max_abs"]) <= 1e-8 or abs(g) > 1e-8 + 1e-12:
            raise AssertionError("Original full scalar stationarity gate failed")
        for key in ("objective", "gradient", "gradient_max_abs", "status"):
            if getattr(row, key) != saved[key]:
                raise AssertionError("State/scalar " + key + " mismatch")
        validate_scalar_accounting(saved)
        status = saved["status"]
        if not selected:
            if status != "EMPTY_HISTORY" or a != 0:
                raise AssertionError("Canonical empty-history zero required")
            empty += 1
        elif status == "BALANCED_AT_ZERO":
            if saved["gradient_at_zero"] != 0 or a != 0:
                raise AssertionError("Balanced exact-zero rule differs")
            balanced += 1
        elif status != "FITTED":
            raise AssertionError("Known nonempty scalar status required")
        independent = independent_calibration_fit(eta, y, w)
        close(a, independent["intercept"], "Independent scalar optimum", rtol=1e-7, atol=1e-6)
        maxg = max(maxg, independent["gradient_max_abs"])
        rootaudits.append({"origin": str(date.date()), **independent})
        original = admitted_issued[ordinal]
        if (
            row.baseline_logit != original["baseline_logit"]
            or row.baseline_probability != original["baseline_probability"]
        ):
            raise AssertionError("Original logit/probability state bits differ")
        prob = _finite(row.calibrated_probability)
        if not 0 <= prob <= 1:
            raise AssertionError("Strict calibrated probability domain")
        predicted = (
            original["baseline_probability"]
            if a == 0
            else float(expit(_finite(original["baseline_logit"] + a)))
        )
        if prob != predicted:
            raise AssertionError("Exact issued correction/zero-identity replay differs")
        independent_p = (
            original["baseline_probability"]
            if independent["intercept"] == 0
            else float(expit(original["baseline_logit"] + independent["intercept"]))
        )
        close(prob, independent_p, "Independent corrected probability", rtol=1e-10, atol=1e-12)
        if date in scored and groups["calibrated"].loc[date, "probability"] != prob:
            raise AssertionError("State/issued candidate probability bits differ")
    return {
        "forecasts_verified": len(panel),
        "new_forecasts": len(scored),
        "reused_control_forecasts": 2 * len(scored),
        "new_monthly_fits": 0,
        "original_monthly_fits_replayed": len(upstream_fits),
        "independent_calibration_fits_verified": len(apps),
        "common_scored_origins": len(scored),
        "common_application_origins": len(apps),
        "application_states_verified": len(states),
        "admitted_records_verified": len(records),
        "arrival_sessions_verified": len(arrivals),
        "empty_history_states": empty,
        "balanced_history_states": balanced,
        "maximum_independent_scalar_gradient": maxg,
        "independent_solver_audits": rootaudits,
    }


UPSTREAM_REPORT = "reports/range_alert/"
UPSTREAM_DATA = "data/range_alert/"
OUTPUT_PATHS = tuple(
    sorted(
        [
            UPSTREAM_DATA + name
            for name in (
                "upstream_admission.json",
                "source_audit.json",
                "support_audit.json",
                "features.parquet",
                "targets.parquet",
                "forecasts.parquet",
                "states.parquet",
                "fits.json",
            )
        ]
        + [UPSTREAM_REPORT + "metrics.json", UPSTREAM_REPORT + "trial_ledger.jsonl"]
    )
)
ANCHOR_PATHS = tuple(
    sorted(
        ["range_alert.yaml"]
        + [
            UPSTREAM_REPORT + name
            for name in (
                "manifest.json",
                "freeze_record.json",
                "verification.json",
                "publication_audit.json",
                "publication_review.json",
                "NEXT_RESEARCH_DIRECTION.md",
            )
        ]
    )
)


def strict_json(payload, object_required=True):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
    if object_required and type(value) is not dict:
        raise ValueError("JSON object required")
    return value


def identity(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise AssertionError(label + " identity differs")


def source_path(root, name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
        or any(part in ("", "..", ".") for part in name.split("/"))
    ):
        raise ValueError("Canonical relative source path required")
    path = Path(root).resolve()
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("Source symlinks not admitted")
    return path


def hash_mapping(value):
    import re

    if type(value) is not dict:
        raise ValueError("Raw byte hash map required")
    for name, signature in value.items():
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or "\x00" in name
            or name.startswith("/")
            or any(part in ("", "..", ".") for part in name.split("/"))
        ):
            raise ValueError("Canonical relative hash-map path required")
        if not isinstance(signature, str) or re.fullmatch("[0-9a-f]{64}", signature) is None:
            raise ValueError("Lowercase SHA256 required")
    return value


def collect_closure(root, expected):
    import re

    root = Path(root)
    anchors = hash_mapping(expected)
    if set(anchors) != set(ANCHOR_PATHS):
        raise AssertionError("Exact seven immutable prior anchors required")
    files = {}
    snapshots = {}
    manifests = {}

    def add(mapping):
        for name, signature in hash_mapping(mapping).items():
            if name in files and files[name] != signature:
                raise AssertionError("Conflicting transitive source hash: " + name)
            files[name] = signature

    def read(name, keep=True, fresh=False):
        if not fresh and name in snapshots:
            return snapshots[name]
        payload = source_path(root, name).read_bytes()
        if sha256(payload).hexdigest() != files[name]:
            raise AssertionError("Pinned bytes changed: " + name)
        if keep:
            snapshots[name] = payload
        return payload

    def obj(name):
        return strict_json(read(name))

    if (root / UPSTREAM_REPORT / "failure.json").exists():
        raise AssertionError("Failed upstream cannot be admitted")
    add(anchors)
    for name in sorted(anchors):
        read(name)
    manifest, freeze, verified, publication, review = [
        obj(UPSTREAM_REPORT + name + ".json")
        for name in (
            "manifest",
            "freeze_record",
            "verification",
            "publication_audit",
            "publication_review",
        )
    ]
    ph = anchors["range_alert.yaml"]
    for one in (manifest, freeze, verified, publication, review):
        identity(one["protocol_sha256"], ph, "Prior protocol")
    identity(verified["status"], "VERIFIED", "Prior verification")
    identity(publication["status"], "VERIFIED_REPORT_AUDITED", "Prior publication")
    identity(review["status"], "APPROVED_VERIFIED_INTERPRETATION", "Prior review")
    identity(review["figure_visually_reviewed"], True, "Prior visual review")
    for name, key in [
        ("manifest", "manifest_sha256"),
        ("freeze_record", "freeze_sha256"),
        ("verification", "verification_sha256"),
    ]:
        identity(publication[key], anchors[UPSTREAM_REPORT + name + ".json"], key)
    identity(freeze["code"], manifest["code"], "Entire prior frozen code")
    add(freeze["prefit_design"])
    add({UPSTREAM_REPORT + "full_repository_tests.txt": freeze["checks"]["full_log_sha256"]})
    if (
        type(freeze["checks"]["full_repository_tests"]) is not int
        or freeze["checks"]["full_repository_tests"] < 1
    ):
        raise AssertionError("Positive prewritten test count required")
    verifier = manifest["code"]["src/verify_range_alert.py"]
    identity(verified["verifier_sha256"], verifier, "Original verifier")
    identity(publication["verifier_sha256"], verifier, "Published verifier")
    outputs = hash_mapping(verified["verified_output_hashes"])
    if set(outputs) != set(OUTPUT_PATHS):
        raise AssertionError("Exact ten original verified outputs required")
    identity(publication["verified_output_hashes"], outputs, "Published output hashes")
    add(outputs)
    artifacts = {
        UPSTREAM_REPORT + name: signature
        for name, signature in hash_mapping(publication["report_artifact_hashes"]).items()
    }
    if not {
        UPSTREAM_REPORT + name
        for name in (
            "SUMMARY.md",
            "NEXT_RESEARCH_DIRECTION.md",
            "publication_review.json",
            "verification.json",
            "metrics.json",
            "trial_ledger.jsonl",
        )
    } <= set(artifacts):
        raise AssertionError("Full reviewed publication reference required")
    add(artifacts)
    identity(
        review["summary_sha256"], artifacts[UPSTREAM_REPORT + "SUMMARY.md"], "Reviewed summary"
    )
    add(review["document_hashes"])
    add(review["figure_hashes"])
    groups = ("code", "inputs", "preserved", "existing_artifacts_sha256")
    while True:
        pending = sorted(
            name
            for name in files
            if re.fullmatch(r"reports/[^/]+/manifest\.json", name) and name not in manifests
        )
        if not pending:
            break
        for name in pending:
            item = obj(name)
            counts = {}
            for group in groups:
                mapping = item.get(group, {})
                add(mapping)
                counts[group] = len(mapping)
            manifests[name] = counts
    entries = {key: len(manifest[key]) for key in ("code", "inputs", "preserved")}
    identity(
        publication["manifest_entries_checked"], entries, "Prior published manifest counts"
    )
    identity(
        verified["artifact_hashes_checked"],
        {**entries, "outputs": 10},
        "Prior verified manifest counts",
    )
    forecast = verified["forecast_reconstruction"]
    counts = {
        "forecasts": forecast["forecasts_verified"],
        "monthly_fits": forecast["monthly_fits_verified"],
        "common_scored_origins": forecast["common_scored_origins"],
        "common_application_origins": forecast["common_application_origins"],
        "bounded_reference_rows": verified["source_reconstruction"]["bounded_reference_rows"],
        "cumulative_hypotheses": 129,
    }
    if any(type(value) is not int or value < 1 for value in counts.values()):
        raise AssertionError("Positive literal original counts required")
    if (
        counts["forecasts"] != 3 * counts["common_scored_origins"]
        or not counts["bounded_reference_rows"]
        >= counts["common_application_origins"]
        >= counts["common_scored_origins"]
    ):
        raise AssertionError("Original complete cohort counts differ")
    for key, value in [
        ("current_wave_verified_forecasts", counts["forecasts"]),
        ("current_wave_verified_monthly_fits", counts["monthly_fits"]),
        ("current_wave_verified_applications", counts["common_application_origins"]),
        ("current_wave_scored_origins", counts["common_scored_origins"]),
        ("cumulative_hypotheses", 129),
    ]:
        identity(publication[key], value, key)
    identity(verified["inference"]["new_hypotheses_verified"], 2, "Original new family")
    identity(
        verified["inference"]["cumulative_hypotheses_verified"],
        129,
        "Original cumulative family",
    )
    for name in sorted(files):
        read(name, keep=name in OUTPUT_PATHS)
    audit = {
        "status": "PINNED_WAVE18_VERIFIED_OUTPUTS",
        "anchors": dict(sorted(anchors.items())),
        "files": dict(sorted(files.items())),
        "manifest_paths": sorted(manifests),
        "manifest_groups": dict(sorted(manifests.items())),
        "verified_output_hashes": dict(sorted(outputs.items())),
        "publication_artifact_hashes": dict(sorted(artifacts.items())),
        "counts": {
            "files": len(files),
            "manifests": len(manifests),
            "verified_outputs": len(outputs),
            "publication_artifacts": len(artifacts),
        },
        "upstream_identity": {
            "protocol_sha256": ph,
            "manifest_sha256": anchors[UPSTREAM_REPORT + "manifest.json"],
            "freeze_sha256": anchors[UPSTREAM_REPORT + "freeze_record.json"],
            "verification_sha256": anchors[UPSTREAM_REPORT + "verification.json"],
            "verifier_sha256": verifier,
            "publication_audit_sha256": anchors[UPSTREAM_REPORT + "publication_audit.json"],
            "publication_review_sha256": anchors[UPSTREAM_REPORT + "publication_review.json"],
            "prospectus_sha256": anchors[UPSTREAM_REPORT + "NEXT_RESEARCH_DIRECTION.md"],
        },
        "upstream_verified_counts": counts,
        "historical_models_refitted": False,
        "source_values_reparsed": False,
        "loaded_from_checked_snapshots": True,
    }
    for name in sorted(files):
        read(name, keep=False, fresh=True)
    if (root / UPSTREAM_REPORT / "failure.json").exists():
        raise AssertionError("Prior terminal failure appeared during admission")
    return audit, snapshots


def admit_upstream(root, expected, registered_pins):
    from collections import Counter

    audit, payloads = collect_closure(root, expected)
    for name, signature in audit["files"].items():
        if registered_pins.get(name) != signature:
            raise AssertionError("Closure file absent from current registered inputs: " + name)
    loaded = {"protocol": yaml.safe_load(payloads["range_alert.yaml"])}
    if type(loaded["protocol"]) is not dict:
        raise AssertionError("Original protocol object required")
    for name in ("features", "targets", "forecasts", "states"):
        loaded[name] = pd.read_parquet(io.BytesIO(payloads[UPSTREAM_DATA + name + ".parquet"]))
    for name in ("upstream_admission", "source_audit", "support_audit"):
        loaded[name] = strict_json(payloads[UPSTREAM_DATA + name + ".json"])
    loaded["fits"] = strict_json(payloads[UPSTREAM_DATA + "fits.json"], object_required=False)
    if type(loaded["fits"]) is not list:
        raise AssertionError("Original fit list required")
    loaded["metrics"] = strict_json(payloads[UPSTREAM_REPORT + "metrics.json"])
    loaded["ledger"] = [
        strict_json(line)
        for line in payloads[UPSTREAM_REPORT + "trial_ledger.jsonl"].splitlines()
        if line
    ]
    counts = audit["upstream_verified_counts"]
    for name, key in [
        ("features", "bounded_reference_rows"),
        ("targets", "bounded_reference_rows"),
        ("forecasts", "forecasts"),
        ("states", "common_application_origins"),
        ("fits", "monthly_fits"),
    ]:
        identity(len(loaded[name]), counts[key], "Exact loaded " + name + " count")
    metrics = loaded["metrics"]
    if metrics.get("status") == "UNEVALUABLE" or metrics.get("whole_wave_aborted") is True:
        raise AssertionError("Original canonical failure is not admissible")
    for key, value in [
        ("protocol_sha256", audit["upstream_identity"]["protocol_sha256"]),
        ("hypothesis_count", 2),
        ("cumulative_hypothesis_count", 129),
        ("new_forecasts", counts["forecasts"]),
        ("new_monthly_fits", counts["monthly_fits"]),
        ("common_scored_origins", counts["common_scored_origins"]),
        ("common_application_origins", counts["common_application_origins"]),
    ]:
        identity(metrics[key], value, "Original metrics " + key)
    identity(len(metrics["rows"]), 2, "Original two comparisons")
    identity(len(metrics["inherited_rows"]), 127, "Original inherited comparisons")
    identity(
        dict(Counter(row["event"] for row in loaded["ledger"])),
        {"registered": 2, "inherited": 127, "evaluated": 2},
        "Original complete terminal ledger",
    )
    for name, signature in audit["files"].items():
        if sha256(source_path(root, name).read_bytes()).hexdigest() != signature:
            raise AssertionError("Original input changed during decode")
    if (Path(root) / UPSTREAM_REPORT / "failure.json").exists():
        raise AssertionError("Upstream failure appeared during output decoding")
    return audit, loaded


def validate_scalar_accounting(saved):
    iterations, calls = saved["iterations"], saved["function_calls"]
    if type(iterations) is not int or type(calls) is not int:
        raise AssertionError("Literal integer numerical iteration/evaluation counts required")
    if saved["status"] in ("EMPTY_HISTORY", "BALANCED_AT_ZERO"):
        if iterations != 0 or calls != 0:
            raise AssertionError(
                "Canonical zero branches perform no numerical root iterations"
            )
    elif saved["status"] == "FITTED":
        if not 1 <= iterations <= 200 or not 1 <= calls <= min(202, iterations + 2):
            raise AssertionError("Fixed Brent iteration/evaluation budget differs")
    else:
        raise AssertionError("Unknown scalar solver status")


def phase_statistics(panel, control, phase, code, protocol):
    section = protocol["index"]
    first, last = section[phase]
    selected = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
    if phase == "development":
        selected = selected.loc[
            selected.available_date <= section["development_target_available_by"]
        ]
    groups = {
        name: selected.loc[selected.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    rows, benchmark = groups["calibrated"], groups[control]
    if any(not part.index.equals(rows.index) for part in groups.values()) or len(rows) < 127:
        raise AssertionError(
            "Complete original cohort and at least127 phase observations required"
        )
    difference = paired_difference(rows.probability, benchmark.probability, rows.y)
    values = normalized_difference(difference)
    mean = float(values.mean())
    delta = _finite(mean * 1.0)
    hac = independent_hac(values, maxlags=126)
    blocks = {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + code * 10000 + width
        samples = explicit_bootstrap_means(
            values, width, protocol["inference"]["bootstrap_draws"], seed
        )[:, 0]
        if not np.isfinite(samples).all():
            raise ValueError("Nonfinite independent normalized bootstrap samples")
        probability = float(
            (1 + np.count_nonzero(abs(samples - mean) >= abs(mean))) / (len(samples) + 1)
        )
        blocks[str(width)] = {
            "p": probability,
            "ci95": [_finite(value * 1.0) for value in np.quantile(samples, [0.025, 0.975])],
        }
    hac = {
        "se": _finite(hac["se"] * 1.0),
        "p": _finite(hac["p"]),
        "ci95": [_finite(value * 1.0) for value in hac["ci95"]],
        "mde80_nominal": _finite(hac["mde80_nominal"] * 1.0),
    }
    intervals = [hac["ci95"], *(row["ci95"] for row in blocks.values())]
    result = {
        "name": phase,
        "first_origin": str(rows.index[0].date()),
        "last_origin": str(rows.index[-1].date()),
        "n": len(rows),
        "class_support": support(rows.y, 30),
        "delta": delta,
        "candidate_loss": _finite(rows.loss.mean()),
        "control_loss": _finite(benchmark.loss.mean()),
        "block_inference": blocks,
        "hac126": hac,
        "ci95_envelope": [
            min(pair[0] for pair in intervals),
            max(pair[1] for pair in intervals),
        ],
        "p_conservative": max(hac["p"], *(row["p"] for row in blocks.values())),
        "nominal_mde_effect_ratio": _finite(hac["mde80_nominal"] / EFFECT),
        "annual": [],
        "stability": [],
        "nonoverlap_phases": [{"phase": 0, "n": len(rows), "delta": delta}],
    }
    for year in sorted(set(rows.index.year)):
        mask = rows.index.year == year
        result["annual"].append(
            {
                "year": int(year),
                "n": int(mask.sum()),
                "delta": _finite(values[mask].mean() * 1.0),
            }
        )
    if phase == "evaluation":
        for start, end in section["evaluation_stability"]:
            mask = (rows.index >= start) & (rows.index <= end)
            if not mask.any():
                raise AssertionError("Fixed evaluation slice empty")
            result["stability"].append(
                {
                    "start": start,
                    "end": end,
                    "n": int(mask.sum()),
                    "delta": _finite(values[mask].mean() * 1.0),
                }
            )
    return result


def inherited_rows(root, protocol, *, pins=None):
    root = Path(root)
    if pins is None:
        pins = json.loads((root / "reports/issued_calibration/manifest.json").read_bytes())[
            "inputs"
        ]
    output = []
    for name in protocol["comparisons"]["inherited_sources"]:
        signature = pins[name]
        decoded = read_json_snapshot(root, name, signature)
        for number, row in enumerate(decoded["rows"]):
            output.append(
                {
                    "study": row.get("study", Path(name).parent.name or Path(name).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": name,
                    "source_sha256": signature,
                    "source_row_index": number,
                    **{key: row[key] for key in ("measure", "score") if key in row},
                }
            )
    if len(output) != 129:
        raise AssertionError("All129 inherited comparisons must remain identified")
    return output


def verify_metrics(root, panel, protocol, metrics, application_n, *, admitted_inputs=None):
    validate_panel(panel)
    n = panel.origin.nunique()
    if (
        metrics["new_monthly_fits"] != 0
        or metrics["new_forecasts"] != n
        or metrics["reused_control_forecasts"] != 2 * n
        or metrics["combined_forecasts"] != len(panel)
        or metrics["common_application_origins"] != application_n
        or metrics["common_scored_origins"] != n
    ):
        raise AssertionError("Exact new/reused forecast and zero-refit counts differ")
    section = protocol["index"]
    counts = {}
    calibration = {}
    phase_union = np.zeros(len(panel), dtype=bool)
    for name in ("development", "evaluation"):
        first, last = section[name]
        mask = panel.origin.between(first, last)
        phase_union |= mask
        if not panel.loc[mask, "phase"].eq(name).all():
            raise AssertionError("Literal phase membership differs")
        base = panel.loc[mask & panel.model.eq("recent_frequency")]
        counts[name] = support(base.y, 30)
        if len(base) < 127:
            raise AssertionError("Literal phase bandwidth unsupported")
        calibration[name] = {}
        for model in MODELS:
            rows = panel.loc[mask & panel.model.eq(model)]
            frequency = float(rows.y.mean())
            probability = float(rows.probability.mean())
            calibration[name][model] = {
                "n": len(rows),
                "observed_frequency": frequency,
                "mean_probability": probability,
                "calibration_gap": probability - frequency,
                "brier": float(rows.loss.mean()),
            }
    if (
        not phase_union.all()
        or not panel.horizon.eq(1).all()
        or (panel.train_n < section["minimum_train"]).any()
        or (
            panel.loc[panel.phase.eq("development"), "available_date"]
            > section["development_target_available_by"]
        ).any()
        or (panel.available_date > section["latest_target"]).any()
    ):
        raise AssertionError("Literal inference sample support/date fences differ")
    counts["evaluation_slices"] = []
    for start, end in section["evaluation_stability"]:
        values = panel.loc[
            panel.origin.between(start, end) & panel.model.eq("recent_frequency"), "y"
        ]
        counts["evaluation_slices"].append({"start": start, "end": end, **support(values, 15)})
    same_tree(metrics["class_support"], counts, "Declared phase and slice class support")
    compare_tree(metrics["calibration"], calibration, "Descriptive calibration in the large")
    rows = metrics["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Both fixed Brier controls required")
    probabilities = []
    effects = []
    for row in rows:
        if (
            row["study"] != "issued_calibration"
            or row["horizon"] != 1
            or row["score"] != "brier"
        ):
            raise AssertionError("Strict-sign Brier comparison identity differs")
        phases = [
            phase_statistics(panel, row["control"], phase, code, protocol)
            for code, phase in enumerate(("development", "evaluation"))
        ]
        compare_tree(row["phases"], phases, "Independent Brier phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        compare_tree(row["p_conservative"], probability, "Conjunction of both phases", "p")
        probabilities.append(probability)
        effects.append(
            all(phase["delta"] <= -EFFECT for phase in phases)
            and len(phases[1]["stability"]) == 2
            and all(one["delta"] < 0 for one in phases[1]["stability"])
        )
    prior = inherited_rows(root, protocol, pins=admitted_inputs)
    same_tree(metrics["inherited_rows"], prior, "All inherited comparisons retained")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-2:]
    passed = []
    for number, row in enumerate(rows):
        compare_tree(row["p_holm_wave"], wave[number], "Two-hypothesis wave correction", "p")
        compare_tree(
            row["p_holm_cumulative"],
            cumulative[number],
            "131-hypothesis cumulative correction",
            "p",
        )
        one = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05)
        if row["verdict"] != ("PASSES_ALL_GATES" if one else "DOES_NOT_QUALIFY"):
            raise AssertionError("Fixed statistical/effect/stability gate differs")
        passed.append(one)
    leads = ["calibrated"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 2
        or metrics["cumulative_hypothesis_count"] != 131
    ):
        raise AssertionError("Complete candidate and hypothesis family required")
    return {
        "new_hypotheses_verified": 2,
        "cumulative_hypotheses_verified": 131,
        "phase_comparisons_verified": 4,
        "bootstrap_runs_verified": 12,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "calibration_rows_verified": 6,
        "class_support": counts,
        "leads": leads,
    }


def verify_ledger(root, metrics, prior, final_event, *, signature=None):
    report = Path(root) / "reports/issued_calibration"
    payload = (report / "trial_ledger.jsonl").read_bytes()
    if signature is not None and sha256(payload).hexdigest() != signature:
        raise AssertionError("Trial ledger bytes changed before decoding")
    ledger = [json.loads(line) for line in payload.splitlines() if line]
    for event, expected in (("inherited", prior), (final_event, metrics["rows"])):
        actual = [
            {key: value for key, value in row.items() if key != "event"}
            for row in ledger
            if row["event"] == event
        ]
        compare_tree(actual, expected, "Independent " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (
        len(ledger) != 133
        or len(registered) != 2
        or len(prior) != 129
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "issued_calibration"
            or row["horizon"] != 1
            or row["score"] != "brier"
            for row in registered
        )
    ):
        raise AssertionError("Complete129inherited+2registered+2terminal ledger required")
    return {"inherited": 129, "registered": 2, final_event: 2}


def verify_manifest_coverage(root, protocol, manifest, closure):
    root = Path(root)
    code = {
        str(path.relative_to(root))
        for folder in ("src", "tests")
        for path in (root / folder).rglob("*.py")
    }
    inputs = set(closure["files"]) | set(protocol["comparisons"]["inherited_sources"])
    name = "reports/issued_calibration/freeze_record.json"
    frozen = read_json_snapshot(root, name, manifest["inputs"][name])
    inputs.add(name)
    inputs.update(frozen["prefit_design"])
    identity(frozen["protocol_sha256"], manifest["protocol_sha256"], "New prefit protocol")
    identity(frozen["code"], manifest["code"], "New prefit full code inventory")
    for path, signature in frozen["prefit_design"].items():
        read_snapshot(root, path, signature)
    read_snapshot(
        root,
        "reports/issued_calibration/full_repository_tests.txt",
        frozen["checks"]["full_log_sha256"],
    )
    preserved = {
        str(path.relative_to(root))
        for path in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if path.is_file()
        and path != root / "issued_calibration.yaml"
        and root / "reports/issued_calibration" not in path.parents
        and str(path.relative_to(root)) not in inputs
    }
    if (
        set(manifest["code"]) != code
        or set(manifest["inputs"]) != inputs
        or set(manifest["preserved"]) != preserved
    ):
        raise AssertionError(
            "Full new and historical input/code/publication inventory differs"
        )
    if inputs & preserved:
        raise AssertionError("Duplicate input/publication accounting")
    for path, signature in closure["files"].items():
        if manifest["inputs"].get(path) != signature:
            raise AssertionError("Original closure differs from current input pin")


def verify(root=ROOT):
    root = Path(root)
    protocol_path = root / "issued_calibration.yaml"
    payload = protocol_path.read_bytes()
    protocol = yaml.safe_load(payload)
    validate_protocol(protocol)
    signature = sha256(payload).hexdigest()
    report = root / "reports/issued_calibration"
    out = root / "data/issued_calibration"
    mpath = report / "manifest.json"
    mpayload = mpath.read_bytes()
    mhash = sha256(mpayload).hexdigest()
    manifest = strict_json(mpayload)
    identity(manifest["protocol_sha256"], signature, "New registration protocol")
    for group in ("code", "inputs", "preserved"):
        previous.pins_checked(root, manifest[group])
    closure, _ = collect_closure(root, protocol["upstream"]["anchors"])
    verify_manifest_coverage(root, protocol, manifest, closure)
    paths = [
        out / name
        for name in (
            "upstream_admission.json",
            "forecasts.parquet",
            "states.parquet",
            "calibration_audit.json",
        )
    ] + [report / "metrics.json", report / "trial_ledger.jsonl"]
    snapshots = {str(path.relative_to(root)): digest(path) for path in paths}

    def decode(path):
        name = str(path.relative_to(root))
        return strict_json(read_snapshot(root, name, snapshots[name]))

    def parquet(path):
        name = str(path.relative_to(root))
        return pd.read_parquet(io.BytesIO(read_snapshot(root, name, snapshots[name])))

    metrics = decode(report / "metrics.json")
    if (
        (report / "failure.json").exists()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != signature
        or metrics["evidence_class"] != protocol["evidence_class"]
    ):
        raise AssertionError("Only an intact new scored publication is admissible")
    proof, loaded = admit_upstream(root, protocol["upstream"]["anchors"], manifest["inputs"])
    identity(
        decode(out / "upstream_admission.json"),
        proof,
        "Complete independent original source admission",
    )
    oldindex = loaded["protocol"]["index"]
    expected_index = {**oldindex, "models": list(MODELS)}
    identity(
        protocol["index"],
        expected_index,
        "Unchanged original calendar/target/feature index contract",
    )
    panel = parquet(out / "forecasts.parquet")
    states = parquet(out / "states.parquet")
    audit = decode(out / "calibration_audit.json")
    checked = verify_forecasts(
        loaded["features"],
        loaded["targets"],
        loaded["forecasts"],
        loaded["fits"],
        loaded["states"],
        panel,
        states,
        audit,
        protocol,
    )
    result = {
        "status": "VERIFIED",
        "protocol_sha256": signature,
        "verifier_sha256": digest(Path(__file__)),
        "upstream_admission": {
            "status": proof["status"],
            "historical_models_refitted": False,
            "source_values_reparsed": False,
        },
        "forecast_reconstruction": checked,
        "primitive_brier_scores_verified": len(panel),
        "paired_brier_differences_verified": 2 * checked["common_scored_origins"],
        "inference": verify_metrics(
            root, panel, protocol, metrics, len(states), admitted_inputs=manifest["inputs"]
        ),
        "ledger_events_verified": verify_ledger(
            root,
            metrics,
            metrics["inherited_rows"],
            "evaluated",
            signature=snapshots[str((report / "trial_ledger.jsonl").relative_to(root))],
        ),
        "verified_output_hashes": snapshots,
        "artifact_hashes_checked": {
            **{group: len(manifest[group]) for group in ("code", "inputs", "preserved")},
            "outputs": len(snapshots),
        },
        "limitations": [
            "A single causal intercept calibration of the archived SPX daily risk-alert baseline; forecast-error memory is reused information, not a new exogenous channel or causal structural predictor.",
            "Original monthly coefficients and both controls are preserved; replaying issued logits is not a baseline refit or a retrospective training-probability seed.",
            "Only genuinely issued baseline records contribute after their labels mature. The fixed recent-frequency control intentionally retains its broader known-label information set.",
            "Unnormalized63-session weights and .01 ridge preserve early low information mass; unknown and unissued arrivals age history without artificial labels or phase resets.",
            "Finite original logits define stable Bernoulli residuals even when expit rounds to0 or1; no inverse-logit clipping, numerical underflow repair, or optimizer fallback.",
            "Adaptive selection on repeatedly reused archival history remains exploratory; Brier improvement is neither profit nor proof of conditional calibration or untouched future validation.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        previous.pins_checked(root, manifest[group])
    previous.pins_checked(root, snapshots)
    verify_manifest_coverage(root, protocol, manifest, closure)
    if digest(protocol_path) != signature or digest(mpath) != mhash:
        raise AssertionError("Protocol/manifest changed during verification")
    if (root / UPSTREAM_REPORT / "failure.json").exists():
        raise AssertionError("Upstream failure appeared before verification commit")
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False), flush=True)
