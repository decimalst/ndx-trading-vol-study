"""Independent wave22 publication entry and long-horizon return inference.

Checked source metadata/raw snapshot admission is shared and explicitly rehashed.
All new feature/model/application reconstruction is independent; inference uses
frozen independent bootstrap/HAC primitives without producer inference imports.
"""

from __future__ import annotations

import errno
import io
import json
import math
import os
import re
import stat
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import issued_calibration_admission as checked
from . import peak_age_admission as shared
from . import verify_event_cluster as byte_checks
from .verify_index_hinge import explicit_bootstrap_means, independent_hac, nonoverlap_rows
from .verify_iterative_signal_search import holm, same

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/peak_age"
OUT = "data/peak_age"
PROTOCOL = "peak_age.yaml"
MODELS = ("mean", "baseline", "depth", "peak_age")
CONTROLS = ("baseline", "depth", "mean")
COMPARISONS = tuple(("peak_age", control, 21) for control in CONTROLS)
WAVE_ALPHA = 0.05 / (22 * 23)
PANEL_COLUMNS = (
    "origin",
    "fit_origin",
    "feature_cutoff_date",
    "target_end",
    "available_date",
    "horizon",
    "phase",
    "model",
    "prediction",
    "y",
    "loss",
    "train_n",
)
OUTPUT_PATHS = tuple(
    [
        OUT + "/" + name
        for name in (
            "upstream_admission.json",
            "features.parquet",
            "targets.parquet",
            "feature_states.parquet",
            "forecasts.parquet",
            "fits.json",
            "application_states.parquet",
            "support_audit.json",
        )
    ]
    + [REPORT + "/metrics.json", REPORT + "/trial_ledger.jsonl"]
)
strict_json = byte_checks.strict_json
source_path = byte_checks.source_path
hash_mapping = byte_checks.hash_mapping
digest = byte_checks.digest
read_snapshot = byte_checks.read_snapshot
read_json_snapshot = byte_checks.read_json_snapshot
pins_checked = byte_checks.pins_checked
identity = checked._same

CONTRACT = {
    "study_id": "peak_age_wave22",
    "specified_on": "2026-09-08",
    "status": "specified_before_historical_admission_features_support_fits_or_scores",
    "wave": 22,
    "evidence_class": "exploratory_reused_archival_SPX_return_history",
    "objective": "Test trailing closing-price peak age as a sequential correction beyond drawdown "
    "depth, window return and fixed market controls",
    "sources": {
        "daily": "data/research_paths/spx_daily.parquet",
        "vix": "data/free_sources/raw/cboe/VIX_History.csv",
        "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
        "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
    },
    "source_contract": {
        "price": "raw SPX OHLC; Yahoo price index, no dividends or risk-free "
        "subtraction; no fund execution claim",
        "vintage": "existing archival Yahoo and Cboe extracts; historical revisions "
        "and exact release latency unverified",
        "vix9d": "January2011-October2013 back-calculated archival training values; "
        "not available at original observation dates",
        "missing": "preserve observed SPX reference calendar before rolling; no "
        "filling; strict complete windows",
        "timing": "every market feature ends at previous observed SPX session; entry "
        "weekday only calendar feature",
        "variance_proxy": "max(Garman-Klass daily variance,1e-10) plus squared raw "
        "overnight log return",
        "scale": "annualization252 is explicit convention; VIX30calendar-day variance "
        "and trailing22trading-session OHLC proxy differ",
        "interpretation": "log implied-versus-trailing-proxy gap; neither measured "
        "variance risk premium nor replication of high-frequency "
        "premium research",
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
        "minimum_train": 1000,
        "market_lag": 1,
        "horizons": [21],
        "models": ["mean", "baseline", "depth", "peak_age"],
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
            "I_square",
            "R_square",
        ],
        "depth": [
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
            "I_square",
            "R_square",
            "drawdown",
            "drawdown_sq",
            "window_return",
        ],
        "common": [
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
            "peak_age",
            "drawdown",
            "drawdown_sq",
            "window_return",
        ],
        "target": "log(raw_close[t+21]/raw_close[t]); target end and availability at observed "
        "session t+21 close",
        "fit_schedule": "First common-feature-complete origin of each month within origin/phase "
        "fences before query-label filtering; all four arms use identical "
        "mature training and full application origins",
        "training_availability": "Earlier common-complete origins with finite labels available "
        "no later than previous observed session of fit; whole-trial "
        "failure below1000",
        "missing": "Keep full reference calendar; exact252-close windows; no "
        "fill/compress/window shortening; no arithmetic-fault row dropping",
    },
    "peak": {
        "window_closes": 252,
        "return_intervals": 251,
        "cutoff": "Previous observed SPX session k; literal positions k-251..k",
        "tie": "Latest exact stored float64 close equal to window maximum P",
        "age": "(k-latest_peak_position)/251; fixed scale one",
        "depth": "Stable log ratio P/C[k]",
        "depth_square": "Single multiplication of returned depth",
        "window_return": "Stable log ratio C[k]/C[k-251]",
        "log_ratio": "Positive finite float64 x,y; exact equality returns0; frexp "
        "mantissas/exponents mx,ex,my,ey, d=ex-ey; abs(d)<=1 uses "
        "log1p((ldexp(mx,d)-my)/my), otherwise fsum(log(mx/my),d*log(2)); finite "
        "result must have correct nonzero sign for unequal prices",
        "arithmetic": "No overflow or nonzero-product underflow in required operations; "
        "incomplete source windows remain missing; arithmetic invalidity on "
        "otherwise observed inputs aborts trial",
        "interpretation": "Age of latest maximum inside rolling252-close window, not "
        "all-time-high or recovery duration",
    },
    "model": {
        "ridge": 0.01,
        "baseline_slopes": 16,
        "depth_slopes": 19,
        "curvature": "I_square and R_square centered on exact common training rows only",
        "scaling": "Population mean/std on common training rows; all declared nuisance scales "
        "finite and >1e-12; no unused hinge and no feature deletion",
        "intercept": "Unpenalized common training target mean",
        "objective": "Mean squared error plus .01 squared norm of standardized slopes; "
        "separately refit baseline and depth",
        "scalar": "Freeze current monthly depth fit; residuals are y minus its in-sample "
        "training predictions; age centered by math.fsum/n, exactconstant uses first "
        "age; beta=mean(centered_age*residual)/(mean(centered_age^2)+.01)",
        "application": "depth_prediction+beta*(query_age-training_age_mean); no additional "
        "intercept, no joint refit, no age standardization",
        "constant": "Exactconstant training age or exactlyzero numerator yields beta0 and exact "
        "copy of depth query predictions, even if query age differs",
        "attribution": "Sequential correction can retune ridge shrinkage inside nuisance span; "
        "no Frisch-Waugh residualization or orthogonality claim",
        "fallback": "None; any source/support/model/numerical/inference/publication failure "
        "aborts all three hypotheses at p1",
    },
    "comparisons": {
        "new_hypotheses": 3,
        "inherited_hypotheses": 137,
        "cumulative_hypotheses": 140,
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
            "reports/event_cluster_replay/metrics.json",
        ],
        "contrasts": [
            ["peak_age", "baseline", 21],
            ["peak_age", "depth", 21],
            ["peak_age", "mean", 21],
        ],
        "gate": "All three controls, both phases >=.25percent relative MSE improvement; "
        "negative paired differences in both fixed later slices and every "
        "nonempty21-offset in both phases; wave and cumulative Holm",
    },
    "inference": {
        "loss": "squared 21-session cumulative log-price-return prediction error",
        "effect_relative": 0.0025,
        "blocks": [126, 252, 504],
        "hac_lags": 504,
        "minimum_phase_observations": 505,
        "bootstrap_draws": 399999,
        "seed": 20260928,
        "seed_rule": "seed+21*1000000+phase_code*10000+block; development0,evaluation1; "
        "same drawdesign across controls",
        "wave_alpha": 9.881422924901186e-05,
        "cumulative_alpha": 0.05,
        "p_phase": "Maximum two-sided centered-null circular block bootstrap p and Bartlett "
        "HAC504 p",
        "p_hypothesis": "Maximum development/evaluation p",
        "multiplicity": "Holm3 against wave_alpha and cumulativeHolm140 against .05",
        "offsets": "All offsets0..20 anchored to full bounded SPX calendar position "
        "modulo21 before any filtering; each must be nonempty and improve",
        "resolution": "1/400000 below one tenth strictest raw Holm3 wave cutoff1/30360",
        "limits": "Repeated archival selection, overlapping targets and original "
        "source-vintage limits; no untouched confirmation or executable trading "
        "claim",
    },
    "verification": {
        "ridge_method": "Independent augmented least squares",
        "scalar_method": "Independent one-column augmented least squares conditional on "
        "validated saved depth fit",
        "gradient_max_abs": 1e-10,
        "coefficient_rtol": 1e-07,
        "coefficient_atol": 1e-12,
        "forecast_rtol": 1e-08,
        "forecast_atol": 1e-12,
        "derived_rtol": 1e-10,
        "derived_atol": 1e-12,
        "date_rule": "Native naive ms/us/ns normalized-midnight dates, exact instants "
        "and NaT masks, checked lossless ns conversion with roundtrip; all "
        "nondates keep declared types and finite-value checks",
        "coverage": "Entire bounded feature/target/state tables, every monthly common "
        "training/application cohort, all saved coefficients and gradients, "
        "all unscored applications, scored predictions/losses, independent "
        "three-contrast inference, complete ledger and "
        "input/output/preservation identities",
        "feature_rtol": 1e-10,
        "feature_atol": 1e-12,
        "target_rtol": 1e-10,
        "target_atol": 1e-13,
    },
    "upstream": {
        "anchors": {
            "event_cluster_replay.yaml": "b3eeea877c536a25cde0edca69cd7cc1474a425a2458ac36978adfeacee7e76f",
            "reports/event_cluster_replay/publication_audit.json": "1e962011d0ccf66776a05a09ba09e8420f5d48f4c646e6bc8616ec07c81bdac5",
        },
        "role": "Verified terminal wave21 plus complete transitive source and failed-family "
        "preservation closure; decode only checked bounded SPX OHLC and Cboe "
        "buffers; no old forecasts used as newly fitted controls",
    },
    "outputs": {"data": "data/peak_age", "reports": "reports/peak_age"},
}


def validate_protocol(protocol):
    identity(protocol, CONTRACT, "literal typed complete peak-age protocol")


def same_tree(actual, expected, label):
    if type(expected) is dict:
        if type(actual) is not dict or actual.keys() != expected.keys():
            raise ValueError("Independent inference schema differs: " + label)
        for key in expected:
            same_tree(actual[key], expected[key], label + "." + key)
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError("Independent inference list differs: " + label)
        for a, b in zip(actual, expected, strict=True):
            same_tree(a, b, label)
    elif type(expected) is float:
        if (
            type(actual) is not float
            or not math.isfinite(actual)
            or not math.isfinite(expected)
        ):
            raise ValueError("Finite literal floating point diagnostic required: " + label)
        same(actual, expected, label, rtol=2e-8, atol=1e-14)
    else:
        identity(actual, expected, label)


def validate_panel(panel):
    if type(panel) is not pd.DataFrame or tuple(panel.columns) != PANEL_COLUMNS or panel.empty:
        raise ValueError("Complete nonempty declared four-arm panel schema required")
    if (
        panel.duplicated(["origin", "model"]).any()
        or set(panel.model) != set(MODELS)
        or not pd.api.types.is_integer_dtype(panel.horizon.dtype)
        or not panel.horizon.eq(21).all()
        or not pd.api.types.is_integer_dtype(panel.train_n.dtype)
        or not panel.train_n.ge(2).all()
    ):
        raise ValueError("Literal unique complete four-arm21-session predictions required")
    if not panel.reset_index(drop=True).equals(
        panel.sort_values(["origin", "model"]).reset_index(drop=True)
    ):
        raise ValueError("Canonical origin/model ordering required")
    for name in (
        "origin",
        "fit_origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
    ):
        s = panel[name]
        if (
            not pd.api.types.is_datetime64_dtype(s.dtype)
            or s.isna().any()
            or s.dt.tz is not None
            or s.dt.unit not in ("ms", "us", "ns")
            or not s.equals(s.dt.normalize())
        ):
            raise ValueError("Native known naive midnight panel dates required")
    if (
        not panel.feature_cutoff_date.lt(panel.origin).all()
        or not panel.fit_origin.le(panel.origin).all()
        or not panel.target_end.gt(panel.origin).all()
        or not panel.target_end.equals(panel.available_date)
    ):
        raise ValueError("Causal issued/target dates required")
    if not set(panel.phase).issubset({"development", "evaluation"}):
        raise ValueError("Only two declared historical phases required")
    for name in ("prediction", "y", "loss"):
        if panel[name].dtype != np.dtype("float64") or not np.isfinite(panel[name]).all():
            raise ValueError("Finite float64 saved numerical forecasts required")
    residual = panel.y.to_numpy() - panel.prediction.to_numpy()
    with np.errstate(all="ignore"):
        loss = residual * residual
    if (
        not np.isfinite(loss).all()
        or ((residual != 0) & (loss == 0)).any()
        or not np.array_equal(panel.loss.to_numpy(), loss)
    ):
        raise ValueError("Every exact finite squared error must independently reconstruct")
    reference = panel.loc[panel.model == "mean"].set_index("origin")
    fields = (
        "fit_origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "horizon",
        "phase",
        "y",
        "train_n",
    )
    for model in MODELS:
        part = panel.loc[panel.model == model].set_index("origin")
        if not reference.index.equals(part.index) or any(
            not reference[n].equals(part[n]) for n in fields
        ):
            raise ValueError(
                "All four arms require identical origin, label and timing identities"
            )


def effect_passes(phases, horizon):
    return (
        horizon == 21
        and len(phases) == 2
        and [phase["name"] for phase in phases] == ["development", "evaluation"]
        and all(
            phase["n"] >= 505 and phase["delta"] < 0 and phase["gain_relative"] >= 0.0025
            for phase in phases
        )
        and len(phases[1]["stability"]) == 2
        and all(part["n"] > 0 and part["delta"] < 0 for part in phases[1]["stability"])
        and all(
            len(phase["nonoverlap_phases"]) == 21
            and [part["phase"] for part in phase["nonoverlap_phases"]] == list(range(21))
            and all(
                part["n"] > 0 and part["delta"] is not None and part["delta"] < 0
                for part in phase["nonoverlap_phases"]
            )
            for phase in phases
        )
    )


def phase_statistics(panel, control, horizon, phase, code, protocol, calendar):
    validate_panel(panel)
    if (
        control not in CONTROLS
        or horizon != 21
        or (phase, code) not in (("development", 0), ("evaluation", 1))
    ):
        raise ValueError("Fixed control, horizon and phase seed identity required")
    section = protocol["index"]
    first, last = section[phase]
    selected = panel.loc[
        (panel.horizon == horizon) & (panel.origin >= first) & (panel.origin <= last)
    ]
    if phase == "development":
        selected = selected.loc[
            selected.available_date <= section["development_target_available_by"]
        ]
    wide = selected.pivot(index="origin", columns="model", values="prediction").sort_index()
    actual = selected.loc[selected.model == control].set_index("origin").reindex(wide.index).y
    candidate_loss, control_loss = (
        (actual - wide.peak_age).to_numpy() ** 2,
        (actual - wide[control]).to_numpy() ** 2,
    )
    difference = candidate_loss - control_loss
    if len(difference) <= max(protocol["inference"]["blocks"]) or not control_loss.mean() > 0:
        raise AssertionError("Insufficient inference observations or zero control loss")
    delta, hac = float(difference.mean()), independent_hac(difference, maxlags=504)
    blocks = {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + horizon * 1000000 + code * 10000 + width
        samples = explicit_bootstrap_means(
            difference, width, protocol["inference"]["bootstrap_draws"], seed
        )[:, 0]
        p = float(
            (1 + np.count_nonzero(np.abs(samples - delta) >= abs(delta))) / (len(samples) + 1)
        )
        blocks[str(width)] = {"p": p, "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}
    intervals = [hac["ci95"], *(part["ci95"] for part in blocks.values())]
    result = {
        "name": phase,
        "first_origin": str(wide.index[0].date()),
        "last_origin": str(wide.index[-1].date()),
        "n": len(wide),
        "delta": delta,
        "candidate_loss": float(candidate_loss.mean()),
        "control_loss": float(control_loss.mean()),
        "gain_relative": float(1 - candidate_loss.mean() / control_loss.mean()),
        "block_inference": blocks,
        "hac504": hac,
        "ci95_envelope": [
            min(item[0] for item in intervals),
            max(item[1] for item in intervals),
        ],
        "p_conservative": max(hac["p"], *(part["p"] for part in blocks.values())),
        "annual": [],
        "stability": [],
        "nonoverlap_phases": nonoverlap_rows(wide.index, difference, calendar, horizon),
    }
    for year in sorted(set(wide.index.year)):
        mask = wide.index.year == year
        result["annual"].append(
            {"year": int(year), "n": int(mask.sum()), "delta": float(difference[mask].mean())}
        )
    if phase == "evaluation":
        for start, end in section["evaluation_stability"]:
            mask = (wide.index >= start) & (wide.index <= end)
            if not mask.any():
                raise AssertionError("Fixed stability slice is empty")
            result["stability"].append(
                {
                    "start": start,
                    "end": end,
                    "n": int(mask.sum()),
                    "delta": float(difference[mask].mean()),
                }
            )
    return result


def require_failed_metrics(study, record):
    count = {"civil_quarter": 2, "civil_quarter_replay": 2, "event_cluster": 3}[study]
    if (
        record.get("status") != "UNEVALUABLE"
        or record.get("whole_wave_aborted") is not True
        or record.get("leads") != []
        or type(record.get("rows")) is not list
        or len(record["rows"]) != count
    ):
        raise ValueError("Complete canonical failed family must remain unevaluable: " + study)
    for row in record["rows"]:
        for key in ("p_conservative", "p_holm_wave", "p_holm_cumulative"):
            value = row[key]
            if type(value) not in (int, float) or value != 1:
                raise ValueError("Every inherited failed-family p-value must remain one")


def inherited_rows(root, protocol, *, pins=None):
    root = Path(root)
    if pins is None:
        pins = strict_json((root / REPORT / "manifest.json").read_bytes())["inputs"]
    pins = hash_mapping(pins)
    names = protocol["comparisons"]["inherited_sources"]
    identity(
        names,
        CONTRACT["comparisons"]["inherited_sources"],
        "complete declared prior source list",
    )
    rows = []
    for name in names:
        signature = pins[name]
        document = read_json_snapshot(root, name, signature)
        study = Path(name).parent.name
        if study in ("civil_quarter", "civil_quarter_replay", "event_cluster"):
            require_failed_metrics(study, document)
        for number, row in enumerate(document["rows"]):
            value = row["p_conservative"]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Finite real inherited p-value required")
            rows.append(
                {
                    "study": row.get("study", study or Path(name).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": value,
                    "source": name,
                    "source_sha256": signature,
                    "source_row_index": number,
                    **{key: row[key] for key in ("measure", "score") if key in row},
                }
            )
    if len(rows) != 137:
        raise ValueError("All137 canonical prior hypotheses must remain")
    return rows


def verify_metrics(
    root,
    panel,
    protocol,
    metrics,
    calendar,
    application_n,
    monthly_schedules,
    *,
    admitted_inputs=None,
):
    validate_panel(panel)
    dates = pd.DatetimeIndex(calendar)
    shared._dates(dates)
    positions = dates.get_indexer(panel.origin)
    if (positions < 1).any() or (positions + 21 >= len(dates)).any():
        raise ValueError("All scored clocks must fit the full bounded reference calendar")
    for name, offset in (
        ("feature_cutoff_date", -1),
        ("target_end", 21),
        ("available_date", 21),
    ):
        expected = dates[positions + offset]
        if not np.array_equal(panel[name].to_numpy(), expected.to_numpy()):
            raise ValueError(
                "Exact independently reconstructed observed-session clock differs: " + name
            )
    section = protocol["index"]
    development = panel.origin.between(*section["development"])
    evaluation = panel.origin.between(*section["evaluation"])
    if (
        not (development | evaluation).all()
        or not np.array_equal(panel.phase, np.where(development, "development", "evaluation"))
        or not panel.train_n.ge(section["minimum_train"]).all()
        or not panel.available_date.le(pd.Timestamp(section["latest_target"])).all()
        or not panel.loc[development, "available_date"]
        .le(pd.Timestamp(section["development_target_available_by"]))
        .all()
    ):
        raise ValueError("Literal phase, maturity and minimum training support required")
    checked._integer(monthly_schedules)
    checked._integer(application_n)
    n = int(panel.origin.nunique())
    if application_n < n:
        raise ValueError("Full applications must include all scored origins")
    counts = {
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 140,
        "newly_generated_forecasts": 4 * n,
        "common_scored_origins": n,
        "common_application_origins": application_n,
        "full_application_predictions": 4 * application_n,
        "monthly_schedules": monthly_schedules,
        "ridge_models_fitted": 2 * monthly_schedules,
        "scalar_models_fitted": monthly_schedules,
        "training_means_computed": monthly_schedules,
    }
    for key, value in counts.items():
        identity(metrics[key], value, "verified count " + key)
    rows = metrics["rows"]
    identity(
        [(r["candidate"], r["control"], r["horizon"]) for r in rows],
        list(COMPARISONS),
        "complete ordered family",
    )
    probabilities = []
    effects = []
    for row in rows:
        identity(row["study"], "peak_age", "new study identity")
        identity(row["score"], "mse", "new score identity")
        phases = [
            phase_statistics(panel, row["control"], 21, phase, code, protocol, calendar)
            for code, phase in enumerate(("development", "evaluation"))
        ]
        same_tree(row["phases"], phases, "independent paired MSE inference and all offsets")
        probability = max(phase["p_conservative"] for phase in phases)
        same_tree(row["p_conservative"], probability, "both-phase conjunction")
        probabilities.append(probability)
        effects.append(effect_passes(phases, 21))
    prior = inherited_rows(root, protocol, pins=admitted_inputs)
    identity(metrics["inherited_rows"], prior, "all canonical inherited trials")
    wave = holm(probabilities)
    cumulative = holm([r["p_conservative"] for r in prior] + probabilities)[-3:]
    passes = []
    for i, row in enumerate(rows):
        same_tree(row["p_holm_wave"], float(wave[i]), "three-way wave Holm")
        same_tree(row["p_holm_cumulative"], float(cumulative[i]), "140-way cumulative Holm")
        passed = bool(effects[i] and wave[i] < WAVE_ALPHA and cumulative[i] < 0.05)
        identity(
            row["verdict"],
            "COMPARISON_GATE_PASS" if passed else "DOES_NOT_QUALIFY",
            "complete comparison gate",
        )
        passes.append(passed)
    leads = [21] if all(passes) else []
    identity(metrics["leads"], leads, "all three controls required for a lead")
    return {
        "new_hypotheses_verified": 3,
        "cumulative_hypotheses_verified": 140,
        "phase_comparisons_verified": 6,
        "bootstrap_runs_verified": 18,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "hac_maxlags": 504,
        "nonoverlap_offsets_verified": 126,
        "leads": leads,
    }


def verify_ledger(root, metrics, prior, final_event, *, signature=None):
    payload = source_path(root, REPORT + "/trial_ledger.jsonl").read_bytes()
    if signature is not None and sha256(payload).hexdigest() != signature:
        raise ValueError("Trial ledger bytes changed before decoding")
    ledger = [strict_json(line) for line in payload.splitlines()]
    if len(prior) != 137 or final_event not in ("evaluated", "failed"):
        raise ValueError("Complete137 inherited rows and declared final event required")
    expected = [
        {
            "event": "registered",
            "study": "peak_age",
            "candidate": a,
            "control": b,
            "horizon": h,
            "score": "mse",
            "protocol_sha256": metrics["protocol_sha256"],
        }
        for a, b, h in COMPARISONS
    ]
    expected += [{"event": "inherited", **r} for r in prior]
    expected += [{"event": final_event, **r} for r in metrics["rows"]]
    identity(ledger, expected, "exact ordered143-event trial ledger")
    return {"registered": 3, "inherited": 137, final_event: 3}


def require_upstream_success(root):
    for study in ("range_alert", "issued_calibration", "event_cluster_replay", "index_hinge"):
        path = Path(root) / "reports" / study / "failure.json"
        if path.exists() or path.is_symlink():
            raise ValueError(
                "Successful upstream failure marker blocks verification: " + study
            )


def require_bound_failure(root, proof):
    require_upstream_success(root)
    files = hash_mapping(proof["files"])
    name = "reports/event_cluster/"
    failure, metrics, verification = [
        read_json_snapshot(root, name + part, files[name + part])
        for part in ("failure.json", "metrics.json", "verification.json")
    ]
    require_failed_metrics("event_cluster", metrics)
    identity(failure, metrics, "exact required old canonical failure")
    signature = files["event_cluster.yaml"]
    identity(metrics["protocol_sha256"], signature, "failed wave20 protocol")
    identity(
        verification,
        {
            "status": "FAILED",
            "protocol_sha256": signature,
            "error": "Independent verification failed: AssertionError: every saved application state differs from independent reconstruction",
        },
        "identified unchanged old verification failure",
    )
    require_upstream_success(root)


def collect_closure(root, expected):
    require_upstream_success(root)
    expected = hash_mapping(expected)
    for name, signature in expected.items():
        read_snapshot(root, name, signature)
    collected = shared._collect(root, expected)
    proof = collected.audit
    identity(proof["anchors"], expected, "literal completed source anchors")
    files = hash_mapping(proof["files"])
    pins_checked(root, files)
    for name, signature in expected.items():
        identity(files[name], signature, "source anchor identity")
    paths = sorted(n for n in files if re.fullmatch(r"reports/[^/]+/manifest\.json", n))
    identity(proof["manifest_paths"], paths, "complete source manifest inventory")
    counts = {}
    for name in paths:
        document = read_json_snapshot(root, name, files[name])
        counts[name] = {}
        for group in ("code", "inputs", "preserved", "existing_artifacts_sha256"):
            refs = hash_mapping(document.get(group, {}))
            counts[name][group] = len(refs)
            for path, signature in refs.items():
                if files.get(path) != signature:
                    raise ValueError("Missing transitive admitted source: " + path)
    identity(proof["manifest_groups"], counts, "independent transitive group coverage")
    identity(
        proof["counts"],
        {
            "files": len(files),
            "manifests": len(paths),
            "raw_sources": 4,
            "preceding_cumulative_hypotheses": 137,
            "preceding_ledger_events": 140,
        },
        "complete preceding closure counts",
    )
    identity(proof["source_paths"], CONTRACT["sources"], "four exact declared raw sources")
    name = "reports/event_cluster_replay/verification.json"
    preceding = read_json_snapshot(root, name, files[name])
    identity(preceding["status"], "VERIFIED", "preceding successful independent proof")
    identity(
        preceding["inference"]["cumulative_hypotheses_verified"],
        137,
        "preceding verified family",
    )
    identity(
        preceding["verified_output_hashes"],
        proof["preceding_verified_output_hashes"],
        "preceding output proof",
    )
    require_bound_failure(root, proof)
    return proof


def admit_upstream(root, expected, registered_pins):
    proof = collect_closure(root, expected)
    registered = hash_mapping(registered_pins)
    for name, signature in proof["files"].items():
        if registered.get(name) != signature:
            raise ValueError("Complete source registration missing: " + name)
    actual, daily, iv = shared.admit_upstream(root, expected, registered)
    identity({k: actual[k] for k in proof}, proof, "checked raw admission metadata unchanged")
    pins_checked(root, proof["files"])
    require_bound_failure(root, proof)
    return actual, daily, iv


def _test_log(payload, expected=None):
    text = payload.decode("utf-8")
    matches = re.findall(r"Ran (\d+) tests? in [0-9.]+s", text)
    if not matches or "\nOK" not in text or "FAILED (" in text:
        raise ValueError("Passing preregistered test execution evidence required")
    count = int(matches[-1])
    checked._integer(count)
    if expected is not None:
        identity(count, expected, "frozen full-suite execution count")
    return count


def verify_manifest_coverage(root, protocol, manifest, closure):
    root = Path(root)
    code = {
        str(p.relative_to(root))
        for folder in ("src", "tests")
        for p in (root / folder).rglob("*.py")
    }
    name = REPORT + "/freeze_record.json"
    frozen = read_json_snapshot(root, name, manifest["inputs"][name])
    identity(
        frozen["protocol_sha256"],
        manifest["protocol_sha256"],
        "registered new freeze protocol",
    )
    identity(frozen["code"], manifest["code"], "entire new code freeze")
    pins_checked(root, frozen["prefit_design"])
    full = read_snapshot(
        root, REPORT + "/full_repository_tests.txt", frozen["checks"]["full_log_sha256"]
    )
    _test_log(full, frozen["checks"]["full_repository_tests"])
    inputs = (
        set(closure["files"])
        | set(protocol["comparisons"]["inherited_sources"])
        | {name}
        | set(frozen["prefit_design"])
    )
    preserved = {
        str(p.relative_to(root))
        for p in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if p.is_file()
        and p != root / PROTOCOL
        and root / REPORT not in p.parents
        and str(p.relative_to(root)) not in inputs
    }
    if (
        set(manifest["code"]) != code
        or set(manifest["inputs"]) != inputs
        or set(manifest["preserved"]) != preserved
    ):
        raise ValueError("Complete source, code and preservation inventory required")
    for name, signature in closure["files"].items():
        if manifest["inputs"].get(name) != signature:
            raise ValueError("Changed admitted source identity: " + name)


def verify_pipeline(*args):
    from .peak_age_verification import verify_pipeline as independent

    return independent(*args)


def verify(root=ROOT):
    root = Path(root)
    report = root / REPORT
    payload = source_path(root, PROTOCOL).read_bytes()
    signature = sha256(payload).hexdigest()
    protocol = yaml.safe_load(payload)
    validate_protocol(protocol)
    manifest_bytes = source_path(root, REPORT + "/manifest.json").read_bytes()
    manifest_signature = sha256(manifest_bytes).hexdigest()
    manifest = strict_json(manifest_bytes)
    identity(manifest["protocol_sha256"], signature, "registered new protocol")
    for group in ("code", "inputs", "preserved"):
        pins_checked(root, manifest[group])
    closure = collect_closure(root, protocol["upstream"]["anchors"])
    verify_manifest_coverage(root, protocol, manifest, closure)
    identity(
        manifest["code"]["src/verify_peak_age.py"],
        digest(Path(__file__)),
        "executing frozen independent verifier",
    )
    check_bytes = source_path(root, REPORT + "/pre_run_checks.txt").read_bytes()
    check_signature = sha256(check_bytes).hexdigest()
    focused = _test_log(check_bytes)
    buffers = {name: source_path(root, name).read_bytes() for name in OUTPUT_PATHS}
    snapshots = {name: sha256(data).hexdigest() for name, data in buffers.items()}
    metrics = strict_json(buffers[REPORT + "/metrics.json"])
    if (
        (report / "failure.json").exists()
        or (report / "failure.json").is_symlink()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics.get("whole_wave_aborted") is True
    ):
        raise ValueError("Only an intact newly scored attempt may be independently verified")
    identity(metrics["protocol_sha256"], signature, "new scored protocol identity")
    identity(
        metrics["evidence_class"], protocol["evidence_class"], "new scored evidence class"
    )
    audit, daily, iv = admit_upstream(
        root, protocol["upstream"]["anchors"], manifest["inputs"]
    )
    identity(
        strict_json(buffers[OUT + "/upstream_admission.json"]),
        audit,
        "saved exact raw-source admission",
    )
    tables = {
        name: pd.read_parquet(io.BytesIO(buffers[OUT + "/" + name + ".parquet"]))
        for name in (
            "features",
            "targets",
            "feature_states",
            "forecasts",
            "application_states",
        )
    }
    fits = strict_json(buffers[OUT + "/fits.json"], object_required=False)
    if type(fits) is not list:
        raise ValueError("Ordered complete monthly fit records required")
    support = strict_json(buffers[OUT + "/support_audit.json"])
    panel = tables["forecasts"]
    validate_panel(panel)
    reconstructed = verify_pipeline(
        daily,
        iv,
        protocol,
        tables["features"],
        tables["targets"],
        tables["feature_states"],
        panel,
        fits,
        tables["application_states"],
        support,
    )
    n = int(panel.origin.nunique())
    a = len(tables["application_states"])
    k = len(fits)
    counts = {
        "forecasts_verified": 4 * n,
        "primitive_squared_losses_verified": 4 * n,
        "application_origins_verified": a,
        "application_predictions_verified": 4 * a,
        "common_scored_origins": n,
        "unscored_origins_verified": a - n,
        "monthly_fits_verified": k,
        "new_nuisance_fits_verified": 2 * k,
        "new_scalar_fits_verified": k,
        "training_mean_fits_verified": k,
    }
    for key, value in counts.items():
        identity(reconstructed[key], value, "independent reconstruction coverage " + key)
    result = {
        "status": "VERIFIED",
        "protocol_sha256": signature,
        "manifest_sha256": manifest_signature,
        "verifier_sha256": digest(Path(__file__)),
        "upstream_admission": {
            "status": audit["status"],
            "metadata_implementation": "shared checked source traversal and raw snapshot admission with independent anchor, full-file and manifest-reference rehashes",
            "numeric_post_cutoff_values_parsed": False,
            "prior_forecast_tables_decoded": False,
            "historical_vintage_certified": False,
        },
        "forecast_reconstruction": reconstructed,
        "primitive_squared_losses_verified": len(panel),
        "paired_squared_loss_differences_verified": 3 * n,
        "inference": verify_metrics(
            root,
            panel,
            protocol,
            metrics,
            daily.index,
            a,
            k,
            admitted_inputs=manifest["inputs"],
        ),
        "ledger_events_verified": verify_ledger(
            root,
            metrics,
            metrics["inherited_rows"],
            "evaluated",
            signature=snapshots[REPORT + "/trial_ledger.jsonl"],
        ),
        "verified_output_hashes": snapshots,
        "artifact_hashes_checked": {
            **{group: len(manifest[group]) for group in ("code", "inputs", "preserved")},
            "outputs": 10,
        },
        "focused_pre_run_tests": focused,
        "pre_run_checks_sha256": check_signature,
        "limitations": [
            "Sequential age correction can retune ridge shrinkage within the nuisance span; any improvement is model-relative and does not establish an orthogonal information channel.",
            "All four arms were refit on identical new common cohorts; this still reuses archival history with overlapping21-session labels.",
            "Raw SPX price-index returns exclude dividends and risk-free subtraction; source-vintage, VIX9D back-calculation and execution limits remain.",
            "All three controls, both phases, both fixed evaluation slices, every offset, and wave/cumulative correction remain required for a lead.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        pins_checked(root, manifest[group])
    pins_checked(root, snapshots)
    verify_manifest_coverage(root, protocol, manifest, closure)
    read_snapshot(root, PROTOCOL, signature)
    read_snapshot(root, REPORT + "/manifest.json", manifest_signature)
    read_snapshot(root, REPORT + "/pre_run_checks.txt", check_signature)
    pins_checked(root, closure["files"])
    require_bound_failure(root, closure)
    if (report / "failure.json").exists() or (report / "failure.json").is_symlink():
        raise ValueError("New failure marker appeared before independent verification commit")
    _atomic_report_write(
        root,
        "verification.json",
        (json.dumps(result, indent=2, allow_nan=False) + "\n").encode(),
    )
    return result


def _report_directory(root, *, create=False):
    directory = Path(root).resolve()
    for part in REPORT.split("/"):
        directory = directory / part
        if directory.is_symlink():
            raise ValueError("Refusing to mutate a symlinked report directory")
        if create:
            directory.mkdir(exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("An existing regular report directory is required")
    return directory


def _regular_bytes(path):
    """Capture only a regular current leaf; never treat a link target as history."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == errno.ELOOP:
            return None
        raise
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Regular current report files are required for preservation")
        return stream.read()


def _atomic_report_write(root, name, payload):
    """Install a new regular leaf instead of following an existing file link."""
    if Path(name).name != name or name in ("", ".", "..") or type(payload) is not bytes:
        raise ValueError("One report basename and exact byte payload required")
    directory = _report_directory(root)
    descriptor, temporary = tempfile.mkstemp(prefix=".peak-age-entry-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _report_directory(root)
        os.replace(temporary, directory / name)
    finally:
        Path(temporary).unlink(missing_ok=True)


def invalidate_publication(root, error):
    report = _report_directory(root, create=True)
    original = _regular_bytes(report / "metrics.json")
    prior = None
    if original is not None:
        try:
            prior = strict_json(original)
        except (ValueError, UnicodeDecodeError):
            pass
    signature = prior.get("protocol_sha256") if prior else None
    message = "Independent verification failed: " + type(error).__name__ + ": " + str(error)
    failure = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 140,
        "protocol_sha256": signature,
        "rows": [
            {
                "study": "peak_age",
                "candidate": candidate,
                "control": control,
                "horizon": horizon,
                "score": "mse",
                "phases": [],
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "status": "INVALID_RUN",
                "verdict": "UNEVALUABLE",
                "error": message,
            }
            for candidate, control, horizon in COMPARISONS
        ],
    }
    payload = (json.dumps(failure, indent=2, allow_nan=False) + "\n").encode()
    _atomic_report_write(root, "metrics.json", payload)
    _atomic_report_write(root, "failure.json", payload)
    _atomic_report_write(
        root,
        "verification.json",
        (
            json.dumps(
                {"status": "FAILED", "error": message, "protocol_sha256": signature}, indent=2
            )
            + "\n"
        ).encode(),
    )
    _atomic_report_write(
        root,
        "results.md",
        (
            "# SPX trailing peak-age return forecasts\n\nUNEVALUABLE: independent verification failed. All three hypotheses remain p=1; no lead.\n\n"
            + message
            + "\n"
        ).encode(),
    )
    old_ledger = _regular_bytes(report / "trial_ledger.jsonl") or b""
    ledger = old_ledger + (b"\n" if old_ledger and not old_ledger.endswith(b"\n") else b"")
    ledger += (
        "".join(
            json.dumps(
                {"event": "verification_failed", **row}, sort_keys=True, allow_nan=False
            )
            + "\n"
            for row in failure["rows"]
        )
    ).encode()
    _atomic_report_write(root, "trial_ledger.jsonl", ledger)
    if (
        prior is not None
        and prior.get("status") != "UNEVALUABLE"
        and _regular_bytes(report / "unpublished_scored_metrics.json") is None
    ):
        _atomic_report_write(
            root,
            "unpublished_scored_metrics.json",
            (
                json.dumps(
                    {
                        "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                        "not_for_inherited_inference_or_promotion": True,
                        "scored_metrics": prior,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode(),
        )
    if (
        original is not None
        and prior is None
        and _regular_bytes(report / "unpublished_invalid_metrics.txt") is None
    ):
        _atomic_report_write(root, "unpublished_invalid_metrics.txt", original)
    return failure


def verify_with_failure_guard(root=ROOT):
    try:
        return verify(root)
    except Exception as error:
        invalidate_publication(root, error)
        raise


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False))
