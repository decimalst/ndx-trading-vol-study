"""Independent fixed-marginal cross-moment fit and publication verification.

Only frozen read-only verifiers are reused. No new producer, scorer or runner
is imported, and no earlier writing or invalidation entrypoint is called.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import verify_cross_moment as previous
from . import verify_joint_risk as original
from .verify_cross_moment import (
    _finite,
    compare_tree,
    digest,
    effect_passes,
    explicit_bootstrap_means,
    holm,
    independent_hac,
    normalized_difference,
    paired_difference,
    same_tree,
)

ALL_FEATURES = original.ALL_FEATURES
MARGINAL_COLUMNS = original.MARGINAL_COLUMNS
CAP = 0.995
MODELS = (
    "constant_matrix",
    "constant_correlation",
    "dynamic_correlation",
    "aligned_constant",
    "aligned_dynamic",
)
COMPARISONS = (
    ("aligned_dynamic", "aligned_constant"),
    ("aligned_dynamic", "constant_matrix"),
    ("aligned_dynamic", "dynamic_correlation"),
)
SCALE = 1e-10
WAVE_ALPHA = 0.05 / (12 * 13)
ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_LIMITATIONS = [
    "Same frozen issued QQQ ETF and SPX price-index forecasts on reused history; this is a new score, not new predictive information or new model fitting.",
    "Signed daily residual products are noisy cross-moment proxies affected by shared mean error and issued marginal second moments; no measured high-frequency covariance claim.",
    "Fixed decimal-return units and absolute effect threshold; nominal MDE is diagnostic and nonqualification is not equivalence or proof of no predictability.",
    "The original joint-score verdict and all upstream source, model and publication artifacts remain unchanged; no profit claim.",
]


def validate_upstream(root=ROOT):
    root = Path(root)
    protocol_path = root / "target_aligned.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    info = protocol["upstream"]
    captured = json.loads((root / "reports/target_aligned/manifest.json").read_text())
    if previous.digest(protocol_path) != captured["protocol_sha256"]:
        raise AssertionError("New manifest must precede chained upstream admission")
    oldreport, olddata = root / info["reports"], root / info["data"]
    required = {
        str(path.relative_to(root))
        for folder in (oldreport, olddata)
        for path in folder.rglob("*")
        if path.is_file()
    }
    required.add(info["protocol"])
    if not required.issubset(captured["inputs"]) or (oldreport / "failure.json").exists():
        raise AssertionError(
            "Every original successful publication and private output must be pinned"
        )
    for filename, key in (
        (root / info["protocol"], "protocol_sha256"),
        (oldreport / "manifest.json", "manifest_sha256"),
        (oldreport / "verification.json", "verification_sha256"),
    ):
        if previous.digest(filename) != info[key]:
            raise AssertionError("Original wave11 anchor identity differs")
    oldp = yaml.safe_load((root / info["protocol"]).read_text())
    oldm = json.loads((oldreport / "manifest.json").read_text())
    oldrecord = json.loads((oldreport / "verification.json").read_text())
    json.dumps(oldrecord, allow_nan=False)
    if (
        info["required_status"] != "VERIFIED"
        or oldrecord["status"] != "VERIFIED"
        or oldrecord["protocol_sha256"] != info["protocol_sha256"]
        or oldrecord["verifier_sha256"] != info["verifier_sha256"]
        or oldm["protocol_sha256"] != info["protocol_sha256"]
        or oldm["code"]["src/verify_cross_moment.py"] != info["verifier_sha256"]
    ):
        raise AssertionError("Original verifier/registration success identity differs")
    all_new = {}
    for group in ("code", "inputs", "preserved"):
        previous._pins(root, captured[group])
        for name, signature in captured[group].items():
            if name in all_new and all_new[name] != signature:
                raise AssertionError("Conflicting new manifest pins")
            all_new[name] = signature
    if not set(oldm["code"]).issubset(captured["code"]) or not set(oldm["inputs"]).issubset(
        captured["inputs"]
    ):
        raise AssertionError("All original code and input identities must remain pinned")
    for group in ("code", "inputs", "preserved"):
        previous._pins(root, oldm[group])
        if any(all_new.get(name) != signature for name, signature in oldm[group].items()):
            raise AssertionError(
                "An original preserved artifact is absent from the new registration"
            )
    previous.validate_protocol(oldp)
    nested = previous.validate_upstream(root)
    previous.same_tree(
        json.loads((olddata / "upstream_admission.json").read_text()),
        nested,
        "Nested original read-only admission",
    )
    original_path = oldp["upstream"]["forecasts"]
    issued = previous.read_issued_snapshot(root, original_path, all_new[original_path])
    expected = previous.score_panel(issued)
    scored = previous.read_issued_snapshot(
        root, info["forecasts"], captured["inputs"][info["forecasts"]]
    )
    if tuple(
        scored.columns
    ) != previous.INPUT_COLUMNS + previous.PRODUCT_COLUMNS or not scored.loc[
        :, previous.INPUT_COLUMNS
    ].equals(issued):
        raise AssertionError("All old issued rows must remain exact")
    for column in previous.PRODUCT_COLUMNS:
        previous.primitive_equal(
            scored[column], expected[column], "Original product " + column
        )
    if (
        len(issued) != info["forecasts_expected"]
        or issued.origin.nunique() != info["scored_origins_expected"]
    ):
        raise AssertionError("Entire previous score cohort required")
    oldmetrics = json.loads((oldreport / "metrics.json").read_text())
    if (
        oldmetrics["protocol_sha256"] != info["protocol_sha256"]
        or oldmetrics["evidence_class"] != oldp["evidence_class"]
        or oldmetrics.get("status") == "UNEVALUABLE"
    ):
        raise AssertionError("Prior scored result is not admissible")
    rebuilt = {
        "status": "VERIFIED",
        "issued_forecasts_verified": len(issued),
        "common_scored_origins": int(issued.origin.nunique()),
        "primitive_score_cells_verified": len(issued) * 3,
        "paired_product_differences_verified": int(issued.origin.nunique()) * 2,
        "upstream_admission": {"status": nested["status"], "prior_files_written": False},
        "new_model_fits": 0,
        "new_forecasts": 0,
        "inference": previous.verify_metrics(root, expected, oldp, oldmetrics),
        "ledger_events_verified": previous.verify_ledger(
            root, oldmetrics, oldmetrics["inherited_rows"], "evaluated"
        ),
        "protocol_sha256": info["protocol_sha256"],
        "verifier_sha256": info["verifier_sha256"],
        "limitations": PREVIOUS_LIMITATIONS,
    }
    previous.compare_tree(oldrecord, rebuilt, "Entire original wave11 VERIFIED record")
    final_required = {
        str(path.relative_to(root))
        for folder in (oldreport, olddata)
        for path in folder.rglob("*")
        if path.is_file()
    }
    final_required.add(info["protocol"])
    if (
        final_required != required
        or previous.digest(protocol_path) != captured["protocol_sha256"]
    ):
        raise AssertionError("Admission inventory or protocol changed while reconstructing")
    for group in ("code", "inputs", "preserved"):
        previous._pins(root, captured[group])
    return {
        "status": "UPSTREAM_VERIFIED_READ_ONLY",
        "prior_files_written": False,
        "previous_manifest_entries_verified": sum(
            len(oldm[group]) for group in ("code", "inputs", "preserved")
        ),
        "pinned_previous_artifacts": len(required),
        "input_hashes": captured["inputs"],
        "original_verification": rebuilt,
        "nested_joint_risk_admission": nested,
        "upstream_protocol_sha256": info["protocol_sha256"],
        "upstream_manifest_sha256": info["manifest_sha256"],
        "upstream_verification_sha256": info["verification_sha256"],
    }


def validate_protocol(p):
    expected = {
        "upstream": {
            "protocol": "cross_moment.yaml",
            "reports": "reports/cross_moment",
            "data": "data/cross_moment",
            "forecasts": "data/cross_moment/scored_forecasts.parquet",
            "features": "data/joint_risk/features.parquet",
            "targets": "data/joint_risk/targets.parquet",
            "fits": "data/joint_risk/fits.json",
            "joint_protocol": "joint_risk.yaml",
            "protocol_sha256": "edeb72e2d7708c0a5227318d1ea54742050f9be0b761941ce598cdda28c26ed0",
            "verifier_sha256": "d1cb8c3fc69e44646359b839aac160acf6ed4be6cc043e2f2ca23f7c979430fa",
            "manifest_sha256": "28a3fa3975bd58789d6859850579a472671443937e4de641351eb8f8217c969a",
            "verification_sha256": "b3745d90db160e1102f2c3b987627add1d890958e38b52bb30499b77f5c13fa0",
            "required_status": "VERIFIED",
            "forecasts_expected": 7386,
            "scored_origins_expected": 2462,
            "monthly_fits_expected": 118,
        },
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
            "evaluation_stability": [
                ["2020-01-02", "2022-12-31"],
                ["2023-01-01", "2025-10-17"],
            ],
            "horizons": [1],
            "models": list(MODELS),
            "market_lag": 1,
            "minimum_train": 1000,
        },
        "scoring": {
            "effect_threshold_absolute": SCALE,
            "new_model_fits": 236,
            "conditional_rho_max": CAP,
            "constant_matrix_rho_max": 0.999999,
            "coherence_eps_multiplier": 64,
        },
        "comparisons": {
            "new_hypotheses": 3,
            "inherited_hypotheses": 114,
            "cumulative_hypotheses": 117,
            "controls": [control for _, control in COMPARISONS],
            "contrasts": [
                [candidate, control, "product_mse"] for candidate, control in COMPARISONS
            ],
        },
        "inference": {
            "blocks": [21, 63, 126],
            "hac_lags": 126,
            "minimum_phase_observations": 127,
            "bootstrap_draws": 99999,
            "seed": 20260918,
            "inference_scale": SCALE,
        },
        "verification": {
            "product_comparison_eps_multiplier": 256,
            "coherence_eps_multiplier": 64,
            "dimensionless_relative_tolerance": 1e-9,
            "probability_absolute_tolerance": 1e-14,
            "coefficient_relative_tolerance": 1e-10,
            "coefficient_absolute_tolerance": 1e-12,
            "fit_statistic_eps_multiplier": 256,
            "fit_gradient_eps_multiplier": 512,
            "marginal_replay_relative_tolerance": 1e-10,
            "marginal_replay_absolute_tolerance": 1e-12,
        },
        "fitting": {
            "new_models": ["aligned_constant", "aligned_dynamic"],
            "cap": CAP,
            "training_staging": "current_monthly_frozen_fit_training_residuals",
            "minimum_correlation_eigenvalue": 0.005,
            "new_monthly_fits_expected": 118,
            "new_scalar_fits_expected": 236,
            "new_forecasts_expected": 4924,
            "combined_forecast_rows_expected": 12310,
            "coefficient_bounds": {"a": [-CAP, CAP], "b": [-1.0, 1.0]},
            "gradient_eps_multiplier": 512,
        },
        "outputs": {
            "data": "data/target_aligned",
            "reports": "reports/target_aligned",
            "forecasts": "data/target_aligned/forecasts.parquet",
            "fits": "data/target_aligned/fits.json",
            "admission": "data/target_aligned/upstream_admission.json",
        },
    }
    if p["wave"] != 12 or p["wave_alpha"] != WAVE_ALPHA:
        raise AssertionError("Fixed wave12 alpha allocation differs")
    for group, values in expected.items():
        for key, value in values.items():
            if p[group][key] != value:
                raise AssertionError(
                    "Fixed target-alignment contract changed: " + group + "." + key
                )


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    info = protocol["upstream"]
    old = json.loads((root / info["reports"] / "manifest.json").read_text())
    code = {
        str(path.relative_to(root))
        for folder in ("src", "tests")
        for path in (root / folder).rglob("*.py")
    } | set(old["code"])
    inputs = set(old["inputs"]) | {info["protocol"]}
    for folder in (info["reports"], info["data"]):
        inputs.update(
            str(path.relative_to(root))
            for path in (root / folder).rglob("*")
            if path.is_file()
        )
    preserved = {
        str(path.relative_to(root))
        for path in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if path.is_file()
        and path != root / "target_aligned.yaml"
        and root / "reports/target_aligned" not in path.parents
        and str(path.relative_to(root)) not in inputs
    }
    if (
        set(manifest["code"]) != code
        or set(manifest["inputs"]) != inputs
        or set(manifest["preserved"]) != preserved
    ):
        raise AssertionError(
            "All original and new code, upstream bytes and prior publications must be frozen"
        )
    if set(manifest["inputs"]) & set(manifest["preserved"]):
        raise AssertionError(
            "Manifest inputs and preserved publications must not be counted twice"
        )


def _real(value):
    if np.iscomplexobj(value):
        raise ValueError("Real finite values required")
    result = np.asarray(value, float)
    if not np.isfinite(result).all():
        raise ValueError("Real finite values required")
    return result


def _mean(value):
    try:
        total = math.fsum(float(x) for x in value)
        result = total / len(value)
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError("Mean arithmetic is not representable") from error
    if not math.isfinite(result) or (total != 0 and result == 0):
        raise ValueError("Mean arithmetic is not representable")
    return result


def _multiply(left, right):
    return np.asarray(
        [previous.checked_product(a, b) for a, b in zip(left, right, strict=True)]
    )


def _divide(numerator, denominator):
    numerator, denominator = float(numerator), float(denominator)
    if not np.isfinite([numerator, denominator]).all() or denominator == 0:
        raise ValueError("Finite representable quotient required")
    result = numerator / denominator
    if not math.isfinite(result) or (numerator != 0 and result == 0):
        raise ValueError("Nonfinite quotient or nonzero quotient underflow")
    return result


def independent_scalar_fit(residual, diagonal, z):
    e, h, z = _real(residual), _real(diagonal), _real(z)
    if (
        e.ndim != 2
        or e.shape[1] != 2
        or h.shape != e.shape
        or z.shape != (len(e),)
        or len(e) < 2
        or (h <= 0).any()
    ):
        raise ValueError(
            "Finite aligned paired residuals, positive diagonals and state required"
        )
    d_raw = _multiply(np.sqrt(h[:, 0]), np.sqrt(h[:, 1]))
    y_raw = _multiply(e[:, 0], e[:, 1])
    common_unit = _mean(d_raw)
    if common_unit <= 0:
        raise ValueError("Strictly positive common training unit required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        d, y = d_raw / common_unit, y_raw / common_unit
    if not np.isfinite([d, y]).all() or np.any(d == 0) or np.any((y_raw != 0) & (y == 0)):
        raise ValueError("Nonfinite normalized value or nonzero input underflow")
    denominator = _mean(_multiply(d, d))
    numerator = _mean(_multiply(d, y))
    if denominator <= 0:
        raise ValueError("Positive representable constant norm required")
    unconstrained = _divide(numerator, denominator)
    if not math.isfinite(unconstrained):
        raise ValueError("Unconstrained coefficient arithmetic overflow")
    a = max(-CAP, min(CAP, unconstrained))
    headroom = CAP - abs(a)
    w = _multiply(_multiply(np.full(len(d), headroom), d), np.tanh(z))
    error = _real(y - _multiply(np.full(len(d), a), d))
    constant_objective = _mean(_multiply(error, error))
    flat = bool(np.all(w == 0))
    slope_denominator = _mean(_multiply(w, w))
    slope_numerator = _mean(_multiply(w, error))
    if flat:
        b = 0.0
        slope_unconstrained = None
    else:
        if slope_denominator <= 0:
            raise ValueError("Nonzero state regressor has unrepresentable squared norm")
        slope_unconstrained = _divide(slope_numerator, slope_denominator)
        if not math.isfinite(slope_unconstrained):
            raise ValueError("Slope coefficient arithmetic is not representable")
        b = max(-1.0, min(1.0, slope_unconstrained))
    candidate_error = _real(error - _multiply(np.full(len(w), b), w))
    candidate_objective = _mean(_multiply(candidate_error, candidate_error))
    return {
        "a": a,
        "b": b,
        "common_unit": common_unit,
        "headroom": headroom,
        "slope_status": "FLAT_OBJECTIVE" if flat else "IDENTIFIED",
        "train_n": len(e),
        "constant_numerator": numerator,
        "constant_denominator": denominator,
        "constant_unconstrained": unconstrained,
        "constant_objective": constant_objective,
        "slope_numerator": slope_numerator,
        "slope_denominator": slope_denominator,
        "slope_unconstrained": slope_unconstrained,
        "slope_objective": candidate_objective,
    }


def replay_moments(frame, audits):
    if tuple(frame.columns) != MARGINAL_COLUMNS:
        raise AssertionError("Exact frozen35-column marginal design required")
    x = _real(frame)
    mean, variance = audits["mean"], audits["variance"]
    if (
        tuple(mean["columns"]) != MARGINAL_COLUMNS
        or tuple(variance["columns"]) != MARGINAL_COLUMNS
    ):
        raise AssertionError("Saved coefficient column identity differs")
    means, scales, beta = map(_real, (mean["means"], mean["scales"], mean["beta"]))
    vmeans, vscales, vbeta = map(
        _real, (variance["means"], variance["scales"], variance["scaled_beta"])
    )
    if (
        means.shape != (x.shape[1],)
        or scales.shape != means.shape
        or beta.shape != means.shape
        or vmeans.shape != (x.shape[1] - 1,)
        or vscales.shape != vmeans.shape
        or vbeta.shape != means.shape
        or (scales <= 0).any()
        or (vscales <= 0).any()
        or not math.isfinite(variance["train_mean"])
        or variance["train_mean"] <= 0
    ):
        raise ValueError("Saved normalization and positive residual unit are invalid")
    design = (x - means) / scales
    vdesign = np.column_stack([np.ones(len(x)), (x[:, 1:] - vmeans) / vscales])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mu = design @ beta
        h = np.exp(np.log(variance["train_mean"]) + vdesign @ vbeta)
    if not np.isfinite([mu, h]).all() or (h <= 0).any():
        raise ValueError("Saved-coefficient marginal replay is nonfinite or nonpositive")
    return mu, h


def fit_statistic(actual, expected, label, *, exact_masks=True):
    if actual is None or expected is None:
        if actual != expected:
            raise AssertionError(label + ": optional scalar mismatch")
        return
    if exact_masks and (
        (actual == 0) != (expected == 0) or np.sign(actual) != np.sign(expected)
    ):
        raise AssertionError(label + ": exact primitive zero/sign mask differs")
    if (
        not math.isfinite(actual)
        or not math.isfinite(expected)
        or abs(actual - expected) > 256 * previous.unit(expected)
    ):
        raise AssertionError(label + ": independent fit statistic differs")


def coefficient(actual, expected, label):
    if not np.isfinite([actual, expected]).all() or abs(
        actual - expected
    ) > 1e-12 + 1e-10 * abs(expected):
        raise AssertionError(label + ": bounded optimum differs")


def gradient_equal(actual, expected, allowance, label):
    if (
        not np.isfinite([actual, expected, allowance]).all()
        or allowance < 0
        or abs(actual - expected) > allowance
    ):
        raise AssertionError(
            label + ": signed gradient exceeds fixed two-term roundoff allowance"
        )


def verify_scalar_audit(e, h, z, audit):
    expected = independent_scalar_fit(e, h, z)
    if set(audit) != {
        "train_n",
        "common_unit",
        "a",
        "b",
        "headroom",
        "slope_status",
        "constant",
        "dynamic",
    }:
        raise AssertionError("Complete fixed two-stage audit required")
    for key in ("a", "b"):
        coefficient(audit[key], expected[key], "Independent " + key)
    if (
        audit["train_n"] != len(e)
        or abs(audit["a"]) > CAP
        or abs(audit["b"]) > 1
        or audit["headroom"] != CAP - abs(audit["a"])
        or audit["slope_status"] != expected["slope_status"]
        or (audit["slope_status"] == "FLAT_OBJECTIVE" and audit["b"] != 0)
    ):
        raise AssertionError("Stage bounds, headroom or canonical flat status differ")
    fit_statistic(audit["common_unit"], expected["common_unit"], "Common unit")
    d_raw = _multiply(np.sqrt(h[:, 0]), np.sqrt(h[:, 1]))
    y_raw = _multiply(e[:, 0], e[:, 1])
    d, y = d_raw / expected["common_unit"], y_raw / expected["common_unit"]
    remaining = _real(y - _multiply(np.full(len(d), audit["a"]), d))
    w = _multiply(_multiply(np.full(len(d), audit["headroom"]), d), np.tanh(z))
    for name, regressor, response, bound, value in (
        ("constant", d, y, CAP, audit["a"]),
        ("dynamic", w, remaining, 1.0, audit["b"]),
    ):
        saved = audit[name]
        denominator = _mean(_multiply(regressor, regressor))
        numerator = _mean(_multiply(regressor, response))
        flat = name == "dynamic" and np.all(w == 0)
        ratio = None if flat else _divide(numerator, denominator)
        error = _real(response - _multiply(np.full(len(d), value), regressor))
        objective = _mean(_multiply(error, error))
        first = previous.checked_product(2.0, previous.checked_product(value, denominator))
        second = previous.checked_product(2.0, numerator)
        gradient = _finite(first - second)
        projected = (
            0.0
            if (value == -bound and gradient >= 0) or (value == bound and gradient <= 0)
            else gradient
        )
        tolerance = sum(512 * previous.unit(term) for term in (first, second))
        reconstructed = {
            "numerator": numerator,
            "denominator": denominator,
            "unconstrained_coefficient": ratio,
            "coefficient": value,
            "bounds": [-bound, bound],
            "objective": objective,
            "gradient": gradient,
            "projected_gradient": projected,
            "gradient_tolerance": tolerance,
        }
        if (
            set(saved) != set(reconstructed)
            or saved["bounds"] != [-bound, bound]
            or saved["coefficient"] != value
        ):
            raise AssertionError("Exact bounded scalar audit identity differs")
        for key in set(reconstructed) - {"bounds", "coefficient"}:
            if key in ("gradient", "projected_gradient"):
                gradient_equal(saved[key], reconstructed[key], tolerance, name + "." + key)
            else:
                fit_statistic(saved[key], reconstructed[key], name + "." + key)
        if (
            not math.isfinite(tolerance)
            or abs(projected) > tolerance
            or abs(saved["projected_gradient"]) > saved["gradient_tolerance"]
        ):
            raise AssertionError(
                "Scale-aware projected KKT residual exceeds fixed roundoff bound"
            )
    return {
        "scalar_fits_verified": 2,
        "flat_slope": expected["slope_status"] == "FLAT_OBJECTIVE",
    }


def score_panel(panel):
    columns = previous.INPUT_COLUMNS + previous.PRODUCT_COLUMNS
    if (
        tuple(panel.columns) != columns
        or set(panel.model) != set(MODELS)
        or panel.duplicated(["origin", "model", "horizon"]).any()
    ):
        raise AssertionError("Entire five-model issued schema required")
    old = panel.loc[panel.model.isin(original.MODELS)].reset_index(drop=True)
    checked_old = previous.score_panel(old.loc[:, previous.INPUT_COLUMNS])
    for column in previous.PRODUCT_COLUMNS:
        previous.primitive_equal(
            old[column], checked_old[column], "Original retained " + column
        )
    # Independent frozen cohort/timing checks apply to the two new conditional
    # arms after a temporary name mapping; no value is replaced or refitted.
    trio = panel.loc[
        panel.model.isin(["constant_matrix", "aligned_constant", "aligned_dynamic"])
    ].copy()
    trio["model"] = trio.model.replace(
        {"aligned_constant": "constant_correlation", "aligned_dynamic": "dynamic_correlation"}
    )
    checked = previous.score_panel(trio.loc[:, previous.INPUT_COLUMNS])
    result = panel.copy()
    for column in previous.PRODUCT_COLUMNS:
        previous.primitive_equal(trio[column], checked[column], "New conditional " + column)
        new_index = trio.index[trio.model != "constant_matrix"]
        result.loc[new_index, column] = checked.loc[new_index, column]
    base = old.loc[old.model == "constant_correlation"].set_index("origin").sort_index()
    fields = [
        column
        for column in columns
        if column not in ("origin", "model", "rho", "loss", "forecast_product", "product_mse")
    ]
    for name in ("aligned_constant", "aligned_dynamic"):
        rows = panel.loc[panel.model == name].set_index("origin").sort_index()
        if not rows.index.equals(base.index) or not rows.loc[:, fields].equals(
            base.loc[:, fields]
        ):
            raise AssertionError(
                "All exact issued labels, means, diagonals and timing must remain shared"
            )
        actual = rows[["y_qqq", "y_spx"]].to_numpy()
        mu = rows[["mu_qqq", "mu_spx"]].to_numpy()
        h = rows[["h_qqq", "h_spx"]].to_numpy()
        expected = original.matrix_score(actual - mu, h, rows.rho.to_numpy())
        previous.same(rows.loss, expected, "Auxiliary matrix field", rtol=1e-7, atol=1e-10)
    return result


def verify_forecasts(features, targets, issued, oldfits, panel, fits, protocol):
    if not features.index.equals(targets.index) or tuple(features.columns) != ALL_FEATURES + (
        "feature_cutoff_date",
    ):
        raise AssertionError("Original complete source feature/calendar schema required")
    checked = score_panel(panel)
    retained = checked.loc[checked.model.isin(original.MODELS)].reset_index(drop=True)
    if not retained.equals(issued.reset_index(drop=True)):
        raise AssertionError(
            "Every original23-column forecast row must remain exactly unchanged"
        )
    section = protocol["index"]
    applications, scored, _ = original.eligible_entries(features, targets, section)
    cohorts = {
        name: panel.loc[panel.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    if any(not rows.index.equals(scored) for rows in cohorts.values()):
        raise AssertionError("All five models require the complete original scoreable cohort")
    months = applications.to_period("M").unique()
    if len(fits) != len(months) or len(oldfits) != len(months):
        raise AssertionError(
            "All original monthly fits retained, including unscored applications"
        )
    flat = 0
    train_cells = 0
    for number, month in enumerate(months):
        app = applications[applications.to_period("M") == month]
        entry = app[0]
        mask = original.training_mask(features, targets, entry)
        if mask.sum() < section["minimum_train"]:
            raise AssertionError("Original minimum mature common training sample required")
        dates = features.index
        cutoff = dates[dates.get_loc(entry) - 1]
        metadata = {
            "fit_origin": str(entry.date()),
            "fit_cutoff_date": str(cutoff.date()),
            "train_n": int(mask.sum()),
            "train_first_origin": str(dates[mask][0].date()),
            "train_last_origin": str(dates[mask][-1].date()),
            "train_last_target": str(targets.loc[mask, "target_end"].max().date()),
            "train_last_available": str(targets.loc[mask, "available_date"].max().date()),
            "application_n": len(app),
        }
        old, saved = oldfits[number], fits[number]
        if (
            {key: value for key, value in old.items() if key != "model_audit"} != metadata
            or {
                key: value
                for key, value in saved.items()
                if key not in ("model_audit", "upstream_fit_index", "upstream_fit_sha256")
            }
            != metadata
            or saved["upstream_fit_index"] != number
            or saved["upstream_fit_sha256"]
            != sha256(json.dumps(old, sort_keys=True, allow_nan=False).encode()).hexdigest()
        ):
            raise AssertionError(
                "Original monthly training cohort, application schedule or saved-fit pin differs"
            )
        tr, ap, transformation = original.transform(features.loc[mask], features.loc[app])
        oldaudit = old["model_audit"]
        same_tree(oldaudit["transform"], transformation, "Original corr22 transform")
        scale = float(tr.corr22.std(ddof=0))
        if (
            oldaudit["residual_staging"] != "current_fit_training_residuals"
            or not scale > 1e-12
        ):
            raise AssertionError(
                "Fixed current-fit training residual convention and positive scale required"
            )
        previous.same(
            oldaudit["corr22_scale"], scale, "Original state scale", rtol=1e-10, atol=1e-12
        )
        combined = pd.concat([tr, ap])
        mus = []
        hs = []
        for asset in ("qqq", "spx"):
            audits = oldaudit["moments"][asset]
            for name in ("mean", "variance"):
                if audits[name]["train_n"] != mask.sum() or audits[name]["alpha"] != 0.01:
                    raise AssertionError(
                        "Saved marginal fit must use the same complete mature training rows"
                    )
            mu, h = replay_moments(combined, audits)
            mus.append(mu)
            hs.append(h)
        mu, h = np.column_stack(mus), np.column_stack(hs)
        residual = _real(targets.loc[mask, ["y_qqq", "y_spx"]].to_numpy() - mu[: mask.sum()])
        train_z = _real(
            (tr.corr22.to_numpy() - transformation["corr22_mean"]) / oldaudit["corr22_scale"]
        )
        application_z = _real(
            (ap.corr22.to_numpy() - transformation["corr22_mean"]) / oldaudit["corr22_scale"]
        )
        verdict = verify_scalar_audit(residual, h[: mask.sum()], train_z, saved["model_audit"])
        flat += int(verdict["flat_slope"])
        train_cells += int(mask.sum()) * 4
        keep = app.isin(scored)
        chosen = app[keep]
        if not len(chosen):
            continue
        base = cohorts["constant_correlation"].loc[chosen]
        previous.same(
            mu[mask.sum() :][keep],
            base[["mu_qqq", "mu_spx"]],
            "Replay exact issued means",
            rtol=1e-10,
            atol=1e-12,
        )
        previous.same(
            h[mask.sum() :][keep],
            base[["h_qqq", "h_spx"]],
            "Replay exact issued diagonals",
            rtol=1e-10,
            atol=1e-12,
        )
        a, b, headroom = (saved["model_audit"][key] for key in ("a", "b", "headroom"))
        increment = _multiply(
            np.full(len(app), previous.checked_product(b, headroom)), np.tanh(application_z)
        )
        dynamic = _real(a + increment)
        if (abs(dynamic) > CAP).any():
            raise AssertionError("Structural application correlation bound differs")
        for name, rho in (
            ("aligned_constant", np.full(len(app), a)),
            ("aligned_dynamic", dynamic),
        ):
            previous.primitive_equal(
                cohorts[name].loc[chosen, "rho"], rho[keep], "Issued bounded " + name
            )
    return {
        "monthly_fits_verified": len(fits),
        "scalar_fits_verified": 2 * len(fits),
        "flat_slope_fits_retained": flat,
        "marginal_training_cells_replayed": train_cells,
        "combined_forecasts_verified": len(panel),
        "original_forecasts_preserved": len(issued),
        "new_forecasts_verified": len(panel) - len(issued),
        "common_scored_origins": len(scored),
        "common_application_origins": len(applications),
    }


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
    rows, benchmark = groups["aligned_dynamic"], groups[control]
    if any(not part.index.equals(rows.index) for part in groups.values()) or len(rows) < 127:
        raise AssertionError(
            "Complete original cohort and at least127 phase observations required"
        )
    difference = paired_difference(
        rows.forecast_product, benchmark.forecast_product, rows.realized_product
    )
    values = normalized_difference(difference)
    mean = float(values.mean())
    delta = _finite(mean * SCALE)
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
            "ci95": [_finite(value * SCALE) for value in np.quantile(samples, [0.025, 0.975])],
        }
    hac = {
        "se": _finite(hac["se"] * SCALE),
        "p": _finite(hac["p"]),
        "ci95": [_finite(value * SCALE) for value in hac["ci95"]],
        "mde80_nominal": _finite(hac["mde80_nominal"] * SCALE),
    }
    intervals = [hac["ci95"], *(row["ci95"] for row in blocks.values())]
    result = {
        "name": phase,
        "first_origin": str(rows.index[0].date()),
        "last_origin": str(rows.index[-1].date()),
        "n": len(rows),
        "delta": delta,
        "candidate_loss": _finite(rows.product_mse.mean()),
        "control_loss": _finite(benchmark.product_mse.mean()),
        "block_inference": blocks,
        "hac126": hac,
        "ci95_envelope": [
            min(pair[0] for pair in intervals),
            max(pair[1] for pair in intervals),
        ],
        "p_conservative": max(hac["p"], *(row["p"] for row in blocks.values())),
        "nominal_mde_effect_ratio": _finite(hac["mde80_nominal"] / SCALE),
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
                "delta": _finite(values[mask].mean() * SCALE),
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
                    "delta": _finite(values[mask].mean() * SCALE),
                }
            )
    return result


def inherited_rows(root, protocol):
    output = []
    for source in protocol["comparisons"]["inherited_sources"]:
        for number, row in enumerate(json.loads((root / source).read_text())["rows"]):
            output.append(
                {
                    "study": row.get("study", Path(source).parent.name or Path(source).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": source,
                    "source_sha256": digest(root / source),
                    "source_row_index": number,
                    **{name: row[name] for name in ("measure", "score") if name in row},
                }
            )
    if len(output) != 114:
        raise AssertionError("All114 inherited comparisons must remain identified")
    return output


def verify_metrics(root, panel, protocol, metrics):
    rows = metrics["rows"]
    if (
        metrics["new_monthly_fits"] != protocol["fitting"]["new_monthly_fits_expected"]
        or metrics["new_scalar_fits"] != 2 * metrics["new_monthly_fits"]
        or metrics["new_forecasts"]
        != int(panel.model.isin(["aligned_constant", "aligned_dynamic"]).sum())
        or metrics["reused_forecasts"] != int(panel.model.isin(original.MODELS).sum())
        or metrics["combined_forecast_rows"] != len(panel)
        or metrics["common_scored_origins"] != panel.origin.nunique()
    ):
        raise AssertionError(
            "New scalar fits/forecasts must be counted separately from exact reused rows"
        )
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError(
            "All three registered cross-moment comparisons required in fixed order"
        )
    probabilities, effects = [], []
    for row in rows:
        if (
            row["study"] != "target_aligned"
            or row["horizon"] != 1
            or row["score"] != "product_mse"
        ):
            raise AssertionError("Direct residual cross-moment comparison identity differs")
        phases = [
            phase_statistics(panel, row["control"], phase, code, protocol)
            for code, phase in enumerate(("development", "evaluation"))
        ]
        compare_tree(row["phases"], phases, "Independent cross-moment phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        compare_tree(
            row["p_conservative"], probability, "Both-phase conjunction probability", "p"
        )
        probabilities.append(probability)
        effects.append(effect_passes(phases))
    prior = inherited_rows(root, protocol)
    same_tree(metrics["inherited_rows"], prior, "Complete cumulative trial identities")
    wave, cumulative = (
        holm(probabilities),
        holm([row["p_conservative"] for row in prior] + probabilities)[-3:],
    )
    passed = []
    for number, row in enumerate(rows):
        compare_tree(row["p_holm_wave"], wave[number], "Three-comparison wave Holm", "p")
        compare_tree(
            row["p_holm_cumulative"], cumulative[number], "117-comparison cumulative Holm", "p"
        )
        eligible = bool(
            effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05
        )
        if row["verdict"] != ("COMPARISON_GATE_PASS" if eligible else "DOES_NOT_QUALIFY"):
            raise AssertionError("Cross-moment statistical or fixed effect gate differs")
        passed.append(eligible)
    leads = ["aligned_dynamic"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 3
        or metrics["cumulative_hypothesis_count"] != 117
    ):
        raise AssertionError("One target-aligned increment must pass all three controls")
    return {
        "new_hypotheses_verified": 3,
        "cumulative_hypotheses_verified": 117,
        "phase_comparisons_verified": 6,
        "bootstrap_runs_verified": 18,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "leads": leads,
    }


def verify_ledger(root, metrics, prior, final_event):
    report = Path(root) / "reports/target_aligned"
    ledger = [
        json.loads(line)
        for line in (report / "trial_ledger.jsonl").read_text().splitlines()
        if line
    ]
    for event, expected in (("inherited", prior), (final_event, metrics["rows"])):
        actual = [
            {key: value for key, value in row.items() if key != "event"}
            for row in ledger
            if row["event"] == event
        ]
        compare_tree(actual, expected, "Independent " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (
        len(ledger) != 120
        or len(registered) != 3
        or len(prior) != 114
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "target_aligned"
            or row["horizon"] != 1
            or row["score"] != "product_mse"
            for row in registered
        )
    ):
        raise AssertionError("Complete114inherited+3registered+3terminal ledger required")
    return {"inherited": 114, "registered": 3, final_event: 3}


def read_json_snapshot(root, name, signature):
    payload = (Path(root) / name).read_bytes()
    if sha256(payload).hexdigest() != signature:
        raise AssertionError("Saved coefficient bytes changed before replay")
    result = json.loads(payload)
    json.dumps(result, allow_nan=False)
    return result


def verify(root=ROOT):
    root = Path(root)
    protocol_path = root / "target_aligned.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    validate_protocol(protocol)
    report, out = root / "reports/target_aligned", root / "data/target_aligned"
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Frozen new protocol changed")
    verify_manifest_coverage(root, protocol, manifest)
    for group in ("code", "inputs", "preserved"):
        previous._pins(root, manifest[group])
    proof = validate_upstream(root)
    same_tree(
        json.loads((out / "upstream_admission.json").read_text()),
        proof,
        "Complete chained read-only admission",
    )
    data = {}
    for key in ("features", "targets", "forecasts"):
        name = protocol["upstream"][key]
        data[key] = previous.read_issued_snapshot(root, name, manifest["inputs"][name])
    name = protocol["upstream"]["fits"]
    oldfits = read_json_snapshot(root, name, manifest["inputs"][name])
    panel = pd.read_parquet(out / "forecasts.parquet")
    fits = json.loads((out / "fits.json").read_text())
    json.dumps(fits, allow_nan=False)
    reconstruction = verify_forecasts(
        data["features"], data["targets"], data["forecasts"], oldfits, panel, fits, protocol
    )
    counts = protocol["fitting"]
    for actual, key in (
        (len(panel), "combined_forecast_rows_expected"),
        (len(fits), "new_monthly_fits_expected"),
        (2 * len(fits), "new_scalar_fits_expected"),
        (len(panel) - len(data["forecasts"]), "new_forecasts_expected"),
    ):
        if actual != counts[key]:
            raise AssertionError("Whole fixed model/fold/cohort count differs")
    if (
        len(data["forecasts"]) != protocol["upstream"]["forecasts_expected"]
        or panel.origin.nunique() != protocol["upstream"]["scored_origins_expected"]
    ):
        raise AssertionError("All original common score origins required")
    metrics = json.loads((report / "metrics.json").read_text())
    json.dumps(metrics, allow_nan=False)
    if (
        (report / "failure.json").exists()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != manifest["protocol_sha256"]
        or metrics["evidence_class"] != protocol["evidence_class"]
    ):
        raise AssertionError("New publication failed or its registered identity differs")
    result = {
        "status": "VERIFIED",
        "upstream_admission": {"status": proof["status"], "prior_files_written": False},
        "forecast_reconstruction": reconstruction,
        "primitive_score_cells_verified": len(panel) * 3,
        "paired_product_differences_verified": int(panel.origin.nunique()) * 3,
        "inference": verify_metrics(root, score_panel(panel), protocol, metrics),
        "ledger_events_verified": verify_ledger(
            root, metrics, metrics["inherited_rows"], "evaluated"
        ),
        "protocol_sha256": manifest["protocol_sha256"],
        "verifier_sha256": digest(Path(__file__)),
        "limitations": [
            "Two new scalar fits per original monthly fold; shared marginal coefficients replayed without refitting and all original issued rows preserved.",
            "Current-fit training residuals are in-sample at the monthly fit cutoff; only their conditional expected product equals covariance plus mean-error product.",
            "The staged bounded affine model changes both target alignment and dependence functional form, so success cannot isolate a loss-function change or establish true correlation dynamics.",
            "Reused archival QQQ ETF/SPX index daily products, no measured high-frequency covariance or trading-profit claim; nominal MDE is diagnostic and nonqualification is not equivalence.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        previous._pins(root, manifest[group])
    verify_manifest_coverage(root, protocol, manifest)
    if digest(protocol_path) != manifest["protocol_sha256"]:
        raise AssertionError("Protocol changed during verification")
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def invalidate_publication(root, error):
    report = Path(root) / "reports/target_aligned"
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
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 117,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "target_aligned",
                "candidate": candidate,
                "control": control,
                "score": "product_mse",
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
        "# Direct QQQ–SPX residual cross-moment prediction\n\nUNEVALUABLE: independent verification failed. All three comparisons retained with p=1; no lead.\n\n"
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


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False), flush=True)
