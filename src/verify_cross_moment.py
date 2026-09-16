"""Independent direct residual-product scoring and read-only upstream admission.

No new scoring or runner implementation is imported. Prior verified model
reconstruction is reused only through read-only functions; the old writers
and failure guards are never called.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import verify_joint_risk as upstream
from .verify_international_volatility import same_tree
from .verify_iterative_signal_search import digest, holm, same
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("constant_matrix", "constant_correlation", "dynamic_correlation")
COMPARISONS = (
    ("dynamic_correlation", "constant_correlation"),
    ("dynamic_correlation", "constant_matrix"),
)
WAVE_ALPHA = 0.05 / (11 * 12)
SCALE = 1e-10
INPUT_COLUMNS = (
    "origin",
    "model",
    "horizon",
    "feature_cutoff_date",
    "target_end",
    "available_date",
    "y_qqq",
    "y_spx",
    "mu_qqq",
    "mu_spx",
    "h_qqq",
    "h_spx",
    "rho",
    "loss",
    "fit_origin",
    "fit_cutoff_date",
    "train_n",
    "train_last_target",
    "train_last_available",
    "phase",
)
PRODUCT_COLUMNS = ("realized_product", "forecast_product", "product_mse")
UPSTREAM_LIMITATIONS = (
    "QQQ ETF and SPX price-index archival daily return residual second moments; no measured high-frequency covariance or exact Nasdaq100-index claim.",
    "Shared diagonal or mean misspecification can reward dynamic dependence without true correlation predictability; current-fit training residual staging.",
    "The GK measurement gate protects risk-history predictors; zero and signed return targets remain valid. No execution or profit claim.",
    "Reused history and back-calculated early VIX9D inputs remain exploratory limitations.",
)


def _pins(root, expected):
    for name, signature in expected.items():
        path = Path(root) / name
        if not path.is_file() or digest(path) != signature:
            raise AssertionError("Pinned upstream artifact changed or missing: " + name)


def read_issued_snapshot(root, name, signature):
    payload = (Path(root) / name).read_bytes()
    if sha256(payload).hexdigest() != signature:
        raise AssertionError("Issued forecast bytes differ from their pre-scoring pin")
    return pd.read_parquet(BytesIO(payload))


def validate_upstream(root=ROOT):
    """Reconstruct the original VERIFIED record without writing any file."""
    root = Path(root)
    protocol_path = root / "cross_moment.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    info = protocol["upstream"]
    manifest_path = root / "reports/cross_moment/manifest.json"
    captured = json.loads(manifest_path.read_text())
    if captured["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("New registration must be pinned before upstream admission")
    report, data = root / info["reports"], root / info["data"]
    old_protocol_path = root / info["protocol"]
    original_manifest_path, original_verification_path = (
        report / "manifest.json",
        report / "verification.json",
    )
    required = {
        str(path.relative_to(root))
        for directory in (report, data)
        for path in directory.rglob("*")
        if path.is_file()
    }
    required.add(info["protocol"])
    for name in (
        "source_audit.json",
        "measurement_audit.json",
        "features.parquet",
        "targets.parquet",
        "forecasts.parquet",
        "fits.json",
    ):
        if not (data / name).is_file():
            raise AssertionError("All six issued upstream output artifacts required")
    for name in ("manifest.json", "verification.json", "metrics.json", "trial_ledger.jsonl"):
        if not (report / name).is_file():
            raise AssertionError("Complete original upstream publication required")
    if not required.issubset(captured["inputs"]):
        raise AssertionError(
            "Every upstream protocol, report and private artifact must be pinned before admission"
        )
    _pins(root, captured["inputs"])
    if (
        (report / "failure.json").exists()
        or info["required_status"] != "VERIFIED"
        or digest(old_protocol_path) != info["protocol_sha256"]
        or digest(original_manifest_path) != info["manifest_sha256"]
        or digest(original_verification_path) != info["verification_sha256"]
    ):
        raise AssertionError("Original successful upstream registration identity required")
    old_protocol = yaml.safe_load(old_protocol_path.read_text())
    old_manifest = json.loads(original_manifest_path.read_text())
    original = json.loads(original_verification_path.read_text())
    json.dumps(original, allow_nan=False)
    if (
        original["status"] != "VERIFIED"
        or original["protocol_sha256"] != info["protocol_sha256"]
        or original["verifier_sha256"] != info["verifier_sha256"]
        or old_manifest["protocol_sha256"] != info["protocol_sha256"]
        or old_manifest["code"]["src/verify_joint_risk.py"] != info["verifier_sha256"]
    ):
        raise AssertionError(
            "Original VERIFIED marker, protocol and independent verifier identity disagree"
        )
    upstream.validate_protocol(old_protocol)
    upstream.verify_manifest_coverage(root, old_protocol, old_manifest)
    if not set(old_manifest["inputs"]).issubset(captured["inputs"]):
        raise AssertionError("All original raw input pins required in new admission")
    frozen = {}
    for group in ("code", "inputs", "preserved"):
        _pins(root, old_manifest[group])
        for name, signature in old_manifest[group].items():
            if name in frozen and frozen[name] != signature:
                raise AssertionError("Conflicting original manifest pins")
            frozen[name] = signature
    qqq, spx, iv, source_audit = upstream.load_source_tables(root, old_protocol)
    same_tree(
        json.loads((data / "source_audit.json").read_text()),
        source_audit,
        "Read-only original source reconstruction",
    )
    measured = upstream.measurement_audit(qqq, spx)
    upstream.require_measurement(measured)
    same_tree(
        json.loads((data / "measurement_audit.json").read_text()),
        measured,
        "Original whole-reference measurement gate",
    )
    features, targets = upstream.feature_target_tables(qqq, spx, iv)
    for name, expected, numeric in (
        ("features", features, upstream.ALL_FEATURES),
        ("targets", targets, upstream.TARGETS),
    ):
        actual = pd.read_parquet(data / (name + ".parquet"))
        if not actual.index.equals(expected.index) or tuple(actual.columns) != tuple(
            expected.columns
        ):
            raise AssertionError("Original full-table schema/calendar differs")
        same(
            actual.loc[:, numeric],
            expected.loc[:, numeric],
            "Original exact raw feature/target reconstruction",
            rtol=1e-10,
            atol=1e-12,
        )
        for column in set(expected) - set(numeric):
            if not actual[column].equals(expected[column]):
                raise AssertionError("Original source/target maturity differs")
    issued = pd.read_parquet(data / "forecasts.parquet")
    fits = json.loads((data / "fits.json").read_text())
    metrics = json.loads((report / "metrics.json").read_text())
    if (
        metrics["protocol_sha256"] != info["protocol_sha256"]
        or metrics.get("status") == "UNEVALUABLE"
    ):
        raise AssertionError("Original scoring outcome is not an admissible verified study")
    reconstruction = upstream.verify_forecasts(features, targets, issued, fits, old_protocol)
    for key, expected_key in (
        ("forecasts_verified", "forecasts_expected"),
        ("common_scored_origins", "scored_origins_expected"),
        ("monthly_fits_verified", "monthly_fits_expected"),
    ):
        if reconstruction[key] != info[expected_key]:
            raise AssertionError("Entire issued cohort must match its predeclared size")
    rebuilt = {
        "status": "VERIFIED",
        "raw_feature_rows_verified": len(features),
        "raw_feature_columns_verified": len(upstream.ALL_FEATURES),
        "measurement_audit": measured,
        "forecast_reconstruction": reconstruction,
        "inference": upstream.verify_metrics(root, issued, old_protocol, metrics),
        "ledger_events_verified": upstream.verify_ledger(
            root, metrics, metrics["inherited_rows"], "evaluated"
        ),
        "protocol_sha256": info["protocol_sha256"],
        "verifier_sha256": info["verifier_sha256"],
        "limitations": list(UPSTREAM_LIMITATIONS),
    }
    same_tree(
        original, rebuilt, "Original complete VERIFIED record reconstructed without writer"
    )
    _pins(root, frozen)
    _pins(root, captured["inputs"])
    final_required = {
        str(path.relative_to(root))
        for directory in (report, data)
        for path in directory.rglob("*")
        if path.is_file()
    }
    final_required.add(info["protocol"])
    if final_required != required:
        raise AssertionError("Upstream artifact inventory changed during admission")
    if digest(protocol_path) != captured["protocol_sha256"]:
        raise AssertionError("New registration changed during read-only admission")
    return {
        "status": "UPSTREAM_VERIFIED_READ_ONLY",
        "upstream_protocol_sha256": info["protocol_sha256"],
        "upstream_verification_sha256": info["verification_sha256"],
        "upstream_manifest_sha256": info["manifest_sha256"],
        "upstream_manifest_entries_verified": sum(
            len(old_manifest[group]) for group in ("code", "inputs", "preserved")
        ),
        "pinned_upstream_artifacts": len(required),
        "input_hashes": dict(captured["inputs"]),
        "reconstructed_verification": rebuilt,
        "new_model_fits": 0,
        "prior_files_written": False,
    }


def checked_product(left, right):
    if np.iscomplexobj(left) or np.iscomplexobj(right):
        raise ValueError("Real product inputs required")
    left, right = float(left), float(right)
    product = left * right
    if (
        not math.isfinite(left)
        or not math.isfinite(right)
        or not math.isfinite(product)
        or (left != 0 and right != 0 and product == 0)
    ):
        raise ValueError("Nonfinite product or nonzero inputs underflowing to zero")
    return product


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite arithmetic intermediate")
    return value


def unit(value):
    value = abs(float(value))
    if not math.isfinite(value):
        raise ValueError("Finite numerical comparison scale required")
    return max(np.finfo(float).eps * value, value - float(np.nextafter(value, 0.0)))


def primitive_equal(actual, expected, label):
    if np.iscomplexobj(actual) or np.iscomplexobj(expected):
        raise AssertionError(label + ": real primitive values required")
    a, e = np.asarray(actual, float), np.asarray(expected, float)
    if a.shape != e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
        raise AssertionError(label + ": finite aligned values required")
    if not np.array_equal(a == 0, e == 0) or not np.array_equal(np.sign(a), np.sign(e)):
        raise AssertionError(label + ": exact zero/sign masks differ")
    for left, right in zip(a.flat, e.flat):
        if abs(float(left) - float(right)) > 256 * unit(right):
            raise AssertionError(label + ": declared relative/ULP tolerance exceeded")


def score_panel(panel):
    if (
        tuple(panel.columns) != INPUT_COLUMNS
        or panel.empty
        or panel.duplicated(["origin", "model", "horizon"]).any()
    ):
        raise AssertionError("Exact entire issued panel schema and unique rows required")
    if set(panel.model) != set(MODELS) or set(panel.horizon) != {1}:
        raise AssertionError("All three issued one-session arms required")
    numeric = [
        "y_qqq",
        "y_spx",
        "mu_qqq",
        "mu_spx",
        "h_qqq",
        "h_spx",
        "rho",
        "loss",
        "train_n",
    ]
    if any(np.iscomplexobj(panel[column]) for column in numeric):
        raise AssertionError("Real issued numbers required")
    if not np.isfinite(panel[numeric]).all(axis=None) or (panel[["h_qqq", "h_spx"]] <= 0).any(
        axis=None
    ):
        raise AssertionError("Finite issued numbers and positive diagonals required")
    bounds = np.where(panel.model == "constant_matrix", 1 - 1e-6, 0.995)
    if (abs(panel.rho.to_numpy()) > bounds).any():
        raise AssertionError("Original normalized correlation bound exceeded")
    date_columns = (
        "origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "fit_origin",
        "fit_cutoff_date",
        "train_last_target",
        "train_last_available",
    )
    if any(
        not pd.api.types.is_datetime64_any_dtype(panel[column]) or panel[column].isna().any()
        for column in date_columns
    ):
        raise AssertionError(
            "Every issued source, application and maturity date must be defined"
        )
    aligned = {
        name: panel.loc[panel.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    first = aligned["constant_correlation"]
    equal = [
        "y_qqq",
        "y_spx",
        "mu_qqq",
        "mu_spx",
        "horizon",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "fit_origin",
        "fit_cutoff_date",
        "train_n",
        "train_last_target",
        "train_last_available",
        "phase",
    ]
    for name, rows in aligned.items():
        if not rows.index.equals(first.index) or any(
            not rows[field].equals(first[field]) for field in equal
        ):
            raise AssertionError("Exact same issued targets/means/cohort/fit timing required")
    for field in ("h_qqq", "h_spx"):
        if not aligned["dynamic_correlation"][field].equals(first[field]):
            raise AssertionError("Conditional issued diagonals must remain identical")
    if not panel.fit_origin.dt.to_period("M").equals(panel.origin.dt.to_period("M")):
        raise AssertionError("Issued fit must belong to its application month")
    fields = [
        "fit_origin",
        "fit_cutoff_date",
        "train_n",
        "train_last_target",
        "train_last_available",
    ]
    if any(
        group[fields].nunique(dropna=False).gt(1).any()
        for _, group in first.groupby(first.index.to_period("M"))
    ):
        raise AssertionError(
            "Exactly one fixed issued training fit per application month required"
        )
    dates = panel.origin
    if (
        (dates < "2016-01-04") | (dates > "2025-10-17") | (panel.target_end > "2025-10-20")
    ).any():
        raise AssertionError("Original origin or target fence crossed")
    if (
        not panel.available_date.equals(panel.target_end)
        or (panel.feature_cutoff_date >= dates).any()
        or (panel.target_end <= dates).any()
        or (panel.fit_origin > dates).any()
        or (panel.fit_cutoff_date >= panel.fit_origin).any()
        or (panel.fit_cutoff_date > panel.feature_cutoff_date).any()
        or (panel.train_last_available > panel.fit_cutoff_date).any()
        or (panel.train_last_target > panel.fit_cutoff_date).any()
        or (panel.train_n < 1000).any()
    ):
        raise AssertionError("Issued causal timing or training maturity differs")
    phase = np.where(dates <= "2019-12-31", "development", "evaluation")
    if (
        not np.array_equal(panel.phase, phase)
        or ((panel.phase == "development") & (panel.available_date > "2019-12-31")).any()
    ):
        raise AssertionError("Original phase maturity policy differs")
    derived = []
    for row in panel.itertuples(index=False):
        e1, e2 = _finite(row.y_qqq - row.mu_qqq), _finite(row.y_spx - row.mu_spx)
        target = checked_product(e1, e2)
        geometric = checked_product(math.sqrt(row.h_qqq), math.sqrt(row.h_spx))
        forecast = checked_product(geometric, row.rho)
        error = _finite(target - forecast)
        derived.append((target, forecast, checked_product(error, error)))
    result = panel.copy()
    result[list(PRODUCT_COLUMNS)] = np.asarray(derived)
    return result


def paired_difference(candidate, control, target):
    if any(np.iscomplexobj(value) for value in (candidate, control, target)):
        raise ValueError("Real one-dimensional product inputs required")
    first, second, y = (np.asarray(value, float) for value in (candidate, control, target))
    if (
        first.ndim != 1
        or first.shape != second.shape
        or first.shape != y.shape
        or not np.isfinite([first, second, y]).all()
    ):
        raise ValueError(
            "Finite aligned one-dimensional product forecasts and target required"
        )
    answer = []
    for one, zero, actual in zip(first, second, y):
        one, zero, actual = float(one), float(zero), float(actual)
        a, b = _finite(one - actual), _finite(zero - actual)
        loss1, loss0 = checked_product(a, a), checked_product(b, b)
        value = 0.0 if one == zero else checked_product(_finite(one - zero), _finite(a + b))
        direct = _finite(loss1 - loss0)
        tolerance = sum(64 * unit(term) for term in (loss1, loss0, value))
        if not math.isfinite(tolerance) or abs(value - direct) > tolerance:
            raise ValueError("Factored and direct squared-loss difference disagree")
        answer.append(value)
    return np.asarray(answer)


def validate_protocol(protocol):
    """Literal statistical/numerical contract, independent of the new runner."""
    if protocol["wave"] != 11 or protocol["wave_alpha"] != WAVE_ALPHA:
        raise AssertionError("Wave11 alpha allocation changed")
    fixed = {
        "upstream": {
            "protocol": "joint_risk.yaml",
            "reports": "reports/joint_risk",
            "data": "data/joint_risk",
            "forecasts": "data/joint_risk/forecasts.parquet",
            "features": "data/joint_risk/features.parquet",
            "protocol_sha256": "358d863853d79c876476e67bcf91e4c7fc855f2f107c8df58ede074f19a809ef",
            "verifier_sha256": "ea4ac89c0d345506bb45ea1dd5add092497edeabdcfa83995e2e596f8596b4a4",
            "manifest_sha256": "c80e77d076cefd98c0e6d7df6c9dd213638ec486e718a9713055552575dc7848",
            "verification_sha256": "d62544e15e9575371a95fd7f1d5fdcb5e034cc49f7e78268a50ae5ffc5d3c3e3",
            "required_status": "VERIFIED",
            "forecasts_expected": 7386,
            "scored_origins_expected": 2462,
            "monthly_fits_expected": 118,
        },
        "index": {
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
        },
        "scoring": {
            "effect_threshold_absolute": SCALE,
            "new_model_fits": 0,
            "conditional_rho_max": 0.995,
            "constant_matrix_rho_max": 1 - 1e-6,
            "coherence_eps_multiplier": 64,
        },
        "comparisons": {
            "new_hypotheses": 2,
            "inherited_hypotheses": 112,
            "cumulative_hypotheses": 114,
            "controls": ["constant_correlation", "constant_matrix"],
            "contrasts": [
                [candidate, control, "product_mse"] for candidate, control in COMPARISONS
            ],
        },
        "inference": {
            "blocks": [21, 63, 126],
            "hac_lags": 126,
            "minimum_phase_observations": 127,
            "bootstrap_draws": 99999,
            "seed": 20260917,
            "inference_scale": SCALE,
        },
        "verification": {
            "product_comparison_eps_multiplier": 256,
            "coherence_eps_multiplier": 64,
            "dimensionless_relative_tolerance": 1e-9,
            "probability_absolute_tolerance": 1e-14,
        },
        "outputs": {
            "data": "data/cross_moment",
            "reports": "reports/cross_moment",
            "scores": "data/cross_moment/scored_forecasts.parquet",
            "admission": "data/cross_moment/upstream_admission.json",
        },
    }
    for group, values in fixed.items():
        for key, expected in values.items():
            if protocol[group][key] != expected:
                raise AssertionError(
                    "Fixed cross-moment contract changed: " + group + "." + key
                )


def normalized_difference(difference):
    if np.iscomplexobj(difference):
        raise ValueError("Real differences required")
    difference = np.asarray(difference, float)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        normalized = difference / SCALE
        mean = normalized.mean()
        centered_ss = np.sum((normalized - mean) ** 2)
    if (
        difference.ndim != 1
        or not len(difference)
        or not np.isfinite(normalized).all()
        or not np.isfinite(mean)
        or not np.isfinite(centered_ss)
        or (np.any(normalized != normalized[0]) and centered_ss == 0)
    ):
        raise ValueError("Normalized mean or centered sum of squares is not representable")
    return normalized


def compare_tree(actual, expected, label, key=""):
    """No unit absolute floor for dimensional diagnostics, strict probabilities."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(label + ": mapping schema differs")
        for name in expected:
            compare_tree(actual[name], expected[name], label + "." + name, name)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(label + ": sequence schema differs")
        for number, value in enumerate(expected):
            compare_tree(actual[number], value, label + f"[{number}]", key)
    elif isinstance(expected, (float, np.floating)):
        left, right = float(actual), float(expected)
        tolerance = (
            1e-14
            if key == "p" or key.startswith("p_")
            else 1e-9 * abs(right) + 256 * unit(right)
        )
        if (
            not math.isfinite(left)
            or not math.isfinite(right)
            or abs(left - right) > tolerance
        ):
            raise AssertionError(label + ": independent finite numeric comparison differs")
    elif actual != expected:
        raise AssertionError(label + ": exact value differs")


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
    rows, benchmark = groups["dynamic_correlation"], groups[control]
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


def effect_passes(phases):
    return (
        len(phases) == 2
        and all(phase["delta"] <= -SCALE for phase in phases)
        and len(phases[1]["stability"]) == 2
        and all(row["delta"] < 0 for row in phases[1]["stability"])
    )


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
    if len(output) != 112:
        raise AssertionError("All112 inherited comparisons must remain identified")
    return output


def verify_metrics(root, panel, protocol, metrics):
    rows = metrics["rows"]
    if (
        metrics["new_model_fits"] != 0
        or metrics["new_forecasts"] != 0
        or metrics["scored_issued_forecast_rows"] != len(panel)
        or metrics["common_scored_origins"] != panel.origin.nunique()
    ):
        raise AssertionError(
            "Only the entire issued cohort may receive new scores; no new fits or forecasts"
        )
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError(
            "Both registered cross-moment comparisons required in fixed order"
        )
    probabilities, effects = [], []
    for row in rows:
        if (
            row["study"] != "cross_moment"
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
        holm([row["p_conservative"] for row in prior] + probabilities)[-2:],
    )
    passed = []
    for number, row in enumerate(rows):
        compare_tree(row["p_holm_wave"], wave[number], "Two-comparison wave Holm", "p")
        compare_tree(
            row["p_holm_cumulative"], cumulative[number], "114-comparison cumulative Holm", "p"
        )
        eligible = bool(
            effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05
        )
        if row["verdict"] != ("COMPARISON_GATE_PASS" if eligible else "DOES_NOT_QUALIFY"):
            raise AssertionError("Cross-moment statistical or fixed effect gate differs")
        passed.append(eligible)
    leads = ["dynamic_correlation"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 2
        or metrics["cumulative_hypothesis_count"] != 114
    ):
        raise AssertionError("One joint-return increment must pass both controls")
    return {
        "new_hypotheses_verified": 2,
        "cumulative_hypotheses_verified": 114,
        "phase_comparisons_verified": 4,
        "bootstrap_runs_verified": 12,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "leads": leads,
    }


def verify_ledger(root, metrics, prior, final_event):
    report = Path(root) / "reports/cross_moment"
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
        len(ledger) != 116
        or len(registered) != 2
        or len(prior) != 112
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "cross_moment"
            or row["horizon"] != 1
            or row["score"] != "product_mse"
            for row in registered
        )
    ):
        raise AssertionError("Complete112inherited+2registered+2terminal ledger required")
    return {"inherited": 112, "registered": 2, final_event: 2}


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    old = json.loads((root / protocol["upstream"]["reports"] / "manifest.json").read_text())
    required_code = {
        str(path.relative_to(root))
        for directory in ("src", "tests")
        for path in (root / directory).rglob("*.py")
    }
    required_code.update(old["code"])
    required_code.update(
        (
            "src/cross_moment_score.py",
            "src/cross_moment_search.py",
            "src/verify_cross_moment.py",
            "tests/test_cross_moment_score.py",
            "tests/test_cross_moment_search.py",
            "tests/test_cross_moment_publication.py",
            "tests/test_verify_cross_moment.py",
        )
    )
    required_inputs = set(old["inputs"]) | {protocol["upstream"]["protocol"]}
    for directory in (protocol["upstream"]["reports"], protocol["upstream"]["data"]):
        required_inputs.update(
            str(path.relative_to(root))
            for path in (root / directory).rglob("*")
            if path.is_file()
        )
    required_preserved = {
        str(path.relative_to(root))
        for path in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if path.is_file()
        and path != root / "cross_moment.yaml"
        and root / "reports/cross_moment" not in path.parents
        and str(path.relative_to(root)) not in required_inputs
    }
    if (
        set(manifest["code"]) != required_code
        or set(manifest["inputs"]) != required_inputs
        or set(manifest["preserved"]) != required_preserved
    ):
        raise AssertionError(
            "Entire source/test, upstream/raw input, and prior publication manifest coverage required"
        )
    if set(manifest["inputs"]) & set(manifest["preserved"]):
        raise AssertionError("Upstream inputs cannot also be counted as preserved artifacts")


def verify(root=ROOT):
    root = Path(root)
    protocol_path = root / "cross_moment.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    validate_protocol(protocol)
    report, data = root / "reports/cross_moment", root / "data/cross_moment"
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("New frozen protocol identity differs")
    verify_manifest_coverage(root, protocol, manifest)
    for group in ("code", "inputs", "preserved"):
        _pins(root, manifest[group])
    proof = validate_upstream(root)
    same_tree(
        json.loads((data / "upstream_admission.json").read_text()),
        proof,
        "Full read-only admission evidence",
    )
    issued_path = protocol["upstream"]["forecasts"]
    issued = read_issued_snapshot(root, issued_path, manifest["inputs"][issued_path])
    expected = score_panel(issued)
    scored = pd.read_parquet(data / "scored_forecasts.parquet")
    if (
        tuple(scored.columns) != INPUT_COLUMNS + PRODUCT_COLUMNS
        or not scored.index.equals(issued.index)
        or not scored.loc[:, INPUT_COLUMNS].equals(issued)
    ):
        raise AssertionError("All exact issued rows, values and schema must remain unchanged")
    for column in PRODUCT_COLUMNS:
        primitive_equal(scored[column], expected[column], "Independent " + column)
    if (
        len(scored) != protocol["upstream"]["forecasts_expected"]
        or scored.origin.nunique() != protocol["upstream"]["scored_origins_expected"]
    ):
        raise AssertionError("Full fixed issued cohort required; no new score filtering")
    metrics = json.loads((report / "metrics.json").read_text())
    json.dumps(metrics, allow_nan=False)
    if (
        (report / "failure.json").exists()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != manifest["protocol_sha256"]
        or metrics["evidence_class"] != protocol["evidence_class"]
        or metrics["new_model_fits"] != 0
    ):
        raise AssertionError("New scoring failed or its registered identity differs")
    result = {
        "status": "VERIFIED",
        "issued_forecasts_verified": len(issued),
        "common_scored_origins": int(issued.origin.nunique()),
        "primitive_score_cells_verified": len(issued) * 3,
        "paired_product_differences_verified": int(issued.origin.nunique()) * 2,
        "upstream_admission": {"status": proof["status"], "prior_files_written": False},
        "new_model_fits": 0,
        "new_forecasts": 0,
        "inference": verify_metrics(root, expected, protocol, metrics),
        "ledger_events_verified": verify_ledger(
            root, metrics, metrics["inherited_rows"], "evaluated"
        ),
        "protocol_sha256": manifest["protocol_sha256"],
        "verifier_sha256": digest(Path(__file__)),
        "limitations": [
            "Same frozen issued QQQ ETF and SPX price-index forecasts on reused history; this is a new score, not new predictive information or new model fitting.",
            "Signed daily residual products are noisy cross-moment proxies affected by shared mean error and issued marginal second moments; no measured high-frequency covariance claim.",
            "Fixed decimal-return units and absolute effect threshold; nominal MDE is diagnostic and nonqualification is not equivalence or proof of no predictability.",
            "The original joint-score verdict and all upstream source, model and publication artifacts remain unchanged; no profit claim.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        _pins(root, manifest[group])
    verify_manifest_coverage(root, protocol, manifest)
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Registration changed during verification")
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def invalidate_publication(root, error):
    report = Path(root) / "reports/cross_moment"
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
        "cumulative_hypothesis_count": 114,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "cross_moment",
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
        "# Direct QQQ–SPX residual cross-moment prediction\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n"
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
