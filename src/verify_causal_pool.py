"""Independent full-calendar probability-pool verification.

Filter states use explicit weighted sums. Only frozen read-only verification
helpers are imported; no new producer, scorer or runner is used.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import verify_cross_moment as scalar
from . import verify_sign_memory as previous
from .verify_cross_moment import (
    _finite as frozen_finite,
)
from .verify_cross_moment import (
    compare_tree,
    digest,
    explicit_bootstrap_means,
    holm,
    independent_hac,
    paired_difference,
    same_tree,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("frozen_baseline", "recent_frequency", "pooled")
COMPARISONS = (("pooled", "frozen_baseline"), ("pooled", "recent_frequency"))
DELTA = 2 ** (-1 / 63)
EFFECT = 0.0005
WAVE_ALPHA = 0.05 / (14 * 15)
PANEL_COLUMNS = previous.PANEL_COLUMNS
STATE_COLUMNS = (
    "origin",
    "feature_cutoff_date",
    "source_fit_origin",
    "seed_fit_origin",
    "seed_cutoff_date",
    "seed_last_available",
    "seed_probability",
    "seed_train_n",
    "S",
    "W",
    "recent_frequency",
    "latest_consumed_available",
    "cumulative_updates",
    "elapsed_sessions",
    "scored",
)
multiply = scalar.checked_product


def _finite(value):
    if np.iscomplexobj(value):
        raise ValueError("Real scalar arithmetic required")
    return frozen_finite(value)


def read_snapshot(root, name, signature):
    payload = (Path(root) / name).read_bytes()
    if sha256(payload).hexdigest() != signature:
        raise AssertionError("Registered bytes changed before decoding: " + name)
    return payload


def read_json_snapshot(root, name, signature):
    result = json.loads(read_snapshot(root, name, signature))
    json.dumps(result, allow_nan=False)
    return result


def divide(numerator, denominator):
    numerator, denominator = _finite(numerator), _finite(denominator)
    if denominator <= 0:
        raise ValueError("Strictly positive denominator required")
    result = _finite(numerator / denominator)
    if numerator != 0 and result == 0:
        raise ValueError("Nonzero quotient underflowed")
    return result


def state_equal(actual, expected, elapsed, label):
    actual, expected = _finite(actual), _finite(expected)
    if (actual == 0) != (expected == 0) or np.sign(actual) != np.sign(expected):
        raise AssertionError(label + ": exact zero/sign masks differ")
    if not isinstance(elapsed, (int, np.integer)) or elapsed < 0:
        raise AssertionError("Nonnegative integer elapsed sessions required")
    bound = _finite(64 * (int(elapsed) + 1) * scalar.unit(expected))
    if abs(actual - expected) > bound:
        raise AssertionError(label + ": declared accumulated-roundoff bound exceeded")


def target_alignment(targets, reference):
    reference = pd.DatetimeIndex(reference)
    if (
        reference.has_duplicates
        or reference.hasnans
        or not reference.is_monotonic_increasing
        or reference.tz is not None
        or not reference.equals(reference.normalize())
        or not targets.index.equals(reference)
        or tuple(targets.columns) != ("y", "target_end", "available_date")
    ):
        raise ValueError(
            "Full ordered normalized reference calendar and exact target schema required"
        )
    maturity = pd.Series(reference, index=reference).shift(-1)
    for column in ("target_end", "available_date"):
        if not targets[column].equals(maturity.rename(column)):
            raise ValueError("Exact next-reference-session target availability required")
    finite = targets.y.notna()
    if finite.any():
        previous.binary(targets.loc[finite, "y"])
    if (finite & targets.available_date.isna()).any():
        raise ValueError("Known binary labels require availability dates")
    return reference


def explicit_states(
    targets, reference, seed_cutoff, seed_probability, seed_last_available, last_cutoff
):
    reference = target_alignment(targets, reference)
    seed_cutoff, seed_last_available, last_cutoff = map(
        pd.Timestamp, (seed_cutoff, seed_last_available, last_cutoff)
    )
    if (
        any(date not in reference for date in (seed_cutoff, seed_last_available, last_cutoff))
        or not seed_last_available <= seed_cutoff <= last_cutoff
    ):
        raise ValueError("Ordered observed seed and cutoff dates required")
    seed_probability = _finite(seed_probability)
    if not 0 < seed_probability < 1:
        raise ValueError("Original first supported frequency must be interior")
    start, stop = reference.get_loc(seed_cutoff), reference.get_loc(last_cutoff)
    # Availability row k corresponds exactly to the preceding target origin.
    labels = targets.y.shift(1)
    observed = []
    output = []
    latest = seed_last_available
    for position in range(start, stop + 1):
        elapsed = position - start
        if elapsed and pd.notna(labels.iloc[position]):
            observed.append((position, float(labels.iloc[position])))
            latest = reference[position]
        initial = _finite(DELTA**elapsed)
        if initial == 0:
            raise ValueError("Nonzero initial weight underflowed")
        if observed:
            events = np.asarray(observed, float)
            powers = np.power(DELTA, position - events[:, 0])
            weights = (1 - DELTA) * powers
            terms = weights * events[:, 1]
            if (
                not np.isfinite(powers).all()
                or not np.isfinite(weights).all()
                or not np.isfinite(terms).all()
                or (powers == 0).any()
                or (weights == 0).any()
                or ((events[:, 1] != 0) & (terms == 0)).any()
            ):
                raise ValueError("Nonfinite weighted terms or nonzero-weight underflow")
        else:
            weights = np.empty(0)
            terms = np.empty(0)
        s = _finite(math.fsum([multiply(seed_probability, initial), *terms]))
        w = _finite(math.fsum([initial, *weights]))
        q = divide(s, w)
        if not 0 <= s <= w or not 0 <= q <= 1:
            raise ValueError("Valid positive weighted frequency state required")
        output.append(
            {
                "feature_cutoff_date": reference[position],
                "S": s,
                "W": w,
                "recent_frequency": q,
                "latest_consumed_available": latest,
                "cumulative_updates": len(observed),
                "elapsed_sessions": elapsed,
            }
        )
    return pd.DataFrame(output).set_index("feature_cutoff_date")


PREVIOUS_LIMITATIONS = [
    "Strict same-sign probability includes exact-zero returns in the complement and retains missing paired returns as unknown; no volatility-magnitude or covariance target.",
    "The excess-agreement increment is a staged logistic model with frozen baseline logits; regularized training NLL and evaluation Brier share a probability target but need not improve together under misspecification.",
    "Reused archival QQQ ETF/SPX index history, unverified historical source vintages and synchronized auction definitions, and early back-calculated VIX9D remain exploratory limitations.",
    "Calibration in the large and nominal minimum-detectable effect are descriptive diagnostics; nonqualification is not equivalence and no execution or profit claim is made.",
]


def table_equal(actual, expected, numeric, label):
    if tuple(actual.columns) != tuple(expected.columns) or not actual.index.equals(
        expected.index
    ):
        raise AssertionError(label + ": exact schema and dates required")
    a, e = (
        previous.real(actual.loc[:, numeric], missing=True),
        previous.real(expected.loc[:, numeric], missing=True),
    )
    if not np.array_equal(np.isnan(a), np.isnan(e)) or not np.allclose(
        a, e, rtol=1e-10, atol=1e-12, equal_nan=True
    ):
        raise AssertionError(label + ": independent numeric values or missingness differ")
    for column in set(expected) - set(numeric):
        if not actual[column].equals(expected[column]):
            raise AssertionError(label + ": exact metadata differs")


def reconstruct_previous(root, info, pins):
    oldp = yaml.safe_load(read_snapshot(root, info["protocol"], info["protocol_sha256"]))
    previous.validate_protocol(oldp)
    oldreport, olddata = root / info["reports"], root / info["data"]
    oldm = read_json_snapshot(
        root, str((oldreport / "manifest.json").relative_to(root)), info["manifest_sha256"]
    )

    def saved_json(path):
        name = str(path.relative_to(root))
        return previous.previous.read_json_snapshot(root, name, pins[name])

    def saved_parquet(path):
        name = str(path.relative_to(root))
        return scalar.read_issued_snapshot(root, name, pins[name])

    nested = previous.validate_upstream(root)
    same_tree(
        saved_json(olddata / "upstream_admission.json"),
        nested,
        "Complete original upstream proof",
    )
    qqq, spx, iv, audit = previous.load_source_tables(root, oldp, oldm)
    same_tree(
        saved_json(oldreport / "source_audit.json"), audit, "Original immutable source audit"
    )
    measured = previous.original.measurement_audit(qqq, spx)
    same_tree(
        saved_json(oldreport / "measurement_audit.json"),
        measured,
        "Original full-source measurement gate",
    )
    previous.original.require_measurement(measured)
    features, targets = previous.feature_target_tables(qqq, spx, iv)
    table_equal(
        saved_parquet(olddata / "features.parquet"),
        features,
        previous.ALL_FEATURES,
        "Original features",
    )
    oldtargets = saved_parquet(olddata / "targets.parquet")
    table_equal(oldtargets, targets, ("y",), "Original targets")
    if not np.array_equal(oldtargets.y, targets.y, equal_nan=True):
        raise AssertionError("Original strict labels must match exactly")
    panel = saved_parquet(olddata / "forecasts.parquet")
    fits = saved_json(olddata / "fits.json")
    metrics = saved_json(oldreport / "metrics.json")
    if (
        metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != info["protocol_sha256"]
        or metrics["evidence_class"] != oldp["evidence_class"]
    ):
        raise AssertionError("Original successful score identity differs")
    result = {
        "status": "VERIFIED",
        "upstream_admission": {"status": nested["status"], "prior_files_written": False},
        "raw_feature_rows_verified": len(features),
        "raw_feature_columns_verified": len(previous.ALL_FEATURES),
        "measurement_audit": measured,
        "forecast_reconstruction": previous.verify_forecasts(
            features, targets, panel, fits, oldp
        ),
        "primitive_brier_scores_verified": len(panel),
        "paired_brier_differences_verified": int(panel.origin.nunique()) * 2,
        "inference": previous.verify_metrics(root, panel, oldp, metrics, len(fits)),
        "ledger_events_verified": previous.verify_ledger(
            root, metrics, metrics["inherited_rows"], "evaluated"
        ),
        "protocol_sha256": info["protocol_sha256"],
        "verifier_sha256": info["verifier_sha256"],
        "limitations": PREVIOUS_LIMITATIONS,
    }
    return result, nested


def exact_float(actual, expected, label):
    a, e = previous.real(actual), previous.real(expected)
    if a.shape != e.shape or not np.array_equal(a.view(np.uint64), e.view(np.uint64)):
        raise AssertionError(label + ": exact float64 replay differs")


def validate_panel(panel):
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.empty
        or tuple(panel.columns) != PANEL_COLUMNS
        or panel.duplicated(["origin", "model"]).any()
        or set(panel.model) != set(MODELS)
    ):
        raise AssertionError("Exact complete three-model probability schema required")
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
            raise AssertionError(
                "Identical issued labels and metadata required for both components and pool"
            )
    for name in (
        "origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "fit_origin",
        "fit_cutoff_date",
        "train_last_target",
        "train_last_available",
    ):
        dates = pd.DatetimeIndex(panel[name])
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise AssertionError("Normalized finite date metadata required")
    exact_float(
        panel.loss, previous.brier_loss(panel.probability, panel.y), "Issued Brier scores"
    )


def verify_forecasts(features, targets, issued, fits, panel, states, protocol):
    section = protocol["index"]
    reference = target_alignment(targets, features.index)
    if tuple(features.columns) != previous.ALL_FEATURES + ("feature_cutoff_date",):
        raise AssertionError("Exact frozen feature calendar required")
    predecessor = pd.Series(reference, index=reference).shift()
    if not features.feature_cutoff_date.equals(predecessor.rename("feature_cutoff_date")):
        raise AssertionError("Frozen previous-session feature cutoffs differ")
    applications, scored = previous.eligible_entries(features, targets, section)
    if not len(applications):
        raise AssertionError("Original application cohort empty")
    previous.validate_panel(issued)
    validate_panel(panel)
    original_groups = {
        name: issued.loc[issued.model == name].set_index("origin").sort_index()
        for name in previous.MODELS
    }
    groups = {
        name: panel.loc[panel.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    if any(
        not rows.index.equals(scored) for rows in [*groups.values(), *original_groups.values()]
    ):
        raise AssertionError("Every original scored origin must remain exactly present")
    baseline = (
        issued.loc[issued.model == "baseline"]
        .sort_values("origin")
        .reset_index(drop=True)
        .copy()
    )
    baseline["model"] = "frozen_baseline"
    preserved = (
        panel.loc[panel.model == "frozen_baseline"]
        .sort_values("origin")
        .reset_index(drop=True)
    )
    if not preserved.equals(baseline):
        raise AssertionError("Frozen baseline rows must be copied exactly")
    for name in ("y", "probability", "loss"):
        exact_float(preserved[name], baseline[name], "Original " + name)
    if tuple(states.columns) != STATE_COLUMNS or not pd.DatetimeIndex(states.origin).equals(
        applications
    ):
        raise AssertionError("State audit must cover every original application exactly")
    previous.real(
        states.loc[
            :,
            [
                "seed_probability",
                "seed_train_n",
                "S",
                "W",
                "recent_frequency",
                "cumulative_updates",
                "elapsed_sessions",
            ],
        ]
    )
    months = applications.to_period("M").unique()
    if len(fits) != len(months):
        raise AssertionError("All original fit identities required, including unscored months")
    source_fit = {}
    for number, month in enumerate(months):
        app = applications[applications.to_period("M") == month]
        entry = app[0]
        cutoff = predecessor.loc[entry]
        fit = fits[number]
        if (
            pd.Timestamp(fit["fit_origin"]) != entry
            or pd.Timestamp(fit["fit_cutoff_date"]) != cutoff
            or fit["application_n"] != len(app)
            or fit["train_n"] < section["minimum_train"]
            or pd.Timestamp(fit["train_last_available"]) > cutoff
            or fit["train_last_available"] != fit["train_last_target"]
            or fit["model_audit"]["train_n"] != fit["train_n"]
            or fit["model_audit"]["application_n"] != len(app)
        ):
            raise AssertionError(
                "Original monthly source identity or supported training metadata differs"
            )
        frequency = fit["model_audit"]["frequency"]
        if (
            frequency["train_n"] != fit["train_n"]
            or frequency["application_n"] != len(app)
            or not 0 < _finite(frequency["probability"]) < 1
        ):
            raise AssertionError("Original supported training frequency identity differs")
        chosen = app[app.isin(scored)]
        exact_float(
            original_groups["frequency"].loc[chosen, "probability"],
            np.full(len(chosen), frequency["probability"]),
            "Original monthly frequency seed provenance",
        )
        for date in app:
            source_fit[date] = entry
        for rows in original_groups.values():
            expected = {
                "feature_cutoff_date": predecessor.loc[chosen].to_numpy(),
                "target_end": targets.loc[chosen, "target_end"].to_numpy(),
                "available_date": targets.loc[chosen, "available_date"].to_numpy(),
                "y": targets.loc[chosen, "y"].to_numpy(),
                "fit_origin": np.full(len(chosen), entry.to_datetime64()),
                "fit_cutoff_date": np.full(len(chosen), cutoff.to_datetime64()),
                "train_n": np.full(len(chosen), fit["train_n"]),
                "horizon": np.ones(len(chosen), int),
                "train_last_target": np.full(
                    len(chosen), pd.Timestamp(fit["train_last_target"]).to_datetime64()
                ),
                "train_last_available": np.full(
                    len(chosen), pd.Timestamp(fit["train_last_available"]).to_datetime64()
                ),
                "phase": np.where(
                    chosen <= pd.Timestamp(section["development"][1]),
                    "development",
                    "evaluation",
                ),
            }
            for column, value in expected.items():
                if not np.array_equal(rows.loc[chosen, column], value):
                    raise AssertionError(
                        "Original issued " + column + " differs from admitted source identity"
                    )
    seed = fits[0]
    seed_cutoff = pd.Timestamp(seed["fit_cutoff_date"])
    seed_last = pd.Timestamp(seed["train_last_available"])
    seed_probability = float(seed["model_audit"]["frequency"]["probability"])
    full = explicit_states(
        targets,
        reference,
        seed_cutoff,
        seed_probability,
        seed_last,
        predecessor.loc[applications[-1]],
    )
    for row in states.itertuples(index=False):
        date = pd.Timestamp(row.origin)
        cutoff = predecessor.loc[date]
        expected = full.loc[cutoff]
        elapsed = int(expected.elapsed_sessions)
        for name, value in (
            ("feature_cutoff_date", cutoff),
            ("source_fit_origin", source_fit[date]),
            ("seed_fit_origin", applications[0]),
            ("seed_cutoff_date", seed_cutoff),
            ("seed_last_available", seed_last),
            ("latest_consumed_available", expected.latest_consumed_available),
            ("cumulative_updates", int(expected.cumulative_updates)),
            ("elapsed_sessions", elapsed),
            ("seed_train_n", seed["train_n"]),
            ("scored", date in scored),
        ):
            if getattr(row, name) != value:
                raise AssertionError("Full-calendar state " + name + " differs")
        exact_float(row.seed_probability, seed_probability, "Original fixed seed probability")
        s, w, q = map(_finite, (row.S, row.W, row.recent_frequency))
        if not 0 <= s <= w or w <= 0 or not 0 <= q <= 1:
            raise AssertionError("Strict state domain required before numerical comparison")
        for name, value in (("S", s), ("W", w), ("recent_frequency", q)):
            state_equal(value, expected[name], elapsed, "Explicit " + name)
        exact_float(q, divide(s, w), "Saved state quotient")
        if date in scored:
            p = float(groups["frozen_baseline"].loc[date, "probability"])
            exact_float(
                groups["recent_frequency"].loc[date, "probability"],
                q,
                "Issued recent frequency",
            )
            pool = _finite(multiply(0.5, p) + multiply(0.5, q))
            if not 0 <= pool <= 1:
                raise AssertionError("Valid pooled probability required")
            exact_float(
                groups["pooled"].loc[date, "probability"],
                pool,
                "Separate half-product pooled probability",
            )
    return {
        "new_monthly_fits": 0,
        "new_forecasts_verified": 2 * len(scored),
        "preserved_forecasts_verified": len(scored),
        "forecasts_verified": len(panel),
        "common_scored_origins": len(scored),
        "common_application_origins": len(applications),
        "application_states_verified": len(states),
        "full_calendar_states_reconstructed": len(full),
        "post_seed_label_updates_verified": int(full.iloc[-1].cumulative_updates),
    }


def validate_upstream(root=ROOT):
    root = Path(root)
    protocol_path = root / "causal_pool.yaml"
    protocol_payload = protocol_path.read_bytes()
    protocol = yaml.safe_load(protocol_payload)
    info = protocol["upstream"]
    captured = json.loads((root / "reports/causal_pool/manifest.json").read_text())
    if sha256(protocol_payload).hexdigest() != captured["protocol_sha256"]:
        raise AssertionError("New manifest must precede upstream admission")
    oldreport, olddata = root / info["reports"], root / info["data"]

    def inventory():
        return {
            str(path.relative_to(root))
            for folder in (oldreport, olddata)
            for path in folder.rglob("*")
            if path.is_file()
        } | {info["protocol"]}

    required = inventory()
    if not required.issubset(captured["inputs"]) or (oldreport / "failure.json").exists():
        raise AssertionError("Every original successful publication/output must be pinned")
    for path, key in (
        (root / info["protocol"], "protocol_sha256"),
        (oldreport / "manifest.json", "manifest_sha256"),
        (oldreport / "verification.json", "verification_sha256"),
    ):
        if digest(path) != info[key]:
            raise AssertionError("Original wave13 anchor identity differs")
    oldm = read_json_snapshot(
        root, str((oldreport / "manifest.json").relative_to(root)), info["manifest_sha256"]
    )
    record = read_json_snapshot(
        root,
        str((oldreport / "verification.json").relative_to(root)),
        info["verification_sha256"],
    )
    json.dumps(record, allow_nan=False)
    if (
        info["required_status"] != "VERIFIED"
        or record["status"] != "VERIFIED"
        or record["protocol_sha256"] != info["protocol_sha256"]
        or record["verifier_sha256"] != info["verifier_sha256"]
        or oldm["protocol_sha256"] != info["protocol_sha256"]
        or oldm["code"]["src/verify_sign_memory.py"] != info["verifier_sha256"]
    ):
        raise AssertionError("Original verification and registration identity differs")
    pins = {}
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, captured[group])
        for name, signature in captured[group].items():
            if name in pins and pins[name] != signature:
                raise AssertionError("Conflicting current pins")
            pins[name] = signature
    if not set(oldm["code"]).issubset(captured["code"]) or not set(oldm["inputs"]).issubset(
        captured["inputs"]
    ):
        raise AssertionError("Original code/input identities must remain registered")
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, oldm[group])
        if any(pins.get(name) != signature for name, signature in oldm[group].items()):
            raise AssertionError(
                "Original preserved artifact absent from current registration"
            )
    rebuilt, nested = reconstruct_previous(root, info, pins)
    compare_tree(record, rebuilt, "Entire original wave13 VERIFIED record")
    if inventory() != required or digest(protocol_path) != captured["protocol_sha256"]:
        raise AssertionError("Original inventory or new protocol changed during admission")
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, captured[group])
    return {
        "status": "UPSTREAM_VERIFIED_READ_ONLY",
        "prior_files_written": False,
        "previous_manifest_entries_verified": sum(
            len(oldm[g]) for g in ("code", "inputs", "preserved")
        ),
        "pinned_previous_artifacts": len(required),
        "input_hashes": captured["inputs"],
        "original_verification": rebuilt,
        "nested_sign_memory_admission": nested,
        "upstream_protocol_sha256": info["protocol_sha256"],
        "upstream_manifest_sha256": info["manifest_sha256"],
        "upstream_verification_sha256": info["verification_sha256"],
    }


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    info = protocol["upstream"]
    old = read_json_snapshot(root, info["reports"] + "/manifest.json", info["manifest_sha256"])
    code = {
        str(path.relative_to(root))
        for folder in ("src", "tests")
        for path in (root / folder).rglob("*.py")
    } | set(old["code"])
    inputs = (
        set(old["inputs"])
        | {info["protocol"]}
        | set(protocol["comparisons"]["inherited_sources"])
    )
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
        and path != root / "causal_pool.yaml"
        and root / "reports/causal_pool" not in path.parents
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


def invalidate_publication(root, error):
    report = Path(root) / "reports/causal_pool"
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
        "cumulative_hypothesis_count": 121,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "causal_pool",
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
        "# QQQ–SPX fixed causal probability pool\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n"
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


CONTRACT = {
    "study_id": "causal_pool_wave14",
    "specified_on": "2026-09-07",
    "status": "specified_before_new_filter_states_forecasts_or_scores",
    "wave": 14,
    "wave_alpha": 0.0002380952380952381,
    "objective": "Test one fixed probability pool against both its frozen conditional baseline and causal "
    "recent-frequency component",
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
        "admission": "Reconstruct complete original VERIFIED record through frozen read-only "
        "functions; anchor historical inventory by its exact original manifest hash, "
        "never by enumerating the expanded current source tree; pin all previous code, "
        "inputs, preserved and output/report artifacts",
    },
    "target": {
        "event": "Both next observed SPX-session raw log(close/open) returns strictly positive, or both "
        "strictly negative",
        "zero": "Either or both exactly zero gives class0; complement includes opposite directions and "
        "ties",
        "missing": "Any missing return gives unknown, never class0; reject infinity or invalid observed "
        "source",
        "sign_arithmetic": "Direct comparisons, never multiplication of returns or epsilon sign "
        "threshold",
        "positive_rescaling": "Within-session positive price unit changes preserve each raw return; no "
        "total-return or executable auction equivalence",
        "fit_schedule": "Monthly first feature-complete origin before future query-label filtering; "
        "expanding training labels mature by prior SPX close; retain unscored "
        "applications and folds",
    },
    "source_contract": {
        "reference_calendar": "Exact full bounded SPX reference calendar from frozen wave13 "
        "features; no compressed dates or new source acquisition",
        "source_and_measurement": "All frozen wave13 raw-source, whole-measurement and "
        "feature/target gates remain enforced by complete read-only "
        "independent replay",
        "immutable_read": "Decode each admitted parquet/json input from a single hash-checked "
        "byte snapshot; final rehash all pins; no previous file writes",
        "limitations": "Archival QQQ ETF versus SPX price index, vendor revisions and "
        "unsynchronized opens, unverified historical publication latency and "
        "back-calculated early VIX9D remain; no new untouched validation",
    },
    "pooling": {
        "half_life_sessions": 63,
        "decay": "2**(-1/63)",
        "baseline_weight": 0.5,
        "recent_weight": 0.5,
        "baseline": "Exact original issued baseline probability on every original scored origin; no "
        "refit, memory-arm substitution or reconstructed in-sample predictions",
        "seed": "At first original feature-complete monthly application origin preceding-session cutoff "
        "k0, S=f0,W=1; f0 exact original first fit frequency on its eligible mature training "
        "subset",
        "seed_provenance": "Record first fit train_last_available separately from state cutoff k0; "
        "excluded old labels intentionally absent; never refeed any target "
        "available<=k0",
        "update": "Every subsequent full reference session k: S=delta*S,W=delta*W, then for its unique "
        "finite binary target add (1-delta)*y to S and (1-delta) to W; advance only through "
        "previous-session cutoff of prediction",
        "labels": "Full immutable target table, including labels whose original origin was unscored or "
        "feature-incomplete; require exact next-reference-session target_end=available_date, "
        "duplicate or misaligned availability invalid",
        "missing": "Decay numerator and mass on every reference session; missing paired return adds "
        "nothing, never class0; zero raw return remains valid nonevent; no compressing "
        "observations",
        "continuity": "Single initialization and no resets across phases, feature gaps, all-unscored "
        "months or fixed evaluation slices; no burn-in or scored-origin deletion",
        "outputs": "For every original application save state cutoff, latest included label "
        "availability, cumulative post-seed observation update count, S,W,q and original fit "
        "identity. Only original scored origins receive three probability rows; unscored "
        "applications have state audit only",
        "arithmetic": "IEEE float64 in fixed operation order: delta*S and delta*W; (1-delta)*y; "
        "additions; q=S/W; separate 0.5*p and 0.5*q products then sum. All state "
        "finite,0<=S<=W,W>0,probabilities in[0,1]. Reject unsupported nonzero "
        "products/divisions underflowing tozero; allow nonzero subnormals and exactzero "
        "outcomes; no clipping, reset, omission or alternative memory",
    },
    "scoring": {
        "loss": "brier",
        "effect_threshold_absolute": 0.0005,
        "paired_difference": "(p_candidate-p_control)*((p_candidate-y)+(p_control-y)), after finite "
        "individual squared loss validation",
        "arithmetic": "Binary finite y and finite probabilities in[0,1]; loss in[0,1]; exactzero errors "
        "valid; nonzero squares/products underflowing tozero reject; "
        "coherence64epsilon/downwardULP no arbitrary absolute floor",
        "functional": "Conditional strict-agreement probability; expected Brier pi(1-pi)+(p-pi)^2; "
        "bounded loss needs no return fourth moments",
        "effect_reference": "Absolute squared-probability error decrease .0005 bothphases; fixed "
        "statistical reference, not profit or direct probability-point improvement",
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
        "p_phase": "Maximum two-sided centered-null circular-block bootstrap p and Bartlett HAC126 p",
        "p_hypothesis": "Maximum development/evaluation p",
        "multiplicity": "Holm2 at.05/(14*15), separately cumulativeHolm121 at.05",
        "gate": "Bothphase absolute Brier decrease>=.0005 against BOTHcontrols; negative differences "
        "in bothfixed evaluation slices; waveHolm<.05/210,cumulativeHolm<.05",
        "support": "Original fits retain at least50events/50nonevents;30/class perphase;15/class "
        "perfixed evaluationslice; any insufficient support aborts wholefamily; no "
        "droppedfolds",
        "power": "Nominal HAC80percent MDE diagnostic only, divided by fixed .0005 effect; no "
        "postscore power gate or equivalence claim",
        "resolution": "Minimum p1/100000 below one tenth strictest two-comparison raw wave "
        "cutoff1/8400",
        "failure_rule": "All2new hypotheses UNEVALUABLEp1 after any "
        "admission/measurement/support/fit/scoring/verification/publication failure; "
        "preserve all diagnostics and old artifacts; no outcome-dependent repair",
        "calibration": "Descriptive calibration in the large only: perphase/model n, observed binary "
        "frequency, mean issued probability, probability-minus-frequency gap and "
        "Brier. No bins, calibration fitting, hypothesis, selection or promotion gate; "
        "not a full conditional calibration claim.",
    },
    "verification": {
        "state_relative_roundoff_multiplier": 64,
        "state_roundoff": "At k full reference sessions after k0, "
        "abs(actual-explicit)<=64*(k+1)*max(eps*abs(explicit),abs(explicit)-nextafter(abs(explicit),0)); "
        "exactzero and sign masks, strict state/probability domain first",
        "independent_state": "Explicit compensated sum of seed*delta**k and observed "
        "(1-delta)*delta**age weights; denominator seed mass delta**k plus "
        "same weights; do not call producer recurrence",
        "saved_replay": "Exact float64 bits for q from saved S/W, pooled from two separate half "
        "products, preserved baseline and all Brier losses; no tolerance "
        "permitting domain violation",
        "inference_relative_tolerance": 1e-09,
        "inference_absolute_tolerance": 1e-14,
        "reconstruction": "Complete frozen upstream proof; independent application and scored "
        "cohorts, every state/probability/primitive Brier/difference, four phase "
        "analyses, six calibration rows, complete121family ledger, immutable "
        "output snapshots and final pin rehash",
    },
    "outputs": {
        "data": "data/causal_pool",
        "reports": "reports/causal_pool",
        "forecasts": "data/causal_pool/forecasts.parquet",
        "states": "data/causal_pool/states.parquet",
        "admission": "data/causal_pool/upstream_admission.json",
    },
}


def validate_protocol(p):
    if p != CONTRACT:
        raise ValueError("Frozen causal-pool protocol differs")


def verify(root=ROOT):
    root = Path(root)
    protocol_path = root / "causal_pool.yaml"
    payload = protocol_path.read_bytes()
    protocol = yaml.safe_load(payload)
    validate_protocol(protocol)
    protocol_hash = sha256(payload).hexdigest()
    report, out = root / "reports/causal_pool", root / "data/causal_pool"
    manifest_path = report / "manifest.json"
    manifest_payload = manifest_path.read_bytes()
    manifest_hash = sha256(manifest_payload).hexdigest()
    manifest = json.loads(manifest_payload)
    if manifest["protocol_sha256"] != protocol_hash:
        raise AssertionError("Frozen protocol and manifest identity differ")
    verify_manifest_coverage(root, protocol, manifest)
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, manifest[group])
    paths = [
        out / name
        for name in ("upstream_admission.json", "forecasts.parquet", "states.parquet")
    ] + [report / "metrics.json"]
    if (report / "trial_ledger.jsonl").exists():
        paths.append(report / "trial_ledger.jsonl")
    snapshots = {str(path.relative_to(root)): digest(path) for path in paths}
    proof = validate_upstream(root)
    name = str((out / "upstream_admission.json").relative_to(root))
    same_tree(
        read_json_snapshot(root, name, snapshots[name]),
        proof,
        "Complete original read-only admission",
    )
    inputs = {}
    for key in ("features", "targets", "forecasts"):
        name = protocol["upstream"][key]
        inputs[key] = scalar.read_issued_snapshot(root, name, manifest["inputs"][name])
    name = protocol["upstream"]["fits"]
    fits = read_json_snapshot(root, name, manifest["inputs"][name])
    name = str((out / "forecasts.parquet").relative_to(root))
    panel = scalar.read_issued_snapshot(root, name, snapshots[name])
    name = str((out / "states.parquet").relative_to(root))
    states = scalar.read_issued_snapshot(root, name, snapshots[name])
    name = str((report / "metrics.json").relative_to(root))
    metrics = read_json_snapshot(root, name, snapshots[name])
    if (
        (report / "failure.json").exists()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != protocol_hash
        or metrics["evidence_class"] != protocol["evidence_class"]
    ):
        raise AssertionError("New publication failed or its identity differs")
    result = {
        "status": "VERIFIED",
        "upstream_admission": {"status": proof["status"], "prior_files_written": False},
        "forecast_reconstruction": verify_forecasts(
            inputs["features"],
            inputs["targets"],
            inputs["forecasts"],
            fits,
            panel,
            states,
            protocol,
        ),
        "primitive_brier_scores_verified": len(panel),
        "paired_brier_differences_verified": int(panel.origin.nunique()) * 2,
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
        "protocol_sha256": protocol_hash,
        "verifier_sha256": digest(Path(__file__)),
        "limitations": [
            "One fixed pool of an unchanged conditional probability and a causal unconditional event-rate filter; no fitted calibration coefficient or new conditional feature.",
            "The initial mass summarizes only the first original eligible training subset; subsequent labels use the full calendar and mature target table, including previously unscored origins.",
            "Adaptively selected exploratory comparison on reused archival QQQ ETF/SPX price-index history; preserved multiplicity and causal replay do not provide a new untouched validation sample.",
            "Strict raw directional agreement, not volatility magnitude, covariance, executable direction or profit; calibration in the large and nominal detectable effect remain descriptive.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, manifest[group])
    scalar._pins(root, snapshots)
    verify_manifest_coverage(root, protocol, manifest)
    if digest(protocol_path) != protocol_hash or digest(manifest_path) != manifest_hash:
        raise AssertionError("Protocol or manifest changed during verification")
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def inherited_rows(root, protocol, *, pins=None):
    root = Path(root)
    if pins is None:
        pins = json.loads((root / "reports/causal_pool/manifest.json").read_bytes())["inputs"]
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
    if len(output) != 119:
        raise AssertionError("All119 inherited comparisons must remain identified")
    return output


def verify_metrics(root, panel, protocol, metrics, application_n, *, admitted_inputs=None):
    validate_panel(panel)
    if (
        metrics["new_monthly_fits"] != 0
        or metrics["new_forecasts"] != 2 * panel.origin.nunique()
        or metrics["preserved_forecasts"] != panel.origin.nunique()
        or metrics["combined_forecasts"] != len(panel)
        or metrics["common_application_origins"] != application_n
        or metrics["common_scored_origins"] != panel.origin.nunique()
    ):
        raise AssertionError("New full fit/forecast cohort counts differ")
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
        if row["study"] != "causal_pool" or row["horizon"] != 1 or row["score"] != "brier":
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
            "121-hypothesis cumulative correction",
            "p",
        )
        one = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05)
        if row["verdict"] != ("PASSES_ALL_GATES" if one else "DOES_NOT_QUALIFY"):
            raise AssertionError("Fixed statistical/effect/stability gate differs")
        passed.append(one)
    leads = ["pooled"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 2
        or metrics["cumulative_hypothesis_count"] != 121
    ):
        raise AssertionError("Complete candidate and hypothesis family required")
    return {
        "new_hypotheses_verified": 2,
        "cumulative_hypotheses_verified": 121,
        "phase_comparisons_verified": 4,
        "bootstrap_runs_verified": 12,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "calibration_rows_verified": 6,
        "class_support": counts,
        "leads": leads,
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
    rows, benchmark = groups["pooled"], groups[control]
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


def verify_ledger(root, metrics, prior, final_event, *, signature=None):
    report = Path(root) / "reports/causal_pool"
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
        len(ledger) != 123
        or len(registered) != 2
        or len(prior) != 119
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "causal_pool"
            or row["horizon"] != 1
            or row["score"] != "brier"
            for row in registered
        )
    ):
        raise AssertionError("Complete119inherited+2registered+2terminal ledger required")
    return {"inherited": 119, "registered": 2, final_event: 2}


support = previous.support
normalized_difference = previous.normalized_difference


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False), flush=True)
