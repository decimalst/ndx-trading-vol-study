"""Independent profiled baseline verification with unchanged scientific gates.

Only the independent baseline solve changes. Source admission and original
mathematical checks are reused explicitly; no frozen module globals are patched.
"""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq

from . import profiled_quarter_solver as profiled
from . import verify_civil_quarter as frozen
from . import verify_civil_quarter_replay as previous

ROOT = Path(__file__).resolve().parents[1]
WAVE_ALPHA = 0.05 / (17 * 18)
EFFECT = 0.005
MODELS = frozen.MODELS
COMPARISONS = frozen.COMPARISONS
ALL_FEATURES = frozen.ALL_FEATURES
OLD = frozen.OLD
BASE = frozen.BASE
MEMORY = frozen.MEMORY
finite = frozen.finite
positive = frozen.positive
multiply = frozen.multiply
divide = frozen.divide
exp_checked = frozen.exp_checked
checked_mean = frozen.checked_mean
design_product = frozen.design_product
transform = frozen.transform
positive_objective = frozen.positive_objective
scalar_objective = frozen.scalar_objective
normalized_equal = frozen.normalized_equal
same = frozen.same
same_tree = frozen.same_tree
civil_support = frozen.civil_support
civil_rank = frozen.civil_rank
eligible_entries = frozen.eligible_entries
validate_panel = frozen.validate_panel
augment_features = frozen.augment_features
preflight = frozen.preflight
phase_statistics = frozen.phase_statistics
compare_tree = frozen.compare_tree
inference_equal = frozen.inference_equal
table_equal = frozen.table_equal
holm = frozen.holm
normalized_difference = frozen.normalized_difference
digest = frozen.digest
read_snapshot = previous.read_snapshot
read_json_snapshot = previous.read_json_snapshot
strict_json = previous.strict_json
safe_path = previous.safe_path
valid_hash = previous.valid_hash
snapshot_sources = previous.snapshot_sources
collect_sources = previous.collect_sources
reconstruct_calendar = previous.reconstruct_calendar
reconstruct_previous = previous.reconstruct_previous
verify_failed_attempt = previous.verify_failed_attempt
scalar = previous.scalar


def independent_fit(training, y, application):
    x, a, geometry = transform(training, application)
    y = positive(y)
    unit = float(positive(checked_mean(y)))
    q = divide(y, unit)
    beta, solver_audit = profiled.solve_baseline(x, q)
    beta = finite(beta)
    value, gradient, _ = positive_objective(beta, x, q)
    maximum = float(np.max(np.abs(gradient)))
    if maximum > 1e-8 + 1e-12:
        raise AssertionError("Independent baseline stationarity failed")
    eta = design_product(x, beta)
    z = training[MEMORY].to_numpy(float) - geometry["quarter_mean"]
    g0 = scalar_objective(0, eta, q, z)[1]
    radius = max(1.0, abs(g0) / 0.02)
    endpoints = [scalar_objective(b, eta, q, z)[1] for b in (-radius, radius)]
    if endpoints[0] > 0 or endpoints[1] < 0:
        raise AssertionError("Independent strictly-convex root bracket failed")
    b = float(
        brentq(
            lambda b: scalar_objective(b, eta, q, z)[1],
            -radius,
            radius,
            xtol=1e-12,
            rtol=1e-14,
            maxiter=200,
        )
    )
    quarter_value, quarter_gradient, curvature = scalar_objective(b, eta, q, z)
    if abs(quarter_gradient) > 1e-8 + 1e-12:
        raise AssertionError("Independent quarter stationarity failed")
    az = application[MEMORY].to_numpy(float) - geometry["quarter_mean"]
    predictions = {
        "mean": np.repeat(unit, len(application)),
        "baseline": exp_checked(np.log(unit) + design_product(a, beta)),
        "quarter": exp_checked(np.log(unit) + design_product(a, beta) + multiply(b, az)),
    }
    return {
        "beta": beta,
        "b": b,
        "geometry": geometry,
        "train_mean": unit,
        "baseline_objective": value,
        "baseline_solver_audit": {
            **solver_audit,
            "unprofiled_gradient": gradient.tolist(),
            "unprofiled_gradient_max_abs": maximum,
        },
        "baseline_gradient_max_abs": maximum,
        "quarter_objective": quarter_value,
        "quarter_gradient": quarter_gradient,
        "quarter_curvature": curvature,
        "bracket": [-radius, radius],
        "bracket_gradients": endpoints,
        "predictions": predictions,
    }


def verify_fit(training, y, application, audits):
    if set(audits) != set(MODELS):
        raise AssertionError("Every registered model audit required")
    independently = independent_fit(training, y, application)
    x, ax, geometry = transform(training, application)
    unit = independently["train_mean"]
    q = divide(positive(y), unit)
    for name, audit in audits.items():
        if audit["train_n"] != len(training) or audit["application_n"] != len(application):
            raise AssertionError("Training/application audit counts differ")
        same(
            audit["train_mean"], unit, "Shared exact target normalization", rtol=1e-12, atol=0
        )
        if name != "mean" and (
            audit["alpha"] != 0.01
            or not audit["start"] == [0.0] * (31 if name == "baseline" else 1)
            or not 0 <= audit["iterations"] < 200
            or audit["backtracks"] < 0
            or audit["numerical_trial_rejections"] < 0
            or not 0 <= float(finite(audit["gradient_max_abs"])) <= 1e-8
        ):
            raise AssertionError("Fixed optimizer and saved stationarity contract differs")
    mean = audits["mean"]
    if mean["columns"] != ["const"] or mean["gradient_max_abs"] != 0:
        raise AssertionError("Same-row arithmetic mean control differs")
    same(mean["beta"], [np.log(unit)], "Mean log-risk intercept", rtol=1e-12, atol=1e-12)
    base = audits["baseline"]
    if base["columns"] != list(BASE):
        raise AssertionError("Full31-column baseline required")
    same(
        base["means"], geometry["means"], "Training-only mixed centers", rtol=1e-12, atol=1e-14
    )
    same(
        base["scales"],
        geometry["scales"],
        "Old population/new fixed-one scales",
        rtol=1e-12,
        atol=1e-14,
    )
    beta = finite(base["scaled_beta"])
    if beta.shape != (31,):
        raise AssertionError("Full31 normalized coefficients required")
    native_beta = beta.copy()
    native_beta[0] += np.log(unit)
    same(
        base["beta"],
        native_beta,
        "Native/normalized intercept identity",
        rtol=1e-10,
        atol=1e-12,
    )
    value, gradient, _ = positive_objective(beta, x, q)
    maximum = float(np.max(np.abs(gradient)))
    if maximum > 1e-8 + 1e-12:
        raise AssertionError("Saved baseline fails independent stationarity")
    same(
        base["gradient"],
        gradient,
        "Independent saved baseline gradient",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        base["gradient_max_abs"],
        maximum,
        "Saved baseline full gradient",
        rtol=1e-4,
        atol=1e-12,
    )
    same(
        base["objective_scaled"],
        value,
        "Normalized penalized baseline objective",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        base["objective"],
        value + np.log(unit),
        "Native penalized baseline objective",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        beta,
        independently["beta"],
        "Separate profiled baseline optimum",
        rtol=1e-7,
        atol=1e-6,
    )
    quarter = audits["quarter"]
    if (
        quarter["columns"] != [MEMORY]
        or quarter["scale"] != 1
        or quarter["baseline_frozen"] is not True
    ):
        raise AssertionError("One fixed-scale slope on frozen baseline required")
    same(
        quarter["mean"],
        geometry["quarter_mean"],
        "Training-only quarter center",
        rtol=1e-12,
        atol=1e-14,
    )
    same_tree(quarter["support"], civil_support(training), "Independent train civil support")
    same_tree(quarter["civil_rank"], civil_rank(training), "Independent rank15 geometry")
    same_tree(
        quarter["identification"], geometry["identification"], "Independent novelty residual"
    )
    eta = design_product(x, beta)
    z = training[MEMORY].to_numpy(float) - geometry["quarter_mean"]
    b = float(finite(quarter["b"]))
    value, gradient, curvature = scalar_objective(b, eta, q, z)
    g0 = scalar_objective(0.0, eta, q, z)[1]
    radius = max(1.0, abs(g0) / 0.02)
    endpoints = [scalar_objective(v, eta, q, z)[1] for v in (-radius, radius)]
    if (
        endpoints[0] > 0
        or endpoints[1] < 0
        or abs(gradient) > 1e-8 + 1e-12
        or curvature < 0.02
    ):
        raise AssertionError("Saved scalar strict-convex certificate failed")
    same(
        quarter["bracket"],
        [-radius, radius],
        "Deterministic derivative bracket",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["bracket_gradients"],
        endpoints,
        "Independent endpoint derivative signs",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["initial_gradient"],
        g0,
        "Independent zero-slope gradient",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["gradient"],
        [gradient],
        "Independent saved scalar gradient",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["gradient_max_abs"],
        abs(gradient),
        "Saved scalar stationarity",
        rtol=1e-4,
        atol=1e-12,
    )
    same(
        quarter["curvature"],
        curvature,
        "Independent positive scalar curvature",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["objective_scaled"],
        value,
        "Normalized scalar objective",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["objective"],
        value + np.log(unit),
        "Native scalar objective",
        rtol=1e-10,
        atol=1e-12,
    )
    # This root uses the saved and checked baseline, isolating scalar tolerance.
    separate_b = brentq(
        lambda v: scalar_objective(v, eta, q, z)[1],
        -radius,
        radius,
        xtol=1e-12,
        rtol=1e-14,
        maxiter=200,
    )
    same(b, separate_b, "Separate brent scalar optimum", rtol=1e-7, atol=1e-6)
    az = application[MEMORY].to_numpy(float) - geometry["quarter_mean"]
    replay = {
        "mean": np.repeat(unit, len(application)),
        "baseline": exp_checked(np.log(unit) + design_product(ax, beta)),
        "quarter": exp_checked(np.log(unit) + design_product(ax, beta) + multiply(b, az)),
    }
    for name in MODELS:
        normalized_equal(
            replay[name],
            independently["predictions"][name],
            unit,
            "Independent normalized " + name + " predictions",
            independent=True,
        )
    return replay, {
        "models_verified": 3,
        "baseline_solver_audit": independently["baseline_solver_audit"],
        "producer_baseline_gradient": maximum,
        "independent_baseline_gradient": independently["baseline_gradient_max_abs"],
        "producer_quarter_gradient": abs(gradient),
        "independent_quarter_gradient": abs(scalar_objective(separate_b, eta, q, z)[1]),
    }


def verify_forecasts(features, targets, panel, fits, protocol):
    validate_panel(panel)
    section = protocol["index"]
    entries, scored, complete = eligible_entries(features, targets, section)
    for name in MODELS:
        rows = panel.loc[panel.model == name].sort_values("origin")
        if not pd.DatetimeIndex(rows.origin).equals(scored):
            raise AssertionError("Exact common scored cohort differs")
        same(rows.y, targets.loc[scored, "y"], "Saved unchanged risk target", rtol=0, atol=0)
        for column in ("available_date", "target_end"):
            if not np.array_equal(rows[column], targets.loc[scored, column]):
                raise AssertionError("Target maturity differs")
        if not np.array_equal(
            rows.feature_cutoff_date, features.loc[scored, "feature_cutoff_date"]
        ):
            raise AssertionError("Prior-session market cutoff differs")
        if not np.array_equal(
            rows.phase,
            np.where(scored <= section["development"][1], "development", "evaluation"),
        ):
            raise AssertionError("Exact fixed phases differ")
    fit_origins = entries[~entries.to_period("M").duplicated()]
    if len(fits) != len(fit_origins) or [
        pd.Timestamp(record["fit_origin"]) for record in fits
    ] != list(fit_origins):
        raise AssertionError(
            "All monthly feature-admitted fits required, including unscored months"
        )
    maxima = {
        "producer_baseline_gradient": 0.0,
        "independent_baseline_gradient": 0.0,
        "producer_quarter_gradient": 0.0,
        "independent_quarter_gradient": 0.0,
    }
    train_total = 0
    solver_audits = []
    for entry, record in zip(fit_origins, fits, strict=True):
        applications = entries[entries.to_period("M") == entry.to_period("M")]
        query = scored[scored.to_period("M") == entry.to_period("M")]
        cutoff = features.loc[entry, "feature_cutoff_date"]
        mask = (
            complete
            & targets.y.notna()
            & (features.index < entry)
            & (targets.available_date <= cutoff)
        )
        origins = features.index[mask]
        if (
            len(origins) < section["minimum_train"]
            or record["train_n"] != len(origins)
            or record["application_n"] != len(applications)
        ):
            raise AssertionError("Exact mature common training/application counts differ")
        expected = {
            "fit_cutoff_date": cutoff,
            "train_first_origin": origins[0],
            "train_last_origin": origins[-1],
            "train_last_target": targets.loc[mask, "target_end"].max(),
            "train_last_available": targets.loc[mask, "available_date"].max(),
        }
        if any(pd.Timestamp(record[key]) != value for key, value in expected.items()):
            raise AssertionError("Exact mature training date evidence differs")
        replay, proof = verify_fit(
            features.loc[mask],
            targets.loc[mask, "y"],
            features.loc[applications],
            record["model_audit"],
        )
        solver_audits.append(
            {"fit_origin": str(entry.date()), "audit": proof["baseline_solver_audit"]}
        )
        train_total += len(origins)
        for key in maxima:
            maxima[key] = max(maxima[key], proof[key])
        for name in MODELS:
            rows = panel.loc[panel.model.eq(name) & panel.origin.isin(query)].sort_values(
                "origin"
            )
            expected_predictions = replay[name][applications.get_indexer(query)]
            normalized_equal(
                rows.prediction,
                expected_predictions,
                record["model_audit"][name]["train_mean"],
                "Strict saved native forecast replay in normalized units",
            )
            if not rows.fit_origin.eq(entry).all() or not rows.train_n.eq(len(origins)).all():
                raise AssertionError("Issued monthly fit identity differs")
            for key in ("fit_cutoff_date", "train_last_target", "train_last_available"):
                if not rows[key].eq(expected[key]).all():
                    raise AssertionError("Issued causal fit evidence differs")
    return {
        "forecasts_verified": len(panel),
        "baseline_solver_audits": solver_audits,
        "monthly_fits_verified": len(fits),
        "common_scored_origins": len(scored),
        "common_application_origins": len(entries),
        "training_observation_instances_verified": train_total,
        "independent_baseline_fits_verified": len(fits),
        "independent_scalar_fits_verified": len(fits),
        **maxima,
    }


def invalidate_publication(root, error):
    report = Path(root) / "reports/profiled_quarter"
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
        "cumulative_hypothesis_count": 127,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "profiled_quarter",
                "candidate": candidate,
                "control": control,
                "score": "proper_variance",
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
        "# SPX civil quarter-end risk increment\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n"
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


def inherited_rows(root, protocol, *, pins=None):
    root = Path(root)
    if pins is None:
        pins = json.loads((root / "reports/profiled_quarter/manifest.json").read_bytes())[
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
    if len(output) != 125:
        raise AssertionError("All125 inherited comparisons must remain identified")
    return output


def verify_metrics(
    root,
    panel,
    features,
    protocol,
    metrics,
    monthly_fits,
    application_n,
    *,
    admitted_inputs=None,
):
    validate_panel(panel)
    reference = pd.DatetimeIndex(features.index)
    if (
        reference.has_duplicates
        or reference.hasnans
        or not reference.is_monotonic_increasing
        or not panel.origin.isin(reference).all()
    ):
        raise AssertionError("Every score origin requires its full ordered source calendar")
    values = features.loc[panel.origin, ALL_FEATURES].to_numpy()
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise AssertionError("All scored origins require complete real registered inputs")
    dates = pd.Series(reference, index=reference)
    for column, shift in (
        ("feature_cutoff_date", 1),
        ("target_end", -1),
        ("available_date", -1),
    ):
        if not np.array_equal(panel[column], dates.shift(shift).loc[panel.origin]):
            raise AssertionError("Exact reference-calendar score timing required")
    n = int(panel.origin.nunique())
    if (
        metrics["new_monthly_fits"] != monthly_fits
        or metrics["new_forecasts"] != len(panel)
        or metrics["common_scored_origins"] != n
        or metrics["common_application_origins"] != application_n
    ):
        raise AssertionError("Complete new monthly fit and forecast accounting differs")
    section = protocol["index"]
    union = np.zeros(len(panel), dtype=bool)
    support = {}
    for name in ("development", "evaluation"):
        first, last = section[name]
        mask = panel.origin.between(first, last)
        union |= mask
        if not panel.loc[mask, "phase"].eq(name).all():
            raise AssertionError("Literal phase membership differs")
        origins = pd.DatetimeIndex(panel.loc[mask & panel.model.eq("baseline"), "origin"])
        if len(origins) < 127:
            raise AssertionError("Fixed phase bandwidth unsupported")
        support[name] = civil_support(features.loc[origins], "phase")
    if (
        not union.all()
        or (panel.train_n < section["minimum_train"]).any()
        or (panel.available_date > section["latest_target"]).any()
        or (
            panel.loc[panel.phase.eq("development"), "available_date"]
            > section["development_target_available_by"]
        ).any()
    ):
        raise AssertionError("Fixed complete inference sample or maturity fence differs")
    support["evaluation_slices"] = []
    for start, end in section["evaluation_stability"]:
        origins = pd.DatetimeIndex(
            panel.loc[panel.model.eq("baseline") & panel.origin.between(start, end), "origin"]
        )
        support["evaluation_slices"].append(
            {"start": start, "end": end, **civil_support(features.loc[origins], "slice")}
        )
    same_tree(metrics["civil_support"], support, "Every fixed phase/slice civil class count")
    rows = metrics["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Both fixed quarter contrasts required")
    probabilities, effects = [], []
    for row in rows:
        if (
            row["study"] != "profiled_quarter"
            or row["horizon"] != 1
            or row["score"] != "proper_variance"
        ):
            raise AssertionError("Proper-risk comparison identity differs")
        phases = [
            phase_statistics(panel, features, row["control"], phase, code, protocol)
            for code, phase in enumerate(("development", "evaluation"))
        ]
        inference_equal(row["phases"], phases, "Independent proper-risk phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        inference_equal(
            row["p_conservative"], probability, "Both-phase conjunction probability", "p"
        )
        probabilities.append(probability)
        effects.append(
            all(phase["delta"] <= -EFFECT for phase in phases)
            and len(phases[1]["stability"]) == 2
            and all(one["delta"] < 0 for one in phases[1]["stability"])
        )
    prior = inherited_rows(root, protocol, pins=admitted_inputs)
    same_tree(metrics["inherited_rows"], prior, "Entire prior125 hypothesis identity")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-2:]
    passed = []
    for number, row in enumerate(rows):
        inference_equal(row["p_holm_wave"], wave[number], "Wave17 Holm2", "p")
        inference_equal(
            row["p_holm_cumulative"], cumulative[number], "Cumulative Holm127", "p"
        )
        one = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05)
        if row["verdict"] != ("PASSES_ALL_GATES" if one else "DOES_NOT_QUALIFY"):
            raise AssertionError("Fixed effect/statistical/stability gates differ")
        passed.append(one)
    leads = ["quarter"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 2
        or metrics["cumulative_hypothesis_count"] != 127
    ):
        raise AssertionError("Both-control candidate and complete family required")
    return {
        "new_hypotheses_verified": 2,
        "cumulative_hypotheses_verified": 127,
        "phase_comparisons_verified": 4,
        "bootstrap_runs_verified": 12,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "civil_support": support,
        "leads": leads,
    }


def verify_ledger(root, metrics, prior, final_event, *, signature=None):
    report = Path(root) / "reports/profiled_quarter"
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
        len(ledger) != 129
        or len(registered) != 2
        or len(prior) != 125
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "profiled_quarter"
            or row["horizon"] != 1
            or row["score"] != "proper_variance"
            for row in registered
        )
    ):
        raise AssertionError("Complete125inherited+2registered+2terminal ledger required")
    return {"inherited": 125, "registered": 2, final_event: 2}


def verify_failed_verification(root, info, pins):
    """Admit failed wave16 by immutable records; never decode diagnostic scores."""
    root = Path(root)
    report, data = info["reports"], info["data"]
    if (
        info["required_status"] != "UNEVALUABLE"
        or info["verification_status"] != "FAILED"
        or info["inherited_hypotheses"] != 123
        or info["registered_hypotheses"] != 2
        or info["cumulative_hypotheses"] != 125
        or info["terminal_event"] != "verification_failed"
        or info["generated_forecasts"] != 6297
        or info["generated_monthly_fits"] != 101
    ):
        raise AssertionError("Exact failed verification registration required")
    anchors = {info["protocol"]: info["protocol_sha256"]}
    for key, filename in (
        ("manifest", "manifest.json"),
        ("freeze_record", "freeze_record.json"),
        ("failure", "failure.json"),
        ("verification", "verification.json"),
        ("verification_failure_audit", "verification_failure_audit.json"),
        ("numerical_failure_diagnosis", "numerical_failure_diagnosis.json"),
        ("publication_audit", "publication_audit.json"),
        ("trial_ledger", "trial_ledger.jsonl"),
    ):
        anchors[report + "/" + filename] = info[key + "_sha256"]
    for name, signature in anchors.items():
        if pins.get(name) != signature:
            raise AssertionError("Failed verification anchor not in current registration")
        read_snapshot(root, name, signature)
    oldp = yaml.safe_load(read_snapshot(root, info["protocol"], info["protocol_sha256"]))
    previous.validate_protocol(oldp)

    def saved(name):
        path = report + "/" + name
        return read_json_snapshot(root, path, pins[path])

    manifest, freeze = saved("manifest.json"), saved("freeze_record.json")
    if (
        manifest["protocol_sha256"] != info["protocol_sha256"]
        or freeze["protocol_sha256"] != info["protocol_sha256"]
        or freeze["code"] != manifest["code"]
        or manifest["code"].get("src/verify_civil_quarter_replay.py")
        != info["verifier_sha256"]
    ):
        raise AssertionError("Original failed verifier and frozen code identity differs")
    for group in ("code", "inputs", "preserved"):
        for name, signature in manifest[group].items():
            if pins.get(name) != signature:
                raise AssertionError("Original failed verification dependency is missing")
            read_snapshot(root, name, signature)
    for name, signature in freeze["prefit_design"].items():
        if pins.get(name) != signature:
            raise AssertionError("Original failed design is no longer registered")
        read_snapshot(root, name, signature)
    failure, metrics, verification = (
        saved("failure.json"),
        saved("metrics.json"),
        saved("verification.json"),
    )
    if (
        metrics != failure
        or failure["status"] != "UNEVALUABLE"
        or failure["whole_wave_aborted"] is not True
        or failure["leads"]
        or failure["hypothesis_count"] != 2
        or failure["cumulative_hypothesis_count"] != 125
        or failure["protocol_sha256"] != info["protocol_sha256"]
        or verification["status"] != "FAILED"
        or verification["protocol_sha256"] != info["protocol_sha256"]
    ):
        raise AssertionError("Failed verification must remain canonical two-p1 family")
    rows = failure["rows"]
    if [(r["candidate"], r["control"]) for r in rows] != list(COMPARISONS):
        raise AssertionError("Both failed verification comparisons required in order")
    for row in rows:
        if (
            row["study"] != "civil_quarter_replay"
            or row["score"] != "proper_variance"
            or row["horizon"] != 1
            or row["phases"]
            or row["status"] != "INVALID_RUN"
            or row["verdict"] != "UNEVALUABLE"
            or row["error"] != verification["error"]
            or not row["error"]
            or any(row[k] != 1 for k in ("p_conservative", "p_holm_wave", "p_holm_cumulative"))
        ):
            raise AssertionError("Failed verification cannot become null evidence or a lead")
    prior = previous.inherited_rows(root, oldp, pins=manifest["inputs"])
    payload = read_snapshot(root, report + "/trial_ledger.jsonl", info["trial_ledger_sha256"])
    lines = [line for line in payload.splitlines() if line]
    # Discard all numeric tokens and nested contents when reading event labels.
    # Evaluated records are never otherwise decoded or used for inference.
    events = [
        json.loads(
            line,
            parse_float=lambda _: None,
            parse_int=lambda _: None,
            object_pairs_hook=lambda pairs: {k: v for k, v in pairs if k == "event"},
        ).get("event")
        for line in lines
    ]
    expected_events = (
        ["inherited"] * 123
        + ["registered"] * 2
        + ["evaluated"] * 2
        + ["verification_failed"] * 2
    )
    if events != expected_events:
        raise AssertionError(
            "Exact failed verification129-event chronological history required"
        )
    inherited = [
        {k: v for k, v in strict_json(line).items() if k != "event"}
        for line, event in zip(lines, events, strict=True)
        if event == "inherited"
    ]
    compare_tree(inherited, prior, "Original failed verification inherited123 identities")
    terminal = [
        {k: v for k, v in strict_json(line).items() if k != "event"}
        for line, event in zip(lines, events, strict=True)
        if event == "verification_failed"
    ]
    same_tree(terminal, rows, "Original final failure rows equal canonical p1 metrics")
    registered = [
        strict_json(line)
        for line, event in zip(lines, events, strict=True)
        if event == "registered"
    ]
    if [(r["candidate"], r["control"]) for r in registered] != list(COMPARISONS) or any(
        r["study"] != "civil_quarter_replay"
        or r["protocol_sha256"] != info["protocol_sha256"]
        or r["horizon"] != 1
        or r["score"] != "proper_variance"
        for r in registered
    ):
        raise AssertionError("Original failed verification registration differs")
    output_hashes = info["unverified_output_hashes"]
    required = {
        data + "/" + name
        for name in (
            "features.parquet",
            "targets.parquet",
            "forecasts.parquet",
            "fits.json",
            "source_closure.json",
            "support_audit.json",
            "upstream_admission.json",
        )
    }
    inventory = {str(p.relative_to(root)) for p in (root / data).rglob("*") if p.is_file()}
    if set(output_hashes) != required or inventory != required:
        raise AssertionError("Complete seven unverified outputs must remain identified")
    for name, signature in output_hashes.items():
        if pins.get(name) != signature:
            raise AssertionError("Unverified output hash absent from current registration")
        read_snapshot(root, name, signature)
    audit = saved("verification_failure_audit.json")
    counts = dict(Counter(events))
    canonical = audit["canonical_publication"]
    if (
        audit["status"] != "VERIFICATION_FAILURE_AND_PRESERVATION_AUDITED"
        or audit["protocol_sha256"] != info["protocol_sha256"]
        or audit["manifest_sha256"] != info["manifest_sha256"]
        or audit["freeze_record_sha256"] != info["freeze_record_sha256"]
        or any(
            audit[k] is not False
            for k in (
                "diagnostic_scores_read",
                "models_fitted",
                "verifier_rerun",
                "frozen_files_written",
            )
        )
        or canonical["status"] != "UNEVALUABLE"
        or canonical["verification_status"] != "FAILED"
        or any(
            canonical[k] is not True
            for k in (
                "metrics_equal_failure_json",
                "terminal_rows_equal_canonical_metrics",
                "whole_wave_aborted",
            )
        )
        or canonical["hypotheses"] != 2
        or canonical["cumulative_hypotheses"] != 125
        or canonical["all_new_pvalues"] != 1
        or canonical["phases_per_comparison"] != 0
        or canonical["leads"]
        or audit["ledger"]["events"] != 129
        or audit["ledger"]["event_counts"] != counts
        or audit["ledger"]["order_verified"] is not True
        or audit["ledger"]["inherited_and_evaluated_score_payloads_decoded"] is not False
    ):
        raise AssertionError("Independent failure preservation audit identity differs")
    outputs = audit["generated_outputs"]
    preservation = audit["current_preservation"]
    if (
        outputs["files_sha256"] != output_hashes
        or outputs["forecast_rows"] != info["generated_forecasts"]
        or outputs["monthly_fit_records"] != info["generated_monthly_fits"]
        or outputs["independently_verified_forecasts"] is not None
        or outputs["independently_verified_monthly_fits"] is not None
        or any(
            preservation[k]
            for k in ("pin_mismatches", "end_pin_mismatches", "prefit_design_mismatches")
        )
        or preservation["freeze_code_equals_manifest_code"] is not True
        or preservation["manifest_group_counts"]
        != {g: len(manifest[g]) for g in ("code", "inputs", "preserved")}
    ):
        raise AssertionError("Failed generated counts are not independent forecast validation")
    for name, signature in canonical["terminal_file_hashes"].items():
        if pins.get(name) != signature:
            raise AssertionError("Original canonical failure audit hash differs")
        read_snapshot(root, name, signature)
    publication = saved("publication_audit.json")
    if (
        publication["status"] != "UNEVALUABLE_REPORT_AUDITED"
        or publication["protocol_sha256"] != info["protocol_sha256"]
        or publication["manifest_sha256"] != info["manifest_sha256"]
        or publication["frozen_manifest_entries_checked"]
        != {g: len(manifest[g]) for g in ("code", "inputs", "preserved")}
        or publication["prefit_code_and_design_pins_unchanged"] is not True
        or publication["prior_failed_attempt_publication_unchanged"] is not True
        or publication["ledger_events"] != 129
        or publication["ledger_event_counts"] != counts
        or publication["generated_unverified_forecasts"] != info["generated_forecasts"]
        or publication["generated_unverified_monthly_fits"] != info["generated_monthly_fits"]
        or publication["cumulative_hypotheses"] != 125
        or publication["leads"]
        or {
            data + "/" + name: signature
            for name, signature in publication["private_generated_artifact_hashes"].items()
        }
        != output_hashes
    ):
        raise AssertionError("Original failed verification publication differs")
    for name, signature in publication["report_artifact_hashes"].items():
        path = report + "/" + name
        if pins.get(path) != signature:
            raise AssertionError("Original failed report artifact hash differs")
        read_snapshot(root, path, signature)
    return {
        "status": "FAILED_VERIFICATION_PRESERVED",
        "prior_files_written": False,
        "protocol_sha256": info["protocol_sha256"],
        "manifest_sha256": info["manifest_sha256"],
        "failure_sha256": info["failure_sha256"],
        "verification_sha256": info["verification_sha256"],
        "verification_failure_audit_sha256": info["verification_failure_audit_sha256"],
        "publication_audit_sha256": info["publication_audit_sha256"],
        "ledger_events_verified": counts,
        "unverified_output_hashes": output_hashes,
        "generated_forecasts": info["generated_forecasts"],
        "generated_monthly_fits": info["generated_monthly_fits"],
        "diagnostic_scores_read": False,
        "hypotheses_retained": 2,
        "all_new_pvalues": 1.0,
        "leads": [],
    }


# Literal contract is refreshed only before the final freeze.
CONTRACT = {
    "study_id": "profiled_quarter_wave17",
    "specified_on": "2026-09-07",
    "status": "specified_before_new_profiled_verification_replay",
    "wave": 17,
    "wave_alpha": 0.00016339869281045753,
    "objective": "Replay the civil quarter-end SPX risk question with independently profiled baseline "
    "verification and retain both failed attempts",
    "evidence_class": "exploratory_reused_history_archival_SPX_daily_risk_civil_calendar_interaction_profiled_verification",
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
        "path": "reports/civil_quarter_replay/NEXT_NUMERICAL_VERIFICATION.md",
        "sha256": "4e1d508593b78175634c39143160c1e8c7fc3c0956132b90bc0183c6484808bc",
    },
    "source_contract": {
        "admission": "Reconstruct complete original wave14 and wave8 VERIFIED records "
        "through frozen read-only functions with full active source "
        "dependency closure; separately preserve failed wave15 source "
        "admission and failed wave16 numerical verification with "
        "canonical p1 ledgers and exact original artifact hashes. Never "
        "invoke old writers or treat unverified diagnostic outputs as "
        "source observations.",
        "immutable_read": "Check selected metadata and all source bytes against "
        "registered hashes before decoding; stage immutable checked "
        "bytes for all original wave8 inputs, inherited metrics and "
        "the complete active document closure; normalize only "
        "documentary source_path prefixes back to originalroot; "
        "rehash every registered artifact before publication.",
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
        "inherited_hypotheses": 125,
        "cumulative_hypotheses": 127,
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
        "seed": 20260923,
        "inference_scale": 1.0,
        "seed_rule": "seed+phase_code*10000+block; development0,evaluation1",
        "p_phase": "Maximum two-sided centered-null circular-block bootstrap and "
        "BartlettHAC126 p",
        "p_hypothesis": "Maximum development/evaluation p",
        "multiplicity": "Holm2 at.05/(17*18), separately cumulativeHolm127 at.05",
        "gate": "Bothphase proper-score delta<=-.005 againstBOTHcontrols; bothfixed "
        "evaluationslice deltas<0; waveHolm<.05/306,cumulativeHolm<.05",
        "power": "Nominal HAC80percent minimumdetectableeffect dividedby.005; "
        "ordinary5percent diagnostic, noequivalence or adjustedpowerclaim",
        "failure_rule": "Any admission/support/fit/score/verification/publication failure "
        "leaves bothregistered hypotheses UNEVALUABLEp1; retain "
        "alloldresults and diagnosticfailures, no outcome-dependent repair",
    },
    "verification": {
        "baseline_method": "profiled_newton_directional_root",
        "baseline_initialization": "zero_slopes_analytic_intercept",
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
        "reconstruction": "Independent active source closure, original successful proof "
        "records and both failed wave15/wave16 p1 registrations, civil "
        "calendar, common maturity/support/rank/novelty, training "
        "transforms, profiled independent baseline with original "
        "full-gradient certification, scalar quarter root, strict "
        "saved native prediction replay normalized by exact trainmean, "
        "primitive proper losses/stable paired gaps, four-phase "
        "inference and complete127-family ledger; no new producer "
        "import",
        "profiled_solver": {
            "method": "profiled_newton_directional_brent",
            "alpha": 0.01,
            "controls": {
                "internal_tolerance": 1e-10,
                "full_tolerance": 1.0001e-08,
                "max_iterations": 500,
                "max_bracket_evaluations": 60,
                "max_root_iterations": 200,
                "root_xtol": 1e-14,
                "root_rtol": 1e-14,
                "small_difference": 0.5,
            },
            "design_path": "reports/profiled_quarter/SOLVER_DESIGN.md",
            "design_sha256": "a66eba8e65b0e01f3e6309d7c407ef4e52c7e98ee88e2ee95911c4074cefe299",
        },
    },
    "outputs": {
        "data": "data/profiled_quarter",
        "reports": "reports/profiled_quarter",
        "features": "data/profiled_quarter/features.parquet",
        "targets": "data/profiled_quarter/targets.parquet",
        "forecasts": "data/profiled_quarter/forecasts.parquet",
        "fits": "data/profiled_quarter/fits.json",
        "support_audit": "data/profiled_quarter/support_audit.json",
        "admission": "data/profiled_quarter/upstream_admission.json",
        "source_closure": "data/profiled_quarter/source_closure.json",
    },
    "failed_attempt": {
        "protocol": "civil_quarter.yaml",
        "reports": "reports/civil_quarter",
        "data": "data/civil_quarter",
        "protocol_sha256": "8b9fdabcbaf3eae80116cc1efa02e7d32372d9a628faf74996fceefe142da109",
        "manifest_sha256": "212578e5418c5f10e8e4847bbebf335bc3dcbd7b0db63aac853a4902a5605156",
        "freeze_record_sha256": "55eefd0f50d3486034e72c2cdc5ba4d4ed8b0015f39f1d92257c911ce99021ce",
        "failure_sha256": "67eb0c458001b726f1456453388283a99f22a2508424da4fdf56d535e9b0d159",
        "admission_failure_audit_sha256": "358a1655c1266ad0bb88ae0cd0f18e183693f83245e0f5afbe95484b7b9fd897",
        "publication_audit_sha256": "2553daee409f3e7ffad2c8204a20bf28a6d3aea9669e073fc0f468f30166d4a9",
        "trial_ledger_sha256": "d44aa0c1b9480cbc9dafae5cea26c0cbfaf1edfb584dcac41f856f8f417591a5",
        "required_status": "UNEVALUABLE",
        "inherited_hypotheses": 121,
        "registered_hypotheses": 2,
        "cumulative_hypotheses": 123,
        "terminal_event": "unevaluable",
        "new_forecasts": 0,
        "new_monthly_fits": 0,
        "admission": "Independently bind the entire failed protocol, freeze/manifest "
        "and current historical inventories, exact two p1 rows with no "
        "phases/leads, 121 inherited+2registered+2unevaluable ledger "
        "events and empty private outputs; reproduce no old writer and "
        "never infer null evidence from failed source packaging",
    },
    "source_closure": {
        "audit": "reports/civil_quarter/source_dependency_audit.json",
        "audit_sha256": "bf83a9bda409729a14fed45419ad081b9969d73766a91ed6a9fb2bc16a6853bb",
        "omitted_dependencies": {
            "data/source_discovery/bls_plan_capture/capture_manifest.json": "ab2a4d167d7e93e403af682dc30aca40cd05497816e626308fa7d8caa6f7c941",
            "data/source_discovery/bls_plan_capture/cpi_09112025.web-extract.txt": "5c5880834bf10c136631c68dfd023f8276ee29e795ad49d4caed95d0646e521c",
            "data/source_discovery/bls_plan_capture/cpi_12152015.web-extract.txt": "a29d4c780e626d86167de0584172b9e84c730cf2b91815372ebdba54dea40c00",
            "data/source_discovery/bls_plan_capture/empsit_09052025.web-extract.txt": "6483aaebdd8a4e320e5ca433ceaea8a2a9dcca9fb08c90cab87f71f16b1c863e",
            "data/source_discovery/bls_plan_capture/empsit_12042015.web-extract.txt": "6bacb96bc54d0242fe5a4c34a98a0b3c3b52d6216305e92a167240eddb2b5454",
        },
        "registration": "Register both successful upstream inventories, complete "
        "failed wave15 and wave16 inputs/reports/protocols and all "
        "five previously omitted declared documents before any real "
        "new closure collection or fitting",
        "active_roots": "Only selected final CPI/payroll ledgers, original annual FOMC "
        "plan table/coverage and final payroll artifact_sha256 "
        "capture-manifest binding; validate same metadata byte "
        "snapshots against registered pins before decoding",
        "runtime": "Every selected BLS source publication within the original fence "
        "including excluded/reissued/ambiguous records; all annual FOMC row "
        "references checked against coverage; enumerate complete "
        "source-reader closure, no fixed observed count or dropped document",
        "documentary": "Include "
        "data/source_discovery/bls_plan_capture/capture_manifest.json "
        "via exact final payroll artifact_sha256 binding. Retained "
        "SUPERSEDED intermediate declarations remain unchanged "
        "provenance, not active alternate hashes",
        "integrity": "Strict normalized relative paths beneath root, no "
        "absolute/traversal/symlink escape; unambiguous full SHA256 "
        "bindings; reject missing files, conflicting active declarations, "
        "current-pin disagreement or hash mismatch before text/JSON "
        "decoding",
        "snapshot": "Read and verify each staged input once into immutable bytes and "
        "write those exact snapshots at same-relative paths; preserve old "
        "parser/source gates and normalize only documentary source_path "
        "prefixes for audit comparison",
        "failure": "Any postregistration source closure/admission failure preserves "
        "both new hypotheses UNEVALUABLEp1; no outcome-dependent source "
        "deletion or gate repair",
    },
    "model_reuse": {
        "producer": {
            "src/civil_quarter_features.py": "1f04b9c191ffaec234e94e15ec2cf07aa87b262ecbd1fd3a90094c3aca554984",
            "src/civil_quarter_models.py": "9b0cc7375d7ee9b5af644559a2138158c76a603152a7fb7feb11b3bf6afa70d5",
            "src/civil_quarter_score.py": "310e80614a24cdc8e010d4bfea838cf913430f3b026e9542cbf9fc8854b1421e",
        },
        "independent_math": {
            "src/verify_civil_quarter.py": "8db7b064cc18bd22449f2c8ba5c03ca6c9a976a548d945c168ff678e8443dc4b",
            "src/profiled_quarter_solver.py": "3bdd8aa1f8f1ea521c50e1ac1b7102111c0c57c5945a0553489759289d3745b9",
        },
        "contract": "Reuse exact frozen wave15 producer feature/model/score "
        "implementations through explicit pure-function interfaces. "
        "Index/civil/support/fitting/scoring settings and independent "
        "acceptance gates are unchanged. Only independent baseline "
        "optimization is newly profiled; never patch module globals or alter "
        "prior files.",
    },
    "failed_verification": {
        "protocol": "civil_quarter_replay.yaml",
        "reports": "reports/civil_quarter_replay",
        "data": "data/civil_quarter_replay",
        "protocol_sha256": "99ff8102daf12a451f4bf2c50ddd4326dcf6b79873ad31a399c319b708ebac22",
        "manifest_sha256": "f4a680cb8036d1dc71b624eddff6bf0937f230f839e507a8deb8e89fe5da9dc9",
        "freeze_record_sha256": "d71d23623e7eee5bc97f7b2b480aa2d63853c74876f48791f311a01e6692fe9b",
        "failure_sha256": "ac6ab77e85fd037535f4031c0124539b3b7c088ec8f42a810ca7eea668787e08",
        "verification_sha256": "50022adbe5ecd4b23e17ae5972d5ec904064fb4d81a6ef0a1aa0be77558af0db",
        "verifier_sha256": "6a4de5f7504991c80a5bbc5fd315d18df61aad492a23d1edd39f25571ce49911",
        "verification_failure_audit_sha256": "49bb09229a2317dc5a33d0711389d1e0f554be4e17ba390676892a52dbe7c708",
        "numerical_failure_diagnosis_sha256": "13be3e97d9edfd71c3ba0a887c22ba5b0ecdbc0802255f7409ae886c9ec6d9e6",
        "publication_audit_sha256": "c0a667d96453ef9dbaa34c8c666ed2cad95c8af15e031bb2068f775418009a0c",
        "trial_ledger_sha256": "dc399e2c7d8048d12e6fed1f5f9549aef6e466ad5f93d1ff7febdaff401c6bab",
        "required_status": "UNEVALUABLE",
        "verification_status": "FAILED",
        "inherited_hypotheses": 123,
        "registered_hypotheses": 2,
        "cumulative_hypotheses": 125,
        "terminal_event": "verification_failed",
        "generated_forecasts": 6297,
        "generated_monthly_fits": 101,
        "unverified_output_hashes": {
            "data/civil_quarter_replay/features.parquet": "20c194f921087fa7f8e048e5ae2321dc2eddc5bfbc4b3fa059394a9dd7c151ff",
            "data/civil_quarter_replay/fits.json": "7075d573f98dd689c7f9357fd3f87a392eaea7c43713438796497433307cf1a8",
            "data/civil_quarter_replay/forecasts.parquet": "6c51fef0b230c443d14b65ecc33be832672d10437e6bb73268ba412af1e34075",
            "data/civil_quarter_replay/source_closure.json": "7e34f54d03baf70eba2e927ad9889ac4a54dcd59d589ab46a1862147e093ab83",
            "data/civil_quarter_replay/support_audit.json": "7e31adc7346fd3bd6ea40e6df744394c0ae847e95a972fa70c72705159aee3e1",
            "data/civil_quarter_replay/targets.parquet": "cb8425322f2d0ad3a3847a071cbbc81723397d9a85391d19471a020054b5bc1a",
            "data/civil_quarter_replay/upstream_admission.json": "5e8f4907d07e3a68170b382fb648d1e0a1fb92ef49562fe02957ce32127eb718",
        },
        "admission": "Bind complete frozen protocol/manifest/failure/publication "
        "and all generated artifact bytes; canonical two p1 rows "
        "with empty phases/leads, 123 "
        "inherited+2registered+2evaluated "
        "diagnostics+2verification_failed events; retain 6297 "
        "generated forecasts and101 fits as unverified, never "
        "consume their scores or rerun the failed verifier",
    },
}


def validate_protocol(protocol):
    if protocol != CONTRACT:
        raise AssertionError("Complete registered profiled verification contract differs")
    if (
        protocol["verification"]["profiled_solver"].get("specification_status")
        == "SYNTHETIC_DESIGN_PENDING_FINAL_CONSTANTS"
    ):
        raise AssertionError("Independent solver specification is not yet final")


def validate_solver_registration(root, protocol, manifest):
    section = protocol["verification"]["profiled_solver"]
    controls = {
        "internal_tolerance": profiled.INTERNAL_TOLERANCE,
        "full_tolerance": profiled.FULL_TOLERANCE,
        "max_iterations": profiled.MAX_ITERATIONS,
        "max_bracket_evaluations": profiled.MAX_BRACKET_EVALUATIONS,
        "max_root_iterations": profiled.MAX_ROOT_ITERATIONS,
        "root_xtol": profiled.ROOT_XTOL,
        "root_rtol": profiled.ROOT_RTOL,
        "small_difference": profiled.SMALL_DIFFERENCE,
    }
    if (
        section["method"] != "profiled_newton_directional_brent"
        or section["alpha"] != profiled.ALPHA
        or section["controls"] != controls
        or manifest["inputs"].get(section["design_path"]) != section["design_sha256"]
    ):
        raise AssertionError("Independent solver controls and design must be registered")
    read_snapshot(root, section["design_path"], section["design_sha256"])
    for group in ("producer", "independent_math"):
        for name, signature in protocol["model_reuse"][group].items():
            if manifest["code"].get(name) != signature:
                raise AssertionError("Every reused mathematical source must remain pinned")
            read_snapshot(root, name, signature)
    if "src/profiled_quarter_solver.py" not in protocol["model_reuse"]["independent_math"]:
        raise AssertionError("The new independent solver requires its own declared code pin")


def validate_upstream(root=ROOT):
    root = Path(root)
    protocol_payload = (root / "profiled_quarter.yaml").read_bytes()
    protocol = yaml.safe_load(protocol_payload)
    validate_protocol(protocol)
    manifest_payload = (root / "reports/profiled_quarter/manifest.json").read_bytes()
    captured = strict_json(manifest_payload)
    if captured["protocol_sha256"] != sha256(protocol_payload).hexdigest():
        raise AssertionError("New registration must precede closure collection and admission")
    pins = {}
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, captured[group])
        for name, signature in captured[group].items():
            if name in pins and pins[name] != signature:
                raise AssertionError("Conflicting registered replay artifact hashes")
            pins[name] = signature
    validate_solver_registration(root, protocol, captured)
    source = protocol["feature_source"]
    oldp = yaml.safe_load(read_snapshot(root, source["protocol"], source["protocol_sha256"]))
    closure = collect_sources(root, oldp, captured["inputs"])
    if any(
        captured["inputs"].get(name) != signature
        for name, signature in closure["files"].items()
    ):
        raise AssertionError(
            "The full active closure must be directly registered before collection"
        )
    for name, signature in protocol["source_closure"]["omitted_dependencies"].items():
        if (
            closure["files"].get(name) != signature
            or captured["inputs"].get(name) != signature
        ):
            raise AssertionError(
                "Every fixed previously omitted declaration must be recovered exactly"
            )
    audit_name = protocol["source_closure"]["audit"]
    if pins.get(audit_name) != protocol["source_closure"]["audit_sha256"]:
        raise AssertionError("Original dependency audit identity differs")
    read_json_snapshot(root, audit_name, pins[audit_name])
    closure_name = protocol["outputs"]["source_closure"]
    closure_hash = digest(root / closure_name)
    same_tree(
        read_json_snapshot(root, closure_name, closure_hash),
        closure,
        "Independent complete active source closure",
    )

    def inventory(info):
        return {info["protocol"]} | {
            str(path.relative_to(root))
            for folder in (info["reports"], info["data"])
            for path in (root / folder).rglob("*")
            if path.is_file()
        }

    inventories = {
        key: inventory(protocol[key])
        for key in ("upstream", "feature_source", "failed_attempt", "failed_verification")
    }
    for key in inventories:
        info = protocol[key]
        if not inventories[key].issubset(captured["inputs"]):
            raise AssertionError(
                "Entire historical protocol/report/private inventory must be pinned"
            )
        oldm = read_json_snapshot(
            root, info["reports"] + "/manifest.json", info["manifest_sha256"]
        )
        if not set(oldm["code"]).issubset(captured["code"]) or not set(
            oldm["inputs"]
        ).issubset(captured["inputs"]):
            raise AssertionError("Original code and input registration must remain complete")
        for group in ("code", "inputs", "preserved"):
            for name, signature in oldm[group].items():
                if pins.get(name) != signature:
                    raise AssertionError(
                        "Original preserved artifact missing from replay registration"
                    )
                read_snapshot(root, name, signature)
    proofs = {}
    for key, module in (("upstream", "causal_pool"), ("feature_source", "calendar_variance")):
        info = protocol[key]
        oldm = read_json_snapshot(
            root, info["reports"] + "/manifest.json", info["manifest_sha256"]
        )
        record = read_json_snapshot(
            root, info["reports"] + "/verification.json", info["verification_sha256"]
        )
        if (
            (root / info["reports"] / "failure.json").exists()
            or info["required_status"] != "VERIFIED"
            or record["status"] != "VERIFIED"
            or record["protocol_sha256"] != info["protocol_sha256"]
            or record["verifier_sha256"] != info["verifier_sha256"]
            or oldm["protocol_sha256"] != info["protocol_sha256"]
            or oldm["code"]["src/verify_" + module + ".py"] != info["verifier_sha256"]
        ):
            raise AssertionError("Successful historical verification identity differs")
        read_snapshot(root, info["protocol"], info["protocol_sha256"])
        nested = None
        if key == "upstream":
            rebuilt, nested = reconstruct_previous(root, info, pins)
        else:
            rebuilt = reconstruct_calendar(root, info, pins, closure)
        compare_tree(record, rebuilt, "Complete original " + module + " VERIFIED proof")
        proofs[key] = {
            "original_verification": rebuilt,
            "previous_manifest_entries_verified": sum(
                len(oldm[g]) for g in ("code", "inputs", "preserved")
            ),
            "pinned_previous_artifacts": len(inventories[key]),
            "protocol_sha256": info["protocol_sha256"],
            "manifest_sha256": info["manifest_sha256"],
            "verification_sha256": info["verification_sha256"],
            **({"nested_admission": nested} if nested is not None else {}),
        }
    proofs["failed_attempt"] = verify_failed_attempt(root, protocol["failed_attempt"], pins)
    proofs["failed_verification"] = verify_failed_verification(
        root, protocol["failed_verification"], pins
    )
    for key, initial in inventories.items():
        if inventory(protocol[key]) != initial:
            raise AssertionError("Historical inventory changed during read-only admission")
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, captured[group])
    if (
        digest(root / "profiled_quarter.yaml") != sha256(protocol_payload).hexdigest()
        or digest(root / "reports/profiled_quarter/manifest.json")
        != sha256(manifest_payload).hexdigest()
        or digest(root / closure_name) != closure_hash
    ):
        raise AssertionError("Registered replay/closure identity changed during admission")
    return {
        "status": "UPSTREAM_VERIFIED_READ_ONLY",
        "prior_files_written": False,
        "input_hashes": captured["inputs"],
        "source_closure": closure,
        "proofs": proofs,
    }


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    code = {
        str(path.relative_to(root))
        for folder in ("src", "tests")
        for path in (root / folder).rglob("*.py")
    }
    inputs = set(protocol["comparisons"]["inherited_sources"]) | set(
        protocol["source_closure"]["omitted_dependencies"]
    )
    inputs.add(protocol["verification"]["profiled_solver"]["design_path"])
    for key in ("upstream", "feature_source", "failed_attempt", "failed_verification"):
        info = protocol[key]
        old = read_json_snapshot(
            root, info["reports"] + "/manifest.json", info["manifest_sha256"]
        )
        code.update(old["code"])
        inputs.update(old["inputs"])
        inputs.add(info["protocol"])
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
        and path != root / "profiled_quarter.yaml"
        and root / "reports/profiled_quarter" not in path.parents
        and str(path.relative_to(root)) not in inputs
    }
    if (
        set(manifest["code"]) != code
        or set(manifest["inputs"]) != inputs
        or set(manifest["preserved"]) != preserved
        or set(manifest["inputs"]) & set(manifest["preserved"])
    ):
        raise AssertionError(
            "Exact new code, dual-upstream inputs and preserved corpus required"
        )


def verify(root=ROOT):
    root = Path(root)
    payload = (root / "profiled_quarter.yaml").read_bytes()
    protocol = yaml.safe_load(payload)
    validate_protocol(protocol)
    protocol_hash = sha256(payload).hexdigest()
    report, out = root / "reports/profiled_quarter", root / "data/profiled_quarter"
    manifest_payload = (report / "manifest.json").read_bytes()
    manifest_hash = sha256(manifest_payload).hexdigest()
    manifest = json.loads(manifest_payload)
    if manifest["protocol_sha256"] != protocol_hash:
        raise AssertionError("Frozen protocol and manifest differ")
    verify_manifest_coverage(root, protocol, manifest)
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, manifest[group])
    names = [
        str((out / name).relative_to(root))
        for name in (
            "upstream_admission.json",
            "source_closure.json",
            "support_audit.json",
            "features.parquet",
            "targets.parquet",
            "forecasts.parquet",
            "fits.json",
        )
    ] + [
        str((report / name).relative_to(root))
        for name in ("metrics.json", "trial_ledger.jsonl")
    ]
    snapshots = {name: digest(root / name) for name in names}

    def saved_json(name):
        return read_json_snapshot(root, name, snapshots[name])

    def saved_parquet(name):
        return scalar.read_issued_snapshot(root, name, snapshots[name])

    oldinfo = protocol["feature_source"]
    oldp = yaml.safe_load(read_snapshot(root, oldinfo["protocol"], oldinfo["protocol_sha256"]))
    closure = collect_sources(root, oldp, manifest["inputs"])
    same_tree(
        saved_json("data/profiled_quarter/source_closure.json"),
        closure,
        "Independent complete source closure artifact",
    )
    proof = validate_upstream(root)
    same_tree(
        saved_json("data/profiled_quarter/upstream_admission.json"),
        proof,
        "Dual original admission proof",
    )
    source = protocol["feature_source"]
    original = scalar.read_issued_snapshot(
        root, source["features"], manifest["inputs"][source["features"]]
    )
    targets = scalar.read_issued_snapshot(
        root, source["targets"], manifest["inputs"][source["targets"]]
    )
    features = augment_features(original)
    table_equal(
        saved_parquet("data/profiled_quarter/features.parquet"),
        features,
        ALL_FEATURES,
        "Independent raw civil augmentation",
    )
    table_equal(
        saved_parquet("data/profiled_quarter/targets.parquet"),
        targets,
        ("y",),
        "Unchanged original target table",
    )
    support = preflight(features, targets, protocol)
    same_tree(
        saved_json("data/profiled_quarter/support_audit.json"),
        support,
        "Every preoptimization support/rank/novelty check",
    )
    panel = saved_parquet("data/profiled_quarter/forecasts.parquet")
    fits = saved_json("data/profiled_quarter/fits.json")
    metrics = saved_json("reports/profiled_quarter/metrics.json")
    if (
        (report / "failure.json").exists()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != protocol_hash
        or metrics["evidence_class"] != protocol["evidence_class"]
    ):
        raise AssertionError("New publication failed or identity differs")
    result = {
        "status": "VERIFIED",
        "upstream_admission": {"status": proof["status"], "prior_files_written": False},
        "source_closure_verified": closure["counts"],
        "failed_attempt_preserved": proof["proofs"]["failed_attempt"],
        "failed_verification_preserved": proof["proofs"]["failed_verification"],
        "raw_feature_rows_verified": len(features),
        "raw_feature_columns_verified": len(ALL_FEATURES),
        "support_audit_verified": {
            "monthly_fits": support["monthly_fits"],
            "phases": len(support["phases"]),
            "evaluation_slices": len(support["phases"][1]["slices"]),
        },
        "forecast_reconstruction": verify_forecasts(features, targets, panel, fits, protocol),
        "primitive_proper_scores_verified": len(panel),
        "paired_proper_differences_verified": int(panel.origin.nunique()) * 2,
        "inference": verify_metrics(
            root,
            panel,
            features,
            protocol,
            metrics,
            len(fits),
            support["common_application_origins"],
            admitted_inputs=manifest["inputs"],
        ),
        "ledger_events_verified": verify_ledger(
            root,
            metrics,
            metrics["inherited_rows"],
            "evaluated",
            signature=snapshots["reports/profiled_quarter/trial_ledger.jsonl"],
        ),
        "protocol_sha256": protocol_hash,
        "verifier_sha256": digest(Path(__file__)),
        "limitations": [
            "Independent intercept-profiled baseline verification; original full-gradient and prediction gates retained. One staged civil interaction conditional on a newly fitted market, macro, month, ordinary month-end and December year-end baseline; no joint refit in the quarter arm.",
            "Nominal next-weekday civil arithmetic ignores holidays and early closes; unchanged target maturity uses the next actual observed SPX session.",
            "The daily floored Garman-Klass plus overnight-squared-return target remains an OHLC risk proxy; original-plan missingness, archival revisions and back-calculated VIX9D persist.",
            "Adaptively selected exploratory reuse of prior historical periods; proper-score gains are absolute natural-log-score units, not percentages, causal mechanisms or profit.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, manifest[group])
    scalar._pins(root, snapshots)
    verify_manifest_coverage(root, protocol, manifest)
    if (
        digest(root / "profiled_quarter.yaml") != protocol_hash
        or digest(report / "manifest.json") != manifest_hash
    ):
        raise AssertionError("Protocol or manifest changed during verification")
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False), flush=True)
