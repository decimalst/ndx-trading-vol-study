"""Independent wave21 entry and inference for immutable failed-wave20 outputs.

Only source metadata traversal is shared. All feature/stage replay and inference
are frozen independent implementations, with the separately declared date seam.
No original or candidate producer model is fit by this module.
"""

from __future__ import annotations

import json
import math
import re
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml

from . import verify_event_cluster as frozen

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/event_cluster_replay"
OUT = "data/event_cluster_replay"
PROTOCOL = "event_cluster_replay.yaml"
MODELS = frozen.MODELS
PANEL_COLUMNS = frozen.PANEL_COLUMNS
COMPARISONS = frozen.COMPARISONS
EFFECT = 0.0005
WAVE_ALPHA = 0.05 / (21 * 22)
phase_statistics = frozen.phase_statistics
validate_panel = frozen.validate_panel
compare_tree = frozen.compare_tree
same_tree = frozen.same_tree
holm = frozen.holm
support = frozen.support
strict_json = frozen.strict_json
source_path = frozen.source_path
hash_mapping = frozen.hash_mapping
digest = frozen.digest
read_snapshot = frozen.read_snapshot
read_json_snapshot = frozen.read_json_snapshot
pins_checked = frozen.pins_checked
require_upstream_success = frozen.require_upstream_success

CONTRACT = {'study_id': 'event_cluster_replay_wave21',
 'specified_on': '2026-09-07',
 'status': 'specified_before_retained_output_reconstruction_or_inference',
 'wave': 21,
 'evidence_class': 'exploratory_technical_replay_of_unverified_archival_SPX_event_clustering',
 'objective': 'Complete independent verification of exact retained wave20 producer output '
              'under lossless application-state date representation',
 'original_index': {'asset': 'SPX',
                    'source_end': '2025-10-20',
                    'sealed_start': '2025-11-03',
                    'origin_start': '2016-01-04',
                    'origin_end': '2025-10-17',
                    'latest_target': '2025-10-20',
                    'development': ['2016-01-04', '2019-12-31'],
                    'development_target_available_by': '2019-12-31',
                    'evaluation': ['2020-01-02', '2025-10-17'],
                    'evaluation_stability': [['2020-01-02', '2022-12-31'],
                                             ['2023-01-01', '2025-10-17']],
                    'horizons': [1],
                    'market_lag': 1,
                    'minimum_train': 1000,
                    'models': ['baseline', 'recent_frequency', 'location'],
                    'minimum_train_per_class': 50,
                    'minimum_phase_per_class': 30,
                    'minimum_slice_per_class': 15,
                    'minimum_phase_observations': 127,
                    'baseline': ['const',
                                 'I',
                                 'R',
                                 'ret_d',
                                 'ret_w',
                                 'ret_m',
                                 'ret_q',
                                 'lr_d',
                                 'lr_w',
                                 'term',
                                 'lvvix',
                                 'entry_dow_1',
                                 'entry_dow_2',
                                 'entry_dow_3',
                                 'entry_dow_4',
                                 'neg_d',
                                 'neg_w',
                                 'neg_m',
                                 'skew',
                                 'I_square',
                                 'R_square',
                                 'skew_square',
                                 'intraday',
                                 'intraday_sq',
                                 'overnight',
                                 'overnight_sq'],
                    'all_features': ['const',
                                     'I',
                                     'R',
                                     'ret_d',
                                     'ret_w',
                                     'ret_m',
                                     'ret_q',
                                     'lr_d',
                                     'lr_w',
                                     'term',
                                     'lvvix',
                                     'entry_dow_1',
                                     'entry_dow_2',
                                     'entry_dow_3',
                                     'entry_dow_4',
                                     'neg_d',
                                     'neg_w',
                                     'neg_m',
                                     'skew',
                                     'intraday',
                                     'intraday_sq',
                                     'overnight',
                                     'overnight_sq',
                                     'range_extremity']},
 'history': {'window': 22,
             'ordered_by': 'available_date on full original SPX calendar, oldest first, '
                           'through previous-close cutoff',
             'origin_position_rule': 'at i>=23 use y.iloc[i-23:i-1], whose available '
                                     'positions are i-22..i-1',
             'labels': 'all known original target labels, including '
                       'unissued/unscored/feature-incomplete origins; not historical forecast '
                       'errors',
             'unknown': 'every original training/application window must contain22 known '
                        'labels; otherwise whole-trial failure before fitting; never fill, '
                        'compress or drop original rows',
             'nuisance': ['event_fraction22',
                          'event_fraction22_sq',
                          'last_event',
                          'expected_adjacency22',
                          'linear_recency22'],
             'nuisance_formula': ['N/22',
                                  'N*N/484',
                                  'last_event',
                                  '(N-last_event)*(N-1)/441',
                                  'sum((2*i-23)*e_i for i=1..22)/462'],
             'candidate': 'adjacency_fraction22',
             'candidate_formula': 'C/21; C=sum(e_i*e_(i-1) for i=2..22)',
             'diagnostic': 'excess_adjacency22=(21*C-(N-last_event)*(N-1))/441; never enters '
                           'fit or promotion',
             'arithmetic': 'exact Python integer numerators followed by single fixed '
                           'division; N*N/484 is not square of rounded N/22',
             'centering': 'training-only math.fsum/n; exactconstant uses firstvalue; '
                          'fixedscale1 and no rank/column deletion'},
 'models': ['baseline', 'recent_frequency', 'nuisance', 'cluster'],
 'model': {'baseline': 'original wave18 saved26-coefficient monthly fit; no refitting; '
                       'original issuedquery probabilities strictlyreplayed and retained',
           'training_offset': 'apply that same savedmonthlyfit to its original '
                              'maturetrainingrows; in-sample currentfit offsets, not '
                              'historicalissuances',
           'nuisance': 'mean Bernoulli NLL with frozenbaselineoffset +freeintercept +five '
                       'centeredboundedslopes, .01sum(slope^2)',
           'candidate': 'mean Bernoulli NLL with frozennuisanceoffset +one centeredC/21 '
                        'slope, .01coefficient^2; no newintercept',
           'ridge': 0.01,
           'producer_nuisance': {'method': 'frozen sign_memory_models._newton',
                                 'start': 'six literalzeros',
                                 'maxiter': 200,
                                 'max_backtracks': 60,
                                 'armijo': 0.0001,
                                 'gradient_tolerance': 1e-08},
           'producer_scalar': {'method': 'brentq',
                               'bracket': '+/- (mean(abs(centered_candidate))+1)/.02',
                               'xtol': 1e-12,
                               'rtol': 1e-14,
                               'maxiter': 200,
                               'gradient_tolerance': 1e-08},
           'constant': 'center exactconstants exactly; constantnuisances '
                       'retainzerocoefficients; constantadjacency retainsscalarzero and '
                       'nuisanceprobability',
           'probability': 'copy checkedparent probability when addedquerycorrection '
                          'exactlyzero; otherwise expit(parentlogit+correction); no clipping',
           'scalar_arithmetic': 'finitefloat64; signedsoftplus and signedresidual remain '
                                'nonzero for finite logits; checked nonzero products '
                                'rejectunderflow; compensated scalar reductions',
           'nuisance_arithmetic': 'frozen logistic_state numerical contract; finite '
                                  'objective, fullgradient and Hessian; no optional '
                                  'unusedcurvature admission gate',
           'fallback': 'none; no empiricalrepair or tolerance change'},
 'comparisons': {'new_hypotheses': 3,
                 'inherited_hypotheses': 134,
                 'cumulative_hypotheses': 137,
                 'inherited_sources': ['reports/orthogonal_round2/metrics.json',
                                       'reports/model_memory_study/combined_metrics.json',
                                       'reports/iterative_signal_search/metrics.json',
                                       'reports/international_volatility/metrics.json',
                                       'reports/overnight_index/metrics.json',
                                       'reports/macro_overnight/metrics.json',
                                       'reports/measurement_memory/metrics.json',
                                       'reports/index_hinge/metrics.json',
                                       'reports/tail_shape/metrics.json',
                                       'reports/calendar_variance/metrics.json',
                                       'reports/relative_risk/metrics.json',
                                       'reports/joint_risk/metrics.json',
                                       'reports/cross_moment/metrics.json',
                                       'reports/target_aligned/metrics.json',
                                       'reports/sign_memory/metrics.json',
                                       'reports/causal_pool/metrics.json',
                                       'reports/civil_quarter/metrics.json',
                                       'reports/civil_quarter_replay/metrics.json',
                                       'reports/profiled_quarter/metrics.json',
                                       'reports/range_alert/metrics.json',
                                       'reports/issued_calibration/metrics.json',
                                       'reports/event_cluster/metrics.json'],
                 'contrasts': [['cluster', 'baseline', 'brier'],
                               ['cluster', 'nuisance', 'brier'],
                               ['cluster', 'recent_frequency', 'brier']],
                 'gate': 'allthreecontrols must improve in bothphases by>=.0005 Brier, '
                         'bothfixedlateslices negative, bothmultiplicity gates; '
                         'nootherpromotion contrast'},
 'inference': {'loss': 'brier',
               'effect_absolute': 0.0005,
               'blocks': [21, 63, 126],
               'hac_lags': 126,
               'bootstrap_draws': 399999,
               'seed': 20260926,
               'seed_rule': 'seed+phase_code*10000+block; development0,evaluation1; same '
                            'drawdesign acrosscontrols',
               'wave_alpha': 0.00010822510822510823,
               'cumulative_alpha': 0.05,
               'p_phase': 'maximum two-sided centered-null circularblock bootstrap and '
                          'BartlettHAC126p',
               'p_hypothesis': 'maximumdevelopment/evaluationp',
               'multiplicity': 'Holm3 versus wave_alpha and cumulativeHolm137 versus.05',
               'resolution': '1/400000 below one tenth minimumHolm3rawwavecutoff1/27720',
               'support': {'minimum_train': 1000,
                           'train_per_class': 50,
                           'phase_observations': 127,
                           'phase_per_class': 30,
                           'slice_per_class': 15},
               'failure': 'any new source, representation, reconstruction, support, '
                          'arithmetic, solver, forecast, inference or publication failure '
                          'keeps all3 UNEVALUABLE p1; retain original wave20 failure and all '
                          'earlier hypotheses',
               'calibration': 'descriptivein-the-large meanprob,eventrate,gap,Brier '
                              'perphase/model; no fittedcalibrationbins',
               'power': 'nominal HAC80percentMDE/.0005 descriptive; not equivalence or '
                        'post-hocselection'},
 'verification': {'nuisance': {'method': 'root_hybr',
                               'start': 'six literalzeros',
                               'xtol': 1e-10,
                               'maxfev': 2000,
                               'factor': 1.0,
                               'analytic_jacobian': True,
                               'full_gradient_tolerance': 1.0001e-08},
                  'scalar': {'method': 'bisect',
                             'bracket': '+/- (mean(abs(centered_candidate))+1)/.02',
                             'xtol': 1e-12,
                             'rtol': 1e-14,
                             'maxiter': 200,
                             'full_gradient_tolerance': 1.0001e-08},
                  'coefficient_rtol': 1e-07,
                  'coefficient_atol': 1e-06,
                  'replay_rtol': 1e-10,
                  'replay_atol': 1e-12,
                  'inference_rtol': 1e-09,
                  'inference_atol': 1e-14,
                  'conditional_scalar': 'independentlysolve using saved '
                                        'independentlyvalidated nuisancebeta, '
                                        'preservingactualfrozenoffsetobjective',
                  'scope': 'fullnewhistory, originalcohortsandissuedcontrols, '
                           'allnewmonthlystages/applicationpredictionsincludingunscored, '
                           'scores,inference,ledger,sourceclosure'},
 'upstream': {'anchors': {'event_cluster.yaml': '824a9beda7cf293568dd2d16bbd3b76a8235f359a78c764147d75731c22b734c',
                          'reports/event_cluster/publication_audit.json': '0d901a938c6a88ff4dbf1bb68d114640a50279e38826be29c97ea167ca56be15',
                          'reports/event_cluster/failure_review.json': 'ae554fd105b5a48b50739af4a6e64c3a8895c3e94bb14fbeb0230490d88d3408'},
              'admission': 'complete exact failed wave20 preservation envelope and retained '
                           'output hashes; full original verified wave19/wave18 closure; '
                           'checked buffers before decoding'},
 'outputs': {'data': 'data/event_cluster_replay', 'reports': 'reports/event_cluster_replay'},
 'replay': {'retained_producer_protocol_sha256': '824a9beda7cf293568dd2d16bbd3b76a8235f359a78c764147d75731c22b734c',
            'input_role': 'exact pinned UNVERIFIED wave20 producer artifacts; original '
                          'wave18/19 source/control proofs remain verified',
            'newly_generated_forecasts': 0,
            'new_producer_fits': 0,
            'no_producer_entrypoint_or_optimizer_calls': True,
            'sequence': 'register, admit checked retained/source buffers, full independent '
                        'history/cohort/stage/application reconstruction, then score and '
                        'infer; separate guarded verifier repeats reconstruction/inference',
            'mathematics': 'unchanged wave20 history, baseline replay, nuisance/scalar '
                           'objectives, original coefficients, centers, solver budgets and '
                           'numeric tolerances; verification optima never replace retained '
                           'predictions',
            'artifact_policy': 'no rewriting, coercion, replacement or reserialization of '
                               'retained source files; ephemeral date comparison views only',
            'old_failure_policy': 'keep canonical wave20 UNEVALUABLE/FAILED and all3p1 '
                                  'forever; its evaluated/unpublished scores are opaque '
                                  'preservation evidence only',
            'counting': 'report retained candidate/control rows, retained monthly schedules '
                        'and independently verified stage optima separately from zero new '
                        'producer fits/forecasts'},
 'date_representation': {'scope': 'only seven declared full-application-state datetime '
                                  'columns at final exact-frame comparison; all other '
                                  'frame/index comparisons unchanged',
                         'fields': ['origin',
                                    'feature_cutoff_date',
                                    'source_fit_origin',
                                    'window_first_available',
                                    'window_last_available',
                                    'window_first_origin',
                                    'window_last_origin'],
                         'accepted_units': ['ms', 'us', 'ns'],
                         'timezone': 'naive only',
                         'known_values': 'normalized midnight calendar dates; unchanged '
                                         'calendar, source, phase and maturity fences',
                         'comparison': 'checked exact nanosecond integer representation with '
                                       'lossless roundtrip to original unit; identical '
                                       'unknown masks; reject overflow or any rounding',
                         'unchanged_structure': 'exact columns/order/index/row order/names; '
                                                'exact numeric boolean integer and nondate '
                                                'dtypes/values; no blanket dtype suppression',
                         'rejected': 'object/string/timezone/subday dates, changed instants, '
                                     'unknown masks, overflow, row/schema/numeric dtype '
                                     'changes',
                         'audit': 'record original units and successful lossless comparisons '
                                  'without modifying retained artifacts'}}


def validate_protocol(protocol):
    same_tree(protocol, CONTRACT, "literal typed event-cluster replay protocol")


def require_failed_metrics(decoded):
    if (decoded.get("status") != "UNEVALUABLE"
        or decoded.get("whole_wave_aborted") is not True
        or decoded.get("leads") != []):
        raise ValueError("Canonical failed wave20 identity required")
    rows = decoded.get("rows")
    if type(rows) is not list or len(rows) != 3:
        raise ValueError("Exactly three failed wave20 hypotheses required")
    for row, (candidate, control) in zip(rows, COMPARISONS, strict=True):
        same_tree({k:row[k] for k in ("study","candidate","control","horizon","score")},
                  {"study":"event_cluster","candidate":candidate,"control":control,
                   "horizon":1,"score":"brier"}, "failed wave20 hypothesis identity")
        if row.get("verdict") != "UNEVALUABLE":
            raise ValueError("Old failure remains unevaluable")
        for key in ("p_conservative", "p_holm_wave", "p_holm_cumulative"):
            value = row[key]
            if type(value) not in (int,float) or value != 1:
                raise ValueError("All three old canonical failure p-values must remain1")


def inherited_rows(root, protocol, *, pins=None):
    root = Path(root)
    if pins is None:
        pins = strict_json((root / REPORT / "manifest.json").read_bytes())["inputs"]
    names = protocol["comparisons"]["inherited_sources"]
    if len(names) != len(set(names)) or names.count("reports/event_cluster/metrics.json") != 1:
        raise ValueError("Exactly one identified failed wave20 canonical metrics input required")
    output = []
    for name in names:
        signature = hash_mapping(pins)[name]
        decoded = read_json_snapshot(root, name, signature)
        if name == "reports/event_cluster/metrics.json":
            require_failed_metrics(decoded)
        for number, row in enumerate(decoded["rows"]):
            pvalue = row["p_conservative"]
            if type(pvalue) not in (int,float) or not math.isfinite(pvalue) or not 0 <= pvalue <= 1:
                raise ValueError("Finite real inherited p-value required")
            output.append({
                "study": row.get("study",Path(name).parent.name or Path(name).stem),
                "candidate":row["candidate"],"control":row.get("control","baseline"),
                "horizon":row["horizon"],"p_conservative":pvalue,"source":name,
                "source_sha256":signature,"source_row_index":number,
                **{k:row[k] for k in ("measure","score") if k in row},
            })
    if len(output) != 134:
        raise ValueError("All134 prior hypotheses including the failed three must remain")
    return output


def verify_metrics(
    root, panel, protocol, metrics, application_n, monthly_fits, *, admitted_inputs=None
):
    validate_panel(panel)
    if type(monthly_fits) is not int or monthly_fits <= 0:
        raise ValueError("Positive literal verified monthly fit count required")
    n = panel.origin.nunique()
    if type(application_n) is not int or application_n < n:
        raise ValueError("Literal complete application count required")
    expected_counts = {
        "newly_generated_forecasts": 0,
        "new_producer_fits": 0,
        "retained_candidate_scored_forecasts": 2 * n,
        "original_control_forecasts": 2 * n,
        "combined_retained_forecasts": len(panel),
        "common_application_origins": application_n,
        "common_scored_origins": n,
        "retained_monthly_schedules": monthly_fits,
        "independent_stage_fits_verified": 2 * monthly_fits,
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 137,
    }
    same_tree(
        {key: metrics[key] for key in expected_counts},
        expected_counts,
        "literal output/family counts",
    )
    section = protocol["original_index"]
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
        raise AssertionError("All three fixed Brier controls required")
    probabilities = []
    effects = []
    for row in rows:
        if row["study"] != "event_cluster_replay" or row["horizon"] != 1 or row["score"] != "brier":
            raise AssertionError("Event-cluster Brier comparison identity differs")
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
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-3:]
    passed = []
    for number, row in enumerate(rows):
        compare_tree(row["p_holm_wave"], wave[number], "Three-hypothesis wave correction", "p")
        compare_tree(
            row["p_holm_cumulative"],
            cumulative[number],
            "137-hypothesis cumulative correction",
            "p",
        )
        one = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05)
        if row["verdict"] != ("PASSES_ALL_GATES" if one else "DOES_NOT_QUALIFY"):
            raise AssertionError("Fixed statistical/effect/stability gate differs")
        passed.append(one)
    leads = ["cluster"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 3
        or metrics["cumulative_hypothesis_count"] != 137
    ):
        raise AssertionError("Complete candidate and hypothesis family required")
    return {
        "new_hypotheses_verified": 3,
        "cumulative_hypotheses_verified": 137,
        "phase_comparisons_verified": 6,
        "bootstrap_runs_verified": 18,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "calibration_rows_verified": 8,
        "class_support": counts,
        "leads": leads,
    }


def verify_ledger(root, metrics, prior, final_event, *, signature=None):
    report = Path(root) / "reports/event_cluster_replay"
    payload = (report / "trial_ledger.jsonl").read_bytes()
    if signature is not None and sha256(payload).hexdigest() != signature:
        raise AssertionError("Trial ledger bytes changed before decoding")
    ledger = [strict_json(line) for line in payload.splitlines() if line]
    for event, expected in (("inherited", prior), (final_event, metrics["rows"])):
        actual = [
            {key: value for key, value in row.items() if key != "event"}
            for row in ledger
            if row["event"] == event
        ]
        same_tree(actual, expected, "Independent " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (
        len(ledger) != 140
        or len(registered) != 3
        or len(prior) != 134
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "event_cluster_replay"
            or row["horizon"] != 1
            or row["score"] != "brier"
            for row in registered
        )
    ):
        raise AssertionError("Complete134inherited+3registered+3terminal ledger required")
    return {"inherited": 134, "registered": 3, final_event: 3}


def verify_manifest_coverage(root_path, protocol, manifest, closure):
    root_path = Path(root_path)
    code = {
        str(path.relative_to(root_path))
        for folder in ("src", "tests")
        for path in (root_path / folder).rglob("*.py")
    }
    inputs = set(closure["files"]) | set(protocol["comparisons"]["inherited_sources"])
    name = REPORT + "/freeze_record.json"
    frozen = read_json_snapshot(root_path, name, manifest["inputs"][name])
    inputs.add(name)
    inputs.update(frozen["prefit_design"])
    same_tree(frozen["protocol_sha256"], manifest["protocol_sha256"], "new prefit protocol")
    same_tree(frozen["code"], manifest["code"], "new prefit entire code inventory")
    pins_checked(root_path, frozen["prefit_design"])
    read_snapshot(
        root_path, REPORT + "/full_repository_tests.txt", frozen["checks"]["full_log_sha256"]
    )
    preserved = {
        str(path.relative_to(root_path))
        for path in [*root_path.glob("*.yaml"), *(root_path / "reports").rglob("*")]
        if path.is_file()
        and path != root_path / PROTOCOL
        and root_path / REPORT not in path.parents
        and str(path.relative_to(root_path)) not in inputs
    }
    if (
        set(manifest["code"]) != code
        or set(manifest["inputs"]) != inputs
        or set(manifest["preserved"]) != preserved
        or inputs & preserved
    ):
        raise ValueError("Complete registered source/code/preservation inventory required")
    for name, signature in closure["files"].items():
        if manifest["inputs"].get(name) != signature:
            raise ValueError("Upstream closure differs from registered input pin: " + name)


def require_bound_failure(root, proof):
    """This identified failed object is required; original successful failures block."""
    require_upstream_success(root)
    files = hash_mapping(proof["files"])
    names = ["reports/event_cluster/" + name for name in
             ("failure.json", "metrics.json", "verification.json")]
    failure, metrics, verification = [read_json_snapshot(root, name, files[name]) for name in names]
    require_failed_metrics(metrics)
    same_tree(failure, metrics, "unchanged failed-wave20 canonical marker")
    signature = proof["anchors"]["event_cluster.yaml"]
    same_tree(metrics["protocol_sha256"], signature, "failed-wave20 protocol identity")
    same_tree(verification, {
        "status": "FAILED",
        "error": "Independent verification failed: AssertionError: every saved application state "
                 "differs from independent reconstruction",
        "protocol_sha256": signature,
    }, "one identified failed-wave20 verification")
    require_upstream_success(root)


def collect_closure(root, expected):
    """Shared documentary traversal; independently checked anchors and full pins.

    No unverified output table or unpublished scored payload is decoded here.
    """
    from . import event_cluster_replay_admission as shared

    require_upstream_success(root)
    expected = hash_mapping(expected)
    for name, signature in expected.items():
        read_snapshot(root, name, signature)
    collected = shared._collect(root, expected)
    proof = collected.audit
    same_tree(proof["anchors"], expected, "literal failed-wave20 documentary anchors")
    for name, value in {
        "status": "PINNED_FAILED_WAVE20_WITH_VERIFIED_WAVE19_WAVE18",
        "retained_outputs_numerically_verified": False,
        "historical_models_refitted": False,
        "source_values_reparsed": False,
        "retained_outputs_transformed": False,
        "loaded_from_checked_snapshots": True,
        "unpublished_scored_payload_decoded": False,
    }.items():
        same_tree(proof[name], value, "source admission role " + name)
    files = hash_mapping(proof["files"])
    pins_checked(root, files)
    for name, signature in expected.items():
        same_tree(files[name], signature, "admitted anchor bytes")
    paths = sorted(name for name in files if re.fullmatch(r"reports/[^/]+/manifest\.json", name))
    same_tree(proof["manifest_paths"], paths, "complete transitive manifest inventory")
    counts = {}
    for name in paths:
        document = read_json_snapshot(root, name, files[name])
        counts[name] = {}
        for group in ("code", "inputs", "preserved", "existing_artifacts_sha256"):
            refs = hash_mapping(document.get(group, {}))
            counts[name][group] = len(refs)
            for path, signature in refs.items():
                if files.get(path) != signature:
                    raise ValueError("Registered transitive source reference missing: " + path)
    same_tree(proof["manifest_groups"], counts, "independent manifest group counts")
    outputs = hash_mapping(proof["retained_output_hashes"])
    retained_names = {"data/event_cluster/" + name for name in (
        "forecasts.parquet", "states.parquet", "memory.parquet", "fits.json",
        "support_audit.json", "upstream_admission.json")}
    if set(outputs) != retained_names:
        raise ValueError("Exactly six retained producer input paths required")
    for name, signature in outputs.items():
        same_tree(files[name], signature, "retained unverified input pin")
    record_name = "reports/issued_calibration/verification.json"
    original_record = read_json_snapshot(root, record_name, files[record_name])
    if original_record.get("status") != "VERIFIED":
        raise ValueError("Original successful wave19 verification required")
    same_tree(original_record["inference"]["cumulative_hypotheses_verified"], 131,
              "verified original family")
    original = proof["original_admission"]
    same_tree(original_record["verified_output_hashes"], original["verified_output_hashes"],
              "original verified output identities")
    for name, signature in hash_mapping(original["files"]).items():
        same_tree(files[name], signature, "original successful source closure")
    same_tree(proof["counts"], {"files": len(files), "manifests": len(paths),
                               "retained_outputs": 6}, "complete source counts")
    require_bound_failure(root, proof)
    return proof


def admit_upstream(root, expected, registered_pins):
    from . import event_cluster_replay_admission as shared

    proof = collect_closure(root, expected)
    registered = hash_mapping(registered_pins)
    for name, signature in proof["files"].items():
        if registered.get(name) != signature:
            raise ValueError("Full source closure missing from registered pins: " + name)
    actual, original, retained = shared.admit_upstream(root, expected, registered)
    same_tree(actual, proof, "unchanged checked-snapshot admission")
    if set(retained) != {"forecasts", "states", "memory", "fits", "support", "upstream_admission"}:
        raise ValueError("Exact retained producer artifact mapping required")
    same_tree(retained["upstream_admission"], proof["original_admission"],
              "retained successful upstream proof")
    metadata = proof["failed_wave_counts"]
    for name, value in {
        "combined_saved_forecasts": len(retained["forecasts"]),
        "producer_reported_monthly_schedules": len(retained["fits"]),
        "state_rows": len(retained["states"]), "memory_rows": len(retained["memory"]),
    }.items():
        same_tree(metadata[name], value, "retained metadata " + name)
    pins_checked(root, proof["files"])
    require_bound_failure(root, proof)
    return proof, original, retained


def compare_reconstruction_audit(saved, expected):
    """Compare independently regenerated proof, allowing solver diagnostic variation.

    Diagnostics cannot substitute for coefficient/objective/gradient comparison.
    Full retained coefficients and probabilities are separately verified by the
    mathematical reconstruction, never accepted from this derived proof alone.
    """
    def compare(a, b, path=()):
        key = path[-1] if path else ""
        label = ".".join(map(str, path))
        if type(b) is dict:
            if type(a) is not dict or set(a) != set(b):
                raise ValueError("Exact reconstruction audit schema required: " + label)
            for name in b:
                compare(a[name], b[name], (*path, name))
        elif type(b) is list:
            if type(a) is not list or len(a) != len(b):
                raise ValueError("Exact reconstruction list required: " + label)
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                compare(x, y, (*path, i))
        elif key == "message" and "independent_nuisance" in path:
            if type(a) is not str or not a or type(b) is not str or not b:
                raise ValueError("Nonempty solver diagnostic message required")
        elif key in ("function_evaluations", "jacobian_evaluations") and "independent_nuisance" in path:
            if type(a) is not int or type(b) is not int or min(a, b) < 1:
                raise ValueError("Positive literal solver diagnostic count required")
        elif key in ("iterations", "function_calls") and "independent_cluster" in path:
            if type(a) is not int or type(b) is not int:
                raise ValueError("Literal independent bisection diagnostics required")
            # Status-specific zero/fitted constraints are checked below as well.
            limit = 200 if key == "iterations" else 202
            if not 0 <= a <= limit or not 0 <= b <= limit:
                raise ValueError("Independent scalar diagnostic outside fixed budget")
        elif "start" in path or key in ("alpha", "xtol", "rtol", "factor"):
            same_tree(a, b, "exact solver setting " + label)
        elif type(b) is float:
            if type(a) not in (int, float) or not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Finite real reconstruction evidence required: " + label)
            if (isinstance(key, str) and key.endswith("gradient_max_abs")
                and (not 0 <= a <= 1.0001e-8 or not 0 <= b <= 1.0001e-8)):
                raise ValueError("Original full gradient acceptance unchanged")
            coefficient = "coefficients" in path or key == "coefficient"
            frozen.close(a, b, "saved independent reconstruction " + label,
                         rtol=1e-7 if coefficient else 1e-10,
                         atol=1e-6 if coefficient else 1e-12)
        else:
            same_tree(a, b, "exact reconstruction evidence " + label)

    compare(saved, expected)
    for proof in (saved, expected):
        for record in proof["monthly_stage_audits"]:
            audit = record["audit"]
            nuisance, scalar = audit["independent_nuisance"], audit["independent_cluster"]
            if nuisance["success"] is not True or scalar["success"] is not True:
                raise ValueError("Successful independent reconstruction required")
            g = frozen.real(nuisance["gradient"], "independent full nuisance gradient")
            if g.shape != (6,) or np.max(np.abs(g)) > 1.0001e-8:
                raise ValueError("Independent full nuisance gradient failed")
            if abs(frozen.scalar(scalar["gradient"])) > 1.0001e-8:
                raise ValueError("Independent full scalar gradient failed")
            frozen.close(nuisance["gradient_max_abs"], float(np.max(np.abs(g))),
                         "reconstruction gradient maximum")
            frozen.close(scalar["gradient_max_abs"], abs(scalar["gradient"]),
                         "reconstruction scalar gradient maximum")
            if scalar["status"] == "FITTED":
                if not 0 <= scalar["iterations"] <= 200 or not 2 <= scalar["function_calls"] <= 202:
                    raise ValueError("Declared independent scalar budget differs")
            elif scalar["status"] in ("EXACT_CONSTANT_INPUT", "BALANCED_AT_ZERO"):
                if scalar["coefficient"] != 0 or scalar["iterations"] != 0 or scalar["function_calls"] != 0:
                    raise ValueError("Canonical independent zero branch required")
            else:
                raise ValueError("Unknown independent scalar status")


def verify_pipeline(*args):
    from .event_cluster_replay_verification import verify_pipeline as reconstruct

    return reconstruct(*args)


def verify(root=ROOT):
    root = Path(root)
    protocol_path = root / PROTOCOL
    payload = protocol_path.read_bytes()
    protocol_hash = sha256(payload).hexdigest()
    protocol = yaml.safe_load(payload)
    validate_protocol(protocol)
    report, out = root / REPORT, root / OUT
    manifest_payload = (report / "manifest.json").read_bytes()
    manifest_hash = sha256(manifest_payload).hexdigest()
    manifest = strict_json(manifest_payload)
    same_tree(manifest["protocol_sha256"], protocol_hash, "new registered protocol hash")
    for group in ("code", "inputs", "preserved"):
        pins_checked(root, manifest[group])
    closure = collect_closure(root, protocol["upstream"]["anchors"])
    verify_manifest_coverage(root, protocol, manifest, closure)
    output_names = [OUT + "/upstream_admission.json", OUT + "/reconstruction_audit.json",
                    REPORT + "/metrics.json", REPORT + "/trial_ledger.jsonl"]
    buffers = {name: source_path(root, name).read_bytes() for name in output_names}
    snapshots = {name: sha256(data).hexdigest() for name, data in buffers.items()}
    metrics = strict_json(buffers[REPORT + "/metrics.json"])
    if ((report / "failure.json").exists() or (report / "failure.json").is_symlink()
        or metrics.get("status") == "UNEVALUABLE" or metrics.get("whole_wave_aborted") is True
        or metrics["protocol_sha256"] != protocol_hash
        or metrics["evidence_class"] != protocol["evidence_class"]):
        raise ValueError("Only an intact new scored publication may be independently verified")
    proof, loaded, retained = admit_upstream(root, protocol["upstream"]["anchors"], manifest["inputs"])
    same_tree(strict_json(buffers[OUT + "/upstream_admission.json"]), proof,
              "saved complete upstream admission")
    same_tree(protocol["original_index"], loaded["protocol"]["index"],
              "unchanged original source/index contract")
    checked = verify_pipeline(
        loaded["features"], loaded["targets"], loaded["forecasts"], loaded["fits"],
        loaded["states"], protocol["original_index"], retained["forecasts"], retained["fits"],
        retained["states"], retained["memory"], retained["support"],
    )
    compare_reconstruction_audit(strict_json(buffers[OUT + "/reconstruction_audit.json"]), checked)
    panel = retained["forecasts"]
    result = {
        "status": "VERIFIED", "protocol_sha256": protocol_hash,
        "verifier_sha256": digest(Path(__file__)),
        "upstream_admission": {
            "status": proof["status"], "retained_input_role": "UNVERIFIED_WAVE20_PRODUCER_OUTPUT",
            "historical_models_refitted": False, "source_values_reparsed": False,
            "metadata_implementation": "shared checked source traversal with independent anchor and full-file rehashes",
        },
        "forecast_reconstruction": checked,
        "primitive_brier_scores_verified": len(panel),
        "paired_brier_differences_verified": 3 * checked["common_scored_origins"],
        "inference": verify_metrics(root, panel, protocol, metrics, len(retained["states"]),
                                    len(retained["fits"]), admitted_inputs=manifest["inputs"]),
        "ledger_events_verified": verify_ledger(root, metrics, metrics["inherited_rows"], "evaluated",
                                                signature=snapshots[REPORT + "/trial_ledger.jsonl"]),
        "verified_output_hashes": snapshots,
        "verified_retained_input_hashes": closure["retained_output_hashes"],
        "artifact_hashes_checked": {
            **{group: len(manifest[group]) for group in ("code", "inputs", "preserved")},
            "outputs": len(snapshots),
        },
        "limitations": [
            "Wave20 remains UNEVALUABLE/FAILED with three canonical p=1 hypotheses; this is a separately registered technical replay.",
            "No producer coefficients or predictions were regenerated or replaced; independently solved optima only verify retained outputs.",
            "Only the seven declared application-state date columns admit lossless ms/us/ns comparison; all other frozen checks remain unchanged.",
            "The same archival alert observations are reused adaptively; this is neither a new predictive sample nor a fresh independent replication.",
            "Model-relative adjacency does not establish conditional orthogonality, a causal regime, historical execution timing or trading profitability.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        pins_checked(root, manifest[group])
    pins_checked(root, snapshots)
    verify_manifest_coverage(root, protocol, manifest, closure)
    if digest(protocol_path) != protocol_hash or digest(report / "manifest.json") != manifest_hash:
        raise ValueError("New protocol/manifest changed during independent verification")
    require_upstream_success(root)
    pins_checked(root, closure["files"])
    # Re-read the bound old failure trio after the last full hash loop, together
    # with the absence of failure markers for both successful original studies.
    require_bound_failure(root, closure)
    if (report / "failure.json").exists() or (report / "failure.json").is_symlink():
        raise ValueError("New failure marker appeared before verification commit")
    (report / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def invalidate_publication(root, error):
    report = Path(root) / REPORT
    report.mkdir(parents=True, exist_ok=True)
    path = report / "metrics.json"
    original_bytes = path.read_bytes() if path.exists() else None
    prior, invalid = None, None
    if original_bytes is not None:
        try:
            prior = strict_json(original_bytes)
        except (UnicodeDecodeError, ValueError):
            invalid = original_bytes
    protocol_hash = prior.get("protocol_sha256") if prior else None
    message = "Independent verification failed: " + type(error).__name__ + ": " + str(error)
    failure = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 137,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "event_cluster_replay",
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
    payload = json.dumps(failure, indent=2, allow_nan=False) + "\n"
    path.write_text(payload)
    (report / "failure.json").write_text(payload)
    (report / "verification.json").write_text(
        json.dumps(
            {"status": "FAILED", "error": message, "protocol_sha256": protocol_hash}, indent=2
        )
        + "\n"
    )
    (report / "results.md").write_text(
        "# SPX event-adjacency risk-alert forecasts\n\nUNEVALUABLE: independent verification failed. All three comparisons retained with p=1; no lead.\n\n"
        + message
        + "\n"
    )
    # Guaranteed canonical family/ledger invalidation precedes optional backups.
    ledger_path = report / "trial_ledger.jsonl"
    old_ledger = ledger_path.read_bytes() if ledger_path.exists() else b""
    with ledger_path.open("a") as stream:
        if old_ledger and not old_ledger.endswith(b"\n"):
            stream.write("\n")
        for row in failure["rows"]:
            stream.write(
                json.dumps(
                    {"event": "verification_failed", **row}, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
    backup = report / "unpublished_scored_metrics.json"
    if prior is not None and prior.get("status") != "UNEVALUABLE" and not backup.exists():
        backup.write_text(
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
        )
    invalid_backup = report / "unpublished_invalid_metrics.txt"
    if invalid is not None and not invalid_backup.exists():
        invalid_backup.write_bytes(invalid)
    return failure


def verify_with_failure_guard(root=ROOT):
    try:
        return verify(root)
    except Exception as error:
        invalidate_publication(root, error)
        raise


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False))
