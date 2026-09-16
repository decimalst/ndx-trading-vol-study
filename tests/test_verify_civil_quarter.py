"""Prewritten independent civil geometry, convex-fit and failure contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import verify_civil_quarter as verify


def original_fixture(n=1700):
    dates = pd.bdate_range("2009-01-01", periods=n)
    rng = np.random.default_rng(731)
    frame = pd.DataFrame(rng.normal(size=(n, 18)), index=dates, columns=verify.OLD)
    frame["const"] = 1.0
    frame["feature_cutoff_date"] = pd.Series(dates, index=dates).shift()
    return frame


class CivilContracts(unittest.TestCase):
    def test_nominal_weekday_leap_and_rollover_edges(self):
        dates = pd.to_datetime(
            [
                "2024-02-23",
                "2024-02-28",
                "2024-03-22",
                "2024-03-27",
                "2024-06-27",
                "2024-09-26",
                "2024-12-26",
                "2024-12-31",
            ]
        )
        frame = verify.civil_features(dates)
        self.assertEqual(frame.quarter_end5.tolist(), [0, 0, 0, 1, 1, 1, 0, 0])
        self.assertEqual(frame.month_end5.tolist(), [1, 1, 0, 1, 1, 1, 1, 0])
        self.assertEqual(frame.year_end5.tolist(), [0, 0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(frame.month_2.iloc[0], 1)
        self.assertEqual(frame.loc[dates[-1], list(verify.MONTHS)].sum(), 0)

    def test_future_observed_holiday_calendar_cannot_change_civil_prefix(self):
        dates = pd.to_datetime(["2024-05-24", "2024-05-28", "2024-05-29"])
        original = verify.civil_features(dates)
        shortened = verify.civil_features(dates[:1])
        pd.testing.assert_frame_equal(original.iloc[:1], shortened, check_exact=True)
        self.assertEqual(original.month_end5.iloc[0], 1)  # Nominal Memorial Day May27.

    def test_augment_preserves_old_unknown_values_cutoff_and_column_order(self):
        old = original_fixture(100)
        old.iloc[30, 2] = np.nan
        new = verify.augment_features(old)
        pd.testing.assert_frame_equal(new.loc[:, old.columns], old, check_exact=True)
        self.assertEqual(tuple(new.columns), verify.ALL_FEATURES + ("feature_cutoff_date",))

    def test_support_no_row_deletion_and_exact_rank_requirement(self):
        frame = verify.augment_features(original_fixture())
        support = verify.civil_support(frame, "train")
        self.assertEqual(support["n"], len(frame))
        self.assertEqual(verify.civil_rank(frame)["rank"], 15)
        for column in ("quarter_end5", "year_end5", "month_2"):
            bad = frame.copy()
            bad[column] = 0.0
            with self.assertRaises(ValueError):
                verify.civil_support(bad, "train")
            with self.assertRaises(ValueError):
                verify.civil_rank(bad)
        bad = frame.copy()
        bad.loc[bad.index[20], "month_2"] = np.nan
        with self.assertRaises(ValueError):
            verify.civil_support(bad, "train")

    def test_training_only_mixed_geometry_and_novelty_projection(self):
        frame = verify.augment_features(original_fixture())
        training, application = frame.iloc[:1300], frame.iloc[1300:]
        x, a, audit = verify.transform(training, application)
        self.assertEqual(x.shape, (1300, 31))
        np.testing.assert_allclose(x[:, 1:18].std(axis=0), 1, atol=1e-13)
        np.testing.assert_allclose(
            x[:, 18:],
            training.loc[:, verify.NUISANCE] - training.loc[:, verify.NUISANCE].mean(),
            atol=1e-15,
        )
        changed = application.copy()
        changed.loc[:, verify.OLD[1:]] *= 100
        x2, _, audit2 = verify.transform(training, changed)
        np.testing.assert_array_equal(x, x2)
        self.assertEqual(audit, audit2)
        self.assertGreater(audit["quarter_residual_relative_norm"], 1e-8)
        self.assertEqual(a.shape[1], 31)

    def test_old_zero_scale_or_redundant_quarter_is_not_dropped(self):
        frame = verify.augment_features(original_fixture())
        bad = frame.copy()
        bad[verify.OLD[1]] = 0.0
        with self.assertRaises(ValueError):
            verify.transform(bad, bad.iloc[:3])
        bad = frame.copy()
        bad[verify.OLD[1]] = bad.quarter_end5
        with self.assertRaises(ValueError):
            verify.transform(bad, bad.iloc[:3])


class PositiveMomentContracts(unittest.TestCase):
    def test_saved_producer_audits_replay_and_coefficient_geometry_tampering_reject(self):
        import copy

        from src import civil_quarter_models as model

        frame = verify.augment_features(original_fixture())
        train, app = frame.iloc[:1400], frame.iloc[1400:]
        y = np.exp(
            -9 + 0.1 * train.quarter_end5.to_numpy() + np.sin(np.arange(len(train))) * 0.2
        )
        predictions, audits = model.fit_predict(train, y, app)
        replay, proof = verify.verify_fit(train, y, app, audits)
        self.assertEqual(proof["models_verified"], 3)
        for name in verify.MODELS:
            np.testing.assert_allclose(predictions[name], replay[name], rtol=1e-10, atol=0)
        for case in ("beta", "center", "quarter", "support", "bracket"):
            changed = copy.deepcopy(audits)
            if case == "beta":
                changed["baseline"]["scaled_beta"][2] += 0.1
            elif case == "center":
                changed["baseline"]["means"][20] += 0.1
            elif case == "quarter":
                changed["quarter"]["b"] += 0.1
            elif case == "support":
                changed["quarter"]["support"]["groups"]["quarter_end5"]["ones"] += 1
            else:
                changed["quarter"]["bracket"][1] += 1
            with self.assertRaises((AssertionError, ValueError)):
                verify.verify_fit(train, y, app, changed)

    def test_analytic_gradient_and_curvature_against_finite_differences(self):
        rng = np.random.default_rng(23)
        x = np.column_stack([np.ones(80), rng.normal(size=(80, 4))])
        y = np.exp(rng.normal(scale=0.4, size=80))
        beta = rng.normal(scale=0.1, size=5)
        f, g, h = verify.positive_objective(beta, x, y)
        for j in range(5):
            step = np.eye(5)[j] * 1e-5
            fp, gp, _ = verify.positive_objective(beta + step, x, y)
            fm, gm, _ = verify.positive_objective(beta - step, x, y)
            self.assertAlmostEqual((fp - fm) / 2e-5, g[j], places=8)
            np.testing.assert_allclose((gp - gm) / 2e-5, h[:, j], rtol=1e-7, atol=1e-8)
        self.assertTrue(np.isfinite(f))
        self.assertGreater(np.linalg.eigvalsh(h).min(), 0)

    def test_independent_convex_fit_scalar_bracket_and_target_unit_covariance(self):
        frame = verify.augment_features(original_fixture())
        rng = np.random.default_rng(19)
        y = np.exp(
            -9 + 0.15 * frame.quarter_end5.to_numpy() + rng.normal(scale=0.2, size=len(frame))
        )
        fit = verify.independent_fit(frame.iloc[:1400], y[:1400], frame.iloc[1400:])
        scaled = verify.independent_fit(frame.iloc[:1400], y[:1400] * 100, frame.iloc[1400:])
        np.testing.assert_allclose(fit["beta"], scaled["beta"], rtol=1e-7, atol=1e-6)
        self.assertAlmostEqual(fit["b"], scaled["b"], places=7)
        self.assertLessEqual(fit["baseline_gradient_max_abs"], 1e-8 + 1e-12)
        self.assertLessEqual(abs(fit["quarter_gradient"]), 1e-8 + 1e-12)
        self.assertLessEqual(fit["bracket_gradients"][0], 0)
        self.assertGreaterEqual(fit["bracket_gradients"][1], 0)
        for model in verify.MODELS:
            np.testing.assert_allclose(
                fit["predictions"][model] * 100,
                scaled["predictions"][model],
                rtol=1e-7,
                atol=1e-12,
            )

    def test_strict_positive_finite_real_arithmetic(self):
        for invalid in (
            np.array([0.0]),
            np.array([-1.0]),
            np.array([np.inf]),
            np.array([np.nan]),
            np.array([1 + 1j]),
        ):
            with self.assertRaises((ValueError, AssertionError)):
                verify.proper_score(np.array([1.0]), invalid)
        with self.assertRaises(ValueError):
            verify.positive_objective(np.array([1000.0]), np.ones((2, 1)), np.ones(2))

    def test_nonzero_design_penalty_and_mean_underflow_are_not_silently_zero(self):
        with self.assertRaises(ValueError):
            verify.positive_objective(
                np.array([0.0, 1e-200]), np.array([[1.0, 1e-200], [1.0, 1e-200]]), np.ones(2)
            )
        with self.assertRaises(ValueError):
            verify.checked_mean(np.array([np.nextafter(0.0, 1.0), 0.0]))

    def test_score_arrays_require_exact_nonempty_alignment(self):
        for y, h in (
            (np.ones(3), np.ones(1)),
            (np.array([]), np.array([])),
            (np.ones((2, 2)), np.ones((2, 2))),
        ):
            with self.assertRaises(ValueError):
                verify.proper_score(y, h)

    def test_stable_paired_proper_loss_antisymmetry_and_exact_nesting(self):
        y = np.array([1e-4, 1e-7, 0.1])
        a = np.array([1e-4 * (1 + 1e-9), 2e-7, 0.08])
        b = np.array([1e-4, 1e-7, 0.1])
        d = verify.paired_difference(a, b, y)
        np.testing.assert_allclose(d, -verify.paired_difference(b, a, y), rtol=1e-14, atol=0)
        np.testing.assert_array_equal(verify.paired_difference(a, a, y), np.zeros(3))
        np.testing.assert_allclose(
            d, verify.proper_score(y, a) - verify.proper_score(y, b), rtol=1e-5, atol=1e-14
        )


class PublicationContracts(unittest.TestCase):
    def test_every_protocol_section_and_extra_fields_are_literal(self):
        import copy

        verify.validate_protocol(copy.deepcopy(verify.CONTRACT))
        for name in verify.CONTRACT:
            changed = copy.deepcopy(verify.CONTRACT)
            changed[name] = {"altered": True}
            with self.assertRaises(AssertionError):
                verify.validate_protocol(changed)
        changed = copy.deepcopy(verify.CONTRACT)
        changed["new_unregistered_field"] = True
        with self.assertRaises(AssertionError):
            verify.validate_protocol(changed)

    def test_independent_inference_matches_complete_synthetic_producer_and_rejects_gates(self):
        import copy

        from src import civil_quarter_search as search
        from tests.test_civil_quarter_search import panel

        rows, features = panel()
        protocol = copy.deepcopy(verify.CONTRACT)
        protocol["inference"]["bootstrap_draws"] = 101
        protocol["comparisons"]["inherited_sources"] = ["prior.json"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = {
                "rows": [
                    {
                        "candidate": str(i),
                        "control": "baseline",
                        "horizon": 1,
                        "p_conservative": 1.0,
                    }
                    for i in range(121)
                ]
            }
            (root / "prior.json").write_text(json.dumps(old))
            pins = {"prior.json": verify.digest(root / "prior.json")}
            prior = verify.inherited_rows(root, protocol, pins=pins)
            metrics = search.evaluate(rows, features, protocol, 118, prior=prior)
            metrics["common_application_origins"] = rows.origin.nunique() + 1
            proof = verify.verify_metrics(
                root,
                rows,
                features,
                protocol,
                metrics,
                118,
                metrics["common_application_origins"],
                admitted_inputs=pins,
            )
            self.assertEqual(proof["phase_comparisons_verified"], 4)
            self.assertEqual(proof["bootstrap_runs_verified"], 12)
            self.assertEqual(proof["cumulative_hypotheses_verified"], 123)
            bad_features = features.copy()
            bad_features.loc[rows.origin.iloc[0], "lrv_d"] = np.nan
            with self.assertRaises(AssertionError):
                verify.verify_metrics(
                    root,
                    rows,
                    bad_features,
                    protocol,
                    metrics,
                    118,
                    metrics["common_application_origins"],
                    admitted_inputs=pins,
                )
            for fault in ("delta", "holm", "support", "mde", "lead"):
                bad = copy.deepcopy(metrics)
                if fault == "delta":
                    bad["rows"][0]["phases"][0]["delta"] += 0.01
                elif fault == "holm":
                    bad["rows"][0]["p_holm_wave"] = 0
                elif fault == "support":
                    bad["civil_support"]["development"]["groups"]["quarter_end5"]["ones"] += 1
                elif fault == "mde":
                    bad["rows"][0]["phases"][0]["nominal_mde_effect_ratio"] += 1
                else:
                    bad["leads"] = ["quarter"]
                with self.assertRaises(AssertionError):
                    verify.verify_metrics(
                        root,
                        rows,
                        features,
                        protocol,
                        bad,
                        118,
                        metrics["common_application_origins"],
                        admitted_inputs=pins,
                    )
            (root / "prior.json").write_text('{"rows": []}')
            with self.assertRaises(AssertionError):
                verify.inherited_rows(root, protocol, pins=pins)

    def test_whole_entrypoint_reads_declared_artifacts_and_checks_final_protocol(self):
        import copy

        import yaml

        for fault in (None, "support", "protocol"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report, out = root / "reports/civil_quarter", root / "data/civil_quarter"
                report.mkdir(parents=True)
                out.mkdir(parents=True)
                protocol = copy.deepcopy(verify.CONTRACT)
                (root / "civil_quarter.yaml").write_text(yaml.safe_dump(protocol))
                phash = verify.digest(root / "civil_quarter.yaml")
                original = original_fixture(80)
                features = verify.augment_features(original)
                dates = original.index
                maturity = pd.Series(dates, index=dates).shift(-1)
                target = pd.DataFrame(
                    {"y": 0.001, "target_end": maturity, "available_date": maturity},
                    index=dates,
                )
                target.loc[dates[-1], "y"] = np.nan
                pins = {}
                for name, frame in (
                    (protocol["feature_source"]["features"], original),
                    (protocol["feature_source"]["targets"], target),
                ):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(path)
                    pins[name] = verify.digest(path)
                (report / "manifest.json").write_text(
                    json.dumps(
                        {"protocol_sha256": phash, "code": {}, "inputs": pins, "preserved": {}}
                    )
                )
                proof = {"status": "UPSTREAM_VERIFIED_READ_ONLY", "prior_files_written": False}
                support = {
                    "monthly_fits": 0,
                    "common_application_origins": 1,
                    "phases": [{"slices": []}, {"slices": [{}, {}]}],
                }
                (out / "upstream_admission.json").write_text(json.dumps(proof))
                (out / "support_audit.json").write_text(
                    json.dumps(
                        support if fault != "support" else {**support, "monthly_fits": 1}
                    )
                )
                features.to_parquet(out / "features.parquet")
                target.to_parquet(out / "targets.parquet")
                pd.DataFrame({"origin": [dates[30]] * 3}).to_parquet(out / "forecasts.parquet")
                (out / "fits.json").write_text("[]")
                (report / "metrics.json").write_text(
                    json.dumps(
                        {
                            "protocol_sha256": phash,
                            "evidence_class": protocol["evidence_class"],
                            "inherited_rows": [],
                        }
                    )
                )
                (report / "trial_ledger.jsonl").write_text("")

                def replay(*args, fault=fault, root=root):
                    if fault == "protocol":
                        with (root / "civil_quarter.yaml").open("a") as stream:
                            stream.write("# injected mutation\n")
                    return {"forecasts_verified": 3}

                with (
                    patch.object(verify, "verify_manifest_coverage"),
                    patch.object(verify, "validate_upstream", return_value=proof),
                    patch.object(verify, "preflight", return_value=support),
                    patch.object(verify, "verify_forecasts", side_effect=replay),
                    patch.object(verify, "verify_metrics", return_value={"leads": []}),
                    patch.object(
                        verify,
                        "verify_ledger",
                        return_value={"inherited": 121, "registered": 2, "evaluated": 2},
                    ),
                ):
                    if fault:
                        with self.assertRaises(AssertionError):
                            verify.verify_with_failure_guard(root)
                        self.assertEqual(
                            json.loads((report / "verification.json").read_text())["status"],
                            "FAILED",
                        )
                    else:
                        result = verify.verify(root)
                        self.assertEqual(result["status"], "VERIFIED")
                        self.assertEqual(result["primitive_proper_scores_verified"], 3)
                        self.assertEqual(result["paired_proper_differences_verified"], 2)

    def test_ledger_requires_all125_events_and_hash_bound_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / "reports/civil_quarter"
            report.mkdir(parents=True)
            prior = [{"candidate": str(i), "p_conservative": 1.0} for i in range(121)]
            rows = [
                {
                    "study": "civil_quarter",
                    "candidate": candidate,
                    "control": control,
                    "score": "proper_variance",
                    "horizon": 1,
                    "p_conservative": 1.0,
                }
                for candidate, control in verify.COMPARISONS
            ]
            metrics = {"protocol_sha256": "frozen", "rows": rows}
            ledger = (
                [{"event": "inherited", **row} for row in prior]
                + [{"event": "registered", "protocol_sha256": "frozen", **row} for row in rows]
                + [{"event": "evaluated", **row} for row in rows]
            )
            path = report / "trial_ledger.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in ledger) + "\n")
            signature = verify.digest(path)
            self.assertEqual(
                verify.verify_ledger(root, metrics, prior, "evaluated", signature=signature),
                {"inherited": 121, "registered": 2, "evaluated": 2},
            )
            path.write_text(path.read_text() + "{}\n")
            with self.assertRaises(AssertionError):
                verify.verify_ledger(root, metrics, prior, "evaluated", signature=signature)

    def test_dual_upstream_admission_is_read_only_and_rejects_source_or_issued_tamper(self):
        import yaml

        for fault in (
            None,
            "raw.bin",
            "data/calendar_variance/features.parquet",
            "reports/causal_pool/verification.json",
        ):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "reports/civil_quarter").mkdir(parents=True)
                (root / "src").mkdir()
                (root / "raw.bin").write_bytes(b"frozen raw")
                info, records, manifests = {}, {}, {}
                for key, name in (
                    ("upstream", "causal_pool"),
                    ("feature_source", "calendar_variance"),
                ):
                    report, data = root / "reports" / name, root / "data" / name
                    report.mkdir(parents=True)
                    data.mkdir(parents=True)
                    (root / (name + ".yaml")).write_text("fixed: true\n")
                    (root / "src" / ("verify_" + name + ".py")).write_text("frozen verifier")
                    (data / "features.parquet").write_bytes(b"frozen issued")
                    protocol_hash = verify.digest(root / (name + ".yaml"))
                    verifier_hash = verify.digest(root / "src" / ("verify_" + name + ".py"))
                    old = {
                        "protocol_sha256": protocol_hash,
                        "code": {"src/verify_" + name + ".py": verifier_hash},
                        "inputs": {"raw.bin": verify.digest(root / "raw.bin")},
                        "preserved": {},
                    }
                    (report / "manifest.json").write_text(json.dumps(old))
                    record = {
                        "status": "VERIFIED",
                        "protocol_sha256": protocol_hash,
                        "verifier_sha256": verifier_hash,
                    }
                    (report / "verification.json").write_text(json.dumps(record))
                    info[key] = {
                        "protocol": name + ".yaml",
                        "reports": "reports/" + name,
                        "data": "data/" + name,
                        "protocol_sha256": protocol_hash,
                        "manifest_sha256": verify.digest(report / "manifest.json"),
                        "verification_sha256": verify.digest(report / "verification.json"),
                        "verifier_sha256": verifier_hash,
                        "required_status": "VERIFIED",
                    }
                    records[key], manifests[key] = record, old
                (root / "civil_quarter.yaml").write_text(yaml.safe_dump(info))
                inputs = {"raw.bin": verify.digest(root / "raw.bin")}
                code = {}
                for key in info:
                    code.update(manifests[key]["code"])
                    inputs[info[key]["protocol"]] = info[key]["protocol_sha256"]
                    for location in (info[key]["reports"], info[key]["data"]):
                        inputs.update(
                            {
                                str(p.relative_to(root)): verify.digest(p)
                                for p in (root / location).rglob("*")
                                if p.is_file()
                            }
                        )
                manifest = {
                    "protocol_sha256": verify.digest(root / "civil_quarter.yaml"),
                    "code": code,
                    "inputs": inputs,
                    "preserved": {},
                }
                (root / "reports/civil_quarter/manifest.json").write_text(json.dumps(manifest))
                if fault:
                    with (root / fault).open("ab") as stream:
                        stream.write(b"changed")
                before = {
                    str(p.relative_to(root)): p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file()
                }
                with (
                    patch.object(
                        verify, "reconstruct_previous", return_value=(records["upstream"], {})
                    ),
                    patch.object(
                        verify, "reconstruct_calendar", return_value=records["feature_source"]
                    ),
                ):
                    if fault:
                        with self.assertRaises(AssertionError):
                            verify.validate_upstream(root)
                    else:
                        self.assertEqual(
                            verify.validate_upstream(root)["status"],
                            "UPSTREAM_VERIFIED_READ_ONLY",
                        )
                self.assertEqual(
                    before,
                    {
                        str(p.relative_to(root)): p.read_bytes()
                        for p in root.rglob("*")
                        if p.is_file()
                    },
                )

    def test_guard_retains_entire_new_family_before_optional_diagnostics(self):
        for content in (
            '{"rows": [], "lead": true}',
            "{bad",
            '{"value": NaN}',
            '{"value": Infinity}',
        ):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                report = root / "reports/civil_quarter"
                report.mkdir(parents=True)
                (report / "metrics.json").write_text(content)
                old = root / "reports/causal_pool/frozen.txt"
                old.parent.mkdir(parents=True)
                old.write_bytes(b"preserve")
                with (
                    patch.object(
                        verify, "verify", side_effect=AssertionError("injected discrepancy")
                    ),
                    self.assertRaises(AssertionError),
                ):
                    verify.verify_with_failure_guard(root)
                metrics = json.loads((report / "metrics.json").read_text())
                self.assertEqual(metrics["status"], "UNEVALUABLE")
                self.assertEqual(metrics["leads"], [])
                self.assertEqual(metrics["cumulative_hypothesis_count"], 123)
                self.assertEqual(len(metrics["rows"]), 2)
                for row in metrics["rows"]:
                    self.assertEqual(
                        [
                            row[k]
                            for k in ("p_conservative", "p_holm_wave", "p_holm_cumulative")
                        ],
                        [1, 1, 1],
                    )
                self.assertEqual(
                    len((report / "trial_ledger.jsonl").read_text().splitlines()), 2
                )
                self.assertEqual(old.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
