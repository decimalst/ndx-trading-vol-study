"""Prewritten synthetic issued-logit calibration and publication contracts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit

from src import verify_issued_calibration as v


class ScalarContracts(unittest.TestCase):
    def test_empty_seed_and_exact_balanced_symmetry(self):
        fit = v.independent_calibration_fit([], [], [])
        self.assertEqual(fit["intercept"], 0.0)
        self.assertEqual(fit["status"], "EMPTY_HISTORY")
        fit = v.independent_calibration_fit([0.7, -0.7], [0.0, 1.0], [0.1, 0.1])
        self.assertEqual(fit["intercept"], 0.0)
        self.assertEqual(fit["status"], "BALANCED_AT_ZERO")

    def test_unnormalized_weights_early_record_not_full_sample(self):
        a = v.independent_calibration_fit([0.0], [1.0], [0.01])["intercept"]
        b = v.independent_calibration_fit([0.0], [1.0], [1.0])["intercept"]
        self.assertGreater(a, 0)
        self.assertLess(a, b)
        self.assertLess(a, 0.25)

    def test_derivatives_curvature_bracket_and_unique_optimum(self):
        eta = np.array([-2.0, 0.5, 3.0])
        y = np.array([1.0, 0.0, 1.0])
        w = np.array([0.02, 0.1, 0.3])
        a = 0.4
        f, g, h = v.weighted_objective(a, eta, y, w)
        eps = 1e-5
        fd = (
            v.weighted_objective(a + eps, eta, y, w)[0]
            - v.weighted_objective(a - eps, eta, y, w)[0]
        ) / (2 * eps)
        self.assertAlmostEqual(g, fd, places=9)
        self.assertGreaterEqual(h, 0.02)
        fit = v.independent_calibration_fit(eta, y, w)
        self.assertTrue(fit["converged"])
        self.assertLessEqual(fit["gradient_max_abs"], 1.0001e-8)
        self.assertLess(fit["bracket_gradients"][0], 0)
        self.assertGreater(fit["bracket_gradients"][1], 0)
        self.assertLessEqual(fit["objective"], f)

    def test_finite_rounded_endpoint_uses_original_stable_residual(self):
        eta = 40.0
        self.assertEqual(expit(eta), 1.0)
        _, g, _ = v.weighted_objective(0.0, [eta], [1.0], [0.1])
        self.assertLess(g, 0.0)
        self.assertNotEqual(g, 0.1 * (expit(eta) - 1))

    def test_nonzero_terms_cannot_underflow_or_be_removed(self):
        for eta, y, w in [
            ([1000.0], [1.0], [0.1]),
            ([0.0], [0.0], [0.0]),
            ([40.0], [1.0], [np.nextafter(0.0, 1.0)]),
        ]:
            with self.subTest(eta=eta, w=w), self.assertRaises(ValueError):
                v.weighted_objective(0.0, eta, y, w)

    def test_invalid_shape_domain_or_mass_rejects(self):
        for eta, y, w in [
            ([0.0], [0.5], [0.1]),
            ([0.0], [1.0], [1.1]),
            ([0j], [1.0], [0.1]),
            ([0.0, 1.0], [1.0], [0.1]),
        ]:
            with self.subTest(eta=eta), self.assertRaises(ValueError):
                v.independent_calibration_fit(eta, y, w)

    def test_explicit_weight_ages_advance_across_missing_sessions(self):
        dates = pd.bdate_range("2016-01-01", periods=12)
        arrivals = pd.DatetimeIndex([dates[2], dates[6]])
        w = v.explicit_weights(dates, arrivals, dates[9])
        np.testing.assert_array_equal(w, (1 - v.DELTA) * np.power(v.DELTA, [7.0, 3.0]))
        later = v.explicit_weights(dates, arrivals, dates[10])
        np.testing.assert_allclose(later, w * v.DELTA, rtol=1e-15)
        with self.assertRaises(ValueError):
            v.explicit_weights(dates, arrivals, dates[4])

    def test_bisection_failure_has_no_fallback(self):
        with (
            patch.object(
                v, "bisect", side_effect=RuntimeError("fixed budget exhausted")
            ) as solve,
            self.assertRaises(RuntimeError),
        ):
            v.independent_calibration_fit([0.0], [1.0], [0.1])
        self.assertEqual(solve.call_count, 1)


class IssuanceContracts(unittest.TestCase):
    def test_original_finite_logit_replay_never_inverts_endpoint(self):
        from tests.test_verify_range_alert import bars

        daily, iv = bars(200)
        features, _ = v.previous.feature_target_tables(daily, iv)
        features = features.dropna()
        train, app = features.iloc[:80], features.iloc[80:]
        *_, geometry = v.previous.transform(train, app)
        beta = np.zeros(26)
        beta[0] = 40
        audit = {
            "transform": geometry,
            "baseline": {
                "columns": list(v.previous.BASE),
                "means": geometry["means"],
                "scales": geometry["scales"],
                "beta": beta.tolist(),
            },
        }
        logits, p = v.replay_saved_logit(app, audit)
        np.testing.assert_array_equal(logits, np.full(len(app), 40.0))
        np.testing.assert_array_equal(p, np.ones(len(app)))
        altered = copy.deepcopy(audit)
        altered["baseline"]["beta"][1] = float("inf")
        with self.assertRaises(ValueError):
            v.replay_saved_logit(app, altered)

    def test_arrivals_exclude_nonissuance_and_unknown_separately(self):
        dates = pd.bdate_range("2016-01-01", periods=8)
        maturity = pd.Series(dates, index=dates).shift(-1)
        targets = pd.DataFrame(
            {
                "y": [0.0, 1.0, np.nan, 1.0, 0.0, 1.0, 0.0, np.nan],
                "target_end": maturity,
                "available_date": maturity,
            },
            index=dates,
        )
        issued = pd.DataFrame(
            {"origin": [dates[1], dates[2], dates[4]], "baseline_logit": [0.0, 0.2, -0.3]}
        )
        arrivals, records = v.arrival_history(targets, dates, issued, dates[0], dates[6])
        self.assertEqual(
            [r["status"] for r in arrivals],
            [
                "NO_ISSUED_FORECAST",
                "ADMITTED",
                "UNKNOWN_LABEL",
                "NO_ISSUED_FORECAST",
                "ADMITTED",
                "NO_ISSUED_FORECAST",
            ],
        )
        self.assertEqual(
            [r["origin"] for r in records], [str(dates[1].date()), str(dates[4].date())]
        )
        changed = targets.copy()
        changed.loc[dates[6], "y"] = 1.0
        self.assertEqual(
            (arrivals, records), v.arrival_history(changed, dates, issued, dates[0], dates[6])
        )


class PublicationContracts(unittest.TestCase):
    def test_typed_protocol_and_extra_field_rejects(self):
        for f in (lambda p: p["index"].update(market_lag=True), lambda p: p.update(extra=1)):
            p = copy.deepcopy(v.CONTRACT)
            f(p)
            with self.assertRaises(ValueError):
                v.validate_protocol(p)

    def test_failure_invalidates_only_new_wave_and_retains_nonfinite_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = root / "reports/issued_calibration"
            report.mkdir(parents=True)
            prior = root / "reports/range_alert/metrics.json"
            prior.parent.mkdir(parents=True)
            prior.write_bytes(b"prior unchanged")
            raw = b'{"rows":[],"bad":NaN}'
            (report / "metrics.json").write_bytes(raw)
            with (
                patch.object(v, "verify", side_effect=ValueError("synthetic failure")),
                self.assertRaises(ValueError),
            ):
                v.verify_with_failure_guard(root)
            metrics = json.loads((report / "metrics.json").read_bytes())
            self.assertEqual(metrics["cumulative_hypothesis_count"], 131)
            self.assertEqual(metrics["leads"], [])
            self.assertEqual(len(metrics["rows"]), 2)
            self.assertTrue(
                all(
                    row["p_conservative"]
                    == row["p_holm_wave"]
                    == row["p_holm_cumulative"]
                    == 1
                    for row in metrics["rows"]
                )
            )
            self.assertEqual((report / "unpublished_invalid_metrics.txt").read_bytes(), raw)
            self.assertEqual(prior.read_bytes(), b"prior unchanged")
            self.assertEqual(len((report / "trial_ledger.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()


class IndependentAdmissionContracts(unittest.TestCase):
    def fixture(self, path):
        from tests.test_issued_calibration_admission import Fixture

        return Fixture(path)

    def test_full_metadata_closure_and_same_byte_loaded_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            fixture = self.fixture(r)
            audit, snapshots = v.collect_closure(r, fixture.expected)
            self.assertIn("private/older/artifact.bin", audit["files"])
            self.assertEqual(audit["counts"]["verified_outputs"], 10)
            with patch.object(
                pd, "read_parquet", side_effect=AssertionError("no metadata decode")
            ):
                second, _ = v.collect_closure(r, fixture.expected)
            self.assertEqual(audit, second)
            checked, loaded = v.admit_upstream(r, fixture.expected, audit["files"])
            self.assertEqual(checked, audit)
            self.assertEqual(len(loaded["forecasts"]), 3)
            self.assertEqual(
                snapshots["data/range_alert/source_audit.json"],
                (r / "data/range_alert/source_audit.json").read_bytes(),
            )

    def test_missing_registered_pin_stops_before_parquet_decoding(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            fixture = self.fixture(r)
            with (
                patch.object(pd, "read_parquet") as decoder,
                self.assertRaises(AssertionError),
            ):
                v.admit_upstream(r, fixture.expected, {})
            decoder.assert_not_called()

    def test_old_transitive_hash_or_success_identity_tamper_rejects(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            fixture = self.fixture(r)
            (r / "private/older/artifact.bin").write_bytes(b"changed")
            with self.assertRaises(AssertionError):
                v.collect_closure(r, fixture.expected)

    def test_duplicate_json_keys_and_symlink_are_rejected(self):
        with self.assertRaises(ValueError):
            v.strict_json(b'{"a":1,"a":2}')
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            fixture = self.fixture(r)
            p = r / "private/older/artifact.bin"
            payload = p.read_bytes()
            p.unlink()
            other = r / "source-copy"
            other.write_bytes(payload)
            p.symlink_to(other)
            with self.assertRaises(ValueError):
                v.collect_closure(r, fixture.expected)


class ScalarAccountingContracts(unittest.TestCase):
    def test_literal_iteration_and_function_call_status_contract(self):
        fitted = {"status": "FITTED", "iterations": 5, "function_calls": 7}
        empty = {"status": "EMPTY_HISTORY", "iterations": 0, "function_calls": 0}
        v.validate_scalar_accounting(fitted)
        v.validate_scalar_accounting(empty)
        for changed in (
            {**fitted, "function_calls": True},
            {**fitted, "function_calls": 0},
            {**fitted, "function_calls": 203},
            {**fitted, "iterations": 0},
            {**empty, "function_calls": 1},
            {**empty, "iterations": 1},
        ):
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                v.validate_scalar_accounting(changed)


class WholeEntryContracts(unittest.TestCase):
    def test_complete_new_output_paths_and_serialized_hash_identity(self):
        from hashlib import sha256

        import yaml

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = root / "reports/issued_calibration"
            out = root / "data/issued_calibration"
            report.mkdir(parents=True)
            out.mkdir(parents=True)
            protocol = copy.deepcopy(v.CONTRACT)
            raw = yaml.safe_dump(protocol).encode()
            ph = sha256(raw).hexdigest()
            (root / "issued_calibration.yaml").write_bytes(raw)
            manifest = {"protocol_sha256": ph, "code": {}, "inputs": {}, "preserved": {}}
            (report / "manifest.json").write_text(json.dumps(manifest))
            audit = {"status": "synthetic", "files": {}}
            (out / "upstream_admission.json").write_text(json.dumps(audit))
            (out / "calibration_audit.json").write_text("{}")
            pd.DataFrame({"placeholder": [0, 1, 2]}).to_parquet(out / "forecasts.parquet")
            pd.DataFrame({"origin": [pd.Timestamp("2016-01-04")]}).to_parquet(
                out / "states.parquet"
            )
            metrics = {
                "protocol_sha256": ph,
                "evidence_class": protocol["evidence_class"],
                "inherited_rows": [],
            }
            (report / "metrics.json").write_text(json.dumps(metrics))
            (report / "trial_ledger.jsonl").write_text("")
            loaded = {
                name: pd.DataFrame() for name in ("features", "targets", "forecasts", "states")
            }
            loaded["fits"] = []
            loaded["protocol"] = copy.deepcopy(protocol)
            loaded["protocol"]["index"]["models"] = list(v.previous.MODELS)
            proof = {
                "forecasts_verified": 3,
                "new_forecasts": 1,
                "reused_control_forecasts": 2,
                "new_monthly_fits": 0,
                "common_scored_origins": 1,
                "common_application_origins": 1,
            }
            with (
                patch.object(v, "collect_closure", return_value=(audit, {})),
                patch.object(v, "verify_manifest_coverage"),
                patch.object(v, "admit_upstream", return_value=(audit, loaded)),
                patch.object(v, "verify_forecasts", return_value=proof) as vf,
                patch.object(v, "verify_metrics", return_value={"new_hypotheses_verified": 2}),
                patch.object(v, "verify_ledger", return_value={}) as ledger,
            ):
                result = v.verify(root)
                (report / "verification.json").unlink()

                def mark_failed(*args, **kwargs):
                    old = root / "reports/range_alert"
                    old.mkdir(parents=True, exist_ok=True)
                    (old / "failure.json").write_text("{}")
                    return proof

                vf.side_effect = mark_failed
                with self.assertRaises(AssertionError):
                    v.verify(root)
                self.assertFalse((report / "verification.json").exists())
            self.assertEqual(result["status"], "VERIFIED")
            self.assertEqual(len(result["verified_output_hashes"]), 6)
            self.assertEqual(vf.call_args.args[5].columns.tolist(), ["placeholder"])
            self.assertEqual(vf.call_args.args[6].columns.tolist(), ["origin"])
            self.assertIn("signature", ledger.call_args.kwargs)
            self.assertFalse((report / "verification.json").exists())


class PrefitPeerReviewContracts(unittest.TestCase):
    def test_representable_likelihood_gradient_not_rejected_by_unused_curvature(self):
        value, gradient, bound = v.weighted_objective(0.0, [700.0], [0.0], [1e-40])
        self.assertGreater(value, 0)
        self.assertGreater(gradient, 0)
        self.assertEqual(bound, 0.02)
        eta = np.array([-0.7, 0.2])
        y = np.array([0.0, 1.0])
        w = np.array([0.1, 0.2])
        eps = 1e-5
        curvature = (
            v.weighted_objective(eps, eta, y, w)[1] - v.weighted_objective(-eps, eta, y, w)[1]
        ) / (2 * eps)
        self.assertGreaterEqual(curvature, 0.02)
        self.assertLessEqual(curvature, 0.02 + 0.25 * w.sum())

    def test_saved_coordinate_nonzero_division_underflow_rejected(self):
        app = pd.DataFrame(np.ones((1, 24)), columns=v.previous.ALL_FEATURES)
        app["intraday"] = 1e-300
        means = np.zeros(26)
        scales = np.ones(26)
        scales[v.previous.BASE.index("intraday")] = 1e100
        audit = {
            "transform": {
                "columns": list(v.previous.BASE),
                "curvature_means": {"I": 1.0, "R": 1.0, "skew": 1.0},
            },
            "baseline": {
                "columns": list(v.previous.BASE),
                "means": means.tolist(),
                "scales": scales.tolist(),
                "beta": np.zeros(26).tolist(),
            },
        }
        with self.assertRaises(ValueError):
            v.replay_saved_logit(app, audit)

    def test_postdecode_upstream_failure_marker_is_not_admissible(self):
        from tests.test_issued_calibration_admission import Fixture

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture = Fixture(root)
            audit, _ = v.collect_closure(root, fixture.expected)
            original = pd.read_parquet

            def decoder(*args, **kwargs):
                (root / "reports/range_alert/failure.json").write_text("{}")
                return original(*args, **kwargs)

            with (
                patch.object(pd, "read_parquet", side_effect=decoder),
                self.assertRaises(AssertionError),
            ):
                v.admit_upstream(root, fixture.expected, audit["files"])
