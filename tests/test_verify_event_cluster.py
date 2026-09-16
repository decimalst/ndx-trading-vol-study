"""Prewritten independent event-cluster primitives; generated inputs only."""

import copy
import itertools
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit

from src import verify_event_cluster as v


def event_row(positions):
    e = np.zeros(22)
    e[np.array(positions, dtype=int) - 1] = 1
    return e


def targets(n=90):
    index = pd.bdate_range("2010-01-04", periods=n + 2).delete([15, 39])
    next_date = pd.Series(index, index=index).shift(-1)
    y = (np.arange(n) % 3 == 0).astype(float)
    y[-1] = np.nan
    return pd.DataFrame({"y": y, "target_end": next_date, "available_date": next_date})


def histories(n=240):
    rng = np.random.default_rng(526)
    index = pd.bdate_range("2010-01-04", periods=n)
    rows = [v.independent_summary(rng.integers(0, 2, 22)) for _ in index]
    return pd.DataFrame(rows, index=index).loc[:, v.FEATURES]


def baseline_fixture():
    f = pd.DataFrame(0.1, index=pd.bdate_range("2010-01-04", periods=4), columns=v.OLD_RAW)
    f["const"] = 1.0
    means, scales, beta = [0.0] * 26, [1.0] * 26, [0.0] * 26
    beta[0] = 40.0
    audit = {
        "baseline": {
            "columns": list(v.OLD_BASE),
            "means": means,
            "scales": scales,
            "beta": beta,
        },
        "transform": {
            "columns": list(v.OLD_BASE),
            "means": means.copy(),
            "scales": scales.copy(),
            "curvature_means": {"I": 0.05, "R": 0.2, "skew": 0.3},
        },
    }
    return f, audit


class SummaryContracts(unittest.TestCase):
    def test_integer_formulas_and_identifiability(self):
        a = v.independent_summary(event_row([1, 2, 6]))
        b = v.independent_summary(event_row([1, 3, 5]))
        for key in v.NUISANCE:
            self.assertEqual(a[key], b[key])
        self.assertEqual(a["adjacency_fraction22"], 1 / 21)
        self.assertEqual(b["adjacency_fraction22"], 0)
        self.assertEqual(a["event_fraction22_sq"], 9 / 484)
        self.assertEqual(a["expected_adjacency22"], 6 / 441)
        self.assertEqual(a["excess_adjacency22"], 15 / 441)
        self.assertEqual(a["linear_recency22"], -51 / 462)
        for key in v.AUDIT_FIELDS:
            self.assertIs(type(a[key]), int)

    def test_conditional_permutation_numerator_sums_exactly_zero(self):
        for q in range(4):
            for last in (0, 1):
                total = 0
                for first in itertools.combinations(range(1, 22), q):
                    s = v.independent_summary(event_row([*first, *([22] if last else [])]))
                    total += s["excess_adjacency_numerator"]
                self.assertEqual(total, 0)

    def test_zero_and_all_one_are_valid(self):
        for e, count, adjacency in ((np.zeros(22), 0, 0), (np.ones(22), 22, 21)):
            a = v.independent_summary(e)
            self.assertEqual(a["event_count22"], count)
            self.assertEqual(a["adjacent_pairs22"], adjacency)
            self.assertEqual(a["excess_adjacency22"], 0)

    def test_binary_schema_strict(self):
        invalid = [
            np.zeros(21),
            np.zeros(23),
            np.zeros((22, 1)),
            ["0"] * 22,
            [False] * 22,
            [False] + [0] * 21,
            np.zeros(22, complex),
            [np.nan] + [0] * 21,
            [np.inf] + [0] * 21,
            [0.5] + [0] * 21,
        ]
        for values in invalid:
            with self.subTest(values=str(values)[:30]), self.assertRaises(ValueError):
                v.independent_summary(values)

    def test_full_calendar_matches_separate_producer(self):
        from src import event_cluster_features as producer

        t = targets()
        m = v.reconstruct_memory(t)
        pd.testing.assert_frame_equal(m, producer.build_memory(t))
        self.assertTrue(m.iloc[:23][list(v.SUMMARY_COLUMNS)].isna().all().all())
        self.assertEqual(m.iloc[23].window_first_origin, t.index[0])
        self.assertEqual(m.iloc[23].window_last_available, t.index[22])
        self.assertEqual(m.iloc[23].event_count22, int(t.y.iloc[:22].sum()))

    def test_missing_history_is_not_compressed_or_dropped(self):
        t = targets()
        t.iloc[30, t.columns.get_loc("y")] = np.nan
        m = v.reconstruct_memory(t)
        self.assertTrue(m.iloc[32:54][list(v.SUMMARY_COLUMNS)].isna().all().all())
        self.assertFalse(m.iloc[32:54][list(v.DATE_COLUMNS)].isna().any().any())
        self.assertTrue(m.iloc[54][list(v.FEATURES)].notna().all())
        with self.assertRaises(ValueError):
            v.require_original_histories(m, t.index[23:31], t.index[32:34])
        with self.assertRaises(ValueError):
            v.require_original_histories(m, t.index[23:34], t.index[54:56])
        v.require_original_histories(m, t.index[23:31], t.index[54:56])

    def test_future_labels_do_not_change_earlier_histories(self):
        t = targets()
        before = v.reconstruct_memory(t)
        changed = t.copy()
        changed.iloc[60:-1, 0] = 1 - changed.iloc[60:-1, 0]
        pd.testing.assert_frame_equal(
            before.iloc[:62], v.reconstruct_memory(changed).iloc[:62]
        )

    def test_dates_terminal_and_schema_are_strict(self):
        for mutation in ("available", "terminal", "columns", "timezone"):
            t = targets()
            if mutation == "available":
                t.iloc[40, 2] = t.index[40]
            elif mutation == "terminal":
                t.iloc[-1, 0] = 0
            elif mutation == "columns":
                t = t[["y", "available_date", "target_end"]]
            else:
                t.index = t.index.tz_localize("UTC")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                v.reconstruct_memory(t)


class GeometryContracts(unittest.TestCase):
    def test_transform_fixed_centers_and_query_invariance(self):
        from src import event_cluster_models as producer

        f = histories()
        train, query = f.iloc[:220].copy(), f.iloc[220:]
        train.iloc[:, 0] = 0.1
        result = v.independent_transform(train, query)
        expected = producer.transform(train, query)
        for a, b in zip(result[:-1], expected[:-1]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(result[-1], expected[-1])
        self.assertEqual(result[-1]["means"][1], 0.1)
        np.testing.assert_array_equal(result[0][:, 1], np.zeros(220))
        reverse = v.independent_transform(train, query.iloc[::-1])
        self.assertEqual(result[-1], reverse[-1])
        np.testing.assert_array_equal(result[0], reverse[0])

    def test_transform_rejects_invalid_schema_values(self):
        f = histories()
        for bad in (
            f.iloc[:, :-1],
            f.astype(str),
            f.astype(complex),
            f.assign(last_event=np.nan),
        ):
            with self.assertRaises(ValueError):
                v.independent_transform(bad.iloc[:220], f.iloc[220:])

    def test_saved_baseline_geometry_replay_and_identity(self):
        from src import event_cluster_models as producer

        f, audit = baseline_fixture()
        eta, x = v.independent_baseline_logits(f, audit)
        other, design = producer.baseline_logits(f, audit)
        np.testing.assert_array_equal(eta, other)
        pd.testing.assert_frame_equal(x, design)
        self.assertTrue(np.all(eta == 40))
        self.assertTrue(np.all(expit(eta) == 1))
        for field, value in (("means", 0.01), ("scales", 0), ("columns", "bad")):
            bad = copy.deepcopy(audit)
            bad["baseline"][field][1] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                v.independent_baseline_logits(f, bad)

    def test_saved_coordinate_division_underflow_rejects(self):
        f, audit = baseline_fixture()
        f["intraday"] = 1e-300
        k = list(v.OLD_BASE).index("intraday")
        for key in ("baseline", "transform"):
            audit[key]["scales"][k] = 1e100
        with self.assertRaises(ValueError):
            v.independent_baseline_logits(f, audit)


class OptimizationContracts(unittest.TestCase):
    def setUp(self):
        f = histories()
        self.train, self.apply = f.iloc[:220], f.iloc[220:]
        self.y = (np.arange(220) % 3 == 0).astype(float)
        self.eta = np.linspace(-1.5, 0.5, 220)
        self.query = np.linspace(-2, 1, 20)
        self.x, self.q, self.z, self.qz, _ = v.independent_transform(self.train, self.apply)

    def test_nuisance_objective_gradient_and_jacobian(self):
        beta = np.array([0.1, -0.2, 0.3, 0.1, -0.1, 0.2])
        value, gradient, jac = v.nuisance_objective(beta, self.x, self.y, self.eta)
        self.assertTrue(math.isfinite(value))
        step = 1e-5
        numeric_g, numeric_j = np.zeros(6), np.zeros((6, 6))
        for k in range(6):
            d = np.eye(6)[k] * step
            plus = v.nuisance_objective(beta + d, self.x, self.y, self.eta)
            minus = v.nuisance_objective(beta - d, self.x, self.y, self.eta)
            numeric_g[k] = (plus[0] - minus[0]) / (2 * step)
            numeric_j[:, k] = (plus[1] - minus[1]) / (2 * step)
        np.testing.assert_allclose(gradient, numeric_g, atol=2e-10, rtol=1e-7)
        np.testing.assert_allclose(jac, numeric_j, atol=2e-10, rtol=1e-7)
        self.assertGreater(np.linalg.eigvalsh(jac).min(), 0)

    def test_independent_root_matches_producer_newton_and_original_full_gradient(self):
        from src import event_cluster_models as producer

        predictions, audit = producer.fit_stages(
            self.train,
            pd.Series(self.y, index=self.train.index),
            self.apply,
            self.eta,
            self.query,
            expit(self.query),
        )
        beta, solved = v.independent_nuisance_fit(self.x, self.y, self.eta)
        np.testing.assert_allclose(beta, audit["nuisance"]["beta"], atol=1e-6, rtol=1e-7)
        self.assertEqual(solved["start"], [0.0] * 6)
        self.assertEqual(solved["method"], "hybr")
        self.assertLessEqual(solved["gradient_max_abs"], 1e-8 + 1e-12)
        proof = v.verify_stages(
            self.train,
            pd.Series(self.y, index=self.train.index),
            self.apply,
            self.eta,
            self.query,
            expit(self.query),
            predictions,
            audit,
        )
        self.assertEqual(proof["train_n"], 220)
        self.assertEqual(proof["application_n"], 20)

    def test_scalar_bisection_matches_brent_and_derivative(self):
        from src import event_cluster_models as producer

        b, audit = v.independent_scalar_fit(self.z, self.y, self.eta)
        expected, _ = producer.fit_scalar(self.z, self.y, self.eta)
        self.assertAlmostEqual(b, expected, delta=2e-11)
        step = 1e-5
        numeric = (
            v.scalar_objective(b + step, self.z, self.y, self.eta)[0]
            - v.scalar_objective(b - step, self.z, self.y, self.eta)[0]
        ) / (2 * step)
        self.assertAlmostEqual(audit["gradient"], numeric, delta=2e-11)
        self.assertEqual(audit["method"], "bisect")

    def test_constant_adjacency_varying_expectation_cannot_retune_parent(self):
        from src import event_cluster_models as producer

        rows = [
            v.independent_summary(event_row(list(range(1, 2 * (1 + i % 9), 2))))
            for i in range(220)
        ]
        full = pd.DataFrame(rows, index=self.train.index)
        self.assertGreater(full.expected_adjacency22.nunique(), 1)
        self.assertGreater(full.excess_adjacency22.nunique(), 1)
        train = full.loc[:, v.FEATURES]
        self.assertTrue(train.adjacency_fraction22.eq(0).all())
        self.assertTrue(self.apply.adjacency_fraction22.gt(0).any())
        y = pd.Series(self.y, index=train.index)
        predictions, audit = producer.fit_stages(
            train, y, self.apply, self.eta, self.query, expit(self.query)
        )
        self.assertEqual(audit["cluster"]["coefficient"], 0)
        np.testing.assert_array_equal(predictions["cluster"], predictions["nuisance"])
        v.verify_stages(
            train, y, self.apply, self.eta, self.query, expit(self.query), predictions, audit
        )
        _, _, z, _, _ = v.independent_transform(train, self.apply)
        b, scalar = v.independent_scalar_fit(z, self.y, self.eta)
        self.assertEqual(b, 0)
        self.assertEqual(scalar["status"], "EXACT_CONSTANT_INPUT")

    def test_scalar_signed_underflow_but_no_extra_nuisance_gate(self):
        value, gradient = v.scalar_objective(
            0, np.array([1.0]), np.array([1.0]), np.array([40.0])
        )
        self.assertGreater(value, 0)
        self.assertEqual(gradient, -float(expit(-40)))
        with self.assertRaises(ValueError):
            v.scalar_objective(0, np.array([1.0]), np.array([1.0]), np.array([1000.0]))
        with self.assertRaises(ValueError):
            v.scalar_objective(0, np.array([1e-300]), np.array([1.0]), np.array([100.0]))
        x = np.column_stack([np.ones(2), np.zeros((2, 5))])
        objective, gradient, _ = v.nuisance_objective(
            np.zeros(6), x, np.array([0.0, 1.0]), np.array([-1000.0, 1000.0])
        )
        self.assertEqual(objective, 0)
        np.testing.assert_array_equal(gradient, np.zeros(6))

    def test_independent_root_failure_is_terminal_no_fallback(self):
        fake = SimpleNamespace(
            success=False, x=np.zeros(6), status=4, message="no progress", nfev=10, njev=2
        )
        with patch.object(v, "root", return_value=fake) as solver:
            with self.assertRaises(ValueError):
                v.independent_nuisance_fit(self.x, self.y, self.eta)
            self.assertEqual(solver.call_count, 1)
        fake.success = True
        with patch.object(v, "root", return_value=fake), self.assertRaises(ValueError):
            v.independent_nuisance_fit(self.x, self.y, self.eta)

    def test_scalar_false_convergence_is_rejected(self):
        fake = (20.0, SimpleNamespace(converged=True, iterations=1, function_calls=3))
        with patch.object(v, "bisect", return_value=fake) as solver:
            with self.assertRaises(ValueError):
                v.independent_scalar_fit(self.z, self.y, self.eta)
            self.assertEqual(solver.call_count, 1)

    def test_saved_stage_audits_and_probabilities_cannot_be_changed(self):
        from src import event_cluster_models as producer

        y = pd.Series(self.y, index=self.train.index)
        predictions, audit = producer.fit_stages(
            self.train, y, self.apply, self.eta, self.query, expit(self.query)
        )
        for mutation in (
            "beta",
            "support",
            "center",
            "scalar",
            "calls",
            "negative_max",
            "probability",
        ):
            a, p = copy.deepcopy(audit), copy.deepcopy(predictions)
            if mutation == "beta":
                a["nuisance"]["beta"][1] += 0.01
            elif mutation == "support":
                a["support"]["events"] += 1
            elif mutation == "center":
                a["transform"]["memory_mean"] += 0.01
            elif mutation == "scalar":
                a["cluster"]["coefficient"] += 0.01
            elif mutation == "calls":
                a["cluster"]["function_calls"] = True
            elif mutation == "negative_max":
                a["cluster"]["gradient_max_abs"] = -1e-16
            else:
                p["cluster"][0] += 0.01
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                v.verify_stages(
                    self.train, y, self.apply, self.eta, self.query, expit(self.query), p, a
                )


class RegisteredLayerContracts(unittest.TestCase):
    def test_literal_protocol_and_boolean_mutation(self):
        from pathlib import Path

        import yaml

        p = yaml.safe_load(Path("event_cluster.yaml").read_bytes())
        v.validate_protocol(p)
        p["original_index"]["market_lag"] = True
        with self.assertRaises(ValueError):
            v.validate_protocol(p)
        p = copy.deepcopy(v.CONTRACT)
        p["unexpected"] = 1
        with self.assertRaises(ValueError):
            v.validate_protocol(p)

    def test_source_metadata_and_original_values_admitted_from_checked_snapshots(self):
        import tempfile
        from pathlib import Path

        from src import event_cluster_admission as producer
        from tests.test_event_cluster_admission import Fixture

        with tempfile.TemporaryDirectory() as d:
            root_path = Path(d)
            f = Fixture(root_path)
            pins = producer.collect_input_pins(root_path, f.expected)
            expected, _ = producer.admit_upstream(root_path, f.expected, pins)
            audit, loaded = v.admit_upstream(root_path, f.expected, pins)
            self.assertEqual(audit, expected)
            self.assertIn("features", loaded)
            self.assertFalse(audit["historical_models_refitted"])
            name = next(iter(pins))
            (root_path / name).write_bytes(b"changed")
            with self.assertRaises((ValueError, AssertionError)):
                v.admit_upstream(root_path, f.expected, pins)

    def test_shared_admission_cannot_omit_registered_or_anchor_hashes(self):
        import tempfile
        from pathlib import Path

        from src import event_cluster_admission as source
        from tests.test_event_cluster_admission import Fixture

        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            f = Fixture(r)
            pins = source.collect_input_pins(r, f.expected)
            bad = dict(pins)
            del bad[next(iter(f.expected))]
            with self.assertRaises((ValueError, AssertionError)):
                v.admit_upstream(r, f.expected, bad)

    def test_marker_after_shared_admission_cannot_pass(self):
        import tempfile
        from pathlib import Path

        from src import event_cluster_admission as source
        from tests.test_event_cluster_admission import Fixture

        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            f = Fixture(r)
            pins = source.collect_input_pins(r, f.expected)
            original = source.admit_upstream

            def inject(*args, **kwargs):
                answer = original(*args, **kwargs)
                (r / "reports/range_alert/failure.json").write_text("{}")
                return answer

            with (
                patch.object(source, "admit_upstream", side_effect=inject),
                self.assertRaises((ValueError, AssertionError)),
            ):
                v.admit_upstream(r, f.expected, pins)

    def test_failure_rows_remain_parseable_after_unterminated_bad_ledger(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            report = r / "reports/event_cluster"
            report.mkdir(parents=True)
            ledger = report / "trial_ledger.jsonl"
            ledger.write_bytes(b"broken original record")
            v.invalidate_publication(r, ValueError("synthetic malformed ledger"))
            lines = ledger.read_text().splitlines()
            self.assertEqual(lines[0], "broken original record")
            self.assertEqual(len(lines), 4)
            self.assertTrue(
                all(json.loads(line)["event"] == "verification_failed" for line in lines[1:])
            )

    def test_inherited_values_are_bound_before_decoding(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            (r / "prior.json").write_text("{bad")
            p = {"comparisons": {"inherited_sources": ["prior.json"]}}
            with self.assertRaisesRegex((ValueError, AssertionError), "hash|pin|changed|SHA"):
                v.inherited_rows(r, p, pins={"prior.json": "0" * 64})

    def test_failure_guard_canonical_three_p1_with_malformed_numeric_diagnostics(self):
        import json
        import tempfile
        from pathlib import Path

        for original in (b"{broken", b'{"bad": NaN}', b'{"leads": ["cluster"], "rows": []}'):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as d:
                r = Path(d)
                report = r / "reports/event_cluster"
                report.mkdir(parents=True)
                (report / "metrics.json").write_bytes(original)
                (report / "results.md").write_text("old apparent success")
                (report / "trial_ledger.jsonl").write_text(
                    "".join(
                        json.dumps({"event": "evaluated", "ordinal": i}) + "\n"
                        for i in range(137)
                    )
                )
                previous = r / "reports/issued_calibration/metrics.json"
                previous.parent.mkdir()
                previous.write_bytes(b"prior immutable")
                with (
                    patch.object(v, "verify", side_effect=ValueError("synthetic discrepancy")),
                    self.assertRaises(ValueError),
                ):
                    v.verify_with_failure_guard(r)
                failed = json.loads((report / "metrics.json").read_bytes())
                self.assertEqual(failed["status"], "UNEVALUABLE")
                self.assertEqual(failed["hypothesis_count"], 3)
                self.assertEqual(failed["cumulative_hypothesis_count"], 134)
                self.assertEqual(failed["leads"], [])
                self.assertTrue(
                    all(
                        row[key] == 1
                        for row in failed["rows"]
                        for key in ("p_conservative", "p_holm_wave", "p_holm_cumulative")
                    )
                )
                self.assertEqual(
                    json.loads((report / "verification.json").read_bytes())["status"], "FAILED"
                )
                self.assertEqual(
                    len((report / "trial_ledger.jsonl").read_text().splitlines()), 140
                )
                self.assertNotIn("old apparent success", (report / "results.md").read_text())
                self.assertEqual(previous.read_bytes(), b"prior immutable")


def inference_fixture():
    from src import event_cluster_search as runner

    p = copy.deepcopy(runner.CONTRACT)
    dates = pd.bdate_range("2010-01-04", periods=421)
    origins = dates[1:-1]
    section = p["original_index"]
    section["development"] = [str(dates[1].date()), str(dates[140].date())]
    section["development_target_available_by"] = str(dates[141].date())
    section["evaluation"] = [str(dates[141].date()), str(dates[419].date())]
    section["evaluation_stability"] = [
        [str(dates[141].date()), str(dates[279].date())],
        [str(dates[280].date()), str(dates[419].date())],
    ]
    section["latest_target"] = str(dates[-1].date())
    p["inference"]["bootstrap_draws"] = 199
    rng = np.random.default_rng(8420)
    y = (rng.random(len(origins)) < 0.35).astype(float)
    base = 0.33 + 0.08 * np.sin(np.arange(len(origins)) * 0.07)
    series = {
        "baseline": base,
        "recent_frequency": 0.34 + 0.04 * np.cos(np.arange(len(origins)) * 0.05),
        "nuisance": base * 0.95 + 0.015,
        "cluster": base + 0.008 * np.sin(np.arange(len(origins)) * 0.03),
    }
    frames = []
    for model, probability in series.items():
        frame = pd.DataFrame(
            {
                "origin": origins,
                "model": model,
                "horizon": 1,
                "feature_cutoff_date": dates[:-2],
                "target_end": dates[2:],
                "available_date": dates[2:],
                "y": y,
                "probability": probability,
                "loss": (probability - y) ** 2,
                "fit_origin": origins,
                "fit_cutoff_date": dates[:-2],
                "train_n": 1000,
                "train_last_target": dates[:-2],
                "train_last_available": dates[:-2],
                "phase": np.where(origins <= dates[140], "development", "evaluation"),
            }
        )
        # Monthly fit metadata must be constant and causal within each month.
        for _, members in frame.groupby(frame.origin.dt.to_period("M")).groups.items():
            first = frame.loc[members[0]]
            for key in (
                "fit_origin",
                "fit_cutoff_date",
                "train_last_target",
                "train_last_available",
            ):
                frame.loc[members, key] = first[key]
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True).loc[:, v.PANEL_COLUMNS]
    prior = [
        {
            "study": "synthetic",
            "candidate": str(i),
            "control": "baseline",
            "horizon": 1,
            "p_conservative": 1.0,
            "source": "synthetic.json",
            "source_sha256": "a" * 64,
            "source_row_index": i,
        }
        for i in range(131)
    ]
    metrics = runner.evaluate(panel, dates, p, 21, prior=prior)
    metrics["common_application_origins"] = len(origins) + 1
    return panel, p, metrics, prior


class InferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.p, cls.metrics, cls.prior = inference_fixture()

    def verify(self, metrics=None):
        with patch.object(v, "inherited_rows", return_value=self.prior):
            return v.verify_metrics(
                None,
                self.panel,
                self.p,
                self.metrics if metrics is None else metrics,
                420,
                21,
                admitted_inputs={},
            )

    def test_actual_six_phase_eighteen_bootstraps_and_full_family(self):
        proof = self.verify()
        self.assertEqual(proof["phase_comparisons_verified"], 6)
        self.assertEqual(proof["bootstrap_runs_verified"], 18)
        self.assertEqual(proof["cumulative_hypotheses_verified"], 134)
        self.assertEqual(proof["calibration_rows_verified"], 8)

    def test_counts_uncertainty_support_and_family_mutation_rejected(self):
        for field in (
            "count",
            "boolcount",
            "prior",
            "phase",
            "interval",
            "calibration",
            "holmp",
            "lead",
        ):
            m = copy.deepcopy(self.metrics)
            if field == "count":
                m["new_monthly_fits"] += 1
            elif field == "boolcount":
                m["original_baseline_fits"] = False
            elif field == "prior":
                m["inherited_rows"][0]["p_conservative"] = 0.2
            elif field == "phase":
                m["rows"][1]["phases"][0]["class_support"]["events"] += 1
            elif field == "interval":
                m["rows"][2]["phases"][1]["ci95_envelope"][0] += 0.001
            elif field == "calibration":
                m["calibration"]["development"]["cluster"]["mean_probability"] += 0.01
            elif field == "holmp":
                m["rows"][0]["p_holm_cumulative"] = 0.000001
            else:
                m["leads"] = ["cluster"] if not m["leads"] else []
            with self.subTest(field=field), self.assertRaises((AssertionError, ValueError)):
                self.verify(m)

    def test_three_registered_plus131inherited_plus3evaluated_exact_ledger(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            folder = r / "reports/event_cluster"
            folder.mkdir(parents=True)
            rows = [
                {
                    "event": "registered",
                    "study": "event_cluster",
                    "candidate": a,
                    "control": b,
                    "score": "brier",
                    "horizon": 1,
                    "protocol_sha256": self.metrics["protocol_sha256"],
                }
                for a, b in v.COMPARISONS
            ]
            rows += [{"event": "inherited", **row} for row in self.prior]
            rows += [{"event": "evaluated", **row} for row in self.metrics["rows"]]
            path = folder / "trial_ledger.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            signature = v.digest(path)
            proof = v.verify_ledger(
                r, self.metrics, self.prior, "evaluated", signature=signature
            )
            self.assertEqual(proof, {"inherited": 131, "registered": 3, "evaluated": 3})
            path.write_text(path.read_text() + "{}\n")
            with self.assertRaises((ValueError, AssertionError)):
                v.verify_ledger(r, self.metrics, self.prior, "evaluated", signature=signature)

    def test_strict_endpoint_domain_before_tolerance(self):
        for value in (-1e-16, 1 + 2e-16):
            bad = self.panel.copy()
            bad.loc[0, "probability"] = value
            bad.loc[0, "loss"] = (value - bad.loc[0, "y"]) ** 2
            with self.assertRaises((ValueError, AssertionError)):
                v.validate_panel(bad)


class CompleteEntryContracts(unittest.TestCase):
    def tree(self, root_path):
        import json
        from pathlib import Path

        import yaml

        from src import event_cluster_admission as source
        from tests.test_event_cluster_admission import Fixture

        class CompleteSourceFixture(Fixture):
            def put(self, name, value):
                if name == "range_alert.yaml":
                    value = yaml.safe_dump(
                        {
                            **yaml.safe_load(value),
                            "index": copy.deepcopy(v.CONTRACT["original_index"]),
                        }
                    ).encode()
                return super().put(name, value)

        f = CompleteSourceFixture(root_path)
        closure, loaded = source.admit_upstream(root_path, f.expected)
        report = root_path / "reports/event_cluster"
        out = root_path / "data/event_cluster"
        report.mkdir(parents=True)
        out.mkdir(parents=True)
        p = copy.deepcopy(v.CONTRACT)
        p["upstream"]["anchors"] = f.expected
        p["original_index"] = loaded["protocol"]["index"]
        p["comparisons"]["inherited_sources"] = []
        (root_path / "event_cluster.yaml").write_text(yaml.safe_dump(p))
        signature = v.digest(root_path / "event_cluster.yaml")
        (report / "full_repository_tests.txt").write_text("synthetic prefit log")
        (report / "VERIFIER_DESIGN.md").write_text("synthetic design")
        # Include the actual verifier source as a supplied code artifact, not an imported temporary module.
        (root_path / "src/verify_event_cluster.py").write_bytes(Path(v.__file__).read_bytes())
        code = {
            str(path.relative_to(root_path)): v.digest(path)
            for folder in ("src", "tests")
            for path in (root_path / folder).rglob("*.py")
        }
        designs = {
            "reports/event_cluster/VERIFIER_DESIGN.md": v.digest(report / "VERIFIER_DESIGN.md")
        }
        freeze = {
            "protocol_sha256": signature,
            "code": code,
            "prefit_design": designs,
            "checks": {"full_log_sha256": v.digest(report / "full_repository_tests.txt")},
        }
        (report / "freeze_record.json").write_text(json.dumps(freeze))
        inputs = {
            **closure["files"],
            **designs,
            "reports/event_cluster/freeze_record.json": v.digest(
                report / "freeze_record.json"
            ),
        }
        preserved = {
            str(path.relative_to(root_path)): v.digest(path)
            for path in [*root_path.glob("*.yaml"), *(root_path / "reports").rglob("*")]
            if path.is_file()
            and path != root_path / "event_cluster.yaml"
            and report not in path.parents
            and str(path.relative_to(root_path)) not in inputs
        }
        manifest = {
            "protocol_sha256": signature,
            "code": code,
            "inputs": inputs,
            "preserved": preserved,
        }
        (report / "manifest.json").write_text(json.dumps(manifest))
        (out / "upstream_admission.json").write_text(json.dumps(closure))
        for name, table in [
            ("forecasts", loaded["forecasts"]),
            ("states", loaded["states"]),
            ("memory", loaded["features"]),
        ]:
            table.to_parquet(out / (name + ".parquet"))
        (out / "fits.json").write_text("[]")
        (out / "support_audit.json").write_text('{"synthetic_support": true}')
        metrics = {
            "protocol_sha256": signature,
            "evidence_class": p["evidence_class"],
            "rows": [],
            "inherited_rows": [],
        }
        (report / "metrics.json").write_text(json.dumps(metrics))
        (report / "trial_ledger.jsonl").write_text('{"event":"synthetic"}\n')
        checked = {
            "forecasts_verified": 4,
            "new_forecasts": 2,
            "reused_control_forecasts": 2,
            "new_monthly_fits": 1,
            "common_scored_origins": 1,
            "common_application_origins": 2,
        }
        return p, metrics, checked

    def run_tree(self, r, checked, pipeline_side_effect=None, inference_side_effect=None):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch.object(v, "validate_protocol"))
        stack.enter_context(
            patch.object(
                v, "verify_pipeline", return_value=checked, side_effect=pipeline_side_effect
            )
        )
        stack.enter_context(
            patch.object(
                v,
                "verify_metrics",
                return_value={"synthetic": True},
                side_effect=inference_side_effect,
            )
        )
        stack.enter_context(
            patch.object(
                v,
                "verify_ledger",
                return_value={"inherited": 131, "registered": 3, "evaluated": 3},
            )
        )
        with stack:
            return v.verify(r)

    def test_complete_paths_real_admission_snapshots_and_eight_verified_outputs(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            _, _, checked = self.tree(r)
            result = self.run_tree(r, checked)
            self.assertEqual(result["status"], "VERIFIED")
            self.assertEqual(len(result["verified_output_hashes"]), 8)
            for name, signature in result["verified_output_hashes"].items():
                self.assertEqual(v.digest(r / name), signature)
            self.assertTrue((r / "reports/event_cluster/verification.json").exists())
            self.assertEqual(result["forecast_reconstruction"], checked)

    def test_output_mutation_during_reconstruction_blocks_commit(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            _, _, checked = self.tree(r)

            def mutate(*args):
                (r / "data/event_cluster/fits.json").write_text("[{}]")
                return checked

            with self.assertRaises((ValueError, AssertionError)):
                self.run_tree(r, checked, pipeline_side_effect=mutate)
            self.assertFalse((r / "reports/event_cluster/verification.json").exists())

    def test_new_upstream_failure_at_final_commit_blocks_success(self):
        import tempfile
        from pathlib import Path

        for prior in ("range_alert", "issued_calibration"):
            with self.subTest(prior=prior), tempfile.TemporaryDirectory() as d:
                r = Path(d)
                _, _, checked = self.tree(r)

                def mutate(*args, r=r, prior=prior, **kwargs):
                    (r / f"reports/{prior}/failure.json").write_text("{}")
                    return {"synthetic": True}

                with self.assertRaises((ValueError, AssertionError)):
                    self.run_tree(r, checked, inference_side_effect=mutate)
                self.assertFalse((r / "reports/event_cluster/verification.json").exists())

    def test_missing_memory_output_fails_before_pipeline(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            _, _, checked = self.tree(r)
            (r / "data/event_cluster/memory.parquet").unlink()
            with self.assertRaises((FileNotFoundError, ValueError, AssertionError)):
                self.run_tree(r, checked)


if __name__ == "__main__":
    unittest.main()
