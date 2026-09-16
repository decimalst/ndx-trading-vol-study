"""Prewritten generated expert, issuance and independent gate contracts."""

import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from src import verify_precision_gate_forecasts as verify

MARKET = (
    "const",
    "lrv_d",
    "lrv_w",
    "lrv_m",
    "lev_d",
    "lev_w",
    "lev_m",
    "liv",
    "lvix",
    "term",
    "xasset_stress",
    "market_stress",
)


def raw_fixture():
    index = pd.bdate_range("2010-01-04", "2013-04-10")
    rng = np.random.default_rng(384091)
    n = len(index)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    opened = close * np.exp(rng.normal(0, 0.008, n))
    daily = pd.DataFrame(
        {
            "open": opened,
            "high": np.maximum(opened, close) * np.exp(rng.uniform(0.002, 0.016, n)),
            "low": np.minimum(opened, close) * np.exp(-rng.uniform(0.002, 0.016, n)),
            "close": close,
            "adj close": close * np.exp(np.cumsum(rng.normal(0, 0.0001, n))),
            "volume": np.exp(rng.normal(14, 0.4, n)),
        },
        index=index,
    )
    cross = pd.DataFrame(
        80 * np.exp(np.cumsum(rng.normal(0, 0.012, (n, 5)), axis=0)),
        index=index,
        columns=("hyg", "tlt", "gld", "uso", "uup"),
    )
    iv = pd.DataFrame(
        np.exp(rng.normal(3, 0.2, (n, 3))), index=index, columns=("vxn", "vix", "vix9d")
    )
    config = dict(
        issuance_start="2010-01-04",
        origin_start="2012-01-01",
        origin_end="2013-04-10",
        source_end="2013-04-10",
        development=["2012-01-01", "2012-12-31"],
        evaluation=["2013-01-01", "2013-04-10"],
        minimum_train=160,
        adaptive_half_life=63,
        gate_window=90,
        gate_minimum_train=30,
    )
    return dict(daily=daily, cross=cross, iv=iv), config


def produce(sources, config):
    # Production routines are called only on generated agreement/corruption cases.
    from src.claims_release_market import build_market_features, make_five_session_targets
    from src.precision_gate_pipeline import build_panel

    features = build_market_features(**sources)
    targets = make_five_session_targets(features.rv_total)
    return features, targets, build_panel(features, targets, config)


def gate_fixture():
    rng = np.random.default_rng(670044)
    base = np.exp(rng.normal(-7, 0.4, 90))
    adaptive = base * np.exp(rng.normal(0, 0.7, 90))
    state = rng.normal(0, 1, 90)
    w = (1 - np.tanh(state)) / 2 * 0.2 + (1 + np.tanh(state)) / 2 * 0.7
    y = base / (1 - w + w * base / adaptive) * np.exp(rng.normal(-0.02, 0.2, 90))
    return base, adaptive, y, state


class IndependentGateTests(unittest.TestCase):
    def test_certificate_matches_separate_optimum_and_literal_forecast(self):
        from src.precision_gate import fit_gate

        b, a, y, s = gate_fixture()
        for constant in (False, True):
            audit = fit_gate(b, a, y, s, constant=constant)
            coeff = verify._verify_gate(b, a, y, s, audit, constant=constant)
            np.testing.assert_array_equal(coeff, audit["coefficients"])
            z = np.tanh(s)
            w = (1 - z) / 2 * coeff[0] + (1 + z) / 2 * coeff[1]
            expected = b / (1 - w + w * b / a)
            np.testing.assert_allclose(
                verify._gate_prediction(b, a, s, coeff), expected, rtol=1e-14
            )
            if constant:
                self.assertEqual(coeff[0], coeff[1])

    def test_interior_optimum_known_analytically(self):
        from src.precision_gate import fit_gate

        x = np.array([[0.75, 0.25], [0.25, 0.75]])
        c = np.array([0.25, 0.65])
        b = np.ones(2)
        a = b / 2
        s = np.arctanh([-0.5, 0.5])
        y = 1 / (1 + x @ c) - 0.02 * np.linalg.solve(x.T, c)
        audit = fit_gate(b, a, y, s)
        np.testing.assert_allclose(
            verify._verify_gate(b, a, y, s, audit), c, atol=1e-7, rtol=0
        )

    def test_gate_gradient_objective_stationarity_and_identity_tampering(self):
        from src.precision_gate import fit_gate

        b, a, y, s = gate_fixture()
        original = fit_gate(b, a, y, s)
        for key in (
            "coefficients",
            "objective",
            "gradient",
            "optimization_gradient",
            "projected_gradient_max_abs",
            "constant",
            "n_train",
            "n_iterations",
            "solver",
            "success",
        ):
            audit = copy.deepcopy(original)
            if key == "coefficients" or key in ("gradient", "optimization_gradient"):
                audit[key][0] += 0.01
            elif key in ("constant", "success"):
                audit[key] = not audit[key]
            elif key == "solver":
                audit[key] = "SLSQP"
            elif key == "n_iterations":
                audit[key] = 1001
            else:
                audit[key] += 0.1
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify._verify_gate(b, a, y, s, audit)

    def test_identical_experts_boundary_and_tiny_ratio_prediction(self):
        from src.precision_gate import fit_gate

        b, _a, y, s = gate_fixture()
        audit = fit_gate(b, b, y, s)
        np.testing.assert_array_equal(verify._verify_gate(b, b, y, s, audit), [0.0, 0.0])
        np.testing.assert_allclose(
            verify._gate_prediction([1e-300], [1.0], [0.0], [1.0, 1.0]), [1.0]
        )
        for level, expected in [(10.0, 0.0), (0.1, 1.0)]:
            base = np.ones(40)
            adaptive = np.full(40, 0.5)
            target = np.full(40, level)
            state = np.linspace(-2, 2, 40)
            audit = fit_gate(base, adaptive, target, state)
            np.testing.assert_array_equal(
                verify._verify_gate(base, adaptive, target, state, audit), [expected] * 2
            )

    def test_separate_optimizer_contract_failure_and_input_preservation(self):
        from src.precision_gate import fit_gate

        b, a, y, s = gate_fixture()
        audit = fit_gate(b, a, y, s)
        old = [x.copy() for x in (b, a, y, s)]
        with patch.object(verify, "minimize", wraps=verify.minimize) as optimizer:
            verify._verify_gate(b, a, y, s, audit)
        self.assertEqual(optimizer.call_count, 1)
        self.assertEqual(optimizer.call_args.kwargs["method"], "SLSQP")
        self.assertEqual(
            optimizer.call_args.kwargs["options"], {"ftol": 1e-14, "maxiter": 2000}
        )
        with (
            patch.object(
                verify,
                "minimize",
                return_value=OptimizeResult(success=False, x=np.zeros(2), message="injected"),
            ),
            self.assertRaises(ValueError),
        ):
            verify._verify_gate(b, a, y, s, audit)
        for actual, prior in zip((b, a, y, s), old):
            np.testing.assert_array_equal(actual, prior)


class IndependentExpertTests(unittest.TestCase):
    def test_weighted_qr_scale_and_smearing_against_literal_weighted_fit(self):
        rng = np.random.default_rng(3771)
        train = pd.DataFrame(rng.normal(size=(140, 12)), columns=MARKET)
        train["const"] = 1.0
        query = train.iloc[:6].copy()
        target = pd.Series(np.exp(rng.normal(-8, 0.5, 140)), index=train.index)
        positions = np.arange(140) * 3
        cutoff = positions[-1] + 7
        for half_life in (None, 63):
            forecast, audit = verify._expert_expected(
                train, target, query, positions, cutoff, half_life
            )
            weight = (
                np.ones(140)
                if half_life is None
                else np.exp2(-(cutoff - positions) / half_life)
            )
            x = train.to_numpy()
            rawbeta = np.linalg.lstsq(
                np.sqrt(weight)[:, None] * x, np.sqrt(weight) * np.log(target), rcond=1e-12
            )[0]
            residual = np.log(target) - x @ rawbeta
            smear = np.sum(weight * np.exp(residual)) / weight.sum()
            expected = np.exp(query.to_numpy() @ rawbeta) * smear
            np.testing.assert_allclose(forecast, expected, atol=1e-14, rtol=1e-10)
            means = np.sum(weight[:, None] * x[:, 1:], axis=0) / weight.sum()
            scales = np.sqrt(
                np.sum(weight[:, None] * (x[:, 1:] - means) ** 2, axis=0) / weight.sum()
            )
            np.testing.assert_allclose(
                list(audit["feature_means"].values()), means, rtol=1e-12, atol=1e-14
            )
            np.testing.assert_allclose(
                list(audit["feature_scales"].values()), scales, rtol=1e-12
            )
            self.assertAlmostEqual(audit["log_smearing"], math.log(smear), places=12)

    def test_calendar_gaps_change_decay_but_query_mutation_does_not_refit(self):
        rng = np.random.default_rng(991)
        train = pd.DataFrame(rng.normal(size=(70, 12)), columns=MARKET)
        train["const"] = 1.0
        y = pd.Series(np.exp(rng.normal(-8, 0.4, 70)), index=train.index)
        query = train.iloc[:4].copy()
        positions = np.arange(70) * 4
        a, fit = verify._expert_expected(train, y, query, positions, 300, 63)
        b, _ = verify._expert_expected(train, y, query, np.arange(70), 300, 63)
        self.assertFalse(np.allclose(a, b, rtol=1e-5, atol=0))
        query.iloc[-1, 1:] += 1
        changed, other = verify._expert_expected(train, y, query, positions, 300, 63)
        self.assertEqual(fit, other)
        np.testing.assert_array_equal(a[:-1], changed[:-1])
        train["lrv_m"] = 1.0
        with self.assertRaises(ValueError):
            verify._expert_expected(train, y, query, positions, 300, 63)


class FullIssuanceVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources, cls.config = raw_fixture()
        cls.features, cls.targets, cls.produced = produce(cls.sources, cls.config)

    def check(self, **changes):
        args = dict(
            sources=self.sources,
            features=self.features,
            targets=self.targets,
            produced=self.produced,
            config=self.config,
        )
        return verify.verify_forecasts(**(args | changes))

    def test_full_generated_agreement_with_warmup_coldstart_and_all_applications(self):
        proof = self.check()
        self.assertEqual(proof["status"], "VERIFIED")
        self.assertEqual(proof["calendar_rows"], len(self.features))
        self.assertEqual(
            proof["application_forecasts_verified"], 4 * len(self.produced["applications"])
        )
        self.assertGreater(
            proof["application_forecasts_verified"], proof["forecasts_verified"]
        )
        self.assertIn("expert_warmup", set(self.produced["coverage"].status))
        self.assertEqual(
            set(self.produced["applications"].gate_status), {"cold_start", "fitted"}
        )

    def test_no_pipeline_producer_can_be_called_by_verifier(self):
        with (
            patch(
                "src.precision_gate_pipeline.build_panel",
                side_effect=AssertionError("producer replay"),
            ),
            patch("src.precision_gate.fit_gate", side_effect=AssertionError("producer gate")),
        ):
            self.assertEqual(self.check()["status"], "VERIFIED")

    def test_features_targets_full_calendar_and_state_are_checked(self):
        for key, col in [
            ("features", "lrv_d"),
            ("features", "rv_total"),
            ("targets", "y"),
            ("targets", "target_end"),
        ]:
            frame = getattr(self, key).copy()
            i = frame.index[-20]
            frame.loc[i, col] += pd.Timedelta(days=1) if col == "target_end" else 0.01
            with self.subTest(key=key, col=col), self.assertRaises(ValueError):
                self.check(**{key: frame})
        produced = copy.deepcopy(self.produced)
        produced["applications"].loc[0, "state"] += 0.01
        with self.assertRaises(ValueError):
            self.check(produced=produced)
        with self.assertRaises(ValueError):
            self.check(features=self.features.iloc[1:])

    def test_all_four_outputs_and_unscored_predictions_checked(self):
        for name, col in [
            ("applications", "pred_adaptive"),
            ("applications", "pred_constant"),
            ("applications", "pred_contextual"),
            ("panel", "prediction"),
            ("coverage", "status"),
            ("coverage", "offset"),
            ("schedules", "training_cutoff"),
        ]:
            produced = copy.deepcopy(self.produced)
            i = len(produced[name]) - 1
            old = produced[name].loc[i, col]
            produced[name].loc[i, col] = (
                "scored"
                if col == "status"
                else old + pd.Timedelta(days=1)
                if col == "training_cutoff"
                else old + 1
            )
            with self.subTest(name=name, col=col), self.assertRaises(ValueError):
                self.check(produced=produced)
        for name in ("applications", "panel", "coverage", "schedules"):
            produced = copy.deepcopy(self.produced)
            produced[name] = produced[name].iloc[:-1]
            with self.subTest(drop=name), self.assertRaises(ValueError):
                self.check(produced=produced)

    def test_calendar_ceiling_before_numerical_market_reconstruction(self):
        sources = copy.deepcopy(self.sources)
        sources["iv"].loc[pd.Timestamp("2025-10-21")] = 1.0
        with patch.object(
            verify, "_market_targets", side_effect=AssertionError("decoded")
        ) as arithmetic:
            with self.assertRaises(ValueError):
                self.check(sources=sources)
            arithmetic.assert_not_called()

    def test_first_query_missing_label_does_not_move_monthly_schedule(self):
        sources = copy.deepcopy(self.sources)
        first = self.produced["applications"].iloc[0].fit_origin
        position = sources["daily"].index.get_loc(first)
        sources["daily"].loc[sources["daily"].index[position + 1], "high"] = np.nan
        f, t, out = produce(sources, self.config)
        self.assertTrue(pd.isna(t.loc[first, "y"]))
        self.assertEqual(out["applications"].iloc[0].fit_origin, first)
        self.assertEqual(
            self.check(sources=sources, features=f, targets=t, produced=out)["status"],
            "VERIFIED",
        )

    def test_generated_inputs_and_saved_outputs_remain_unchanged(self):
        sources = copy.deepcopy(self.sources)
        features = self.features.copy()
        targets = self.targets.copy()
        produced = copy.deepcopy(self.produced)
        self.check()
        for key in sources:
            pd.testing.assert_frame_equal(self.sources[key], sources[key])
        pd.testing.assert_frame_equal(self.features, features)
        pd.testing.assert_frame_equal(self.targets, targets)
        for key in ("applications", "panel", "coverage", "schedules"):
            pd.testing.assert_frame_equal(self.produced[key], produced[key])
        self.assertEqual(self.produced["fits"], produced["fits"])

    def test_literal_issued_gate_membership_window_and_maturity(self):
        issued = self.produced["applications"].set_index("origin")
        index = self.features.index
        for fit in self.produced["fits"]:
            first = index.get_loc(pd.Timestamp(fit["fit_origin"]))
            if fit["status"] == "expert_warmup":
                self.assertEqual(fit["application_origins"], [])
                continue
            expected = [
                i
                for i, day in enumerate(index)
                if first - self.config["gate_window"] <= i < first
                and day in issued.index
                and np.isfinite(self.targets.y.iloc[i])
                and pd.notna(self.targets.target_end.iloc[i])
                and self.targets.target_end.iloc[i] <= index[first - 1]
            ]
            self.assertEqual(fit["gate_train_positions"], expected)
            self.assertEqual(
                fit["gate_train_origins"], [index[i].strftime("%Y-%m-%d") for i in expected]
            )
            self.assertEqual(fit["gate_n"], len(expected))
            self.assertTrue(
                all(
                    day not in fit["planned_application_origins"]
                    for day in fit["gate_train_origins"]
                )
            )

    def test_gate_membership_and_weight_audit_tampering(self):
        for field in (
            "train_positions",
            "gate_train_origins",
            "gate_train_positions",
            "weights",
            "gate_coefficients",
        ):
            produced = copy.deepcopy(self.produced)
            fit = next(row for row in produced["fits"] if row["gate_status"] == "fitted")
            if field in ("train_positions", "gate_train_positions"):
                fit[field][-1] += 1
            elif field == "gate_train_origins":
                fit[field][-1] = fit["fit_origin"]
            elif field == "weights":
                fit["expert_audits"]["adaptive"]["weights"][0] *= 1.05
            else:
                fit["gate_audits"]["contextual"]["coefficients"][0] += 0.01
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(produced=produced)

    def test_cold_start_is_retained_and_both_gates_equal_base_exactly(self):
        apps = self.produced["applications"]
        cold = apps.loc[apps.gate_status == "cold_start"]
        self.assertGreater(len(cold), 0)
        np.testing.assert_array_equal(cold.pred_base, cold.pred_constant)
        np.testing.assert_array_equal(cold.pred_base, cold.pred_contextual)
        produced = copy.deepcopy(self.produced)
        fit = next(row for row in produced["fits"] if row["gate_status"] == "cold_start")
        fit["gate_audits"]["constant"]["coefficients"] = [0.1, 0.1]
        with self.assertRaises(ValueError):
            self.check(produced=produced)

    def test_scoring_period_insufficiency_is_fatal_to_independent_verification(self):
        config = self.config | {"minimum_train": 10000}
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            self.check(config=config)


if __name__ == "__main__":
    unittest.main()
