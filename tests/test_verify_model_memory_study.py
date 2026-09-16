"""Pre-score synthetic contracts for the independent model-memory verifier."""
import copy
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import SplineTransformer

from src import verify_model_memory_study as v


def estimator_fixture():
    """External estimator-library fixture, without producer model helpers."""
    rng = np.random.default_rng(2031)
    train = pd.DataFrame(rng.normal(size=(140, 12)), columns=v.BASELINE)
    query = pd.DataFrame(rng.normal(size=(5, 12)), columns=v.BASELINE)
    train["const"], query["const"] = 1, 1
    y = np.exp(-8 + 0.2 * train["lrv_d"].to_numpy() + rng.normal(0, 0.3, len(train)))
    ages = np.arange(len(train), 0, -1)
    raw, apply = train.iloc[:, 1:].to_numpy(), query.iloc[:, 1:].to_numpy()
    mean, scale = raw.mean(0), raw.std(0)
    z, zquery = (raw - mean) / scale, (apply - mean) / scale
    names = ["lrv_d", "lrv_w", "lrv_m", "liv", "lvix", "term"]
    selected = [list(v.BASELINE[1:]).index(name) for name in names]
    remaining = [i for i in range(11) if i not in selected]
    spline = SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False, extrapolation="linear")
    b = np.column_stack([spline.fit_transform(z[:, selected]), z[:, remaining]])
    bq = np.column_stack([spline.transform(zquery[:, selected]), zquery[:, remaining]])
    bmean, bscale = b.mean(0), b.std(0)
    audits, predictions = {}, {}
    for name in v.COMPONENTS:
        design, design_query = ((b - bmean) / bscale, (bq - bmean) / bscale) if name == "spline_gamma" else (z, zquery)
        weights = np.exp2(-ages / 252) if name.startswith("adaptive") else np.ones(len(train))
        audit = {"input_columns": list(v.BASELINE[1:]), "input_center": mean.tolist(), "input_scale": scale.tolist(),
                 "converged": True, "n_train": len(train), "weight_sum": float(sum(weights)),
                 "effective_sample_size": float(sum(weights)**2 / sum(weights**2))}
        if name == "spline_gamma":
            audit.update({"spline_knots": np.column_stack([s.t for s in spline.bsplines_]).tolist(),
                          "design_center": bmean.tolist(), "design_scale": bscale.tolist()})
        if name in ("baseline", "adaptive_ols"):
            x = np.column_stack([np.ones(len(train)), design])
            beta = np.linalg.lstsq(x * np.sqrt(weights[:, None]), np.log(y) * np.sqrt(weights), rcond=None)[0]
            smear = sum(np.exp(np.log(y) - x @ beta) * weights) / sum(weights)
            audit.update({"coefficients": beta[1:].tolist(), "intercept": float(beta[0]), "smearing_factor": float(smear)})
            predictions[name] = np.exp(beta[0] + design_query @ beta[1:]) * smear
        else:
            alpha = 0.01 if name in ("ridge_gamma", "spline_gamma") else 0.0
            units = float(np.median(y))
            model = GammaRegressor(alpha=alpha, solver="newton-cholesky", tol=1e-9, max_iter=2000).fit(design, y / units, sample_weight=weights)
            intercept = model.intercept_ + np.log(units)
            weighted_score = weights * (1 - y / np.exp(intercept + design @ model.coef_)) / sum(weights)
            gradient = np.r_[sum(weighted_score), design.T @ weighted_score + alpha * model.coef_]
            audit.update({"coefficients": model.coef_.tolist(), "intercept": float(intercept), "alpha": alpha,
                          "target_scale": units, "gradient_inf_norm": float(max(abs(gradient))), "n_iter": model.n_iter_})
            predictions[name] = np.exp(intercept + design_query @ model.coef_)
        audits[name] = audit
    protocol = {"estimators": {"spline": {"variables": names}, "adaptive_half_life_sessions": 252,
                               "ridge_alpha": 0.01, "gamma_max_iter": 2000}}
    return train, y, query, ages, audits, protocol, predictions


class IndependentEstimatorVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = estimator_fixture()

    def test_all_model_maps_gradients_and_independent_gamma_refits(self):
        predictions, gradient = v.verify_model_fit(*self.fixture[:6], refit_gamma=True)
        for name, expected in self.fixture[6].items():
            np.testing.assert_allclose(predictions[name], expected, rtol=1e-7)
        self.assertLess(gradient, 2e-7)

    def test_changed_coefficients_are_detected_for_every_model(self):
        for name in v.COMPONENTS:
            arguments = list(self.fixture[:6])
            audits = copy.deepcopy(arguments[4])
            audits[name]["coefficients"][0] += 0.01
            arguments[4] = audits
            with self.subTest(model=name), self.assertRaises(AssertionError):
                v.verify_model_fit(*arguments)

    def test_future_fitted_scaler_knots_and_smearing_are_detected(self):
        for name, field in (("gamma", "input_center"), ("spline_gamma", "design_scale"), ("baseline", "smearing_factor")):
            arguments = list(self.fixture[:6])
            audits = copy.deepcopy(arguments[4])
            if field == "smearing_factor":
                audits[name][field] *= 1.01
            else:
                audits[name][field][0] += 0.01
            arguments[4] = audits
            with self.subTest(field=field), self.assertRaises(AssertionError):
                v.verify_model_fit(*arguments)


def layer_fixture():
    """Assemble valid recorded layers without calling verifier/producer helpers."""
    rng = np.random.default_rng(9641)
    dates = pd.bdate_range("2022-01-03", periods=220)
    features = pd.DataFrame(rng.normal(size=(220, 12)), index=dates, columns=v.BASELINE)
    features["const"] = 1
    latents = pd.DataFrame(rng.normal(size=(220, 6)), index=dates)
    history, scoring = np.arange(10, 206), np.arange(184, 206)
    models = list(v.COMPONENTS) + ["global_calibration", "observable_memory", "latent_memory", "equal_ensemble", "dynamic_ensemble"]
    protocol = {"sample": {"horizons": [1, 5], "score_start": str(dates[184].date()), "score_end": str(dates[205].date())},
                "models": models,
                "memory": {"lookback_sessions": 100, "additional_gap_after_target_sessions": 5,
                           "min_records": 15, "neighbors": 4, "shrinkage": 0.25, "pca_components": 2},
                "ensemble": {"lookback_sessions": 40, "min_records": 15, "half_life_sessions": 63,
                             "temperature": 5, "equal_weight_floor": 0.1}}
    components, forecasts, audits, neighbors = [], [], [], []
    for horizon in (1, 5):
        predictions = np.exp(rng.normal(-8, 0.2, size=(len(history), 6)))
        actual = np.exp(rng.normal(-8, 0.3, size=len(history)))
        for j, i in enumerate(history):
            for k, name in enumerate(v.COMPONENTS):
                components.append({"origin": dates[i], "horizon": horizon, "model": name,
                                   "target_end": dates[i + horizon], "y": actual[j], "prediction": predictions[j, k],
                                   "fit_origin": dates[i], "train_n": i - horizon, "train_last_target": dates[i]})
        months = {}
        for i in scoring:
            months.setdefault(dates[i].to_period("M"), []).append(i)
        for positions in months.values():
            fit = positions[0]
            fitting = [j for j, i in enumerate(history) if fit - 100 <= i < fit and i + horizon + 5 <= fit]
            x = features.iloc[history][list(v.BASELINE[1:])].to_numpy()
            center, scale = x[fitting].mean(0), x[fitting].std(0)
            observed_keys = (x - center) / scale
            z = latents.iloc[history].to_numpy()
            latent_center = z[fitting].mean(0)
            _, singular, rotation = np.linalg.svd(z[fitting] - latent_center, full_matrices=False)
            retained_scale = np.sqrt(sum(singular[:2]**2) / (len(fitting) - 1))
            latent_keys = (z - latent_center) @ rotation[:2].T / retained_scale
            for i in positions:
                query = int(np.flatnonzero(history == i)[0])
                pool = [j for j, old in enumerate(history) if i - 100 <= old < i and old + horizon + 5 <= i]
                experts = [j for j, old in enumerate(history) if i - 40 <= old < i and old + horizon <= i]
                ratio = actual / predictions[:, 0]
                correction = 0.75 + 0.25 * sum(ratio[j] for j in pool) / len(pool)
                row = {"origin": dates[i], "horizon": horizon, "key_fit_origin": dates[fit], "pool_n": len(pool),
                       "ensemble_n": len(experts), "max_memory_target": dates[max(history[j] + horizon for j in pool)],
                       "max_ensemble_target": dates[max(history[j] + horizon for j in experts)], "global_multiplier": correction}
                final = dict(zip(v.COMPONENTS, predictions[query], strict=True))
                final["global_calibration"] = final["baseline"] * correction
                for kind, keys in (("observable", observed_keys), ("latent", latent_keys)):
                    distance = [sum((keys[j] - keys[query])**2) for j in pool]
                    ranked = sorted(range(len(pool)), key=lambda k: (distance[k], k))[:4]
                    selected = [pool[k] for k in ranked]
                    correction = 0.75 + 0.25 * sum(ratio[j] for j in selected) / 4
                    row[f"{kind}_multiplier"] = correction
                    final[f"{kind}_memory"] = final["baseline"] * correction
                    for rank, k in enumerate(ranked):
                        neighbors.append({"origin": dates[i], "horizon": horizon, "kind": kind, "rank": rank,
                                          "neighbor_origin": dates[history[pool[k]]], "distance": distance[k], "ratio": ratio[pool[k]]})
                expert_ratio = actual[experts, None] / predictions[experts]
                losses = expert_ratio - np.log(expert_ratio) - 1
                temporal = 2**(-(i - history[experts]) / 63)
                mean_losses = sum(losses[k] * temporal[k] for k in range(len(experts))) / sum(temporal)
                raw_weights = np.exp(-5 * mean_losses)
                weights = 0.1 / 6 + 0.9 * raw_weights / sum(raw_weights)
                for name, weight in zip(v.COMPONENTS, weights, strict=True):
                    row[f"weight_{name}"] = weight
                final["equal_ensemble"] = sum(predictions[query]) / 6
                final["dynamic_ensemble"] = sum(weights * predictions[query])
                audits.append(row)
                for name, prediction in final.items():
                    forecasts.append({"origin": dates[i], "horizon": horizon, "model": name, "target_end": dates[i + horizon],
                                      "y": actual[query], "prediction": prediction, "fit_origin": dates[fit],
                                      "train_n": i - horizon, "train_last_target": dates[i]})
    return features, latents, pd.DataFrame(components), pd.DataFrame(forecasts), pd.DataFrame(audits), pd.DataFrame(neighbors), protocol


class FullLayerVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = layer_fixture()

    def verify(self, replacement=None):
        fixture = list(self.fixture)
        if replacement:
            index, value = replacement
            fixture[index] = value
        return v.verify_layers(*fixture)

    def test_accepts_independently_constructed_svd_memory_and_dynamic_ensemble(self):
        result = self.verify()
        self.assertEqual(result["final_forecasts_checked"], len(self.fixture[3]))
        self.assertEqual(result["neighbor_records_checked"], len(self.fixture[5]))

    def test_rejects_single_forecast_and_whole_origin_omission(self):
        frame = self.fixture[3]
        for bad in (frame.iloc[1:], frame.loc[frame["origin"] != frame["origin"].min()]):
            with self.assertRaises(AssertionError):
                self.verify((3, bad))

    def test_rejects_forecast_target_or_multiplier_tampering(self):
        for which, column in ((3, "prediction"), (3, "y"), (4, "global_multiplier"), (4, "latent_multiplier"), (4, "weight_gamma")):
            bad = self.fixture[which].copy()
            bad.loc[bad.index[0], column] *= 1.1
            with self.subTest(column=column), self.assertRaises(AssertionError):
                self.verify((which, bad))

    def test_rejects_neighbor_distance_ratio_and_identity_corruption(self):
        for column in ("distance", "ratio", "neighbor_origin"):
            bad = self.fixture[5].copy()
            if column == "neighbor_origin":
                bad.loc[0, column] = bad.loc[0, "origin"]
            else:
                bad.loc[0, column] *= 1.1
            with self.subTest(column=column), self.assertRaises(AssertionError):
                self.verify((5, bad))

    def test_rejects_changed_fitting_date_or_maturity_audit(self):
        for column in ("key_fit_origin", "max_memory_target", "max_ensemble_target"):
            bad = self.fixture[4].copy()
            bad.loc[0, column] += pd.Timedelta(days=1)
            with self.subTest(column=column), self.assertRaises(AssertionError):
                self.verify((4, bad))


class MemoryReconstructionTests(unittest.TestCase):
    def test_retrieval_requires_completed_target_and_exact_extra_session_gap(self):
        dates = pd.bdate_range("2024-01-02", periods=60)
        origins = dates[:45]
        ends = dates[5:50]
        positions = v.eligible_positions(dates, origins, ends, dates[40], 30, 22)
        np.testing.assert_array_equal(positions, np.arange(10, 14))
        # A five-session target ending at position18 matures only at40.
        previous = v.eligible_positions(dates, origins, ends, dates[39], 30, 22)
        self.assertNotIn(13, previous)

    def test_bad_timestamp_and_negative_horizon_fail_closed(self):
        dates = pd.bdate_range("2024-01-02", periods=40)
        with self.assertRaises(AssertionError):
            v.eligible_positions(dates, dates[:10], dates[:10], "2024-01-06", 20, 2)
        with self.assertRaises(AssertionError):
            v.eligible_positions(dates, dates[1:11], dates[:10], dates[30], 20, 2)

    def test_ratio_is_arithmetic_oos_ratio_with_convex_shrinkage(self):
        correction = v.ratio_correction([2, 8, 3], [1, 2, 3], 0.25)
        self.assertAlmostEqual(correction, 0.75 + 0.25 * 7 / 3)
        self.assertNotAlmostEqual(correction, 0.75 + 0.25 * np.exp(np.log([2, 4, 1]).mean()))
        self.assertEqual(v.ratio_correction([0.001], [1000], 0), 1)

    def test_missing_zero_negative_or_empty_memory_is_rejected(self):
        for actual, forecast in [([], []), ([1], [0]), ([-1], [1]), ([np.nan], [1]), ([1], [np.inf])]:
            with self.subTest(actual=actual, forecast=forecast), self.assertRaises(AssertionError):
                v.ratio_correction(actual, forecast, 0.25)

    def test_neighbor_ties_use_oldest_row_and_report_squared_distance(self):
        keys = [[1, 0], [-1, 0], [0, 2], [0.1, 0.1]]
        index, distance = v.nearest_indices(keys, [0, 0], 3)
        np.testing.assert_array_equal(index, [3, 0, 1])
        np.testing.assert_allclose(distance, [0.02, 1, 1])

    def test_missing_keys_and_silent_neighbor_reduction_rejected(self):
        with self.assertRaises(AssertionError):
            v.nearest_indices([[0, 0]], [0, 0], 2)
        with self.assertRaises(AssertionError):
            v.nearest_indices([[0, np.nan]], [0, 0], 1)

    def test_ensemble_equal_losses_are_equal_weights(self):
        np.testing.assert_allclose(v.expert_weights(np.ones((5, 6)), np.arange(5)), np.repeat(1 / 6, 6))

    def test_ensemble_uses_decay_floor_and_finished_losses(self):
        losses = np.array([[0, 1], [1, 0]])
        actual = v.expert_weights(losses, [63, 0], floor_mass=0.1)
        expected_scores = np.exp(-5 * np.array([2 / 3, 1 / 3]))
        expected = 0.05 + 0.9 * expected_scores / expected_scores.sum()
        np.testing.assert_allclose(actual, expected)
        with self.assertRaises(AssertionError):
            v.expert_weights(losses, [-1, 0])

    def test_target_requires_every_future_session_and_preserves_end_date(self):
        dates = pd.bdate_range("2024-01-02", periods=8)
        features = pd.DataFrame({"rv_total": [1, 2, 3, np.nan, 5, 6, 7, 8]}, index=dates)
        target = v.target_table(features, 2)
        self.assertEqual(target.loc[dates[0], "y"], 2.5)
        self.assertTrue(np.isnan(target.loc[dates[1], "y"]))
        self.assertEqual(target.loc[dates[0], "target_end"], dates[2])
        self.assertTrue(target.iloc[-2:]["y"].isna().all())


class IndependentInferenceTests(unittest.TestCase):
    def test_qlike_identity_and_positivity(self):
        self.assertEqual(v.qlike(1, 1), 0)
        self.assertAlmostEqual(v.qlike(2, 1), 1 - np.log(2))
        with self.assertRaises(AssertionError):
            v.qlike(1, -1)

    def test_circular_bootstrap_matches_explicit_reference_including_tail(self):
        values = np.arange(17, dtype=float)
        block, draws, seed = 6, 7, 21
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, len(values), size=(draws, 2))
        tails = rng.integers(0, len(values), size=draws)
        expected = []
        for i in range(draws):
            indices = [(int(s) + j) % 17 for s in starts[i] for j in range(block)]
            indices += [(int(tails[i]) + j) % 17 for j in range(5)]
            expected.append(sum(values[j] for j in indices) / 17)
        np.testing.assert_allclose(v.explicit_bootstrap_means(values, block, draws, seed)[:, 0], expected)

    def test_hac_matches_direct_bartlett_covariance(self):
        values = np.random.default_rng(31).normal(size=40)
        centered, n = values - values.mean(), len(values)
        variance = sum(centered**2) / n
        for lag in range(1, 9):
            variance += 2 * (1 - lag / 9) * sum(centered[lag:] * centered[:-lag]) / n
        result = v.independent_hac(values, maxlags=8)
        self.assertAlmostEqual(result["se"], np.sqrt(variance / n), places=13)


if __name__ == "__main__":
    unittest.main()
