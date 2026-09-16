"""Synthetic CPU and timing contracts, written before the xLSTM adaptation.

No market observations, forecasts, or score files are accessed by this suite.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src import reference_xlstm as study

FEATURES = (
    "lrv_d", "lrv_w", "lrv_m", "lev_d", "lev_w", "lev_m",
    "liv", "lvix", "term", "xasset_stress", "market_stress",
)


def fixture(n=590):
    rng = np.random.default_rng(32011)
    dates = pd.bdate_range("2010-01-04", periods=n)
    raw = rng.normal(0, .3, (n, len(FEATURES)))
    frame = pd.DataFrame(raw, columns=FEATURES, index=dates)
    frame["rv_total"] = np.exp(-8 + .2 * np.sin(np.arange(n) / 30) + rng.normal(0, .08, n))
    frame["lrv_d"] = np.log(frame.rv_total)
    frame["lrv_w"] = np.log(frame.rv_total.rolling(5, min_periods=1).mean())
    frame["lrv_m"] = np.log(frame.rv_total.rolling(22, min_periods=1).mean())
    return frame


class WindowContractTests(unittest.TestCase):
    def test_windows_use_consecutive_real_sessions_and_include_origin_only(self):
        frame = fixture(60)
        origins = frame.index[[21, 35, 48]]
        got = study.input_windows(frame, origins)
        self.assertEqual(got.shape, (3, 22, 11))
        for i, position in enumerate([21, 35, 48]):
            np.testing.assert_array_equal(got[i], frame.iloc[position - 21:position + 1][list(FEATURES)])
        changed = frame.copy()
        changed.iloc[49:, :11] += 1e5
        np.testing.assert_array_equal(study.input_windows(changed, origins), got)

    def test_targets_use_next_one_and_five_complete_sessions(self):
        frame = fixture(60)
        y, ends = study.targets(frame.rv_total)
        for position in (0, 12, 50):
            for j, h in enumerate((1, 5)):
                self.assertAlmostEqual(y.iloc[position, j], frame.rv_total.iloc[position + 1:position + h + 1].mean(), places=14)
                self.assertEqual(ends.iloc[position, j], frame.index[position + h])
        self.assertTrue(np.isnan(y.iloc[-1]).all())
        self.assertTrue(np.isnan(y.iloc[-4, 1]))

    def test_joint_label_maturity_and_common_origin_mask_precede_scaling(self):
        frame = fixture()
        fit_origin = frame.index[575]
        eligible = frame.index.delete([110, 200, 300])
        data = study.training_windows(frame, fit_origin, eligible)
        expected_dates = frame.index[21:571].difference(frame.index[[110, 200, 300]])
        pd.testing.assert_index_equal(data["origins"], expected_dates)
        self.assertLess(data["origins"].max(), fit_origin)
        self.assertLessEqual(data["target_ends"].to_numpy().max(), fit_origin.to_datetime64())
        self.assertEqual(data["windows"].shape[0], len(expected_dates))

    def test_training_pool_has_a_fixed_last1500_cap_and_minimum500(self):
        frame = fixture(1900)
        data = study.training_windows(frame, frame.index[1890], frame.index)
        self.assertEqual(len(data["origins"]), 1500)
        self.assertEqual(data["origins"][-1], frame.index[1885])
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_DATA"):
            study.training_windows(frame, frame.index[400], frame.index)

    def test_incomplete_or_misaligned_input_windows_fail_without_imputation(self):
        frame = fixture(60)
        for origins in (frame.index[:1], pd.DatetimeIndex(["1990-01-01"]), frame.index[[25, 25]]):
            with self.assertRaises(ValueError):
                study.input_windows(frame, origins)
        frame.loc[frame.index[20], "lvix"] = np.nan
        with self.assertRaises(ValueError):
            study.input_windows(frame, frame.index[[30]])


class NeuralContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_qlike_and_gradient_are_exact_in_log_parameterization(self):
        y = torch.tensor([[.8, 1.2], [1.5, .3]], dtype=torch.float64)
        log_prediction = torch.tensor([[.1, -.2], [.3, .4]], dtype=torch.float64, requires_grad=True)
        loss = study.qlike_from_log(y, log_prediction)
        ratio = y / torch.exp(log_prediction)
        self.assertAlmostEqual(loss.item(), (ratio - torch.log(ratio) - 1).mean().item(), places=13)
        loss.backward()
        torch.testing.assert_close(log_prediction.grad, (1 - ratio) / y.numel())

    def test_real_vanilla_forward_backward_and_optimizer_updates_both_models(self):
        with torch.random.fork_rng():
            torch.manual_seed(712)
            x = torch.randn(8, 22, 11)
            y = torch.exp(.2 * torch.randn(8, 2))
            for name in ("nlinear", "xlstm"):
                model = study.SmallVarianceMixer(name)
                initial = {key: value.detach().clone() for key, value in model.named_parameters()}
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                out = model(x)
                self.assertEqual(out.shape, (8, 2))
                self.assertEqual(out.dtype, torch.float32)
                self.assertEqual(out.device.type, "cpu")
                self.assertTrue(torch.isfinite(out).all())
                loss = study.qlike_from_log(y, out)
                optimizer.zero_grad()
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(torch.any(g != 0) for g in gradients))
                optimizer.step()
                self.assertTrue(any(not torch.equal(initial[key], value) for key, value in model.named_parameters()))
                if name == "xlstm":
                    self.assertEqual(model.mixer.config.backend, "vanilla")
                    self.assertEqual(model.mixer.config.dtype, "float32")
                    self.assertTrue(torch.any(model.memory_tokens.grad != 0))

    def test_forward_has_no_cross_sample_state_and_matched_linear_parameters(self):
        x = torch.randn(4, 22, 11)
        with torch.random.fork_rng():
            torch.manual_seed(20260906)
            linear = study.SmallVarianceMixer("nlinear")
            torch.manual_seed(20260906)
            full = study.SmallVarianceMixer("xlstm")
            for name in ("temporal.weight", "temporal.bias", "pre_encoding.weight", "pre_encoding.bias", "readout.weight", "readout.bias"):
                torch.testing.assert_close(linear.state_dict()[name], full.state_dict()[name], rtol=0, atol=0)
            for model in (linear, full):
                batch = model(x).detach()
                for row in range(len(x)):
                    torch.testing.assert_close(model(x[row:row + 1])[0], batch[row], rtol=3e-6, atol=3e-7)

    def test_fixed_training_is_repeatable_and_future_values_cannot_change_fit(self):
        frame = fixture()
        origin = frame.index[570]
        fit1 = study.fit_window_models(frame, origin, frame.index)
        changed = frame.copy()
        changed.loc[frame.index[571]:, list(FEATURES)] += 1000
        changed.loc[frame.index[571]:, "rv_total"] *= 1000
        fit2 = study.fit_window_models(changed, origin, frame.index)
        self.assertEqual(fit1["audit"], fit2["audit"])
        train = study.training_windows(frame, origin, frame.index)
        np.testing.assert_allclose(fit1["input_center"], train["windows"].reshape(-1, 11).mean(axis=0))
        np.testing.assert_allclose(fit1["input_scale"], train["windows"].reshape(-1, 11).std(axis=0))
        predictions = study.predict_window_models(fit1, frame, frame.index[570:580])
        for name in ("nlinear", "xlstm"):
            self.assertEqual(predictions[name].shape, (10, 2))
            self.assertTrue(np.isfinite(predictions[name]).all())
            self.assertTrue((predictions[name] > 0).all())
            audit = fit1["audit"]["models"][name]
            self.assertEqual(audit["epochs"], 12)
            self.assertEqual(audit["batch_size"], 128)
            self.assertEqual(audit["seed"], 20260906)
            self.assertEqual(audit["optimizer"], "Adam")
            self.assertEqual(audit["learning_rate"], .001)
            self.assertEqual(audit["steps"], 12 * int(np.ceil(len(train["origins"]) / 128)))
            for key in fit1["models"][name].state_dict():
                torch.testing.assert_close(fit1["models"][name].state_dict()[key], fit2["models"][name].state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
