"""Prewritten instrumentation contracts; only tiny deterministic torch mocks."""
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import unittest
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src import tirex_intervention_adapter as adapter_module
from src.tirex_intervention_adapter import TirexAdapter


@dataclass
class FakeSeries:
    target: torch.Tensor
    past_covariates: object = None
    future_covariates: object = None


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.arange(1, 1409).float()[None].expand(512, -1))
        self.received = []

    def forward(self, x):
        self.received.append(x.detach().clone())
        return x.sum(-1, keepdim=True).expand(-1, -1, 512) / 1408


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.ModuleList()
        for _ in range(12):
            block = nn.Module()
            block.time_mixer = nn.Module()
            block.time_mixer.ffn = nn.Module()
            block.time_mixer.ffn.proj_down = Projection()
            self.stack.append(block)
        self.stack_out_norm = nn.Identity()
        # Deliberately permuted metadata catches positional quantile assumptions.
        self.register_buffer("quantiles", torch.tensor([.9, .3, .5, .1, .7]))
        self.context_len = 2048
        self.future_len = 320
        self.input_patch_size = self.output_patch_size = 32
        self.embedding_dim = 512


class Wrapper:
    def __init__(self):
        self.model = Core()
        self.calls = []
        self.raise_after = False
        self.repeat_layer = False
        self.skip_layer = False
        self.bad_shape = False
        self.bad_output = False
        self.bad_finite = False
        self.original_inputs = []

    def forecast(self, timeseries, **kwargs):
        self.calls.append((timeseries, kwargs, torch.is_grad_enabled(), self.model.training))
        x = torch.stack([s.target[0] for s in timeseries])
        channels = 1407 if self.bad_shape else 1408
        z = (torch.arange(74)[None, :, None] + torch.arange(channels)[None, None, :] / 10000
             + x[:, -1, None, None]).float().expand(len(x), 74, channels).clone()
        self.original_inputs.append(z)
        out = None
        for layer in (0, 5, 11):
            if layer == 5 and self.skip_layer:
                continue
            out = self.model.stack[layer].time_mixer.ffn.proj_down(z)
            if self.repeat_layer and layer == 5:
                out = self.model.stack[layer].time_mixer.ffn.proj_down(z)
            # Carry effects of each selected stage to the next.
            if channels == 1408:
                z = z + out[..., :1] / 10
        out = self.model.stack_out_norm(out)
        if self.raise_after:
            raise RuntimeError("deliberate model failure")
        levels = self.model.quantiles
        forecasts = [(out[i, 63, 0] + levels)[None, :, None] for i in range(len(x))]
        if self.bad_output:
            forecasts[0] = forecasts[0].squeeze(0)
        if self.bad_finite:
            forecasts[0][0, 0, 0] = torch.nan
        return forecasts


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.wrapper = Wrapper()
        self.adapter = TirexAdapter.from_model(self.wrapper, FakeSeries)
        self.contexts = np.stack([np.linspace(-1, 1, 2048), np.linspace(2, 4, 2048)])

    def assert_hooks_empty(self):
        for block in self.wrapper.model.stack:
            self.assertFalse(block.time_mixer.ffn.proj_down._forward_pre_hooks)
        self.assertFalse(self.wrapper.model.stack_out_norm._forward_hooks)

    def test_quantile_metadata_and_native_units(self):
        result = self.adapter.forecast(self.contexts)
        self.assertEqual(result["quantiles"].shape, (2, 3))
        np.testing.assert_allclose(np.diff(result["quantiles"], axis=1), .4, atol=1e-5)
        self.assertGreater(float(result["quantiles"].min()), 60)  # no exp or variance conversion
        self.assertEqual(result["residual"].shape, (2, 512))
        self.assertEqual(set(result["activations"]), {0, 5, 11})
        for activation in result["activations"].values():
            self.assertEqual(activation.shape, (2, 1408))

    def test_eval_no_grad_one_batch_and_tta_disabled(self):
        self.wrapper.model.train()
        self.adapter.forecast(self.contexts)
        self.assertEqual(len(self.wrapper.calls), 1)
        series, kwargs, grad, training = self.wrapper.calls[0]
        self.assertFalse(grad)
        self.assertFalse(training)
        self.assertFalse(kwargs["tta_sign_flip"])
        self.assertFalse(kwargs["tta_diff"])
        self.assertEqual(kwargs["batch_size"], 2)
        self.assertEqual(kwargs["prediction_length"], 1)
        self.assertFalse(kwargs["yield_per_batch"])
        self.assertEqual(tuple(series[0].target.shape), (1, 2048))

    def test_capture_is_token63_not_last_future_token(self):
        result = self.adapter.forecast(self.contexts)
        expected = self.wrapper.model.stack[0].time_mixer.ffn.proj_down.received[0][:, 63]
        np.testing.assert_array_equal(result["activations"][0], expected.numpy())

    def test_patch_changes_exact_channel_token_and_matching_batch_rows(self):
        values = np.array([-5., 20.])
        original = values.copy()
        contexts = self.contexts.copy()
        baseline = self.adapter.forecast(contexts)
        projection = self.wrapper.model.stack[5].time_mixer.ffn.proj_down
        before = projection.received[-1].numpy().copy()
        result = self.adapter.forecast(contexts, patch={"layer": 5, "unit": 7, "values": values})
        after = projection.received[-1].numpy()
        expected = before.copy()
        expected[:, 63, 7] = values
        np.testing.assert_array_equal(after, expected)
        np.testing.assert_array_equal(result["activations"][5][:, 7], values)
        self.assertFalse(np.array_equal(baseline["quantiles"], result["quantiles"]))
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(contexts, self.contexts)
        self.assert_hooks_empty()

    def test_noop_swap_reproduces_output_exactly(self):
        baseline = self.adapter.forecast(self.contexts)
        result = self.adapter.forecast(self.contexts, patch={
            "layer": 5, "unit": 11, "values": baseline["activations"][5][:, 11]})
        np.testing.assert_array_equal(result["quantiles"], baseline["quantiles"])
        np.testing.assert_array_equal(result["residual"], baseline["residual"])

    def test_repeated_hook_call_is_rejected_and_cleaned(self):
        self.wrapper.repeat_layer = True
        with self.assertRaisesRegex(RuntimeError, "once|multiple|count"):
            self.adapter.forecast(self.contexts)
        self.assert_hooks_empty()

    def test_missing_hook_call_is_rejected_and_cleaned(self):
        self.wrapper.skip_layer = True
        with self.assertRaisesRegex(RuntimeError, "once|missing|count"):
            self.adapter.forecast(self.contexts)
        self.assert_hooks_empty()

    def test_hook_cleanup_after_model_exception(self):
        self.wrapper.raise_after = True
        with self.assertRaisesRegex(RuntimeError, "deliberate model failure"):
            self.adapter.forecast(self.contexts, patch={"layer": 0, "unit": 1, "values": [4., 5.]})
        self.assert_hooks_empty()
        self.wrapper.raise_after = False
        self.assertEqual(self.adapter.forecast(self.contexts)["quantiles"].shape, (2, 3))

    def test_wrong_activation_shape_rejected_and_cleaned(self):
        self.wrapper.bad_shape = True
        with self.assertRaisesRegex(ValueError, "shape"):
            self.adapter.forecast(self.contexts)
        self.assert_hooks_empty()

    def test_invalid_contexts_do_not_call_model(self):
        for contexts in (np.ones(2048), np.ones((1, 100)), np.empty((0, 2048)),
                         np.full((1, 2048), np.nan), np.full((1, 2048), np.inf)):
            with self.subTest(shape=np.shape(contexts)), self.assertRaises(ValueError):
                self.adapter.forecast(contexts)
        self.assertFalse(self.wrapper.calls)
        self.assert_hooks_empty()

    def test_invalid_patch_rejected_before_model(self):
        patches = [
            {"layer": 1, "unit": 0, "values": [1, 2]},
            {"layer": 0, "unit": -1, "values": [1, 2]},
            {"layer": 0, "unit": 1408, "values": [1, 2]},
            {"layer": 0, "unit": 2.5, "values": [1, 2]},
            {"layer": 0, "unit": 1, "values": [1]},
            {"layer": 0, "unit": 1, "values": [[1], [2]]},
            {"layer": 0, "unit": 1, "values": [1, np.nan]},
            {"layer": 0, "unit": 1},
        ]
        for patch in patches:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.adapter.forecast(self.contexts, patch=patch)
        self.assertFalse(self.wrapper.calls)
        self.assert_hooks_empty()

    def test_wrong_output_shape_rejected_after_cleanup(self):
        self.wrapper.bad_output = True
        with self.assertRaisesRegex(ValueError, "shape"):
            self.adapter.forecast(self.contexts)
        self.assert_hooks_empty()

    def test_nonfinite_output_rejected_after_cleanup(self):
        self.wrapper.bad_finite = True
        with self.assertRaisesRegex(ValueError, "finite"):
            self.adapter.forecast(self.contexts)
        self.assert_hooks_empty()

    def test_capture_false_can_patch_without_retaining_activations(self):
        result = self.adapter.forecast(self.contexts, capture=False,
            patch={"layer": 11, "unit": 2, "values": [-2, 3]})
        self.assertEqual(result["activations"], {})
        self.assertIsNone(result["residual"])
        self.assert_hooks_empty()

    def test_weight_norms_are_down_projection_column_norms(self):
        for norms in self.adapter.weight_norms.values():
            np.testing.assert_allclose(norms, np.arange(1, 1409) * np.sqrt(512), rtol=1e-6)

    def test_duplicate_layers_and_unsupported_indices_rejected(self):
        for layers in ((0, 0), (-1,), (12,), (2.5,), ()):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                TirexAdapter(layers=layers)

    def test_forecast_requires_explicit_load(self):
        with self.assertRaisesRegex(RuntimeError, "load"):
            TirexAdapter().forecast(self.contexts)

    def test_first_forecast_token_matches_output_window_slicing(self):
        # Match TiRex._predict: discard the last output patch, then retain
        # future_len values. Each toy output value carries its generating token.
        for context, future, patch in ((2048, 320, 32), (64, 16, 8), (96, 32, 16)):
            token_count = (context + future) // patch
            output = np.repeat(np.arange(token_count), patch)
            selected = output[:-patch][-future:]
            calculated = adapter_module.forecast_token_index(context, future, patch, patch)
            self.assertEqual(calculated, int(selected[0]))
            self.assertTrue(np.all(selected[:patch] == calculated))
        self.assertEqual(adapter_module.forecast_token_index(2048, 320, 32, 32), 63)

    def test_first_forecast_token_rejects_unsupported_geometry(self):
        for args in ((0, 320, 32, 32), (2048, 0, 32, 32), (2049, 320, 32, 32),
                     (2048, 321, 32, 32), (2048, 320, 32, 16), (2048, 320, 0, 0),
                     (2048, 320, 32.0, 32)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                adapter_module.forecast_token_index(*args)


if __name__ == "__main__":
    unittest.main()
