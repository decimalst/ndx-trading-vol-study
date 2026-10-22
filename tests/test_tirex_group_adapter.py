"""Prewritten contracts for simultaneous native-unit group interventions."""

import unittest

import numpy as np
import torch

from src.tirex_group_adapter import GroupTirexAdapter
from src.tirex_intervention_adapter import TirexAdapter
from tests.test_tirex_intervention_adapter import FakeSeries, Wrapper


class GroupAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.wrapper = Wrapper()
        self.adapter = GroupTirexAdapter.from_model(self.wrapper, FakeSeries)
        self.contexts = np.stack([np.linspace(-1 + i, 1 + i, 2048) for i in range(5)])

    def assert_hooks_empty(self):
        for block in self.wrapper.model.stack:
            self.assertFalse(block.time_mixer.ffn.proj_down._forward_pre_hooks)
        self.assertFalse(self.wrapper.model.stack_out_norm._forward_hooks)

    def test_group_changes_only_selected_channels_at_token63(self):
        units = [2, 19, 1407]
        values = np.arange(15, dtype=np.float32).reshape(5, 3) - 100
        baseline = self.adapter.forecast(self.contexts)
        proj = self.wrapper.model.stack[5].time_mixer.ffn.proj_down
        before = proj.received[-1].numpy().copy()
        result = self.adapter.forecast(self.contexts, patch={
            "layer": 5, "units": units, "values": values})
        expected = before.copy()
        expected[:, 63, units] = values
        np.testing.assert_array_equal(proj.received[-1].numpy(), expected)
        np.testing.assert_array_equal(result["activations"][5][:, units], values)
        self.assertFalse(np.array_equal(baseline["quantiles"], result["quantiles"]))
        self.assert_hooks_empty()

    def test_group_can_replace_all_channels_without_other_token_changes(self):
        units = list(range(1408))
        values = np.tile(np.arange(1408, dtype=np.float32), (5, 1)) - 700
        self.adapter.forecast(self.contexts)
        proj = self.wrapper.model.stack[11].time_mixer.ffn.proj_down
        before = proj.received[-1].numpy().copy()
        result = self.adapter.forecast(self.contexts, patch={
            "layer": 11, "units": units, "values": values})
        expected = before.copy()
        expected[:, 63] = values
        np.testing.assert_array_equal(proj.received[-1].numpy(), expected)
        np.testing.assert_array_equal(result["activations"][11], values)

    def test_permuting_units_and_corresponding_columns_is_equivalent(self):
        units = [5, 119, 1000, 1407]
        values = np.arange(20).reshape(5, 4) - 80
        first = self.adapter.forecast(self.contexts, patch={
            "layer": 5, "units": units, "values": values})
        order = [2, 0, 3, 1]
        second = self.adapter.forecast(self.contexts, patch={
            "layer": 5, "units": [units[i] for i in order], "values": values[:, order]})
        np.testing.assert_array_equal(first["quantiles"], second["quantiles"])
        np.testing.assert_array_equal(first["residual"], second["residual"])
        for layer in self.adapter.layers:
            np.testing.assert_array_equal(first["activations"][layer], second["activations"][layer])

    def test_same_value_group_swap_is_bit_exact(self):
        units = [0, 17, 104, 985]
        baseline = self.adapter.forecast(self.contexts)
        for layer in self.adapter.layers:
            with self.subTest(layer=layer):
                result = self.adapter.forecast(self.contexts, patch={
                    "layer": layer, "units": units,
                    "values": baseline["activations"][layer][:, units]})
                np.testing.assert_array_equal(result["quantiles"], baseline["quantiles"])
                np.testing.assert_array_equal(result["residual"], baseline["residual"])
                for current in self.adapter.layers:
                    np.testing.assert_array_equal(
                        result["activations"][current], baseline["activations"][current])

    def test_cardinality_one_matches_original_scalar_adapter_bit_exactly(self):
        scalar = TirexAdapter.from_model(self.wrapper, FakeSeries)
        values = np.array([-30., 2., 80., 111., -99.])
        for layer in self.adapter.layers:
            with self.subTest(layer=layer):
                original = scalar.forecast(self.contexts, patch={
                    "layer": layer, "unit": 391, "values": values})
                group = self.adapter.forecast(self.contexts, patch={
                    "layer": layer, "units": [391], "values": values[:, None]})
                np.testing.assert_array_equal(original["quantiles"], group["quantiles"])
                np.testing.assert_array_equal(original["residual"], group["residual"])
                for current in self.adapter.layers:
                    np.testing.assert_array_equal(
                        original["activations"][current], group["activations"][current])

    def test_no_patch_matches_original_adapter(self):
        scalar = TirexAdapter.from_model(self.wrapper, FakeSeries)
        first = scalar.forecast(self.contexts)
        second = self.adapter.forecast(self.contexts)
        np.testing.assert_array_equal(first["quantiles"], second["quantiles"])
        np.testing.assert_array_equal(first["residual"], second["residual"])

    def test_donor_rows_align_when_batch_is_split_into_chunks(self):
        baseline = self.adapter.forecast(self.contexts)
        units = [2, 53, 800]
        donor_row = np.array([4, 0, 3, 1, 2])
        values = baseline["activations"][5][donor_row][:, units]
        whole = self.adapter.forecast(self.contexts, patch={
            "layer": 5, "units": units, "values": values})
        chunks = []
        for start in range(0, len(self.contexts), 2):
            chunks.append(self.adapter.forecast(self.contexts[start:start + 2], patch={
                "layer": 5, "units": units, "values": values[start:start + 2]}))
        np.testing.assert_array_equal(
            whole["quantiles"], np.concatenate([item["quantiles"] for item in chunks]))
        np.testing.assert_array_equal(
            whole["activations"][5],
            np.concatenate([item["activations"][5] for item in chunks]))
        np.testing.assert_array_equal(whole["activations"][5][:, units], values)

    def test_contexts_donors_units_and_original_tensors_are_not_mutated(self):
        units = [0, 2, 7]
        values = np.arange(15, dtype=np.float32).reshape(5, 3) - 40
        original_values = values.copy()
        original_contexts = self.contexts.copy()
        values.flags.writeable = False
        self.adapter.forecast(self.contexts)
        original_tensor = self.wrapper.original_inputs[-1].clone()
        self.adapter.forecast(self.contexts, patch={"layer": 0, "units": units, "values": values})
        np.testing.assert_array_equal(values, original_values)
        np.testing.assert_array_equal(self.contexts, original_contexts)
        np.testing.assert_array_equal(self.wrapper.original_inputs[-1].numpy(), original_tensor.numpy())
        self.assertEqual(units, [0, 2, 7])

    def test_invalid_unit_groups_fail_before_model_calls(self):
        for units in ([], [1, 1], [-1], [1408], [True], [2.5], "2", (1, 2), None):
            with self.subTest(units=units), self.assertRaises(ValueError):
                self.adapter.forecast(self.contexts, patch={
                    "layer": 5, "units": units, "values": np.ones((5, 2))})
        self.assertFalse(self.wrapper.calls)
        self.assert_hooks_empty()

    def test_group_values_require_exact_N_by_K_shape_without_broadcast(self):
        for shape in ((2,), (5,), (1, 2), (5, 1), (2, 5), (5, 2, 1), (), (5, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.adapter.forecast(self.contexts, patch={
                    "layer": 5, "units": [2, 7], "values": np.ones(shape)})
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.adapter.forecast(self.contexts, patch={
                    "layer": 5, "units": [2, 7], "values": np.full((5, 2), value)})
        self.assertFalse(self.wrapper.calls)
        self.assert_hooks_empty()

    def test_invalid_layers_and_keys_fail_before_model_calls(self):
        for patch in (
            {"layer": 2, "units": [1, 7], "values": np.ones((5, 2))},
            {"layer": True, "units": [1, 7], "values": np.ones((5, 2))},
            {"layer": 5, "units": [1, 7]},
            {"layer": 5, "units": [1, 7], "values": np.ones((5, 2)), "token": 63},
        ):
            with self.subTest(keys=list(patch)), self.assertRaises(ValueError):
                self.adapter.forecast(self.contexts, patch=patch)
        self.assertFalse(self.wrapper.calls)

    def test_group_cleanup_after_model_exception_then_reusable(self):
        self.wrapper.raise_after = True
        with self.assertRaisesRegex(RuntimeError, "deliberate model failure"):
            self.adapter.forecast(self.contexts, patch={
                "layer": 5, "units": [1, 2], "values": np.zeros((5, 2))})
        self.assert_hooks_empty()
        self.wrapper.raise_after = False
        self.assertEqual(self.adapter.forecast(self.contexts)["quantiles"].shape, (5, 3))

    def test_group_repeated_and_missing_hook_calls_fail_closed(self):
        for attr in ("repeat_layer", "skip_layer"):
            setattr(self.wrapper, attr, True)
            with self.subTest(attr=attr), self.assertRaisesRegex(RuntimeError, "once"):
                self.adapter.forecast(self.contexts, patch={
                    "layer": 5, "units": [1, 2], "values": np.zeros((5, 2))})
            self.assert_hooks_empty()
            setattr(self.wrapper, attr, False)

    def test_group_eval_nograd_and_tta_disabled(self):
        self.wrapper.model.train()
        self.adapter.forecast(self.contexts, patch={
            "layer": 5, "units": [1, 2], "values": np.zeros((5, 2))})
        self.assertEqual(len(self.wrapper.calls), 1)
        _, kwargs, grad, training = self.wrapper.calls[0]
        self.assertFalse(grad)
        self.assertFalse(training)
        self.assertFalse(kwargs["tta_sign_flip"])
        self.assertFalse(kwargs["tta_diff"])

    def test_group_capture_false_preserves_forecast(self):
        patch = {"layer": 5, "units": [1, 2], "values": np.zeros((5, 2))}
        captured = self.adapter.forecast(self.contexts, patch=patch)
        uncaptured = self.adapter.forecast(self.contexts, patch=patch, capture=False)
        np.testing.assert_array_equal(captured["quantiles"], uncaptured["quantiles"])
        self.assertEqual(uncaptured["activations"], {})
        self.assertIsNone(uncaptured["residual"])
        self.assert_hooks_empty()


if __name__ == "__main__":
    unittest.main()
