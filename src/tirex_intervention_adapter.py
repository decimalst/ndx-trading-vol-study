"""CPU-only, pinned-local TiRex-2 forecasts with temporary native-unit hooks.

Inputs and forecast quantiles stay in native log-variance units.  This adapter
never exponentiates forecasts and never trains, downloads, or modifies weights.
The hook is the INPUT to the native gated FFN's down projection, at context
token 63.  Input histories have 2048 observations and no covariates.
"""
from __future__ import annotations

import os

# TiRex decorates its residual block class at import time. Disable compilation
# before importing torch/TiRex so ordinary Python hooks remain observable.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import hashlib
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import torch

REVISION = "05e5b26db52bfb256f1ae1bdf785589850482de3"
CONTEXT_LENGTH = 2048
TOKEN_INDEX = 63
TOKEN_COUNT = 74
UNIT_COUNT = 1408
EMBEDDING_DIM = 512


def forecast_token_index(context_length, future_length, input_patch_size, output_patch_size):
    """Locate the token producing the first retained forecast in native slicing.

    TiRex emits one output patch per input token, removes its final output
    patch, and retains the trailing future_length values. Thus the first
    forecast is emitted by the final context token, not the final future token.
    """
    values = (context_length, future_length, input_patch_size, output_patch_size)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("context, future and patch lengths must be positive integers")
    if (input_patch_size != output_patch_size or context_length % input_patch_size
            or future_length % input_patch_size):
        raise ValueError("patch lengths must match and divide context and future lengths")
    total_tokens = (context_length + future_length) // input_patch_size
    retained_start = total_tokens * output_patch_size - output_patch_size - future_length
    return retained_start // output_patch_size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TirexAdapter:
    """An explicitly loaded model; one intervention and one forward pass per call."""

    def __init__(self, checkpoint_revision=REVISION, layers=(0, 5, 11), threads=4):
        self.layers = tuple(layers)
        if (not self.layers or len(set(self.layers)) != len(self.layers)
                or any(type(x) is not int or not 0 <= x < 12 for x in self.layers)):
            raise ValueError("layers must be distinct integer indices in 0..11")
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        if (not isinstance(checkpoint_revision, str) or len(checkpoint_revision) != 40
                or any(c not in "0123456789abcdef" for c in checkpoint_revision)):
            raise ValueError("checkpoint_revision must be a full lowercase git commit SHA")
        self.checkpoint_revision = checkpoint_revision
        self.threads = threads
        self.model = None
        self._timeseries_type = None
        self.weight_norms = {}
        self.metadata = {}
        self._forecast_active = False

    @classmethod
    def from_model(cls, model, timeseries_type, layers=(0, 5, 11)):
        """Inject a wrapper for deterministic instrumentation tests, without loading."""
        obj = cls(layers=layers, threads=1)
        obj.model = model
        obj._timeseries_type = timeseries_type
        obj._inspect_model()
        obj.metadata = {"test_injected_model": True}
        return obj

    def _inspect_model(self):
        core = self.model.model
        expected = {"context_len": 2048, "future_len": 320, "input_patch_size": 32,
                    "output_patch_size": 32, "embedding_dim": 512}
        for field, value in expected.items():
            if getattr(core, field, None) != value:
                raise ValueError(f"model {field} must equal {value}")
        first_forecast_token = forecast_token_index(core.context_len, core.future_len,
                                                   core.input_patch_size, core.output_patch_size)
        if first_forecast_token != TOKEN_INDEX:
            raise ValueError("first forecast token does not match intervention token")
        if len(core.stack) != 12:
            raise ValueError("model must have exactly 12 blocks")
        self._quantile_levels = np.asarray(core.quantiles.detach().cpu(), dtype=float)
        self._quantile_indices = []
        for q in (.1, .5, .9):
            matches = np.flatnonzero(np.isclose(self._quantile_levels, q, atol=1e-6, rtol=0))
            if len(matches) != 1:
                raise ValueError(f"model quantile {q} must have exactly one metadata match")
            self._quantile_indices.append(int(matches[0]))
        for layer in self.layers:
            proj = core.stack[layer].time_mixer.ffn.proj_down
            if tuple(proj.weight.shape) != (EMBEDDING_DIM, UNIT_COUNT):
                raise ValueError("down-projection weight shape must equal (512, 1408)")
            self.weight_norms[layer] = torch.linalg.vector_norm(
                proj.weight.detach().double(), dim=0).cpu().numpy().copy()
        core.eval()

    def load(self):
        """Resolve the exact cached snapshot, hash its bytes and load on CPU only."""
        if self.model is not None:
            return self
        from huggingface_hub.constants import HF_HUB_CACHE
        from tirex2 import TimeseriesType, load_model

        checkpoint = (Path(HF_HUB_CACHE) / "models--NX-AI--TiRex-2" / "snapshots"
                      / self.checkpoint_revision)
        config_path = checkpoint / "model-config.yaml"
        weights_path = checkpoint / "model.ckpt"
        for path in (config_path, weights_path):
            if not path.is_file():
                raise FileNotFoundError(f"required pinned local checkpoint file missing: {path}")
        torch.set_num_threads(self.threads)
        self.model = load_model(str(checkpoint), device="cpu", use_flex_attention=False)
        self._timeseries_type = TimeseriesType
        self._inspect_model()
        versions = {}
        for name in ("torch", "numpy", "tirex-2", "xlstm", "huggingface-hub"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        self.metadata = {
            "checkpoint_id": "NX-AI/TiRex-2", "checkpoint_revision": self.checkpoint_revision,
            "snapshot_path": str(checkpoint), "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path), "weights_path": str(weights_path.resolve()),
            "weights_sha256": _sha256(weights_path), "versions": versions,
            "python": platform.python_version(), "platform": platform.platform(),
            "device": "cpu", "threads": torch.get_num_threads(), "compile_disabled": True,
            "tta_sign_flip": False, "tta_diff": False, "use_flex_attention": False,
            "context_length": CONTEXT_LENGTH, "token_index": TOKEN_INDEX,
            "token_count": TOKEN_COUNT, "unit_count": UNIT_COUNT, "layers": list(self.layers),
            "quantile_levels_native": self._quantile_levels.tolist(),
            "selected_quantile_levels": [.1, .5, .9],
            "selected_quantile_indices": self._quantile_indices,
            "tirex_import_package": "tirex2", "tirex_distribution": "tirex-2",
            "forecast_token_index": forecast_token_index(
                self.model.model.context_len, self.model.model.future_len,
                self.model.model.input_patch_size, self.model.model.output_patch_size),
            "units": "native input units (pilot: log variance)",
            "hook": "stack[layer].time_mixer.ffn.proj_down forward_pre_hook",
        }
        return self

    def _validate_patch(self, patch, batch_size):
        if patch is None:
            return None
        if not isinstance(patch, dict) or set(patch) != {"layer", "unit", "values"}:
            raise ValueError("patch must contain exactly layer, unit, values")
        layer, unit = patch["layer"], patch["unit"]
        if type(layer) is not int or layer not in self.layers:
            raise ValueError("patch layer must be one of the captured native layers")
        if type(unit) is not int or not 0 <= unit < UNIT_COUNT:
            raise ValueError("patch unit must be an integer in 0..1407")
        values = np.asarray(patch["values"], dtype=np.float32)
        if values.shape != (batch_size,) or not np.isfinite(values).all():
            raise ValueError("patch values must be a finite vector matching the batch")
        return {"layer": layer, "unit": unit, "values": torch.from_numpy(values.copy())}

    def forecast(self, contexts, patch=None, capture=True):
        """Return Nx3 quantiles, layer->Nx1408 activations and Nx512 residual.

        Captured activations are the values actually consumed by proj_down
        (after any intervention). With capture=False, activations={} and
        residual=None. Hooks are removed even if validation or inference fails.
        """
        if self.model is None:
            raise RuntimeError("load the model before forecasting")
        if self._forecast_active:
            raise RuntimeError("concurrent or recursive forecasts are not supported")
        contexts = np.asarray(contexts, dtype=np.float32)
        if (contexts.ndim != 2 or contexts.shape[1] != CONTEXT_LENGTH
                or contexts.shape[0] == 0 or not np.isfinite(contexts).all()):
            raise ValueError("contexts must have finite shape (N>0, 2048)")
        batch_size = len(contexts)
        patch = self._validate_patch(patch, batch_size)
        tensor = torch.from_numpy(contexts.copy())
        timeseries = [self._timeseries_type(target=row[None].clone(),
                      past_covariates=None, future_covariates=None) for row in tensor]
        core = self.model.model
        activations, counts, handles = {}, {}, []
        residual = None

        def pre_hook(layer):
            def hook(_module, args):
                counts[layer] += 1
                if counts[layer] != 1:
                    raise RuntimeError(f"layer {layer} hook must execute exactly once")
                if len(args) != 1 or tuple(args[0].shape) != (batch_size, TOKEN_COUNT, UNIT_COUNT):
                    raise ValueError("native FFN activation shape does not match (N,74,1408)")
                value = args[0]
                if patch is not None and patch["layer"] == layer:
                    value = value.clone()
                    value[:, TOKEN_INDEX, patch["unit"]] = patch["values"].to(value)
                if capture:
                    activations[layer] = value[:, TOKEN_INDEX].detach().cpu().numpy().copy()
                return (value,)
            return hook

        def residual_hook(_module, _args, output):
            nonlocal residual
            counts["residual"] += 1
            if counts["residual"] != 1:
                raise RuntimeError("residual hook must execute exactly once")
            if tuple(output.shape) != (batch_size, TOKEN_COUNT, EMBEDDING_DIM):
                raise ValueError("normalized residual shape does not match (N,74,512)")
            residual = output[:, TOKEN_INDEX].detach().cpu().numpy().copy()

        self._forecast_active = True
        try:
            for layer in self.layers:
                counts[layer] = 0
                module = core.stack[layer].time_mixer.ffn.proj_down
                handles.append(module.register_forward_pre_hook(pre_hook(layer)))
            if capture:
                counts["residual"] = 0
                handles.append(core.stack_out_norm.register_forward_hook(residual_hook))
            core.eval()
            with torch.no_grad():
                output = self.model.forecast(timeseries, prediction_length=1, output_type="torch",
                    batch_size=batch_size, yield_per_batch=False, tta_sign_flip=False, tta_diff=False)
            if any(count != 1 for count in counts.values()):
                raise RuntimeError(f"every registered hook must execute exactly once; counts={counts}")
        finally:
            for handle in handles:
                handle.remove()
            self._forecast_active = False
        if not isinstance(output, list) or len(output) != batch_size:
            raise ValueError("forecast output list shape must match input batch")
        expected_shape = (1, len(self._quantile_levels), 1)
        if any(tuple(value.shape) != expected_shape for value in output):
            raise ValueError(f"forecast output shape must equal {expected_shape}")
        all_quantiles = np.stack([value.detach().cpu().numpy()[0, :, 0] for value in output])
        if not np.isfinite(all_quantiles).all():
            raise ValueError("forecast output must be finite")
        if any(not np.isfinite(values).all() for values in activations.values()):
            raise ValueError("captured activations must be finite")
        if residual is not None and not np.isfinite(residual).all():
            raise ValueError("captured residual must be finite")
        return {"quantiles": all_quantiles[:, self._quantile_indices].copy(),
                "activations": activations, "residual": residual}
