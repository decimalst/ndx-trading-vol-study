"""Fixed CPU xLSTM-Mixer-inspired and matched NLinear variance experiments.

This is an adaptation, not a paper reproduction: 11 existing market features
are temporally compressed, then mixed across variates using vanilla sLSTM.
No data is loaded and no empirical experiment runs merely by importing this
module. Empirical orchestration must freeze its complete comparison family.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from torch import nn
from xlstm.blocks.slstm.layer import sLSTMLayer, sLSTMLayerConfig

FEATURE_COLUMNS = (
    "lrv_d", "lrv_w", "lrv_m", "lev_d", "lev_w", "lev_m",
    "liv", "lvix", "term", "xasset_stress", "market_stress",
)
MODEL_NAMES = ("nlinear", "xlstm")
CONTEXT = 22
HORIZONS = (1, 5)
HIDDEN = 16
HEADS = 4
MEMORY_TOKENS = 2
MAX_TRAIN = 1500
MIN_TRAIN = 500
EPOCHS = 12
BATCH_SIZE = 128
LEARNING_RATE = .001
SEED = 20260906


def _features(frame):
    if (not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.has_duplicates or not frame.index.is_monotonic_increasing):
        raise ValueError("A unique chronological DatetimeIndex is required")
    if not set(FEATURE_COLUMNS).issubset(frame.columns):
        raise ValueError("All eleven declared baseline features are required")
    return frame.loc[:, FEATURE_COLUMNS].to_numpy(float)


def input_windows(features, origins):
    """Return the 22 actual consecutive sessions ending at each given origin."""
    values = _features(features)
    dates = pd.DatetimeIndex(origins)
    if dates.has_duplicates or not dates.is_monotonic_increasing or dates.isna().any():
        raise ValueError("Unique chronological forecast origins are required")
    positions = features.index.get_indexer(dates)
    if len(positions) == 0 or (positions < CONTEXT - 1).any():
        raise ValueError("Forecast origin has an incomplete session context")
    windows = np.stack([values[pos - CONTEXT + 1:pos + 1] for pos in positions])
    if not np.isfinite(windows).all():
        raise ValueError("Missing input context; no imputation or session deletion is permitted")
    return windows


def targets(rv):
    """Next-h arithmetic mean variance and date of its last constituent."""
    if (not isinstance(rv.index, pd.DatetimeIndex) or rv.index.has_duplicates
            or not rv.index.is_monotonic_increasing):
        raise ValueError("A unique chronological variance calendar is required")
    values, endings = {}, {}
    dates = pd.Series(rv.index, index=rv.index)
    for horizon in HORIZONS:
        future = pd.concat([rv.shift(-step) for step in range(1, horizon + 1)], axis=1)
        target = future.mean(axis=1, skipna=False)
        values[horizon] = target.where(np.isfinite(future).all(axis=1) & (future > 0).all(axis=1))
        endings[horizon] = dates.shift(-horizon)
    return pd.DataFrame(values), pd.DataFrame(endings)


def training_windows(features, fit_origin, eligible_origins):
    """Select the latest 1,500 eligible windows with both labels completed."""
    values = _features(features)
    fit_origin = pd.Timestamp(fit_origin)
    if fit_origin not in features.index:
        raise ValueError("Fit origin must be an observed session")
    eligible_dates = pd.DatetimeIndex(eligible_origins)
    if eligible_dates.has_duplicates or eligible_dates.isna().any():
        raise ValueError("Common eligible origins must be unique and known")
    if (features.index.get_indexer(eligible_dates) < 0).any():
        raise ValueError("Common origin absent from actual session calendar")
    y, ends = targets(features.rv_total)
    finite_rows = pd.Series(np.isfinite(values).all(axis=1), index=features.index)
    full_windows = finite_rows.rolling(CONTEXT, min_periods=CONTEXT).sum() == CONTEXT
    mask = (full_windows & features.index.isin(eligible_dates) & (features.index < fit_origin)
            & np.isfinite(y).all(axis=1) & (y > 0).all(axis=1)
            & (ends <= fit_origin).all(axis=1))
    origins = features.index[mask][-MAX_TRAIN:]
    if len(origins) < MIN_TRAIN:
        raise ValueError(f"INSUFFICIENT_DATA: {len(origins)} complete neural training windows")
    return {"windows": input_windows(features, origins), "targets": y.loc[origins].to_numpy(float),
            "origins": origins, "target_ends": ends.loc[origins]}


class SmallVarianceMixer(nn.Module):
    """NLinear compression followed optionally by two-view scalar-memory mixing."""

    def __init__(self, variant):
        super().__init__()
        if variant not in MODEL_NAMES:
            raise ValueError("Unknown fixed neural arm")
        self.variant = variant
        # Common layers are initialized first and in the same order in both arms.
        self.temporal = nn.Linear(CONTEXT, len(HORIZONS))
        self.pre_encoding = nn.Linear(len(HORIZONS), HIDDEN)
        self.readout = nn.Linear(len(FEATURE_COLUMNS) * HIDDEN, len(HORIZONS))
        if variant == "xlstm":
            self.memory_tokens = nn.Parameter(.01 * torch.randn(MEMORY_TOKENS, HIDDEN))
            self.mixer = sLSTMLayer(sLSTMLayerConfig(
                embedding_dim=HIDDEN, num_heads=HEADS, backend="vanilla",
                dtype="float32", dtype_b="float32", dtype_r="float32", dtype_w="float32",
                dtype_g="float32", dtype_s="float32", dtype_a="float32",
                enable_automatic_mixed_precision=False, conv1d_kernel_size=0, dropout=0.,
            ))
            self.mixer.reset_parameters()

    def forward(self, windows):
        if windows.ndim != 3 or tuple(windows.shape[1:]) != (CONTEXT, len(FEATURE_COLUMNS)):
            raise ValueError("Neural input must have shape batch,22,11")
        last = windows[:, -1:, :].detach()
        temporal = self.temporal((windows - last).transpose(1, 2)) + last.transpose(1, 2)
        tokens = self.pre_encoding(temporal)
        if self.variant == "xlstm":
            memory = self.memory_tokens.unsqueeze(0).expand(len(tokens), -1, -1)
            forward = self.mixer(torch.cat([memory, tokens], dim=1))[:, MEMORY_TOKENS:]
            backward = self.mixer(torch.cat([memory, tokens.flip(1)], dim=1))[:, MEMORY_TOKENS:].flip(1)
            # Both views mix already observed variates; neither accesses a future
            # session. Recurrent state resets inside every independent call.
            tokens = tokens + .5 * (forward + backward)
        return self.readout(tokens.flatten(start_dim=1))


def qlike_from_log(y, log_prediction):
    """Mean QLIKE over samples and the two jointly trained horizons."""
    if y.shape != log_prediction.shape or not torch.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Positive finite targets must align with log-variance predictions")
    log_ratio = torch.log(y) - log_prediction
    loss = (torch.exp(log_ratio) - log_ratio - 1).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite neural QLIKE loss")
    return loss


@contextmanager
def _deterministic_cpu():
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _training_loss(model, windows, y):
    total = 0.
    with torch.no_grad():
        for start in range(0, len(windows), BATCH_SIZE):
            batch = windows[start:start + BATCH_SIZE]
            total += len(batch) * qlike_from_log(y[start:start + BATCH_SIZE], model(batch)).item()
    return total / len(windows)


def fit_window_models(features, fit_origin, eligible_origins):
    """Train both fixed arms on causally eligible completed windows only."""
    data = training_windows(features, fit_origin, eligible_origins)
    flat = data["windows"].reshape(-1, len(FEATURE_COLUMNS))
    center, scale = flat.mean(axis=0), flat.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or (scale <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale neural input")
    target_scale = np.median(data["targets"], axis=0)
    windows = torch.tensor((data["windows"] - center) / scale, dtype=torch.float32)
    y = torch.tensor(data["targets"] / target_scale, dtype=torch.float32)
    models, model_audit = {}, {}
    with _deterministic_cpu():
        for name in MODEL_NAMES:
            torch.manual_seed(SEED)
            model = SmallVarianceMixer(name).to(device="cpu", dtype=torch.float32)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                         betas=(.9, .999), eps=1e-8, weight_decay=0.)
            initial_loss = _training_loss(model, windows, y)
            model.train()
            steps, max_gradient = 0, 0.
            for _ in range(EPOCHS):
                # Consecutive batches in chronological order, identical each
                # epoch; there is no shuffle, held-out score or early stopping.
                for start in range(0, len(windows), BATCH_SIZE):
                    batch = windows[start:start + BATCH_SIZE]
                    optimizer.zero_grad(set_to_none=True)
                    loss = qlike_from_log(y[start:start + BATCH_SIZE], model(batch))
                    loss.backward()
                    parameters = [p for p in model.parameters() if p.requires_grad]
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                        raise RuntimeError("Missing or nonfinite neural training gradient")
                    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1., error_if_nonfinite=True)
                    max_gradient = max(max_gradient, float(gradient_norm))
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in parameters):
                        raise RuntimeError("Nonfinite trained neural parameter")
                    steps += 1
            model.eval()
            final_loss = _training_loss(model, windows, y)
            models[name] = model
            model_audit[name] = {
                "parameters": sum(p.numel() for p in model.parameters()),
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED,
                "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                "adam_betas": [.9, .999], "adam_eps": 1e-8, "weight_decay": 0.,
                "gradient_clip_norm": 1., "max_preclip_gradient_norm": max_gradient,
                "steps": steps, "initial_training_qlike": initial_loss,
                "final_training_qlike": final_loss, "state_sha256": _state_hash(model),
                "device": "cpu", "dtype": "float32", "threads": 1,
                "backend": "vanilla" if name == "xlstm" else None,
                "dropout": 0., "early_stopping": False, "shuffle": False,
            }
    origins, ends = data["origins"], data["target_ends"]
    audit = {
        "fit_origin": str(pd.Timestamp(fit_origin).date()),
        "train_n": len(origins), "train_first_origin": str(origins[0].date()),
        "train_last_origin": str(origins[-1].date()),
        "train_last_target": str(pd.Timestamp(ends.to_numpy().max()).date()),
        "training_origins": [str(date.date()) for date in origins],
        "feature_columns": list(FEATURE_COLUMNS), "context": CONTEXT,
        "hidden": HIDDEN, "heads": HEADS, "memory_tokens": MEMORY_TOKENS,
        "input_center": center.tolist(), "input_scale": scale.tolist(),
        "scaling_rule": "population mean/std of flattened eligible training windows",
        "target_scale": target_scale.tolist(), "horizons": list(HORIZONS),
        "torch_version": torch.__version__, "xlstm_version": importlib.metadata.version("xlstm"),
        "models": model_audit,
    }
    return {"models": models, "input_center": center, "input_scale": scale,
            "target_scale": target_scale, "audit": audit}


def predict_window_models(fitted, features, origins):
    """Predict supplied current/future origins without reading any targets."""
    dates = pd.DatetimeIndex(origins)
    if len(dates) == 0 or (dates < pd.Timestamp(fitted["audit"]["fit_origin"])).any():
        raise ValueError("Predictions must be at or after the recorded fit origin")
    raw = input_windows(features, dates)
    windows = torch.tensor((raw - fitted["input_center"]) / fitted["input_scale"], dtype=torch.float32)
    predictions = {}
    with _deterministic_cpu(), torch.no_grad():
        for name in MODEL_NAMES:
            model = fitted["models"][name]
            model.eval()
            log_prediction = np.concatenate([
                model(windows[start:start + BATCH_SIZE]).numpy().astype(float)
                for start in range(0, len(windows), BATCH_SIZE)
            ])
            with np.errstate(over="raise", under="raise", invalid="raise"):
                prediction = np.exp(log_prediction) * fitted["target_scale"]
            if not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise RuntimeError("Nonpositive or nonfinite neural variance forecast")
            predictions[name] = prediction
    return predictions


def yearly_forecasts(features, eligible_origins, forecast_origins, progress=None, fit_callback=None):
    """Produce honest yearly-refit forecast rows and their complete fit audit.

    The caller supplies the fixed common origins and protects its evaluation
    fences. There are no file writes here. Optional callbacks can record live
    progress and persist each year's trained states before they are released.
    """
    dates = pd.DatetimeIndex(forecast_origins)
    if dates.has_duplicates or not dates.is_monotonic_increasing or len(dates) == 0:
        raise ValueError("Unique chronological forecast origins are required")
    if not dates.isin(pd.DatetimeIndex(eligible_origins)).all():
        raise ValueError("Forecast origin lies outside the declared common origin pool")
    input_windows(features, dates)
    y, ends = targets(features.rv_total)
    if not np.isfinite(y.loc[dates]).all().all() or ends.loc[dates].isna().any().any():
        raise ValueError("Requested forecast origins lack complete evaluation labels")
    records, audits = [], []
    for year in dates.year.unique():
        query = dates[dates.year == year]
        fit_origin = query[0]
        fitted = fit_window_models(features, fit_origin, eligible_origins)
        predicted = predict_window_models(fitted, features, query)
        if fit_callback is not None:
            fit_callback(fit_origin, fitted)
        audit = fitted["audit"]
        audits.append(audit)
        for name in MODEL_NAMES:
            for j, horizon in enumerate(HORIZONS):
                records.append(pd.DataFrame({
                    "origin": query, "horizon": horizon, "model": name,
                    "prediction": predicted[name][:, j], "y": y.loc[query, horizon].to_numpy(),
                    "target_end": ends.loc[query, horizon].to_numpy(), "fit_origin": fit_origin,
                    "train_n": audit["train_n"], "train_last_target": pd.Timestamp(audit["train_last_target"]),
                }))
        if progress is not None:
            progress({"year": int(year), "fit_origin": audit["fit_origin"],
                      "train_n": audit["train_n"], "forecast_origins": len(query),
                      "optimizer_steps_per_model": EPOCHS * math.ceil(audit["train_n"] / BATCH_SIZE)})
    return pd.concat(records, ignore_index=True), audits
