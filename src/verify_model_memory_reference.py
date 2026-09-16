"""Independent audit of reference-model additions and the combined family.

The established independent verifier supplies only independent reconstruction
and inference helpers. No reference or core producer is imported here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

from . import verify_model_memory_study as independent

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MODELS = ("nlinear", "xlstm", "moirai_univariate_gamma", "moirai_augmented_gamma")
REFERENCE_CONTRASTS = [(name, "baseline", "primary") for name in REFERENCE_MODELS] + [
    ("xlstm", "nlinear", "mechanism"),
    ("moirai_augmented_gamma", "gamma", "mechanism"),
    ("moirai_augmented_gamma", "moirai_univariate_gamma", "mechanism"),
]


def verify_common_panel(core, combined, reference_models=REFERENCE_MODELS):
    required = {"origin", "horizon", "model", "prediction", "y", "target_end"}
    if not required.issubset(core.columns) or not required.issubset(combined.columns):
        raise AssertionError("reference panel lacks required forecast columns")
    core, combined = core.copy(), combined.copy()
    for frame in (core, combined):
        for column in ("origin", "target_end"):
            frame[column] = pd.to_datetime(frame[column])
        if frame.duplicated(["origin", "horizon", "model"]).any():
            raise AssertionError("duplicate reference forecast key")
        if not np.isfinite(frame[["y", "prediction"]]).all().all() or (frame[["y", "prediction"]] <= 0).any().any():
            raise AssertionError("reference predictions/targets violate positivity")
    models = set(core["model"]) | set(reference_models)
    if set(combined["model"]) != models or set(core["horizon"]) != set(combined["horizon"]):
        raise AssertionError("reference model/horizon family changed")
    count = 0
    for horizon in sorted(set(core["horizon"])):
        base = core.loc[(core["horizon"] == horizon) & (core["model"] == "baseline")].set_index("origin").sort_index()
        for model in models:
            rows = combined.loc[(combined["horizon"] == horizon) & (combined["model"] == model)].set_index("origin").sort_index()
            if not rows.index.equals(base.index) or not rows["target_end"].equals(base["target_end"]):
                raise AssertionError("reference model common origin/target population differs")
            independent.same(rows["y"], base["y"], f"{model} common targets", rtol=1e-11)
            if model in set(core["model"]):
                original = core.loc[(core["horizon"] == horizon) & (core["model"] == model)].set_index("origin").sort_index()
                independent.same(rows["prediction"], original["prediction"], f"preserved core forecast {model}", rtol=0, atol=0)
            count += len(rows)
    if count != len(combined):
        raise AssertionError("unchecked reference forecast row")
    return {"common_panel_verification": "PASS", "combined_forecast_rows": count,
            "reference_rows": int(combined["model"].isin(reference_models).sum())}


def joint_holm_rows(core_metrics, reference_metrics):
    """Independently calculate the 46-way family from unadjusted paired tests."""
    rows = core_metrics["rows"] + reference_metrics["rows"]
    if len(core_metrics["rows"]) != 32 or len(reference_metrics["rows"]) != 14:
        raise AssertionError("combined multiplicity family must retain all 46 hypotheses")
    keys = [(int(row["horizon"]), row["candidate"], row["control"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise AssertionError("duplicate combined hypothesis")
    raw = np.asarray([row["p_conservative"] for row in rows], float)
    if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise AssertionError("invalid combined-family p-value")
    corrected = multipletests(raw, method="holm")[1]
    result = []
    for row, value in zip(rows, corrected, strict=True):
        verdict = "EXPLORATORY_SHORTLIST" if (value < 0.05 and row["improvement_pct"] >= 1
                   and all(period["delta"] < 0 for period in row["periods"])) else "INCONCLUSIVE"
        result.append({"horizon": int(row["horizon"]), "candidate": row["candidate"], "control": row["control"],
                       "p_holm_joint": float(value), "verdict_joint": verdict})
    return result


def neural_training_data(features, eligible_origins, fit_origin, context=22, maximum=1500, minimum=500):
    values = features[list(independent.BASELINE[1:])].to_numpy(float)
    complete_window = pd.Series(np.isfinite(values).all(axis=1), index=features.index).rolling(context).sum() == context
    targets = [independent.target_table(features, horizon) for horizon in (1, 5)]
    fit_origin = pd.Timestamp(fit_origin)
    eligible = features.index.isin(pd.DatetimeIndex(eligible_origins)) & complete_window
    for target in targets:
        eligible &= (target["y"] > 0) & np.isfinite(target["y"]) & (target["target_end"] <= fit_origin)
    eligible &= features.index < fit_origin
    indices = np.flatnonzero(eligible)[-maximum:]
    if len(indices) < minimum:
        raise AssertionError("independent neural training warmup insufficient")
    # Preserve the source's per-window column strides. Float32 recurrent
    # training can otherwise choose different reduction kernels after casting.
    windows = np.stack([values[i - context + 1:i + 1] for i in indices])
    y = np.column_stack([target.iloc[indices]["y"] for target in targets])
    last_end = max(target.iloc[indices]["target_end"].max() for target in targets)
    return features.index[indices], windows, y, last_end


def make_neural_replay(variant):
    """Architecture assembled independently, using the pinned upstream sLSTM."""
    import torch
    from torch import nn
    from xlstm.blocks.slstm.layer import sLSTMLayer, sLSTMLayerConfig

    class Replay(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = nn.Linear(22, 2)
            self.pre_encoding = nn.Linear(2, 16)
            self.readout = nn.Linear(176, 2)
            if variant == "xlstm":
                self.memory_tokens = nn.Parameter(torch.randn(2, 16) * 0.01)
                self.mixer = sLSTMLayer(sLSTMLayerConfig(
                    embedding_dim=16, num_heads=4, backend="vanilla", dtype="float32",
                    dtype_b="float32", dtype_r="float32", dtype_w="float32", dtype_g="float32",
                    dtype_s="float32", dtype_a="float32", enable_automatic_mixed_precision=False,
                    conv1d_kernel_size=0, dropout=0.0))
                self.mixer.reset_parameters()

        def forward(self, inputs):
            last = inputs[:, -1:, :].detach()
            coordinates = self.temporal((inputs - last).transpose(1, 2)) + last.transpose(1, 2)
            tokens = self.pre_encoding(coordinates)
            if variant == "xlstm":
                memory = self.memory_tokens[None].expand(len(inputs), -1, -1)
                forward = self.mixer(torch.cat((memory, tokens), dim=1))[:, 2:]
                reverse = self.mixer(torch.cat((memory, torch.flip(tokens, (1,))), dim=1))[:, 2:]
                tokens = tokens + (forward + torch.flip(reverse, (1,))) * 0.5
            return self.readout(tokens.reshape(len(inputs), -1))

    if variant not in ("nlinear", "xlstm"):
        raise AssertionError("unknown fixed replay architecture")
    return Replay()


def tensor_state_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def independently_train_neural(variant, windows, targets, seed=20260906, epochs=12):
    import torch
    torch.manual_seed(seed)
    model = make_neural_replay(variant).to(dtype=torch.float32, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    model.train()
    for _ in range(epochs):
        for start in range(0, len(windows), 128):
            optimizer.zero_grad(set_to_none=True)
            logratio = torch.log(targets[start:start + 128]) - model(windows[start:start + 128])
            loss = (torch.exp(logratio) - logratio - 1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
    model.eval()
    return model


def verify_neural(features, latents, historical_core, neural, fits, checkpoint_root, retrain_first=True):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    neural = neural.copy()
    for column in ("origin", "target_end", "fit_origin", "train_last_target"):
        neural[column] = pd.to_datetime(neural[column])
    if neural.duplicated(["origin", "horizon", "model"]).any() or set(neural["model"]) != {"nlinear", "xlstm"}:
        raise AssertionError("neural forecast key/model population differs")
    if not np.isfinite(neural[["y", "prediction"]]).all().all() or (neural[["y", "prediction"]] <= 0).any().any():
        raise AssertionError("neural finite positivity differs")
    complete = np.isfinite(features[list(independent.BASELINE) + list(independent.OTHER_INPUTS)]).all(axis=1) & features.index.isin(latents.index)
    eligible_origins = features.index[complete]
    reference = historical_core.loc[(historical_core["horizon"] == 1) & (historical_core["model"] == "baseline")].copy()
    reference["origin"] = pd.to_datetime(reference["origin"])
    dates = pd.DatetimeIndex(reference["origin"].sort_values())
    exclusions = []
    for date in dates:
        train_dates, _, _, _ = neural_training_data(features, eligible_origins, date, minimum=1)
        if len(train_dates) >= 500:
            break
        exclusions.append({"origin": str(date.date()), "train_n": len(train_dates)})
    if exclusions != [{"origin": date, "train_n": count} for date, count in
                      [("2013-02-01", 496), ("2013-02-04", 497), ("2013-02-05", 498), ("2013-02-06", 499)]]:
        raise AssertionError("initial neural warmup differs from the preserved pre-fit clarification")
    dates = dates[len(exclusions):]
    keyed_fits = {pd.Timestamp(row["fit_origin"]): row for row in fits}
    if len(keyed_fits) != len(fits):
        raise AssertionError("duplicate neural fit audit")
    for horizon in (1, 5):
        for name in ("nlinear", "xlstm"):
            actual = pd.DatetimeIndex(neural.loc[(neural["horizon"] == horizon) & (neural["model"] == name), "origin"].sort_values())
            if not actual.equals(dates):
                raise AssertionError("neural historical origins differ from fixed common origins")
    checked, retrained, used, maximum_error = 0, 0, set(), 0.0
    for year in dates.year.unique():
        query = dates[dates.year == year]
        fit_origin = query[0]
        audit = keyed_fits[fit_origin]
        used.add(fit_origin)
        train_dates, raw, targets, last_end = neural_training_data(features, eligible_origins, fit_origin)
        if (list(audit["training_origins"]) != [str(d.date()) for d in train_dates]
                or audit["train_n"] != len(train_dates) or pd.Timestamp(audit["train_last_target"]) != last_end
                or audit["feature_columns"] != list(independent.BASELINE[1:]) or audit["context"] != 22
                or audit["hidden"] != 16 or audit["heads"] != 4 or audit["memory_tokens"] != 2):
            raise AssertionError("neural architecture or completed training-origin audit differs")
        flattened = raw.reshape(-1, 11)
        center, scale = flattened.mean(0), flattened.std(0)
        units = np.median(targets, axis=0)
        independent.same(audit["input_center"], center, "neural training center", rtol=1e-11)
        independent.same(audit["input_scale"], scale, "neural training scale", rtol=1e-11)
        independent.same(audit["target_scale"], units, "neural completed target scale", rtol=1e-11)
        raw_values = features[list(independent.BASELINE[1:])].to_numpy(float)
        query_positions = features.index.get_indexer(query)
        query_windows = np.stack([raw_values[i - 21:i + 1] for i in query_positions])
        if not np.isfinite(query_windows).all():
            raise AssertionError("neural query windows have missing actual sessions")
        query_tensor = torch.tensor((query_windows - center) / scale, dtype=torch.float32)
        for name in ("nlinear", "xlstm"):
            model_audit = audit["models"][name]
            checkpoint = checkpoint_root / f"{fit_origin.date()}_{name}.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if tensor_state_digest(state) != model_audit["state_sha256"]:
                raise AssertionError("trained neural state identity changed")
            model = make_neural_replay(name)
            model.load_state_dict(state, strict=True)
            model.eval()
            if model_audit["epochs"] != 12 or model_audit["seed"] != 20260906 or model_audit["early_stopping"] or model_audit["shuffle"]:
                raise AssertionError("neural fixed training settings differ")
            if retrain_first and len(used) == 1:
                x = torch.tensor((raw - center) / scale, dtype=torch.float32)
                y = torch.tensor(targets / units, dtype=torch.float32)
                trained = independently_train_neural(name, x, y)
                if tensor_state_digest(trained.state_dict()) != model_audit["state_sha256"]:
                    raise AssertionError("first neural fit did not reproduce from independently selected windows and labels")
                retrained += 1
            with torch.no_grad():
                logs = np.concatenate([model(query_tensor[start:start + 128]).numpy().astype(float)
                                       for start in range(0, len(query), 128)])
            prediction = np.exp(logs) * units
            for column, horizon in enumerate((1, 5)):
                block = neural.loc[(neural["horizon"] == horizon) & (neural["model"] == name)].set_index("origin").reindex(query)
                target = independent.target_table(features, horizon).loc[query]
                if not (block["fit_origin"] == fit_origin).all() or not (block["train_n"] == len(train_dates)).all() or not (block["train_last_target"] == last_end).all() or not block["target_end"].equals(target["target_end"]):
                    raise AssertionError("neural forecast provenance/target dates differ")
                independent.same(block["y"], target["y"], "neural reconstructed target", rtol=1e-11)
                independent.same(block["prediction"], prediction[:, column], "independent neural architecture replay", rtol=1e-7)
                maximum_error = max(maximum_error, float(np.max(abs(block["prediction"].to_numpy() - prediction[:, column]))))
                checked += len(block)
    if used != set(keyed_fits) or checked != len(neural):
        raise AssertionError("unchecked neural fit/forecast")
    return {"neural_predictions_replayed": checked, "neural_annual_models_checked": 2 * len(fits),
            "neural_models_independently_retrained": retrained, "maximum_neural_replay_error": maximum_error,
            "independent_neural_warmup_exclusions": exclusions,
            "neural_verification_scope": "Every causal training-window/scaler audit and saved forecast replayed using separately assembled architecture; first annual pair retrained from independently selected completed targets"}


def verify_calibration(features, latents, summaries, forecasts, fits, core_protocol):
    complete = np.isfinite(features[list(independent.BASELINE) + list(independent.OTHER_INPUTS)]).all(axis=1) & features.index.isin(latents.index)
    rows = forecasts.copy()
    for name in ("origin", "fit_origin", "target_end", "train_last_target"):
        rows[name] = pd.to_datetime(rows[name])
    audits = {(int(row["horizon"]), row["model"], pd.Timestamp(row["fit_origin"])): row for row in fits}
    if len(audits) != len(fits):
        raise AssertionError("duplicate Moirai calibration fit audit")
    used, checked, refits, maximum_gradient = set(), 0, 0, 0.0
    for horizon in (1, 5):
        target = independent.target_table(features, horizon)
        base = rows.loc[(rows["model"] == "baseline") & (rows["horizon"] == horizon)].set_index("origin").sort_index()
        summary = summaries[f"moirai_log_h{horizon}"].reindex(features.index)
        for month in base.index.to_period("M").unique():
            query = base.index[base.index.to_period("M") == month]
            fit_origin = query[0]
            train_dates = features.index[complete & (features.index < fit_origin) & (target["target_end"] <= fit_origin)
                                         & np.isfinite(target["y"]) & (target["y"] > 0)]
            if len(train_dates) < core_protocol["sample"]["minimum_train"]:
                raise AssertionError("Moirai calibration training warmup insufficient")
            y = target.loc[train_dates, "y"].to_numpy()
            for model in REFERENCE_MODELS[2:]:
                key = (horizon, model, fit_origin)
                used.add(key)
                entry, audit = audits[key], audits[key]["audit"]
                xraw = summary.loc[train_dates].to_numpy()[:, None]
                qraw = summary.loc[query].to_numpy()[:, None]
                if model == "moirai_augmented_gamma":
                    xraw = np.column_stack([features.loc[train_dates, list(independent.BASELINE[1:])], xraw])
                    qraw = np.column_stack([features.loc[query, list(independent.BASELINE[1:])], qraw])
                if not np.isfinite(xraw).all() or not np.isfinite(qraw).all():
                    raise AssertionError("Moirai calibration lacks a declared input")
                center, scale = xraw.mean(0), xraw.std(0)
                if (scale <= 1e-12).any():
                    raise AssertionError("zero-scale Moirai calibration predictor")
                x, q = (xraw - center) / scale, (qraw - center) / scale
                last = target.loc[train_dates, "target_end"].max()
                if entry["train_n"] != len(train_dates) or pd.Timestamp(entry["train_last_target"]) != last:
                    raise AssertionError("Moirai calibration eligibility audit differs")
                independent.same(audit["input_center"], center, "Moirai training scaler mean", rtol=1e-11)
                independent.same(audit["input_scale"], scale, "Moirai training scaler scale", rtol=1e-11)
                independent.same(audit["target_scale"], np.median(y), "Moirai target unit normalization", rtol=1e-11)
                if (audit["alpha"] != 0 or not audit["converged"] or audit["n_train"] != len(y)
                        or audit["n_iter"] >= 2000 or audit["solver"] != "newton-cholesky"):
                    raise AssertionError("Moirai Gamma objective/convergence settings differ")
                coefficient = np.asarray(audit["coefficients"])
                linear = x @ coefficient + audit["intercept"]
                residual_score = (1 - y / np.exp(linear)) / len(y)
                gradient = np.r_[residual_score.sum(), x.T @ residual_score]
                norm = float(max(abs(gradient)))
                if norm > 2e-7:
                    raise AssertionError("Moirai convex Gamma stationarity failed")
                independent.same(audit["gradient_inf_norm"], norm, "Moirai independent Gamma gradient", atol=2e-12)
                maximum_gradient = max(maximum_gradient, norm)
                prediction = np.exp(q @ coefficient + audit["intercept"])
                if fit_origin.month == 1:
                    beta = independent.independently_fit_gamma(x, y, np.ones(len(y)), 0)
                    independent.same(np.exp(beta[0] + q @ beta[1:]), prediction, "independent Moirai Gamma scipy refit", rtol=5e-6)
                    refits += 1
                saved = rows.loc[(rows["horizon"] == horizon) & (rows["model"] == model)].set_index("origin").reindex(query)
                if (not (saved["fit_origin"] == fit_origin).all() or not (saved["train_n"] == len(train_dates)).all()
                        or not (saved["train_last_target"] == last).all() or not saved["target_end"].equals(target.loc[query, "target_end"])):
                    raise AssertionError("Moirai forecast provenance differs")
                independent.same(saved["prediction"], prediction, "independent Moirai calibrated forecast", rtol=2e-7)
                independent.same(saved["y"], target.loc[query, "y"], "independent Moirai target", rtol=1e-11)
                checked += len(saved)
    if used != set(audits) or checked != int(rows["model"].isin(REFERENCE_MODELS[2:]).sum()):
        raise AssertionError("unchecked Moirai calibration fit/forecast")
    return {"moirai_calibrated_forecasts_checked": checked, "moirai_monthly_calibrations_checked": len(fits),
            "moirai_independent_scipy_refits": refits, "maximum_moirai_gamma_gradient": maximum_gradient}


def verify_moirai_contexts(features, summaries, contexts, reference_protocol):
    if not isinstance(summaries.index, pd.DatetimeIndex) or summaries.index.has_duplicates or not summaries.index.is_monotonic_increasing:
        raise AssertionError("invalid Moirai summary dates")
    if list(summaries.columns) != ["moirai_log_h1", "moirai_log_h5"] or not np.isfinite(summaries).all().all():
        raise AssertionError("invalid Moirai summary values/schema")
    start = pd.Timestamp(reference_protocol["moirai"]["extraction_start"])
    end = pd.Timestamp(reference_protocol["sample"]["score_end"])
    expected_dates = features.index[(features.index >= start) & (features.index <= end)]
    context = reference_protocol["moirai"]["context_sessions"]
    variance = features["rv_total"]
    valid = pd.Series(np.isfinite(variance) & (variance > 0), index=features.index).rolling(context).sum() == context
    expected_dates = expected_dates[valid.loc[expected_dates]]
    if not summaries.index.equals(expected_dates) or not contexts.index.equals(expected_dates):
        raise AssertionError("Moirai complete context population differs")
    for column in ("context_start", "context_end"):
        contexts[column] = pd.to_datetime(contexts[column])
    positions = features.index.get_indexer(expected_dates)
    if (not np.array_equal(contexts["context_rows"].to_numpy(), np.repeat(context, len(expected_dates)))
            or not pd.DatetimeIndex(contexts["context_start"]).equals(features.index[positions - context + 1])
            or not pd.DatetimeIndex(contexts["context_end"]).equals(expected_dates)):
        raise AssertionError("Moirai context uses future, missing, padded or nonconsecutive sessions")
    return {"moirai_contexts_checked": len(contexts), "moirai_context_sessions": context}


MOIRAI_REPLAY_SCRIPT = r'''
import hashlib, importlib.util, json, sys, tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

root = Path(sys.argv[1])
local = root / "data/model_memory_reference"
report = root / "reports/model_memory_study"
manifest = json.loads((report / "moirai_extraction_manifest.json").read_text())
provenance = manifest["provenance"]
revision = provenance["official_code_revision"]
archive_path = local / "vendor" / f"uni2ts-{revision}.tar.gz"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for label, path in (("code_archive", archive_path), ("config.json", local / "moirai2_checkpoint/config.json"),
                    ("model.safetensors", local / "moirai2_checkpoint/model.safetensors")):
    assert digest(path) == provenance["artifact_sha256"][label], f"Changed official artifact: {label}"
package = Path(importlib.util.find_spec("uni2ts").origin).parent
source_hashes = {}
with tarfile.open(archive_path) as archive:
    prefix = f"uni2ts-{revision}/src/uni2ts/"
    for member in archive.getmembers():
        if member.isfile() and member.name.startswith(prefix) and member.name.endswith(".py"):
            relative = member.name[len(prefix):]
            assert ".." not in Path(relative).parts
            expected = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            assert digest(package / relative) == expected, f"Installed upstream source changed: {relative}"
            source_hashes[relative] = expected
assert len(source_hashes) == provenance["checked_installed_python_files"]
assert hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest() == provenance["installed_source_tree_sha256"]
source = pd.read_parquet(root / "data/orthogonal_round2/features.parquet", columns=["rv_total"]).loc[:"2025-10-10", "rv_total"]
source_hash = hashlib.sha256(source.to_csv(float_format="%.17g", date_format="%Y-%m-%d").encode()).hexdigest()
assert source_hash == manifest["bounded_rv_sha256"]
cached = np.load(local / "moirai_quantiles.npz", allow_pickle=False)
origins = pd.DatetimeIndex(cached["origins"])
positions = source.index.get_indexer(origins)
logged = np.log(source.to_numpy())
windows = np.array([logged[pos - 511:pos + 1] for pos in positions], dtype="<f8")
assert windows.shape == (len(origins), 512) and np.isfinite(windows).all()
assert hashlib.sha256(windows.tobytes()).hexdigest() == manifest["context_matrix_sha256"]
torch.manual_seed(20260906)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
module = Moirai2Module(**json.loads((local / "moirai2_checkpoint/config.json").read_text()))
module.load_state_dict(load_file(str(local / "moirai2_checkpoint/model.safetensors")), strict=True)
model = Moirai2Forecast(prediction_length=5, target_dim=1, feat_dynamic_real_dim=0,
                       past_feat_dynamic_real_dim=0, context_length=512, module=module).to("cpu").eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
output = []
with torch.inference_mode():
    for start in range(0, len(windows), 32):
        result = model.predict([row.astype(np.float32) for row in windows[start:start + 32]])
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        output.append(np.asarray(result))
prediction = np.concatenate(output)
if prediction.ndim == 4 and prediction.shape[-1] == 1:
    prediction = prediction[..., 0]
expected = cached["predictions"]
assert prediction.shape == expected.shape == (len(origins), 9, 5)
assert np.allclose(prediction, expected, rtol=1e-6, atol=1e-6)
print(json.dumps({"moirai_raw_forecasts_replayed": len(origins), "moirai_quantile_values_checked": int(expected.size),
                  "maximum_moirai_native_replay_error": float(np.max(np.abs(prediction - expected))),
                  "moirai_installed_official_files_checked": len(source_hashes),
                  "moirai_fresh_context_hash": hashlib.sha256(windows.tobytes()).hexdigest()}))
'''


def verify_moirai_extraction(root, features, protocol):
    from scipy.special import logsumexp
    local, report = root / "data/model_memory_reference", root / "reports/model_memory_study"
    manifest = json.loads((report / "moirai_extraction_manifest.json").read_text())
    audit = json.loads((report / "moirai_extraction_audit.json").read_text())
    if audit["provenance"] != manifest["provenance"] or manifest["status"] != "FROZEN_BEFORE_HISTORICAL_EXTRACTION":
        raise AssertionError("Moirai frozen extraction identity differs")
    provenance = audit["provenance"]
    if (provenance["official_code_revision"] != protocol["moirai"]["code_revision"]
            or provenance["official_model_revision"] != protocol["moirai"]["checkpoint_revision"]):
        raise AssertionError("Moirai pinned code/checkpoint revision differs")
    for relative, expected in manifest["protocol_sha256"].items():
        if independent.digest(root / relative) != expected:
            raise AssertionError("Moirai pre-extraction protocol changed")
    for field, relative in (("output_sha256", "data/model_memory_reference/moirai_features.parquet"),
                            ("contexts_sha256", "data/model_memory_reference/moirai_contexts.parquet"),
                            ("quantiles_sha256", "data/model_memory_reference/moirai_quantiles.npz"),
                            ("manifest_sha256", "reports/model_memory_study/moirai_extraction_manifest.json")):
        if audit[field] != independent.digest(root / relative):
            raise AssertionError(f"Moirai {field} differs")
    for field, relative in (("dependency_lock_sha256", "data/model_memory_reference/moirai2_requirements.lock.txt"),
                            ("install_report_sha256", "data/model_memory_reference/moirai2_install_report.json"),
                            ("adapter_sha256", "src/reference_moirai.py"), ("tests_sha256", "tests/test_reference_moirai.py"),
                            ("plan_sha256", "reports/model_memory_study/MOIRAI_PLAN.md")):
        if provenance[field] != independent.digest(root / relative):
            raise AssertionError(f"Moirai provenance {field} differs")
    summaries = pd.read_parquet(local / "moirai_features.parquet")
    contexts = pd.read_parquet(local / "moirai_contexts.parquet")
    result = verify_moirai_contexts(features, summaries, contexts, protocol)
    quantiles = np.load(local / "moirai_quantiles.npz", allow_pickle=False)
    if not pd.DatetimeIndex(quantiles["origins"]).equals(summaries.index):
        raise AssertionError("Moirai quantile origin alignment differs")
    levels = quantiles["quantile_levels"]
    independent.same(levels, np.arange(1, 10) / 10, "Moirai registered quantile grid", rtol=1e-7)
    median = quantiles["predictions"][:, int(np.argmin(abs(levels - 0.5))), :].astype(float)
    independent.same(summaries["moirai_log_h1"], median[:, 0], "Moirai one-session median summary", rtol=1e-10)
    independent.same(summaries["moirai_log_h5"], logsumexp(median, axis=1) - np.log(5), "Moirai five-session median-log summary", rtol=1e-10)
    completed = subprocess.run([str(local / "moirai2_env/bin/python"), "-c", MOIRAI_REPLAY_SCRIPT, str(root)],
                               cwd=root, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise AssertionError(f"Independent upstream Moirai replay failed: {completed.stderr[-4000:]}")
    result.update(json.loads(completed.stdout.strip().splitlines()[-1]))
    return result


def verify(root=ROOT):
    report = root / "reports/model_memory_study"
    data = root / "data/model_memory_reference"
    protocol = yaml.safe_load((root / "model_memory_reference.yaml").read_text())
    core_protocol = yaml.safe_load((root / "model_memory_study.yaml").read_text())
    manifest = json.loads((report / "reference_manifest.json").read_text())
    if manifest["protocol_sha256"] != independent.digest(root / "model_memory_reference.yaml"):
        raise AssertionError("reference protocol identity differs")
    for relative, expected in manifest["hashes"].items():
        if independent.digest(root / relative) != expected:
            raise AssertionError(f"reference frozen input/code identity differs: {relative}")
    neural_manifest = json.loads((data / "neural_manifest.json").read_text())
    for category in ("code", "protocols", "inputs"):
        for relative, expected in neural_manifest[category].items():
            if independent.digest(root / relative) != expected:
                raise AssertionError(f"neural frozen {category} identity differs: {relative}")
    for category in ("outputs", "checkpoints"):
        for relative, expected in neural_manifest[category].items():
            if independent.digest(data / relative) != expected:
                raise AssertionError(f"neural saved {category} identity differs: {relative}")
    for field, filename in (("initial_manifest_sha256", "neural_manifest_initial_insufficient.json"),
                            ("warmup_exclusions_sha256", "neural_warmup_exclusions.json"),
                            ("pre_run_checks_sha256", "neural_pre_run_checks.txt")):
        if independent.digest(data / filename) != neural_manifest[field]:
            raise AssertionError("neural chronology-only warmup/provenance identity differs")
    raw_features = independent.reconstruct_features(root, core_protocol)
    features = pd.read_parquet(root / core_protocol["inputs"]["features"]).loc[:core_protocol["sample"]["latest_target"]]
    if not raw_features.index.equals(features.index):
        raise AssertionError("reference source feature calendar differs")
    for column in raw_features:
        independent.same(features[column], raw_features[column], f"reference source {column}", rtol=1e-10)
    latents = pd.read_parquet(root / core_protocol["inputs"]["latents"]).loc[:core_protocol["sample"]["score_end"]]
    core = pd.read_parquet(root / "data/model_memory_study/forecasts.parquet")
    combined = pd.read_parquet(data / "forecasts.parquet")
    result = verify_common_panel(core, combined)
    result.update(verify_moirai_extraction(root, features, protocol))
    neural = pd.read_parquet(data / "neural_forecasts.parquet")
    fits = json.loads((data / "neural_fits.json").read_text())
    historical_core = pd.read_parquet(root / "data/model_memory_study/components.parquet")
    result.update(verify_neural(features, latents, historical_core, neural, fits, data / "checkpoints"))
    exclusions = json.loads((data / "neural_warmup_exclusions.json").read_text())
    if [{"origin": row["origin"], "train_n": row["train_n"]} for row in exclusions] != result["independent_neural_warmup_exclusions"]:
        raise AssertionError("saved neural exclusions differ from independently verified eligibility")
    summaries = pd.read_parquet(data / "moirai_features.parquet")
    calibrations = json.loads((data / "calibration_fits.json").read_text())
    result.update(verify_calibration(features, latents, summaries, combined, calibrations, core_protocol))
    core_metrics = json.loads((report / "metrics.json").read_text())
    reference_metrics = json.loads((report / "reference_metrics.json").read_text())
    if reference_metrics["protocol_sha256"] != manifest["protocol_sha256"]:
        raise AssertionError("reference metric protocol identity differs")
    result.update(independent.verify_metrics(combined, features, reference_metrics, core_protocol,
                                            comparisons=REFERENCE_CONTRASTS, hypothesis_count=14))
    expected = joint_holm_rows(core_metrics, reference_metrics)
    recorded = json.loads((report / "combined_metrics.json").read_text())
    if recorded["hypothesis_count"] != 46 or recorded["reference_protocol_sha256"] != manifest["protocol_sha256"]:
        raise AssertionError("combined family identity/count differs")
    observed = {(int(row["horizon"]), row["candidate"], row["control"]): row for row in recorded["rows"]}
    if len(observed) != 46 or len(recorded["rows"]) != 46:
        raise AssertionError("combined hypothesis key population differs")
    original = {(int(row["horizon"]), row["candidate"], row["control"]): row for row in core_metrics["rows"] + reference_metrics["rows"]}
    for row in expected:
        key = (row["horizon"], row["candidate"], row["control"])
        saved = observed[key]
        independent.same(saved["p_holm_all46"], row["p_holm_joint"], "independent full46 Holm", rtol=1e-10)
        if saved["verdict_all46"] != row["verdict_joint"]:
            raise AssertionError("combined46 joint gate differs")
        if {name: value for name, value in saved.items() if name not in ("p_holm_all46", "verdict_all46")} != original[key]:
            raise AssertionError("combined report changed an original metric/uncertainty field")
    core_verification = json.loads((report / "verification.json").read_text())
    if core_verification["status"] != "PASS" or core_verification["protocol_sha256"] != independent.digest(root / "model_memory_study.yaml"):
        raise AssertionError("combined family lacks matching passed core verification")
    result.update({"status": "PASS", "combined_hypotheses_checked": 46,
                   "reference_protocol_sha256": independent.digest(root / "model_memory_reference.yaml"),
                   "verifier_sha256": independent.digest(Path(__file__)),
                   "verifier_tests_sha256": independent.digest(root / "tests/test_verify_model_memory_reference.py"),
                   "limitations": "All reference predictions replayed; first neural annual pair independently retrained, later training checked by independently reconstructed eligibility/maps and saved-state replay. Unknown foundation checkpoint pretraining overlap and reused historical sample prevent confirmatory claims."})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify(args.root)
    destination = args.root / "reports/model_memory_study/reference_verification.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
