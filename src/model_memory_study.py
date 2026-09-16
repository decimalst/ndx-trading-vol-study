"""Fixed broad model/memory comparison, isolated from existing studies."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import eigh
from scipy.special import softmax

from . import orthogonal_round2 as parent

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "model_memory_study.yaml"
OUT = ROOT / "data/model_memory_study"
REPORT = ROOT / "reports/model_memory_study"
COMPONENTS = ("baseline", "gamma", "adaptive_ols", "adaptive_gamma", "ridge_gamma", "spline_gamma")
MODELS = COMPONENTS + ("global_calibration", "observable_memory", "latent_memory", "equal_ensemble", "dynamic_ensemble")
BASE = parent.BASE


def validate_protocol(p):
    s = p["sample"]
    if (tuple(p["component_models"]) != COMPONENTS or tuple(p["models"]) != MODELS
            or tuple(p["baseline_columns"]) != BASE or s["horizons"] != [1, 5]
            or s["score_start"] != "2016-01-04" or s["score_end"] != "2025-10-10"
            or s["latest_target"] != "2025-10-20" or s["sealed_start"] != "2025-11-03"
            or p["comparisons"]["total_hypotheses"] != 32):
        raise ValueError("Protocol differs from fixed implemented family or fences")
    if (p["memory"]["neighbors"] != 64 or p["memory"]["pca_components"] != 8
            or p["memory"]["shrinkage"] != .25
            or p["memory"]["lookback_sessions"] != 1260
            or p["memory"]["min_records"] != 252
            or p["memory"]["additional_gap_after_target_sessions"] != 22
            or p["ensemble"]["half_life_sessions"] != 63
            or p["ensemble"]["temperature"] != 5
            or p["ensemble"]["equal_weight_floor"] != .1
            or p["ensemble"]["lookback_sessions"] != 252
            or p["ensemble"]["min_records"] != 126
            or p["inference"]["blocks"] != [21, 63, 126]
            or p["inference"]["bootstrap_draws"] != 4999
            or p["inference"]["seed"] != 20260907):
        raise ValueError("Fixed memory/ensemble parameters changed")


def positions(dates, sessions):
    sessions = pd.DatetimeIndex(sessions)
    dates = pd.DatetimeIndex(dates)
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("Unique chronological sessions required")
    if dates.isna().any():
        raise ValueError("Unknown target timestamp")
    pos = sessions.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError("Timestamp absent from session calendar")
    return pos


def eligible_memory(origins, target_ends, query, sessions, lookback=1260, gap=22):
    if len(origins) != len(target_ends) or lookback < 1 or gap < 0:
        raise ValueError("Invalid memory parameters")
    op = positions(origins, sessions)
    tp = positions(target_ends, sessions)
    qp = positions([query], sessions)[0]
    if (tp <= op).any():
        raise ValueError("Target must finish after historical origin")
    return (op < qp) & (tp <= qp - gap) & (op >= qp - lookback)


def raw_keys(train, apply):
    tr, ap = np.asarray(train, float), np.asarray(apply, float)
    if (tr.ndim != 2 or ap.ndim != 2 or tr.shape[1] != ap.shape[1]
            or len(tr) < 2 or not np.isfinite(tr).all() or not np.isfinite(ap).all()):
        raise ValueError("Finite matching key matrices required")
    mu, sd = tr.mean(axis=0), tr.std(axis=0, ddof=0)
    if (sd <= 1e-12).any():
        raise ValueError("Zero-scale observable key")
    return (tr - mu) / sd, (ap - mu) / sd


def latent_keys(train_z, apply_z, n_components=8):
    tr, ap = np.asarray(train_z, float), np.asarray(apply_z, float)
    if (tr.ndim != 2 or ap.ndim != 2 or tr.shape[1] != ap.shape[1]
            or not np.isfinite(tr).all() or not np.isfinite(ap).all()
            or not 1 <= n_components <= min(tr.shape[1], len(tr) - 1)):
        raise ValueError("Invalid PCA key matrices or dimension")
    mu = tr.mean(axis=0)
    centered = tr - mu
    cov = centered.T @ centered / (len(tr) - 1)
    vals, vec = eigh(cov, subset_by_index=[tr.shape[1] - n_components, tr.shape[1] - 1])
    vals, vec = vals[::-1], vec[:, ::-1]
    if vals[-1] <= max(vals[0], 1) * 1e-12:
        raise ValueError("Insufficient PCA rank")
    # Sign convention affects reproducibility, never Euclidean neighbors.
    for j in range(vec.shape[1]):
        if vec[np.argmax(np.abs(vec[:, j])), j] < 0:
            vec[:, j] *= -1
    scale = np.sqrt(vals.sum())
    return centered @ vec / scale, (ap - mu) @ vec / scale


def neighbor_indices(keys, query, k=64):
    keys, query = np.asarray(keys, float), np.asarray(query, float)
    if (keys.ndim != 2 or query.ndim != 1 or keys.shape[1] != len(query)
            or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(keys)
            or not np.isfinite(keys).all() or not np.isfinite(query).all()):
        raise ValueError("Invalid retrieval query")
    distance = np.sum((keys - query) ** 2, axis=1)
    return np.argsort(distance, kind="stable")[:k]


def ratio_correction(ratios, shrinkage=.25):
    ratios = np.asarray(ratios, float)
    if (ratios.ndim != 1 or len(ratios) == 0 or not np.isfinite(ratios).all()
            or (ratios <= 0).any() or not 0 <= shrinkage <= 1):
        raise ValueError("Positive finite ratios and convex shrinkage required")
    return float(1 - shrinkage + shrinkage * ratios.mean())


def ensemble_weights(losses, ages, half_life=63, temperature=5, floor=.1):
    losses, ages = np.asarray(losses, float), np.asarray(ages, float)
    if (losses.ndim != 2 or len(losses) == 0 or losses.shape[1] == 0
            or ages.shape != (len(losses),) or not np.isfinite(losses).all()
            or not np.isfinite(ages).all() or (ages < 0).any() or (losses < -1e-12).any()
            or not np.isfinite([half_life, temperature, floor]).all()
            or half_life <= 0 or temperature < 0 or not 0 <= floor <= 1):
        raise ValueError("Invalid expert history or weighting parameters")
    weights = np.exp2(-ages / half_life)
    if weights.sum() <= 0:
        raise ValueError("Degenerate history weights")
    mean_loss = np.average(losses, weights=weights, axis=0)
    return floor / losses.shape[1] + (1 - floor) * softmax(-temperature * mean_loss)


def load_inputs(p):
    features = pd.read_parquet(ROOT / p["inputs"]["features"])
    features = features.loc[:p["sample"]["latest_target"]]
    z = pd.read_parquet(ROOT / p["inputs"]["latents"])
    z = z.loc[:p["sample"]["score_end"]]
    if not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise ValueError("Invalid feature calendar")
    if z.shape[1] != 512 or z.index.has_duplicates or not np.isfinite(z).all().all():
        raise ValueError("Unexpected latent cache")
    return features, z


def component_forecasts(features, z, p):
    from .model_memory_estimators import fit_models

    all_complete = (np.isfinite(features.loc[:, parent.ALL_FEATURES]).all(axis=1)
                    & features.index.isin(z.index))
    records, fits = [], []
    for h in p["sample"]["horizons"]:
        targets = parent.make_targets(features.rv_total, h)
        candidate = (all_complete & np.isfinite(targets.y)
                     & (targets.target_end <= pd.Timestamp(p["sample"]["latest_target"]))
                     & (features.index >= pd.Timestamp(p["sample"]["historical_forecast_start"]))
                     & (features.index <= pd.Timestamp(p["sample"]["score_end"])))
        dates = features.index[candidate]
        months = dates.to_period("M")
        for mi, month in enumerate(months.unique()):
            query = dates[months == month]
            origin = query[0]
            mask = (all_complete & np.isfinite(targets.y) & (features.index < origin)
                    & (targets.target_end <= origin))
            if mask.sum() < p["sample"]["minimum_train"]:
                raise ValueError(f"INSUFFICIENT_DATA at {origin}: {int(mask.sum())}")
            age = positions([origin], features.index)[0] - positions(features.index[mask], features.index)
            result = fit_models(features.loc[mask, BASE], targets.loc[mask, "y"].to_numpy(),
                                features.loc[query, BASE], age)
            last_target = targets.loc[mask, "target_end"].max()
            fits.append({"horizon": h, "fit_origin": str(origin.date()), "train_n": int(mask.sum()),
                         "train_last_target": str(last_target.date()), "model_audit": result["audit"]})
            if set(result["predictions"]) != set(COMPONENTS):
                raise ValueError("Component model mismatch")
            for model in COMPONENTS:
                pred = np.asarray(result["predictions"][model])
                parent.qlike(targets.loc[query, "y"], pred)
                records.append(pd.DataFrame({"origin": query, "horizon": h, "model": model,
                    "target_end": targets.loc[query, "target_end"].to_numpy(),
                    "y": targets.loc[query, "y"].to_numpy(), "prediction": pred,
                    "fit_origin": origin, "train_n": int(mask.sum()), "train_last_target": last_target}))
            if mi % 12 == 0:
                print(f"component h{h}: {origin.date()}, {int(mask.sum())} training rows", flush=True)
    return pd.concat(records, ignore_index=True), fits


def memory_forecasts(features, z, components, p):
    output, audits, neighbors = [], [], []
    sessions = features.index
    for h in p["sample"]["horizons"]:
        fc = components.loc[components.horizon == h]
        wide = fc.pivot(index="origin", columns="model", values="prediction").sort_index().loc[:, COMPONENTS]
        base = fc.loc[fc.model == "baseline"].set_index("origin").sort_index()
        target_ends = pd.DatetimeIndex(base.target_end)
        y = base.y.to_numpy()
        ratios = y / wide.baseline.to_numpy()
        loss_history = np.column_stack([parent.qlike(y, wide[c]) for c in COMPONENTS])
        raw = features.loc[wide.index, BASE[1:]].to_numpy()
        latent = z.loc[wide.index].to_numpy(float)
        query_dates = wide.index[wide.index >= pd.Timestamp(p["sample"]["score_start"])]
        qpos_all = positions(wide.index, sessions)
        months = query_dates.to_period("M")
        for month in months.unique():
            dates = query_dates[months == month]
            key_fit = dates[0]
            train = eligible_memory(wide.index, target_ends, key_fit, sessions)
            if train.sum() < p["memory"]["min_records"]:
                raise ValueError("INSUFFICIENT_DATA: memory warm-up")
            # Fit monthly maps only on admissible historical records. All applied
            # rows are merely transformed; each query searches its own causal pool.
            _, raw_map = raw_keys(raw[train], raw)
            _, latent_map = latent_keys(latent[train], latent, p["memory"]["pca_components"])
            for origin in dates:
                qi = wide.index.get_loc(origin)
                pool = eligible_memory(wide.index, target_ends, origin, sessions)
                inds = np.flatnonzero(pool)
                if len(inds) < p["memory"]["min_records"]:
                    raise ValueError("INSUFFICIENT_DATA: daily memory pool")
                pred_base = float(wide.baseline.iloc[qi])
                multipliers = {"global_calibration": ratio_correction(ratios[pool])}
                for kind, keys, model in [("observable", raw_map, "observable_memory"),
                                          ("latent", latent_map, "latent_memory")]:
                    near = inds[neighbor_indices(keys[pool], keys[qi], p["memory"]["neighbors"])]
                    distance = np.sum((keys[near] - keys[qi]) ** 2, axis=1)
                    multipliers[model] = ratio_correction(ratios[near])
                    neighbors.append(pd.DataFrame({"origin": origin, "horizon": h, "kind": kind,
                        "rank": np.arange(len(near)), "neighbor_origin": wide.index[near],
                        "distance": distance, "ratio": ratios[near]}))
                epool = eligible_memory(wide.index, target_ends, origin, sessions,
                                         lookback=p["ensemble"]["lookback_sessions"], gap=0)
                if epool.sum() < p["ensemble"]["min_records"]:
                    raise ValueError("INSUFFICIENT_DATA: expert-history warm-up")
                ages = positions([origin], sessions)[0] - qpos_all[epool]
                weights = ensemble_weights(loss_history[epool], ages)
                forecasts = {m: pred_base * c for m, c in multipliers.items()}
                forecasts["equal_ensemble"] = float(wide.iloc[qi].mean())
                forecasts["dynamic_ensemble"] = float(weights @ wide.iloc[qi].to_numpy())
                for model, pred in forecasts.items():
                    output.append({"origin": origin, "horizon": h, "model": model,
                        "target_end": target_ends[qi], "y": y[qi], "prediction": pred,
                        "fit_origin": key_fit, "train_n": int(pool.sum()),
                        "train_last_target": target_ends[pool].max()})
                audits.append({"origin": origin, "horizon": h, "key_fit_origin": key_fit,
                    "pool_n": int(pool.sum()), "global_multiplier": multipliers["global_calibration"],
                    "observable_multiplier": multipliers["observable_memory"],
                    "latent_multiplier": multipliers["latent_memory"],
                    "max_memory_target": target_ends[pool].max(), "ensemble_n": int(epool.sum()),
                    "max_ensemble_target": target_ends[epool].max(),
                    **{f"weight_{m}": float(w) for m, w in zip(COMPONENTS, weights)}})
            if month.month == 1:
                print(f"memory h{h}: {month}, pool={int(train.sum())}", flush=True)
    scored = components.loc[components.origin >= pd.Timestamp(p["sample"]["score_start"])]
    return (pd.concat([scored, pd.DataFrame(output)], ignore_index=True),
            pd.DataFrame(audits), pd.concat(neighbors, ignore_index=True))


def evaluate(forecasts, features, p, *, contrasts=None, models=MODELS):
    if contrasts is None:
        contrasts = [(m, "baseline", "primary") for m in MODELS[1:]]
        contrasts += [(a, b, "mechanism") for a, b in p["comparisons"]["mechanisms"]]
    rows = []
    for h in p["sample"]["horizons"]:
        f = forecasts.loc[forecasts.horizon == h]
        wide = f.pivot(index="origin", columns="model", values="prediction").sort_index()
        if set(wide) != set(models) or not np.isfinite(wide).all().all():
            raise ValueError("Common-arm sample failed")
        y = f.drop_duplicates("origin").set_index("origin").y.reindex(wide.index)
        losses = {m: parent.qlike(y, wide[m]) for m in models}
        differences = np.column_stack([losses[a] - losses[b] for a, b, _ in contrasts])
        boot = {b: parent.bootstrap_means(differences, b, p["inference"]["bootstrap_draws"],
                                          p["inference"]["seed"] + h * 1000 + b)
                for b in p["inference"]["blocks"]}
        for j, (candidate, control, role) in enumerate(contrasts):
            d = differences[:, j]
            delta = float(d.mean())
            block_info = {}
            for b, samples in boot.items():
                bp = (1 + int((np.abs(samples[:, j] - delta) >= abs(delta)).sum())) / (len(samples) + 1)
                block_info[str(b)] = {"p": bp, "ci95": np.quantile(samples[:, j], [.025, .975]).tolist()}
            hac = parent.hac_summary(d)
            intervals = [x["ci95"] for x in block_info.values()] + [hac["ci95"]]
            periods = []
            for start, end in p["inference"]["stability_periods"]:
                ix = (wide.index >= start) & (wide.index <= end)
                periods.append({"start": start, "end": end, "n": int(ix.sum()), "delta": float(d[ix].mean())})
            annual = [{"year": int(yr), "n": int((wide.index.year == yr).sum()),
                       "delta": float(d[wide.index.year == yr].mean())} for yr in sorted(set(wide.index.year))]
            phase = positions(wide.index, features.index) % h
            phases = [{"phase": i, "n": int((phase == i).sum()), "delta": float(d[phase == i].mean())} for i in range(h)]
            rows.append({"candidate": candidate, "control": control, "role": role, "horizon": h,
                "n": len(d), "delta": delta, "baseline_qlike": float(losses[control].mean()),
                "candidate_qlike": float(losses[candidate].mean()),
                "improvement_pct": float(-100 * delta / losses[control].mean()),
                "win_rate": float((d < 0).mean()), "block_inference": block_info, "hac126": hac,
                "ci95_envelope": [min(x[0] for x in intervals), max(x[1] for x in intervals)],
                "p_conservative": max(hac["p"], *(x["p"] for x in block_info.values())),
                "periods": periods, "annual": annual, "phases": phases,
                "best_component_qlike": min(float(losses[m].mean()) for m in COMPONENTS)})
    if len(rows) != p["comparisons"]["total_hypotheses"]:
        raise ValueError("Comparison family changed")
    adjusted = parent.holm_adjust([r["p_conservative"] for r in rows])
    for r, hp in zip(rows, adjusted):
        r["p_holm"] = float(hp)
        passed = hp < .05 and r["improvement_pct"] >= 1 and all(x["delta"] < 0 for x in r["periods"])
        r["verdict"] = "EXPLORATORY_SHORTLIST" if passed else "INCONCLUSIVE"
        r["beats_best_component_point"] = r["candidate_qlike"] < r["best_component_qlike"]
    return {"evidence_class": p["evidence_class"], "rows": rows,
            "protocol_sha256": parent.digest(PROTOCOL), "hypothesis_count": len(rows)}


def report(metrics, audit):
    rows = metrics["rows"]
    lines = ["# Broad modeling and memory experiment", "",
        "Exploratory on previously inspected history. Eleven models, two horizons, 32 prespecified comparisons.",
        "Positive improvement means lower QLIKE. All arms use the same target and scoring origins.", "",
        "| model | h1 improvement | h5 improvement | h1 adjusted p | h5 adjusted p |",
        "|---|---:|---:|---:|---:|"]
    for model in MODELS[1:]:
        r = [next(x for x in rows if x["candidate"] == model and x["control"] == "baseline" and x["horizon"] == h) for h in [1, 5]]
        lines.append(f"| {model} | {r[0]['improvement_pct']:+.2f}% | {r[1]['improvement_pct']:+.2f}% | {r[0]['p_holm']:.4f} | {r[1]['p_holm']:.4f} |")
    lines += ["", "## Mechanism controls", "",
        "Memory must beat global recalibration. Latent versus observable retrieval isolates the representation.",
        "Dynamic weighting must beat equal weighting and the best component point estimate before an ensemble claim.", "",
        "| candidate vs control | h | improvement | adjusted p | verdict |", "|---|---:|---:|---:|---|"]
    for r in rows:
        if r["role"] == "mechanism":
            lines.append(f"| {r['candidate']} vs {r['control']} | {r['horizon']} | {r['improvement_pct']:+.2f}% | {r['p_holm']:.4f} | {r['verdict']} |")
    lines += ["", "## Uncertainty and stability", "",
        "Candidate-minus-control gaps below: negative favors candidate. Intervals envelope block21/63/126 and HAC126;",
        "they are nominal and not simultaneous. Adjusted p-values use Holm over all 32 core comparisons; the reference extension also reports the complete 46-comparison family.", "",
        "| comparison | h | gap | 95% envelope | 2016–19 | 2020–22 | 2023–25 | verdict |",
        "|---|---:|---:|---|---:|---:|---:|---|"]
    for r in rows:
        lo, hi = r["ci95_envelope"]
        period = " | ".join(f"{x['delta']:+.5f}" for x in r["periods"])
        lines.append(f"| {r['candidate']} vs {r['control']} | {r['horizon']} | {r['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {period} | {r['verdict']} |")
    lines += ["", "## Memory and ensemble audit", "",
        f"- Memory records per query: {audit.pool_n.min()}–{audit.pool_n.max()}; exactly 64 neighbors per retrieval arm.",
        "- Memory target must finish at least 22 QQQ sessions before query; lookback 1260 sessions. Global and local corrections use identical pools and 25% shrinkage.",
        "- Monthly observable scaling and eight-dimensional PCA use only records eligible at the month's first scoring origin. No whitening; no future-fitted geometry.",
        "- Ensemble weights use completed expert losses only: 252-session lookback, 63-session half-life, 10% equal-weight floor.",
        "- Historical forecasts begin 2013-02-01 to warm up the memory. Scoring 2016-01-04–2025-10-10, targets at or before 2025-10-20.",
        "- Same-history inputs do not create new external information; pretrained latent-cache exposure remains unknown.",
        "- The target is daily GK-plus-overnight variance, not measured five-minute RV. No economic-return or equivalence claim is made.",
        "- All annual, phase, MDE, component-loss and inference details are in metrics.json. Neighbor traces and fits are local under data/model_memory_study.",
        "- Reproduce: make model-memory-study; independent check: make verify-model-memory-study.", ""]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "results.md").write_text("\n".join(lines))


def check_contracts():
    tests = ["tests.test_model_memory_estimators", "tests.test_model_memory_study",
             "tests.test_verify_model_memory_study", "tests.test_round2_inference",
             "tests.test_methodology", "tests.test_signal_safety"]
    result = subprocess.run([sys.executable, "-m", "unittest", *tests, "-q"],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "pre_run_checks.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    print(result.stderr.strip(), flush=True)


def run():
    p = yaml.safe_load(PROTOCOL.read_text())
    validate_protocol(p)
    check_contracts()
    inputs = {v: parent.digest(ROOT / v) for v in p["inputs"].values()}
    code = {f: parent.digest(ROOT / f) for f in ["src/model_memory_study.py", "src/model_memory_estimators.py",
        "tests/test_model_memory_study.py", "tests/test_model_memory_estimators.py",
        "src/verify_model_memory_study.py", "tests/test_verify_model_memory_study.py"]}
    protected = [f for f in list(ROOT.glob("*.yaml")) + list((ROOT / "reports").rglob("*"))
                 if f.is_file() and "model_memory" not in str(f)]
    frozen = {str(f.relative_to(ROOT)): parent.digest(f) for f in protected}
    manifest = {"created_utc": datetime.now(UTC).isoformat(), "protocol_sha256": parent.digest(PROTOCOL),
                "inputs": inputs, "code": code, "existing_artifacts_sha256": frozen,
                "evidence_class": p["evidence_class"]}
    path = REPORT / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        for key in ["inputs", "code", "protocol_sha256"]:
            if old[key] != manifest[key]:
                raise ValueError(f"Previous empirical run differs in {key}; do not overwrite")
    parent.dump(path, manifest)
    features, z = load_inputs(p)
    OUT.mkdir(parents=True, exist_ok=True)
    components, fits = component_forecasts(features, z, p)
    components.to_parquet(OUT / "components.parquet", index=False)
    parent.dump(OUT / "estimator_fits.json", fits)
    forecasts, audit, neighbors = memory_forecasts(features, z, components, p)
    forecasts.to_parquet(OUT / "forecasts.parquet", index=False)
    audit.to_parquet(OUT / "memory_audit.parquet", index=False)
    neighbors.to_parquet(OUT / "neighbors.parquet", index=False)
    metrics = evaluate(forecasts, features, p)
    for f, expected in frozen.items():
        if parent.digest(ROOT / f) != expected:
            raise ValueError(f"Existing artifact changed: {f}")
    parent.dump(REPORT / "metrics.json", metrics)
    report(metrics, audit)
    print(json.dumps([{k: r[k] for k in ("candidate", "control", "horizon", "improvement_pct", "p_holm", "verdict")}
                      for r in metrics["rows"] if r["role"] == "primary"], indent=2), flush=True)


if __name__ == "__main__":
    run()
