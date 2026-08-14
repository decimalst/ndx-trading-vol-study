"""Frozen GBM functional-form study on the HAR-IV information set.

The workflow is deliberately two-stage::

    python -m src.gbm_study discover
    python -m src.gbm_study confirm

``discover`` scores only the 2016-2019 origins and writes an interaction lock.
``confirm`` refuses to run without a lock matching the frozen protocol, scores
2020-2025, and renders the final diagnostic report.  No clean-phase NDX origin
is read by either command.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from . import envcheck

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "gbm_study.yaml"
FEATURES = ("lrv_d", "lrv_w", "lrv_m", "liv")


def load_protocol(path: str | pathlib.Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    p = pathlib.Path(path)
    with p.open() as handle:
        spec = yaml.safe_load(handle)
    if tuple(spec["information_set"]["feature_order"]) != FEATURES:
        raise ValueError("protocol information set does not match frozen HAR-IV features")
    return spec


def protocol_sha256(path: str | pathlib.Path = DEFAULT_PROTOCOL) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def resolve_repo_path(value: str) -> pathlib.Path:
    return ROOT / value


def build_design(master: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the four-column HAR-IV design and its one-session target."""
    if not master.index.is_monotonic_increasing or master.index.has_duplicates:
        raise ValueError("master index must be unique and increasing")
    out = pd.DataFrame(index=pd.DatetimeIndex(master.index))
    out["lrv_d"] = np.log(master["rv_total"])
    out["lrv_w"] = np.log(master["rv_total"].rolling(5).mean())
    out["lrv_m"] = np.log(master["rv_total"].rolling(22).mean())
    out["liv"] = np.log(master["vxn"])
    out["y_next"] = np.log(master["rv_total"]).shift(-1)
    out["actual_var"] = master["rv_total"].shift(-1)
    dates = pd.Series(master.index, index=master.index)
    out["target_date"] = dates.shift(-1)
    return out


def complete_rows(design: pd.DataFrame) -> pd.DataFrame:
    return design.dropna(subset=[*FEATURES, "y_next", "actual_var", "target_date"])


def split_origins(design: pd.DataFrame, spec: dict[str, Any], split: str) -> pd.DatetimeIndex:
    if split == "all_diagnostic":
        lo = spec["fences"]["diagnostic_start"]
        hi = spec["fences"]["diagnostic_end"]
    else:
        lo = spec["splits"][split]["start"]
        hi = spec["splits"][split]["end"]
    rows = complete_rows(design)
    idx = rows.index[(rows.index >= pd.Timestamp(lo)) & (rows.index <= pd.Timestamp(hi))]
    assert_leakage_fences(rows.loc[idx], spec)
    return pd.DatetimeIndex(idx)


def assert_leakage_fences(rows: pd.DataFrame, spec: dict[str, Any]) -> None:
    if rows.empty:
        raise ValueError("score sample is empty")
    clean = pd.Timestamp(spec["fences"]["sealed_clean_start"])
    latest_target = pd.Timestamp(spec["fences"]["latest_allowed_target_date"])
    if rows.index.max() >= clean:
        raise ValueError("sealed NDX clean origin entered GBM study")
    target_dates = pd.to_datetime(rows["target_date"])
    if target_dates.max() > latest_target or target_dates.max() >= clean:
        raise ValueError("sealed NDX clean target entered GBM study")
    if not np.all(target_dates.values > rows.index.values):
        raise ValueError("target is not strictly after its origin")


def training_rows(design: pd.DataFrame, origin: pd.Timestamp,
                  min_rows: int) -> pd.DataFrame:
    """Rows whose one-session targets are known at ``origin``."""
    train = complete_rows(design)
    train = train.loc[train.index < pd.Timestamp(origin)]
    if len(train) < min_rows:
        raise ValueError(f"only {len(train)} training rows at {origin}; need {min_rows}")
    if not (train.index < pd.Timestamp(origin)).all():
        raise AssertionError("training target boundary failed")
    return train


def make_gbm(spec: dict[str, Any]):
    """Instantiate exactly the registered tree; importing sklearn is deferred."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    p = spec["estimators"]["gbm"]
    return HistGradientBoostingRegressor(
        loss=p["loss"],
        learning_rate=float(p["learning_rate"]),
        max_iter=int(p["max_iter"]),
        max_leaf_nodes=int(p["max_leaf_nodes"]),
        max_depth=int(p["max_depth"]),
        min_samples_leaf=int(p["min_samples_leaf"]),
        l2_regularization=float(p["l2_regularization"]),
        max_bins=int(p["max_bins"]),
        early_stopping=bool(p["early_stopping"]),
        random_state=int(p["random_state"]),
    )


def smearing_variance(log_prediction: float, residuals: np.ndarray) -> float:
    resid = np.asarray(residuals, dtype=float)
    resid = resid[np.isfinite(resid)]
    if not len(resid):
        raise ValueError("cannot smear without residuals")
    return float(np.exp(float(log_prediction)) * np.mean(np.exp(resid)))


def fit_ols(train: pd.DataFrame, x: np.ndarray,
            columns: Iterable[str]) -> tuple[float, float]:
    cols = list(columns)
    X = np.column_stack([np.ones(len(train)), train[cols].to_numpy(float)])
    y = train["y_next"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xrow = np.r_[1.0, np.asarray(x, dtype=float)]
    mu = float(xrow @ beta)
    return mu, smearing_variance(mu, resid)


def apply_locked_term(frame: pd.DataFrame, lock: dict[str, Any]) -> np.ndarray:
    term = lock["locked_term"]
    values: list[np.ndarray] = []
    for item in term["features"]:
        raw = frame[item["name"]].to_numpy(float)
        hinge = np.maximum(
            float(item["direction"]) * (raw - float(item["threshold"]))
            / float(item["scale"]),
            0.0,
        )
        values.append(hinge)
    return values[0] * values[1]


def forecast_origins(design: pd.DataFrame, origins: pd.DatetimeIndex,
                     spec: dict[str, Any],
                     lock: dict[str, Any] | None = None) -> pd.DataFrame:
    """Expanding, every-origin HAR-IV and fixed-GBM forecasts."""
    rows: list[dict[str, Any]] = []
    min_rows = int(spec["fences"]["min_train_rows"])
    for n, origin in enumerate(origins, start=1):
        train = training_rows(design, origin, min_rows)
        x = design.loc[origin, list(FEATURES)].to_numpy(float)

        har_mu, har_var = fit_ols(train, x, FEATURES)

        model = make_gbm(spec)
        X_train = train.loc[:, FEATURES].to_numpy(float)
        y_train = train["y_next"].to_numpy(float)
        model.fit(X_train, y_train)
        gbm_mu = float(model.predict(x.reshape(1, -1))[0])
        gbm_resid = y_train - model.predict(X_train)
        gbm_var = smearing_variance(gbm_mu, gbm_resid)

        aug_mu = np.nan
        aug_var = np.nan
        if lock is not None:
            aug_train = train.copy()
            aug_train["locked_interaction"] = apply_locked_term(aug_train, lock)
            x_frame = pd.DataFrame([dict(zip(FEATURES, x))])
            x_aug = np.r_[x, apply_locked_term(x_frame, lock)[0]]
            aug_mu, aug_var = fit_ols(
                aug_train, x_aug, [*FEATURES, "locked_interaction"]
            )

        rows.append({
            "origin": origin,
            "target_date": pd.Timestamp(design.loc[origin, "target_date"]),
            "actual_var": float(design.loc[origin, "actual_var"]),
            "actual_log_var": float(design.loc[origin, "y_next"]),
            "har_iv_log": har_mu,
            "har_iv_var": har_var,
            "gbm_log": gbm_mu,
            "gbm_var": gbm_var,
            "augmented_log": aug_mu,
            "augmented_var": aug_var,
        })
        if n % 100 == 0 or n == len(origins):
            print(f"forecasted {n}/{len(origins)} origins through {origin.date()}", flush=True)
    out = pd.DataFrame(rows).set_index("origin")
    return out


def _partial_dependence_surface(model, background: pd.DataFrame,
                                feature_a: str, feature_b: str,
                                quantiles: np.ndarray) -> dict[str, Any]:
    ia, ib = FEATURES.index(feature_a), FEATURES.index(feature_b)
    base = background.loc[:, FEATURES].to_numpy(float)
    grid_a = np.quantile(base[:, ia], quantiles)
    grid_b = np.quantile(base[:, ib], quantiles)
    surface = np.empty((len(grid_a), len(grid_b)), dtype=float)
    work = base.copy()
    for i, a in enumerate(grid_a):
        for j, b in enumerate(grid_b):
            work[:, :] = base
            work[:, ia] = a
            work[:, ib] = b
            surface[i, j] = float(np.mean(model.predict(work)))
    grand = float(surface.mean())
    interaction = surface - surface.mean(axis=1, keepdims=True) \
        - surface.mean(axis=0, keepdims=True) + grand
    centered = surface - grand
    denominator = float(np.sum(centered ** 2))
    score = float(np.sum(interaction ** 2) / denominator) if denominator > 0 else 0.0
    loc_flat = int(np.argmax(np.abs(interaction)))
    loc = np.unravel_index(loc_flat, interaction.shape)
    return {
        "pair": [feature_a, feature_b],
        "score": score,
        "grid_a": grid_a.tolist(),
        "grid_b": grid_b.tolist(),
        "surface": surface.tolist(),
        "interaction": interaction.tolist(),
        "location": [int(loc[0]), int(loc[1])],
        "location_value": float(interaction[loc]),
    }


def exact_interaction_probe(model, background: pd.DataFrame,
                            spec: dict[str, Any]) -> dict[str, Any]:
    """Exact registered fallback for locating tree interactions."""
    grid = np.asarray(spec["interaction_probe"]["grid_quantiles"], dtype=float)
    pair_results = []
    for a, b in itertools.combinations(FEATURES, 2):
        pair_results.append(_partial_dependence_surface(model, background, a, b, grid))
    # Stable sort preserves feature-order enumeration on a numerical tie.
    selected = sorted(pair_results, key=lambda item: -item["score"])[0]
    a, b = selected["pair"]
    i, j = selected["location"]
    thresholds = [selected["grid_a"][i], selected["grid_b"][j]]
    term_features = []
    for name, threshold in zip((a, b), thresholds):
        series = background[name].to_numpy(float)
        median = float(np.median(series))
        q25, q75 = np.quantile(series, [0.25, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.std(series))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        term_features.append({
            "name": name,
            "threshold": float(threshold),
            "direction": 1 if threshold >= median else -1,
            "scale": scale,
            "discovery_median": median,
        })
    return {
        "method": spec["interaction_probe"]["method"],
        "pair_scores": pair_results,
        "selected_pair": selected["pair"],
        "selected_score": selected["score"],
        "selected_location": selected["location"],
        "selected_interaction_value": selected["location_value"],
        "locked_term": {
            "kind": "product_of_one_sided_iqr_scaled_hinges",
            "features": term_features,
        },
    }


def discover_interaction(design: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    snapshot = pd.Timestamp(spec["interaction_probe"]["snapshot_origin"])
    train = training_rows(design, snapshot, int(spec["fences"]["min_train_rows"]))
    model = make_gbm(spec)
    model.fit(train.loc[:, FEATURES].to_numpy(float), train["y_next"].to_numpy(float))
    lo = pd.Timestamp(spec["splits"]["discovery"]["start"])
    hi = pd.Timestamp(spec["splits"]["discovery"]["end"])
    background = complete_rows(design).loc[lo:hi, list(FEATURES)]
    result = exact_interaction_probe(model, background, spec)
    result.update({
        "protocol_sha256": protocol_sha256(),
        "snapshot_origin": snapshot.isoformat(),
        "fit_last_row": train.index.max().isoformat(),
        "fit_rows": int(len(train)),
        "background_start": background.index.min().isoformat(),
        "background_end": background.index.max().isoformat(),
        "background_rows": int(len(background)),
        "feature_order": list(FEATURES),
        "confirmation_results_read": False,
    })
    return result


def qlike(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if np.any(actual <= 0) or np.any(forecast <= 0):
        raise ValueError("QLIKE requires positive actuals and forecasts")
    ratio = actual / forecast
    return ratio - np.log(ratio) - 1.0


def dm_h1(candidate_loss: np.ndarray, baseline_loss: np.ndarray) -> dict[str, float | int]:
    diff = np.asarray(candidate_loss, dtype=float) - np.asarray(baseline_loss, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 10:
        return {"statistic": np.nan, "p_value": np.nan, "n": n}
    variance = float(np.var(diff, ddof=0))
    statistic = float(np.mean(diff) / np.sqrt(variance / n)) if variance > 0 else 0.0
    harvey = math.sqrt((n - 1) / n)
    statistic *= harvey
    return {
        "statistic": statistic,
        "p_value": float(2 * stats.t.sf(abs(statistic), df=n - 1)),
        "n": n,
    }


def moving_block_means(values: np.ndarray, block: int, draws: int,
                       seed: int, center: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        raise ValueError("empty bootstrap input")
    if center:
        values = values - values.mean()
    block = min(int(block), n)
    starts = np.arange(n - block + 1)
    blocks_needed = int(math.ceil(n / block))
    rng = np.random.default_rng(seed)
    output = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        picked = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[s:s + block] for s in picked])[:n]
        output[draw] = float(sample.mean())
    return output


def paired_comparison(frame: pd.DataFrame, candidate_col: str,
                      spec: dict[str, Any], seed_offset: int = 0) -> dict[str, Any]:
    use = frame.dropna(subset=["actual_var", "har_iv_var", candidate_col])
    actual = use["actual_var"].to_numpy(float)
    base_loss = qlike(actual, use["har_iv_var"].to_numpy(float))
    candidate_loss = qlike(actual, use[candidate_col].to_numpy(float))
    diff = candidate_loss - base_loss
    boot_spec = spec["scoreboard"]["robust_inference"]
    seed = int(boot_spec["seed"]) + int(seed_offset)
    boot = moving_block_means(
        diff, int(boot_spec["block_sessions"]), int(boot_spec["draws"]), seed
    )
    null_boot = moving_block_means(
        diff, int(boot_spec["block_sessions"]), int(boot_spec["draws"]), seed,
        center=True,
    )
    observed = float(diff.mean())
    two_sided_p = float((1 + np.sum(np.abs(null_boot) >= abs(observed)))
                        / (len(null_boot) + 1))
    ci95 = np.quantile(boot, [0.025, 0.975])
    ci90 = np.quantile(boot, [0.05, 0.95])
    base_mean = float(base_loss.mean())
    candidate_mean = float(candidate_loss.mean())
    margin = 0.03 * base_mean
    dm = dm_h1(candidate_loss, base_loss)
    return {
        "n": int(len(use)),
        "baseline_qlike": base_mean,
        "candidate_qlike": candidate_mean,
        "mean_difference": observed,
        "improvement_pct": float(100 * (base_mean - candidate_mean) / base_mean),
        "win_count": int(np.sum(candidate_loss < base_loss)),
        "win_rate": float(np.mean(candidate_loss < base_loss)),
        "dm_statistic": dm["statistic"],
        "dm_p_value": dm["p_value"],
        "block_bootstrap_p_value": two_sided_p,
        "block_ci95": [float(ci95[0]), float(ci95[1])],
        "block_ci90": [float(ci90[0]), float(ci90[1])],
        "equivalence_margin": margin,
        "equivalent_3pct": bool(ci90[0] > -margin and ci90[1] < margin),
    }


def _frame_for_split(frame: pd.DataFrame, spec: dict[str, Any], split: str) -> pd.DataFrame:
    if split == "all_diagnostic":
        lo, hi = spec["fences"]["diagnostic_start"], spec["fences"]["diagnostic_end"]
    else:
        lo, hi = spec["splits"][split]["start"], spec["splits"][split]["end"]
    return frame.loc[pd.Timestamp(lo):pd.Timestamp(hi)]


def calculate_metrics(forecasts: pd.DataFrame, lock: dict[str, Any],
                      spec: dict[str, Any]) -> dict[str, Any]:
    gbm = {}
    for offset, split in enumerate(("all_diagnostic", "discovery", "confirmation")):
        gbm[split] = paired_comparison(
            _frame_for_split(forecasts, spec, split), "gbm_var", spec, offset
        )
    confirmation = _frame_for_split(forecasts, spec, "confirmation")
    augmented = paired_comparison(confirmation, "augmented_var", spec, 10)
    primary = gbm["confirmation"]
    if primary["mean_difference"] < 0 and primary["block_bootstrap_p_value"] < 0.05:
        gbm_verdict = "WIN"
    elif primary["equivalent_3pct"]:
        gbm_verdict = "EQUIVALENT_WITHIN_3PCT"
    else:
        gbm_verdict = "INCONCLUSIVE"
    augmented_verdict = (
        "PASS" if augmented["mean_difference"] < 0
        and augmented["block_bootstrap_p_value"] < 0.05 else "DOES_NOT_ADD"
    )
    return {
        "study_id": spec["study_id"],
        "protocol_sha256": protocol_sha256(),
        "feature_order": list(FEATURES),
        "interaction_method": lock["method"],
        "selected_pair": lock["selected_pair"],
        "locked_term": lock["locked_term"],
        "sample": {
            "all_start": forecasts.index.min().isoformat(),
            "all_end": forecasts.index.max().isoformat(),
            "all_n": int(len(forecasts)),
            "last_target_date": pd.Timestamp(forecasts["target_date"].max()).isoformat(),
        },
        "gbm": gbm,
        "augmented_term": {"confirmation": augmented},
        "verdicts": {
            "gbm_functional_form": gbm_verdict,
            "locked_interpretable_term": augmented_verdict,
        },
    }


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def render_report(metrics: dict[str, Any], lock: dict[str, Any]) -> str:
    selected = lock["locked_term"]["features"]
    term_text = " × ".join(
        f"max({int(x['direction']):+d}·({x['name']} − {x['threshold']:.6f})/{x['scale']:.6f}, 0)"
        for x in selected
    )
    lines = [
        "# GBM functional-form study",
        "",
        "**Evidence class: internally split diagnostic study; sealed NDX clean origins were not read.**",
        "",
        "Timing qualification: both models use the same-session published Cboe VXN close, so the functional-form comparison is internally fair. However, that close may contain 16:00–16:15 ET information after the repository's standing 16:00 origin. Treat this frozen run as timing-ambiguous for a strict 16:00 forecast; the separately registered lagged-VXN sensitivity determines whether the substantive result survives a fully known-at-origin input.",
        "",
        f"Protocol SHA-256: `{metrics['protocol_sha256']}`.",
        "",
        "## Same-information-set forecast comparison",
        "",
        "| split | n | HAR-IV QLIKE | GBM QLIKE | improvement | DM p | block p | 95% block interval | win rate | equivalent within 3% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for split in ("all_diagnostic", "discovery", "confirmation"):
        x = metrics["gbm"][split]
        lines.append(
            f"| {split} | {x['n']} | {_fmt(x['baseline_qlike'])} | "
            f"{_fmt(x['candidate_qlike'])} | {x['improvement_pct']:+.3f}% | "
            f"{x['dm_p_value']:.4g} | {x['block_bootstrap_p_value']:.4g} | "
            f"[{x['block_ci95'][0]:+.6f}, {x['block_ci95'][1]:+.6f}] | "
            f"{100*x['win_rate']:.1f}% | {'yes' if x['equivalent_3pct'] else 'no'} |"
        )
    lines += [
        "",
        f"Frozen functional-form verdict: **{metrics['verdicts']['gbm_functional_form']}**.",
        "",
        "Negative mean differences favor the candidate. The block p-value and interval use paired 21-session moving blocks; DM is h=1 and is reported for continuity with the existing harness.",
        "",
        "## Exact interaction fallback",
        "",
        "SHAP was not installed before the protocol freeze. The registered fallback exactly double-centers each fitted two-feature partial-dependence surface on the discovery quantile grid.",
        "",
        "| pair | interaction score | selected |",
        "|---|---:|:---:|",
    ]
    for item in lock["pair_scores"]:
        pair = " × ".join(item["pair"])
        lines.append(
            f"| {pair} | {item['score']:.6f} | "
            f"{'yes' if item['pair'] == lock['selected_pair'] else 'no'} |"
        )
    lines += [
        "",
        f"Locked term: `{term_text}`.",
        "",
        "The pair, location, directions, thresholds, and IQR scales were written to the interaction lock before any confirmation forecast was computed.",
        "",
        "## Interpretable-term confirmation",
        "",
        "| n | HAR-IV QLIKE | HAR-IV + locked term QLIKE | improvement | DM p | block p | 95% block interval | win rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    a = metrics["augmented_term"]["confirmation"]
    lines += [
        f"| {a['n']} | {_fmt(a['baseline_qlike'])} | {_fmt(a['candidate_qlike'])} | "
        f"{a['improvement_pct']:+.3f}% | {a['dm_p_value']:.4g} | "
        f"{a['block_bootstrap_p_value']:.4g} | "
        f"[{a['block_ci95'][0]:+.6f}, {a['block_ci95'][1]:+.6f}] | "
        f"{100*a['win_rate']:.1f}% |",
        "",
        f"Frozen interpretable-term verdict: **{metrics['verdicts']['locked_interpretable_term']}**.",
        "",
        "## Scope",
        "",
        "This closes only the fixed HAR-IV functional-form question on the already-open diagnostic history. It does not test additional signals, optimize a GBM, or constitute evidence from the sealed clean phase.",
        "",
    ]
    return "\n".join(lines)


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_fenced_master(spec: dict[str, Any]) -> pd.DataFrame:
    """Read only rows through the last permitted target at parquet scan time."""
    latest = pd.Timestamp(spec["fences"]["latest_allowed_target_date"])
    frame = pd.read_parquet(
        resolve_repo_path(spec["fences"]["source"]),
        filters=[("date", "<=", latest)],
    )
    if frame.index.max() > latest:
        raise ValueError("parquet predicate admitted a post-fence row")
    return frame


def run_discovery(spec: dict[str, Any]) -> None:
    data_dir = resolve_repo_path(spec["outputs"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    design = build_design(load_fenced_master(spec))
    origins = split_origins(design, spec, "discovery")
    forecasts = forecast_origins(design, origins, spec)
    forecasts["phase"] = "discovery"
    lock = discover_interaction(design, spec)
    lock_path = resolve_repo_path(spec["outputs"]["interactions"])
    if lock_path.exists():
        old = json.loads(lock_path.read_text())
        if old != lock:
            raise RuntimeError("existing interaction lock differs; refusing to overwrite")
    forecasts.to_parquet(data_dir / "discovery_forecasts.parquet")
    _write_json(lock_path, lock)
    print(f"wrote discovery forecasts ({len(forecasts)}) and frozen interaction lock")


def validate_lock(lock: dict[str, Any], spec: dict[str, Any]) -> None:
    if lock.get("protocol_sha256") != protocol_sha256():
        raise ValueError("interaction lock does not match frozen protocol")
    if tuple(lock.get("feature_order", [])) != FEATURES:
        raise ValueError("interaction lock feature order changed")
    if lock.get("confirmation_results_read") is not False:
        raise ValueError("interaction lock was not created before confirmation")
    if len(lock.get("pair_scores", [])) != 6:
        raise ValueError("interaction lock must contain all six feature pairs")


def validate_artifacts(forecasts: pd.DataFrame, metrics: dict[str, Any],
                       lock: dict[str, Any], spec: dict[str, Any]) -> None:
    validate_lock(lock, spec)
    required = {
        "target_date", "actual_var", "har_iv_var", "gbm_var", "augmented_var", "phase"
    }
    if not required.issubset(forecasts.columns):
        raise ValueError(f"forecast artifact missing {sorted(required - set(forecasts.columns))}")
    assert_leakage_fences(forecasts, spec)
    expected = split_origins(build_design(load_fenced_master(spec)), spec, "all_diagnostic")
    if not forecasts.index.equals(expected):
        raise ValueError("forecast artifact is not the exact diagnostic common sample")
    discovery = forecasts["phase"] == "discovery"
    confirmation = forecasts["phase"] == "confirmation"
    if forecasts.loc[discovery, "augmented_var"].notna().any():
        raise ValueError("locked term leaked into discovery forecasts")
    if forecasts.loc[confirmation, "augmented_var"].isna().any():
        raise ValueError("confirmation augmented forecasts are incomplete")
    if metrics.get("protocol_sha256") != protocol_sha256():
        raise ValueError("metrics do not match frozen protocol")
    if metrics["sample"]["all_n"] != len(forecasts):
        raise ValueError("metrics sample count does not match forecasts")


def run_confirmation(spec: dict[str, Any]) -> None:
    outputs = spec["outputs"]
    data_dir = resolve_repo_path(outputs["data_dir"])
    discovery_path = data_dir / "discovery_forecasts.parquet"
    lock_path = resolve_repo_path(outputs["interactions"])
    if not discovery_path.exists() or not lock_path.exists():
        raise FileNotFoundError("run discover and freeze its lock before confirm")
    discovery = pd.read_parquet(discovery_path)
    lock = json.loads(lock_path.read_text())
    validate_lock(lock, spec)

    design = build_design(load_fenced_master(spec))
    origins = split_origins(design, spec, "confirmation")
    confirmation = forecast_origins(design, origins, spec, lock=lock)
    confirmation["phase"] = "confirmation"
    forecasts = pd.concat([discovery, confirmation]).sort_index()
    if forecasts.index.has_duplicates:
        raise ValueError("discovery and confirmation origins overlap")
    metrics = calculate_metrics(forecasts, lock, spec)
    validate_artifacts(forecasts, metrics, lock, spec)

    confirmation.to_parquet(data_dir / "confirmation_forecasts.parquet")
    forecasts.to_parquet(resolve_repo_path(outputs["forecasts"]))
    _write_json(resolve_repo_path(outputs["metrics"]), metrics)
    report = render_report(metrics, lock)
    report_path = resolve_repo_path(outputs["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"wrote {outputs['forecasts']}, {outputs['metrics']}, and {outputs['report']}")


def verify_saved(spec: dict[str, Any]) -> None:
    outputs = spec["outputs"]
    forecasts = pd.read_parquet(resolve_repo_path(outputs["forecasts"]))
    metrics = json.loads(resolve_repo_path(outputs["metrics"]).read_text())
    lock = json.loads(resolve_repo_path(outputs["interactions"]).read_text())
    validate_artifacts(forecasts, metrics, lock, spec)
    recomputed = calculate_metrics(forecasts, lock, spec)
    if recomputed != metrics:
        raise ValueError("saved GBM metrics do not recompute from forecasts. "
                         + envcheck.pin_advice())
    rendered = render_report(metrics, lock)
    if resolve_repo_path(outputs["report"]).read_text() != rendered:
        raise ValueError("saved report does not match metrics")
    print("GBM STUDY VERIFICATION PASS")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("discover", "confirm", "verify"))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    args = parser.parse_args(argv)
    spec = load_protocol(args.protocol)
    if pathlib.Path(args.protocol).resolve() != DEFAULT_PROTOCOL.resolve():
        raise ValueError("empirical commands require the frozen repository protocol")
    if args.command == "discover":
        run_discovery(spec)
    elif args.command == "confirm":
        run_confirmation(spec)
    else:
        verify_saved(spec)


if __name__ == "__main__":
    main()
