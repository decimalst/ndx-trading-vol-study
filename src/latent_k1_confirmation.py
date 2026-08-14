"""Separate 99-control formal randomization diagnostic for sparse TiRex k=1.

The protocol in ``latent_k1_confirmation.yaml`` was frozen after the original
ten-control ladder had been inspected but before this run.  This module never
mutates that ladder or its artifacts.  Every annual coordinate is selected on
completed training labels and scored only on the existing held-out fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config
from . import latent_probe_study as latent
from . import representation_study as base

ROOT = config.ROOT
PROTOCOL_PATH = ROOT / "latent_k1_confirmation.yaml"
OUTPUT_DIR = ROOT / "data" / "representation_study"
REPORT_DIR = ROOT / "reports" / "representation_study"
FORECASTS_PATH = OUTPUT_DIR / "latent_k1_confirmation_forecasts.parquet"
CONTROLS_PATH = OUTPUT_DIR / "latent_k1_confirmation_controls.parquet"
SELECTIONS_PATH = OUTPUT_DIR / "latent_k1_confirmation_selections.parquet"
CORRELATIONS_PATH = OUTPUT_DIR / "latent_k1_coordinate_correlations.parquet"
PHASE_PATH = OUTPUT_DIR / "latent_k1_confirmation_phase_metrics.parquet"
METRICS_PATH = OUTPUT_DIR / "latent_k1_confirmation_metrics.json"
VERIFICATION_PATH = OUTPUT_DIR / "latent_k1_confirmation_verification.json"
REPORT_PATH = REPORT_DIR / "latent_k1_confirmation.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(relative: str) -> Path:
    return ROOT / relative


def control_seeds(protocol: dict) -> list[int]:
    cfg = protocol["control_task"]["seeds"]
    return list(range(int(cfg["start"]), int(cfg["stop_inclusive"]) + 1))


def load_protocol() -> dict:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    if protocol.get("status") != "frozen_before_99_control_run":
        raise ValueError("the 99-control protocol is not frozen")
    if protocol.get("evidence_role") != "post_result_formal_randomization_diagnostic":
        raise ValueError("post-result provenance changed")
    probe = protocol["probe"]
    if probe.get("rung") != "sparse_k1_only" or float(probe.get("ridge", np.nan)) != 1.0:
        raise ValueError("the single sparse-k=1 probe changed")
    if probe.get("input_dimensions") != "all_512_before_selection":
        raise ValueError("selection must start from all 512 coordinates")
    if not probe.get("forbid_pca") or not probe.get("forbid_test_label_selection"):
        raise ValueError("PCA and test-label selection must remain forbidden")
    seeds = control_seeds(protocol)
    control = protocol["control_task"]
    if int(control.get("draws", 0)) != 99 or seeds != list(range(4300, 4399)):
        raise ValueError("the 99 predetermined controls changed")
    if control.get("family") != "one_sparse_k1_rung_only":
        raise ValueError("the formal run must contain exactly one rung")
    if control.get("familywise_correction") != "none_single_registered_test":
        raise ValueError("single-test correction rule changed")
    if protocol["scoreboard"].get("primary") != "five_phase_unweighted_mean_auc":
        raise ValueError("phase-mean AUC must remain primary")
    if int(protocol["scoreboard"].get("phases", 0)) != 5:
        raise ValueError("the scoreboard must retain all five phases")
    if not protocol["sample"].get("forbid_clean_origins"):
        raise ValueError("the clean fence is mandatory")
    return protocol


def exact_randomization_test(
    actual: float, controls: Iterable[float], *, alpha: float
) -> dict[str, float | int | bool]:
    values = np.asarray(list(controls), dtype=float)
    if len(values) != 99 or not np.isfinite(values).all():
        raise ValueError("the formal randomization test requires exactly 99 finite controls")
    exceedances = int(np.sum(values >= float(actual)))
    exact_p = float((exceedances + 1) / (len(values) + 1))
    return {
        "draws": int(len(values)),
        "exceedances": exceedances,
        "exact_p": exact_p,
        "minimum_attainable_p": float(1 / (len(values) + 1)),
        "alpha": float(alpha),
        "formal_evidence": bool(exact_p <= float(alpha)),
    }


def _validate_binary_labels(labels: np.ndarray, rows: int) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    if y.ndim != 1 or len(y) != int(rows) or set(np.unique(y)) != {0, 1}:
        raise ValueError("selector needs aligned binary training labels with both classes")
    return y


def select_top1(X_train: np.ndarray, y_train: np.ndarray) -> tuple[int, float]:
    """Select on training data only; the interface deliberately has no test input."""
    X = np.asarray(X_train, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("selector needs a finite two-dimensional training matrix")
    y = _validate_binary_labels(y_train, len(X))
    effects = latent.standardized_mean_difference(X, y)
    dimension = int(np.argmax(np.abs(effects)))
    return dimension, float(effects[dimension])


def select_top1_many(
    X_train: np.ndarray, training_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized training-only selector for independently labelled controls."""
    X = np.asarray(X_train, dtype=float)
    Y = np.asarray(training_labels, dtype=int)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError("control selectors need aligned training matrices")
    if not np.isfinite(X).all() or not np.isin(Y, [0, 1]).all():
        raise ValueError("control selectors need finite inputs and binary labels")
    count_one = Y.sum(axis=0).astype(float)
    count_zero = len(Y) - count_one
    if (count_one == 0).any() or (count_zero == 0).any():
        raise ValueError("every control training path needs both classes")
    scale = X.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    sums_one = X.T @ Y
    sums_zero = X.sum(axis=0)[:, None] - sums_one
    effects = (sums_one / count_one - sums_zero / count_zero) / scale[:, None]
    selected = np.argmax(np.abs(effects), axis=0).astype(int)
    chosen_effects = effects[selected, np.arange(Y.shape[1])]
    return selected, np.asarray(chosen_effects, dtype=float)


def prior_session_vxn(vxn: pd.Series, origins: pd.DatetimeIndex) -> pd.Series:
    """Shift on the complete VXN source calendar, then reindex scored origins."""
    source = vxn.astype(float).sort_index()
    source.index = pd.DatetimeIndex(source.index).tz_localize(None).normalize()
    shifted = source.shift(1)
    result = shifted.reindex(pd.DatetimeIndex(origins))
    result.name = "prior_session_vxn_level"
    return result


def fold_correlations(
    frame: pd.DataFrame,
    *,
    train_index: pd.DatetimeIndex,
    fold_year: int,
    variables: Iterable[str],
    orientation: float = 1.0,
) -> pd.DataFrame:
    """Correlate only held-out values; report raw and training-effect orientation."""
    held_out = frame.copy()
    if len(pd.DatetimeIndex(held_out.index).intersection(pd.DatetimeIndex(train_index))):
        raise ValueError("training and held-out characterization rows overlap")
    if "coordinate" not in held_out:
        raise ValueError("characterization frame is missing its selected coordinate")
    if float(orientation) not in (-1.0, 1.0):
        raise ValueError("orientation must be the sign of the training event effect")
    rows = []
    for variable in variables:
        if variable not in held_out:
            raise ValueError(f"characterization variable is missing: {variable}")
        sample = held_out[["coordinate", variable]].dropna()
        if len(sample) < 3 or sample.nunique().min() < 2:
            raw_pearson = raw_spearman = float("nan")
        else:
            raw_pearson = float(sample["coordinate"].corr(sample[variable], method="pearson"))
            raw_spearman = float(sample["coordinate"].corr(sample[variable], method="spearman"))
        rows.append(
            {
                "fold_year": int(fold_year),
                "variable": variable,
                "n": int(len(sample)),
                "pearson": raw_pearson,
                "spearman": raw_spearman,
                "event_oriented_pearson": float(orientation * raw_pearson),
                "event_oriented_spearman": float(orientation * raw_spearman),
            }
        )
    return pd.DataFrame(rows)


def _validate_hashes(protocol: dict) -> dict[str, str]:
    observed: dict[str, str] = {}
    mapping = {
        "representation_protocol": ("path", "expected_sha256"),
        "embeddings": ("path", "expected_sha256"),
        "classical_forecasts": ("path", "expected_sha256"),
        "history_panel": ("path", "expected_sha256"),
        "vxn": ("path", "expected_sha256"),
    }
    for name, (path_key, hash_key) in mapping.items():
        item = protocol["inputs"][name]
        path = _path(item[path_key])
        digest = _sha256(path)
        if digest != item[hash_key]:
            raise RuntimeError(f"frozen input hash changed: {name}")
        observed[name] = digest
    manifest_cfg = protocol["inputs"]["embeddings"]
    manifest_path = _path(manifest_cfg["manifest_path"])
    manifest_digest = _sha256(manifest_path)
    if manifest_digest != manifest_cfg["manifest_expected_sha256"]:
        raise RuntimeError("frozen embedding manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("output_sha256") != observed["embeddings"]:
        raise RuntimeError("embedding manifest no longer binds the embedding matrix")
    if manifest.get("checkpoint_revision") != manifest_cfg["checkpoint_revision"]:
        raise RuntimeError("embedding checkpoint revision changed")
    observed["embeddings_manifest"] = manifest_digest
    return observed


def _load_inputs(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    hashes = _validate_hashes(protocol)
    original_protocol = latent.load_protocol()
    embeddings = latent.load_embeddings(original_protocol)
    classical = pd.read_parquet(_path(protocol["inputs"]["classical_forecasts"]["path"])).sort_index()
    history = pd.read_parquet(_path(protocol["inputs"]["history_panel"]["path"])).sort_index()
    vxn_frame = pd.read_parquet(_path(protocol["inputs"]["vxn"]["path"])).sort_index()
    vxn_column = "close" if "close" in vxn_frame else vxn_frame.columns[-1]
    vxn = vxn_frame[vxn_column].astype(float)
    expected = protocol["inputs"]
    if embeddings.shape != (
        int(expected["embeddings"]["expected_rows"]),
        int(expected["embeddings"]["expected_dimensions"]),
    ):
        raise RuntimeError("embedding shape changed")
    if len(classical) != int(expected["classical_forecasts"]["expected_rows"]):
        raise RuntimeError("classical common-row count changed")
    folds = sorted(classical["fold_year"].astype(int).unique().tolist())
    if folds != list(
        range(
            int(protocol["sample"]["first_scored_year"]),
            int(protocol["sample"]["final_scored_year"]) + 1,
        )
    ) or len(folds) != int(protocol["sample"]["expected_folds"]):
        raise RuntimeError("the existing annual forward-fold set changed")
    if classical.index.has_duplicates or not classical.index.is_monotonic_increasing:
        raise RuntimeError("classical origins must be unique and chronological")
    clean = pd.Timestamp(protocol["sample"]["clean_start"])
    if (classical.index >= clean).any():
        raise RuntimeError("the formal diagnostic entered the sealed clean window")
    return embeddings, classical, history, vxn, hashes


def _characterization_features(
    history: pd.DataFrame, vxn: pd.Series, origins: pd.DatetimeIndex
) -> pd.DataFrame:
    frame = pd.DataFrame(index=pd.DatetimeIndex(history.index))
    frame["log_rv_level"] = history["log_rv"].astype(float)
    frame["trailing_5_session_mean_log_rv"] = frame["log_rv_level"].rolling(5).mean()
    frame["trailing_22_session_mean_log_rv"] = frame["log_rv_level"].rolling(22).mean()
    returns = history["ret_cc"].astype(float)
    frame["trailing_1_session_return"] = returns
    frame["trailing_5_session_return"] = returns.rolling(5).sum()
    frame["trailing_22_session_return"] = returns.rolling(22).sum()
    result = frame.reindex(origins)
    result["prior_session_vxn_level"] = prior_session_vxn(vxn, origins)
    return result


def _phase_summary(frame: pd.DataFrame, score: str, protocol: dict, *, event: str = "event") -> dict:
    phases = latent.phase_ranking_metrics(
        frame,
        score_column=score,
        event_column=event,
        horizon=int(protocol["scoreboard"]["phases"]),
        top_fraction=float(protocol["scoreboard"]["top_fraction"]),
    )
    mean = latent.phase_mean(phases)
    return {**mean, "dispersion": latent.phase_dispersion(phases)}


def _correlation_summary(correlations: pd.DataFrame) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for variable, group in correlations.groupby("variable", sort=False):
        item: dict[str, object] = {
            "folds": int(len(group)),
            "valid_folds": int(group["event_oriented_pearson"].notna().sum()),
            "observations": int(group["n"].sum()),
        }
        for metric in ("event_oriented_pearson", "event_oriented_spearman"):
            values = group[metric].dropna().astype(float)
            item[metric] = {
                "median": float(values.median()) if len(values) else None,
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
            }
        result[str(variable)] = item
    return result


def _selection_stability(actual_selections: pd.DataFrame) -> dict:
    counts = Counter(actual_selections["dimension"].astype(int))
    folds = len(actual_selections)
    return {
        "folds": int(folds),
        "unique_coordinates": int(len(counts)),
        "top_coordinates": [
            {"coordinate": int(coordinate), "folds": int(count), "fraction": float(count / folds)}
            for coordinate, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "interpretation": "annual training folds select multiple coordinates; this is not one universal neuron",
    }


def render_report(metrics: dict, selections: pd.DataFrame, correlations: pd.DataFrame) -> str:
    actual = metrics["actual"]
    control = metrics["controls"]
    randomization = metrics["randomization_test"]
    verdict = "FORMAL SELECTIVITY EVIDENCE" if randomization["formal_evidence"] else "NO FORMAL SELECTIVITY EVIDENCE"
    lines = [
        "# TiRex sparse-k=1: separate 99-control diagnostic",
        "",
        f"**Post-result formal-randomization verdict: {verdict}.** This is not a pristine historical holdout: the single k=1 rung was registered after the earlier ten-control ladder had been seen. It does not change that frozen ladder or its verdict.",
        "",
        "## Registered result",
        "",
        f"- Held-out origins: **{metrics['origins']:,}** across **{metrics['folds']}** annual forward folds; latest origin **{metrics['last_origin']}**, before the sealed clean window.",
        f"- Actual sparse-k=1 phase-mean AUC: **{actual['auc']:.4f}**; top-decile lift: **{actual['top_decile_lift']:.2f}x**.",
        f"- 99-control AUC median / 95th percentile: **{control['auc_median']:.4f} / {control['auc_95th_percentile']:.4f}**.",
        f"- Corrected exact randomization p: **{randomization['exact_p']:.4f}** ({randomization['exceedances']} controls at least as high as actual; attainable minimum **{randomization['minimum_attainable_p']:.2f}**).",
        f"- Actual AUC phase range: **{actual['dispersion']['auc']['min']:.4f}-{actual['dispersion']['auc']['max']:.4f}**; lift range: **{actual['dispersion']['top_decile_lift']['min']:.2f}x-{actual['dispersion']['top_decile_lift']['max']:.2f}x**.",
        "",
        "The exact p-value is the registered inference. There is one registered rung, so no ladder-wide family correction is invoked. Controls are first-order Markov paths estimated from completed fold-training labels; each control reselects its own coordinate using only its synthetic training labels.",
        "",
        "## Was k=1 selected out of sample?",
        "",
        "Yes. For every annual fold, all 512 dimensions were ranked on completed training labels, one coordinate was selected, and its ridge-logistic score was evaluated only on that year's held-out origins. The verifier reconstructs both actual and all 2,376 fold-by-control selections.",
        "",
        "## Coordinate identity and stability",
        "",
        f"The 24 folds selected **{metrics['selection_stability']['unique_coordinates']}** distinct coordinates. The recurring coordinates were:",
        "",
        "| coordinate | folds | fraction |",
        "|---:|---:|---:|",
    ]
    for item in metrics["selection_stability"]["top_coordinates"]:
        lines.append(f"| z{item['coordinate']:03d} | {item['folds']} | {item['fraction']:.1%} |")
    lines.extend([
        "",
        "This is a fold-specific selector, not evidence for one universal TiRex neuron. Signs below are oriented using the training-only event/non-event effect so positive values have a common event-facing interpretation.",
        "",
        "## Held-out coordinate characterization",
        "",
        "| held-out variable | valid folds | median Pearson [min, max] | median Spearman [min, max] |",
        "|---|---:|---:|---:|",
    ])
    for variable, item in metrics["coordinate_characterization"].items():
        pearson = item["event_oriented_pearson"]
        spearman = item["event_oriented_spearman"]
        if pearson["median"] is None:
            p_text = s_text = "n/a"
        else:
            p_text = f"{pearson['median']:.3f} [{pearson['min']:.3f}, {pearson['max']:.3f}]"
            s_text = f"{spearman['median']:.3f} [{spearman['min']:.3f}, {spearman['max']:.3f}]"
        lines.append(f"| {variable} | {item['valid_folds']} | {p_text} | {s_text} |")
    lines.extend([
        "",
        "Every correlation uses the coordinate chosen before that fold and only that fold's held-out origins. VXN is shifted on its complete source calendar before reindexing, so it is at least one full published session old; VXN rows before September 2009 remain missing.",
        "",
        "## Scope",
        "",
        "A positive result means the registered single-coordinate representation separates the actual transition-proximity label from the structurally matched synthetic controls under this reused historical sample. It does not establish incremental information beyond direct RV history, causal neuron semantics, checkpoint pretraining cleanliness, or a pristine out-of-sample discovery.",
        "",
        "## Reproducibility",
        "",
        f"- Protocol SHA-256: `{metrics['protocol_sha256']}`",
        f"- Embeddings SHA-256: `{metrics['input_hashes']['embeddings']}`",
        f"- Classical origins SHA-256: `{metrics['input_hashes']['classical_forecasts']}`",
        f"- Forecast artifact: `{FORECASTS_PATH.relative_to(ROOT)}`",
        f"- Control artifact: `{CONTROLS_PATH.relative_to(ROOT)}`",
        f"- Per-fold selection artifact: `{SELECTIONS_PATH.relative_to(ROOT)}`",
        f"- Held-out correlation artifact: `{CORRELATIONS_PATH.relative_to(ROOT)}`",
        "",
    ])
    return "\n".join(lines)


def _fold_inputs(
    year: int,
    classical_fold: pd.DataFrame,
    y: pd.Series,
    features: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    cutoffs = pd.DatetimeIndex(classical_fold["cutoff"].unique())
    if len(cutoffs) != 1:
        raise RuntimeError(f"fold {year} does not have exactly one cutoff")
    cutoff = cutoffs[0]
    latent._strict_validate_classical_fold(
        y, classical_fold, cutoff=cutoff, horizon=5, quantile=.8
    )
    train, test = latent.build_annual_fold_tables(
        y=y,
        features=features,
        embeddings=embeddings,
        classical_fold=classical_fold,
        cutoff=cutoff,
        horizon=5,
        quantile=.8,
    )
    return train, test, cutoff


def _produce_fold(
    *,
    year: int,
    classical_fold: pd.DataFrame,
    y: pd.Series,
    features: pd.DataFrame,
    embeddings: pd.DataFrame,
    characterization: pd.DataFrame,
    seeds: list[int],
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train, test, cutoff = _fold_inputs(year, classical_fold, y, features, embeddings)
    z_columns = latent.latent_columns()
    X_train = latent._matrix(train, z_columns)
    X_test = latent._matrix(test, z_columns)
    actual_y = train["event"].astype(int).to_numpy()
    dimension, effect = select_top1(X_train, actual_y)
    actual_probe = latent.fit_probe(
        "sparse_k1", X_train[:, [dimension]], actual_y, ridge=ridge, seed=42
    )
    actual = test[
        ["target_end", "event", "trigger_date", "fold_year", "cutoff", "ranking_phase"]
    ].copy()
    actual["selected_dimension"] = dimension
    actual["selected_training_effect"] = effect
    actual["p_latent_sparse_k1_99"] = actual_probe.predict(X_test[:, [dimension]])
    actual["phase"] = actual["ranking_phase"].astype(int)

    paths = [
        latent.markov_surrogate_path(
            actual_y, train_length=len(train), test_length=len(test), seed=seed
        )
        for seed in seeds
    ]
    control_train_labels = np.column_stack([path.train for path in paths])
    control_dimensions, control_effects = select_top1_many(X_train, control_train_labels)
    control_parts = []
    selection_rows = [
        {
            "kind": "actual",
            "fold_year": year,
            "seed": pd.NA,
            "dimension": dimension,
            "standardized_mean_difference": effect,
            "selected_on": "completed_actual_training_labels",
        }
    ]
    for column, (seed, path) in enumerate(zip(seeds, paths)):
        selected = int(control_dimensions[column])
        probe = latent.fit_probe(
            "sparse_k1", X_train[:, [selected]], path.train, ridge=ridge, seed=42
        )
        control_parts.append(
            pd.DataFrame(
                {
                    "origin": test.index,
                    "fold_year": year,
                    "seed": seed,
                    "event_control": path.test.astype(bool),
                    "phase": test["ranking_phase"].to_numpy(dtype=int),
                    "selected_dimension": selected,
                    "p_control": probe.predict(X_test[:, [selected]]),
                }
            )
        )
        selection_rows.append(
            {
                "kind": "control",
                "fold_year": year,
                "seed": seed,
                "dimension": selected,
                "standardized_mean_difference": float(control_effects[column]),
                "selected_on": "that_controls_synthetic_training_labels",
            }
        )

    char = characterization.reindex(test.index).copy()
    char["prior_fold_empirical_rv_percentile"] = test["rv_percentile_prior"].astype(float)
    char["coordinate"] = X_test[:, dimension]
    variables = [
        "log_rv_level",
        "prior_fold_empirical_rv_percentile",
        "trailing_5_session_mean_log_rv",
        "trailing_22_session_mean_log_rv",
        "trailing_1_session_return",
        "trailing_5_session_return",
        "trailing_22_session_return",
        "prior_session_vxn_level",
    ]
    correlations = fold_correlations(
        char,
        train_index=pd.DatetimeIndex(train.index),
        fold_year=year,
        variables=variables,
        orientation=1.0 if effect >= 0 else -1.0,
    )
    correlations["dimension"] = dimension
    correlations["training_effect"] = effect
    correlations["coordinate_orientation"] = 1 if effect >= 0 else -1
    vxn_rows = correlations.loc[
        correlations["variable"] == "prior_session_vxn_level", "n"
    ]
    minimum_vxn_rows = 20
    if len(vxn_rows) != 1 or (int(vxn_rows.iloc[0]) not in (0,) and int(vxn_rows.iloc[0]) < minimum_vxn_rows):
        raise RuntimeError("a partially observed VXN fold falls below the frozen minimum")
    meta = {
        "fold_year": year,
        "cutoff": str(cutoff.date()),
        "train_rows": len(train),
        "test_rows": len(test),
        "selected_dimension": dimension,
    }
    return actual, pd.concat(control_parts, ignore_index=True), pd.DataFrame(selection_rows), correlations, meta


def _score_outputs(
    protocol: dict,
    actual: pd.DataFrame,
    controls: pd.DataFrame,
    selections: pd.DataFrame,
    correlations: pd.DataFrame,
    hashes: dict[str, str],
    folds: list[dict],
) -> tuple[dict, pd.DataFrame]:
    actual_phases = latent.phase_ranking_metrics(
        actual,
        score_column="p_latent_sparse_k1_99",
        horizon=5,
        top_fraction=.1,
    )
    actual_phases["kind"] = "actual"
    actual_phases["seed"] = pd.NA
    control_summaries = []
    phase_parts = [actual_phases]
    for seed, group in controls.groupby("seed", sort=True):
        group = group.sort_values("origin")
        phases = latent.phase_ranking_metrics(
            group,
            score_column="p_control",
            event_column="event_control",
            horizon=5,
            top_fraction=.1,
        )
        phases["kind"] = "control"
        phases["seed"] = int(seed)
        phase_parts.append(phases)
        summary = latent.phase_mean(phases)
        control_summaries.append({"seed": int(seed), **summary})
    phase_metrics = pd.concat(phase_parts, ignore_index=True)
    actual_summary = {
        **latent.phase_mean(actual_phases),
        "dispersion": latent.phase_dispersion(actual_phases),
    }
    control_auc = np.asarray([item["auc"] for item in control_summaries], dtype=float)
    control_lift = np.asarray([item["top_decile_lift"] for item in control_summaries], dtype=float)
    exact = exact_randomization_test(
        actual_summary["auc"],
        control_auc,
        alpha=float(protocol["control_task"]["alpha"]),
    )
    actual_selections = selections.loc[selections["kind"] == "actual"].sort_values("fold_year")
    metrics = {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "evidence_role": protocol["evidence_role"],
        "input_hashes": hashes,
        "origins": int(len(actual)),
        "first_origin": str(actual.index.min().date()),
        "last_origin": str(actual.index.max().date()),
        "folds": int(len(folds)),
        "fold_metadata": folds,
        "actual": actual_summary,
        "controls": {
            "draws": int(len(control_summaries)),
            "auc_median": float(np.median(control_auc)),
            "auc_95th_percentile": float(np.quantile(control_auc, .95)),
            "auc_min": float(control_auc.min()),
            "auc_max": float(control_auc.max()),
            "top_decile_lift_median": float(np.median(control_lift)),
            "by_seed": control_summaries,
        },
        "randomization_test": exact,
        "selection_stability": _selection_stability(actual_selections),
        "coordinate_characterization": _correlation_summary(correlations),
        "interpretation": (
            "This post-result single-rung diagnostic tests formal selectivity against 99 "
            "matched controls; it is not a pristine holdout and does not test incremental "
            "value beyond direct RV-history features."
        ),
    }
    return metrics, phase_metrics


def run_study() -> dict:
    protocol = load_protocol()
    for path in (FORECASTS_PATH, CONTROLS_PATH, SELECTIONS_PATH, CORRELATIONS_PATH, PHASE_PATH, METRICS_PATH, REPORT_PATH):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite the registered 99-control artifact: {path}")
    embeddings, classical, history, vxn, hashes = _load_inputs(protocol)
    y = history["log_rv"].dropna().sort_index()
    features = base.build_history_features(y)
    characterization = _characterization_features(history, vxn, pd.DatetimeIndex(classical.index))
    seeds = control_seeds(protocol)
    ridge = float(protocol["probe"]["ridge"])
    actual_parts = []
    control_parts = []
    selection_parts = []
    correlation_parts = []
    fold_metadata = []
    for year, classical_fold in classical.groupby("fold_year", sort=True):
        values = _produce_fold(
            year=int(year),
            classical_fold=classical_fold.sort_index(),
            y=y,
            features=features,
            embeddings=embeddings,
            characterization=characterization,
            seeds=seeds,
            ridge=ridge,
        )
        actual, controls, selections, correlations, meta = values
        actual_parts.append(actual)
        control_parts.append(controls)
        selection_parts.append(selections)
        correlation_parts.append(correlations)
        fold_metadata.append(meta)
        print(f"k1 confirmation fold {int(year)} complete ({len(actual):,} held-out origins)", flush=True)
    actual = pd.concat(actual_parts).sort_index()
    controls = pd.concat(control_parts, ignore_index=True).sort_values(["seed", "origin"], kind="mergesort")
    selections = pd.concat(selection_parts, ignore_index=True).sort_values(["kind", "fold_year", "seed"], kind="mergesort")
    correlations = pd.concat(correlation_parts, ignore_index=True).sort_values(["fold_year", "variable"], kind="mergesort")
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(actual.index), pd.DatetimeIndex(classical.index), check_names=False
    )
    if len(controls) != len(actual) * len(seeds):
        raise RuntimeError("99-control output does not cover every common row")
    metrics, phases = _score_outputs(
        protocol, actual, controls, selections, correlations, hashes, fold_metadata
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    actual.to_parquet(FORECASTS_PATH, compression="zstd")
    controls.to_parquet(CONTROLS_PATH, index=False, compression="zstd")
    selections.to_parquet(SELECTIONS_PATH, index=False, compression="zstd")
    correlations.to_parquet(CORRELATIONS_PATH, index=False, compression="zstd")
    phases.to_parquet(PHASE_PATH, index=False, compression="zstd")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    REPORT_PATH.write_text(render_report(metrics, selections, correlations))
    return metrics


def _compare_float_frame(left: pd.DataFrame, right: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        np.testing.assert_allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )


def verify_results(*, write: bool = True) -> dict:
    protocol = load_protocol()
    embeddings, classical, history, vxn, hashes = _load_inputs(protocol)
    metrics = json.loads(METRICS_PATH.read_text())
    actual = pd.read_parquet(FORECASTS_PATH).sort_index()
    controls = pd.read_parquet(CONTROLS_PATH).sort_values(["seed", "origin"], kind="mergesort")
    selections = pd.read_parquet(SELECTIONS_PATH).sort_values(["kind", "fold_year", "seed"], kind="mergesort")
    correlations = pd.read_parquet(CORRELATIONS_PATH).sort_values(["fold_year", "variable"], kind="mergesort")
    phases_saved = pd.read_parquet(PHASE_PATH)
    checks: dict[str, bool] = {
        "protocol_hash": metrics.get("protocol_sha256") == _sha256(PROTOCOL_PATH),
        "input_hashes": metrics.get("input_hashes") == hashes,
        "common_rows": len(actual) == len(classical) == int(protocol["sample"]["expected_origins"]),
        "clean_fence": bool(actual.index.max() < pd.Timestamp(protocol["sample"]["clean_start"])),
        "control_coverage": bool(
            len(controls) == len(actual) * 99
            and controls.groupby("seed").size().eq(len(actual)).all()
        ),
        "one_actual_selection_per_fold": bool(
            selections.loc[selections["kind"] == "actual"].groupby("fold_year").size().eq(1).all()
        ),
        "one_control_selection_per_fold_seed": bool(
            selections.loc[selections["kind"] == "control"].groupby(["fold_year", "seed"]).size().eq(1).all()
        ),
    }
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(actual.index), pd.DatetimeIndex(classical.index), check_names=False
    )
    for column in ("event", "target_end", "trigger_date", "fold_year", "cutoff", "ranking_phase"):
        pd.testing.assert_series_equal(
            actual[column].reset_index(drop=True),
            classical[column].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )
    y = history["log_rv"].dropna().sort_index()
    features = base.build_history_features(y)
    characterization = _characterization_features(history, vxn, pd.DatetimeIndex(classical.index))
    seeds = control_seeds(protocol)
    rebuilt_actual = []
    rebuilt_controls = []
    rebuilt_selections = []
    rebuilt_correlations = []
    rebuilt_folds = []
    for year, classical_fold in classical.groupby("fold_year", sort=True):
        values = _produce_fold(
            year=int(year),
            classical_fold=classical_fold.sort_index(),
            y=y,
            features=features,
            embeddings=embeddings,
            characterization=characterization,
            seeds=seeds,
            ridge=float(protocol["probe"]["ridge"]),
        )
        fold_actual, fold_controls, fold_selections, fold_correlations_frame, meta = values
        rebuilt_actual.append(fold_actual)
        rebuilt_controls.append(fold_controls)
        rebuilt_selections.append(fold_selections)
        rebuilt_correlations.append(fold_correlations_frame)
        rebuilt_folds.append(meta)
    expected_actual = pd.concat(rebuilt_actual).sort_index()
    expected_controls = pd.concat(rebuilt_controls, ignore_index=True).sort_values(["seed", "origin"], kind="mergesort")
    expected_selections = pd.concat(rebuilt_selections, ignore_index=True).sort_values(["kind", "fold_year", "seed"], kind="mergesort")
    expected_correlations = pd.concat(rebuilt_correlations, ignore_index=True).sort_values(["fold_year", "variable"], kind="mergesort")
    _compare_float_frame(expected_actual, actual, ["selected_training_effect", "p_latent_sparse_k1_99"])
    np.testing.assert_array_equal(
        expected_actual["selected_dimension"].to_numpy(), actual["selected_dimension"].to_numpy()
    )
    for column in ("origin", "fold_year", "seed", "event_control", "phase", "selected_dimension"):
        pd.testing.assert_series_equal(
            expected_controls[column].reset_index(drop=True),
            controls[column].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )
    _compare_float_frame(expected_controls, controls, ["p_control"])
    for column in ("kind", "fold_year", "dimension", "selected_on"):
        pd.testing.assert_series_equal(
            expected_selections[column].reset_index(drop=True),
            selections[column].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )
    np.testing.assert_allclose(
        pd.to_numeric(expected_selections["seed"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(selections["seed"], errors="coerce").to_numpy(dtype=float),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    _compare_float_frame(expected_selections, selections, ["standardized_mean_difference"])
    for column in ("fold_year", "variable", "n", "dimension", "coordinate_orientation"):
        pd.testing.assert_series_equal(
            expected_correlations[column].reset_index(drop=True),
            correlations[column].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )
    _compare_float_frame(
        expected_correlations,
        correlations,
        ["pearson", "spearman", "event_oriented_pearson", "event_oriented_spearman", "training_effect"],
    )
    recomputed, phases = _score_outputs(
        protocol,
        expected_actual,
        expected_controls,
        expected_selections,
        expected_correlations,
        hashes,
        rebuilt_folds,
    )
    checks["training_only_actual_selection_reconstructed"] = True
    checks["control_training_label_reselection_reconstructed"] = True
    checks["held_out_predictions_reconstructed"] = True
    checks["coordinate_characterization_reconstructed"] = True
    checks["metrics_reproduced"] = recomputed == metrics
    phase_columns = [column for column in phases.columns if column != "seed"]
    pd.testing.assert_frame_equal(
        phases[phase_columns].reset_index(drop=True),
        phases_saved[phase_columns].reset_index(drop=True),
        check_dtype=False,
    )
    np.testing.assert_allclose(
        pd.to_numeric(phases["seed"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(phases_saved["seed"], errors="coerce").to_numpy(dtype=float),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    checks["phase_metrics_reproduced"] = True
    checks["report_reproduced"] = REPORT_PATH.read_text() == render_report(metrics, selections, correlations)
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "origins": len(actual),
        "control_rows": len(controls),
        "fold_control_selections_reconstructed": int(24 * 99),
    }
    if write:
        VERIFICATION_PATH.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS":
        raise RuntimeError(f"k1 confirmation verification failed: {checks}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "run", "verify"])
    args = parser.parse_args()
    if args.command == "validate":
        protocol = load_protocol()
        _validate_hashes(protocol)
        print("latent k1 99-control protocol and input hashes valid")
    elif args.command == "run":
        metrics = run_study()
        print(json.dumps(metrics["randomization_test"], indent=2))
    else:
        print(json.dumps(verify_results(), indent=2))


if __name__ == "__main__":
    main()
