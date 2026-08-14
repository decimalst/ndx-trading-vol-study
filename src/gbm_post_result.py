"""Frozen post-result diagnostics for the GBM functional-form study.

The command intentionally performs outcome-conditioned attribution.  It never
changes the parent study's estimator, interaction lock, metrics, or verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.gbm_study import (
    FEATURES,
    build_design,
    complete_rows,
    forecast_origins,
    make_gbm,
    paired_comparison,
    qlike,
    split_origins,
    training_rows,
)

from . import envcheck

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "gbm_post_result.yaml"
PAIR_ORDER = (
    "gbm_confirmation",
    "locked_term_confirmation",
    "timing_safe_gbm_confirmation",
    "earnings_with_iv_diagnostic",
    "earnings_without_iv_diagnostic",
)


def resolve_repo_path(value: str) -> pathlib.Path:
    return ROOT / value


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: str | pathlib.Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    with pathlib.Path(path).open() as handle:
        spec = yaml.safe_load(handle)
    if spec["status"] != "frozen_before_diagnostic_query":
        raise ValueError("post-result protocol is not frozen")
    if int(spec["loss_decomposition"]["deciles"]) != 10:
        raise ValueError("loss decomposition must use ten frozen bins")
    return spec


def validate_input_hashes(spec: dict[str, Any]) -> None:
    records = [spec["inputs"][name] for name in (
        "master", "gbm_forecasts", "gbm_metrics", "gbm_interactions"
    )]
    records.extend(spec["inputs"]["earnings_forecasts"].values())
    forbidden = spec["inputs"]["forbid_filename_suffix"]
    for record in records:
        path = resolve_repo_path(record["path"])
        if path.name.endswith(forbidden):
            raise ValueError(f"forbidden smoothed-methodology artifact: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"input hash mismatch for {path}: {actual}")


def assign_realized_deciles(actual: pd.Series, bins: int = 10) -> pd.Series:
    """Stable, equal-count outcome bins with date/order breaking exact ties."""
    values = pd.Series(actual, copy=True)
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("realized values must be finite and complete")
    if len(values) < bins:
        raise ValueError("not enough rows for requested realized-variance bins")
    ranks = values.rank(method="first", ascending=True)
    labels = pd.qcut(ranks, q=bins, labels=np.arange(1, bins + 1))
    return pd.Series(labels.astype(int), index=values.index, name="realized_decile")


def build_timing_safe_design(master: pd.DataFrame) -> pd.DataFrame:
    """Parent HAR-IV design with only VXN moved behind the 16:00 boundary."""
    design = build_design(master)
    design["liv"] = np.log(master["vxn"].shift(1))
    return design


def timing_safe_assessment(comparisons: dict[str, dict[str, Any]]) -> str:
    required = ("all_diagnostic", "discovery", "confirmation")
    if set(comparisons) != set(required):
        raise ValueError("timing-safe assessment requires all three frozen splits")
    point_no_better = all(comparisons[name]["mean_difference"] >= 0 for name in required)
    confirmation_win = (
        comparisons["confirmation"]["mean_difference"] < 0
        and comparisons["confirmation"]["block_bootstrap_p_value"] < 0.05
    )
    return (
        "SURVIVES_AT_1600_SAFE_BOUNDARY"
        if point_no_better and not confirmation_win
        else "DOES_NOT_SURVIVE_AT_1600_SAFE_BOUNDARY"
    )


def decompose_pair(frame: pd.DataFrame, *, pair: str, baseline_col: str,
                   candidate_col: str, bins: int = 10,
                   tolerance: float = 1e-12) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ["actual_var", baseline_col, candidate_col]
    use = frame.dropna(subset=required).copy()
    if use.empty:
        raise ValueError(f"empty common sample for {pair}")
    if use.index.has_duplicates or not use.index.is_monotonic_increasing:
        raise ValueError("pair sample index must be unique and increasing")
    use["baseline_loss"] = qlike(
        use["actual_var"].to_numpy(float), use[baseline_col].to_numpy(float)
    )
    use["candidate_loss"] = qlike(
        use["actual_var"].to_numpy(float), use[candidate_col].to_numpy(float)
    )
    use["loss_difference"] = use["candidate_loss"] - use["baseline_loss"]
    use["candidate_win"] = use["loss_difference"] < 0
    use["realized_decile"] = assign_realized_deciles(use["actual_var"], bins)
    use["pair"] = pair

    total_sum = float(use["loss_difference"].sum())
    rows: list[dict[str, Any]] = []
    for decile, group in use.groupby("realized_decile", sort=True):
        gap_sum = float(group["loss_difference"].sum())
        rows.append({
            "realized_decile": int(decile),
            "n": int(len(group)),
            "actual_var_min": float(group["actual_var"].min()),
            "actual_var_max": float(group["actual_var"].max()),
            "baseline_mean_qlike": float(group["baseline_loss"].mean()),
            "candidate_mean_qlike": float(group["candidate_loss"].mean()),
            "mean_loss_difference": float(group["loss_difference"].mean()),
            "sum_loss_difference": gap_sum,
            "candidate_win_rate": float(group["candidate_win"].mean()),
            "fraction_total_gap": (
                float(gap_sum / total_sum) if abs(total_sum) > tolerance else None
            ),
        })
    deciles = pd.DataFrame(rows).set_index("realized_decile")
    reconstructed_sum = float(deciles["sum_loss_difference"].sum())
    reconstructed_mean = float(
        np.average(deciles["mean_loss_difference"], weights=deciles["n"])
    )
    total_mean = float(use["loss_difference"].mean())
    if not np.isclose(reconstructed_sum, total_sum, rtol=0.0, atol=tolerance):
        raise AssertionError("decile sums do not reconcile to aggregate gap")
    if not np.isclose(reconstructed_mean, total_mean, rtol=0.0, atol=tolerance):
        raise AssertionError("decile means do not reconcile to aggregate mean")
    baseline_mean = float(use["baseline_loss"].mean())
    candidate_mean = float(use["candidate_loss"].mean())
    improvement = (
        float(100 * (baseline_mean - candidate_mean) / baseline_mean)
        if abs(baseline_mean) > tolerance else None
    )
    summary = {
        "pair": pair,
        "n": int(len(use)),
        "origin_start": pd.Timestamp(use.index.min()).isoformat(),
        "origin_end": pd.Timestamp(use.index.max()).isoformat(),
        "baseline_mean_qlike": baseline_mean,
        "candidate_mean_qlike": candidate_mean,
        "mean_loss_difference": total_mean,
        "sum_loss_difference": total_sum,
        "candidate_win_rate": float(use["candidate_win"].mean()),
        "improvement_pct": improvement,
        "top_decile_sum_loss_difference": float(deciles.loc[bins, "sum_loss_difference"]),
        "top_decile_fraction_total_gap": deciles.loc[bins, "fraction_total_gap"],
        "reconciliation": {
            "sum_absolute_error": abs(reconstructed_sum - total_sum),
            "mean_absolute_error": abs(reconstructed_mean - total_mean),
            "tolerance": tolerance,
        },
        "deciles": {
            str(int(idx)): {
                key: (None if pd.isna(value) else int(value) if key == "n" else float(value))
                for key, value in row.items()
            }
            for idx, row in deciles.iterrows()
        },
    }
    keep = use[[
        "actual_var", baseline_col, candidate_col, "baseline_loss",
        "candidate_loss", "loss_difference", "candidate_win", "realized_decile", "pair"
    ]].copy()
    keep = keep.rename(columns={baseline_col: "baseline_forecast",
                                candidate_col: "candidate_forecast"})
    return keep, summary


def score_shap_interactions(values: np.ndarray,
                            features: tuple[str, ...] = FEATURES) -> list[dict[str, Any]]:
    cube = np.asarray(values, dtype=float)
    if cube.ndim == 4 and cube.shape[-1] == 1:
        cube = cube[..., 0]
    if cube.ndim != 3 or cube.shape[1:] != (len(features), len(features)):
        raise ValueError(f"unexpected SHAP interaction shape {cube.shape}")
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(features):
        for j in range(i + 1, len(features)):
            rows.append({
                "pair": [a, features[j]],
                "mean_absolute_interaction": float(np.mean(np.abs(cube[:, i, j]))),
            })
    return sorted(rows, key=lambda item: -item["mean_absolute_interaction"])


def feature_correlations(background: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            rows.append({
                "feature_a": a,
                "feature_b": b,
                "pearson": float(background[a].corr(background[b], method="pearson")),
                "spearman": float(background[a].corr(background[b], method="spearman")),
            })
    return pd.DataFrame(rows)


def _read_forecast_window(path: pathlib.Path, start: pd.Timestamp,
                          end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[
        ("origin", ">=", start), ("origin", "<=", end)
    ])
    if frame.empty or frame.index.min() < start or frame.index.max() > end:
        raise ValueError(f"forecast predicate failed for {path}")
    return frame.sort_index()


def load_master_through(spec: dict[str, Any]) -> pd.DataFrame:
    end = pd.Timestamp(spec["fences"]["latest_allowed_target_date"])
    path = resolve_repo_path(spec["inputs"]["master"]["path"])
    frame = pd.read_parquet(path, filters=[("date", "<=", end)])
    if frame.index.max() > end:
        raise ValueError("master predicate admitted a post-fence row")
    return frame.sort_index()


def load_pair_frames(spec: dict[str, Any],
                     timing_safe: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, str, str]]:
    start = pd.Timestamp(spec["fences"]["diagnostic_start"])
    confirm = pd.Timestamp(spec["fences"]["confirmation_start"])
    end = pd.Timestamp(spec["fences"]["diagnostic_end"])
    gbm = _read_forecast_window(
        resolve_repo_path(spec["inputs"]["gbm_forecasts"]["path"]), confirm, end
    )
    if set(gbm["phase"].unique()) != {"confirmation"}:
        raise ValueError("GBM confirmation slice contains another phase")
    pairs: dict[str, tuple[pd.DataFrame, str, str]] = {
        "gbm_confirmation": (gbm, "har_iv_var", "gbm_var"),
        "locked_term_confirmation": (gbm, "har_iv_var", "augmented_var"),
        "timing_safe_gbm_confirmation": (
            timing_safe.loc[confirm:end], "har_iv_var", "gbm_var"
        ),
    }

    master = load_master_through(spec)
    actual = master["rv_total"].shift(-1).rename("actual_var")
    target_date = pd.Series(master.index, index=master.index).shift(-1).rename("target_date")
    earnings = spec["inputs"]["earnings_forecasts"]
    loaded = {
        name: _read_forecast_window(resolve_repo_path(item["path"]), start, end)
        for name, item in earnings.items()
    }
    definitions = {
        "earnings_with_iv_diagnostic": ("har_iv", "har_iv_x"),
        "earnings_without_iv_diagnostic": ("har", "har_x"),
    }
    for pair, (baseline_name, candidate_name) in definitions.items():
        common = pd.concat([
            actual,
            target_date,
            loaded[baseline_name]["mean_var"].rename("baseline_var"),
            loaded[candidate_name]["mean_var"].rename("candidate_var"),
        ], axis=1, join="inner").dropna()
        if common.index.min() < start or common.index.max() > end:
            raise ValueError("earnings pair escaped diagnostic origin fence")
        if pd.to_datetime(common["target_date"]).max() > pd.Timestamp(
            spec["fences"]["latest_allowed_target_date"]
        ):
            raise ValueError("earnings pair escaped target fence")
        pairs[pair] = (common, "baseline_var", "candidate_var")
    return pairs


def run_timing_safe_sensitivity(spec: dict[str, Any], master: pd.DataFrame
                                ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-run both frozen forms with prior-session VXN and no reselection."""
    parent = yaml.safe_load((ROOT / "gbm_study.yaml").read_text())
    design = build_timing_safe_design(master)
    origins = split_origins(design, parent, "all_diagnostic")
    forecasts = forecast_origins(design, origins, parent)
    comparisons: dict[str, dict[str, Any]] = {}
    for offset, split in enumerate(("all_diagnostic", "discovery", "confirmation")):
        if split == "all_diagnostic":
            subset = forecasts
        else:
            lo = pd.Timestamp(parent["splits"][split]["start"])
            hi = pd.Timestamp(parent["splits"][split]["end"])
            subset = forecasts.loc[lo:hi]
        comparisons[split] = paired_comparison(subset, "gbm_var", parent, offset)
    result = {
        "evidence_class": spec["timing_safe_sensitivity"]["evidence_class"],
        "vxn_rule": "log(VXN close).shift(1) for both estimators",
        "n": int(len(forecasts)),
        "origin_start": forecasts.index.min().isoformat(),
        "origin_end": forecasts.index.max().isoformat(),
        "last_target_date": pd.Timestamp(forecasts["target_date"].max()).isoformat(),
        "comparisons": comparisons,
        "substantive_assessment": timing_safe_assessment(comparisons),
        "parent_verdict_changed": False,
    }
    return forecasts, result


def run_shap_audit(spec: dict[str, Any], master: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    design = build_design(master)
    snapshot = pd.Timestamp(spec["correlation_audit"]["snapshot_origin"])
    train = training_rows(design, snapshot, 500)
    lo = pd.Timestamp(spec["fences"]["diagnostic_start"])
    hi = pd.Timestamp("2019-12-31")
    background = complete_rows(design).loc[lo:hi, list(FEATURES)]
    correlations = feature_correlations(background)
    selected = spec["correlation_audit"]["pd_selected_pair"]
    selected_row = correlations[
        ((correlations["feature_a"] == selected[0]) &
         (correlations["feature_b"] == selected[1])) |
        ((correlations["feature_a"] == selected[1]) &
         (correlations["feature_b"] == selected[0]))
    ].iloc[0]
    result: dict[str, Any] = {
        "evidence_class": "post-result diagnostic",
        "status": "not_run",
        "pd_selected_pair": selected,
        "pd_pair_pearson": float(selected_row["pearson"]),
        "pd_pair_spearman": float(selected_row["spearman"]),
        "snapshot_origin": snapshot.isoformat(),
        "fit_rows": int(len(train)),
        "fit_last_row": train.index.max().isoformat(),
        "background_rows": int(len(background)),
        "background_start": background.index.min().isoformat(),
        "background_end": background.index.max().isoformat(),
        "method": spec["correlation_audit"]["shap"],
    }
    try:
        import shap

        model = make_gbm(yaml.safe_load((ROOT / "gbm_study.yaml").read_text()))
        model.fit(train.loc[:, FEATURES].to_numpy(float), train["y_next"].to_numpy(float))
        explainer = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent", model_output="raw"
        )
        raw = explainer.shap_interaction_values(
            background.loc[:, FEATURES].to_numpy(float)
        )
        scores = score_shap_interactions(np.asarray(raw), FEATURES)
        result.update({
            "status": "success",
            "shap_version": shap.__version__,
            "scores": scores,
            "selected_pair": scores[0]["pair"],
            "agrees_with_pd": set(scores[0]["pair"]) == set(selected),
        })
    except Exception as exc:  # explicitly frozen transparent-failure behavior
        try:
            import shap
            version = shap.__version__
        except Exception:
            version = None
        result.update({
            "status": "unsupported_or_failed",
            "shap_version": version,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
    return result, correlations


def render_report(metrics: dict[str, Any], shap_result: dict[str, Any]) -> str:
    lines = [
        "# GBM post-result reviewer diagnostics",
        "",
        "**Evidence class: post-result, outcome-conditioned descriptive diagnostic.**",
        "",
        f"Protocol SHA-256: `{metrics['protocol_sha256']}`.",
        "",
        "The parent verdict remains **INCONCLUSIVE**. Substantively, GBM was point-estimate worse than HAR-IV on all three frozen splits (about 4%); nothing here changes that registered verdict.",
        "",
        "The frozen comparison used the same same-origin Cboe VXN close for both models, so it is an internally common-information comparison, but it is timing-ambiguous relative to the repository's standing 16:00 origin. It is not labeled leakage-free or timing-safe here.",
        "",
        "## 16:00 timing-safe VXN sensitivity",
        "",
        "Both estimators were rerun unchanged except that `liv` uses the preceding complete session's VXN close. There was no tuning or interaction reselection.",
        "",
        "| split | n | HAR-IV QLIKE | GBM QLIKE | improvement | block p | 95% block interval | win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    timing = metrics["timing_safe_sensitivity"]
    for split in ("all_diagnostic", "discovery", "confirmation"):
        row = timing["comparisons"][split]
        lines.append(
            f"| {split} | {row['n']} | {row['baseline_qlike']:.6f} | "
            f"{row['candidate_qlike']:.6f} | {row['improvement_pct']:+.3f}% | "
            f"{row['block_bootstrap_p_value']:.4g} | "
            f"[{row['block_ci95'][0]:+.6f}, {row['block_ci95'][1]:+.6f}] | "
            f"{100*row['win_rate']:.1f}% |"
        )
    lines += [
        "",
        f"Pre-specified sensitivity assessment: **{timing['substantive_assessment']}**. The original formal verdict remains **INCONCLUSIVE**.",
        "",
        "## QLIKE loss gap by realized-variance decile",
        "",
        "Candidate minus baseline loss is reported below; negative values favor the candidate. Fractions attribute the aggregate paired gap and can be negative or exceed 100% when deciles offset one another.",
        "",
    ]
    for pair in PAIR_ORDER:
        result = metrics["pairs"][pair]
        lines += [
            f"### {pair.replace('_', ' ')}",
            "",
            f"Common origins: {result['n']} ({result['origin_start'][:10]} through {result['origin_end'][:10]}). Overall candidate improvement: {result['improvement_pct']:+.3f}%; win rate: {100*result['candidate_win_rate']:.1f}%; top-decile share of aggregate gap: {_fmt_fraction(result['top_decile_fraction_total_gap'])}.",
            "",
            "| RV decile | n | baseline QLIKE | candidate QLIKE | mean diff | sum diff | win rate | fraction of total gap |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for decile in map(str, range(1, 11)):
            row = result["deciles"][decile]
            lines.append(
                f"| {decile} | {row['n']} | {row['baseline_mean_qlike']:.6f} | "
                f"{row['candidate_mean_qlike']:.6f} | {row['mean_loss_difference']:+.6f} | "
                f"{row['sum_loss_difference']:+.6f} | {100*row['candidate_win_rate']:.1f}% | "
                f"{_fmt_fraction(row['fraction_total_gap'])} |"
            )
        lines += [""]
    gbm = metrics["pairs"]["gbm_confirmation"]
    locked = metrics["pairs"]["locked_term_confirmation"]
    timing_pair = metrics["pairs"]["timing_safe_gbm_confirmation"]
    earnings_iv = metrics["pairs"]["earnings_with_iv_diagnostic"]
    earnings_no_iv = metrics["pairs"]["earnings_without_iv_diagnostic"]
    lines += [
        "## Diagnostic reading",
        "",
        f"The proposed asymmetric-loss mechanism is strongly supported for the GBM comparison, though not literally in all other nine bins: GBM improves mean QLIKE in realized-variance deciles 2–8, loses in deciles 1, 9, and 10, and decile 10 contributes {100*gbm['top_decile_fraction_total_gap']:.1f}% of the net deficit. The many-small-gains/rare-large-loss shape therefore explains the worse mean despite a {100*gbm['candidate_win_rate']:.1f}% daily win rate.",
        "",
        f"The frozen locked term shows the same concentrated failure more cleanly: it improves deciles 1–8, is nearly flat in decile 9, and decile 10 contributes {100*locked['top_decile_fraction_total_gap']:.1f}% of its net deficit. The timing-safe GBM sensitivity also improves deciles 2–8 while its top decile contributes {100*timing_pair['top_decile_fraction_total_gap']:.1f}% of the deficit.",
        "",
        f"The transfer is not universal. The earnings-with-IV comparison is a near-zero cancellation ({earnings_iv['improvement_pct']:+.3f}%), and it improves rather than loses in the top decile, so its {100*earnings_iv['top_decile_fraction_total_gap']:.1f}% ratio is unstable against a tiny aggregate denominator and does not support the mechanism. Without IV, the top decile contributes {100*earnings_no_iv['top_decile_fraction_total_gap']:.1f}% of the deficit, but several middle deciles also lose. The defensible finding is therefore concentrated QLIKE tail fragility for added functional flexibility and the locked term, with partial—not four-way—replication.",
        "",
    ]
    lines += [
        "Every table reconciles to its full-sample paired mean and sum within the frozen 1e-12 tolerance. These are realized-outcome bins, so they explain where an observed loss gap occurred; they do not define a usable ex-ante rule.",
        "",
        "## Correlated-feature and SHAP audit",
        "",
        f"The partial-dependence-selected pair `lrv_w × lrv_m` has discovery-background Pearson correlation {shap_result['pd_pair_pearson']:.4f} and Spearman correlation {shap_result['pd_pair_spearman']:.4f}. Both are overlapping averages of the same RV series. Partial dependence can therefore extrapolate into weakly supported combinations, making this selection plausibly a correlation artifact.",
        "",
    ]
    if shap_result["status"] == "success":
        lines += [
            f"A fixed post-result TreeExplainer interaction audit (SHAP {shap_result['shap_version']}) selected `{' × '.join(shap_result['selected_pair'])}`; it {'agreed' if shap_result['agrees_with_pd'] else 'did not agree'} with the PD pair.",
            "",
            "| SHAP pair | mean absolute interaction | selected |",
            "|---|---:|:---:|",
        ]
        for i, row in enumerate(shap_result["scores"]):
            lines.append(
                f"| {' × '.join(row['pair'])} | {row['mean_absolute_interaction']:.8f} | {'yes' if i == 0 else 'no'} |"
            )
        lines.append("")
    else:
        lines += [
            f"The fixed TreeExplainer audit failed transparently under SHAP {shap_result.get('shap_version')}: `{shap_result.get('exception_type')}: {shap_result.get('exception_message')}`. No substitute selector was introduced.",
            "",
        ]
    lines += [
        "The SHAP audit reused the exact frozen discovery-snapshot estimator and full discovery background, with no confirmation reselection, hyperparameter search, or substitute method. It is diagnostic evidence only.",
        "",
    ]
    return "\n".join(lines)


def _fmt_fraction(value: float | None) -> str:
    return "n/a" if value is None else f"{100*value:+.1f}%"


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(spec: dict[str, Any]) -> None:
    validate_input_hashes(spec)
    master = load_master_through(spec)
    timing_safe, timing_metrics = run_timing_safe_sensitivity(spec, master)
    all_rows = []
    metrics = {
        "study_id": spec["study_id"],
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "evidence_class": spec["evidence_class"],
        "parent_frozen_verdict": spec["immutability"]["frozen_parent_verdict"],
        "parent_same_origin_vxn_timing_status": spec["immutability"]["parent_timing_status"],
        "timing_safe_sensitivity": timing_metrics,
        "pairs": {},
    }
    for pair, (frame, baseline, candidate) in load_pair_frames(spec, timing_safe).items():
        rows, result = decompose_pair(
            frame, pair=pair, baseline_col=baseline, candidate_col=candidate,
            bins=int(spec["loss_decomposition"]["deciles"]),
            tolerance=float(spec["loss_decomposition"]["reconciliation_tolerance"]),
        )
        all_rows.append(rows)
        metrics["pairs"][pair] = result

    shap_result, correlations = run_shap_audit(spec, master)
    outputs = spec["outputs"]
    loss_rows = pd.concat(all_rows).rename_axis("origin").reset_index()
    loss_rows.to_parquet(resolve_repo_path(outputs["loss_rows"]), index=False)
    timing_safe.to_parquet(resolve_repo_path(outputs["timing_safe_forecasts"]))
    correlations.to_csv(resolve_repo_path(outputs["correlations"]), index=False)
    write_json(resolve_repo_path(outputs["metrics"]), metrics)
    write_json(resolve_repo_path(outputs["shap"]), shap_result)
    report = render_report(metrics, shap_result)
    report_path = resolve_repo_path(outputs["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"wrote post-result diagnostics for {len(metrics['pairs'])} pairs")


def verify(spec: dict[str, Any]) -> None:
    validate_input_hashes(spec)
    metrics = json.loads(resolve_repo_path(spec["outputs"]["metrics"]).read_text())
    shap_result = json.loads(resolve_repo_path(spec["outputs"]["shap"]).read_text())
    if metrics["parent_frozen_verdict"] != "INCONCLUSIVE":
        raise ValueError("post-result artifact changed parent verdict")
    if metrics["timing_safe_sensitivity"]["parent_verdict_changed"] is not False:
        raise ValueError("timing sensitivity rewrote parent verdict")
    master = load_master_through(spec)
    timing_path = resolve_repo_path(spec["outputs"]["timing_safe_forecasts"])
    timing_safe = pd.read_parquet(timing_path).sort_index()
    parent = yaml.safe_load((ROOT / "gbm_study.yaml").read_text())
    expected_origins = split_origins(
        build_timing_safe_design(master), parent, "all_diagnostic"
    )
    if not timing_safe.index.equals(expected_origins):
        raise ValueError("timing-safe forecasts no longer match exact frozen origins")
    comparisons: dict[str, dict[str, Any]] = {}
    for offset, split in enumerate(("all_diagnostic", "discovery", "confirmation")):
        subset = timing_safe if split == "all_diagnostic" else timing_safe.loc[
            pd.Timestamp(parent["splits"][split]["start"]):
            pd.Timestamp(parent["splits"][split]["end"])
        ]
        comparisons[split] = paired_comparison(subset, "gbm_var", parent, offset)
    if comparisons != metrics["timing_safe_sensitivity"]["comparisons"]:
        raise ValueError("timing-safe metrics do not recompute from saved forecasts. "
                         + envcheck.pin_advice())
    if timing_safe_assessment(comparisons) != metrics["timing_safe_sensitivity"]["substantive_assessment"]:
        raise ValueError("timing-safe assessment no longer follows its frozen rule")
    rebuilt_pairs: dict[str, dict[str, Any]] = {}
    for pair, (frame, baseline, candidate) in load_pair_frames(spec, timing_safe).items():
        _rows, result = decompose_pair(
            frame,
            pair=pair,
            baseline_col=baseline,
            candidate_col=candidate,
            bins=int(spec["loss_decomposition"]["deciles"]),
            tolerance=float(spec["loss_decomposition"]["reconciliation_tolerance"]),
        )
        rebuilt_pairs[pair] = result
    if rebuilt_pairs != metrics["pairs"]:
        raise ValueError("loss-decile metrics do not recompute from locked inputs. "
                         + envcheck.pin_advice())
    correlations = pd.read_csv(resolve_repo_path(spec["outputs"]["correlations"]))
    expected_correlations = feature_correlations(
        complete_rows(build_design(master)).loc[
            pd.Timestamp(spec["fences"]["diagnostic_start"]):pd.Timestamp("2019-12-31"),
            list(FEATURES),
        ]
    )
    pd.testing.assert_frame_equal(correlations, expected_correlations)
    for result in rebuilt_pairs.values():
        if result["reconciliation"]["sum_absolute_error"] > result["reconciliation"]["tolerance"]:
            raise ValueError("saved diagnostic fails sum reconciliation")
        if result["reconciliation"]["mean_absolute_error"] > result["reconciliation"]["tolerance"]:
            raise ValueError("saved diagnostic fails mean reconciliation")
    expected = render_report(metrics, shap_result)
    if resolve_repo_path(spec["outputs"]["report"]).read_text() != expected:
        raise ValueError("saved post-result report does not match artifacts")
    print("GBM POST-RESULT VERIFICATION PASS")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    args = parser.parse_args(argv)
    if pathlib.Path(args.protocol).resolve() != DEFAULT_PROTOCOL.resolve():
        raise ValueError("empirical commands require the repository-frozen protocol")
    spec = load_protocol(args.protocol)
    if args.command == "run":
        run(spec)
    else:
        verify(spec)


if __name__ == "__main__":
    main()
