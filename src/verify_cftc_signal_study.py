"""Independent verifier for the frozen CFTC Nasdaq-positioning study.

This module intentionally shares no implementation code with the study runner.
It rebuilds release availability, annual thresholds, five-session labels,
history features, training-fold metadata, and ranking metrics directly from the
persisted source panels.  Model probabilities are treated as forecast outputs;
their bounds and every metric calculated from them are verified independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "free_signal_study.yaml"
DEFAULT_FORECASTS = Path("data/free_signal_study/cftc_positioning_forecasts.parquet")
DEFAULT_METRICS = Path("data/free_signal_study/cftc_positioning_metrics.json")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_index(values: Iterable) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _validate_protocol(protocol: dict) -> None:
    if protocol.get("status") != "frozen_before_empirical_run":
        raise AssertionError("CFTC protocol is not frozen before the empirical run")
    if protocol.get("evidence_class") != "post_program_diagnostic":
        raise AssertionError("CFTC evidence class drifted")
    fences = protocol.get("fences", {})
    if not fences.get("forbid_clean_origins") or not fences.get("no_forward_fill"):
        raise AssertionError("sealed-window or no-fill fence is disabled")
    if pd.Timestamp(fences["final_origin"]) >= pd.Timestamp(fences["clean_start"]):
        raise AssertionError("CFTC protocol overlaps the sealed clean window")

    spec = protocol.get("cftc_positioning", {})
    if spec.get("official_dataset_id") != "gpe5-46if":
        raise AssertionError("CFTC protocol no longer names the official TFF view")
    if spec.get("contract_code") != "20974+":
        raise AssertionError("CFTC consolidated Nasdaq contract code drifted")
    if spec.get("report") != "TFF futures only":
        raise AssertionError("CFTC report family drifted")
    if spec.get("feature") != "leveraged_money_net_open_interest_share":
        raise AssertionError("CFTC positioning feature drifted")
    availability = str(spec.get("availability", {}).get("ordinary_rule", ""))
    if "10 calendar days" not in availability or "on or after" not in availability:
        raise AssertionError("CFTC availability is not the frozen conservative rule")
    if len(spec.get("availability", {}).get("excluded_report_date_ranges", [])) != 2:
        raise AssertionError("CFTC backlog exclusions drifted")
    if not spec.get("fitting", {}).get("one_origin_per_release"):
        raise AssertionError("CFTC weekly one-origin rule is disabled")
    target = spec.get("target", {})
    if int(target.get("horizon_sessions", 0)) != 5:
        raise AssertionError("CFTC target horizon drifted")
    if not math.isclose(float(target.get("stress_quantile", -1)), 0.8):
        raise AssertionError("CFTC stress threshold drifted")
    if not target.get("calm_origins_only"):
        raise AssertionError("CFTC target is no longer calm-origin conditional")


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        protocol = yaml.safe_load(source)
    _validate_protocol(protocol)
    return protocol


def reconstruct_release_origins(
    reports: pd.DataFrame,
    sessions: Iterable,
    protocol: dict,
) -> pd.DataFrame:
    """Rebuild conservative public-availability origins from weekly reports."""
    spec = protocol["cftc_positioning"]
    required = {
        "report_date", "contract_code", "lev_long", "lev_short", "open_interest"
    }
    missing = sorted(required - set(reports.columns))
    if missing:
        raise AssertionError(f"CFTC source panel is missing {missing}")
    frame = reports.loc[:, sorted(required)].copy()
    frame["report_date"] = pd.to_datetime(
        frame["report_date"], errors="raise"
    ).dt.normalize()
    if frame["report_date"].duplicated().any():
        raise AssertionError("CFTC source has duplicate weekly report dates")
    codes = frame["contract_code"].astype(str).str.strip()
    if not codes.eq(str(spec["contract_code"])).all():
        raise AssertionError("CFTC source contains a different or overlapping contract")
    frame["contract_code"] = codes
    for column in ("lev_long", "lev_short", "open_interest"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise AssertionError(f"CFTC source has nonfinite {column}")
    if (frame["open_interest"] <= 0).any():
        raise AssertionError("CFTC source has nonpositive open interest")

    keep = pd.Series(True, index=frame.index)
    for start, end in spec["availability"]["excluded_report_date_ranges"]:
        keep &= ~frame["report_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    frame = frame.loc[keep].sort_values("report_date").copy()
    if not frame["report_date"].is_monotonic_increasing:
        raise AssertionError("CFTC report dates are not sorted")
    frame["available_date"] = frame["report_date"] + pd.Timedelta(days=10)

    market_sessions = _normal_index(sessions).sort_values().unique()
    if market_sessions.has_duplicates or not market_sessions.is_monotonic_increasing:
        raise AssertionError("market sessions are not unique and sorted")
    positions = market_sessions.searchsorted(
        pd.DatetimeIndex(frame["available_date"]), side="left"
    )
    valid = positions < len(market_sessions)
    frame = frame.loc[valid].copy()
    frame["origin"] = market_sessions.take(positions[valid])
    if frame["origin"].duplicated().any():
        raise AssertionError("multiple CFTC releases map to one origin")
    if (frame["origin"] < frame["available_date"]).any():
        raise AssertionError("CFTC release was used before conservative availability")
    frame["lev_net_share"] = (
        frame["lev_long"] - frame["lev_short"]
    ) / frame["open_interest"]
    if not np.isfinite(frame["lev_net_share"].to_numpy(dtype=float)).all():
        raise AssertionError("CFTC leverage share is nonfinite")
    return frame.reset_index(drop=True)


def reconstruct_fold_targets(
    log_rv: pd.Series,
    *,
    cutoff: pd.Timestamp,
    origins: pd.DatetimeIndex,
    horizon: int,
    quantile: float,
) -> pd.DataFrame:
    """Construct annual-fold labels from an independent positional loop."""
    values = pd.to_numeric(log_rv, errors="raise").astype(float).copy()
    values.index = _normal_index(values.index)
    values = values.sort_index()
    if values.index.duplicated().any() or values.isna().any():
        raise AssertionError("log-RV history must be complete, unique, and sorted")
    training = values.loc[:pd.Timestamp(cutoff)]
    if training.empty:
        raise AssertionError("annual target fold has no threshold history")
    threshold = float(training.quantile(float(quantile)))
    positions = pd.Series(np.arange(len(values), dtype=int), index=values.index)
    rows: list[dict[str, Any]] = []
    for origin in _normal_index(origins):
        if origin not in positions.index:
            continue
        position = int(positions.loc[origin])
        future = values.iloc[position + 1 : position + 1 + int(horizon)]
        if len(future) != int(horizon) or future.isna().any():
            continue
        exceedance = future > threshold
        event = bool(exceedance.any())
        rows.append(
            {
                "origin": origin,
                "target_end": future.index[-1],
                "threshold": threshold,
                "calm": bool(values.loc[origin] <= threshold),
                "event": event,
                "trigger_date": future.index[int(np.flatnonzero(exceedance)[0])]
                if event
                else pd.NaT,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("origin").sort_index()


def history_features(log_rv: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(log_rv, errors="raise").astype(float).copy()
    values.index = _normal_index(values.index)
    values = values.sort_index()
    return pd.DataFrame(
        {
            "log_rv_d": values,
            "log_rv_w": values.rolling(5).mean(),
            "log_rv_m": values.rolling(22).mean(),
        },
        index=values.index,
    )


def join_scorable(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Drop missing model inputs, never a valid negative's NaT trigger date."""
    missing = sorted(set(feature_columns) - set(features.columns))
    if missing:
        raise AssertionError(f"history features are missing {missing}")
    joined = labels.join(features[feature_columns])
    return joined.dropna(subset=feature_columns)


def _completed_origins(
    sessions: pd.DatetimeIndex, cutoff: pd.Timestamp, horizon: int
) -> pd.DatetimeIndex:
    ordered = pd.DatetimeIndex(sessions).sort_values()
    cutoff_position = int(ordered.searchsorted(pd.Timestamp(cutoff), side="right") - 1)
    final_position = cutoff_position - int(horizon)
    return ordered[: final_position + 1] if final_position >= 0 else pd.DatetimeIndex([])


def reconstruct_scored_rows(
    reports: pd.DataFrame,
    history: pd.DataFrame,
    protocol: dict,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """Rebuild every row/label and each annual training-fold audit statistic."""
    if "log_rv" not in history:
        raise AssertionError("locked history has no log_rv")
    y = pd.to_numeric(history["log_rv"], errors="raise").dropna().astype(float)
    y.index = _normal_index(y.index)
    y = y.sort_index()
    if y.index.duplicated().any():
        raise AssertionError("locked history has duplicate sessions")
    releases = reconstruct_release_origins(reports, y.index, protocol)
    release_by_origin = releases.set_index("origin").sort_index()
    if release_by_origin.index.duplicated().any():
        raise AssertionError("CFTC origin mapping is not one release to one origin")
    features = history_features(y)

    spec = protocol["cftc_positioning"]
    horizon = int(spec["target"]["horizon_sessions"])
    quantile = float(spec["target"]["stress_quantile"])
    columns = list(spec["fitting"]["baseline_features"])
    minimum = int(spec["fitting"]["minimum_training_releases"])
    first_year = int(spec["fitting"]["first_score_year"])
    final_origin = pd.Timestamp(protocol["fences"]["final_origin"])
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []

    for year in range(first_year, final_origin.year + 1):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = min(pd.Timestamp(f"{year}-12-31"), final_origin)
        prior_sessions = y.index[y.index < year_start]
        if not len(prior_sessions):
            continue
        cutoff = prior_sessions[-1]
        complete = _completed_origins(y.index, cutoff, horizon)
        train_origins = release_by_origin.index[
            (release_by_origin.index < year_start)
            & release_by_origin.index.isin(complete)
        ]
        train_targets = reconstruct_fold_targets(
            y,
            cutoff=cutoff,
            origins=train_origins,
            horizon=horizon,
            quantile=quantile,
        )
        if train_targets.empty:
            continue
        train_targets = train_targets.loc[train_targets["calm"]]
        train = join_scorable(train_targets[["event"]], features, columns)
        train["lev_net_share"] = release_by_origin.loc[
            train.index, "lev_net_share"
        ]
        train = train.dropna(subset=[*columns, "event", "lev_net_share"])
        if len(train) < minimum or train["event"].nunique() != 2:
            continue
        feature_mean = float(train["lev_net_share"].mean())
        feature_std = float(train["lev_net_share"].std(ddof=0))
        if not np.isfinite(feature_std) or feature_std <= 0:
            raise AssertionError("CFTC training leverage feature has zero scale")

        test_origins = release_by_origin.index[
            (release_by_origin.index >= year_start)
            & (release_by_origin.index <= year_end)
        ]
        targets = reconstruct_fold_targets(
            y,
            cutoff=cutoff,
            origins=test_origins,
            horizon=horizon,
            quantile=quantile,
        )
        if targets.empty:
            continue
        test = join_scorable(targets.loc[targets["calm"]], features, columns)
        if test.empty:
            continue
        test["lev_net_share"] = release_by_origin.loc[
            test.index, "lev_net_share"
        ]
        test["cftc_lev_net_z"] = (
            test["lev_net_share"] - feature_mean
        ) / feature_std
        test["fold_year"] = year
        test["training_cutoff"] = cutoff
        parts.append(test)
        folds.append(
            {
                "year": year,
                "training_cutoff": str(cutoff.date()),
                "training_releases": int(len(train)),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
            }
        )

    if not parts:
        raise AssertionError("independent reconstruction produced no scored rows")
    expected = pd.concat(parts).sort_index()
    expected.index.name = "origin"
    return expected, folds, releases


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    positives = int(positive.sum())
    negatives = int((~positive).sum())
    if not positives or not negatives:
        return math.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    return float(
        (ranks[positive].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def ranking_summary(
    frame: pd.DataFrame, model: str, *, top_fraction: float
) -> dict[str, Any]:
    score_column = f"p_{model}"
    if "event" not in frame or score_column not in frame:
        raise AssertionError(f"forecast rows lack event or {score_column}")
    labels = frame["event"].astype(int).to_numpy()
    scores = pd.to_numeric(frame[score_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise AssertionError(f"{model} probability is nonfinite or outside [0, 1]")
    if not np.isin(labels, [0, 1]).all():
        raise AssertionError("event labels are not binary")
    base_rate = float(labels.mean())
    count = max(1, int(math.ceil(len(labels) * float(top_fraction))))
    order = np.argsort(-scores, kind="mergesort")
    top_rate = float(labels[order[:count]].mean())
    lift = float(top_rate / base_rate) if base_rate > 0 else math.nan
    return {
        "n": int(len(frame)),
        "positives": int(labels.sum()),
        "base_rate": base_rate,
        "auc": _auc(labels, scores),
        "top_decile_lift": lift,
        "top_decile_event_rate": top_rate,
    }


def _assert_close(label: str, actual: Any, expected: Any, tolerance: float = 1e-12) -> None:
    if isinstance(expected, bool):
        if bool(actual) is not expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, np.integer)):
        if int(actual) != int(expected):
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (float, np.floating)):
        left, right = float(actual), float(expected)
        if math.isnan(left) and math.isnan(right):
            return
        if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
            raise AssertionError(f"{label}: {left!r} != {right!r}")
        return
    if actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def _compare_summary(label: str, actual: dict, expected: dict) -> None:
    for key, value in expected.items():
        if key not in actual:
            raise AssertionError(f"{label}.{key} is missing")
        _assert_close(f"{label}.{key}", actual[key], value)


def _assert_series_equal(label: str, actual: pd.Series, expected: pd.Series) -> None:
    try:
        pd.testing.assert_series_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise AssertionError(f"{label} differs: {error}") from error


def verify_artifacts(
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    root = Path(root)
    protocol = load_protocol(protocol_path)
    checks = 1
    spec = protocol["cftc_positioning"]
    source_path = root / spec["source"]
    history_path = root / spec["target"]["source"]
    forecast_path = root / DEFAULT_FORECASTS
    metrics_path = root / DEFAULT_METRICS
    for path in (source_path, history_path, forecast_path, metrics_path):
        if not path.is_file():
            raise AssertionError(f"required CFTC artifact is missing: {path}")

    source = pd.read_parquet(source_path)
    history = pd.read_parquet(history_path)
    forecasts = pd.read_parquet(forecast_path).sort_index()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_digest = sha256(source_path)
    if metrics.get("source_sha256") != source_digest:
        raise AssertionError("reported CFTC source hash does not match the source panel")
    checks += 1

    expected, expected_folds, releases = reconstruct_scored_rows(
        source, history, protocol
    )
    if not forecasts.index.is_unique or not forecasts.index.is_monotonic_increasing:
        raise AssertionError("forecast origins are not unique and sorted")
    forecast_index = _normal_index(forecasts.index)
    if not forecast_index.equals(expected.index):
        missing = expected.index.difference(forecast_index)
        extra = forecast_index.difference(expected.index)
        raise AssertionError(
            f"forecast row origins differ; missing={list(missing)}, extra={list(extra)}"
        )
    forecasts.index = forecast_index
    forecasts.index.name = "origin"
    checks += 1

    final_origin = pd.Timestamp(protocol["fences"]["final_origin"])
    clean_start = pd.Timestamp(protocol["fences"]["clean_start"])
    if (forecasts.index > final_origin).any() or (forecasts.index >= clean_start).any():
        raise AssertionError("forecast origin crossed the sealed-window fence")
    target_ends = pd.to_datetime(forecasts["target_end"], errors="raise")
    if (target_ends > final_origin).any() or (target_ends >= clean_start).any():
        raise AssertionError("forecast target crossed the sealed-window fence")
    checks += 1

    release_index = pd.DatetimeIndex(releases["origin"])
    if release_index.has_duplicates:
        raise AssertionError("weekly release map repeats an origin")
    if not forecasts.index.isin(release_index).all():
        raise AssertionError("a forecast origin is not tied to exactly one CFTC release")
    checks += 1

    date_columns = ("target_end", "trigger_date", "training_cutoff")
    for column in date_columns:
        if column not in forecasts:
            raise AssertionError(f"forecast output lacks {column}")
        _assert_series_equal(
            column,
            pd.to_datetime(forecasts[column], errors="coerce"),
            pd.to_datetime(expected[column], errors="coerce"),
        )
    checks += 1
    for column in ("calm", "event", "fold_year"):
        _assert_series_equal(column, forecasts[column], expected[column])
    if not forecasts["calm"].astype(bool).all():
        raise AssertionError("non-calm origin was scored")
    checks += 1
    for column in (
        "threshold", "log_rv_d", "log_rv_w", "log_rv_m",
        "lev_net_share", "cftc_lev_net_z",
    ):
        _assert_series_equal(column, forecasts[column], expected[column])
    checks += 1

    events = forecasts["event"].astype(bool)
    trigger = pd.to_datetime(forecasts["trigger_date"], errors="coerce")
    if trigger.loc[events].isna().any():
        raise AssertionError("positive event lacks its first trigger session")
    if trigger.loc[~events].notna().any():
        raise AssertionError("negative event has a trigger session")
    negative_nat = int((~events & trigger.isna()).sum())
    if negative_nat != int((~events).sum()):
        raise AssertionError("valid negative labels with NaT trigger were dropped")
    checks += 1

    top_fraction = float(spec["scoreboard"]["top_fraction"])
    calculated = {
        model: ranking_summary(forecasts, model, top_fraction=top_fraction)
        for model in ("baseline", "augmented")
    }
    for model, summary in calculated.items():
        _compare_summary(f"models.{model}", metrics["models"][model], summary)
        checks += 1
    delta_auc = calculated["augmented"]["auc"] - calculated["baseline"]["auc"]
    delta_lift = (
        calculated["augmented"]["top_decile_lift"]
        - calculated["baseline"]["top_decile_lift"]
    )
    _assert_close("delta_auc", metrics["delta_auc"], delta_auc)
    _assert_close("delta_top_decile_lift", metrics["delta_top_decile_lift"], delta_lift)
    registered = bool(
        calculated["augmented"]["auc"] > calculated["baseline"]["auc"]
        and calculated["augmented"]["top_decile_lift"]
        > calculated["baseline"]["top_decile_lift"]
    )
    _assert_close("registered_success", metrics["registered_success"], registered)
    checks += 1

    if metrics.get("first_origin") != str(forecasts.index.min().date()):
        raise AssertionError("reported first CFTC origin is wrong")
    if metrics.get("last_origin") != str(forecasts.index.max().date()):
        raise AssertionError("reported last CFTC origin is wrong")
    if metrics.get("evidence_class") != protocol["evidence_class"]:
        raise AssertionError("reported CFTC evidence class drifted")
    checks += 1

    break_date = pd.Timestamp(spec["structural_break"]["date"])
    sensitivity_samples = {
        "pre_micro_inclusion": forecasts.loc[forecasts.index < break_date],
        "post_micro_inclusion": forecasts.loc[forecasts.index >= break_date],
    }
    stored_sensitivity = metrics.get("structural_break_sensitivity", {})
    for name, sample in sensitivity_samples.items():
        if len(sample) and sample["event"].nunique() == 2:
            if name not in stored_sensitivity:
                raise AssertionError(f"missing structural-break sensitivity {name}")
            for model in ("baseline", "augmented"):
                summary = ranking_summary(sample, model, top_fraction=top_fraction)
                _compare_summary(
                    f"structural_break_sensitivity.{name}.{model}",
                    stored_sensitivity[name][model],
                    summary,
                )
    checks += 1

    stored_folds = metrics.get("folds", [])
    if len(stored_folds) != len(expected_folds):
        raise AssertionError("reported annual CFTC fold count is wrong")
    for position, (stored, rebuilt) in enumerate(zip(stored_folds, expected_folds)):
        for key, expected_value in rebuilt.items():
            if key not in stored:
                raise AssertionError(f"folds[{position}].{key} is missing")
            _assert_close(f"folds[{position}].{key}", stored[key], expected_value)
    checks += 1

    return {
        "status": "PASS",
        "checks": checks,
        "rows": int(len(forecasts)),
        "positives": int(events.sum()),
        "negative_labels_with_nat_trigger": negative_nat,
        "first_origin": str(forecasts.index.min().date()),
        "last_origin": str(forecasts.index.max().date()),
        "source_sha256": source_digest,
        "baseline_auc": calculated["baseline"]["auc"],
        "augmented_auc": calculated["augmented"]["auc"],
        "baseline_top_decile_lift": calculated["baseline"]["top_decile_lift"],
        "augmented_top_decile_lift": calculated["augmented"]["top_decile_lift"],
        "registered_success": registered,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    result = verify_artifacts(args.protocol, root=args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
