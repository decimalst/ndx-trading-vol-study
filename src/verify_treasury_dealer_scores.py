"""Independent Treasury score reconstruction using explicit circular index tables.

The producer evaluator and prefix-sum inference are never imported. Shared
frozen independent date/panel, scalar-validation, Holm and comparison utilities
are disclosed in SCORE_CONTRACT.md; protocol validation is metadata only.
"""

import copy
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.treasury_dealer_protocol import CLOCK_CLASS, CLOCK_LIMITATION, validate
from src.verify_claims_release_scores import _compare, _holm, _integer, _real
from src.verify_commodity_implied_scores import _panel

PHASES = ("development", "evaluation")
CONTROLS = ("matched", "market")
TENORS = ("2", "3", "5", "7", "10", "30")


def _indexed_inference(
    differences, mask, *, blocks=(21, 63, 126), hac_lags=126, draws=399999, seed=20261001
):
    values = np.asarray(differences)
    selected = np.asarray(mask)
    if (
        values.ndim != 1
        or values.dtype.kind not in "fiu"
        or selected.ndim != 1
        or selected.dtype.kind != "b"
        or len(values) != len(selected)
        or len(values) < 2
    ):
        raise ValueError("Matching full-calendar real values and boolean mask required")
    values = values.astype(float, copy=True)
    n, observed = len(values), int(selected.sum())
    if observed < 2 or np.isinf(values).any() or not np.isfinite(values[selected]).all():
        raise ValueError("At least two finite selected observations required")
    _integer(draws, "draws")
    _integer(seed, "seed", 0)
    lags = min(_integer(hac_lags, "HAC lags", 0), n - 1)
    lengths = tuple(blocks)
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("Nonempty distinct block lengths required")
    for length in lengths:
        if _integer(length, "block") > n:
            raise ValueError("Block exceeds phase calendar")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = float(np.mean(values[selected]))
            influence = np.zeros(n)
            influence[selected] = (values[selected] - mean) / (observed / n)
            variance = float(np.dot(influence, influence) / n)
            # Explicit lag products; no producer HAC or covariance helper.
            for lag in range(1, lags + 1):
                covariance = float(np.dot(influence[lag:], influence[:-lag]) / n)
                variance += 2 * (1 - lag / (lags + 1)) * covariance
            if not math.isfinite(variance) or variance < 0:
                raise ValueError("Invalid reconstructed HAC variance")
            se = math.sqrt(variance / n)
            hac_p = float(2 * norm.sf(abs(mean) / se)) if se else float(mean == 0)
            hac = {
                "se": se,
                "p": hac_p,
                "ci95": [mean - 1.96 * se, mean + 1.96 * se],
                "mde80_nominal": float((norm.ppf(0.975) + norm.ppf(0.8)) * se),
            }
            weighted = np.where(selected, values, 0.0)
            block_results = {}
            for length in lengths:
                full, remainder = divmod(n, length)

                def tables(width):
                    positions = (np.arange(n)[:, None] + np.arange(width)[None, :]) % n
                    return weighted[positions].sum(axis=1), selected[positions].sum(axis=1)

                sums_table, counts_table = tables(length)
                sums = np.empty(draws)
                counts = np.empty(draws, dtype=np.int64)
                rng = np.random.default_rng(seed + length)
                # Complete-block RNG draws precede every remainder draw.
                # Tables use explicit source positions, never cumulative sums.
                for start in range(0, draws, 1024):
                    stop = min(draws, start + 1024)
                    indices = rng.integers(0, n, size=(stop - start, full))
                    sums[start:stop] = sums_table[indices].sum(axis=1)
                    counts[start:stop] = counts_table[indices].sum(axis=1)
                if remainder:
                    tail_sums, tail_counts = tables(remainder)
                    for start in range(0, draws, 1024):
                        stop = min(draws, start + 1024)
                        indices = rng.integers(0, n, size=stop - start)
                        sums[start:stop] += tail_sums[indices]
                        counts[start:stop] += tail_counts[indices]
                if (counts == 0).any():
                    raise ValueError("zero-support bootstrap replicate")
                sampled = sums / counts
                if not np.isfinite(sampled).all():
                    raise ValueError("Nonfinite independently sampled means")
                probability = (1 + int((abs(sampled - mean) >= abs(mean)).sum())) / (draws + 1)
                block_results[str(length)] = {
                    "p": float(probability),
                    "ci95": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
                }
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("Invalid independent inference arithmetic") from error
    if not np.isfinite([mean, *hac["ci95"], hac["mde80_nominal"]]).all():
        raise ValueError("Nonfinite independent inference result")
    return {
        "mean": mean,
        "n": observed,
        "full_calendar_n": n,
        "hac": hac,
        "block_inference": block_results,
        "p_conservative": max(hac_p, *(r["p"] for r in block_results.values())),
    }


def _prior_probabilities(prior):
    if type(prior) is not list or len(prior) != 144:
        raise ValueError("Exactly144 preserved prior hypotheses required")
    probabilities, sources, seen = [], {}, set()
    for row in prior:
        if type(row) is not dict:
            raise ValueError("Prior records required")
        probabilities.append(_real(row.get("p_conservative"), "prior probability", 0, 1))
        for key in ("p_holm_wave", "p_holm_cumulative"):
            if key in row:
                _real(row[key], key, 0, 1)
        for key in ("study", "candidate", "control", "source"):
            if type(row.get(key)) is not str or not row[key]:
                raise ValueError("Prior identity/provenance is missing")
        _integer(row.get("horizon"), "prior horizon")
        index = _integer(row.get("source_row_index"), "source row index", 0)
        digest = row.get("source_sha256")
        if type(digest) is not str or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Prior source SHA256 required")
        source = row["source"]
        if (source, index) in seen or (source in sources and sources[source] != digest):
            raise ValueError("Prior source identities conflict")
        seen.add((source, index))
        sources[source] = digest
    return probabilities


def _checked_event_counts(row, prefix, minimum):
    counts = row.get(prefix + "events_per_tenor")
    if type(counts) is not dict or set(counts) != set(TENORS):
        raise ValueError("Six exact event tenor counts required")
    values = [_integer(counts[t], "tenor count", minimum) for t in TENORS]
    total = _integer(row.get(prefix + "event_n"), "event total", 0)
    if total != sum(values):
        raise ValueError("Event total differs from tenor sum")
    return total


def _verify_support(common, calendar, activation, report, protocol):
    if (
        type(report) is not dict
        or set(report)
        != {"status", "passed", "monthly", "phases", "slices", "offsets", "discrepancies"}
        or report["status"] != "SUPPORT_PASS"
        or report["passed"] is not True
        or report["discrepancies"] != []
    ):
        raise ValueError("Complete independently consistent passing support required")
    spec, bounds = protocol["support"], protocol["forecast"]
    rows = []
    for field in ("phases", "offsets"):
        if type(report[field]) is not dict or set(report[field]) != set(PHASES):
            raise ValueError("Two exact support phases required")
    if type(report["slices"]) is not list or len(report["slices"]) != 2:
        raise ValueError("Two exact support slices required")
    for phase in PHASES:
        indices = common.index[common.phase == phase]
        rows.append((report["phases"][phase], bounds[phase], indices, "phase", {}))
        offsets = report["offsets"][phase]
        if type(offsets) is not list or len(offsets) != 5:
            raise ValueError("Five global support offsets required")
        for offset, row in enumerate(offsets):
            selected = indices[calendar.get_indexer(indices) % 5 == offset]
            rows.append((row, bounds[phase], selected, "offset", {"offset": offset}))
    for index, limits in enumerate(bounds["stability"]):
        indices = common.index[(common.index >= limits[0]) & (common.index <= limits[1])]
        rows.append(
            (report["slices"][index], limits, indices, "slice", {"slice_index": index})
        )
    for row, limits, indices, scope, extra in rows:
        daily = len(indices)
        active = int(activation.loc[indices].sum())
        daily_floor = spec[scope + "_daily"]
        active_floor = spec[
            "phase_active_dates_per_offset"
            if scope == "offset"
            else scope + "_activation_dates"
        ]
        tenor_floor = 0 if scope == "offset" else spec[scope + "_events_per_tenor"]
        if daily < daily_floor or active < active_floor:
            raise ValueError("INSUFFICIENT_DATA: reconstructed score support")
        if type(row) is not dict or _checked_event_counts(row, "", tenor_floor) < active:
            raise ValueError("Activation events are inconsistent")
        expected = {
            "origin_start": limits[0],
            "origin_end": limits[1],
            "daily_n": daily,
            "activation_dates": active,
            "event_n": row["event_n"],
            "events_per_tenor": row["events_per_tenor"],
            "passed": True,
            "failures": [],
        } | extra
        _compare(row, expected, "support." + scope, exact=True)
    months = pd.period_range(bounds["origin_start"], bounds["origin_end"], freq="M")
    monthly = report["monthly"]
    if type(monthly) is not list or len(monthly) != len(months):
        raise ValueError("Complete monthly support schedule required")
    month_fields = {
        "month",
        "status",
        "fit_origin",
        "training_cutoff",
        "requested_n",
        "application_n",
        "train_n",
        "train_activation_dates",
        "train_event_n",
        "train_events_per_tenor",
        "passed",
        "failures",
    }
    for row, month in zip(monthly, months, strict=True):
        if (
            type(row) is not dict
            or set(row) != month_fields
            or row["month"] != str(month)
            or row["passed"] is not True
            or row["failures"] != []
        ):
            raise ValueError("Exact ordered passing monthly support required")
        selected = calendar[
            (calendar >= bounds["origin_start"])
            & (calendar <= bounds["origin_end"])
            & (calendar.to_period("M") == month)
        ]
        scored = common[common.index.to_period("M") == month]
        requested_n = _integer(row["requested_n"], "monthly requested", 0)
        application_n = _integer(row["application_n"], "monthly applications", 0)
        if requested_n != len(selected) or not len(scored) <= application_n <= requested_n:
            raise ValueError("Monthly support counts conflict with calendar/panel")
        if row["status"] == "SUPPORTED":
            train = _integer(row["train_n"], "monthly train", bounds["minimum_train"])
            train_active = _integer(
                row["train_activation_dates"],
                "training activity",
                spec["training_activation_dates"],
            )
            if (
                train_active > train
                or _checked_event_counts(row, "train_", spec["training_events_per_tenor"])
                < train_active
                or application_n == 0
            ):
                raise ValueError("Invalid training event support")
            fit, cutoff = pd.Timestamp(row["fit_origin"]), pd.Timestamp(row["training_cutoff"])
            position = calendar.get_indexer([fit])[0]
            if fit not in selected or position < 1 or calendar[position - 1] != cutoff:
                raise ValueError("Monthly support clock differs from full calendar")
            for field, value in (
                ("fit_origin", fit),
                ("training_cutoff", cutoff),
                ("train_n", train),
            ):
                if not scored[field].eq(value).all():
                    raise ValueError("Monthly support differs from paired fit metadata")
        else:
            expected_status = (
                "NO_REQUESTED_ORIGINS" if requested_n == 0 else "NO_COMPLETE_ORIGIN"
            )
            if (
                row["status"] != expected_status
                or application_n
                or len(scored)
                or any(
                    row[k] is not None
                    for k in (
                        "fit_origin",
                        "training_cutoff",
                        "train_n",
                        "train_activation_dates",
                        "train_event_n",
                        "train_events_per_tenor",
                    )
                )
            ):
                raise ValueError("Invalid unfit month support evidence")


def verify_scores(panel, calendar, activation_mask, metrics, prior, protocol, support_report):
    validate(protocol)
    if type(metrics) is not dict:
        raise ValueError("Scored metrics dictionary required")
    qualification = {}
    if {"source_clock_class", "source_clock_limitation"} & set(metrics):
        qualification = {
            "source_clock_class": CLOCK_CLASS,
            "source_clock_limitation": CLOCK_LIMITATION,
        }
        _compare(
            {key: metrics[key] for key in qualification if key in metrics},
            qualification,
            "source_clock_qualification",
            exact=True,
        )
    old_p = _prior_probabilities(prior)
    if (
        not isinstance(activation_mask, pd.Series)
        or not activation_mask.index.equals(calendar)
        or activation_mask.dtype != np.dtype(bool)
    ):
        raise ValueError("Exact original-calendar boolean activity series required")
    if (
        not isinstance(panel, pd.DataFrame)
        or "treasury_cutoff_date" not in panel
        or "commodity_cutoff_date" in panel
    ):
        raise ValueError("Treasury-specific scored cutoff metadata required")
    forecast = protocol["forecast"]
    limits = {p: tuple(pd.Timestamp(x) for x in forecast[p]) for p in PHASES}
    checked = _panel(
        panel.rename(columns={"treasury_cutoff_date": "commodity_cutoff_date"}),
        calendar,
        limits,
        pd.Timestamp(forecast["source_end"]),
        forecast["minimum_train"],
    )
    if (checked.commodity_cutoff_date < pd.Timestamp("2010-01-01")).any():
        raise ValueError("Treasury cutoff precedes fixed source floor")
    common = checked[checked.model == "candidate"].set_index("origin")
    _verify_support(common, calendar, activation_mask, support_report, protocol)
    options = protocol["inference"]
    rows = []
    for control in CONTROLS:
        other = checked[checked.model == control].set_index("origin")
        phases = []
        for phase_code, phase in enumerate(PHASES):
            frame = common[common.phase == phase]
            reference = other.loc[frame.index]
            differences = frame.loss.to_numpy() - reference.loss.to_numpy()
            start, end = limits[phase]
            dates = calendar[(calendar >= start) & (calendar <= end)]
            positions = dates.get_indexer(frame.index)
            full_values = np.full(len(dates), np.nan)
            full_values[positions] = differences
            daily = np.zeros(len(dates), dtype=bool)
            daily[positions] = True
            active = daily & activation_mask.loc[dates].to_numpy()
            phase_result = {"name": phase}
            for endpoint_code, (endpoint, mask) in enumerate(
                (("daily", daily), ("active", active))
            ):
                result = _indexed_inference(
                    full_values,
                    mask,
                    blocks=options["blocks"],
                    hac_lags=options["hac_lags"],
                    draws=options["bootstrap_draws"],
                    seed=options["seed"] + phase_code * 10000 + endpoint_code * 1000000,
                )
                included = mask[positions]
                intervals = [
                    result["hac"]["ci95"],
                    *[b["ci95"] for b in result["block_inference"].values()],
                ]
                result |= {
                    "ci95_envelope": [
                        min(i[0] for i in intervals),
                        max(i[1] for i in intervals),
                    ],
                    "candidate_loss": float(frame.loss.to_numpy()[included].mean()),
                    "control_loss": float(reference.loss.to_numpy()[included].mean()),
                    "first_origin": str(frame.index[included][0].date()),
                    "last_origin": str(frame.index[included][-1].date()),
                }
                phase_result[endpoint] = result

            def group(selection, differences=differences, frame=frame):
                d = differences[selection]
                a = activation_mask.loc[frame.index[selection]].to_numpy()
                return {
                    "daily_n": len(d),
                    "active_n": int(a.sum()),
                    "daily_delta": float(d.mean()) if len(d) else None,
                    "active_delta": float(d[a].mean()) if a.any() else None,
                }

            offsets = calendar.get_indexer(frame.index) % 5
            phase_result["offsets"] = [{"offset": k} | group(offsets == k) for k in range(5)]
            phase_result["stability"] = (
                [
                    {"start": a, "end": b} | group((frame.index >= a) & (frame.index <= b))
                    for a, b in forecast["stability"]
                ]
                if phase == "evaluation"
                else []
            )
            phase_result["annual"] = [
                {
                    "year": int(year),
                    "calendar_origins": int((dates.year == year).sum()),
                    "missing_origins": int((dates.year == year).sum())
                    - int((frame.index.year == year).sum()),
                }
                | group(frame.index.year == year)
                for year in sorted(set(dates.year))
            ]
            phase_result["p_conservative"] = max(
                phase_result[e]["p_conservative"] for e in ("daily", "active")
            )
            phases.append(phase_result)
        rows.append(
            {
                "study": "treasury_dealer",
                "candidate": "candidate",
                "control": control,
                "horizon": 5,
                "score": "qlike",
                "phases": phases,
                "p_conservative": max(p["p_conservative"] for p in phases),
            }
        )
    probabilities = [r["p_conservative"] for r in rows]
    wave, cumulative = _holm(probabilities), _holm(old_p + probabilities)[-2:]
    for row, pw, pc in zip(rows, wave, cumulative, strict=True):
        passed = (
            pw < protocol["comparisons"]["wave_alpha"]
            and pc < protocol["comparisons"]["cumulative_alpha"]
        )
        for phase in row["phases"]:
            passed = (
                passed and phase["daily"]["mean"] <= -0.005 and phase["active"]["mean"] < 0
            )
            passed = passed and all(
                g["daily_delta"] < 0 and g["active_delta"] < 0
                for g in phase["offsets"] + phase["stability"]
            )
        row |= {
            "p_holm_wave": float(pw),
            "p_holm_cumulative": float(pc),
            "verdict": "COMPARISON_GATE_PASS" if passed else "DOES_NOT_QUALIFY",
        }
    expected = {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["treasury_dealer"]
        if all(r["verdict"] == "COMPARISON_GATE_PASS" for r in rows)
        else [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 146,
        "common_scored_origins": len(common),
        "common_active_origins": int(activation_mask.loc[common.index].sum()),
        "support_report": copy.deepcopy(support_report),
    } | qualification
    _compare(metrics, expected)
    return {
        "status": "VERIFIED",
        "comparisons_verified": 2,
        "endpoints_verified": 8,
        "bootstrap_runs_verified": 24,
        "hac_runs_verified": 8,
        "inherited_comparisons_verified": 144,
        "common_scored_origins": len(common),
        "common_active_origins": expected["common_active_origins"],
    }
