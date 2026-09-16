"""Independent four-arm score reconstruction; no gate producer imports."""

import copy
import json
import re

import numpy as np
import pandas as pd

from src.precision_gate_protocol import EVIDENCE_CLASS, EVIDENCE_LIMITATION
from src.verify_claims_release_scores import (
    _calendar,
    _compare,
    _dates,
    _holm,
    _integer,
    _real,
)
from src.verify_treasury_dealer_scores import _indexed_inference

CONTROLS = ("base", "adaptive", "constant")
PHASES = ("development", "evaluation")


def _contract(protocol):
    fixed = {
        "forecast": {
            "issuance_start": "2010-01-04",
            "origin_start": "2016-01-01",
            "origin_end": "2025-10-20",
            "source_end": "2025-10-20",
            "development": ["2016-01-01", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "stability": [["2020-01-01", "2022-12-31"], ["2023-01-01", "2025-10-20"]],
            "horizon": 5,
            "minimum_train": 1000,
            "adaptive_half_life": 252,
            "gate_window": 1260,
            "gate_minimum_train": 252,
        },
        "support": {
            "phase_daily": 505,
            "slice_daily": 252,
            "offset_daily": 63,
            "phase_fitted_gate": 252,
            "slice_fitted_gate": 126,
        },
        "inference": {
            "blocks": [21, 63, 126],
            "hac_lags": 126,
            "bootstrap_draws": 399999,
            "seed": 20261003,
        },
        "comparisons": {
            "wave": 26,
            "inherited": 146,
            "new": 3,
            "cumulative": 149,
            "contrasts": [
                ["contextual", "base"],
                ["contextual", "adaptive"],
                ["contextual", "constant"],
            ],
            "wave_alpha": 0.05 / (26 * 27),
            "cumulative_alpha": 0.05,
        },
    }
    if type(protocol) is not dict:
        raise ValueError("Fixed precision protocol required")
    for section, values in fixed.items():
        if type(protocol.get(section)) is not dict:
            raise ValueError("Missing independent scientific contract")
        for key, expected in values.items():
            actual = protocol[section].get(key)
            try:
                if json.dumps(actual, allow_nan=False) != json.dumps(expected):
                    raise ValueError("Independent scientific contract mismatch: " + key)
            except (TypeError, OverflowError) as error:
                raise ValueError("Invalid scientific contract") from error


def _prior_p(prior):
    if type(prior) is not list or len(prior) != 146:
        raise ValueError("Exactly146 inherited hypothesis records required")
    probabilities, identities, hashes = [], set(), {}
    for row in prior:
        if type(row) is not dict:
            raise ValueError("Inherited record dictionary required")
        probabilities.append(_real(row.get("p_conservative"), "prior probability", 0, 1))
        for name in ("p_holm_wave", "p_holm_cumulative"):
            if name in row:
                _real(row[name], name, 0, 1)
        for field in ("study", "candidate", "control", "source"):
            if type(row.get(field)) is not str or not row[field]:
                raise ValueError("Prior identity and provenance required")
        _integer(row.get("horizon"), "prior horizon")
        index = _integer(row.get("source_row_index"), "source row index", 0)
        digest, source = row.get("source_sha256"), row["source"]
        if type(digest) is not str or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Prior lowercase SHA256 required")
        if (source, index) in identities or (source in hashes and hashes[source] != digest):
            raise ValueError("Conflicting inherited source identities")
        identities.add((source, index))
        hashes[source] = digest
    return probabilities


def _checked_panel(panel, calendar):
    _calendar(calendar)
    if (calendar > pd.Timestamp("2025-10-20")).any():
        raise ValueError("Numerical calendar exceeds fixed ceiling")
    required = {
        "origin",
        "model",
        "prediction",
        "y",
        "loss",
        "target_end",
        "phase",
        "offset",
        "fit_origin",
        "training_cutoff",
        "train_n",
        "gate_n",
        "gate_status",
        "state",
    }
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.columns.has_duplicates
        or not set(panel) >= required
        or panel.empty
    ):
        raise ValueError("Complete nonempty four-arm scored panel required")
    p = panel.copy(deep=True)
    for field in ("origin", "target_end", "fit_origin", "training_cutoff"):
        if not pd.api.types.is_datetime64_any_dtype(p[field].dtype):
            raise ValueError("Native panel dates required")
        p[field] = _dates(p[field])
    if (
        set(p.model) != {"contextual", *CONTROLS}
        or p.duplicated(["origin", "model"]).any()
        or not p.groupby("origin").size().eq(4).all()
    ):
        raise ValueError("Exactly four distinct arm rows per common origin required")
    for column in ("prediction", "y", "loss", "state"):
        if pd.api.types.is_bool_dtype(p[column].dtype) or not pd.api.types.is_numeric_dtype(
            p[column].dtype
        ):
            raise ValueError("Nonboolean real scored numeric fields required")
        values = p[column].to_numpy(dtype=float, na_value=np.nan)
        if np.iscomplexobj(p[column].to_numpy()) or not np.isfinite(values).all():
            raise ValueError("Finite real scored values required")
    for column in ("offset", "train_n", "gate_n"):
        if (
            not pd.api.types.is_integer_dtype(p[column].dtype)
            or pd.api.types.is_bool_dtype(p[column].dtype)
            or p[column].isna().any()
        ):
            raise ValueError("Literal integer gate/training counts and offsets required")
    if (
        (p.y <= 0).any()
        or (p.prediction <= 0).any()
        or (p.train_n < 1000).any()
        or (p.gate_n < 0).any()
        or (p.gate_n > 1260).any()
    ):
        raise ValueError("Invalid target/forecast or issued-history count")
    if not p.gate_status.eq(
        pd.Series(np.where(p.gate_n >= 252, "fitted", "cold_start"), index=p.index)
    ).all():
        raise ValueError("Gate status contradicts fixed issued-history floor")
    metadata = [c for c in p if c not in {"model", "prediction", "loss"}]
    for column in metadata:
        if (
            p[column].isna().any()
            or not p.groupby("origin")[column].nunique(dropna=False).eq(1).all()
        ):
            raise ValueError("Mismatched four-arm common metadata: " + column)
    position = calendar.get_indexer(p.origin)
    fit = calendar.get_indexer(p.fit_origin)
    if (
        (position < 1).any()
        or (position + 5 >= len(calendar)).any()
        or (fit < 1).any()
        or (fit > position).any()
    ):
        raise ValueError("Origin/fit/endpoint must exist on original calendar")
    if not np.array_equal(
        p.target_end.to_numpy(), calendar[position + 5].to_numpy()
    ) or not np.array_equal(p.offset.to_numpy(), position % 5):
        raise ValueError("Target or offset uses a changed calendar")
    if not np.array_equal(
        p.training_cutoff.to_numpy(), calendar[fit - 1].to_numpy()
    ) or not np.array_equal(
        pd.DatetimeIndex(p.origin).to_period("M"),
        pd.DatetimeIndex(p.fit_origin).to_period("M"),
    ):
        raise ValueError("Fit month or strict previous-session cutoff differs")
    development = (p.origin >= "2016-01-01") & (p.origin <= "2019-12-31")
    evaluation = (p.origin >= "2020-01-01") & (p.origin <= "2025-10-20")
    if (
        not (development | evaluation).all()
        or not np.array_equal(
            p.phase.to_numpy(), np.where(development, "development", "evaluation")
        )
        or (p.loc[development, "target_end"] > "2019-12-31").any()
    ):
        raise ValueError("Literal phase and maturity boundaries violated")
    try:
        with np.errstate(over="raise", under="raise", divide="raise", invalid="raise"):
            relative = p.y.to_numpy(float) / p.prediction.to_numpy(float)
            reconstructed_loss = relative - np.log(relative) - 1
    except FloatingPointError as error:
        raise ValueError("Invalid independently reconstructed QLIKE") from error
    if not np.isfinite(reconstructed_loss).all() or not np.allclose(
        reconstructed_loss, p.loss, atol=1e-10, rtol=1e-8
    ):
        raise ValueError("Saved QLIKE differs from independent reconstruction")
    p["loss"] = reconstructed_loss
    return p.sort_values(["origin", "model"])


def _support(common, calendar):
    for phase in PHASES:
        frame = common[common.phase == phase]
        if len(frame) < 505 or int(frame.gate_status.eq("fitted").sum()) < 252:
            raise ValueError("INSUFFICIENT_DATA: phase daily/fitted support")
        offset = calendar.get_indexer(frame.index) % 5
        if any(int((offset == k).sum()) < 63 for k in range(5)):
            raise ValueError("INSUFFICIENT_DATA: phase global-offset support")
    for a, b in (("2020-01-01", "2022-12-31"), ("2023-01-01", "2025-10-20")):
        frame = common.loc[a:b]
        if len(frame) < 252 or int(frame.gate_status.eq("fitted").sum()) < 126:
            raise ValueError("INSUFFICIENT_DATA: slice daily/fitted support")


def verify_scores(panel, calendar, metrics, prior, protocol):
    _contract(protocol)
    old = _prior_p(prior)
    if type(metrics) is not dict:
        raise ValueError("Scored metrics dictionary required")
    qualification = {
        "evidence_class": EVIDENCE_CLASS,
        "evidence_limitation": EVIDENCE_LIMITATION,
    }
    _compare(
        {k: metrics[k] for k in qualification if k in metrics},
        qualification,
        "evidence",
        exact=True,
    )
    p = _checked_panel(panel, calendar)
    common = p[p.model == "contextual"].set_index("origin")
    _support(common, calendar)
    rows = []
    for control in CONTROLS:
        reference = p[p.model == control].set_index("origin")
        phases = []
        for code, phase in enumerate(PHASES):
            candidate = common[common.phase == phase]
            other = reference.loc[candidate.index]
            d = candidate.loss.to_numpy() - other.loss.to_numpy()
            a, b = protocol["forecast"][phase]
            full = calendar[(calendar >= a) & (calendar <= b)]
            where = full.get_indexer(candidate.index)
            selected = np.zeros(len(full), dtype=bool)
            selected[where] = True
            values = np.full(len(full), np.nan)
            values[where] = d
            inference = _indexed_inference(
                values,
                selected,
                blocks=[21, 63, 126],
                hac_lags=126,
                draws=399999,
                seed=20261003 + code * 10000,
            )
            fitted = candidate.gate_status.eq("fitted").to_numpy()
            intervals = [
                inference["hac"]["ci95"],
                *[v["ci95"] for v in inference["block_inference"].values()],
            ]

            def group(mask, differences=d, fitted=fitted):
                n = int(mask.sum())
                trained = int(fitted[mask].sum())
                return {
                    "n": n,
                    "mean": float(differences[mask].mean()) if n else None,
                    "fitted_gate_n": trained,
                    "cold_start_n": n - trained,
                }

            offsets = calendar.get_indexer(candidate.index) % 5
            inference |= {
                "name": phase,
                "fitted_gate_n": int(fitted.sum()),
                "cold_start_n": int((~fitted).sum()),
                "candidate_loss": float(candidate.loss.to_numpy().mean()),
                "control_loss": float(other.loss.to_numpy().mean()),
                "first_origin": str(candidate.index[0].date()),
                "last_origin": str(candidate.index[-1].date()),
                "ci95_envelope": [min(v[0] for v in intervals), max(v[1] for v in intervals)],
                "offsets": [{"offset": k} | group(offsets == k) for k in range(5)],
                "stability": [
                    {"start": start, "end": end}
                    | group((candidate.index >= start) & (candidate.index <= end))
                    for start, end in protocol["forecast"]["stability"]
                ]
                if phase == "evaluation"
                else [],
                "annual": [
                    {
                        "year": int(year),
                        "calendar_origins": int((full.year == year).sum()),
                        "missing_origins": int((full.year == year).sum())
                        - int((candidate.index.year == year).sum()),
                    }
                    | group(candidate.index.year == year)
                    for year in sorted(set(full.year))
                ],
            }
            phases.append(inference)
        rows.append(
            {
                "study": "precision_gate",
                "candidate": "contextual",
                "control": control,
                "horizon": 5,
                "score": "qlike",
                "phases": phases,
                "p_conservative": max(r["p_conservative"] for r in phases),
            }
        )
    new = [r["p_conservative"] for r in rows]
    wave, cumulative = _holm(new), _holm(old + new)[-3:]
    for row, pw, pc in zip(rows, wave, cumulative, strict=True):
        passes = pw <= 0.05 / (26 * 27) and pc <= 0.05
        for phase in row["phases"]:
            passes = passes and phase["mean"] <= -0.005
            passes = passes and all(
                item["mean"] < 0 for item in phase["offsets"] + phase["stability"]
            )
        row |= {
            "p_holm_wave": float(pw),
            "p_holm_cumulative": float(pc),
            "verdict": "COMPARISON_GATE_PASS" if passes else "DOES_NOT_QUALIFY",
        }
    expected = {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "leads": ["precision_gate"]
        if all(r["verdict"] == "COMPARISON_GATE_PASS" for r in rows)
        else [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 149,
        "common_scored_origins": len(common),
        "fitted_gate_scored_origins": int(common.gate_status.eq("fitted").sum()),
    } | qualification
    _compare(metrics, expected)
    return {
        "status": "VERIFIED",
        "comparisons_verified": 3,
        "endpoints_verified": 6,
        "bootstrap_runs_verified": 18,
        "hac_runs_verified": 6,
        "inherited_comparisons_verified": 146,
        "common_scored_origins": len(common),
        "fitted_gate_scored_origins": expected["fitted_gate_scored_origins"],
    }
