"""Source-gated relative QQQ ETF versus SPX index intraday risk measurements.

The full reference-calendar measurement gate precedes every numerical feature
and label build. Fitting, registration, and scoring belong to the caller.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import index_hinge

ROOT = index_hinge.ROOT
BASE = ("const", *[f"{asset}_{kind}_{suffix}" for asset in ("qqq", "spx")
                  for kind in ("lg", "lt", "neg") for suffix in ("d", "w", "m")],
        "lvxn", "lvix", "term", "lvvix", *[f"entry_dow_{day}" for day in range(1, 5)])
ALL_FEATURES = BASE+("corr22",)
MODELS = ("mean", "baseline", "correlation")
GK_FLOOR = 1e-10
CORRELATION_TOLERANCE = 1e-12


def _raw_gk(daily):
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        return .5*np.log(daily.high/daily.low)**2-(2*np.log(2)-1)*np.log(daily.close/daily.open)**2


def measurement_audit(qqq, spx):
    """Audit every observed paired-period OHLC measurement before any row mask.

    Missing source observations remain unknown. Finite values at or below the
    existing GK floor fail the entire measurement family; they are not clipped
    or silently removed. Invalid OHLC and nonfinite computed GK are errors.
    """
    for frame in (qqq, spx):
        index_hinge._daily_contract(frame)
    if spx.empty:
        raise ValueError("INSUFFICIENT_DATA: empty SPX reference calendar")
    audit = {"status": "PASS", "gk_floor": GK_FLOOR, "reference_rows": len(spx), "per_asset": {}}
    for name, source in (("qqq", qqq), ("spx", spx)):
        aligned = source.reindex(spx.index)
        observed = aligned.loc[:, index_hinge.OHLC].notna().all(axis=1)
        raw = _raw_gk(aligned.loc[observed])
        if not np.isfinite(raw).all():
            raise ValueError(f"Nonfinite computed raw GK measurement for {name}")
        floor_hits = int((raw <= GK_FLOOR).sum())
        entry = {"observed_complete_rows": int(observed.sum()),
                 "missing_ohlc_rows": int((~observed).sum()), "finite_gk_rows": len(raw),
                 "floor_hit_rows": floor_hits, "nonpositive_gk_rows": int((raw <= 0).sum()),
                 "small_positive_gk_rows": int(((raw > 0) & (raw <= GK_FLOOR)).sum()),
                 "minimum_raw_gk": float(raw.min()) if len(raw) else None}
        audit["per_asset"][name] = entry
        if floor_hits or not len(raw):
            audit["status"] = "INSUFFICIENT_MEASUREMENT"
    return audit


def require_measurement(audit):
    """Fail after the caller has had the opportunity to persist the audit."""
    valid = audit.get("status") == "PASS" and audit.get("gk_floor") == GK_FLOOR
    for asset in ("qqq", "spx"):
        one = audit.get("per_asset", {}).get(asset, {})
        minimum = one.get("minimum_raw_gk")
        valid = (valid and one.get("finite_gk_rows", 0) > 0 and one.get("floor_hit_rows") == 0
                 and minimum is not None and np.isfinite(minimum) and minimum > GK_FLOOR)
    if not valid:
        raise ValueError("INSUFFICIENT_DATA: full-reference raw GK measurement gate failed; no clipping or row filtering")


def _correlation22(qqq_day, spx_day):
    """Strict centered two-pass Pearson correlation on complete paired windows."""
    if not qqq_day.index.equals(spx_day.index):
        raise ValueError("Correlation inputs must share the full reference calendar")
    pair = np.column_stack([qqq_day.to_numpy(float), spx_day.to_numpy(float)])
    values = np.full(len(pair), np.nan)
    for end in range(21, len(pair)):
        block = pair[end-21:end+1]
        if not np.isfinite(block).all():
            continue
        # Exact constants can leave a rounding residual after mean-centering.
        # Detect them in the original values; no variance epsilon is introduced.
        if (block == block[0]).all(axis=0).any():
            continue
        centered = block-block.mean(axis=0)
        a, b = centered[:, 0], centered[:, 1]
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        if denominator == 0:
            continue
        if not np.isfinite(denominator):
            raise ValueError("Nonfinite correlation denominator")
        correlation = float(np.dot(a, b)/denominator)
        if not np.isfinite(correlation) or abs(correlation) > 1+CORRELATION_TOLERANCE:
            raise ValueError("Invalid correlation beyond fixed floating-point tolerance")
        values[end] = correlation
    return pd.Series(values, index=qqq_day.index, name="corr22")


def build_features(qqq, spx, iv):
    """Build the declared common design only after full-source measurement passes.

    QQQ retains its native source history for predecessor identity. No observed
    SPX reference row is compressed away before rolling. All market predictors
    end at the previous reference session; target dates do not choose features.
    """
    require_measurement(measurement_audit(qqq, spx))
    index_hinge._dates(iv.index)
    index_hinge._positive_market(iv, ("vxn", "vix", "vix9d", "vvix"))
    reference = spx.index
    dates = pd.Series(reference, index=reference)
    previous_reference = dates.shift(1)
    feature, gk, day = pd.DataFrame(index=reference), {}, {}
    for asset, source in (("qqq", qqq), ("spx", spx)):
        aligned = source.reindex(reference)
        previous_source = pd.Series(source.index, index=source.index).shift(1).reindex(reference)
        matching = previous_source.eq(previous_reference) & previous_reference.notna()
        previous_close = source.close.shift(1).reindex(reference).where(matching)
        gk[asset] = _raw_gk(aligned)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            day[asset] = np.log(aligned.close/aligned.open)
            total = gk[asset]+np.log(aligned.open/previous_close)**2
            negative = (-np.log(aligned.close/previous_close)).clip(lower=0)
            for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
                feature[f"{asset}_lg_{suffix}"] = np.log(gk[asset].rolling(width, min_periods=width).mean())
                feature[f"{asset}_lt_{suffix}"] = np.log(total.rolling(width, min_periods=width).mean())
                feature[f"{asset}_neg_{suffix}"] = negative.rolling(width, min_periods=width).mean()
    aligned_iv = iv.reindex(reference)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for output, name in (("lvxn", "vxn"), ("lvix", "vix"), ("lvvix", "vvix")):
            feature[output] = np.log(aligned_iv[name])
        feature["term"] = np.log(aligned_iv.vix9d/aligned_iv.vix)
    feature["corr22"] = _correlation22(day["qqq"], day["spx"])
    feature = feature.shift(1)
    feature["const"] = 1.
    for weekday in range(1, 5):
        feature[f"entry_dow_{weekday}"] = (reference.weekday == weekday).astype(float)
    feature = feature.loc[:, ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
    feature["feature_cutoff_date"] = previous_reference
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        target = (np.log(gk["qqq"])-np.log(gk["spx"])).shift(-1)
    targets = pd.DataFrame({"y": target.where(np.isfinite(target)),
                            "target_end": dates.shift(-1), "available_date": dates.shift(-1)}, index=reference)
    return feature, targets


def load_sources(protocol, root=ROOT):
    """Load only the bounded raw OHLC and four daily IV source columns."""
    root = Path(root)
    spx, iv, audit = index_hinge.load_sources(protocol, root)
    end = pd.Timestamp(protocol["index"]["source_end"])
    qqq_path, vxn_path = root/protocol["sources"]["qqq"], root/protocol["sources"]["vxn"]
    qqq = pd.read_parquet(qqq_path, columns=list(index_hinge.OHLC), filters=[("date", "<=", end)]).loc[:end]
    index_hinge._daily_contract(qqq)
    if qqq.empty:
        raise ValueError("INSUFFICIENT_DATA: empty bounded QQQ source")
    raw = pd.read_csv(vxn_path, usecols=["DATE", "CLOSE"], dtype=str)
    dates = pd.to_datetime(raw.DATE, format="%m/%d/%Y", errors="raise")
    selected = dates <= end
    values = pd.to_numeric(raw.loc[selected, "CLOSE"], errors="raise").to_numpy(float)
    vxn = pd.Series(values, index=pd.DatetimeIndex(dates.loc[selected], name="date"), name="vxn")
    index_hinge._dates(vxn.index)
    index_hinge._positive_market(vxn.to_frame(), ["vxn"])
    if vxn.empty:
        raise ValueError("INSUFFICIENT_DATA: empty bounded VXN source")
    sources = {**audit["sources"], "qqq": index_hinge._source_audit(qqq_path, qqq, spx.index),
               "vxn": {**index_hinge._source_audit(vxn_path, vxn.to_frame(), spx.index),
                       "provider_date_field": "DATE", "provider_value_field": "CLOSE"}}
    audit = {key: value for key, value in audit.items() if key != "gap_interpretation"}
    audit.update({"sources": sources, "raw_columns": list(ALL_FEATURES),
                  "target_interpretation": "next observed session log raw intraday GK of QQQ minus log raw intraday GK of SPX; ETF/index relative proxy, not portfolio variance",
                  "timing_assumption": "market predictors through prior observed SPX session; each raw previous-close calculation requires its source predecessor to equal the reference predecessor",
                  "asset_identity": {"qqq": "QQQ ETF vendor raw OHLC", "spx": "Yahoo ^GSPC price-index raw OHLC"},
                  "measurement_gate": "all finite raw GK on full SPX reference calendar strictly greater than 1e-10 before any feature/label mask"})
    return qqq, spx, pd.concat([iv, vxn], axis=1).sort_index(), audit
