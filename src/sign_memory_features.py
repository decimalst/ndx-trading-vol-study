"""Source-gated strict raw intraday sign agreement on the full SPX calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import joint_risk_features as joint

ROOT = joint.ROOT
OLD_FEATURES = joint.ALL_FEATURES
BOUNDED = ("qqq_pos22", "qqq_neg22", "spx_pos22", "spx_neg22", "independent22")
MEMORY = "excess22"
ALL_FEATURES = OLD_FEATURES + BOUNDED + (MEMORY,)
MODELS = ("frequency", "baseline", "memory")
SIGN_COLUMNS = ("qqq_pos", "qqq_neg", "spx_pos", "spx_neg", "agreement")
measurement_audit = joint.measurement_audit
require_measurement = joint.require_measurement


def sign_observations(qqq_returns, spx_returns):
    """Return n×5 float indicators in SIGN_COLUMNS order from aligned 1D arrays.

    Input order is positional. A NaN in either return makes all five indicators
    unknown. Observed infinities, complex returns, or unequal shapes reject.
    Every exact zero, including negative zero, is a valid nonagreement. Direct
    sign comparisons preserve subnormal return signs without multiplication.
    """
    if np.iscomplexobj(qqq_returns) or np.iscomplexobj(spx_returns):
        raise ValueError("Real paired raw returns required")
    try:
        qqq = np.asarray(qqq_returns, dtype=float)
        spx = np.asarray(spx_returns, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Numeric paired raw returns required") from error
    if qqq.ndim != 1 or spx.shape != qqq.shape:
        raise ValueError("Equal-length one-dimensional paired returns required")
    if np.isinf(qqq).any() or np.isinf(spx).any():
        raise ValueError("Observed paired returns must be finite")
    observed = np.isfinite(qqq) & np.isfinite(spx)
    qpos, qneg, spos, sneg = qqq > 0, qqq < 0, spx > 0, spx < 0
    result = np.column_stack((qpos, qneg, spos, sneg, (qpos & spos) | (qneg & sneg))).astype(float)
    result[~observed] = np.nan
    return result


def build_features(qqq, spx, iv):
    """Preserve old controls and add strict paired22 sign summaries ending t−1.

    The frozen joint builder enforces the full-source measurement gate before
    any new sign construction. No source calendar intersection, target-driven
    masking, filling, or global training centering occurs here.
    """
    features, paired_targets = joint.build_features(qqq, spx, iv)
    reference = spx.index
    observations = pd.DataFrame(
        sign_observations(
            joint._signed_intraday(qqq.reindex(reference)).to_numpy(),
            joint._signed_intraday(spx).to_numpy(),
        ),
        index=reference,
        columns=SIGN_COLUMNS,
    )
    fractions = observations.rolling(22, min_periods=22).mean().shift(1)
    for feature, column in zip(BOUNDED[:4], SIGN_COLUMNS[:4], strict=True):
        features[feature] = fractions[column]
    features["independent22"] = (
        fractions.qqq_pos * fractions.spx_pos + fractions.qqq_neg * fractions.spx_neg
    )
    features[MEMORY] = fractions.agreement - features.independent22
    targets = pd.DataFrame(index=reference)
    targets["y"] = observations.agreement.shift(-1)
    targets["target_end"] = paired_targets.target_end
    targets["available_date"] = paired_targets.available_date
    return features.loc[:, ALL_FEATURES + ("feature_cutoff_date",)], targets


def load_sources(protocol, root=ROOT):
    """Reuse exact frozen source loading; source audit contains no new sign counts."""
    qqq, spx, iv, old = joint.load_sources(protocol, root)
    audit = {
        **old,
        "raw_columns": list(ALL_FEATURES),
        "target_interpretation": "strict same-sign next observed SPX session raw log(close/open) returns for QQQ ETF and SPX price index; any exact zero is nonagreement, missing pair unknown; directional agreement, not volatility magnitude or covariance",
    }
    return qqq, spx, iv, audit
