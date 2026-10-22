"""Post-result descriptive sensitivity supplement; no changed decision criteria."""

import argparse
import json
from pathlib import Path

import numpy as np


def effect_diagnostics(base, forecast, pair_ids, tau):
    base, forecast = np.asarray(base, dtype=float), np.asarray(forecast, dtype=float)
    if (
        base.shape != forecast.shape
        or base.ndim != 3
        or base.shape[1:] != (2, 3)
        or len(base) != len(pair_ids)
    ):
        raise ValueError("incompatible forecasts/identifiers")
    if not np.isfinite(base).all() or not np.isfinite(forecast).all():
        raise ValueError("forecasts must be finite")
    delta = forecast[:, :, 1] - base[:, :, 1]
    e = (delta[:, 0] - delta[:, 1]) / 2
    negative = (delta[:, 0] < -tau) & (delta[:, 1] > tau)
    positive = (delta[:, 0] > tau) & (delta[:, 1] < -tau)
    row = {
        "n": len(e),
        "mean_e": float(e.mean()),
        "median_e": float(np.median(e)),
        "negative_raw_paired_effect_count": int((e < 0).sum()),
        "both_negative_raw_orientation_beyond_tau": int(negative.sum()),
        "both_positive_raw_orientation_beyond_tau": int(positive.sum()),
        "mixed_or_one_neutral_beyond_tau": int((~(negative | positive)).sum()),
        "both_recipient_negative_donor_positive_strict_zero": int(
            ((delta[:, 0] < 0) & (delta[:, 1] > 0)).sum()
        ),
        "quantiles_e": dict(
            zip(
                ["min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max"],
                np.quantile(
                    e, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
                ).tolist(),
                strict=True,
            )
        ),
        "recipient_quantiles": np.quantile(
            delta[:, 0], [0, 0.05, 0.5, 0.95, 1]
        ).tolist(),
        "donor_quantiles": np.quantile(delta[:, 1], [0, 0.05, 0.5, 0.95, 1]).tolist(),
    }
    order = np.argsort(np.abs(e), kind="stable")[::-1]
    row["largest_abs_pair_ids"] = [str(pair_ids[i]) for i in order[:5]]
    row["largest_abs_pair_effects"] = e[order[:5]].tolist()
    for count, name in ((1, "one"), (5, "five"), (16, "sixteen")):
        if len(e) > count:
            row[f"mean_without_{name}_largest_abs"] = float(
                np.delete(e, order[:count]).mean()
            )
    row["descriptive_note"] = (
        "Post-result robustness diagnostics only; registered mean includes every pair. Largest-absolute exclusions are sensitivity descriptions, not revised tests or replacement estimates."
    )
    return row


def extract(run):
    primary = json.loads((run / "primary_summary.json").read_text())
    tau = primary["tau"]
    rows = []
    for claim in primary["claims"]:
        gid, family, variant = claim["group_id"], claim["family"], claim["variant"]
        with (
            np.load(run / f"validation_{family}_{variant}.npz") as base,
            np.load(run / f"validation_swap_{gid}.npz") as swap,
        ):
            lookup = {identity: i for i, identity in enumerate(swap["pair_id"])}
            q = swap["quantiles"][[lookup[identity] for identity in base["pair_id"]]]
            rows.append(
                {
                    "group_id": gid,
                    "variant": variant,
                    **effect_diagnostics(base["quantiles"], q, base["pair_id"], tau),
                }
            )
    return {
        "analysis_type": "Explicitly post-result descriptive supplement, not preregistered confirmatory inference",
        "tau": tau,
        "rows": rows,
        "definitions": {
            "e": "(recipient median shift minus donor median shift)/2; raw sign",
            "both_negative_raw_orientation_beyond_tau": "recipient shift < -tau and donor shift > tau",
            "strict_zero": "recipient shift < 0 and donor shift > 0; differs from both shifts exceeding numerical floor",
            "sensitivity_exclusions": "Remove cases in descending absolute e solely to describe tail sensitivity; primary inference retains all cases.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(extract(args.run_dir), indent=2) + "\n")


if __name__ == "__main__":
    main()
