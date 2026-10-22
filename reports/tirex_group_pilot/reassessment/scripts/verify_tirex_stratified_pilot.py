"""Internal numerical check using a second implementation, not peer review."""

import numpy as np
from scipy.stats import t as student


def independent_mean_test(values, tau):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or not np.isfinite(tau)
        or tau < 0
    ):
        raise ValueError("need finite vector n>=2 and nonnegative finite tau")
    n = len(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    if se == 0:
        p = 0.0 if abs(mean) > tau else 1.0
    else:
        statistic = (abs(mean) - tau) / se
        p = float(min(1, 2 * student.sf(statistic, df=n - 1)))
    return {"n": n, "mean": mean, "se": se, "p": p}


def independent_holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
    ):
        raise ValueError("pvalues must be finite probabilities")
    ordered = np.argsort(values, kind="stable")
    output = np.empty(len(values))
    previous = 0.0
    for position, index in enumerate(ordered):
        previous = max(
            previous, min(1.0, float(values[index] * (len(values) - position)))
        )
        output[index] = previous
    return output


def independent_tau(baseline, comparisons):
    baseline = np.asarray(baseline, dtype=np.float32)
    if not np.isfinite(baseline).all():
        raise ValueError("baseline nonfinite")
    maximum = 0.0
    for before, after in comparisons:
        before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
        if (
            before.shape != after.shape
            or not np.isfinite(before).all()
            or not np.isfinite(after).all()
        ):
            raise ValueError("comparison invalid")
        maximum = max(maximum, float(np.max(np.abs(after - before))))
    spacing = float(np.max(np.abs(np.spacing(baseline))))
    return {
        "tau": 10 * max(maximum, spacing),
        "maximum_jitter": maximum,
        "maximum_float32_spacing": spacing,
    }


def independent_readout(calibration_feature, calibration_actual, feature, actual):
    cf, ca = (
        np.asarray(calibration_feature, dtype=float),
        np.asarray(calibration_actual, dtype=float),
    )
    centered = cf - cf.mean()
    slope = float(np.sum(centered * (ca - ca.mean())) / np.sum(centered**2))
    intercept = float(ca.mean() - slope * cf.mean())
    prediction = intercept + slope * np.asarray(feature, dtype=float)
    actual = np.asarray(actual, dtype=float)
    actual_diff = actual[:, 1] - actual[:, 0]
    predict_diff = prediction[:, 1] - prediction[:, 0]

    def r2(y, estimate):
        total = np.sum((y - y.mean()) ** 2)
        return None if total == 0 else float(1 - np.sum((y - estimate) ** 2) / total)

    return {
        "coefficients": [intercept, slope],
        "n_pairs": len(actual),
        "level_r2": r2(actual, prediction),
        "paired_difference_r2": r2(actual_diff, predict_diff),
        "level_rmse": float(np.sqrt(np.mean((actual - prediction) ** 2))),
        "paired_difference_rmse": float(
            np.sqrt(np.mean((actual_diff - predict_diff) ** 2))
        ),
    }


class Checks:
    def __init__(self):
        self.count = 0

    def truth(self, value, label):
        if not bool(value):
            raise AssertionError(label)
        self.count += 1

    def equal(self, actual, expected, label):
        np.testing.assert_allclose(
            actual, expected, rtol=1e-11, atol=1e-15, err_msg=label
        )
        self.count += 1


def _sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path):
    import json

    return json.loads(path.read_text())


def _array_file(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _align(values, source, target):
    source, target = list(source), list(target)
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError("duplicate pair IDs")
    mapping = {name: index for index, name in enumerate(source)}
    return values[[mapping[name] for name in target]]


def _descriptive(base, swapped, acts, units):
    change = np.asarray(swapped, dtype=float) - np.asarray(base, dtype=float)
    displacement = acts[:, 1, units].astype(float) - acts[:, 0, units].astype(float)
    return {
        "n_pairs": len(base),
        "mean_e": float(((change[:, 0, 1] - change[:, 1, 1]) / 2).mean()),
        "mean_abs_median_shift": float(np.abs(change[:, :, 1]).mean()),
        "mean_recipient_quantile_change": change[:, 0].mean(axis=0),
        "mean_donor_quantile_change": change[:, 1].mean(axis=0),
        "mean_abs_width_change": float(
            np.abs(change[:, :, 2] - change[:, :, 0]).mean()
        ),
        "input_displacement_rms": float(
            np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
        ),
        "mean_native_median_contrast": float(
            (base[:, 1, 1].astype(float) - base[:, 0, 1]).mean()
        ),
        "forecast_crossing_count": int(
            np.any(np.diff(swapped, axis=2) < 0, axis=2).sum()
        ),
        "base_crossing_count": int(np.any(np.diff(base, axis=2) < 0, axis=2).sum()),
    }


def _verify_stats(check, base, swapped, saved, tau):
    from scipy.stats import binomtest

    delta = swapped.astype(float) - base.astype(float)
    qe = (delta[:, 0] - delta[:, 1]) / 2
    effect = qe[:, 1]
    independently = independent_mean_test(effect, tau)
    for name, value in [
        ("n_pairs", len(base)),
        ("deadband", tau),
        ("primary_pvalue", independently["p"]),
        ("mean_e", effect.mean()),
        ("mean_signed", effect.mean()),
        ("mean_e_sd", effect.std(ddof=1)),
        ("mean_e_se", independently["se"]),
        ("e", effect),
        ("recipient_median_delta", delta[:, 0, 1]),
        ("donor_median_delta", delta[:, 1, 1]),
        ("mean_recipient_median_shift", delta[:, 0, 1].mean()),
        ("mean_donor_median_shift", delta[:, 1, 1].mean()),
        ("mean_abs_median_shift", np.abs(delta[:, :, 1]).mean()),
        (
            "mean_native_donor_recipient_contrast",
            (base[:, 1, 1].astype(float) - base[:, 0, 1]).mean(),
        ),
        ("mean_quantile_e", qe.mean(axis=0)),
        ("mean_recipient_quantile_shift", delta[:, 0].mean(axis=0)),
        ("mean_donor_quantile_shift", delta[:, 1].mean(axis=0)),
        ("base_crossing_count", np.any(np.diff(base, axis=-1) < 0, axis=-1).sum()),
        ("crossing_count", np.any(np.diff(swapped, axis=-1) < 0, axis=-1).sum()),
    ]:
        check.equal(saved[name], value, name)
    pos, neg = int((effect > tau).sum()), int((effect < -tau).sum())
    check.truth(
        saved["sign_counts"]
        == {
            "positive": pos,
            "negative": neg,
            "neutral": len(effect) - pos - neg,
            "nonneutral": pos + neg,
            "total": len(effect),
        },
        "sign counts",
    )
    check.equal(
        saved["sign_pvalue"],
        binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0,
        "sign p",
    )
    width = delta[:, :, 2] - delta[:, :, 0]
    we = (width[:, 0] - width[:, 1]) / 2
    for name, value in [
        ("mean_width_e", we.mean()),
        ("mean_recipient_width_shift", width[:, 0].mean()),
        ("mean_donor_width_shift", width[:, 1].mean()),
        ("mean_abs_width_shift", np.abs(width).mean()),
    ]:
        check.equal(saved[name], value, name)
    indices = np.random.default_rng(saved["seed"]).integers(
        0, len(base), size=(saved["bootstrap_replicates"], len(base))
    )
    for name, values in [
        ("mean_e_ci95", effect),
        ("recipient_median_ci95", delta[:, 0, 1]),
        ("donor_median_ci95", delta[:, 1, 1]),
        ("abs_median_ci95", np.abs(delta[:, :, 1]).mean(axis=1)),
        ("quantile_e_ci95", qe),
        ("recipient_quantile_ci95", delta[:, 0]),
        ("donor_quantile_ci95", delta[:, 1]),
        ("width_e_ci95", we),
        ("recipient_width_ci95", width[:, 0]),
        ("donor_width_ci95", width[:, 1]),
        ("abs_width_ci95", np.abs(width).mean(axis=1)),
    ]:
        expected = np.percentile(
            np.mean(values[indices], axis=1), [2.5, 97.5], axis=0
        ).T
        check.equal(saved[name], expected, name)
    return independently["p"]


def verify(run, source_root, prior_runs):
    import hashlib

    from scipy.stats import norm

    c = Checks()
    protocol = _json(run / "protocol.json")
    manifest = _json(run / "manifest.json")
    for name, digest in manifest.items():
        c.truth(_sha(run / name) == digest, f"manifest {name}")
    pre = _json(run / "pre_execution.json")
    for name, digest in pre["source_hashes"].items():
        c.truth(_sha(source_root / name) == digest, f"source hash {name}")
    c.truth(
        _sha(run / "protocol.json") == pre["protocol_sha256"], "protocol frozen hash"
    )
    initial = _json(run / "intervention_lock.json")
    lock = _json(run / "validation_lock.json")
    summary = _json(run / "summary.json")
    metadata = _json(run / "model_metadata.json")
    c.truth(
        metadata["checkpoint_revision"] == protocol["checkpoint_revision"],
        "checkpoint revision",
    )
    c.truth(
        metadata["context_length"] == 2048
        and metadata["token_index"] == 63
        and metadata["unit_count"] == 1408,
        "model mechanics",
    )
    for receipt_name in ("instrumentation.json", "group_instrumentation.json"):
        receipt = _json(run / receipt_name)
        for row in receipt["checks"]:
            c.truth(row["passed"], "instrumentation reported pass")
            c.truth(
                row["max_forecast_abs_error"]
                <= protocol["instrumentation"]["max_forecast_abs_error"],
                "instrumentation forecast tolerance",
            )
            for error in row["activation_relative_errors"].values():
                c.truth(
                    error
                    <= protocol["instrumentation"]["max_activation_relative_error"],
                    "instrumentation activation tolerance",
                )
    c.truth(initial["interventions"] == protocol["interventions"], "frozen identities")
    c.truth(
        _sha(run / "intervention_lock.json") == lock["intervention_lock_sha256"],
        "identity lock hash",
    )
    c.truth(
        _sha(run / "validation_lock.json") == summary["validation_lock_sha256"],
        "validation lock hash",
    )
    c.truth(
        pre["timestamp_before_model_load"]
        <= initial["timestamp_before_calibration"]
        <= lock["timestamp_before_validation"]
        <= summary["finished_utc"],
        "declared chronology",
    )
    c.truth(
        _sha(run / "numerical_calibration.json")
        == lock["numerical_calibration_sha256"],
        "numeric lock hash",
    )
    baseline = {}
    families = ("persistence", "recency", "recovery")
    variants = ("short", "long")
    input_hashes = {}
    ids_by_split = {}
    for split in ("calibration", "validation"):
        hashes, ids = set(), set()
        expected = protocol[f"{split}_pairs_per_cell"]
        for family in families:
            for variant in variants:
                path = run / f"{split}_{family}_{variant}.npz"
                data = _array_file(path)
                baseline[split, family, variant] = data
                c.truth(len(data["pair_id"]) == expected, "balanced cell size")
                c.truth(
                    np.all(data["family"] == family)
                    and np.all(data["variant"] == variant),
                    "cell metadata",
                )
                c.truth(
                    np.all(data["window"] == (76 if variant == "short" else 180)),
                    "motif length",
                )
                c.truth(
                    np.all((data["amplitude"] >= 0.8) & (data["amplitude"] <= 2.8)),
                    "amplitude support",
                )
                c.truth(
                    np.all((data["noise_sd"] >= 0.04) & (data["noise_sd"] <= 0.3)),
                    "noise support",
                )
                c.truth(
                    np.array_equal(np.sort(data["recipient"]), np.sort(data["donor"])),
                    "identical input histograms",
                )
                c.truth(
                    np.array_equal(data["recipient"][:, -1], data["donor"][:, -1]),
                    "identical terminal input",
                )
                for side in ("recipient", "donor"):
                    c.equal(data[side].mean(axis=1), -9, "input location")
                    c.equal(data[side].std(axis=1), 0.6, "input scale")
                    for row in data[side]:
                        digest = hashlib.sha256(
                            row.astype(np.float32).tobytes()
                        ).hexdigest()
                        c.truth(digest not in hashes, "unique contexts within split")
                        hashes.add(digest)
                for identity in data["pair_id"]:
                    c.truth(identity not in ids, "unique IDs within split")
                    ids.add(identity)
                c.truth(
                    data["quantiles"].shape == (expected, 2, 3),
                    "baseline forecast axes",
                )
                c.truth(np.isfinite(data["quantiles"]).all(), "finite baseline")
                if split == "calibration":
                    c.truth(
                        _sha(path) == lock["calibration_file_hashes"][path.name],
                        "calibration file lock",
                    )
        input_hashes[split], ids_by_split[split] = hashes, ids
    c.truth(
        not (input_hashes["calibration"] & input_hashes["validation"]),
        "fresh calibration versus validation",
    )
    c.truth(
        not (ids_by_split["calibration"] & ids_by_split["validation"]),
        "distinct calibration/validation IDs",
    )
    for prior in prior_runs:
        for split in ("discovery", "validation"):
            old = _array_file(prior / f"{split}.npz")
            oldhash = {
                hashlib.sha256(row.astype(np.float32).tobytes()).hexdigest()
                for side in ("recipient", "donor")
                for row in old[side]
            }
            for new in ("calibration", "validation"):
                c.truth(
                    not (input_hashes[new] & oldhash), "fresh versus prior contexts"
                )
                c.truth(
                    not (ids_by_split[new] & set(old["pair_id"])),
                    "fresh versus prior IDs",
                )
    numeric = _json(run / "numerical_calibration.json")
    nulls = _array_file(run / "calibration_nulls.npz")
    nullbase = np.concatenate(
        [baseline["calibration", f, v]["quantiles"] for f in families for v in variants]
    ).reshape(-1, 3)
    nullids = np.concatenate(
        [baseline["calibration", f, v]["pair_id"] for f in families for v in variants]
    )
    c.truth(np.array_equal(nullids, nulls["pair_id"]), "null pair IDs")
    c.equal(nulls["baseline_quantiles"], nullbase, "null baseline from saved cells")
    comparisons = [(nullbase, nulls[name]) for name in numeric["null_max_abs_errors"]]
    rebuilt = independent_tau(nullbase, comparisons)
    tau = rebuilt["tau"]
    c.equal(numeric["deadband"], tau, "reconstructed tau")
    c.equal(lock["tau"], tau, "frozen tau")
    c.equal(summary["tau"], tau, "summary tau")
    c.equal(numeric["observed_max_jitter"], rebuilt["maximum_jitter"], "maximum jitter")
    c.equal(
        numeric["max_float32_spacing"],
        rebuilt["maximum_float32_spacing"],
        "float32 spacing",
    )
    for name, value in numeric["null_max_abs_errors"].items():
        c.equal(
            value,
            np.abs(nulls[name].astype(float) - nullbase).max(),
            "named null error",
        )
    # One-feature surrogate is checked by a closed-form covariance fit instead
    # of the producer's least-squares solve, then scored without refitting.
    ca = np.concatenate(
        [baseline["calibration", "recency", v]["layer_0"][:, :, 666] for v in variants]
    )
    cf = np.concatenate(
        [
            np.stack(
                (
                    baseline["calibration", "recency", v]["recipient"],
                    baseline["calibration", "recency", v]["donor"],
                ),
                axis=1,
            )[:, :, -32:].mean(axis=-1)
            for v in variants
        ]
    )
    readout = _json(run / "recent_mean_diagnostic.json")
    c.truth(
        readout["frozen_model"] == lock["recent_mean_readout"], "frozen surrogate copy"
    )
    for variant in ("all", "short", "long"):
        choices = variants if variant == "all" else (variant,)
        a = np.concatenate(
            [
                baseline["validation", "recency", v]["layer_0"][:, :, 666]
                for v in choices
            ]
        )
        f = np.concatenate(
            [
                np.stack(
                    (
                        baseline["validation", "recency", v]["recipient"],
                        baseline["validation", "recency", v]["donor"],
                    ),
                    axis=1,
                )[:, :, -32:].mean(axis=-1)
                for v in choices
            ]
        )
        rebuilt_readout = independent_readout(cf, ca, f, a)
        c.equal(
            readout["frozen_model"]["coefficients"],
            rebuilt_readout.pop("coefficients"),
            "closed-form surrogate fit",
        )
        for name, value in rebuilt_readout.items():
            if value is None:
                c.truth(
                    readout["validation"][variant][name] is None,
                    "undefined surrogate R2",
                )
            else:
                c.equal(
                    readout["validation"][variant][name],
                    value,
                    "surrogate score " + name,
                )
    primary = _json(run / "primary_summary.json")["claims"]
    c.truth(primary == summary["primary"], "primary summary copies")
    c.truth(len(primary) == 18, "exactly eighteen primary claims")
    independent_p = []
    groups = {g["group_id"]: g for g in protocol["interventions"]}
    for index, group in enumerate(protocol["interventions"]):
        gid, family = group["group_id"], group["family"]
        for split in ("calibration", "validation"):
            swap = _array_file(run / f"{split}_swap_{gid}.npz")
            c.truth(np.array_equal(swap["units"], group["units"]), "intervention units")
            for cell, variant in enumerate(variants):
                data = baseline[split, family, variant]
                q = _align(swap["quantiles"], swap["pair_id"], data["pair_id"])
                if split == "calibration":
                    saved = lock["calibration"][gid][variant]["summary"]
                    c.truth(
                        saved["seed"] == protocol["seeds"]["bootstrap"] + index,
                        "calibration bootstrap seed",
                    )
                else:
                    item = primary[2 * index + cell]
                    c.truth(
                        (item["group_id"], item["variant"]) == (gid, variant),
                        "claim order",
                    )
                    c.truth(
                        item["calibration"] == lock["calibration"][gid][variant],
                        "reported calibration cell copy",
                    )
                    saved = item["summary"]
                    c.truth(
                        saved["seed"]
                        == protocol["seeds"]["bootstrap"] + 100 + 2 * index + cell,
                        "validation bootstrap seed",
                    )
                    for name, value in _descriptive(
                        data["quantiles"],
                        q,
                        data[f"layer_{group['layer']}"],
                        group["units"],
                    ).items():
                        c.equal(
                            item["description"][name],
                            value,
                            "primary description " + name,
                        )
                p = _verify_stats(c, data["quantiles"], q, saved, tau)
                if split == "validation":
                    independent_p.append(p)
                else:
                    sd = np.std(
                        (
                            q[:, 0, 1].astype(float)
                            - data["quantiles"][:, 0, 1]
                            - q[:, 1, 1].astype(float)
                            + data["quantiles"][:, 1, 1]
                        )
                        / 2,
                        ddof=1,
                    )
                    mde = tau + (
                        norm.ppf(1 - 0.05 / 36) + norm.ppf(0.8)
                    ) * sd / np.sqrt(protocol["validation_pairs_per_cell"])
                    c.equal(
                        lock["calibration"][gid][variant]["mde80"],
                        mde,
                        "calibration MDE",
                    )
    adjusted = independent_holm(independent_p)
    for item, pvalue in zip(primary, adjusted, strict=True):
        c.equal(item["holm_pvalue"], pvalue, "Holm p")
        qualify = (
            pvalue < protocol["primary"]["alpha"]
            and item["summary"]["crossing_count"] == 0
            and item["summary"]["base_crossing_count"] == 0
        )
        c.truth(
            item["numerical_influence_detected"] == qualify, "numerical influence label"
        )
    # Reconstruct random reference membership directly from seed and exclusions.
    rng = np.random.default_rng(protocol["seeds"]["random_groups"])
    rebuilt_references = []
    for group in [
        g for g in protocol["interventions"] if g["group_id"].endswith("_k8")
    ]:
        excluded = set(group["units"]) | ({103} if group["layer"] == 0 else set())
        allowed = [unit for unit in range(1408) if unit not in excluded]
        for draw in range(protocol["random_reference"]["draws"]):
            units = rng.choice(allowed, size=8, replace=False).tolist()
            for size in (4, 8):
                rebuilt_references.append(
                    {
                        "group_id": f"random_{group['family']}_{draw:02d}_k{size}",
                        "family": group["family"],
                        "layer": group["layer"],
                        "units": units[:size],
                        "draw": draw,
                        "size": size,
                    }
                )
    c.truth(
        rebuilt_references == lock["random_reference"],
        "random membership seed reconstruction",
    )
    references = _json(run / "random_reference_summary.json")
    for row in references["references"]:
        group = row["reference"]
        c.truth(group in rebuilt_references, "saved reference membership")
        q = _array_file(run / f"validation_{group['group_id']}.npz")
        for variant in variants:
            data = baseline["validation", group["family"], variant]
            n = protocol["dose_response"]["pairs_per_cell"]
            forecast = _align(q["quantiles"], q["pair_id"], data["pair_id"][:n])
            for name, value in _descriptive(
                data["quantiles"][:n],
                forecast,
                data[f"layer_{group['layer']}"][:n],
                group["units"],
            ).items():
                c.equal(
                    row["by_variant"][variant][name],
                    value,
                    "reference description " + name,
                )
    for row in references["candidates_on_identical_subset"]:
        group = groups[row["group_id"]]
        q = _array_file(run / f"validation_swap_{group['group_id']}.npz")
        for variant in variants:
            data = baseline["validation", group["family"], variant]
            n = protocol["dose_response"]["pairs_per_cell"]
            forecast = _align(q["quantiles"], q["pair_id"], data["pair_id"][:n])
            for name, value in _descriptive(
                data["quantiles"][:n],
                forecast,
                data[f"layer_{group['layer']}"][:n],
                group["units"],
            ).items():
                c.equal(
                    row["by_variant"][variant][name],
                    value,
                    "same subset candidate " + name,
                )
    dose = _json(run / "dose_summary.json")["results"]
    dose_data = {
        (row["group_id"], row["alpha"]): _array_file(run / row["filename"])
        for row in dose
    }
    c.truth(len(dose_data) == 44 and len(dose) == 44, "all dose combinations")
    n = protocol["dose_response"]["pairs_per_cell"]
    db = np.concatenate(
        [baseline["validation", "recovery", v]["quantiles"][:n] for v in variants]
    ).astype(float)
    d_ids = np.concatenate(
        [baseline["validation", "recovery", v]["pair_id"][:n] for v in variants]
    )
    for row in dose:
        gid, alpha = row["group_id"], row["alpha"]
        saved = dose_data[gid, alpha]
        c.truth(np.array_equal(saved["pair_id"], d_ids), "dose subset pair alignment")
        c.equal(saved["alpha"], alpha, "dose alpha")
        expected_units = (
            groups[gid]["units"] if gid in groups else [int(gid.split("_U")[1])]
        )
        c.truth(saved["units"].tolist() == expected_units, "dose unit identities")
        c.truth(alpha in protocol["dose_response"]["alphas"], "registered dose")
        delta = saved["quantiles"].astype(float) - db
        unit = dose_data[gid, 1]["quantiles"].astype(float) - db
        curve = delta - alpha * unit
        for name, value in [
            ("mean_abs_median_curvature", np.abs(curve[:, :, 1]).mean()),
            ("mean_recipient_median_curvature", curve[:, 0, 1].mean()),
            ("mean_donor_median_curvature", curve[:, 1, 1].mean()),
        ]:
            c.equal(row[name], value, "dose " + name)
        if len(saved["units"]) > 1:
            summed = sum(
                dose_data[f"recovery_U{u}", alpha]["quantiles"].astype(float) - db
                for u in saved["units"]
            )
            residual = delta - summed
            for name, value in [
                ("mean_abs_joint_minus_sum_median", np.abs(residual[:, :, 1]).mean()),
                ("mean_recipient_joint_minus_sum_median", residual[:, 0, 1].mean()),
                ("mean_donor_joint_minus_sum_median", residual[:, 1, 1].mean()),
            ]:
                c.equal(row[name], value, "dose " + name)
        if row["reused_primary"]:
            original = _array_file(run / f"validation_swap_{gid}.npz")
            c.equal(
                saved["quantiles"],
                _align(original["quantiles"], original["pair_id"], d_ids),
                "unit dose primary reuse",
            )
        for i, variant in enumerate(variants):
            data = baseline["validation", "recovery", variant]
            for name, value in _descriptive(
                db[i * n : (i + 1) * n],
                saved["quantiles"][i * n : (i + 1) * n],
                data["layer_5"][:n],
                saved["units"].tolist(),
            ).items():
                c.equal(
                    row["by_variant"][variant][name], value, "dose description " + name
                )
    execution = _json(run / "execution_records.json")
    c.truth(
        [row["completed"] for row in execution] == list(range(1, len(execution) + 1)),
        "execution sequence",
    )
    for record in execution:
        c.truth((run / record["filename"]).exists(), "executed archive exists")
        c.truth(
            lock["timestamp_before_validation"]
            <= record["timestamp"]
            <= summary["finished_utc"],
            "validation execution chronology",
        )
    return {
        "passed": True,
        "checks": c.count,
        "verification_scope": "Internal numerical check by second implementation; no independent scientific replication or peer review",
        "tau": tau,
        "n_primary_claims": len(primary),
        "n_primary_detected": sum(
            row["numerical_influence_detected"] for row in primary
        ),
        "checked": [
            "all manifest/source hashes",
            "frozen identities and threshold chronology",
            "all fresh contexts and paired histograms",
            "tau from saved no-op/repeat/batch arrays",
            "all primary and calibration statistics with bootstrap intervals",
            "calibration MDE",
            "recent-mean surrogate fit and held-out scores",
            "Holm18 labels",
            "seeded random reference identities and raw descriptions",
            "dose curvature and joint-minus-singleton calculations",
        ],
        "limitations": [
            "Does not rerun the model",
            "Does not by itself verify all inference code paths",
        ],
    }


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_dir, args.source_root, args.prior_run)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
