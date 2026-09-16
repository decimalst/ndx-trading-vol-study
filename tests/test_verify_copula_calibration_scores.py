"""Prewritten generated contracts for independent wave28 score verification."""

import copy
import unittest

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, multivariate_t, norm, t

from src.treasury_dealer_inference import masked_mean_inference
from src.verify_copula_calibration_scores import verify_scores

CELLS = ("orig_gaussian", "orig_t8", "cal_gaussian", "cal_t8")
NAMES = (
    "original_gap",
    "calibrated_gap",
    "interaction",
    "gaussian_calibration",
    "t8_calibration",
    "qqq_calibration",
    "spx_calibration",
)


def fixture():
    calendar = pd.bdate_range("2018-11-01", periods=230).as_unit("ns")
    iso = lambda d: str(d.date())
    protocol = {
        "evidence_class": "SYNTHETIC_FIXTURE",
        "evidence_limitation": "Generated data only; no market observations.",
        "forecast": {
            "origin_start": iso(calendar[1]),
            "origin_end": iso(calendar[-2]),
            "source_end": iso(calendar[-1]),
            "horizon": 1,
            "development": [iso(calendar[1]), iso(calendar[100])],
            "evaluation": [iso(calendar[101]), iso(calendar[-1])],
            "stability": [
                [iso(calendar[101]), iso(calendar[165])],
                [iso(calendar[166]), iso(calendar[-1])],
            ],
        },
        "support": {"phase_daily": 30, "slice_daily": 20, "offset_daily": 5},
        "inference": {
            "blocks": [2, 5],
            "hac_lags": 3,
            "bootstrap_draws": 31,
            "seed": 9201,
            "phase_codes": {"development": 0, "evaluation": 1},
        },
        "comparisons": {
            "wave": 28,
            "inherited": 151,
            "new": 7,
            "cumulative": 158,
            "wave_alpha": 0.05 / (28 * 29),
            "cumulative_alpha": 0.05,
            "ordered_contrasts": list(NAMES),
        },
    }
    positions = np.array([i for i in range(30, 229) if i not in (45, 88, 100, 145, 166)])
    rng = np.random.default_rng(728122)
    w = rng.normal(size=(len(positions), 2)) * [1.15, 0.91] + [0.13, -0.09]
    w[:, 1] += 0.5 * w[:, 0]
    z = t.ppf(norm.cdf(w), 8)
    h = np.full(z.shape, 0.001)
    a, b = np.array([0.1, -0.11]), np.array([1.12, 0.93])
    v, zcal = (w - a) / b, t.ppf(norm.cdf((w - a) / b), 8)
    original = t.logpdf(z, 8) - np.log(np.sqrt(0.75 * h))
    calibrated = original + norm.logpdf(v) - norm.logpdf(w) - np.log(b)
    p = pd.DataFrame(
        {
            "origin": calendar[positions],
            "target_end": calendar[positions + 1],
            "available_date": calendar[positions + 1],
            "offset": positions % 5,
            "phase": np.where(positions <= 100, "development", "evaluation"),
            "horizon": 1,
        }
    )
    for j, asset in enumerate(("qqq", "spx")):
        for name, value in {
            "mu": np.zeros(len(p)),
            "h": h[:, j],
            "y": z[:, j] * np.sqrt(0.75 * h[:, j]),
            "a": np.full(len(p), a[j]),
            "b": np.full(len(p), b[j]),
            "marginal_original": original[:, j],
            "marginal_calibrated": calibrated[:, j],
            "normal_original": w[:, j],
            "normal_calibrated": v[:, j],
            "pit_original": norm.cdf(w[:, j]),
            "pit_calibrated": norm.cdf(v[:, j]),
        }.items():
            p[f"{name}_{asset}"] = value
    for cell, rho, coords, marginal in zip(
        CELLS,
        [0.4, 0.45, 0.41, 0.46],
        [w, z, v, zcal],
        [original, original, calibrated, calibrated],
        strict=True,
    ):
        p["rho_" + cell] = rho
        copula = (
            multivariate_normal.logpdf(coords, cov=[[1, rho], [rho, 1]])
            - norm.logpdf(coords).sum(axis=1)
            if "gaussian" in cell
            else multivariate_t.logpdf(coords, shape=[[1, rho], [rho, 1]], df=8)
            - t.logpdf(coords, 8).sum(axis=1)
        )
        p["loss_" + cell] = -marginal.sum(axis=1) - copula
    p["d_original_gap"] = p.loss_orig_t8 - p.loss_orig_gaussian
    p["d_calibrated_gap"] = p.loss_cal_t8 - p.loss_cal_gaussian
    p["d_interaction"] = p.d_calibrated_gap - p.d_original_gap
    p["d_gaussian_calibration"] = p.loss_cal_gaussian - p.loss_orig_gaussian
    p["d_t8_calibration"] = p.loss_cal_t8 - p.loss_orig_t8
    for asset in ("qqq", "spx"):
        p[f"d_{asset}_calibration"] = (
            p[f"marginal_original_{asset}"] - p[f"marginal_calibrated_{asset}"]
        )
    prior = [
        {
            "study": "prior",
            "candidate": str(i),
            "control": "base",
            "horizon": 1,
            "source": "prior.json",
            "source_sha256": "a" * 64,
            "source_row_index": i,
            "p_conservative": 0.1 + 0.8 * i / 150,
        }
        for i in range(151)
    ]
    return p, calendar, prior, protocol


def metrics_fixture(panel, calendar, prior, protocol):
    """Producer-style generated receipts; expectations come from the independent verifier."""
    rows, diagnostics, joint = [], {}, {}
    for name in NAMES:
        phases = []
        for code, phase in enumerate(("development", "evaluation")):
            frame = panel[panel.phase == phase]
            start, end = protocol["forecast"][phase]
            full = calendar[(calendar >= start) & (calendar <= end)]
            diff = frame["d_" + name].to_numpy()
            values = pd.Series(diff, index=frame.origin).reindex(full).to_numpy()
            inf = protocol["inference"]
            r = masked_mean_inference(
                values,
                full.isin(frame.origin),
                blocks=inf["blocks"],
                hac_lags=inf["hac_lags"],
                draws=inf["bootstrap_draws"],
                seed=inf["seed"] + code * 10000,
            )
            intervals = [r["hac"]["ci95"]] + [x["ci95"] for x in r["block_inference"].values()]

            def group(mask, diff=diff):
                x = diff[np.asarray(mask)]
                return {"n": len(x), "mean": float(x.mean()) if len(x) else None}

            r.update(
                name=phase,
                ci95_envelope=[min(x[0] for x in intervals), max(x[1] for x in intervals)],
                offsets=[{"offset": k, **group(frame.offset == k)} for k in range(5)],
                stability=[
                    {"start": a, "end": b, **group(frame.origin.between(a, b))}
                    for a, b in protocol["forecast"]["stability"]
                ]
                if phase == "evaluation"
                else [],
                annual=[
                    {"year": int(y), **group(frame.origin.dt.year == y)}
                    for y in sorted(set(full.year))
                ],
            )
            phases.append(r)
        rows.append(
            {
                "contrast": name,
                "phases": phases,
                "p_conservative": max(x["p_conservative"] for x in phases),
            }
        )

    def holm(p):
        order = sorted(range(len(p)), key=lambda i: p[i])
        result = [0.0] * len(p)
        high = 0.0
        for k, i in enumerate(order):
            high = max(high, p[i] * (len(p) - k))
            result[i] = min(1.0, high)
        return result

    wave = holm([r["p_conservative"] for r in rows])
    allp = holm([r["p_conservative"] for r in prior + rows])[-7:]
    for row, pw, pc in zip(rows, wave, allp, strict=True):
        means = [x["mean"] for x in row["phases"]]
        row.update(
            p_holm_wave=pw,
            p_holm_cumulative=pc,
            adjusted_difference_detected=bool(
                means[0] * means[1] > 0
                and pw <= protocol["comparisons"]["wave_alpha"]
                and pc <= 0.05
            ),
        )
    for phase in ("development", "evaluation"):
        f = panel[panel.phase == phase]
        joint[phase] = {c: float(f["loss_" + c].mean()) for c in CELLS}
        diagnostics[phase] = {}
        for asset in ("qqq", "spx"):
            diagnostics[phase][asset] = {}
            for system in ("original", "calibrated"):
                normal = f[f"normal_{system}_{asset}"].to_numpy()
                pit = f[f"pit_{system}_{asset}"].to_numpy()
                centered = normal - normal.mean()
                variance = float(np.mean(centered**2))
                diagnostics[phase][asset][system] = {
                    "mean_loss": float(-f[f"marginal_{system}_{asset}"].mean()),
                    "pit_lower": float(np.mean(pit < 0.025)),
                    "pit_upper": float(np.mean(pit > 0.975)),
                    "normal_mean": float(normal.mean()),
                    "normal_variance": variance,
                    "normal_skew": float(np.mean(centered**3) / variance**1.5),
                }
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": copy.deepcopy(prior),
        "hypothesis_count": 7,
        "cumulative_hypothesis_count": 158,
        "leads": [],
        "common_scored_origins": len(panel),
        "calibration_diagnostics": diagnostics,
        "full_joint_losses": joint,
    }


class ScoreVerificationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.calendar, cls.prior, cls.protocol = fixture()
        cls.metrics = metrics_fixture(cls.panel, cls.calendar, cls.prior, cls.protocol)

    def check(self, panel=None, metrics=None, calendar=None, prior=None, protocol=None):
        return verify_scores(
            self.panel if panel is None else panel,
            self.calendar if calendar is None else calendar,
            self.metrics if metrics is None else metrics,
            self.prior if prior is None else prior,
            self.protocol if protocol is None else protocol,
        )

    def test_generated_panel_full_calendar_and_all_seven_contrasts_verify(self):
        self.assertEqual(self.check()["status"], "VERIFIED")
        self.assertGreater(
            self.metrics["rows"][0]["phases"][0]["full_calendar_n"],
            self.metrics["rows"][0]["phases"][0]["n"],
        )

    def test_actual_producer_evaluator_agrees_on_the_generated_fixture(self):
        from src.copula_calibration_run import evaluate

        metrics = evaluate(self.panel, self.calendar, self.prior, self.protocol)
        self.assertEqual(self.check(metrics=metrics)["status"], "VERIFIED")

    def test_count_scope_order_and_unearned_promotion_fail(self):
        for key, value in [
            ("hypothesis_count", 6),
            ("cumulative_hypothesis_count", 157),
            ("common_scored_origins", len(self.panel) - 1),
            ("leads", ["cal_t8"]),
            ("status", "UNEVALUABLE"),
        ]:
            bad = copy.deepcopy(self.metrics)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(metrics=bad)
        for field in ("rows", "inherited_rows"):
            bad = copy.deepcopy(self.metrics)
            bad[field] = bad[field][::-1]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(metrics=bad)

    def test_each_contrast_sign_or_value_tampering_fails(self):
        for name in NAMES:
            bad = self.panel.copy()
            bad.loc[0, "d_" + name] += 0.1
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check(panel=bad)

    def test_cohort_calendar_phase_endpoint_and_global_offset_tampering_fail(self):
        variants = [
            self.panel.iloc[1:].copy(),
            pd.concat([self.panel, self.panel.iloc[:1]], ignore_index=True),
            self.panel.iloc[::-1].copy(),
        ]
        for field, value in [
            ("offset", 4),
            ("target_end", self.panel.origin.iloc[0]),
            ("phase", "evaluation"),
        ]:
            bad = self.panel.copy()
            bad.loc[0, field] = value
            variants.append(bad)
        for i, bad in enumerate(variants):
            with self.subTest(i=i), self.assertRaises(ValueError):
                self.check(panel=bad)
        with self.assertRaises(ValueError):
            self.check(calendar=self.calendar.delete(50))

    def test_reported_uncertainty_diagnostics_and_adjustments_are_independently_checked(self):
        for mutation in (
            "mean",
            "p",
            "count_p",
            "envelope",
            "offset",
            "slice",
            "annual",
            "holm",
            "detector",
            "diagnostic",
            "joint",
        ):
            bad = copy.deepcopy(self.metrics)
            phase = bad["rows"][0]["phases"][0]
            if mutation == "mean":
                phase["mean"] += 0.02
            elif mutation == "p":
                phase["hac"]["p"] += 1e-5
            elif mutation == "count_p":
                phase["block_inference"]["2"]["p"] = np.nextafter(
                    phase["block_inference"]["2"]["p"], 0.0
                )
            elif mutation == "envelope":
                phase["ci95_envelope"][0] -= 0.01
            elif mutation == "offset":
                phase["offsets"][1]["n"] += 1
            elif mutation == "slice":
                bad["rows"][0]["phases"][1]["stability"][0]["mean"] += 0.01
            elif mutation == "annual":
                phase["annual"][0]["mean"] += 0.01
            elif mutation == "holm":
                bad["rows"][0]["p_holm_cumulative"] *= 0.5
            elif mutation == "detector":
                bad["rows"][0]["adjusted_difference_detected"] = True
            elif mutation == "diagnostic":
                bad["calibration_diagnostics"]["development"]["qqq"]["calibrated"][
                    "normal_skew"
                ] += 0.01
            else:
                bad["full_joint_losses"]["evaluation"]["cal_t8"] += 0.01
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check(metrics=bad)

    def test_coherently_omitted_jacobian_fails_even_with_regenerated_metrics(self):
        bad = self.panel.copy()
        for asset in ("qqq", "spx"):
            bad[f"marginal_calibrated_{asset}"] = bad[f"marginal_original_{asset}"]
        for name in ("cal_gaussian", "cal_t8"):
            for asset in ("qqq", "spx"):
                bad["loss_" + name] += (
                    self.panel[f"marginal_calibrated_{asset}"]
                    - self.panel[f"marginal_original_{asset}"]
                )
        bad.d_calibrated_gap = bad.loss_cal_t8 - bad.loss_cal_gaussian
        bad.d_interaction = bad.d_calibrated_gap - bad.d_original_gap
        bad.d_gaussian_calibration = bad.loss_cal_gaussian - bad.loss_orig_gaussian
        bad.d_t8_calibration = bad.loss_cal_t8 - bad.loss_orig_t8
        bad.d_qqq_calibration = 0.0
        bad.d_spx_calibration = 0.0
        with self.assertRaises(ValueError):
            self.check(
                panel=bad,
                metrics=metrics_fixture(bad, self.calendar, self.prior, self.protocol),
            )

    def test_support_and_inherited_probability_mutations_fail(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["support"]["phase_daily"] = 10000
        with self.assertRaises(ValueError):
            self.check(protocol=protocol)
        for value in (np.nan, -0.01, 1.1, True):
            prior = copy.deepcopy(self.prior)
            prior[0]["p_conservative"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(prior=prior)


if __name__ == "__main__":
    unittest.main()
