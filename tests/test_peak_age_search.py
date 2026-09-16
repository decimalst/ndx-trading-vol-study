"""Prewritten family/inference tests using invented forecasts only."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import peak_age_search as run


def inference_fixture():
    p = copy.deepcopy(run.CONTRACT)
    calendar = pd.bdate_range("2000-01-03", periods=2400)
    p["index"].update(
        development=[str(calendar[100].date()), str(calendar[720].date())],
        development_target_available_by=str(calendar[720].date()),
        evaluation=[str(calendar[1000].date()), str(calendar[1700].date())],
        evaluation_stability=[
            [str(calendar[1000].date()), str(calendar[1299].date())],
            [str(calendar[1300].date()), str(calendar[1700].date())],
        ],
        source_end=str(calendar[-1].date()),
        latest_target=str(calendar[-1].date()),
    )
    p["inference"]["bootstrap_draws"] = 31
    rng = np.random.default_rng(707)
    rows = []
    for phase, positions in [
        ("development", np.arange(100, 700)),
        ("evaluation", np.arange(1000, 1600)),
    ]:
        y = rng.normal(0, 0.02, len(positions))
        for model, offset in [
            ("mean", 0.02),
            ("baseline", 0.013),
            ("depth", 0.012),
            ("peak_age", 0.006),
        ]:
            prediction = y + offset + rng.normal(0, 0.001, len(positions))
            for n, position in enumerate(positions):
                rows.append(
                    {
                        "origin": calendar[position],
                        "fit_origin": calendar[1],
                        "feature_cutoff_date": calendar[position - 1],
                        "target_end": calendar[position + 21],
                        "available_date": calendar[position + 21],
                        "horizon": 21,
                        "phase": phase,
                        "model": model,
                        "prediction": prediction[n],
                        "y": y[n],
                        "loss": (y[n] - prediction[n]) * (y[n] - prediction[n]),
                        "train_n": 1000,
                    }
                )
    return (
        pd.DataFrame(rows, columns=run.pipeline.PANEL_COLUMNS)
        .sort_values(["origin", "model"])
        .reset_index(drop=True),
        calendar,
        p,
    )


def row(control="baseline"):
    return {
        "study": "peak_age",
        "candidate": "peak_age",
        "control": control,
        "horizon": 21,
        "score": "mse",
        "p_holm_wave": 1e-6,
        "p_holm_cumulative": 0.001,
        "phases": [
            {
                "name": name,
                "n": 600,
                "delta": -0.001,
                "gain_relative": 0.01,
                "nonoverlap_phases": [
                    {"phase": i, "n": 20, "delta": -0.001} for i in range(21)
                ],
                "stability": [{"n": 200, "delta": -0.001}, {"n": 200, "delta": -0.001}]
                if name == "evaluation"
                else [],
            }
            for name in ("development", "evaluation")
        ],
    }


class SearchContracts(unittest.TestCase):
    def test_literal_full_contract(self):
        run.validate(copy.deepcopy(run.CONTRACT))
        self.assertEqual(run.CONTRACT["comparisons"]["cumulative_hypotheses"], 140)
        self.assertEqual(run.CONTRACT["index"]["horizons"], [21])

    def test_every_leaf_or_extra_key_mutation_rejects(self):
        def leaves(x, path=()):
            if isinstance(x, dict):
                for k, v in x.items():
                    yield from leaves(v, path + (k,))
            elif isinstance(x, list):
                for k, v in enumerate(x):
                    yield from leaves(v, path + (k,))
            else:
                yield path, x

        for path, value in leaves(run.CONTRACT):
            p = copy.deepcopy(run.CONTRACT)
            node = p
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value + " changed" if isinstance(value, str) else "changed"
            with self.subTest(path=path), self.assertRaises(ValueError):
                run.validate(p)
        p = copy.deepcopy(run.CONTRACT)
        p["unregistered_variant"] = True
        with self.assertRaises(ValueError):
            run.validate(p)
        p = copy.deepcopy(run.CONTRACT)
        p["index"]["market_lag"] = True
        with self.assertRaises(ValueError):
            run.validate(p)

    def test_all_three_controls_and_single_horizon_required(self):
        rows = [row(c) for c in ("baseline", "depth", "mean")]
        self.assertEqual(run.candidate_leads(rows), [21])
        rows[1]["phases"][0]["delta"] = 0
        self.assertEqual(run.candidate_leads(rows), [])
        for partial in (rows[:2], rows + [row()], rows[:2] + [row("unknown")]):
            with self.assertRaises(ValueError):
                run.candidate_leads(partial)

    def test_every_offset_and_each_joint_gate_required(self):
        mutations = [
            lambda r: r.update(p_holm_wave=run.WAVE_ALPHA),
            lambda r: r.update(p_holm_cumulative=0.05),
            lambda r: r["phases"][0].update(gain_relative=0.002499),
            lambda r: r["phases"][0].update(n=504),
            lambda r: r["phases"][1]["stability"][0].update(delta=0),
            lambda r: r["phases"][0]["nonoverlap_phases"][20].update(delta=0),
            lambda r: r["phases"][1]["nonoverlap_phases"][0].update(n=0),
            lambda r: r["phases"][0]["nonoverlap_phases"].pop(),
        ]
        for mutation in mutations:
            r = row()
            mutation(r)
            with self.subTest(mutation=mutation):
                self.assertFalse(run.passes(r))

    def test_failure_keeps_all_three_and_family140(self):
        m = run.failure_metrics(ValueError("broken"), "abc")
        self.assertEqual(m["hypothesis_count"], 3)
        self.assertEqual(m["cumulative_hypothesis_count"], 140)
        self.assertEqual(m["leads"], [])
        self.assertEqual(m["status"], "UNEVALUABLE")
        for r in m["rows"]:
            self.assertEqual(r["phases"], [])
            self.assertEqual(
                [r[k] for k in ("p_conservative", "p_holm_wave", "p_holm_cumulative")],
                [1.0, 1.0, 1.0],
            )

    def test_inherited_count_and_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file = root / "prior.json"
            file.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "candidate": str(i),
                                "control": "baseline",
                                "horizon": 21,
                                "p_conservative": 1.0,
                            }
                            for i in range(137)
                        ]
                    }
                )
            )
            p = copy.deepcopy(run.CONTRACT)
            p["comparisons"]["inherited_sources"] = ["prior.json"]
            with patch.object(run, "ROOT", root):
                self.assertEqual(
                    len(run.inherited(p, {"prior.json": run.inference.digest(file)})), 137
                )
                with self.assertRaises(ValueError):
                    run.inherited(p, {"prior.json": "0" * 64})
                file.write_text(json.dumps({"rows": []}))
                with self.assertRaises(ValueError):
                    run.inherited(p)

    def test_generated_inference_has_all_controls_phases_offsets_and_counts(self):
        panel, calendar, p = inference_fixture()
        prior = [{"p_conservative": 1.0} for _ in range(137)]
        m = run.evaluate(panel, calendar, p, 10, 1205, prior=prior)
        self.assertEqual(len(m["rows"]), 3)
        self.assertEqual(m["newly_generated_forecasts"], 4800)
        self.assertEqual(m["full_application_predictions"], 4820)
        self.assertEqual(m["ridge_models_fitted"], 20)
        self.assertEqual(m["scalar_models_fitted"], 10)
        for r in m["rows"]:
            self.assertEqual(r["horizon"], 21)
            for phase in r["phases"]:
                self.assertEqual(phase["n"], 600)
                self.assertLess(phase["delta"], 0)
                self.assertEqual(
                    {x["phase"] for x in phase["nonoverlap_phases"]}, set(range(21))
                )
                self.assertEqual(set(phase["block_inference"]), {"126", "252", "504"})

    def test_short_phase_and_mismatched_or_future_rows_reject(self):
        panel, calendar, p = inference_fixture()
        prior = [{"p_conservative": 1.0} for _ in range(137)]
        for kind in ("short", "label", "calendar", "count"):
            q = panel.copy()
            cal = calendar
            if kind == "short":
                q = q.loc[
                    ~q.origin.isin(q.loc[q.phase == "development", "origin"].unique()[:100])
                ].reset_index(drop=True)
            elif kind == "label":
                q.loc[0, "y"] += 1
            elif kind == "calendar":
                cal = calendar[1:]
            else:
                prior = prior[:-1]
            if kind == "calendar":
                cal = calendar.delete(100)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                run.evaluate(q, cal, p, 10, 1205, prior=prior)

    def test_literal_fit_application_counts_required(self):
        panel, calendar, p = inference_fixture()
        prior = [{"p_conservative": 1.0} for _ in range(137)]
        for k, a in ((True, 1205), (0, 1205), (10, True), (10, 100)):
            with self.subTest(k=k, a=a), self.assertRaises(ValueError):
                run.evaluate(panel, calendar, p, k, a, prior=prior)


if __name__ == "__main__":
    unittest.main()
