"""Guard high-level claims that previously drifted from generated reports."""
from pathlib import Path
import json
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReadmeConsistencyContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text()

    def test_power_counts_match_frozen_reports(self):
        self.assertIn("12 in the frozen ≥5% heavy-earnings slice", self.readme)
        self.assertIn("157 in the\ndiagnostic window", self.readme)
        self.assertNotIn("8 in the heavy-earnings slice", self.readme)

    def test_clean_phase_forbids_monthly_peeking(self):
        self.assertIn("500 scored origins or\n  2027-06-30", self.readme)
        self.assertIn("No monthly peeking", self.readme)
        self.assertNotIn("re-evaluate monthly", self.readme)

    def test_headline_rv_estimator_is_unambiguous(self):
        self.assertIn(
            "every original QQQ headline table in this repository was produced with\n"
            "  `SOURCE=daily`",
            self.readme,
        )
        self.assertIn("not five-minute realized variance", self.readme)

    def test_leverage_absorption_is_in_summary(self):
        self.assertIn("103.19 without VXN to 62.63 with it", self.readme)
        self.assertIn("about 40% attenuation", self.readme)

    def test_weight_language_describes_current_point_in_time_state(self):
        self.assertIn("latest\n  point-in-time reconstruction available before that announcement", self.readme)
        self.assertNotIn("uses approximately-current weights", self.readme)

    def test_five_path_headlines_match_persisted_metrics(self):
        data = ROOT / "data/research_paths"
        horizon = json.loads((data / "horizon_curve_metrics.json").read_text())
        five = next(row for row in horizon["rows"] if row["horizon"] == 5)
        self.assertIn(f"{five['qlike_improvement_pct']:.2f}%", self.readme)
        vrp = json.loads((data / "vrp_term_structure_metrics.json").read_text())
        premiums = [row["premium_mean"] for row in vrp["rows"]]
        self.assertIn(
            f"{premiums[0]:.2f}, {premiums[1]:.2f}, and {premiums[2]:.2f}",
            self.readme,
        )
        single = json.loads((data / "single_name_earnings_metrics.json").read_text())
        effect = single["equal_asset_pool"]["equal_asset_effect_log_variance"]
        self.assertIn(f"{effect:.2f}-log-variance", self.readme)
        spx = json.loads((data / "spx_term_slope_metrics.json").read_text())
        self.assertIn(f"{abs(spx['dislocation_only']['improvement_pct']):.2f}%", self.readme)

    def test_every_local_report_link_resolves(self):
        targets = re.findall(r"\((reports/[^)]+)\)", self.readme)
        for target in targets:
            path = target.split("#", 1)[0]
            self.assertTrue((ROOT / path).exists(), path)


if __name__ == "__main__":
    unittest.main()
