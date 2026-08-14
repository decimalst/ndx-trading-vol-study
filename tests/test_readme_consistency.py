"""Guard high-level claims that previously drifted from generated reports.

Scope note (2026-08-13): README.md was cut from 477 lines to ~150 and the run
order, data sourcing, method and signal-study sections moved to `docs/`. These
assertions are about the DOCUMENTATION SET, not about one file, so they search
README.md plus docs/*.md together -- otherwise a claim could be "fixed" simply
by relocating it, which is exactly the drift this contract exists to prevent.

Whitespace is normalised before matching so a reflowed paragraph does not
silently drop a guarded claim.
"""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


class ReadmeConsistencyContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text()
        parts = [cls.readme]
        for p in sorted((ROOT / "docs").glob("*.md")):
            parts.append(p.read_text())
        cls.docs = "\n\n".join(parts)
        cls.norm = _norm(cls.docs)

    def assertDocs(self, needle: str, msg: str = ""):
        self.assertIn(_norm(needle), self.norm, msg or needle)

    def assertNotDocs(self, needle: str):
        self.assertNotIn(_norm(needle), self.norm, needle)

    def test_power_counts_match_frozen_reports(self):
        self.assertDocs("12 in the frozen ≥5% heavy-earnings slice")
        self.assertDocs("157 in the diagnostic window")
        self.assertNotDocs("8 in the heavy-earnings slice")

    def test_clean_phase_forbids_monthly_peeking(self):
        self.assertDocs("500 scored origins or 2027-06-30")
        self.assertDocs("No monthly peeking")
        self.assertNotDocs("re-evaluate monthly")

    def test_headline_rv_estimator_is_unambiguous(self):
        self.assertDocs(
            "every original QQQ headline table in this repository was produced "
            "with `SOURCE=daily`")
        self.assertDocs("not five-minute realized variance")

    def test_leverage_absorption_is_in_summary(self):
        self.assertDocs("103.19 → 62.63")
        self.assertDocs("~40% of it")

    def test_weight_language_describes_current_point_in_time_state(self):
        self.assertDocs(
            "latest point-in-time reconstruction available before that announcement")
        self.assertNotDocs("uses approximately-current weights")

    def test_five_path_headlines_match_persisted_metrics(self):
        data = ROOT / "data/research_paths"
        horizon = json.loads((data / "horizon_curve_metrics.json").read_text())
        five = next(row for row in horizon["rows"] if row["horizon"] == 5)
        self.assertDocs(f"{five['qlike_improvement_pct']:.2f}%")
        vrp = json.loads((data / "vrp_term_structure_metrics.json").read_text())
        premiums = [row["premium_mean"] for row in vrp["rows"]]
        self.assertDocs(
            f"{premiums[0]:.2f}, {premiums[1]:.2f}, and {premiums[2]:.2f}")
        single = json.loads((data / "single_name_earnings_metrics.json").read_text())
        effect = single["equal_asset_pool"]["equal_asset_effect_log_variance"]
        self.assertDocs(f"{effect:.2f}-log-variance")
        spx = json.loads((data / "spx_term_slope_metrics.json").read_text())
        self.assertDocs(f"{abs(spx['dislocation_only']['improvement_pct']):.2f}%")

    # ---- claims added by the 2026-08-13 corrections -------------------------
    # Each of these was a stated conclusion that the corrections overturned. If
    # one disappears from the documentation set, the retraction has been lost.

    def test_inconclusive_is_distinguished_from_null(self):
        self.assertDocs("`inconclusive` means the design could not tell")
        self.assertDocs("A failure to reject is not evidence of no effect")

    def test_spent_clean_draw_is_disclosed(self):
        self.assertDocs("no longer have a clean draw")
        self.assertDocs("192 clean origins were")

    def test_unreachable_gate_is_disclosed(self):
        self.assertDocs("cannot fire at its own trigger")

    def test_bootstrap_miscalibration_is_disclosed(self):
        self.assertDocs("anti-conservative")

    def test_estimator_dependence_is_disclosed(self):
        self.assertDocs("estimator-dependent")
        self.assertDocs("QLIKE is not scale-invariant")

    def test_ranking_retraction_is_on_the_front_page(self):
        """The three retracted ranking claims must be in README itself, not
        only in a linked report -- a reader going top-down hits the table
        first."""
        self.assertIn("0.8317", self.readme)
        self.assertIn("zero within-year information", _norm(self.readme))

    def test_convergence_standard_is_stated(self):
        self.assertDocs("zero conclusion-changing findings")

    def test_every_local_link_resolves(self):
        targets = re.findall(r"\((reports/[^)]+|docs/[^)]+)\)", self.docs)
        for target in targets:
            path = target.split("#", 1)[0]
            self.assertTrue((ROOT / path).exists(), path)


if __name__ == "__main__":
    unittest.main()
