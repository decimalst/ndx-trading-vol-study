"""Environment-prerequisite checks. NOT contracts about the code.

These assert that large, git-ignored inputs exist on this machine: the 512-dim
TiRex embedding chunk store, the raw HuggingFace option-flow shards, and the
locally cached Chronos-2 / TiRex-2 model snapshots. They cannot pass on a clean
clone and they cannot pass on a CI runner, because the inputs are deliberately
not in the repository.

They used to live in the ordinary suite behind guards that checked the WRONG
precondition -- `METRICS_PATH.exists()` is true on a clean clone because the
metrics JSON is committed, while the verify path it gates then reaches for
`latent_embedding_chunks/manifest.json`, which `.gitignore` excludes. So the
guard passed, the code errored, and a fresh checkout saw a hard failure that had
nothing to do with the science.

That is the same defect this repository has now corrected several times: a fence
checking something adjacent to the thing it is protecting. Splitting them out is
cleaner than adding more guards, because "do I have a 512-dimensional embedding
cache on this box" is not a claim about whether the code is correct.

    make test        # code contracts. CI runs this.
    make test-env    # this file.

The filename does not start with `test_` on purpose. `unittest discover`
recurses into any subdirectory that is a package, so putting these in
`tests/env/` was NOT enough on its own -- the default suite still collected
them. The `env_test_` prefix is what actually excludes them from
`discover -p 'test_*.py'`, which is the pattern CI runs.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _need(*relative: str) -> None:
    """Skip with a precise reason when a required local input is absent."""
    missing = [r for r in relative if not (ROOT / r).exists()]
    if missing:
        raise unittest.SkipTest(
            "missing local (git-ignored) inputs: " + ", ".join(missing))


class LatentChunkStore(unittest.TestCase):
    """Needs data/representation_study/latent_embedding_chunks/ (git-ignored)."""

    def test_frozen_k1_outputs_verify(self):
        _need("data/representation_study/latent_k1_confirmation_metrics.json",
              "data/representation_study/latent_embedding_chunks/manifest.json")
        from src import latent_k1_confirmation as study
        self.assertEqual(study.verify_results(write=False)["status"], "PASS")


class HuggingFaceOptionFlowShards(unittest.TestCase):
    """Needs data/free_sources/raw/huggingface/ (git-ignored)."""

    def test_current_artifacts_pass_or_verify_the_frozen_insufficient_gate(self):
        _need("data/free_sources/raw/huggingface")
        from src import verify_free_signal_option_flow as verifier
        _need("data/free_signal_study/hf_option_flow_metrics.json")
        got = verifier.verify_artifacts()
        self.assertIn(got["status"], {"PASS", "INSUFFICIENT_DATA"})
        self.assertGreaterEqual(got["checks"], 12)
        if got["status"] == "INSUFFICIENT_DATA":
            self.assertIn("zero historical scale", got["gate_reason"])
            self.assertEqual(got["zero_scale_component"],
                             "near_expiry_volume_share_7d")
            self.assertEqual(got["finite_composite_rows"], {"QQQ": 0, "SPY": 0})


class ModelSnapshotCache(unittest.TestCase):
    """Needs the local HuggingFace hub cache and huggingface_hub installed."""

    def test_cached_snapshots_resolve_required_executable_files(self):
        try:
            import huggingface_hub
        except ModuleNotFoundError:
            self.skipTest("huggingface_hub is not installed (torch extra)")
        from src.noise_robustness import cached_snapshot_path
        try:
            chronos = cached_snapshot_path(
                "amazon/chronos-2",
                "29ec3766d36d6f73f0696f85560a422f50e8498c", ("config.json",))
            tirex = cached_snapshot_path(
                "NX-AI/TiRex-2",
                "05e5b26db52bfb256f1ae1bdf785589850482de3",
                ("model-config.yaml", "model.ckpt"))
        except Exception as exc:
            self.skipTest(f"model snapshots not cached locally: {exc}")
        self.assertTrue((chronos / "config.json").exists())
        self.assertTrue((tirex / "model.ckpt").exists())


if __name__ == "__main__":
    unittest.main()
