"""Preference-Conditioned PPO 已暂停归档的回归保护。"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "preference_ppo_archive",
    os.path.join(ROOT, "tools", "verify_preference_ppo_archive.py"),
)
assert SPEC is not None and SPEC.loader is not None
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


class TestPreferencePpoArchive(unittest.TestCase):
    def test_v1_v2_v3_are_frozen_negative_results_and_test_v5_is_sealed(self) -> None:
        report = ARCHIVE.verify_archive()
        self.assertEqual(set(report["versions"]), {"v1", "v2", "v3"})
        self.assertEqual(report["test_v5"], "sealed")
        for version in report["versions"].values():
            self.assertEqual(version["mechanism_gate"], "failed")
            self.assertIn("sealed", version["test_status"])
            self.assertEqual(len(version["checkpoint_hashes"]), 3)

    def test_multiseed_baseline_conclusion_remains_positive(self) -> None:
        report = ARCHIVE.verify_archive()
        self.assertEqual(report["baseline_ppo"]["status"], "preserved")
        self.assertIn("five-seed", report["baseline_ppo"]["conclusion"])

    def test_missing_historical_absolute_checkpoint_path_falls_back_to_seed_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata_path = os.path.join(temporary, "metadata.json")
            fallback = os.path.join(temporary, "policy.pt")
            with open(fallback, "wb") as handle:
                handle.write(b"frozen checkpoint fixture")
            resolved = ARCHIVE.resolve_checkpoint(metadata_path, {
                "checkpoint": r"C:\retired-machine\output\seed_907\policy.pt",
            })
            self.assertEqual(os.path.normcase(resolved["path"]),
                             os.path.normcase(fallback))
            self.assertEqual(
                resolved["resolution"],
                "metadata_seed_directory_policy_pt_fallback",
            )
