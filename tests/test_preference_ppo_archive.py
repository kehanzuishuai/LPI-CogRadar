"""Preference-Conditioned PPO 已暂停归档的回归保护。"""
from __future__ import annotations

import importlib.util
import os
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
