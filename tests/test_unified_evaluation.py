"""统一评测的纯逻辑守卫；完整 validation 由命令入口运行。"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resource_management.unified_evaluation import METHODS, summary, validate_records


class TestUnifiedEvaluationContract(unittest.TestCase):
    def test_seven_methods_are_fixed(self) -> None:
        self.assertEqual(len(METHODS), 7)
        self.assertEqual(METHODS[:3], ("round_robin", "edf", "rule"))
        self.assertEqual(METHODS[-2:], ("ppo_baseline", "ppo_freshness_uncertainty"))

    def test_single_training_checkpoint_has_no_fake_ci(self) -> None:
        result = summary([1.0])
        self.assertEqual(result["n"], 1)
        self.assertIsNone(result["std"])
        self.assertIsNone(result["ci95_low"])

    def test_validation_rejects_runtime_or_boundary_failures(self) -> None:
        row = {
            "method": "rule", "scenario": "s", "environment_seed": 1,
            "conservation_ok": True, "resource_violation_rate": 0.0,
            "truth_payload_violations": 0, "duplicate_runtime_tasks": 0,
            "runtime_mode": "plan_controlled_feedback", "task_gating": "expose_all",
        }
        self.assertEqual(validate_records([row]), [])
        row["truth_payload_violations"] = 1
        self.assertEqual(len(validate_records([row])), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
