"""固定 development 全局航迹评测入口的轻量回归。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluate_global_tracking import (
    COMMUNICATION_MODES,
    run_development_evaluation,
)


class TestGlobalTrackingDevelopmentEvaluation(unittest.TestCase):
    def test_fixed_development_evaluator_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_development_evaluation(Path(directory), seeds=(41,), steps=8)
            self.assertEqual(report["scope"], "fixed_development_seeds_only")
            self.assertEqual(len(report["rows"]), len(COMMUNICATION_MODES))
            self.assertTrue(all(row["resource_conserved"] for row in report["rows"]))
            self.assertTrue(all(row["runtime_truth_payload_violations"] == 0
                                for row in report["rows"]))
            for name in ("global_tracking_development.csv",
                         "global_tracking_development.json",
                         "global_tracking_development.html"):
                self.assertTrue((Path(directory) / name).is_file())
            saved = json.loads((Path(directory) / "global_tracking_development.json").read_text(
                encoding="utf-8"))
            self.assertEqual(saved["modes"], list(COMMUNICATION_MODES))
            self.assertIn("No PPO training", " ".join(saved["honesty_boundaries"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
