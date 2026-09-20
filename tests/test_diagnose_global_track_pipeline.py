"""Global Track Pipeline Diagnostics 的无副作用报告回归。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from global_fusion import GLOBAL_SHARE_MODE_TRACK
from tools.diagnose_global_track_pipeline import run_pipeline_diagnosis


class TestGlobalTrackPipelineDiagnosis(unittest.TestCase):
    def test_writes_funnel_lifecycle_and_truth_isolation_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = run_pipeline_diagnosis(
                output, seeds=(41,), steps=18, modes=(GLOBAL_SHARE_MODE_TRACK,),
            )
            for name in (
                "global_track_pipeline_events.csv",
                "global_track_pipeline_summary.csv",
                "global_track_pipeline_lifecycle.csv",
                "global_track_pipeline.json",
                "global_track_pipeline.md",
            ):
                self.assertTrue((output / name).is_file(), name)

            run = report["runs"][0]
            message = run["pipeline_funnel"]["message_event_funnel"]
            lifecycle = run["pipeline_funnel"]["global_track_lifecycle"]
            self.assertGreater(run["pipeline_funnel"]["unique_local_track_funnel"]
                               ["local_track_created"], 0)
            self.assertGreater(message["generated_attempts"], 0)
            self.assertGreaterEqual(message["sent"], message["arrived"])
            self.assertGreaterEqual(message["arrived"], message["associated"])
            self.assertEqual(message["associated"], message["ci_fused"])
            self.assertGreater(lifecycle["final_global_tracks"], 0)
            self.assertAlmostEqual(
                lifecycle["single_source_ratio"] + lifecycle["multi_source_ratio"],
                1.0,
            )
            self.assertTrue(run["runtime_guards"]["resource_conserved"])
            self.assertEqual(run["runtime_guards"]["runtime_truth_payload_violations"], 0)

            # 外层离线指标可有 coverage 聚合，但运行时 events 和生命周期绝不携带 truth ID。
            runtime_serialized = json.dumps({
                "events": run["events"],
                "lifecycle": run["global_track_lifecycles"],
            }, ensure_ascii=False).lower()
            self.assertNotIn("truth_id", runtime_serialized)
            self.assertEqual(
                run["offline_truth_evaluation"]["provenance"],
                "offline_truth_evaluation_only",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
