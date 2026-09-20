"""Global Track v1.3 A–K 端到端与最终稳健性验收运行器回归。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.run_global_track_acceptance import SCENARIOS, run_acceptance


class TestGlobalTrackAcceptanceRunner(unittest.TestCase):
    def test_runs_all_scenarios_and_enforces_v13_final_foundation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = run_acceptance(output, tuple(SCENARIOS), seed=20260920)
            self.assertTrue((output / "global_track_acceptance_report.json").is_file())
            self.assertTrue((output / "global_track_acceptance_report.md").is_file())
            self.assertEqual(len(report["scenarios"]), 11)
            by_id = {item["scenario_id"]: item for item in report["scenarios"]}
            self.assertTrue(by_id["dual_radar_single_target"]["acceptance"]["passed"])
            self.assertTrue(by_id["dual_radar_two_targets"]["acceptance"]["passed"])
            self.assertTrue(by_id["handover_disconnect_reconnect"]["acceptance"]["passed"])
            self.assertIsNone(by_id["two_targets_crossing"]["acceptance"]["passed"])
            for scenario_id in (
                "track_message_delay_reordering", "radar_reconnect_new_local_id",
                "leave_drop_reenter", "single_node_false_local_track",
                "link_outage_loss_recovery", "separated_multi_target_concurrency",
                "asynchronous_radar_refresh",
            ):
                self.assertTrue(by_id[scenario_id]["acceptance"]["passed"])
            self.assertTrue(report["foundation_freeze"]["frozen"])
            self.assertTrue(report["foundation_freeze"]["v1x_final_freeze"])
            self.assertTrue(report["foundation_freeze"]["no_more_foundation_scenarios"])
            self.assertEqual(report["foundation_freeze"]["next_stage"], "Tower View")
            self.assertEqual(
                report["overall_judgement"]["basic_acceptance_failed_scenarios"], [])

            single = by_id["dual_radar_single_target"]
            self.assertEqual(single["runtime_trace"]["ci"]["final_tracks"][0]
                             ["active_ci_source_count"], 2)
            dual = by_id["dual_radar_two_targets"]
            self.assertEqual(dual["runtime_trace"]["local_tracks"]
                             ["track_message_generated_unique"], 4)
            self.assertEqual(len(dual["runtime_trace"]["ci"]["final_tracks"]), 2)
            self.assertGreater(dual["runtime_trace"]["share_execution"]
                               ["multi_track_share_executions"], 0)
            handover = by_id["handover_disconnect_reconnect"]
            self.assertGreater(handover["offline_evaluation"]
                               ["global_track_coverage"], 0.0)
            self.assertIsNotNone(handover["offline_evaluation"]["rmse_m"])

            crossing = by_id["two_targets_crossing"]
            self.assertIn("complex_association_capability_limit",
                          crossing["judgement"]["categories"])
            reorder = by_id["track_message_delay_reordering"]
            self.assertGreater(reorder["runtime_trace"]["track_message_state"]
                               ["out_of_order_arrival_count"], 0)
            self.assertGreater(reorder["runtime_trace"]["robustness"]
                               ["stale_message_rejection_count"], 0)
            reconnect = by_id["radar_reconnect_new_local_id"]
            self.assertGreater(reconnect["runtime_trace"]["robustness"]
                               ["reconnect_count"], 0)
            reentry = by_id["leave_drop_reenter"]
            self.assertEqual(reentry["runtime_trace"]["robustness"]["drop_count"], 1)
            fake = by_id["single_node_false_local_track"]
            self.assertGreater(fake["runtime_trace"]["robustness"]
                               ["same_source_competitor_rejection_count"], 0)
            outage = by_id["link_outage_loss_recovery"]
            self.assertGreater(outage["runtime_trace"]["robustness"]
                               ["transport_rejection_reasons"].get("link_outage", 0), 0)
            outage_sources = [row["active_source_count"] for row in
                              outage["runtime_trace"]["robustness"]
                              ["active_source_count_history"]]
            self.assertIn(0, outage_sources)
            self.assertEqual(outage_sources[-1], 2)
            concurrency = by_id["separated_multi_target_concurrency"]
            self.assertEqual(concurrency["runtime_trace"]["local_tracks"]
                             ["track_message_generated_unique"], 6)
            self.assertEqual(len(concurrency["runtime_trace"]["ci"]["final_tracks"]), 3)
            self.assertEqual(concurrency["offline_evaluation"]["global_id_switches"], 0)
            self.assertEqual(concurrency["offline_evaluation"]["fragmentation"], 0)
            asynchronous = by_id["asynchronous_radar_refresh"]
            temporal = asynchronous["runtime_trace"]["temporal_projection"]
            self.assertGreater(temporal["asynchronous_ci_events"], 0)
            self.assertGreater(temporal["positive_projection_events"], 0)
            self.assertLessEqual(temporal["max_projection_formula_error_m"], 1e-9)
            for scenario in report["scenarios"]:
                trace = scenario["runtime_trace"]
                self.assertTrue(trace["resource_conserved"])
                self.assertEqual(trace["runtime_truth_payload_violations"], 0)
                self.assertGreater(trace["ci"]["ci_fusion_events"], 0)
                self.assertTrue(trace["share_execution"]
                                ["all_accounted_bytes_equal_sent_bytes"])
                self.assertTrue(all(row["strictly_increasing"]
                                    for row in trace["track_message_state"]
                                    ["by_local_track"]))
                serialized = json.dumps({
                    "runtime_trace": trace,
                    "judgement": scenario["judgement"],
                }, ensure_ascii=False).lower()
                self.assertNotIn("truth_id", serialized)

    def test_each_scenario_can_run_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = run_acceptance(
                Path(temporary), ("handover_disconnect_reconnect",), seed=20260920,
            )
            self.assertEqual(report["scenario_ids"], ["handover_disconnect_reconnect"])
            self.assertEqual(report["scenarios"][0]["scenario_id"],
                             "handover_disconnect_reconnect")
            self.assertFalse(report["foundation_freeze"]["frozen"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
