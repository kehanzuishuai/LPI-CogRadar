"""Global Track / Track-to-Track Fusion v1 的协议、隔离与生命周期回归。"""

from __future__ import annotations

import json
import copy
import unittest

from communication import (
    CommBus,
    CommConfig,
    PayloadViolation,
    SHARE_CONSTRAINED,
    SHARE_IDEAL,
    TrackMessage,
)
from global_fusion import (
    GLOBAL_TRACK_ENDPOINT,
    GLOBAL_TRACK_MODE_OFF,
    GLOBAL_TRACK_MODE_TRACK_FUSION,
    GLOBAL_SHARE_MODE_EVENT_TRACK,
    GLOBAL_SHARE_MODE_MEASUREMENT,
    GLOBAL_SHARE_MODE_NO_SHARE,
    GLOBAL_SHARE_MODE_TRACK,
    GlobalTrackManager,
    GlobalTrackConfig,
    GlobalObservation,
    GLOBAL_OBSERVATION_SCHEMA_VERSION,
    global_observation_diagnostics,
)
from resource_management.closed_loop import RUNTIME_MODE_FEEDBACK, run_closed_loop
from resource_management.scheduling import SchedulerPolicy


def _message(message_id: str, source: str, local: str, seq: int,
             position=(0.0, 0.0, 0.0), now: float = 0.0,
             covariance=(100.0, 100.0, 100.0),
             velocity=(10.0, 0.0, 0.0)) -> TrackMessage:
    return TrackMessage(
        message_id=message_id, source_node_id=source, local_track_id=local,
        sequence_no=seq, state_timestamp_s=now, send_time_s=now,
        position_m=position, velocity_mps=velocity,
        covariance_position_m2=covariance,
        track_status="confirmed", information_age_s=0.0,
        source_provenance={"source_sensors": ["RADAR_A"]},
        dst_platform_id=GLOBAL_TRACK_ENDPOINT,
    )


class TestTrackMessageProtocol(unittest.TestCase):
    def test_track_message_uses_real_link_delay_and_blocks_truth_fields(self) -> None:
        # 构造期才会审计字段，故用完整构造验证真值通道被拒。
        with self.assertRaises(PayloadViolation):
            TrackMessage(
                message_id="bad", source_node_id="A", local_track_id="A-T1",
                sequence_no=1, state_timestamp_s=0.0, send_time_s=0.0,
                position_m=(0.0, 0.0, 0.0), velocity_mps=(0.0, 0.0, 0.0),
                covariance_position_m2=(1.0, 1.0, 1.0), track_status="confirmed",
                information_age_s=0.0, source_provenance={"truth_id": "T1"},
                dst_platform_id=GLOBAL_TRACK_ENDPOINT,
            )

        bus = CommBus(["A"], CommConfig(
            policy=SHARE_CONSTRAINED, base_delay_s=1.0, seed=7,
        ), extra_endpoint_ids=[GLOBAL_TRACK_ENDPOINT])
        manager = GlobalTrackManager()
        message = _message("GT000001", "A", "A-T1", 1)
        bus.publish_track(message, 0.0)
        self.assertEqual(manager.ingest_arrived(bus, 0.0), 0)
        self.assertEqual(manager.ingest_arrived(bus, 1.0), 1)
        self.assertEqual(len(manager.tracks), 1)
        self.assertEqual(bus.log[0].size_bytes, 128.0)

    def test_covariance_intersection_is_conservative_and_stable(self) -> None:
        bus = CommBus(["A", "B"], CommConfig(policy=SHARE_IDEAL, seed=5),
                      extra_endpoint_ids=[GLOBAL_TRACK_ENDPOINT])
        manager = GlobalTrackManager()
        # 两个互补方向的 local covariance：CI 可以改善整体 trace，但每一维
        # 都不会比最小单来源方差更小（禁止独立性假设造成的过度自信）。
        cov_a = (100.0, 10_000.0, 100.0)
        cov_b = (10_000.0, 100.0, 10_000.0)
        bus.publish_track(_message("GT000011", "A", "A-T1", 11,
                                   covariance=cov_a), 0.0)
        manager.ingest_arrived(bus, 0.0)
        global_id = manager.local_to_global[("A", "A-T1")]
        single = tuple(manager.tracks[global_id].covariance_position_m2)
        self.assertEqual(single, cov_a, "单节点必须退化为本节点估计")
        bus.publish_track(_message("GT000012", "B", "B-T7", 12,
                                   position=(5.0, 0.0, 0.0), covariance=cov_b), 1.0)
        manager.predict_to(1.0)
        manager.ingest_arrived(bus, 1.0)
        track = manager.tracks[global_id]
        self.assertEqual(track.fusion_method, "covariance_intersection_conservative")
        self.assertAlmostEqual(sum(track.fusion_weights.values()), 1.0, places=7)
        self.assertTrue(all(value > 0.0 for value in track.covariance_position_m2))
        self.assertTrue(any(
            row.get("event") == "ci_fused"
            and row.get("message_id") == "GT000012"
            and row.get("fusion_method") == "covariance_intersection_conservative"
            for row in manager.audit_log
        ), "每个已接受 TrackMessage 都必须保留对应的 CI 审计事件")
        self.assertLess(sum(track.covariance_position_m2), min(sum(cov_a), sum(cov_b)))
        self.assertTrue(all(value >= min(left, right)
                            for value, left, right in zip(
                                track.covariance_position_m2, cov_a, cov_b)))

        # 同协方差来源不能被当作独立量测：CI 保持 100，而不是错误地缩到 50。
        equal_bus = CommBus(["A", "B"], CommConfig(policy=SHARE_IDEAL, seed=6),
                            extra_endpoint_ids=[GLOBAL_TRACK_ENDPOINT])
        equal_manager = GlobalTrackManager()
        for source, local, sequence in (("A", "A-T1", 1), ("B", "B-T1", 2)):
            equal_bus.publish_track(_message(f"EQ{sequence}", source, local, sequence,
                                             covariance=(100.0, 100.0, 100.0)), 0.0)
        equal_manager.ingest_arrived(equal_bus, 0.0)
        equal_track = next(iter(equal_manager.tracks.values()))
        for value in equal_track.covariance_position_m2:
            self.assertAlmostEqual(value, 100.0, places=9)

    def test_delayed_track_is_projected_to_fusion_time_without_false_freshness(self) -> None:
        bus = CommBus(["A"], CommConfig(policy=SHARE_IDEAL, seed=77),
                      extra_endpoint_ids=[GLOBAL_TRACK_ENDPOINT])
        manager = GlobalTrackManager()
        # 状态在 t=0 形成、t=2 才抵达；中央只能用报文速度和声明过程噪声推演，
        # 不能把 t=0 位置伪装成 t=2 观测。
        bus.publish_track(_message("delayed", "A", "A-T1", 1,
                                   position=(0.0, 0.0, 0.0), now=0.0,
                                   covariance=(100.0, 100.0, 100.0),
                                   velocity=(10.0, 0.0, 0.0)), 0.0)
        manager.ingest_arrived(bus, 2.0)
        track = next(iter(manager.tracks.values()))
        self.assertEqual(track.last_state_time_s, 2.0)
        self.assertEqual(track.position_m, (20.0, 0.0, 0.0))
        self.assertEqual(track.covariance_position_m2, (300.0, 300.0, 300.0))
        self.assertEqual(track.information_age_s(2.0), 2.0)
        audit = next(row for row in manager.audit_log
                     if row.get("event") == "ci_fused")
        self.assertEqual(audit["message_state_timestamp_s"], 0.0)
        self.assertEqual(audit["fused_at_s"], 2.0)
        self.assertEqual(audit["projection_delta_s"], 2.0)
        self.assertEqual(audit["message_position_m"], [0.0, 0.0, 0.0])
        self.assertEqual(audit["effective_projection_velocity_mps"],
                         [10.0, 0.0, 0.0])
        self.assertEqual(audit["projected_message_position_m"],
                         [20.0, 0.0, 0.0])

    def test_association_reconnect_and_audit_are_stable(self) -> None:
        bus = CommBus(["A", "B"], CommConfig(policy=SHARE_IDEAL, seed=3),
                      extra_endpoint_ids=[GLOBAL_TRACK_ENDPOINT])
        manager = GlobalTrackManager()
        bus.publish_track(_message("GT000001", "A", "A-T1", 1), 0.0)
        manager.ingest_arrived(bus, 0.0)
        global_id = manager.local_to_global[("A", "A-T1")]

        # 另一节点的相邻 local track 经空间门控关联到同一 stable global ID。
        bus.publish_track(_message("GT000002", "B", "B-T9", 2,
                                   position=(100.0, 0.0, 0.0)), 1.0)
        manager.predict_to(1.0)
        manager.ingest_arrived(bus, 1.0)
        self.assertEqual(manager.local_to_global[("B", "B-T9")], global_id)

        # 节点短暂失联后仍先 coast；重接入时映射优先恢复原 global ID。
        manager.predict_to(2.0)
        self.assertEqual(manager.tracks[global_id].status, "coasting")
        bus.publish_track(_message("GT000003", "A", "A-T1", 3,
                                   position=(25.0, 0.0, 0.0), now=3.0), 3.0)
        manager.predict_to(3.0)
        manager.ingest_arrived(bus, 3.0)
        self.assertEqual(manager.local_to_global[("A", "A-T1")], global_id)
        # 乱序/重复序号不能覆写较新的关联，且必须留下拒绝证据。
        bus.publish_track(_message("GT000004", "A", "A-T1", 2,
                                   position=(20.0, 0.0, 0.0), now=4.0), 4.0)
        manager.predict_to(4.0)
        manager.ingest_arrived(bus, 4.0)
        self.assertEqual(manager.local_to_global[("A", "A-T1")], global_id)
        self.assertTrue(any(row["reason"] == "mapping_reused"
                            for row in manager.audit_log
                            if row["event"] == "association"))
        self.assertTrue(any(row["reason"] == "duplicate_or_out_of_order_sequence"
                            for row in manager.audit_log
                            if row["event"] == "association"))
        self.assertNotIn("truth", json.dumps(manager.report(3.0)).lower())

    def test_higher_sequence_with_older_state_timestamp_is_rejected(self) -> None:
        bus = CommBus(["A"], CommConfig(policy=SHARE_IDEAL, seed=13),
                      extra_endpoint_ids=[GLOBAL_TRACK_ENDPOINT])
        manager = GlobalTrackManager()
        bus.publish_track(_message("new", "A", "A-local", 1,
                                   position=(100.0, 0.0, 0.0), now=5.0), 5.0)
        manager.ingest_arrived(bus, 5.0)
        global_id = manager.local_to_global[("A", "A-local")]
        before = manager.tracks[global_id].position_m
        bus.publish_track(_message("old-state-new-seq", "A", "A-local", 2,
                                   position=(-900.0, 0.0, 0.0), now=4.0), 6.0)
        manager.predict_to(6.0)
        manager.ingest_arrived(bus, 6.0)
        self.assertNotEqual(manager.tracks[global_id].position_m[0], -900.0)
        self.assertEqual(manager.tracks[global_id].position_m, before)
        self.assertTrue(any(row.get("reason") == "non_monotonic_state_timestamp"
                            for row in manager.audit_log))

    def test_global_track_drops_and_tombstone_cannot_be_reused(self) -> None:
        bus = CommBus(["A"], CommConfig(policy=SHARE_IDEAL, seed=14),
                      extra_endpoint_ids=[GLOBAL_TRACK_ENDPOINT])
        manager = GlobalTrackManager(GlobalTrackConfig(max_coast_s=3.0))
        bus.publish_track(_message("first", "A", "A-old", 1, now=0.0), 0.0)
        manager.ingest_arrived(bus, 0.0)
        old_id = manager.local_to_global[("A", "A-old")]
        manager.predict_to(0.0)
        manager.predict_to(4.0)
        self.assertNotIn(old_id, manager.tracks)
        self.assertEqual(manager.report(4.0)["dropped_tracks"][0]
                         ["global_track_id"], old_id)
        bus.publish_track(_message("reentry", "A", "A-new", 1,
                                   position=(40.0, 0.0, 0.0), now=4.0), 4.0)
        manager.ingest_arrived(bus, 4.0)
        self.assertNotEqual(manager.local_to_global[("A", "A-new")], old_id)

    def test_tower_observation_is_separate_readonly_rm_obs_v2(self) -> None:
        bus = CommBus(["A", "B"], CommConfig(policy=SHARE_IDEAL, seed=9),
                      extra_endpoint_ids=[GLOBAL_TRACK_ENDPOINT])
        manager = GlobalTrackManager()
        bus.publish_track(_message("GT100001", "A", "A-T1", 1), 0.0)
        bus.publish_track(_message("GT100002", "B", "B-T9", 2,
                                   position=(20.0, 0.0, 0.0)), 0.0)
        manager.ingest_arrived(bus, 0.0)
        observation = GlobalObservation.from_manager(manager, 0.0)
        payload = observation.to_dict()
        self.assertEqual(payload["schema_version"], GLOBAL_OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(len(payload["tracks"]), 1)
        self.assertEqual(payload["tracks"][0]["coverage_state"], "handover")
        self.assertTrue(payload["tracks"][0]["handover_recent"])
        self.assertEqual(set(payload["tracks"][0]["source_node_ids"]), {"A", "B"})
        self.assertEqual(global_observation_diagnostics(observation)["n_global_tracks"], 1)
        self.assertNotIn("truth", json.dumps(payload).lower())


class TestDeterministicTowerAcceptance(unittest.TestCase):
    """阶段 7：不依赖大场景随机性的塔台最小端到端验收。"""

    def _ideal_bus(self) -> CommBus:
        return CommBus(["A", "B"], CommConfig(policy=SHARE_IDEAL, seed=31),
                       extra_endpoint_ids=[GLOBAL_TRACK_ENDPOINT])

    def test_a_same_target_different_local_ids_forms_one_global_id(self) -> None:
        bus, manager = self._ideal_bus(), GlobalTrackManager()
        bus.publish_track(_message("A1", "A", "A-local-1", 1,
                                   position=(100.0, 0.0, 0.0)), 0.0)
        bus.publish_track(_message("B1", "B", "B-local-9", 2,
                                   position=(120.0, 0.0, 0.0)), 0.0)
        self.assertEqual(manager.ingest_arrived(bus, 0.0), 2)
        self.assertEqual(len(manager.tracks), 1)
        self.assertEqual(manager.local_to_global[("A", "A-local-1")],
                         manager.local_to_global[("B", "B-local-9")])
        self.assertEqual(manager.ingest_arrived(bus, 0.0), 0,
                         "同一已到达消息只能消费一次")

    def test_b_handover_keeps_global_id_when_a_stops_and_b_continues(self) -> None:
        bus, manager = self._ideal_bus(), GlobalTrackManager()
        bus.publish_track(_message("A1", "A", "A-local", 1), 0.0)
        manager.ingest_arrived(bus, 0.0)
        global_id = manager.local_to_global[("A", "A-local")]
        manager.predict_to(1.0)
        bus.publish_track(_message("B1", "B", "B-local", 2,
                                   position=(10.0, 0.0, 0.0), now=1.0), 1.0)
        manager.ingest_arrived(bus, 1.0)
        manager.predict_to(2.0)
        bus.publish_track(_message("B2", "B", "B-local", 3,
                                   position=(20.0, 0.0, 0.0), now=2.0), 2.0)
        manager.ingest_arrived(bus, 2.0)
        self.assertEqual(len(manager.tracks), 1)
        self.assertEqual(manager.local_to_global[("B", "B-local")], global_id)
        track = manager.tracks[global_id]
        self.assertEqual(track.last_reporting_source_node_id, "B")
        self.assertGreaterEqual(track.handover_count, 1)

    def test_c_expired_and_out_of_order_messages_cannot_overwrite_newer_state(self) -> None:
        delayed_bus = CommBus(["A"], CommConfig(
            policy=SHARE_CONSTRAINED, base_delay_s=2.0, expiry_s=1.0, seed=32,
        ), extra_endpoint_ids=[GLOBAL_TRACK_ENDPOINT])
        delayed_manager = GlobalTrackManager()
        delayed_bus.publish_track(_message("late", "A", "A-local", 1), 0.0)
        self.assertTrue(delayed_bus.log[0].dropped)
        self.assertEqual(delayed_bus.log[0].drop_reason, "expired")
        self.assertEqual(delayed_manager.ingest_arrived(delayed_bus, 2.0), 0)
        self.assertTrue(any(row["reason"] == "expired"
                            for row in delayed_manager.audit_log))

        bus, manager = self._ideal_bus(), GlobalTrackManager()
        bus.publish_track(_message("new", "A", "A-local", 5,
                                   position=(500.0, 0.0, 0.0), now=5.0), 5.0)
        manager.ingest_arrived(bus, 5.0)
        global_id = manager.local_to_global[("A", "A-local")]
        before = manager.tracks[global_id].position_m
        bus.publish_track(_message("old", "A", "A-local", 4,
                                   position=(-500.0, 0.0, 0.0), now=4.0), 6.0)
        manager.predict_to(6.0)
        manager.ingest_arrived(bus, 6.0)
        self.assertNotEqual(manager.tracks[global_id].position_m[0], -500.0)
        self.assertTrue(any(row["reason"] == "duplicate_or_out_of_order_sequence"
                            for row in manager.audit_log))

    def test_d_near_distinct_targets_remain_separate(self) -> None:
        bus = self._ideal_bus()
        manager = GlobalTrackManager(GlobalTrackConfig(gate_distance_m=100.0,
                                                        reconnect_gate_distance_m=200.0))
        bus.publish_track(_message("T1", "A", "A-target-1", 1,
                                   position=(0.0, 0.0, 0.0)), 0.0)
        bus.publish_track(_message("T2", "B", "B-target-2", 2,
                                   position=(180.0, 0.0, 0.0)), 0.0)
        manager.ingest_arrived(bus, 0.0)
        self.assertEqual(len(manager.tracks), 2)
        self.assertNotEqual(manager.local_to_global[("A", "A-target-1")],
                            manager.local_to_global[("B", "B-target-2")])

    def test_retained_and_active_sources_are_distinct_after_timeout(self) -> None:
        bus = self._ideal_bus()
        manager = GlobalTrackManager(GlobalTrackConfig(max_source_age_s=2.0))
        bus.publish_track(_message("A1", "A", "A-T1", 1, now=0.0), 0.0)
        bus.publish_track(_message("B1", "B", "B-T1", 1, now=0.0), 0.0)
        manager.ingest_arrived(bus, 0.0)
        manager.predict_to(0.0)
        manager.predict_to(3.0)
        track = manager.report(3.0)["tracks"][0]
        self.assertEqual(track["participating_source_nodes"], ["A", "B"])
        self.assertEqual(track["active_source_nodes"], [])

        bus.publish_track(_message("A2", "A", "A-T2", 1,
                                   position=(30.0, 0.0, 0.0), now=3.0), 3.0)
        manager.ingest_arrived(bus, 3.0)
        track = manager.report(3.0)["tracks"][0]
        self.assertEqual(track["participating_source_nodes"], ["A", "B"])
        self.assertEqual(track["active_source_nodes"], ["A"])
        self.assertEqual(track["global_track_id"], "GLOBAL_TRACK_1")


class TestGlobalTrackClosedLoop(unittest.TestCase):
    def test_off_mode_is_exactly_the_default_export(self) -> None:
        default = run_closed_loop(SchedulerPolicy.RULE, seed=17, steps=12,
                                  runtime_mode=RUNTIME_MODE_FEEDBACK)
        explicit_off = run_closed_loop(
            SchedulerPolicy.RULE, seed=17, steps=12,
            runtime_mode=RUNTIME_MODE_FEEDBACK,
            global_track_mode=GLOBAL_TRACK_MODE_OFF,
        )
        baseline = copy.deepcopy(default.to_dict())
        disabled = copy.deepcopy(explicit_off.to_dict())
        # 规划 wall-clock 是唯一的非确定性测量；它不属于路径语义。
        for payload in (baseline, disabled):
            payload["planning_time_s"] = 0.0
            payload["metrics"].pop("compute_time_s", None)
            payload["metrics"]["evaluation_vector"]["values"].pop(
                "compute_time", None)
        self.assertEqual(baseline, disabled)
        self.assertNotIn("global_track_report", baseline)

    def test_track_mode_keeps_measurement_share_and_accounts_track_bytes(self) -> None:
        result = run_closed_loop(
            SchedulerPolicy.RULE, seed=42, steps=16,
            runtime_mode=RUNTIME_MODE_FEEDBACK,
            global_track_mode=GLOBAL_TRACK_MODE_TRACK_FUSION,
        )
        shares = [row for row in result.runtime_log if row.get("event") == "share"]
        self.assertTrue(shares)
        self.assertTrue(any(row["n_measurement_messages"] == 1
                            and row["n_track_messages"] == 1 for row in shares))
        self.assertTrue(all(row["accounted_comm_bytes"] == row["sent_comm_bytes"]
                            for row in shares))
        self.assertEqual(
            result.metrics["comm_overhead_bytes"],
            sum(row["accounted_comm_bytes"] for row in shares),
            "TrackMessage 字节必须进入 UnifiedExecutor 的 COMM_BYTE 账本",
        )
        self.assertTrue(result.conservation["all_conserved"])
        report = result.global_track_report
        self.assertIsNotNone(report)
        self.assertGreater(report["n_global_tracks"], 0)
        self.assertTrue(report["audit"])
        self.assertFalse(any("truth" in key.lower()
                             for message in report["audit"]
                             for key in message))

    def test_predeclared_communication_modes_use_real_runtime_and_ledger(self) -> None:
        outputs = {}
        for mode in (
            GLOBAL_SHARE_MODE_NO_SHARE,
            GLOBAL_SHARE_MODE_MEASUREMENT,
            GLOBAL_SHARE_MODE_TRACK,
            GLOBAL_SHARE_MODE_EVENT_TRACK,
        ):
            outputs[mode] = run_closed_loop(
                SchedulerPolicy.RULE, seed=42, steps=16,
                runtime_mode=RUNTIME_MODE_FEEDBACK,
                global_track_mode=GLOBAL_TRACK_MODE_TRACK_FUSION,
                global_share_mode=mode,
            )
            self.assertTrue(outputs[mode].conservation["all_conserved"])

        no_share_rows = [row for row in outputs[GLOBAL_SHARE_MODE_NO_SHARE].runtime_log
                         if row.get("event") == "share"]
        measurement_rows = [row for row in outputs[GLOBAL_SHARE_MODE_MEASUREMENT].runtime_log
                            if row.get("event") == "share"]
        track_rows = [row for row in outputs[GLOBAL_SHARE_MODE_TRACK].runtime_log
                      if row.get("event") == "share"]
        event_rows = [row for row in outputs[GLOBAL_SHARE_MODE_EVENT_TRACK].runtime_log
                      if row.get("event") == "share"]
        self.assertEqual(no_share_rows, [])
        self.assertTrue(measurement_rows and all(
            row["n_measurement_messages"] == 1 and row["n_track_messages"] == 0
            for row in measurement_rows))
        self.assertTrue(track_rows and all(
            row["n_measurement_messages"] == 0 and row["n_track_messages"] >= 1
            for row in track_rows))
        self.assertTrue(event_rows and all(
            row["n_measurement_messages"] == 0 and row["n_track_messages"] >= 1
            and row["event_trigger_reasons"] for row in event_rows))
        self.assertLessEqual(len(event_rows), len(track_rows))
        for result, rows in ((outputs[GLOBAL_SHARE_MODE_MEASUREMENT], measurement_rows),
                             (outputs[GLOBAL_SHARE_MODE_TRACK], track_rows),
                             (outputs[GLOBAL_SHARE_MODE_EVENT_TRACK], event_rows)):
            self.assertEqual(result.metrics["comm_overhead_bytes"],
                             sum(row["sent_comm_bytes"] for row in rows))

    def test_one_share_batches_all_pending_local_tracks_with_exact_bytes(self) -> None:
        result = run_closed_loop(
            SchedulerPolicy.RULE, seed=20260920, steps=18,
            runtime_mode=RUNTIME_MODE_FEEDBACK,
            global_track_mode=GLOBAL_TRACK_MODE_TRACK_FUSION,
            global_share_mode=GLOBAL_SHARE_MODE_TRACK,
            target_specs=[
                {"target_id": "BATCH_T1", "x": -1600.0, "y": 0.0,
                 "vx": 0.0, "vy": 100.0},
                {"target_id": "BATCH_T2", "x": 1600.0, "y": 0.0,
                 "vx": 0.0, "vy": -100.0},
            ],
        )
        rows = [row for row in result.runtime_log
                if row.get("event") == "share"
                and row.get("n_track_messages", 0) > 1]
        self.assertTrue(rows, "双目标至少应有一次 share 同时发送多个 local tracks")
        for row in rows:
            self.assertEqual(len(row["track_local_ids"]), row["n_track_messages"])
            self.assertEqual(len(set(row["track_local_ids"])),
                             row["n_track_messages"])
            self.assertEqual(row["accounted_comm_bytes"],
                             row["sent_comm_bytes"])
            self.assertEqual(row["sent_comm_bytes"],
                             128.0 * row["n_track_messages"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
