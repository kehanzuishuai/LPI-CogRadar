"""融合结果 → 资源调度适配层测试（v4.5）。

验收重点（用户明确）
--------------------
> **断开远端消息后，调度器不能继续"知道"远端的新信息。**

因此本文件的中心是那条端到端用例 `TestRemoteDisconnectAcceptance`：
节点 B 的航迹由**真实的 `FusionCenter`** 输出（喂的是测量，不是真值），
经**真实 `CommBus`** 送到中央；断开链路后断言
中央视图**冻结**、年龄增长、且**不跟随底层真值**。

其余用例覆盖用户点名的边界：未到达消息 / 航迹删除 / 空列表 /
节点数量变化 / 记录重放 / 未知对象不得提前建任务 /
定长适配器不得混用旧 checkpoint。

不需要 torch。
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment_config as ec  # noqa: E402
from communication import (  # noqa: E402
    CommBus,
    CommConfig,
    SHARE_CONSTRAINED,
    SHARE_IDEAL,
    SHARE_NONE,
)
from communication.message import (  # noqa: E402
    MESSAGE_KIND_NODE_OBSERVATION,
    MeasurementMessage,
)
from fusion import FusionCenter, FusionConfig  # noqa: E402
from engine.geometry import Vec3  # noqa: E402
from resource_management.observation import (  # noqa: E402
    CentralObservationStore,
    FixedLengthAdapter,
    LEGACY_OBSERVATION_MODES,
    SCHEMA_VERSION,
    field_metadata,
    node_observation_from_fusion,
    observation_payload,
    observation_truth_violations,
    publish_node_observation,
)
from resource_management.tasks import (  # noqa: E402
    DuplicateTaskError,
    QueueTaskKind,
    TaskQueue,
    TaskStatus,
    UnknownObjectError,
)
from resource_management.units import ResourceUnit  # noqa: E402
from resource_management.model import NodeState, ResourceBudget  # noqa: E402


class _Measurement:
    """最小测量对象：只含**测量量**，没有任何真值字段。"""

    def __init__(self, sensor_id: str, candidate_id: str, time_s: float,
                 range_m: float, azimuth_deg: float = 0.0,
                 sensor_kind: str = "radar") -> None:
        self.sensor_id = sensor_id
        self.candidate_id = candidate_id
        self.sensor_kind = sensor_kind
        self.time_s = time_s
        self.range_m = range_m
        self.azimuth_deg = azimuth_deg
        self.elevation_deg = 0.0
        self.range_rate_mps = 0.0
        self.std_range_m = 20.0
        self.std_az_deg = 0.3
        self.std_el_deg = 0.3
        self.confidence = 1.0
        self.msg_id = ""
        self.platform_id = ""


def _node_state(node_id: str, slots: float = 5.0) -> NodeState:
    return NodeState(node_id=node_id, update_period_s=1.0,
                     budget=ResourceBudget(capacity={
                         ResourceUnit.SAMPLE_SLOT: slots,
                         ResourceUnit.PROCESSING_OP: 5.0,
                         ResourceUnit.COMM_BYTE: 4096.0,
                     }))


class _RemoteProbe:
    """节点 B 的"世界"：一个沿 +x 匀速运动的目标。

    ⚠️ 它只用来**生成测量**喂给节点 B 的 `FusionCenter`（扮演传感器）。
    调度器与中央观测**永远看不到**这个真值——这正是被测的隔离关系。
    """

    def __init__(self, x0: float = 3000.0, speed: float = 60.0) -> None:
        self.x0, self.speed = x0, speed

    def position_at(self, t: float) -> float:
        return self.x0 + self.speed * t

    def measurement_for(self, t: float) -> _Measurement:
        # 传感器在原点，目标沿 +y 方向？不：沿 +x，方位角 = 90°
        # 极坐标 → 直角：range·sin(az), range·cos(az)；az=90° → +x
        return _Measurement(sensor_id="S_B", candidate_id="C_B", time_s=t,
                            range_m=self.position_at(t), azimuth_deg=90.0)


def _remote_fusion(t: float, probe: _RemoteProbe,
                   center: FusionCenter) -> None:
    """把探针产生的测量喂进节点 B 的融合中心（**只喂测量**）。"""
    center.predict_to(t)
    center.update([probe.measurement_for(t)], t, {"S_B": Vec3(0.0, 0.0, 0.0)})


class TestRemoteDisconnectAcceptance(unittest.TestCase):
    """**验收核心**：断开远端消息后，中央不得继续知道远端的新信息。"""

    def _build(self, policy: str):
        bus = CommBus(["LOCAL", "REMOTE"], CommConfig(policy=policy, seed=7))
        probe = _RemoteProbe()
        remote_fusion = FusionCenter("REMOTE", FusionConfig(),
                                     own_sensor_ids={"S_B"})
        store = CentralObservationStore(["LOCAL", "REMOTE"])
        remote_node = _node_state("REMOTE")
        return bus, probe, remote_fusion, store, remote_node

    def _tick(self, bus, probe, remote_fusion, store, remote_node, t: float,
              publish: bool) -> None:
        _remote_fusion(t, probe, remote_fusion)
        observation = node_observation_from_fusion(
            remote_fusion, remote_node, now_s=t)
        if publish:
            publish_node_observation(observation, bus, "REMOTE", now_s=t)
        store.ingest_arrived(bus, "LOCAL", t)

    def test_central_view_freezes_after_disconnect(self) -> None:
        bus, probe, remote_fusion, store, remote_node = self._build(SHARE_IDEAL)

        # --- 阶段 1：链路正常（t=1..4），中央能看到远端航迹在推进
        for t in (1.0, 2.0, 3.0, 4.0):
            self._tick(bus, probe, remote_fusion, store, remote_node, t,
                       publish=True)
        view_ok = store.observe(4.0)
        self.assertTrue(view_ok.node_valid_mask[1], "远端摘要应当已到达")
        remote_tracks = view_ok.nodes[1].track_ids()
        self.assertTrue(remote_tracks, "远端应当已经有航迹")
        position_online = view_ok.nodes[1].tracks[0].position[0]
        self.assertAlmostEqual(position_online, probe.position_at(4.0), delta=60.0)

        # --- 阶段 2：**断开**链路（t=5..10），但底层目标仍在运动
        for t in (5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
            self._tick(bus, probe, remote_fusion, store, remote_node, t,
                       publish=False)

        view_off = store.observe(10.0)
        position_offline = view_off.nodes[1].tracks[0].position[0]

        # ① 中央的远端位置**冻结**在最后一次到达时的值
        self.assertAlmostEqual(position_offline, position_online, places=6,
                               msg="断开后中央不再收到摘要，位置必须冻结")
        # ② 它**没有**跟随底层真值：真值已经走了 360 m
        truth_now = probe.position_at(10.0)
        self.assertGreater(truth_now - position_offline, 300.0,
                           "中央视图竟然跟上了真值——说明存在隐藏真值路径")
        # ③ 节点摘要年龄**随全局时钟增长**
        age = view_off.node_information_age_s[1]
        self.assertAlmostEqual(age, 6.0, places=6,
                               msg="摘要年龄应为 10-4=6 s")
        # ④ 缺失信息被显式标出，而不是悄悄补齐
        self.assertTrue(any("summary_age" in item
                            for item in view_off.missing), view_off.missing)

    def test_no_share_policy_yields_no_remote_knowledge_at_all(self) -> None:
        bus, probe, remote_fusion, store, remote_node = self._build(SHARE_NONE)
        for t in (1.0, 2.0, 3.0):
            self._tick(bus, probe, remote_fusion, store, remote_node, t,
                       publish=True)
        view = store.observe(3.0)
        self.assertFalse(view.node_valid_mask[1],
                         "no_share 下中央不该收到任何远端摘要")
        self.assertEqual(view.all_track_ids(), [],
                         "中央不该看到任何远端航迹")
        self.assertTrue(any("never_arrived" in item for item in view.missing))

    def test_not_yet_arrived_message_is_not_readable(self) -> None:
        """**未到达**的消息内容不得被读取（延迟链路下逐条验证）。

        ⚠️ 必须用 `SHARE_CONSTRAINED`：`SHARE_IDEAL` 的链路在
        `CommBus._build_links` 里**硬编码** `base_delay_s=0`，
        配置里的延迟会被忽略——第一版测试就因此没测到"在途"。
        """
        bus = CommBus(["LOCAL", "REMOTE"],
                      CommConfig(policy=SHARE_CONSTRAINED, seed=3,
                                 base_delay_s=5.0))
        probe = _RemoteProbe()
        fusion = FusionCenter("REMOTE", FusionConfig(), own_sensor_ids={"S_B"})
        store = CentralObservationStore(["LOCAL", "REMOTE"])
        node = _node_state("REMOTE")
        _remote_fusion(1.0, probe, fusion)
        publish_node_observation(node_observation_from_fusion(fusion, node, 1.0),
                                 bus, "REMOTE", now_s=1.0)
        # t=2 时消息还在路上：**总线本身**就不会把它交出来
        delivered = bus.consume("LOCAL", 2.0)
        self.assertEqual(delivered, [], "在途消息被总线交付了")
        store.ingest_arrived(bus, "LOCAL", 2.0)
        self.assertEqual(store.n_ingested, 0, "未到达的消息被读取了")
        view = store.observe(2.0)
        self.assertFalse(view.node_valid_mask[1])

        # 第二层防线：即便调用方绕过总线直接递消息，store 也必须拒绝
        in_flight = bus.in_flight(2.0)
        self.assertEqual(len(in_flight), 1, "应当有 1 条在途消息")
        self.assertFalse(store.ingest(in_flight[0], 2.0),
                         "store 接受了未到达的消息——二层防线失效")
        self.assertEqual(store.rejected_not_arrived, 1)
        self.assertEqual(store.n_ingested, 0)

        # t=6 之后应当已到达
        store.ingest_arrived(bus, "LOCAL", 7.0)
        self.assertEqual(store.n_ingested, 1)
        self.assertTrue(store.observe(7.0).node_valid_mask[1])


class TestTrackDeletionAndEmptyList(unittest.TestCase):
    def test_track_deletion_removes_it_from_observation(self) -> None:
        fusion = FusionCenter("A", FusionConfig(), own_sensor_ids={"S"})
        node = _node_state("A")
        # 喂足够的测量建立航迹
        for t in (1.0, 2.0):
            fusion.predict_to(t)
            fusion.update([_Measurement("S", "C1", t, 5000.0)], t,
                          {"S": Vec3(0.0, 0.0, 0.0)})
        observation = node_observation_from_fusion(fusion, node, 2.0)
        self.assertTrue(observation.track_ids())
        # 连续漏检直到航迹被删除（drop_after_misses=5）
        for t in range(3, 12):
            fusion.predict_to(float(t))
            fusion.update([], float(t), {"S": Vec3(0.0, 0.0, 0.0)})
        after = node_observation_from_fusion(fusion, node, 11.0)
        self.assertEqual(after.track_ids(), [],
                         "航迹被删除后必须从观测里消失")
        self.assertIn("no_track_local", after.missing)

    def test_empty_observation_is_valid_and_creates_only_sample_task(self) -> None:
        """空列表：结构有效、掩码为空，且**只**产生与目标无关的采样任务。"""
        from resource_management.observation import NodeObservation

        empty = NodeObservation(node_id="A", observed_at_s=1.0,
                                tracks=[], track_valid_mask=[],
                                missing=["no_track_local"])
        self.assertEqual(empty.track_ids(), [])
        self.assertEqual(json.dumps(empty.to_dict()) is not None, True)
        queue = TaskQueue()
        created = queue.create_from_observation(empty, 1.0)
        kinds = [task.kind for task in created]
        self.assertEqual(kinds, [QueueTaskKind.PREDEFINED_SAMPLE],
                         "空观测下只能下发预定义采样，不得为看不见的目标造任务")
        # 空观测也不能建"更新任务"
        with self.assertRaises(UnknownObjectError):
            queue.enqueue_for_observation(empty, QueueTaskKind.ESTIMATE_UPDATE,
                                          "u1")

    def test_no_task_for_unknown_object(self) -> None:
        """**未知对象不得依据 truth_id 提前创建任务**。"""
        from resource_management.observation import NodeObservation

        observation = NodeObservation(node_id="A", observed_at_s=1.0,
                                      tracks=[], track_valid_mask=[])
        queue = TaskQueue()
        for truth_like in ("TGT1", "ESM1", "TGT_ESCORT"):
            with self.assertRaises(UnknownObjectError, msg=truth_like):
                queue.enqueue_for_observation(
                    observation, QueueTaskKind.PROCESS, f"p-{truth_like}",
                    targets=(truth_like,))
        self.assertEqual(queue.tasks, [])


class TestNodeCountChanges(unittest.TestCase):
    def test_variable_node_list_and_mask(self) -> None:
        store = CentralObservationStore(["A", "B"])
        # 只有 A 送达
        message = MeasurementMessage(
            msg_id="M1", src_platform_id="A", src_sensor_id="S_A", seq=1,
            generated_at=1.0, sent_at=1.0, arrived_at=1.0,
            kind=MESSAGE_KIND_NODE_OBSERVATION,
            payload={"node_id": "A", "observed_at_s": 1.0, "n_tracks": 1,
                     "track_0_id": "A-T1", "track_0_x": 10.0,
                     "track_0_y": 20.0, "track_0_z": 0.0,
                     "track_0_vx": 0.0, "track_0_vy": 0.0, "track_0_vz": 0.0,
                     "track_0_sigma_x": 5.0, "track_0_sigma_y": 5.0,
                     "track_0_sigma_z": 5.0, "track_0_last_meas_s": 1.0,
                     "track_0_last_fusion_s": 1.0, "track_0_age_s": 0.0,
                     "track_0_coasting": False, "track_0_n_sources": 1,
                     "track_0_sensors": "S_A", "track_0_platforms": "A",
                     "remaining_sample_slot": 3.0},
        )
        store.ingest(message, now_s=1.0)
        view = store.observe(1.0)
        self.assertEqual(view.node_ids(), ["A", "B"])
        self.assertEqual(view.node_valid_mask, [True, False])
        self.assertEqual(view.all_track_ids(), ["A-T1"])
        self.assertIsNone(view.node_information_age_s[1])

    def test_node_added_later_appears_as_missing_until_arrival(self) -> None:
        store = CentralObservationStore(["A", "B", "C"])
        view = store.observe(5.0)
        self.assertEqual(len(view.nodes), 3)
        self.assertEqual(view.node_valid_mask, [False, False, False])
        self.assertEqual(len(view.missing), 3)


class TestReplayDeterminism(unittest.TestCase):
    """记录重放：同一序列重放必须得到**逐位相同**的观测与任务集。"""

    def _run(self) -> tuple:
        bus = CommBus(["LOCAL", "REMOTE"],
                      CommConfig(policy=SHARE_IDEAL, seed=11))
        probe = _RemoteProbe()
        fusion = FusionCenter("REMOTE", FusionConfig(), own_sensor_ids={"S_B"})
        store = CentralObservationStore(["LOCAL", "REMOTE"])
        node = _node_state("REMOTE")
        queue = TaskQueue()
        for t in (1.0, 2.0, 3.0, 4.0):
            _remote_fusion(t, probe, fusion)
            observation = node_observation_from_fusion(fusion, node, t)
            publish_node_observation(observation, bus, "REMOTE", now_s=t)
            store.ingest_arrived(bus, "LOCAL", t)
            if t <= 2.0:
                queue.create_from_observation(observation, t)
        view = store.observe(4.0)
        return (json.dumps(view.to_dict(), sort_keys=True, ensure_ascii=False),
                json.dumps([task.to_dict() for task in queue.tasks],
                           sort_keys=True, ensure_ascii=False))

    def test_replay_is_bit_identical(self) -> None:
        first_view, first_tasks = self._run()
        second_view, second_tasks = self._run()
        self.assertEqual(first_view, second_view)
        self.assertEqual(first_tasks, second_tasks)


class TestTaskQueueContract(unittest.TestCase):
    def _observation(self, node_id: str = "A", track_id: str = "A-T1"):
        from resource_management.observation import (
            NodeObservation,
            TrackObservation,
        )

        return NodeObservation(
            node_id=node_id, observed_at_s=1.0,
            tracks=[TrackObservation(
                track_id=track_id, position=(1.0, 2.0, 3.0),
                velocity=(0.0, 0.0, 0.0), sigma_position=(10.0, 10.0, 10.0),
                last_measurement_time_s=1.0, last_fusion_time_s=1.0,
                information_age_s=0.0, coasting=False, n_sources=1,
                source_sensor_ids=("S",), platforms=(node_id,),
                local_updates=1, remote_updates=0)],
            track_valid_mask=[True])

    def test_tasks_carry_release_deadline_cost_node_and_status(self) -> None:
        queue = TaskQueue()
        observation = self._observation()
        task = queue.enqueue_for_observation(
            observation, QueueTaskKind.ESTIMATE_UPDATE, "u1",
            deadline_s=6.0)
        self.assertAlmostEqual(task.release_time_s, 1.0)
        self.assertAlmostEqual(task.deadline_s, 6.0)
        self.assertEqual(task.node_id, "A")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertGreater(task.estimated_cost[ResourceUnit.PROCESSING_OP], 0.0)
        self.assertEqual(task.targets, ("A-T1",))
        self.assertIn("A-T1", task.created_from,
                      "溯源字段必须记下选中的目标，否则无法回答'这个任务为什么存在'")

    def test_duplicate_key_rejected(self) -> None:
        queue = TaskQueue()
        observation = self._observation()
        queue.enqueue_for_observation(observation, QueueTaskKind.PROCESS,
                                      "p1", idempotency_key="k1")
        with self.assertRaises(DuplicateTaskError):
            queue.enqueue_for_observation(observation, QueueTaskKind.PROCESS,
                                          "p2", idempotency_key="k1")

    def test_due_and_expire(self) -> None:
        queue = TaskQueue()
        observation = self._observation()
        queue.enqueue_for_observation(observation, QueueTaskKind.PROCESS, "p1",
                                      release_time_s=5.0, deadline_s=8.0)
        self.assertEqual(queue.due(4.0), [])
        self.assertEqual(len(queue.due(5.0)), 1)
        expired = queue.expire_overdue(9.0)
        self.assertEqual([task.task_id for task in expired], ["p1"])
        self.assertEqual(queue.due(9.0), [], "超期任务不再释放")

    def test_four_kinds_supported(self) -> None:
        queue = TaskQueue()
        observation = self._observation()
        created = queue.create_from_observation(observation, 1.0)
        kinds = {task.kind for task in created}
        self.assertIn(QueueTaskKind.PREDEFINED_SAMPLE, kinds)
        self.assertIn(QueueTaskKind.ESTIMATE_UPDATE, kinds)
        self.assertIn(QueueTaskKind.SHARE, kinds)
        # PROCESS 由调用方按需补下（同样是"绑定已有航迹"的语义）
        queue.enqueue_for_observation(observation, QueueTaskKind.PROCESS,
                                      "proc-1", targets=("A-T1",))
        kinds = {task.kind for task in queue.tasks}
        self.assertEqual(len(kinds), 4)

    def test_task_to_executor_task_request(self) -> None:
        queue = TaskQueue()
        observation = self._observation()
        task = queue.enqueue_for_observation(
            observation, QueueTaskKind.ESTIMATE_UPDATE, "u1")
        request = task.to_task_request(start_s=1.0)
        self.assertEqual(request.node_id, "A")
        self.assertEqual(request.start_s, 1.0)
        self.assertEqual(request.entities, ("A-T1",))


class TestFixedLengthAdapter(unittest.TestCase):
    def test_adapter_has_its_own_schema_and_dim(self) -> None:
        adapter = FixedLengthAdapter(slots=4)
        self.assertTrue(adapter.schema_version.startswith(SCHEMA_VERSION))
        self.assertEqual(len(adapter.feature_names()), adapter.output_dim)

    def test_adapter_refuses_legacy_checkpoint_dim(self) -> None:
        """**不得**把变长观测塞进旧路径的维度。"""
        adapter = FixedLengthAdapter(slots=4)
        for legacy_dim in (12, 16, 53):
            with self.assertRaises(ValueError) as ctx:
                adapter.check_checkpoint_dim(legacy_dim)
            self.assertIn("不匹配", str(ctx.exception))
        adapter.check_checkpoint_dim(adapter.output_dim)   # 自己的维度可用

    def test_encode_is_fixed_length_and_uses_mask(self) -> None:
        from resource_management.observation import NodeObservation

        adapter = FixedLengthAdapter(slots=2)
        empty = NodeObservation(node_id="A", observed_at_s=0.0)
        vector = adapter.encode(empty)
        self.assertEqual(len(vector), adapter.output_dim)
        self.assertEqual(vector[0], 0.0, "空观测的槽位掩码必须为 0")
        self.assertEqual(adapter.overflow_count(empty), 0)

    def test_overflow_is_reported_not_silently_dropped(self) -> None:
        from resource_management.observation import (
            NodeObservation,
            TrackObservation,
        )

        adapter = FixedLengthAdapter(slots=2)
        tracks = [TrackObservation(
            track_id=f"T{i}", position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0), sigma_position=(1.0, 1.0, 1.0),
            last_measurement_time_s=0.0, last_fusion_time_s=0.0,
            information_age_s=0.0, coasting=False, n_sources=1,
            source_sensor_ids=(), platforms=(), local_updates=0,
            remote_updates=0) for i in range(5)]
        observation = NodeObservation(node_id="A", observed_at_s=0.0,
                                      tracks=tracks,
                                      track_valid_mask=[True] * 5)
        self.assertEqual(adapter.overflow_count(observation), 3,
                         "超出槽位的航迹必须被显式报出，不能静默丢弃")
        self.assertEqual(len(adapter.encode(observation)), adapter.output_dim)


class TestFieldMetadataAndTruthIsolation(unittest.TestCase):
    def test_every_field_has_unit_frame_visibility_provenance(self) -> None:
        for name, spec in field_metadata().items():
            for key in ("unit", "frame", "visibility", "provenance"):
                self.assertTrue(spec[key], f"{name}.{key} 未标注")
            self.assertIn(spec["visibility"],
                          ("local", "shared", "derived"))
            self.assertIn(spec["frame"], ("ENU", "none"))

    def test_observation_module_does_not_import_truth_layers(self) -> None:
        """用 **AST 扫 import**，而不是子串匹配。

        子串匹配会把文档字符串里"本模块不 import `engine.simulator`"这句
        **声明**当成违规——本工程已经因此误报过一次
        （见 `validation/checks.py::check_truth_isolation`）。
        只有真正的 import 语句才算依赖。
        """
        import ast

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "resource_management", "observation.py")
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        modules: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        for forbidden in ("engine", "engine.simulator", "engine.scene",
                          "sensor", "torch"):
            offenders = [name for name in modules
                         if name == forbidden
                         or name.startswith(forbidden + ".")]
            self.assertEqual(
                offenders, [],
                f"适配层不得依赖真值/物理层：import 了 {offenders}")

    def test_serialized_observation_has_no_truth_channel(self) -> None:
        fusion = FusionCenter("A", FusionConfig(), own_sensor_ids={"S"})
        for t in (1.0, 2.0):
            fusion.predict_to(t)
            fusion.update([_Measurement("S", "C1", t, 5000.0)], t,
                          {"S": Vec3(0.0, 0.0, 0.0)})
        observation = node_observation_from_fusion(fusion, _node_state("A"), 2.0)
        payload = observation.to_dict()
        self.assertEqual(observation_truth_violations(payload), [])
        self.assertEqual(
            observation_truth_violations(json.loads(json.dumps(payload))), [])

    def test_payload_whitelist_blocks_truth_fields(self) -> None:
        from communication.message import PayloadViolation, assert_payload_clean

        fusion = FusionCenter("A", FusionConfig(), own_sensor_ids={"S"})
        fusion.predict_to(1.0)
        fusion.update([_Measurement("S", "C1", 1.0, 5000.0)], 1.0,
                      {"S": Vec3(0.0, 0.0, 0.0)})
        payload = observation_payload(
            node_observation_from_fusion(fusion, _node_state("A"), 1.0))
        assert_payload_clean(payload, MESSAGE_KIND_NODE_OBSERVATION)
        payload["track_0_truth_id"] = "TGT1"
        with self.assertRaises(PayloadViolation):
            assert_payload_clean(payload, MESSAGE_KIND_NODE_OBSERVATION)

    def test_legacy_paths_are_marked_and_separate(self) -> None:
        self.assertEqual(set(LEGACY_OBSERVATION_MODES),
                         {"full", "pomdp", "ideal", "realistic"})
        payload = CentralObservationStore(["A"]).observe(0.0).to_dict()
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(set(payload["legacy_observation_modes"]),
                         set(LEGACY_OBSERVATION_MODES))


class TestIntegrationWithRealEnvFusion(unittest.TestCase):
    """用**真实的 env + FusionCenter** 验证适配层读到的是真融合输出。"""

    def test_local_observation_matches_fusion_tracks(self) -> None:
        env = ec.make_env(observation_mode="realistic")
        env.reset(seed=42)
        for _ in range(6):
            env.step(6)
        tracker = env.tracker
        self.assertIsNotNone(tracker, "realistic 模式应有 FusionCenter")
        node = _node_state("LOCAL")
        observation = node_observation_from_fusion(tracker, node,
                                                   env.sim.current_time)
        # 观测的航迹 ID 必须与融合中心的航迹 ID 完全一致（稳定 ID 透传）
        self.assertEqual(observation.track_ids(),
                         [track.track_id for track in tracker.tracks
                          if track.last_measurement_time is not None])
        if observation.tracks:
            first = observation.tracks[0]
            source = tracker.tracks[0]
            self.assertAlmostEqual(first.position[0], source.position.x, places=6)
            self.assertAlmostEqual(first.position[1], source.position.y, places=6)
            self.assertGreaterEqual(first.information_age_s, 0.0)
        # 本地资源余量来自 NodeState，且带单位
        self.assertIn(ResourceUnit.SAMPLE_SLOT.value, observation.remaining)


if __name__ == "__main__":
    unittest.main(verbosity=2)
