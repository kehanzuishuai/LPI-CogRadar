"""ExecutionPlan 真正反向控制 Sensor -> Comm -> Fusion 的端到端测试。"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication import CommBus, CommConfig, SHARE_CONSTRAINED  # noqa: E402
from engine.simulator import Simulator  # noqa: E402
from fusion import FusionCenter, FusionConfig  # noqa: E402
from resource_management.clock import GlobalClock  # noqa: E402
from resource_management.closed_loop import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_NODE_BUDGETS,
    NODE_LAYOUT,
    RUNTIME_MODE_FEEDBACK,
    RUNTIME_MODE_LEGACY,
    RuntimeExecutionError,
    RuntimeExecutor,
    _radar,
    _sensor,
    _target,
    run_closed_loop,
)
from resource_management.executor import UnifiedExecutor  # noqa: E402
from resource_management.ledger import ResourceLedger  # noqa: E402
from resource_management.model import (  # noqa: E402
    ExecutionPlan,
    NodeState,
    ResourceBudget,
    TaskRequest,
)
from resource_management.scheduling import SchedulerPolicy  # noqa: E402
from resource_management.units import (  # noqa: E402
    BUDGET_UNITS,
    ResourceUnit,
    TaskKind,
)
from sensor.config import build_suite_from_config  # noqa: E402


class RuntimeFixture:
    def __init__(self) -> None:
        with open(BASE_CONFIG, "r", encoding="utf-8") as handle:
            base = json.load(handle)
        radar_base, target_base = base["radar"], base["targets"][0]
        overrides = {
            "radars": [
                _radar(radar_base, node_id, layout["x"], layout["y"],
                       layout["heading_deg"])
                for node_id, layout in NODE_LAYOUT.items()
            ],
            "targets": [
                _target(target_base, "TGT_1", 1500.0, -5000.0, 0.0, 100.0),
                _target(target_base, "TGT_2", -1200.0, -3000.0, 0.0, 100.0),
            ],
            "sensors": [
                _sensor(f"SENSOR_{node_id}", node_id,
                        layout["max_range_m"], az_fov_deg=45.0)
                for node_id, layout in NODE_LAYOUT.items()
            ],
            "occluders": [],
        }
        self.sim = Simulator(BASE_CONFIG)
        self.sim.load_config()
        self.sim.apply_overrides(extra=overrides)
        self.sim.reset(seed=42)
        raw = dict(getattr(self.sim, "_raw_config", {}) or {})
        self.suite = build_suite_from_config(
            self.sim.scene, raw, seed=42, noise_scale=1.0
        )
        self.positions = {
            sensor.sensor_id:
                self.sim.scene.by_id(sensor.config.mounting_id).position
            for sensor in self.suite.sensors
        }
        self.clock = GlobalClock(now_s=0.0)
        self.ledger = ResourceLedger()
        self.accounting = UnifiedExecutor(self.clock, self.ledger)
        capacity = {
            ResourceUnit.SAMPLE_SLOT: 20.0,
            ResourceUnit.PROCESSING_OP: 40.0,
            ResourceUnit.COMM_BYTE: 4096.0,
        }
        for node_id in NODE_LAYOUT:
            self.accounting.register_node(NodeState(
                node_id=node_id,
                update_period_s=1.0,
                budget=ResourceBudget(capacity=dict(capacity)),
            ))
        self.own_sensor = {
            node_id: f"SENSOR_{node_id}" for node_id in NODE_LAYOUT
        }
        self.centers = {
            node_id: FusionCenter(
                node_id, FusionConfig(), own_sensor_ids={self.own_sensor[node_id]}
            )
            for node_id in NODE_LAYOUT
        }
        self.bus = CommBus(
            list(NODE_LAYOUT),
            CommConfig(
                policy=SHARE_CONSTRAINED,
                base_delay_s=1.0,
                jitter_s=0.0,
                loss_prob=0.0,
                expiry_s=10.0,
                message_size_bytes=128.0,
                seed=42,
            ),
        )
        self.runtime = RuntimeExecutor(
            self.accounting, self.sim.scene, self.suite, self.centers,
            self.bus, self.positions, self.own_sensor,
        )
        self.counter = 0

    def tick(self) -> float:
        self.clock.advance(1.0, "test feedback tick")
        self.sim.scene.advance_all(1.0)
        self.runtime.begin_tick(self.clock.now_s)
        return self.clock.now_s

    def submit(self, node_id: str, kind: TaskKind) -> ExecutionPlan:
        self.counter += 1
        task = TaskRequest(
            task_id=f"{kind.value}-{node_id}-{self.counter}",
            node_id=node_id,
            kind=kind,
            start_s=self.clock.now_s,
            idempotency_key=f"runtime-{self.counter}",
        )
        plan = ExecutionPlan(
            plan_id=f"P{self.counter}", submit_time_s=self.clock.now_s,
            tasks=[task],
        )
        result = self.runtime.submit(plan)
        if result.n_applied != 1:
            raise AssertionError(result.to_dict())
        return plan


class TestPlanControlledRuntime(unittest.TestCase):
    def test_sampling_pause_prediction_share_arrival_and_recovery(self) -> None:
        world = RuntimeFixture()

        # t=1：只计划 A 采样。B 没有 sample 任务，不得暗中扫描。
        world.tick()
        first_plan = world.submit("NODE_A", TaskKind.SAMPLE)
        sensor_a = world.suite.by_id("SENSOR_NODE_A")
        sensor_b = world.suite.by_id("SENSOR_NODE_B")
        self.assertEqual(sensor_a.stats["scans"], 1)
        self.assertEqual(sensor_b.stats["scans"], 0)
        first_sample_events = [
            row for row in world.runtime.event_log
            if row.get("event") == "sample" and row.get("node_id") == "NODE_A"
        ]
        self.assertGreater(first_sample_events[-1]["n_measurements"], 0)

        # 同一计划不能第二次产生副作，账本和扫描数都不变。
        ledger_len = len(world.ledger.entries)
        with self.assertRaises(RuntimeExecutionError):
            world.runtime.submit(first_plan)
        self.assertEqual(len(world.ledger.entries), ledger_len)
        self.assertEqual(sensor_a.stats["scans"], 1)

        # t=2：只有 process 才能用已采样数据创建/更新航迹。
        world.tick()
        world.submit("NODE_A", TaskKind.PROCESS)
        self.assertTrue(world.centers["NODE_A"].tracks)
        track = world.centers["NODE_A"].tracks[0]
        initial_updates = track.local_updates
        sigma_after_update = max(
            track.sigma_position.x, track.sigma_position.y,
            track.sigma_position.z,
        )

        # t=3：share 任务才真正发送；之前 CommBus 一直为空。
        self.assertEqual(len(world.bus.log), 0)
        world.tick()
        world.submit("NODE_A", TaskKind.SHARE)
        self.assertEqual(len(world.bus.log), 1)
        message = world.bus.log[0]
        self.assertEqual(message.size_bytes, 128.0)
        self.assertFalse(any(
            str(key).startswith(("truth", "err_", "is_false_alarm"))
            for key in message.payload
        ))

        # 同一 t=3 尚未到达：B 即使 process 也不能融合未来消息。
        world.submit("NODE_B", TaskKind.PROCESS)
        before_arrival = world.runtime.event_log[-1]
        self.assertEqual(before_arrival["n_remote_measurements"], 0)
        self.assertFalse(world.centers["NODE_B"].tracks)

        # t=4 到达后才能被 B 的 Fusion 消费。
        world.tick()
        world.submit("NODE_B", TaskKind.PROCESS)
        after_arrival = world.runtime.event_log[-1]
        self.assertEqual(after_arrival["n_remote_measurements"], 1)
        self.assertTrue(after_arrival["fusion_updated"])
        self.assertTrue(world.centers["NODE_B"].tracks)

        # t=5：A 连续 3 tick 未采样，只允许预测。
        world.tick()
        self.assertEqual(sensor_a.stats["scans"], 1)
        sigma_after_pause = max(
            track.sigma_position.x, track.sigma_position.y,
            track.sigma_position.z,
        )
        self.assertGreater(sigma_after_pause, sigma_after_update)
        self.assertGreater(world.clock.now_s - track.last_measurement_time, 0.0)

        # t=6 恢复采样，t=7 执行 process 后航迹获得新测量更新。
        world.tick()
        world.submit("NODE_A", TaskKind.SAMPLE)
        self.assertEqual(sensor_a.stats["scans"], 2)
        world.tick()
        world.submit("NODE_A", TaskKind.PROCESS)
        track = world.centers["NODE_A"].tracks[0]
        self.assertGreater(track.local_updates, initial_updates)
        self.assertAlmostEqual(track.last_measurement_time, 6.0)

        totals = world.ledger.totals_by_node()["NODE_A"]
        self.assertEqual(totals["consumed_comm_byte"], 128.0)
        self.assertTrue(world.accounting.conservation_report()["all_conserved"])

    def test_integrated_feedback_mode_records_real_effects(self) -> None:
        result = run_closed_loop(
            SchedulerPolicy.RULE,
            seed=42,
            steps=12,
            runtime_mode=RUNTIME_MODE_FEEDBACK,
        )
        runtime = result.metrics["runtime_feedback"]
        self.assertGreater(runtime["n_sensor_measurements"], 0)
        self.assertGreater(runtime["n_fusion_measurements"], 0)
        self.assertGreater(runtime["n_messages_sent"], 0)
        self.assertEqual(runtime["sent_comm_bytes"],
                         runtime["accounted_comm_bytes"])
        self.assertEqual(runtime["n_duplicate_runtime_tasks"], 0)
        self.assertEqual(runtime["truth_payload_violations"], 0)
        self.assertTrue(result.conservation["all_conserved"])


class TestClosedLoopEndToEnd(unittest.TestCase):
    """闭环级端到端证据（用户在提示词里点名的四项证明）。

    与上面的 fixture 测试的分工：fixture 证明**单个机制**正确，
    这里证明**整条闭环在真实场景里成立**：
    融合航迹 → 资源调度 → 节点实际行为 → 新测量 → 通信 → 融合航迹。
    """

    STEPS = 16
    #: 停采样窗口。取 [3, 6]：前 3 tick 建立航迹，中间 4 tick 让 σ 明显增长，
    #: 之后留够"恢复"所需的 tick——恢复要经过 process → share → sample 三步，
    #: 窗口贴到末尾会让"恢复"这一半因步数不够而假失败（踩过一次）。
    OUTAGE = (3.0, 6.0)

    def _run(self, **kwargs):
        params = dict(policy=SchedulerPolicy.RULE, seed=42, steps=self.STEPS,
                      runtime_mode=RUNTIME_MODE_FEEDBACK)
        params.update(kwargs)
        return run_closed_loop(**params)

    # --- 证 1：关掉某雷达采样 → 测量数下降、不确定度上升 ---

    def test_sampling_outage_pauses_measurements_and_grows_uncertainty(self) -> None:
        result = self._run(mechanisms={
            "unavailable_windows": {"NODE_B": [self.OUTAGE]}})
        runtime = result.metrics["runtime_feedback"]

        # ① 整轮里 B 的传感器扫描次数显著少于 A（A 全程可用）
        scans = runtime["sensor_scans_by_node"]
        self.assertGreater(scans["NODE_A"], scans["NODE_B"])
        self.assertEqual(scans["NODE_B"],
                         runtime["n_sample_events"] - scans["NODE_A"],
                         "逐节点扫描数之和必须等于 sample 事件总数")

        # ② 不可用窗口内 B 一次扫描都没有
        start, end = self.OUTAGE
        in_window = [row for row in result.runtime_log
                     if row.get("event") == "sample"
                     and row.get("node_id") == "NODE_B"
                     and start <= float(row["time_s"]) <= end]
        self.assertEqual(in_window, [],
                         "B 在不可用窗口内不得触发任何传感器扫描")
        self.assertTrue(any(row.get("event") == "sample"
                            and row.get("node_id") == "NODE_B"
                            and float(row["time_s"]) > end
                            for row in result.runtime_log),
                        "窗口结束后 B 应当恢复采样")

        # ③ 窗口内 B 的航迹信息年龄与位置 σ **单调上升**（只预测、不更新）
        #
        # 统计范围**严格在窗口内**：窗口之后第一个 tick 节点已恢复，
        # σ 本来就该回落，把它算进来会让"单调上升"必然失败。
        series = self._node_series(result, "NODE_B")
        inside = [row for row in series if start < row["time_s"] <= end]
        self.assertGreaterEqual(len(inside), 3)
        ages = [row["mean_track_age_s"] for row in inside
                if row["mean_track_age_s"] is not None]
        sigmas = [row["mean_sigma_m"] for row in inside
                  if row["mean_sigma_m"] is not None]
        self.assertTrue(ages and sigmas)
        self.assertEqual(ages, sorted(ages), f"年龄必须单调增长：{ages}")
        self.assertEqual(sigmas, sorted(sigmas), f"σ 必须单调增长：{sigmas}")
        self.assertGreater(sigmas[-1], sigmas[0],
                           "停采样后不确定度必须真的变大")

    # --- 证 2：恢复调度 → 测量与航迹更新恢复 ---

    def test_recovery_resumes_sensor_updates(self) -> None:
        result = self._run(mechanisms={
            "unavailable_windows": {"NODE_B": [self.OUTAGE]}})
        start, end = self.OUTAGE
        b_samples = [float(row["time_s"]) for row in result.runtime_log
                     if row.get("event") == "sample"
                     and row.get("node_id") == "NODE_B"]
        self.assertTrue(b_samples and b_samples[-1] > end,
                        f"B 在窗口结束后必须再次采样：{b_samples}")
        # 恢复后 B 的 σ 必须回落（新的测量经 process 进入融合）
        series = self._node_series(result, "NODE_B")
        after = [row for row in series
                 if row["time_s"] > end and row["mean_sigma_m"] is not None]
        self.assertTrue(after)
        peak = max(row["mean_sigma_m"] for row in series
                   if row["mean_sigma_m"] is not None
                   and start < row["time_s"] <= end)
        self.assertLess(min(row["mean_sigma_m"] for row in after), peak,
                        "恢复调度后不确定度必须回落")

    # --- 证 3：只有执行 share 任务才发送消息 ---

    def test_share_task_gates_real_message_sending(self) -> None:
        normal = self._run()
        runtime = normal.metrics["runtime_feedback"]
        self.assertGreater(runtime["n_share_events"], 0)
        self.assertGreater(runtime["n_messages_sent"], 0)
        # 每条 share 事件最多一条真实消息；发送字节与账本字节一致
        self.assertLessEqual(runtime["n_messages_sent"],
                             runtime["n_share_events"])
        self.assertEqual(runtime["sent_comm_bytes"],
                         runtime["accounted_comm_bytes"])
        self.assertEqual(runtime["sent_comm_bytes"],
                         normal.metrics["comm_overhead_bytes"])

        # 通信预算置零 → 一个 share 任务都无法执行 → 一条消息都不发
        blocked = self._run(node_budgets={
            node_id: {**budget, ResourceUnit.COMM_BYTE: 0.0}
            for node_id, budget in DEFAULT_NODE_BUDGETS.items()})
        blocked_runtime = blocked.metrics["runtime_feedback"]
        self.assertEqual(blocked_runtime["n_share_events"], 0)
        self.assertEqual(blocked_runtime["n_messages_sent"], 0)
        self.assertEqual(blocked_runtime["sent_comm_bytes"], 0.0)
        self.assertEqual(blocked.metrics["comm_overhead_bytes"], 0.0)

    # --- 证 4：资源守恒、无重复执行、无真值泄漏 ---

    def test_runtime_invariants_hold_over_a_full_run(self) -> None:
        result = self._run()
        runtime = result.metrics["runtime_feedback"]

        # 每任务只执行一次：运行时任务数 == 唯一去重键数
        self.assertEqual(runtime["n_runtime_tasks"],
                         runtime["n_unique_task_keys"])
        self.assertEqual(runtime["n_duplicate_runtime_tasks"], 0)
        # 事件分类必须覆盖全部运行时任务
        self.assertEqual(
            runtime["n_sample_events"] + runtime["n_process_events"]
            + runtime["n_share_events"],
            runtime["n_runtime_tasks"])
        # 真值隔离：由载荷键扫描 + 中央观测序列化扫描统计，两者都为 0
        self.assertEqual(runtime["observed_payload_violations"], 0)
        self.assertEqual(runtime["observed_context_violations"], 0)
        self.assertEqual(runtime["truth_payload_violations"], 0)
        # 资源守恒
        self.assertTrue(result.metrics["conservation_all"])
        self.assertTrue(result.conservation["all_conserved"])

    def test_fusion_only_consumes_measurements_that_arrived(self) -> None:
        result = self._run()
        # 第一条 process 之前不可能有远端测量（消息还没到）
        first_process = next(row for row in result.runtime_log
                             if row.get("event") == "process")
        self.assertEqual(first_process["n_remote_measurements"], 0)
        # 事件的远端测量数不得超过"截至该时刻已发送"的消息数
        sent = 0
        for row in result.runtime_log:
            if row.get("event") == "share":
                sent += int(row.get("n_messages", 0))
            if row.get("event") == "process":
                self.assertLessEqual(
                    int(row["n_remote_measurements"]), sent,
                    f"t={row['time_s']} 融合了尚未发送的远端测量")
        # 离线误差表必须显式标注来源，调度器看不到它
        self.assertTrue(result.estimate_error_samples)
        self.assertTrue(all(sample["provenance"]
                            == "offline_truth_nearest_track"
                            for sample in result.estimate_error_samples))

    # --- 证 5：旧路径保留，且"只有新路径才让调度影响感知链" ---

    def test_only_feedback_mode_lets_scheduling_change_track_quality(self) -> None:
        """这是**闭环成立与否的判据**。

        * 旧路径（观测优先）：每 tick 无条件扫描与融合 → 三个策略的
          估计质量与误差**完全相同**，调度只改任务统计；
        * 新路径（计划控制）：不同策略产生**不同的测量数与估计质量**。
        """
        def quality(mode):
            out = {}
            for policy in (SchedulerPolicy.ROUND_ROBIN,
                           SchedulerPolicy.EDF,
                           SchedulerPolicy.RULE):
                result = run_closed_loop(policy, seed=42, steps=12,
                                         runtime_mode=mode)
                vector = result.metrics["evaluation_vector"]["values"]
                runtime = result.metrics.get("runtime_feedback") or {}
                out[policy.value] = (
                    round(float(vector["estimate_quality"]), 9),
                    int(runtime.get("n_sensor_measurements", -1)),
                )
            return out

        legacy = quality(RUNTIME_MODE_LEGACY)
        feedback = quality(RUNTIME_MODE_FEEDBACK)
        # 旧路径：质量对策略不敏感（这正是"闭环没成立"的症状）
        self.assertEqual(len({value[0] for value in legacy.values()}), 1,
                         f"旧路径的估计质量本应与策略无关：{legacy}")
        # 新路径：至少两个策略在测量数或质量上不同
        self.assertGreater(len({value for value in feedback.values()}), 1,
                           f"新路径下调度必须改变感知链：{feedback}")
        self.assertTrue(all(value[1] >= 0 for value in feedback.values()))

    def _node_series(self, result, node_id: str):
        rows = []
        for row in result.node_ages:
            if node_id not in row["node_ids"]:
                continue
            index = row["node_ids"].index(node_id)
            rows.append({
                "time_s": float(row["time_s"]),
                "mean_track_age_s": row["mean_track_age_s"][index],
                "mean_sigma_m": row["mean_sigma_m"][index],
                "n_tracks": row["n_tracks"][index],
            })
        return rows


if __name__ == "__main__":
    unittest.main(verbosity=2)
