"""非学习型资源调度的闭环运行器（v4.5）。

分层（**这个划分本身就是被测对象**）
------------------------------------
```
closed_loop.py（本模块，**唯一允许碰真值的地方**）
   ├─ 世界：Simulator / SensorSuite / 目标运动（真值只在这里）
   ├─ 每个节点：自己的 FusionCenter ← 只吃本节点传感器的**测量**
   ├─ 通信：CommBus ← 只搬**节点观测摘要**，中央只读已到达的
   └─ 调用：scheduler.plan() → executor.submit()
scheduling.py / observation.py / tasks.py（**不 import engine/sensor/fusion**）
```

调度器能看到的只有：`CentralObservation`（融合输出 + 已到达的通信状态）
与 `TaskQueue`。它**看不到**真值，也没有接口去改传感器/融合器内部字段——
本模块把世界与调度器隔开，正是为了让"调度输入确实来自融合与通信链"
成为可检验的结构事实，而不是一句声明。

已接入的场景机制（用户点名的四种）
----------------------------------
| 机制 | 实现 | 观察点 |
| --- | --- | --- |
| **节点不可用** | `unavailable_windows`：窗口内该节点不产出观测 | 该节点任务被 `not_eligible` 推迟 |
| **覆盖交接** | 两个雷达作用距离/位置不同，目标穿过交界 | 航迹在不同节点间接力，任务随之改派 |
| **通信延迟** | `CommConfig.base_delay_s`（受限共享） | 中央的节点摘要年龄增大；无到达则不规划 |
| **观测偏差** | 某节点传感器注入距离/方位偏差 | 该节点航迹协方差与残差变大 → 规则调度优先补它 |

⚠️ 本模块**不追求**"规则调度比轮询更好"。实际出现的资源冲突、服务不足与
失败情形一律保留为下一阶段的基准案例，不做粉饰。
"""

from __future__ import annotations

import json as _json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment_config as ec
from communication import CommBus, CommConfig, SHARE_CONSTRAINED, SHARE_IDEAL
from communication.message import MESSAGE_KIND_NODE_OBSERVATION
from engine.geometry import Vec3
from engine.simulator import Simulator
from fusion import FusionCenter, FusionConfig
from sensor.config import build_suite_from_config

from resource_management.clock import GlobalClock
from resource_management.executor import UnifiedExecutor
from resource_management.ledger import ResourceLedger
from resource_management.model import NodeState, PlanStatus, ResourceBudget
from resource_management.observation import (
    SCHEMA_VERSION,
    CentralObservation,
    CentralObservationStore,
    node_observation_from_fusion,
    observation_truth_violations,
    publish_node_observation,
)
from resource_management.scheduling import (
    SchedulerPolicy,
    SchedulingConfig,
    build_scheduler,
)
from resource_management.tasks import (
    DuplicateTaskError,
    QueueTaskKind,
    TaskQueue,
    TaskStatus,
    UnknownObjectError,
)
from resource_management.model import ExecutionPlan, ExecutionResult, Outcome
from resource_management.units import BUDGET_UNITS, ResourceUnit, TaskKind

BASE_CONFIG = ec.CONFIG_PATH

RUNTIME_MODE_LEGACY = "legacy_observation_first"
RUNTIME_MODE_FEEDBACK = "plan_controlled_feedback"
RUNTIME_MODES = (RUNTIME_MODE_LEGACY, RUNTIME_MODE_FEEDBACK)

#: 任务派生门控方式。
#:
#: `loop_gate`（默认）：沿用闭环的**手工门控**——每 tick 只派生
#:   一种可执行的任务类型（有数据可处理就派 process，否则有数据可发就派
#:   share，否则派 sample）。旧路径与规则/优化参考都用它，**逐位不变**。
#:
#: `expose_all`：把 sample / process / share 三种候选**同时**暴露出来，
#:   让**策略**去选提交哪一种。学习型调度器必须用这个模式——
#:   否则"提交什么任务"已经被门控替策略决定了，动作空间里没有可学的东西。
#:   ⚠️ 它改变的是**候选任务的构成**，不改传感器/通信/融合语义；
#:   代价是候选更多、过期更多，因此**两种模式的完成率不可直接比较**。
TASK_GATING_LOOP = "loop_gate"
TASK_GATING_EXPOSE_ALL = "expose_all"
TASK_GATING_MODES = (TASK_GATING_LOOP, TASK_GATING_EXPOSE_ALL)

#: 各任务类型的**基线**截止余量（秒）。这是闭环的权威定义，
#: `rl_resource.scenarios` 从**这里**导入后按 `load_multiplier` 缩放，
#: 避免两处各写一份（两处各写一份 = 迟早对不上）。
BASELINE_DEADLINE_OFFSETS: Dict[QueueTaskKind, float] = {
    QueueTaskKind.SHARE: 2.0,
    QueueTaskKind.ESTIMATE_UPDATE: 3.0,
    QueueTaskKind.PROCESS: 3.0,
    QueueTaskKind.PREDEFINED_SAMPLE: 6.0,
}

#: 默认机制配置（全部显式，便于复现与对照）
DEFAULT_MECHANISMS: Dict[str, Any] = {
    #: 节点不可用窗口 `{node_id: [(start_s, end_s), ...]}`
    "unavailable_windows": {},
    #: 通信策略（`ideal_share` / `constrained_share`）
    "share_policy": SHARE_IDEAL,
    #: 受限共享的链路参数
    "comm": {"base_delay_s": 1.2, "jitter_s": 0.4, "loss_prob": 0.0,
             "expiry_s": 10.0},
    #: 观测偏差注入 `{node_id: {...SensorConfig 偏差字段...}}`
    "bias": {},
}


def _radar(base: Dict[str, Any], radar_id: str, x: float, y: float,
           heading_deg: float) -> Dict[str, Any]:
    cfg = dict(base)
    cfg.update({
        "radar_id": radar_id, "x": x, "y": y, "z": 0.0,
        "heading_deg": heading_deg, "pitch_deg": 0.0, "roll_deg": 0.0,
        "velocity_x": 0.0, "velocity_y": 0.0, "velocity_z": 0.0,
        "timestamp_s": 0.0, "platform_id": radar_id, "is_active": True,
    })
    return cfg


def _target(base: Dict[str, Any], target_id: str, x: float, y: float,
            vx: float, vy: float) -> Dict[str, Any]:
    cfg = dict(base)
    cfg.update({
        "target_id": target_id, "x": x, "y": y, "z": 0.0,
        "vx": vx, "vy": vy, "vz": 0.0,
        "heading_deg": math.degrees(math.atan2(vx, vy)) % 360.0,
        "pitch_deg": 0.0, "roll_deg": 0.0, "timestamp_s": 0.0,
        "platform_id": target_id, "is_active": True,
    })
    return cfg


def _sensor(sensor_id: str, mounting_id: str, max_range_m: float,
            az_fov_deg: float = 60.0, **overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "sensor_id": sensor_id, "mounting_id": mounting_id,
        "sensor_kind": "radar", "max_range_m": max_range_m,
        "min_range_m": 0.0, "az_fov_deg": az_fov_deg, "el_fov_deg": 30.0,
        "update_period_s": 1.0, "range_sigma_rel": 0.01,
        "range_sigma_abs_m": 5.0, "az_sigma_deg": 0.5, "el_sigma_deg": 0.5,
        "range_rate_sigma_mps": 1.0, "snr50_db": 6.0, "pd_slope_db": 2.0,
        "false_alarm_rate": 0.0, "tx_power_w": 18.0, "peak_gain_db": 30.0,
        "wavelength_m": 0.1, "bandwidth_hz": 1.0e6, "noise_figure_db": 3.0,
        "system_loss_db": 3.0, "temperature_k": 290.0,
        "observes_kind": "target", "provides_range": True,
        # 闭环验收要的是"分工与记账"，不是检测概率统计：
        # 确定性检测让"看不到"只由几何（距离/视场）决定，便于归因。
        "force_detection": True, "seed": 42,
    }
    cfg.update(overrides)
    return cfg


#: 两节点几何 + 传感器包线（**故意只部分重叠**，从而产生覆盖交接）
NODE_LAYOUT: Dict[str, Dict[str, Any]] = {
    "NODE_A": {"x": 0.0, "y": -6000.0, "heading_deg": 0.0,
               "max_range_m": 11000.0},
    "NODE_B": {"x": 0.0, "y": 6000.0, "heading_deg": 180.0,
               "max_range_m": 11000.0},
}

#: 默认节点资源预算（教学成本模型，非真实装备参数）。
#:
#: 定标方法是**需求侧定标**，不是为了让某个策略好看：
#: 1. 任务由可见航迹逐 tick 派生（每目标每 tick 采样 + 更新 + 共享），
#:    24 tick 两节点合计约 200+ 条，而执行器每节点每 tick 只能落 1 条，
#:    所以**服务上限**（= 节点数 × tick 数）才是主要约束；
#: 2. 因此预算取"略高于服务上限所需"的量级：每个节点 24 tick 最多消耗
#:    24 采样槽 / 24 处理操作 / 3 KB 通信，取 18~30 一档可让预算在
#:    后半程**真正绑定**（出现 `insufficient_resource` 拒绝），
#:    否则账本永远单调、资源管理就只是个摆设；
#: 3. **三个基线用同一组预算**，差异只来自排序规则。
DEFAULT_NODE_BUDGETS: Dict[str, Dict[ResourceUnit, float]] = {
    "NODE_A": {ResourceUnit.SAMPLE_SLOT: 30.0,
               ResourceUnit.PROCESSING_OP: 40.0,
               ResourceUnit.COMM_BYTE: 12288.0},
    "NODE_B": {ResourceUnit.SAMPLE_SLOT: 18.0,
               ResourceUnit.PROCESSING_OP: 26.0,
               ResourceUnit.COMM_BYTE: 8192.0},
}


@dataclass
class LoopResult:
    """一次闭环运行的全部记录（**结论必须能从这里逐条追溯**）。"""

    policy: str
    seed: int
    steps: int
    config: Dict[str, Any] = field(default_factory=dict)
    mechanisms: Dict[str, Any] = field(default_factory=dict)
    #: 逐 tick 的决策记录（含"为什么这么分工"）
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    #: 逐 tick 的计划状态
    plan_log: List[Dict[str, Any]] = field(default_factory=list)
    #: 每个节点的任务时间线（任务 → 时刻 → 状态 → 原因）
    timelines: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    #: 逐节点观测年龄（用于看通信延迟的影响）
    node_ages: List[Dict[str, Any]] = field(default_factory=list)
    #: 离线评估专用估计误差；调度器永远看不到这份真值对齐表
    estimate_error_samples: List[Dict[str, Any]] = field(default_factory=list)
    #: 新真闭环的逐任务运行时副作；旧路径为空。
    runtime_log: List[Dict[str, Any]] = field(default_factory=list)
    #: 被判"长期未获服务"的任务 ID（与"过期"互斥）
    starved_task_ids: List[str] = field(default_factory=list)
    #: 规划耗时（秒）——**计算耗时**维度；不含执行与仿真推进
    planning_time_s: float = 0.0
    #: 优化参考最近一次规划的完整交代（规则基线为 None）
    optimizer_outcome: Optional[Dict[str, Any]] = None
    #: 优化参考逐 tick 规划交代的汇总（含最优性声明的自审）
    optimizer_summary: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    queue_summary: Dict[str, Any] = field(default_factory=dict)
    conservation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "steps": self.steps,
            "config": self.config,
            "mechanisms": {key: str(value) for key, value in
                           self.mechanisms.items()},
            "metrics": self.metrics,
            "queue_summary": self.queue_summary,
            "conservation": self.conservation,
            "plan_log": self.plan_log,
            "timelines": self.timelines,
            "node_ages": self.node_ages,
            "estimate_error_samples": self.estimate_error_samples,
            "runtime_log": self.runtime_log,
            "starved_task_ids": self.starved_task_ids,
            "planning_time_s": round(self.planning_time_s, 9),
            "optimizer_outcome": self.optimizer_outcome,
            "optimizer_summary": self.optimizer_summary,
            "decisions": self.decisions,
        }


class RuntimeExecutionError(RuntimeError):
    """运行时执行层拒绝不一致计划或重复副作。"""


class _SharedMeasurement:
    """从 CommBus 白名单载荷还原的融合输入（永远不含真值）。"""

    def __init__(self, payload: Dict[str, Any], msg_id: str,
                 platform_id: str) -> None:
        for key, value in payload.items():
            setattr(self, key, value)
        self.msg_id = str(msg_id)
        self.platform_id = str(platform_id)
        self.truth_id = None
        self.is_false_alarm = False
        covariance = [payload.get("cov_xx"), payload.get("cov_yy"),
                      payload.get("cov_zz")]
        self.covariance = (
            [[float(covariance[0]), 0.0, 0.0],
             [0.0, float(covariance[1]), 0.0],
             [0.0, 0.0, float(covariance[2])]]
            if all(isinstance(value, (int, float)) for value in covariance)
            else None
        )


class RuntimeExecutor:
    """`ExecutionPlan` 的唯一运行时副作入口。

    `UnifiedExecutor` 先完成校验、去重和资源记账；本类随后仅对
    `APPLIED` 任务执行一次传感器、通信和融合副作，不再扣费。
    """

    def __init__(
        self,
        accounting: UnifiedExecutor,
        scene: Any,
        suite: Any,
        centers: Dict[str, Any],
        bus: CommBus,
        sensor_positions: Dict[str, Vec3],
        own_sensor: Dict[str, str],
    ) -> None:
        self.accounting = accounting
        self.scene = scene
        self.suite = suite
        self.centers = centers
        self.bus = bus
        self.sensor_positions = sensor_positions
        self.own_sensor = own_sensor
        self.local_pending: Dict[str, List[Any]] = {
            node_id: [] for node_id in centers
        }
        self.share_outbox: Dict[str, List[Any]] = {
            node_id: [] for node_id in centers
        }
        self.event_log: List[Dict[str, Any]] = []
        self._submitted_plan_ids: set = set()
        self._runtime_task_keys: set = set()
        self._received_total: Dict[str, int] = {
            node_id: 0 for node_id in centers
        }

    def begin_tick(self, now_s: float) -> None:
        """无条件执行的只有航迹预测；不采样、不发送、不融合新测量。"""
        for node_id, center in self.centers.items():
            before = [max(track.sigma_position.x, track.sigma_position.y,
                          track.sigma_position.z)
                      for track in center.tracks]
            center.predict_to(now_s)
            after = [max(track.sigma_position.x, track.sigma_position.y,
                         track.sigma_position.z)
                     for track in center.tracks]
            self.event_log.append({
                "time_s": round(now_s, 6), "node_id": node_id,
                "event": "predict_only", "task_id": "",
                "n_tracks": len(center.tracks),
                "sigma_before_max_m": max(before) if before else None,
                "sigma_after_max_m": max(after) if after else None,
            })

    def has_shareable(self, node_id: str) -> bool:
        return bool(self.share_outbox.get(node_id))

    def sensor_scan_counts(self) -> Dict[str, int]:
        """逐节点传感器**真实扫描次数**（来自传感器自身的统计计数器）。

        这是"计划真的控制了感知链"的硬证据：未拿到 sample 任务的节点，
        计数必须保持不变。
        """
        counts: Dict[str, int] = {}
        for node_id in sorted(self.centers):
            sensor = self.suite.by_id(self.own_sensor[node_id])
            counts[node_id] = int(getattr(sensor, "stats", {}).get("scans", 0))
        return counts

    def unique_task_keys(self) -> int:
        """已登记的唯一去重键数（与运行时任务数相等 ⟺ 无重复执行）。"""
        return len(self._runtime_task_keys)

    def has_processable(self, node_id: str, now_s: float) -> bool:
        return bool(self.local_pending.get(node_id)) or bool(
            self.bus.deliverable_count(node_id, now_s)
        )

    @staticmethod
    def _observable_quality_proxy(center: Any, now_s: float) -> Dict[str, Any]:
        """仅由融合航迹计算的质量代理；不读取真值或离线误差。

        它用于 v2 的延迟通信回报审计：年龄越小、位置协方差越小越好。
        0 航迹记为 0，使“远端测量首次建立可见航迹”也可被观察到。
        """
        tracks = list(center.tracks)
        if not tracks:
            return {"quality_proxy": 0.0, "mean_information_age_s": None,
                    "mean_sigma_max_m": None, "n_tracks": 0}
        ages = [max(0.0, float(now_s) - float(track.last_measurement_time))
                for track in tracks]
        sigmas = [max(track.sigma_position.x, track.sigma_position.y,
                      track.sigma_position.z) for track in tracks]
        age_quality = sum(1.0 / (1.0 + value) for value in ages) / len(ages)
        # 200 m 是 v2 协议冻结的、仅用于可观测代理的尺度；绝不由 test 拟合。
        sigma_quality = sum(1.0 / (1.0 + value / 200.0)
                            for value in sigmas) / len(sigmas)
        return {"quality_proxy": 0.5 * (age_quality + sigma_quality),
                "mean_information_age_s": sum(ages) / len(ages),
                "mean_sigma_max_m": sum(sigmas) / len(sigmas),
                "n_tracks": len(tracks)}

    def submit(self, plan: ExecutionPlan) -> ExecutionResult:
        if plan.plan_id in self._submitted_plan_ids:
            raise RuntimeExecutionError(
                f"计划 {plan.plan_id!r} 已进入过 RuntimeExecutor，禁止重复副作"
            )
        self._submitted_plan_ids.add(plan.plan_id)
        result = self.accounting.submit(plan)
        if result.plan_id != plan.plan_id:
            raise RuntimeExecutionError("记账结果与运行时计划 ID 不一致")
        by_task = {task.task_id: task for task in plan.tasks}
        for outcome in result.outcomes:
            if outcome.outcome is not Outcome.APPLIED:
                continue
            task = by_task.get(outcome.task_id)
            if task is None:
                raise RuntimeExecutionError(
                    f"执行结果引用了计划外任务 {outcome.task_id!r}"
                )
            if task.dedup_key in self._runtime_task_keys:
                raise RuntimeExecutionError(
                    f"运行时去重键 {task.dedup_key!r} 已执行"
                )
            self._runtime_task_keys.add(task.dedup_key)
            self._apply(task, plan.plan_id)
        return result

    def _apply(self, task: Any, plan_id: str) -> None:
        now = self.accounting.clock.now_s
        if task.kind is TaskKind.SAMPLE:
            self._sample(task, plan_id, now)
        elif task.kind is TaskKind.PROCESS:
            self._process(task, plan_id, now)
        elif task.kind is TaskKind.SHARE:
            self._share(task, plan_id, now)
        else:
            self.event_log.append({
                "time_s": round(now, 6), "node_id": task.node_id,
                "event": "idle", "task_id": task.task_id,
                "plan_id": plan_id,
            })

    def _sample(self, task: Any, plan_id: str, now: float) -> None:
        sensor_id = self.own_sensor[task.node_id]
        report = self.suite.by_id(sensor_id).observe(
            self.scene, now, self.suite.occlusion, {}
        )
        fresh = list(report.detections) + list(report.false_alarms)
        self.local_pending[task.node_id].extend(fresh)
        self.share_outbox[task.node_id].extend(fresh)
        self.event_log.append({
            "time_s": round(now, 6), "node_id": task.node_id,
            "event": "sample", "task_id": task.task_id,
            "plan_id": plan_id, "sensor_id": sensor_id,
            "sensor_updated": bool(report.updated),
            "n_measurements": len(fresh),
        })

    def _process(self, task: Any, plan_id: str, now: float) -> None:
        before_quality = self._observable_quality_proxy(
            self.centers[task.node_id], now)
        messages = self.bus.consume(task.node_id, now)
        self.bus.assert_only_arrived(messages, now)
        remote = [
            _SharedMeasurement(message.payload, message.msg_id,
                               message.src_platform_id)
            for message in messages
        ]
        local = list(self.local_pending[task.node_id])
        measurements = local + remote
        if measurements:
            self.centers[task.node_id].update(
                measurements,
                now,
                self.sensor_positions,
                remote_measurement_flags=(
                    [False] * len(local) + [True] * len(remote)
                ),
            )
            self.local_pending[task.node_id].clear()
        self._received_total[task.node_id] += len(remote)
        after_quality = self._observable_quality_proxy(
            self.centers[task.node_id], now)
        self.event_log.append({
            "time_s": round(now, 6), "node_id": task.node_id,
            "event": "process", "task_id": task.task_id,
            "plan_id": plan_id, "n_local_measurements": len(local),
            "n_remote_measurements": len(remote),
            "n_measurements": len(measurements),
            "fusion_updated": bool(measurements),
            "quality_proxy_before": before_quality["quality_proxy"],
            "quality_proxy_after": after_quality["quality_proxy"],
            "mean_information_age_before_s": before_quality["mean_information_age_s"],
            "mean_information_age_after_s": after_quality["mean_information_age_s"],
            "mean_sigma_before_m": before_quality["mean_sigma_max_m"],
            "mean_sigma_after_m": after_quality["mean_sigma_max_m"],
        })

    def _share(self, task: Any, plan_id: str, now: float) -> None:
        outbox = self.share_outbox[task.node_id]
        measurement = outbox[0] if outbox else None
        peers = sorted(node_id for node_id in self.centers
                       if node_id != task.node_id)
        sent = []
        if measurement is not None and peers:
            sent = self.bus.publish(
                task.node_id,
                self.own_sensor[task.node_id],
                [measurement],
                now=now,
                dst_platform_ids=[peers[0]],
            )
            if sent:
                outbox.pop(0)
        accounted = float(task.effective_cost().get(
            ResourceUnit.COMM_BYTE, 0.0
        ))
        sent_bytes = sum(float(message.size_bytes) for message in sent)
        self.event_log.append({
            "time_s": round(now, 6), "node_id": task.node_id,
            "event": "share", "task_id": task.task_id,
            "plan_id": plan_id, "n_messages": len(sent),
            "accounted_comm_bytes": accounted,
            "sent_comm_bytes": sent_bytes,
            "payload_truth_fields": sorted(
                key for message in sent for key in message.payload
                if str(key).startswith(("truth", "err_", "is_false_alarm"))
            ),
        })

    def central_observation(self, now_s: float) -> CentralObservation:
        """集中调度控制面：只读节点资源与本地融合输出，不读真值。"""
        nodes = []
        for node_id in sorted(self.centers):
            nodes.append(node_observation_from_fusion(
                self.centers[node_id], self.accounting.node(node_id), now_s,
                n_messages_arrived=self._received_total[node_id],
                n_messages_inflight=sum(
                    1 for message in self.bus.in_flight(now_s)
                    if message.dst_platform_id == node_id
                ),
            ))
        return CentralObservation(
            schema_version=SCHEMA_VERSION,
            observed_at_s=float(now_s),
            nodes=nodes,
            node_valid_mask=[True] * len(nodes),
            node_information_age_s=[0.0] * len(nodes),
            node_content_age_s=[0.0] * len(nodes),
            missing=[],
            n_messages_ingested=sum(self._received_total.values()),
        )


#: 目标几何库（**只改场景，不改物理**）。
#:
#: 第 1、2 条就是闭环一直使用的基线几何，顺序与坐标**不得改动**
#: （改了就破坏旧路径的逐位复现）。第 3 条供"高负载"场景使用：
#: 它在 t=0 同时对两个节点可见（A：方位偏 16.7°、距离 5.2 km；
#: B：偏 12.1°、距离 7.2 km），因此真的会提高每 tick 的派生任务数。
TARGET_LIBRARY: Tuple[Dict[str, Any], ...] = (
    # 沿 x=1500 从 y=-5000 向 +y 运动 → 依次进入 A、重叠区、B
    {"target_id": "TGT_1", "x": 1500.0, "y": -5000.0, "vx": 0.0, "vy": 400.0},
    # 第二个目标全程在 A 侧，用于产生并发任务
    {"target_id": "TGT_2", "x": -1200.0, "y": -3000.0, "vx": 0.0, "vy": 250.0},
    # 高负载场景追加：t=0 即对两节点同时可见
    {"target_id": "TGT_3", "x": -1500.0, "y": -1000.0, "vx": 0.0, "vy": 300.0},
)


def _build_world(
    policy: SchedulerPolicy,
    seed: int,
    steps: int,
    scheduling_config: Optional[SchedulingConfig] = None,
    mechanisms: Optional[Dict[str, Any]] = None,
    node_budgets: Optional[Dict[str, Dict[ResourceUnit, float]]] = None,
    optimizer_config: Any = None,
    runtime_mode: str = RUNTIME_MODE_LEGACY,
    target_count: int = 2,
) -> Dict[str, Any]:
    """构造闭环世界（**不含主循环**）。真值只在这里。

    抽出来的理由：学习型调度器需要"自己一步一步驱动"同一个世界，
    如果它另写一份构造代码，学到的策略与规则基线比的就不是同一件事了。
    两条路径共用本函数，因此世界、传感器、通信、融合、执行器**逐位相同**。
    """
    mech = dict(DEFAULT_MECHANISMS)
    mech.update(mechanisms or {})
    unavailable = {node: list(windows) for node, windows
                   in (mech.get("unavailable_windows") or {}).items()}
    bias = dict(mech.get("bias") or {})

    import json as _json_base

    with open(BASE_CONFIG, "r", encoding="utf-8") as handle:
        base = _json_base.load(handle)
    radar_base, target_base = base["radar"], base["targets"][0]

    overrides = {
        "radars": [_radar(radar_base, node_id, layout["x"], layout["y"],
                          layout["heading_deg"])
                   for node_id, layout in NODE_LAYOUT.items()],
        "targets": [
            _target(target_base, spec["target_id"], spec["x"], spec["y"],
                    spec["vx"], spec["vy"])
            for spec in TARGET_LIBRARY[:max(1, min(len(TARGET_LIBRARY),
                                                   int(target_count)))]
        ],
        "sensors": [
            _sensor(f"SENSOR_{node_id}", node_id, layout["max_range_m"],
                    az_fov_deg=45.0, **bias.get(node_id, {}))
            for node_id, layout in NODE_LAYOUT.items()
        ],
        "occluders": [],
    }

    sim = Simulator(BASE_CONFIG)
    sim.load_config()
    sim.apply_overrides(extra=overrides)
    sim.reset(seed=seed)
    raw_config = dict(getattr(sim, "_raw_config", {}) or {})
    suite = build_suite_from_config(sim.scene, raw_config, seed=seed,
                                    noise_scale=1.0)
    sensor_positions = {
        sensor.sensor_id: sim.scene.by_id(sensor.config.mounting_id).position
        for sensor in suite.sensors
    }

    # ---------- 时钟 / 执行器 / 队列（三者对所有策略完全相同） ----------
    clock = GlobalClock(now_s=0.0)
    ledger = ResourceLedger()
    executor = UnifiedExecutor(clock, ledger)
    queue = TaskQueue()
    scheduler = build_scheduler(policy, scheduling_config, optimizer_config)

    budgets = dict(DEFAULT_NODE_BUDGETS)
    budgets.update(node_budgets or {})
    for node_id in NODE_LAYOUT:
        executor.register_node(NodeState(
            node_id=node_id, update_period_s=1.0,
            budget=ResourceBudget(capacity=dict(budgets[node_id]))))

    # ---------- 每节点自己的融合中心（只吃本节点传感器的测量） ----------
    own_sensor = {node_id: f"SENSOR_{node_id}" for node_id in NODE_LAYOUT}
    centers = {
        node_id: FusionCenter(node_id, FusionConfig(),
                              own_sensor_ids={own_sensor[node_id]},
                              lifecycle=None)
        for node_id in NODE_LAYOUT
    }

    share_policy = mech.get("share_policy", SHARE_IDEAL)
    comm_kwargs = dict(mech.get("comm") or {}) \
        if share_policy == SHARE_CONSTRAINED else {}
    bus = CommBus(list(NODE_LAYOUT), CommConfig(policy=share_policy, seed=seed,
                                                **comm_kwargs))
    store = CentralObservationStore(list(NODE_LAYOUT))

    result = LoopResult(policy=policy.value, seed=seed, steps=steps,
                        config=_config_dict(scheduler.config),
                        mechanisms=mech)
    result.config["runtime_mode"] = runtime_mode
    # 逐节点三类单位的**容量**（实测资源占用率的分母；只读，不参与调度）
    result.config["node_budgets"] = {
        node_id: {unit.value: float(node.budget.capacity.get(unit, 0.0))
                  for unit in BUDGET_UNITS}
        for node_id, node in sorted(executor.nodes.items())}
    resolved_optimizer = getattr(scheduler, "optimizer_config", None)
    if resolved_optimizer is not None:
        result.config["optimizer"] = {
            "kind": resolved_optimizer.kind.value,
            "horizon_ticks": int(getattr(scheduler, "_horizon", 1)),
            "budget": resolved_optimizer.budget.to_dict(),
            "objective": resolved_optimizer.objective.describe(),
            "prediction_assumptions":
                getattr(scheduler, "prediction", None).assumptions()
                if getattr(scheduler, "prediction", None) is not None else [],
        }
    result.timelines = {node_id: [] for node_id in NODE_LAYOUT}
    #: 被判"长期未获服务"的 task_id（与"过期"互斥，见主循环第 ⑥ 步）
    starved_ids: set = set()
    return {
        "sim": sim, "suite": suite, "sensor_positions": sensor_positions,
        "clock": clock, "ledger": ledger, "executor": executor, "queue": queue,
        "scheduler": scheduler, "centers": centers, "bus": bus, "store": store,
        "result": result, "starved_ids": starved_ids,
        "unavailable": unavailable, "mech": mech, "own_sensor": own_sensor,
        "node_ids": tuple(sorted(centers)),
    }


class _ConfigOnlyPlanner:
    """只提供 `config` 的占位调度器（学习型路径自己产出计划）。

    存在的理由：`FeedbackLoopDriver.commit()` 收尾时需要
    `scheduler.config.starvation_threshold_s`。学习型路径不调用 `plan()`，
    因此这里显式让它在被误调用时**报错**，而不是悄悄退回某个规则基线。
    """

    def __init__(self, config: Optional[SchedulingConfig] = None) -> None:
        self.config = config or SchedulingConfig()
        self.planning_time_s = 0.0
        self.last_outcome = None

    def plan(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "学习型路径不应调用占位调度器：计划由策略产出，"
            "再交给 UnifiedExecutor + RuntimeExecutor 校验与执行")


def _build_feedback_world(
    *,
    seed: int,
    steps: int,
    mechanisms: Optional[Dict[str, Any]] = None,
    node_budgets: Optional[Dict[str, Dict[ResourceUnit, float]]] = None,
    deadline_offsets: Optional[Dict[str, float]] = None,
    scheduler_config: Optional[SchedulingConfig] = None,
    policy: SchedulerPolicy = SchedulerPolicy.RULE,
    task_gating: str = TASK_GATING_EXPOSE_ALL,
    share_requires_visible_track: bool = True,
    target_count: int = 2,
    keep_decisions: bool = True,
) -> Dict[str, Any]:
    """构造**真闭环**世界 + `RuntimeExecutor` + `FeedbackLoopDriver`。

    供学习环境使用：环境自己按 tick 调 `driver.begin()` / `driver.commit()`，
    中间插入策略决策。世界构造与 `run_closed_loop` 共用 `_build_world`，
    因此两条路径的传感器/通信/融合/执行器逐位相同。
    """
    world = _build_world(
        policy=policy, seed=seed, steps=steps,
        scheduling_config=scheduler_config, mechanisms=mechanisms,
        node_budgets=node_budgets, optimizer_config=None,
        runtime_mode=RUNTIME_MODE_FEEDBACK, target_count=target_count)
    runtime = RuntimeExecutor(
        world["executor"], world["sim"].scene, world["suite"], world["centers"],
        world["bus"], world["sensor_positions"], world["own_sensor"])
    world["runtime"] = runtime
    world["planner"] = _ConfigOnlyPlanner(scheduler_config)
    world["driver"] = FeedbackLoopDriver(
        policy=policy, steps=steps, unavailable=world["unavailable"],
        sim=world["sim"], clock=world["clock"], ledger=world["ledger"],
        executor=world["executor"], queue=world["queue"],
        scheduler=world["planner"], runtime=runtime, result=world["result"],
        keep_decisions=keep_decisions, starved_ids=world["starved_ids"],
        task_gating=task_gating, deadline_offsets=deadline_offsets,
        share_requires_visible_track=share_requires_visible_track)
    return world


def run_closed_loop(
    policy: SchedulerPolicy = SchedulerPolicy.RULE,
    seed: int = 42,
    steps: int = 24,
    scheduling_config: Optional[SchedulingConfig] = None,
    mechanisms: Optional[Dict[str, Any]] = None,
    node_budgets: Optional[Dict[str, Dict[ResourceUnit, float]]] = None,
    optimizer_config: Any = None,
    keep_decisions: bool = True,
    runtime_mode: str = RUNTIME_MODE_LEGACY,
    target_count: int = 2,
    task_gating: str = TASK_GATING_LOOP,
    deadline_offsets: Optional[Dict[str, float]] = None,
) -> LoopResult:
    """跑一次闭环：世界 → 融合 → 通信 → 调度 → 执行 → 记账。

    **同一份可见观测、同一个任务队列、同一个执行器**：规则基线与优化参考
    在这里只有 `build_scheduler(policy)` 不同，其余代码路径完全一致。

    `optimizer_config`（`optimization.OptimizerConfig`）只在优化参考时生效，
    它带来计算预算与预测假设；两条路径的任务定义、资源预算、信息权限、
    执行器**完全相同**。
    """
    if runtime_mode not in RUNTIME_MODES:
        raise ValueError(f"runtime_mode 必须是 {list(RUNTIME_MODES)}")
    world = _build_world(
        policy=policy, seed=seed, steps=steps,
        scheduling_config=scheduling_config, mechanisms=mechanisms,
        node_budgets=node_budgets, optimizer_config=optimizer_config,
        runtime_mode=runtime_mode, target_count=target_count)
    sim = world["sim"]
    suite = world["suite"]
    sensor_positions = world["sensor_positions"]
    clock = world["clock"]
    ledger = world["ledger"]
    executor = world["executor"]
    queue = world["queue"]
    scheduler = world["scheduler"]
    centers = world["centers"]
    bus = world["bus"]
    store = world["store"]
    result = world["result"]
    starved_ids = world["starved_ids"]
    unavailable = world["unavailable"]
    own_sensor = world["own_sensor"]

    if runtime_mode == RUNTIME_MODE_FEEDBACK:
        runtime = RuntimeExecutor(
            executor, sim.scene, suite, centers, bus,
            sensor_positions, world["own_sensor"],
        )
        return _run_feedback_loop(
            policy=policy,
            steps=steps,
            unavailable=unavailable,
            sim=sim,
            clock=clock,
            ledger=ledger,
            executor=executor,
            queue=queue,
            scheduler=scheduler,
            runtime=runtime,
            result=result,
            keep_decisions=keep_decisions,
            starved_ids=starved_ids,
            task_gating=task_gating,
            deadline_offsets=deadline_offsets,
        )

    # ---------- 主循环 ----------
    for step in range(steps):
        clock.advance(1.0, f"tick {step + 1}")
        now = clock.now_s
        # **先把场景推进一个 tick**，再观测。
        #
        # 第一版漏了这一行：目标全程静止，于是"沿 x=1500 从 y=-5000 向 +y 运动
        # → 依次进入 A、重叠区、B"的注释是假的，覆盖交接在结构上不可能发生
        # （两个节点每 tick 都看到同一个目标，可见目标数恒为 1）。
        # `advance_all` 同时推进位置与时间戳，保持 Scene 的时间同步约束。
        sim.scene.advance_all(1.0)
        report = suite.observe(sim.scene, now, {})

        # ① 各节点用自己的传感器测量更新自己的融合中心
        for node_id in NODE_LAYOUT:
            window = _in_window(unavailable.get(node_id, []), now)
            if window:
                # 节点不可用：本 tick 不产出观测（**不影响其他节点**）
                continue
            measurements = [m for sensor_report in report.reports
                            if sensor_report.sensor_id == own_sensor[node_id]
                            for m in (sensor_report.detections
                                      + sensor_report.held)]
            centers[node_id].predict_to(now)
            centers[node_id].update(measurements, now, sensor_positions)

        # 离线评估：用最近航迹计算位置误差。这一段位于唯一允许读
        # 真值的 closed_loop 编排层，只写入最终指标，不进入 CentralObservation。
        for node_id, center in centers.items():
            estimated = [track.position for track in center.tracks]
            for target in sim.scene.targets:
                if not estimated:
                    continue
                error = min((position - target.position).norm()
                            for position in estimated)
                result.estimate_error_samples.append({
                    "time_s": round(now, 6),
                    "node_id": node_id,
                    "target_id": target.target_id,
                    "position_error_m": round(error, 6),
                    "provenance": "offline_truth_nearest_track",
                })

        # ② 各节点把**融合观测摘要**发到总线
        node_states = {node_id: executor.node(node_id)
                       for node_id in NODE_LAYOUT}
        for node_id in NODE_LAYOUT:
            window = _in_window(unavailable.get(node_id, []), now)
            if window:
                continue
            observation = node_observation_from_fusion(
                centers[node_id], node_states[node_id], now_s=now,
                n_messages_arrived=0, n_messages_inflight=0)
            publish_node_observation(
                observation,
                bus,
                node_id,
                now_s=now,
                include_research_extension=bool(
                    world["mech"].get("information_research_extension", False)
                ),
            )

        # ③ 中央只读**已到达**的摘要
        for node_id in NODE_LAYOUT:
            store.ingest_arrived(bus, node_id, now)
        central = store.observe(now)
        result.node_ages.append({
            "time_s": round(now, 6),
            "valid": list(central.node_valid_mask),
            "ages": [None if age is None else round(age, 6)
                     for age in central.node_information_age_s],
            # 内容年龄（= now − 摘要生成时刻）与到达年龄分开记：
            # 通信延迟只体现在这一项上。
            "content_ages": [None if age is None else round(age, 6)
                             for age in central.node_content_age_s],
            # 逐节点**可见航迹条数**：交接时目标从 A 的视野进入 B 的视野，
            # 这一列会随时间变化，是"分工随时间变化"的直接证据。
            "n_tracks": [len(node.track_ids()) if valid else None
                         for node, valid
                         in zip(central.nodes, central.node_valid_mask)],
            # 逐节点**估计质量**支撑量：可见航迹的平均信息年龄与平均位置σ。
            # 只从**已到达的**摘要统计（中央看得见什么就统计什么）。
            "mean_track_age_s": [
                _mean_or_none([track.information_age_s
                               for track, valid in zip(node.tracks,
                                                       node.track_valid_mask)
                               if valid]) if valid else None
                for node, valid in zip(central.nodes, central.node_valid_mask)],
            "mean_sigma_m": [
                _mean_or_none([max(track.sigma_position)
                               for track, valid in zip(node.tracks,
                                                       node.track_valid_mask)
                               if valid]) if valid else None
                for node, valid in zip(central.nodes, central.node_valid_mask)],
            "node_ids": [node.node_id for node in central.nodes],
        })

        # ④ 从**每个节点的可见观测**派生任务（未知对象不建任务）
        #
        # 截止时间按任务类型给**不同余量**，让截止时间真正表达服务需求：
        # 共享的数据最易过期（2s）、更新次之（3s）、采样最松（6s）。
        # 若三类都用同一余量，EDF 会退化成先到先服务，与轮询无从区分——
        # 第一版就是这样，三个基线里有两个完全同值。
        for node, valid in zip(central.nodes, central.node_valid_mask):
            if not valid:
                continue
            try:
                queue.create_from_observation(
                    node, now, deadline_offsets={
                        QueueTaskKind.SHARE: 2.0,
                        QueueTaskKind.ESTIMATE_UPDATE: 3.0,
                        QueueTaskKind.PREDEFINED_SAMPLE: 6.0,
                    })
            except (DuplicateTaskError, UnknownObjectError) as exc:
                # 只有**领域内**的失败才允许被记成"任务创建失败"。
                # 传统路径不做 sample/process/share 门控（那是反馈模式的事），
                # 因此这里既不需要 runtime 上下文，也不该吞掉编程错误——
                # 见下方 except 的说明。
                result.decisions.append({
                    "time_s": now, "node_id": node.node_id,
                    "decision": "task_creation_failed", "reason": str(exc)})

        # ⑤ 调度 → 标准 ExecutionPlan → 执行器校验与记账
        planning = scheduler.plan(central, queue, now,
                                  plan_id=f"{policy.value}-{step + 1:04d}")
        if keep_decisions:
            for decision in planning.decisions:
                payload = decision.to_dict()
                payload["time_s"] = round(now, 6)
                result.decisions.append(payload)
        for decision in planning.decisions:
            if decision.decision in ("planned", "deferred", "abandoned",
                                     "suppressed_duplicate_node",
                                     "not_eligible"):
                result.timelines.setdefault(decision.node_id, []).append({
                    "time_s": round(now, 6),
                    "task_id": decision.task_id,
                    "kind": decision.kind,
                    "decision": decision.decision,
                    "reason": decision.reasons[0] if decision.reasons else "",
                    "priority": round(decision.priority, 6),
                })

        if planning.plan is not None:
            execution = executor.submit(planning.plan)
            result.plan_log.append({
                "time_s": round(now, 6),
                "plan_id": planning.plan.plan_id,
                "status": execution.status.value,
                "n_applied": execution.n_applied,
                "n_rejected": execution.n_rejected,
                "nodes": planning.plan.node_ids(),
                # 逐条被拒任务的**原因**与逐条校验问题（**约束违反**的证据）
                "rejected": [item.to_dict() for item in execution.outcomes
                             if item.outcome.value != "applied"],
                "issues": [issue.to_dict() for issue in execution.issues],
            })
            for outcome in execution.outcomes:
                queue.mark(
                    outcome.task_id,
                    TaskStatus.COMPLETED if outcome.outcome.value == "applied"
                    else TaskStatus.REJECTED,
                    reason=(outcome.reason if outcome.outcome.value == "applied"
                            else f"执行器拒绝：{outcome.reason}"),
                    plan_id=planning.plan.plan_id)

        # ⑥ 超期处理（**不删除任务**，只标状态；它们仍计入完成率分母）
        close_tick(queue, now, scheduler.config.starvation_threshold_s,
                   starved_ids)

    # 运行结束时仍未离开队列、且已等待超过阈值的任务同样计入
    final_now = clock.now_s
    for task in queue.tasks:
        if task.status in (TaskStatus.PENDING, TaskStatus.SUBMITTED):
            waiting = final_now - task.release_time_s
            if (waiting >= scheduler.config.starvation_threshold_s
                    and task.task_id not in starved_ids):
                starved_ids.add(task.task_id)
                task.reason = (task.reason + "；"
                               f"运行结束时已等待 {waiting:.2f}s，"
                               "超过长期未获服务阈值").strip("；")
    starved = len(starved_ids)

    result.starved_task_ids = sorted(starved_ids)
    result.queue_summary = queue.summary()
    # 规划耗时（**计算耗时**维度的来源）：由调度器自己累计，这里只搬运
    result.planning_time_s = float(getattr(scheduler, "planning_time_s", 0.0))
    result.optimizer_outcome = (
        scheduler.last_outcome.to_dict()
        if getattr(scheduler, "last_outcome", None) is not None else None)
    result.optimizer_summary = (
        scheduler.outcome_summary()
        if hasattr(scheduler, "outcome_summary") else None)
    result.metrics = _compute_metrics(queue, result, ledger,
                                      now_s=final_now, starved=starved)
    result.conservation = executor.conservation_report()
    result.metrics["conservation_all"] = result.conservation["all_conserved"]
    return result


def _run_feedback_loop(
    policy: SchedulerPolicy,
    steps: int,
    unavailable: Dict[str, List[Sequence[float]]],
    sim: Any,
    clock: GlobalClock,
    ledger: ResourceLedger,
    executor: UnifiedExecutor,
    queue: TaskQueue,
    scheduler: Any,
    runtime: RuntimeExecutor,
    result: LoopResult,
    keep_decisions: bool,
    starved_ids: set,
    task_gating: str = TASK_GATING_LOOP,
    deadline_offsets: Optional[Dict[str, float]] = None,
) -> LoopResult:
    """计划控制的真闭环：先调度，再由 RuntimeExecutor 产生副作。

    本函数只是 `FeedbackLoopDriver` 的薄封装。抽成驱动类的原因是
    **学习型调度器需要两段式**：先拿到本 tick 的观测，再由策略决定动作，
    最后才提交计划。规则基线是一段式的（观测与决策在一次 `plan()` 里完成），
    但两条路径必须走**同一份** tick 逻辑，否则"学习基线"与"规则基线"
    比的就不是同一件事了。
    """
    driver = FeedbackLoopDriver(
        policy=policy, steps=steps, unavailable=unavailable, sim=sim,
        clock=clock, ledger=ledger, executor=executor, queue=queue,
        scheduler=scheduler, runtime=runtime, result=result,
        keep_decisions=keep_decisions, starved_ids=starved_ids,
        task_gating=task_gating, deadline_offsets=deadline_offsets,
    )
    for _ in range(steps):
        driver.tick()
    return driver.finalize()


class FeedbackLoopDriver:
    """计划控制闭环的**可交互**驱动：`begin()` → （策略决策）→ `commit(plan)`。

    `tick()` 是两者的组合，供规则/优化参考这类"内部自行决策"的调度器使用。
    """

    def __init__(self, policy: SchedulerPolicy, steps: int,
                 unavailable: Dict[str, List[Sequence[float]]],
                 sim: Any, clock: GlobalClock, ledger: ResourceLedger,
                 executor: UnifiedExecutor, queue: TaskQueue, scheduler: Any,
                 runtime: RuntimeExecutor, result: LoopResult,
                 keep_decisions: bool, starved_ids: set,
                 task_gating: str = TASK_GATING_LOOP,
                 deadline_offsets: Optional[Dict[str, float]] = None,
                 share_requires_visible_track: bool = True) -> None:
        if task_gating not in TASK_GATING_MODES:
            raise ValueError(f"task_gating 必须是 {list(TASK_GATING_MODES)}")
        self.task_gating = task_gating
        self.share_requires_visible_track = bool(share_requires_visible_track)
        # 截止余量（秒）。传 None 时用基线值——**基线与旧路径逐位一致**。
        self.deadline_offsets = dict(BASELINE_DEADLINE_OFFSETS)
        if deadline_offsets:
            for key, value in deadline_offsets.items():
                kind = (key if isinstance(key, QueueTaskKind)
                        else QueueTaskKind(str(key)))
                self.deadline_offsets[kind] = float(value)
        self.policy = policy
        self.steps = int(steps)
        self.unavailable = unavailable
        self.sim = sim
        self.clock = clock
        self.ledger = ledger
        self.executor = executor
        self.queue = queue
        self.scheduler = scheduler
        self.runtime = runtime
        self.result = result
        self.keep_decisions = keep_decisions
        self.starved_ids = starved_ids
        self.step_index = 0
        self.observation_violations = 0
        self.current_observation: Optional[CentralObservation] = None
        self.current_time_s = 0.0
        #: 逐 tick 的决策统计（学习基线用它统计非法动作率等）
        self.tick_stats: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def begin(self) -> CentralObservation:
        """推进时钟与场景，执行无条件预测，派生任务，返回本 tick 的中央观测。"""
        self.clock.advance(1.0, f"feedback tick {self.step_index + 1}")
        now = self.clock.now_s
        self.step_index += 1
        self.current_time_s = now
        self.sim.scene.advance_all(1.0)

        for node_id in sorted(self.runtime.centers):
            available = not _in_window(self.unavailable.get(node_id, []), now)
            node = self.executor.node(node_id)
            if node is not None and node.available != available:
                self.executor.set_availability(
                    node_id,
                    available,
                    reason=("configured_unavailable_window"
                            if not available else ""),
                )

        # 无计划也必须随时间预测，但这里不产生测量或融合更新。
        self.runtime.begin_tick(now)
        central_before = self.runtime.central_observation(now)

        for node, valid in zip(central_before.nodes,
                               central_before.node_valid_mask):
            if not valid:
                continue
            unresolved_share = any(
                task.node_id == node.node_id
                and task.kind is QueueTaskKind.SHARE
                and task.status in (TaskStatus.PENDING, TaskStatus.SUBMITTED)
                for task in self.queue.tasks
            )
            try:
                processable = self.runtime.has_processable(node.node_id, now)
                shareable = self.runtime.has_shareable(node.node_id)
                if self.task_gating == TASK_GATING_EXPOSE_ALL:
                    # 把三种候选同时暴露，交给策略选（学习型调度器用）。
                    # 仍然沿用**真实可行性**：没有数据可处理就不派 process，
                    # outbox 为空就不派 share——否则会派生必然空转的任务。
                    allow_share = shareable
                    allow_process = processable
                    allow_sample = True
                else:
                    allow_share = (shareable and not processable
                                   and not unresolved_share)
                    allow_process = processable
                    allow_sample = (not processable and not shareable)
                self.queue.create_from_observation(
                    node,
                    now,
                    deadline_offsets=self.deadline_offsets,
                    allow_share=allow_share,
                    share_requires_visible_track=self.share_requires_visible_track,
                    allow_process=allow_process,
                    allow_sample=allow_sample,
                    allow_estimate_update=False,
                )
            except (DuplicateTaskError, UnknownObjectError) as exc:
                self.result.decisions.append({
                    "time_s": now, "node_id": node.node_id,
                    "decision": "task_creation_failed", "reason": str(exc),
                })
        self.current_observation = central_before
        return central_before

    # ------------------------------------------------------------------

    def commit(self, plan: Optional[ExecutionPlan]) -> Dict[str, Any]:
        """提交本 tick 的计划（可为 None）并做收尾记录。返回本 tick 的统计。"""
        now = self.current_time_s
        stats: Dict[str, Any] = {
            "step": self.step_index, "time_s": round(now, 6),
            "n_planned": 0, "n_applied": 0, "n_rejected": 0,
            "plan_status": "none",
        }
        if plan is not None:
            execution = self.runtime.submit(plan)
            stats.update({
                "n_planned": len(plan.tasks),
                "n_applied": execution.n_applied,
                "n_rejected": execution.n_rejected,
                "plan_status": execution.status.value,
            })
            self.result.plan_log.append({
                "time_s": round(now, 6),
                "plan_id": plan.plan_id,
                "status": execution.status.value,
                "n_applied": execution.n_applied,
                "n_rejected": execution.n_rejected,
                "nodes": plan.node_ids(),
                "runtime_mode": RUNTIME_MODE_FEEDBACK,
                "rejected": [
                    item.to_dict() for item in execution.outcomes
                    if item.outcome is not Outcome.APPLIED
                ],
                "issues": [issue.to_dict() for issue in execution.issues],
            })
            for outcome in execution.outcomes:
                self.queue.mark(
                    outcome.task_id,
                    (TaskStatus.COMPLETED
                     if outcome.outcome is Outcome.APPLIED
                     else TaskStatus.REJECTED),
                    reason=(outcome.reason
                            if outcome.outcome is Outcome.APPLIED
                            else f"执行器拒绝：{outcome.reason}"),
                    plan_id=plan.plan_id,
                )

        central_after = self.runtime.central_observation(now)
        # 每 tick 做一次真值隔离守卫：中央控制面快照里不得出现真值通道。
        # 这不是"声明"，而是对**实际序列化结果**的扫描。
        self.observation_violations += len(observation_truth_violations(
            _json.dumps(central_after.to_dict(), ensure_ascii=False)))
        _record_node_quality(self.result, central_after, now)
        _record_offline_errors(self.result, self.sim, self.runtime.centers, now)
        close_tick(self.queue, now, self.scheduler.config.starvation_threshold_s,
                   self.starved_ids)
        self.tick_stats.append(stats)
        return stats

    # ------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """一段式 tick：观测 → `scheduler.plan()` → 提交（规则/优化参考用）。"""
        central_before = self.begin()
        planning = self.scheduler.plan(
            central_before, self.queue, self.current_time_s,
            plan_id=f"{self.policy.value}-feedback-{self.step_index:04d}",
        )
        if self.keep_decisions:
            for decision in planning.decisions:
                payload = decision.to_dict()
                payload["time_s"] = round(self.current_time_s, 6)
                self.result.decisions.append(payload)
        for decision in planning.decisions:
            if decision.decision in (
                "planned", "deferred", "abandoned",
                "suppressed_duplicate_node", "not_eligible",
            ):
                self.result.timelines.setdefault(decision.node_id, []).append({
                    "time_s": round(self.current_time_s, 6),
                    "task_id": decision.task_id,
                    "kind": decision.kind,
                    "decision": decision.decision,
                    "reason": decision.reasons[0] if decision.reasons else "",
                    "priority": round(decision.priority, 6),
                })
        return self.commit(planning.plan)

    # ------------------------------------------------------------------

    def finalize(self) -> LoopResult:
        result = self.result
        result.runtime_log = list(self.runtime.event_log)
        final_now = self.clock.now_s
        starved_ids = self.starved_ids
        for task in self.queue.tasks:
            if task.status in (TaskStatus.PENDING, TaskStatus.SUBMITTED):
                waiting = final_now - task.release_time_s
                if (waiting >= self.scheduler.config.starvation_threshold_s
                        and task.task_id not in starved_ids):
                    starved_ids.add(task.task_id)
                    task.reason = (task.reason + "；"
                                   f"运行结束时已等待 {waiting:.2f}s，"
                                   "超过长期未获服务阈值").strip("；")
        starved = len(starved_ids)
        result.starved_task_ids = sorted(starved_ids)
        result.queue_summary = self.queue.summary()
        result.planning_time_s = float(
            getattr(self.scheduler, "planning_time_s", 0.0))
        result.optimizer_outcome = (
            self.scheduler.last_outcome.to_dict()
            if getattr(self.scheduler, "last_outcome", None) is not None
            else None
        )
        result.optimizer_summary = (
            self.scheduler.outcome_summary()
            if hasattr(self.scheduler, "outcome_summary") else None
        )
        result.metrics = _compute_metrics(
            self.queue, result, self.ledger, now_s=final_now, starved=starved)
        result.conservation = self.executor.conservation_report()
        result.metrics["conservation_all"] = \
            result.conservation["all_conserved"]
        result.metrics["runtime_feedback"] = _runtime_feedback_metrics(
            result.runtime_log,
            scan_counts=self.runtime.sensor_scan_counts(),
            observation_violations=self.observation_violations,
            unique_task_keys=self.runtime.unique_task_keys(),
        )
        return result



def _record_node_quality(result: LoopResult, central: CentralObservation,
                         now: float) -> None:
    result.node_ages.append({
        "time_s": round(now, 6),
        "valid": list(central.node_valid_mask),
        "ages": [None if age is None else round(age, 6)
                 for age in central.node_information_age_s],
        "content_ages": [None if age is None else round(age, 6)
                         for age in central.node_content_age_s],
        "n_tracks": [len(node.track_ids()) if valid else None
                     for node, valid in zip(central.nodes,
                                            central.node_valid_mask)],
        "mean_track_age_s": [
            _mean_or_none([
                track.information_age_s
                for track, track_valid in zip(node.tracks,
                                              node.track_valid_mask)
                if track_valid
            ]) if valid else None
            for node, valid in zip(central.nodes, central.node_valid_mask)
        ],
        "mean_sigma_m": [
            _mean_or_none([
                max(track.sigma_position)
                for track, track_valid in zip(node.tracks,
                                              node.track_valid_mask)
                if track_valid
            ]) if valid else None
            for node, valid in zip(central.nodes, central.node_valid_mask)
        ],
        "node_ids": [node.node_id for node in central.nodes],
    })


def _record_offline_errors(result: LoopResult, sim: Any,
                           centers: Dict[str, Any], now: float) -> None:
    for node_id, center in centers.items():
        estimated = [track.position for track in center.tracks]
        for target in sim.scene.targets:
            if not estimated:
                continue
            error = min((position - target.position).norm()
                        for position in estimated)
            result.estimate_error_samples.append({
                "time_s": round(now, 6),
                "node_id": node_id,
                "target_id": target.target_id,
                "position_error_m": round(error, 6),
                "provenance": "offline_truth_nearest_track",
            })


def _runtime_feedback_metrics(rows: Sequence[Dict[str, Any]],
                              scan_counts: Optional[Dict[str, int]] = None,
                              observation_violations: int = 0,
                              unique_task_keys: int = 0,
                              ) -> Dict[str, Any]:
    """真闭环的运行时统计。

    ⚠️ 这些数必须**从事件里算出来**，不能硬编码。
    第一版把 `n_duplicate_runtime_tasks` 与 `truth_payload_violations`
    直接写成 0——那等于"用一个常数证明没有重复执行/没有真值泄漏"，
    是自证而不是证据。现在两者都由实际事件、去重集合与载荷键扫描统计得出。
    """
    events = [row.get("event") for row in rows]
    n_runtime_tasks = sum(1 for event in events
                          if event in ("sample", "process", "share", "idle"))
    duplicate_keys = max(0, n_runtime_tasks - int(unique_task_keys))
    payload_violations = sum(
        len(row.get("payload_truth_fields") or []) for row in rows)
    return {
        "n_sensor_measurements": sum(
            int(row.get("n_measurements", 0))
            for row in rows if row.get("event") == "sample"
        ),
        "n_fusion_measurements": sum(
            int(row.get("n_measurements", 0))
            for row in rows if row.get("event") == "process"
        ),
        "n_fusion_updates": sum(
            1 for row in rows
            if row.get("event") == "process" and row.get("fusion_updated")
        ),
        "n_messages_sent": sum(
            int(row.get("n_messages", 0))
            for row in rows if row.get("event") == "share"
        ),
        "sent_comm_bytes": sum(
            float(row.get("sent_comm_bytes", 0.0))
            for row in rows if row.get("event") == "share"
        ),
        "accounted_comm_bytes": sum(
            float(row.get("accounted_comm_bytes", 0.0))
            for row in rows if row.get("event") == "share"
        ),
        "n_runtime_tasks": n_runtime_tasks,
        "n_unique_task_keys": int(unique_task_keys),
        # 每任务只执行一次：运行时任务数必须等于已登记的唯一去重键数
        "n_duplicate_runtime_tasks": duplicate_keys,
        # 真值隔离：由实际发送载荷的键扫描 + 中央观测序列化扫描统计，
        # 两者都必须是 0，且都不是常数。
        "truth_payload_violations": payload_violations + int(
            observation_violations),
        "observed_payload_violations": payload_violations,
        "observed_context_violations": int(observation_violations),
        "n_predict_only_events": sum(1 for event in events
                                     if event == "predict_only"),
        "n_sample_events": sum(1 for event in events if event == "sample"),
        "n_process_events": sum(1 for event in events if event == "process"),
        "n_share_events": sum(1 for event in events if event == "share"),
        "sensor_scans_by_node": dict(scan_counts or {}),
        "definition": (
            "n_sensor_measurements 只统计**被计划触发的**传感器扫描产出；"
            "n_messages_sent 只统计**被 share 任务触发的**真实发送；"
            "n_duplicate_runtime_tasks = 运行时任务数 − 唯一去重键数（恒应为 0）；"
            "truth_payload_violations 由发送载荷的键扫描统计（恒应为 0）"),
    }


def close_tick(queue: TaskQueue, now_s: float, starvation_threshold_s: float,
               starved_ids: set) -> List[str]:
    """一个 tick 的收尾：**先判长期未获服务，再判过期，并保证两者互斥**。

    为什么顺序与互斥都要写死
    ------------------------
    "长期未获服务"与"过期"是两条不同的语义，不能同一条任务都算：

    * **过期**：截止时间到了，任务失去意义；
    * **长期未获服务**：任务一直有效，但始终没轮到它（典型的饿死）。

    如果先做过期、再统计"仍在排队且等待超阈值"的任务，那么带截止时间的任务
    永远不会被判长期未获服务——实测默认场景里所有任务都带 2~6s 截止余量，
    而阈值是 8s，于是 `n_starved` 恒为 0，这条语义**形同不存在**。
    反过来若两个计数都记同一条任务，完成率分母会被重复计入（偏低也是错）。

    因此这里的规则是：等待已超过阈值、且**不是**因截止时间到期而离开队列的，
    才判长期未获服务。

    返回本 tick 新判定的 task_id 列表。
    """
    pending_too_long = [
        task.task_id for task in queue.tasks
        if task.status in (TaskStatus.PENDING, TaskStatus.SUBMITTED)
        and now_s - task.release_time_s >= starvation_threshold_s]
    queue.expire_overdue(now_s)
    by_id = {task.task_id: task for task in queue.tasks}
    newly: List[str] = []
    for task_id in pending_too_long:
        task = by_id.get(task_id)
        if task is None or task.status is TaskStatus.EXPIRED:
            continue              # 先到期 → 归入"过期"，不重复计入
        if task.task_id in starved_ids:
            continue
        starved_ids.add(task.task_id)
        newly.append(task.task_id)
        task.reason = (task.reason + "；"
                       f"长期未获服务：等待 "
                       f"{now_s - task.release_time_s:.2f}s 仍未轮到，"
                       f"超过阈值 {starvation_threshold_s:g}s").strip("；")
    return newly


def _in_window(windows: Sequence[Sequence[float]], now_s: float) -> bool:
    for start, end in windows:
        if start <= now_s <= end:
            return True
    return False


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _config_dict(config: SchedulingConfig) -> Dict[str, Any]:
    return {
        "task_levels": {kind.value: level
                        for kind, level in config.task_levels.items()},
        "max_information_age_s": config.max_information_age_s,
        "max_sigma_position_m": config.max_sigma_position_m,
        "starvation_threshold_s": config.starvation_threshold_s,
        "abandon_after_s": config.abandon_after_s,
        "max_tasks_per_node_per_tick": config.max_tasks_per_node_per_tick,
        "allow_multi_node_same_task": config.allow_multi_node_same_task,
        "charge_duplicate_as_overhead": config.charge_duplicate_as_overhead,
        "weights": {"level": config.weight_level,
                    "waiting": config.weight_waiting,
                    "freshness": config.weight_freshness,
                    "quality": config.weight_quality,
                    "source_inconsistency": config.weight_source_inconsistency},
        "feature_gates": {
            "freshness": config.use_information_age,
            "uncertainty": config.use_estimate_uncertainty,
            "source_consistency": config.use_source_consistency,
        },
    }


def _compute_metrics(queue: TaskQueue, result: LoopResult, ledger: Any,
                     now_s: float, starved: int) -> Dict[str, Any]:
    """闭环指标。**完成率的分母包含放弃/过期/饿死**，不能靠删任务做高。"""
    completed = [task for task in queue.tasks
                 if task.status is TaskStatus.COMPLETED]
    expired = [task for task in queue.tasks if task.status is TaskStatus.EXPIRED]
    abandoned = [task for task in queue.tasks
                 if task.status is TaskStatus.CANCELLED]
    rejected = [task for task in queue.tasks if task.status is TaskStatus.REJECTED]
    denominator = (len(completed) + len(expired) + len(abandoned)
                   + len(rejected) + starved)

    waits: List[float] = []
    for task in completed:
        waits.append(max(0.0, _completion_time(result, task) - task.release_time_s))

    violations = 0
    for task in queue.tasks:
        if task.deadline_s is None or task.status is not TaskStatus.COMPLETED:
            continue
        if _completion_time(result, task) > task.deadline_s + 1e-9:
            violations += 1

    per_node: Dict[str, Dict[str, Any]] = {}
    for node_id, timeline in result.timelines.items():
        planned = sum(1 for row in timeline if row["decision"] == "planned")
        deferred = sum(1 for row in timeline
                       if row["decision"] in ("deferred", "not_eligible"))
        per_node[node_id] = {
            "n_planned": planned,
            "n_deferred_or_not_eligible": deferred,
            "n_abandoned": sum(1 for row in timeline
                               if row["decision"] == "abandoned"),
            "n_suppressed_duplicate": sum(
                1 for row in timeline
                if row["decision"] == "suppressed_duplicate_node"),
            "kinds_planned": sorted({row["kind"] for row in timeline
                                     if row["decision"] == "planned"}),
            "timeline_len": len(timeline),
        }

    # 逐节点资源占用与通信开销：**从真实账本读**，不从计划日志反推
    occupancy: Dict[str, Dict[str, float]] = {}
    totals = ledger.totals_by_node()
    for node_id, bucket in totals.items():
        occupancy[node_id] = {
            f"{unit.value}_consumed": bucket.get(f"consumed_{unit.value}", 0.0)
            for unit in BUDGET_UNITS}
    consumed_comm = sum(bucket.get(f"consumed_{unit.value}", 0.0)
                        for bucket in totals.values()
                        for unit in (ResourceUnit.COMM_BYTE,))
    n_rejections = {node_id: bucket.get("n_rejected", 0.0)
                    for node_id, bucket in totals.items()}

    return {
        "n_tasks_total": len(queue.tasks),
        "n_completed": len(completed),
        "n_expired": len(expired),
        "n_abandoned": len(abandoned),
        "n_rejected_by_executor": len(rejected),
        "n_starved": starved,
        "completion_denominator": denominator,
        "completion_rate": (len(completed) / denominator) if denominator else 0.0,
        "deadline_violations": violations,
        "completed_on_time": sum(
            1 for task in completed
            if task.deadline_s is None
            or _completion_time(result, task) <= task.deadline_s + 1e-9),
        "mean_waiting_s": (sum(waits) / len(waits)) if waits else 0.0,
        "max_waiting_s": max(waits) if waits else 0.0,
        "mean_estimate_quality": _mean_estimate_quality(result),
        "position_error_mean_m": _position_error_summary(result)["mean_m"],
        "position_error_rmse_m": _position_error_summary(result)["rmse_m"],
        "position_error_max_m": _position_error_summary(result)["max_m"],
        "position_error_provenance": "offline_truth_nearest_track",
        "compute_time_s": float(result.planning_time_s),
        "per_node": per_node,
        "per_node_occupancy": occupancy,
        "per_node_executor_rejections": n_rejections,
        "comm_overhead_bytes": consumed_comm,
        "service_capacity": _service_capacity(result, queue, per_node),
        "evaluation_vector": _measured_vector(
            result, completed=len(completed), on_time=sum(
                1 for task in completed
                if task.deadline_s is None
                or _completion_time(result, task) <= task.deadline_s + 1e-9),
            denominator=denominator, occupancy=occupancy,
            comm_bytes=consumed_comm).to_dict(),
        "note": ("完成率分母 = 完成 + 过期 + 主动放弃 + 执行器拒绝 + 长期未获服务。"
                 "主动放弃**不会**让分母变小，因此删掉难任务只会让完成率下降。"
                 "另见 `service_capacity`：当派生的任务数远大于"
                 "「节点数 × tick 数 × 每节点每 tick 限额」时，完成率由**服务上限**"
                 "决定，而不是由调度规则决定——此时完成率不能当作策略优劣的证据。"
                 "`evaluation_vector` 是**实测**六维向量（provenance=measured），"
                 "与优化参考内部的**预测**向量严格区分，不得混用。"),
    }


def _position_error_summary(result: LoopResult) -> Dict[str, float]:
    values = [float(row["position_error_m"])
              for row in result.estimate_error_samples]
    if not values:
        return {"mean_m": 0.0, "rmse_m": 0.0, "max_m": 0.0}
    return {
        "mean_m": sum(values) / len(values),
        "rmse_m": math.sqrt(sum(value * value for value in values) / len(values)),
        "max_m": max(values),
    }


def _mean_estimate_quality(result: LoopResult) -> float:
    """实测估计质量：逐 tick 逐节点取 1/(1+平均信息年龄)，再对全部样本取均值。

    只看**已到达的**节点摘要（中央看得见什么就统计什么）；
    一个节点都没有到达的 tick 不进样本，而不是记 0——
    记 0 会把"没有信息"错误地算成"信息质量最差"。
    """
    samples: List[float] = []
    for row in result.node_ages:
        for age in row.get("mean_track_age_s", []):
            if age is None:
                continue
            samples.append(1.0 / (1.0 + float(age)))
    return (sum(samples) / len(samples)) if samples else 0.0


def _measured_vector(result: LoopResult, completed: int, on_time: int,
                     denominator: int, occupancy: Dict[str, Dict[str, float]],
                     comm_bytes: float) -> Any:
    """组装**实测**六维评价向量（维度定义取自 `optimization.EVALUATION_METRICS`）。

    维度定义只允许有一处实现：这里复用 `optimization` 的 `EvaluationVector`，
    因此"规则基线与优化参考用同一把尺子量"这件事是结构保证，不是约定。
    """
    from resource_management.optimization import EvaluationVector

    ratios: List[float] = []
    per_unit: Dict[str, Dict[str, float]] = {}
    for node_id, bucket in sorted(occupancy.items()):
        per_unit[node_id] = {}
        for unit in BUDGET_UNITS:
            consumed = float(bucket.get(f"{unit.value}_consumed", 0.0))
            per_unit[node_id][unit.value] = round(consumed, 9)
    node_ids = sorted(result.timelines.keys())
    for node_id in node_ids:
        # 容量从**逐节点配额的记录**里取（与执行器注册时同一份配置）
        capacities = result.config.get("node_budgets", {}).get(node_id) or {}
        for unit in BUDGET_UNITS:
            capacity = float(capacities.get(unit.value, 0.0) or 0.0)
            consumed = float(per_unit.get(node_id, {}).get(unit.value, 0.0))
            if capacity > 0:
                ratios.append(consumed / capacity)
    return EvaluationVector(
        values={
            "service_completion": (completed / denominator)
            if denominator else 0.0,
            "task_timeliness": (on_time / completed) if completed else 0.0,
            "estimate_quality": _mean_estimate_quality(result),
            "resource_consumption": (sum(ratios) / len(ratios))
            if ratios else 0.0,
            "communication_overhead": float(comm_bytes),
            "compute_time": float(result.planning_time_s),
        },
        per_unit=per_unit,
        provenance="measured",
    )


def _service_capacity(result: LoopResult, queue: TaskQueue,
                      per_node: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """服务能力对照：把"做得完做不完"与"先做哪一件"两件事分开。

    本版本执行器只支持立即执行，因此**每节点每 tick 最多一个任务**（见
    `SchedulingConfig.max_tasks_per_node_per_tick` 的说明）。任务却是由可见
    航迹逐 tick 派生的（每目标每 tick 采样 + 更新 + 共享），需求速率通常远高于
    服务上限。于是"完成率低"首先是一个**供需关系**，不是调度失误——
    不把这一点写清楚，指标表会诱导出错误结论。
    """
    node_ids = set(result.timelines.keys())
    cap = int(result.config.get("max_tasks_per_node_per_tick", 1) or 1)
    maximum = len(node_ids) * max(0, result.steps) * cap
    planned = sum(bucket["n_planned"] for bucket in per_node.values())
    return {
        "n_nodes": len(node_ids),
        "steps": result.steps,
        "max_tasks_per_node_per_tick": cap,
        "max_serviceable_tasks": maximum,
        "n_planned": planned,
        "service_utilization": (planned / maximum) if maximum else 0.0,
        "n_tasks_created": len(queue.tasks),
        "demand_per_tick": (len(queue.tasks) / result.steps)
        if result.steps else 0.0,
        "demand_over_capacity": (len(queue.tasks) / maximum)
        if maximum else 0.0,
        "interpretation": ("service_utilization = 1.0 表示每个节点每 tick 都用满"
                           "了执行上限；demand_over_capacity > 1 表示派生任务数"
                           "超过任何调度策略可能完成的量，此时完成率由服务上限"
                           "封顶，**不能**用来说明某策略更差或更好。"),
    }


def _completion_time(result: LoopResult, task: Any) -> float:
    """任务完成时刻：取账本/计划日志里该任务最后一次 applied 的记录。"""
    plan_id = getattr(task, "submitted_plan_id", "")
    for row in result.plan_log:
        if row["plan_id"] == plan_id:
            return float(row["time_s"])
    return float(result.steps)
