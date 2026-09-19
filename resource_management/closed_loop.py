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
    CentralObservationStore,
    node_observation_from_fusion,
    publish_node_observation,
)
from resource_management.scheduling import (
    SchedulerPolicy,
    SchedulingConfig,
    build_scheduler,
)
from resource_management.tasks import QueueTaskKind, TaskQueue, TaskStatus
from resource_management.units import BUDGET_UNITS, ResourceUnit

BASE_CONFIG = ec.CONFIG_PATH

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
            "starved_task_ids": self.starved_task_ids,
            "planning_time_s": round(self.planning_time_s, 9),
            "optimizer_outcome": self.optimizer_outcome,
            "optimizer_summary": self.optimizer_summary,
            "decisions": self.decisions,
        }


def run_closed_loop(
    policy: SchedulerPolicy = SchedulerPolicy.RULE,
    seed: int = 42,
    steps: int = 24,
    scheduling_config: Optional[SchedulingConfig] = None,
    mechanisms: Optional[Dict[str, Any]] = None,
    node_budgets: Optional[Dict[str, Dict[ResourceUnit, float]]] = None,
    optimizer_config: Any = None,
    keep_decisions: bool = True,
) -> LoopResult:
    """跑一次闭环：世界 → 融合 → 通信 → 调度 → 执行 → 记账。

    **同一份可见观测、同一个任务队列、同一个执行器**：规则基线与优化参考
    在这里只有 `build_scheduler(policy)` 不同，其余代码路径完全一致。

    `optimizer_config`（`optimization.OptimizerConfig`）只在优化参考时生效，
    它带来计算预算与预测假设；两条路径的任务定义、资源预算、信息权限、
    执行器**完全相同**。
    """
    mech = dict(DEFAULT_MECHANISMS)
    mech.update(mechanisms or {})
    unavailable = {node: list(windows) for node, windows
                   in (mech.get("unavailable_windows") or {}).items()}
    bias = dict(mech.get("bias") or {})

    # ---------- 世界（真值只在这里） ----------
    import json

    with open(BASE_CONFIG, "r", encoding="utf-8") as handle:
        base = json.load(handle)
    radar_base, target_base = base["radar"], base["targets"][0]

    overrides = {
        "radars": [_radar(radar_base, node_id, layout["x"], layout["y"],
                          layout["heading_deg"])
                   for node_id, layout in NODE_LAYOUT.items()],
        "targets": [
            # 沿 x=1500 从 y=-5000 向 +y 运动 → 依次进入 A、重叠区、B
            _target(target_base, "TGT_1", 1500.0, -5000.0, 0.0, 400.0),
            # 第二个目标全程在 A 侧，用于产生并发任务
            _target(target_base, "TGT_2", -1200.0, -3000.0, 0.0, 250.0),
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
    # 逐节点三类单位的**容量**（实测资源占用率的分母；只读，不参与调度）
    result.config["node_budgets"] = {
        node_id: {unit.value: float(node.budget.capacity.get(unit, 0.0))
                  for unit in BUDGET_UNITS}
        for node_id, node in sorted(executor.nodes.items())}
    optimizer_config = getattr(scheduler, "optimizer_config", None)
    if optimizer_config is not None:
        result.config["optimizer"] = {
            "kind": optimizer_config.kind.value,
            "horizon_ticks": int(getattr(scheduler, "_horizon", 1)),
            "budget": optimizer_config.budget.to_dict(),
            "objective": optimizer_config.objective.describe(),
            "prediction_assumptions":
                getattr(scheduler, "prediction", None).assumptions()
                if getattr(scheduler, "prediction", None) is not None else [],
        }
    result.timelines = {node_id: [] for node_id in NODE_LAYOUT}
    #: 被判"长期未获服务"的 task_id（与"过期"互斥，见主循环第 ⑥ 步）
    starved_ids: set = set()

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
            publish_node_observation(observation, bus, node_id, now_s=now)

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
            except Exception as exc:  # noqa: BLE001
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
                    "quality": config.weight_quality},
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
