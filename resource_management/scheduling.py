"""非学习型教学资源调度基线（v4.5）。

三条纪律
--------
1. **所有基线共用同一份可见观测、同一个任务队列、同一个执行器**。
   三个策略类只实现"怎么排序、给谁"，不碰抓观测、不碰建任务、不碰记账。
2. **只输出标准 `ExecutionPlan`**，由 `UnifiedExecutor` 统一校验与记账。
   调度器**不得**直接修改传感器或融合器的任何内部字段——
   本模块只 import 观测与队列，不 import `engine` / `sensor` / `fusion`
   （由测试用 AST 扫 import 钉住）。
3. **优先级来自显式的教学任务等级与可观测服务需求**，
   **不设**军事目标价值、对抗效能、威胁度这类目标。

服务需求的"可观测"含义
----------------------
调度只能用**观测里真的有的量**判断"该不该服务"：

| 需求 | 观测来源 | 判据 |
| --- | --- | --- |
| 新鲜度需求 | `TrackObservation.information_age_s` | 年龄越大越该更新 |
| 估计质量需求 | `TrackObservation.sigma_position` | 协方差越大越该更新 |
| 等待时间 | 任务入队时刻 | 等得越久越该服务（防饿死） |
| 截止时间 | 任务 `deadline_s` | EDF 用 |

没有任何一项需要目标价值或对抗效能——那些量在观测里也不存在。

三类失败语义必须分清（否则完成率会被做假）
------------------------------------------
| 名称 | 含义 | 计入完成率分母？ |
| --- | --- | --- |
| **任务完成** `completed` | 已提交且执行器返回 applied | ✅（分子） |
| **任务过期** `expired` | 超过 `deadline_s` 仍未完成 | ✅ |
| **主动放弃** `abandoned` | 调度器**主动**决定不再服务（记录原因） | ✅ |
| **重复请求** `duplicate` | 去重键相同的再次请求 | ❌（不是新任务），但**单独统计** |
| **长期未获服务** `starved` | 等待超过 `starvation_threshold_s` | ✅ |

`completion_rate = completed / (completed + expired + abandoned + starved
+ rejected_by_executor)` —— **主动放弃不会把分母变小**，
所以"删掉难任务"只会让完成率下降，不可能虚高。
`tests/test_resource_scheduling.py` 直接钉住这条。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from resource_management.model import ExecutionPlan, TaskRequest
from resource_management.observation import (
    CentralObservation,
    NodeObservation,
    TrackObservation,
)
from resource_management.tasks import (
    QueueTaskKind,
    QueuedTask,
    TaskQueue,
    TaskStatus,
)
from resource_management.units import ResourceUnit, format_cost


class SchedulerPolicy(str, Enum):
    """调度策略标识。

    前三个是**规则基线**（`BASELINE_POLICIES`），后两个是**优化参考**
    （见 `optimization.py`）。它们**不是**"更强的同类方法"：
    优化参考只用当前可见观测 + 显式预测模型，且受计算预算约束。
    """

    ROUND_ROBIN = "round_robin"
    EDF = "edf"
    RULE = "rule"
    #: 小规模**完全枚举**（只在能完全枚举时才谈精确最优）
    ENUMERATION = "enumeration"
    #: 带计算预算的**滚动规划**（只称优化参考，不声明最优性）
    ROLLING_HORIZON = "rolling_horizon"


#: 规则基线（v4.5 冻结的三个对照方法）
BASELINE_POLICIES: Tuple[SchedulerPolicy, ...] = (
    SchedulerPolicy.ROUND_ROBIN,
    SchedulerPolicy.EDF,
    SchedulerPolicy.RULE,
)

#: 优化参考（非学习；受计算预算约束）
OPTIMIZATION_POLICIES: Tuple[SchedulerPolicy, ...] = (
    SchedulerPolicy.ENUMERATION,
    SchedulerPolicy.ROLLING_HORIZON,
)

POLICY_CN: Dict[SchedulerPolicy, str] = {
    SchedulerPolicy.ROUND_ROBIN: "轮询调度",
    SchedulerPolicy.EDF: "最早截止时间优先",
    SchedulerPolicy.RULE: "规则调度（等待时间 + 数据新鲜度 + 估计质量）",
    SchedulerPolicy.ENUMERATION: "枚举优化参考（小规模完全枚举）",
    SchedulerPolicy.ROLLING_HORIZON: "滚动规划优化参考（带计算预算）",
}

#: 决策去向
DECISION_PLANNED = "planned"
DECISION_DEFERRED = "deferred"
DECISION_ABANDONED = "abandoned"
DECISION_SUPPRESSED_DUPLICATE_NODE = "suppressed_duplicate_node"
DECISION_NOT_ELIGIBLE = "not_eligible"


@dataclass
class SchedulingConfig:
    """调度配置（**全部是教学设定，可整体替换**）。

    ⚠️ `task_levels` 是"教学任务等级"，只表达**服务优先级**，
    与任何军事目标价值、威胁度或对抗效能无关。
    """

    #: 教学任务等级（数值越大越优先）。执行器任务类型之外不引入任何价值维度。
    task_levels: Dict[QueueTaskKind, int] = field(default_factory=lambda: {
        QueueTaskKind.ESTIMATE_UPDATE: 3,   # 已有估计的更新：维持已有服务
        QueueTaskKind.PROCESS: 3,
        QueueTaskKind.PREDEFINED_SAMPLE: 2,  # 预定义采样：有节奏地补新数据
        QueueTaskKind.SHARE: 1,              # 共享：让别人也能用
    })
    #: 服务需求（可观测）：信息年龄超过它就算"该服务"
    max_information_age_s: float = 3.0
    #: 服务需求（可观测）：位置标准差超过它就算"该服务"（米）
    max_sigma_position_m: float = 150.0
    #: 长期未获服务判定（秒）
    starvation_threshold_s: float = 8.0
    #: 主动放弃：等待超过它且仍无法服务则放弃（None = 永不主动放弃）
    abandon_after_s: Optional[float] = 20.0
    #: 每个 tick 每个节点最多提交几个任务。
    #:
    #: **本版本只能取 1**，且这不是保守取值而是执行器的硬约束：执行器要求
    #: 计划内所有任务 `start_s == now`（只支持立即执行），而同一节点上两段
    #: 从 `now` 开始的占用必然重叠 → 触发 `PLAN_FATAL` 的占用冲突 → **整份计划
    #: 被拒、一个任务也不执行**。
    #:
    #: 这个坑很隐蔽：把上限设成 2 及以上时，结果是"计划数翻倍、完成数归零"，
    #: 而错误只出现在执行器拒绝记录里，指标表上看起来像"策略变差了"。
    #: 因此这里直接**拒绝**非法取值，而不是安静地接受它。
    max_tasks_per_node_per_tick: int = 1
    #: 每个 tick 全局最多提交几个任务（None = 不限）
    max_tasks_per_tick: Optional[int] = None

    # --- 同一任务能否由多个节点处理（用户要求写成配置规则）---
    #: True = 允许多个节点在同一 tick 处理"同一任务"；
    #: False = 只保留优先级最高的那个节点，其余记为
    #: `suppressed_duplicate_node`（原因入账，不计完成率分子）
    allow_multi_node_same_task: bool = False
    #: 允许时，多节点处理是否**计为重复开销**（真实成本照扣，另记一笔重复量）
    charge_duplicate_as_overhead: bool = True

    # --- 规则调度的权重（显式，便于解释）---
    #
    # **等级权重为什么是 0.3 而不是 1.0**（这是一次真实的缺陷修复，不是调参）：
    # 等级差最大为 3−1=2，另三项之和上限为 0.5+0.6+0.4=1.5。
    # 当 `weight_level = 1.0` 时，等级差 2.0 > 1.5，意味着**等级永远压过服务需求**，
    # 规则调度退化成"严格优先级"：实测 seed=42/24 tick 下，48 个执行名额
    # **全部**给了等级 3 的 ESTIMATE_UPDATE，SHARE 完成 0/42、SAMPLE 完成 0/34，
    # 即低等级服务被永久饿死。
    # 取 0.3 后等级差为 0.6 < 1.5：等级仍然起作用，但一个已长期等待且数据已旧的
    # 低等级任务**可以**翻越高等级的新任务——这正是"等待时间"这一项存在的意义。
    # 注意这并不保证任何策略更好，只是让"长期未获服务"成为可被避免而非必然的状态。
    weight_level: float = 0.3
    weight_waiting: float = 0.5
    weight_freshness: float = 0.6
    weight_quality: float = 0.4

    def validate(self) -> None:
        if self.max_tasks_per_node_per_tick < 1:
            raise ValueError("max_tasks_per_node_per_tick 至少为 1")
        if self.max_tasks_per_node_per_tick > 1:
            raise ValueError(
                "max_tasks_per_node_per_tick 只能是 1：执行器只支持立即执行"
                "（计划内 start_s 必须等于当前时钟），同一节点上两段从当前时刻"
                "开始的占用必然重叠，会触发 PLAN_FATAL 的占用冲突并使**整份计划"
                "被拒**（表现为计划数增加、完成数归零）。"
                "要提高服务吞吐请调大节点资源预算，或降低任务派生速率，"
                "不要调大这个上限。"
            )
        if self.starvation_threshold_s <= 0:
            raise ValueError("starvation_threshold_s 必须为正")
        if self.abandon_after_s is not None and self.abandon_after_s <= 0:
            raise ValueError("abandon_after_s 必须为正或 None")
        for kind in QueueTaskKind:
            if kind not in self.task_levels:
                raise ValueError(f"task_levels 缺少 {kind.value}")


@dataclass
class SchedulingDecision:
    """一次调度决策——**"为什么这么分工"就写在这里**。"""

    task_id: str
    node_id: str
    kind: str
    decision: str
    priority: float = 0.0
    #: 逐项理由（人类可读），至少一条
    reasons: List[str] = field(default_factory=list)
    #: 结构化证据：打分用到的每个量（可被 AI 与报告直接引用）
    evidence: Dict[str, Any] = field(default_factory=dict)
    policy: str = ""
    deferred_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "kind": self.kind,
            "decision": self.decision,
            "priority": round(self.priority, 6),
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
            "policy": self.policy,
            "deferred_reason": self.deferred_reason,
        }


@dataclass
class SelectionResult:
    """一次**选择**的结果：挑中的任务 + 为什么挑它们（含未挑中的原因）。"""

    chosen: List[QueuedTask] = field(default_factory=list)
    decisions: List[SchedulingDecision] = field(default_factory=list)


@dataclass
class PlanningResult:
    """一个 tick 的规划结果：标准 `ExecutionPlan` + 全部决策记录。"""

    plan: Optional[ExecutionPlan]
    decisions: List[SchedulingDecision] = field(default_factory=list)
    abandoned: List[SchedulingDecision] = field(default_factory=list)
    deferred: List[SchedulingDecision] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict() if self.plan else None,
            "decisions": [item.to_dict() for item in self.decisions],
            "n_planned": sum(1 for item in self.decisions
                             if item.decision == DECISION_PLANNED),
            "n_abandoned": len(self.abandoned),
            "n_deferred": len(self.deferred),
        }


# ----------------------------------------------------------------------
# 观测查询小工具（三个策略共用，保证"同一份可见观测"）
# ----------------------------------------------------------------------


def _track_index(observation: NodeObservation
                 ) -> Dict[str, TrackObservation]:
    return {track.track_id: track for track, valid
            in zip(observation.tracks, observation.track_valid_mask) if valid}


def task_urgency(task: QueuedTask, observation: Optional[NodeObservation]
                 ) -> Dict[str, Any]:
    """计算一个任务的**可观测服务需求**（三个策略共用同一套量）。

    返回的量全部来自观测或任务自身队列信息，**不含真值、不含目标价值**。
    """
    tracks = _track_index(observation) if observation is not None else {}
    ages: List[float] = []
    sigmas: List[float] = []
    for target in task.targets:
        track = tracks.get(target)
        if track is None:
            continue
        ages.append(track.information_age_s)
        sigmas.append(max(track.sigma_position))
    return {
        "information_age_s": (max(ages) if ages else None),
        "sigma_position_m": (max(sigmas) if sigmas else None),
        "n_targets": len(task.targets),
        "n_targets_visible": len(ages),
    }


class SchedulerBase:
    """调度器基类：**统一入口、统一观测、统一队列、统一执行器输出的计划**。"""

    policy: SchedulerPolicy = SchedulerPolicy.ROUND_ROBIN

    def __init__(self, config: Optional[SchedulingConfig] = None) -> None:
        self.config = config or SchedulingConfig()
        self.config.validate()
        self._round_robin_cursor = 0
        #: 规划耗时（**计算耗时**维度）。**所有**调度器（含规则基线）都在
        #: 同一处计时，因此这一维度是对称测量的——若只给优化参考计时，
        #: 规则基线的耗时会被记成 0，对比就失真了。
        self.planning_time_s: float = 0.0
        self.planning_calls: int = 0
        self.last_planning_time_s: float = 0.0

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return POLICY_CN[self.policy]

    def _order_nodes(self, observation: CentralObservation) -> List[str]:
        """节点顺序（轮询策略用它实现"轮流"）。"""
        node_ids = [node.node_id for node, valid
                    in zip(observation.nodes, observation.node_valid_mask)
                    if valid]
        if not node_ids:
            return []
        start = self._round_robin_cursor % len(node_ids)
        ordered = node_ids[start:] + node_ids[:start]
        self._round_robin_cursor = (self._round_robin_cursor + 1) % len(node_ids)
        return ordered

    def _priority(self, task: QueuedTask, urgency: Dict[str, Any],
                  now_s: float) -> Tuple[float, List[str], Dict[str, Any]]:
        """子类实现：返回 (优先级, 理由, 证据)。"""
        raise NotImplementedError

    # ------------------------------------------------------------------

    def plan(
        self,
        observation: CentralObservation,
        queue: TaskQueue,
        now_s: float,
        plan_id: str,
    ) -> PlanningResult:
        """生成一份标准 `ExecutionPlan`（不执行、不记账）。

        这是**模板方法**：候选筛选（不可服务 / 主动放弃）由 `_classify_due`
        统一完成，选择逻辑由 `_select` 提供。规则基线与优化参考因此共享
        同一套"谁能被服务、什么情况算放弃"的语义——差别只在**选择哪一条**。

        计时也在这里统一做：`compute_time` 维度对两条路径**对称测量**。
        """
        started = time.perf_counter()
        try:
            return self._plan_inner(observation, queue, now_s, plan_id)
        finally:
            self.last_planning_time_s = time.perf_counter() - started
            self.planning_time_s += self.last_planning_time_s
            self.planning_calls += 1

    def _plan_inner(self, observation: CentralObservation, queue: TaskQueue,
                    now_s: float, plan_id: str) -> PlanningResult:
        due = queue.due(now_s)
        if not due:
            return PlanningResult(plan=None)

        node_order = self._order_nodes(observation)
        by_node: Dict[str, NodeObservation] = {
            node.node_id: node for node, valid
            in zip(observation.nodes, observation.node_valid_mask) if valid
        }

        candidates, decisions, abandoned, deferred = self._classify_due(
            due, by_node, now_s)
        if not candidates:
            return PlanningResult(plan=None, decisions=decisions,
                                  abandoned=abandoned, deferred=deferred)

        selection = self._select(candidates, observation, by_node, now_s,
                                 node_order)
        decisions.extend(selection.decisions)
        chosen = selection.chosen

        if not chosen:
            return PlanningResult(plan=None, decisions=decisions,
                                  abandoned=abandoned, deferred=deferred)

        # --- 输出**标准 ExecutionPlan**（唯一的输出形式）---
        requests: List[TaskRequest] = [
            task.to_task_request(start_s=now_s) for task in chosen]
        plan = ExecutionPlan(plan_id=plan_id, submit_time_s=now_s,
                             tasks=requests,
                             note=f"[{self.policy.value}] {self.name}")
        for task in chosen:
            task.status = TaskStatus.SUBMITTED
            task.submitted_plan_id = plan_id
            task.reason = (f"已由 {self.policy.value} 排入计划 {plan_id}；"
                           f"优先级 "
                           f"{self._priority(task, task_urgency(task, by_node.get(task.node_id)), now_s)[0]:.4f}")
        return PlanningResult(plan=plan, decisions=decisions,
                              abandoned=abandoned, deferred=deferred)

    # ------------------------------------------------------------------

    def _classify_due(
        self,
        due: Sequence[QueuedTask],
        by_node: Dict[str, NodeObservation],
        now_s: float,
    ) -> Tuple[List[QueuedTask], List[SchedulingDecision],
               List[SchedulingDecision], List[SchedulingDecision]]:
        """① 候选筛选：不可服务 → `not_eligible`；等太久 → `abandoned`。

        **规则基线与优化参考共用这一份实现**：如果把这段逻辑各写一遍，
        "什么情况算放弃"迟早会在两条路径上分叉，那时两种方法的对比就不再
        是"选择策略不同"，而是"问题定义不同"——这正是本阶段要避免的。
        """
        decisions: List[SchedulingDecision] = []
        abandoned: List[SchedulingDecision] = []
        deferred: List[SchedulingDecision] = []
        candidates: List[QueuedTask] = []
        for task in due:
            waiting = now_s - task.release_time_s
            observation_for_node = by_node.get(task.node_id)
            eligible = self._eligible(task, observation_for_node)
            if not eligible[0]:
                decision = SchedulingDecision(
                    task_id=task.task_id, node_id=task.node_id,
                    kind=task.kind.value, decision=DECISION_NOT_ELIGIBLE,
                    priority=0.0, policy=self.policy.value,
                    reasons=[eligible[1]],
                    evidence={"waiting_s": round(waiting, 6)})
                decisions.append(decision)
                deferred.append(decision)
                continue
            if (self.config.abandon_after_s is not None
                    and waiting > self.config.abandon_after_s):
                decision = SchedulingDecision(
                    task_id=task.task_id, node_id=task.node_id,
                    kind=task.kind.value, decision=DECISION_ABANDONED,
                    priority=0.0, policy=self.policy.value,
                    reasons=[
                        f"等待 {waiting:.2f}s 已超过主动放弃阈值 "
                        f"{self.config.abandon_after_s:g}s",
                        "该节点在这段时间内始终没有可用的执行窗口",
                    ],
                    evidence={"waiting_s": round(waiting, 6),
                              "abandon_after_s": self.config.abandon_after_s})
                decisions.append(decision)
                abandoned.append(decision)
                task.status = TaskStatus.CANCELLED
                task.reason = (f"调度器主动放弃：等待 {waiting:.2f}s 超阈值"
                               f"（**仍计入完成率分母**）")
                continue
            candidates.append(task)
        return candidates, decisions, abandoned, deferred

    def _duplicate_rule(
        self,
        scored: List[Tuple[float, QueuedTask]],
        decisions: List[SchedulingDecision],
    ) -> List[Tuple[float, QueuedTask]]:
        """③ 同一任务多节点规则（配置化）——**两条选择路径共用**。

        ⚠️ 只有**绑定目标**的任务才谈得上"同一任务"。
        无目标任务（预定义采样、共享）的对象是"本节点自己的数据"，
        两个节点各自采样/共享**不是重复**。
        第一版把 `targets=()` 也当成同一个键，结果把两个节点合法的
        共享任务互相抑制了（实测一轮 55 次误抑制），
        还顺带把"规则调度通信量更低"这个结论变成了假象。
        """
        if self.config.allow_multi_node_same_task:
            if not self.config.charge_duplicate_as_overhead:
                return scored
            # 允许时：把"同一目标被多个节点同时处理"记成**重复开销**
            seen: Dict[Tuple[str, Tuple[str, ...]], List[str]] = {}
            for _priority, task in scored:
                if not task.targets:
                    continue
                key = (task.kind.value, tuple(task.targets))
                seen.setdefault(key, []).append(task.node_id)
            for key, node_ids in seen.items():
                if len(node_ids) < 2:
                    continue
                decisions.append(SchedulingDecision(
                    task_id=f"duplicate:{key[0]}:{list(key[1])}",
                    node_id=",".join(sorted(node_ids)),
                    kind=key[0], decision=DECISION_PLANNED,
                    priority=0.0, policy=self.policy.value,
                    reasons=[
                        f"配置允许同一任务多节点处理：{sorted(node_ids)} 同时承接 "
                        f"{key[0]} targets={list(key[1])}",
                        "按配置 charge_duplicate_as_overhead=True，"
                        "本次重复计入**重复开销**（真实成本仍由执行器照扣）",
                    ],
                    evidence={"duplicate_nodes": sorted(node_ids),
                              "kind": key[0], "targets": list(key[1]),
                              "charged_as_overhead": True}))
            return scored

        best_by_key: Dict[Tuple[str, Tuple[str, ...]], Tuple[float, str]] = {}
        for priority, task in scored:
            if not task.targets:
                continue          # 无目标任务不参与重复判定
            key = (task.kind.value, tuple(task.targets))
            current = best_by_key.get(key)
            if current is None or priority > current[0]:
                best_by_key[key] = (priority, task.node_id)
        kept: List[Tuple[float, QueuedTask]] = []
        for row in scored:
            priority, task = row
            if not task.targets:
                kept.append(row)
                continue
            key = (task.kind.value, tuple(task.targets))
            winner = best_by_key.get(key)
            if winner is not None and winner[1] != task.node_id:
                decisions.append(SchedulingDecision(
                    task_id=task.task_id, node_id=task.node_id,
                    kind=task.kind.value,
                    decision=DECISION_SUPPRESSED_DUPLICATE_NODE,
                    priority=priority, policy=self.policy.value,
                    reasons=[
                        f"同一任务（kind={task.kind.value},"
                        f"targets={list(task.targets)}）本 tick 已由节点 "
                        f"{winner[1]} 承接；配置 "
                        "allow_multi_node_same_task=False 时只保留优先级"
                        "最高的节点",
                    ],
                    evidence={"winner_node": winner[1],
                              "winner_priority": round(winner[0], 6)}))
                continue
            kept.append(row)
        return kept

    def _select(
        self,
        candidates: List[QueuedTask],
        observation: CentralObservation,
        by_node: Dict[str, NodeObservation],
        now_s: float,
        node_order: List[str],
    ) -> "SelectionResult":
        """选择逻辑：**默认 = 打分排序 + 每节点限额截断**。

        三个规则基线只替换 `_priority`；优化参考（`optimization.py`）替换
        整个 `_select`。两条路径共用 `_classify_due` 与 `_duplicate_rule`，
        因此"谁可以被服务""什么算重复"完全一致，差别只在**挑哪一条**。
        """
        decisions: List[SchedulingDecision] = []

        # --- ② 打分 ---
        context: Dict[int, Tuple[int, List[str], Dict[str, Any], str]] = {}
        scored: List[Tuple[float, QueuedTask]] = []
        for order_index, task in enumerate(candidates):
            observation_for_node = by_node.get(task.node_id)
            urgency = task_urgency(task, observation_for_node)
            priority, reasons, evidence = self._priority(task, urgency, now_s)
            evidence["waiting_s"] = round(now_s - task.release_time_s, 6)
            evidence["node_rank"] = node_order.index(task.node_id) \
                if task.node_id in node_order else -1
            scored.append((priority, task))
            context[id(task)] = (order_index, reasons, evidence, task.node_id)

        # --- ③ 同一任务多节点规则（配置化，与优化参考共用）---
        scored = self._duplicate_rule(scored, decisions)

        # --- ④ 取计划：先按优先级排，再按"每节点每 tick 限额"截断 ---
        #
        # 排序只用 (优先级, 释放时刻, 入队序)：
        # **不再用 deadline 当并列时的次级键**——否则轮询基线的并列任务
        # 会被 deadline 排成 EDF 的样子，"轮询"就名不副实了
        # （实测过：轮询与 EDF 因此在同一条任务上选出相同结果）。
        order_of = {id(task): rank for rank, task in enumerate(candidates)}
        scored.sort(key=lambda row: (-row[0], row[1].release_time_s,
                                     order_of[id(row[1])]))
        per_node: Dict[str, int] = {}
        chosen: List[QueuedTask] = []
        for priority, task in scored:
            _index, reasons, evidence, node_id = context[id(task)]
            used = per_node.get(node_id, 0)
            if used >= self.config.max_tasks_per_node_per_tick:
                decisions.append(SchedulingDecision(
                    task_id=task.task_id, node_id=node_id,
                    kind=task.kind.value, decision=DECISION_DEFERRED,
                    priority=priority, policy=self.policy.value,
                    reasons=[
                        f"节点 {node_id} 本 tick 已排满"
                        f"（上限 {self.config.max_tasks_per_node_per_tick}）",
                        "一个节点在同一时刻只能执行一个任务",
                    ],
                    evidence=dict(evidence),
                    deferred_reason="node_quota"))
                continue
            if (self.config.max_tasks_per_tick is not None
                    and len(chosen) >= self.config.max_tasks_per_tick):
                decisions.append(SchedulingDecision(
                    task_id=task.task_id, node_id=node_id,
                    kind=task.kind.value, decision=DECISION_DEFERRED,
                    priority=priority, policy=self.policy.value,
                    reasons=[f"全局本 tick 任务上限 "
                             f"{self.config.max_tasks_per_tick} 已满"],
                    evidence=dict(evidence), deferred_reason="global_quota"))
                continue
            per_node[node_id] = used + 1
            chosen.append(task)
            decisions.append(SchedulingDecision(
                task_id=task.task_id, node_id=node_id, kind=task.kind.value,
                decision=DECISION_PLANNED, priority=priority,
                policy=self.policy.value, reasons=list(reasons),
                evidence=dict(evidence,
                              estimated_cost=task.estimated_cost_text(),
                              deadline_s=task.deadline_s)))
        return SelectionResult(chosen=chosen, decisions=decisions)

    # ------------------------------------------------------------------

    def _eligible(self, task: QueuedTask,
                  observation: Optional[NodeObservation]) -> Tuple[bool, str]:
        """节点是否**真的能**被服务（全部依据可见观测）。"""
        if observation is None:
            # 该节点本 tick 没有到达的摘要 → 调度器不敢动它
            return False, (f"节点 {task.node_id} 本 tick 没有到达的观测摘要，"
                           "调度器不依据缺失信息下计划")
        if not observation.available:
            return False, f"节点 {task.node_id} 观测显示为不可用"
        if observation.observed_at_s < task.release_time_s - 1e-9:
            return False, (f"观测时刻 {observation.observed_at_s:g}s 早于任务释放时刻 "
                           f"{task.release_time_s:g}s")
        remaining = observation.remaining
        for unit in (ResourceUnit.SAMPLE_SLOT, ResourceUnit.PROCESSING_OP,
                     ResourceUnit.COMM_BYTE):
            need = task.estimated_cost.get(unit, 0.0)
            if need > 0.0 and remaining.get(unit.value, 0.0) < need - 1e-9:
                return False, (f"节点 {task.node_id} 的 {unit.value} 余量 "
                               f"{remaining.get(unit.value, 0.0):g} 不足以覆盖"
                               f"预估成本 {need:g}")
        return True, ""


# ----------------------------------------------------------------------
# 三个基线（共用同一份观测/队列/执行器，差别只在排序与选择）
# ----------------------------------------------------------------------


class RoundRobinScheduler(SchedulerBase):
    """轮询调度：按节点轮流，选出各节点最早的 due 任务。

    它是**最弱的基线**：不看紧迫性、不看新鲜度，只保证"每个节点都轮得到"。
    用它当对照，才能说清"规则调度带来的分工差异"是不是来自规则本身。
    """

    policy = SchedulerPolicy.ROUND_ROBIN

    def _priority(self, task: QueuedTask, urgency: Dict[str, Any],
                  now_s: float) -> Tuple[float, List[str], Dict[str, Any]]:
        # 轮询不按任务内容打分：同一节点内按入队先后（释放时间）
        priority = -float(task.release_time_s)
        return priority, [
            "轮询基线：不评估任务紧迫性，按节点轮流 + 节点内先到先服务",
            f"释放时刻 {task.release_time_s:g}s（越早入队越优先）",
        ], {"policy_note": "round_robin"}


class EdfScheduler(SchedulerBase):
    """最早截止时间优先（EDF）。

    只看 `deadline_s`；没有截止时间的任务排在最后。
    """

    policy = SchedulerPolicy.EDF

    def _priority(self, task: QueuedTask, urgency: Dict[str, Any],
                  now_s: float) -> Tuple[float, List[str], Dict[str, Any]]:
        slack = (task.deadline_s - now_s) if task.deadline_s is not None else None
        priority = -slack if slack is not None else -1e9
        reasons = []
        if slack is None:
            reasons.append("该任务没有截止时间，排在所有有截止时间的任务之后")
        else:
            reasons.append(f"距截止时间还有 {slack:.2f}s（越小越优先）")
        reasons.append("EDF 只认截止时间，不评估数据新鲜度或估计质量")
        return priority, reasons, {
            "slack_s": None if slack is None else round(slack, 6),
            "deadline_s": task.deadline_s,
        }


class RuleScheduler(SchedulerBase):
    """规则调度：**等待时间 + 数据新鲜度 + 估计质量**（+ 教学任务等级）。

    打分（全部可观测、全部可解释）：

        priority = w_level   × task_level
                 + w_wait    × min(1, waiting / starvation_threshold)
                 + w_fresh   × min(1, age / max_information_age)
                 + w_quality × min(1, sigma / max_sigma_position)

    三项服务需求的分母都在配置里，因此"为什么这个任务优先"可以逐项算出来。
    **没有**军事目标价值、威胁度、对抗效能之类的项。
    """

    policy = SchedulerPolicy.RULE

    def _priority(self, task: QueuedTask, urgency: Dict[str, Any],
                  now_s: float) -> Tuple[float, List[str], Dict[str, Any]]:
        cfg = self.config
        level = cfg.task_levels[task.kind]
        waiting = max(0.0, now_s - task.release_time_s)
        wait_norm = min(1.0, waiting / cfg.starvation_threshold_s)
        age = urgency.get("information_age_s")
        sigma = urgency.get("sigma_position_m")
        age_norm = (min(1.0, age / cfg.max_information_age_s)
                    if age is not None else 0.0)
        sigma_norm = (min(1.0, sigma / cfg.max_sigma_position_m)
                      if sigma is not None else 0.0)
        priority = (cfg.weight_level * level
                    + cfg.weight_waiting * wait_norm
                    + cfg.weight_freshness * age_norm
                    + cfg.weight_quality * sigma_norm)

        reasons = [
            f"教学任务等级 {level}（{task.kind.value}）× 权重 "
            f"{cfg.weight_level:g} = {cfg.weight_level * level:.3f}",
            f"等待 {waiting:.2f}s → 归一化 {wait_norm:.3f}（阈值 "
            f"{cfg.starvation_threshold_s:g}s）× 权重 {cfg.weight_waiting:g}"
            f" = {cfg.weight_waiting * wait_norm:.3f}",
        ]
        if age is None:
            reasons.append("该任务不绑定航迹（无可观测的信息年龄项）")
        else:
            reasons.append(
                f"信息年龄 {age:.2f}s → 归一化 {age_norm:.3f}（阈值 "
                f"{cfg.max_information_age_s:g}s）× 权重 "
                f"{cfg.weight_freshness:g} = {cfg.weight_freshness * age_norm:.3f}")
        if sigma is None:
            reasons.append("无可见航迹协方差（估计质量项计 0）")
        else:
            reasons.append(
                f"位置标准差 {sigma:.1f}m → 归一化 {sigma_norm:.3f}（阈值 "
                f"{cfg.max_sigma_position_m:g}m）× 权重 "
                f"{cfg.weight_quality:g} = {cfg.weight_quality * sigma_norm:.3f}")
        return priority, reasons, {
            "task_level": level,
            "waiting_s": round(waiting, 6),
            "wait_norm": round(wait_norm, 6),
            "information_age_s": (None if age is None else round(age, 6)),
            "age_norm": round(age_norm, 6),
            "sigma_position_m": (None if sigma is None else round(sigma, 6)),
            "sigma_norm": round(sigma_norm, 6),
            "priority_total": round(priority, 6),
        }


def build_scheduler(policy: SchedulerPolicy,
                    config: Optional[SchedulingConfig] = None,
                    optimizer_config: Any = None) -> SchedulerBase:
    """按策略名构造调度器（**规则基线与优化参考同一入口**）。

    优化参考的两个策略在这里**惰性导入** `optimization.py`，避免
    `scheduling.py` 依赖它（依赖方向是单向的：优化层用规则层，反之不成立）。
    """
    table: Dict[SchedulerPolicy, Any] = {
        SchedulerPolicy.ROUND_ROBIN: RoundRobinScheduler,
        SchedulerPolicy.EDF: EdfScheduler,
        SchedulerPolicy.RULE: RuleScheduler,
    }
    if policy in OPTIMIZATION_POLICIES:
        from resource_management.optimization import build_optimizer
        return build_optimizer(policy, config, optimizer_config)
    if policy not in table:
        raise KeyError(
            f"未知策略 {policy!r}；规则基线={[p.value for p in BASELINE_POLICIES]}，"
            f"优化参考={[p.value for p in OPTIMIZATION_POLICIES]}")
    return table[policy](config)
