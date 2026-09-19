"""统一数据结构：节点状态、资源预算、任务、计划、执行结果。

一个计划（`ExecutionPlan`）同时描述**多个节点**的采样/处理/共享/空闲任务；
执行器（`executor.py`）在**唯一全局时钟**的同一时刻校验并执行它。

三条不变量（由执行器强制，测试逐条钉住）
----------------------------------------
1. **资源守恒**：对每个预算单位都有
   `consumed + reserved + remaining == capacity`，且三者均非负；
2. **原子性**：*计划级致命错误*（非法对象 / 重复任务 / 占用冲突）时
   **一个单位都不扣**，状态快照逐位不变；
3. **节点局部性**：*节点级资源不足*只影响该节点，其余节点的任务照常执行，
   且**不得**因此结束整个网络任务。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from resource_management.units import (
    BUDGET_UNITS,
    DEFAULT_DURATION_S,
    TASK_KIND_CN,
    TEACHING_COST_MODEL,
    ResourceUnit,
    TaskKind,
    TeachingCost,
    format_cost,
)


# ----------------------------------------------------------------------
# 校验问题与结果状态
# ----------------------------------------------------------------------


class ProblemScope(str, Enum):
    """问题的处置范围——决定了"整个计划被拒"还是"只影响一个节点"。"""

    #: 计划级致命：非法对象 / 重复任务 / 占用冲突 → 整份计划不执行，零扣费
    PLAN_FATAL = "plan_fatal"
    #: 节点局部：该节点资源不足 → 只拒该节点的任务，其余节点继续
    NODE_LOCAL = "node_local"


class Outcome(str, Enum):
    APPLIED = "applied"
    REJECTED = "rejected"


class PlanStatus(str, Enum):
    #: 全部任务执行成功
    APPLIED = "applied"
    #: 计划级致命错误 → 一个单位都没扣
    REJECTED = "rejected"
    #: 部分节点任务被拒（节点局部原因），其余已执行
    PARTIAL = "partial"


@dataclass(frozen=True)
class ValidationIssue:
    """一条校验问题；`detail` 必须能回答"为什么没执行"。"""

    scope: ProblemScope
    code: str
    node_id: str
    task_id: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.value,
            "code": self.code,
            "node_id": self.node_id,
            "task_id": self.task_id,
            "detail": self.detail,
        }


# ----------------------------------------------------------------------
# 资源预算
# ----------------------------------------------------------------------


@dataclass
class ResourceBudget:
    """一个节点的三类预算：容量 / 已消耗 / 已预留。

    为什么要区分 `consumed` 与 `reserved`：任务有**占用区间**。
    `start_s == now` 的任务立即消耗；`start_s > now` 的任务先把成本
    **预留**下来（否则同一份预算会被两个未来任务重复承诺）。
    时钟推进到 `start_s` 时，预留转为消耗。
    """

    capacity: Dict[ResourceUnit, float] = field(default_factory=dict)
    consumed: Dict[ResourceUnit, float] = field(default_factory=dict)
    reserved: Dict[ResourceUnit, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for unit in BUDGET_UNITS:
            self.capacity.setdefault(unit, 0.0)
            self.consumed.setdefault(unit, 0.0)
            self.reserved.setdefault(unit, 0.0)

    # ------------------------------------------------------------------

    def remaining(self, unit: ResourceUnit) -> float:
        return float(self.capacity.get(unit, 0.0)) \
            - float(self.consumed.get(unit, 0.0)) \
            - float(self.reserved.get(unit, 0.0))

    def can_afford(self, cost: Mapping[ResourceUnit, float]) -> Tuple[bool, List[str]]:
        """能否负担该成本；返回 (是否够, 逐单位缺口说明)。"""
        shortfalls: List[str] = []
        for unit in BUDGET_UNITS:
            need = float(cost.get(unit, 0.0) or 0.0)
            if need <= 0.0:
                continue
            available = self.remaining(unit)
            if need > available + 1e-9:
                shortfalls.append(
                    f"{unit.value}：需要 {need:g}，可用 {available:g}"
                )
        return (not shortfalls), shortfalls

    def consume(self, cost: Mapping[ResourceUnit, float]) -> None:
        """立即消耗（只应由执行器在校验通过后调用）。"""
        for unit in BUDGET_UNITS:
            self.consumed[unit] = self.consumed.get(unit, 0.0) \
                + float(cost.get(unit, 0.0) or 0.0)

    def reserve(self, cost: Mapping[ResourceUnit, float]) -> None:
        for unit in BUDGET_UNITS:
            self.reserved[unit] = self.reserved.get(unit, 0.0) \
                + float(cost.get(unit, 0.0) or 0.0)

    def release_reservation(self, cost: Mapping[ResourceUnit, float]) -> None:
        for unit in BUDGET_UNITS:
            self.reserved[unit] = max(
                0.0, self.reserved.get(unit, 0.0)
                - float(cost.get(unit, 0.0) or 0.0))

    def activate_reservation(self, cost: Mapping[ResourceUnit, float]) -> None:
        """预留 → 消耗（时钟推进到任务开始时刻时调用）。"""
        self.release_reservation(cost)
        self.consume(cost)

    def conservation_residual(self) -> Dict[str, float]:
        """守恒残差：每个单位应为 0（容量 = 消耗 + 预留 + 剩余）。"""
        return {
            unit.value: round(
                self.capacity.get(unit, 0.0)
                - self.consumed.get(unit, 0.0)
                - self.reserved.get(unit, 0.0)
                - self.remaining(unit), 12)
            for unit in BUDGET_UNITS
        }

    def is_conserved(self, tol: float = 1e-9) -> bool:
        return all(abs(value) <= tol
                   for value in self.conservation_residual().values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            unit.value: {
                "capacity": round(self.capacity.get(unit, 0.0), 9),
                "consumed": round(self.consumed.get(unit, 0.0), 9),
                "reserved": round(self.reserved.get(unit, 0.0), 9),
                "remaining": round(self.remaining(unit), 9),
            }
            for unit in BUDGET_UNITS
        }


# ----------------------------------------------------------------------
# 节点状态
# ----------------------------------------------------------------------


@dataclass
class Estimate:
    """一个历史估计：**只有采样任务能刷新它**。

    空闲任务不产生采样报告，因此 `last_update_s` 不变，
    而 `age_of(now)` 会随时间增加——这正是"历史估计可以保留但信息会变旧"。
    """

    entity_id: str
    range_m: float
    updated_at_s: float
    source_task_id: str

    def age_of(self, now_s: float) -> float:
        return max(0.0, now_s - self.updated_at_s)

    def to_dict(self, now_s: float) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "range_m": round(self.range_m, 3),
            "updated_at_s": round(self.updated_at_s, 6),
            "age_s": round(self.age_of(now_s), 6),
            "source_task_id": self.source_task_id,
        }


@dataclass
class NodeState:
    """一个传感器节点的完整状态。

    | 字段 | 含义 |
    | --- | --- |
    | `available` | 节点可用性（不可用时不接受任何任务） |
    | `update_period_s` | 采样更新周期：两次采样之间至少要隔这么久 |
    | `occupancy` | 任务占用区间 `(start_s, end_s, task_id)` 列表 |
    | `budget` | 三类预算（容量/消耗/预留） |
    | `estimates` | 历史估计（带信息年龄，只由采样任务刷新） |
    | `ledger` | 本节点的资源账本（消耗/拒绝逐条记录） |
    """

    node_id: str
    available: bool = True
    update_period_s: float = 1.0
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    #: 占用区间；区间由执行器维护，允许查询"这个时段谁在占着"
    occupancy: List[Tuple[float, float, str]] = field(default_factory=list)
    estimates: Dict[str, Estimate] = field(default_factory=dict)
    #: 该节点最后一次采样时刻（用于更新周期判定）
    last_sample_s: Optional[float] = None
    #: 不可用原因（可追溯）
    unavailable_reason: str = ""
    ledger: Any = None          # 由 nodes.py 注入 ResourceLedger
    stats: Dict[str, int] = field(default_factory=lambda: {
        "submitted": 0, "applied": 0, "rejected": 0,
    })

    # ------------------------------------------------------------------

    @property
    def busy_until_s(self) -> float:
        return max((end for _start, end, _task in self.occupancy), default=0.0)

    def is_busy_at(self, time_s: float) -> bool:
        return any(start <= time_s < end
                   for start, end, _task in self.occupancy)

    def conflicting_interval(
        self, start_s: float, end_s: float
    ) -> Optional[Tuple[float, float, str]]:
        """与给定区间重叠的既有占用（无重叠返回 None）。

        区间按 `[start, end)` 处理：首尾相接不算冲突
        （否则"每 1 秒采样一次、每次占 1 秒"会永远冲突）。
        """
        for existing in self.occupancy:
            ex_start, ex_end, _task = existing
            if start_s < ex_end - 1e-12 and ex_start < end_s - 1e-12:
                return existing
        return None

    def add_occupancy(self, start_s: float, end_s: float, task_id: str) -> None:
        self.occupancy.append((float(start_s), float(end_s), task_id))
        self.occupancy.sort(key=lambda item: item[0])

    def prune_occupancy(self, now_s: float) -> None:
        """丢掉已经完全过去的占用区间（保持账本可读）。"""
        self.occupancy = [item for item in self.occupancy
                          if item[1] > now_s - 1e-12]

    def information_ages(self, now_s: float) -> Dict[str, float]:
        return {entity: estimate.age_of(now_s)
                for entity, estimate in self.estimates.items()}

    def max_information_age_s(self, now_s: float) -> float:
        ages = self.information_ages(now_s).values()
        return max(ages) if ages else 0.0

    def to_dict(self, now_s: float) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "update_period_s": self.update_period_s,
            "busy_until_s": round(self.busy_until_s, 6),
            "occupancy": [
                {"start_s": round(s, 6), "end_s": round(e, 6), "task_id": tid}
                for s, e, tid in self.occupancy
            ],
            "budget": self.budget.to_dict(),
            "estimates": [estimate.to_dict(now_s)
                          for estimate in self.estimates.values()],
            "max_information_age_s": round(self.max_information_age_s(now_s), 6),
            "stats": dict(self.stats),
        }


# ----------------------------------------------------------------------
# 任务 / 计划 / 结果
# ----------------------------------------------------------------------


@dataclass
class TaskRequest:
    """一个节点上的一次任务请求。

    字段
    ----
    `task_id`          计划内唯一；重复即非法对象（计划级致命）
    `node_id`          目标节点
    `kind`             采样 / 处理 / 共享 / 空闲
    `start_s`          开始时刻（必须 ≥ 当前时钟；等于则立即执行）
    `duration_s`       占用时长；默认取该类型的教学设定值
    `cost`             成本字典；留空则按 `TEACHING_COST_MODEL` 取
    `idempotency_key`  **去重键**：同一键只允许成功执行一次（重放会被拒）
    `entities`         该任务涉及的对象（采样/处理针对哪些实体）
    `note`             自由备注
    """

    task_id: str
    node_id: str
    kind: TaskKind
    start_s: float
    duration_s: Optional[float] = None
    cost: Optional[Dict[ResourceUnit, float]] = None
    idempotency_key: str = ""
    entities: Tuple[str, ...] = ()
    note: str = ""

    def effective_duration_s(self) -> float:
        if self.duration_s is not None:
            return float(self.duration_s)
        return float(DEFAULT_DURATION_S.get(self.kind, 0.0))

    def effective_cost(self) -> Dict[ResourceUnit, float]:
        if self.cost is not None:
            return {unit: float(self.cost.get(unit, 0.0) or 0.0)
                    for unit in ResourceUnit}
        model: TeachingCost = TEACHING_COST_MODEL.get(
            self.kind, TeachingCost())
        return model.as_dict()

    @property
    def dedup_key(self) -> str:
        """去重键：显式给了就用它，否则退化为 `node:task_id`。"""
        return self.idempotency_key or f"{self.node_id}:{self.task_id}"

    def end_s(self) -> float:
        return self.start_s + self.effective_duration_s()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "kind": self.kind.value,
            "kind_cn": TASK_KIND_CN.get(self.kind, self.kind.value),
            "start_s": round(self.start_s, 6),
            "duration_s": round(self.effective_duration_s(), 6),
            "end_s": round(self.end_s(), 6),
            "cost": {unit.value: value
                     for unit, value in self.effective_cost().items()},
            "cost_text": format_cost(self.effective_cost()),
            "idempotency_key": self.dedup_key,
            "entities": list(self.entities),
            "note": self.note,
        }


@dataclass
class ExecutionPlan:
    """一份**多节点**执行计划：同一个提交时刻，作用于多个节点。"""

    plan_id: str
    submit_time_s: float
    tasks: List[TaskRequest] = field(default_factory=list)
    note: str = ""

    def node_ids(self) -> List[str]:
        seen: List[str] = []
        for task in self.tasks:
            if task.node_id not in seen:
                seen.append(task.node_id)
        return seen

    def tasks_of(self, node_id: str) -> List[TaskRequest]:
        return [task for task in self.tasks if task.node_id == node_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "submit_time_s": round(self.submit_time_s, 6),
            "n_tasks": len(self.tasks),
            "nodes": self.node_ids(),
            "tasks": [task.to_dict() for task in self.tasks],
            "note": self.note,
        }


@dataclass
class TaskOutcome:
    """单个任务的执行结果——**"为什么没执行"就在这里**。"""

    task_id: str
    node_id: str
    kind: TaskKind
    outcome: Outcome
    reason_code: str = ""
    reason: str = ""
    consumed: Dict[str, float] = field(default_factory=dict)
    reserved: Dict[str, float] = field(default_factory=dict)
    produced_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "kind": self.kind.value,
            "outcome": self.outcome.value,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "consumed": dict(self.consumed),
            "reserved": dict(self.reserved),
            "produced_samples": self.produced_samples,
        }


@dataclass
class ExecutionResult:
    """一份计划的执行结果：状态 + 逐任务结果 + 逐节点账本增量。"""

    plan_id: str
    submit_time_s: float
    status: PlanStatus
    outcomes: List[TaskOutcome] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)
    ledger_entry_ids: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def n_applied(self) -> int:
        return sum(1 for item in self.outcomes
                   if item.outcome is Outcome.APPLIED)

    @property
    def n_rejected(self) -> int:
        return sum(1 for item in self.outcomes
                   if item.outcome is Outcome.REJECTED)

    def rejected_reasons(self) -> Dict[str, List[str]]:
        """节点 → 该节点被拒任务的原因（验收要求的"为什么没执行"）。"""
        grouped: Dict[str, List[str]] = {}
        for item in self.outcomes:
            if item.outcome is Outcome.REJECTED:
                grouped.setdefault(item.node_id, []).append(
                    f"{item.task_id}：{item.reason}")
        for issue in self.issues:
            if issue.scope is ProblemScope.NODE_LOCAL:
                continue
            grouped.setdefault(issue.node_id or "(计划)", []).append(
                f"{issue.task_id or '-'}：{issue.detail}")
        return grouped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "submit_time_s": round(self.submit_time_s, 6),
            "status": self.status.value,
            "n_tasks": len(self.outcomes),
            "n_applied": self.n_applied,
            "n_rejected": self.n_rejected,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "issues": [issue.to_dict() for issue in self.issues],
            "ledger_entry_ids": list(self.ledger_entry_ids),
            "note": self.note,
        }
