"""统一执行器：唯一全局时钟 + 先校验后执行 + 逐节点记账。

执行模型（两阶段，**顺序很重要**）
----------------------------------
```
submit(plan)
  ├─ 阶段 1  全量校验（只读，绝不改状态）
  │    ① 非法对象：未知节点 / 未知任务类型 / 未知单位 / 负成本 / 负时长
  │    ② 重复任务：计划内 task_id 重复、或去重键已成功执行过
  │    ③ 占用冲突：与既有占用区间或计划内其他任务重叠
  │    ④ 时间非法：start_s < 当前时钟；本版本要求 start_s == now（立即执行）
  │    ⑤ 资源不足：**按节点分别判定**（节点局部，不是计划级致命）
  │
  └─ 阶段 2  执行与记账（仅在**没有计划级致命问题**时进入）
       逐节点：立即任务 → 消耗；未来任务 → 预留；空闲 → 零成本零采样
       被拒任务 → 只记一条 `reject` 账本，**不扣任何单位**
```

两条必须分清的失败语义
----------------------
| 类别 | 例子 | 处置 |
| --- | --- | --- |
| **计划级致命** | 未知节点、重复 task_id、占用冲突、时间非法 | **整份计划不执行**，一个单位都不扣，状态零污染 |
| **节点局部** | 该节点采样时隙不够 | **只拒该节点的任务**，其余节点照常执行；不得因此结束整个网络任务 |

把两者混成一种，就会出现"因为 C 节点没配额，A/B 节点的采样也一起没了"——
这正是用户要求禁止的行为。

空闲语义（`TaskKind.IDLE`）
---------------------------
零成本、零时长占用、**不产生新的采样报告**；历史估计保留，
但其**信息年龄**随全局时钟增加（`estimate.age_of(now)`）。
执行器在空闲条目里记录当时的 `information_age_s`，便于追溯
"空闲期间信息旧了多少"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from resource_management.clock import GlobalClock
from resource_management.ledger import (
    ENTRY_ACTIVATE,
    ENTRY_CONSUME,
    ENTRY_IDLE,
    ENTRY_REJECT,
    ENTRY_RESERVE,
    ResourceLedger,
)
from resource_management.model import (
    Estimate,
    ExecutionPlan,
    ExecutionResult,
    NodeState,
    Outcome,
    PlanStatus,
    ProblemScope,
    TaskOutcome,
    TaskRequest,
    ValidationIssue,
)
from resource_management.units import (
    BUDGET_UNITS,
    TASK_KIND_CN,
    ResourceUnit,
    TaskKind,
    format_cost,
)

#: 拒绝原因码（账本与结果里都用它，便于机器统计）
REASON_UNKNOWN_NODE = "unknown_node"
REASON_NODE_UNAVAILABLE = "node_unavailable"
REASON_UNKNOWN_KIND = "unknown_task_kind"
REASON_ILLEGAL_COST = "illegal_cost"
REASON_ILLEGAL_DURATION = "illegal_duration"
REASON_ILLEGAL_TIME = "illegal_start_time"
REASON_DUPLICATE_TASK_IN_PLAN = "duplicate_task_in_plan"
REASON_DUPLICATE_SUBMISSION = "duplicate_submission"
REASON_OCCUPANCY_CONFLICT = "occupancy_conflict"
REASON_UPDATE_PERIOD = "update_period_not_reached"
REASON_INSUFFICIENT_RESOURCE = "insufficient_resource"

#: 具体到某个节点、由该节点自身状态导致的拒绝（节点局部）
NODE_LOCAL_REASONS: Tuple[str, ...] = (
    REASON_NODE_UNAVAILABLE,
    REASON_INSUFFICIENT_RESOURCE,
    REASON_UPDATE_PERIOD,
)
#: 计划级致命：出现任意一条就整份计划不执行
PLAN_FATAL_REASONS: Tuple[str, ...] = (
    REASON_UNKNOWN_NODE,
    REASON_UNKNOWN_KIND,
    REASON_ILLEGAL_COST,
    REASON_ILLEGAL_DURATION,
    REASON_ILLEGAL_TIME,
    REASON_DUPLICATE_TASK_IN_PLAN,
    REASON_DUPLICATE_SUBMISSION,
    REASON_OCCUPANCY_CONFLICT,
)


@dataclass
class _Reservation:
    """一份未来任务的预留（时钟推进到 `start_s` 时激活）。"""

    node_id: str
    task_id: str
    plan_id: str
    kind: TaskKind
    start_s: float
    end_s: float
    cost: Dict[ResourceUnit, float]


class UnifiedExecutor:
    """多节点统一执行器。**唯一**的时间来源是构造时传入的 `GlobalClock`。"""

    def __init__(
        self,
        clock: GlobalClock,
        ledger: Optional[ResourceLedger] = None,
    ) -> None:
        self.clock = clock
        self.ledger = ledger if ledger is not None else ResourceLedger()
        self.nodes: Dict[str, NodeState] = {}
        #: 已成功执行过的去重键（多次提交去重）
        self._executed_keys: Set[str] = set()
        self._reservations: List[_Reservation] = []
        self.plan_log: List[Dict[str, Any]] = []
        # 预留激活挂到时钟上：这是**唯一**允许在推进时产生副作用的通道
        self.clock.subscribe(self._on_clock_advance)

    # ------------------------------------------------------------------
    # 节点管理
    # ------------------------------------------------------------------

    def register_node(self, node: NodeState) -> NodeState:
        if node.node_id in self.nodes:
            raise ValueError(f"节点 {node.node_id} 已注册")
        node.ledger = self.ledger
        self.nodes[node.node_id] = node
        return node

    def node(self, node_id: str) -> Optional[NodeState]:
        return self.nodes.get(node_id)

    def set_availability(self, node_id: str, available: bool,
                         reason: str = "") -> None:
        """切换节点可用性。不可用**只影响该节点**，不影响其他节点。"""
        node = self.nodes[node_id]
        node.available = bool(available)
        node.unavailable_reason = "" if available else (reason or "未说明原因")
        self.ledger.record(
            time_s=self.clock.now_s, node_id=node_id, plan_id="-",
            task_id="-", kind=TaskKind.IDLE, entry_type=ENTRY_REJECT,
            remaining_after={unit: node.budget.remaining(unit)
                             for unit in BUDGET_UNITS},
            reason_code=REASON_NODE_UNAVAILABLE if not available else "",
            reason=(f"节点被置为不可用：{node.unavailable_reason}"
                    if not available else "节点恢复可用"),
        )

    # ------------------------------------------------------------------
    # 时钟推进
    # ------------------------------------------------------------------

    def advance(self, dt_s: float, reason: str = "") -> float:
        """推进唯一全局时钟。**每个 tick 只调用一次**，与节点数无关。"""
        return self.clock.advance(dt_s, reason)

    def _on_clock_advance(self, previous_s: float, now_s: float) -> None:
        """时钟推进回调：到期预留转消耗、清掉过期占用。

        ⚠️ 这里**只动节点自己的账**，不碰任何世界实体：
        没有目标、没有通信队列、没有别的实体被"顺便"推进一次。
        """
        due = [item for item in self._reservations if item.start_s <= now_s + 1e-12]
        self._reservations = [item for item in self._reservations
                              if item.start_s > now_s + 1e-12]
        for item in due:
            node = self.nodes.get(item.node_id)
            if node is None:
                continue
            node.budget.activate_reservation(item.cost)
            self.ledger.record(
                time_s=now_s, node_id=item.node_id, plan_id=item.plan_id,
                task_id=item.task_id, kind=item.kind, entry_type=ENTRY_ACTIVATE,
                delta=item.cost,
                remaining_after={unit: node.budget.remaining(unit)
                                 for unit in BUDGET_UNITS},
                reason_code="", reason="预留转为消耗（任务开始时刻到达）",
            )
        for node in self.nodes.values():
            node.prune_occupancy(now_s)

    # ------------------------------------------------------------------
    # 提交
    # ------------------------------------------------------------------

    def submit(self, plan: ExecutionPlan) -> ExecutionResult:
        """校验并执行一份计划（见模块文档的两阶段模型）。"""
        now = self.clock.now_s
        issues: List[ValidationIssue] = []

        # ---------- 阶段 1① 非法对象 ----------
        seen_task_ids: Dict[str, str] = {}
        for task in plan.tasks:
            if task.node_id not in self.nodes:
                issues.append(ValidationIssue(
                    ProblemScope.PLAN_FATAL, REASON_UNKNOWN_NODE,
                    task.node_id, task.task_id,
                    f"未知节点 {task.node_id!r}（已注册：{sorted(self.nodes)}）"))
                continue
            if not isinstance(task.kind, TaskKind):
                issues.append(ValidationIssue(
                    ProblemScope.PLAN_FATAL, REASON_UNKNOWN_KIND,
                    task.node_id, task.task_id,
                    f"未知任务类型 {task.kind!r}"))
                continue
            cost = task.effective_cost()
            bad_units = [unit.value for unit, value in cost.items()
                         if value < 0.0]
            if bad_units:
                issues.append(ValidationIssue(
                    ProblemScope.PLAN_FATAL, REASON_ILLEGAL_COST,
                    task.node_id, task.task_id,
                    f"成本不能为负：{bad_units}"))
            if task.effective_duration_s() < 0.0:
                issues.append(ValidationIssue(
                    ProblemScope.PLAN_FATAL, REASON_ILLEGAL_DURATION,
                    task.node_id, task.task_id,
                    f"时长不能为负：{task.effective_duration_s():g}s"))
            # 时间：本版本只支持立即执行（同一提交时刻），避免引入调度器语义
            if abs(task.start_s - now) > 1e-9:
                issues.append(ValidationIssue(
                    ProblemScope.PLAN_FATAL, REASON_ILLEGAL_TIME,
                    task.node_id, task.task_id,
                    f"start_s={task.start_s:g} 与当前时钟 {now:g} 不一致；"
                    "本版本只支持**立即执行**（同一时刻提交）"))
            # ---------- 阶段 1② 重复任务 ----------
            previous = seen_task_ids.get(task.task_id)
            if previous is not None:
                issues.append(ValidationIssue(
                    ProblemScope.PLAN_FATAL, REASON_DUPLICATE_TASK_IN_PLAN,
                    task.node_id, task.task_id,
                    f"计划内 task_id 重复（另一次在节点 {previous}）"))
            else:
                seen_task_ids[task.task_id] = task.node_id
            if task.dedup_key in self._executed_keys:
                issues.append(ValidationIssue(
                    ProblemScope.PLAN_FATAL, REASON_DUPLICATE_SUBMISSION,
                    task.node_id, task.task_id,
                    f"去重键 {task.dedup_key!r} 已成功执行过（重复提交）"))

        # ---------- 阶段 1③ 占用冲突（计划内 + 与既有占用） ----------
        intervals: Dict[str, List[Tuple[float, float, str]]] = {}
        for task in plan.tasks:
            node = self.nodes.get(task.node_id)
            if node is None or not isinstance(task.kind, TaskKind):
                continue
            start, end = task.start_s, task.end_s()
            existing = node.conflicting_interval(start, end)
            if existing is not None:
                issues.append(ValidationIssue(
                    ProblemScope.PLAN_FATAL, REASON_OCCUPANCY_CONFLICT,
                    task.node_id, task.task_id,
                    f"与既有占用 [{existing[0]:g},{existing[1]:g}) "
                    f"（任务 {existing[2]}）重叠"))
            for other_start, other_end, other_task in intervals.get(task.node_id, []):
                if start < other_end - 1e-12 and other_start < end - 1e-12:
                    issues.append(ValidationIssue(
                        ProblemScope.PLAN_FATAL, REASON_OCCUPANCY_CONFLICT,
                        task.node_id, task.task_id,
                        f"与本计划内任务 {other_task} 的占用区间重叠"))
            intervals.setdefault(task.node_id, []).append((start, end, task.task_id))
            # 更新周期：两次采样之间至少隔 update_period_s
            if (task.kind is TaskKind.SAMPLE and node.last_sample_s is not None
                    and task.start_s - node.last_sample_s
                    < node.update_period_s - 1e-9):
                issues.append(ValidationIssue(
                    ProblemScope.NODE_LOCAL, REASON_UPDATE_PERIOD,
                    task.node_id, task.task_id,
                    f"距上次采样仅 {task.start_s - node.last_sample_s:g}s，"
                    f"小于更新周期 {node.update_period_s:g}s"))

        # ---------- 阶段 1④ 节点可用性 + 资源是否足够（节点局部） ----------
        node_cost: Dict[str, Dict[ResourceUnit, float]] = {}
        for task in plan.tasks:
            if not isinstance(task.kind, TaskKind):
                continue
            cost = task.effective_cost()
            bucket = node_cost.setdefault(task.node_id, {unit: 0.0
                                                         for unit in BUDGET_UNITS})
            for unit in BUDGET_UNITS:
                bucket[unit] += cost.get(unit, 0.0)

        for node_id, total_cost in node_cost.items():
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if not node.available:
                for task in plan.tasks_of(node_id):
                    issues.append(ValidationIssue(
                        ProblemScope.NODE_LOCAL, REASON_NODE_UNAVAILABLE,
                        node_id, task.task_id,
                        f"节点不可用：{node.unavailable_reason or '未说明原因'}"))
                continue
            affordable, shortfalls = node.budget.can_afford(total_cost)
            if not affordable:
                for task in plan.tasks_of(node_id):
                    issues.append(ValidationIssue(
                        ProblemScope.NODE_LOCAL, REASON_INSUFFICIENT_RESOURCE,
                        node_id, task.task_id,
                        "资源不足：" + "；".join(shortfalls)))

        fatal = [issue for issue in issues
                 if issue.scope is ProblemScope.PLAN_FATAL]
        local = [issue for issue in issues
                 if issue.scope is ProblemScope.NODE_LOCAL]

        # ---------- 计划级致命：整份计划不执行，零扣费、零污染 ----------
        if fatal:
            result = ExecutionResult(
                plan_id=plan.plan_id, submit_time_s=now,
                status=PlanStatus.REJECTED, outcomes=[], issues=issues,
                note=("计划级致命错误 → **未执行任何任务、未扣任何单位**。"
                      f"原因：{fatal[0].detail}"),
            )
            self._log_plan(plan, result, applied_units=0)
            return result

        # ---------- 阶段 2：执行 + 逐节点记账 ----------
        rejected_by_task: Dict[str, ValidationIssue] = {}
        for issue in local:
            rejected_by_task.setdefault(issue.task_id, issue)

        outcomes: List[TaskOutcome] = []
        entry_ids: List[str] = []
        for task in plan.tasks:
            node = self.nodes[task.node_id]
            node.stats["submitted"] += 1
            issue = rejected_by_task.get(task.task_id)
            cost = task.effective_cost()
            if issue is not None:
                node.stats["rejected"] += 1
                entry = self.ledger.record(
                    time_s=now, node_id=task.node_id, plan_id=plan.plan_id,
                    task_id=task.task_id, kind=task.kind,
                    entry_type=ENTRY_REJECT,
                    remaining_after={unit: node.budget.remaining(unit)
                                     for unit in BUDGET_UNITS},
                    reason_code=issue.code, reason=issue.detail,
                    information_age_s=node.max_information_age_s(now),
                )
                entry_ids.append(entry.entry_id)
                outcomes.append(TaskOutcome(
                    task_id=task.task_id, node_id=task.node_id, kind=task.kind,
                    outcome=Outcome.REJECTED, reason_code=issue.code,
                    reason=issue.detail))
                continue

            produced = self._apply_task(node, task, plan, now, cost)
            entry_type = ENTRY_RESERVE if task.start_s > now + 1e-12 else (
                ENTRY_IDLE if task.kind is TaskKind.IDLE else ENTRY_CONSUME)
            entry = self.ledger.record(
                time_s=now, node_id=task.node_id, plan_id=plan.plan_id,
                task_id=task.task_id, kind=task.kind, entry_type=entry_type,
                delta=cost if entry_type in (ENTRY_CONSUME, ENTRY_RESERVE)
                else None,
                remaining_after={unit: node.budget.remaining(unit)
                                 for unit in BUDGET_UNITS},
                reason_code="", reason=self._apply_reason(task, produced),
                produced_samples=produced,
                information_age_s=node.max_information_age_s(now),
            )
            entry_ids.append(entry.entry_id)
            node.stats["applied"] += 1
            self._executed_keys.add(task.dedup_key)
            outcomes.append(TaskOutcome(
                task_id=task.task_id, node_id=task.node_id, kind=task.kind,
                outcome=Outcome.APPLIED, reason="已执行",
                consumed={unit.value: cost[unit] for unit in BUDGET_UNITS
                          if entry_type is ENTRY_CONSUME and cost[unit]},
                reserved={unit.value: cost[unit] for unit in BUDGET_UNITS
                          if entry_type is ENTRY_RESERVE and cost[unit]},
                produced_samples=produced,
            ))

        status = PlanStatus.PARTIAL if local else PlanStatus.APPLIED
        result = ExecutionResult(
            plan_id=plan.plan_id, submit_time_s=now, status=status,
            outcomes=outcomes, issues=issues, ledger_entry_ids=entry_ids,
            note=("部分节点任务被拒（节点局部原因），其余节点已执行"
                  if local else "全部任务已执行"),
        )
        self._log_plan(plan, result, applied_units=sum(
            value for outcome in outcomes for value in outcome.consumed.values()))
        return result

    # ------------------------------------------------------------------

    def _apply_task(self, node: NodeState, task: TaskRequest,
                    plan: ExecutionPlan, now: float,
                    cost: Dict[ResourceUnit, float]) -> int:
        """把任务落到节点状态上；返回产生的采样报告数。"""
        start, end = task.start_s, task.end_s()
        if task.start_s > now + 1e-12:
            node.budget.reserve(cost)
            self._reservations.append(_Reservation(
                node_id=node.node_id, task_id=task.task_id,
                plan_id=plan.plan_id, kind=task.kind,
                start_s=start, end_s=end, cost=dict(cost)))
            # 未来任务同样占用区间（否则同一时段会被重复承诺）
            node.add_occupancy(start, end, task.task_id)
            return 0

        node.budget.consume(cost)
        if end > start + 1e-12:
            node.add_occupancy(start, end, task.task_id)

        produced = 0
        if task.kind is TaskKind.SAMPLE:
            # 只有采样会产出报告并刷新历史估计
            node.last_sample_s = now
            for entity in (task.entities or ("unknown",)):
                node.estimates[entity] = Estimate(
                    entity_id=entity, range_m=float(len(entity) * 1000.0),
                    updated_at_s=now, source_task_id=task.task_id)
                produced += 1
        elif task.kind is TaskKind.IDLE:
            # 空闲：不产生任何采样报告；历史估计保留，年龄随时钟增长
            produced = 0
        return produced

    @staticmethod
    def _apply_reason(task: TaskRequest, produced: int) -> str:
        kind_cn = TASK_KIND_CN.get(task.kind, task.kind.value)
        if task.kind is TaskKind.IDLE:
            return (f"空闲：零成本、不产生采样报告；历史估计保留，"
                    f"信息年龄随时间增加")
        if task.kind is TaskKind.SAMPLE:
            return f"采样完成，产生 {produced} 条新报告"
        return f"{kind_cn}完成"

    def _log_plan(self, plan: ExecutionPlan, result: ExecutionResult,
                  applied_units: float) -> None:
        self.plan_log.append({
            "plan_id": plan.plan_id,
            "time_s": round(self.clock.now_s, 6),
            "n_tasks": len(plan.tasks),
            "nodes": plan.node_ids(),
            "status": result.status.value,
            "n_applied": result.n_applied,
            "n_rejected": result.n_rejected,
            "applied_units": round(float(applied_units), 9),
            "note": result.note,
        })

    # ------------------------------------------------------------------

    def conservation_report(self) -> Dict[str, Any]:
        """全节点资源守恒自检：每个节点每个单位残差都应为 0。"""
        report: Dict[str, Any] = {"nodes": {}, "all_conserved": True}
        for node_id, node in self.nodes.items():
            residual = node.budget.conservation_residual()
            conserved = node.budget.is_conserved()
            report["nodes"][node_id] = {
                "residual": residual, "conserved": conserved,
            }
            report["all_conserved"] = report["all_conserved"] and conserved
        return report

    def snapshot(self) -> Dict[str, Any]:
        """世界状态快照（用于"失败计划零污染"的逐位比对）。"""
        now = self.clock.now_s
        return {
            "now_s": now,
            "nodes": {node_id: node.to_dict(now)
                      for node_id, node in self.nodes.items()},
            "n_ledger_entries": len(self.ledger.entries),
            "n_executed_keys": len(self._executed_keys),
            "n_reservations": len(self._reservations),
        }
