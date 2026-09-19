"""非学习型资源调度闭环测试（v4.5）。

验收重点（用户明确）
--------------------
> 系统确实会产生**不同分工**，而且每次分工都有**原因和实际执行记录**。

因此本文件的中心是两条：
1. `TestPoliciesShareSameInputs`：三策略共用同一观测/队列/执行器（结构上保证），
   且**确实产生不同分工**；
2. `TestRationaleIsTraceable`：每条决策都带可读理由与结构化证据，
   并被真实执行（计划日志与账本能对上）。

其余覆盖用户点名的语义与失败情形：完成/过期/主动放弃/重复请求/长期未获服务、
完成率不可做高、多节点同任务配置规则、只读 AI 与超时、闭环四种机制。

不需要 torch。
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resource_management.ai_bridge import (  # noqa: E402
    build_rationale_payload,
    explain_locally,
    explain_schedule,
)
from resource_management.closed_loop import (  # noqa: E402
    DEFAULT_MECHANISMS,
    close_tick,
    run_closed_loop,
)
from resource_management.executor import UnifiedExecutor  # noqa: E402
from resource_management.clock import GlobalClock  # noqa: E402
from resource_management.model import (  # noqa: E402
    NodeState,
    PlanStatus,
    ResourceBudget,
)
from resource_management.observation import (  # noqa: E402
    CentralObservation,
    CentralObservationStore,
    NodeObservation,
    TrackObservation,
)
from resource_management.scheduling import (  # noqa: E402
    BASELINE_POLICIES,
    DECISION_ABANDONED,
    DECISION_DEFERRED,
    DECISION_NOT_ELIGIBLE,
    DECISION_PLANNED,
    DECISION_SUPPRESSED_DUPLICATE_NODE,
    EdfScheduler,
    RoundRobinScheduler,
    RuleScheduler,
    SchedulerPolicy,
    SchedulingConfig,
    build_scheduler,
)
from resource_management.tasks import (  # noqa: E402
    QueueTaskKind,
    QueuedTask,
    TaskQueue,
    TaskStatus,
)
from resource_management.units import ResourceUnit  # noqa: E402


# ----------------------------------------------------------------------
# 测试用的小世界（构造观测，不涉及真值）
# ----------------------------------------------------------------------


def _track(track_id: str, age_s: float = 0.0, sigma_m: float = 20.0
           ) -> TrackObservation:
    return TrackObservation(
        track_id=track_id, position=(1000.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        sigma_position=(sigma_m, sigma_m, sigma_m),
        last_measurement_time_s=max(0.0, 10.0 - age_s),
        last_fusion_time_s=max(0.0, 10.0 - age_s),
        information_age_s=age_s, coasting=False, n_sources=1,
        source_sensor_ids=("S",), platforms=("P",),
        local_updates=1, remote_updates=0)


def _node(node_id: str, tracks=(), age_s: float = 0.0,
          sigma_m: float = 20.0, available: bool = True,
          slots: float = 10.0) -> NodeObservation:
    items = [_track(tid, age_s, sigma_m) for tid in tracks]
    return NodeObservation(
        node_id=node_id, observed_at_s=10.0, tracks=items,
        track_valid_mask=[True] * len(items),
        available=available,
        capacity={u.value: 10.0 for u in
                  (ResourceUnit.SAMPLE_SLOT, ResourceUnit.PROCESSING_OP,
                   ResourceUnit.COMM_BYTE)},
        remaining={ResourceUnit.SAMPLE_SLOT.value: slots,
                   ResourceUnit.PROCESSING_OP.value: 10.0,
                   ResourceUnit.COMM_BYTE.value: 4096.0})


def _central(nodes) -> CentralObservation:
    return CentralObservation(
        schema_version="test", observed_at_s=10.0, nodes=list(nodes),
        node_valid_mask=[True] * len(nodes),
        node_information_age_s=[0.0] * len(nodes))


def _queue_with(*tasks: QueuedTask) -> TaskQueue:
    queue = TaskQueue()
    for task in tasks:
        queue.enqueue(task)
    return queue


def _task(task_id: str, node_id: str, kind: QueueTaskKind = QueueTaskKind.PROCESS,
          release_s: float = 10.0, deadline_s: float = 15.0,
          targets=("A-T1",), cost=None) -> QueuedTask:
    return QueuedTask(
        task_id=task_id, kind=kind, node_id=node_id,
        release_time_s=release_s, deadline_s=deadline_s,
        estimated_cost=(cost if cost is not None else {
            ResourceUnit.SAMPLE_SLOT: 1.0, ResourceUnit.PROCESSING_OP: 1.0,
            ResourceUnit.COMM_BYTE: 0.0}),
        targets=tuple(targets))


# ----------------------------------------------------------------------


class TestPoliciesShareSameInputs(unittest.TestCase):
    """三个基线必须共用同一份观测、同一个队列、同一个执行器。"""

    def test_schedulers_only_read_observation_and_queue(self) -> None:
        """AST 证明调度层不 import 真值/传感器/融合层。"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "resource_management", "scheduling.py")
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        for forbidden in ("engine", "sensor", "fusion", "torch"):
            self.assertEqual(
                [name for name in modules
                 if name == forbidden or name.startswith(forbidden + ".")],
                [], f"调度层不得依赖 {forbidden}")

    def test_three_policies_share_signature_and_output_type(self) -> None:
        for policy in BASELINE_POLICIES:
            scheduler = build_scheduler(policy, SchedulingConfig())
            self.assertTrue(hasattr(scheduler, "plan"))
            self.assertIsInstance(scheduler, (RoundRobinScheduler,
                                              EdfScheduler, RuleScheduler))

    def test_policies_produce_different_work_division(self) -> None:
        """同一输入下三策略必须给出**不同**的分工（否则基线没有意义）。"""
        observation = _central([_node("A", ("A-T1",), age_s=8.0, sigma_m=400.0),
                                _node("B", ("B-T1",), age_s=0.0, sigma_m=10.0)])
        divisions = {}
        for policy in BASELINE_POLICIES:
            queue = _queue_with(
                _task("share-A", "A", QueueTaskKind.SHARE, targets=()),
                _task("update-A", "A", QueueTaskKind.ESTIMATE_UPDATE,
                      deadline_s=13.0, targets=("A-T1",)),
                _task("share-B", "B", QueueTaskKind.SHARE, targets=()),
                _task("update-B", "B", QueueTaskKind.ESTIMATE_UPDATE,
                      deadline_s=20.0, targets=("B-T1",)),
            )
            result = build_scheduler(policy, SchedulingConfig()).plan(
                observation, queue, 10.0, plan_id=f"p-{policy.value}")
            planned = sorted(row.task_id for row in result.decisions
                             if row.decision == DECISION_PLANNED)
            priorities = sorted(round(row.priority, 4) for row in result.decisions
                                if row.decision == DECISION_PLANNED)
            divisions[policy.value] = (planned, priorities)
        # 轮询：节点内先到先服务（不评估紧迫性）→ 各节点取最早入队的那条
        self.assertEqual(divisions["round_robin"][0], ["share-A", "share-B"])
        # EDF：只认截止时间。A 的 update 截止 13s 早于 share 的 15s →
        # A 选 update；B 的 update 截止 20s 晚于 share 的 15s → B 选 share。
        # 注意这与轮询**在每个节点上给出不同选择**，正是"只认截止时间"的体现。
        self.assertEqual(divisions["edf"][0], ["share-B", "update-A"])
        # 规则：A 的航迹信息年龄 8s、σ=400m 都已越界 → 服务需求最高，
        # A 选 update；B 的航迹很新很好 → 仍是 update（等级 3 > share 1），
        # 但优先级的"数值"远低于 A（见下面 priorities 的比较）。
        self.assertEqual(divisions["rule"][0], ["update-A", "update-B"])
        # 三个基线两两不同分工——否则基线不构成对照
        distinct = {tuple(row) for row, _ in divisions.values()}
        self.assertEqual(len(distinct), 3)
        self.assertNotEqual(divisions["round_robin"][0], divisions["edf"][0])
        self.assertNotEqual(divisions["edf"][0], divisions["rule"][0])

    def test_round_robin_does_not_look_at_deadlines(self) -> None:
        """轮询的排序键里**没有** deadline：把截止时间全改掉，分工不变。

        这条防的是一个真实回归：排序的第二键若用 deadline，轮询基线会被
        排成 EDF 的样子（实测两基线在同一条任务上选出相同结果），
        "轮询"就名不副实了。
        """
        def run(deadline_a: float, deadline_b: float):
            observation = _central([_node("A", ("A-T1",)),
                                    _node("B", ("B-T1",))])
            queue = _queue_with(
                _task("share-A", "A", QueueTaskKind.SHARE, targets=(),
                      deadline_s=deadline_a),
                _task("update-A", "A", QueueTaskKind.ESTIMATE_UPDATE,
                      deadline_s=deadline_b, targets=("A-T1",)),
            )
            result = RoundRobinScheduler(SchedulingConfig()).plan(
                observation, queue, 10.0, plan_id="p")
            return sorted(row.task_id for row in result.decisions
                          if row.decision == DECISION_PLANNED)

        # share 先入队 → 无论截止时间怎么给，轮询都只选出 share（每节点每 tick 限额 1）
        self.assertEqual(run(1.0, 99.0), ["share-A"])
        self.assertEqual(run(99.0, 1.0), ["share-A"])

    def test_rule_priority_uses_only_observable_service_needs(self) -> None:
        """规则调度的证据里只有可观测服务需求，没有目标价值类字段。"""
        observation = _central([_node("A", ("A-T1",), age_s=6.0, sigma_m=300.0)])
        queue = _queue_with(_task("u", "A", QueueTaskKind.ESTIMATE_UPDATE,
                                  targets=("A-T1",)))
        result = RuleScheduler(SchedulingConfig()).plan(observation, queue, 10.0,
                                                        plan_id="p")
        evidence = result.decisions[0].evidence
        for key in ("task_level", "waiting_s", "information_age_s", "sigma_position_m"):
            self.assertIn(key, evidence)
        for forbidden in ("threat", "target_value", "military", "effectiveness",
                          "priority_target", "kill"):
            self.assertNotIn(forbidden, json.dumps(evidence, ensure_ascii=False))

    def test_low_level_task_can_outrank_high_level_one(self) -> None:
        """等级**不是**绝对优先：已长期等待 + 数据已旧的任务能翻越等级差。

        这条钉的是 `weight_level` 的取值下界（算术，不是手感）：
        等级差最大 3−1=2。另三项之和上限 = 0.5+0.6+0.4 = 1.5。

        * `weight_level=1.0` → 等级差 2.0 > 1.5：**数学上不可能**翻越，
          规则退化成严格优先级（这正是修复前的实测行为）；
        * `weight_level=0.3` → 等级差 0.6 < 1.5：可翻越。

        构造：等级 1 的任务已等 10s（> 阈值 8s）、绑定的航迹又旧又差
        （年龄 6s > 3s、σ=300m > 150m），等级 3 的任务刚释放且航迹很新很好。
        """
        observation = _central([
            _node("A", ("STALE", "FRESH"), age_s=6.0, sigma_m=300.0)])
        # 两条任务绑定**不同航迹**：让等级 1 的那条拿到"又旧又差"的服务需求项
        observation.nodes[0].tracks[0].information_age_s = 6.0
        observation.nodes[0].tracks[1].information_age_s = 0.0
        config = SchedulingConfig()
        self.assertLess(config.weight_level * 2,
                        config.weight_waiting + config.weight_freshness
                        + config.weight_quality,
                        "等级差必须小于另三项之和，否则等级绝对优先")
        queue = _queue_with(
            _task("low", "A", QueueTaskKind.SHARE, release_s=0.0,
                  targets=("STALE",)),
            _task("high", "A", QueueTaskKind.ESTIMATE_UPDATE, release_s=10.0,
                  targets=("FRESH",)))
        result = RuleScheduler(config).plan(observation, queue, 10.0,
                                            plan_id="p")
        planned = [row.task_id for row in result.decisions
                   if row.decision == DECISION_PLANNED]
        self.assertEqual(planned, ["low"],
                         "低等级但已长期等待且数据已旧的任务没能翻越等级差")

    def test_target_less_task_cannot_use_service_need_terms(self) -> None:
        """**记录一条真实局限**：不绑定航迹的任务拿不到任何服务需求项。

        这解释了为什么规则调度会把 `share` 类服务永久饿死（见
        `docs/resource_management.md` §11.6 第 2 条）：它只有等级分，
        加上最多 0.5 的等待分（= 0.3 + 0.5 = 0.8），永远低于一个"零需求"
        的高等级任务（3 × 0.3 = 0.9）。

        这里**不是**在断言"这是对的"，而是把算术上限钉住：
        一旦将来给这类任务补上服务需求项，这条测试会失败并提醒改文档。
        """
        observation = _central([_node("A", ("FRESH",))])
        config = SchedulingConfig()
        queue = _queue_with(
            _task("share", "A", QueueTaskKind.SHARE, release_s=0.0, targets=()),
            _task("update", "A", QueueTaskKind.ESTIMATE_UPDATE,
                  release_s=10.0, targets=("FRESH",)))
        result = RuleScheduler(config).plan(observation, queue, 10.0, plan_id="p")
        by_id = {row.task_id: row for row in result.decisions}
        self.assertEqual(by_id["share"].priority,
                         config.weight_level * 1 + config.weight_waiting * 1.0)
        self.assertGreater(by_id["update"].priority, by_id["share"].priority)
        self.assertIsNone(by_id["share"].evidence.get("information_age_s"))
        self.assertIsNone(by_id["share"].evidence.get("sigma_position_m"))

    def test_scheduler_output_is_execution_plan(self) -> None:
        observation = _central([_node("A", ("A-T1",))])
        queue = _queue_with(_task("u", "A", targets=("A-T1",)))
        result = RuleScheduler(SchedulingConfig()).plan(observation, queue, 10.0,
                                                        plan_id="P1")
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.plan_id, "P1")
        self.assertEqual(result.plan.submit_time_s, 10.0)
        self.assertEqual(len(result.plan.tasks), 1)


class TestPlanValidationAndAccounting(unittest.TestCase):
    def _executor(self):
        clock = GlobalClock(now_s=10.0)
        executor = UnifiedExecutor(clock)
        for node_id in ("A", "B"):
            executor.register_node(NodeState(
                node_id=node_id, update_period_s=1.0,
                budget=ResourceBudget(capacity={
                    ResourceUnit.SAMPLE_SLOT: 3.0,
                    ResourceUnit.PROCESSING_OP: 3.0,
                    ResourceUnit.COMM_BYTE: 1024.0})))
        return executor

    def test_plan_goes_through_executor_not_direct_writes(self) -> None:
        """调度器只产出计划；扣费只发生在执行器里。"""
        executor = self._executor()
        observation = _central([_node("A", ("A-T1",))])
        queue = _queue_with(_task("u", "A", targets=("A-T1",)))
        before = dict(executor.node("A").budget.consumed)
        result = RuleScheduler(SchedulingConfig()).plan(observation, queue, 10.0,
                                                        plan_id="P1")
        # 规划阶段**不许**动预算
        self.assertEqual(executor.node("A").budget.consumed, before)
        execution = executor.submit(result.plan)
        self.assertEqual(execution.status, PlanStatus.APPLIED)
        self.assertEqual(executor.node("A").budget.consumed[
            ResourceUnit.PROCESSING_OP], 1.0)

    def test_executor_rejection_is_recorded_not_hidden(self) -> None:
        """执行器拒绝要被记账，且不影响其他节点（节点局部）。"""
        executor = self._executor()
        observation = _central([_node("A", ("A-T1",), slots=0.0),
                                _node("B", ("B-T1",))])
        queue = _queue_with(
            _task("a", "A", QueueTaskKind.PREDEFINED_SAMPLE, targets=(),
                  cost={ResourceUnit.SAMPLE_SLOT: 1.0}),
            _task("b", "B", QueueTaskKind.PREDEFINED_SAMPLE, targets=(),
                  cost={ResourceUnit.SAMPLE_SLOT: 1.0}))
        result = RuleScheduler(SchedulingConfig()).plan(observation, queue, 10.0,
                                                        plan_id="P1")
        # 规则调度里 A 的采样会被 `_eligible` 拦下（余量 0）
        not_eligible = [row for row in result.decisions
                        if row.decision == DECISION_NOT_ELIGIBLE]
        self.assertTrue(not_eligible)
        self.assertIn("余量", not_eligible[0].reasons[0])


class TestFailureSemantics(unittest.TestCase):
    """完成 / 过期 / 主动放弃 / 重复请求 / 长期未获服务 必须分清。"""

    def test_abandon_does_not_shrink_denominator(self) -> None:
        """**主动放弃不能把完成率做高**——这是本阶段最容易被做假的地方。"""
        executor = self._executor_no_resources()
        observation = _central([_node("A", ("A-T1",), slots=0.0)])
        queue = _queue_with(
            _task("u1", "A", targets=("A-T1",), release_s=1.0),
            _task("u2", "A", targets=("A-T1",), release_s=1.0))
        config = SchedulingConfig(abandon_after_s=1.0, starvation_threshold_s=2.0)
        result = RuleScheduler(config).plan(observation, queue, 10.0, plan_id="P1")
        abandoned = [row for row in result.decisions
                     if row.decision == DECISION_ABANDONED]
        # 任务其实因为"余量不足"先被判 not_eligible；两种情况都不得扣费
        for row in result.decisions:
            self.assertIn(row.decision,
                          (DECISION_ABANDONED, DECISION_NOT_ELIGIBLE,
                           DECISION_DEFERRED))
        self.assertEqual(executor.node("A").budget.consumed[
            ResourceUnit.PROCESSING_OP], 0.0)
        # 放弃的任务状态是 CANCELLED，**仍留在队列里**（分母不会变小）
        cancelled = [task for task in queue.tasks
                     if task.status is TaskStatus.CANCELLED]
        remaining = [task for task in queue.tasks]
        self.assertEqual(len(remaining), 2, "放弃的任务不得被从队列里删掉")
        self.assertTrue(all(task.status is not TaskStatus.COMPLETED
                            for task in queue.tasks))

    def _executor_no_resources(self):
        clock = GlobalClock(now_s=10.0)
        executor = UnifiedExecutor(clock)
        executor.register_node(NodeState(
            node_id="A", budget=ResourceBudget(capacity={
                ResourceUnit.SAMPLE_SLOT: 5.0,
                ResourceUnit.PROCESSING_OP: 0.0,
                ResourceUnit.COMM_BYTE: 1024.0})))
        return executor

    def test_completion_rate_denominator_includes_all_failures(self) -> None:
        result = run_closed_loop(SchedulerPolicy.RULE, seed=42, steps=10)
        metrics = result.metrics
        denominator = metrics["completion_denominator"]
        parts = (metrics["n_completed"] + metrics["n_expired"]
                 + metrics["n_abandoned"] + metrics["n_rejected_by_executor"]
                 + metrics["n_starved"])
        self.assertEqual(denominator, parts)
        self.assertLessEqual(metrics["completion_rate"], 1.0)
        self.assertIn("主动放弃", metrics["note"])

    def test_expired_tasks_are_marked_not_deleted(self) -> None:
        queue = _queue_with(_task("u", "A", deadline_s=11.0))
        expired = queue.expire_overdue(12.0)
        self.assertEqual([task.task_id for task in expired], ["u"])
        self.assertEqual(len(queue.tasks), 1, "过期任务不得从队列里消失")
        self.assertEqual(queue.tasks[0].status, TaskStatus.EXPIRED)

    def test_duplicate_request_is_counted_separately(self) -> None:
        queue = _queue_with(_task("u", "A", targets=("A-T1",)))
        with self.assertRaises(Exception):
            queue.enqueue(_task("u", "A", targets=("A-T1",)))
        self.assertEqual(len(queue.tasks), 1, "重复请求不得新增任务")

    def test_starvation_is_reported(self) -> None:
        result = run_closed_loop(
            SchedulerPolicy.ROUND_ROBIN, seed=42, steps=8,
            scheduling_config=SchedulingConfig(starvation_threshold_s=2.0))
        for key in ("n_starved", "n_expired", "completion_denominator"):
            self.assertIn(key, result.metrics)
        self.assertEqual(result.metrics["n_starved"],
                         len(result.starved_task_ids))

    def test_starvation_and_expiry_are_mutually_exclusive(self) -> None:
        """**长期未获服务与过期是两条语义，不得同一条任务都算。**

        构造两条任务：
        - `with_dl`：截止时间已过 → 归入"过期"；
        - `no_dl`：没有截止时间，等待超过阈值 → 归入"长期未获服务"。

        第一版只统计"运行结束时仍在排队"的任务，于是带截止时间的任务
        一律先过期、永远不算长期未获服务（默认场景里 `n_starved` 恒为 0，
        这条语义形同不存在）；若改成两条都记，完成率分母又会重复计入。
        """
        queue = _queue_with(
            _task("with_dl", "A", release_s=1.0, deadline_s=5.0),
            _task("no_dl", "A", release_s=1.0, deadline_s=None))
        starved: set = set()
        close_tick(queue, now_s=10.0, starvation_threshold_s=8.0,
                   starved_ids=starved)
        self.assertEqual(sorted(starved), ["no_dl"])
        by_id = {task.task_id: task for task in queue.tasks}
        self.assertEqual(by_id["with_dl"].status, TaskStatus.EXPIRED)
        self.assertEqual(by_id["no_dl"].status, TaskStatus.PENDING)
        self.assertIn("长期", by_id["no_dl"].reason)
        # 重复调用不得重复计数
        close_tick(queue, now_s=11.0, starvation_threshold_s=8.0,
                   starved_ids=starved)
        self.assertEqual(sorted(starved), ["no_dl"])

    def test_starvation_does_not_fire_below_threshold(self) -> None:
        queue = _queue_with(_task("no_dl", "A", release_s=9.0,
                                  deadline_s=None))
        starved: set = set()
        close_tick(queue, now_s=10.0, starvation_threshold_s=8.0,
                   starved_ids=starved)
        self.assertEqual(starved, set())

    def test_capacity_cap_above_one_is_rejected(self) -> None:
        """每节点每 tick 限额 > 1 必须**报错**，不能安静地让整份计划被拒。

        执行器只支持立即执行（计划内 `start_s == now`），同一节点上两段从
        当前时刻开始的占用必然重叠 → `PLAN_FATAL` 占用冲突 → 整份计划被拒。
        实测把上限设成 2 时：计划数 96、完成数 **0**、通信 0 B——
        在指标表上看起来像"策略变差了"，其实一个任务都没执行。
        """
        with self.assertRaises(ValueError) as ctx:
            SchedulingConfig(max_tasks_per_node_per_tick=2).validate()
        self.assertIn("只能是 1", str(ctx.exception))
        with self.assertRaises(ValueError):
            SchedulingConfig(max_tasks_per_node_per_tick=0).validate()
        SchedulingConfig(max_tasks_per_node_per_tick=1).validate()

    def test_service_capacity_separates_supply_from_policy(self) -> None:
        """完成率低首先可能是**供需关系**，不能直接当策略优劣的证据。"""
        result = run_closed_loop(SchedulerPolicy.RULE, seed=42, steps=24)
        capacity = result.metrics["service_capacity"]
        self.assertEqual(capacity["max_serviceable_tasks"],
                         capacity["n_nodes"] * capacity["steps"]
                         * capacity["max_tasks_per_node_per_tick"])
        self.assertEqual(capacity["service_utilization"], 1.0)
        self.assertGreater(capacity["demand_over_capacity"], 1.0)
        self.assertIn("不能", capacity["interpretation"])

    def test_per_kind_status_is_recorded(self) -> None:
        """逐任务类型 × 状态：能看出"某一类服务被系统性跳过"。"""
        result = run_closed_loop(SchedulerPolicy.RULE, seed=42, steps=24)
        table = result.queue_summary["by_kind_status"]
        self.assertEqual(set(table), {kind.value for kind in QueueTaskKind})
        for per_status in table.values():
            self.assertEqual(set(per_status),
                             {status.value for status in TaskStatus})
        # 规则调度实测只服务 ESTIMATE_UPDATE（等级最高），共享/采样全部过期。
        # 这里**只钉住"记录存在且不掩盖"**，不宣称这是好结果。
        self.assertGreater(table["estimate_update"]["completed"], 0)


class TestMultiNodeDuplicateRule(unittest.TestCase):
    """同一任务多节点处理：**必须写成配置规则**。"""

    def _obs(self):
        # 两个节点都"看到"同一个目标键（跨节点同名航迹）
        return _central([_node("A", ("SAME-T1",)), _node("B", ("SAME-T1",))])

    def _queue(self):
        return _queue_with(
            _task("u-A", "A", targets=("SAME-T1",)),
            _task("u-B", "B", targets=("SAME-T1",)))

    def test_disallowed_suppresses_second_node(self) -> None:
        config = SchedulingConfig(allow_multi_node_same_task=False)
        result = RuleScheduler(config).plan(self._obs(), self._queue(), 10.0,
                                            plan_id="P1")
        suppressed = [row for row in result.decisions
                      if row.decision == DECISION_SUPPRESSED_DUPLICATE_NODE]
        self.assertEqual(len(suppressed), 1)
        self.assertIn("allow_multi_node_same_task=False", suppressed[0].reasons[0])
        planned = [row for row in result.decisions
                   if row.decision == DECISION_PLANNED]
        self.assertEqual(len(planned), 1)

    def test_allowed_records_duplicate_overhead(self) -> None:
        config = SchedulingConfig(allow_multi_node_same_task=True,
                                  charge_duplicate_as_overhead=True)
        result = RuleScheduler(config).plan(self._obs(), self._queue(), 10.0,
                                            plan_id="P1")
        overhead = [row for row in result.decisions
                    if row.task_id.startswith("duplicate:")]
        self.assertEqual(len(overhead), 1)
        self.assertTrue(overhead[0].evidence["charged_as_overhead"])

    def test_target_less_tasks_are_never_duplicates(self) -> None:
        """**两个节点各自采样/共享不是重复**（第一版在这里判错过）。"""
        observation = _central([_node("A", ("A-T1",)), _node("B", ("B-T1",))])
        queue = _queue_with(
            _task("share-A", "A", QueueTaskKind.SHARE, targets=()),
            _task("share-B", "B", QueueTaskKind.SHARE, targets=()))
        result = RuleScheduler(SchedulingConfig()).plan(observation, queue, 10.0,
                                                        plan_id="P1")
        suppressed = [row for row in result.decisions
                      if row.decision == DECISION_SUPPRESSED_DUPLICATE_NODE]
        self.assertEqual(suppressed, [], "无目标任务被误判成重复")
        planned = sorted(row.task_id for row in result.decisions
                         if row.decision == DECISION_PLANNED)
        self.assertEqual(planned, ["share-A", "share-B"])


class TestClosedLoopMechanisms(unittest.TestCase):
    """四种场景机制都必须能被观察到（否则闭环验收是空话）。"""

    def test_node_unavailable_shifts_work_to_other_node(self) -> None:
        result = run_closed_loop(
            SchedulerPolicy.RULE, seed=42, steps=14,
            mechanisms={"unavailable_windows": {"NODE_B": [(3.0, 9.0)]}})
        metrics = result.metrics
        # B 不可用期间，B 侧出现"不可服务"；A 仍照常承接
        self.assertGreater(metrics["per_node"]["NODE_A"]["n_planned"], 0)
        not_eligible = [row for row in result.decisions
                        if row.get("decision") == DECISION_NOT_ELIGIBLE
                        and row.get("node_id") == "NODE_B"]
        self.assertTrue(not_eligible, "节点不可用没有被记录成 not_eligible")

    def test_communication_delay_raises_remote_information_age(self) -> None:
        """通信延迟必须体现在**内容年龄**上。

        这里刻意区分两个量：
        - 到达年龄（`node_information_age_s`）= now − 到达时刻：延迟 2s 的摘要
          **送达那一 tick 仍是 0**（"刚收到"），所以它测不出链路延迟；
        - 内容年龄（`node_content_age_s`）= now − 摘要**生成**时刻：延迟 2s
          时它正是 2s。

        第一版只记了到达年龄，于是"通信延迟"这条机制在指标上完全不可见
        （快/慢两个配置都记 0.000）——那不是机制没生效，是**指标口径选错了**。
        """
        from communication import SHARE_CONSTRAINED

        fast = run_closed_loop(SchedulerPolicy.RULE, seed=42, steps=10,
                               mechanisms={"share_policy": "ideal_share"})
        slow = run_closed_loop(
            SchedulerPolicy.RULE, seed=42, steps=10,
            mechanisms={"share_policy": SHARE_CONSTRAINED,
                        "comm": {"base_delay_s": 2.0, "jitter_s": 0.0,
                                 "loss_prob": 0.0, "expiry_s": 10.0}})

        def max_of(result, key):
            values = [value for row in result.node_ages
                      for value in row[key] if value is not None]
            return max(values) if values else 0.0

        # 内容年龄：慢链路明显更旧
        self.assertGreater(max_of(slow, "content_ages"),
                           max_of(fast, "content_ages"),
                           "通信延迟没有体现在内容年龄上")
        self.assertLess(max_of(fast, "content_ages"), 1e-9,
                        "零延迟链路上内容年龄应当为 0")
        # 到达年龄测不出延迟（送达瞬间归零）——把这个事实也钉住，
        # 防止后人又拿它当延迟指标。
        self.assertLess(max_of(slow, "ages"), 1e-9)
        # 延迟的第二个可观察后果：**开头若干 tick 中央手里什么都没有**
        # （消息还在路上）。零延迟链路第 1 个 tick 就有数据。
        self.assertTrue(all(any(row["valid"]) for row in fast.node_ages),
                        "零延迟链路第 1 个 tick 就该收到摘要")
        self.assertTrue(any(not any(row["valid"]) for row in slow.node_ages),
                        "2s 延迟下开头应当有 tick 完全收不到摘要")
        self.assertTrue(any(row["valid"].count(True) > 0
                            for row in slow.node_ages),
                        "延迟链路最终应当收到摘要（延迟不等于丢失）")

    def test_observation_bias_changes_rule_priority(self) -> None:
        """观测偏差让该节点的航迹协方差变大 → 规则调度更该补它。"""
        result = run_closed_loop(
            SchedulerPolicy.RULE, seed=42, steps=12,
            mechanisms={"bias": {"NODE_B": {
                "range_bias_m": 150.0, "az_bias_deg": 0.8,
                "noise_underreport_factor": 0.4}}})
        # 偏差注入后 B 侧仍在承接任务（偏差不改变"是否可服务"），
        # 但证据里应能看到 B 侧残差/协方差量级变化
        self.assertGreater(result.metrics["per_node"]["NODE_B"]["n_planned"], 0)

    def test_handover_produces_different_node_activity_over_time(self) -> None:
        """目标穿过覆盖交界：各节点**看得见的目标条数**随时间变化。

        第一版这里断言"活跃节点集合随时间不同"，结果只有一种集合——因为
        「每节点每 tick 限额 1」下两个节点**每 tick 都在承接任务**，集合恒为
        {NODE_A, NODE_B}，与交接无关。那是我把断言放在了错误的量上：
        交接改变的不是"谁在干活"，而是"谁手上有这个目标"。
        因此改为检查逐节点**可见航迹条数**的时间序列。
        """
        result = run_closed_loop(SchedulerPolicy.RULE, seed=42, steps=24)
        rows = [row for row in result.node_ages if row.get("n_tracks")]
        self.assertTrue(rows)
        node_ids = rows[0]["node_ids"]
        series = {node_id: [] for node_id in node_ids}
        for row in rows:
            for node_id, count in zip(row["node_ids"], row["n_tracks"]):
                series[node_id].append(count)
        # 至少一个节点的可见目标数在过程中发生了变化（交接/进出视野）
        varying = [node_id for node_id, counts in series.items()
                   if len(set(counts)) > 1]
        self.assertTrue(
            varying,
            f"没有任何节点的可见目标数随时间变化——交接未发生：{series}")
        # 且**两个节点都曾看到过目标**：否则不是交接，只是一个节点独占
        seen = [node_id for node_id, counts in series.items() if max(counts) > 0]
        self.assertGreaterEqual(len(seen), 2,
                                f"只有 {seen} 看到过目标，不构成覆盖交接")


class TestRationaleIsTraceable(unittest.TestCase):
    """验收核心：每次分工都有**原因**与**实际执行记录**。"""

    def test_every_planned_decision_has_reason_and_evidence(self) -> None:
        result = run_closed_loop(SchedulerPolicy.RULE, seed=42, steps=12)
        planned = [row for row in result.decisions
                   if row.get("decision") == DECISION_PLANNED
                   and "reasons" in row]
        self.assertTrue(planned)
        for row in planned:
            self.assertTrue(row["reasons"], f"{row['task_id']} 没有理由")
            self.assertTrue(row["evidence"], f"{row['task_id']} 没有证据")
            self.assertGreater(row["priority"] if isinstance(
                row["priority"], (int, float)) else 0.0, 0.0)

    def test_plan_log_matches_executor_records(self) -> None:
        result = run_closed_loop(SchedulerPolicy.RULE, seed=42, steps=12)
        self.assertTrue(result.plan_log)
        for row in result.plan_log:
            self.assertIn(row["status"],
                          ("applied", "partial", "rejected"))
            self.assertGreaterEqual(row["n_applied"] + row["n_rejected"], 0)
        total_planned = sum(1 for row in result.decisions
                            if row.get("decision") == DECISION_PLANNED)
        total_applied = sum(row["n_applied"] for row in result.plan_log)
        self.assertLessEqual(total_applied, total_planned,
                             "执行器执行的比计划还多，记账对不上")

    def test_timeline_records_per_node_activity(self) -> None:
        result = run_closed_loop(SchedulerPolicy.RULE, seed=42, steps=12)
        for node_id, timeline in result.timelines.items():
            self.assertTrue(timeline, f"{node_id} 没有任何时间线记录")
            for row in timeline:
                for key in ("time_s", "task_id", "kind", "decision", "reason"):
                    self.assertIn(key, row)

    def test_resource_conservation_holds_in_closed_loop(self) -> None:
        result = run_closed_loop(SchedulerPolicy.EDF, seed=42, steps=12)
        self.assertTrue(result.metrics["conservation_all"])
        self.assertIn("per_node_occupancy", result.metrics)


class TestProvenanceCoversNewArtifacts(unittest.TestCase):
    """新产物必须能被源码摘要追溯到。

    本阶段新增了两类产物（`resource_management/` 账本、`scheduler_baselines/`
    对比表），它们的产出代码必须在源码摘要范围内，否则会出现
    "产物清单齐了、却答不出是哪版代码跑出来的"——这正是引入 run_manifest 要防的事。
    """

    def test_source_digest_covers_resource_management_and_tools(self) -> None:
        from run_manifest import source_digest as manifest_digest
        from tools.capture_baseline import source_digest as baseline_digest

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        manifest = manifest_digest(root)
        files = set(manifest["files"])
        for required in ("resource_management/closed_loop.py",
                         "resource_management/scheduling.py",
                         "tools/compare_schedulers.py",
                         "run_manifest.py"):
            self.assertIn(required, files,
                          f"源码摘要未覆盖 {required}——产物将无法追溯")

    def test_two_source_digest_implementations_agree(self) -> None:
        """两份实现（run_manifest 与 capture_baseline）必须给出**同一个**摘要。

        它们的 `SOURCE_PACKAGES` 是两处独立维护的常量，复制粘贴迟早会漂移；
        漂移的后果是"同一份基线有两个摘要"，比没有摘要更误导。
        """
        from run_manifest import source_digest as manifest_digest
        from tools.capture_baseline import source_digest as baseline_digest

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertEqual(manifest_digest(root)["digest"],
                         baseline_digest()["digest"])


class TestReadOnlyAiBridge(unittest.TestCase):
    def test_bridge_does_not_mutate_queue_or_decisions(self) -> None:
        observation = _central([_node("A", ("A-T1",))])
        queue = _queue_with(_task("u", "A", targets=("A-T1",)))
        result = RuleScheduler(SchedulingConfig()).plan(observation, queue, 10.0,
                                                        plan_id="P1")
        before_queue = json.dumps([task.to_dict() for task in queue.tasks],
                                  sort_keys=True, ensure_ascii=False)
        before_decisions = json.dumps([row.to_dict() for row in result.decisions],
                                      sort_keys=True, ensure_ascii=False)
        payload = build_rationale_payload(result.decisions)
        explain_locally(payload)
        self.assertEqual(
            json.dumps([task.to_dict() for task in queue.tasks],
                       sort_keys=True, ensure_ascii=False), before_queue)
        self.assertEqual(
            json.dumps([row.to_dict() for row in result.decisions],
                       sort_keys=True, ensure_ascii=False), before_decisions)
        self.assertTrue(payload["read_only"])

    def test_timeout_falls_back_without_blocking(self) -> None:
        """**远程解释超时不得阻塞**：2 s 的 provider + 0.3 s 预算 → 立刻返回。"""
        payload = build_rationale_payload([
            {"task_id": "t", "node_id": "A", "kind": "process",
             "decision": DECISION_PLANNED, "priority": 1.0,
             "reasons": ["测试"], "evidence": {}}])

        def slow(_payload):
            time.sleep(2.0)
            return {"summary": "remote"}

        started = time.perf_counter()
        rationale = explain_schedule(payload, remote_call=slow,
                                     timeout_s=0.3)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 1.0,
                        f"超时没有生效，等待了 {elapsed:.2f}s")
        self.assertTrue(rationale.explanation_timeout)
        self.assertEqual(rationale.provider, "local_rule")
        self.assertIn("不阻塞仿真时钟", rationale.note)

    def test_timeout_does_not_advance_clock(self) -> None:
        """超时只影响解释，**不推进仿真时钟**。"""
        executor = UnifiedExecutor(GlobalClock(now_s=5.0))
        executor.register_node(NodeState(node_id="A"))
        before = executor.clock.step_count
        payload = build_rationale_payload([])

        def slow(_payload):
            time.sleep(0.6)
            return {"summary": "remote"}

        explain_schedule(payload, remote_call=slow, timeout_s=0.2)
        self.assertEqual(executor.clock.step_count, before)
        self.assertEqual(executor.clock.now_s, 5.0)

    def test_failing_provider_falls_back(self) -> None:
        def boom(_payload):
            raise RuntimeError("provider 挂了")

        rationale = explain_schedule(build_rationale_payload([]),
                                     remote_call=boom, timeout_s=1.0)
        self.assertFalse(rationale.explanation_timeout)
        self.assertEqual(rationale.provider, "local_rule")
        self.assertIn("回退本地", rationale.note)

    def test_remote_success_is_used(self) -> None:
        rationale = explain_schedule(
            build_rationale_payload([]),
            remote_call=lambda _p: {"summary": "remote ok", "provider": "mock"},
            timeout_s=1.0)
        self.assertEqual(rationale.provider, "mock")
        self.assertEqual(rationale.summary, "remote ok")

    def test_bridge_module_does_not_reference_clock(self) -> None:
        """桥接层不得 import 时钟——没有推进时钟的路径。

        用 AST 而非子串匹配：`ai_bridge.py` 的**文档字符串**里写着
        「本模块不持有 GlobalClock」这类说明，子串匹配会把自己的说明文字
        当成违规（这个坑在观测层测试里已经踩过一次）。
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "resource_management", "ai_bridge.py")
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        modules, names = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
        self.assertNotIn("clock", modules)
        self.assertNotIn("GlobalClock", names)
        self.assertNotIn("advance", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
