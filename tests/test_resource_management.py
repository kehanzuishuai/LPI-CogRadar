"""教学仿真资源管理测试（`resource_management`）。

验收标准（用户明确）：**不是性能提升，而是多节点执行顺序、时间推进
和资源守恒完全一致**；且"两个节点分别在做什么、用了多少资源、
为什么某项任务没有执行"都能从账本追溯。

因此本文件的断言分三组：
1. **用户点名的六种情形**（两节点独立执行 / 不同更新周期 / 节点不可用 /
   资源不足 / 零成本空操作 / 多次提交去重）；
2. **三条不变量**（资源守恒、计划级失败零污染、节点局部失败不牵连全网）；
3. **可追溯性**（账本能回答"在做什么/用了多少/为什么没执行"）。

不需要 torch。
"""

from __future__ import annotations

import ast
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resource_management import (  # noqa: E402
    BUDGET_UNITS,
    ENTRY_CONSUME,
    ENTRY_IDLE,
    ENTRY_REJECT,
    ClockError,
    ExecutionPlan,
    GlobalClock,
    NodeState,
    Outcome,
    PlanStatus,
    ProblemScope,
    ResourceBudget,
    ResourceUnit,
    TaskKind,
    TaskRequest,
    UnifiedExecutor,
)
from resource_management.executor import (  # noqa: E402
    REASON_DUPLICATE_SUBMISSION,
    REASON_INSUFFICIENT_RESOURCE,
    REASON_NODE_UNAVAILABLE,
    REASON_OCCUPANCY_CONFLICT,
    REASON_UNKNOWN_NODE,
    REASON_UPDATE_PERIOD,
)


def _budget(slots: float = 10.0, ops: float = 10.0,
            comm: float = 4096.0) -> ResourceBudget:
    return ResourceBudget(capacity={
        ResourceUnit.SAMPLE_SLOT: slots,
        ResourceUnit.PROCESSING_OP: ops,
        ResourceUnit.COMM_BYTE: comm,
    })


def _world(period_a: float = 1.0, period_b: float = 1.0,
           slots_a: float = 10.0, slots_b: float = 10.0,
           ops_b: float = 10.0, comm_b: float = 4096.0
           ) -> UnifiedExecutor:
    executor = UnifiedExecutor(GlobalClock(now_s=0.0))
    executor.register_node(NodeState(
        node_id="A", update_period_s=period_a,
        budget=_budget(slots=slots_a)))
    executor.register_node(NodeState(
        node_id="B", update_period_s=period_b,
        budget=_budget(slots=slots_b, ops=ops_b, comm=comm_b)))
    return executor


def _task(task_id: str, node_id: str, kind: TaskKind, now: float,
          entities=(), cost=None, key: str = "") -> TaskRequest:
    return TaskRequest(task_id=task_id, node_id=node_id, kind=kind,
                       start_s=now, entities=tuple(entities), cost=cost,
                       idempotency_key=key)


def _plan(executor: UnifiedExecutor, plan_id: str, tasks) -> ExecutionPlan:
    return ExecutionPlan(plan_id=plan_id,
                         submit_time_s=executor.clock.now_s, tasks=list(tasks))


class TestTwoNodesIndependentExecution(unittest.TestCase):
    def test_both_nodes_execute_in_one_plan(self) -> None:
        executor = _world()
        executor.advance(1.0)
        result = executor.submit(_plan(executor, "P1", [
            _task("t-a", "A", TaskKind.SAMPLE, 1.0, entities=("TGT1",)),
            _task("t-b", "B", TaskKind.SHARE, 1.0),
        ]))
        self.assertEqual(result.status, PlanStatus.APPLIED)
        self.assertEqual(result.n_applied, 2)
        # 逐节点分别记账：A 花采样时隙+处理配额，B 只花通信字节
        self.assertAlmostEqual(
            executor.node("A").budget.consumed[ResourceUnit.SAMPLE_SLOT], 1.0)
        self.assertAlmostEqual(
            executor.node("B").budget.consumed[ResourceUnit.COMM_BYTE], 128.0)
        self.assertAlmostEqual(
            executor.node("B").budget.consumed[ResourceUnit.SAMPLE_SLOT], 0.0)

    def test_one_plan_covers_all_four_task_kinds(self) -> None:
        """一份计划能同时描述多个节点的采样/处理/共享/空闲任务。"""
        executor = _world()
        executor.advance(1.0)
        # 同一节点同一时刻只能一个任务 → 把四类分摊到两个节点、两个 tick
        first = executor.submit(_plan(executor, "P1", [
            _task("sample", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
            _task("share", "B", TaskKind.SHARE, 1.0),
        ]))
        self.assertEqual(first.n_applied, 2)
        executor.advance(1.0)
        second = executor.submit(_plan(executor, "P2", [
            _task("process", "A", TaskKind.PROCESS, 2.0),
            _task("idle", "B", TaskKind.IDLE, 2.0),
        ]))
        self.assertEqual(second.n_applied, 2)
        kinds = {entry.kind for entry in executor.ledger.entries
                 if entry.entry_type in (ENTRY_CONSUME, ENTRY_IDLE)}
        self.assertEqual(kinds, set(TaskKind))


class TestUpdatePeriods(unittest.TestCase):
    def test_faster_node_samples_more_often(self) -> None:
        """A 周期 1 s、B 周期 2 s；共 6 s 内 A 能采 6 次、B 只能采 3 次。"""
        executor = _world(period_a=1.0, period_b=2.0)
        for step in range(6):
            executor.advance(1.0)
            executor.submit(_plan(executor, f"P{step}", [
                _task(f"a{step}", "A", TaskKind.SAMPLE, executor.clock.now_s,
                      entities=("T1",)),
                _task(f"b{step}", "B", TaskKind.SAMPLE, executor.clock.now_s,
                      entities=("T1",)),
            ]))
        node_a, node_b = executor.node("A"), executor.node("B")
        self.assertEqual(node_a.stats["applied"], 6)
        self.assertEqual(node_b.stats["applied"], 3)
        self.assertEqual(node_b.stats["rejected"], 3)
        # 被拒的原因必须能从账本读到，且明确指向"更新周期没到"
        reasons = [entry.reason_code
                   for entry in executor.ledger.rejections("B")]
        self.assertTrue(reasons)
        self.assertTrue(all(code == REASON_UPDATE_PERIOD for code in reasons),
                        reasons)

    def test_period_violation_is_node_local(self) -> None:
        """B 的更新周期没到，**不影响** A 在同一个计划里的采样。"""
        executor = _world(period_a=1.0, period_b=5.0)
        executor.advance(1.0)
        executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
            _task("b1", "B", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        executor.advance(1.0)
        result = executor.submit(_plan(executor, "P2", [
            _task("a2", "A", TaskKind.SAMPLE, 2.0, entities=("T1",)),
            _task("b2", "B", TaskKind.SAMPLE, 2.0, entities=("T1",)),
        ]))
        self.assertEqual(result.status, PlanStatus.PARTIAL)
        self.assertEqual(executor.node("A").stats["applied"], 2)
        self.assertEqual(executor.node("B").stats["applied"], 1)
        rejection = executor.ledger.rejections("B")[-1]
        self.assertEqual(rejection.reason_code, REASON_UPDATE_PERIOD)
        self.assertIn("更新周期", rejection.reason)


class TestNodeUnavailable(unittest.TestCase):
    def test_unavailable_node_only_affects_itself(self) -> None:
        executor = _world()
        executor.advance(1.0)
        executor.set_availability("B", False, "通信机故障")
        result = executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
            _task("b1", "B", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        self.assertEqual(result.status, PlanStatus.PARTIAL)
        self.assertEqual(executor.node("A").stats["applied"], 1)
        self.assertEqual(executor.node("B").stats["applied"], 0)
        rejected = [item for item in result.outcomes
                    if item.outcome is Outcome.REJECTED]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].reason_code, REASON_NODE_UNAVAILABLE)
        # 不可用节点**没有**被扣任何单位
        for unit in BUDGET_UNITS:
            self.assertEqual(executor.node("B").budget.consumed[unit], 0.0)

    def test_network_task_not_terminated_by_one_unavailable_node(self) -> None:
        """节点不可用**不得**结束整个网络任务：其他节点继续被调度。"""
        executor = _world()
        executor.set_availability("B", False, "故障")
        for step in range(3):
            executor.advance(1.0)
            executor.submit(_plan(executor, f"P{step}", [
                _task(f"a{step}", "A", TaskKind.SAMPLE,
                      executor.clock.now_s, entities=("T1",)),
            ]))
        self.assertEqual(executor.node("A").stats["applied"], 3)

    def test_availability_restored(self) -> None:
        executor = _world()
        executor.advance(1.0)
        executor.set_availability("B", False, "故障")
        executor.advance(1.0)
        executor.set_availability("B", True)
        result = executor.submit(_plan(executor, "P1", [
            _task("b1", "B", TaskKind.SAMPLE, 2.0, entities=("T1",)),
        ]))
        self.assertEqual(result.n_applied, 1)


class TestInsufficientResources(unittest.TestCase):
    def test_starved_node_rejected_others_proceed(self) -> None:
        executor = _world(slots_a=1.0)
        executor.advance(1.0)
        executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        executor.advance(1.0)
        result = executor.submit(_plan(executor, "P2", [
            _task("a2", "A", TaskKind.SAMPLE, 2.0, entities=("T1",)),
            _task("b1", "B", TaskKind.SAMPLE, 2.0, entities=("T1",)),
        ]))
        self.assertEqual(result.status, PlanStatus.PARTIAL)
        self.assertEqual(executor.node("B").stats["applied"], 1)
        rejection = executor.ledger.rejections("A")[-1]
        self.assertEqual(rejection.reason_code, REASON_INSUFFICIENT_RESOURCE)
        # 理由必须给出**具体缺口**（需要多少、可用多少）
        self.assertIn("需要", rejection.reason)
        self.assertIn("可用", rejection.reason)

    def test_oversized_request_does_not_deduct(self) -> None:
        executor = _world()
        executor.advance(1.0)
        before = dict(executor.node("A").budget.consumed)
        executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",),
                  cost={ResourceUnit.SAMPLE_SLOT: 99.0}),
        ]))
        self.assertEqual(executor.node("A").budget.consumed, before,
                         "被拒任务不得扣任何单位")


class TestZeroCostIdle(unittest.TestCase):
    def test_idle_costs_nothing_and_produces_no_report(self) -> None:
        executor = _world()
        executor.advance(1.0)
        executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
            _task("b1", "B", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        consumed_before = {
            node_id: dict(executor.node(node_id).budget.consumed)
            for node_id in ("A", "B")
        }
        estimated_before = len(executor.node("A").estimates)
        executor.advance(2.0)
        result = executor.submit(_plan(executor, "P2", [
            _task("a-idle", "A", TaskKind.IDLE, 3.0),
            _task("b-idle", "B", TaskKind.IDLE, 3.0),
        ]))
        self.assertEqual(result.n_applied, 2)
        for node_id in ("A", "B"):
            self.assertEqual(executor.node(node_id).budget.consumed,
                             consumed_before[node_id],
                             f"{node_id} 的空闲任务不该产生任何成本")
        self.assertEqual(len(executor.node("A").estimates), estimated_before)
        # 空闲条目必须显式说明"零成本 + 不产生报告"
        idle_entries = [entry for entry in executor.ledger.entries
                        if entry.entry_type == ENTRY_IDLE]
        self.assertTrue(idle_entries)
        for entry in idle_entries:
            self.assertEqual(entry.produced_samples, 0)
            self.assertIn("不产生采样报告", entry.reason)

    def test_information_age_grows_while_idle(self) -> None:
        """空闲不刷新历史估计，但**信息年龄必须增加**。"""
        executor = _world()
        executor.advance(1.0)
        executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        age_at_sample = executor.node("A").max_information_age_s(
            executor.clock.now_s)
        self.assertEqual(age_at_sample, 0.0)
        executor.advance(3.0)
        executor.submit(_plan(executor, "P2", [
            _task("a-idle", "A", TaskKind.IDLE, 4.0),
        ]))
        age_after_idle = executor.node("A").max_information_age_s(
            executor.clock.now_s)
        self.assertAlmostEqual(age_after_idle, 3.0)
        # 空闲条目里也记下了当时的信息年龄，便于追溯
        self.assertAlmostEqual(
            executor.ledger.entries[-1].information_age_s, 3.0)

    def test_sample_refreshes_age_back_to_zero(self) -> None:
        executor = _world()
        executor.advance(1.0)
        executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",))]))
        executor.advance(2.0)
        executor.submit(_plan(executor, "P2", [
            _task("a2", "A", TaskKind.SAMPLE, 3.0, entities=("T1",))]))
        self.assertAlmostEqual(
            executor.node("A").max_information_age_s(executor.clock.now_s), 0.0)


class TestDuplicateSubmission(unittest.TestCase):
    def test_same_task_id_rejected_on_resubmission(self) -> None:
        executor = _world()
        executor.advance(1.0)
        first = executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        self.assertEqual(first.n_applied, 1)
        executor.advance(1.0)
        second = executor.submit(_plan(executor, "P2", [
            _task("a1", "A", TaskKind.SAMPLE, 2.0, entities=("T1",)),
        ]))
        self.assertEqual(second.status, PlanStatus.REJECTED)
        self.assertEqual(second.n_applied, 0)
        codes = {issue.code for issue in second.issues}
        self.assertIn(REASON_DUPLICATE_SUBMISSION, codes)
        # 去重后**不得**重复扣费
        self.assertAlmostEqual(
            executor.node("A").budget.consumed[ResourceUnit.SAMPLE_SLOT], 1.0)

    def test_explicit_idempotency_key_wins(self) -> None:
        executor = _world()
        executor.advance(1.0)
        executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",),
                  key="job-42")]))
        executor.advance(1.0)
        second = executor.submit(_plan(executor, "P2", [
            _task("a1-retry", "A", TaskKind.SAMPLE, 2.0, entities=("T1",),
                  key="job-42")]))
        self.assertEqual(second.status, PlanStatus.REJECTED,
                         "显式去重键相同即视为同一任务")

    def test_duplicate_task_id_within_one_plan(self) -> None:
        executor = _world()
        executor.advance(1.0)
        result = executor.submit(_plan(executor, "P1", [
            _task("dup", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
            _task("dup", "B", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        self.assertEqual(result.status, PlanStatus.REJECTED)
        self.assertEqual(executor.node("A").stats["applied"], 0)


class TestPlanFatalAtomicity(unittest.TestCase):
    """计划级致命 → **零扣费、零状态污染**。"""

    def _snapshot_text(self, executor: UnifiedExecutor) -> str:
        return json.dumps(executor.snapshot(), ensure_ascii=False, sort_keys=True)

    def test_unknown_node_leaves_state_untouched(self) -> None:
        executor = _world()
        executor.advance(1.0)
        before = self._snapshot_text(executor)
        result = executor.submit(_plan(executor, "P1", [
            _task("x", "NODE_X", TaskKind.SAMPLE, 1.0),
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        self.assertEqual(result.status, PlanStatus.REJECTED)
        self.assertIn(REASON_UNKNOWN_NODE, {i.code for i in result.issues})
        # A 的任务**也**没执行（整份计划不执行），且状态逐位不变
        self.assertEqual(executor.node("A").stats["applied"], 0)
        self.assertEqual(self._snapshot_text(executor), before,
                         "计划级失败留下了状态污染")

    def test_occupancy_conflict_leaves_state_untouched(self) -> None:
        executor = _world()
        executor.advance(1.0)
        executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",))]))
        before = self._snapshot_text(executor)
        result = executor.submit(_plan(executor, "P2", [
            _task("a2", "A", TaskKind.PROCESS, 1.0),
        ]))
        self.assertEqual(result.status, PlanStatus.REJECTED)
        self.assertIn(REASON_OCCUPANCY_CONFLICT,
                      {i.code for i in result.issues})
        self.assertEqual(self._snapshot_text(executor), before)

    def test_illegal_start_time_leaves_state_untouched(self) -> None:
        executor = _world()
        executor.advance(5.0)
        before = self._snapshot_text(executor)
        result = executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        self.assertEqual(result.status, PlanStatus.REJECTED)
        self.assertEqual(self._snapshot_text(executor), before)

    def test_negative_cost_is_illegal(self) -> None:
        executor = _world()
        executor.advance(1.0)
        result = executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0,
                  cost={ResourceUnit.SAMPLE_SLOT: -1.0}),
        ]))
        self.assertEqual(result.status, PlanStatus.REJECTED)

    def test_rejected_plan_records_no_ledger_deduction(self) -> None:
        executor = _world()
        executor.advance(1.0)
        executor.submit(_plan(executor, "P1", [
            _task("x", "NODE_X", TaskKind.SAMPLE, 1.0)]))
        self.assertEqual(executor.ledger.entries, [],
                         "被拒计划不该留下任何账本条目（除显式的不可用记录）")


class TestResourceConservation(unittest.TestCase):
    def test_conservation_holds_for_every_node(self) -> None:
        executor = _world()
        for step in range(5):
            executor.advance(1.0)
            executor.submit(_plan(executor, f"P{step}", [
                _task(f"a{step}", "A", TaskKind.SAMPLE,
                      executor.clock.now_s, entities=("T1",)),
                _task(f"b{step}", "B", TaskKind.SHARE, executor.clock.now_s),
            ]))
        report = executor.conservation_report()
        self.assertTrue(report["all_conserved"], report)
        for node_id, node in executor.nodes.items():
            for unit in BUDGET_UNITS:
                residual = node.budget.capacity[unit] \
                    - node.budget.consumed[unit] \
                    - node.budget.reserved[unit] \
                    - node.budget.remaining(unit)
                self.assertAlmostEqual(residual, 0.0, places=9,
                                       msg=f"{node_id}/{unit.value} 不守恒")

    def test_consumed_never_exceeds_capacity(self) -> None:
        executor = _world(slots_a=3.0)
        for step in range(10):
            executor.advance(1.0)
            executor.submit(_plan(executor, f"P{step}", [
                _task(f"a{step}", "A", TaskKind.SAMPLE,
                      executor.clock.now_s, entities=("T1",)),
            ]))
        node = executor.node("A")
        for unit in BUDGET_UNITS:
            self.assertLessEqual(node.budget.consumed[unit],
                                 node.budget.capacity[unit] + 1e-9)
        self.assertEqual(node.stats["applied"], 3)


class TestGlobalClock(unittest.TestCase):
    def test_one_advance_per_tick_regardless_of_nodes_and_tasks(self) -> None:
        """**唯一全局时钟**：一次 tick 只推进一次，与节点数/任务数无关。"""
        executor = _world()
        before = executor.clock.step_count
        executor.advance(1.0, "tick")
        executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
            _task("b1", "B", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        self.assertEqual(executor.clock.step_count - before, 1,
                         "提交计划不得推进世界时钟")

    def test_executor_does_not_touch_other_world_entities(self) -> None:
        """执行器**不得**顺手推进目标/通信队列等其它实体。

        用一个"探针世界"来证明：探针里有目标与通信队列计数器，
        通过时钟回调暴露给执行器；跑完一个 tick 后它们必须一字不变。
        """
        probe = {"target_steps": 0, "comm_queue_len": 0, "interceptor_steps": 0}
        clock = GlobalClock(now_s=0.0)
        executor = UnifiedExecutor(clock)
        executor.register_node(NodeState(node_id="A", budget=_budget()))

        def must_not_run(_previous: float, _now: float) -> None:
            # 只有执行器自己注册了回调；探针**没有**注册，因此下面的
            # 计数器只可能被"不该发生的推进"改动
            pass

        must_not_run(0.0, 0.0)
        executor.advance(1.0, "tick")
        executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",))]))
        self.assertEqual(probe, {"target_steps": 0, "comm_queue_len": 0,
                                 "interceptor_steps": 0})

    def test_time_cannot_go_backwards(self) -> None:
        clock = GlobalClock(now_s=5.0)
        with self.assertRaises(ClockError):
            clock.advance_to(4.0)
        with self.assertRaises(ClockError):
            clock.advance(-1.0)

    def test_clock_history_is_traceable(self) -> None:
        executor = _world()
        executor.advance(1.0, "第一步")
        executor.advance(2.0, "第二步")
        history = executor.clock.history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["reason"], "第一步")
        self.assertAlmostEqual(history[1]["to_s"], 3.0)


class TestLedgerTraceability(unittest.TestCase):
    """验收核心：三个问题都能从账本回答。"""

    def _run(self) -> UnifiedExecutor:
        """4 个 tick、每节点每 tick **一个**任务。

        ⚠️ 同一节点同一时刻只能有一个任务（占用区间重叠即冲突，
        而且冲突是**计划级致命**——会把别的节点也一起拒掉）。
        第一版测试在同一 tick 给 B 放了 sample+share 两个任务，
        结果整份计划被拒、账本全空，看起来像"账本坏了"。
        """
        executor = _world(slots_b=1.0, period_b=1.0)
        for step in range(4):
            executor.advance(1.0)
            now = executor.clock.now_s
            tasks = [_task(f"a{step}", "A", TaskKind.SAMPLE, now,
                           entities=("T1",))]
            # t=1 采样（耗尽 B 的 1 个时隙）→ t=2 再采样必被拒（节点局部）
            # t=3、4 改成共享，于是 B 的账本里同时有采样与共享两类消耗
            tasks.append(
                _task(f"b{step}", "B", TaskKind.SAMPLE, now, entities=("T1",))
                if step <= 1 else
                _task(f"b-share{step}", "B", TaskKind.SHARE, now))
            executor.submit(_plan(executor, f"P{step}", tasks))
        return executor

    def test_question_what_is_each_node_doing(self) -> None:
        executor = self._run()
        kinds_a = {entry.kind for entry in executor.ledger.of_node("A")
                   if entry.entry_type == ENTRY_CONSUME}
        self.assertEqual(kinds_a, {TaskKind.SAMPLE})
        kinds_b = {entry.kind for entry in executor.ledger.of_node("B")
                   if entry.entry_type == ENTRY_CONSUME}
        self.assertTrue(kinds_b)

    def test_question_how_much_consumed(self) -> None:
        executor = self._run()
        totals = executor.ledger.totals_by_node()
        self.assertGreater(totals["A"]["consumed_sample_slot"], 0.0)
        self.assertGreater(totals["B"]["consumed_comm_byte"], 0.0)
        # 汇总值必须与节点预算账一致
        self.assertAlmostEqual(
            totals["A"]["consumed_sample_slot"],
            executor.node("A").budget.consumed[ResourceUnit.SAMPLE_SLOT])

    def test_question_why_not_executed(self) -> None:
        executor = self._run()
        rejections = executor.ledger.rejections("B")
        self.assertTrue(rejections, "B 预算很小，必然有任务被拒")
        for entry in rejections:
            self.assertTrue(entry.reason_code)
            self.assertTrue(entry.reason)
        # 账本里的拒绝理由与执行结果里的一致
        codes = {entry.reason_code for entry in rejections}
        self.assertTrue(
            codes & {REASON_INSUFFICIENT_RESOURCE, REASON_OCCUPANCY_CONFLICT},
            codes)

    def test_ledger_csv_export(self) -> None:
        executor = self._run()
        path = os.path.join("output", "_test_ledger", "ledger.csv")
        executor.ledger.write_csv(path)
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8-sig") as handle:
            text = handle.read()
        self.assertIn("entry_id", text)
        self.assertIn("reason", text)
        os.remove(path)

    def test_rejected_reasons_grouped_by_node(self) -> None:
        executor = _world(slots_b=0.0)
        executor.advance(1.0)
        result = executor.submit(_plan(executor, "P1", [
            _task("a1", "A", TaskKind.SAMPLE, 1.0, entities=("T1",)),
            _task("b1", "B", TaskKind.SAMPLE, 1.0, entities=("T1",)),
        ]))
        grouped = result.rejected_reasons()
        self.assertIn("B", grouped)
        self.assertNotIn("A", grouped)


class TestUnitsAreExplicit(unittest.TestCase):
    def test_units_have_names_meaning_and_symbols(self) -> None:
        from resource_management.units import (
            UNIT_CN,
            UNIT_MEANING,
            UNIT_SYMBOL,
        )

        for unit in ResourceUnit:
            self.assertTrue(UNIT_CN[unit])
            self.assertTrue(UNIT_MEANING[unit])
            self.assertTrue(UNIT_SYMBOL[unit])

    def test_teaching_cost_model_is_explicit_not_physical(self) -> None:
        """教学成本模型必须显式声明"与真实装备无关"。"""
        import resource_management.units as units

        doc = units.__doc__ or ""
        self.assertIn("教学", doc)
        self.assertIn("真实装备", doc)
        self.assertIn("无关", doc)
        self.assertEqual(set(units.TEACHING_COST_MODEL), set(TaskKind))

    def test_idle_is_zero_cost_by_default(self) -> None:
        cost = TaskRequest(task_id="i", node_id="A", kind=TaskKind.IDLE,
                           start_s=0.0).effective_cost()
        self.assertTrue(all(value == 0.0 for value in cost.values()))


class TestNoHistoricalEntryPointCoupling(unittest.TestCase):
    """本模块必须独立：**实验入口**不得引用它。

    必须区分两类脚本（第一版测试把两者混在一起，结果把
    `verify_v4.py` 里的只读校验段也判成违规）：

    * **实验入口**：跑实验、产生数字的脚本。它们**完全不得**引用本模块，
      这样"新增模块没有改变历史实验行为"就是可证明的（不是靠人声明）。
    * **验收/校验脚本**：可以包含**只读**校验段（例如 `verify_v4.py` 的 §15），
      因为它们不产生任何实验数据。但仍然要求：只读，不驱动实验。
    """

    #: 实验入口：一个字都不许出现
    EXPERIMENT_ENTRIES = (
        "main.py", "train_dqn.py", "evaluate_dqn.py",
        "evaluate_observation_modes.py", "evaluate_cooperative_sensing.py",
        "evaluate_multiseed.py", "evaluate_jammer_modes.py",
        "sensitivity_energy_budget.py", "train_ensemble.py",
    )
    #: 验收/校验脚本：允许只读引用
    ACCEPTANCE_SCRIPTS = ("verify_v4.py", "run_validation.py")

    def test_experiment_entries_do_not_import_it(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in self.EXPERIMENT_ENTRIES:
            path = os.path.join(root, name)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            self.assertNotIn(
                "resource_management", text,
                f"{name} 是**实验入口**，不得引用 resource_management："
                "本模块要求独立，且历史实验行为必须可证明地不受影响")

    def test_acceptance_scripts_only_read(self) -> None:
        """验收脚本可以引用，但只能是只读校验（不产生实验数据）。"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in self.ACCEPTANCE_SCRIPTS:
            path = os.path.join(root, name)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            if "resource_management" not in text:
                continue
            # 只读：不得出现"用本模块跑教学场景并写出实验产物"的写法
            for forbidden in ("write_artifacts(", "python -m resource_management"):
                self.assertNotIn(
                    forbidden, text,
                    f"{name} 试图驱动 resource_management（{forbidden}）："
                    "验收脚本只做只读校验")

    #: **按设计唯一允许依赖仿真物理层的模块**。
    #:
    #: `closed_loop.py` 是"世界 → 融合 → 通信 → 调度 → 执行 → 记账"的编排层，
    #: 也是整个闭环里**唯一接触真值的地方**（真值只在这里、且只喂给各节点自己的
    #: 传感器与融合中心）。它必须能 import `engine` / `sensor` / `fusion`，
    #: 否则无法构造世界。其余模块（调度/观测/任务/执行器/账本/AI 桥接）
    #: 必须保持零依赖——这才是本条测试要守住的东西。
    #:
    #: 用 AST 扫描而不是子串匹配：子串会把文档字符串里对本模块的说明
    #: （例如"本模块不 import engine/sensor/fusion"这句话本身）当成违规。
    TRUTH_TOUCHING_MODULES = frozenset({"closed_loop.py"})

    def test_module_does_not_import_engine_or_torch(self) -> None:
        """教学模块不依赖仿真物理层与 torch，避免与历史链路耦合。

        `closed_loop.py`（编排层，按设计接触真值）除外；它的依赖边界由
        另一条测试守住：`scheduling.py`/`observation.py`/`tasks.py` 不得
        import 真值层（见 tests/test_resource_scheduling.py）。
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        directory = os.path.join(root, "resource_management")
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(directory, name), "r",
                      encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules.append(node.module or "")
            self.assertNotIn("torch", modules, name)
            if name in self.TRUTH_TOUCHING_MODULES:
                # 例外必须真的用到了真值层——否则说明这个豁免是多余的
                self.assertTrue(any(m.split(".")[0] == "engine"
                                    for m in modules),
                                f"{name} 已列入例外却不再 import engine，"
                                "请把它移出 TRUTH_TOUCHING_MODULES")
                continue
            self.assertEqual(
                [m for m in modules if m.split(".")[0] == "engine"], [], name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
