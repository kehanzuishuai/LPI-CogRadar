"""非学习优化参考与阶段冻结的测试（v4.5 大阶段一收尾）。

验收重点
--------
1. **"精确最优"必须靠独立暴力枚举验证**：测试自己枚举全部可行计划、
   自己求最大目标值，再与优化器的结果比对。用优化器自己的搜索去"验证"
   优化器是自证，不算证据。
2. **`exact=True` 只在完全枚举且未触预算时出现**；任何提前退出都必须
   降级为"优化参考"或"预算耗尽"，且声明文本必须是规范文本的逐字渲染。
3. **信息不越权**：优化层不 import 真值层；预测模型不含未来故障一类信息；
   改掉未来不改变当前的规划（在 acceptance 里做运行时对照）。
4. **冻结契约**：口径摘要一致；改一处就会变。

不需要 torch。
"""

from __future__ import annotations

import ast
import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resource_management.acceptance import (  # noqa: E402
    check_information_boundary,
    check_multi_node_execution,
    check_plan_controlled_feedback,
    check_queue_traceability,
    check_reproducibility,
    check_resource_conservation,
    run_acceptance,
)
from resource_management.contract_v1 import (  # noqa: E402
    CONTRACT_VERSION,
    FROZEN_DIGEST,
    ContractFrozenError,
    canonical_json,
    contract_digest,
    contract_snapshot,
    verify_frozen,
)
from resource_management.observation import (  # noqa: E402
    CentralObservation,
    NodeObservation,
    TrackObservation,
)
from resource_management.optimization import (  # noqa: E402
    CANONICAL_CLAIMS,
    CLAIM_KIND_BUDGET,
    CLAIM_KIND_EXACT,
    CLAIM_KIND_LOOKAHEAD,
    ComputeBudget,
    ConsequenceModel,
    EnumerationOptimizer,
    EvaluationVector,
    METRIC_KEYS,
    ObjectiveSpec,
    OptimizationScheduler,
    OptimizerConfig,
    OptimizerKind,
    PREDICTION_ASSUMPTIONS,
    PredictionModel,
    RollingHorizonOptimizer,
    build_optimizer,
    default_optimizer_config,
    pareto_frontier,
)
from resource_management.scheduling import (  # noqa: E402
    BASELINE_POLICIES,
    DECISION_NOT_ELIGIBLE,
    DECISION_PLANNED,
    OPTIMIZATION_POLICIES,
    SchedulerPolicy,
    SchedulingConfig,
    build_scheduler,
)
from resource_management.tasks import (  # noqa: E402
    QueueTaskKind,
    QueuedTask,
    TaskQueue,
)
from resource_management.units import (  # noqa: E402
    ResourceUnit,
    TEACHING_COST_MODEL,
)


# ----------------------------------------------------------------------
# 测试用的小世界
# ----------------------------------------------------------------------


def _track(track_id: str, age_s: float, sigma_m: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id, position=(1000.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        sigma_position=(sigma_m, sigma_m, sigma_m),
        last_measurement_time_s=max(0.0, 10.0 - age_s),
        last_fusion_time_s=max(0.0, 10.0 - age_s),
        information_age_s=age_s, coasting=False, n_sources=1,
        source_sensor_ids=("S",), platforms=("P",),
        local_updates=1, remote_updates=0)


def _node(node_id: str, tracks, remaining=None) -> NodeObservation:
    remaining = remaining or {
        ResourceUnit.SAMPLE_SLOT.value: 10.0,
        ResourceUnit.PROCESSING_OP.value: 10.0,
        ResourceUnit.COMM_BYTE.value: 4096.0}
    return NodeObservation(
        node_id=node_id, observed_at_s=10.0, tracks=list(tracks),
        track_valid_mask=[True] * len(tracks),
        available=True,
        capacity={ResourceUnit.SAMPLE_SLOT.value: 10.0,
                  ResourceUnit.PROCESSING_OP.value: 10.0,
                  ResourceUnit.COMM_BYTE.value: 4096.0},
        remaining=dict(remaining))


def _central(nodes) -> CentralObservation:
    return CentralObservation(
        schema_version="test", observed_at_s=10.0, nodes=list(nodes),
        node_valid_mask=[True] * len(nodes),
        node_information_age_s=[0.0] * len(nodes))


def _task(task_id: str, node_id: str, kind=QueueTaskKind.ESTIMATE_UPDATE,
          targets=("T1",), deadline_s: float = 15.0,
          release_s: float = 10.0) -> QueuedTask:
    return QueuedTask(
        task_id=task_id, kind=kind, node_id=node_id,
        release_time_s=release_s, deadline_s=deadline_s,
        estimated_cost=TEACHING_COST_MODEL[
            {"predefined_sample": "sample", "estimate_update": "process",
             "process": "process", "share": "share"}[kind.value]].as_dict(),
        targets=tuple(targets))


def _queue(*tasks) -> TaskQueue:
    queue = TaskQueue()
    for task in tasks:
        queue.enqueue(task)
    return queue


# ----------------------------------------------------------------------
# ① 预测模型与后果模型
# ----------------------------------------------------------------------


class TestPredictionModel(unittest.TestCase):
    def test_age_and_sigma_growth_are_monotone(self) -> None:
        model = PredictionModel(horizon_ticks=3)
        self.assertAlmostEqual(model.age_after(2.0, 3), 5.0)
        previous = model.sigma_after(20.0, 0)
        for ticks in range(1, 5):
            current = model.sigma_after(20.0, ticks)
            self.assertGreater(current, previous)
            previous = current
        self.assertAlmostEqual(model.sigma_after(20.0, 0), 20.0)

    def test_model_declares_no_future_events(self) -> None:
        """模型必须**显式声明**它不含未来故障信息（用户点名要求）。"""
        names = {item.name for item in PREDICTION_ASSUMPTIONS}
        self.assertIn("no_future_events", names)
        describe = PredictionModel().describe()
        self.assertIn("不读未来故障", describe["information_boundary"])
        text = canonical_json(describe)
        self.assertIn("unavailable(t + k) = []", text)

    def test_optimization_layer_does_not_import_truth_layers(self) -> None:
        """AST 证明优化层不 import 真值/传感器/融合层。"""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "resource_management",
            "optimization.py")
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
                [m for m in modules
                 if m == forbidden or m.startswith(forbidden + ".")],
                [], f"优化层不得依赖 {forbidden}")


class TestConsequenceModel(unittest.TestCase):
    def _setup(self):
        node = _node("A", [_track("T1", age_s=6.0, sigma_m=300.0)])
        by_node = {"A": node}
        candidates = [_task("upd", "A")]
        model = ConsequenceModel(prediction=PredictionModel(),
                                 objective=ObjectiveSpec(), horizon_ticks=1)
        return model, candidates, by_node

    def test_serving_a_stale_track_improves_predicted_quality(self) -> None:
        """性质检查：服务一条又旧又差的航迹，预测估计质量必须变好。"""
        model, candidates, by_node = self._setup()
        idle = model.evaluate([], candidates, by_node, 10.0)
        served = model.evaluate(candidates, candidates, by_node, 10.0)
        self.assertGreater(served.values["estimate_quality"],
                           idle.values["estimate_quality"])
        self.assertGreater(served.values["service_completion"],
                           idle.values["service_completion"])

    def test_estimate_quality_is_horizon_mean_not_terminal(self) -> None:
        """**期末效应回归**：估计质量取时域平均，不取末端值。

        第一版只取末端值，导致"晚一个 tick 再刷新"在末端更新鲜，
        模型于是在奖励拖延（实测束搜索里"什么都不做"压过合法的更新任务）。
        取平均后，早刷新严格不差于晚刷新。
        """
        node = _node("A", [_track("T1", age_s=3.0, sigma_m=100.0)])
        model = ConsequenceModel(prediction=PredictionModel(horizon_ticks=3),
                                 objective=ObjectiveSpec(), horizon_ticks=3)
        candidates = [_task("upd", "A")]
        early = model.evaluate(candidates, candidates, {"A": node}, 10.0)
        self.assertEqual(early.detail["estimate_quality_scope"], "horizon_mean")
        self.assertIn("terminal_mean_age_s", early.detail)

    def test_completion_scope_is_deadline_within_horizon(self) -> None:
        """完成度分母只算时域内可判定去留的任务（否则对选择不敏感）。"""
        node = _node("A", [_track("T1", age_s=6.0, sigma_m=300.0)])
        model = ConsequenceModel(prediction=PredictionModel(), objective=ObjectiveSpec())
        near = _task("near", "A", deadline_s=10.5)
        far = _task("far", "A", deadline_s=40.0)
        vector = model.evaluate([], [near, far], {"A": node}, 10.0)
        self.assertEqual(vector.detail["completion_scope"],
                         "deadline_within_horizon")
        self.assertEqual(vector.detail["n_resolvable"], 1)

    def test_resource_shortage_makes_plan_infeasible(self) -> None:
        node = _node("A", [_track("T1", age_s=6.0, sigma_m=300.0)],
                     remaining={ResourceUnit.SAMPLE_SLOT.value: 0.0,
                                ResourceUnit.PROCESSING_OP.value: 0.0,
                                ResourceUnit.COMM_BYTE.value: 0.0})
        optimizer = EnumerationOptimizer()
        ok, why = optimizer._affordable([_task("upd", "A")], {"A": node})
        self.assertFalse(ok)
        self.assertTrue(any("可用" in item for item in why))


# ----------------------------------------------------------------------
# ② 精确性：靠**独立暴力枚举**验证
# ----------------------------------------------------------------------


class TestExactnessAgainstBruteForce(unittest.TestCase):
    """测试自己枚举、自己求最大，再与优化器的结果比对。"""

    def _world(self):
        nodes = [
            _node("A", [_track("T1", age_s=8.0, sigma_m=400.0),
                        _track("T2", age_s=1.0, sigma_m=30.0)]),
            _node("B", [_track("T3", age_s=2.0, sigma_m=90.0)]),
        ]
        by_node = {node.node_id: node for node in nodes}
        candidates = [
            _task("a-upd-T1", "A", targets=("T1",), deadline_s=11.0),
            _task("a-upd-T2", "A", targets=("T2",), deadline_s=16.0),
            _task("a-share", "A", kind=QueueTaskKind.SHARE, targets=()),
            _task("b-upd-T3", "B", targets=("T3",), deadline_s=12.0),
            _task("b-upd-T3b", "B", targets=("T3",), deadline_s=18.0),
        ]
        return _central(nodes), by_node, candidates

    def test_enumerated_optimum_equals_brute_force(self) -> None:
        observation, by_node, candidates = self._world()
        queue = _queue(*candidates)
        config = OptimizerConfig(kind=OptimizerKind.ENUMERATION,
                                 horizon_ticks=1, compute_pareto=True)
        optimizer = EnumerationOptimizer(optimizer_config=config)
        result = optimizer.plan(observation, queue, 10.0, plan_id="P1")
        outcome = optimizer.last_outcome

        # --- 独立暴力枚举：不复用优化器的搜索，只用同一个后果模型 ---
        by_node_tasks = {}
        for task in candidates:
            by_node_tasks.setdefault(task.node_id, []).append(task)
        node_ids = sorted(by_node_tasks)
        best_score, best_plan = None, None
        n_total = 0
        options = [[None] + sorted(by_node_tasks[node_id],
                                   key=lambda t: t.task_id)
                   for node_id in node_ids]
        for combination in itertools.product(*options):
            n_total += 1
            pick = [task for task in combination if task is not None]
            vector = optimizer.consequence.evaluate(pick, candidates, by_node,
                                                    10.0)
            score = optimizer.optimizer_config.objective.score(vector)
            ids = tuple(sorted(task.task_id for task in pick))
            if (best_score is None or score > best_score + 1e-12
                    or (abs(score - best_score) <= 1e-12 and ids < best_plan)):
                best_score, best_plan = score, ids

        self.assertEqual(n_total, outcome.n_combinations_total,
                         "优化器报告的组合总数与独立枚举不一致")
        self.assertEqual(n_total, outcome.n_combinations_evaluated,
                         "优化器没有把全部组合评估完")
        self.assertTrue(outcome.exact, "完全枚举后必须标记 exact")
        self.assertEqual(outcome.claim_kind, CLAIM_KIND_EXACT)
        self.assertEqual(tuple(sorted(outcome.best_plan)), best_plan,
                         "优化器的选择与独立暴力枚举的最优解不一致")
        self.assertAlmostEqual(outcome.objective, best_score, places=9)

        planned = sorted(row.task_id for row in result.decisions
                         if row.decision == DECISION_PLANNED)
        self.assertEqual(tuple(planned), best_plan)

    def test_per_node_quota_is_respected(self) -> None:
        """执行器只支持立即执行 → 每节点每 tick 至多 1 条，优化器必须遵守。"""
        observation, _by_node, candidates = self._world()
        queue = _queue(*candidates)
        optimizer = EnumerationOptimizer()
        result = optimizer.plan(observation, queue, 10.0, plan_id="P1")
        per_node = {}
        for row in result.decisions:
            if row.decision == DECISION_PLANNED:
                per_node[row.node_id] = per_node.get(row.node_id, 0) + 1
        self.assertTrue(per_node)
        self.assertTrue(all(count == 1 for count in per_node.values()),
                        f"每节点每 tick 应恰好 1 条：{per_node}")

    def test_pareto_frontier_is_returned_for_small_cases(self) -> None:
        """小规模枚举必须给出帕累托前沿（向量评价，不给综合分）。"""
        observation, _by_node, candidates = self._world()
        queue = _queue(*candidates)
        optimizer = EnumerationOptimizer(
            optimizer_config=OptimizerConfig(compute_pareto=True))
        optimizer.plan(observation, queue, 10.0, plan_id="P1")
        outcome = optimizer.last_outcome
        self.assertTrue(outcome.pareto_frontier)
        for point in outcome.pareto_frontier:
            self.assertEqual(set(point["values"]), set(METRIC_KEYS))
            self.assertEqual(point["provenance"], "predicted")


# ----------------------------------------------------------------------
# ③ 最优性声明不得越界
# ----------------------------------------------------------------------


class TestOptimalityClaimDiscipline(unittest.TestCase):
    def test_claims_are_canonical_and_consistent(self) -> None:
        observation, _by_node, candidates = TestExactnessAgainstBruteForce()._world()
        for policy in OPTIMIZATION_POLICIES:
            queue = _queue(*candidates)
            optimizer = build_scheduler(policy, SchedulingConfig())
            optimizer.plan(observation, queue, 10.0, plan_id="P")
            summary = optimizer.outcome_summary()
            audit = summary["claim_audit"]
            self.assertTrue(audit["all_claims_canonical"], policy.value)
            self.assertTrue(audit["exact_iff_exact_claim"], policy.value)
            self.assertTrue(audit["all_exact_claims_backed"], policy.value)
            for outcome in optimizer.outcomes:
                self.assertIn(outcome.claim_kind, CANONICAL_CLAIMS)
                self.assertEqual(outcome.optimality_claim,
                                 CANONICAL_CLAIMS[outcome.claim_kind].format(
                                     **outcome.claim_args))

    def test_exact_claim_contains_the_negation(self) -> None:
        """精确声明必须**自带**"不是理论上界"这句话（防止被读成上界）。"""
        text = CANONICAL_CLAIMS[CLAIM_KIND_EXACT]
        self.assertIn("不是", text)
        self.assertIn("精确最优", text)

    def test_no_policy_claims_global_optimality(self) -> None:
        """除"小问题精确最优"外，任何声明都必须**自带否定**，不得主张上界。

        检查前先去掉 Markdown 粗体标记：声明里写的是 `**没有**全局最优性证明`，
        直接子串匹配会被那两个星号骗过去（这个坑在本项目里已经踩过多次）。
        """
        for kind, text in CANONICAL_CLAIMS.items():
            plain = text.replace("**", "")
            if kind == CLAIM_KIND_EXACT:
                self.assertIn("不是", plain)
                self.assertIn("精确最优", plain)
                continue
            self.assertTrue(
                any(marker in plain for marker in
                    ("没有全局最优性证明", "不构成理论上界",
                     "不构成任何最优性声明")),
                f"{kind} 的声明没有自带的否定表述：{text}")
            # 主张式措辞检查：短语本身允许出现，但**前文必须有否定词**。
            # 直接用 `assertNotIn("是全局最优")` 会把"既**不是**全局最优"
            # 也判成违规——那又是一次关键词误报。
            self._assert_only_negated(plain, kind)

    def _assert_only_negated(self, plain: str, kind: str) -> None:
        negators = ("不", "非", "没", "无")
        for phrase in ("是全局最优", "达到理论上界", "即为理论上界",
                       "是该问题的理论上界"):
            start = 0
            while True:
                index = plain.find(phrase, start)
                if index < 0:
                    break
                prefix = plain[max(0, index - 6):index]
                self.assertTrue(
                    any(neg in prefix for neg in negators),
                    f"{kind}：「{phrase}」以主张式出现（前文 {prefix!r}）")
                start = index + 1

    def test_budget_stopping_enumeration_reports_partial_progress(self) -> None:
        """预算在枚举中途耗尽：必须报出"评估了几条 / 共几条"。

        这条钉的是一个**诚实性缺陷**：第一版把 `n_combinations_total`
        一律记成 -1（"未知"），于是报告里丢掉"评估了 2 / 共 12"这个关键事实。
        枚举路径的总数是**先算出来的**，没有理由不报。
        """
        observation, _by_node, candidates = TestExactnessAgainstBruteForce()._world()
        queue = _queue(*candidates)
        optimizer = EnumerationOptimizer(optimizer_config=OptimizerConfig(
            kind=OptimizerKind.ENUMERATION, horizon_ticks=1,
            budget=ComputeBudget(max_expansions=10**6, time_limit_s=1e-9)))
        optimizer.plan(observation, queue, 10.0, plan_id="P")
        outcome = optimizer.last_outcome
        self.assertFalse(outcome.exact, "预算耗尽时不得声明精确最优")
        self.assertEqual(outcome.claim_kind, CLAIM_KIND_BUDGET)
        self.assertGreater(outcome.n_combinations_total, 0,
                           "枚举路径必须报出真实组合总数")
        self.assertLess(outcome.n_combinations_evaluated,
                        outcome.n_combinations_total)
        self.assertTrue(outcome.budget["exhausted"])

    def test_beam_downgrade_does_not_fake_a_total(self) -> None:
        """组合数超预算 → 降级束搜索，**不虚报**组合总数（记 -1 表示未知）。"""
        observation, _by_node, candidates = TestExactnessAgainstBruteForce()._world()
        queue = _queue(*candidates)
        optimizer = EnumerationOptimizer(optimizer_config=OptimizerConfig(
            kind=OptimizerKind.ENUMERATION, horizon_ticks=1,
            budget=ComputeBudget(max_expansions=2, time_limit_s=10.0)))
        optimizer.plan(observation, queue, 10.0, plan_id="P")
        outcome = optimizer.last_outcome
        self.assertFalse(outcome.exact)
        self.assertEqual(outcome.claim_kind, CLAIM_KIND_BUDGET)
        self.assertEqual(outcome.n_combinations_total, -1,
                         "束搜索路径不得虚报组合总数")
        self.assertTrue(outcome.explanations.get("downgraded_from_exact"))

    def test_budget_exhaustion_downgrades_exact(self) -> None:
        """预算耗尽一律**降级**：不得再声明精确最优（由上面两条分别覆盖两条路径）。"""
        observation, _by_node, candidates = TestExactnessAgainstBruteForce()._world()
        queue = _queue(*candidates)
        optimizer = EnumerationOptimizer(optimizer_config=OptimizerConfig(
            kind=OptimizerKind.ENUMERATION,
            budget=ComputeBudget(max_expansions=2, time_limit_s=1e-9),
            horizon_ticks=1))
        optimizer.plan(observation, queue, 10.0, plan_id="P")
        outcome = optimizer.last_outcome
        self.assertFalse(outcome.exact)
        self.assertEqual(outcome.claim_kind, CLAIM_KIND_BUDGET)

    def test_rolling_horizon_never_claims_exact(self) -> None:
        """滚动规划含前瞻推演 → **永远**不声明精确最优。"""
        observation, _by_node, candidates = TestExactnessAgainstBruteForce()._world()
        queue = _queue(*candidates)
        optimizer = RollingHorizonOptimizer(optimizer_config=OptimizerConfig(
            kind=OptimizerKind.ROLLING_HORIZON, horizon_ticks=3))
        optimizer.plan(observation, queue, 10.0, plan_id="P")
        self.assertFalse(optimizer.last_outcome.exact)
        self.assertEqual(optimizer.last_outcome.claim_kind,
                         CLAIM_KIND_LOOKAHEAD)

    def test_two_optimizers_are_not_the_same_method(self) -> None:
        """两种优化参考的默认配置必须**语义不同**（否则只是两个名字）。"""
        enum_cfg = default_optimizer_config(OptimizerKind.ENUMERATION)
        roll_cfg = default_optimizer_config(OptimizerKind.ROLLING_HORIZON)
        self.assertEqual(enum_cfg.horizon_ticks, 1)
        self.assertGreater(roll_cfg.horizon_ticks, 1)
        self.assertNotEqual(enum_cfg.kind, roll_cfg.kind)

    def test_horizon_is_clamped_by_budget(self) -> None:
        config = OptimizerConfig(
            kind=OptimizerKind.ROLLING_HORIZON, horizon_ticks=99,
            budget=ComputeBudget(max_horizon_ticks=2))
        optimizer = OptimizationScheduler(None, config)
        self.assertEqual(optimizer._horizon, 2)
        self.assertTrue(optimizer._horizon_clipped)


# ----------------------------------------------------------------------
# ④ 与规则基线共用定义
# ----------------------------------------------------------------------


class TestSharesDefinitionsWithBaselines(unittest.TestCase):
    def test_all_schedulers_share_plan_contract(self) -> None:
        observation, _by_node, candidates = TestExactnessAgainstBruteForce()._world()
        for policy in list(BASELINE_POLICIES) + list(OPTIMIZATION_POLICIES):
            queue = _queue(*candidates)
            scheduler = build_scheduler(policy, SchedulingConfig())
            result = scheduler.plan(observation, queue, 10.0, plan_id="P")
            self.assertTrue(hasattr(result, "decisions"), policy.value)
            if result.plan is not None:
                self.assertEqual(result.plan.submit_time_s, 10.0)
                for task in result.plan.tasks:
                    self.assertEqual(task.start_s, 10.0)
                    self.assertIn(task.node_id, ("A", "B"))

    def test_not_eligible_semantics_is_shared(self) -> None:
        """节点本 tick 没有到达摘要 → 两种路径都必须判 `not_eligible`。"""
        observation = _central([_node("B", [_track("T3", 1.0, 20.0)])])
        for policy in (SchedulerPolicy.RULE, SchedulerPolicy.ENUMERATION):
            queue = _queue(_task("a-upd", "A", targets=("T1",)))
            scheduler = build_scheduler(policy, SchedulingConfig())
            result = scheduler.plan(observation, queue, 10.0, plan_id="P")
            decisions = [row.decision for row in result.decisions]
            self.assertIn(DECISION_NOT_ELIGIBLE, decisions, policy.value)
            self.assertIsNone(result.plan)

    def test_infeasible_target_is_visible_to_both(self) -> None:
        """优化参考只能在观测允许的对象上建计划（共用同一套观测）。"""
        observation = _central([_node("A", [_track("T1", 1.0, 20.0)])])
        queue = _queue(_task("a-upd", "A", targets=("T1",)))
        scheduler = build_scheduler(SchedulerPolicy.ENUMERATION,
                                    SchedulingConfig())
        result = scheduler.plan(observation, queue, 10.0, plan_id="P")
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.tasks[0].entities, ("T1",))


# ----------------------------------------------------------------------
# ⑤ 向量评价与帕累托
# ----------------------------------------------------------------------


class TestEvaluationVector(unittest.TestCase):
    def test_template_is_ignored_in_objective_but_present(self) -> None:
        vector = EvaluationVector(values={key: 0.0 for key in METRIC_KEYS})
        spec = ObjectiveSpec()
        # 计算耗时权重为 0：同一次规划内它对候选无区分能力
        self.assertEqual(spec.weights["compute_time"], 0.0)
        self.assertIn("compute_time", vector.values)

    def test_dominance_respects_direction(self) -> None:
        better = EvaluationVector(values={
            "service_completion": 0.5, "task_timeliness": 0.5,
            "estimate_quality": 0.5, "resource_consumption": 0.1,
            "communication_overhead": 100.0, "compute_time": 1.0})
        worse = EvaluationVector(values={
            "service_completion": 0.4, "task_timeliness": 0.5,
            "estimate_quality": 0.5, "resource_consumption": 0.2,
            "communication_overhead": 200.0, "compute_time": 2.0})
        self.assertTrue(better.dominates(worse))
        self.assertFalse(worse.dominates(better))

    def test_tradeoff_is_not_dominated(self) -> None:
        """一维更好、另一维更差 → 互不支配（这正是要用向量的原因）。"""
        a = EvaluationVector(values={
            "service_completion": 0.5, "task_timeliness": 0.2,
            "estimate_quality": 0.5, "resource_consumption": 0.1,
            "communication_overhead": 0.0, "compute_time": 0.01})
        b = EvaluationVector(values={
            "service_completion": 0.4, "task_timeliness": 0.9,
            "estimate_quality": 0.5, "resource_consumption": 0.1,
            "communication_overhead": 1000.0, "compute_time": 0.2})
        self.assertFalse(a.dominates(b))
        self.assertFalse(b.dominates(a))
        self.assertEqual(len(pareto_frontier([a, b])), 2)


class TestBudgetDeterminism(unittest.TestCase):
    """预算必须是**确定性**的，否则"可复现"这条验收条件根本不成立。

    用墙钟当预算的方法天然不可复现：实测同一配置连跑四次，展开数分别是
    1315/1270/1312/1292（机器负载不同 → 停止点不同 → 计划可能不同）。
    这条不变量是验收第 5 项的地基。
    """

    def _world(self):
        from tests.test_resource_optimization import (
            TestExactnessAgainstBruteForce)
        return TestExactnessAgainstBruteForce()._world()

    def _fresh(self):
        """每次都重建世界：`plan()` 会把选中任务置为 SUBMITTED，
        复用同一批任务对象会让第二次调用的候选变少（这个坑踩过一次）。"""
        observation, _by_node, candidates = self._world()
        return observation, _queue(*candidates)

    def test_default_budget_is_deterministic(self) -> None:
        expansions = []
        for _ in range(3):
            observation, queue = self._fresh()
            optimizer = RollingHorizonOptimizer()
            optimizer.plan(observation, queue, 10.0, plan_id="P")
            summary = optimizer.outcome_summary()
            self.assertTrue(summary["deterministic"])
            self.assertEqual(summary["n_time_limit_triggered"], 0)
            expansions.append(summary["total_expansions"])
        self.assertEqual(len(set(expansions)), 1,
                         f"默认预算下展开数应完全一致：{expansions}")
        self.assertGreater(expansions[0], 0)

    def test_tiny_time_limit_is_flagged_non_deterministic(self) -> None:
        """墙钟安全阀触发时必须**如实标记**，不能悄悄继续。"""
        observation, queue = self._fresh()
        optimizer = RollingHorizonOptimizer(optimizer_config=OptimizerConfig(
            kind=OptimizerKind.ROLLING_HORIZON, horizon_ticks=1,
            budget=ComputeBudget(time_limit_s=1e-9, max_expansions=10**6)))
        optimizer.plan(observation, queue, 10.0, plan_id="P")
        summary = optimizer.outcome_summary()
        self.assertFalse(summary["deterministic"])
        self.assertGreaterEqual(summary["n_time_limit_triggered"], 1)
        self.assertIn("可复现基准", summary["determinism_note"])
        self.assertIn("deterministic=False", summary["determinism_note"])
        self.assertTrue(optimizer.last_outcome.budget["time_limit_triggered"])
        self.assertIn("不可复现", optimizer.last_outcome.budget[
            "exhausted_reason"])

    def test_time_limit_none_means_no_wall_clock_valve(self) -> None:
        observation, queue = self._fresh()
        optimizer = RollingHorizonOptimizer(optimizer_config=OptimizerConfig(
            kind=OptimizerKind.ROLLING_HORIZON, horizon_ticks=1,
            budget=ComputeBudget(time_limit_s=None, max_expansions=10**6)))
        optimizer.plan(observation, queue, 10.0, plan_id="P")
        budget = optimizer.last_outcome.budget
        self.assertIsNone(budget["time_limit_s"])
        self.assertFalse(budget["time_limit_triggered"])
        self.assertTrue(budget["deterministic"])


# ----------------------------------------------------------------------
# ⑥ 冻结契约
# ----------------------------------------------------------------------


class TestContractFreeze(unittest.TestCase):
    def test_frozen_digest_matches_current_code(self) -> None:
        result = verify_frozen(strict=False)
        self.assertTrue(result["ok"], result["hint"])
        self.assertEqual(result["recorded"], FROZEN_DIGEST)
        self.assertEqual(CONTRACT_VERSION, "resource-contract-v1")

    def test_digest_is_deterministic(self) -> None:
        self.assertEqual(contract_digest(), contract_digest())

    def test_any_change_changes_the_digest(self) -> None:
        """改一处口径摘要就变——这是"冻结"的实际含义。"""
        snapshot = contract_snapshot()
        baseline = contract_digest(snapshot)
        mutated = json_roundtrip(snapshot)
        mutated["completion_semantics"]["starvation_definition"] += "（被改过）"
        self.assertNotEqual(contract_digest(mutated), baseline)

    def test_verify_frozen_raises_in_strict_mode(self) -> None:
        if verify_frozen(strict=False)["ok"]:
            self.skipTest("当前口径与摘要一致，无需触发严格模式")
        with self.assertRaises(ContractFrozenError):
            verify_frozen(strict=True)

    def test_snapshot_covers_the_four_frozen_items(self) -> None:
        snapshot = contract_snapshot()
        for key in ("observation_schema", "completion_semantics",
                    "resource_model", "baseline_config", "evaluation"):
            self.assertIn(key, snapshot)
        self.assertEqual(snapshot["observation_schema"]["schema_version"],
                         "rm-obs-1.0")
        self.assertIn("完成 + 过期",
                      snapshot["completion_semantics"][
                          "completion_rate_denominator"])
        self.assertEqual(snapshot["baseline_config"]["baseline_run"]["steps"],
                         24)


def json_roundtrip(payload):
    import json
    return json.loads(json.dumps(payload, ensure_ascii=False))


# ----------------------------------------------------------------------
# ⑦ 阶段验收检查
# ----------------------------------------------------------------------


class TestAcceptanceChecks(unittest.TestCase):
    def test_all_six_checks_are_defined_and_named(self) -> None:
        report = run_acceptance(seeds=(42,), steps=12, quick=True)
        names = [check["name_cn"] for check in report["checks"]]
        for expected in ("多节点执行真实生效", "资源不超支", "信息不越权",
                         "任务队列可追溯", "规则与优化参考可复现",
                         "调度反向控制感知链（真闭环）"):
            self.assertIn(expected, names)
        self.assertEqual(report["n_checks"], 6)

    def test_each_check_passes_on_the_baseline_config(self) -> None:
        report = run_acceptance(seeds=(42,), steps=12, quick=True)
        failed = [check["name_cn"] for check in report["checks"]
                  if not check["ok"]]
        self.assertEqual(failed, [], f"验收未通过：{failed}")
        self.assertTrue(report["all_passed"])

    def test_information_boundary_is_a_runtime_comparison(self) -> None:
        """信息不越权必须有**运行时**对照，不能只有静态扫描。"""
        check = check_information_boundary(seed=42, steps=12, cut_tick=5)
        self.assertTrue(check.ok, check.evidence)
        self.assertTrue(all(check.evidence["identical_before_cut"]))
        self.assertTrue(check.evidence["differs_inside_window"],
                        "扰动没有生效，这条检查就是空转")
        self.assertEqual(len(check.evidence["ticks_compared"]), 5)

    def test_queue_traceability_counts_reconcile(self) -> None:
        check = check_queue_traceability(seed=42, steps=12)
        self.assertTrue(check.ok, check.evidence)
        self.assertTrue(check.evidence["status_sum_equals_total"])
        self.assertEqual(check.evidence["decisions_without_reason"], 0)

    def test_multi_node_check_requires_real_consumption(self) -> None:
        check = check_multi_node_execution(seed=42, steps=12)
        self.assertTrue(check.ok, check.evidence)
        self.assertGreaterEqual(
            len(check.evidence["nodes_with_actual_consumption"]), 2)

    def test_resource_check_covers_all_policies(self) -> None:
        check = check_resource_conservation(seeds=(42,), steps=12)
        self.assertTrue(check.ok, check.evidence)
        self.assertEqual(check.evidence["n_violations"], 0)

    def test_plan_controlled_feedback_proves_the_loop_is_closed(self) -> None:
        """第 6 项验收：**调度真的能改变航迹质量**。

        这是"闭环成立与否"的判据，不是"多了个开关"：
        旧路径下三策略估计质量完全相同（闭环未成立），
        新路径下至少两个策略不同。
        """
        check = check_plan_controlled_feedback(seed=42, steps=16)
        self.assertTrue(check.ok, check.evidence)
        evidence = check.evidence
        self.assertTrue(evidence["legacy_quality_identical_across_policies"])
        self.assertTrue(evidence["feedback_quality_differs_across_policies"])
        self.assertEqual(evidence["scans_inside_outage"], 0)
        self.assertGreater(evidence["scans_after_outage"], 0)
        self.assertTrue(evidence["sigma_monotone_increase"])
        self.assertTrue(evidence["no_duplicate_execution"])
        self.assertTrue(evidence["no_truth_leak"])
        self.assertTrue(evidence["comm_bytes_match_ledger"])
        self.assertTrue(evidence["resource_conserved"])

    def test_reproducibility_requires_deterministic_search(self) -> None:
        """可复现检查必须**显式**要求优化参考的搜索是确定性的。"""
        from resource_management.acceptance import check_reproducibility
        check = check_reproducibility(seeds=(42,), steps=12)
        self.assertTrue(check.ok, check.evidence)
        self.assertIn("non_deterministic_runs", check.evidence)
        self.assertEqual(check.evidence["non_deterministic_runs"], [])
        self.assertEqual(check.evidence["mismatches"], [])
        self.assertIn("确定", check.evidence["note"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
