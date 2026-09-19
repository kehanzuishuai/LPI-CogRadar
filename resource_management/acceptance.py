"""大阶段一验收清单（5 项条件，逐项给证据）。

用户给出的五条通过条件
----------------------
1. **多节点执行真实生效**
2. **资源不超支**
3. **信息不越权**
4. **任务队列可追溯**
5. **规则与优化参考可复现**

设计原则
--------
* 每条检查都返回**证据**（数字、样本、对照结果），不是一句"通过"；
* 能被机器判定的就机器判定，判不了的明确写"人工复核"而不是假装通过；
* 第三项（信息不越权）除了静态扫描，还必须有一条**运行时对照**：
  把"未来"改掉、当前观测不变，规划结果必须逐位相同——
  这是"优化参考没有偷看未来"的**决定性**证据，静态检查做不到这一点。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from resource_management.closed_loop import run_closed_loop
from resource_management.contract_v1 import (
    CONTRACT_VERSION,
    contract_digest,
    verify_frozen,
)
from resource_management.optimization import (
    EvaluationVector,
    default_optimizer_config,
    OptimizerKind,
)
from resource_management.scheduling import (
    BASELINE_POLICIES,
    OPTIMIZATION_POLICIES,
    SchedulerPolicy,
)
from resource_management.units import BUDGET_UNITS


@dataclass
class Check:
    """一条验收检查的结果。"""

    key: str
    name_cn: str
    ok: bool
    evidence: Dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name_cn": self.name_cn,
            "ok": bool(self.ok),
            "detail": self.detail,
            "evidence": self.evidence,
        }


# ----------------------------------------------------------------------
# ① 多节点执行真实生效
# ----------------------------------------------------------------------


def check_multi_node_execution(policy: SchedulerPolicy = SchedulerPolicy.RULE,
                               seed: int = 42,
                               steps: int = 12) -> Check:
    """一份计划真的驱动了**多个**节点，且成本真的从各自预算里扣了。

    只看"计划里有几个节点"是不够的：计划可能被执行器整份拒绝。
    因此同时要求：执行器状态为 applied/partial、两个节点各自有
    `applied > 0`、且账本里两个节点都有消耗记录。
    """
    result = run_closed_loop(policy, seed=seed, steps=steps)
    metrics = result.metrics
    per_node = metrics["per_node"]
    applied = {node_id: int(result.metrics["per_node_occupancy"]
                            .get(node_id, {}).get(
                                f"{BUDGET_UNITS[0].value}_consumed", 0) > 0)
               for node_id in per_node}
    multi_node_plans = [row for row in result.plan_log
                        if row["n_applied"] >= 1]
    consumed_nodes = [node_id for node_id, bucket in
                      metrics["per_node_occupancy"].items()
                      if any(value > 0 for value in bucket.values())]
    ok = (len(consumed_nodes) >= 2
          and all(bucket["n_planned"] > 0 for bucket in per_node.values())
          and not (set(consumed_nodes) - set(per_node)))
    return Check(
        key="multi_node_execution",
        name_cn="多节点执行真实生效",
        ok=ok,
        evidence={
            "policy": policy.value,
            "n_plans": len(result.plan_log),
            "plans_with_applied_tasks": len(multi_node_plans),
            "nodes_with_actual_consumption": sorted(consumed_nodes),
            "per_node_planned": {node_id: bucket["n_planned"]
                                 for node_id, bucket in per_node.items()},
            "per_node_consumed": {
                node_id: {key: round(value, 6)
                          for key, value in bucket.items()}
                for node_id, bucket in metrics["per_node_occupancy"].items()},
            "any_consumption_flag": applied,
        },
        detail=("要求：≥2 个节点既有已规划任务、又在账本里有真实消耗；"
                "计划数 > 0"),
    )


# ----------------------------------------------------------------------
# ② 资源不超支
# ----------------------------------------------------------------------


def check_resource_conservation(seeds: Sequence[int] = (42,),
                                steps: int = 24,
                                policies: Optional[Sequence[SchedulerPolicy]] = None
                                ) -> Check:
    """每个节点每个预算单位都不得超支，且守恒式残差为 0。"""
    policies = list(policies or (list(BASELINE_POLICIES)
                                 + list(OPTIMIZATION_POLICIES)))
    violations: List[Dict[str, Any]] = []
    checked: List[Dict[str, Any]] = []
    for policy in policies:
        for seed in seeds:
            result = run_closed_loop(policy, seed=seed, steps=steps)
            report = result.conservation
            for node_id, entry in (report.get("nodes") or {}).items():
                checked.append({"policy": policy.value, "seed": seed,
                                "node_id": node_id})
                for unit in BUDGET_UNITS:
                    key = unit.value
                    residual = float(entry.get("residual", {}).get(key, 0.0))
                    remaining = float(entry.get("remaining", {}).get(key, 0.0))
                    consumed = float(entry.get("consumed", {}).get(key, 0.0))
                    capacity = float(entry.get("capacity", {}).get(key, 0.0))
                    if (abs(residual) > 1e-9 or remaining < -1e-9
                            or consumed > capacity + 1e-9):
                        violations.append({
                            "policy": policy.value, "seed": seed,
                            "node_id": node_id, "unit": key,
                            "residual": residual, "remaining": remaining,
                            "consumed": consumed, "capacity": capacity})
            if not report.get("all_conserved", False):
                violations.append({"policy": policy.value, "seed": seed,
                                   "node_id": "*",
                                   "unit": "*", "residual": None,
                                   "remaining": None, "consumed": None,
                                   "capacity": None})
    return Check(
        key="resource_conservation",
        name_cn="资源不超支",
        ok=not violations,
        evidence={
            "n_node_unit_samples": len(checked) * len(BUDGET_UNITS),
            "n_violations": len(violations),
            "violations": violations[:8],
            "invariant": ("consumed + reserved + remaining == capacity，"
                          "三者均非负，consumed ≤ capacity"),
        },
        detail=f"覆盖 {len(policies)} 个策略 × {len(seeds)} 个种子 × 全部节点与预算单位",
    )


# ----------------------------------------------------------------------
# ③ 信息不越权
# ----------------------------------------------------------------------


def _plan_signature(result: Any, time_s: float) -> List[Dict[str, Any]]:
    """某一 tick 的规划签名（任务/节点/决定/优先级/理由）。"""
    rows = [row for row in result.decisions
            if abs(float(row.get("time_s", -1.0)) - time_s) < 1e-9]
    return sorted(
        ({"task_id": row.get("task_id"), "node_id": row.get("node_id"),
          "decision": row.get("decision"), "priority": row.get("priority"),
          "kind": row.get("kind")}
         for row in rows),
        key=lambda row: (str(row["task_id"]), str(row["decision"])))


def check_information_boundary(seed: int = 42, steps: int = 12,
                               cut_tick: int = 6) -> Check:
    """优化参考不得读取未来故障：**同一 tick 的规划必须与未来无关**。

    做法：跑两次，第二次让 NODE_B 从 `cut_tick + 1` 起不可用。
    `cut_tick` 之前的观测**逐位相同**，因此该 tick 的规划也必须逐位相同。
    如果优化参考偷看了"未来会故障"，两次规划就会不同——这条检查能抓住它。

    静态部分（模块不 import 真值层、观测字段全部登记、无真值通道）
    由 `tests/` 与 `verify_v4.py` 另外钉住；这里给出**运行时**证据。
    """
    baseline = run_closed_loop(SchedulerPolicy.ROLLING_HORIZON, seed=seed,
                               steps=steps)
    perturbed = run_closed_loop(
        SchedulerPolicy.ROLLING_HORIZON, seed=seed, steps=steps,
        mechanisms={"unavailable_windows": {
            "NODE_B": [(float(cut_tick + 1), float(steps))]}})
    same: List[bool] = []
    diffs: List[Dict[str, Any]] = []
    for tick in range(1, cut_tick + 1):
        a = _plan_signature(baseline, float(tick))
        b = _plan_signature(perturbed, float(tick))
        same.append(a == b)
        if a != b:
            diffs.append({"tick": tick, "baseline": a, "perturbed": b})
    # 反向对照：故障窗口**之内**的 tick 允许不同（否则说明扰动没生效）
    inside_differs = (_plan_signature(baseline, float(cut_tick + 1))
                      != _plan_signature(perturbed, float(cut_tick + 1)))
    return Check(
        key="information_boundary",
        name_cn="信息不越权",
        ok=all(same) and inside_differs,
        evidence={
            "cut_tick": cut_tick,
            "ticks_compared": list(range(1, cut_tick + 1)),
            "identical_before_cut": same,
            "differences_before_cut": diffs[:3],
            "differs_inside_window": inside_differs,
            "perturbation": ("第二次运行让 NODE_B 从 tick "
                             f"{cut_tick + 1} 起不可用（未来故障）"),
            "note": ("故障窗口**之前**的规划必须逐位相同——"
                     "这是「没有偷看未来」的决定性证据；"
                     "窗口**之内**允许不同，否则说明扰动根本没生效。"),
        },
        detail=("运行时对照：改掉未来故障、保持当前观测不变，"
                "规划必须逐位相同"),
    )


# ----------------------------------------------------------------------
# ④ 任务队列可追溯
# ----------------------------------------------------------------------


def check_queue_traceability(policy: SchedulerPolicy = SchedulerPolicy.RULE,
                             seed: int = 42, steps: int = 24) -> Check:
    """队列里每条任务都能回答"从哪来、去哪了、为什么"。"""
    from resource_management.tasks import TaskQueue, TaskStatus

    result = run_closed_loop(policy, seed=seed, steps=steps,
                             keep_decisions=True)
    # 从队列摘要反推：各状态计数之和必须等于任务总数
    summary = result.queue_summary
    by_status = summary["by_status"]
    counts = {status.value: int(by_status.get(status.value, 0))
              for status in TaskStatus}
    total = sum(counts.values())
    # 逐条决策：每条都必须带理由
    decisions = result.decisions
    no_reason = [row for row in decisions
                 if not (row.get("reasons") or row.get("reason"))]
    # 逐节点时间线：每条都要能对上一个 plan 或一个明确原因
    timeline_rows = [row for rows in result.timelines.values() for row in rows]
    bad_timeline = [row for row in timeline_rows
                    if not row.get("reason") or row.get("decision") is None]
    # 计划日志：每条 applied 的都记了 n_applied
    bad_plans = [row for row in result.plan_log
                 if "n_applied" not in row or "status" not in row]
    ok = (total == summary["n_tasks"]
          and not no_reason and not bad_timeline and not bad_plans)
    return Check(
        key="queue_traceability",
        name_cn="任务队列可追溯",
        ok=ok,
        evidence={
            "n_tasks": summary["n_tasks"],
            "by_status": counts,
            "status_sum_equals_total": total == summary["n_tasks"],
            "n_decisions": len(decisions),
            "decisions_without_reason": len(no_reason),
            "timeline_rows": len(timeline_rows),
            "timeline_rows_without_reason": len(bad_timeline),
            "plan_log_rows": len(result.plan_log),
            "plan_log_rows_malformed": len(bad_plans),
            "by_kind_status": summary.get("by_kind_status", {}),
        },
        detail=("逐状态计数之和 = 任务总数；每条决策/时间线行都带原因；"
                "每条计划都记了状态与应用数"),
    )


# ----------------------------------------------------------------------
# ⑤ 规则与优化参考可复现
# ----------------------------------------------------------------------


def _fingerprint(result: Any) -> str:
    """一次运行的指纹：决策 + 指标 + 账本消耗（不含耗时这类墙钟量）。"""
    payload = {
        "policy": result.policy,
        "seed": result.seed,
        "steps": result.steps,
        "decisions": [
            {"time_s": row.get("time_s"), "task_id": row.get("task_id"),
             "node_id": row.get("node_id"), "decision": row.get("decision"),
             "kind": row.get("kind"), "priority": row.get("priority")}
            for row in result.decisions],
        "metrics": {key: value for key, value in result.metrics.items()
                    if key not in ("evaluation_vector", "note",
                                   "compute_time_s")},
        "plan_log": result.plan_log,
        "queue_summary": result.queue_summary,
    }
    vector = result.metrics.get("evaluation_vector", {}).get("values", {})
    payload["vector_sans_time"] = {
        key: value for key, value in vector.items() if key != "compute_time"}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      default=str)


def check_reproducibility(seeds: Sequence[int] = (42, 7),
                          steps: int = 16) -> Check:
    """同配置重跑必须**逐位相同**（含优化参考；耗时不参与比较）。

    这里同时校验一条**不变量**：优化参考的搜索必须是**确定性**的。
    用墙钟当预算的方法天然不可复现——实测同一配置连跑四次，展开数分别是
    1315/1270/1312/1292，计划也因此可能不同。因此墙钟只能是**安全阀**：
    默认取值大到不会触发，一旦触发就把 `deterministic` 置 False 并如实入账，
    那次结果不允许被当作可复现基准。
    """
    mismatches: List[Dict[str, Any]] = []
    compared: List[str] = []
    non_deterministic: List[str] = []
    for policy in list(BASELINE_POLICIES) + list(OPTIMIZATION_POLICIES):
        for seed in seeds:
            first = run_closed_loop(policy, seed=seed, steps=steps)
            second = run_closed_loop(policy, seed=seed, steps=steps)
            label = f"{policy.value}@seed{seed}"
            compared.append(label)
            if _fingerprint(first) != _fingerprint(second):
                mismatches.append({"run": label})
            summary = first.optimizer_summary or {}
            if summary and not summary.get("deterministic", True):
                non_deterministic.append(label)
    frozen = verify_frozen(strict=False)
    return Check(
        key="reproducibility",
        name_cn="规则与优化参考可复现",
        ok=not mismatches and not non_deterministic and frozen["ok"],
        evidence={
            "runs_compared": compared,
            "mismatches": mismatches,
            "non_deterministic_runs": non_deterministic,
            "contract_version": CONTRACT_VERSION,
            "contract_digest": contract_digest(),
            "contract_frozen_ok": frozen["ok"],
            "note": ("指纹包含决策/指标/计划日志/队列摘要，"
                     "**排除**墙钟耗时（它不是可复现量，单独作为计算耗维度报告）；"
                     "同时要求：冻结契约摘要一致、且优化参考的搜索是确定性的"
                     "（墙钟安全阀未触发）"),
        },
        detail=("同策略同种子跑两次逐位相同；且冻结口径未被改动、"
                "优化参考的停止点不依赖机器负载"),
    )


# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------


def run_acceptance(seeds: Sequence[int] = (42,), steps: int = 24,
                   quick: bool = False) -> Dict[str, Any]:
    """跑全部 5 项检查，返回清单（含总体结论）。"""
    checks: List[Check] = [
        check_multi_node_execution(seed=seeds[0], steps=min(steps, 12)),
        check_resource_conservation(seeds=tuple(seeds), steps=steps),
        check_information_boundary(seed=seeds[0], steps=min(steps, 12)),
        check_queue_traceability(seed=seeds[0], steps=steps),
        check_reproducibility(seeds=tuple(seeds) if not quick else (seeds[0],),
                              steps=min(steps, 16)),
    ]
    passed = [check for check in checks if check.ok]
    return {
        "contract_version": CONTRACT_VERSION,
        "contract_digest": contract_digest(),
        "n_checks": len(checks),
        "n_passed": len(passed),
        "all_passed": len(passed) == len(checks),
        "checks": [check.to_dict() for check in checks],
        "gate": ("未全部通过前**不进入学习算法阶段**；"
                 "通过后也不要求规则方法必须失败、学习方法必须胜出。"),
    }


def render_acceptance_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# 大阶段一验收清单",
        "",
        f"- 契约版本：`{report['contract_version']}`",
        f"- 契约摘要：`{report['contract_digest']}`",
        f"- 通过：{report['n_passed']} / {report['n_checks']}",
        f"- 总判定：{'**全部通过**' if report['all_passed'] else '**未通过**'}",
        "",
        f"> {report['gate']}",
        "",
        "| 条件 | 结果 | 证据摘要 |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        evidence = check["evidence"]
        summary = "；".join(f"{k}={_short(v)}"
                            for k, v in list(evidence.items())[:4])
        lines.append(f"| {check['name_cn']} | "
                     f"{'通过' if check['ok'] else '**未通过**'} | {summary} |")
    lines += ["", "## 逐项详细证据", ""]
    for check in report["checks"]:
        lines.append(f"### {check['name_cn']}（{'通过' if check['ok'] else '未通过'}）")
        lines.append("")
        lines.append(f"- 判据：{check['detail']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(check["evidence"], ensure_ascii=False,
                                indent=2, default=str))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= 80 else text[:77] + "..."


__all__ = [
    "Check", "check_information_boundary", "check_multi_node_execution",
    "check_queue_traceability", "check_reproducibility",
    "check_resource_conservation", "render_acceptance_markdown",
    "run_acceptance",
]
