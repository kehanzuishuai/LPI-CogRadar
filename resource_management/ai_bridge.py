"""调度理由 → 只读 AI 诊断（v4.5）。

三条纪律
--------
1. **只读**：本桥只接收调度决策的**副本**（`SchedulingDecision` 或它的
   `to_dict()` 结果），不持有 `TaskQueue` / `SchedulerBase` / `UnifiedExecutor`
   的引用，因此**结构上不可能**修改计划、队列或资源账本。
2. **只解释已发生的事**：解释内容只覆盖"已经排入的计划"与
   "被拒/被推迟/被放弃的任务及其原因"。不做预测，不建议改计划。
3. **远程解释超时不得阻塞仿真时钟**：`explain_schedule()` 带
   **墙钟超时**；超时即回退本地规则解释并标注 `explanation_timeout=True`。
   时钟推进只由 `GlobalClock` 负责，本模块**不 import** 时钟，
   也没有任何推进时钟的路径。

为什么超时要单独实现
--------------------
远程 provider 慢是**外部**问题，不能让它卡住仿真节奏。
本工程早先只做了"provider 异常回退"，没有做"provider 慢"的处理——
慢与失败是两件事：异常会被 `try/except` 立刻捕获，
而"慢"会让调用方一直等下去。因此这里用
`concurrent.futures` 给每次远程解释加一个**硬超时**。
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from resource_management.scheduling import (
    DECISION_ABANDONED,
    DECISION_DEFERRED,
    DECISION_NOT_ELIGIBLE,
    DECISION_PLANNED,
    DECISION_SUPPRESSED_DUPLICATE_NODE,
)

#: 决策去向 → 中文（解释里一律用它）
DECISION_CN: Dict[str, str] = {
    DECISION_PLANNED: "已排入计划",
    DECISION_DEFERRED: "本 tick 推迟",
    DECISION_ABANDONED: "主动放弃",
    DECISION_SUPPRESSED_DUPLICATE_NODE: "因多节点重复规则被抑制",
    DECISION_NOT_ELIGIBLE: "该节点当前不可服务",
}

#: 默认墙钟预算（秒）
DEFAULT_TIMEOUT_S = 0.5


@dataclass
class Rationale:
    """一次调度解释的结果。"""

    summary: str
    findings: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = "local_rule"
    #: 是否因**超时**而回退（与"provider 抛异常"不同）
    explanation_timeout: bool = False
    elapsed_s: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "findings": list(self.findings),
            "provider": self.provider,
            "explanation_timeout": self.explanation_timeout,
            "elapsed_s": round(self.elapsed_s, 6),
            "note": self.note,
        }


def build_rationale_payload(decisions: Sequence[Any],
                            plan_meta: Optional[Dict[str, Any]] = None
                            ) -> Dict[str, Any]:
    """把调度决策整理成**只读**的解释载荷。

    `decisions` 可以是 `SchedulingDecision` 列表或它们已序列化的字典列表；
    两者都只被**读取**，不会被修改。
    """
    rows: List[Dict[str, Any]] = []
    for item in decisions:
        payload = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        rows.append({
            "task_id": payload.get("task_id", ""),
            "node_id": payload.get("node_id", ""),
            "kind": payload.get("kind", ""),
            "decision": payload.get("decision", ""),
            "decision_cn": DECISION_CN.get(payload.get("decision", ""),
                                           payload.get("decision", "")),
            "priority": payload.get("priority", 0.0),
            "reasons": list(payload.get("reasons", []) or []),
            "evidence": dict(payload.get("evidence", {}) or {}),
            "time_s": payload.get("time_s"),
            "policy": payload.get("policy", ""),
        })
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["decision"], []).append(row)
    return {
        "read_only": True,
        "n_decisions": len(rows),
        "by_decision": {key: len(value) for key, value in grouped.items()},
        "decisions": rows,
        "plan_meta": dict(plan_meta or {}),
    }


def explain_locally(payload: Dict[str, Any]) -> Rationale:
    """本地规则解释（**永远可用**，远程 provider 不可用/超时都用它）。"""
    rows = list(payload.get("decisions") or [])
    planned = [row for row in rows if row["decision"] == DECISION_PLANNED]
    deferred = [row for row in rows if row["decision"] == DECISION_DEFERRED]
    abandoned = [row for row in rows if row["decision"] == DECISION_ABANDONED]
    suppressed = [row for row in rows
                  if row["decision"] == DECISION_SUPPRESSED_DUPLICATE_NODE]
    ineligible = [row for row in rows
                  if row["decision"] == DECISION_NOT_ELIGIBLE]

    findings: List[Dict[str, Any]] = []
    by_node: Dict[str, List[str]] = {}
    for row in planned:
        by_node.setdefault(row["node_id"], []).append(row["kind"])
    for node_id, kinds in sorted(by_node.items()):
        counts: Dict[str, int] = {}
        for kind in kinds:
            counts[kind] = counts.get(kind, 0) + 1
        findings.append({
            "code": "PLANNED_WORK_DIVISION",
            "node_id": node_id,
            "message": (f"节点 {node_id} 在本 tick 承接 "
                        + "、".join(f"{kind}×{count}"
                                    for kind, count in sorted(counts.items()))),
            "evidence": {"counts": counts},
        })
    for row in planned[:8]:
        findings.append({
            "code": "PLANNED_REASON",
            "node_id": row["node_id"],
            "task_id": row["task_id"],
            "message": (f"{row['task_id']}（{row['kind']}）优先级 "
                        f"{row['priority']:.4f}；理由："
                        + "；".join(row["reasons"][:2])),
            "evidence": row["evidence"],
        })
    for row in deferred:
        findings.append({
            "code": "DEFERRED",
            "node_id": row["node_id"], "task_id": row["task_id"],
            "message": (f"{row['task_id']} 本 tick 未执行："
                        + (row["reasons"][0] if row["reasons"] else "未知原因")),
            "evidence": row["evidence"],
        })
    for row in ineligible:
        findings.append({
            "code": "NOT_ELIGIBLE",
            "node_id": row["node_id"], "task_id": row["task_id"],
            "message": (f"{row['task_id']} 所在节点当前不可服务："
                        + (row["reasons"][0] if row["reasons"] else "")),
            "evidence": row["evidence"],
        })
    for row in abandoned:
        findings.append({
            "code": "ABANDONED",
            "node_id": row["node_id"], "task_id": row["task_id"],
            "message": (f"{row['task_id']} 被**主动放弃**："
                        + "；".join(row["reasons"])),
            "evidence": row["evidence"],
        })
    for row in suppressed:
        findings.append({
            "code": "DUPLICATE_SUPPRESSED",
            "node_id": row["node_id"], "task_id": row["task_id"],
            "message": (f"{row['task_id']} 因多节点重复规则被抑制："
                        + "；".join(row["reasons"])),
            "evidence": row["evidence"],
        })

    summary = (
        f"共 {len(rows)} 条调度决策：已排入 {len(planned)}、推迟 {len(deferred)}、"
        f"不可服务 {len(ineligible)}、主动放弃 {len(abandoned)}、"
        f"重复抑制 {len(suppressed)}。"
        "本解释只覆盖**已经发生**的计划与拒绝原因，不预测、不改计划。"
    )
    return Rationale(summary=summary, findings=findings, provider="local_rule",
                     note="本地规则解释（结构化证据驱动）")


def explain_schedule(
    payload: Dict[str, Any],
    remote_call: Optional[Callable[[Dict[str, Any]], Any]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Rationale:
    """解释一次调度；远程调用带**硬墙钟超时**，超时回退本地。

    `remote_call` 是"调用远程 provider"的可注入函数（生产环境里是
    `AIDiagnosisService` 的一次调用）。它**慢**与它**抛异常**是两件事：

    * 抛异常 → 立刻回退（`try/except`）；
    * 慢 → 到 `timeout_s` 就放弃等待，回退本地并把
      `explanation_timeout=True` 记进结果。

    无论哪种情况，函数都**不会**去动任何时钟：
    超时只影响本次解释，不影响仿真时间线。
    """
    started = time.perf_counter()
    if remote_call is None:
        rationale = explain_locally(payload)
        rationale.elapsed_s = time.perf_counter() - started
        return rationale

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(remote_call, payload)
        try:
            result = future.result(timeout=max(0.0, float(timeout_s)))
        except concurrent.futures.TimeoutError:
            rationale = explain_locally(payload)
            rationale.explanation_timeout = True
            rationale.elapsed_s = time.perf_counter() - started
            rationale.note = (
                f"远程解释在 {timeout_s:g}s 内没有返回 → 已放弃等待并回退本地"
                "规则解释。**超时只影响本次解释，不阻塞仿真时钟**。")
            return rationale
        except Exception as exc:  # noqa: BLE001 - provider 失败也要能继续
            rationale = explain_locally(payload)
            rationale.elapsed_s = time.perf_counter() - started
            rationale.note = f"远程解释失败（{type(exc).__name__}）→ 回退本地。"
            return rationale
    finally:
        # 不等待慢线程结束：`shutdown(wait=False)` 保证调用方立刻继续
        executor.shutdown(wait=False)

    rationale = _adapt_remote(result, payload)
    rationale.elapsed_s = time.perf_counter() - started
    return rationale


def _adapt_remote(result: Any, payload: Dict[str, Any]) -> Rationale:
    """把远程结果收进统一结构；结果不合规就回退本地。"""
    if isinstance(result, Rationale):
        return result
    if isinstance(result, dict):
        summary = str(result.get("summary", "")).strip()
        findings = list(result.get("findings", []) or [])
        if summary:
            return Rationale(summary=summary, findings=findings,
                             provider=str(result.get("provider", "remote")))
    local = explain_locally(payload)
    local.note = "远程解释返回了无法识别的结构 → 回退本地规则解释。"
    return local
