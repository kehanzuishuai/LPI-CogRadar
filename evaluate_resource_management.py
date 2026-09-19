"""资源管理阶段评估：规则基线与优化参考的逐任务对照（v4.5 大阶段一收尾）。

它回答的问题
------------
1. 规则基线与优化参考**各自排了哪些任务**（逐任务计划）；
2. 没执行的任务**为什么**没执行（拒绝原因 / 未选中原因 / 约束违反）；
3. 每种方法的**运行时间**与**计算预算**使用情况；
4. 六维评价向量对照（**不给综合分**）；
5. 优化参考的最优性声明是否**越界**（自审结果一并落盘）；
6. 阶段验收清单的 5 项条件。

概念纪律（用户明确要求）
------------------------
* "有前瞻的参考方法"**不自动等于**"理论上界"：只有完全枚举的小问题
  才谈精确最优，且只对**该问题**成立；滚动规划一律只称"优化参考"；
* 优化参考与规则基线**共用**任务定义、资源预算、信息权限、执行器；
  它不得通过读取未来真实测量、未来故障或隐藏对象状态取得优势；
* 评价一律给**向量**（完成度/及时性/估计质量/资源消耗/通信开销/计算耗时），
  综合分只是搜索内部的声明式偏好，不作为结论；
* 计算预算是硬约束，防止某个方法无限推演。

产物（`output/runs/<run_id>/resource_eval/`）
--------------------------------------------
| 文件 | 内容 |
| --- | --- |
| `metrics.csv` | 逐（场景 × 策略 × 种子）的指标与六维向量 |
| `plans_<场景>_<策略>.csv` | **逐任务计划**（时刻/节点/任务/决定/理由/优先级） |
| `rejections_<场景>_<策略>.csv` | **拒绝原因**（执行器拒绝 + 未选中 + 不可服务） |
| `violations_<场景>_<策略>.csv` | **约束违反**（执行器校验问题逐条） |
| `runtime.csv` | 逐次运行的规划耗时与计算预算使用 |
| `comparison.md` | 人读对照表（含诚实结论与已知局限） |
| `acceptance/checklist.json` / `.md` | 阶段验收清单 |
| `contract_snapshot.json` | 冻结契约快照（口径可核对） |
| `manifest.json` | run_id / 配置摘要 / 源码摘要 / 产物 sha256 |

用法
----
    python evaluate_resource_management.py                     # 默认全量
    python evaluate_resource_management.py --quick             # 少种子少场景
    python evaluate_resource_management.py --acceptance        # 只跑验收清单
    python evaluate_resource_management.py --freeze            # 重算并写出契约快照
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logging_utils import ensure_utf8_console  # noqa: E402

from resource_management.acceptance import (  # noqa: E402
    render_acceptance_markdown,
    run_acceptance,
)
from resource_management.closed_loop import run_closed_loop  # noqa: E402
from resource_management.contract_v1 import (  # noqa: E402
    CONTRACT_VERSION,
    contract_digest,
    verify_frozen,
    write_contract_doc,
)
from resource_management.optimization import (  # noqa: E402
    METRIC_KEYS,
    METRIC_BY_KEY,
)
from resource_management.scheduling import (  # noqa: E402
    BASELINE_POLICIES,
    OPTIMIZATION_POLICIES,
    SchedulerPolicy,
)
from tools.compare_schedulers import SCENES  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SEEDS: Sequence[int] = (42, 7, 13)
DEFAULT_STEPS = 24
#: 默认只跑"规则基线 + 两个优化参考"三方法（其余基线在 compare_schedulers 里）
DEFAULT_POLICIES: Sequence[SchedulerPolicy] = (
    SchedulerPolicy.RULE,
    SchedulerPolicy.ENUMERATION,
    SchedulerPolicy.ROLLING_HORIZON,
)


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------


def _write_csv(path: str, rows: Sequence[Dict[str, Any]],
               fieldnames: Optional[Sequence[str]] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys or ["empty"]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames),
                               extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flat(row.get(key)) for key in fieldnames})
    return path


def _flat(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _rejection_rows(result: Any, scene: str) -> List[Dict[str, Any]]:
    """逐条"为什么没执行"：执行器拒绝 + 调度器未选中/不可服务/放弃。"""
    rows: List[Dict[str, Any]] = []
    for entry in result.plan_log:
        for item in entry.get("rejected", []):
            rows.append({
                "scene": scene, "source": "executor",
                "time_s": entry["time_s"], "plan_id": entry["plan_id"],
                "task_id": item.get("task_id"),
                "node_id": item.get("node_id"),
                "kind": item.get("kind"),
                "reason_code": item.get("reason_code"),
                "reason": item.get("reason"),
            })
    for row in result.decisions:
        decision = row.get("decision")
        if decision == "planned":
            continue
        rows.append({
            "scene": scene,
            "source": "scheduler",
            "time_s": row.get("time_s"),
            "plan_id": "",
            "task_id": row.get("task_id"),
            "node_id": row.get("node_id"),
            "kind": row.get("kind"),
            "reason_code": row.get("deferred_reason") or decision,
            "reason": (row.get("reasons") or [""])[0],
        })
    return rows


def _violation_rows(result: Any, scene: str) -> List[Dict[str, Any]]:
    """逐条**约束违反**（执行器校验问题）。"""
    rows: List[Dict[str, Any]] = []
    for entry in result.plan_log:
        for issue in entry.get("issues", []):
            rows.append({
                "scene": scene, "time_s": entry["time_s"],
                "plan_id": entry["plan_id"],
                "scope": issue.get("scope"), "code": issue.get("code"),
                "node_id": issue.get("node_id"),
                "task_id": issue.get("task_id"),
                "detail": issue.get("detail"),
            })
    return rows


def _plan_rows(result: Any, scene: str) -> List[Dict[str, Any]]:
    """逐任务计划（含理由与结构化证据的关键字段）。"""
    rows: List[Dict[str, Any]] = []
    for row in result.decisions:
        if row.get("decision") != "planned":
            continue
        evidence = row.get("evidence") or {}
        rows.append({
            "scene": scene, "time_s": row.get("time_s"),
            "node_id": row.get("node_id"), "task_id": row.get("task_id"),
            "kind": row.get("kind"),
            "priority": row.get("priority"),
            "objective": evidence.get("objective"),
            "exact": evidence.get("exact"),
            "n_combinations_evaluated": evidence.get(
                "n_combinations_evaluated"),
            "estimated_cost": evidence.get("estimated_cost"),
            "deadline_s": evidence.get("deadline_s"),
            "reason_1": (row.get("reasons") or [""])[0],
            "reason_2": (row.get("reasons") or ["", ""])[1]
            if len(row.get("reasons") or []) > 1 else "",
        })
    return rows


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------


def evaluate(seeds: Sequence[int] = DEFAULT_SEEDS,
             steps: int = DEFAULT_STEPS,
             scenes: Optional[Sequence[str]] = None,
             policies: Sequence[SchedulerPolicy] = DEFAULT_POLICIES,
             out_dir: Optional[str] = None,
             acceptance: bool = False,
             quiet: bool = False) -> Dict[str, Any]:
    scene_names = list(scenes or ("base", "node_outage"))
    records: List[Dict[str, Any]] = []
    run_artifacts: List[Dict[str, Any]] = []

    for scene in scene_names:
        mechanisms = SCENES[scene]
        for policy in policies:
            for seed in seeds:
                result = run_closed_loop(policy, seed=seed, steps=steps,
                                         mechanisms=mechanisms)
                metrics = result.metrics
                vector = dict(metrics.get("evaluation_vector", {})
                              .get("values", {}))
                summary = result.optimizer_summary or {}
                records.append({
                    "scene": scene, "policy": policy.value, "seed": seed,
                    "n_tasks": metrics["n_tasks_total"],
                    "n_completed": metrics["n_completed"],
                    "n_completed_on_time": metrics["completed_on_time"],
                    "n_expired": metrics["n_expired"],
                    "n_abandoned": metrics["n_abandoned"],
                    "n_rejected_by_executor":
                        metrics["n_rejected_by_executor"],
                    "n_starved": metrics["n_starved"],
                    "n_deadline_violations": metrics["deadline_violations"],
                    "completion_denominator":
                        metrics["completion_denominator"],
                    "n_plans": len(result.plan_log),
                    "n_plans_partial": sum(
                        1 for row in result.plan_log
                        if row["status"] != "applied"),
                    "n_constraint_violations": sum(
                        len(row.get("issues", [])) for row in result.plan_log),
                    "planning_time_s": round(metrics["compute_time_s"], 9),
                    "planning_calls": getattr(result, "planning_calls", 0),
                    "n_exact_plans": summary.get("n_exact"),
                    "n_budget_exhausted": summary.get("n_budget_exhausted"),
                    "total_expansions": summary.get("total_expansions"),
                    "max_overrun_s": summary.get("max_overrun_s"),
                    "claim_kinds": summary.get("claim_kinds"),
                    "claim_audit_ok": (
                        all(value for key, value
                            in (summary.get("claim_audit") or {}).items()
                            if isinstance(value, bool))
                        if summary else None),
                    **{key: vector.get(key) for key in METRIC_KEYS},
                })
                run_artifacts.append({"scene": scene, "policy": policy.value,
                                      "seed": seed, "result": result})

    report = {
        "contract_version": CONTRACT_VERSION,
        "contract_digest": contract_digest(),
        "contract_frozen": verify_frozen(strict=False),
        "seeds": list(seeds),
        "steps": steps,
        "scenes": scene_names,
        "policies": [policy.value for policy in policies],
        "records": records,
        "metric_definitions": [METRIC_BY_KEY[key].to_dict()
                               for key in METRIC_KEYS],
        "comparison": _compare(records),
        "honest_notes": _honest_notes(records),
    }
    if acceptance:
        report["acceptance"] = run_acceptance(seeds=tuple(seeds), steps=steps,
                                              quick=len(seeds) == 1)
    if out_dir:
        report["artifacts"] = _write_outputs(report, run_artifacts, out_dir,
                                             quiet=quiet)
    return report


def _compare(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """逐场景比较：谁在哪些维度上更好（用**向量**，不给综合分）。"""
    out: Dict[str, Any] = {}
    for scene in sorted({row["scene"] for row in records}):
        rows = [row for row in records if row["scene"] == scene]
        by_policy: Dict[str, Dict[str, float]] = {}
        for row in rows:
            bucket = by_policy.setdefault(row["policy"],
                                          {key: 0.0 for key in METRIC_KEYS})
            for key in METRIC_KEYS:
                bucket[key] += float(row.get(key) or 0.0)
        for policy, bucket in by_policy.items():
            n = sum(1 for row in rows if row["policy"] == policy) or 1
            for key in METRIC_KEYS:
                bucket[key] /= n
        # 支配关系（按方向对齐后比较）
        dominance: Dict[str, List[str]] = {}
        for policy, bucket in by_policy.items():
            beats: List[str] = []
            for other, other_bucket in by_policy.items():
                if other == policy:
                    continue
                worse_or_equal = True
                strictly_better = False
                for key in METRIC_KEYS:
                    spec = METRIC_BY_KEY[key]
                    mine = bucket[key] if spec.direction == "higher" \
                        else -bucket[key]
                    theirs = other_bucket[key] if spec.direction == "higher" \
                        else -other_bucket[key]
                    if mine < theirs - 1e-12:
                        worse_or_equal = False
                        break
                    if mine > theirs + 1e-12:
                        strictly_better = True
                if worse_or_equal and strictly_better:
                    beats.append(other)
            dominance[policy] = beats
        out[scene] = {"means": by_policy, "dominates": dominance}
    return out


def _honest_notes(records: Sequence[Dict[str, Any]]) -> List[str]:
    """必须与结果一起报的局限性（不是免责声明，是可核对的结论）。"""
    notes = [
        ("完成度由**服务上限**封顶：执行器只支持立即执行，每个节点每 tick "
         "最多落 1 条任务，而任务由可见航迹逐 tick 派生（实测需求/能力约 6 倍）。"
         "因此完成度差异很小，**不能**单独用来判定方法优劣。"),
        ("`estimate_quality` 在当前教学沙盒里对调度**不敏感**：节点摘要每 tick "
         "**无条件**发布，中央看到的航迹新鲜度不由被调度任务决定。"
         "实测三个方法的该维度数值完全相同。这是建模缺口"
         "（共享/刷新的收益链路尚未接进资源约束），**不是**调度器的成绩或过错。"),
        ("优化参考的排名依据是**预测**向量（显式预测模型 + 声明式 rollout），"
         "而报告里的向量是**实测**向量；两者不一回事。"
         "预测模型含「航迹始终可见」等乐观假设，因此预测目标值高**不代表**"
         "实测更好——报告必须同时给出两者。"),
        ("计算耗时是**对称测量**的：规则基线与优化参考都走同一处计时，"
         "因此该维度的对比有效。优化参考的耗时高 1 个数量级是**真实代价**，"
         "必须与收益一起读。"),
        ("'有前瞻'不等于'理论上界'：只有**完全枚举**的单 tick 小问题"
         "才谈得上精确最优，且只对该问题成立。滚动规划只称优化参考。"),
        ("未通过阶段验收的 5 项条件前**不进入学习算法阶段**；"
         "通过后也**不要求**规则方法必须失败、学习方法必须胜出。"),
    ]
    # 只有当实测确认"该维度无差异"时才加这条，避免把猜测写成结论
    for scene in sorted({row["scene"] for row in records}):
        rows = [row for row in records if row["scene"] == scene]
        for key in METRIC_KEYS:
            values = {round(float(row.get(key) or 0.0), 9) for row in rows}
            if len(values) == 1 and len(rows) > 1:
                notes.append(
                    f"[实测] 场景 `{scene}` 中维度 `{key}` 在所有策略上取值**完全相同**"
                    f"（{values.pop()}），该维度无法用于区分方法。")
    return notes


def _write_outputs(report: Dict[str, Any], runs: Sequence[Dict[str, Any]],
                   out_dir: str, quiet: bool = False) -> Dict[str, Any]:
    from run_manifest import RunManifest

    manifest = RunManifest(
        tool="resource_eval",
        command="python evaluate_resource_management.py",
        config={"seeds": report["seeds"], "steps": report["steps"],
                "scenes": report["scenes"], "policies": report["policies"],
                "contract_digest": report["contract_digest"]},
        seeds=tuple(report["seeds"]), root=ROOT)
    directory = os.path.join(manifest.run_dir(), "resource_eval")
    os.makedirs(directory, exist_ok=True)
    written: List[str] = []

    # --- 汇总 ---
    metrics_path = os.path.join(directory, "metrics.csv")
    _write_csv(metrics_path, report["records"])
    written.append(metrics_path)

    runtime_path = os.path.join(directory, "runtime.csv")
    _write_csv(runtime_path, [
        {"scene": row["scene"], "policy": row["policy"], "seed": row["seed"],
         "planning_time_s": row["planning_time_s"],
         "n_plans": row["n_plans"],
         "n_exact_plans": row["n_exact_plans"],
         "n_budget_exhausted": row["n_budget_exhausted"],
         "total_expansions": row["total_expansions"],
         "max_overrun_s": row["max_overrun_s"],
         "claim_audit_ok": row["claim_audit_ok"]}
        for row in report["records"]])
    written.append(runtime_path)

    # --- 逐任务计划 / 拒绝原因 / 约束违反 ---
    for run in runs:
        stem = f"{run['scene']}_{run['policy']}_s{run['seed']}"
        for name, rows in (
                (f"plans_{stem}.csv", _plan_rows(run["result"], run["scene"])),
                (f"rejections_{stem}.csv",
                 _rejection_rows(run["result"], run["scene"])),
                (f"violations_{stem}.csv",
                 _violation_rows(run["result"], run["scene"]))):
            path = os.path.join(directory, name)
            _write_csv(path, rows)
            written.append(path)

    # --- 对照报告 ---
    md_path = os.path.join(directory, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    written.append(md_path)
    json_path = os.path.join(directory, "summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in report.items()
                   if key not in ("artifacts",)},
                  handle, ensure_ascii=False, indent=2, default=str)
    written.append(json_path)

    # --- 验收清单 ---
    acceptance = report.get("acceptance")
    if acceptance:
        acc_dir = os.path.join(directory, "acceptance")
        os.makedirs(acc_dir, exist_ok=True)
        acc_json = os.path.join(acc_dir, "checklist.json")
        with open(acc_json, "w", encoding="utf-8") as handle:
            json.dump(acceptance, handle, ensure_ascii=False, indent=2,
                      default=str)
        acc_md = os.path.join(acc_dir, "checklist.md")
        with open(acc_md, "w", encoding="utf-8") as handle:
            handle.write(render_acceptance_markdown(acceptance))
        written += [acc_json, acc_md]

    # --- 冻结契约快照 ---
    contract_path = os.path.join(directory, "contract_snapshot.json")
    write_contract_doc(contract_path)
    written.append(contract_path)

    for path in written:
        manifest.claim(path)
    for path in written:
        manifest.record(path)
    manifest_path = manifest.write()
    if not quiet:
        print(f"产物目录：{directory}")
    return {"run_id": manifest.run_id, "directory": directory,
            "manifest": manifest_path, "n_files": len(written)}


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# 资源管理阶段评估：规则基线 vs 优化参考",
        "",
        f"- 契约：`{CONTRACT_VERSION}`（摘要 `{report['contract_digest']}`，"
        f"冻结校验 {'通过' if report['contract_frozen']['ok'] else '**未通过**'}）",
        f"- 种子：{report['seeds']}；tick 数：{report['steps']}",
        f"- 场景：{report['scenes']}",
        f"- 方法：{report['policies']}",
        "",
        "## 1. 六维评价向量（逐场景 × 方法，种子均值）",
        "",
        "| 场景 | 方法 | 完成度 | 及时性 | 估计质量 | 资源消耗 | 通信(B) | 计算耗时(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scene, block in sorted(report["comparison"].items()):
        for policy, bucket in sorted(block["means"].items()):
            lines.append(
                f"| {scene} | {policy} "
                f"| {bucket['service_completion']:.4f} "
                f"| {bucket['task_timeliness']:.4f} "
                f"| {bucket['estimate_quality']:.4f} "
                f"| {bucket['resource_consumption']:.4f} "
                f"| {bucket['communication_overhead']:.0f} "
                f"| {bucket['compute_time']:.4f} |")

    lines += ["", "## 2. 服务结果与执行情况（种子均值）", "",
              "| 场景 | 方法 | 任务数 | 完成 | 按时 | 过期 | 放弃 | 执行器拒绝 "
              "| 长期未获服务 | 截止违背 | 计划数 | 非全应用计划 | 约束违反 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
              "--- | --- | --- |"]
    keys = ("n_tasks", "n_completed", "n_completed_on_time", "n_expired",
            "n_abandoned", "n_rejected_by_executor", "n_starved",
            "n_deadline_violations", "n_plans", "n_plans_partial",
            "n_constraint_violations")
    for scene in sorted({row["scene"] for row in report["records"]}):
        for policy in report["policies"]:
            rows = [row for row in report["records"]
                    if row["scene"] == scene and row["policy"] == policy]
            if not rows:
                continue
            means = {key: sum(float(row[key]) for row in rows) / len(rows)
                     for key in keys}
            lines.append("| " + " | ".join(
                [scene, policy] + [f"{means[key]:.0f}" for key in keys]) + " |")

    lines += ["", "## 3. 计算预算使用（优化参考）", "",
              "| 场景 | 方法 | 规划总耗时(s) | 完全枚举次数 | 触发预算次数 "
              "| 展开总数 | 最大超支(s) | 声明自审 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["records"]:
        if row["n_exact_plans"] is None:
            continue
        lines.append(
            f"| {row['scene']} | {row['policy']} "
            f"| {row['planning_time_s']:.4f} | {row['n_exact_plans']} "
            f"| {row['n_budget_exhausted']} | {row['total_expansions']} "
            f"| {row['max_overrun_s']:.6f} "
            f"| {'通过' if row['claim_audit_ok'] else '**未通过**'} |")

    lines += ["", "## 4. 支配关系（按方向对齐后的帕累托比较）", ""]
    for scene, block in sorted(report["comparison"].items()):
        for policy, beats in sorted(block["dominates"].items()):
            if beats:
                lines.append(f"- `{scene}`：**{policy}** 支配 {beats}")
        if not any(block["dominates"].values()):
            lines.append(f"- `{scene}`：**没有任何方法支配其他方法**"
                         "（多维取舍，这正是要用向量而非综合分的原因）")

    lines += ["", "## 5. 六维定义（逐字）", "",
              "| 维度 | 单位 | 方向 | 定义 |", "| --- | --- | --- | --- |"]
    for spec in report["metric_definitions"]:
        lines.append(f"| {spec['name_cn']} | {spec['unit']} "
                     f"| {spec['direction']} | {spec['definition']} |")

    lines += ["", "## 6. 必须一起读的局限（逐条可核对）", ""]
    for index, note in enumerate(report["honest_notes"], 1):
        lines.append(f"{index}. {note}")

    acceptance = report.get("acceptance")
    if acceptance:
        lines += ["", "## 7. 阶段验收清单", "",
                  f"- 通过：{acceptance['n_passed']} / {acceptance['n_checks']}",
                  f"- 总判定：{'**全部通过**' if acceptance['all_passed'] else '**未通过**'}",
                  ""]
        for check in acceptance["checks"]:
            lines.append(f"- [{'x' if check['ok'] else ' '}] {check['name_cn']}"
                         + ("" if check["ok"] else f"  ← {check['detail']}"))
    lines.append("")
    return "\n".join(lines)


def print_table(report: Dict[str, Any]) -> None:
    header = (f"{'场景':<16}{'方法':<17}{'完成度':>8}{'及时性':>8}{'质量':>8}"
              f"{'资源':>8}{'通信B':>8}{'耗时s':>9}{'枚举':>6}{'预算':>6}")
    print(header)
    print("-" * len(header))
    for row in report["records"]:
        print(f"{row['scene']:<16}{row['policy']:<17}"
              f"{row['service_completion']:>8.4f}"
              f"{row['task_timeliness']:>8.4f}"
              f"{row['estimate_quality']:>8.4f}"
              f"{row['resource_consumption']:>8.4f}"
              f"{row['communication_overhead']:>8.0f}"
              f"{row['planning_time_s']:>9.4f}"
              f"{(row['n_exact_plans'] if row['n_exact_plans'] is not None else 0):>6}"
              f"{(row['n_budget_exhausted'] if row['n_budget_exhausted'] is not None else 0):>6}")
    print()
    for scene, block in sorted(report["comparison"].items()):
        for policy, beats in sorted(block["dominates"].items()):
            if beats:
                print(f"  支配[{scene}]：{policy} ⊃ {beats}")
    if report.get("acceptance"):
        acc = report["acceptance"]
        print(f"  阶段验收：{acc['n_passed']}/{acc['n_checks']} "
              f"{'全部通过' if acc['all_passed'] else '未通过'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_console()
    parser = argparse.ArgumentParser(
        description="资源管理阶段评估（规则基线 vs 优化参考）")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--scenes", type=str, default="",
                        help="逗号分隔场景名；缺省 base,node_outage")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--acceptance", action="store_true",
                        help="一并运行阶段验收清单")
    parser.add_argument("--freeze", action="store_true",
                        help="只重算并写出冻结契约快照后退出")
    parser.add_argument("--quick", action="store_true",
                        help="少种子少场景（自检用）")
    args = parser.parse_args(argv)

    if args.freeze:
        frozen = verify_frozen(strict=False)
        print(f"契约版本：{CONTRACT_VERSION}")
        print(f"记录摘要：{frozen['recorded']}")
        print(f"当前摘要：{frozen['current']}")
        print(f"一致：{frozen['ok']}")
        path = write_contract_doc(os.path.join("docs", "resource_contract_v1.json"))
        print(f"快照已写出：{path}")
        if not frozen["ok"]:
            print(f"⚠ {frozen['hint']}")
            return 1
        return 0

    seeds = args.seeds
    scenes = [name.strip() for name in args.scenes.split(",") if name.strip()]
    if args.quick:
        seeds = seeds[:1]
        scenes = scenes or ["base"]
    unknown = [name for name in scenes if name not in SCENES]
    if unknown:
        parser.error(f"未知场景 {unknown}；可选 {sorted(SCENES)}")

    report = evaluate(seeds=seeds, steps=args.steps,
                      scenes=scenes or None,
                      out_dir=None if args.no_write else "output",
                      acceptance=args.acceptance or not args.no_write)
    print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
