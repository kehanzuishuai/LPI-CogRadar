"""非学习型资源调度基线对比（v4.5 · 资源管理阶段）。

验收口径（用户明确）
--------------------
> 系统确实会产生**不同分工**，而且每次分工都有**原因和实际执行记录**。

因此本工具不比"谁更好"，而是回答三件事：
1. 三个策略在**同一份**可见观测、同一个任务队列、同一个执行器下，
   是否产生**不同的分工**；
2. 每个策略的分工差异是否有**可读原因**（逐条决策的理由与结构化证据）；
3. 分工是否**真的被执行**（计划日志、账本、任务时间线三者能对上）。

本工具**不预设**规则调度必须赢。事实上在 24 tick、每节点每 tick 限额 1 的
默认配置下，三个基线的完成率可能完全相同——这属于诚实结论，照实记录。
"为了让规则调度显得有效而挑选指标"是被禁止的。

产物（落在 `output/runs/<run_id>/scheduler_baselines/`）
------------------------------------------------------
| 文件 | 内容 |
| --- | --- |
| `summary.json` | 全部场景 × 策略 × 种子的逐次指标与聚合 |
| `summary.md` | 人读的对比表（含"分工差异"与"执行证据"两节） |
| `decisions_<scene>_<policy>.csv` | **逐条决策**：时刻/节点/任务/决定/理由/优先级 |
| `timeline_<scene>_<policy>.csv` | **逐节点任务时间线**（分工与执行记录） |
| `node_ages_<scene>_<policy>.csv` | 逐 tick 观测到达年龄/内容年龄/可见航迹数 |
| `manifest.json` | run_id / 配置摘要 / 源码摘要 / 产物 sha256 |

用法
----
    python tools/compare_schedulers.py                    # 默认场景 × 3 策略
    python tools/compare_schedulers.py --seeds 42 7 13
    python tools/compare_schedulers.py --scenes base,node_outage
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import ensure_utf8_console  # noqa: E402

from communication import SHARE_CONSTRAINED  # noqa: E402
from resource_management.closed_loop import (  # noqa: E402
    NODE_LAYOUT,
    run_closed_loop,
)
from resource_management.scheduling import (  # noqa: E402
    SchedulerPolicy,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SEEDS: Sequence[int] = (42, 7, 13)
DEFAULT_STEPS = 24

#: 三个非学习基线（顺序即报告顺序）
POLICIES: Sequence[SchedulerPolicy] = (
    SchedulerPolicy.ROUND_ROBIN,
    SchedulerPolicy.EDF,
    SchedulerPolicy.RULE,
)

#: 场景机制集合。全部使用**已有机制**，不新增物理：
#: - `base`：理想共享、无节点故障、无观测偏差（对照）
#: - `constrained_comm`：受限链路 + 1.2s 基础延迟（信息变旧）
#: - `node_outage`：NODE_B 在 [3,9] 不可用（覆盖交班）
#: - `bias`：NODE_B 观测偏差（估计质量变差）
SCENES: Dict[str, Dict[str, Any]] = {
    "base": {},
    "constrained_comm": {
        "share_policy": SHARE_CONSTRAINED,
        "comm": {"base_delay_s": 1.2, "jitter_s": 0.4, "loss_prob": 0.0,
                 "expiry_s": 10.0},
    },
    "node_outage": {"unavailable_windows": {"NODE_B": [(3.0, 9.0)]}},
    "bias": {"bias": {"NODE_B": {"range_bias_m": 150.0, "az_bias_deg": 0.8,
                                 "noise_underreport_factor": 0.4}}},
}

#: 汇总到对比表的标量指标
SCALAR_METRICS = (
    "n_tasks_total", "n_completed", "n_expired", "n_abandoned",
    "n_rejected_by_executor", "n_starved", "completion_rate",
    "deadline_violations", "mean_waiting_s", "max_waiting_s",
    "comm_overhead_bytes",
)

#: 由逐节点统计**求和**得到的指标（不在 `metrics` 顶层，需单独聚合）
DERIVED_METRICS = ("n_planned", "n_deferred_or_not_eligible")


# ----------------------------------------------------------------------
# 落盘小工具
# ----------------------------------------------------------------------


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    """写 CSV；没有行时也写出表头为空文件（便于"确实没有记录"被看见）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["empty"],
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _flatten(v) for k, v in row.items()})
    return path


def _flatten(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------


def run_comparison(seeds: Sequence[int] = DEFAULT_SEEDS,
                   steps: int = DEFAULT_STEPS,
                   scenes: Optional[Sequence[str]] = None,
                   policies: Sequence[SchedulerPolicy] = POLICIES,
                   out_dir: Optional[str] = None,
                   quiet: bool = False) -> Dict[str, Any]:
    """跑完全部场景 × 策略 × 种子，返回汇总（并落盘，除非 `out_dir` 为空）。"""
    scene_names = list(scenes or SCENES.keys())
    records: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []

    for scene in scene_names:
        mechanisms = SCENES[scene]
        for policy in policies:
            per_seed: List[Dict[str, Any]] = []
            for seed in seeds:
                result = run_closed_loop(
                    policy, seed=seed, steps=steps, mechanisms=mechanisms,
                    keep_decisions=True)
                metrics = dict(result.metrics)
                row = {
                    "scene": scene, "policy": policy.value, "seed": seed,
                    **{key: metrics.get(key) for key in SCALAR_METRICS},
                    "n_planned": sum(
                        bucket["n_planned"]
                        for bucket in metrics["per_node"].values()),
                    "n_deferred_or_not_eligible": sum(
                        bucket["n_deferred_or_not_eligible"]
                        for bucket in metrics["per_node"].values()),
                    "kinds_planned": {
                        node_id: bucket["kinds_planned"]
                        for node_id, bucket in metrics["per_node"].items()},
                    "by_kind_status": result.queue_summary["by_kind_status"],
                    "service_capacity": metrics.get("service_capacity"),
                    "conservation_all": metrics.get("conservation_all"),
                }
                per_seed.append(row)
                runs.append({
                    "scene": scene, "policy": policy.value, "seed": seed,
                    "metrics": metrics, "result": result})

            aggregate = {"scene": scene, "policy": policy.value,
                         "n_seeds": len(per_seed)}
            for key in SCALAR_METRICS + DERIVED_METRICS:
                values = [row[key] for row in per_seed
                          if isinstance(row.get(key), (int, float))]
                aggregate[key] = (statistics.fmean(values) if values else None)
                aggregate[f"{key}_std"] = (
                    statistics.pstdev(values) if len(values) > 1 else 0.0)
            # 分工差异：逐节点的**任务种类集合**，直接对照"分工是否不同"
            aggregate["kinds_planned_by_node"] = _merge_kinds(per_seed)
            aggregate["service_capacity"] = per_seed[0].get("service_capacity")
            aggregate["per_seed"] = per_seed
            records.append(aggregate)

    summary = {
        "seeds": list(seeds),
        "steps": steps,
        "scenes": scene_names,
        "policies": [policy.value for policy in policies],
        "records": records,
        "work_division_differs": _work_division_differs(records),
        "seeds_affect_result": _seeds_affect_result(records),
        "honesty_note": (
            "本表不预设任何策略更好。默认配置下三个基线的完成率可能相同——"
            "差异体现在**分工方式与等待时间**上，而非完成率。"
            "禁止通过删除难任务来抬高完成率：完成率分母含过期/放弃/拒绝/饿死。"),
    }

    if out_dir:
        summary["artifacts"] = _write_outputs(summary, runs, out_dir,
                                              seeds, steps, scene_names,
                                              policies, quiet=quiet)
    return summary


def _merge_kinds(per_seed: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    merged: Dict[str, set] = {}
    for row in per_seed:
        for node_id, kinds in (row.get("kinds_planned") or {}).items():
            merged.setdefault(node_id, set()).update(kinds)
    return {node_id: sorted(kinds) for node_id, kinds in sorted(merged.items())}


def _work_division_differs(records: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    """逐场景判断：三个策略的分工（节点 → 任务种类集合）是否两两不同。"""
    out: Dict[str, bool] = {}
    for scene in {row["scene"] for row in records}:
        signatures = {
            row["policy"]: json.dumps(row["kinds_planned_by_node"],
                                      sort_keys=True, ensure_ascii=False)
            for row in records if row["scene"] == scene}
        out[scene] = len(set(signatures.values())) > 1
    return out


def _seeds_affect_result(records: Sequence[Dict[str, Any]]
                         ) -> Dict[str, bool]:
    """逐场景判断：种子是否真的改变了结果。

    实测（默认闭环场景）：**只有带链路抖动的 `constrained_comm` 会变**。
    `base` / `node_outage` / `bias` 逐种子的全部指标完全相同——因为检测被强制
    成功（`force_detection=True`）且虚警率为 0，噪声只影响量测精度、不改变
    「哪条航迹可见」，于是任务集合、截止时间与调度决策都与种子无关；
    只有抖动延迟会改变摘要到达时刻，进而改变任务集合。

    这一点必须**逐场景**说清楚：写成全局布尔会把"这三个场景的多种子不是
    独立重复实验"这个事实藏掉，让读者以为整张表都有统计意义。
    """
    out: Dict[str, bool] = {}
    for scene in {row["scene"] for row in records}:
        varies = False
        for row in records:
            if row["scene"] != scene:
                continue
            for key, value in row.items():
                if (key.endswith("_std") and isinstance(value, (int, float))
                        and value > 0.0):
                    varies = True
        out[scene] = varies
    return dict(sorted(out.items()))


def _write_outputs(summary: Dict[str, Any], runs: Sequence[Dict[str, Any]],
                   out_dir: str, seeds: Sequence[int], steps: int,
                   scene_names: Sequence[str],
                   policies: Sequence[SchedulerPolicy],
                   quiet: bool = False) -> Dict[str, Any]:
    from run_manifest import RunManifest

    manifest = RunManifest(
        tool="scheduler_baselines",
        command="python tools/compare_schedulers.py",
        config={"scenes": {name: _stringify(SCENES[name])
                           for name in scene_names},
                "policies": [p.value for p in policies],
                "steps": steps,
                "node_layout": NODE_LAYOUT},
        seeds=tuple(seeds), root=ROOT)
    directory = os.path.join(manifest.run_dir(), "scheduler_baselines")
    os.makedirs(directory, exist_ok=True)
    written: List[str] = []

    json_path = manifest.artifact_path(
        os.path.join("scheduler_baselines", "summary.json"))
    manifest.claim(json_path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in summary.items() if k != "artifacts"},
                  handle, ensure_ascii=False, indent=2)
    written.append(json_path)

    md_path = manifest.artifact_path(
        os.path.join("scheduler_baselines", "summary.md"))
    manifest.claim(md_path)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(summary))
    written.append(md_path)

    for run in runs:
        result = run["result"]
        stem = f"{run['scene']}_{run['policy']}"
        decisions = _write_csv(
            os.path.join(directory, f"decisions_{stem}.csv"),
            result.decisions)
        timeline = _write_csv(
            os.path.join(directory, f"timeline_{stem}.csv"),
            [dict(row, node_id=node_id)
             for node_id, rows in sorted(result.timelines.items())
             for row in rows])
        ages = _write_csv(
            os.path.join(directory, f"node_ages_{stem}.csv"),
            result.node_ages)
        for path in (decisions, timeline, ages):
            manifest.claim(path)
            written.append(path)

    for path in written:
        manifest.record(path)
    manifest_path = manifest.write()
    if not quiet:
        print(f"产物目录：{directory}")
    return {"run_id": manifest.run_id, "directory": directory,
            "manifest": manifest_path, "n_files": len(written)}


def _stringify(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: (str(value) if not isinstance(value, (int, float, str, bool,
                                                       type(None)))
                  else value)
            for key, value in sorted(config.items())}


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------


def render_markdown(summary: Dict[str, Any]) -> str:
    seeds = summary["seeds"]
    lines = [
        "# 非学习型资源调度基线对比",
        "",
        f"- 种子：{list(seeds)}（共 {len(seeds)} 个，"
        "样本量小，只报均值与总体标准差，**不宣称置信区间**）",
        f"- 每轮 tick 数：{summary['steps']}",
        f"- 策略：{summary['policies']}",
        "",
    ]
    if summary.get("seeds_affect_result"):
        invariant = [scene for scene, varies
                     in summary["seeds_affect_result"].items() if not varies]
        if invariant:
            lines += [
                "> ⚠️ **下列场景的种子不影响结果**：" + "、".join(
                    f"`{scene}`" for scene in invariant)
                + "。这些场景里逐种子的全部指标完全相同（标准差全为 0）——"
                "检测被强制成功且虚警率为 0，噪声只影响量测精度、不改变"
                "「哪条航迹可见」。因此这些场景的多行种子**不构成统计证据**，"
                "不能当成多次独立重复实验；只有逐种子有差异的场景"
                "（通常是带链路抖动的那一个）才谈得上重复。",
                "",
            ]
    lines += [
        "## 1. 逐场景指标（种子均值）",
        "",
        "| 场景 | 策略 | 任务数 | 完成 | 完成率 | 过期 | 放弃 | 执行器拒绝 | "
        "长期未获服务 | 截止违背 | 平均等待(s) | 通信开销(B) | 已规划数 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- |",
    ]
    for row in summary["records"]:
        lines.append(
            f"| {row['scene']} | {row['policy']} | {_f(row['n_tasks_total'])} "
            f"| {_f(row['n_completed'])} | {_f(row['completion_rate'], 4)} "
            f"| {_f(row['n_expired'])} | {_f(row['n_abandoned'])} "
            f"| {_f(row['n_rejected_by_executor'])} | {_f(row['n_starved'])} "
            f"| {_f(row['deadline_violations'])} "
            f"| {_f(row['mean_waiting_s'], 3)} "
            f"| {_f(row['comm_overhead_bytes'])} | {_f(row['n_planned'])} |")

    lines += ["", "## 2. 分工差异（验收重点）", "",
              "| 场景 | 策略 | 各节点实际承接的任务种类 |", "| --- | --- | --- |"]
    for row in summary["records"]:
        kinds = row["kinds_planned_by_node"]
        text = "；".join(f"{node}: {'/'.join(v) or '（无）'}"
                        for node, v in kinds.items())
        lines.append(f"| {row['scene']} | {row['policy']} | {text} |")

    lines += ["", "### 是否产生不同分工", ""]
    for scene, differs in summary["work_division_differs"].items():
        lines.append(f"- `{scene}`：{'是——三策略分工不完全相同' if differs else '**否——三策略分工相同**'}")

    lines += ["", "## 2b. 逐任务类型服务分布（首个种子，完成/过期）", "",
              "只看总完成率看不出「哪一类服务被系统性跳过」。", "",
              "| 场景 | 策略 | 采样 完成/过期 | 更新 完成/过期 | 处理 完成/过期 "
              "| 共享 完成/过期 |", "| --- | --- | --- | --- | --- | --- |"]
    for row in summary["records"]:
        per_seed = row["per_seed"][0]
        table = per_seed.get("by_kind_status") or {}

        def cell(kind: str) -> str:
            entry = table.get(kind) or {}
            return f"{entry.get('completed', 0)}/{entry.get('expired', 0)}"

        lines.append(
            f"| {row['scene']} | {row['policy']} | {cell('predefined_sample')} "
            f"| {cell('estimate_update')} | {cell('process')} "
            f"| {cell('share')} |")

    lines += ["", "## 2c. 服务上限 vs 需求（首个种子）", "",
              "| 场景 | 策略 | 节点×tick×限额 | 已规划 | 服务利用率 | 派生任务数 "
              "| 需求/tick | 需求/能力 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary["records"]:
        capacity = row["per_seed"][0].get("service_capacity") or {}
        if not capacity:
            continue
        lines.append(
            f"| {row['scene']} | {row['policy']} "
            f"| {capacity.get('max_serviceable_tasks')} "
            f"| {capacity.get('n_planned')} "
            f"| {_f(capacity.get('service_utilization'), 3)} "
            f"| {capacity.get('n_tasks_created')} "
            f"| {_f(capacity.get('demand_per_tick'), 2)} "
            f"| {_f(capacity.get('demand_over_capacity'), 2)} |")
    lines += ["",
              "**需求/能力 > 1 时，完成率由服务上限封顶**：本场景实测该比值约 6.3，"
              "因此完成率约 0.16~0.20 是供需关系的直接结果，"
              "**不能**用来说明某个策略更差。策略差异体现在"
              "「这 48 个执行名额给了哪一类服务」和等待/截止违背上。"]

    lines += [
        "",
        "## 3. 诚实结论",
        "",
        summary["honesty_note"],
        "",
        "读表提醒：",
        "",
        "1. **完成率相同不代表调度无效**。本版本执行器只支持立即执行，因此每个"
        "节点每 tick 最多落一个任务；任务却由可见航迹逐 tick 派生，需求约为服务"
        "上限的 6 倍。完成量由服务上限封顶，策略只改变**先做哪一件**，"
        "于是差异出现在等待时间、截止违背与任务种类上（见 §2b/§2c）。",
        "2. **规则调度可能把某一类服务永久饿死**。实测 `rule` 在默认场景下"
        "把 48 个名额全部给了等级最高的 `estimate_update`，`share` 完成 0 条"
        "（42 条过期）、`predefined_sample` 完成 0 条——因为 `share` 不绑定航迹，"
        "没有信息年龄与协方差项，只有等级分，而所有任务在同一 tick 内等待时间"
        "相同，于是它永远排不到前面。这是**如实记录的设计局限**，不是要掩饰的"
        "问题；要改变它需要给「从未被服务的服务类别」单独加项。",
        "3. **EDF 的截止违背数可能最多**，这不是笔误：EDF 优先做「最紧急」的任务，"
        "但紧急任务往往正是预算最吃紧、最难按时做完的那些。照实报告。",
        "4. **长期未获服务（饿死）与过期是互斥的两条语义**。默认场景里所有任务都"
        "带 2~6s 截止余量，而饿死阈值是 8s，因此任务总是先过期、"
        "`n_starved` 恒为 0——这不代表语义不存在，而是两条语义在本配置下不会"
        "同时触发；该语义由定向用例单独验证（见测试 "
        "`test_starvation_and_expiry_are_mutually_exclusive`）。",
        "5. **种子影响是分场景的**（见 §0 警示）：默认闭环里只有带链路抖动的场景"
        "逐种子有差异，其余场景逐种子完全同值，因此不能把「3 个种子」当作"
        "统计重复。要做统计意义上的策略比较，必须先打开概率检测/虚警"
        "（`force_detection=False`、`false_alarm_rate>0`）或随机化场景。",
        "6. 任何「某策略更好」的说法都必须附上**具体指标 + 该指标的取样方式**；"
        "没有逐条决策记录支撑的分工描述一律不算结论。",
        "",
    ]
    return "\n".join(lines)


def _f(value: Any, digits: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(summary: Dict[str, Any]) -> None:
    header = (f"{'场景':<18}{'策略':<12}{'完成':>5}{'完成率':>9}{'过期':>6}"
              f"{'放弃':>6}{'违背':>6}{'等待s':>8}{'通信B':>9}{'规划':>6}")
    print(header)
    print("-" * len(header))
    for row in summary["records"]:
        print(f"{row['scene']:<18}{row['policy']:<12}"
              f"{_f(row['n_completed']):>5}"
              f"{_f(row['completion_rate'], 3):>9}"
              f"{_f(row['n_expired']):>6}{_f(row['n_abandoned']):>6}"
              f"{_f(row['deadline_violations']):>6}"
              f"{_f(row['mean_waiting_s'], 2):>8}"
              f"{_f(row['comm_overhead_bytes']):>9}"
              f"{_f(row['n_planned']):>6}")
    print()
    for scene, differs in summary["work_division_differs"].items():
        print(f"  分工是否因策略而异 [{scene}]：{'是' if differs else '否'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_console()
    parser = argparse.ArgumentParser(description="非学习型调度基线对比")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--scenes", type=str, default="",
                        help="逗号分隔的场景名；缺省跑全部")
    parser.add_argument("--no-write", action="store_true",
                        help="只打印，不落盘（自检用）")
    args = parser.parse_args(argv)

    scenes = [name.strip() for name in args.scenes.split(",") if name.strip()]
    unknown = [name for name in scenes if name not in SCENES]
    if unknown:
        parser.error(f"未知场景 {unknown}；可选 {sorted(SCENES)}")

    summary = run_comparison(seeds=args.seeds, steps=args.steps,
                             scenes=scenes or None,
                             out_dir=None if args.no_write else "output")
    print_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
