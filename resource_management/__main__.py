"""教学演示：两个传感器节点的资源管理与统一执行（`python -m resource_management`）。

场景
----
两个节点、**不同更新周期**、不同预算，共用一个全局时钟：

| 节点 | 更新周期 | 采样时隙 | 处理配额 | 通信字节 |
| --- | --- | --- | --- | --- |
| `NODE_A` | 1.0 s | 6 | 6 | 512 |
| `NODE_B` | 2.0 s | 3 | 2 | 256 |

演示依次覆盖用户点名的六种情形，并打印：
* **每个 tick 的时间推进次数**（证明"多个节点只推进一次"）；
* **逐节点账本**（在做什么 / 用了多少 / 为什么没执行）；
* **资源守恒自检**（逐节点逐单位残差必须为 0）。
"""

from __future__ import annotations

import os
import sys

#: 本模块可独立运行（`python -m resource_management`），
#: 但控制台编码兜底在全工程只有一份实现，因此复用 `logging_utils`。
#: Windows 控制台默认 GBK，会打印 ⚠ / → 等符号，不兜底会在中途抛
#: `UnicodeEncodeError`（本工程已踩过多次）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_utils import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

from resource_management.clock import GlobalClock  # noqa: E402
from resource_management.executor import UnifiedExecutor
from resource_management.model import (
    ExecutionPlan,
    NodeState,
    ResourceBudget,
    TaskRequest,
)
from resource_management.units import (
    BUDGET_UNITS,
    UNIT_CN,
    UNIT_MEANING,
    ResourceUnit,
    TaskKind,
)


def build_world() -> UnifiedExecutor:
    clock = GlobalClock(now_s=0.0)
    executor = UnifiedExecutor(clock)
    executor.register_node(NodeState(
        node_id="NODE_A", update_period_s=1.0,
        budget=ResourceBudget(capacity={
            ResourceUnit.SAMPLE_SLOT: 6.0,
            ResourceUnit.PROCESSING_OP: 6.0,
            ResourceUnit.COMM_BYTE: 512.0,
        })))
    executor.register_node(NodeState(
        node_id="NODE_B", update_period_s=2.0,
        budget=ResourceBudget(capacity={
            ResourceUnit.SAMPLE_SLOT: 3.0,
            ResourceUnit.PROCESSING_OP: 2.0,
            ResourceUnit.COMM_BYTE: 256.0,
        })))
    return executor


def task(task_id: str, node_id: str, kind: TaskKind, now: float,
         entities=(), note="", cost=None) -> TaskRequest:
    return TaskRequest(task_id=task_id, node_id=node_id, kind=kind,
                       start_s=now, entities=tuple(entities), note=note,
                       cost=cost)


def main() -> int:
    executor = build_world()
    clock = executor.clock
    print("=" * 96)
    print("教学仿真资源管理演示（多节点 + 统一执行器）")
    print("⚠️ 资源成本是**教学模型**，与真实装备效能无关")
    print("=" * 96)
    print("单位说明：")
    for unit in BUDGET_UNITS:
        print(f"  {unit.value:<18} {UNIT_CN[unit]}：一个单位 = "
              f"{UNIT_MEANING[unit]}")
    print()

    #: 每个 tick：**先推进唯一全局时钟，再按当前时刻构造任务并提交**。
    #: 顺序反了的话（先建任务后推进），所有任务的 `start_s` 都会落后于时钟，
    #: 被"只支持立即执行"的校验合法地拒掉——第一版演示脚本就是这么写的，
    #: 结果七个计划全被拒。这个坑本身值得写进教学材料：
    #: **计划必须与提交时刻对齐**。
    #:
    #: 另外：本模型里一个节点在同一时刻只能干一件事（占用区间重叠即冲突），
    #: 因此同一 tick 内每个节点最多给一个任务；空闲任务时长为 0、不占区间。
    steps: list = [
        ("① 两节点同时采样（各用自己的更新周期）", 1.0, None, [
            ("A-sample-1", "NODE_A", TaskKind.SAMPLE, ("TGT1",)),
            ("B-sample-1", "NODE_B", TaskKind.SAMPLE, ("TGT1",)),
        ]),
        ("② 一份计划描述**两个节点**的不同任务（A 处理 / B 共享）", 2.0, None, [
            ("A-process-1", "NODE_A", TaskKind.PROCESS, ()),
            ("B-share-1", "NODE_B", TaskKind.SHARE, ()),
        ]),
        ("③ 零成本空操作：两个节点都空闲（不产生采样报告）", 2.0, None, [
            ("A-idle-1", "NODE_A", TaskKind.IDLE, ()),
            ("B-idle-1", "NODE_B", TaskKind.IDLE, ()),
        ]),
        ("④ 重复提交去重：把 ① 的两个任务原样再交一次", 2.0, None, [
            ("A-sample-1", "NODE_A", TaskKind.SAMPLE, ("TGT1",)),
            ("B-sample-1", "NODE_B", TaskKind.SAMPLE, ("TGT1",)),
        ]),
        ("⑤ NODE_B 不可用（只影响 B，A 照常执行）", 2.0, "disable_b", [
            ("A-sample-2", "NODE_A", TaskKind.SAMPLE, ("TGT1",)),
            ("B-sample-2", "NODE_B", TaskKind.SAMPLE, ("TGT1",)),
        ]),
        ("⑥ 非法对象：未知节点（计划级致命 → 整份不执行、零扣费）", 2.0, None, [
            ("X-sample-1", "NODE_X", TaskKind.SAMPLE, ()),
            ("A-sample-3", "NODE_A", TaskKind.SAMPLE, ("TGT1",)),
        ]),
        # ⑦ 用一个**明显超出余量**的请求演示"节点局部资源不足"。
        # 注意：不能在同一节点同一时刻放两个任务——那会触发**占用冲突**
        # （计划级致命），把别的节点也一起拒掉，反而看不出"只影响该节点"。
        ("⑦ 资源不足：A 要 99 个采样时隙（只拒 A，B 照常执行）", 2.0, "enable_b", [
            ("A-oversized", "NODE_A", TaskKind.SAMPLE, ("TGT1",),
             {"sample_slot": 99.0, "processing_op": 0.0, "comm_byte": 0.0}),
            ("B-sample-3", "NODE_B", TaskKind.SAMPLE, ("TGT1",), None),
        ]),
    ]

    step_count_before = clock.step_count
    total_tasks = 0
    for index, (title, dt, action, specs) in enumerate(steps):
        clock.advance(dt, title)
        if action == "disable_b":
            executor.set_availability("NODE_B", False, "通信机故障（教学示例）")
        elif action == "enable_b":
            executor.set_availability("NODE_B", True)
        tasks = []
        for spec in specs:
            task_id, node_id, kind, entities = spec[:4]
            cost = spec[4] if len(spec) > 4 else None
            tasks.append(task(task_id, node_id, kind, clock.now_s,
                              entities=entities, cost=cost))
        total_tasks += len(tasks)
        plan = ExecutionPlan(plan_id=f"P{index + 1:02d}",
                             submit_time_s=clock.now_s, tasks=tasks)
        result = executor.submit(plan)
        print(f"--- {title}")
        print(f"    时钟 t={clock.now_s:g}s｜状态={result.status.value}"
              f"｜执行 {result.n_applied} / 拒绝 {result.n_rejected}")
        for outcome in result.outcomes:
            mark = "OK " if outcome.outcome.value == "applied" else "REJ"
            print(f"      [{mark}] {outcome.node_id} {outcome.task_id:<12} "
                  f"{outcome.kind.value:<8} {outcome.reason}")
        for issue in result.issues:
            if issue.task_id in {o.task_id for o in result.outcomes}:
                continue          # 已在上面的逐任务行里用可读理由显示
            print(f"      · [{issue.scope.value}] {issue.task_id or '-'}："
                  f"{issue.detail}")
        if result.n_applied == 0 and not result.outcomes:
            print("      （整份计划未执行：零扣费、状态零污染）")
        print()

    print("=" * 96)
    print(f"时间推进：{len(steps)} 个 tick 共调用 clock.advance "
          f"{clock.step_count - step_count_before} 次"
          f"（与节点数 2、任务数 {total_tasks} 无关：**一次 tick 只推进一次**）")
    print("=" * 96)

    for node_id in ("NODE_A", "NODE_B"):
        print()
        print(executor.ledger.format_node(node_id))

    print()
    print("=" * 96)
    print("逐节点资源汇总（账本汇总，单位已标注）")
    print("=" * 96)
    totals = executor.ledger.totals_by_node()
    for node_id, bucket in sorted(totals.items()):
        parts = []
        for unit in BUDGET_UNITS:
            parts.append(f"消耗{bucket[f'consumed_{unit.value}']:g}"
                         f"{unit.value.split('_')[0]}")
        print(f"  {node_id}: " + "，".join(parts)
              + f"；拒绝 {bucket['n_rejected']:g} 次；"
              f"产生采样报告 {bucket['n_samples_produced']:g} 条")

    print()
    report = executor.conservation_report()
    print("资源守恒自检（容量 = 消耗 + 预留 + 剩余，残差必须为 0）：")
    for node_id, entry in sorted(report["nodes"].items()):
        print(f"  {node_id}: conserved={entry['conserved']} "
              f"residual={entry['residual']}")
    print(f"  全部守恒：{report['all_conserved']}")

    print()
    print("信息年龄（空闲不刷新估计，年龄随时钟增长）：")
    for node_id in ("NODE_A", "NODE_B"):
        node = executor.node(node_id)
        ages = node.information_ages(clock.now_s)
        print(f"  {node_id}: " + (", ".join(
            f"{entity} 年龄 {age:.1f}s" for entity, age in sorted(ages.items()))
            or "（无历史估计）"))

    # ---------- 落盘：逐节点资源账本 + 执行日志（带 run_id 隔离） ----------
    from resource_management.reporting import write_artifacts

    artifacts = write_artifacts(executor, out_dir="output")
    print()
    print("=" * 96)
    print(f"run_id            ：{artifacts['run_id']}")
    print(f"账本（逐条）      ：{artifacts['ledger_csv']}")
    print(f"账本（逐节点汇总）：{artifacts['by_node_csv']}")
    print(f"执行日志          ：{artifacts['execution_log_csv']}")
    print(f"节点状态          ：{artifacts['nodes_json']}")
    print(f"运行清单          ：{artifacts['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
