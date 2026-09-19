"""逐节点资源账本（审计的核心）。

验收要求
--------
> 两个节点分别在做什么、用了多少资源、为什么某项任务没有执行，
> 都能**从账本追溯**。

因此账本不是"日志"，而是**结构化记录**：每条条目都带
`node_id / task_id / plan_id / kind / 时间 / 单位增量 / 前后余额 / 原因`，
于是三个问题都能用一次过滤回答：

* 某节点在做什么 → 按 `node_id` 过滤，看 `APPLIED` 条目的 `kind` 与占用区间；
* 用了多少资源   → 按单位汇总 `delta`；
* 为什么没执行   → 看 `REJECTED_*` 条目的 `reason_code` / `reason`。

⚠️ 账本**只记录，不判定**。所有"能不能执行"的判断都在执行器里，
账本不参与决策——否则审计材料会与决策逻辑互相污染。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from resource_management.units import (
    BUDGET_UNITS,
    UNIT_CN,
    UNIT_SYMBOL,
    ResourceUnit,
    TaskKind,
    format_cost,
)

#: 账本条目类型
ENTRY_CONSUME = "consume"            # 立即消耗
ENTRY_RESERVE = "reserve"            # 预留（未来任务）
ENTRY_ACTIVATE = "activate"          # 预留转消耗
ENTRY_RELEASE = "release"            # 预留被释放（任务被取消/计划被拒）
ENTRY_REJECT = "reject"              # 任务被拒（无扣费）
ENTRY_IDLE = "idle"                  # 空闲（零成本，零采样）


@dataclass
class LedgerEntry:
    """一条账本条目。`reason` 一律用中文可读句子，能直接回答"为什么"。"""

    entry_id: str
    time_s: float
    node_id: str
    plan_id: str
    task_id: str
    kind: TaskKind
    entry_type: str
    #: 逐单位增量（消耗/预留为正；释放/拒绝为 0）
    delta: Dict[str, float] = field(default_factory=dict)
    #: 记账后的余额快照（逐单位 remaining）
    remaining_after: Dict[str, float] = field(default_factory=dict)
    reason_code: str = ""
    reason: str = ""
    produced_samples: int = 0
    information_age_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "entry_id": self.entry_id,
            "time_s": round(self.time_s, 6),
            "node_id": self.node_id,
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "entry_type": self.entry_type,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "produced_samples": self.produced_samples,
            "information_age_s": round(self.information_age_s, 6),
        }
        for unit in BUDGET_UNITS:
            payload[f"delta_{UNIT_SYMBOL[unit]}"] = round(
                self.delta.get(unit.value, 0.0), 9)
            payload[f"remaining_{UNIT_SYMBOL[unit]}"] = round(
                self.remaining_after.get(unit.value, 0.0), 9)
        return payload


class ResourceLedger:
    """逐节点账本的集合（全局一份，按 `node_id` 可过滤）。"""

    def __init__(self) -> None:
        self.entries: List[LedgerEntry] = []
        self._counter = 0

    # ------------------------------------------------------------------

    def record(
        self,
        time_s: float,
        node_id: str,
        plan_id: str,
        task_id: str,
        kind: TaskKind,
        entry_type: str,
        delta: Optional[Dict[ResourceUnit, float]] = None,
        remaining_after: Optional[Dict[ResourceUnit, float]] = None,
        reason_code: str = "",
        reason: str = "",
        produced_samples: int = 0,
        information_age_s: float = 0.0,
    ) -> LedgerEntry:
        self._counter += 1
        entry = LedgerEntry(
            entry_id=f"L{self._counter:05d}",
            time_s=float(time_s),
            node_id=node_id,
            plan_id=plan_id,
            task_id=task_id,
            kind=kind,
            entry_type=entry_type,
            delta={unit.value: float((delta or {}).get(unit, 0.0))
                   for unit in BUDGET_UNITS},
            remaining_after={unit.value: float((remaining_after or {}).get(unit, 0.0))
                             for unit in BUDGET_UNITS},
            reason_code=reason_code,
            reason=reason,
            produced_samples=int(produced_samples),
            information_age_s=float(information_age_s),
        )
        self.entries.append(entry)
        return entry

    # ------------------------------------------------------------------

    def of_node(self, node_id: str) -> List[LedgerEntry]:
        return [entry for entry in self.entries if entry.node_id == node_id]

    def rejections(self, node_id: Optional[str] = None) -> List[LedgerEntry]:
        """所有被拒条目——"为什么某项任务没有执行"的直接答案。"""
        return [entry for entry in self.entries
                if entry.entry_type == ENTRY_REJECT
                and (node_id is None or entry.node_id == node_id)]

    def totals_by_node(self) -> Dict[str, Dict[str, float]]:
        """逐节点汇总：消耗 / 预留 / 拒绝次数。"""
        totals: Dict[str, Dict[str, float]] = {}
        for entry in self.entries:
            bucket = totals.setdefault(entry.node_id, {
                **{f"consumed_{unit.value}": 0.0 for unit in BUDGET_UNITS},
                **{f"reserved_{unit.value}": 0.0 for unit in BUDGET_UNITS},
                "n_rejected": 0.0, "n_entries": 0.0,
                "n_samples_produced": 0.0,
            })
            bucket["n_entries"] += 1
            if entry.entry_type == ENTRY_CONSUME:
                for unit in BUDGET_UNITS:
                    bucket[f"consumed_{unit.value}"] += entry.delta.get(unit.value, 0.0)
            elif entry.entry_type == ENTRY_RESERVE:
                for unit in BUDGET_UNITS:
                    bucket[f"reserved_{unit.value}"] += entry.delta.get(unit.value, 0.0)
            elif entry.entry_type == ENTRY_REJECT:
                bucket["n_rejected"] += 1
            bucket["n_samples_produced"] += entry.produced_samples
        return totals

    def format_node(self, node_id: str, limit: int = 40) -> str:
        """逐节点账本的可读文本（终端演示与报告都用它）。"""
        rows = self.of_node(node_id)
        lines = [f"-------- 节点账本 {node_id}（{len(rows)} 条）--------"]
        if not rows:
            lines.append("  （无记录）")
            return "\n".join(lines)
        lines.append(f"  {'时间':>7} {'任务':<10} {'类型':<10} {'动作':<9} "
                     f"{'增量':<16} {'剩余(slot/op/B)':<20} 说明")
        for entry in rows[-limit:]:
            delta_text = format_cost({ResourceUnit(unit): value
                                      for unit, value in entry.delta.items()})
            remaining = "/".join(
                f"{entry.remaining_after.get(unit.value, 0.0):g}"
                for unit in BUDGET_UNITS
            )
            lines.append(
                f"  {entry.time_s:>7.2f} {entry.task_id:<10} "
                f"{entry.kind.value:<10} {entry.entry_type:<9} "
                f"{delta_text:<16} {remaining:<20} {entry.reason}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def write_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        rows = [entry.to_dict() for entry in self.entries]
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path
