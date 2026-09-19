"""资源账本与执行日志落盘（复用工程的运行隔离契约）。

产物（全部落在 `output/runs/<run_id>/resource_management/` 下）
----------------------------------------------------------------
| 文件 | 内容 |
| --- | --- |
| `ledger.csv` | **逐条**账本（一行 = 一次消耗/预留/拒绝/空闲） |
| `ledger_by_node.csv` | **逐节点**汇总（用了多少资源、拒绝几次、产出几条报告） |
| `execution_log.csv` | **逐计划**执行日志（状态、执行/拒绝数、涉及节点） |
| `nodes.json` | 各节点最终状态（可用性、占用区间、预算、信息年龄） |
| `manifest.json` | run_id / 配置摘要 / 源码摘要 / 产物清单（含 sha256） |

为什么复用 `run_manifest`：本工程刚建立"不同实验不得覆盖同一份汇总报告"
的纪律（见 `docs/data_contract.md` §3.3）。教学模块同样适用——
账本是结论的支撑材料，必须能回答"这份账本是哪次跑出来的"。
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from resource_management.units import BUDGET_UNITS, ResourceUnit


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
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


def write_artifacts(executor: Any, out_dir: str,
                    command: str = "python -m resource_management",
                    config: Optional[Dict[str, Any]] = None,
                    isolate_run: bool = True) -> Dict[str, Any]:
    """把账本、逐节点汇总、执行日志与节点状态写到磁盘。

    `isolate_run=True` 时落在 `output/runs/<run_id>/resource_management/`，
    并写 `manifest.json`（run_id / 配置摘要 / 源码摘要 / 产物清单）。
    """
    from run_manifest import RunManifest

    manifest = RunManifest(
        tool="resource_management",
        command=command,
        config=config or {
            "nodes": sorted(executor.nodes),
            "capacities": {
                node_id: {unit.value: node.budget.capacity.get(unit, 0.0)
                          for unit in BUDGET_UNITS}
                for node_id, node in sorted(executor.nodes.items())
            },
        },
        seeds=(),
    )
    directory = os.path.join(manifest.run_dir() if isolate_run else out_dir,
                             "resource_management")
    os.makedirs(directory, exist_ok=True)
    now = executor.clock.now_s

    ledger_path = _write_csv(
        os.path.join(directory, "ledger.csv"),
        [entry.to_dict() for entry in executor.ledger.entries])

    totals = executor.ledger.totals_by_node()
    by_node_rows: List[Dict[str, Any]] = []
    for node_id, node in sorted(executor.nodes.items()):
        bucket = totals.get(node_id, {})
        row: Dict[str, Any] = {
            "node_id": node_id,
            "available": node.available,
            "unavailable_reason": node.unavailable_reason,
            "update_period_s": node.update_period_s,
            "submitted": node.stats["submitted"],
            "applied": node.stats["applied"],
            "rejected": node.stats["rejected"],
            "max_information_age_s": round(node.max_information_age_s(now), 6),
        }
        for unit in BUDGET_UNITS:
            key = unit.value
            row[f"capacity_{key}"] = node.budget.capacity.get(unit, 0.0)
            row[f"consumed_{key}"] = node.budget.consumed.get(unit, 0.0)
            row[f"reserved_{key}"] = node.budget.reserved.get(unit, 0.0)
            row[f"remaining_{key}"] = round(node.budget.remaining(unit), 9)
            row[f"conserved_{key}"] = abs(
                node.budget.conservation_residual()[key]) <= 1e-9
        row["consumed_sample_slots_total"] = bucket.get(
            f"consumed_{ResourceUnit.SAMPLE_SLOT.value}", 0.0)
        by_node_rows.append(row)
    by_node_path = _write_csv(
        os.path.join(directory, "ledger_by_node.csv"), by_node_rows)

    log_path = _write_csv(
        os.path.join(directory, "execution_log.csv"), executor.plan_log)

    nodes_path = os.path.join(directory, "nodes.json")
    with open(nodes_path, "w", encoding="utf-8") as handle:
        json.dump({
            "now_s": now,
            "nodes": {node_id: node.to_dict(now)
                      for node_id, node in sorted(executor.nodes.items())},
            "conservation": executor.conservation_report(),
        }, handle, ensure_ascii=False, indent=2)

    for path in (ledger_path, by_node_path, log_path, nodes_path):
        manifest.record(path)
    manifest_path = manifest.write()

    return {
        "run_id": manifest.run_id,
        "directory": directory,
        "ledger_csv": ledger_path,
        "by_node_csv": by_node_path,
        "execution_log_csv": log_path,
        "nodes_json": nodes_path,
        "manifest": manifest_path,
    }
