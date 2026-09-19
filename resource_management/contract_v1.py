"""resource-contract-v1：大阶段一的**冻结契约**。

冻结什么
--------
1. **观测 schema**（字段、单位、坐标系、可见范围、来源）与它的 `SCHEMA_VERSION`；
2. **任务完成口径**（完成/过期/主动放弃/重复请求/长期未获服务，以及完成率分母）；
3. **基准配置**（节点布局、资源预算、场景机制、种子与 tick 数）；
4. **资源单位与教学成本模型**（哪些是预算单位、哪些是时间占用量）。

为什么需要"冻结"
----------------
大阶段一结束时要能回答"资源管理问题已经被定义清楚并能运行"。
如果口径还在动，那么"规则基线 vs 优化参考"的对比就随时可能变成
"两套问题定义的对比"——那不是结论，是误会。
因此这里把口径做成**可校验的摘要**：任何一处改动都会让
`verify_frozen()` 失败，必须显式升版本号并同步文档。

冻结**不等于**不能改
--------------------
改了会失败，不是不能改。正确做法是：
① 确认改动是必要的；② 升 `CONTRACT_VERSION`；
③ 更新 `docs/resource_contract_v1.json` 与 `docs/resource_contract_v1.md`；
④ 重跑基线与验收。禁止的是**悄悄改**。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List

from resource_management.observation import (
    FIELD_SPECS,
    LEGACY_OBSERVATION_MODES,
    SCHEMA_VERSION,
    field_metadata,
)
from resource_management.tasks import QUEUE_TASK_KIND_CN, QueueTaskKind, TaskStatus
from resource_management.units import (
    BUDGET_UNITS,
    DEFAULT_DURATION_S,
    TASK_KIND_CN,
    TEACHING_COST_MODEL,
    UNIT_CN,
    UNIT_MEANING,
    UNIT_SYMBOL,
    ResourceUnit,
)

#: 契约版本。**任何**冻结内容的改动都必须同时升这个号。
CONTRACT_VERSION = "resource-contract-v1"

#: 冻结摘要（对 `contract_snapshot()` 的规范化 JSON 求 sha256 的前 32 位）。
#:
#: 它由 `python -m scripts.freeze_contract`（或
#: `evaluate_resource_management.py --freeze`）生成后写回这里。
#: 之所以把摘要写死在源码里：这样"口径被改了"这件事会在**导入即校验**时
#: 暴露，而不是等到有人对比两个数字时才发现。
FROZEN_DIGEST = "380dad61d2efb15aaf6dadd532b831d1"

#: 冻结口径用到的一组固定运行参数（"基准配置"的一部分）
BASELINE_RUN = {
    #: 规则/优化参考对比用的固定种子（**少量固定种子**，不宣称统计意义）
    "seeds": [42, 7, 13],
    #: 每轮 tick 数
    "steps": 24,
    #: 单 tick 时长（秒）
    "tick_s": 1.0,
}


# ----------------------------------------------------------------------
# 快照
# ----------------------------------------------------------------------


def observation_schema_snapshot() -> Dict[str, Any]:
    """冻结观测 schema：版本 + 每字段的单位/坐标系/可见范围/来源。"""
    fields: Dict[str, Dict[str, Any]] = {}
    for name, spec in sorted(FIELD_SPECS.items()):
        fields[name] = {
            "unit": spec.unit,
            "frame": spec.frame,
            "visibility": spec.visibility,
            "provenance": spec.provenance,
            "note": spec.note,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "legacy_observation_modes": list(LEGACY_OBSERVATION_MODES),
        "n_fields": len(fields),
        "fields": fields,
    }


def completion_semantics_snapshot() -> Dict[str, Any]:
    """冻结任务完成口径（含完成率分母）。"""
    return {
        "task_statuses": [status.value for status in TaskStatus],
        "status_cn": {
            "pending": "待释放（未到释放时间）",
            "released": "已释放",
            "submitted": "已排入计划",
            "completed": "完成（计入分子）",
            "rejected": "被执行器拒绝",
            "expired": "过期（**不删除**，仍计入分母）",
            "cancelled": "主动放弃（**仍在队列**，分母不变）",
        },
        "completion_rate_denominator": (
            "完成 + 过期 + 主动放弃 + 执行器拒绝 + 长期未获服务"),
        "starvation_definition": (
            "等待超过 starvation_threshold_s，且**不是**因截止时间到期"
            "而离开队列；与'过期'互斥"),
        "expiry_definition": "now > deadline_s 且未完成/未取消",
        "abandon_definition": (
            "等待超过 abandon_after_s 且该节点仍无法服务；状态置 cancelled，"
            "任务保留在队列中"),
        "duplicate_definition": (
            "同 task_id / 同去重键已成功执行过 → 计划级致命拒绝；"
            "同一 (kind, targets) 在多个节点出现由配置规则决定"),
        "no_deletion_rule": (
            "任何失败/放弃/过期都**不得**从队列中删除任务——"
            "删掉难任务只会让完成率下降，不可能把它做高"),
    }


def resource_model_snapshot() -> Dict[str, Any]:
    """冻结资源单位与教学成本模型。"""
    return {
        "budget_units": [unit.value for unit in BUDGET_UNITS],
        "units": {
            unit.value: {
                "symbol": UNIT_SYMBOL[unit],
                "name_cn": UNIT_CN[unit],
                "meaning": UNIT_MEANING[unit],
                "is_budget_unit": unit in BUDGET_UNITS,
            }
            for unit in ResourceUnit
        },
        "teaching_cost_model": {
            kind.value: TEACHING_COST_MODEL[kind].as_dict()
            for kind in TASK_KIND_CN
        },
        "default_duration_s": {
            kind.value: float(DEFAULT_DURATION_S.get(kind, 0.0))
            for kind in TASK_KIND_CN
        },
        "task_kind_cn": {kind.value: TASK_KIND_CN[kind] for kind in TASK_KIND_CN},
        "queue_task_kind_cn": {kind.value: QUEUE_TASK_KIND_CN[kind]
                               for kind in QueueTaskKind},
        "conservation_invariant": (
            "对每个预算单位：consumed + reserved + remaining == capacity，"
            "且三者均非负"),
        "note": ("教学成本模型是**教学设定**，与真实装备参数无关；"
                 "occupancy_second 是时间占用量，**不是**预算单位"),
    }


def baseline_config_snapshot() -> Dict[str, Any]:
    """冻结基准配置：节点布局、预算、场景机制、调度配置默认值。"""
    from resource_management.closed_loop import (
        DEFAULT_MECHANISMS,
        DEFAULT_NODE_BUDGETS,
        NODE_LAYOUT,
    )
    from resource_management.scheduling import (
        BASELINE_POLICIES,
        OPTIMIZATION_POLICIES,
        SchedulingConfig,
    )

    config = SchedulingConfig()
    return {
        "node_layout": {
            node_id: {key: float(value) if isinstance(value, (int, float))
                      else value
                      for key, value in layout.items()}
            for node_id, layout in sorted(NODE_LAYOUT.items())
        },
        "node_budgets": {
            node_id: {unit.value: float(value)
                      for unit, value in sorted(
                          budgets.items(), key=lambda item: item[0].value)}
            for node_id, budgets in sorted(DEFAULT_NODE_BUDGETS.items())
        },
        "mechanisms": {
            "unavailable_windows": DEFAULT_MECHANISMS["unavailable_windows"],
            "share_policy": str(DEFAULT_MECHANISMS["share_policy"]),
            "comm": dict(DEFAULT_MECHANISMS["comm"]),
            "bias": DEFAULT_MECHANISMS["bias"],
        },
        "scheduling_defaults": {
            "task_levels": {kind.value: int(level)
                            for kind, level in sorted(
                                config.task_levels.items(),
                                key=lambda item: item[0].value)},
            "max_information_age_s": config.max_information_age_s,
            "max_sigma_position_m": config.max_sigma_position_m,
            "starvation_threshold_s": config.starvation_threshold_s,
            "abandon_after_s": config.abandon_after_s,
            "max_tasks_per_node_per_tick": config.max_tasks_per_node_per_tick,
            "max_tasks_per_tick": config.max_tasks_per_tick,
            "allow_multi_node_same_task": config.allow_multi_node_same_task,
            "charge_duplicate_as_overhead":
                config.charge_duplicate_as_overhead,
            "weights": {
                "level": config.weight_level,
                "waiting": config.weight_waiting,
                "freshness": config.weight_freshness,
                "quality": config.weight_quality,
            },
        },
        "policies": {
            "baselines": [policy.value for policy in BASELINE_POLICIES],
            "optimization_references":
                [policy.value for policy in OPTIMIZATION_POLICIES],
        },
        "baseline_run": dict(BASELINE_RUN),
        "delay_offsets_s": {
            "share": 2.0, "estimate_update": 3.0, "predefined_sample": 6.0,
        },
    }


def evaluation_snapshot() -> Dict[str, Any]:
    """冻结**评价维度定义**（6 维；不给综合分）。"""
    from resource_management.optimization import (
        EVALUATION_METRICS, ObjectiveSpec)
    return {
        "metrics": [spec.to_dict() for spec in EVALUATION_METRICS],
        "objective_default": ObjectiveSpec().describe(),
        "rule": ("评价一律给完整向量；综合分只是搜索内部的**声明式偏好**，"
                 "不得作为「方法好不好」的结论"),
    }


def contract_snapshot() -> Dict[str, Any]:
    """完整冻结快照（摘要就是它的规范化 JSON 的 sha256）。"""
    return {
        "contract_version": CONTRACT_VERSION,
        "observation_schema": observation_schema_snapshot(),
        "completion_semantics": completion_semantics_snapshot(),
        "resource_model": resource_model_snapshot(),
        "baseline_config": baseline_config_snapshot(),
        "evaluation": evaluation_snapshot(),
    }


def canonical_json(snapshot: Dict[str, Any]) -> str:
    return json.dumps(snapshot, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def contract_digest(snapshot: Dict[str, Any] | None = None) -> str:
    payload = canonical_json(snapshot if snapshot is not None
                             else contract_snapshot())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# ----------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------


class ContractFrozenError(RuntimeError):
    """冻结口径被改动（必须显式升版本号并同步文档）。"""


def verify_frozen(strict: bool = True) -> Dict[str, Any]:
    """校验当前代码的口径与冻结摘要一致。

    返回 `{"ok": bool, "recorded": ..., "current": ..., "hint": ...}`。
    `strict=True` 时不一致直接抛 `ContractFrozenError`。
    """
    current = contract_digest()
    ok = (FROZEN_DIGEST == current)
    result = {
        "contract_version": CONTRACT_VERSION,
        "recorded": FROZEN_DIGEST,
        "current": current,
        "ok": ok,
        "hint": (
            "口径已改动。若改动是有意的，请：① 升 CONTRACT_VERSION；"
            "② 重算并写回 FROZEN_DIGEST；"
            "③ 同步 docs/resource_contract_v1.json 与 "
            "docs/resource_contract_v1.md；④ 重跑基线与阶段验收。"
            "禁止悄悄改口径。"
            if not ok else "口径与冻结摘要一致"),
    }
    if strict and not ok:
        raise ContractFrozenError(result["hint"])
    return result


def write_contract_doc(path: str) -> str:
    """把冻结快照写成 JSON（供人工核对与跨版本对比）。"""
    snapshot = contract_snapshot()
    payload = {
        "contract_version": CONTRACT_VERSION,
        "digest": contract_digest(snapshot),
        "snapshot": snapshot,
    }
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


__all__ = [
    "BASELINE_RUN", "CONTRACT_VERSION", "ContractFrozenError", "FROZEN_DIGEST",
    "baseline_config_snapshot", "canonical_json", "completion_semantics_snapshot",
    "contract_digest", "contract_snapshot", "evaluation_snapshot",
    "observation_schema_snapshot", "resource_model_snapshot", "verify_frozen",
    "write_contract_doc",
]
