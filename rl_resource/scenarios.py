"""场景目录 → 真闭环机制的映射（学习型调度器的环境构造）。

⚠️ 这个映射是**本轮首次实现**，必须说明白
-------------------------------------------
`config/learning_splits_v1.json` 里的 `scenario_catalog` 在此之前**只被校验、
从未被任何环境消费**（`resource_management/learning_env.py` 是一个独立的纯队列
校核环境，不读场景目录）。因此"`load_multiplier` 到底意味着什么"此前**没有定义**。

本模块第一次为它给出可执行的语义。为了让这件事可追溯：

* 映射**不修改** `learning_splits_v1.json`（那是摘要受保护的冻结登记表）；
* 映射本身由 `SCENARIO_MAPPING_VERSION` + `mapping_digest()` 单独标识，
  改动映射会让摘要变化；
* 每个字段的语义、理由与**限制**在下表里逐条写清，不留下"大概是这个意思"。

字段映射
--------
| 目录字段 | 映射到闭环的什么 | 说明与限制 |
| --- | --- | --- |
| `budget_multiplier` | `node_budgets` 各单位的容量 × 该系数 | 直接、无歧义 |
| `node_outage_windows` | `mechanisms["unavailable_windows"]` | 目录里是**tick 序号**；闭环用秒，本工程的 tick 固定为 1 s，故 1:1 映射 |
| `comm_delay_ticks` | `mechanisms["comm"]["base_delay_s"]` | 同样 1 tick = 1 s |
| `comm_drop_probability` | `mechanisms["comm"]["loss_prob"]` | 触发受限共享链路（`SHARE_CONSTRAINED`） |
| `sensor_bias_sigma_multiplier` | 远端节点 NODE_B 的传感器偏置 | 见下 |
| `load_multiplier` | **各任务类型的截止余量 ÷ 该系数** | ⚠️ 见下，这是**服务压力**而非任务数量 |

三条必须写明的限制
------------------

1. **`load_multiplier` 不是"任务数量倍数"**。真闭环里每节点每 tick 的候选任务
   类型上限就是 3 种（sample / process / share），目标数只影响 SHARE 是否可用
   （实测目标数 1→3 只让任务总数从 54 变到 56）。所以"多 40% 的任务"在结构上
   做不到。本实现把它定义为**服务压力**：截止余量 ÷ m，即同样的容量面对
   "更早失去意义的任务"。m>1 更难，m<1 更容易。**不得**把它读成"任务多了 40%"。

2. **`sensor_bias_sigma_multiplier` 实现为"来源不一致"而不是"噪声更大"**。
   场景名是 `rm_validation_sensor_bias`（偏差），而传感器配置里的可用旋钮是
   偏置项而非 σ 缩放。因此 m>0 时给远端节点注入
   `range_bias_m = 150·m`、`az_bias_deg = 0.8·m`、
   `noise_underreport_factor = 1/(1+m)`（自报 σ 被低估）。
   它**只污染测量、不动真值**，且调度器看不到偏差标签——这正是研究分支里
   "来源一致性指示量"要处理的情形。

3. **跨场景数字不可直接比较**。不同场景的预算、截止余量、链路都不同，
   因此只允许比较"同一场景内不同方法"，以及"训练/验证分区上的聚合量"。

另外：`target_count` **不作为负载旋钮**。它保留在闭环接口里是为了场景完整性
（几何覆盖），本模块固定使用基线几何（2 个目标）以保持与既有实验可比。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from communication import SHARE_CONSTRAINED, SHARE_IDEAL
from resource_management.closed_loop import (
    BASELINE_DEADLINE_OFFSETS,
    DEFAULT_NODE_BUDGETS,
    NODE_LAYOUT,
)
from resource_management.learning_protocol import (
    DEFAULT_SPLIT_PATH,
    load_registry,
)
from resource_management.tasks import QueueTaskKind
from resource_management.units import BUDGET_UNITS, ResourceUnit

#: 映射版本。**改动任何映射语义都要升这个号**。
SCENARIO_MAPPING_VERSION = "resource-rl-scenario-mapping-v1"

#: 闭环的 tick 时长（秒）。目录里的 tick 序号按它换算成秒。
TICK_SECONDS = 1.0

#: 远端节点（注入传感器偏差的那个）
REMOTE_NODE_ID = "NODE_B"


@dataclass(frozen=True)
class ScenarioSpec:
    """场景目录里的一条记录（只读快照）。"""

    name: str
    load_multiplier: float
    budget_multiplier: float
    comm_delay_ticks: int
    comm_drop_probability: float
    node_outage_windows: Tuple[Tuple[int, int], ...]
    sensor_bias_sigma_multiplier: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "load_multiplier": self.load_multiplier,
            "budget_multiplier": self.budget_multiplier,
            "comm_delay_ticks": self.comm_delay_ticks,
            "comm_drop_probability": self.comm_drop_probability,
            "node_outage_windows": [list(window)
                                    for window in self.node_outage_windows],
            "sensor_bias_sigma_multiplier":
                self.sensor_bias_sigma_multiplier,
        }


def load_scenarios(path: str = DEFAULT_SPLIT_PATH) -> Dict[str, ScenarioSpec]:
    """读取场景目录（**只读**，不修改冻结登记表）。"""
    registry = load_registry(path)
    out: Dict[str, ScenarioSpec] = {}
    for name, raw in (registry.get("scenario_catalog") or {}).items():
        out[str(name)] = ScenarioSpec(
            name=str(name),
            load_multiplier=float(raw["load_multiplier"]),
            budget_multiplier=float(raw["budget_multiplier"]),
            comm_delay_ticks=int(raw["comm_delay_ticks"]),
            comm_drop_probability=float(raw["comm_drop_probability"]),
            node_outage_windows=tuple(
                (int(window[0]), int(window[1]))
                for window in raw["node_outage_windows"]),
            sensor_bias_sigma_multiplier=float(
                raw["sensor_bias_sigma_multiplier"]),
        )
    return out


def mapping_digest(path: str = DEFAULT_SPLIT_PATH) -> str:
    """映射摘要：版本号 + 基线常数 + 目录内容 + 映射后的闭环参数。"""
    payload = {
        "mapping_version": SCENARIO_MAPPING_VERSION,
        "tick_seconds": TICK_SECONDS,
        "baseline_deadline_offsets": {
            kind.value: value
            for kind, value in sorted(BASELINE_DEADLINE_OFFSETS.items(),
                                      key=lambda item: item[0].value)},
        "remote_node_id": REMOTE_NODE_ID,
        "scenarios": {
            name: closed_loop_kwargs(spec, path=path)
            for name, spec in sorted(load_scenarios(path).items())},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


# ----------------------------------------------------------------------
# 映射本体
# ----------------------------------------------------------------------


def scaled_budgets(budget_multiplier: float
                   ) -> Dict[str, Dict[ResourceUnit, float]]:
    """按 `budget_multiplier` 缩放逐节点容量（**不修改**默认值本身）。"""
    if budget_multiplier <= 0.0:
        raise ValueError("budget_multiplier 必须为正")
    return {
        node_id: {unit: float(value) * float(budget_multiplier)
                  for unit, value in budgets.items()}
        for node_id, budgets in DEFAULT_NODE_BUDGETS.items()
    }


def deadline_offsets(load_multiplier: float
                     ) -> Dict[QueueTaskKind, float]:
    """`load_multiplier` → 截止余量缩放（**服务压力**，不是任务数量）。"""
    if load_multiplier <= 0.0:
        raise ValueError("load_multiplier 必须为正")
    return {kind: value / float(load_multiplier)
            for kind, value in BASELINE_DEADLINE_OFFSETS.items()}


def sensor_bias(multiplier: float
                ) -> Dict[str, Dict[str, float]]:
    """`sensor_bias_sigma_multiplier` → 远端节点传感器偏置（**只污染测量**）。"""
    if multiplier <= 0.0:
        return {}
    return {REMOTE_NODE_ID: {
        "range_bias_m": 150.0 * float(multiplier),
        "az_bias_deg": 0.8 * float(multiplier),
        # 自报 σ 被低估：这是"来源不一致"的来源，不是"噪声更大"
        "noise_underreport_factor": 1.0 / (1.0 + float(multiplier)),
    }}


def closed_loop_kwargs(spec: ScenarioSpec,
                       path: str = DEFAULT_SPLIT_PATH) -> Dict[str, Any]:
    """把一条场景记录翻译成 `run_closed_loop` 的关键字参数。"""
    constrained = (spec.comm_delay_ticks > 0
                   or spec.comm_drop_probability > 0.0)
    # 目录里的中断窗口只有一对 [start, end]，**没写作用在哪个节点**。
    # 本实现把它作用在**远端节点**（与 sensor_bias 同一目标），理由：
    # 远端掉线才会制造"覆盖/交接"压力，也正是研究分支关心的"来源消失"情形。
    # 若将来要按节点分别指定，需要扩展登记表——那是契约变更，不能悄悄做。
    mechanisms: Dict[str, Any] = {
        "share_policy": SHARE_CONSTRAINED if constrained else SHARE_IDEAL,
        "unavailable_windows": (
            {REMOTE_NODE_ID: [(start * TICK_SECONDS, end * TICK_SECONDS)
                              for start, end in spec.node_outage_windows]}
            if spec.node_outage_windows else {}),
        "bias": sensor_bias(spec.sensor_bias_sigma_multiplier),
    }
    if constrained:
        mechanisms["comm"] = {
            "base_delay_s": spec.comm_delay_ticks * TICK_SECONDS,
            "jitter_s": 0.0,
            "loss_prob": spec.comm_drop_probability,
            "expiry_s": 10.0,
        }
    # 空的中断窗口不要留在机制里（保持 mechanisms 的可读性）
    mechanisms["unavailable_windows"] = {
        node_id: windows
        for node_id, windows in mechanisms["unavailable_windows"].items()
        if windows}
    return {
        "mechanisms": mechanisms,
        "node_budgets": scaled_budgets(spec.budget_multiplier),
        "deadline_offsets": {
            kind.value: value for kind, value
            in deadline_offsets(spec.load_multiplier).items()},
        "scenario": spec.to_dict(),
    }


def describe_mapping(path: str = DEFAULT_SPLIT_PATH) -> Dict[str, Any]:
    """人读的映射说明（供日志与文档核对）。"""
    scenarios = load_scenarios(path)
    return {
        "mapping_version": SCENARIO_MAPPING_VERSION,
        "digest": mapping_digest(path),
        "tick_seconds": TICK_SECONDS,
        "baseline_deadline_offsets": {
            kind.value: value
            for kind, value in BASELINE_DEADLINE_OFFSETS.items()},
        "budget_units": [unit.value for unit in BUDGET_UNITS],
        "remote_node_id": REMOTE_NODE_ID,
        "scenarios": {
            name: closed_loop_kwargs(spec, path=path)
            for name, spec in sorted(scenarios.items())},
        "limitations": [
            "load_multiplier 实现为截止余量缩放（服务压力），不是任务数量倍数："
            "闭环每节点每 tick 的候选任务类型上限为 3，结构上无法按倍数增加任务。",
            "sensor_bias_sigma_multiplier 实现为远端节点的测量偏置 + 自报 σ 低估"
            "（来源不一致），不是 σ 的整体缩放；只污染测量、不动真值。",
            "跨场景的数字不可直接比较（预算、截止余量、链路都不同）。",
        ],
    }


def scenario_names_for_split(split: str, *,
                             release_test: bool = False,
                             path: str = DEFAULT_SPLIT_PATH
                             ) -> Tuple[str, ...]:
    """取某个分区的场景名（测试分区默认抛错，见 learning_protocol）。"""
    from resource_management.learning_protocol import get_split
    return get_split(split, path=path, release_test=release_test).scenarios


def spec_for_scenario(name: str, path: str = DEFAULT_SPLIT_PATH) -> ScenarioSpec:
    scenarios = load_scenarios(path)
    if name not in scenarios:
        raise KeyError(f"未知场景 {name!r}；可用：{sorted(scenarios)}")
    return scenarios[name]


__all__ = [
    "BASELINE_DEADLINE_OFFSETS", "REMOTE_NODE_ID",
    "SCENARIO_MAPPING_VERSION", "TICK_SECONDS", "ScenarioSpec",
    "closed_loop_kwargs", "deadline_offsets", "describe_mapping",
    "load_scenarios", "mapping_digest", "scaled_budgets", "scenario_names_for_split",
    "sensor_bias", "spec_for_scenario",
]
