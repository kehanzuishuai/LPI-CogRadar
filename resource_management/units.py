"""资源单位与**教学成本模型**（`resource_management` 的基础层）。

为什么单位必须写死在代码里
--------------------------
"节点用了 3 个资源"这种话在工程上没有意义，除非说清 3 个**什么**。
因此本模块把所有资源单位定义成枚举，并为每个单位写明：

* 中文名与符号；
* **一个单位在这个教学仿真里代表什么操作**；
* 它**不对应**任何真实装备指标（这是纪律，不是免责声明）。

⚠️ **教学成本模型与真实装备效能无关**
--------------------------------------
下面 `TEACHING_COST_MODEL` 里的数字是**为了把多节点资源竞争演示清楚**
而人为设定的，单位之间没有换算关系，也没有标定过任何真实雷达。
**不得**用它们推断真实装备的采样能力、处理吞吐或通信带宽。
要让成本"更真实"，正确做法是在配置里替换成实测标定值，
而不是把这里的数字当成物理量去引用。

四类资源（前三类是预算，第四类是时间占用）
------------------------------------------
============================  ==================  ================================
单位                          含义                一个单位代表
============================  ==================  ================================
`SAMPLE_SLOT`                 采样时隙            一次驻留观测（对场景扫一遍）
`PROCESSING_OP`               处理配额            一次关联 + 一次滤波更新
`COMM_BYTE`                   通信字节            一个字节的共享载荷
`OCCUPANCY_SECOND`            占用秒（时间资源）   节点被某个任务独占 1 秒
============================  ==================  ================================

`OCCUPANCY_SECOND` 不是"预算"而是**时间占用**：它不会被消耗掉，
而是记录成占用区间；把两者分开是必要的，否则"节点忙但资源充足"
与"节点闲但资源耗尽"这两种完全不同的故障会被混成一种。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Tuple


class ResourceUnit(str, Enum):
    """可被预算约束的资源单位。"""

    SAMPLE_SLOT = "sample_slot"
    PROCESSING_OP = "processing_op"
    COMM_BYTE = "comm_byte"
    #: 时间资源：以"节点被独占的秒数"计，记入占用区间而非消耗预算
    OCCUPANCY_SECOND = "occupancy_second"


#: 单位 → 中文名（报表与错误信息里一律用它，不用英文键）
UNIT_CN: Dict[ResourceUnit, str] = {
    ResourceUnit.SAMPLE_SLOT: "采样时隙",
    ResourceUnit.PROCESSING_OP: "处理配额",
    ResourceUnit.COMM_BYTE: "通信字节",
    ResourceUnit.OCCUPANCY_SECOND: "占用秒",
}

#: 单位 → 符号（紧凑报表用）
UNIT_SYMBOL: Dict[ResourceUnit, str] = {
    ResourceUnit.SAMPLE_SLOT: "slot",
    ResourceUnit.PROCESSING_OP: "op",
    ResourceUnit.COMM_BYTE: "B",
    ResourceUnit.OCCUPANCY_SECOND: "s",
}

#: 单位 → "一个单位代表什么"（写进契约文档与报表表头）
UNIT_MEANING: Dict[ResourceUnit, str] = {
    ResourceUnit.SAMPLE_SLOT: "一次驻留观测（对场景扫一遍）",
    ResourceUnit.PROCESSING_OP: "一次关联 + 一次滤波更新",
    ResourceUnit.COMM_BYTE: "一字节共享载荷",
    ResourceUnit.OCCUPANCY_SECOND: "节点被该任务独占 1 秒",
}

#: 预算类单位（会被消耗/预留）；`OCCUPANCY_SECOND` 不在其中
BUDGET_UNITS: Tuple[ResourceUnit, ...] = (
    ResourceUnit.SAMPLE_SLOT,
    ResourceUnit.PROCESSING_OP,
    ResourceUnit.COMM_BYTE,
)


class TaskKind(str, Enum):
    """任务类型：一个计划可以同时描述多个节点的这几类任务。"""

    SAMPLE = "sample"
    PROCESS = "process"
    SHARE = "share"
    IDLE = "idle"


TASK_KIND_CN: Dict[TaskKind, str] = {
    TaskKind.SAMPLE: "采样",
    TaskKind.PROCESS: "处理",
    TaskKind.SHARE: "共享",
    TaskKind.IDLE: "空闲",
}


@dataclass(frozen=True)
class TeachingCost:
    """一类任务的**教学**单位成本。

    ⚠️ 与真实装备效能无关：这些数字只用来把资源竞争演示清楚。
    要"更真实"就在配置里换成实测标定值，不要把这里当物理量引用。
    """

    sample_slots: float = 0.0
    processing_ops: float = 0.0
    comm_bytes: float = 0.0

    def as_dict(self) -> Dict[ResourceUnit, float]:
        return {
            ResourceUnit.SAMPLE_SLOT: float(self.sample_slots),
            ResourceUnit.PROCESSING_OP: float(self.processing_ops),
            ResourceUnit.COMM_BYTE: float(self.comm_bytes),
        }


#: 教学成本模型：**人为设定、可整体替换**，不代表任何真实装备
TEACHING_COST_MODEL: Dict[TaskKind, TeachingCost] = {
    # 采样：消耗一个采样时隙；产出的原始点迹需要一次处理
    TaskKind.SAMPLE: TeachingCost(sample_slots=1.0, processing_ops=1.0),
    # 处理：只消耗处理配额（对已有测量/历史估计做一次更新）
    TaskKind.PROCESS: TeachingCost(processing_ops=1.0),
    # 共享：把测量打包发出去；每条消息按固定载荷计费
    TaskKind.SHARE: TeachingCost(comm_bytes=128.0),
    # 空闲：**零成本**，且不产生任何采样报告（见 executor 的空闲语义）
    TaskKind.IDLE: TeachingCost(),
}

#: 每类任务的默认持续时间（秒）——同样是教学设定，不是装备指标
DEFAULT_DURATION_S: Dict[TaskKind, float] = {
    TaskKind.SAMPLE: 1.0,
    TaskKind.PROCESS: 1.0,
    TaskKind.SHARE: 0.5,
    TaskKind.IDLE: 0.0,
}


def format_cost(cost: Mapping[ResourceUnit, float]) -> str:
    """把成本字典排版成 `1slot+1op` 这种紧凑可读形式（0 值省略）。"""
    parts = []
    for unit in ResourceUnit:
        value = float(cost.get(unit, 0.0) or 0.0)
        if value == 0.0:
            continue
        parts.append(f"{value:g}{UNIT_SYMBOL[unit]}")
    return "+".join(parts) if parts else "0（无成本）"


def format_amount(unit: ResourceUnit, value: float) -> str:
    return f"{value:g} {UNIT_CN[unit]}（{UNIT_SYMBOL[unit]}）"
