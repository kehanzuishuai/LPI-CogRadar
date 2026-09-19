"""目标（被探测对象）模型。

目标提供雷达方程所需的距离与雷达截面积 RCS，并可按匀速直线机动。
第一版不对目标做隐身/起伏建模，RCS 为常数。

v4.1 多平台升级
---------------
继承 `SceneEntity`，因此获得了统一标识（`entity_id`）、三维位置与速度、
姿态、时间戳、平台归属，以及**全工程唯一的**距离实现。
距离计算本文件已不再自己实现——`range_to()` 由基类提供并统一走
`engine.geometry`。旧字段名（`target_id` / `x` / `y` / `rcs_m2` / `vx` / `vy`）
与旧构造方式（`Target(**config)`）完全不变。
"""

from __future__ import annotations

from dataclasses import dataclass

from models.entity import KIND_TARGET, SceneEntity


@dataclass
class Target(SceneEntity):
    target_id: str
    x: float
    y: float
    rcs_m2: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    is_active: bool = True

    # --- v4.1 多平台扩展（全部带默认值，旧配置无需改动）---
    z: float = 0.0
    vz: float = 0.0
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    timestamp_s: float = 0.0
    platform_id: str = ""

    #: 类属性（不加注解 => 不是数据字段）
    ENTITY_KIND = KIND_TARGET
    ID_FIELD = "target_id"

    def __post_init__(self) -> None:
        if self.rcs_m2 <= 0:
            raise ValueError(f"[{self.target_id}] rcs_m2 必须为正")
