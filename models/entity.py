"""场景实体基类（v4.1 多平台几何升级）。

升级前，四类实体（Radar / Target / EnemyInterceptor / Jammer）是**各写各的**：
各自声明 `x, y`，各自实现 `update_position()`，其中三个还各自抄了一遍
`range_to()`。这在单雷达小场景里能用，但支撑不了多平台：

* 没有统一的唯一标识，跨类型引用只能靠各自的 `*_id` 字段名硬编码；
* 没有三维位置、没有姿态、没有时间戳，无法表达"谁在什么时刻朝哪看"；
* 没有平台归属，无法表达"这个干扰吊舱和那个目标属于同一架飞机"。

本模块提供 `SceneEntity` 混入基类，**统一**这些能力，同时**不破坏旧接口**：

* 位置仍是扁平的 `x` / `y`（新增 `z`，默认 0），
  因此 `Radar(**config["radar"])`、`radar.x` 这类既有写法全部照旧；
* 速度字段名因历史原因不统一（Target/Jammer/Interceptor 用 `vx,vy`，
  Radar 用 `velocity_x,velocity_y`），基类用 `VELOCITY_FIELDS` 适配，
  而不是强行重命名——重命名会让所有配置文件与外部脚本一起失效；
* 唯一标识通过 `ID_FIELD` 适配（`target_id` / `interceptor_id` / `jammer_id` / `radar_id`），
  对外统一暴露 `entity_id`。

设计取舍：为什么用「类属性默认值」而不是「property」
--------------------------------------------------
基类把 `z` / `timestamp_s` / `platform_id` / 姿态角声明为**普通类属性默认值**，
而不是 `@property`。原因：四个子类都是 `@dataclass`，会在自己的类体里把
这些名字声明成数据字段（字段默认值同样落在类属性上），从而**遮蔽**基类属性。
若基类用 property，遮蔽关系会变得很微妙（读到的是字段还是描述符取决于
子类有没有声明），而且 `advance()` 里对 `z` 的写入会因描述符行为出岔子。

普通类属性则没有这个问题：
* 子类**声明了**该字段 → 用子类字段，可正常读写；
* 子类**没声明** → 回落到基类默认值，`self.z = ...` 也只是给实例加属性，照样工作。

于是可以分步迁移：先把基类挂上去，再逐步给各模型补字段，中间任何一步都不会崩。

⚠️ 兼容性：`SceneEntity.advance()` / `range_to()` 的实现刻意与旧代码逐字等价
（同样的 `math.hypot`、同样的 `+=` 顺序），新增的 `z`/时间戳维度不会影响
`x`/`y` 的数值。`tests/test_entity_geometry.py` 有逐位回归断言钉住这一点。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from engine.geometry import (
    Attitude,
    GeometricRelation,
    Pose,
    Vec3,
    enu_range,
    enu_range_2d,
    relation,
)

#: 实体类型常量
KIND_RADAR = "radar"
KIND_TARGET = "target"
KIND_INTERCEPTOR = "interceptor"
KIND_JAMMER = "jammer"

KIND_CN: Dict[str, str] = {
    KIND_RADAR: "雷达",
    KIND_TARGET: "目标",
    KIND_INTERCEPTOR: "侦察接收机",
    KIND_JAMMER: "干扰源",
}


class SceneEntity:
    """所有场景实体的公共混入基类。

    子类需要：
    * 设置 `ID_FIELD`（其唯一标识字段名）与 `ENTITY_KIND`；
    * 声明 `x` / `y` 数值字段。

    下列名字基类给定了默认值，子类**声明成 dataclass 字段即可覆盖**
    （不声明也能正常工作，只是无法从配置里给值）：

        z, timestamp_s, platform_id, heading_deg, pitch_deg, roll_deg
    """

    #: 该实体的唯一标识字段名（子类覆盖）
    ID_FIELD: str = "entity_id"
    #: 实体类型（子类覆盖）
    ENTITY_KIND: str = "entity"
    #: 速度三元组的字段名，顺序 (x, y, z)
    VELOCITY_FIELDS: Tuple[str, str, str] = ("vx", "vy", "vz")

    # --- 公共字段默认值（子类可声明为 dataclass 字段来覆盖）---
    z: float = 0.0
    timestamp_s: float = 0.0
    platform_id: str = ""
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    is_active: bool = True

    # ------------------------------------------------------------------
    # 唯一标识与归属
    # ------------------------------------------------------------------

    @property
    def entity_id(self) -> str:
        """跨类型唯一的实体标识。

        不重命名子类的历史字段（那会破坏全部配置与外部脚本），
        而是统一暴露一个只读别名。
        """
        return str(getattr(self, self.ID_FIELD))

    @property
    def entity_kind(self) -> str:
        return self.ENTITY_KIND

    @property
    def entity_kind_cn(self) -> str:
        return KIND_CN.get(self.ENTITY_KIND, self.ENTITY_KIND)

    # ------------------------------------------------------------------
    # 位置 / 速度 / 姿态 / 时间
    # ------------------------------------------------------------------

    @property
    def position(self) -> Vec3:
        """ENU 三维位置。"""
        return Vec3(float(self.x), float(self.y), float(self.z))

    @property
    def velocity(self) -> Vec3:
        """ENU 三维速度。"""
        vx_field, vy_field, vz_field = self.VELOCITY_FIELDS
        return Vec3(
            float(getattr(self, vx_field, 0.0)),
            float(getattr(self, vy_field, 0.0)),
            float(getattr(self, vz_field, 0.0)),
        )

    @property
    def speed_mps(self) -> float:
        return self.velocity.norm()

    @property
    def attitude(self) -> Attitude:
        """平台姿态（航向/俯仰/横滚，度）。缺省全零 = 机头指北、水平。"""
        return Attitude(
            heading_deg=float(self.heading_deg),
            pitch_deg=float(self.pitch_deg),
            roll_deg=float(self.roll_deg),
        )

    @property
    def pose(self) -> Pose:
        """当前位姿（位置 + 速度 + 姿态 + 时间戳）。"""
        return Pose(
            position=self.position,
            velocity=self.velocity,
            attitude=self.attitude,
            time_s=float(self.timestamp_s),
            entity_id=self.entity_id,
        )

    def set_pose(
        self,
        position: Optional[Vec3] = None,
        velocity: Optional[Vec3] = None,
        attitude: Optional[Attitude] = None,
        time_s: Optional[float] = None,
    ) -> None:
        """一次性写入位姿分量（未给的分量保持不变）。"""
        if position is not None:
            self.x = position.x
            self.y = position.y
            self.z = position.z
        if velocity is not None:
            vx_field, vy_field, vz_field = self.VELOCITY_FIELDS
            setattr(self, vx_field, velocity.x)
            setattr(self, vy_field, velocity.y)
            setattr(self, vz_field, velocity.z)
        if attitude is not None:
            self.heading_deg = attitude.heading_deg
            self.pitch_deg = attitude.pitch_deg
            self.roll_deg = attitude.roll_deg
        if time_s is not None:
            self.timestamp_s = float(time_s)

    # ------------------------------------------------------------------
    # 运动
    # ------------------------------------------------------------------

    def advance(self, dt: float) -> None:
        """按当前速度推进 dt 秒，并把时间戳前移 dt。

        与旧 `update_position(dt)` 的关系：**逐字等价**
        （`self.x = self.x + vx * dt`），只是额外推进了 `z` 与时间戳。
        旧场景 `vz = 0`，因此 `z` 恒为 0，不影响任何既有数值。
        """
        velocity = self.velocity
        self.x = self.x + velocity.x * dt
        self.y = self.y + velocity.y * dt
        self.z = self.z + velocity.z * dt
        self.timestamp_s = float(self.timestamp_s) + dt

    def update_position(self, dt: float) -> None:
        """保留旧名字，委托给 `advance`（外部脚本仍在调用它）。"""
        self.advance(dt)

    # ------------------------------------------------------------------
    # 几何（**唯一实现**，所有模型共用）
    # ------------------------------------------------------------------

    def range_to(self, x: float, y: float) -> float:
        """到水平坐标 (x, y) 的距离。

        这是**全工程唯一的距离实现**：四类实体不再各写一份。
        刻意走二维路径（`enu_range_2d`），与升级前逐位一致。
        """
        return enu_range_2d(float(self.x), float(self.y), float(x), float(y))

    def range_to_entity(self, other: "SceneEntity") -> float:
        """到另一个实体的**三维**距离（走统一几何模块）。"""
        return enu_range(self.position, other.position)

    def relation_to(
        self, other: "SceneEntity", time_s: Optional[float] = None
    ) -> GeometricRelation:
        """本实体作为 **observer**、`other` 作为 **target** 的有向几何关系。

        语义是"我相对于它看到/作用于它的几何"：
        方位、俯仰、机体方位、视线向量全部是 `self → other` 方向上的量。
        需要反向关系时请显式调用 `other.relation_to(self)`。

        交换双方会变的东西：`los_enu` 取反，`azimuth_deg` 差 180°，
        `elevation_deg` / `elevation_body_deg` 变号。
        **不会变**的是 `range_m`、`range_rate_mps`、`closing`——
        径向速度按定义是"距离变化率"，属于这一对实体，与谁当观察者无关。
        """
        return relation(
            self.entity_id, self.pose, other.entity_id, other.pose, time_s=time_s
        )

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """实体状态快照（供 CSV / JSON 导出与报告使用）。"""
        velocity = self.velocity
        attitude = self.attitude
        return {
            "entity_id": self.entity_id,
            "entity_kind": self.entity_kind,
            "entity_kind_cn": self.entity_kind_cn,
            "platform_id": str(self.platform_id),
            "time_s": float(self.timestamp_s),
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "vx": velocity.x,
            "vy": velocity.y,
            "vz": velocity.z,
            "speed_mps": self.speed_mps,
            "heading_deg": attitude.heading_deg,
            "pitch_deg": attitude.pitch_deg,
            "roll_deg": attitude.roll_deg,
            "is_active": bool(self.is_active),
        }

    def describe_entity(self) -> str:
        """一行人类可读描述，便于日志与调试。"""
        return (
            f"{self.entity_id}[{self.entity_kind_cn}] "
            f"pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}) "
            f"v={self.speed_mps:.1f} m/s "
            f"hdg={self.attitude.heading_deg:.1f}° "
            f"t={float(self.timestamp_s):g}s"
            + (f" platform={self.platform_id}" if self.platform_id else "")
        )
