"""简化遮挡模型（v4.2）。

范围与取舍
----------
本阶段**不做射线追踪**。遮挡用两种解析可解的基本体表示：

* **长方体遮挡区**（AABB，轴对齐）：`box(min_x, min_y, min_z, max_x, max_y, max_z)`
  适合表示建筑群、山体包围盒、禁飞区立方体；
* **球形遮挡区**：`sphere(cx, cy, cz, radius)`
  适合表示山包、烟幕、局部干扰云。

判据是"传感器与目标之间的**线段**是否与障碍体相交"，
两者都有闭式解，代价 O(1)，无需采样、无需迭代：

* 线段 vs AABB：**slab 方法**（逐轴计算进入/离开参数区间后取交）；
* 线段 vs 球：解一元二次方程，取落在 [0,1] 区间内的最小正根。

两个关键细节（都踩过）：
1. **必须只取线段内部的交点**（参数 t ∈ [0, 1]）。只判断"直线相交"是错误的：
   障碍物在传感器**背后**时直线仍然相交，但那不构成遮挡。
2. **必须处理线段起点/终点落在障碍物内部**的情形（例如目标贴在遮挡区表面）。
   此时按"被遮挡"处理（保守），否则会出现"目标在盒子里却看得见"的荒谬结果。

诚实性说明：真实的地形遮挡还要考虑地球曲率、大气折射、多径与绕射。
本模型**都没有**，它只能回答"直线视线是否被简单几何体截断"。
在 README 与本模块文档里都写明这一点，不要把它包装成完整的地形建模。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.geometry import Vec3

#: 判断"线段与障碍体表面相切"时的容差
_PARALLEL_EPS = 1e-12


@dataclass(frozen=True)
class BoxOccluder:
    """轴对齐长方体遮挡区（ENU，米）。"""

    occluder_id: str
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    is_active: bool = True

    def __post_init__(self) -> None:
        if self.max_x < self.min_x or self.max_y < self.min_y or self.max_z < self.min_z:
            raise ValueError(f"[{self.occluder_id}] AABB 的 max 不能小于 min")

    @property
    def center(self) -> Vec3:
        return Vec3(
            0.5 * (self.min_x + self.max_x),
            0.5 * (self.min_y + self.max_y),
            0.5 * (self.min_z + self.max_z),
        )

    def contains(self, point: Vec3, tol: float = 0.0) -> bool:
        return (
            self.min_x - tol <= point.x <= self.max_x + tol
            and self.min_y - tol <= point.y <= self.max_y + tol
            and self.min_z - tol <= point.z <= self.max_z + tol
        )

    def intersects_segment(self, start: Vec3, end: Vec3) -> bool:
        """线段 start→end 是否与本体相交（slab 方法）。

        返回 True 表示**被遮挡**。端点落在体内也算遮挡（保守处理）。
        """
        # 端点已在体内：直接判遮挡
        if self.contains(start) or self.contains(end):
            return True

        t_min, t_max = 0.0, 1.0
        for origin, direction, low, high in (
            (start.x, end.x - start.x, self.min_x, self.max_x),
            (start.y, end.y - start.y, self.min_y, self.max_y),
            (start.z, end.z - start.z, self.min_z, self.max_z),
        ):
            if abs(direction) < _PARALLEL_EPS:
                # 该轴方向无变化：起点必须已经落在板层内，否则永不相交
                if origin < low or origin > high:
                    return False
                continue
            t1 = (low - origin) / direction
            t2 = (high - origin) / direction
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        # 交点参数必须落在线段上（t ∈ [0,1]）
        return t_min <= 1.0 and t_max >= 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "occluder_id": self.occluder_id, "shape": "box",
            "min_x": self.min_x, "min_y": self.min_y, "min_z": self.min_z,
            "max_x": self.max_x, "max_y": self.max_y, "max_z": self.max_z,
            "is_active": self.is_active,
        }


@dataclass(frozen=True)
class SphereOccluder:
    """球形遮挡区（ENU，米）。"""

    occluder_id: str
    center_x: float
    center_y: float
    center_z: float
    radius_m: float
    is_active: bool = True

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            raise ValueError(f"[{self.occluder_id}] radius_m 必须为正")

    @property
    def center(self) -> Vec3:
        return Vec3(self.center_x, self.center_y, self.center_z)

    def contains(self, point: Vec3, tol: float = 0.0) -> bool:
        return (point - self.center).norm() <= self.radius_m + tol

    def intersects_segment(self, start: Vec3, end: Vec3) -> bool:
        """线段与球是否相交（解一元二次方程，只取 t ∈ [0,1] 的根）。"""
        if self.contains(start) or self.contains(end):
            return True

        direction = end - start
        a = direction.norm_squared()
        if a < _PARALLEL_EPS:
            return False  # 退化为一个点
        offset = start - self.center
        b = 2.0 * offset.dot(direction)
        c = offset.norm_squared() - self.radius_m ** 2
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return False
        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)
        if t1 > t2:
            t1, t2 = t2, t1
        # 只在**线段内部**有交点才算遮挡（t 必须落在 [0, 1]）
        return t2 >= 0.0 and t1 <= 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "occluder_id": self.occluder_id, "shape": "sphere",
            "center_x": self.center_x, "center_y": self.center_y,
            "center_z": self.center_z, "radius_m": self.radius_m,
            "is_active": self.is_active,
        }


Occluder = Any  # BoxOccluder | SphereOccluder


def build_occluder(data: Dict[str, Any]) -> Occluder:
    """按配置字典构造遮挡体（`shape` 决定类型）。"""
    shape = str(data.get("shape", "box")).lower()
    if shape == "box":
        return BoxOccluder(
            occluder_id=str(data["occluder_id"]),
            min_x=float(data["min_x"]), min_y=float(data["min_y"]),
            min_z=float(data["min_z"]), max_x=float(data["max_x"]),
            max_y=float(data["max_y"]), max_z=float(data["max_z"]),
            is_active=bool(data.get("is_active", True)),
        )
    if shape == "sphere":
        return SphereOccluder(
            occluder_id=str(data["occluder_id"]),
            center_x=float(data["center_x"]), center_y=float(data["center_y"]),
            center_z=float(data["center_z"]),
            radius_m=float(data["radius_m"]),
            is_active=bool(data.get("is_active", True)),
        )
    raise ValueError(f"未知遮挡体类型 shape={shape!r}，只支持 'box' / 'sphere'")


class OcclusionModel:
    """一组遮挡体，回答"两点之间的视线是否被切断"。"""

    def __init__(self, occluders: Optional[Sequence[Occluder]] = None) -> None:
        self.occluders: List[Occluder] = list(occluders or [])

    def __len__(self) -> int:
        return len(self.occluders)

    @property
    def active(self) -> List[Occluder]:
        return [o for o in self.occluders if o.is_active]

    def is_occluded(
        self, start: Vec3, end: Vec3, ignore: Optional[Sequence[str]] = None
    ) -> bool:
        """视线 start→end 是否被任一激活遮挡体切断。"""
        skip = set(ignore or ())
        for occluder in self.active:
            if occluder.occluder_id in skip:
                continue
            if occluder.intersects_segment(start, end):
                return True
        return False

    def first_blocker(self, start: Vec3, end: Vec3) -> Optional[str]:
        """返回切断视线的第一个遮挡体 ID（用于诊断"到底被谁挡了"）。"""
        for occluder in self.active:
            if occluder.intersects_segment(start, end):
                return occluder.occluder_id
        return None

    def to_list(self) -> List[Dict[str, Any]]:
        return [o.to_dict() for o in self.occluders]

    @classmethod
    def from_config(cls, items: Optional[Sequence[Dict[str, Any]]]) -> "OcclusionModel":
        return cls([build_occluder(item) for item in (items or [])])
