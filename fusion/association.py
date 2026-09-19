"""多源关联：门限 + 最近邻（v4.3）。

做法与理由
----------
本阶段刻意用**最基础**的关联：

1. **门限（gating）**：把航迹位置投影成"预测的极坐标量测"，
   与测量比较，超出 `gate_*` 门限的组合直接排除；
2. **最近邻（贪心指派）**：在通过门限的组合里按代价从小到大贪心分配，
   一条航迹至多吸收一条测量，一条测量至多进一条航迹。

为什么不做 JPDA / MHT：用户明确要求"不需要一开始实现复杂算法"，
而且本项目当前的目标数是 2~3、传感器 2~4，最近邻够用。
**但必须写明局限**：密集目标或漏检连续发生时，最近邻会把测量串到错误航迹上，
表现为"丢轨 + 重复轨迹"，这正是三组实验要观察的量之一。

代价度量用的是**归一化残差**（在各自 σ 尺度上），而不是纯欧氏距离——
否则远距离目标因为量纲大永远被排在后面。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.geometry import Vec3, enu_to_spherical


@dataclass
class AssociationConfig:
    """关联门限（都在 σ 尺度上）。"""

    #: 距离残差门限（米）；None 表示用 sigma 倍数
    gate_range_m: Optional[float] = None
    #: 角度残差门限（度）
    gate_az_deg: float = 5.0
    gate_el_deg: float = 5.0
    #: 若给了 sigma，则用 max(gate, k*sigma) 放宽门限
    sigma_multiplier: float = 3.0
    #: 归一化代价上限（超过即认为不该关联）
    max_cost: float = 9.0

    def validate(self) -> None:
        if self.gate_az_deg <= 0 or self.gate_el_deg <= 0:
            raise ValueError("角度门限必须为正")
        if self.sigma_multiplier <= 0:
            raise ValueError("sigma_multiplier 必须为正")


@dataclass
class AssociationResult:
    """一次关联的结果。"""

    #: (track_id, measurement_index, cost)
    assignments: List[Tuple[str, int, float]] = field(default_factory=list)
    #: 未关联上的测量下标（可能起始新航迹，也可能是虚警）
    unassigned_measurements: List[int] = field(default_factory=list)
    #: 未获得测量的航迹 ID（进入外推）
    unmatched_tracks: List[str] = field(default_factory=list)
    #: 被门限拒绝的组合数（诊断用）
    gated_out: int = 0

    @property
    def n_assigned(self) -> int:
        return len(self.assignments)


def _predict_polar(track_position: Vec3, sensor_position: Vec3) -> Tuple[float, float, float]:
    """把航迹位置的绝对坐标投影成该传感器视角下的 (range, az, el)。"""
    delta = track_position - sensor_position
    spherical = enu_to_spherical(delta)
    return spherical.range_m, spherical.azimuth_deg, spherical.elevation_deg


def _angular_diff(a: float, b: float) -> float:
    delta = a - b
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return abs(delta)


def association_cost(
    track_position: Vec3,
    sensor_position: Vec3,
    measurement: Any,
    config: AssociationConfig,
) -> Optional[float]:
    """计算一条 (航迹, 测量) 组合的归一化代价；超出任何门限返回 None。

    归一化方式：每维残差除以"门限"，再平方求和。
    这样不同量纲（米 / 度）可以比较，且代价 ≈ 门限倍数²。
    """
    pred_range, pred_az, pred_el = _predict_polar(track_position, sensor_position)

    terms: List[float] = []
    # --- 距离（仅当测量含距离时）---
    if getattr(measurement, "range_m", None) is not None:
        sigma_r = float(getattr(measurement, "std_range_m", None) or 0.0)
        gate_r = config.gate_range_m
        if gate_r is None:
            gate_r = max(config.sigma_multiplier * sigma_r, 50.0)
        else:
            gate_r = max(gate_r, config.sigma_multiplier * sigma_r)
        residual = float(measurement.range_m) - pred_range
        if abs(residual) > gate_r:
            return None
        terms.append((residual / gate_r) ** 2)

    # --- 方位 ---
    sigma_az = float(getattr(measurement, "std_az_deg", None) or 0.0)
    gate_az = max(config.gate_az_deg, config.sigma_multiplier * sigma_az)
    residual_az = _angular_diff(float(measurement.azimuth_deg), pred_az)
    if residual_az > gate_az:
        return None
    terms.append((residual_az / gate_az) ** 2)

    # --- 俯仰 ---
    sigma_el = float(getattr(measurement, "std_el_deg", None) or 0.0)
    gate_el = max(config.gate_el_deg, config.sigma_multiplier * sigma_el)
    if getattr(measurement, "elevation_deg", None) is not None:
        residual_el = abs(float(measurement.elevation_deg) - pred_el)
        if residual_el > gate_el:
            return None
        terms.append((residual_el / gate_el) ** 2)

    if not terms:
        return None
    cost = sum(terms)
    if cost > config.max_cost:
        return None
    return cost


def associate(
    tracks: Sequence[Any],
    measurements: Sequence[Any],
    sensor_positions: Dict[str, Vec3],
    config: Optional[AssociationConfig] = None,
) -> AssociationResult:
    """对"航迹 × 测量"做门限 + 最近邻贪心指派。

    参数
    ----
    tracks          : `fusion.track.Track` 列表
    measurements    : 测量记录列表（需有 sensor_id / azimuth_deg / range_m 等字段）
    sensor_positions: `{sensor_id: Vec3}`，把航迹位置投影到该传感器视角要用
    config          : 门限配置

    返回 `AssociationResult`。**只读测量，不访问任何真值。**
    """
    cfg = config or AssociationConfig()
    cfg.validate()

    result = AssociationResult()
    candidates: List[Tuple[float, str, int]] = []

    for track in tracks:
        if track.status == "dropped":
            continue
        for index, measurement in enumerate(measurements):
            sensor_id = str(getattr(measurement, "sensor_id", ""))
            sensor_position = sensor_positions.get(sensor_id)
            if sensor_position is None:
                # 不知道传感器位置就无法做几何一致性判定（例如共享来的测量
                # 带了传感器 ID 但本平台没有该传感器的几何信息）
                continue
            if getattr(measurement, "azimuth_deg", None) is None:
                continue
            cost = association_cost(track.position, sensor_position, measurement, cfg)
            if cost is None:
                result.gated_out += 1
                continue
            candidates.append((cost, track.track_id, index))

    # --- 贪心：代价升序，每航迹/每测量只用一次 ---
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_tracks: set = set()
    used_measurements: set = set()
    for cost, track_id, index in candidates:
        if track_id in used_tracks or index in used_measurements:
            continue
        used_tracks.add(track_id)
        used_measurements.add(index)
        result.assignments.append((track_id, index, cost))

    result.unassigned_measurements = [
        i for i in range(len(measurements)) if i not in used_measurements
    ]
    result.unmatched_tracks = [
        t.track_id for t in tracks
        if t.status != "dropped" and t.track_id not in used_tracks
    ]
    return result
