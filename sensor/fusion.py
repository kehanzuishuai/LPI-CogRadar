"""测量汇聚层：把变长的测量列表打包成**定长、且不含真值**的观测向量（v4.2）。

这是「真值 → 测量 → 算法输入」三层里的最后一层，也是**唯一**算法能看到的东西。

保证（有单元测试钉住）
----------------------
1. **不含真值**：打包过程只读 `MeasurementRecord` 的测量字段，
   绝不调用 `Sensor.truth_of_candidate()`，也不读 `truth_*`。
   `assert_no_truth_leak()` 会扫描打包结果里出现的所有键，发现任何
   `truth` / `err_` 前缀就直接抛错。
2. **不含虚警标记**：`is_false_alarm` 同样**不进**观测向量——
   算法若能看到这个标记就等于开了上帝视角。
3. **定长**：观测维度与当步有几个测量无关，只与 `TrackTableConfig` 有关。
   这样 DQN 的网络结构不随场景变化。

槽位分配
--------
按 `confidence` 降序取前 `max_tracks` 条测量放进固定槽位，
不足的槽位用 `present=0` 填充。排序是确定性的（同分按候选编号字典序），
保证同一输入下观测完全相同。

⚠️ 这个"按置信度取前 K 条"的做法本身是一种**极简的关联/航迹管理**，
它**没有**做真正的数据关联（最近邻 / JPDA / MHT）。因此：
* 同一目标在不同步可能落在不同槽位（因为排序会变）；
* 无法维持稳定的航迹编号。
这一点必须在 README 里写清楚，不能把它说成"跟踪器"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sensor.record import MeasurementRecord

#: 禁止出现在算法输入里的键前缀（真值/误差/虚警标记）
FORBIDDEN_KEY_PREFIXES: Tuple[str, ...] = ("truth", "err_", "is_false_alarm")

#: 每个航迹槽位打包的维度
SLOT_FIELDS: Tuple[str, ...] = (
    "present",           # 1 = 该槽位有测量
    "range_norm",        # 距离 / range_scale（无距离量测时为 0，并由 has_range 区分）
    "has_range",         # 1 = 该测量含距离（被动 ESM 为 0）
    "bearing_norm",      # 相对本机机头的方位 / 180
    "elevation_norm",    # 俯仰 / 90
    "range_rate_norm",   # 径向速度 / vr_scale
    "std_range_norm",    # 距离标准差 / range_scale（无距离为 1）
    "std_az_norm",       # 方位标准差 / 5°
    "confidence",        # 检测置信度
    "age_norm",          # 测量年龄 / age_scale
)


@dataclass
class TrackTableConfig:
    """观测打包配置。维度固定由它决定，与场景无关。"""

    max_tracks: int = 4
    range_scale_m: float = 30000.0
    range_rate_scale_mps: float = 200.0
    age_scale_s: float = 10.0
    std_az_scale_deg: float = 5.0

    @property
    def slot_dim(self) -> int:
        return len(SLOT_FIELDS)

    @property
    def table_dim(self) -> int:
        return self.max_tracks * self.slot_dim

    def validate(self) -> None:
        if self.max_tracks < 1:
            raise ValueError("max_tracks 至少为 1")
        for name in ("range_scale_m", "range_rate_scale_mps", "age_scale_s",
                     "std_az_scale_deg"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正")


@dataclass
class FusedObservation:
    """融合结果：定长向量 + 逐槽位说明 + 缺失统计。

    `vector` 是**唯一**可以交给决策算法的东西。
    `slots` 与 `reason_counts` 只用于诊断、导出与调试，
    但因为它们可能含候选编号（不是真值），仍应避免直接进入网络输入。
    """

    vector: List[float]
    slots: List[Dict[str, Any]] = field(default_factory=list)
    reason_counts: Dict[str, int] = field(default_factory=dict)
    reason_by_dimension: Dict[str, int] = field(default_factory=dict)
    n_measurements: int = 0
    n_fresh: int = 0
    n_tracks_used: int = 0
    observation_quality: float = 0.0
    #: 仅诊断用：本步是否有测量不含距离（被动）
    any_range_less: bool = False

    def to_dict(self, include_slots: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "n_measurements": self.n_measurements,
            "n_fresh": self.n_fresh,
            "n_tracks_used": self.n_tracks_used,
            "observation_quality": self.observation_quality,
            "reason_counts": dict(self.reason_counts),
            "reason_by_dimension": dict(self.reason_by_dimension),
            "any_range_less": self.any_range_less,
            "vector_dim": len(self.vector),
        }
        if include_slots:
            payload["slots"] = [dict(s) for s in self.slots]
        return payload


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value != value:  # NaN
        return low
    return max(low, min(high, value))


def _sort_key(measurement: MeasurementRecord) -> Tuple[float, str]:
    """确定性排序键：置信度降序，同分按候选编号升序。

    加候选编号这一项是为了让"置信度完全相同"时结果依然稳定——
    否则同一输入可能因排序不稳定而给出不同观测，破坏可复现性。
    """
    return (-float(measurement.confidence), str(measurement.candidate_id))


def fuse_measurements(
    measurements: Sequence[MeasurementRecord],
    config: Optional[TrackTableConfig] = None,
    extra_scalars: Optional[Sequence[float]] = None,
    reason_counts: Optional[Dict[str, int]] = None,
    reason_by_dimension: Optional[Dict[str, int]] = None,
    reference_heading_deg: float = 0.0,
    n_fresh: Optional[int] = None,
) -> FusedObservation:
    """把测量列表打包成定长观测。

    参数
    ----
    extra_scalars : 追加在航迹表之后的**自状态**标量（自身功率、剩余能量等）。
                    这些量属于"本平台精确已知"，不是测量，因此单独放在末尾，
                    不与传感器测量混在一起。
    reference_heading_deg : 计算 `bearing_norm` 时用的参考航向
                    （通常取本平台机头方向，使"前方/后方"有物理含义）。
    """
    cfg = config or TrackTableConfig()
    cfg.validate()

    vector: List[float] = []
    slots: List[Dict[str, Any]] = []

    ordered = sorted(measurements, key=_sort_key)
    used = ordered[: cfg.max_tracks]

    for index in range(cfg.max_tracks):
        if index < len(used):
            measurement = used[index]
            has_range = measurement.range_m is not None
            range_value = float(measurement.range_m or 0.0)
            azimuth = measurement.azimuth_deg
            bearing_deg = 0.0
            if azimuth is not None:
                delta = azimuth - reference_heading_deg
                while delta > 180.0:
                    delta -= 360.0
                while delta < -180.0:
                    delta += 360.0
                bearing_deg = delta
            vr = float(measurement.range_rate_mps or 0.0)
            std_range = (
                float(measurement.std_range_m)
                if measurement.std_range_m is not None
                else cfg.range_scale_m
            )
            std_az = float(measurement.std_az_deg or 0.0)
            slot_values = [
                1.0,
                _clamp(range_value / cfg.range_scale_m),
                1.0 if has_range else 0.0,
                _clamp(bearing_deg / 180.0, -1.0, 1.0),
                _clamp(float(measurement.elevation_deg or 0.0) / 90.0, -1.0, 1.0),
                _clamp(vr / cfg.range_rate_scale_mps, -1.0, 1.0),
                _clamp(std_range / cfg.range_scale_m),
                _clamp(std_az / cfg.std_az_scale_deg),
                _clamp(float(measurement.confidence)),
                _clamp(float(measurement.age_s) / cfg.age_scale_s),
            ]
            slots.append({
                "slot": index,
                "candidate_id": measurement.candidate_id,
                "sensor_id": measurement.sensor_id,
                "sensor_kind": measurement.sensor_kind,
                "has_range": has_range,
                "is_fresh": bool(measurement.is_fresh),
                "age_s": float(measurement.age_s),
            })
            _assert_slot_clean(slots[-1])
        else:
            slot_values = [0.0] * cfg.slot_dim
        vector.extend(slot_values)

    if extra_scalars:
        vector.extend(float(v) for v in extra_scalars)

    # --- 观测质量：由"有多少测量、有多新、不确定度多大"综合 ---
    n_meas = len(measurements)
    fresh = n_fresh if n_fresh is not None else sum(
        1 for m in measurements if m.is_fresh
    )
    if n_meas == 0:
        quality = 0.0
    else:
        freshness = fresh / n_meas
        # 距离标准差相对量程越小越可信
        std_penalty = 0.0
        ranged = [m for m in measurements if m.std_range_m is not None]
        if ranged:
            mean_rel = sum(m.std_range_m / cfg.range_scale_m for m in ranged) / len(ranged)
            std_penalty = min(1.0, mean_rel * 20.0)
        # 刻意**不**把 max_tracks 计入质量：那是打包容量的设定值，
        # 与"看得准不准"无关。早先版本用 `n_meas/max_tracks` 当覆盖率，
        # 结果同样的测量质量会因为 max_tracks 调大而下降，是人为伪影。
        quality = _clamp(freshness * (1.0 - 0.5 * std_penalty))

    fused = FusedObservation(
        vector=vector,
        slots=slots,
        reason_counts=dict(reason_counts or {}),
        reason_by_dimension=dict(reason_by_dimension or {}),
        n_measurements=n_meas,
        n_fresh=fresh,
        n_tracks_used=len(used),
        observation_quality=quality,
        any_range_less=any(m.range_m is None for m in measurements),
    )
    return fused


def _assert_slot_clean(slot: Dict[str, Any]) -> None:
    """确保槽位诊断信息里没有混入真值字段。

    槽位是给人和导出用的，但它**贴着**算法输入，很容易在某次改动里
    顺手把 `truth_id` 塞进去而没人发现。这里每次都检查一遍。
    """
    for key in slot:
        for prefix in FORBIDDEN_KEY_PREFIXES:
            if key.startswith(prefix):
                raise AssertionError(
                    f"融合槽位出现禁止字段 {key!r}：真值/误差/虚警标记不得进入算法输入链路"
                )


def assert_no_truth_leak(payload: Dict[str, Any]) -> None:
    """递归扫描一个字典，发现真值类键就抛错。

    供测试与验收脚本使用：把 `FusedObservation.to_dict()` 或任何要交给
    算法/外部工具的结构丢进来即可。
    """
    stack: List[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                for prefix in FORBIDDEN_KEY_PREFIXES:
                    if str(key).startswith(prefix):
                        raise AssertionError(
                            f"检测到真值泄漏：键 {key!r} 出现在算法可见结构里"
                        )
                stack.append(value)
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
