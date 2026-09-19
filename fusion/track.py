"""航迹与其来源溯源（v4.3）。

为什么溯源是硬要求
------------------
用户明确要求"每条融合结果都能追溯到哪些传感器、哪些时间的测量"。
因此 `Track.sources` 不是调试信息，而是**结构化的、可导出的**字段：
每一条贡献都记录

    (sensor_id, 测量时刻, 消息 ID（若来自通信）, 到达时刻, 使用的权重)

有了它才能回答：
* 这条航迹的位置是**谁**在**什么时候**测的？
* 它的精度为什么是这个量级（哪些测量在起作用）？
* 某条航迹是"本地测量支撑的"还是"全靠别平台共享来的"？
  ——这正是协同收益报告要的东西。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from engine.geometry import Vec3

#: 航迹状态
TRACK_TENTATIVE = "tentative"  # 刚起始，证据不足
TRACK_CONFIRMED = "confirmed"  # 已确认
TRACK_COASTING = "coasting"    # 当前无观测，靠外推维持
TRACK_DROPPED = "dropped"      # 已删除


@dataclass
class TrackSource:
    """一条测量的来源记录（**溯源的核心**）。"""

    sensor_id: str
    measurement_time_s: float
    platform_id: str = ""
    #: 若该测量是经通信到达的，记录消息 ID 与到达时刻
    msg_id: str = ""
    arrived_at: Optional[float] = None
    weight: float = 0.0
    age_at_use_s: float = 0.0
    #: 对应生命周期追踪 ID（可端到端溯源到具体测量）
    trace_id: str = ""
    # --- v4.5 系统级压力测试：逐来源的一致性证据 ---
    #: 该测量相对**预测位置**的残差范数（米）——"这条测量离航迹有多远"
    residual_m: float = 0.0
    #: 该测量的马氏距离平方（用预测协方差归一化）——"这个偏离合不合理"
    innovation_mahalanobis_sq: float = 0.0
    #: 该传感器**上报的**量测标准差（米）。v4.5 加：用它把残差归一化，
    #: 才能回答"这个残差相对它自称的精度是不是太大了"——
    #: 噪声低估（谎报精度）就是靠这个比值暴露的。
    reported_sigma_m: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "platform_id": self.platform_id,
            "measurement_time_s": round(self.measurement_time_s, 6),
            "msg_id": self.msg_id,
            "arrived_at": self.arrived_at,
            "weight": round(self.weight, 6),
            "age_at_use_s": round(self.age_at_use_s, 6),
            "trace_id": self.trace_id,
            "residual_m": round(self.residual_m, 6),
            "innovation_mahalanobis_sq": round(self.innovation_mahalanobis_sq, 6),
            "reported_sigma_m": round(self.reported_sigma_m, 6),
        }


@dataclass
class Track:
    """一条融合航迹。

    状态用**笛卡尔位置 + 速度**（ENU，米 / 米每秒），协方差按对角线处理。
    选择笛卡尔而非极坐标是因为加权融合在笛卡尔下是线性最小二乘，
    实现简单且可解释；代价是极坐标测量的误差会经由雅可比非线性传播，
    这在远距离时误差较大（局限已在模块文档里写明）。
    """

    track_id: str
    created_at: float
    position: Vec3
    velocity: Vec3 = Vec3()
    sigma_position: Vec3 = Vec3(1e9, 1e9, 1e9)
    #: 该航迹被哪些平台的测量支撑
    platforms: List[str] = field(default_factory=list)
    #: 溯源链：最近 N 条贡献测量
    sources: List[TrackSource] = field(default_factory=list)
    #: 直接来自本地传感器的更新次数 / 来自通信的更新次数
    local_updates: int = 0
    remote_updates: int = 0
    hits: int = 0
    misses: int = 0
    last_update_time: Optional[float] = None
    last_measurement_time: Optional[float] = None
    status: str = TRACK_TENTATIVE
    #: 是否由本平台自己发起
    is_local_origin: bool = False
    #: 常速度 Kalman 滤波器（状态预测与量测更新的载体）。
    #: 由 `FusionCenter._initiate()` 注入；None 表示该航迹没有滤波器。
    filter: Any = None

    # ------------------------------------------------------------------

    def freshness(self, now: float) -> float:
        """数据新鲜度（0~1，1 = 刚刚更新过）。

        用**测量时刻**而不是到达时刻来算：一条延迟很大的共享测量
        即使"刚到"，它的信息也是旧的。这正是"必须标记数据新鲜度"的意义。
        """
        if self.last_measurement_time is None:
            return 0.0
        age = max(0.0, now - self.last_measurement_time)
        # 以"3 个更新周期"作为衰减尺度（经验值，配置里可调）
        return 1.0 / (1.0 + age / 3.0)

    def measurement_age_s(self, now: float) -> Optional[float]:
        if self.last_measurement_time is None:
            return None
        return max(0.0, now - self.last_measurement_time)

    def source_summary(self) -> Dict[str, Any]:
        """按传感器聚合的贡献摘要（协同收益报告直接用）。"""
        by_sensor: Dict[str, int] = {}
        for source in self.sources:
            by_sensor[source.sensor_id] = by_sensor.get(source.sensor_id, 0) + 1
        return {
            "n_sources": len(self.sources),
            "by_sensor": by_sensor,
            "platforms": sorted(set(self.platforms)),
            "local_updates": self.local_updates,
            "remote_updates": self.remote_updates,
        }

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "track_id": self.track_id,
            "status": self.status,
            "created_at": self.created_at,
            "x": round(self.position.x, 6),
            "y": round(self.position.y, 6),
            "z": round(self.position.z, 6),
            "vx": round(self.velocity.x, 6),
            "vy": round(self.velocity.y, 6),
            "vz": round(self.velocity.z, 6),
            "sigma_x": round(self.sigma_position.x, 6),
            "sigma_y": round(self.sigma_position.y, 6),
            "sigma_z": round(self.sigma_position.z, 6),
            "hits": self.hits,
            "misses": self.misses,
            "local_updates": self.local_updates,
            "remote_updates": self.remote_updates,
            "is_local_origin": self.is_local_origin,
            "last_update_time": self.last_update_time,
            "last_measurement_time": self.last_measurement_time,
            "n_sources": len(self.sources),
            "platforms": sorted(set(self.platforms)),
        }
        if now is not None:
            payload["freshness"] = round(self.freshness(now), 6)
            payload["measurement_age_s"] = self.measurement_age_s(now)
        return payload

    def describe(self) -> str:
        return (
            f"{self.track_id}[{self.status}] "
            f"pos=({self.position.x:.0f}, {self.position.y:.0f}, {self.position.z:.0f}) "
            f"σ=({self.sigma_position.x:.1f}, {self.sigma_position.y:.1f}) "
            f"hits={self.hits} 本地={self.local_updates} 共享={self.remote_updates} "
            f"来源={sorted({s.sensor_id for s in self.sources})}"
        )


@dataclass
class TracksSnapshot:
    """某一时刻的全部航迹（含汇总统计）。"""

    time_s: float
    tracks: List[Track] = field(default_factory=list)
    #: 本步参与融合的测量数（含本地与共享）
    n_measurements: int = 0
    n_local_measurements: int = 0
    n_remote_measurements: int = 0
    #: 因过期/过旧被拒绝的测量数
    n_stale_rejected: int = 0
    #: 因观测对象不匹配被拒绝的测量数（例如 ESM 测的是辐射源而非目标）
    n_kind_rejected: int = 0
    #: 通过准入检查、真正参与关联的测量数（v4.5：用于算时效拒绝率）
    n_accepted_measurements: int = 0

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    @property
    def n_confirmed(self) -> int:
        return sum(1 for t in self.tracks if t.status == TRACK_CONFIRMED)

    def track_by_id(self, track_id: str) -> Optional[Track]:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def freshness_stats(self) -> Dict[str, float]:
        values = [t.freshness(self.time_s) for t in self.tracks]
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": self.time_s,
            "n_tracks": self.n_tracks,
            "n_confirmed": self.n_confirmed,
            "n_measurements": self.n_measurements,
            "n_local_measurements": self.n_local_measurements,
            "n_remote_measurements": self.n_remote_measurements,
            "n_stale_rejected": self.n_stale_rejected,
            "n_kind_rejected": self.n_kind_rejected,
            "n_accepted_measurements": self.n_accepted_measurements,
            "freshness": self.freshness_stats(),
            "tracks": [t.to_dict(now=self.time_s) for t in self.tracks],
        }
