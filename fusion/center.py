"""融合中心：基础多目标跟踪器（v4.4）。

升级内容（相对 v4.3 的"按置信度加权平均"）
------------------------------------------
| 能力 | v4.3（旧） | v4.4（本版） |
| --- | --- | --- |
| 状态 | 仅位置 | **位置 + 速度**（6 维） |
| 预测 | 无 | **常速度 Kalman 预测**（含过程噪声） |
| 关联 | 固定米/度门限 | **马氏距离门限**（用预测协方差归一化）+ 角度兜底 |
| 速度 | 相邻位置差分 | Kalman 滤波估计 |
| 漏检 | 直接记 miss | **短时保持（coasting）**，靠预测外推 |
| track_id | 会重排 | **稳定**（单调递增，不随排序/删除变化） |
| 溯源 | 有 | 有 + **全链路生命周期追踪** |

算法闭环
--------
```
每步：
 1. 全部航迹预测到当前时刻（predict_to(now)，含协方差增长）
 2. 逐条测量：时效检查 → 观测对象检查 → 马氏门限 + 最近邻贪心关联
 3. 关联上 → Kalman 更新；未关联上 → 尝试起始新航迹
 4. 未获测量的航迹：misses+1 → coasting → 达阈值删除
 5. 输出 TracksSnapshot（含逐测量生命周期）
```

⚠️ 真值隔离
------------
本模块**只吃测量对象**，不访问 `Scene` / 实体真值、不读 `truth_id`。
真值只在 `evaluate_cooperative_sensing.py` 里用于离线算
位置/速度 RMSE、轨迹召回率、误关联率、丢轨率、连续性。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.geometry import Vec3, enu_to_spherical
from fusion.kalman import ConstantVelocityKalmanFilter, KalmanConfig
from fusion.lifecycle import (
    REJECT_AZIMUTH,
    REJECT_DROPPED,
    REJECT_ELEVATION,
    REJECT_MAHALANOBIS,
    STAGE_ASSOCIATED,
    STAGE_GATE_REJECTED,
    STAGE_KIND_REJECTED,
    STAGE_STALE_REJECTED,
    STAGE_TRACK_CREATED,
    STAGE_TRACK_UPDATED,
    AssociationCandidate,
    LifecycleLog,
    MeasurementTrace,
)
from fusion.track import (
    TRACK_COASTING,
    TRACK_CONFIRMED,
    TRACK_DROPPED,
    TRACK_TENTATIVE,
    Track,
    TrackSource,
    TracksSnapshot,
)


@dataclass
class FusionConfig:
    """跟踪器配置。"""

    kalman: KalmanConfig = field(default_factory=KalmanConfig)

    #: 马氏距离平方门限（3 自由度，9.0 约对应 97% 置信椭球）
    gate_mahalanobis_sq: float = 9.0
    #: 角度兜底门限（度）：防止预测协方差过大时马氏门限过度宽松
    gate_az_deg: float = 8.0
    gate_el_deg: float = 8.0

    confirm_hits: int = 2
    coast_after_misses: int = 1
    drop_after_misses: int = 5
    max_measurement_age_s: float = 3.0
    seed_sigma_floor_m: float = 10.0
    max_sources: int = 16
    #: 只接受这些类型的传感器测量参与目标航迹
    #: （被动 ESM 观测的是雷达辐射源，不是目标）
    accepted_sensor_kinds: Tuple[str, ...] = ("radar",)

    def validate(self) -> None:
        self.kalman.validate()
        if self.confirm_hits < 1:
            raise ValueError("confirm_hits 至少为 1")
        if self.drop_after_misses < 1:
            raise ValueError("drop_after_misses 至少为 1")
        if self.max_measurement_age_s <= 0:
            raise ValueError("max_measurement_age_s 必须为正")
        if self.gate_mahalanobis_sq <= 0:
            raise ValueError("gate_mahalanobis_sq 必须为正")


def measurement_to_cartesian(
    measurement: Any, sensor_position: Vec3
) -> Optional[Tuple[Vec3, Vec3]]:
    """极坐标测量 → 笛卡尔位置 + 各轴标准差。

    仅对含距离的测量有效（被动传感器没有距离量测，无法定位）。
    误差传播取各向同性横向误差 `R·σ_angle`，与距离误差平方和。
    **局限**：未做完整雅可比，远距离会略微高估横向不确定度。
    """
    if getattr(measurement, "range_m", None) is None:
        return None
    rng_m = float(measurement.range_m)
    az = math.radians(float(getattr(measurement, "azimuth_deg", 0.0) or 0.0))
    el = math.radians(float(getattr(measurement, "elevation_deg", 0.0) or 0.0))
    position = sensor_position + Vec3(
        rng_m * math.cos(el) * math.sin(az),
        rng_m * math.cos(el) * math.cos(az),
        rng_m * math.sin(el),
    )
    sigma_range = float(getattr(measurement, "std_range_m", None) or 0.0)
    lateral = rng_m * math.radians(max(
        float(getattr(measurement, "std_az_deg", None) or 0.0),
        float(getattr(measurement, "std_el_deg", None) or 0.0),
    ))
    sigma_axis = math.hypot(sigma_range, lateral)
    return position, Vec3(sigma_axis, sigma_axis, sigma_axis)


class FusionCenter:
    """单个平台的跟踪器。"""

    def __init__(
        self,
        platform_id: str,
        config: Optional[FusionConfig] = None,
        own_sensor_ids: Optional[Sequence[str]] = None,
        lifecycle: Optional[LifecycleLog] = None,
    ) -> None:
        self.platform_id = platform_id
        self.config = config or FusionConfig()
        self.config.validate()
        self.own_sensor_ids = set(own_sensor_ids or [])
        self.lifecycle = lifecycle if lifecycle is not None else LifecycleLog(False)
        self.tracks: List[Track] = []
        self._track_counter = 0
        self.history: List[TracksSnapshot] = []
        self.retired_track_ids: List[str] = []
        self.stats: Dict[str, int] = {
            "initiated": 0, "confirmed": 0, "dropped": 0,
            "updates_local": 0, "updates_remote": 0,
            "stale_rejected": 0, "kind_rejected": 0, "gate_rejected": 0,
            #: 关联歧义次数：≥2 条候选航迹同时通过门限（v4.5）
            "ambiguous": 0,
        }

    def reset(self) -> None:
        self.tracks.clear()
        self._track_counter = 0
        self.history.clear()
        self.retired_track_ids.clear()
        for key in self.stats:
            self.stats[key] = 0
        self.lifecycle.reset()

    def _next_track_id(self) -> str:
        self._track_counter += 1
        return f"{self.platform_id}-T{self._track_counter}"

    # ------------------------------------------------------------------

    def predict_to(self, time_s: float) -> None:
        """把所有航迹预测到 `time_s`（含协方差增长）。

        ⚠️ 本方法会更新 `track.last_update_time`，因此该字段的含义是
        **「最近一次状态推进（预测或量测更新）的时刻」**，
        而**不能**用来判断"这一帧有没有拿到测量"。
        判"本帧是否有测量"必须用 `update()` 内的逐步身份集合——
        v4.4 曾经用 `abs(last_update_time - now) < 1e-12` 来判断，
        而 `predict_to(now)` 恰好把每一条航迹的该字段都写成了 `now`，
        于是 miss 分支永远走不到：`misses` 恒为 0，coasting 与删除
        **全是死代码**（实测一条航迹连续 27 帧无观测仍为 confirmed、
        misses=0、dropped=0）。多目标压力测试把这个 bug 抓了出来。
        """
        for track in self.tracks:
            reference = track.last_update_time
            if reference is None:
                continue
            dt = time_s - reference
            if dt <= 0.0:
                continue
            track.filter.predict(dt)
            track.position = track.filter.position
            track.velocity = track.filter.velocity
            track.sigma_position = track.filter.position_sigma()
            track.last_update_time = time_s

    # ------------------------------------------------------------------

    def update(
        self,
        measurements: Sequence[Any],
        now: float,
        sensor_positions: Dict[str, Vec3],
        remote_measurement_flags: Optional[Sequence[bool]] = None,
    ) -> TracksSnapshot:
        """用一批测量更新航迹，返回本步快照。

        `measurements` 应为「本地测量 + **已到达的**远端测量」，
        由调用方保证只传已到达的（见 `communication/bus.py`）。
        """
        flags = list(remote_measurement_flags or [False] * len(measurements))
        snapshot = TracksSnapshot(time_s=now)
        snapshot.n_measurements = len(measurements)
        snapshot.n_remote_measurements = sum(1 for f in flags if f)
        snapshot.n_local_measurements = len(measurements) - snapshot.n_remote_measurements

        self.predict_to(now)

        # --- 1) 准入检查 ---
        candidates: List[Tuple[Any, MeasurementTrace, Vec3, Vec3, bool]] = []
        for index, measurement in enumerate(measurements):
            is_remote = flags[index] if index < len(flags) else False
            trace = self.lifecycle.open_trace(
                candidate_id=str(getattr(measurement, "candidate_id", "")),
                sensor_id=str(getattr(measurement, "sensor_id", "")),
                sensor_kind=str(getattr(measurement, "sensor_kind", "")),
                source_platform=str(getattr(measurement, "platform_id", "") or
                                    ("" if is_remote else self.platform_id)),
                msg_id=str(getattr(measurement, "msg_id", "")),
                is_remote=is_remote,
                measured_at=float(getattr(measurement, "time_s", now)),
                generated_at=float(getattr(measurement, "time_s", now)),
            )
            if is_remote:
                self.lifecycle.mark_arrived(trace, now)
            else:
                self.lifecycle.mark_sent(trace, now)

            kind = str(getattr(measurement, "sensor_kind", "radar"))
            if kind not in self.config.accepted_sensor_kinds:
                snapshot.n_kind_rejected += 1
                self.stats["kind_rejected"] += 1
                self.lifecycle.reject(
                    trace, STAGE_KIND_REJECTED,
                    f"sensor_kind={kind} 不在 {list(self.config.accepted_sensor_kinds)}",
                )
                continue

            m_time = float(getattr(measurement, "time_s", now))
            age = now - m_time
            if age > self.config.max_measurement_age_s:
                snapshot.n_stale_rejected += 1
                self.stats["stale_rejected"] += 1
                self.lifecycle.reject(
                    trace, STAGE_STALE_REJECTED,
                    f"age={age:.3f}s > {self.config.max_measurement_age_s:g}s",
                )
                continue

            sensor_id = str(getattr(measurement, "sensor_id", ""))
            sensor_position = sensor_positions.get(sensor_id)
            if sensor_position is None:
                self.lifecycle.reject(
                    trace, STAGE_STALE_REJECTED, f"未知传感器位置 {sensor_id}"
                )
                continue
            converted = measurement_to_cartesian(measurement, sensor_position)
            if converted is None:
                if trace is not None:
                    trace.note = "无距离量测，不参与定位"
                continue
            position, sigma = converted
            candidates.append((measurement, trace, position, sigma, is_remote))

        #: 通过准入检查、真正参与关联的测量数（v4.5）
        snapshot.n_accepted_measurements = len(candidates)

        # --- 2) 逐条顺序关联 + Kalman 更新 ---
        # ⚠️ 这里**必须**逐条处理，不能先批量关联再统一更新。
        # 批量关联时同一时刻的两条测量会竞争同一条航迹：一条胜出、
        # 另一条关联不上 → 起始重复航迹；而且两条信息**没有先后作用在
        # 同一条航迹上**，多传感器融合降低协方差的机制根本没发生。
        # 实测表现是"共享后 RMSE 反而更差、受限共享比理想共享还好"
        # 这种自相矛盾的结果。
        # 逐条处理时，第二条测量会关联到刚被第一条更新过的航迹，
        # 两次 Kalman 更新真正合成到同一条航迹上。
        #
        # `updated_this_step` 用 `id()` 而不是 Track 对象：
        # `Track` 是默认 `eq=True` 的 dataclass，因此不可哈希。
        updated_this_step: set = set()
        for measurement, trace, position, sigma, is_remote in candidates:
            assignments, _unmatched = self._associate(
                [(measurement, trace, position, sigma, is_remote)],
                sensor_positions, audit=True,
            )
            if assignments:
                _ci, track = assignments[0]
                cost = track.filter.mahalanobis_sq(position, sigma)
                # 残差必须在**更新之前**算：更新之后 measured 与 track 就重合了，
                # 残差会永远是 0。v4.5 系统级压力测试靠它做逐来源一致性判断。
                residual_m = (position - track.position).norm()
                reported_sigma_m = float(sigma.x)
                track.filter.update(position, sigma)
                updated_this_step.add(id(track))
                track.position = track.filter.position
                track.velocity = track.filter.velocity
                track.sigma_position = track.filter.position_sigma()
                track.last_measurement_time = float(getattr(measurement, "time_s", now))
                track.last_update_time = now
                track.hits += 1
                track.misses = 0
                if track.hits >= self.config.confirm_hits:
                    if track.status == TRACK_TENTATIVE:
                        self.stats["confirmed"] += 1
                    track.status = TRACK_CONFIRMED
                track.sources.append(self._source(
                    measurement, trace, now, is_remote,
                    residual_m=residual_m, mahalanobis_sq=cost,
                    reported_sigma_m=reported_sigma_m,
                ))
                if len(track.sources) > self.config.max_sources:
                    track.sources = track.sources[-self.config.max_sources:]
                platform_id = str(getattr(measurement, "platform_id", "") or
                                  (self.platform_id if not is_remote else "remote"))
                if platform_id and platform_id not in track.platforms:
                    track.platforms.append(platform_id)
                if is_remote:
                    track.remote_updates += 1
                    self.stats["updates_remote"] += 1
                else:
                    track.local_updates += 1
                    self.stats["updates_local"] += 1
                if trace is not None:
                    trace.mark(STAGE_ASSOCIATED)
                    trace.mark(STAGE_TRACK_UPDATED)
                    trace.track_id = track.track_id
                    trace.association_cost = cost
                    trace.consumed_at = now
            else:
                # 关联不上：起始新航迹（可能是新目标，也可能是门限拒绝后的起点）
                #
                # ⚠️ 这里**不再**重复统计 gate_rejected：
                # `_associate` 已在上面按同一条件（最近候选超过门限）记过一次。
                # 旧版在两处各记一次，导致 `stats["gate_rejected"]` 恰好是真实值的
                # **2 倍**，这个数字被写进了协同感知报告的 `gate_rejected` 列。
                track = self._initiate(position, sigma, measurement, trace, now, is_remote)
                if track is not None:
                    # 新起始的航迹本步就是"有测量支撑"的，不能记成 miss
                    updated_this_step.add(id(track))
                if track is not None and trace is not None:
                    # 测量没能更新已有航迹、而是开了一条新航迹：
                    # 关联审计里的"最终选择"要跟着更新，否则审计会显示"没选中任何航迹"。
                    trace.chosen_track_id = track.track_id
                    trace.mark(STAGE_ASSOCIATED)
                    trace.mark(STAGE_TRACK_CREATED)
                    trace.track_id = track.track_id
                    trace.consumed_at = now

        # --- 5) miss / coasting / 删除 ---
        #
        # 用**本步被更新过的航迹身份集合**判断，而不是比较 `last_update_time`：
        # 后者已经被上面的 `predict_to(now)` 全部写成 `now`（见该方法文档）。
        for track in list(self.tracks):
            if id(track) in updated_this_step:
                continue
            track.misses += 1
            if track.misses >= self.config.drop_after_misses:
                track.status = TRACK_DROPPED
                self.retired_track_ids.append(track.track_id)
                self.stats["dropped"] += 1
                self.tracks.remove(track)
            elif track.misses >= self.config.coast_after_misses:
                track.status = TRACK_COASTING

        snapshot.tracks = list(self.tracks)
        self.history.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------

    def _associate(
        self,
        candidates: Sequence[Tuple[Any, MeasurementTrace, Vec3, Vec3, bool]],
        sensor_positions: Dict[str, Vec3],
        audit: bool = False,
    ) -> Tuple[List[Tuple[int, Track]], List[int]]:
        """马氏门限 + 最近邻贪心，另加角度兜底门限。

        `audit=True` 时把**每一次候选评估**（含被门限拒绝的）记进生命周期，
        这是"关联可审计"的实现点：只记最终结果的话，
        ID switch 只能靠指标反推，无法回答"这一帧为什么关联错了"。
        """
        cfg = self.config
        pairs: List[Tuple[float, int, int]] = []
        #: ci -> 该测量评估过的全部候选（含未过门限的）
        evaluations: Dict[int, List[AssociationCandidate]] = {}
        for ci, (measurement, _trace, position, sigma, _remote) in enumerate(candidates):
            az_meas = getattr(measurement, "azimuth_deg", None)
            el_meas = getattr(measurement, "elevation_deg", None)
            sensor_position = sensor_positions.get(
                str(getattr(measurement, "sensor_id", ""))
            )
            for ti, track in enumerate(self.tracks):
                if track.status == TRACK_DROPPED:
                    if audit:
                        evaluations.setdefault(ci, []).append(AssociationCandidate(
                            track_id=track.track_id,
                            mahalanobis_sq=float("inf"),
                            gate_mahalanobis_sq=cfg.gate_mahalanobis_sq,
                            passed=False, reject_reason=REJECT_DROPPED,
                        ))
                    continue
                cost = track.filter.mahalanobis_sq(position, sigma)
                az_diff: Optional[float] = None
                el_diff: Optional[float] = None
                reason = ""
                if cost > cfg.gate_mahalanobis_sq:
                    reason = REJECT_MAHALANOBIS
                elif az_meas is not None and sensor_position is not None:
                    pred = enu_to_spherical(track.position - sensor_position)
                    az_diff = abs(
                        ((float(az_meas) - pred.azimuth_deg + 180.0) % 360.0) - 180.0
                    )
                    if az_diff > cfg.gate_az_deg:
                        reason = REJECT_AZIMUTH
                    elif el_meas is not None:
                        el_diff = abs(float(el_meas) - pred.elevation_deg)
                        if el_diff > cfg.gate_el_deg:
                            reason = REJECT_ELEVATION
                if audit:
                    evaluations.setdefault(ci, []).append(AssociationCandidate(
                        track_id=track.track_id,
                        mahalanobis_sq=cost,
                        gate_mahalanobis_sq=cfg.gate_mahalanobis_sq,
                        residual_m=(position - track.position).norm(),
                        az_diff_deg=az_diff, el_diff_deg=el_diff,
                        passed=(reason == ""), reject_reason=reason,
                    ))
                if reason:
                    continue
                pairs.append((cost, ci, ti))

        pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        used_c: set = set()
        used_t: set = set()
        assignments: List[Tuple[int, Track]] = []
        for cost, ci, ti in pairs:
            if ci in used_c or ti in used_t:
                continue
            used_c.add(ci)
            used_t.add(ti)
            assignments.append((ci, self.tracks[ti]))

        # 未被关联上的候选：只有在"本来有航迹可关联"时才算门限拒绝，
        # 否则它是新目标，应当起始新航迹。
        # 这一点很容易写错——写成无条件记录的话，首批目标全会被记成
        # gate_rejected，漏斗统计就失真了。
        if self.tracks:
            for ci, (measurement, trace, position, sigma, _r) in enumerate(candidates):
                if ci in used_c:
                    continue
                best = min(
                    (t.filter.mahalanobis_sq(position, sigma) for t in self.tracks
                     if t.status != TRACK_DROPPED),
                    default=float("inf"),
                )
                if best > cfg.gate_mahalanobis_sq:
                    self.lifecycle.reject(
                        trace, STAGE_GATE_REJECTED,
                        f"最近马氏距离²={best:.2f} > {cfg.gate_mahalanobis_sq:g}",
                    )
                    self.stats["gate_rejected"] += 1

        if audit:
            chosen_by_ci = {ci: track.track_id for ci, track in assignments}
            for ci, (measurement, trace, _p, _s, _r) in enumerate(candidates):
                chosen = chosen_by_ci.get(ci, "")
                items = evaluations.get(ci, [])
                for item in items:
                    item.chosen = bool(chosen) and item.track_id == chosen
                self.lifecycle.record_association(trace, items, chosen)
                if items and sum(1 for i in items if i.passed) >= 2:
                    self.stats["ambiguous"] += 1

        unmatched = [ci for ci in range(len(candidates)) if ci not in used_c]
        return assignments, unmatched

    def _initiate(
        self, position: Vec3, sigma: Vec3, measurement: Any,
        trace: Optional[MeasurementTrace], now: float, is_remote: bool,
    ) -> Optional[Track]:
        m_time = float(getattr(measurement, "time_s", now))
        scaled = Vec3(
            max(sigma.x, self.config.seed_sigma_floor_m),
            max(sigma.y, self.config.seed_sigma_floor_m),
            max(sigma.z, self.config.seed_sigma_floor_m),
        )
        kalman = ConstantVelocityKalmanFilter(
            position=position, position_sigma=scaled, config=self.config.kalman
        )
        track = Track(
            track_id=self._next_track_id(),
            created_at=m_time,
            position=position,
            velocity=Vec3(),
            sigma_position=kalman.position_sigma(),
            status=(TRACK_CONFIRMED if self.config.confirm_hits <= 1 else TRACK_TENTATIVE),
            last_update_time=now,
            last_measurement_time=m_time,
            is_local_origin=not is_remote,
            filter=kalman,
        )
        track.hits = 1
        track.sources.append(self._source(measurement, trace, now, is_remote))
        platform_id = str(getattr(measurement, "platform_id", "") or
                          (self.platform_id if not is_remote else "remote"))
        if platform_id:
            track.platforms.append(platform_id)
        if is_remote:
            track.remote_updates += 1
            self.stats["updates_remote"] += 1
        else:
            track.local_updates += 1
            self.stats["updates_local"] += 1
        self.tracks.append(track)
        self.stats["initiated"] += 1
        return track

    def _source(
        self, measurement: Any, trace: Optional[MeasurementTrace],
        now: float, is_remote: bool,
        residual_m: float = 0.0, mahalanobis_sq: float = 0.0,
        reported_sigma_m: float = 0.0,
    ) -> TrackSource:
        m_time = float(getattr(measurement, "time_s", now))
        return TrackSource(
            sensor_id=str(getattr(measurement, "sensor_id", "")),
            measurement_time_s=m_time,
            platform_id=str(getattr(measurement, "platform_id", "") or
                            ("" if is_remote else self.platform_id)),
            msg_id=str(getattr(measurement, "msg_id", "")),
            arrived_at=(now if is_remote else None),
            weight=1.0,
            age_at_use_s=max(0.0, now - m_time),
            trace_id=(trace.trace_id if trace is not None else ""),
            residual_m=residual_m,
            innovation_mahalanobis_sq=mahalanobis_sq,
            reported_sigma_m=reported_sigma_m,
        )

    # ------------------------------------------------------------------

    def snapshot(self, now: float) -> TracksSnapshot:
        return TracksSnapshot(time_s=now, tracks=list(self.tracks))

    def describe(self) -> str:
        return (
            f"跟踪器[{self.platform_id}]：{len(self.tracks)} 条航迹，"
            f"起始 {self.stats['initiated']}，确认 {self.stats['confirmed']}，"
            f"删除 {self.stats['dropped']}，本地更新 {self.stats['updates_local']}，"
            f"共享更新 {self.stats['updates_remote']}"
        )
