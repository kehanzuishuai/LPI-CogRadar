"""系统级压力指标（v4.5 P2 第二阶段）。

与 `metrics.py` 的分工
----------------------
`MultiTargetMetrics` 算的是**关联层**指标（换号/碎裂/重复/假航迹/纯度…），
四个核心场景共用。本模块算的是**系统级**指标，按场景挑着用：

| 场景 | 指标组 |
| --- | --- |
| S5 机动失配 | 机动前后创新量、门限拒绝、coasting 时长、恢复时间、预测误差 |
| S6 交接 | 交接连续性、交接延迟、远端贡献比例、短时双轨 |
| S7 时序 | 乱序率、时效拒绝率、突发长度、恢复时间、中断期/恢复后连续率、航迹年龄 |
| S8 偏差 | 逐来源残差、创新一致性、航迹协方差、传感器健康分、远端贡献比例 |

⚠️ 真值仍然**只在这里**被用于算误差；所有指标不进算法。
⚠️ **不读 truth_id 做决策**：本模块只做离线统计。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 创新一致性判据：3 自由度卡方 95% 分位数（马氏距离² 的正常上界）
CHI2_3DOF_95 = 7.815
#: 创新一致性判据：3 自由度卡方 99% 分位数
CHI2_3DOF_99 = 11.345
#: 传感器健康分判据（v4.5 S8）：
#: * `HEALTH_RATIO_ALARM`：**相对**同场景最佳传感器的倍数（1.5 = 差 50% 以上）
#: * `HEALTH_ABSOLUTE_FLOOR`：绝对下限（归一化残差 2.0，即残差达到自称 σ 的 2 倍）
#:
#: ⚠️ 这是一个**相对**判据，只能回答"哪部传感器明显比同伴差"，
#: **不能**回答"哪部传感器是坏的"——只有两部传感器时，
#: 被标记的那部只是"更差的那部"，偏差可能实际在另一部上。
#: 因此它只作**诊断分支**，绝不自动剔除任何传感器。
HEALTH_RATIO_ALARM = 1.5
HEALTH_ABSOLUTE_FLOOR = 2.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _norm(values: Any) -> float:
    """把 `Vec3` 或 `(x, y, z)` 归一成范数。"""
    if values is None:
        return 0.0
    if hasattr(values, "norm"):
        return float(values.norm())
    try:
        return math.sqrt(sum(float(v) ** 2 for v in values))
    except TypeError:
        return 0.0


@dataclass
class ManeuverMetrics:
    """S5：机动目标模型失配。"""

    maneuvers: Sequence[Any] = ()
    #: 每次机动前后各取多少秒作为观察窗
    window_s: float = 4.0
    #: 机动后认为"已恢复"的判据：残差回到机动前水平的多少倍以内
    recovery_tolerance: float = 1.5

    _frames: List[Dict[str, Any]] = field(default_factory=list)
    _by_target: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def add_frame(self, frame: Dict[str, Any]) -> None:
        self._frames.append(frame)
        for entry in frame.get("target_metrics", []) or []:
            # 失配指标 = max(被接受测量的残差, **被门限拒绝**的最大残差)。
            # 为什么必须取最大：目标一机动，预测位置就偏出去，
            # 那条偏离最大的测量**最可能直接被门限拒掉**——
            # 只看"被接受的测量"会系统性漏掉失配信号（第一版正是如此，
            # 结果机动目标与匀速对照目标的残差比几乎一样，看不出机动）。
            accepted = entry.get("residual_m")
            rejected = entry.get("rejected_residual_m")
            values = [v for v in (accepted, rejected) if v is not None]
            innovations = [
                v for v in (entry.get("innovation_mahalanobis_sq"),
                            entry.get("rejected_mahalanobis_sq"))
                if v is not None
            ]
            self._by_target.setdefault(entry["target_id"], []).append({
                **entry,
                "residual_m": max(values) if values else None,
                "accepted_residual_m": accepted,
                "innovation_mahalanobis_sq": (
                    max(innovations) if innovations else None),
            })

    # ------------------------------------------------------------------

    def _target_series(self, target_id: str, key: str) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for entry in self._by_target.get(target_id, []):
            value = entry.get(key)
            if value is None:
                continue
            out.append((float(entry["time_s"]), float(value)))
        return out

    def _window_stats(self, series: Sequence[Tuple[float, float]],
                      start: float, end: float) -> Dict[str, float]:
        values = [v for t, v in series if start <= t < end]
        if not values:
            return {"n": 0, "mean": 0.0, "max": 0.0}
        return {"n": len(values), "mean": _mean(values), "max": max(values)}

    def result(self) -> Dict[str, Any]:
        maneuvered = sorted({str(getattr(m, "target_id", "")) for m in self.maneuvers})
        out: Dict[str, Any] = {
            "n_maneuvers": len(self.maneuvers),
            "maneuver_labels": [
                m.describe() if hasattr(m, "describe") else str(m)
                for m in self.maneuvers
            ],
        }
        # 机动前/后的创新量与残差对比（按目标分别统计）
        for role, target_id in (("maneuvered", maneuvered[0] if maneuvered else ""),
                                ("reference", self._reference_target())):
            if not target_id:
                continue
            residual = self._target_series(target_id, "residual_m")
            innovation = self._target_series(target_id, "innovation_mahalanobis_sq")
            gate_rejects = self._target_series(target_id, "gate_rejected")
            recovery = self._recovery_times(target_id, residual)
            pre_windows, post_windows = [], []
            for step in self.maneuvers:
                t0 = float(getattr(step, "time_s", 0.0))
                pre_windows.append(self._window_stats(
                    residual, t0 - self.window_s, t0))
                post_windows.append(self._window_stats(
                    residual, t0, t0 + self.window_s))
            pre_mean = _mean([w["mean"] for w in pre_windows]) if pre_windows else 0.0
            post_mean = _mean([w["mean"] for w in post_windows]) if post_windows else 0.0
            out.update({
                f"{role}_target_id": target_id,
                f"{role}_residual_pre_m": pre_mean,
                f"{role}_residual_post_m": post_mean,
                f"{role}_residual_max_m": max(
                    (v for _t, v in residual), default=0.0),
                f"{role}_innovation_mean": _mean([v for _t, v in innovation]),
                f"{role}_innovation_max": max(
                    (v for _t, v in innovation), default=0.0),
                f"{role}_gate_rejected": sum(
                    v for _t, v in gate_rejects),
                f"{role}_recovery_time_s": recovery,
                f"{role}_residual_ratio": (
                    post_mean / pre_mean if pre_mean > 1e-9 else 0.0),
            })
        return out

    def _reference_target(self) -> str:
        maneuvered = {str(getattr(m, "target_id", "")) for m in self.maneuvers}
        for target_id in self._by_target:
            if target_id not in maneuvered:
                return target_id
        return ""

    def _recovery_times(self, target_id: str,
                        series: Sequence[Tuple[float, float]]) -> float:
        """机动后残差回到"机动前水平 × 容差"以内所需的时间（取平均，秒）。

        只统计真值目标与航迹**配对**的帧——没有航迹就没有残差，
        那种情况应当由"丢轨/碎裂"指标去反映，不该混进恢复时间。
        """
        if not series:
            return 0.0
        recoveries: List[float] = []
        for step in self.maneuvers:
            t0 = float(getattr(step, "time_s", 0.0))
            baseline_values = [v for t, v in series
                               if t0 - self.window_s <= t < t0]
            if not baseline_values:
                continue
            threshold = _mean(baseline_values) * self.recovery_tolerance
            recovered_at: Optional[float] = None
            for time_s, value in series:
                if time_s < t0:
                    continue
                if value <= threshold:
                    recovered_at = time_s
                    break
            if recovered_at is not None:
                recoveries.append(recovered_at - t0)
        return _mean(recoveries)


@dataclass
class HandoverMetrics:
    """S6：多雷达覆盖交接。"""

    incoming_sensor_id: str = ""
    target_id: str = ""
    overlap_window_s: Tuple[float, float] = (0.0, 0.0)
    #: A 的可见窗口（本地传感器）
    local_window_s: Tuple[float, float] = (0.0, 0.0)

    _frames: List[Dict[str, Any]] = field(default_factory=list)

    def add_frame(self, frame: Dict[str, Any]) -> None:
        self._frames.append(frame)

    def result(self) -> Dict[str, Any]:
        if not self._frames:
            return {}
        # ① 交接连续性：在**重叠窗口**内，目标是否一直有航迹
        overlap_frames = [f for f in self._frames
                          if self.overlap_window_s[0] <= f["time_s"]
                          <= self.overlap_window_s[1]]
        # ② 交接之后（本地已经看不到）是否仍有航迹
        after_frames = [f for f in self._frames
                        if f["time_s"] > self.local_window_s[1]]
        covered_overlap = sum(1 for f in overlap_frames if f["n_tracks"] > 0)
        covered_after = sum(1 for f in after_frames if f["n_tracks"] > 0)

        # ③ 交接延迟：新传感器首次出测量 → 航迹首次带上它的来源
        first_meas = self._first_time(
            lambda f: f.get("incoming_sensor_detections", 0) > 0
        )
        first_source = self._first_time(
            lambda f: self.incoming_sensor_id in (f.get("track_sensors") or [])
        )
        delay = (first_source - first_meas
                 if first_meas is not None and first_source is not None else None)

        # ④ 远端贡献比例：航迹来源里非本地传感器的占比
        sources = [s for f in self._frames for s in (f.get("track_sources") or [])]
        remote = [s for s in sources if s.get("sensor_id") != self._local_sensor()]
        ratio = (len(remote) / len(sources)) if sources else 0.0

        # ⑤ 短时双轨：重叠窗口内一度出现 2 条以上航迹的帧数
        double_frames = sum(1 for f in overlap_frames if f["n_tracks"] > 1)

        # ⑥ **同一航迹内部**是否真的发生了跨传感器接力。
        # 这是"覆盖接力"与"身份接力"的分界线：
        #   覆盖接力 = 目标一直有航迹（可能是**新**航迹接上的）
        #   身份接力 = **同一条** track_id 先后拿到了两个传感器的来源
        # 只有后者才叫"多雷达协同实现了航迹接力"。
        within_track = 0
        within_track_ids: List[str] = []
        for frame in self._frames:
            for track_id, sensors in (frame.get("track_sensor_map") or {}).items():
                if len(sensors) >= 2:
                    within_track += 1
                    if track_id not in within_track_ids:
                        within_track_ids.append(track_id)
        return {
            "handover_target_id": self.target_id,
            "handover_incoming_sensor": self.incoming_sensor_id,
            "handover_overlap_s": round(
                self.overlap_window_s[1] - self.overlap_window_s[0], 6),
            "handover_continuity": (covered_overlap / len(overlap_frames))
                                   if overlap_frames else 0.0,
            "handover_continuity_after_local": (covered_after / len(after_frames))
                                               if after_frames else 0.0,
            "handover_delay_s": delay,
            "handover_first_measurement_s": first_meas,
            "handover_first_source_s": first_source,
            "remote_contribution_ratio": ratio,
            "n_remote_sources": len(remote),
            "n_track_sources": len(sources),
            "short_double_track_frames": double_frames,
            "n_overlap_frames": len(overlap_frames),
            "n_frames_after_local_lost": len(after_frames),
            "within_track_handover_frames": within_track,
            "within_track_handover_ids": within_track_ids,
            "within_track_handover_achieved": bool(within_track_ids),
        }

    def _local_sensor(self) -> str:
        for frame in self._frames:
            for sensor_id in frame.get("own_sensor_ids") or []:
                return str(sensor_id)
        return ""

    def _first_time(self, predicate) -> Optional[float]:
        for frame in self._frames:
            if predicate(frame):
                return float(frame["time_s"])
        return None


@dataclass
class TimingMetrics:
    """S7：通信时序压力。"""

    outage_windows: Tuple[Tuple[float, float], ...] = ()
    oosm_summary: Dict[str, Any] = field(default_factory=dict)

    _frames: List[Dict[str, Any]] = field(default_factory=list)
    _decisions: List[Dict[str, Any]] = field(default_factory=list)

    def add_frame(self, frame: Dict[str, Any]) -> None:
        self._frames.append(frame)
        self._decisions.extend(frame.get("oosm_decisions") or [])

    # ------------------------------------------------------------------

    def _in_outage(self, time_s: float) -> bool:
        return any(start <= time_s <= end for start, end in self.outage_windows)

    def _after_outage(self, time_s: float, horizon_s: float) -> bool:
        return any(end < time_s <= end + horizon_s for _start, end in self.outage_windows)

    def _covered(self, frame: Dict[str, Any]) -> bool:
        return frame["n_tracks"] > 0 and frame.get("n_assigned_tracks", 0) > 0

    def result(self) -> Dict[str, Any]:
        if not self._frames:
            return {}
        out: Dict[str, Any] = {}

        # --- 乱序 ---
        measurement_times = [d["measurement_time_s"] for d in self._decisions]
        out_of_order = 0
        newest = None
        for value in measurement_times:
            if newest is not None and value < newest - 1e-12:
                out_of_order += 1
            newest = value if newest is None else max(newest, value)
        out["out_of_order_count"] = out_of_order
        out["out_of_order_rate"] = (
            out_of_order / len(measurement_times) if measurement_times else 0.0
        )
        fused_times = [d["fusion_time_s"] for d in self._decisions
                       if d.get("fusion_time_s") is not None]
        out["mean_time_in_system_s"] = _mean([
            (d["fusion_time_s"] - d["measurement_time_s"])
            for d in self._decisions if d.get("fusion_time_s") is not None
        ])
        out["n_fused_measurements"] = len(fused_times)

        # --- 时效拒绝率 ---
        stale = sum(f.get("n_stale_rejected", 0) for f in self._frames)
        accepted = sum(f.get("n_accepted_measurements", 0) for f in self._frames)
        out["stale_rejected"] = stale
        out["stale_rejection_rate"] = (
            stale / (stale + accepted) if (stale + accepted) else 0.0
        )

        # --- 突发长度 ---
        # 定义：**连续多少步**都发生了突发丢包。链路每步通常只发一条消息，
        # 因此"每步丢了几条"恒为 1，没有信息量；有信息量的是突发的**持续时间**。
        runs: List[int] = []
        current = 0
        for frame in self._frames:
            if int(frame.get("burst_dropped_this_step", 0)) > 0:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
        out["burst_loss_length_mean"] = _mean(runs)
        out["burst_loss_length_max"] = max(runs, default=0)
        out["n_bursts"] = len(runs)
        out["n_burst_steps"] = sum(runs)

        # --- 中断期间 / 恢复后的连续率 ---
        outage_frames = [f for f in self._frames if self._in_outage(f["time_s"])]
        after_frames = [f for f in self._frames if self._after_outage(f["time_s"], 6.0)]
        out["n_outage_frames"] = len(outage_frames)
        out["continuity_during_outage"] = (
            sum(1 for f in outage_frames if self._covered(f)) / len(outage_frames)
            if outage_frames else 0.0
        )
        out["continuity_after_recovery"] = (
            sum(1 for f in after_frames if self._covered(f)) / len(after_frames)
            if after_frames else 0.0
        )

        # --- 恢复时间：中断结束后第一次重新覆盖所花的时间 ---
        recovery_times: List[float] = []
        for _start, end in self.outage_windows:
            for frame in self._frames:
                if frame["time_s"] > end and self._covered(frame):
                    recovery_times.append(frame["time_s"] - end)
                    break
            else:
                recovery_times.append(float("nan"))
        out["recovery_time_s"] = (
            _mean([v for v in recovery_times if not math.isnan(v)])
            if any(not math.isnan(v) for v in recovery_times) else None
        )

        # --- 航迹年龄 / coasting 时长 ---
        ages = [a for f in self._frames for a in (f.get("track_ages") or [])]
        out["track_age_mean_s"] = _mean(ages)
        out["track_age_max_s"] = max(ages, default=0.0)
        coasting = [f.get("n_coasting", 0) for f in self._frames]
        out["coasting_frames_total"] = sum(coasting)
        out["coasting_duration_mean_s"] = _mean([c for c in coasting])

        # --- 旧包是否污染：迟到应用的测量数与其残差 ---
        late = [d for d in self._decisions
                if d.get("decision") == "released_late"]
        out["late_applied_measurements"] = len(late)
        out["late_applied_mean_hold_s"] = _mean([d.get("hold_s", 0.0) for d in late])
        out["max_reorder_lag_s"] = max(
            (d.get("hold_s", 0.0) for d in late), default=0.0)
        out.update(self.oosm_summary)
        return out


@dataclass
class BiasMetrics:
    """S8：多传感器系统偏差 / 不一致。"""

    biased_sensor_ids: Tuple[str, ...] = ()
    local_sensor_id: str = ""

    _frames: List[Dict[str, Any]] = field(default_factory=list)

    def add_frame(self, frame: Dict[str, Any]) -> None:
        self._frames.append(frame)

    def result(self) -> Dict[str, Any]:
        if not self._frames:
            return {}
        # --- 逐来源残差与创新一致性 ---
        per_sensor: Dict[str, Dict[str, List[float]]] = {}
        for frame in self._frames:
            for source in frame.get("track_sources") or []:
                bucket = per_sensor.setdefault(str(source.get("sensor_id")), {
                    "residual_m": [], "mahalanobis": [], "age": [],
                    "normalised": [],
                })
                residual = float(source.get("residual_m", 0.0))
                bucket["residual_m"].append(residual)
                bucket["mahalanobis"].append(
                    float(source.get("innovation_mahalanobis_sq", 0.0)))
                bucket["age"].append(float(source.get("age_at_use_s", 0.0)))
                # **归一化残差** = 残差 / 该传感器自己上报的 σ。
                # 这是唯一量纲正确的"残差是不是太大"判据：
                # 直接拿米和"马氏距离开方"比是错的（第一版就这么错过，
                # 健康分算出来 126 这种数字，完全没有物理含义）。
                sigma = float(source.get("reported_sigma_m", 0.0) or 0.0)
                if sigma > 0.0:
                    bucket["normalised"].append(residual / sigma)

        source_wise: Dict[str, Dict[str, float]] = {}
        for sensor_id, bucket in per_sensor.items():
            residuals = bucket["residual_m"]
            mahalanobis = bucket["mahalanobis"]
            normalised = bucket["normalised"]
            inconsistent = sum(1 for value in mahalanobis
                               if value > CHI2_3DOF_95)
            source_wise[sensor_id] = {
                "n_sources": float(len(residuals)),
                "residual_mean_m": _mean(residuals),
                "residual_max_m": max(residuals, default=0.0),
                "innovation_mean": _mean(mahalanobis),
                "innovation_max": max(mahalanobis, default=0.0),
                "innovation_inconsistent_rate": (
                    inconsistent / len(mahalanobis) if mahalanobis else 0.0),
                "normalised_residual_mean": _mean(normalised),
                "normalised_residual_max": max(normalised, default=0.0),
                "age_mean_s": _mean(bucket["age"]),
            }

        # 健康分 = 归一化残差均值（1 左右 = 与自称精度一致，越大越可疑）。
        # 只用**算法可见的量**（残差 + 上报协方差），不读真值——
        # 因此它可以是诊断分支，但不是"真值裁判"。
        health = {sensor_id: float(entry["normalised_residual_mean"])
                  for sensor_id, entry in source_wise.items()}
        # 判据：**明显差于同场景最好的那个传感器**才标记（相对 + 绝对下限）。
        # 单传感器场景没有比较对象，因此永不标记——这是有意的，
        # 避免把"只有一个传感器"误报成"这个传感器有问题"。
        suspicious: List[str] = []
        if len(health) >= 2:
            best = min(health.values())
            suspicious = [
                sensor_id for sensor_id, value in health.items()
                if value > best * HEALTH_RATIO_ALARM
                and value > HEALTH_ABSOLUTE_FLOOR
            ]
        scores = list(health.values())

        # --- 融合精度与协方差 ---
        position_errors = [e for f in self._frames
                           for e in (f.get("position_errors") or [])]
        covariances = [c for f in self._frames for c in (f.get("track_sigmas") or [])]
        velocity_errors = [e for f in self._frames
                           for e in (f.get("velocity_errors") or [])]

        # --- 远端贡献比例 ---
        sources = [s for f in self._frames for s in (f.get("track_sources") or [])]
        remote = [s for s in sources
                  if str(s.get("sensor_id")) != self.local_sensor_id]
        remote_ratio = (len(remote) / len(sources)) if sources else 0.0

        return {
            "biased_sensor_ids": list(self.biased_sensor_ids),
            "source_wise": source_wise,
            "sensor_health_score": health,
            "sensor_health_best": (min(scores) if scores else 0.0),
            "sensor_health_worst": (max(scores) if scores else 0.0),
            "suspicious_sensor_ids": suspicious,
            "health_alarm_ratio": HEALTH_RATIO_ALARM,
            "remote_contribution_ratio": remote_ratio,
            "position_rmse_m": (math.sqrt(_mean([e ** 2 for e in position_errors]))
                                if position_errors else 0.0),
            "position_error_max_m": max(position_errors, default=0.0),
            "velocity_rmse_mps": (math.sqrt(_mean([e ** 2 for e in velocity_errors]))
                                  if velocity_errors else 0.0),
            "track_sigma_mean_m": _mean(covariances),
            "track_sigma_max_m": max(covariances, default=0.0),
            "n_position_samples": len(position_errors),
        }


#: 报告里展示的系统级指标（key, 中文, 小数位）
SYSTEM_METRIC_ORDER: Tuple[Tuple[str, str, int], ...] = (
    # S5
    ("maneuvered_residual_pre_m", "机动前残差(m)", 1),
    ("maneuvered_residual_post_m", "机动后残差(m)", 1),
    ("maneuvered_residual_ratio", "残差放大倍数", 2),
    ("maneuvered_innovation_max", "创新峰值(马氏²)", 1),
    ("maneuvered_recovery_time_s", "恢复时间(s)", 2),
    ("reference_residual_ratio", "对照残差倍数", 2),
    # S6
    ("handover_continuity", "交接连续性", 4),
    ("handover_continuity_after_local", "本地丢失后连续性", 4),
    ("handover_delay_s", "交接延迟(s)", 2),
    ("remote_contribution_ratio", "远端贡献比例", 4),
    ("short_double_track_frames", "短时双轨帧数", 0),
    # S7
    ("out_of_order_rate", "乱序率", 4),
    ("stale_rejection_rate", "时效拒绝率", 4),
    ("burst_loss_length_mean", "突发丢包长度", 2),
    ("continuity_during_outage", "中断期连续性", 4),
    ("continuity_after_recovery", "恢复后连续性", 4),
    ("recovery_time_s", "恢复时间(s)", 2),
    ("late_applied_measurements", "迟到应用测量数", 0),
    ("mean_time_in_system_s", "平均在途时间(s)", 3),
    ("track_age_mean_s", "平均航迹年龄(s)", 2),
    ("coasting_frames_total", "coasting 帧数", 0),
    # S8
    ("position_rmse_m", "位置RMSE(m)", 2),
    ("track_sigma_mean_m", "航迹σ均值(m)", 2),
)
