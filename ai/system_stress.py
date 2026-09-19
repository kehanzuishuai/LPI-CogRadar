"""系统级压力证据构造（v4.5 P2 第二阶段）。

把"机动失配 / 航迹交接 / 通信时序 / 传感器健康"这四件事变成
**结构化、可校验**的证据，供 AI 诊断使用。

⚠️ 三条纪律
-----------
1. **只用算法此刻拿得到的信息**：航迹来源、残差、创新、门限拒绝、
   链路统计、跨传感器包线——全部来自跟踪器与通信总线；
2. **不含真值**：没有 `truth_id`、没有真实目标位置、没有真实速度；
3. **不做判断之外的事**：本模块只**描述**（"残差放大 1.85 倍"），
   不控制任何东西，也不自动剔除传感器。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

#: 机动失配判据：残差放大倍数超过它即认为模型失配（与压力测试阈值一致）
MANEUVER_RESIDUAL_ALARM = 1.5
#: 新息一致性判据：3 自由度卡方 95% 分位
CHI2_3DOF_95 = 7.815
#: 传感器健康分绝对下限（归一化残差 = 残差 / 该传感器自称的 σ）
HEALTH_ABSOLUTE_FLOOR = 2.0


def _source_dict(source: Any) -> Dict[str, Any]:
    return source.to_dict() if hasattr(source, "to_dict") else dict(source)


def maneuver_state_from_tracks(
    tracks: Sequence[Any], now: float,
    residual_history: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """从航迹的逐来源残差判断"运动模型是否失配"。

    `residual_history` 可传入 `{track_id: [历史残差...]}`；
    缺省时只用当前来源残差，仍然能给出"当前残差 / 上报 σ"的归一化量。
    """
    history = residual_history or {}
    entries: List[Dict[str, Any]] = []
    for track in tracks:
        sources = list(getattr(track, "sources", []) or [])
        if not sources:
            continue
        latest = max(sources, key=lambda s: s.measurement_time_s)
        residual = float(getattr(latest, "residual_m", 0.0) or 0.0)
        sigma = float(getattr(latest, "reported_sigma_m", 0.0) or 0.0)
        innovation = float(
            getattr(latest, "innovation_mahalanobis_sq", 0.0) or 0.0
        )
        past = history.get(track.track_id) or []
        baseline = (sum(past) / len(past)) if past else 0.0
        ratio = (residual / baseline) if baseline > 1e-9 else None
        entries.append({
            "track_id": track.track_id,
            "status": track.status,
            "residual_m": round(residual, 3),
            "reported_sigma_m": round(sigma, 3),
            # 归一化残差：唯一量纲正确的"偏离是否过大"判据
            "normalised_residual": round(residual / sigma, 4) if sigma > 0 else None,
            "innovation_mahalanobis_sq": round(innovation, 4),
            "innovation_inconsistent": innovation > CHI2_3DOF_95,
            "residual_ratio_vs_history": (round(ratio, 4) if ratio else None),
            "misses": track.misses,
        })
    mismatch = [
        entry for entry in entries
        if entry["innovation_inconsistent"]
        or (entry["residual_ratio_vs_history"] is not None
            and entry["residual_ratio_vs_history"] > MANEUVER_RESIDUAL_ALARM)
        or (entry["normalised_residual"] is not None
            and entry["normalised_residual"] > 3.0)
    ]
    return {
        "n_tracks": len(entries),
        "tracks": entries,
        "n_mismatch_suspected": len(mismatch),
        "mismatch_track_ids": [entry["track_id"] for entry in mismatch],
        "note": ("运动模型失配是**推断**：常速度滤波器在目标机动后会出现"
                 "残差与创新同时放大的特征；本证据只给出现象，"
                 "不断言目标一定在机动。"),
    }


def handover_state_from_sources(
    tracks: Sequence[Any], sensor_windows: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """从航迹来源判断"是否正在跨传感器交接"。

    `sensor_windows` 可选，给 `{sensor_id: {"first_source_s":.., "last_source_s":..}}`
    之类的跨传感器时间线（由评测侧或场景提供）；缺省时只做来源计数。
    """
    per_sensor: Dict[str, Dict[str, Any]] = {}
    transitions: List[Dict[str, Any]] = []
    for track in tracks:
        sources = sorted(
            (s for s in (getattr(track, "sources", []) or [])),
            key=lambda s: s.measurement_time_s,
        )
        previous: Optional[str] = None
        for source in sources:
            sensor_id = str(getattr(source, "sensor_id", ""))
            bucket = per_sensor.setdefault(sensor_id, {
                "n_sources": 0, "first_s": None, "last_s": None,
                "track_ids": set(),
            })
            bucket["n_sources"] += 1
            bucket["track_ids"].add(track.track_id)
            time_s = float(getattr(source, "measurement_time_s", 0.0) or 0.0)
            bucket["first_s"] = (time_s if bucket["first_s"] is None
                                 else min(bucket["first_s"], time_s))
            bucket["last_s"] = (time_s if bucket["last_s"] is None
                                else max(bucket["last_s"], time_s))
            if previous is not None and sensor_id != previous:
                transitions.append({
                    "track_id": track.track_id,
                    "from_sensor": previous,
                    "to_sensor": sensor_id,
                    "at_s": round(time_s, 3),
                })
            previous = sensor_id
    for bucket in per_sensor.values():
        bucket["track_ids"] = sorted(bucket["track_ids"])
    return {
        "n_sensors_contributing": len(per_sensor),
        "per_sensor": per_sensor,
        "n_handover_transitions": len(transitions),
        "transitions": transitions[:16],
        "handover_in_progress": len(per_sensor) > 1 and len(transitions) > 0,
        "sensor_windows": sensor_windows or {},
        "note": ("交接是**结构性事实**：同一条航迹的来源从传感器 A 变成了 B。"
                 "这不等价于「接力成功」——是否成功要看本地丢失后航迹是否还在"
                 "（见多目标压力测试的 handover continuity）。"),
    }


def timing_state_from_bus(bus: Any, tracker_stats: Optional[Dict[str, int]] = None,
                          oosm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """从通信总线与 OOSM 控制器取通信时序证据。"""
    stats = bus.statistics() if hasattr(bus, "statistics") else {}
    reasons = dict(stats.get("drop_reasons") or {})
    tracker_stats = tracker_stats or {}
    state: Dict[str, Any] = {
        "policy": stats.get("policy", ""),
        "n_links": stats.get("n_links", 0),
        "n_messages": stats.get("n_messages", 0),
        "delivery_rate": round(float(stats.get("delivery_rate", 0.0) or 0.0), 6),
        "latency_mean_s": round(float(stats.get("latency_mean_s", 0.0) or 0.0), 6),
        "out_of_order_rate": round(
            float(stats.get("out_of_order_rate", 0.0) or 0.0), 6),
        "max_reorder_lag_s": round(
            float(stats.get("max_reorder_lag_s", 0.0) or 0.0), 6),
        "drop_reasons": reasons,
        "n_outage_dropped": int(reasons.get("link_outage", 0)),
        "n_burst_dropped": int(reasons.get("burst_loss", 0)),
        "n_congestion_dropped": int(reasons.get("congestion", 0)),
        "n_stale_rejected_by_tracker": int(tracker_stats.get("stale_rejected", 0)),
        "oosm": oosm or {},
    }
    state["link_outage_observed"] = state["n_outage_dropped"] > 0
    state["burst_loss_observed"] = state["n_burst_dropped"] > 0
    state["congestion_observed"] = state["n_congestion_dropped"] > 0
    state["out_of_order_observed"] = state["out_of_order_rate"] > 0.0
    return state


def sensor_health_state_from_tracks(
    tracks: Sequence[Any],
    local_sensor_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """逐传感器的残差一致性（诊断专用，**不自动剔除**任何传感器）。"""
    per_sensor: Dict[str, Dict[str, Any]] = {}
    for track in tracks:
        for source in (getattr(track, "sources", []) or []):
            sensor_id = str(getattr(source, "sensor_id", ""))
            bucket = per_sensor.setdefault(sensor_id, {
                "n_sources": 0, "residual_sum": 0.0, "normalised_sum": 0.0,
                "normalised_count": 0, "inconsistent": 0, "max_innovation": 0.0,
                "is_local": sensor_id in set(local_sensor_ids or []),
            })
            residual = float(getattr(source, "residual_m", 0.0) or 0.0)
            sigma = float(getattr(source, "reported_sigma_m", 0.0) or 0.0)
            innovation = float(
                getattr(source, "innovation_mahalanobis_sq", 0.0) or 0.0)
            bucket["n_sources"] += 1
            bucket["residual_sum"] += residual
            if sigma > 0.0:
                bucket["normalised_sum"] += residual / sigma
                bucket["normalised_count"] += 1
            if innovation > CHI2_3DOF_95:
                bucket["inconsistent"] += 1
            bucket["max_innovation"] = max(bucket["max_innovation"], innovation)

    health: Dict[str, float] = {}
    entries: Dict[str, Dict[str, Any]] = {}
    for sensor_id, bucket in per_sensor.items():
        count = bucket["normalised_count"]
        score = (bucket["normalised_sum"] / count) if count else 0.0
        health[sensor_id] = round(score, 4)
        entries[sensor_id] = {
            "n_sources": bucket["n_sources"],
            "residual_mean_m": round(bucket["residual_sum"] / bucket["n_sources"], 3)
            if bucket["n_sources"] else 0.0,
            "normalised_residual_mean": round(score, 4),
            "innovation_inconsistent_count": bucket["inconsistent"],
            "innovation_max": round(bucket["max_innovation"], 4),
            "is_local": bucket["is_local"],
        }
    # **相对**判据：只有两部以上传感器时才有比较对象（与压力测试口径一致）
    suspicious: List[str] = []
    if len(health) >= 2:
        best = min(health.values())
        suspicious = [sensor_id for sensor_id, value in health.items()
                      if value > best * 1.5 and value > HEALTH_ABSOLUTE_FLOOR]
    return {
        "health_score": health,
        "per_sensor": entries,
        "suspicious_sensor_ids": suspicious,
        "note": ("健康分 = 残差 / 该传感器**自称**的 σ，是**相对**判据："
                 "它只能指出「明显比同伴差的那部」，**不能**断定偏差就在它身上；"
                 "本诊断分支**不自动剔除**任何传感器（否则会掩盖基线问题）。"),
    }


def system_stress_state(
    tracks: Sequence[Any],
    bus: Any = None,
    tracker_stats: Optional[Dict[str, int]] = None,
    oosm: Optional[Dict[str, Any]] = None,
    local_sensor_ids: Optional[Sequence[str]] = None,
    residual_history: Optional[Dict[str, List[float]]] = None,
    sensor_windows: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """汇总成 `StateSnapshot.system_stress_state`（缺数据时为 None）。"""
    if not tracks and bus is None:
        return None
    payload: Dict[str, Any] = {
        "maneuver": maneuver_state_from_tracks(tracks, 0.0, residual_history),
        "handover": handover_state_from_sources(tracks, sensor_windows),
        "sensor_health": sensor_health_state_from_tracks(tracks, local_sensor_ids),
    }
    if bus is not None:
        payload["timing"] = timing_state_from_bus(bus, tracker_stats, oosm)
    return payload
