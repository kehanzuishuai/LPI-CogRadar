"""可解释的 Track-to-Track Fusion v1 全局航迹管理器。

本版仅使用距离门控 + 稳定映射 + 对角协方差 CI，不引入 JPDA/MHT。
所有输入均为 CommBus 已到达的 TrackMessage；此模块没有 Scene、Sensor 或
真值对象的导入路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from communication import CommBus, TrackMessage
from communication.message import MESSAGE_KIND_TRACK


GLOBAL_TRACK_ENDPOINT = "GLOBAL_TRACK_MANAGER"
GLOBAL_TRACK_MODE_OFF = "off"
GLOBAL_TRACK_MODE_TRACK_FUSION = "track_fusion"
GLOBAL_TRACK_MODES = (GLOBAL_TRACK_MODE_OFF, GLOBAL_TRACK_MODE_TRACK_FUSION)


@dataclass(frozen=True)
class GlobalTrackConfig:
    """冻结的 v1 管理参数；数值是可解释门限，不是学习调参。"""

    endpoint_id: str = GLOBAL_TRACK_ENDPOINT
    gate_distance_m: float = 1_500.0
    reconnect_gate_distance_m: float = 3_000.0
    process_noise_m2_per_s: float = 100.0
    max_coast_s: float = 30.0
    message_size_bytes: float = 128.0
    #: CI 的固定网格在协议中声明；不根据实验结果调权重。
    ci_weight_grid_steps: int = 20
    #: 某来源超过该年龄不再参与下一次 CI（global track 仍可 coast）。
    max_source_age_s: float = 10.0
    #: 仅为数值稳定使用的正方差下界，非精度声明。
    covariance_floor_m2: float = 1e-6
    handover_window_s: float = 5.0

    def validate(self) -> None:
        if self.gate_distance_m <= 0 or self.reconnect_gate_distance_m <= 0:
            raise ValueError("全局航迹门限必须为正")
        if self.reconnect_gate_distance_m < self.gate_distance_m:
            raise ValueError("重接入门限不能小于普通关联门限")
        if (self.process_noise_m2_per_s < 0 or self.max_coast_s <= 0
                or self.max_source_age_s <= 0 or self.covariance_floor_m2 <= 0
                or self.handover_window_s <= 0):
            raise ValueError("全局航迹 coast 参数非法")
        if self.message_size_bytes <= 0:
            raise ValueError("TrackMessage 字节数必须为正")
        if self.ci_weight_grid_steps < 2:
            raise ValueError("CI 权重网格至少需要两个区间")


@dataclass(frozen=True)
class _SourceEstimate:
    """一条节点 local track 的最新可见估计；没有真值/离线标签。"""

    source_node_id: str
    local_track_id: str
    position_m: Tuple[float, float, float]
    velocity_mps: Tuple[float, float, float]
    covariance_position_m2: Tuple[float, float, float]
    state_timestamp_s: float
    information_age_s: float
    provenance: Dict[str, Any]


@dataclass
class GlobalTrack:
    global_track_id: str
    position_m: Tuple[float, float, float]
    velocity_mps: Tuple[float, float, float]
    covariance_position_m2: Tuple[float, float, float]
    created_at_s: float
    last_state_time_s: float
    last_measurement_time_s: float
    status: str = "tentative"
    #: 确认态是生命周期事实；coasting 状态不能抹掉“是否曾被确认”。
    ever_confirmed: bool = False
    sources: List[Dict[str, Any]] = field(default_factory=list)
    #: 每节点当前可用的最新 local estimate；CI 不把同一节点历史重复当独立来源。
    source_estimates: Dict[str, _SourceEstimate] = field(default_factory=dict)
    fusion_method: str = "single_source"
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    #: v1.1 当前仍在 freshness 窗口内的来源；与 retained source 明确分离。
    active_source_nodes: List[str] = field(default_factory=list)
    fusion_numerical_guards: List[str] = field(default_factory=list)
    last_reporting_source_node_id: str = ""
    last_handover_at_s: Optional[float] = None
    handover_count: int = 0
    updates: int = 0
    coasts: int = 0

    def information_age_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - self.last_measurement_time_s)

    def to_dict(self, now_s: Optional[float] = None) -> Dict[str, Any]:
        return {
            "global_track_id": self.global_track_id,
            "position_m": list(self.position_m),
            "velocity_mps": list(self.velocity_mps),
            "covariance_position_m2": list(self.covariance_position_m2),
            "created_at_s": self.created_at_s,
            "last_state_time_s": self.last_state_time_s,
            "last_measurement_time_s": self.last_measurement_time_s,
            "information_age_s": (None if now_s is None
                                  else self.information_age_s(now_s)),
            "status": self.status,
            "ever_confirmed": self.ever_confirmed,
            "updates": self.updates,
            "coasts": self.coasts,
            "sources": [dict(item) for item in self.sources],
            "participating_source_nodes": sorted(self.source_estimates),
            "active_source_nodes": sorted(self.active_source_nodes),
            "fusion": {
                "method": self.fusion_method,
                "weights": dict(self.fusion_weights),
                "numerical_guards": list(self.fusion_numerical_guards),
                "note": ("Covariance Intersection is conservative under unknown "
                         "cross-node correlation; it is not claimed statistically optimal."),
            },
            "handover": {
                "last_reporting_source_node_id": self.last_reporting_source_node_id,
                "last_handover_at_s": self.last_handover_at_s,
                "handover_count": self.handover_count,
            },
        }


class GlobalTrackManager:
    """中央 global track 生命周期和审计证据的唯一拥有者。"""

    def __init__(self, config: Optional[GlobalTrackConfig] = None) -> None:
        self.config = config or GlobalTrackConfig()
        self.config.validate()
        self.tracks: Dict[str, GlobalTrack] = {}
        self.local_to_global: Dict[Tuple[str, str], str] = {}
        self._last_sequence_by_local: Dict[Tuple[str, str], int] = {}
        self._last_state_timestamp_by_local: Dict[Tuple[str, str], float] = {}
        self.audit_log: List[Dict[str, Any]] = []
        #: 无真值 global-track 快照，供外层离线评测连续性；不属于调度输入。
        self.history: List[Dict[str, Any]] = []
        self._transport_audit_message_ids: set = set()
        self._counter = 0
        self._last_predict_s: Optional[float] = None
        #: 真正退出 active 容器的航迹墓碑；只含算法可见生命周期证据。
        self.dropped_tracks: List[Dict[str, Any]] = []

    def _next_id(self) -> str:
        self._counter += 1
        return f"GLOBAL_TRACK_{self._counter}"

    @staticmethod
    def _distance(left: Tuple[float, float, float],
                  right: Tuple[float, float, float]) -> float:
        return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) ** 0.5

    def predict_to(self, now_s: float) -> None:
        """无消息时只 coast：位置外推、协方差增长，不凭空创建航迹。"""
        if self._last_predict_s is None:
            self._last_predict_s = float(now_s)
            return
        dt = max(0.0, float(now_s) - self._last_predict_s)
        if dt <= 0.0:
            return
        dropped_ids: List[str] = []
        for track in list(self.tracks.values()):
            track.position_m = tuple(
                position + velocity * dt
                for position, velocity in zip(track.position_m, track.velocity_mps)
            )
            track.covariance_position_m2 = tuple(
                value + self.config.process_noise_m2_per_s * dt
                for value in track.covariance_position_m2
            )
            track.last_state_time_s = float(now_s)
            track.coasts += 1
            track.active_source_nodes = sorted(
                source_node_id
                for source_node_id, estimate in track.source_estimates.items()
                if float(now_s) - estimate.state_timestamp_s
                <= self.config.max_source_age_s
            )
            if track.information_age_s(now_s) > self.config.max_coast_s:
                track.status = "dropped"
                dropped_ids.append(track.global_track_id)
            elif track.status != "dropped":
                track.status = ("stale_coasting" if not track.active_source_nodes
                                else "coasting")
        for global_track_id in dropped_ids:
            track = self.tracks.pop(global_track_id)
            tombstone = {
                "global_track_id": global_track_id,
                "created_at_s": track.created_at_s,
                "dropped_at_s": float(now_s),
                "reason": "max_coast_exceeded",
                "information_age_s": track.information_age_s(now_s),
                "retained_source_nodes": sorted(track.source_estimates),
                "last_measurement_time_s": track.last_measurement_time_s,
                "last_state_time_s": track.last_state_time_s,
                "updates": track.updates,
                "coasts": track.coasts,
                "fusion_method": track.fusion_method,
            }
            self.dropped_tracks.append(tombstone)
            self.audit_log.append({
                "time_s": float(now_s), "event": "global_track_dropped",
                **tombstone,
            })
            # 删除所有指向墓碑的活动映射，防止目标再次出现时错误复活旧 ID。
            for key, mapped_id in list(self.local_to_global.items()):
                if mapped_id == global_track_id:
                    del self.local_to_global[key]
        self._last_predict_s = float(now_s)

    def make_track_message(self, source_node_id: str, local_track: Any,
                           now_s: float, sequence_no: int,
                           src_sensor_id: str = "") -> TrackMessage:
        """把 local FusionCenter 的可见估计转换为无真值 TrackMessage。"""
        state_time = float(local_track.last_measurement_time
                           if local_track.last_measurement_time is not None
                           else now_s)
        provenance = {
            "source_platforms": sorted(set(getattr(local_track, "platforms", []))),
            "source_sensors": sorted({source.sensor_id
                                      for source in getattr(local_track, "sources", [])}),
            "local_updates": int(getattr(local_track, "local_updates", 0)),
            "remote_updates": int(getattr(local_track, "remote_updates", 0)),
            "hits": int(getattr(local_track, "hits", 0)),
        }
        return TrackMessage(
            # sequence_no 按 local track 独立递增；message_id 同时带来源/航迹，
            # 从而不同 outbox 的相同序号也保持全局唯一。
            message_id=(f"GT-{source_node_id}-{local_track.track_id}-"
                        f"{sequence_no:06d}"),
            source_node_id=source_node_id,
            local_track_id=str(local_track.track_id),
            sequence_no=int(sequence_no),
            state_timestamp_s=state_time,
            send_time_s=float(now_s),
            position_m=(float(local_track.position.x), float(local_track.position.y),
                        float(local_track.position.z)),
            velocity_mps=(float(local_track.velocity.x), float(local_track.velocity.y),
                          float(local_track.velocity.z)),
            covariance_position_m2=(
                float(local_track.sigma_position.x) ** 2,
                float(local_track.sigma_position.y) ** 2,
                float(local_track.sigma_position.z) ** 2,
            ),
            track_status=str(local_track.status),
            information_age_s=max(0.0, float(now_s) - state_time),
            source_provenance=provenance,
            size_bytes=self.config.message_size_bytes,
            dst_platform_id=self.config.endpoint_id,
            src_sensor_id=src_sensor_id,
        )

    def publish_local_track(self, source_node_id: str, local_track: Any,
                            bus: CommBus, now_s: float, sequence_no: int,
                            src_sensor_id: str = "") -> List[TrackMessage]:
        message = self.make_track_message(source_node_id, local_track, now_s,
                                          sequence_no, src_sensor_id)
        sent = bus.publish_track(message, now_s)
        self.audit_log.append({
            "time_s": float(now_s), "event": "track_message_sent",
            "message_id": message.message_id, "source_node_id": source_node_id,
            "local_track_id": message.local_track_id,
            "bytes": message.size_bytes, "sent": bool(sent),
            "reason": "comm_link_available" if sent else "no_route_or_sharing_disabled",
        })
        return sent

    def ingest_arrived(self, bus: CommBus, now_s: float) -> int:
        """只消费实际抵达全局端点的 TrackMessage，绝不读未来/在途内容。"""
        # CommLink 在到达前因丢包/过期/队列满丢弃的 TrackMessage 永远不应
        # 进入关联器，但必须留下可审计的拒绝原因。
        for message in bus.log:
            if (getattr(message, "kind", "") == MESSAGE_KIND_TRACK
                    and getattr(message, "dst_platform_id", "") == self.config.endpoint_id
                    and message.dropped
                    and message.msg_id not in self._transport_audit_message_ids):
                self._transport_audit_message_ids.add(message.msg_id)
                self.audit_log.append({
                    "time_s": float(now_s), "event": "transport_rejected",
                    "message_id": message.msg_id,
                    "source_node_id": message.source_node_id,
                    "local_track_id": message.local_track_id,
                    "decision": "rejected",
                    "reason": message.drop_reason or "transport_drop",
                })
        messages = bus.consume_kind(self.config.endpoint_id, now_s, MESSAGE_KIND_TRACK)
        bus.assert_only_arrived(messages, now_s)
        # CommBus 的 log 是发送顺序，不是到达顺序。中央必须先按真实到达时刻
        # 消费；同一节点、同一状态时刻的并发 local tracks 再按可观测更新证据
        # 排序，使成熟航迹先占据来源槽，竞争航迹随后被隔离为 tentative。
        messages.sort(key=lambda item: (
            float(item.arrived_at if item.arrived_at is not None else now_s),
            str(item.source_node_id), float(item.state_timestamp_s),
            -int(item.source_provenance.get("local_updates", 0)),
            -int(item.source_provenance.get("hits", 0)),
            str(item.local_track_id), str(item.message_id),
        ))
        for message in messages:
            self._associate_and_update(message, now_s)
        return len(messages)

    def record_snapshot(self, now_s: float) -> None:
        """记录当前全局估计；调用方不能通过它反向写入 manager。"""
        self.history.append({
            "time_s": float(now_s),
            "tracks": [self.tracks[track_id].to_dict(now_s)
                       for track_id in sorted(self.tracks)],
        })

    def _associate_and_update(self, message: TrackMessage, now_s: float) -> str:
        key = (message.source_node_id, message.local_track_id)
        previous_sequence = self._last_sequence_by_local.get(key)
        if previous_sequence is not None and message.sequence_no <= previous_sequence:
            self.audit_log.append({
                "time_s": float(now_s), "event": "association",
                "message_id": message.message_id,
                "source_node_id": message.source_node_id,
                "local_track_id": message.local_track_id,
                "global_track_id": self.local_to_global.get(key, ""),
                "decision": "rejected",
                "reason": "duplicate_or_out_of_order_sequence",
                "distance_m": None,
                "candidate_global_ids": [], "rejected_candidates": [],
            })
            return self.local_to_global.get(key, "")
        previous_state_time = self._last_state_timestamp_by_local.get(key)
        if (previous_state_time is not None
                and float(message.state_timestamp_s) <= previous_state_time + 1e-12):
            self.audit_log.append({
                "time_s": float(now_s), "event": "association",
                "message_id": message.message_id,
                "source_node_id": message.source_node_id,
                "local_track_id": message.local_track_id,
                "global_track_id": self.local_to_global.get(key, ""),
                "decision": "rejected",
                "reason": "non_monotonic_state_timestamp",
                "distance_m": None,
                "candidate_global_ids": [], "rejected_candidates": [],
            })
            return self.local_to_global.get(key, "")
        mapped_id = self.local_to_global.get(key)
        selected: Optional[GlobalTrack] = None
        reason = ""
        distance: Optional[float] = None
        candidate_distances: List[Tuple[float, str]] = []
        rejected_candidates: List[Dict[str, Any]] = []
        if mapped_id and mapped_id in self.tracks:
            candidate = self.tracks[mapped_id]
            distance = self._distance(candidate.position_m, message.position_m)
            if distance <= self.config.reconnect_gate_distance_m:
                selected, reason = candidate, "mapping_reused"
            else:
                reason = "mapped_track_rejected_reconnect_gate"
                rejected_candidates.append({
                    "global_track_id": mapped_id, "distance_m": distance,
                    "reason": "reconnect_gate_exceeded",
                })
        if selected is None:
            candidates = []
            for track in self.tracks.values():
                candidate_distance = self._distance(track.position_m, message.position_m)
                candidate_distances.append((candidate_distance, track.global_track_id))
                # v1.1：同一来源节点的 local track 可能在失联/漏测后以新 local ID
                # 重建。此时没有 local_to_global 键可复用，但 retained source 足以
                # 证明这是“重接候选”；使用协议中原已冻结的 reconnect gate，
                # 不读取 truth，也不改变普通跨节点关联门限。
                prior_source = track.source_estimates.get(message.source_node_id)
                reconnect_candidate = prior_source is not None
                # 同一节点在同一状态时刻同时上报两个不同 local ID，表示并发竞争
                # 航迹而非掉线后的时序重建。禁止第二条覆盖已有真实来源；它可按
                # 普通门控另建 tentative global track。判据只用报文时戳/来源。
                simultaneous_competitor = (
                    prior_source is not None
                    and prior_source.local_track_id != message.local_track_id
                    and abs(prior_source.state_timestamp_s
                            - float(message.state_timestamp_s)) <= 1e-12
                )
                if simultaneous_competitor:
                    rejected_candidates.append({
                        "global_track_id": track.global_track_id,
                        "distance_m": candidate_distance,
                        "reason": "same_source_simultaneous_competing_local_track",
                    })
                    continue
                gate_m = (self.config.reconnect_gate_distance_m
                          if reconnect_candidate else self.config.gate_distance_m)
                if candidate_distance <= gate_m:
                    candidates.append((candidate_distance, track))
                else:
                    rejected_candidates.append({
                        "global_track_id": track.global_track_id,
                        "distance_m": candidate_distance,
                        "reason": ("reconnect_gate_exceeded"
                                   if reconnect_candidate
                                   else "spatial_gate_exceeded"),
                    })
            if candidates:
                distance, selected = min(candidates, key=lambda item: (item[0], item[1].global_track_id))
                reason = ("source_reconnect_associated"
                          if message.source_node_id in selected.source_estimates
                          else "spatial_gate_associated")
            else:
                selected = GlobalTrack(
                    global_track_id=self._next_id(), position_m=message.position_m,
                    velocity_mps=message.velocity_mps,
                    covariance_position_m2=message.covariance_position_m2,
                    created_at_s=float(now_s), last_state_time_s=float(now_s),
                    last_measurement_time_s=float(message.state_timestamp_s),
                    status="tentative", updates=0,
                )
                self.tracks[selected.global_track_id] = selected
                reason = ("new_track_after_gate_reject" if self.tracks and mapped_id
                          else "new_track")
        fused = self._fuse(selected, message, now_s)
        self.local_to_global[key] = selected.global_track_id
        self._last_sequence_by_local[key] = message.sequence_no
        self._last_state_timestamp_by_local[key] = float(message.state_timestamp_s)
        # 关联成功后的每一次 _fuse 都留下只含算法可见字段的诊断证据。
        # 这不是第二条执行路径：只观察刚刚已经发生的 CI 结果。
        if fused:
            projection_delta_s = max(
                0.0, float(now_s) - float(message.state_timestamp_s))
            effective_source = selected.source_estimates[message.source_node_id]
            effective_velocity_mps = [
                float(value) for value in effective_source.velocity_mps
            ]
            projected_message_position_m = [
                float(position) + projection_delta_s * float(velocity)
                for position, velocity in zip(
                    message.position_m, effective_velocity_mps)
            ]
            self.audit_log.append({
                "time_s": float(now_s), "event": "ci_fused",
                "message_id": message.message_id,
                "source_node_id": message.source_node_id,
                "local_track_id": message.local_track_id,
                "global_track_id": selected.global_track_id,
                "reason": "ci_fusion_complete",
                "fusion_method": selected.fusion_method,
                "participating_source_nodes": sorted(selected.fusion_weights),
                "active_source_nodes": list(selected.active_source_nodes),
                "fusion_weights": dict(selected.fusion_weights),
                "numerical_guards": list(selected.fusion_numerical_guards),
                # v1.3 只读时序证据：证明迟到/异步状态先传播到当前融合时刻，
                # 再进入既有 CI；不改变 _project_source_to 或 CI 数学。
                "message_state_timestamp_s": float(message.state_timestamp_s),
                "message_arrived_at_s": (None if message.arrived_at is None
                                          else float(message.arrived_at)),
                "fused_at_s": float(now_s),
                "projection_delta_s": projection_delta_s,
                "message_position_m": [float(value)
                                       for value in message.position_m],
                "message_velocity_mps": [float(value)
                                         for value in message.velocity_mps],
                "effective_projection_velocity_mps": effective_velocity_mps,
                "projected_message_position_m": projected_message_position_m,
                "active_source_state_timestamps_s": {
                    source_node_id: float(
                        selected.source_estimates[source_node_id].state_timestamp_s)
                    for source_node_id in selected.active_source_nodes
                },
            })
        else:
            self.audit_log.append({
                "time_s": float(now_s), "event": "source_retained_not_active",
                "message_id": message.message_id,
                "source_node_id": message.source_node_id,
                "local_track_id": message.local_track_id,
                "global_track_id": selected.global_track_id,
                "reason": "state_timestamp_exceeded_source_freshness_window",
                "active_source_nodes": list(selected.active_source_nodes),
            })
        self.audit_log.append({
            "time_s": float(now_s), "event": "association",
            "message_id": message.message_id, "source_node_id": message.source_node_id,
            "local_track_id": message.local_track_id,
            "global_track_id": selected.global_track_id,
            "decision": "accepted", "reason": reason,
            "distance_m": distance,
            "candidate_global_ids": [track_id for _value, track_id
                                     in sorted(candidate_distances)],
            "rejected_candidates": rejected_candidates,
        })
        return selected.global_track_id

    def _sanitize_covariance(self, covariance: Tuple[float, float, float]
                             ) -> Tuple[Tuple[float, float, float], List[str]]:
        """确保对角协方差严格为正，避免 CI 的求逆产生虚假置信。"""
        guards: List[str] = []
        stable = []
        for index, value in enumerate(covariance):
            value = float(value)
            if value < 0.0:
                raise ValueError(f"协方差第 {index} 维为负，拒绝融合")
            if value < self.config.covariance_floor_m2:
                guards.append(f"covariance_floor_dim_{index}")
                value = self.config.covariance_floor_m2
            stable.append(value)
        return (tuple(stable), guards)

    def _ci_pair(
        self,
        left_position: Tuple[float, float, float],
        left_velocity: Tuple[float, float, float],
        left_covariance: Tuple[float, float, float],
        right_position: Tuple[float, float, float],
        right_velocity: Tuple[float, float, float],
        right_covariance: Tuple[float, float, float],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float],
               Tuple[float, float, float], float, List[str]]:
        """二元对角 CI：最小 trace 的固定网格搜索，且绝不假设独立性。

        ``omega`` 接近 1 表示偏向 left。CI 协方差是信息矩阵的凸组合，
        不会采用独立高斯融合中会导致过度自信的 ``P1^-1 + P2^-1`` 形式。
        """
        left_covariance, left_guards = self._sanitize_covariance(left_covariance)
        right_covariance, right_guards = self._sanitize_covariance(right_covariance)
        guards = left_guards + right_guards
        choices: List[Tuple[float, float, float, Tuple[float, float, float]]] = []
        for index in range(self.config.ci_weight_grid_steps + 1):
            omega = index / float(self.config.ci_weight_grid_steps)
            covariance = tuple(
                1.0 / (omega / left + (1.0 - omega) / right)
                for left, right in zip(left_covariance, right_covariance)
            )
            # 次关键字靠近 0.5，让完全相等来源的行为对称、可复现。
            choices.append((sum(covariance), abs(omega - 0.5), omega, covariance))
        _trace, _symmetry, selected, covariance = min(
            choices, key=lambda item: (item[0], item[1])
        )
        position = tuple(
            covariance_value * (
                selected * left_position_value / left_variance
                + (1.0 - selected) * right_position_value / right_variance
            )
            for covariance_value, left_position_value, right_position_value,
            left_variance, right_variance in zip(
                covariance, left_position, right_position,
                left_covariance, right_covariance
            )
        )
        # 本项目的 local track 不报告速度协方差；位置 CI 权重因此只作为
        # 对称、可解释的状态权重，不把速度当作独立观测去虚构协方差收缩。
        velocity = tuple(
            selected * left_value + (1.0 - selected) * right_value
            for left_value, right_value in zip(left_velocity, right_velocity)
        )
        return position, velocity, covariance, selected, guards

    def _project_source_to(
        self, estimate: _SourceEstimate, now_s: float,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float],
               Tuple[float, float, float], List[str]]:
        """将已到达 local estimate 推演到同一融合时刻。

        TrackMessage 的状态时间可能早于实际抵达时间。若把它直接同 ``now_s``
        的 global state 混合，会把过期位置标成当前状态。这里仅用报文内的速度
        和既有 coast 过程噪声传播；没有读取任何未来量测或真值。
        """
        delta_s = max(0.0, float(now_s) - estimate.state_timestamp_s)
        covariance, guards = self._sanitize_covariance(
            estimate.covariance_position_m2
        )
        return (
            tuple(position + delta_s * velocity
                  for position, velocity in zip(
                      estimate.position_m, estimate.velocity_mps)),
            estimate.velocity_mps,
            tuple(value + self.config.process_noise_m2_per_s * delta_s
                  for value in covariance),
            guards,
        )

    def _fuse(self, track: GlobalTrack, message: TrackMessage, now_s: float) -> bool:
        covariance, guards = self._sanitize_covariance(message.covariance_position_m2)
        previous_source = track.source_estimates.get(message.source_node_id)
        velocity = tuple(float(value) for value in message.velocity_mps)
        provenance = dict(message.source_provenance)
        # local tracker 在长间隔/高速交接下可能重建 local ID；新生航迹的速度
        # 尚为 0，但同一节点的上一条已到达 estimate 提供了不依赖 truth 的时间
        # 连续性。用两次状态差恢复重接速度，避免把旧时刻位置原样带到 arrival
        # 时刻；连续 local ID 仍完全使用原 local tracker 速度。
        if (previous_source is not None
                and previous_source.local_track_id != message.local_track_id):
            delta_state_s = (float(message.state_timestamp_s)
                             - previous_source.state_timestamp_s)
            if delta_state_s > 0.0:
                velocity = tuple(
                    (float(current) - float(previous)) / delta_state_s
                    for current, previous in zip(
                        message.position_m, previous_source.position_m)
                )
                provenance["reconnect_velocity_from_arrived_state_delta"] = True
                guards.append("reconnect_velocity_reconstructed")
        source = _SourceEstimate(
            source_node_id=message.source_node_id,
            local_track_id=message.local_track_id,
            position_m=tuple(float(value) for value in message.position_m),
            velocity_mps=velocity,
            covariance_position_m2=covariance,
            state_timestamp_s=float(message.state_timestamp_s),
            information_age_s=float(message.information_age_s),
            provenance=provenance,
        )
        # 一个节点的相邻 local ID 不能被误当成独立传感器，始终只保留该节点最新估计。
        track.source_estimates[message.source_node_id] = source
        eligible = [estimate for estimate in track.source_estimates.values()
                    if now_s - estimate.state_timestamp_s
                    <= self.config.max_source_age_s]
        eligible.sort(key=lambda item: (item.source_node_id, item.local_track_id))
        track.active_source_nodes = [estimate.source_node_id for estimate in eligible]
        if not eligible:
            # 抵达时间不能把旧状态伪装成新鲜状态。映射、序号与 retained source
            # 仍保留用于重接审计，但本次不覆盖已 coast 的 global state。
            track.sources.append({
                "source_node_id": message.source_node_id,
                "local_track_id": message.local_track_id,
                "message_id": message.message_id,
                "state_timestamp_s": message.state_timestamp_s,
                "information_age_s": message.information_age_s,
                "provenance": dict(provenance),
                "active_at_fusion": False,
            })
            return False
        first = eligible[0]
        position, velocity, fused_covariance, first_guards = self._project_source_to(
            first, now_s
        )
        source_weights = {first.source_node_id: 1.0}
        ci_guards = list(guards) + first_guards
        for estimate in eligible[1:]:
            estimate_position, estimate_velocity, estimate_covariance, estimate_guards = (
                self._project_source_to(estimate, now_s)
            )
            position, velocity, fused_covariance, omega, pair_guards = self._ci_pair(
                position, velocity, fused_covariance,
                estimate_position, estimate_velocity, estimate_covariance,
            )
            source_weights = {key: value * omega
                              for key, value in source_weights.items()}
            source_weights[estimate.source_node_id] = 1.0 - omega
            ci_guards.extend(estimate_guards + pair_guards)
        track.position_m = position
        track.velocity_mps = velocity
        track.covariance_position_m2 = fused_covariance
        track.last_state_time_s = float(now_s)
        # age 必须反映仍参与融合来源中最新的真实状态，而非恰好最后抵达的旧消息；
        # 消息抵达顺序不能把 global 信息年龄伪装成“更新”。
        track.last_measurement_time_s = max(
            estimate.state_timestamp_s for estimate in eligible
        )
        track.status = "confirmed" if track.updates >= 1 else "tentative"
        if track.status == "confirmed":
            track.ever_confirmed = True
        track.fusion_method = ("single_source" if len(eligible) == 1
                               else "covariance_intersection_conservative")
        track.fusion_weights = {key: round(value, 9)
                                for key, value in sorted(source_weights.items())}
        track.fusion_numerical_guards = sorted(set(ci_guards))
        if (track.last_reporting_source_node_id
                and track.last_reporting_source_node_id != message.source_node_id):
            track.last_handover_at_s = float(now_s)
            track.handover_count += 1
        track.last_reporting_source_node_id = message.source_node_id
        track.updates += 1
        track.sources.append({
            "source_node_id": message.source_node_id,
            "local_track_id": message.local_track_id,
            "message_id": message.message_id,
            "state_timestamp_s": message.state_timestamp_s,
            "information_age_s": message.information_age_s,
            "provenance": dict(provenance),
            "active_at_fusion": True,
        })
        return True

    def report(self, now_s: float) -> Dict[str, Any]:
        return {
            "schema_version": "global-track-v1",
            "endpoint_id": self.config.endpoint_id,
            "n_global_tracks": len(self.tracks),
            "tracks": [self.tracks[key].to_dict(now_s) for key in sorted(self.tracks)],
            "local_to_global": [
                {"source_node_id": source, "local_track_id": local,
                 "global_track_id": global_id}
                for (source, local), global_id in sorted(self.local_to_global.items())
            ],
            "dropped_tracks": [dict(item) for item in self.dropped_tracks],
            "audit": list(self.audit_log),
            "history": list(self.history),
        }
