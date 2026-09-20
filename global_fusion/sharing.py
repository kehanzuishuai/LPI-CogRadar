"""预声明、可解释的 TrackMessage 通信基线。

这不是学习策略：所有阈值属于冻结配置，决策只读取 local track、历史已发送
摘要和全局管理器已有映射，不读取 truth、未来消息或离线标签。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


GLOBAL_SHARE_MODE_NO_SHARE = "no_share"
GLOBAL_SHARE_MODE_MEASUREMENT = "measurement_share"
GLOBAL_SHARE_MODE_TRACK = "track_share"
GLOBAL_SHARE_MODE_EVENT_TRACK = "event_triggered_track_share"
# v1 的兼容模式：现有 track_fusion 不改成 track-only，仍保留 measurement sharing。
GLOBAL_SHARE_MODE_MEASUREMENT_AND_TRACK = "measurement_and_track"
GLOBAL_SHARE_MODES = (
    GLOBAL_SHARE_MODE_NO_SHARE,
    GLOBAL_SHARE_MODE_MEASUREMENT,
    GLOBAL_SHARE_MODE_TRACK,
    GLOBAL_SHARE_MODE_EVENT_TRACK,
    GLOBAL_SHARE_MODE_MEASUREMENT_AND_TRACK,
)


@dataclass(frozen=True)
class EventTriggeredTrackShareConfig:
    """事件共享基线的冻结阈值（教学值，不基于 development 结果回调）。"""

    position_change_threshold_m: float = 250.0
    covariance_trace_threshold_m2: float = 100_000.0
    information_age_threshold_s: float = 3.0
    trigger_on_new_track: bool = True
    trigger_on_handover: bool = True
    trigger_on_refresh_request: bool = True

    def validate(self) -> None:
        if (self.position_change_threshold_m <= 0
                or self.covariance_trace_threshold_m2 <= 0
                or self.information_age_threshold_s <= 0):
            raise ValueError("事件触发 TrackMessage 阈值必须为正")


@dataclass(frozen=True)
class TrackShareDecision:
    node_id: str
    local_track_id: str
    triggered: bool
    reasons: Tuple[str, ...] = ()


@dataclass
class _LastSent:
    local_track_id: str
    position_m: Tuple[float, float, float]
    covariance_trace_m2: float
    information_age_s: float
    measurement_timestamp_s: float
    sent_at_s: float


class EventTriggeredTrackSharePolicy:
    """维护已发送摘要，给 RuntimeExecutor 提供可复现的事件门控。"""

    def __init__(self, config: Optional[EventTriggeredTrackShareConfig] = None) -> None:
        self.config = config or EventTriggeredTrackShareConfig()
        self.config.validate()
        # v1.1：发送状态属于 (node, local track)，不能由同一节点的第一条航迹
        # 覆盖其余航迹。否则多目标场景会反复把每条航迹误判为“new_track”。
        self._last_sent: Dict[Tuple[str, str], _LastSent] = {}
        self._refresh_requests: Set[Tuple[str, str]] = set()
        self.audit_log: List[Dict[str, Any]] = []

    @staticmethod
    def _distance(left: Tuple[float, float, float],
                  right: Tuple[float, float, float]) -> float:
        return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) ** 0.5

    def request_refresh(self, node_id: str, local_track_id: str) -> None:
        """远端可见的刷新请求入口；请求本身不产生发送或资源副作用。"""
        self._refresh_requests.add((str(node_id), str(local_track_id)))

    def evaluate(self, node_id: str, local_track: Any, now_s: float,
                 global_manager: Optional[Any] = None) -> TrackShareDecision:
        """只读判定：重复调用不会消耗请求或改写发送历史。"""
        node_id, local_id = str(node_id), str(local_track.track_id)
        reasons: List[str] = []
        previous = self._last_sent.get((node_id, local_id))
        position = (float(local_track.position.x), float(local_track.position.y),
                    float(local_track.position.z))
        covariance_trace = sum(
            float(value) ** 2 for value in (
                local_track.sigma_position.x, local_track.sigma_position.y,
                local_track.sigma_position.z,
            )
        )
        state_time = (float(local_track.last_measurement_time)
                      if local_track.last_measurement_time is not None else float(now_s))
        information_age = max(0.0, float(now_s) - state_time)
        if self.config.trigger_on_new_track and (
                previous is None or previous.local_track_id != local_id):
            reasons.append("new_track")
        if (previous is not None
                and self._distance(position, previous.position_m)
                >= self.config.position_change_threshold_m):
            reasons.append("state_change")
        # 阈值事件仅在本地有新测量后重新具备资格；否则会把“持续超过”
        # 错误实现成每个调度 tick 都上报。
        refreshed_since_send = (previous is not None
                                and state_time > previous.measurement_timestamp_s)
        if (covariance_trace >= self.config.covariance_trace_threshold_m2
                and (previous is None or refreshed_since_send)):
            reasons.append("covariance_threshold")
        if (information_age >= self.config.information_age_threshold_s
                and (previous is None or refreshed_since_send)):
            reasons.append("information_age_threshold")
        if self.config.trigger_on_refresh_request and (node_id, local_id) in self._refresh_requests:
            reasons.append("remote_refresh_request")
        if self.config.trigger_on_handover and global_manager is not None:
            global_id = global_manager.local_to_global.get((node_id, local_id))
            track = (global_manager.tracks.get(global_id) if global_id else None)
            # handover 是边沿事件而非持续电平；只有上次发送之后新发生的
            # handover 才再次触发，避免两个节点因 last reporter 交替而 ping-pong。
            if (track is not None
                    and track.last_reporting_source_node_id not in ("", node_id)
                    and track.last_handover_at_s is not None
                    and (previous is None or refreshed_since_send)
                    and (previous is None
                         or track.last_handover_at_s > previous.sent_at_s)):
                reasons.append("handover")
        return TrackShareDecision(node_id, local_id, bool(reasons), tuple(reasons))

    def record_sent(self, decision: TrackShareDecision, local_track: Any,
                    now_s: float) -> None:
        if not decision.triggered:
            return
        position = (float(local_track.position.x), float(local_track.position.y),
                    float(local_track.position.z))
        covariance_trace = sum(
            float(value) ** 2 for value in (
                local_track.sigma_position.x, local_track.sigma_position.y,
                local_track.sigma_position.z,
            )
        )
        state_time = (float(local_track.last_measurement_time)
                      if local_track.last_measurement_time is not None else float(now_s))
        self._last_sent[(decision.node_id, decision.local_track_id)] = _LastSent(
            decision.local_track_id, position, covariance_trace,
            max(0.0, float(now_s) - state_time),
            state_time,
            float(now_s),
        )
        self._refresh_requests.discard((decision.node_id, decision.local_track_id))
        self.audit_log.append({
            "time_s": float(now_s), "event": "event_triggered_track_share",
            "node_id": decision.node_id, "local_track_id": decision.local_track_id,
            "reasons": list(decision.reasons),
        })
