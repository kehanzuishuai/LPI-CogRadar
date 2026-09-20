"""塔台的只读 global-track 观测（rm-obs-2.0）。

它与 resource_management.observation.CentralObservation（rm-obs-1.0）完全分离：
当前只为 Rule/诊断/评测提供快照，不改变现有调度器、PPO 或 resource-contract-v1。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from global_fusion.manager import GlobalTrack, GlobalTrackManager


GLOBAL_OBSERVATION_SCHEMA_VERSION = "rm-obs-2.0"


@dataclass(frozen=True)
class GlobalTrackObservation:
    """一个算法可见 global track 的塔台快照，不含真值或未来信息。"""

    global_track_id: str
    position_m: Tuple[float, float, float]
    velocity_mps: Tuple[float, float, float]
    covariance_position_m2: Tuple[float, float, float]
    updated_at_s: float
    information_age_s: float
    source_node_ids: Tuple[str, ...]
    coverage_state: str
    handover_recent: bool
    fusion_method: str
    fusion_weights: Dict[str, float]
    status: str

    @classmethod
    def from_track(cls, track: GlobalTrack, now_s: float,
                   manager: GlobalTrackManager) -> "GlobalTrackObservation":
        active_sources = tuple(sorted(track.active_source_nodes))
        handover_recent = bool(
            track.last_handover_at_s is not None
            and now_s - track.last_handover_at_s <= manager.config.handover_window_s
        )
        if handover_recent:
            coverage_state = "handover"
        elif len(active_sources) >= 2:
            coverage_state = "overlap"
        elif len(active_sources) == 1:
            coverage_state = "single_node"
        else:
            coverage_state = "coasting_no_active_source"
        return cls(
            global_track_id=track.global_track_id,
            position_m=tuple(track.position_m), velocity_mps=tuple(track.velocity_mps),
            covariance_position_m2=tuple(track.covariance_position_m2),
            updated_at_s=float(track.last_state_time_s),
            information_age_s=track.information_age_s(now_s),
            source_node_ids=active_sources,
            coverage_state=coverage_state,
            handover_recent=handover_recent,
            fusion_method=track.fusion_method,
            fusion_weights=dict(track.fusion_weights),
            status=track.status,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_track_id": self.global_track_id,
            "position_m": list(self.position_m),
            "velocity_mps": list(self.velocity_mps),
            "covariance_position_m2": list(self.covariance_position_m2),
            "updated_at_s": self.updated_at_s,
            "information_age_s": self.information_age_s,
            "source_node_ids": list(self.source_node_ids),
            "coverage_state": self.coverage_state,
            "handover_recent": self.handover_recent,
            "fusion_method": self.fusion_method,
            "fusion_weights": dict(self.fusion_weights),
            "status": self.status,
        }


@dataclass(frozen=True)
class GlobalObservation:
    """`rm-obs-2.0` 的只读塔台视图。"""

    schema_version: str
    observed_at_s: float
    global_track_mode: str
    tracks: Tuple[GlobalTrackObservation, ...] = ()
    missing: Tuple[str, ...] = ()
    provenance: str = "actual_arrived_global-track-v1_messages"

    @classmethod
    def from_manager(cls, manager: GlobalTrackManager, now_s: float,
                     global_track_mode: str = "track_fusion") -> "GlobalObservation":
        tracks = tuple(
            GlobalTrackObservation.from_track(manager.tracks[track_id], now_s, manager)
            for track_id in sorted(manager.tracks)
        )
        missing = (() if tracks else ("no_arrived_track_messages",))
        return cls(
            schema_version=GLOBAL_OBSERVATION_SCHEMA_VERSION,
            observed_at_s=float(now_s), global_track_mode=global_track_mode,
            tracks=tracks, missing=missing,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observed_at_s": self.observed_at_s,
            "global_track_mode": self.global_track_mode,
            "tracks": [track.to_dict() for track in self.tracks],
            "missing": list(self.missing),
            "provenance": self.provenance,
        }


def global_observation_diagnostics(observation: GlobalObservation) -> Dict[str, Any]:
    """面向 Rule/报告的只读摘要，不返回动作、不访问调度器内部状态。"""
    tracks = list(observation.tracks)
    return {
        "schema_version": observation.schema_version,
        "n_global_tracks": len(tracks),
        "n_overlap_tracks": sum(track.coverage_state == "overlap" for track in tracks),
        "n_handover_tracks": sum(track.handover_recent for track in tracks),
        "n_coasting_tracks": sum(track.status in ("coasting", "stale_coasting")
                                 for track in tracks),
        "mean_information_age_s": (sum(track.information_age_s for track in tracks) / len(tracks)
                                   if tracks else None),
        "source": "readonly_global_observation",
    }
