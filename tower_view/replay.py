"""从冻结 Global Track 验收链路生成确定性的只读 Tower View 回放。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from global_fusion import GLOBAL_SHARE_MODE_TRACK, GLOBAL_TRACK_MODE_TRACK_FUSION
from resource_management.closed_loop import NODE_LAYOUT, _build_feedback_world
from resource_management.scheduling import SchedulerPolicy, build_scheduler
from tools.run_global_track_acceptance import (
    DEFAULT_SEED,
    SCENARIOS,
    _apply_post_tick_events,
    _apply_pre_tick_events,
)


TOWER_VIEW_SCHEMA_VERSION = "tower-view-v1"
CANONICAL_FLOAT_DECIMALS = 6
TOWER_VIEW_SCENARIOS: Tuple[str, ...] = (
    "dual_radar_single_target",
    "dual_radar_two_targets",
    "handover_disconnect_reconnect",
    "two_targets_crossing",
)

# 这是冻结 v1.3 外层评测的聚合注释，不是运行时输入，也不含目标 ID/位置。
# 它确保 UI 不会把交叉场景的已知负结果隐藏成“看起来正常”。
_FROZEN_EVALUATION_ANNOTATIONS: Dict[str, Dict[str, Any]] = {
    "two_targets_crossing": {
        "global_id_switches": 4,
        "fragmentation": 3,
        "duplicate_global_tracks_mean": 0.292,
        "classification": "complex_association_negative_result",
        "provenance": "global-track-acceptance-v1.3 outer offline evaluation",
    },
}


def _canonical_float(value: float) -> float:
    """把 replay 的所有浮点数冻结为平台无关的显示/哈希精度。"""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("tower-view-v1 不接受 NaN 或 Infinity")
    canonical = round(numeric, CANONICAL_FLOAT_DECIMALS)
    return 0.0 if canonical == 0.0 else canonical


def canonicalize(value: Any) -> Any:
    """递归 canonicalize 回放值并稳定 dict 顺序，覆盖 event/weights 等嵌套字段。"""
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    raise TypeError(f"tower-view-v1 不支持的回放值类型：{type(value)!r}")


def _round_vector(values: Iterable[float]) -> List[float]:
    return [_canonical_float(value) for value in values]


def _radar_nodes(world: Mapping[str, Any]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for node_id in sorted(world["runtime"].centers):
        sensor_id = str(world["runtime"].own_sensor[node_id])
        position = world["sensor_positions"][sensor_id]
        nodes.append({
            "node_id": node_id,
            "sensor_id": sensor_id,
            "position_m": _round_vector((position.x, position.y, position.z)),
            "heading_deg": _canonical_float(NODE_LAYOUT[node_id]["heading_deg"]),
            "available": bool(world["runtime"].accounting.node(node_id).available),
        })
    return nodes


def _local_tracks(world: Mapping[str, Any], now_s: float) -> List[Dict[str, Any]]:
    manager = world["global_track_manager"]
    rows: List[Dict[str, Any]] = []
    for node_id, center in sorted(world["runtime"].centers.items()):
        for track in sorted(center.tracks, key=lambda item: str(item.track_id)):
            local_id = str(track.track_id)
            rows.append({
                "source_node_id": node_id,
                "local_track_id": local_id,
                "mapped_global_track_id": manager.local_to_global.get((node_id, local_id)),
                "position_m": _round_vector(
                    (track.position.x, track.position.y, track.position.z)),
                "velocity_mps": _round_vector(
                    (track.velocity.x, track.velocity.y, track.velocity.z)),
                "sigma_position_m": _round_vector(
                    (track.sigma_position.x, track.sigma_position.y,
                     track.sigma_position.z)),
                "status": str(track.status),
                "information_age_s": (
                    None if track.last_measurement_time is None
                    else _canonical_float(max(
                        0.0, now_s - float(track.last_measurement_time)))
                ),
                "local_updates": int(track.local_updates),
                "remote_updates": int(track.remote_updates),
            })
    return rows


def _global_tracks(world: Mapping[str, Any], now_s: float) -> List[Dict[str, Any]]:
    manager = world["global_track_manager"]
    rows: List[Dict[str, Any]] = []
    for global_id in sorted(manager.tracks):
        track = manager.tracks[global_id]
        payload = track.to_dict(now_s)
        rows.append({
            "global_track_id": global_id,
            "position_m": _round_vector(payload["position_m"]),
            "velocity_mps": _round_vector(payload["velocity_mps"]),
            "covariance_position_m2": _round_vector(
                payload["covariance_position_m2"]),
            "status": str(payload["status"]),
            "information_age_s": _canonical_float(payload["information_age_s"]),
            "source_node_ids": list(payload["participating_source_nodes"]),
            "active_source_node_ids": list(payload["active_source_nodes"]),
            "active_source_count": len(payload["active_source_nodes"]),
            "fusion_method": str(payload["fusion"]["method"]),
            "fusion_weights": dict(payload["fusion"]["weights"]),
            "created_at_s": _canonical_float(payload["created_at_s"]),
            "updates": int(payload["updates"]),
            "coasts": int(payload["coasts"]),
            "handover": dict(payload["handover"]),
        })
    return rows


def _mappings(world: Mapping[str, Any]) -> List[Dict[str, str]]:
    manager = world["global_track_manager"]
    return [
        {
            "source_node_id": source,
            "local_track_id": local,
            "global_track_id": global_id,
        }
        for (source, local), global_id in sorted(manager.local_to_global.items())
    ]


_EVENT_FIELDS = (
    "time_s", "event", "message_id", "source_node_id", "local_track_id",
    "global_track_id", "decision", "reason", "distance_m", "fusion_method",
    "participating_source_nodes", "active_source_nodes", "fusion_weights",
    "projection_delta_s", "handover_count", "dropped_at_s",
)


def _events(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {key: row[key] for key in _EVENT_FIELDS if key in row}
        for row in rows
    ]


def _debug_overlay(world: Mapping[str, Any]) -> List[Dict[str, Any]]:
    # 此导入只能从显式 debug_truth_overlay=True 分支抵达。
    from evaluate_global_tracking import _truth_snapshot

    snapshot = _truth_snapshot(world)
    return [
        {"debug_target_id": target_id, "position_m": _round_vector(position)}
        for target_id, position in sorted(snapshot.items())
    ]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_replay(
    scenario_id: str,
    *,
    seed: int = DEFAULT_SEED,
    debug_truth_overlay: bool = False,
) -> Dict[str, Any]:
    """运行冻结场景并返回不影响仿真的只读回放数据。"""
    if scenario_id not in TOWER_VIEW_SCENARIOS:
        raise ValueError(
            f"Tower View v1 场景必须是 {list(TOWER_VIEW_SCENARIOS)}，收到 {scenario_id!r}"
        )
    scenario = SCENARIOS[scenario_id]
    world = _build_feedback_world(
        seed=int(seed), steps=int(scenario["steps"]),
        mechanisms=dict(scenario["mechanisms"]),
        policy=SchedulerPolicy.RULE, task_gating="loop_gate",
        global_track_mode=GLOBAL_TRACK_MODE_TRACK_FUSION,
        global_share_mode=GLOBAL_SHARE_MODE_TRACK,
        target_specs=list(scenario["targets"]),
        node_budgets=scenario.get("node_budgets"),
        keep_decisions=True,
    )
    world["planner"] = build_scheduler(SchedulerPolicy.RULE)
    world["driver"].scheduler = world["planner"]
    for node_id, period_s in dict(
            scenario.get("sensor_update_period_s", {})).items():
        sensor = world["runtime"].suite.by_id(world["runtime"].own_sensor[node_id])
        sensor.config.update_period_s = float(period_s)

    frames: List[Dict[str, Any]] = []
    audit_cursor = 0
    scenario_evidence: Dict[str, Any] = {
        "local_tracker_resets": [], "false_local_track_injections": [],
        "sensor_update_period_s": dict(
            scenario.get("sensor_update_period_s", {})),
    }
    for _ in range(int(scenario["steps"])):
        _apply_pre_tick_events(
            world, scenario, float(world["clock"].now_s) + 1.0)
        world["driver"].tick()
        now_s = float(world["clock"].now_s)
        _apply_post_tick_events(world, scenario, now_s, scenario_evidence)
        new_events = world["global_track_manager"].audit_log[audit_cursor:]
        audit_cursor += len(new_events)
        frame: Dict[str, Any] = {
            "time_s": _canonical_float(now_s),
            "radar_nodes": _radar_nodes(world),
            "local_tracks": _local_tracks(world, now_s),
            "global_tracks": _global_tracks(world, now_s),
            "mappings": _mappings(world),
            "recent_events": _events(new_events),
        }
        if debug_truth_overlay:
            frame["debug_truth_tracks"] = _debug_overlay(world)
        frames.append(frame)

    result = world["driver"].finalize()
    observed_global_ids = sorted({
        str(track["global_track_id"])
        for frame in frames for track in frame["global_tracks"]
    })
    mapping_history: Dict[Tuple[str, str], List[str]] = {}
    for frame in frames:
        for mapping in frame["mappings"]:
            key = (mapping["source_node_id"], mapping["local_track_id"])
            history = mapping_history.setdefault(key, [])
            global_id = mapping["global_track_id"]
            if not history or history[-1] != global_id:
                history.append(global_id)
    summary = {
        "frame_count": len(frames),
        "duration_s": (frames[-1]["time_s"] if frames else 0.0),
        "global_track_ids_observed": observed_global_ids,
        "mapping_reassignment_count": sum(
            max(0, len(history) - 1) for history in mapping_history.values()),
        "global_track_drop_count": sum(
            event.get("event") == "global_track_dropped"
            for frame in frames for event in frame["recent_events"]
        ),
        "resource_conserved": bool(result.conservation.get("all_conserved", False)),
        "runtime_read_only_provenance": (
            "Sensor/local FusionCenter/RuntimeExecutor/CommBus/GlobalTrackManager snapshots"
        ),
    }
    if scenario_id in _FROZEN_EVALUATION_ANNOTATIONS:
        summary["frozen_evaluation_annotation"] = dict(
            _FROZEN_EVALUATION_ANNOTATIONS[scenario_id])

    replay: Dict[str, Any] = {
        "schema_version": TOWER_VIEW_SCHEMA_VERSION,
        "scenario": {
            "scenario_id": scenario_id,
            "title": str(scenario["title"]),
            "seed": int(seed),
            "steps": int(scenario["steps"]),
            "kind": str(scenario["kind"]),
        },
        "coordinate_frame": {
            "name": "ENU", "position_unit": "m", "velocity_unit": "m/s",
            "display_axes": ["east_x", "north_y"],
        },
        "frames": frames,
        "summary": summary,
    }
    if debug_truth_overlay:
        replay["debug_overlay"] = {
            "enabled": True,
            "warning": "Development-only ground truth overlay; never a runtime input.",
        }
    canonical_replay = canonicalize(replay)
    canonical_replay["frames_sha256"] = hashlib.sha256(
        _canonical_bytes(canonical_replay["frames"])).hexdigest()
    return canonical_replay


def write_replay(
    path: Path,
    scenario_id: str,
    *,
    seed: int = DEFAULT_SEED,
    debug_truth_overlay: bool = False,
) -> Dict[str, Any]:
    replay = build_replay(
        scenario_id, seed=seed, debug_truth_overlay=debug_truth_overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(replay, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return replay
