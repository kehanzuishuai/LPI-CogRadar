"""Tower View v2 只读诊断回放。

本模块只运行冻结的 Global Track v1.x 场景并读取已有快照/审计日志；不会向
Sensor、local FusionCenter、CommBus、GlobalTrackManager、调度器或 PPO 回写。
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from communication.message import MESSAGE_KIND_TRACK
from evaluate_global_tracking import EVALUATION_ASSOCIATION_GATE_M, _distance, _truth_snapshot
from global_fusion import (
    GLOBAL_SHARE_MODE_EVENT_TRACK,
    GLOBAL_SHARE_MODE_MEASUREMENT,
    GLOBAL_SHARE_MODE_NO_SHARE,
    GLOBAL_SHARE_MODE_TRACK,
    GLOBAL_TRACK_MODE_TRACK_FUSION,
)
from resource_management.closed_loop import _build_feedback_world
from resource_management.scheduling import SchedulerPolicy, build_scheduler
from tools.diagnose_global_track_pipeline import (
    _pipeline_events_from_runtime,
    _record_local_track_creations,
)
from tools.run_global_track_acceptance import (
    DEFAULT_SEED,
    SCENARIOS,
    _apply_post_tick_events,
    _apply_pre_tick_events,
)
from tower_view.replay import (
    _canonical_bytes,
    _debug_overlay,
    _global_tracks,
    _local_tracks,
    _mappings,
    _radar_nodes,
    canonicalize,
)


TOWER_VIEW_V2_SCHEMA_VERSION = "tower-view-v2"
TOWER_VIEW_V2_SCENARIOS: Tuple[str, ...] = (
    "dual_radar_single_target",
    "dual_radar_two_targets",
    "handover_disconnect_reconnect",
    "radar_reconnect_new_local_id",
    "leave_drop_reenter",
    "link_outage_loss_recovery",
    "asynchronous_radar_refresh",
    "two_targets_crossing",
)
TOWER_VIEW_V2_MODES: Tuple[str, ...] = (
    GLOBAL_SHARE_MODE_NO_SHARE,
    GLOBAL_SHARE_MODE_MEASUREMENT,
    GLOBAL_SHARE_MODE_TRACK,
    GLOBAL_SHARE_MODE_EVENT_TRACK,
)

_EVENT_TYPE_BY_STAGE = {
    "local_track_created": "LOCAL_TRACK_CREATED",
    "track_message_generated": "TRACK_MESSAGE_GENERATED",
    "track_message_sent": "TRACK_MESSAGE_SENT",
    "arrived": "TRACK_MESSAGE_ARRIVED",
    "arrived_unconsumed_at_horizon": "ARRIVED_UNCONSUMED",
    "transport_rejected": "STALE_REJECTED",
    "global_gate": "GLOBAL_GATE",
    "associated": "ASSOCIATED",
    "gate_candidate_rejected": "GATE_REJECTED",
    "ci_fused": "CI_FUSED",
}


def _copy_json(value: Any) -> Any:
    """只复制 JSON 标量，防止诊断层意外保留运行时对象引用。"""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _frame_communication(world: Mapping[str, Any], now_s: float) -> Dict[str, Any]:
    bus = world["bus"]
    manager = world["global_track_manager"]
    all_messages = [message for message in bus.log
                    if float(getattr(message, "sent_at", now_s)) <= now_s + 1e-12]
    track_messages = [message for message in all_messages
                      if getattr(message, "kind", "") == MESSAGE_KIND_TRACK]
    arrived = [message for message in track_messages
               if not message.dropped and message.arrived_at is not None
               and float(message.arrived_at) <= now_s + 1e-12]
    decisions = [row for row in manager.audit_log
                 if row.get("event") == "association"
                 and float(row.get("time_s", 0.0)) <= now_s + 1e-12]
    accepted = [row for row in decisions if row.get("decision") == "accepted"]
    association_rejected = [row for row in decisions if row.get("decision") == "rejected"]
    transport_rejected = [message for message in track_messages if message.dropped]
    return {
        "sent_messages": len(track_messages),
        "arrived_messages": len(arrived),
        "used_messages": len(accepted),
        "rejected_messages": len(transport_rejected) + len(association_rejected),
        "all_sent_messages": len(all_messages),
        "communication_bytes": sum(float(getattr(message, "size_bytes", 0.0))
                                   for message in all_messages),
        "message_utilization": (len(accepted) / len(track_messages)
                                if track_messages else None),
    }


def _frame_fusion(world: Mapping[str, Any], now_s: float,
                  radar_nodes: Sequence[Mapping[str, Any]],
                  local_tracks: Sequence[Mapping[str, Any]],
                  global_tracks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    manager = world["global_track_manager"]
    ci_count = sum(
        row.get("event") == "ci_fused"
        and float(row.get("time_s", 0.0)) <= now_s + 1e-12
        for row in manager.audit_log
    )
    source_nodes = {
        source for track in global_tracks
        for source in track.get("active_source_node_ids", [])
    }
    return {
        "local_track_count": len(local_tracks),
        "global_track_count": len(global_tracks),
        "single_source_global_count": sum(
            int(track.get("active_source_count", 0)) <= 1 for track in global_tracks),
        "multi_source_global_count": sum(
            int(track.get("active_source_count", 0)) > 1 for track in global_tracks),
        "ci_count": ci_count,
        "active_radar_count": sum(bool(node.get("available")) for node in radar_nodes),
        "active_source_count": len(source_nodes),
    }


def _global_tracks_v2(world: Mapping[str, Any], now_s: float) -> List[Dict[str, Any]]:
    rows = _global_tracks(world, now_s)
    manager = world["global_track_manager"]
    for row in rows:
        track = manager.tracks[row["global_track_id"]]
        row["retained_source_node_ids"] = list(row["source_node_ids"])
        row["last_state_time_s"] = float(track.last_state_time_s)
        row["last_measurement_time_s"] = float(track.last_measurement_time_s)
        row["last_update_time_s"] = float(
            max(track.last_state_time_s, track.last_measurement_time_s))
        row["ever_confirmed"] = bool(track.ever_confirmed)
    return rows


def _offline_metrics_and_events(
    frames: Sequence[Mapping[str, Any]],
    evaluation_positions: Mapping[float, Mapping[str, Tuple[float, float, float]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """外层评测后丢弃目标身份/位置，只返回聚合值和无目标 ID 事件。"""
    total = covered = 0
    squared_errors: List[float] = []
    duplicate_counts: List[int] = []
    previous_assignment: Dict[str, Optional[str]] = {}
    observed_assignments: Dict[str, set] = defaultdict(set)
    events: List[Dict[str, Any]] = []
    id_switches = fragmentation = 0
    for frame in frames:
        now_s = round(float(frame["time_s"]), 6)
        positions = evaluation_positions.get(now_s, {})
        tracks = list(frame.get("global_tracks", []))
        for evaluation_id, position in positions.items():
            total += 1
            candidates = sorted(
                (_distance(track["position_m"], position), track["global_track_id"])
                for track in tracks
            )
            within = [item for item in candidates
                      if item[0] <= EVALUATION_ASSOCIATION_GATE_M]
            duplicate_counts.append(max(0, len(within) - 1))
            current = within[0][1] if within else None
            if current is not None:
                covered += 1
                squared_errors.append(within[0][0] ** 2)
                prior = previous_assignment.get(evaluation_id)
                if prior is not None and prior != current:
                    id_switches += 1
                    events.append({
                        "time_s": now_s, "event_type": "ID_SWITCH",
                        "previous_global_track_id": prior,
                        "global_track_id": current,
                        "evaluation_only": True,
                        "reason": "nearest_global_assignment_changed",
                    })
                if current not in observed_assignments[evaluation_id] and observed_assignments[evaluation_id]:
                    fragmentation += 1
                    events.append({
                        "time_s": now_s, "event_type": "FRAGMENTATION",
                        "global_track_id": current,
                        "evaluation_only": True,
                        "reason": "additional_global_id_observed_for_same_evaluation_series",
                    })
                observed_assignments[evaluation_id].add(current)
                previous_assignment[evaluation_id] = current
    return ({
        "coverage": (covered / total if total else 0.0),
        "rmse_m": ((sum(squared_errors) / len(squared_errors)) ** 0.5
                   if squared_errors else None),
        "id_switch": id_switches,
        "fragmentation": fragmentation,
        "duplicate": (sum(duplicate_counts) / len(duplicate_counts)
                      if duplicate_counts else 0.0),
        "evaluation_note": "aggregate nearest-track evaluation only; never a runtime input",
    }, events)


def _manager_events(manager: Any) -> List[Dict[str, Any]]:
    """把既有审计语义映射为 UI 事件，不改变底层事件或决定。"""
    events: List[Dict[str, Any]] = []
    created_ids: set = set()
    for row in manager.audit_log:
        name = str(row.get("event", ""))
        base = {key: _copy_json(value) for key, value in row.items()
                if key != "event"}
        if name == "association":
            global_id = str(row.get("global_track_id", ""))
            if row.get("decision") == "accepted":
                if global_id and global_id not in created_ids and row.get("reason") in (
                        "new_track", "new_track_after_gate_reject"):
                    created_ids.add(global_id)
                    events.append({**base, "event_type": "GLOBAL_TRACK_CREATED"})
                events.append({**base, "event_type": "ASSOCIATED"})
                if row.get("reason") == "source_reconnect_associated":
                    events.append({**base, "event_type": "RECONNECTED"})
            else:
                event_type = ("STALE_REJECTED" if row.get("reason") in (
                    "duplicate_or_out_of_order_sequence", "non_monotonic_state_timestamp",
                ) else "GATE_REJECTED")
                events.append({**base, "event_type": event_type})
        elif name == "ci_fused":
            events.append({**base, "event_type": "CI_FUSED"})
        elif name == "global_track_dropped":
            events.append({**base, "event_type": "DROPPED"})
        elif name == "transport_rejected":
            events.append({**base, "event_type": "STALE_REJECTED"})
    return events


def _status_events(frames: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    previous_status: Dict[str, str] = {}
    previous_handover: Dict[str, int] = defaultdict(int)
    events: List[Dict[str, Any]] = []
    for frame in frames:
        now_s = frame["time_s"]
        for track in frame.get("global_tracks", []):
            global_id = track["global_track_id"]
            status = str(track.get("status", ""))
            if status in ("coasting", "stale_coasting") and previous_status.get(global_id) != status:
                events.append({"time_s": now_s, "event_type": "COASTING",
                               "global_track_id": global_id, "reason": status})
            handover_count = int(track.get("handover", {}).get("handover_count", 0))
            if handover_count > previous_handover[global_id]:
                events.append({
                    "time_s": now_s, "event_type": "HANDOVER",
                    "global_track_id": global_id,
                    "source_node_id": track.get("handover", {}).get(
                        "last_reporting_source_node_id", ""),
                    "handover_count": handover_count,
                    "reason": "reporting_source_changed",
                })
            previous_status[global_id] = status
            previous_handover[global_id] = handover_count
    return events


def _normalize_pipeline_event(row: Mapping[str, Any]) -> Dict[str, Any]:
    item = {key: _copy_json(value) for key, value in row.items()
            if key not in ("protocol", "mode", "environment_seed")}
    item["event_type"] = _EVENT_TYPE_BY_STAGE.get(
        str(row.get("stage", "")), str(row.get("stage", "")).upper())
    item.pop("stage", None)
    return item


def _evidence_chains(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    chains: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in events:
        message_id = str(row.get("message_id", ""))
        if message_id:
            chains[message_id].append({
                key: value for key, value in row.items()
                if key in ("time_s", "event_type", "decision", "reason",
                           "global_track_id", "source_node_id", "local_track_id",
                           "distance_m", "fusion_method", "projection_delta_s")
            })
    return [
        {"message_id": message_id,
         "steps": sorted(rows, key=lambda row: (
             float(row.get("time_s") or 0.0), str(row.get("event_type", ""))))}
        for message_id, rows in sorted(chains.items())
    ]


def _track_lifecycle(
    frames: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    histories: Dict[str, Dict[str, Any]] = {}
    for frame in frames:
        for track in frame.get("global_tracks", []):
            global_id = str(track["global_track_id"])
            row = histories.setdefault(global_id, {
                "global_track_id": global_id, "states": [], "source_history": [],
            })
            state = {
                "time_s": frame["time_s"], "status": track.get("status"),
                "information_age_s": track.get("information_age_s"),
                "active_source_node_ids": track.get("active_source_node_ids", []),
                "retained_source_node_ids": track.get("retained_source_node_ids", []),
                "fusion_method": track.get("fusion_method"),
            }
            if not row["states"] or row["states"][-1]["status"] != state["status"]:
                row["states"].append(state)
            source_key = (
                tuple(state["active_source_node_ids"]),
                tuple(state["retained_source_node_ids"]),
            )
            if not row["source_history"] or row["source_history"][-1]["source_key"] != source_key:
                row["source_history"].append({
                    "time_s": frame["time_s"], "source_key": source_key,
                    "active_source_node_ids": state["active_source_node_ids"],
                    "retained_source_node_ids": state["retained_source_node_ids"],
                })
    for event in events:
        global_id = str(event.get("global_track_id", ""))
        if not global_id:
            continue
        row = histories.setdefault(global_id, {
            "global_track_id": global_id, "states": [], "source_history": [],
        })
        if event.get("event_type") in ("HANDOVER", "RECONNECTED", "DROPPED"):
            lifecycle_status = {
                "HANDOVER": "handover",
                "RECONNECTED": "reconnect",
                "DROPPED": "dropped",
            }[str(event["event_type"])]
            row["states"].append({
                "time_s": event.get("time_s"),
                "status": lifecycle_status,
                "reason": event.get("reason", ""),
            })
    for row in histories.values():
        row["states"] = sorted(row["states"], key=lambda item: float(item["time_s"]))
        for item in row["source_history"]:
            item.pop("source_key", None)
    return [histories[key] for key in sorted(histories)]


def _attach_events(frames: List[Dict[str, Any]], events: Sequence[Mapping[str, Any]]) -> None:
    for frame in frames:
        frame["events"] = []
    for event in sorted(events, key=lambda row: (
            float(row.get("time_s") or 0.0), str(row.get("event_type", "")))):
        event_time = float(event.get("time_s") or 0.0)
        target = next((frame for frame in frames
                       if float(frame["time_s"]) + 1e-12 >= event_time),
                      frames[-1] if frames else None)
        if target is not None:
            target["events"].append(dict(event))


def build_replay_v2(
    scenario_id: str,
    *,
    mode: str = GLOBAL_SHARE_MODE_TRACK,
    seed: int = DEFAULT_SEED,
    debug_truth_overlay: bool = False,
) -> Dict[str, Any]:
    if scenario_id not in TOWER_VIEW_V2_SCENARIOS:
        raise ValueError(f"Tower View v2 未注册场景 {scenario_id!r}")
    if mode not in TOWER_VIEW_V2_MODES:
        raise ValueError(f"Tower View v2 未注册共享模式 {mode!r}")
    scenario = SCENARIOS[scenario_id]
    world = _build_feedback_world(
        seed=int(seed), steps=int(scenario["steps"]),
        mechanisms=dict(scenario["mechanisms"]),
        policy=SchedulerPolicy.RULE, task_gating="loop_gate",
        global_track_mode=GLOBAL_TRACK_MODE_TRACK_FUSION,
        global_share_mode=mode, target_specs=list(scenario["targets"]),
        node_budgets=scenario.get("node_budgets"), keep_decisions=True,
    )
    world["planner"] = build_scheduler(SchedulerPolicy.RULE)
    world["driver"].scheduler = world["planner"]
    for node_id, period_s in dict(scenario.get("sensor_update_period_s", {})).items():
        sensor = world["runtime"].suite.by_id(world["runtime"].own_sensor[node_id])
        sensor.config.update_period_s = float(period_s)

    frames: List[Dict[str, Any]] = []
    evaluation_positions: Dict[float, Dict[str, Tuple[float, float, float]]] = {}
    local_events: List[Dict[str, Any]] = []
    known_local_tracks: Dict[str, set] = {}
    scenario_evidence: Dict[str, Any] = {
        "local_tracker_resets": [], "false_local_track_injections": [],
        "sensor_update_period_s": dict(scenario.get("sensor_update_period_s", {})),
    }
    for _ in range(int(scenario["steps"])):
        _apply_pre_tick_events(world, scenario, float(world["clock"].now_s) + 1.0)
        world["driver"].tick()
        now_s = float(world["clock"].now_s)
        _apply_post_tick_events(world, scenario, now_s, scenario_evidence)
        _record_local_track_creations(
            world["runtime"].centers, known_local_tracks, now_s,
            mode, int(seed), local_events,
        )
        radar_nodes = _radar_nodes(world)
        local_tracks = _local_tracks(world, now_s)
        global_tracks = _global_tracks_v2(world, now_s)
        frame: Dict[str, Any] = {
            "time_s": now_s,
            "radar_nodes": radar_nodes,
            "local_tracks": local_tracks,
            "global_tracks": global_tracks,
            "mappings": _mappings(world),
            "communication": _frame_communication(world, now_s),
            "fusion": _frame_fusion(
                world, now_s, radar_nodes, local_tracks, global_tracks),
        }
        if debug_truth_overlay:
            frame["debug_truth_tracks"] = _debug_overlay(world)
        frames.append(frame)
        evaluation_positions[round(now_s, 6)] = _truth_snapshot(world)

    result = world["driver"].finalize()
    manager = world["global_track_manager"]
    final_now_s = float(world["clock"].now_s)
    pipeline = [
        _normalize_pipeline_event(row) for row in (
            local_events + _pipeline_events_from_runtime(
                manager, world["bus"], final_now_s, mode, int(seed)))
    ]
    evaluation_metrics, evaluation_events = _offline_metrics_and_events(
        frames, evaluation_positions)
    events = pipeline + _manager_events(manager) + _status_events(frames) + evaluation_events
    unique: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for event in events:
        key = (event.get("time_s"), event.get("event_type"),
               event.get("message_id"), event.get("global_track_id"))
        current = unique.get(key)
        if current is None or len(event) > len(current):
            unique[key] = event
    events = list(unique.values())
    _attach_events(frames, events)
    last_comm = frames[-1]["communication"] if frames else {}
    info_ages = [float(track["information_age_s"])
                 for frame in frames for track in frame["global_tracks"]]
    comparison = {
        **evaluation_metrics,
        "information_age_s": (sum(info_ages) / len(info_ages) if info_ages else None),
        "communication_bytes": float(last_comm.get("communication_bytes", 0.0)),
        "message_utilization": last_comm.get("message_utilization"),
    }
    replay: Dict[str, Any] = {
        "schema_version": TOWER_VIEW_V2_SCHEMA_VERSION,
        "scenario": {
            "scenario_id": scenario_id, "title": str(scenario["title"]),
            "seed": int(seed), "steps": int(scenario["steps"]),
            "kind": str(scenario["kind"]),
        },
        "sharing_mode": mode,
        "coordinate_frame": {
            "name": "ENU", "position_unit": "m", "velocity_unit": "m/s",
            "display_axes": ["east_x", "north_y"],
        },
        "read_only_contract": {
            "runtime_mutation": False,
            "source": "frozen Global Track v1.x snapshots and audit logs",
            "association_parameters_changed": False,
        },
        "frames": frames,
        "events": sorted(events, key=lambda row: (
            float(row.get("time_s") or 0.0), str(row.get("event_type", "")))),
        "message_evidence": _evidence_chains(events),
        "track_lifecycle": _track_lifecycle(frames, events),
        "comparison_metrics": comparison,
        "summary": {
            "frame_count": len(frames),
            "duration_s": frames[-1]["time_s"] if frames else 0.0,
            "resource_conserved": bool(result.conservation.get("all_conserved", False)),
            "event_count": len(events),
            "global_track_ids_observed": sorted({
                track["global_track_id"]
                for frame in frames for track in frame["global_tracks"]
            }),
            "runtime_read_only_provenance": (
                "Sensor/local FusionCenter/RuntimeExecutor/CommBus/GlobalTrackManager snapshots"
            ),
        },
    }
    if debug_truth_overlay:
        replay["debug_overlay"] = {
            "enabled": True,
            "warning": "Development-only ground truth overlay; never a runtime input.",
        }
    canonical = canonicalize(replay)
    canonical["frames_sha256"] = hashlib.sha256(
        _canonical_bytes(canonical["frames"])).hexdigest()
    canonical["replay_sha256"] = hashlib.sha256(
        _canonical_bytes({key: value for key, value in canonical.items()
                          if key not in ("frames_sha256", "replay_sha256")})).hexdigest()
    return canonical


def write_replay_v2(
    path: Path,
    scenario_id: str,
    *,
    mode: str = GLOBAL_SHARE_MODE_TRACK,
    seed: int = DEFAULT_SEED,
    debug_truth_overlay: bool = False,
) -> Dict[str, Any]:
    replay = build_replay_v2(
        scenario_id, mode=mode, seed=seed,
        debug_truth_overlay=debug_truth_overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(replay, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return replay
