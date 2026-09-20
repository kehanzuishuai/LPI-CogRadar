"""Global Track v1.3 确定性端到端与稳健性验收场景。

每个场景走同一条真实链路：

``Sensor -> local FusionCenter -> RuntimeExecutor share -> CommBus ->
GlobalTrackManager gate/CI -> GlobalObservation history``。

不训练 PPO，不改变 CI 网格、门限、通信/资源参数或 action 语义。真值只在此工具的
外层离线质量指标中读取，绝不进入 TrackMessage、GlobalTrackManager 或调度器。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from global_fusion import GLOBAL_SHARE_MODE_TRACK, GLOBAL_TRACK_MODE_TRACK_FUSION
from communication import SHARE_CONSTRAINED
from resource_management.closed_loop import RUNTIME_MODE_FEEDBACK, _build_feedback_world
from resource_management.scheduling import SchedulerPolicy, build_scheduler
from resource_management.units import ResourceUnit
from evaluate_global_tracking import (
    EVALUATION_ASSOCIATION_GATE_M,
    _distance,
    _offline_global_metrics,
    _truth_snapshot,
)
from tools.diagnose_global_track_pipeline import (
    _funnel_summary,
    _global_lifecycle_rows,
    _pipeline_events_from_runtime,
    _record_local_track_creations,
)


ACCEPTANCE_PROTOCOL = "global-track-acceptance-v1.3"
DEFAULT_SEED = 20260920
FOUNDATION_GATE_SCENARIOS: Tuple[str, ...] = (
    "dual_radar_single_target", "dual_radar_two_targets",
    "handover_disconnect_reconnect", "track_message_delay_reordering",
    "radar_reconnect_new_local_id", "leave_drop_reenter",
    "single_node_false_local_track", "link_outage_loss_recovery",
    "separated_multi_target_concurrency", "asynchronous_radar_refresh",
)

# 场景几何在运行前固定；不根据验收结果调坐标、门限、CI 或通信参数。
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "dual_radar_single_target": {
        "title": "双雷达单目标",
        "steps": 18,
        "targets": [
            {"target_id": "SC_A_T1", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 200.0},
        ],
        "mechanisms": {},
        "kind": "basic",
    },
    "dual_radar_two_targets": {
        "title": "双雷达双目标",
        "steps": 18,
        "targets": [
            {"target_id": "SC_B_T1", "x": -1600.0, "y": 0.0, "vx": 0.0, "vy": 100.0},
            {"target_id": "SC_B_T2", "x": 1600.0, "y": 0.0, "vx": 0.0, "vy": -100.0},
        ],
        "mechanisms": {},
        "kind": "basic",
    },
    "two_targets_crossing": {
        "title": "两目标交叉",
        "steps": 12,
        "targets": [
            {"target_id": "SC_C_T1", "x": -1800.0, "y": 0.0, "vx": 300.0, "vy": 0.0},
            {"target_id": "SC_C_T2", "x": 1800.0, "y": 0.0, "vx": -300.0, "vy": 0.0},
        ],
        "mechanisms": {},
        # 交叉被明确保留为关联压力诊断，不使用它反推或掩盖门限。
        "kind": "association_stress",
    },
    "handover_disconnect_reconnect": {
        "title": "handover / 失联 / 重接入",
        "steps": 26,
        "targets": [
            {"target_id": "SC_D_T1", "x": 0.0, "y": -8500.0, "vx": 0.0, "vy": 650.0},
        ],
        "mechanisms": {
            # NODE_B 在交接窗口短时不提供任务；返回后仍按既有规则恢复，不引入专用恢复逻辑。
            "unavailable_windows": {"NODE_B": [(10.0, 15.0)]},
        },
        "kind": "basic",
    },
    "track_message_delay_reordering": {
        "title": "E：TrackMessage 延迟 + 乱序",
        "steps": 30,
        "targets": [
            {"target_id": "SC_E_T1", "x": 0.0, "y": 0.0,
             "vx": 0.0, "vy": 120.0},
        ],
        "mechanisms": {
            "share_policy": SHARE_CONSTRAINED,
            "comm": {"base_delay_s": 0.2, "jitter_s": 0.0,
                     "loss_prob": 0.0, "reorder_prob": 0.5,
                     "reorder_extra_delay_s": 5.0},
        },
        "kind": "basic",
    },
    "radar_reconnect_new_local_id": {
        "title": "F：雷达掉线重启 + 新 local ID",
        "steps": 34,
        "targets": [
            {"target_id": "SC_F_T1", "x": 0.0, "y": 0.0,
             "vx": 0.0, "vy": 120.0},
        ],
        "mechanisms": {"unavailable_windows": {"NODE_A": [(10.0, 18.0)]}},
        "events": {"reset_local_tracker": {"node_id": "NODE_A", "time_s": 9.0}},
        "kind": "basic",
    },
    "leave_drop_reenter": {
        "title": "G：离场 → global drop → 重新进入",
        "steps": 62,
        "targets": [
            {"target_id": "SC_G_T1", "x": 0.0, "y": 0.0,
             "vx": 0.0, "vy": 0.0},
        ],
        "mechanisms": {},
        # 该场景必须覆盖 >30s coast + 重入；按 62 tick 的最大服务上限预先给足
        # 教学预算，避免把资源耗尽误诊为生命周期失败。成本模型本身不变。
        "node_budgets": {
            "NODE_A": {ResourceUnit.SAMPLE_SLOT: 70.0,
                       ResourceUnit.PROCESSING_OP: 70.0,
                       ResourceUnit.COMM_BYTE: 20000.0},
            "NODE_B": {ResourceUnit.SAMPLE_SLOT: 70.0,
                       ResourceUnit.PROCESSING_OP: 70.0,
                       ResourceUnit.COMM_BYTE: 20000.0},
        },
        "events": {"target_inactive_window": [10.0, 46.0]},
        "kind": "basic",
    },
    "single_node_false_local_track": {
        "title": "H：单节点虚假 local track 隔离",
        "steps": 22,
        "targets": [
            {"target_id": "SC_H_T1", "x": 0.0, "y": 0.0,
             "vx": 0.0, "vy": 120.0},
        ],
        "mechanisms": {},
        "events": {"inject_false_local_track": {
            "node_id": "NODE_B", "time_s": 8.0, "offset_x_m": 250.0,
            "local_track_id": "NODE_B-FALSE-1",
        }},
        "kind": "basic",
    },
    "link_outage_loss_recovery": {
        "title": "I：通信链路中断 / 丢包后恢复",
        "steps": 38,
        "targets": [
            {"target_id": "SC_I_T1", "x": 0.0, "y": 0.0,
             "vx": 0.0, "vy": 120.0},
        ],
        "mechanisms": {
            "share_policy": SHARE_CONSTRAINED,
            "comm": {"base_delay_s": 0.2, "jitter_s": 0.0,
                     "loss_prob": 0.0, "outage_windows": [(10.0, 20.0)]},
        },
        "events": {"link_outage_window": [10.0, 20.0]},
        "kind": "basic",
    },
    "separated_multi_target_concurrency": {
        "title": "J：三目标分离并发",
        "steps": 18,
        "targets": [
            {"target_id": "SC_J_T1", "x": -2200.0, "y": 0.0,
             "vx": 0.0, "vy": 80.0},
            {"target_id": "SC_J_T2", "x": 0.0, "y": 0.0,
             "vx": 0.0, "vy": 0.0},
            {"target_id": "SC_J_T3", "x": 2200.0, "y": 0.0,
             "vx": 0.0, "vy": -80.0},
        ],
        "mechanisms": {},
        "expected_target_count": 3,
        "kind": "basic",
    },
    "asynchronous_radar_refresh": {
        "title": "K：异步多雷达刷新",
        "steps": 30,
        "targets": [
            {"target_id": "SC_K_T1", "x": 0.0, "y": -1000.0,
             "vx": 180.0, "vy": 120.0},
        ],
        "mechanisms": {
            # NODE_B 首个 tick 暂停，使 sample→process→share 链与 A 错开；
            # 之后仍走完全相同的 RuntimeExecutor。
            "unavailable_windows": {"NODE_B": [(1.0, 1.0)]},
        },
        "sensor_update_period_s": {"NODE_A": 1.0, "NODE_B": 2.5},
        "kind": "basic",
    },
}


def _without_truth_payload(value: Any) -> bool:
    """递归查验运行时报告中没有 truth 字段；离线指标另行存放。"""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower().startswith(("truth", "true_")):
                return False
            if not _without_truth_payload(nested):
                return False
    elif isinstance(value, (list, tuple)):
        return all(_without_truth_payload(item) for item in value)
    return True


def _global_id_and_source_history(manager: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """从无真值 manager history 派生 global ID/source history。"""
    id_history: List[Dict[str, Any]] = []
    source_history: List[Dict[str, Any]] = []
    for snapshot in manager.history:
        now_s = float(snapshot["time_s"])
        tracks = list(snapshot.get("tracks", []))
        id_history.append({
            "time_s": now_s,
            "global_track_ids": sorted(str(track["global_track_id"]) for track in tracks),
            "n_global_tracks": len(tracks),
        })
        for track in tracks:
            fusion = track.get("fusion", {})
            source_history.append({
                "time_s": now_s,
                "global_track_id": str(track["global_track_id"]),
                # retained 来源可能已 stale；active_ci 只能从本帧 CI 权重读取。
                "retained_source_nodes": sorted(track.get("participating_source_nodes", [])),
                "active_ci_source_nodes": sorted(track.get("active_source_nodes", [])),
                "fusion_method": fusion.get("method", ""),
                "status": track.get("status", ""),
            })
    return id_history, source_history


def _local_track_summary(
    events: Iterable[Mapping[str, Any]], centers: Mapping[str, Any],
) -> Dict[str, Any]:
    created = [row for row in events if row.get("stage") == "local_track_created"]
    generated = [row for row in events if row.get("stage") == "track_message_generated"]
    created_keys = {
        (str(row.get("source_node_id", "")), str(row.get("local_track_id", "")))
        for row in created
    }
    generated_keys = {
        (str(row.get("source_node_id", "")), str(row.get("local_track_id", "")))
        for row in generated
    }
    unreported = sorted(created_keys - generated_keys)
    by_node = Counter(source for source, _local_id in created_keys)
    return {
        "created_total": len(created_keys),
        "created_by_node": dict(sorted(by_node.items())),
        "final_by_node": {
            node_id: len(center.tracks) for node_id, center in sorted(centers.items())
        },
        "track_message_generated_unique": len(generated_keys),
        "track_message_coverage_rate": (
            len(generated_keys) / len(created_keys) if created_keys else None
        ),
        "created_without_track_message": [
            {"source_node_id": source, "local_track_id": local}
            for source, local in unreported
        ],
    }


def _association_summary(manager: Any) -> Dict[str, Any]:
    rows = [row for row in manager.audit_log if row.get("event") == "association"]
    accepted = [row for row in rows if row.get("decision") == "accepted"]
    rejected = [row for row in rows if row.get("decision") == "rejected"]
    return {
        "decisions": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_reasons": dict(sorted(Counter(
            str(row.get("reason", "")) for row in accepted
        ).items())),
        "rejected_reasons": dict(sorted(Counter(
            str(row.get("reason", "")) for row in rejected
        ).items())),
        "history": [dict(row) for row in rows],
    }


def _ci_summary(manager: Any, final_now_s: float) -> Dict[str, Any]:
    rows = [row for row in manager.audit_log if row.get("event") == "ci_fused"]
    multi_events = [row for row in rows
                    if len(row.get("participating_source_nodes", [])) > 1]
    final_report = manager.report(final_now_s)
    final_tracks = []
    for track in final_report.get("tracks", []):
        active_nodes = sorted(track.get("active_source_nodes", []))
        retained_nodes = sorted(track.get("participating_source_nodes", []))
        final_tracks.append({
            "global_track_id": track["global_track_id"],
            "active_ci_source_nodes": active_nodes,
            "active_ci_source_count": len(active_nodes),
            "retained_source_nodes": retained_nodes,
            "retained_source_count": len(retained_nodes),
            "fusion_method": track.get("fusion", {}).get("method", ""),
            "information_age_s": track.get("information_age_s"),
            "status": track.get("status", ""),
            "ever_confirmed": bool(track.get("ever_confirmed", False)),
        })
    return {
        "ci_fusion_events": len(rows),
        "multi_source_ci_events": len(multi_events),
        "final_active_source_count": sum(item["active_ci_source_count"]
                                           for item in final_tracks),
        "final_tracks": final_tracks,
    }


def _track_message_state(bus: Any) -> Dict[str, Any]:
    rows = [message for message in bus.log
            if getattr(message, "kind", "") == "track"]
    by_local: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for message in rows:
        by_local[(str(message.source_node_id), str(message.local_track_id))].append(
            int(message.sequence_no))
    arrival_rows = sorted(
        (message for message in rows
         if not message.dropped and message.arrived_at is not None),
        key=lambda item: (float(item.arrived_at), str(item.message_id)),
    )
    last_by_local: Dict[Tuple[str, str], Tuple[int, float]] = {}
    out_of_order_arrivals: List[Dict[str, Any]] = []
    for message in arrival_rows:
        key = (str(message.source_node_id), str(message.local_track_id))
        previous = last_by_local.get(key)
        current = (int(message.sequence_no), float(message.state_timestamp_s))
        if previous is not None and (current[0] <= previous[0]
                                     or current[1] <= previous[1]):
            out_of_order_arrivals.append({
                "message_id": message.message_id,
                "source_node_id": key[0], "local_track_id": key[1],
                "arrived_at_s": float(message.arrived_at),
                "sequence_no": current[0], "state_timestamp_s": current[1],
                "previous_arrived_sequence_no": previous[0],
                "previous_arrived_state_timestamp_s": previous[1],
            })
        if previous is None or current[0] > previous[0]:
            last_by_local[key] = current
    return {
        "messages": len(rows),
        "out_of_order_arrival_count": len(out_of_order_arrivals),
        "out_of_order_arrivals": out_of_order_arrivals,
        "by_local_track": [
            {
                "source_node_id": source,
                "local_track_id": local,
                "sequence_numbers": sequence_numbers,
                "strictly_increasing": all(
                    right > left for left, right in zip(
                        sequence_numbers, sequence_numbers[1:])
                ),
            }
            for (source, local), sequence_numbers in sorted(by_local.items())
        ],
    }


def _apply_pre_tick_events(world: Mapping[str, Any], scenario: Mapping[str, Any],
                           next_time_s: float) -> None:
    """只在验收编排层切换预声明场景事件，不向运行时暴露 truth。"""
    window = (scenario.get("events") or {}).get("target_inactive_window")
    if window:
        active = not (float(window[0]) <= next_time_s < float(window[1]))
        for target in world["sim"].scene.targets:
            target.is_active = active


def _apply_post_tick_events(world: Mapping[str, Any], scenario: Mapping[str, Any],
                            now_s: float, evidence: Dict[str, Any]) -> None:
    events = scenario.get("events") or {}
    reset = events.get("reset_local_tracker")
    if reset and abs(now_s - float(reset["time_s"])) <= 1e-12:
        node_id = str(reset["node_id"])
        center = world["runtime"].centers[node_id]
        old_ids = [str(track.track_id) for track in center.tracks]
        center.retired_track_ids.extend(old_ids)
        center.tracks.clear()
        world["runtime"].track_outbox[node_id].clear()
        evidence["local_tracker_resets"].append({
            "time_s": now_s, "node_id": node_id,
            "retired_local_track_ids": old_ids,
        })
    injection = events.get("inject_false_local_track")
    if injection and abs(now_s - float(injection["time_s"])) <= 1e-12:
        node_id = str(injection["node_id"])
        center = world["runtime"].centers[node_id]
        if not center.tracks:
            raise RuntimeError("H 场景注入时尚无真实 local track")
        fake = copy.deepcopy(center.tracks[0])
        fake.track_id = str(injection["local_track_id"])
        position_type = type(fake.position)
        fake.position = position_type(
            fake.position.x + float(injection["offset_x_m"]),
            fake.position.y, fake.position.z,
        )
        if fake.filter is not None:
            fake.filter.position = fake.position
        fake.created_at = now_s
        fake.hits = 1
        fake.local_updates = 1
        fake.remote_updates = 0
        center.tracks.append(fake)
        evidence["false_local_track_injections"].append({
            "time_s": now_s, "node_id": node_id,
            "local_track_id": fake.track_id,
            "offset_x_m": float(injection["offset_x_m"]),
        })


def _robustness_summary(manager: Any, bus: Any,
                        scenario_evidence: Mapping[str, Any]) -> Dict[str, Any]:
    associations = [row for row in manager.audit_log
                    if row.get("event") == "association"]
    reconnects = [row for row in associations
                  if row.get("decision") == "accepted"
                  and row.get("reason") == "source_reconnect_associated"]
    stale_rejections = [row for row in associations if row.get("reason") in (
        "duplicate_or_out_of_order_sequence", "non_monotonic_state_timestamp",
    )]
    competing = [candidate for row in associations
                 for candidate in row.get("rejected_candidates", [])
                 if candidate.get("reason") ==
                 "same_source_simultaneous_competing_local_track"]
    report = manager.report(float(manager._last_predict_s or 0.0))
    return {
        "reconnect_count": len(reconnects),
        "reconnect_events": [dict(row) for row in reconnects],
        "stale_message_rejection_count": len(stale_rejections),
        "stale_message_rejections": [dict(row) for row in stale_rejections],
        "same_source_competitor_rejection_count": len(competing),
        "drop_count": len(report.get("dropped_tracks", [])),
        "dropped_tracks": list(report.get("dropped_tracks", [])),
        "active_local_to_global": list(report.get("local_to_global", [])),
        "scenario_events": dict(scenario_evidence),
        "comm_bus_out_of_order_count": int(bus.statistics()["n_out_of_order"]),
        "transport_rejection_count": sum(
            int(value) for value in bus.statistics()["drop_reasons"].values()),
        "transport_rejection_reasons": dict(bus.statistics()["drop_reasons"]),
        "active_source_count_history": [
            {
                "time_s": float(snapshot["time_s"]),
                "global_track_id": str(track["global_track_id"]),
                "active_source_count": len(track.get("active_source_nodes", [])),
                "active_source_nodes": sorted(track.get("active_source_nodes", [])),
                "status": str(track.get("status", "")),
            }
            for snapshot in manager.history
            for track in snapshot.get("tracks", [])
        ],
        "coasting_snapshot_count": sum(
            str(track.get("status", "")) in ("coasting", "stale_coasting")
            for snapshot in manager.history
            for track in snapshot.get("tracks", [])
        ),
    }


def _temporal_projection_summary(manager: Any, bus: Any) -> Dict[str, Any]:
    """只用 TrackMessage/audit 可见字段证明异步时间传播，不读取 truth。"""
    track_messages = [message for message in bus.log
                      if getattr(message, "kind", "") == "track"]
    timestamps_by_source: Dict[str, List[float]] = defaultdict(list)
    for message in track_messages:
        timestamps_by_source[str(message.source_node_id)].append(
            float(message.state_timestamp_s))
    cadence = []
    for source, values in sorted(timestamps_by_source.items()):
        unique = sorted(set(values))
        intervals = [right - left for left, right in zip(unique, unique[1:])]
        cadence.append({
            "source_node_id": source,
            "state_timestamps_s": unique,
            "intervals_s": intervals,
            "mean_interval_s": (sum(intervals) / len(intervals)
                                if intervals else None),
        })
    ci_rows = [row for row in manager.audit_log
               if row.get("event") == "ci_fused"]
    projection_rows = []
    asynchronous_ci_events = 0
    max_formula_error = 0.0
    for row in ci_rows:
        source_times = dict(row.get("active_source_state_timestamps_s", {}))
        if len(set(source_times.values())) > 1:
            asynchronous_ci_events += 1
        position = list(row.get("message_position_m", []))
        velocity = list(row.get("effective_projection_velocity_mps", []))
        projected = list(row.get("projected_message_position_m", []))
        delta = float(row.get("projection_delta_s", 0.0))
        formula_error = max(
            [abs(float(actual) - (float(base) + delta * float(speed)))
             for actual, base, speed in zip(projected, position, velocity)]
            or [0.0]
        )
        max_formula_error = max(max_formula_error, formula_error)
        projection_rows.append({
            "time_s": float(row.get("time_s", 0.0)),
            "message_id": str(row.get("message_id", "")),
            "source_node_id": str(row.get("source_node_id", "")),
            "state_timestamp_s": row.get("message_state_timestamp_s"),
            "arrived_at_s": row.get("message_arrived_at_s"),
            "fused_at_s": row.get("fused_at_s"),
            "projection_delta_s": delta,
            "active_source_state_timestamps_s": source_times,
            "projection_formula_error_m": formula_error,
        })
    return {
        "configured_or_observed_cadence": cadence,
        "asynchronous_ci_events": asynchronous_ci_events,
        "positive_projection_events": sum(
            row["projection_delta_s"] > 0.0 for row in projection_rows),
        "max_projection_formula_error_m": max_formula_error,
        "projection_rows": projection_rows,
    }


def _share_execution_summary(runtime_log: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [row for row in runtime_log if row.get("event") == "share"]
    return {
        "share_executions": len(rows),
        "multi_track_share_executions": sum(
            int(row.get("n_track_messages", 0)) > 1 for row in rows),
        "all_accounted_bytes_equal_sent_bytes": all(
            float(row.get("accounted_comm_bytes", 0.0))
            == float(row.get("sent_comm_bytes", 0.0)) for row in rows
        ),
        "rows": [dict(row) for row in rows],
    }


def _offline_match_diagnosis(
    global_history: Iterable[Mapping[str, Any]],
    local_history: Mapping[float, Sequence[Mapping[str, Any]]],
    truth_history: Mapping[float, Mapping[str, Tuple[float, float, float]]],
) -> Dict[str, Any]:
    """只在验收外层比较最近距离，用于解释 coverage=0；不返回 truth 状态。"""
    global_distances: List[float] = []
    local_distances: List[float] = []
    global_covered = local_covered = total = 0
    for snapshot in global_history:
        now_s = round(float(snapshot["time_s"]), 6)
        truths = truth_history.get(now_s, {})
        global_tracks = list(snapshot.get("tracks", []))
        local_tracks = list(local_history.get(now_s, []))
        for truth_position in truths.values():
            total += 1
            if global_tracks:
                distance = min(_distance(track["position_m"], truth_position)
                               for track in global_tracks)
                global_distances.append(distance)
                global_covered += int(distance <= EVALUATION_ASSOCIATION_GATE_M)
            if local_tracks:
                distance = min(_distance(track["position_m"], truth_position)
                               for track in local_tracks)
                local_distances.append(distance)
                local_covered += int(distance <= EVALUATION_ASSOCIATION_GATE_M)

    def summary(values: Sequence[float], covered: int) -> Dict[str, Any]:
        ordered = sorted(float(value) for value in values)
        return {
            "samples_with_estimate": len(ordered),
            "coverage": (covered / total if total else None),
            "nearest_distance_min_m": (ordered[0] if ordered else None),
            "nearest_distance_median_m": (
                ordered[len(ordered) // 2] if ordered else None),
            "nearest_distance_max_m": (ordered[-1] if ordered else None),
        }

    global_summary = summary(global_distances, global_covered)
    local_summary = summary(local_distances, local_covered)
    if not total:
        cause = "not_applicable_no_truth_samples"
    elif local_summary["coverage"] == 0:
        cause = "local_estimate_outside_offline_gate"
    elif global_summary["coverage"] == 0:
        cause = "global_state_or_lifecycle_outside_offline_gate"
    else:
        cause = "coverage_available"
    return {
        "evaluation_gate_m": EVALUATION_ASSOCIATION_GATE_M,
        "global": global_summary,
        "local": local_summary,
        "coverage_zero_cause": cause,
    }


def _scenario_judgement(
    scenario_id: str,
    scenario: Mapping[str, Any],
    local: Mapping[str, Any],
    association: Mapping[str, Any],
    ci: Mapping[str, Any],
    offline: Mapping[str, Any],
) -> Dict[str, Any]:
    """把证据归为三类问题；交叉仅诊断，不构造“必须成功”的门槛。"""
    categories: List[str] = []
    evidence: List[str] = []
    coverage_rate = local.get("track_message_coverage_rate")
    if coverage_rate is not None and float(coverage_rate) < 1.0:
        categories.append("reporting_mechanism_issue")
        evidence.append(
            f"local 创建 {local['created_total']}，但仅 {local['track_message_generated_unique']} "
            "条 unique local track 生成 TrackMessage。"
        )
    final_tracks = ci.get("final_tracks", [])
    final_single = bool(final_tracks) and all(
        int(track["active_ci_source_count"]) == 1 for track in final_tracks
    )
    if final_single and int(ci.get("multi_source_ci_events", 0)) > 0:
        categories.append("lifecycle_active_source_decay")
        evidence.append(
            "运行中已有多源 CI，但最终所有 global track 的活跃 CI 来源都只剩一个；"
            "需从 source history 判断来源 aging/stale 过滤，而不是称 CI 未执行。"
        )
    if scenario.get("kind") == "association_stress":
        metrics = offline
        if (int(metrics.get("global_id_switches", 0)) > 0
                or int(metrics.get("fragmentation", 0)) > 0
                or float(metrics.get("duplicate_global_tracks_mean", 0.0)) > 0.0):
            categories.append("complex_association_capability_limit")
            evidence.append(
                "交叉场景出现 ID switch、fragmentation 或 duplicate；这是 v1 简单门控关联的"
                "压力结果，保留给后续关联算法研究。"
            )
        else:
            evidence.append(
                "交叉场景当前未触发 ID/碎片/重复指标；这只是在该固定轨迹上的现象，不代表"
                "已具备复杂关联能力。"
            )
    if not categories:
        categories.append("no_first_order_reporting_or_lifecycle_failure_observed")
        evidence.append(
            "本固定场景中未观测到上报覆盖缺口或“多源 CI 后最终单源”的生命周期证据；"
            "不应外推为所有场景成功。"
        )
    return {
        "scenario_id": scenario_id,
        "categories": categories,
        "evidence": evidence,
        "coverage": offline.get("global_track_coverage"),
        "association_accepted": association.get("accepted"),
        "ci_events": ci.get("ci_fusion_events"),
    }


def _acceptance_checks(
    scenario_id: str,
    scenario: Mapping[str, Any],
    runtime_trace: Mapping[str, Any],
    offline: Mapping[str, Any],
) -> Dict[str, Any]:
    """A/B/D/E/F/G/H/I/J/K 固定工程闸门；交叉仅作复杂关联诊断。"""
    if scenario.get("kind") == "association_stress":
        return {
            "applicable": False,
            "passed": None,
            "checks": [],
            "failures": [],
            "boundary": "crossing_is_diagnostic_only_not_a_general_association_acceptance",
        }
    local = runtime_trace["local_tracks"]
    ci = runtime_trace["ci"]
    final_tracks = ci["final_tracks"]
    checks: List[Tuple[str, bool, str]] = [
        ("resource_conserved", bool(runtime_trace["resource_conserved"]),
         "真实 RuntimeExecutor/账本资源守恒"),
        ("truth_payload_clean", int(runtime_trace["runtime_truth_payload_violations"]) == 0,
         "运行时消息和观测无 truth payload"),
        ("local_track_message_coverage", local["track_message_coverage_rate"] == 1.0,
         "每条已创建 local track 至少生成一条 TrackMessage"),
        ("ci_exercised", int(ci["ci_fusion_events"]) > 0,
         "至少一次真实关联后的 CI 审计"),
    ]
    if scenario_id == "dual_radar_single_target":
        checks.extend([
            ("one_global_track", len(final_tracks) == 1,
             "同目标最终应形成一个 global track"),
            ("multisource_ci_exercised", int(ci["multi_source_ci_events"]) > 0,
             "双雷达同目标应至少出现一次多源 CI"),
            ("final_dual_active_sources",
             len(final_tracks) == 1
             and final_tracks[0]["active_ci_source_count"] == 2,
             "持续刷新后最终同一 global track 应保持双 active source"),
        ])
    elif scenario_id == "dual_radar_two_targets":
        checks.extend([
            ("four_local_tracks_reported",
             local["created_total"] == 4
             and local["track_message_generated_unique"] == 4,
             "双节点四条 local track 均必须实际生成 TrackMessage"),
            ("two_global_tracks", len(final_tracks) == 2,
             "双目标最终应保留两条 global track，不能靠遗漏上报掩盖"),
            ("two_dual_source_global_tracks",
             len(final_tracks) == 2 and all(
                 track["active_ci_source_count"] == 2 for track in final_tracks),
             "两个 global track 最终都应具有双 active source"),
            ("multi_track_share_exercised",
             int(runtime_trace["share_execution"]["multi_track_share_executions"]) > 0,
             "至少一次 share 应在真实账本下覆盖多个待上报 local tracks"),
        ])
    elif scenario_id == "handover_disconnect_reconnect":
        retained_both = any(
            {"NODE_A", "NODE_B"}.issubset(set(track["retained_source_nodes"]))
            for track in final_tracks
        )
        nonempty_ids = [set(row["global_track_ids"])
                        for row in runtime_trace["global_id_history"]
                        if row["global_track_ids"]]
        stable_id = bool(nonempty_ids) and len({tuple(sorted(ids)) for ids in nonempty_ids}) == 1
        checks.extend([
            ("handover_multisource_evidence", retained_both,
             "交接/重接入后同一 global track 应保留 A/B 来源证据"),
            ("stable_global_id_history", stable_id,
             "非空 global ID 历史不应在交接期换号"),
            ("handover_has_valid_global_coverage",
             float(offline.get("global_track_coverage", 0.0)) > 0.0
             and offline.get("rmse_m") is not None,
             "handover 不能只保持 ID；必须在离线门限内形成有效 global coverage"),
        ])
    elif scenario_id == "track_message_delay_reordering":
        nonempty_ids = [tuple(row["global_track_ids"])
                        for row in runtime_trace["global_id_history"]
                        if row["global_track_ids"]]
        robustness = runtime_trace["robustness"]
        checks.extend([
            ("real_out_of_order_arrival_exercised",
             runtime_trace["track_message_state"]
             ["out_of_order_arrival_count"] > 0
             and robustness["comm_bus_out_of_order_count"] > 0,
             "真实 CommBus 必须产生至少一次 TrackMessage 乱序到达"),
            ("stale_message_rejected",
             robustness["stale_message_rejection_count"] > 0,
             "晚到旧 sequence/timestamp 必须被 manager 明确拒绝"),
            ("global_id_stable_under_reordering",
             bool(nonempty_ids) and len(set(nonempty_ids)) == 1
             and len(nonempty_ids[0]) == 1,
             "乱序期间同一目标 global ID 不得换号或复制"),
        ])
    elif scenario_id == "radar_reconnect_new_local_id":
        mappings = runtime_trace["robustness"]["active_local_to_global"]
        node_a = [row for row in mappings if row["source_node_id"] == "NODE_A"]
        checks.extend([
            ("local_tracker_restart_exercised",
             bool(runtime_trace["robustness"]["scenario_events"]
                  ["local_tracker_resets"]),
             "必须真实清空掉线节点 local tracker 并产生重建证据"),
            ("new_local_id_after_reconnect", len({row["local_track_id"]
                                                   for row in node_a}) >= 2,
             "NODE_A 恢复后必须出现新的 local_track_id"),
            ("new_local_ids_map_to_original_global",
             len(node_a) >= 2
             and len({row["global_track_id"] for row in node_a}) == 1,
             "NODE_A 重建的 local ID 必须 reconnect 到原 global ID"),
            ("reconnect_audited",
             runtime_trace["robustness"]["reconnect_count"] > 0,
             "重接必须留下 source_reconnect_associated 审计"),
        ])
    elif scenario_id == "leave_drop_reenter":
        dropped_ids = {row["global_track_id"] for row in
                       runtime_trace["robustness"]["dropped_tracks"]}
        active_ids = {row["global_track_id"] for row in final_tracks}
        checks.extend([
            ("global_track_really_dropped",
             runtime_trace["robustness"]["drop_count"] > 0,
             "离场超过冻结 max_coast 后必须从 active 容器真正 drop"),
            ("reentry_creates_new_global_id",
             bool(dropped_ids) and bool(active_ids)
             and dropped_ids.isdisjoint(active_ids),
             "重新进入后必须新建 global ID，不得复活墓碑 ID"),
            ("reentry_has_valid_coverage",
             float(offline.get("global_track_coverage", 0.0)) > 0.0
             and offline.get("rmse_m") is not None,
             "重新进入后应恢复可评测的 global coverage"),
        ])
    elif scenario_id == "single_node_false_local_track":
        mappings = runtime_trace["robustness"]["active_local_to_global"]
        fake = [row for row in mappings
                if row["local_track_id"] == "NODE_B-FALSE-1"]
        genuine = [row for row in mappings
                   if row["source_node_id"] == "NODE_A"]
        false_global_ids = {row["global_track_id"] for row in fake}
        false_final = [track for track in final_tracks
                       if track["global_track_id"] in false_global_ids]
        checks.extend([
            ("false_local_track_injected",
             bool(runtime_trace["robustness"]["scenario_events"]
                  ["false_local_track_injections"]),
             "必须从单节点 local 层真实注入一条并发虚假航迹"),
            ("same_source_competition_audited",
             runtime_trace["robustness"]
             ["same_source_competitor_rejection_count"] > 0,
             "同源同状态时刻竞争必须留下拒绝候选审计"),
            ("false_track_does_not_pollute_real_global",
             bool(fake) and bool(genuine)
             and fake[0]["global_track_id"] not in {
                 row["global_track_id"] for row in genuine},
             "虚假 local track 只能独立 tentative，不得覆盖真实 global track"),
            ("false_track_never_confirmed",
             bool(false_final)
             and all(not track["ever_confirmed"] for track in false_final),
             "独立虚假 global track 不得在单次单源证据下进入 confirmed"),
        ])
    elif scenario_id == "link_outage_loss_recovery":
        window = scenario["events"]["link_outage_window"]
        active_history = runtime_trace["robustness"]["active_source_count_history"]
        before = [row["active_source_count"] for row in active_history
                  if row["time_s"] < float(window[0])]
        during = [row["active_source_count"] for row in active_history
                  if float(window[0]) <= row["time_s"] <= float(window[1]) + 3.0]
        after = [row["active_source_count"] for row in active_history
                 if row["time_s"] > float(window[1]) + 3.0]
        local_history = runtime_trace["local_state_history"]
        local_updates_during = {
            node_id: len({row["state_timestamp_s"] for row in local_history
                          if row["source_node_id"] == node_id
                          and row["state_timestamp_s"] is not None
                          and float(window[0]) <= row["time_s"] <= float(window[1])})
            for node_id in ("NODE_A", "NODE_B")
        }
        nonempty_ids = [tuple(row["global_track_ids"])
                        for row in runtime_trace["global_id_history"]
                        if row["global_track_ids"]]
        checks.extend([
            ("real_link_outage_rejections",
             runtime_trace["robustness"]["transport_rejection_reasons"]
             .get("link_outage", 0) > 0,
             "TrackMessage 必须经真实 CommLink 以 link_outage 被拒绝"),
            ("local_tracking_continues_during_outage",
             all(count >= 2 for count in local_updates_during.values()),
             "中央链路中断期间两节点 local state timestamp 仍持续推进"),
            ("active_sources_degrade_and_recover",
             bool(before) and max(before) == 2
             and bool(during) and min(during) < 2
             and bool(after) and max(after) == 2
             and final_tracks and final_tracks[0]["active_ci_source_count"] == 2,
             "中断时 active source 数下降，恢复后同一航迹重新多源"),
            ("outage_global_id_stable",
             bool(nonempty_ids) and len(set(nonempty_ids)) == 1
             and len(nonempty_ids[0]) == 1
             and runtime_trace["robustness"]["drop_count"] == 0,
             "短时中断只允许 coast，不得换 global ID 或 drop"),
        ])
    elif scenario_id == "separated_multi_target_concurrency":
        expected = int(scenario["expected_target_count"])
        mappings = runtime_trace["robustness"]["active_local_to_global"]
        by_global: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in mappings:
            by_global[str(row["global_track_id"])].append(row)
        sequence_streams = runtime_trace["track_message_state"]["by_local_track"]
        checks.extend([
            ("all_multi_target_local_tracks_reported",
             local["created_total"] == 2 * expected
             and local["track_message_generated_unique"] == 2 * expected,
             "两节点的每条分离目标 local track 都必须独立生成 TrackMessage"),
            ("one_global_per_separated_target",
             len(final_tracks) == expected
             and int(offline.get("global_id_switches", -1)) == 0
             and int(offline.get("fragmentation", -1)) == 0
             and float(offline.get("duplicate_global_tracks_mean", -1.0)) == 0.0,
             "每个分离目标恰有一条稳定 global track，无重复/碎裂/换号"),
            ("mapping_is_one_per_source_per_global",
             len(by_global) == expected and all(
                 len(rows) == 2
                 and {row["source_node_id"] for row in rows}
                 == {"NODE_A", "NODE_B"}
                 for rows in by_global.values()),
             "每条 global track 只能映射 A/B 各一条 local track，禁止跨目标串线"),
            ("independent_sequence_streams",
             len(sequence_streams) == 2 * expected
             and all(row["strictly_increasing"] for row in sequence_streams),
             "每条 local track 使用独立且严格递增的 sequence stream"),
            ("all_multi_target_globals_dual_source",
             len(final_tracks) == expected and all(
                 track["active_ci_source_count"] == 2 for track in final_tracks),
             "所有分离目标最终均保持双 active source"),
        ])
    elif scenario_id == "asynchronous_radar_refresh":
        temporal = runtime_trace["temporal_projection"]
        periods = runtime_trace["robustness"]["scenario_events"]
        nonempty_ids = [tuple(row["global_track_ids"])
                        for row in runtime_trace["global_id_history"]
                        if row["global_track_ids"]]
        checks.extend([
            ("asynchronous_periods_predeclared",
             periods["sensor_update_period_s"]
             == {"NODE_A": 1.0, "NODE_B": 2.5},
             "A/B 使用预声明的不同传感器刷新周期"),
            ("asynchronous_state_timestamps_reach_ci",
             temporal["asynchronous_ci_events"] > 0,
             "至少一次多源 CI 的 A/B state_timestamp_s 不同步"),
            ("state_is_projected_to_fusion_time",
             temporal["positive_projection_events"] > 0
             and temporal["max_projection_formula_error_m"] <= 1e-9,
             "迟到状态必须按有效速度投影至 fused_at_s，不能把旧位置当当前位置"),
            ("async_global_id_stable",
             bool(nonempty_ids) and len(set(nonempty_ids)) == 1
             and len(nonempty_ids[0]) == 1
             and len(final_tracks) == 1
             and final_tracks[0]["active_ci_source_count"] == 2,
             "异步刷新期间 global ID 稳定且最终恢复双源"),
        ])
    normalized = [
        {"name": name, "passed": passed, "requirement": requirement}
        for name, passed, requirement in checks
    ]
    failures = [item["name"] for item in normalized if not item["passed"]]
    return {
        "applicable": True,
        "passed": not failures,
        "checks": normalized,
        "failures": failures,
    }


def run_scenario(scenario_id: str, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """独立运行一个冻结场景，输出无真值 runtime trace + 离线质量聚合。"""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"未知 scenario {scenario_id!r}，可选 {sorted(SCENARIOS)}")
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
    driver = world["driver"]
    # 固定 Rule，只替换学习占位 planner；执行器/场景/资源和通信路径不变。
    world["planner"] = build_scheduler(SchedulerPolicy.RULE)
    driver.scheduler = world["planner"]
    configured_periods = dict(scenario.get("sensor_update_period_s", {}))
    for node_id, period_s in configured_periods.items():
        sensor = world["runtime"].suite.by_id(world["runtime"].own_sensor[node_id])
        sensor.config.update_period_s = float(period_s)
    known_local_tracks: Dict[str, set] = {
        node_id: set() for node_id in world["runtime"].centers
    }
    local_events: List[Dict[str, Any]] = []
    scenario_evidence: Dict[str, Any] = {
        "local_tracker_resets": [], "false_local_track_injections": [],
        "sensor_update_period_s": configured_periods,
    }
    truth_history: Dict[float, Dict[str, Tuple[float, float, float]]] = {}
    local_history: Dict[float, List[Dict[str, Any]]] = {}
    local_state_history: List[Dict[str, Any]] = []
    for _ in range(int(scenario["steps"])):
        _apply_pre_tick_events(
            world, scenario, float(world["clock"].now_s) + 1.0)
        driver.tick()
        now_s = float(world["clock"].now_s)
        _apply_post_tick_events(world, scenario, now_s, scenario_evidence)
        _record_local_track_creations(
            world["runtime"].centers, known_local_tracks, now_s,
            GLOBAL_SHARE_MODE_TRACK, int(seed), local_events,
        )
        # 唯一真值读取点；不可传入任何运行时对象。
        truth_history[round(now_s, 6)] = _truth_snapshot(world)
        local_history[round(now_s, 6)] = [
            {"position_m": (float(track.position.x), float(track.position.y),
                            float(track.position.z))}
            for center in world["runtime"].centers.values()
            for track in center.tracks
        ]
        local_state_history.extend({
            "time_s": now_s,
            "source_node_id": node_id,
            "local_track_id": str(track.track_id),
            "state_timestamp_s": (None if track.last_measurement_time is None
                                  else float(track.last_measurement_time)),
            "status": str(track.status),
            "local_updates": int(track.local_updates),
        } for node_id, center in sorted(world["runtime"].centers.items())
          for track in center.tracks)

    result = driver.finalize()
    manager = world["global_track_manager"]
    final_now_s = float(world["clock"].now_s)
    runtime_events = _pipeline_events_from_runtime(
        manager, world["bus"], final_now_s, GLOBAL_SHARE_MODE_TRACK, int(seed),
    )
    lifecycle = _global_lifecycle_rows(
        manager, final_now_s, GLOBAL_SHARE_MODE_TRACK, int(seed),
    )
    events = local_events + runtime_events
    funnel = _funnel_summary(events, lifecycle)
    offline = _offline_global_metrics(manager.history, truth_history)
    offline["match_diagnosis"] = _offline_match_diagnosis(
        manager.history, local_history, truth_history)
    local = _local_track_summary(events, world["runtime"].centers)
    association = _association_summary(manager)
    ci = _ci_summary(manager, final_now_s)
    id_history, source_history = _global_id_and_source_history(manager)
    runtime = result.metrics.get("runtime_feedback", {})
    runtime_trace = {
        "scenario_id": scenario_id,
        "local_tracks": local,
        "pipeline_funnel": funnel,
        "pipeline_events": events,
        "local_state_history": local_state_history,
        "global_association": association,
        "global_id_history": id_history,
        "source_history": source_history,
        "ci": ci,
        "track_message_state": _track_message_state(world["bus"]),
        "robustness": _robustness_summary(
            manager, world["bus"], scenario_evidence),
        "temporal_projection": _temporal_projection_summary(
            manager, world["bus"]),
        "share_execution": _share_execution_summary(result.runtime_log),
        "global_lifecycle": lifecycle,
        "communication_bytes": float(result.metrics.get("comm_overhead_bytes", 0.0)),
        "resource_conserved": bool(result.conservation.get("all_conserved", False)),
        "runtime_truth_payload_violations": int(runtime.get("truth_payload_violations", 0)),
    }
    if not _without_truth_payload(runtime_trace):
        raise RuntimeError("验收运行时 trace 含禁止的 truth 字段")
    judgement = _scenario_judgement(
        scenario_id, scenario, local, association, ci, offline,
    )
    acceptance = _acceptance_checks(scenario_id, scenario, runtime_trace, offline)
    return {
        "scenario_id": scenario_id,
        "scenario_title": scenario["title"],
        "scenario_kind": scenario["kind"],
        "seed": int(seed),
        "steps": int(scenario["steps"]),
        "runtime_trace": runtime_trace,
        # 此层仅是外部离线评测聚合，刻意不输出 truth 位置或 ID。
        "offline_evaluation": offline,
        "judgement": judgement,
        "acceptance": acceptance,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Global Track v1.3 最终稳健性验收与基础机制冻结报告",
        "",
        "固定 A–K 场景均走 Sensor → local FusionCenter → RuntimeExecutor → CommBus → "
        "GlobalTrackManager 的真实链路。未改 CI/门限/默认物理与 PPO；E 使用预声明乱序通信压力，"
        "G 按 62 tick 服务上限预声明耐久预算，I 使用固定链路中断窗口，J 使用固定分离目标几何，"
        "K 使用预声明异步刷新周期，均不根据结果回调。真值仅用于外层离线指标。",
        "",
        "v1 旧行为记录保留：B/C 均为 4 条 local track 仅 2 条上报；四场景最终均退化为"
        "单 active source；D coverage=0、RMSE=None。v1.1 不回写这些历史结论。",
        "",
        "| 场景 | local 创建 / 上报 | association / CI | final active sources | coast / stale reject / drop / reconnect | utilization | coverage | ID switch | fragmentation | duplicate | RMSE m | bytes | 基础验收 | 判断 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run in report["scenarios"]:
        trace = run["runtime_trace"]
        local = trace["local_tracks"]
        association = trace["global_association"]
        ci = trace["ci"]
        offline = run["offline_evaluation"]
        robust = trace["robustness"]
        active = ", ".join(
            f"{item['global_track_id']}:{item['active_ci_source_count']}"
            for item in ci["final_tracks"]
        ) or "—"
        lines.append(
            f"| `{run['scenario_id']}` | {local['created_total']} / "
            f"{local['track_message_generated_unique']} | {association['accepted']} / "
            f"{ci['ci_fusion_events']} | {active} | "
            f"{robust['coasting_snapshot_count']} / "
            f"{robust['stale_message_rejection_count']} / "
            f"{robust['drop_count']} / {robust['reconnect_count']} | "
            f"{trace['pipeline_funnel']['message_event_funnel']['message_utilization_rate']} | "
            f"{offline['global_track_coverage']} | {offline['global_id_switches']} | "
            f"{offline['fragmentation']} | {offline['duplicate_global_tracks_mean']} | "
            f"{offline['rmse_m']} | {trace['communication_bytes']} | "
            f"{('N/A' if run['acceptance']['passed'] is None else run['acceptance']['passed'])} | "
            f"{', '.join(run['judgement']['categories'])} |"
        )
    lines.extend(["", "## 场景证据与判断", ""])
    for run in report["scenarios"]:
        trace = run["runtime_trace"]
        local = trace["local_tracks"]
        lines.extend([
            f"### {run['scenario_title']} (`{run['scenario_id']}`)",
            "",
            f"- 未上报 local track：`{local['created_without_track_message']}`。",
            f"- source/CI：CI 共 `{trace['ci']['ci_fusion_events']}` 次，其中多源 CI "
            f"`{trace['ci']['multi_source_ci_events']}` 次；最终 active sources："
            f"`{trace['ci']['final_tracks']}`。",
            f"- global ID history 和 source history 已完整写入 JSON；资源守恒="
            f"`{trace['resource_conserved']}`，运行时 truth payload violations="
            f"`{trace['runtime_truth_payload_violations']}`。",
            f"- 一次 share 多航迹执行次数："
            f"`{trace['share_execution']['multi_track_share_executions']}`；逐 local track "
            f"sequence 证据已写入 JSON。",
            f"- 稳健性事件：stale rejection=`{trace['robustness']['stale_message_rejection_count']}`，"
            f"coasting snapshots=`{trace['robustness']['coasting_snapshot_count']}`，"
            f"drop=`{trace['robustness']['drop_count']}`，reconnect="
            f"`{trace['robustness']['reconnect_count']}`，CommBus out-of-order="
            f"`{trace['robustness']['comm_bus_out_of_order_count']}`。",
            f"- 消息利用率："
            f"`{trace['pipeline_funnel']['message_event_funnel']['message_utilization_rate']}`。",
        ])
        for evidence in run["judgement"]["evidence"]:
            lines.append(f"- 判断证据：{evidence}")
        acceptance = run["acceptance"]
        if acceptance["applicable"]:
            lines.append(
                f"- 基础机制验收：`{acceptance['passed']}`；失败项：`{acceptance['failures']}`。"
            )
        else:
            lines.append(f"- 基础机制验收：N/A；边界：`{acceptance['boundary']}`。")
        diagnosis = run["offline_evaluation"].get("match_diagnosis", {})
        lines.append(
            f"- 离线匹配诊断：`{diagnosis.get('coverage_zero_cause')}`；"
            f"global=`{diagnosis.get('global')}`，local=`{diagnosis.get('local')}`。"
        )
        lines.append("")
    summary = report["overall_judgement"]
    lines.extend([
        "## 总体归类",
        "",
        f"- 上报机制问题场景：`{summary['reporting_mechanism_scenarios']}`。",
        f"- 生命周期活跃来源退化场景：`{summary['lifecycle_decay_scenarios']}`。",
        f"- 复杂关联压力场景：`{summary['complex_association_scenarios']}`。",
        f"- A/B/D/E/F/G/H/I/J/K 基础机制未通过场景："
        f"`{summary['basic_acceptance_failed_scenarios']}`。",
        f"- Global Track v1.x 基础机制最终冻结：`{report['foundation_freeze']['frozen']}`；"
        f"缺失场景=`{report['foundation_freeze']['missing_scenarios']}`。",
        "- 交叉场景即使本次没有失败，也只代表固定轨迹未触发指标，不能等价为已解决复杂关联；"
        "不得据此引入或宣称 JPDA。",
        "",
        "运行时报告不含 truth 状态；coverage/ID switch/fragmentation/duplicate/RMSE 是外层离线评测。",
    ])
    return "\n".join(lines) + "\n"


def _overall_judgement(runs: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    by_category: Dict[str, List[str]] = defaultdict(list)
    for run in runs:
        for category in run["judgement"]["categories"]:
            by_category[category].append(str(run["scenario_id"]))
    return {
        "reporting_mechanism_scenarios": by_category["reporting_mechanism_issue"],
        "lifecycle_decay_scenarios": by_category["lifecycle_active_source_decay"],
        "complex_association_scenarios": by_category["complex_association_capability_limit"],
        "basic_acceptance_failed_scenarios": [
            str(run["scenario_id"]) for run in runs
            if run["acceptance"]["applicable"] and not run["acceptance"]["passed"]
        ],
    }


def run_acceptance(
    output_dir: Path,
    scenario_ids: Sequence[str] = tuple(SCENARIOS),
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """运行一个或全部独立场景，并生成统一 JSON/Markdown。"""
    unknown = sorted(set(scenario_ids) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"未知 scenario_ids: {unknown}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [run_scenario(scenario_id, seed) for scenario_id in scenario_ids]
    report = {
        "protocol": ACCEPTANCE_PROTOCOL,
        "scope": "deterministic_global_track_end_to_end_acceptance",
        "seed": int(seed),
        "scenario_ids": list(scenario_ids),
        "honesty_boundaries": [
            "No PPO, JPDA, new probabilistic association algorithm, reward, CI-grid, gate-threshold, default physical-model, or frozen-contract modification.",
            "E uses a predeclared constrained-link reordering stress configuration; G uses a predeclared 62-tick endurance budget; I uses a fixed outage window; J uses fixed separated-target geometry; K uses configured A=1.0s/B=2.5s sensor refresh periods. None is tuned from acceptance outcomes.",
            "In K the sample-process-share task chain yields an observed 3s local-state cadence per source with a 1s phase offset. The evidence establishes asynchronous timestamps and timestamp-aware projection, not literal 1s/2.5s end-to-end reporting throughput.",
            "All association and CI use only arrived TrackMessages; truth is only read by the outer offline evaluator.",
            "Crossing, close formation, systematic bias, and other complex association cases remain enhancement branches: a failure is retained and a non-failure is not a general association claim.",
        ],
        "v1_baseline_reference": {
            "protocol": "global-track-acceptance-v1",
            "preserved_conclusion": (
                "B/C each created 4 local tracks but only 2 unique local tracks reported; "
                "all scenarios ended with single active source, and D coverage was 0 with RMSE not applicable."
            ),
            "source": "pre-v1.1 global_track_acceptance_report",
        },
        "scenarios": runs,
        "overall_judgement": _overall_judgement(runs),
    }
    by_id = {str(run["scenario_id"]): run for run in runs}
    missing = [scenario_id for scenario_id in FOUNDATION_GATE_SCENARIOS
               if scenario_id not in by_id]
    failed = [scenario_id for scenario_id in FOUNDATION_GATE_SCENARIOS
              if scenario_id in by_id
              and not bool(by_id[scenario_id]["acceptance"]["passed"])]
    report["foundation_freeze"] = {
        "required_scenarios": list(FOUNDATION_GATE_SCENARIOS),
        "missing_scenarios": missing,
        "failed_scenarios": failed,
        "frozen": not missing and not failed,
        "v1x_final_freeze": not missing and not failed,
        "no_more_foundation_scenarios": not missing and not failed,
        "next_stage": ("Tower View" if not missing and not failed
                       else "Global Track robustness acceptance remains open"),
        "boundary": ("Crossing remains a retained complex-association negative result; "
                     "close formation and systematic bias remain enhancement branches; "
                     "the foundation freeze is not a JPDA/MHT claim."),
    }
    json_path = output_dir / "global_track_acceptance_report.json"
    markdown_path = output_dir / "global_track_acceptance_report.md"
    # 第一次 v1.1 运行时把已有 v1 报告原样归档；后续 v1.1 重跑不覆盖基线。
    if json_path.is_file():
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if previous.get("protocol") == "global-track-acceptance-v1":
            (output_dir / "global_track_acceptance_report_v1_baseline.json").write_text(
                json.dumps(previous, indent=2, ensure_ascii=False), encoding="utf-8")
            if markdown_path.is_file():
                (output_dir / "global_track_acceptance_report_v1_baseline.md").write_text(
                    markdown_path.read_text(encoding="utf-8"), encoding="utf-8")
        elif previous.get("protocol") == "global-track-acceptance-v1.1":
            (output_dir / "global_track_acceptance_report_v11_baseline.json").write_text(
                json.dumps(previous, indent=2, ensure_ascii=False), encoding="utf-8")
            if markdown_path.is_file():
                (output_dir / "global_track_acceptance_report_v11_baseline.md").write_text(
                    markdown_path.read_text(encoding="utf-8"), encoding="utf-8")
        elif previous.get("protocol") == "global-track-acceptance-v1.2":
            (output_dir / "global_track_acceptance_report_v12_baseline.json").write_text(
                json.dumps(previous, indent=2, ensure_ascii=False), encoding="utf-8")
            if markdown_path.is_file():
                (output_dir / "global_track_acceptance_report_v12_baseline.md").write_text(
                    markdown_path.read_text(encoding="utf-8"), encoding="utf-8")
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("output/global_track_acceptance"))
    parser.add_argument("--scenario", choices=sorted(SCENARIOS),
                        action="append", help="可重复；省略时运行全部场景")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    report = run_acceptance(args.out_dir, args.scenario or tuple(SCENARIOS), args.seed)
    print(json.dumps(report["overall_judgement"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
