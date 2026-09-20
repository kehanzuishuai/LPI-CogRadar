"""Global Track / Track-to-Track Fusion v1 的只读链路漏斗诊断。

运行固定 Rule development 场景，记录：

``local_track_created -> TrackMessage generated/sent/arrived -> global gate
-> associated -> CI fused -> maintained/dropped``。

这不是新的调度器、关联器或评测协议：不改 CI 参数、门限、任务语义、资源账本或 PPO。
真值只在 driver tick 结束后的离线 coverage 计算中读取；所有运行时事件均来自
RuntimeExecutor、CommBus 和 GlobalTrackManager 的无真值日志/快照。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# 允许从仓库根目录以 ``python tools/diagnose_global_track_pipeline.py`` 直接运行，
# 同时不要求把工程安装成 site-package。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from communication.message import MESSAGE_KIND_TRACK
from global_fusion import (
    GLOBAL_SHARE_MODE_EVENT_TRACK,
    GLOBAL_SHARE_MODE_TRACK,
    GLOBAL_TRACK_MODE_TRACK_FUSION,
)
from resource_management.closed_loop import RUNTIME_MODE_FEEDBACK, _build_feedback_world
from resource_management.scheduling import SchedulerPolicy, build_scheduler


DIAGNOSTIC_PROTOCOL = "global-track-pipeline-diagnosis-v1"
DEVELOPMENT_SEEDS: Tuple[int, ...] = (41, 73, 109)
DIAGNOSTIC_MODES: Tuple[str, ...] = (
    GLOBAL_SHARE_MODE_TRACK,
    GLOBAL_SHARE_MODE_EVENT_TRACK,
)
DEFAULT_STEPS = 18
OFFLINE_COVERAGE_GATE_M = 1_500.0


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _offline_truth_snapshot(world: Mapping[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    """唯一读取 Scene truth 的函数；返回值绝不写入 runtime/manager/message。"""
    return {
        target.target_id: (target.position.x, target.position.y, target.position.z)
        for target in world["sim"].scene.targets
    }


def _offline_coverage(
    history: Iterable[Mapping[str, Any]],
    truth_history: Mapping[float, Mapping[str, Tuple[float, float, float]]],
) -> Dict[str, Any]:
    """外层只读对照；只返回聚合数值，不输出任何 truth 位置/ID。"""
    total = covered = 0
    squared_errors: List[float] = []
    for snapshot in history:
        truths = truth_history.get(round(float(snapshot["time_s"]), 6), {})
        tracks = list(snapshot.get("tracks", []))
        for truth_position in truths.values():
            total += 1
            distances = [_distance(track["position_m"], truth_position)
                         for track in tracks]
            if not distances:
                continue
            nearest = min(distances)
            if nearest <= OFFLINE_COVERAGE_GATE_M:
                covered += 1
                squared_errors.append(nearest ** 2)
    return {
        "coverage": (covered / total if total else None),
        "covered_pairs": covered,
        "offline_total_pairs": total,
        "rmse_m": math.sqrt(mean(squared_errors)) if squared_errors else None,
        "provenance": "offline_truth_evaluation_only",
    }


def _event(
    stage: str,
    time_s: Optional[float],
    mode: str,
    seed: int,
    **fields: Any,
) -> Dict[str, Any]:
    """构造纯诊断记录；不得把 truth payload 放入记录。"""
    return {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "stage": stage,
        "time_s": None if time_s is None else round(float(time_s), 6),
        "mode": mode,
        "environment_seed": int(seed),
        **fields,
    }


def _record_local_track_creations(
    centers: Mapping[str, Any],
    known_local_tracks: Dict[str, set],
    now_s: float,
    mode: str,
    seed: int,
    events: List[Dict[str, Any]],
) -> None:
    """由同一运行时的 local FusionCenter 前后快照派生创建事件。

    这里不读取 Scene，也不向 FusionCenter 写数据；因此默认关闭的旧路径没有新增
    副作用。local track 若在一个 tick 内创建又删除（当前 v1 未出现）会在报告中被
    标注为这种快照方法的限制。
    """
    for node_id, center in sorted(centers.items()):
        current = {str(track.track_id): track for track in center.tracks}
        previous = known_local_tracks.setdefault(node_id, set())
        for local_track_id in sorted(set(current) - previous):
            track = current[local_track_id]
            events.append(_event(
                "local_track_created", now_s, mode, seed,
                source_node_id=node_id,
                local_track_id=local_track_id,
                track_status=str(track.status),
                last_measurement_time_s=getattr(track, "last_measurement_time", None),
            ))
        previous.update(current)


def _track_messages(bus: Any) -> Dict[str, Any]:
    return {
        str(message.msg_id): message
        for message in bus.log
        if getattr(message, "kind", "") == MESSAGE_KIND_TRACK
    }


def _pipeline_events_from_runtime(
    manager: Any,
    bus: Any,
    final_now_s: float,
    mode: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """将既有 CommBus/manager audit 转为逐级漏斗事件，不改运行时决定。"""
    events: List[Dict[str, Any]] = []
    messages = _track_messages(bus)
    send_attempts = [row for row in manager.audit_log
                     if row.get("event") == "track_message_sent"]
    association_rows = [row for row in manager.audit_log
                        if row.get("event") == "association"]
    consumed_message_ids = {str(row.get("message_id", ""))
                            for row in association_rows}
    for row in send_attempts:
        message_id = str(row.get("message_id", ""))
        events.append(_event(
            "track_message_generated", row.get("time_s"), mode, seed,
            message_id=message_id,
            source_node_id=row.get("source_node_id", ""),
            local_track_id=row.get("local_track_id", ""),
            generated=True,
            send_attempt_success=bool(row.get("sent", False)),
            reason=row.get("reason", ""),
        ))
        if row.get("sent", False):
            message = messages.get(message_id)
            events.append(_event(
                "track_message_sent", row.get("time_s"), mode, seed,
                message_id=message_id,
                source_node_id=row.get("source_node_id", ""),
                local_track_id=row.get("local_track_id", ""),
                bytes=(None if message is None else float(message.size_bytes)),
                send_log_present=message is not None,
            ))

    for message in messages.values():
        if not message.dropped and message.arrived_at is not None:
            arrival_state = ("arrived" if message.arrived_at <= final_now_s + 1e-12
                             else "in_flight_at_end")
            events.append(_event(
                arrival_state, message.arrived_at, mode, seed,
                message_id=message.msg_id,
                source_node_id=message.source_node_id,
                local_track_id=message.local_track_id,
                latency_s=message.latency_s(),
                bytes=float(message.size_bytes),
            ))
            # 运行时只在下一次 begin_tick 消费已到达消息。最后一个 tick 内到达的
            # 报文若还没有下一次 begin_tick，必须显式计作“已到达但未进 gate”，不能
            # 混入链路丢失或关联拒绝。
            if (arrival_state == "arrived"
                    and str(message.msg_id) not in consumed_message_ids):
                events.append(_event(
                    "arrived_unconsumed_at_horizon", final_now_s, mode, seed,
                    message_id=message.msg_id,
                    source_node_id=message.source_node_id,
                    local_track_id=message.local_track_id,
                    reason="simulation_ended_before_next_begin_tick_ingest",
                ))

    for row in manager.audit_log:
        name = row.get("event")
        if name == "transport_rejected":
            events.append(_event(
                "transport_rejected", row.get("time_s"), mode, seed,
                message_id=row.get("message_id", ""),
                source_node_id=row.get("source_node_id", ""),
                local_track_id=row.get("local_track_id", ""),
                reason=row.get("reason", "transport_drop"),
            ))
        elif name == "association":
            decision = str(row.get("decision", ""))
            events.append(_event(
                "global_gate", row.get("time_s"), mode, seed,
                message_id=row.get("message_id", ""),
                source_node_id=row.get("source_node_id", ""),
                local_track_id=row.get("local_track_id", ""),
                global_track_id=row.get("global_track_id", ""),
                decision=decision,
                reason=row.get("reason", ""),
                candidate_global_count=len(row.get("candidate_global_ids", [])),
                rejected_candidate_count=len(row.get("rejected_candidates", [])),
            ))
            if decision == "accepted":
                events.append(_event(
                    "associated", row.get("time_s"), mode, seed,
                    message_id=row.get("message_id", ""),
                    source_node_id=row.get("source_node_id", ""),
                    local_track_id=row.get("local_track_id", ""),
                    global_track_id=row.get("global_track_id", ""),
                    association_reason=row.get("reason", ""),
                ))
            for rejected in row.get("rejected_candidates", []):
                events.append(_event(
                    "gate_candidate_rejected", row.get("time_s"), mode, seed,
                    message_id=row.get("message_id", ""),
                    global_track_id=rejected.get("global_track_id", ""),
                    distance_m=rejected.get("distance_m"),
                    reason=rejected.get("reason", ""),
                ))
        elif name == "ci_fused":
            sources = list(row.get("participating_source_nodes", []))
            events.append(_event(
                "ci_fused", row.get("time_s"), mode, seed,
                message_id=row.get("message_id", ""),
                source_node_id=row.get("source_node_id", ""),
                local_track_id=row.get("local_track_id", ""),
                global_track_id=row.get("global_track_id", ""),
                fusion_method=row.get("fusion_method", ""),
                participating_source_count=len(sources),
                participating_source_nodes=",".join(sources),
                numerical_guard_count=len(row.get("numerical_guards", [])),
            ))
    return events


def _global_lifecycle_rows(manager: Any, final_now_s: float,
                           mode: str, seed: int) -> List[Dict[str, Any]]:
    """根据 v1 的无真值 history/report 写出 global track 生命周期。"""
    appearances: Dict[str, int] = Counter()
    statuses: Dict[str, List[str]] = defaultdict(list)
    for snapshot in manager.history:
        for track in snapshot.get("tracks", []):
            global_id = str(track["global_track_id"])
            appearances[global_id] += 1
            statuses[global_id].append(str(track.get("status", "")))
    report = manager.report(final_now_s)
    rows: List[Dict[str, Any]] = []
    for track in report.get("tracks", []):
        global_id = str(track["global_track_id"])
        # v1 report 的 participating_source_nodes 是 manager 当前保留的每节点
        # estimate；其中旧来源可能已因 max_source_age 排除出本次 CI。因此同时
        # 记录“保留来源”和“当前 CI 有效来源”，避免把旧缓存写成正在融合。
        retained_source_nodes = list(track.get("participating_source_nodes", []))
        active_ci_source_nodes = sorted(track.get("fusion", {}).get("weights", {}))
        status = str(track.get("status", ""))
        rows.append({
            "protocol": DIAGNOSTIC_PROTOCOL,
            "mode": mode,
            "environment_seed": int(seed),
            "global_track_id": global_id,
            "created_at_s": float(track["created_at_s"]),
            "last_state_time_s": float(track["last_state_time_s"]),
            "information_age_s": float(track["information_age_s"]),
            "lifetime_s": max(0.0, final_now_s - float(track["created_at_s"])),
            "final_status": status,
            "updates": int(track.get("updates", 0)),
            "coasts": int(track.get("coasts", 0)),
            "retained_source_count": len(retained_source_nodes),
            "retained_source_nodes": ",".join(retained_source_nodes),
            "retained_source_class": (
                "multi_source" if len(retained_source_nodes) > 1 else "single_source"
            ),
            "active_ci_source_count": len(active_ci_source_nodes),
            "active_ci_source_nodes": ",".join(active_ci_source_nodes),
            "source_class": (
                "multi_source" if len(active_ci_source_nodes) > 1 else "single_source"
            ),
            "fusion_method": track.get("fusion", {}).get("method", ""),
            "snapshot_appearances": int(appearances.get(global_id, 0)),
            "status_history": ",".join(statuses.get(global_id, [])),
        })
    for track in report.get("dropped_tracks", []):
        global_id = str(track["global_track_id"])
        retained_source_nodes = list(track.get("retained_source_nodes", []))
        rows.append({
            "protocol": DIAGNOSTIC_PROTOCOL,
            "mode": mode,
            "environment_seed": int(seed),
            "global_track_id": global_id,
            "created_at_s": float(track["created_at_s"]),
            "last_state_time_s": float(track["last_state_time_s"]),
            "information_age_s": float(track["information_age_s"]),
            "lifetime_s": max(0.0, float(track["dropped_at_s"])
                              - float(track["created_at_s"])),
            "final_status": "dropped",
            "updates": int(track.get("updates", 0)),
            "coasts": int(track.get("coasts", 0)),
            "retained_source_count": len(retained_source_nodes),
            "retained_source_nodes": ",".join(retained_source_nodes),
            "retained_source_class": (
                "multi_source" if len(retained_source_nodes) > 1 else "single_source"
            ),
            "active_ci_source_count": 0,
            "active_ci_source_nodes": "",
            "source_class": "single_source",
            "fusion_method": track.get("fusion_method", ""),
            "snapshot_appearances": int(appearances.get(global_id, 0)),
            "status_history": ",".join(statuses.get(global_id, []) + ["dropped"]),
        })
    return rows


def _reason_counts(events: Iterable[Mapping[str, Any]], stage: str) -> Dict[str, int]:
    return dict(sorted(Counter(
        str(row.get("reason", "unspecified"))
        for row in events if row.get("stage") == stage
    ).items()))


def _safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _funnel_summary(events: Sequence[Mapping[str, Any]],
                    lifecycle: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """同时保留 unique local-track 与逐消息两个口径，防止重复上报伪造转化率。"""
    def source_key(row: Mapping[str, Any]) -> Tuple[str, str]:
        return str(row.get("source_node_id", "")), str(row.get("local_track_id", ""))

    local_created = {source_key(row) for row in events
                     if row.get("stage") == "local_track_created"}
    generated = [row for row in events if row.get("stage") == "track_message_generated"]
    sent = [row for row in events if row.get("stage") == "track_message_sent"]
    arrived = [row for row in events if row.get("stage") == "arrived"]
    arrived_unconsumed = [row for row in events
                          if row.get("stage") == "arrived_unconsumed_at_horizon"]
    gated = [row for row in events if row.get("stage") == "global_gate"]
    associated = [row for row in events if row.get("stage") == "associated"]
    fused = [row for row in events if row.get("stage") == "ci_fused"]
    generated_local = {source_key(row) for row in generated}
    sent_local = {source_key(row) for row in sent}
    arrived_local = {source_key(row) for row in arrived}
    associated_local = {source_key(row) for row in associated}
    fused_local = {source_key(row) for row in fused}
    multi_final = sum(1 for row in lifecycle if row.get("source_class") == "multi_source")
    single_final = sum(1 for row in lifecycle if row.get("source_class") == "single_source")
    retained_multi_final = sum(
        1 for row in lifecycle if row.get("retained_source_class") == "multi_source"
    )
    dropped_final = sum(1 for row in lifecycle if row.get("final_status") == "dropped")
    maintained_final = len(lifecycle) - dropped_final
    return {
        "unique_local_track_funnel": {
            "local_track_created": len(local_created),
            "track_message_generated": len(generated_local),
            "track_message_sent": len(sent_local),
            "track_message_arrived": len(arrived_local),
            "associated": len(associated_local),
            "ci_fused": len(fused_local),
            "local_to_generated_rate": _safe_rate(len(generated_local), len(local_created)),
            "generated_to_sent_rate": _safe_rate(len(sent_local), len(generated_local)),
            "sent_to_arrived_rate": _safe_rate(len(arrived_local), len(sent_local)),
            "arrived_to_associated_rate": _safe_rate(len(associated_local), len(arrived_local)),
            "associated_to_ci_fused_rate": _safe_rate(len(fused_local), len(associated_local)),
        },
        "message_event_funnel": {
            "generated_attempts": len(generated),
            "sent": len(sent),
            "arrived": len(arrived),
            "arrived_unconsumed_at_horizon": len(arrived_unconsumed),
            "global_gate_decisions": len(gated),
            "associated": len(associated),
            "ci_fused": len(fused),
            "send_rate": _safe_rate(len(sent), len(generated)),
            "arrival_rate": _safe_rate(len(arrived), len(sent)),
            "arrived_to_gate_decision_rate": _safe_rate(len(gated), len(arrived)),
            "gate_acceptance_rate": _safe_rate(len(associated), len(gated)),
            "ci_fusion_rate": _safe_rate(len(fused), len(associated)),
            "message_utilization_rate": _safe_rate(len(associated), len(sent)),
        },
        "global_track_lifecycle": {
            "final_global_tracks": len(lifecycle),
            "maintained_final": maintained_final,
            "dropped_final": dropped_final,
            "single_source_final": single_final,
            "multi_source_final": multi_final,
            "single_source_ratio": _safe_rate(single_final, len(lifecycle)),
            "multi_source_ratio": _safe_rate(multi_final, len(lifecycle)),
            "retained_multi_source_ratio": _safe_rate(retained_multi_final, len(lifecycle)),
            "mean_lifetime_s": (mean(float(row["lifetime_s"]) for row in lifecycle)
                                if lifecycle else None),
            "mean_updates": (mean(int(row["updates"]) for row in lifecycle)
                             if lifecycle else None),
            "mean_coasts": (mean(int(row["coasts"]) for row in lifecycle)
                             if lifecycle else None),
        },
        "rejection_reasons": {
            "generation_or_route": _reason_counts(events, "track_message_generated"),
            "transport": _reason_counts(events, "transport_rejected"),
            "global_gate": _reason_counts(
                [row for row in events
                 if row.get("stage") == "global_gate"
                 and row.get("decision") == "rejected"],
                "global_gate",
            ),
            "gate_candidates": _reason_counts(events, "gate_candidate_rejected"),
        },
    }


def _first_bottleneck(funnel: Mapping[str, Any], offline: Mapping[str, Any]) -> Dict[str, Any]:
    """可复核的规则解释：按漏斗损失定位，不把离线 truth 写回运行时。"""
    unique = funnel["unique_local_track_funnel"]
    message = funnel["message_event_funnel"]
    candidates = [
        ("local_track_to_track_message", unique["local_to_generated_rate"],
         "已形成的 local track 没有获得 TrackMessage 生成/上报机会"),
        ("message_generation_to_send", unique["generated_to_sent_rate"],
         "生成的 TrackMessage 未能进入真实 CommBus 链路"),
        ("transport_send_to_arrival", unique["sent_to_arrived_rate"],
         "已发送 TrackMessage 在链路延迟、丢包、过期或队列中损失"),
        ("arrival_to_global_gate", message["arrived_to_gate_decision_rate"],
         "报文虽已抵达，但在仿真结束前没有进入下一次 begin_tick 的 global gate"),
        ("global_gate_to_association", message["gate_acceptance_rate"],
         "已进入 global gate 的消息被序号或关联路径拒绝，未进入 global track"),
        ("association_to_ci", unique["associated_to_ci_fused_rate"],
         "已关联消息没有留下 CI 融合审计（应调查管理器日志）"),
    ]
    usable = [(name, float(rate), text) for name, rate, text in candidates
              if rate is not None]
    if not usable:
        return {
            "stage": "no_track_pipeline_activity",
            "evidence": "未形成可诊断的 TrackMessage 链；先检查调度是否实际执行 share。",
            "offline_coverage": offline.get("coverage"),
        }
    stage, rate, text = min(usable, key=lambda item: item[1])
    coverage = offline.get("coverage")
    if rate >= 0.95 and coverage is not None and float(coverage) < 0.5:
        return {
            "stage": "post_pipeline_estimation_or_temporal_scope",
            "evidence": (
                "local→消息→到达→关联→CI 漏斗没有显著损失，但离线 coverage 仍低；"
                "当前证据不支持把问题归因于 CommBus、门控拒绝或 CI 未执行。应仅作为"
                "local/global 状态质量、更新时序和本开发场景覆盖范围的待诊断限制。"
            ),
            "lowest_funnel_rate": rate,
            "offline_coverage": coverage,
            "message_utilization_rate": message["message_utilization_rate"],
        }
    return {
        "stage": stage,
        "evidence": text,
        "lowest_funnel_rate": rate,
        "offline_coverage": coverage,
        "message_utilization_rate": message["message_utilization_rate"],
    }


def diagnose_one(mode: str, seed: int, steps: int = DEFAULT_STEPS) -> Dict[str, Any]:
    """运行一条固定 Rule 真闭环，并在外层收集无真值漏斗证据。"""
    world = _build_feedback_world(
        seed=int(seed), steps=int(steps), policy=SchedulerPolicy.RULE,
        task_gating="loop_gate", global_track_mode=GLOBAL_TRACK_MODE_TRACK_FUSION,
        global_share_mode=mode,
    )
    driver = world["driver"]
    # 复用 development evaluator 的固定 Rule；不替换世界、任务队列、执行器或资源模型。
    world["planner"] = build_scheduler(SchedulerPolicy.RULE)
    driver.scheduler = world["planner"]

    known_local_tracks: Dict[str, set] = {
        node_id: set() for node_id in world["runtime"].centers
    }
    local_events: List[Dict[str, Any]] = []
    truth_history: Dict[float, Dict[str, Tuple[float, float, float]]] = {}
    for _ in range(int(steps)):
        driver.tick()
        now_s = float(world["clock"].now_s)
        _record_local_track_creations(
            world["runtime"].centers, known_local_tracks, now_s,
            mode, seed, local_events,
        )
        # 只供离线 coverage；该字典不会传给 Manager/Bus/Observation/调度器。
        truth_history[round(now_s, 6)] = _offline_truth_snapshot(world)

    result = driver.finalize()
    manager = world["global_track_manager"]
    final_now_s = float(world["clock"].now_s)
    runtime_events = _pipeline_events_from_runtime(
        manager, world["bus"], final_now_s, mode, seed,
    )
    lifecycle = _global_lifecycle_rows(manager, final_now_s, mode, seed)
    lifecycle_events = [
        _event(
            ("global_track_dropped" if row["final_status"] == "dropped"
             else "global_track_maintained"),
            final_now_s, mode, seed,
            global_track_id=row["global_track_id"],
            final_status=row["final_status"],
            lifetime_s=row["lifetime_s"],
            source_class=row["source_class"],
        )
        for row in lifecycle
    ]
    events = local_events + runtime_events + lifecycle_events
    offline = _offline_coverage(manager.history, truth_history)
    funnel = _funnel_summary(events, lifecycle)
    diagnosis = _first_bottleneck(funnel, offline)
    runtime = result.metrics.get("runtime_feedback", {})
    return {
        "mode": mode,
        "environment_seed": int(seed),
        "steps": int(steps),
        "pipeline_funnel": funnel,
        "coverage_diagnosis": diagnosis,
        "offline_truth_evaluation": offline,
        "runtime_guards": {
            "resource_conserved": bool(result.conservation.get("all_conserved", False)),
            "runtime_truth_payload_violations": int(
                runtime.get("truth_payload_violations", 0)),
        },
        "events": events,
        "global_track_lifecycles": lifecycle,
    }


def _flatten_summary(run: Mapping[str, Any]) -> Dict[str, Any]:
    unique = run["pipeline_funnel"]["unique_local_track_funnel"]
    message = run["pipeline_funnel"]["message_event_funnel"]
    lifecycle = run["pipeline_funnel"]["global_track_lifecycle"]
    offline = run["offline_truth_evaluation"]
    diagnosis = run["coverage_diagnosis"]
    return {
        "mode": run["mode"],
        "environment_seed": run["environment_seed"],
        "steps": run["steps"],
        "local_tracks_created": unique["local_track_created"],
        "unique_tracks_generated": unique["track_message_generated"],
        "unique_tracks_sent": unique["track_message_sent"],
        "unique_tracks_arrived": unique["track_message_arrived"],
        "unique_tracks_associated": unique["associated"],
        "unique_tracks_ci_fused": unique["ci_fused"],
        "generated_attempts": message["generated_attempts"],
        "messages_sent": message["sent"],
        "messages_arrived": message["arrived"],
        "messages_arrived_unconsumed_at_horizon": message[
            "arrived_unconsumed_at_horizon"],
        "global_gate_decisions": message["global_gate_decisions"],
        "messages_associated": message["associated"],
        "ci_fusion_events": message["ci_fused"],
        "message_utilization_rate": message["message_utilization_rate"],
        "final_global_tracks": lifecycle["final_global_tracks"],
        "single_source_ratio": lifecycle["single_source_ratio"],
        "multi_source_ratio": lifecycle["multi_source_ratio"],
        "retained_multi_source_ratio": lifecycle["retained_multi_source_ratio"],
        "coverage": offline["coverage"],
        "rmse_m": offline["rmse_m"],
        "first_bottleneck": diagnosis["stage"],
        "resource_conserved": run["runtime_guards"]["resource_conserved"],
        "runtime_truth_payload_violations": run["runtime_guards"][
            "runtime_truth_payload_violations"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Global Track Pipeline Diagnosis",
        "",
        f"协议：`{report['protocol']}`；固定 Rule development runs，未修改算法参数、门限、"
        "资源调度或 PPO。",
        "",
        "## 漏斗摘要",
        "",
        "`single / multi` 统计最终时刻实际进入 CI 权重的活跃来源；另在 CSV/JSON 保留"
        "`retained_multi_source_ratio`，它只表示 manager 尚缓存过多个节点来源，不等于仍在融合。",
        "",
        "| mode | seed | local created | generated / sent / arrived | associated / CI | final global | single / multi | coverage | first bottleneck |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            "| {mode} | {environment_seed} | {local_tracks_created} | "
            "{generated_attempts} / {messages_sent} / {messages_arrived} | "
            "{messages_associated} / {ci_fusion_events} | {final_global_tracks} | "
            "{single_source_ratio} / {multi_source_ratio} | {coverage} | {first_bottleneck} |".format(
                **{key: ("—" if value is None else value) for key, value in row.items()}
            )
        )
    lines.extend([
        "",
        "## 每格解释",
        "",
    ])
    for run in report["runs"]:
        diagnosis = run["coverage_diagnosis"]
        funnel = run["pipeline_funnel"]
        lines.extend([
            f"### `{run['mode']}` / seed `{run['environment_seed']}`",
            "",
            f"- 首个瓶颈：`{diagnosis['stage']}`。{diagnosis['evidence']}",
            f"- 消息利用率：`{funnel['message_event_funnel']['message_utilization_rate']}`；"
            f"到达但因 horizon 未进 gate：`{funnel['message_event_funnel']['arrived_unconsumed_at_horizon']}`；"
            f"关联拒绝：`{funnel['rejection_reasons']['global_gate']}`；"
            f"传输拒绝：`{funnel['rejection_reasons']['transport']}`。",
            f"- 生命周期：`{funnel['global_track_lifecycle']}`。v1 当前只有 maintained/coasting/"
            "stale_coasting 状态；没有自动删除 global track 的策略时，`dropped_final=0` 是"
            "实现事实，不应误写成成功率。",
            "",
        ])
    lines.extend([
        "## 数据边界",
        "",
        "运行时 events/lifecycle CSV 只来自 local FusionCenter 快照、RuntimeExecutor、CommBus "
        "和 GlobalTrackManager。真值仅在外层离线 coverage/RMSE 聚合中短暂读取，报告不写出"
        "truth 位置或 ID；没有任何真值进入 TrackMessage、GlobalTrackManager、GlobalObservation 或调度器。",
        "",
        "## 产物",
        "",
        "- `global_track_pipeline_events.csv`：逐级运行时事件。",
        "- `global_track_pipeline_summary.csv`：每 mode × seed 的可比较漏斗。",
        "- `global_track_pipeline_lifecycle.csv`：global track 生命周期。",
        "- `global_track_pipeline.json`：完整结构化诊断。",
    ])
    return "\n".join(lines) + "\n"


def run_pipeline_diagnosis(
    output_dir: Path,
    seeds: Sequence[int] = DEVELOPMENT_SEEDS,
    steps: int = DEFAULT_STEPS,
    modes: Sequence[str] = DIAGNOSTIC_MODES,
) -> Dict[str, Any]:
    """写出 CSV/JSON/MD；仅运行固定 Rule development 闭环。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [diagnose_one(mode, int(seed), int(steps))
            for mode in modes for seed in seeds]
    summaries = [_flatten_summary(run) for run in runs]
    event_rows = [event for run in runs for event in run["events"]]
    lifecycle_rows = [row for run in runs for row in run["global_track_lifecycles"]]
    report = {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "scope": "fixed_rule_development_diagnosis",
        "modes": list(modes),
        "seeds": [int(seed) for seed in seeds],
        "steps": int(steps),
        "offline_coverage_gate_m": OFFLINE_COVERAGE_GATE_M,
        "honesty_boundaries": [
            "No algorithm parameter, CI weight grid, gate threshold, reward, PPO, or training change.",
            "Runtime diagnostics are derived solely from local-track snapshots, RuntimeExecutor, CommBus, and GlobalTrackManager.",
            "Truth is accessed only in an outer offline coverage/RMSE aggregation and is never written to TrackMessage or GlobalTrackManager.",
            "A low coverage value is diagnosed as a funnel observation, not an algorithm-comparison claim.",
        ],
        "runs": runs,
    }
    _write_csv(output_dir / "global_track_pipeline_events.csv", event_rows)
    _write_csv(output_dir / "global_track_pipeline_summary.csv", summaries)
    _write_csv(output_dir / "global_track_pipeline_lifecycle.csv", lifecycle_rows)
    (output_dir / "global_track_pipeline.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "global_track_pipeline.md").write_text(
        _markdown(report, summaries), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("output/global_track_pipeline_diagnosis"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEVELOPMENT_SEEDS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--modes", nargs="+", default=list(DIAGNOSTIC_MODES))
    args = parser.parse_args()
    report = run_pipeline_diagnosis(args.out_dir, args.seeds, args.steps, args.modes)
    print(json.dumps([_flatten_summary(run) for run in report["runs"]],
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
