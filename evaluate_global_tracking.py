"""固定 development seeds 的 Global Track / 塔台层统一对照评测。

只比较可解释通信基线：no_share / measurement_share / track_share /
event_triggered_track_share。不会训练 PPO、改变奖励，或读取任何封存测试集。
真值仅在本脚本的离线指标计算中读取，绝不传给 GlobalTrackManager、
GlobalObservation 或资源调度器。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from global_fusion import (
    GLOBAL_SHARE_MODE_EVENT_TRACK,
    GLOBAL_SHARE_MODE_MEASUREMENT,
    GLOBAL_SHARE_MODE_NO_SHARE,
    GLOBAL_SHARE_MODE_TRACK,
    GLOBAL_TRACK_MODE_TRACK_FUSION,
)
from resource_management.closed_loop import (
    RUNTIME_MODE_FEEDBACK,
    _build_feedback_world,
)
from resource_management.scheduling import SchedulerPolicy, build_scheduler


DEVELOPMENT_SEEDS: Tuple[int, ...] = (41, 73, 109)
COMMUNICATION_MODES: Tuple[str, ...] = (
    GLOBAL_SHARE_MODE_NO_SHARE,
    GLOBAL_SHARE_MODE_MEASUREMENT,
    GLOBAL_SHARE_MODE_TRACK,
    GLOBAL_SHARE_MODE_EVENT_TRACK,
)
EVALUATION_ASSOCIATION_GATE_M = 1_500.0
DEFAULT_STEPS = 18


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _truth_snapshot(world: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    """仅离线评测读取；不写入 manager/observation/调度器。"""
    return {
        target.target_id: (target.position.x, target.position.y, target.position.z)
        for target in world["sim"].scene.targets
    }


def _offline_global_metrics(
    history: Iterable[Dict[str, Any]],
    truth_history: Dict[float, Dict[str, Tuple[float, float, float]]],
) -> Dict[str, Any]:
    """将无真值 global snapshot 与外层真值历史对齐，计算报告指标。"""
    covered_pairs = total_pairs = 0
    squared_errors: List[float] = []
    covariance_traces: List[float] = []
    information_ages: List[float] = []
    assignments: Dict[str, List[Optional[str]]] = {}
    duplicate_counts: List[int] = []
    for snapshot in history:
        now_s = round(float(snapshot["time_s"]), 6)
        truths = truth_history.get(now_s, {})
        tracks = list(snapshot.get("tracks", []))
        for track in tracks:
            covariance_traces.append(sum(float(value)
                                         for value in track["covariance_position_m2"]))
            information_ages.append(float(track["information_age_s"]))
        for truth_id, truth_position in truths.items():
            total_pairs += 1
            candidates = sorted(
                (_distance(track["position_m"], truth_position), track)
                for track in tracks
            )
            within_gate = [item for item in candidates
                           if item[0] <= EVALUATION_ASSOCIATION_GATE_M]
            duplicate_counts.append(max(0, len(within_gate) - 1))
            if not within_gate:
                assignments.setdefault(truth_id, []).append(None)
                continue
            distance, selected = within_gate[0]
            covered_pairs += 1
            squared_errors.append(distance ** 2)
            assignments.setdefault(truth_id, []).append(selected["global_track_id"])

    id_switches = 0
    fragmentation = 0
    for series in assignments.values():
        observed = [value for value in series if value is not None]
        fragmentation += max(0, len(set(observed)) - 1)
        previous: Optional[str] = None
        for value in series:
            if value is not None and previous is not None and value != previous:
                id_switches += 1
            if value is not None:
                previous = value
    return {
        "global_track_coverage": (covered_pairs / total_pairs if total_pairs else 0.0),
        "global_id_switches": id_switches,
        "fragmentation": fragmentation,
        "duplicate_global_tracks_mean": (mean(duplicate_counts)
                                         if duplicate_counts else 0.0),
        "rmse_m": (math.sqrt(mean(squared_errors)) if squared_errors else None),
        "mean_covariance_trace_m2": (mean(covariance_traces)
                                      if covariance_traces else None),
        "mean_information_age_s": (mean(information_ages)
                                   if information_ages else None),
        "offline_truth_provenance": "evaluation_only_nearest_global_track",
    }


def evaluate_one(mode: str, seed: int, steps: int = DEFAULT_STEPS) -> Dict[str, Any]:
    """运行一格固定 development 评测；策略始终是 Rule，不训练任何模型。"""
    start = time.perf_counter()
    world = _build_feedback_world(
        seed=int(seed), steps=int(steps), policy=SchedulerPolicy.RULE,
        task_gating="loop_gate", global_track_mode=GLOBAL_TRACK_MODE_TRACK_FUSION,
        global_share_mode=mode,
    )
    driver = world["driver"]
    # `_build_feedback_world` 服务学习环境，默认放置禁止自行规划的占位器；
    # 本报告明确固定 Rule 基线，因此替换为同配置的真实 RuleScheduler，其他
    # 世界/队列/执行器/RuntimeExecutor 完全不变。
    world["planner"] = build_scheduler(SchedulerPolicy.RULE)
    driver.scheduler = world["planner"]
    truth_history: Dict[float, Dict[str, Tuple[float, float, float]]] = {}
    for _ in range(int(steps)):
        driver.tick()
        truth_history[round(float(world["clock"].now_s), 6)] = _truth_snapshot(world)
    result = driver.finalize()
    wall_clock_s = time.perf_counter() - start
    manager = world["global_track_manager"]
    offline = _offline_global_metrics(manager.history, truth_history)
    sent_track = sum(1 for message in world["bus"].log
                     if getattr(message, "kind", "") == "track")
    accepted_track = sum(1 for row in manager.audit_log
                         if row.get("event") == "association"
                         and row.get("decision") == "accepted")
    runtime = result.metrics.get("runtime_feedback", {})
    return {
        "mode": mode,
        "environment_seed": int(seed),
        "steps": int(steps),
        **offline,
        "communication_bytes": float(result.metrics.get("comm_overhead_bytes", 0.0)),
        "n_track_messages_sent": sent_track,
        "n_track_messages_accepted": accepted_track,
        "message_utilization_rate": (accepted_track / sent_track if sent_track else None),
        "planning_time_s": float(result.planning_time_s),
        "wall_clock_run_s": wall_clock_s,
        "resource_conserved": bool(result.conservation.get("all_conserved", False)),
        "runtime_truth_payload_violations": int(runtime.get("truth_payload_violations", 0)),
        "global_track_count_final": int((result.global_track_report or {}).get(
            "n_global_tracks", 0)),
    }


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    return mean(usable) if usable else None


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = (
        "global_track_coverage", "global_id_switches", "fragmentation",
        "duplicate_global_tracks_mean", "rmse_m", "mean_covariance_trace_m2",
        "mean_information_age_s", "communication_bytes", "planning_time_s",
        "wall_clock_run_s", "message_utilization_rate",
    )
    summary: List[Dict[str, Any]] = []
    for mode in COMMUNICATION_MODES:
        members = [row for row in rows if row["mode"] == mode]
        item: Dict[str, Any] = {"mode": mode, "n_development_seeds": len(members)}
        for key in keys:
            item[f"mean_{key}"] = _mean_or_none([row.get(key) for row in members])
        item["all_resource_conserved"] = all(row["resource_conserved"] for row in members)
        item["all_truth_payload_clean"] = all(
            row["runtime_truth_payload_violations"] == 0 for row in members)
        summary.append(item)
    return summary


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, summary: Sequence[Dict[str, Any]]) -> None:
    fields = sorted({key for row in summary for key in row})
    head = "".join(f"<th>{html.escape(key)}</th>" for key in fields)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in fields
        ) + "</tr>" for row in summary
    )
    path.write_text(
        "<html><meta charset='utf-8'><title>Global Tracking Development Report</title>"
        "<body><h1>Global Tracking Development Report</h1>"
        "<p>Fixed development seeds only; no RL training, no aggregate score, "
        "and offline truth is used solely for evaluation.</p>"
        f"<table border='1'><tr>{head}</tr>{body}</table></body></html>",
        encoding="utf-8",
    )


def run_development_evaluation(
    output_dir: Path,
    seeds: Sequence[int] = DEVELOPMENT_SEEDS,
    steps: int = DEFAULT_STEPS,
) -> Dict[str, Any]:
    """写入 CSV/JSON/HTML；固定输入不会读取训练或封存测试分区。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_one(mode, seed, steps)
            for mode in COMMUNICATION_MODES for seed in seeds]
    report = {
        "protocol": "global-tracking-development-v1",
        "scope": "fixed_development_seeds_only",
        "modes": list(COMMUNICATION_MODES),
        "seeds": list(seeds),
        "steps": int(steps),
        "evaluation_association_gate_m": EVALUATION_ASSOCIATION_GATE_M,
        "honesty_boundaries": [
            "No PPO training, reward modification, JPDA, MARL, or joint power control.",
            "No single composite score or statistical-significance claim.",
            "Truth is accessed only by this offline evaluator, never by global fusion or scheduling.",
        ],
        "rows": rows,
        "summary": summarize(rows),
    }
    _write_csv(output_dir / "global_tracking_development.csv", rows)
    (output_dir / "global_tracking_development.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_html(output_dir / "global_tracking_development.html", report["summary"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("output/global_tracking_development"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEVELOPMENT_SEEDS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    report = run_development_evaluation(args.out_dir, args.seeds, args.steps)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
