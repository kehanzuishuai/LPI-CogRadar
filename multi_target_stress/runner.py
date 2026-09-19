"""压力场景运行器（v4.5 P2：4 核心关联场景 + 4 系统级场景）。

一条运行链路
------------
```
场景几何 / 机动时刻表 / 偏差注入 / 通信时序（scenarios + system_scenarios）
  → 传感器套件（sensor/，只建本地传感器时为"单雷达"路）
  → 每步：本地测量 = detections + held + **虚警**
          远端测量 = 远端 detections（**不含虚警**）→ CommBus → 已到达的才可读
  → **OOSM 控制器**（drop_stale / reorder_buffer / delayed_update）
  → FusionCenter（NN + 卡尔曼，**基线未改**）→ 航迹 + 关联层审计
  → 帧记录（航迹 / 真值 / 关联决策 / 逐来源残差 / 时序决策）
  → 关联层指标 + 系统级指标（机动 / 交接 / 时序 / 偏差）
```

真值隔离
--------
`measurement_truth`（测量 → 真值目标）只在**帧记录**里构造，供离线指标使用；
它**不进** `FusionCenter`，也不进 AI 上下文。
跟踪器拿到的是 `MeasurementRecord` / `SharedMeasurement`，
后者的 `truth_id` 恒为 None。
机动注入只写**真值实体**；偏差只加在**传感器测量**上——两者对算法都不可见。

对照轴
------
* 核心场景（S1–S4）与 S5/S6/S8：沿**共享策略**对照
  （S8 把最后一档换成 `biased_share`）。
* S7：沿 **OOSM 处理策略**对照，共享策略固定为受限共享。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment_config as ec
from communication import (
    SHARE_CONSTRAINED,
    SHARE_IDEAL,
    SHARE_NONE,
    SHARE_POLICY_CN,
    CommBus,
    CommConfig,
)
from fusion import FusionCenter, FusionConfig
from fusion.lifecycle import STAGE_GATE_REJECTED, LifecycleLog
from multi_target_stress.maneuvers import ManeuverInjector
from multi_target_stress.metrics import MultiTargetMetrics, assign_tracks_to_truth
from multi_target_stress.scenarios import BASE_CONFIG_PATH, StressScenario
from multi_target_stress.system_metrics import (
    BiasMetrics,
    HandoverMetrics,
    ManeuverMetrics,
    TimingMetrics,
)
from multi_target_stress.timing import (
    DROP_STALE,
    OOSM_POLICIES,
    OOSM_POLICY_CN,
    OosmController,
)
from sensor.config import build_suite_from_config

#: 核心四路对照（S1–S6）
STRESS_POLICIES: Tuple[str, ...] = ("single", SHARE_NONE, SHARE_IDEAL,
                                    SHARE_CONSTRAINED)

#: S8 的四组：最后一档换成"有偏传感器共享"
BIAS_POLICIES: Tuple[str, ...] = ("single", SHARE_NONE, SHARE_IDEAL,
                                  "biased_share")

#: 受限共享链路参数（与 `evaluate_cooperative_sensing.CONSTRAINED_COMM` 一致）
CONSTRAINED_COMM = {"base_delay_s": 1.2, "jitter_s": 0.4, "loss_prob": 0.25,
                    "expiry_s": 2.5}

LOCAL_ENDPOINT = "LOCAL"
REMOTE_ENDPOINT = "REMOTE"

#: 固定功率档（索引 6 = 18 W），不引入控制策略差异
FIXED_POWER_ACTION = 6


def resolve_scenario(scenario_id: str) -> StressScenario:
    """按 ID 取场景（核心 S1–S4 与系统级 S5–S8 都支持）。"""
    from multi_target_stress.scenarios import SCENARIO_IDS, get_scenario
    from multi_target_stress.system_scenarios import (
        SYSTEM_SCENARIO_IDS,
        get_system_scenario,
    )

    if scenario_id in SCENARIO_IDS:
        return get_scenario(scenario_id)
    if scenario_id in SYSTEM_SCENARIO_IDS:
        return get_system_scenario(scenario_id)
    raise ValueError(
        f"未知场景 {scenario_id!r}；核心 {list(SCENARIO_IDS)}，"
        f"系统级 {list(SYSTEM_SCENARIO_IDS)}"
    )


class SharedMeasurement:
    """从通信消息解包出的测量对象（**不含真值**，供融合中心消费）。"""

    __slots__ = ("sensor_id", "candidate_id", "sensor_kind", "time_s", "range_m",
                 "azimuth_deg", "elevation_deg", "range_rate_mps", "std_range_m",
                 "std_az_deg", "std_el_deg", "confidence", "msg_id", "platform_id",
                 "is_false_alarm", "truth_id", "covariance")

    def __init__(self, payload: Dict[str, Any], msg_id: str = "",
                 platform_id: str = "") -> None:
        for key in ("sensor_id", "candidate_id", "sensor_kind", "time_s", "range_m",
                    "azimuth_deg", "elevation_deg", "range_rate_mps", "std_range_m",
                    "std_az_deg", "std_el_deg", "confidence"):
            setattr(self, key, payload.get(key))
        self.msg_id = msg_id
        self.platform_id = platform_id
        # 共享测量不含真值：这两个字段恒为 None（接口一致性保留）
        self.is_false_alarm = False
        self.truth_id = None
        #: 由载荷里的对角项还原 3×3 协方差，供"延迟更新"按需放大
        cov = [payload.get("cov_xx"), payload.get("cov_yy"), payload.get("cov_zz")]
        self.covariance = (
            [[float(cov[0]), 0.0, 0.0],
             [0.0, float(cov[1]), 0.0],
             [0.0, 0.0, float(cov[2])]]
            if all(isinstance(v, (int, float)) for v in cov) else None
        )


def _sensor_configs(scenario: StressScenario, variant: str,
                    drop_remote: bool) -> Dict[str, Any]:
    """构造传感器配置；只有 S8 的 `biased_share` 变体才会注入偏差。"""
    raw: Dict[str, Any] = {
        "sensors": [dict(item) for item in scenario.overrides.get("sensors", [])],
        "occluders": list(scenario.overrides.get("occluders", [])),
    }
    if scenario.bias_overrides and variant == scenario.bias_variant:
        for item in raw["sensors"]:
            overrides = scenario.bias_overrides.get(item.get("sensor_id"))
            if overrides:
                item.update(overrides)
    if drop_remote:
        raw["sensors"] = [s for s in raw["sensors"]
                          if s.get("sensor_id") != scenario.remote_sensor_id]
    return raw


def _build_bus(scenario: StressScenario, policy: str, seed: int
               ) -> Optional[CommBus]:
    """按场景与共享路构造通信总线（含 S7 的时序压力项）。"""
    if policy == "single":
        return None
    extras: Dict[str, Any] = {}
    if policy in (SHARE_CONSTRAINED, "biased_share"):
        extras.update(CONSTRAINED_COMM)
        if scenario.comm_extras:
            extras.update(scenario.comm_extras)
    config = CommConfig(
        policy=(SHARE_IDEAL if policy == "biased_share" else policy),
        message_size_bytes=128.0, seed=seed, **extras,
    )
    return CommBus([LOCAL_ENDPOINT, REMOTE_ENDPOINT], config)


def run_case(
    scenario_id: str,
    variant: Optional[str] = None,
    seed: int = 42,
    steps: Optional[int] = None,
    scenario: Optional[StressScenario] = None,
    policy: Optional[str] = None,
) -> Dict[str, Any]:
    """跑一个 (场景, 变体, 种子) 组合。

    `variant` 的含义由 `scenario.axis` 决定：`"policy"` → 共享策略；
    `"oosm"` → 乱序测量处理策略。`policy=` 为旧调用方式保留，等价于 `variant`。
    """
    scenario = scenario or resolve_scenario(scenario_id)
    if steps is None:
        steps = scenario.steps
    chosen = policy if policy is not None else variant
    if chosen is None:
        chosen = scenario.variants[0]

    if scenario.axis == "oosm":
        if chosen not in OOSM_POLICIES:
            raise ValueError(
                f"场景 {scenario_id} 的对照轴是 OOSM 策略，取值只能是 "
                f"{list(OOSM_POLICIES)}，收到 {chosen!r}"
            )
        oosm_policy = chosen
        share_policy = scenario.fixed_policy
    else:
        if chosen not in tuple(STRESS_POLICIES) + ("biased_share",):
            raise ValueError(f"未知共享路 {chosen!r}")
        oosm_policy = DROP_STALE
        share_policy = chosen

    env = ec.make_env_for_seed(
        seed, config_path=BASE_CONFIG_PATH, jitter=False,
        observation_mode="realistic", measurement_max_tracks=4,
    )
    env.sim.apply_overrides(extra=scenario.overrides)
    # ⚠️ 套件在构造 env 时按**原配置**建好（传感器 mounting_id 指向 RADAR1），
    # 换掉雷达或注入偏差后必须重建，否则要么找不到平台，要么偏差不生效。
    raw_config = _sensor_configs(scenario, chosen, share_policy == "single")
    env.suite = build_suite_from_config(
        env.sim.scene, raw_config, seed=seed, noise_scale=1.0
    )
    env.reset(seed=seed)
    sim = env.sim
    if env.suite is None:
        raise RuntimeError("realistic 模式应建立传感器套件")

    radar_sensors = [s for s in env.suite.sensors if s.sensor_kind == "radar"]
    local_sensor = next(
        (s for s in radar_sensors if s.sensor_id == scenario.local_sensor_id), None
    )
    if local_sensor is None:
        raise RuntimeError(f"场景 {scenario_id} 缺少本地传感器 "
                           f"{scenario.local_sensor_id}")
    remote_sensors = ([] if share_policy == "single" else
                      [s for s in radar_sensors
                       if s.sensor_id == scenario.remote_sensor_id])
    own_ids = {local_sensor.sensor_id}
    remote_ids = {s.sensor_id for s in remote_sensors}

    sensor_positions = {
        s.sensor_id: env.sim.scene.by_id(s.config.mounting_id).position
        for s in env.suite.sensors
    }

    bus = _build_bus(scenario, share_policy, seed)
    if bus is not None and not remote_ids:
        bus = None

    lifecycle = LifecycleLog(enabled=True)
    center = FusionCenter("LOCAL", FusionConfig(), own_sensor_ids=own_ids,
                          lifecycle=lifecycle)
    core_metrics = MultiTargetMetrics()
    oosm = OosmController(
        policy=oosm_policy,
        window_s=float((scenario.comm_extras or {}).get("oosm_window_s", 2.0)),
    )
    injector = ManeuverInjector(scenario.maneuvers or ())

    frames: List[Dict[str, Any]] = []
    steps_done = 0
    while steps_done < steps:
        _obs, _reward, terminated, truncated, _info = env.step(FIXED_POWER_ACTION)
        now = sim.current_time

        # ① 机动注入（只写真值实体，算法完全不可见）
        injector.apply(sim, now)

        report = env.suite_report()
        if report is None:
            break

        # ② 本地测量：检测 + 沿用 + **虚警**（虚警必须真的进融合才叫压力测试）
        local_meas: List[Any] = []
        for sensor_report in report.reports:
            if sensor_report.sensor_id not in own_ids:
                continue
            local_meas.extend(sensor_report.detections)
            local_meas.extend(sensor_report.held)
            local_meas.extend(sensor_report.false_alarms)

        # ③ 远端测量：只共享**检测**，不共享虚警（保证虚警压力可归因）
        burst_before = _link_stat(bus, "burst_lost")
        if bus is not None and bus.sharing_enabled and remote_ids:
            remote_fresh = [m for sensor_report in report.reports
                            if sensor_report.sensor_id in remote_ids
                            for m in sensor_report.detections]
            if remote_fresh:
                bus.publish(REMOTE_ENDPOINT, remote_fresh[0].sensor_id,
                            remote_fresh, now=now)
        burst_this_step = _link_stat(bus, "burst_lost") - burst_before

        # ④ 决策侧只读**已到达**的消息（consume 是一次投递，不是重复查询）
        shared: List[SharedMeasurement] = []
        if bus is not None:
            shared = [SharedMeasurement(dict(m.payload), msg_id=m.msg_id,
                                        platform_id=m.src_platform_id)
                      for m in bus.consume(LOCAL_ENDPOINT, now)]

        # ⑤ OOSM 控制：决定本步融合哪些测量、以什么不确定度
        measurements = list(local_meas) + list(shared)
        flags = [False] * len(local_meas) + [True] * len(shared)
        fused, fused_flags, decisions = oosm.select(measurements, flags, now)

        snapshot = center.update(fused, now, sensor_positions,
                                 remote_measurement_flags=fused_flags)

        # ⑥ 离线真值标签（**只在帧记录里**，不进跟踪器、不进 AI 上下文）
        measurement_truth: Dict[Tuple[str, str], Optional[str]] = {}
        for sensor_report in report.reports:
            for measurement in (sensor_report.detections
                                + sensor_report.false_alarms):
                measurement_truth[
                    (sensor_report.sensor_id, str(measurement.candidate_id))
                ] = measurement.truth_id

        # ⑦ 关联决策与门限拒绝（来自生命周期审计，不含真值）
        associations = []
        for trace in lifecycle.traces:
            if trace.consumed_at is None or abs(trace.consumed_at - now) > 1e-12:
                continue
            associations.append((trace.sensor_id, trace.candidate_id,
                                 trace.chosen_track_id, trace.is_remote))
        gate_rejected_by_truth: Dict[str, int] = {}
        rejected_residual_by_truth: Dict[str, float] = {}
        rejected_mahalanobis_by_truth: Dict[str, float] = {}
        for trace in lifecycle.traces:
            if STAGE_GATE_REJECTED not in trace.stages or trace.measured_at is None:
                continue
            if abs(trace.measured_at - now) > 1e-9:
                continue
            truth_id = measurement_truth.get((trace.sensor_id, trace.candidate_id))
            if not truth_id:
                continue
            gate_rejected_by_truth[truth_id] = (
                gate_rejected_by_truth.get(truth_id, 0) + 1
            )
            # 被拒时也留残差：目标机动时**正是被拒的那条**最能说明模型失配，
            # 只看"被接受的测量残差"会把失配信号系统性漏掉。
            costs = [c for c in trace.association_candidate_tracks
                     if c.mahalanobis_sq != float("inf")]
            if costs:
                rejected_residual_by_truth[truth_id] = min(
                    rejected_residual_by_truth.get(truth_id, float("inf")),
                    min(c.residual_m for c in costs),
                )
                rejected_mahalanobis_by_truth[truth_id] = min(
                    rejected_mahalanobis_by_truth.get(truth_id, float("inf")),
                    min(c.mahalanobis_sq for c in costs),
                )

        frame = _build_frame(
            now=now, snapshot=snapshot, sim=sim, center=center,
            measurement_truth=measurement_truth, associations=associations,
            decisions=decisions, gate_rejected_by_truth=gate_rejected_by_truth,
            rejected_residual_by_truth=rejected_residual_by_truth,
            rejected_mahalanobis_by_truth=rejected_mahalanobis_by_truth,
            burst_this_step=burst_this_step,
            incoming_sensor_id=scenario.incoming_sensor_id, report=report,
        )
        frames.append(frame)
        core_metrics.add_frame(frame)
        steps_done += 1
        if terminated or truncated:
            break

    result = core_metrics.result()
    # 整轮仍扣在重排缓冲里的测量要留下 `held` 记录，不能悄悄消失
    oosm.finalize()
    funnel = lifecycle.funnel()
    ambiguity = lifecycle.association_ambiguity_stats()
    comm_stats = bus.statistics() if bus is not None else {
        "n_messages": 0, "delivery_rate": 0.0, "latency_mean_s": 0.0,
        "latency_p95_s": 0.0, "drop_reasons": {}, "n_out_of_order": 0,
        "out_of_order_rate": 0.0, "max_reorder_lag_s": 0.0,
    }

    system = _system_metrics(scenario, frames, oosm, chosen)
    result.update({
        "scenario": scenario_id,
        "scenario_title": scenario.title_cn,
        "scenario_group": scenario.group,
        "axis": scenario.axis,
        "variant": chosen,
        "policy": share_policy,
        "policy_cn": (
            "仅本地雷达（单雷达）" if share_policy == "single"
            else ("有偏传感器共享" if chosen == "biased_share"
                  else SHARE_POLICY_CN.get(share_policy, share_policy))
        ),
        "oosm_policy": oosm_policy,
        "oosm_policy_cn": OOSM_POLICY_CN.get(oosm_policy, oosm_policy),
        "seed": seed,
        "steps": steps_done,
        "n_remote_sensors": len(remote_sensors),
        "remote_utilization": funnel["remote"]["utilization"],
        "local_utilization": funnel["local"]["utilization"],
        "gate_rejected": center.stats["gate_rejected"],
        "ambiguous_measurements": center.stats["ambiguous"],
        "ambiguous_rate": ambiguity["ambiguous_rate"],
        "mean_candidate_tracks": ambiguity["mean_candidate_tracks"],
        "tracks_initiated": center.stats["initiated"],
        "tracks_dropped": center.stats["dropped"],
        "n_messages": comm_stats["n_messages"],
        "delivery_rate": comm_stats["delivery_rate"],
        "latency_mean_s": comm_stats["latency_mean_s"],
        "latency_p95_s": comm_stats["latency_p95_s"],
        "comm_out_of_order_rate": comm_stats.get("out_of_order_rate", 0.0),
        "comm_max_reorder_lag_s": comm_stats.get("max_reorder_lag_s", 0.0),
        "drop_reasons": comm_stats.get("drop_reasons", {}),
        "n_maneuvers_applied": len(injector.events),
    })
    return {
        "metrics": result,
        "frames": frames,
        "funnel": funnel,
        "ambiguity": ambiguity,
        "lifecycle": lifecycle,
        "center": center,
        "scenario": scenario,
        "system": system,
        "oosm": oosm,
        "maneuvers": list(injector.events),
    }


# ----------------------------------------------------------------------


def _system_metrics(
    scenario: StressScenario, frames: Sequence[Dict[str, Any]],
    oosm: OosmController, chosen: str,
) -> Dict[str, Any]:
    """按场景挑着算系统级指标（不算用不上的那一组）。"""
    system: Dict[str, Any] = {}
    if scenario.maneuvers:
        accumulator = ManeuverMetrics(maneuvers=scenario.maneuvers)
        for frame in frames:
            accumulator.add_frame(frame)
        system["maneuver"] = accumulator.result()
    if scenario.incoming_sensor_id:
        from multi_target_stress.system_scenarios import (
            handover_overlap_s,
            handover_windows,
        )

        windows = handover_windows()
        accumulator = HandoverMetrics(
            incoming_sensor_id=scenario.incoming_sensor_id,
            target_id=scenario.handover_target_id,
            overlap_window_s=handover_overlap_s(),
            local_window_s=windows["A"],
        )
        for frame in frames:
            accumulator.add_frame(frame)
        system["handover"] = accumulator.result()
        system["handover_windows"] = {k: list(v) for k, v in windows.items()}
    if scenario.axis == "oosm" or scenario.outage_windows:
        accumulator = TimingMetrics(
            outage_windows=tuple(scenario.outage_windows or ()),
            oosm_summary=oosm.summary(),
        )
        for frame in frames:
            accumulator.add_frame(frame)
        system["timing"] = accumulator.result()
    if scenario.bias_overrides:
        accumulator = BiasMetrics(
            biased_sensor_ids=tuple(scenario.bias_overrides.keys()),
            local_sensor_id=scenario.local_sensor_id,
        )
        for frame in frames:
            accumulator.add_frame(frame)
        system["bias"] = accumulator.result()
        system["bias_injected"] = bool(chosen == scenario.bias_variant)
    return system


def _link_stat(bus: Optional[CommBus], key: str) -> int:
    if bus is None:
        return 0
    return sum(int(link.stats.get(key, 0)) for link in bus.links.values())


def _build_frame(
    now: float, snapshot: Any, sim: Any, center: FusionCenter,
    measurement_truth: Dict[Any, Any], associations: Sequence[Any],
    decisions: Sequence[Any], gate_rejected_by_truth: Dict[str, int],
    rejected_residual_by_truth: Dict[str, float],
    rejected_mahalanobis_by_truth: Dict[str, float],
    burst_this_step: int, incoming_sensor_id: str, report: Any,
) -> Dict[str, Any]:
    """组装一帧：既有核心字段 + 系统级指标要用的字段。"""
    tracks = list(snapshot.tracks)
    truth_all = list(sim.targets)
    active = [t for t in truth_all if t.is_active]

    # 离线一对一分配（**评测专用**；跟踪器不参与这一步）
    track_to_truth, truth_to_track = assign_tracks_to_truth(
        [(t.track_id, t.position) for t in tracks],
        [(t.target_id, t.position) for t in active],
    )
    by_id = {t.track_id: t for t in tracks}

    position_errors: List[float] = []
    velocity_errors: List[float] = []
    target_metrics: List[Dict[str, Any]] = []
    for target in active:
        track_id = truth_to_track.get(target.target_id)
        if track_id is None:
            target_metrics.append({
                "target_id": target.target_id, "time_s": now, "track_id": "",
                "residual_m": None, "innovation_mahalanobis_sq": None,
                "position_error_m": None,
                "gate_rejected": gate_rejected_by_truth.get(target.target_id, 0),
                "rejected_residual_m": rejected_residual_by_truth.get(
                    target.target_id),
                "rejected_mahalanobis_sq": rejected_mahalanobis_by_truth.get(
                    target.target_id),
            })
            continue
        track = by_id[track_id]
        error = (track.position - target.position).norm()
        position_errors.append(error)
        velocity_errors.append((track.velocity - target.velocity).norm())
        source = max(track.sources, key=lambda s: s.measurement_time_s,
                     default=None)
        target_metrics.append({
            "target_id": target.target_id, "time_s": now, "track_id": track_id,
            # 残差取该航迹**最近一条来源**的残差——"最近这次测量离预测位置
            # 有多远"，正是运动模型失配的直接观测量
            "residual_m": (source.residual_m if source is not None else None),
            "innovation_mahalanobis_sq": (
                source.innovation_mahalanobis_sq if source is not None else None),
            "position_error_m": error,
            "gate_rejected": gate_rejected_by_truth.get(target.target_id, 0),
            "rejected_residual_m": rejected_residual_by_truth.get(
                target.target_id),
            "rejected_mahalanobis_sq": rejected_mahalanobis_by_truth.get(
                target.target_id),
        })

    incoming_detections = 0
    if incoming_sensor_id:
        incoming_detections = sum(
            len(sr.detections) for sr in report.reports
            if sr.sensor_id == incoming_sensor_id
        )

    return {
        "time_s": now,
        "n_local": snapshot.n_local_measurements,
        "n_remote": snapshot.n_remote_measurements,
        "n_stale_rejected": snapshot.n_stale_rejected,
        "n_kind_rejected": snapshot.n_kind_rejected,
        "n_accepted_measurements": snapshot.n_accepted_measurements,
        "n_tracks": snapshot.n_tracks,
        "n_confirmed": snapshot.n_confirmed,
        "n_coasting": sum(1 for t in tracks if t.status == "coasting"),
        "n_assigned_tracks": len(track_to_truth),
        "own_sensor_ids": sorted(center.own_sensor_ids),
        "track_sensors": sorted({s.sensor_id for t in tracks for s in t.sources}),
        #: 逐航迹的贡献传感器集合：用于区分"覆盖接力"与"身份接力"
        "track_sensor_map": {
            t.track_id: sorted({s.sensor_id for s in t.sources}) for t in tracks
        },
        "tracks": [(t.track_id, t.position, t.velocity, t.status) for t in tracks],
        "truth": [(t.target_id, t.position, t.velocity, t.is_active)
                  for t in truth_all],
        "associations": list(associations),
        "measurement_truth": measurement_truth,
        # --- 系统级 ---
        "target_metrics": target_metrics,
        "track_sources": [
            {**source.to_dict(), "track_id": track.track_id}
            for track in tracks for source in track.sources
        ],
        "track_ages": [max(0.0, now - t.created_at) for t in tracks],
        "track_sigmas": [
            (t.sigma_position.x ** 2 + t.sigma_position.y ** 2
             + t.sigma_position.z ** 2) ** 0.5 for t in tracks
        ],
        "position_errors": position_errors,
        "velocity_errors": velocity_errors,
        "oosm_decisions": [d.to_dict() for d in decisions],
        "burst_dropped_this_step": int(burst_this_step),
        "incoming_sensor_detections": incoming_detections,
    }
