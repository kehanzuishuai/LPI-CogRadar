"""把仿真侧的数据转成 AI 层能吃的**结构化只读快照**。

这一层是「仿真 ↔ AI」的边界：
* 只读取 `Simulator` / `StepResult` / 汇总字典里的**叶子字段**；
* 绝不把 Simulator、Tensor、numpy 数组等实时对象交给 AI 层；
* 输出一定是 `StateSnapshot` 或纯 dict，可直接 JSON 序列化。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ai.snapshot_boundary import (
    ONLINE_UNKNOWN_FIELDS,
    SOURCE_OFFLINE,
    SOURCE_ONLINE,
)
from sensor.record import REASON_PROVENANCE

from .schema import (
    AgentState,
    EnergyState,
    InterceptorState,
    JammerState,
    PowerState,
    StateSnapshot,
    TargetState,
)

#: 对比与报告默认使用的指标顺序（与 metrics.collector 的汇总键一致）
COMPARE_METRICS: List[str] = [
    "horizon_satisfaction_rate",
    "violation_rate",
    "avg_tx_power_w",
    "cumulative_energy_j",
    "avg_intercept_prob",
    "avg_exposure",
    "cumulative_exposure",
    "composite_reward",
]


def snapshot_from_simulator(
    sim: Any,
    result: Any | None = None,
    agent_state: Optional[AgentState] = None,
    recent_violation_rate: Optional[float] = None,
    recent_avg_power_w: Optional[float] = None,
    preview_power_w: Optional[float] = None,
    env: Any | None = None,
    trust: Any | None = None,
) -> StateSnapshot:
    """从仿真器（可附带某一步的结果）构造只读快照。

    参数
    ----
    sim        : Simulator（必须已 load_config / reset）
    result     : 可选，`StepResult`。给了就用它的真实数值；不给就用 preview 现场算。
    agent_state: 可选，智能体侧信息（动作、Q 值、λ 等）。
    preview_power_w: 无 result 时用于现场评估的功率（默认用标称 tx_power_w）。
    env        : 可选，LpiPowerEnv。给了就附带**观测链路状态**（v4.0 POMDP），
                 使 AI 诊断能解释「为什么智能体看到的量不可信」。
    trust      : 可选，TrustState。给了就附带不确定度与回退情况（v4.0 P2），
                 使 AI 诊断能解释「为什么触发回退」。
    """
    sim._ensure_ready()
    assert sim.scenario is not None and sim.radar is not None

    scenario = sim.scenario
    radar = sim.radar

    if result is not None:
        detections = result.detections
        interceptions = result.interceptions
        pd_min = float(result.pd_min)
        task_satisfied = bool(result.task_satisfied)
        task_violated = bool(result.task_violated)
        pint_eff = float(result.intercept_prob)
        pint_inst = float(result.intercept_prob_instant)
        step_index = int(result.step_index)
        time_s = float(result.time)
        tx_power_w = float(result.tx_power_w)
        power_level = int(result.power_level)
        remaining_j = float(result.remaining_energy_j)
        cumulative_j = float(result.cumulative_energy_j)
        reward = float(result.reward)
        jam_ratio = float(result.jam_noise_ratio)
    else:
        pt_w = float(preview_power_w if preview_power_w is not None else radar.tx_power_w)
        evaluation = sim.preview(pt_w)
        detections = evaluation.detections
        interceptions = evaluation.interceptions
        pd_min = float(evaluation.pd_min)
        task_satisfied = bool(evaluation.task_satisfied)
        task_violated = not task_satisfied
        pint_eff = float(evaluation.intercept_prob)
        pint_inst = float(evaluation.intercept_prob_instant)
        step_index = int(sim.step_index)
        time_s = float(sim.current_time)
        tx_power_w = pt_w
        power_level = int(scenario.nearest_level_index(pt_w))
        remaining_j = float(sim.remaining_energy_j)
        cumulative_j = float(sim.cumulative_energy_j)
        reward = float(sim.action_reward(evaluation, pt_w))
        jam_ratio = float(evaluation.jam_noise_ratio)

    targets = [
        TargetState(
            target_id=str(d.target_id),
            range_m=round(float(d.range_m), 3),
            rcs_m2=float(d.rcs_m2),
            snr_db=round(float(d.snr_db), 4),
            pd=round(float(d.pd), 6),
        )
        for d in detections
    ]

    interceptors = [
        InterceptorState(
            interceptor_id=str(i.interceptor_id),
            range_m=round(float(i.range_m), 3),
            beam=str(i.beam),
            snr_db=round(float(i.snr_db), 4),
            pint_inst=round(float(i.pint), 6),
        )
        for i in interceptions
    ]

    jammers: List[JammerState] = []
    for jammer in sim.jammers:
        if not jammer.is_active:
            continue
        controller = jammer.controller
        if controller is not None:
            jammers.append(
                JammerState(
                    jammer_id=str(jammer.jammer_id),
                    active=bool(jam_ratio > 0),
                    mode="adaptive",
                    action=str(controller.mode),
                    action_cn=_mode_cn(controller.mode),
                    jam_noise_ratio=round(jam_ratio, 6),
                    threat=round(float(controller.threat), 6),
                )
            )
        else:
            jammers.append(
                JammerState(
                    jammer_id=str(jammer.jammer_id),
                    active=bool(jam_ratio > 0),
                    mode="fixed",
                    action="window_on" if jam_ratio > 0 else "window_off",
                    action_cn="预置时间窗内工作" if jam_ratio > 0 else "预置时间窗外静默",
                    jam_noise_ratio=round(jam_ratio, 6),
                    threat=0.0,
                )
            )

    power = PowerState(
        level=power_level,
        tx_power_w=round(tx_power_w, 4),
        power_levels_w=[float(v) for v in scenario.power_levels_w],
        feasible_levels=sim.feasible_levels(),
        action_mask=sim.action_mask(),
        previous_level=int(sim.previous_power_level),
        switched=bool(
            sim.previous_power_level >= 0 and sim.previous_power_level != power_level
        ),
    )

    energy = EnergyState(
        budget_j=float(radar.energy_budget_j),
        remaining_j=round(remaining_j, 4),
        cumulative_j=round(cumulative_j, 4),
        fraction_used=round(cumulative_j / radar.energy_budget_j, 6)
        if radar.energy_budget_j > 0
        else 0.0,
        min_step_energy_j=float(sim.min_step_energy_j),
    )

    return StateSnapshot(
        scenario=str(scenario.scenario_name),
        step_index=step_index,
        time=round(time_s, 4),
        horizon_steps=int(scenario.num_steps),
        targets=targets,
        interceptors=interceptors,
        jammers=jammers,
        pd_min=round(pd_min, 6),
        required_pd=float(radar.required_pd),
        task_satisfied=task_satisfied,
        task_violated=task_violated,
        pint_eff=round(pint_eff, 6),
        pint_inst=round(pint_inst, 6),
        exposure=round(float(sim.exposure.value), 6),
        power=power,
        energy=energy,
        agent=agent_state,
        reward=round(reward, 6),
        recent_violation_rate=recent_violation_rate,
        recent_avg_power_w=recent_avg_power_w,
        observability=observability_from_env(env) if env is not None else None,
        trust=trust,
        measurement_state=measurement_state_from_env(env),
        communication_state=communication_state_from_env(env),
        fusion_state=fusion_state_from_env(env),
        cooperation_state=cooperation_state_from_env(env),
        # --- 系统级压力证据（v4.5 P2 第二阶段）---
        # 只吃跟踪器与通信总线里**算法可见的量**：来源残差、创新、门限拒绝、
        # 链路统计。不含真值、不含真实目标位置、不含尚未到达的消息。
        system_stress_state=system_stress_state_from_env(env),
        # --- 信息边界（v4.5 一致性验收）---
        # 本函数读 `sim.targets/interceptors/jammers` 与 `result.pd_min` 等，
        # 因此它是**离线评测快照**：允许含真值，但必须自报来源，
        # 否则会被误当成在线诊断输入——那会让 AI 看到比它要诊断的算法
        # 高得多的信息权限，解释也就不再可信。
        # 在线路径见 `online_snapshot_from_env`。
        information_boundary=SOURCE_OFFLINE,
        boundary_provenance=list(TRUTH_PROVENANCE_FIELDS),
        missing_reason_provenance=dict(REASON_PROVENANCE),
        online_unknown_fields=[],
    )


def system_stress_state_from_env(env: Any) -> Optional[Dict[str, Any]]:
    """构造 `system_stress_state`（无跟踪器也无总线时返回 None）。"""
    if env is None:
        return None
    tracker = getattr(env, "tracker", None)
    bus = getattr(env, "comm_bus", None)
    tracks = list(getattr(env, "tracks", []) or [])
    if not tracks and bus is None:
        return None
    from ai.system_stress import system_stress_state

    return system_stress_state(
        tracks=tracks,
        bus=bus,
        tracker_stats=(tracker.stats if tracker is not None else None),
        local_sensor_ids=sorted(getattr(env, "own_sensor_ids", []) or []),
    )


# ----------------------------------------------------------------------
# v4.5：测量 / 通信 / 融合 / 协同 证据构造
#
# 全部从 `env` 的结构化访问器读取，**不读真值**：
# 不碰 TargetOutcome.truth_id、不碰 Scene 实体位置、不读在途消息内容。
# ----------------------------------------------------------------------


def measurement_state_from_env(env: Any) -> Optional["MeasurementState"]:
    """构造测量层证据。`env` 未接入测量层时返回 None。"""
    from ai.schema import MeasurementState

    if env is None or not hasattr(env, "measurement_state_dict"):
        return None
    if getattr(env, "observation_mode", "full") not in ("ideal", "realistic"):
        return None
    data = env.measurement_state_dict()
    return MeasurementState(
        mode=data["mode"],
        n_measurements=data["n_measurements"],
        n_fresh=data["n_fresh"],
        n_held=data["n_held"],
        n_false_alarms=data["n_false_alarms"],
        observation_quality=data["observation_quality"],
        reason_counts=dict(data["reason_counts"]),
        reason_rates=dict(data["reason_rates"]),
        sensors=list(data["sensors"]),
        candidates=list(data["candidates"]),
    )


def communication_state_from_env(env: Any) -> Optional["CommunicationState"]:
    """构造通信层证据（只统计已发生的事；在途消息**只报数量**）。"""
    from ai.schema import CommunicationState

    if env is None or not hasattr(env, "communication_state_dict"):
        return None
    data = env.communication_state_dict()
    return CommunicationState(
        policy=data.get("policy", "none"),
        policy_cn=data.get("policy_cn", ""),
        n_links=int(data.get("n_links", 0)),
        n_messages_sent=int(data.get("n_messages_sent", 0)),
        n_delivered=int(data.get("n_delivered", 0)),
        n_dropped=int(data.get("n_dropped", 0)),
        delivery_rate=float(data.get("delivery_rate", 0.0)),
        drop_reasons=dict(data.get("drop_reasons", {})),
        latency_mean_s=float(data.get("latency_mean_s", 0.0)),
        latency_p95_s=float(data.get("latency_p95_s", 0.0)),
        n_in_flight=int(data.get("n_in_flight", 0)),
        n_arrived_stale=int(data.get("n_arrived_stale", 0)),
    )


def fusion_state_from_env(env: Any) -> Optional["FusionState"]:
    """构造融合/航迹证据。"""
    from ai.schema import FusionState, TrackState

    if env is None or not hasattr(env, "fusion_state_dict"):
        return None
    data = env.fusion_state_dict()
    if not data.get("enabled"):
        return None
    tracks: List[TrackState] = []
    for item in data.get("tracks", []):
        tracks.append(TrackState(
            track_id=str(item.get("track_id", "")),
            status=str(item.get("status", "unknown")),
            x=float(item.get("x", 0.0)), y=float(item.get("y", 0.0)),
            z=float(item.get("z", 0.0)),
            vx=float(item.get("vx", 0.0)), vy=float(item.get("vy", 0.0)),
            vz=float(item.get("vz", 0.0)),
            sigma_x=float(item.get("sigma_x", 0.0)),
            sigma_y=float(item.get("sigma_y", 0.0)),
            sigma_z=float(item.get("sigma_z", 0.0)),
            sigma_vx=float(item.get("sigma_vx", 0.0)),
            sigma_vy=float(item.get("sigma_vy", 0.0)),
            sigma_vz=float(item.get("sigma_vz", 0.0)),
            hits=int(item.get("hits", 0)), misses=int(item.get("misses", 0)),
            local_updates=int(item.get("local_updates", 0)),
            remote_updates=int(item.get("remote_updates", 0)),
            is_local_origin=bool(item.get("is_local_origin", False)),
            freshness=float(item.get("freshness", 0.0)),
            measurement_age_s=item.get("measurement_age_s"),
            n_sources=int(item.get("n_sources", 0)),
            platforms=list(item.get("platforms", [])),
            source_sensors=list(item.get("source_sensors", [])),
            recent_sources=list(item.get("recent_sources", [])),
            has_remote_contribution=bool(item.get("has_remote_contribution", False)),
        ))
    return FusionState(
        enabled=True,
        platform_id=str(data.get("platform_id", "")),
        n_tracks=int(data.get("n_tracks", 0)),
        n_confirmed=int(data.get("n_confirmed", 0)),
        n_coasting=int(data.get("n_coasting", 0)),
        n_tentative=int(data.get("n_tentative", 0)),
        freshness_mean=float(data.get("freshness_mean", 0.0)),
        tracks=tracks,
        n_local_measurements=int(data.get("n_local_measurements", 0)),
        n_remote_measurements=int(data.get("n_remote_measurements", 0)),
        n_kind_rejected=int(data.get("n_kind_rejected", 0)),
        n_stale_rejected=int(data.get("n_stale_rejected", 0)),
        tracks_initiated=int(data.get("tracks_initiated", 0)),
        tracks_dropped=int(data.get("tracks_dropped", 0)),
        gate_rejected_total=int(data.get("gate_rejected_total", 0)),
    )


def cooperation_state_from_env(env: Any) -> Optional["CooperationState"]:
    """构造协同感知证据（**只报有没有远端贡献，不报收益大小**）。"""
    from ai.schema import CooperationState

    if env is None or not hasattr(env, "cooperation_state_dict"):
        return None
    data = env.cooperation_state_dict()
    if data.get("policy", "none") == "none" and not data.get("sharing_enabled"):
        return None
    return CooperationState(
        sharing_enabled=bool(data.get("sharing_enabled", False)),
        policy=str(data.get("policy", "none")),
        n_tracks_with_remote=int(data.get("n_tracks_with_remote", 0)),
        n_tracks_local_only=int(data.get("n_tracks_local_only", 0)),
        remote_measurements_arrived=int(data.get("remote_measurements_arrived", 0)),
        remote_measurements_used=int(data.get("remote_measurements_used", 0)),
        remote_measurements_rejected=int(data.get("remote_measurements_rejected", 0)),
        remote_utilization=float(data.get("remote_utilization", 0.0)),
        tracks_supported_remotely_only=int(
            data.get("tracks_supported_remotely_only", 0)
        ),
        notes=list(data.get("notes", [])),
    )


def _mode_cn(mode: str) -> str:
    try:
        from models.adaptive_jammer import MODE_CN

        return MODE_CN.get(mode, mode)
    except Exception:  # noqa: BLE001 - 纯展示用途，失败就回退原文
        return mode


def observability_from_env(env: Any, step_info: Optional[Dict[str, Any]] = None) -> Any:
    """从环境读出观测链路状态（v4.0 POMDP）。

    只读取观测模型**上报**的量（估计值与标准差），不读真值：
    AI 诊断要解释的是「智能体以为自己看到什么」，混入真值会让诊断失真。
    """
    from ai.schema import ObservabilityState

    mode = str(getattr(env, "observation_mode", "full"))
    if mode != "pomdp":
        return ObservabilityState(mode="full", observation_quality=1.0)

    estimates = getattr(env, "last_estimates", {}) or {}
    dropped: List[str] = []
    stale: List[str] = []
    sigma: Dict[str, float] = {}
    for name, record in estimates.items():
        if not record.get("observed", True):
            dropped.append(name)
        if record.get("stale", False):
            stale.append(name)
        sigma[name] = float(record.get("sigma", 0.0))

    quality = float(getattr(env, "observation_quality", lambda: 1.0)())
    notes: List[str] = []
    if dropped:
        notes.append(f"本步丢测 {len(dropped)} 项，相关状态沿用上一次测量")
    if stale:
        notes.append(f"本步有 {len(stale)} 项观测存在延迟")
    if not notes:
        notes.append("观测链路正常，但所有量均带测量噪声")

    return ObservabilityState(
        mode=mode,
        observation_quality=quality,
        dropped_fields=sorted(dropped),
        stale_fields=sorted(stale),
        sigma=sigma,
        history_len=int(getattr(env, "history_len", 1)),
        notes=notes,
    )


def trust_from_policy(
    decision_record: Optional[Any] = None,
    policy: Optional[Any] = None,
) -> Any:
    """从不确定度感知策略的决策记录构造可信决策状态。"""
    from ai.schema import TrustState

    if decision_record is None:
        return None

    uncertainty = dict(getattr(decision_record, "uncertainty", {}) or {})
    trust = TrustState(
        uncertainty_source="ensemble" if "q_std_max" in uncertainty else "none",
        q_std_max=uncertainty.get("q_std_max"),
        q_std_at_best=uncertainty.get("q_std_at_best"),
        disagreement=uncertainty.get("disagreement"),
        ood_score=uncertainty.get("ood_score"),
        q_margin=uncertainty.get("q_margin"),
        decision_mode=str(getattr(decision_record, "mode", "ai")),
        decision_mode_cn=str(getattr(decision_record, "mode_cn", "")),
        reason_code=str(getattr(decision_record, "reason_code", "")),
        reason_cn=str(getattr(decision_record, "reason_cn", "")),
        triggered=list(getattr(decision_record, "triggered", ()) or ()),
    )
    if policy is not None and hasattr(policy, "agent"):
        agent = policy.agent
        ensemble_config = getattr(agent, "ensemble_config", None)
        if ensemble_config is not None:
            trust.ensemble_size = int(getattr(ensemble_config, "ensemble_size", 0))
    if policy is not None and hasattr(policy, "summary"):
        summary = policy.summary()
        trust.ai_autonomy_rate = summary.get("ai_autonomy_rate")
        trust.fallback_rate = summary.get("fallback_rate")
        trust.shield_rate = summary.get("shield_rate")
    return trust


def agent_state_from_dqn(
    q_values: Sequence[float],
    action: int,
    power_levels_w: Sequence[float],
    epsilon: Optional[float] = None,
    lambda_cost: Optional[float] = None,
    cost_rate: Optional[float] = None,
    kind: str = "dqn",
) -> AgentState:
    """从 DQN 的一次推理构造智能体状态（只取叶子数值）。"""
    values = [float(v) for v in q_values]
    margin: Optional[float] = None
    if len(values) >= 2:
        ordered = sorted(values, reverse=True)
        margin = ordered[0] - ordered[1]
    return AgentState(
        kind=kind,
        action=int(action),
        tx_power_w=round(float(power_levels_w[action]), 4) if power_levels_w else None,
        q_values=[round(v, 6) for v in values],
        q_margin=round(margin, 6) if margin is not None else None,
        epsilon=epsilon,
        lambda_cost=lambda_cost,
        cost_rate=cost_rate,
    )


def summaries_to_payload(
    summaries: Sequence[Dict[str, Any]],
    metric_order: Optional[Sequence[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把 `metrics.collector` 的汇总列表整理成 compare/report 的入参。"""
    order = list(metric_order or COMPARE_METRICS)
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        row: Dict[str, Any] = {"label": str(summary.get("label", "?"))}
        for metric in order:
            value = summary.get(metric)
            if isinstance(value, (int, float)):
                row[metric] = round(float(value), 6)
        rows.append(row)
    payload: Dict[str, Any] = {"summaries": rows, "metric_order": order}
    if context:
        payload["context"] = dict(context)
    return payload


#: 离线评测快照里**来自真值**的字段（审计用：一眼看清哪些不能进在线路径）
TRUTH_PROVENANCE_FIELDS: Tuple[str, ...] = (
    "targets", "interceptors", "jammers",
    "pd_min", "pint_eff", "pint_inst",
    "task_satisfied", "task_violated", "reward", "exposure",
)


def online_snapshot_from_env(
    env: Any,
    agent_state: Optional[AgentState] = None,
    trust: Any | None = None,
    recent_violation_rate: Optional[float] = None,
    recent_avg_power_w: Optional[float] = None,
) -> Optional[StateSnapshot]:
    """构造**在线诊断**快照：只含本平台已知状态与**已经收到**的证据。

    与 `snapshot_from_simulator` 的区别（这就是本函数存在的全部意义）：

    | 内容 | 离线评测 | 在线诊断 |
    | --- | --- | --- |
    | 真值目标 / 侦察机 / 干扰机清单 | 有 | **置空**（平台没有"目标名册"） |
    | 真值 Pd / Pint / 暴露 | 有 | 用**智能体侧记下来的估计值** |
    | 真实虚警标签、逐原因真值计数 | 有 | **去掉**（见 `_measurement_state_online`） |
    | 已到达的共享测量 / 航迹 / 协同结构 | 有 | 有（本来就是算法可见的） |

    无法确定的内容写进 `online_unknown_fields`，**不编造确定值**；
    若 `observation_mode == "full"`，智能体本来就"看得见真值"，
    此时把这些字段如实计入 `boundary_provenance` —— 与其假装它是在线估计，
    不如标注清楚。
    """
    if env is None:
        return None
    sim = getattr(env, "sim", None)
    if sim is None:
        return None

    mode = str(getattr(env, "observation_mode", ""))
    pd_estimate = float(getattr(env, "_last_pd", 0.0) or 0.0)
    pint_estimate = float(getattr(env, "_last_pint", 0.0) or 0.0)
    truth_coupled: List[str] = []
    if mode == "full":
        truth_coupled = ["pd_min", "pint_eff", "pint_inst"]

    radar = getattr(sim, "radar", None)
    scenario = getattr(sim, "scenario", None)
    levels = list(getattr(sim, "power_levels_w", []) or [])
    previous_level = int(getattr(sim, "previous_power_level", -1))
    executed_power = levels[previous_level] if 0 <= previous_level < len(levels) else 0.0
    budget = float(getattr(radar, "energy_budget_j", 0.0) or 0.0)
    remaining = float(getattr(sim, "remaining_energy_j", 0.0) or 0.0)
    feasible = getattr(sim, "feasible_levels", None)
    mask = getattr(sim, "action_mask", None)

    unknown = list(ONLINE_UNKNOWN_FIELDS)
    if mode != "pomdp":
        # 没有不确定度通道时，暴露量没有估计值可用 → 只能标为未知
        unknown.append("exposure")

    return StateSnapshot(
        scenario=str(getattr(scenario, "scenario_name", "")),
        step_index=int(getattr(sim, "step_index", 0)),
        time=round(float(getattr(sim, "current_time", 0.0)), 4),
        horizon_steps=int(getattr(scenario, "num_steps", 0) or 0),
        targets=[],
        interceptors=[],
        jammers=[],
        pd_min=round(pd_estimate, 6),
        required_pd=float(getattr(radar, "required_pd", 0.0) or 0.0),
        task_satisfied=False,
        task_violated=False,
        pint_eff=round(pint_estimate, 6),
        pint_inst=round(pint_estimate, 6),
        exposure=0.0,
        power=PowerState(
            level=previous_level,
            tx_power_w=round(executed_power, 4),
            power_levels_w=[float(v) for v in levels],
            feasible_levels=list(feasible() if callable(feasible) else []),
            action_mask=list(mask() if callable(mask) else []),
            previous_level=previous_level,
            switched=False,
        ),
        energy=EnergyState(
            budget_j=budget,
            remaining_j=round(remaining, 4),
            cumulative_j=round(max(0.0, budget - remaining), 4),
            fraction_used=round(1.0 - remaining / budget, 6) if budget > 0 else 0.0,
            min_step_energy_j=float(getattr(sim, "min_step_energy_j", 0.0) or 0.0),
        ),
        agent=agent_state,
        reward=0.0,
        recent_violation_rate=recent_violation_rate,
        recent_avg_power_w=recent_avg_power_w,
        observability=observability_from_env(env),
        trust=trust,
        measurement_state=_measurement_state_online(env),
        communication_state=communication_state_from_env(env),
        fusion_state=fusion_state_from_env(env),
        cooperation_state=cooperation_state_from_env(env),
        system_stress_state=system_stress_state_from_env(env),
        information_boundary=SOURCE_ONLINE,
        online_unknown_fields=unknown,
        boundary_provenance=truth_coupled,
        missing_reason_provenance=dict(REASON_PROVENANCE),
    )


def _measurement_state_online(env: Any) -> Optional["MeasurementState"]:
    """在线测量节：去掉**评测专用**内容。

    逐原因计数需要"本帧涉及多少条**真值**实体"才能当分母，
    因此它自带真值信息；真实虚警标签同理（真实雷达分不清
    "这是一条虚警"还是"这是个真目标"）。在线路径只保留
    设备已知 / 可推断的部分，并由 `missing_reason_provenance` 标明推断属性。
    """
    state = measurement_state_from_env(env)
    if state is None:
        return None
    state.reason_counts = {}
    state.reason_rates = {}
    state.n_false_alarms = 0
    return state
