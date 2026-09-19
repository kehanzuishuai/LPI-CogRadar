"""AI 认知诊断层的**结构化 JSON 输入输出协议**。

这是整个 AI 层最重要的文件：它把「仿真/智能体侧」与「大模型侧」解耦。
所有字段都是纯数据（dict / list / str / number / bool），可以直接 JSON 序列化，
**不含任何实时对象引用**（Simulator、Tensor、numpy 数组一律不出现）。

协议要点
--------
* 输入（`StateSnapshot`）只包含**只读状态**：时间、目标距离、干扰强度、当前功率、
  Pd、Pint、累计暴露、剩余能量、DQN 动作与 Q 值等；
  **不含任何动作指令字段**——AI 层不具备控制权，这是刻意的架构约束。
* 输出（`DiagnosisResult` / `ExplainResult` / `CompareResult` / `ReportResult`）
  一律包含 `provider`、`status`、`latency_ms`，便于审计「这条结论是谁生成的」。
* `Finding.evidence` 必须是**可核验的数值**，`Finding.code` 必须是枚举代码；
  自然语言只允许解释这些证据，不允许自造事实。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

#: 严重度
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

#: 结果状态：ok = 正常；degraded = 降级（用了规则兜底）；error = 失败
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_ERROR = "error"

#: 诊断发现代码
FINDING_CODES: Dict[str, str] = {
    "DETECTION_AT_RISK": "探测概率接近或低于任务要求",
    "DETECTION_VIOLATED": "本步探测未达标",
    "ENERGY_LOW": "剩余能量进入警戒区",
    "ENERGY_EXHAUSTED": "剩余能量已不足以继续发射",
    "HIGH_EXPOSURE": "累计暴露量偏高，被识别风险上升",
    "EXPOSURE_RISING": "累计暴露量持续上升",
    "INTERCEPT_RISK_HIGH": "有效截获概率偏高",
    "JAMMING_ESCALATED": "干扰强度上升（或被自适应干扰机升级压制）",
    "POWER_AT_MAX_FEASIBLE": "当前已是可行最高档，功率受限",
    "POWER_INEFFICIENT": "当前功率高于满足探测所需的最低档",
    "ENERGY_OVERRUN_RISK": "按当前功率无法支撑到任务结束",
    "DETECTION_WELL_WITHIN_MARGIN": "探测余量充足",
    "LOW_INTERCEPT_STATE": "处于低截获状态",
    # --- v4.0 部分可观测与可信决策 ---
    "OBSERVATION_DEGRADED": "观测链路退化（测量噪声偏大或存在延迟）",
    "OBSERVATION_MISSING": "本步存在丢测，部分状态量来自上一次测量",
    "ESM_POSITION_UNKNOWN": "侦察机真实位置不可直接观测，只能依赖带误差的估计",
    "EXPOSURE_ESTIMATE_UNCERTAIN": "累计暴露只能估计，存在不可忽略的误差",
    "AI_HIGH_UNCERTAINTY": "集成各成员对该状态的估值分歧较大，AI 决策不可靠",
    "AI_OOD_INPUT": "当前观测偏离训练分布，网络没有外推依据",
    "AI_FALLBACK_TRIGGERED": "不确定度过高，已回退到规则策略",
    "AI_SHIELD_APPLIED": "安全护盾生效：AI 动作被抬升到满足探测要求的最低档位",
    "AI_SMALL_Q_MARGIN": "最优与次优动作 Q 值接近，AI 决策缺乏区分度",
    # --- v4.5：测量 / 通信 / 融合 / 航迹 / 协同 ---
    # 测量层：把"为什么现在没有目标信息"逐原因分开
    "TARGET_OUT_OF_FOV": "目标不在传感器视场内（指向问题）",
    "TARGET_BEYOND_RANGE": "目标超出传感器作用距离（能量/门限问题）",
    "TARGET_OCCLUDED": "目标视线被遮挡（环境问题，提功率无用）",
    "SENSOR_NOT_UPDATED": "传感器尚未到更新时刻（时间问题，旧值仍有效）",
    "MISSED_DETECTION": "本帧检测遗漏（概率问题，下一帧可能就有）",
    "SENSOR_UNAVAILABLE": "传感器不可用（关机/故障）",
    # 通信层：把"这条远端测量有没有到"逐原因分开
    "COMM_PACKET_LOST": "远端消息在链路上丢失",
    "COMM_MESSAGE_EXPIRED": "远端消息晚于过期时限到达，已丢弃",
    "COMM_LINK_DOWN": "通信链路不可用",
    "COMM_QUEUE_FULL": "发送队列积压超限，新消息被丢弃",
    "REMOTE_MEASUREMENT_DELAYED": "远端测量已到达但已过时（时效不足）",
    # 融合/航迹层
    "TRACK_COASTING": "航迹当前无观测支撑，处于预测外推（coasting）",
    "TRACK_MAINTAINED": "航迹有持续观测支撑（本地或远端）",
    "TRACK_FRAGMENTED": "航迹发生断裂后重建，Track ID 已改变",
    "TRACK_UNCERTAIN": "航迹位置不确定度偏大",
    "ASSOCIATION_AMBIGUOUS": "存在多个关联代价接近的候选，关联结果不确定",
    # 协同层
    "REMOTE_SENSOR_CONTRIBUTION": "本地航迹获得了远端传感器的测量贡献",
    "COOPERATIVE_TRACK_RECOVERED": "本地看不见时，远端共享使航迹得以保持",
    # 系统级压力层（v4.5 P2 第二阶段）
    "MANEUVER_MODEL_MISMATCH": "目标机动导致运动模型失配（预测残差显著放大）",
    "TRACK_HANDOVER_IN_PROGRESS": "航迹正在从一个传感器向另一个传感器交接",
    "TRACK_HANDOVER_COMPLETED": "交接完成：新传感器的测量已进入同一条航迹",
    "TRACK_HANDOVER_FAILED": "交接失败：本地包线之外没有任何测量接上",
    "MEASUREMENT_OUT_OF_ORDER": "远端测量乱序到达（先到的反而更新）",
    "COMM_LINK_OUTAGE": "通信链路发生中断（中断窗口内消息全部丢弃）",
    "COMM_BURST_LOSS": "链路发生突发丢包（连续多条消息丢失）",
    "COMM_RECOVERY_CONGESTION": "链路恢复后出现拥塞（额外丢包与延迟）",
    "SENSOR_BIAS_SUSPECTED": "某传感器长期残差异常，可能存在系统偏差",
    "SENSOR_NOISE_UNDERREPORTED": "传感器上报精度优于其实际精度（噪声低估）",
    "INNOVATION_INCONSISTENT": "新息与上报协方差不一致（残差超出卡方上界）",
}


# ----------------------------------------------------------------------
# 输入协议
# ----------------------------------------------------------------------

@dataclass
class TargetState:
    target_id: str
    range_m: float
    rcs_m2: float
    snr_db: float
    pd: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InterceptorState:
    interceptor_id: str
    range_m: float
    beam: str
    snr_db: float
    pint_inst: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JammerState:
    jammer_id: str
    active: bool
    mode: str = "fixed"  # fixed | adaptive
    action: str = ""  # 自适应模式下的当前动作（no_jam/low_power/...）
    action_cn: str = ""
    jam_noise_ratio: float = 0.0
    threat: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PowerState:
    level: int
    tx_power_w: float
    power_levels_w: List[float] = field(default_factory=list)
    feasible_levels: List[int] = field(default_factory=list)
    action_mask: List[bool] = field(default_factory=list)
    previous_level: int = -1
    switched: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnergyState:
    budget_j: float
    remaining_j: float
    cumulative_j: float
    fraction_used: float
    min_step_energy_j: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    """DQN（或其它智能体）的可观测输出。仅作诊断输入，不含控制语义。"""

    kind: str = "unknown"  # dqn | dqn_lagrangian | rule | greedy | lookahead | random
    action: Optional[int] = None
    tx_power_w: Optional[float] = None
    q_values: List[float] = field(default_factory=list)
    q_margin: Optional[float] = None  # 最大 Q 与次大 Q 的差
    epsilon: Optional[float] = None
    lambda_cost: Optional[float] = None  # 安全 RL 的拉格朗日乘子
    cost_rate: Optional[float] = None  # 约束违反率

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservabilityState:
    """观测链路状态（v4.0 POMDP 新增）。

    AI 层需要知道「智能体看到的东西有多可信」，否则无法解释
    「为什么当前 AI 不确定」。所有字段都来自观测模型的上报，
    不包含真值。
    """

    mode: str = "full"  # full | pomdp
    observation_quality: float = 1.0  # 0~1，1 = 全新鲜全精确
    dropped_fields: List[str] = field(default_factory=list)  # 本步丢测的量
    stale_fields: List[str] = field(default_factory=list)  # 沿用旧测量的量
    sigma: Dict[str, float] = field(default_factory=dict)  # 各量的上报标准差
    history_len: int = 1
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "observation_quality": round(self.observation_quality, 6),
            "dropped_fields": list(self.dropped_fields),
            "stale_fields": list(self.stale_fields),
            "sigma": {k: round(float(v), 6) for k, v in self.sigma.items()},
            "history_len": self.history_len,
            "notes": list(self.notes),
        }


@dataclass
class TrustState:
    """可信决策状态：不确定度 + 回退情况（v4.0 P2 新增）。

    这一节是「AI 知不知道自己在犯错」的直接证据来源。
    """

    uncertainty_source: str = "none"  # none | ensemble | single
    ensemble_size: int = 0
    q_std_max: Optional[float] = None  # 集成分歧（认知不确定度）
    q_std_at_best: Optional[float] = None
    disagreement: Optional[float] = None  # 成员投票不一致比例
    ood_score: Optional[float] = None  # 分布外评分
    q_margin: Optional[float] = None

    decision_mode: str = "ai"  # ai | shield | fallback_rule
    decision_mode_cn: str = ""
    reason_code: str = ""
    reason_cn: str = ""
    triggered: List[str] = field(default_factory=list)

    # 累计统计（一个 episode 或一段评测内的比率）
    ai_autonomy_rate: Optional[float] = None
    fallback_rate: Optional[float] = None
    shield_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        def r(value: Optional[float]) -> Optional[float]:
            return None if value is None else round(float(value), 6)

        return {
            "uncertainty_source": self.uncertainty_source,
            "ensemble_size": self.ensemble_size,
            "q_std_max": r(self.q_std_max),
            "q_std_at_best": r(self.q_std_at_best),
            "disagreement": r(self.disagreement),
            "ood_score": r(self.ood_score),
            "q_margin": r(self.q_margin),
            "decision_mode": self.decision_mode,
            "decision_mode_cn": self.decision_mode_cn,
            "reason_code": self.reason_code,
            "reason_cn": self.reason_cn,
            "triggered": list(self.triggered),
            "ai_autonomy_rate": r(self.ai_autonomy_rate),
            "fallback_rate": r(self.fallback_rate),
            "shield_rate": r(self.shield_rate),
        }


@dataclass
class StateSnapshot:
    """送给 AI 诊断层的**只读**状态快照。

    刻意不含任何「建议动作」以外的控制字段：AI 层可以描述、解释、建议，
    但绝不能直接驱动雷达——动作永远由策略/DQN 决定。
    """

    project: str = "LPI-CogRadar"
    schema_version: str = SCHEMA_VERSION
    scenario: str = ""
    step_index: int = 0
    time: float = 0.0
    horizon_steps: int = 0

    targets: List[TargetState] = field(default_factory=list)
    interceptors: List[InterceptorState] = field(default_factory=list)
    jammers: List[JammerState] = field(default_factory=list)

    pd_min: float = 0.0
    required_pd: float = 0.8
    task_satisfied: bool = False
    task_violated: bool = False

    pint_eff: float = 0.0
    pint_inst: float = 0.0
    exposure: float = 0.0

    power: Optional[PowerState] = None
    energy: Optional[EnergyState] = None
    agent: Optional[AgentState] = None
    reward: Optional[float] = None

    recent_violation_rate: Optional[float] = None
    recent_avg_power_w: Optional[float] = None

    # --- v4.0 ---
    observability: Optional[ObservabilityState] = None
    trust: Optional[TrustState] = None

    # --- v4.5：测量 → 通信 → 融合 → 航迹 → 协同 证据链 ---
    measurement_state: Optional[MeasurementState] = None
    communication_state: Optional[CommunicationState] = None
    fusion_state: Optional[FusionState] = None
    cooperation_state: Optional[CooperationState] = None
    #: 系统级压力证据（v4.5 P2 第二阶段）：机动失配 / 交接 / 通信时序 / 传感器健康。
    #: 全部是**算法可见的量**（残差、创新、门限拒绝、来源计数、链路统计），
    #: 不含真值、不含真实目标位置、不含尚未到达的消息。
    system_stress_state: Optional[Dict[str, Any]] = None

    # --- 信息边界（v4.5 一致性验收）---
    #: 快照来源：`"online"`（只含本地已知 + 已收到的证据）或
    #: `"offline_evaluation"`（额外含真值）。**必须显式声明**，
    #: 否则无法判断这份快照能不能作为在线诊断输入。
    information_boundary: str = ""
    #: 在线路径下**明确标为未知**的字段名（不确定就必须说不知道）
    online_unknown_fields: List[str] = field(default_factory=list)
    #: 本快照里哪些字段来自真值（离线路径专用清单，便于审计）
    boundary_provenance: List[str] = field(default_factory=list)
    #: 逐缺失原因的信息来源：device_known / measurement_inferred / evaluation_only
    missing_reason_provenance: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "schema_version": self.schema_version,
            "scenario": self.scenario,
            "step_index": self.step_index,
            "time": self.time,
            "horizon_steps": self.horizon_steps,
            "targets": [t.to_dict() for t in self.targets],
            "interceptors": [i.to_dict() for i in self.interceptors],
            "jammers": [j.to_dict() for j in self.jammers],
            "detection": {
                "pd_min": self.pd_min,
                "required_pd": self.required_pd,
                "task_satisfied": self.task_satisfied,
                "task_violated": self.task_violated,
            },
            "interception": {
                "pint_eff": self.pint_eff,
                "pint_inst": self.pint_inst,
                "exposure": self.exposure,
            },
            "power": self.power.to_dict() if self.power else None,
            "energy": self.energy.to_dict() if self.energy else None,
            "agent": self.agent.to_dict() if self.agent else None,
            "reward": self.reward,
            "recent": {
                "violation_rate": self.recent_violation_rate,
                "avg_power_w": self.recent_avg_power_w,
            },
            "observability": self.observability.to_dict() if self.observability else None,
            "trust": self.trust.to_dict() if self.trust else None,
            "measurement_state":
                self.measurement_state.to_dict() if self.measurement_state else None,
            "communication_state":
                self.communication_state.to_dict() if self.communication_state else None,
            "fusion_state": self.fusion_state.to_dict() if self.fusion_state else None,
            "cooperation_state":
                self.cooperation_state.to_dict() if self.cooperation_state else None,
            "system_stress_state": self.system_stress_state,
            "information_boundary": self.information_boundary,
            "online_unknown_fields": list(self.online_unknown_fields),
            "boundary_provenance": list(self.boundary_provenance),
            "missing_reason_provenance": dict(self.missing_reason_provenance),
        }


# ----------------------------------------------------------------------
# v4.5：测量 / 通信 / 融合 / 航迹 / 协同
# ----------------------------------------------------------------------


@dataclass
class MeasurementState:
    """当前时刻**算法可见**的测量层状态。

    数据来源：`SensorReport.measurements`（候选编号 + 估计值）与
    `SuiteReport.reason_counts()`（逐原因计数）。

    ⚠️ **不含** `TargetOutcome.truth_id`、也不含任何真值位置。
    AI 只能知道"有几个候选、每个候选的估计与不确定度"，
    以及"有多少判定因为什么原因没有数据"。
    """

    mode: str = "full"
    n_measurements: int = 0
    n_fresh: int = 0
    n_held: int = 0
    n_false_alarms: int = 0
    observation_quality: float = 0.0
    #: 逐原因计数（键为 NoDataReason 值）
    reason_counts: Dict[str, int] = field(default_factory=dict)
    #: 逐原因占比（分母为该步全部判定数）
    reason_rates: Dict[str, float] = field(default_factory=dict)
    #: 每个在线传感器的状态（只含配置与可用性，不含真值）
    sensors: List[Dict[str, Any]] = field(default_factory=list)
    #: 候选测量的摘要（candidate_id + 估计 + σ + 置信度 + 年龄）
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "n_measurements": self.n_measurements,
            "n_fresh": self.n_fresh,
            "n_held": self.n_held,
            "n_false_alarms": self.n_false_alarms,
            "observation_quality": round(self.observation_quality, 6),
            "reason_counts": dict(self.reason_counts),
            "reason_rates": {k: round(v, 6) for k, v in self.reason_rates.items()},
            "sensors": [dict(x) for x in self.sensors],
            "candidates": [dict(x) for x in self.candidates],
        }


@dataclass
class CommunicationState:
    """通信层状态（**只统计已发生的事，不含未来消息**）。"""

    policy: str = "none"
    policy_cn: str = ""
    n_links: int = 0
    n_messages_sent: int = 0
    n_delivered: int = 0
    n_dropped: int = 0
    delivery_rate: float = 0.0
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    latency_mean_s: float = 0.0
    latency_p95_s: float = 0.0
    #: 当前在途、尚未到达的消息数（**只报数量，不得读其内容**）
    n_in_flight: int = 0
    #: 已到达但被时效判为过时的远端测量数（本步）
    n_arrived_stale: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "policy_cn": self.policy_cn,
            "n_links": self.n_links,
            "n_messages_sent": self.n_messages_sent,
            "n_delivered": self.n_delivered,
            "n_dropped": self.n_dropped,
            "delivery_rate": round(self.delivery_rate, 6),
            "drop_reasons": dict(self.drop_reasons),
            "latency_mean_s": round(self.latency_mean_s, 6),
            "latency_p95_s": round(self.latency_p95_s, 6),
            "n_in_flight": self.n_in_flight,
            "n_arrived_stale": self.n_arrived_stale,
        }


@dataclass
class TrackState:
    """单条航迹的结构化描述（**AI 解释航迹的证据基础**）。

    全部字段来自跟踪器自己的状态，**不含真值**：
    没有 truth_id、没有真实位置、没有与真值目标的对应关系。
    """

    track_id: str = ""
    status: str = "unknown"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    sigma_x: float = 0.0
    sigma_y: float = 0.0
    sigma_z: float = 0.0
    sigma_vx: float = 0.0
    sigma_vy: float = 0.0
    sigma_vz: float = 0.0
    hits: int = 0
    misses: int = 0
    local_updates: int = 0
    remote_updates: int = 0
    is_local_origin: bool = False
    freshness: float = 0.0
    measurement_age_s: Optional[float] = None
    n_sources: int = 0
    platforms: List[str] = field(default_factory=list)
    #: 支持该航迹的传感器集合（来自溯源，非真值）
    source_sensors: List[str] = field(default_factory=list)
    #: 最近若干条溯源记录（sensor_id / 测量时刻 / msg_id / 年龄 / trace_id）
    recent_sources: List[Dict[str, Any]] = field(default_factory=list)
    #: 是否获得过远端（共享）贡献
    has_remote_contribution: bool = False

    @property
    def position_sigma_norm(self) -> float:
        return (self.sigma_x ** 2 + self.sigma_y ** 2 + self.sigma_z ** 2) ** 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "status": self.status,
            "position": {"x": round(self.x, 3), "y": round(self.y, 3),
                         "z": round(self.z, 3)},
            "velocity": {"vx": round(self.vx, 3), "vy": round(self.vy, 3),
                         "vz": round(self.vz, 3)},
            "sigma_position": {"x": round(self.sigma_x, 3),
                               "y": round(self.sigma_y, 3),
                               "z": round(self.sigma_z, 3)},
            "sigma_velocity": {"vx": round(self.sigma_vx, 3),
                               "vy": round(self.sigma_vy, 3),
                               "vz": round(self.sigma_vz, 3)},
            "hits": self.hits,
            "misses": self.misses,
            "local_updates": self.local_updates,
            "remote_updates": self.remote_updates,
            "is_local_origin": self.is_local_origin,
            "freshness": round(self.freshness, 6),
            "measurement_age_s": self.measurement_age_s,
            "n_sources": self.n_sources,
            "platforms": list(self.platforms),
            "source_sensors": list(self.source_sensors),
            "recent_sources": [dict(x) for x in self.recent_sources],
            "has_remote_contribution": self.has_remote_contribution,
        }


@dataclass
class FusionState:
    """融合/跟踪层状态。"""

    enabled: bool = False
    platform_id: str = ""
    n_tracks: int = 0
    n_confirmed: int = 0
    n_coasting: int = 0
    n_tentative: int = 0
    freshness_mean: float = 0.0
    tracks: List[TrackState] = field(default_factory=list)
    #: 本步融合中心的准入统计
    n_local_measurements: int = 0
    n_remote_measurements: int = 0
    n_kind_rejected: int = 0
    n_stale_rejected: int = 0
    #: 累计航迹管理统计
    tracks_initiated: int = 0
    tracks_dropped: int = 0
    gate_rejected_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "platform_id": self.platform_id,
            "n_tracks": self.n_tracks,
            "n_confirmed": self.n_confirmed,
            "n_coasting": self.n_coasting,
            "n_tentative": self.n_tentative,
            "freshness_mean": round(self.freshness_mean, 6),
            "n_local_measurements": self.n_local_measurements,
            "n_remote_measurements": self.n_remote_measurements,
            "n_kind_rejected": self.n_kind_rejected,
            "n_stale_rejected": self.n_stale_rejected,
            "tracks_initiated": self.tracks_initiated,
            "tracks_dropped": self.tracks_dropped,
            "gate_rejected_total": self.gate_rejected_total,
            "tracks": [t.to_dict() for t in self.tracks],
        }


@dataclass
class CooperationState:
    """协同感知状态：**只报"有没有远端贡献"，不报真值层面的对错**。

    ⚠️ 协同的**收益大小**（RMSE 改善、覆盖提升）必须由离线评测用真值算，
    属于评测通道，**不进** AI 上下文。AI 这里能说的是
    "本地航迹获得/未获得远端贡献""远端测量到达/被丢弃"这类
    可从结构化证据直接读出的事实。
    """

    sharing_enabled: bool = False
    policy: str = "none"
    n_tracks_with_remote: int = 0
    n_tracks_local_only: int = 0
    remote_measurements_arrived: int = 0
    remote_measurements_used: int = 0
    remote_measurements_rejected: int = 0
    #: 远端测量利用率（已到达中真正进入航迹的比例）
    remote_utilization: float = 0.0
    #: 本步是否有航迹"仅靠远端"维持（本地无贡献却拿到了测量）
    tracks_supported_remotely_only: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sharing_enabled": self.sharing_enabled,
            "policy": self.policy,
            "n_tracks_with_remote": self.n_tracks_with_remote,
            "n_tracks_local_only": self.n_tracks_local_only,
            "remote_measurements_arrived": self.remote_measurements_arrived,
            "remote_measurements_used": self.remote_measurements_used,
            "remote_measurements_rejected": self.remote_measurements_rejected,
            "remote_utilization": round(self.remote_utilization, 6),
            "tracks_supported_remotely_only": self.tracks_supported_remotely_only,
            "notes": list(self.notes),
        }


# ----------------------------------------------------------------------
# 输出协议
# ----------------------------------------------------------------------

@dataclass
class Finding:
    code: str
    severity: str
    title: str
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisResult:
    provider: str = "rule"
    model: str = ""
    status: str = STATUS_OK
    severity: str = SEVERITY_INFO
    summary: str = ""
    findings: List[Finding] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.5
    latency_ms: float = 0.0
    generated_at: str = ""
    error: str = ""
    #: v4.5：远端 provider 的证据校验结果（字段/发现码白名单）
    evidence_check_passed: bool = True
    evidence_check_failed: bool = False
    evidence_violations: List[str] = field(default_factory=list)
    #: 因证据校验失败而回退到本地 rule provider 时置 True
    fell_back_to_rule: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "severity": self.severity,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "recommendations": list(self.recommendations),
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "generated_at": self.generated_at,
            "error": self.error,
            "evidence_check_passed": self.evidence_check_passed,
            "evidence_check_failed": self.evidence_check_failed,
            "evidence_violations": list(self.evidence_violations),
            "fell_back_to_rule": self.fell_back_to_rule,
            "schema_version": SCHEMA_VERSION,
        }


@dataclass
class ExplainResult:
    provider: str = "rule"
    model: str = ""
    status: str = STATUS_OK
    step_index: int = 0
    direction: str = "hold"
    explanation: str = ""
    verdict_code: str = ""
    evidence_codes: List[str] = field(default_factory=list)
    used_numbers: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "step_index": self.step_index,
            "direction": self.direction,
            "explanation": self.explanation,
            "verdict_code": self.verdict_code,
            "evidence_codes": list(self.evidence_codes),
            "used_numbers": self.used_numbers,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "schema_version": SCHEMA_VERSION,
        }


@dataclass
class CompareResult:
    provider: str = "rule"
    model: str = ""
    status: str = STATUS_OK
    metric_order: List[str] = field(default_factory=list)
    ranking: List[str] = field(default_factory=list)
    highlights: List[str] = field(default_factory=list)
    tradeoffs: List[str] = field(default_factory=list)
    table: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "metric_order": list(self.metric_order),
            "ranking": list(self.ranking),
            "highlights": list(self.highlights),
            "tradeoffs": list(self.tradeoffs),
            "table": list(self.table),
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "schema_version": SCHEMA_VERSION,
        }


@dataclass
class ReportResult:
    provider: str = "rule"
    model: str = ""
    status: str = STATUS_OK
    title: str = ""
    sections: List[Dict[str, str]] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.5
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "title": self.title,
            "sections": list(self.sections),
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "schema_version": SCHEMA_VERSION,
        }


@dataclass
class ProviderInfo:
    name: str = ""
    model: str = ""
    kind: str = "local"  # local | remote
    requires_api_key: bool = False
    supports: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------

def ok_result_payload(result: Any) -> Dict[str, Any]:
    """统一把结果对象转成 JSON 字典（供 API 返回）。"""
    if hasattr(result, "to_dict"):
        return result.to_dict()
    raise TypeError(f"结果对象不可序列化：{type(result)!r}")
