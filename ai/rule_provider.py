"""RuleProvider：**无需任何 API Key** 的本地认知诊断实现。

它用「规则 + 模板」在所有结构化证据上生成自然语言，
是整个 AI 层的默认 provider 与兜底实现：

* 离线可用、确定性、零依赖、毫秒级；
* 与远程大模型 provider 共享同一套输入输出协议，
  因此「换成真模型」只影响文字表达，不影响任何数值结论；
* 远程 provider 失败时，`AIDiagnosisService` 会自动降级到它。

**它不会编造事实**：每一句结论都直接由 `StateSnapshot` / 反事实证据里的数值拼出。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .provider import AIProvider
from .schema import (
    CompareResult,
    DiagnosisResult,
    ExplainResult,
    FINDING_CODES,
    Finding,
    ProviderInfo,
    ReportResult,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    STATUS_OK,
    StateSnapshot,
)

#: 指标中文名（对比与报告模块共用）
METRIC_CN: Dict[str, str] = {
    "horizon_satisfaction_rate": "探测任务满足率",
    "violation_rate": "约束违反率",
    "avg_tx_power_w": "平均发射功率(W)",
    "cumulative_energy_j": "累计能耗(J)",
    "avg_intercept_prob": "平均截获概率",
    "avg_exposure": "平均累计暴露",
    "cumulative_exposure": "累计暴露",
    "composite_reward": "综合收益",
}

#: 指标方向：True = 越小越好
LOWER_IS_BETTER: Dict[str, bool] = {
    "horizon_satisfaction_rate": False,
    "violation_rate": True,
    "avg_tx_power_w": True,
    "cumulative_energy_j": True,
    "avg_intercept_prob": True,
    "avg_exposure": True,
    "cumulative_exposure": True,
    "composite_reward": False,
}


class RuleProvider(AIProvider):
    """基于规则与模板的本地诊断 provider。"""

    name = "rule"
    model = "rule-engine-v1"

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            model=self.model,
            kind="local",
            requires_api_key=False,
            supports=["diagnose", "explain_decision", "compare_policies", "generate_report"],
            notes="纯规则+模板实现：零依赖、离线可用、结果确定，作为默认与兜底 provider",
        )

    # ------------------------------------------------------------------
    # 能力一：实时态势诊断
    # ------------------------------------------------------------------

    def diagnose(self, snapshot: StateSnapshot) -> DiagnosisResult:
        findings: List[Finding] = []
        s = snapshot

        # --- 探测侧 ---
        pd_margin = s.pd_min - s.required_pd
        if s.task_violated:
            findings.append(
                Finding(
                    code="DETECTION_VIOLATED",
                    severity=SEVERITY_CRITICAL,
                    title="探测任务未达标",
                    message=(
                        f"当前最小探测概率 Pd={s.pd_min:.3f} 低于任务要求 "
                        f"{s.required_pd:.3f}（差 {pd_margin:+.3f}），本步计入约束违反。"
                    ),
                    evidence={"pd_min": s.pd_min, "required_pd": s.required_pd,
                              "margin": round(pd_margin, 6)},
                )
            )
        elif pd_margin < 0.05:
            findings.append(
                Finding(
                    code="DETECTION_AT_RISK",
                    severity=SEVERITY_WARNING,
                    title="探测概率接近门限",
                    message=(
                        f"Pd={s.pd_min:.3f} 仅高于 required_pd={s.required_pd:.3f} "
                        f"{pd_margin:+.3f}，几何或干扰稍有变化就会跌破。"
                    ),
                    evidence={"pd_min": s.pd_min, "required_pd": s.required_pd,
                              "margin": round(pd_margin, 6)},
                )
            )
        else:
            findings.append(
                Finding(
                    code="DETECTION_WELL_WITHIN_MARGIN",
                    severity=SEVERITY_INFO,
                    title="探测余量充足",
                    message=f"Pd={s.pd_min:.3f}，高于要求 {pd_margin:+.3f}。",
                    evidence={"margin": round(pd_margin, 6)},
                )
            )

        # --- 能量侧 ---
        if s.energy is not None:
            e = s.energy
            steps_left = max(0, s.horizon_steps - s.step_index)
            affordable = (
                int(e.remaining_j // e.min_step_energy_j) if e.min_step_energy_j > 0 else 0
            )
            if e.remaining_j <= e.min_step_energy_j * 1.001:
                findings.append(
                    Finding(
                        code="ENERGY_EXHAUSTED",
                        severity=SEVERITY_CRITICAL,
                        title="能量即将耗尽",
                        message=(
                            f"剩余 {e.remaining_j:.2f} J，已不足最低档单步能耗 "
                            f"{e.min_step_energy_j:.2f} J，episode 将终止。"
                        ),
                        evidence={"remaining_j": e.remaining_j,
                                  "min_step_energy_j": e.min_step_energy_j},
                    )
                )
            elif affordable < steps_left:
                findings.append(
                    Finding(
                        code="ENERGY_LOW",
                        severity=SEVERITY_WARNING,
                        title="剩余能量不足以跑完全程",
                        message=(
                            f"剩余 {e.remaining_j:.1f} J 只够最低档再发 {affordable} 步，"
                            f"而任务还剩 {steps_left} 步（预算 {e.budget_j:.0f} J）。"
                        ),
                        evidence={"remaining_j": e.remaining_j, "affordable_steps": affordable,
                                  "steps_left": steps_left},
                    )
                )
            if s.power is not None and s.power.tx_power_w > 0:
                per_step = s.power.tx_power_w
                if per_step * steps_left > e.remaining_j + 1e-9:
                    findings.append(
                        Finding(
                            code="ENERGY_OVERRUN_RISK",
                            severity=SEVERITY_WARNING,
                            title="按当前功率无法支撑到任务结束",
                            message=(
                                f"以当前 {per_step:.1f} W 再发 {steps_left} 步需要 "
                                f"{per_step * steps_left:.1f} J，超过剩余 {e.remaining_j:.1f} J。"
                            ),
                            evidence={"needed_j": round(per_step * steps_left, 3),
                                      "remaining_j": e.remaining_j},
                        )
                    )

        # --- 截获与暴露侧 ---
        if s.exposure >= 0.5:
            findings.append(
                Finding(
                    code="HIGH_EXPOSURE",
                    severity=SEVERITY_WARNING if s.exposure < 0.75 else SEVERITY_CRITICAL,
                    title="累计暴露偏高",
                    message=(
                        f"累计暴露量已达 {s.exposure:.3f}（上限 1.0），"
                        f"敌方侦察证据累积充分，后续被识别与测向的风险显著上升。"
                    ),
                    evidence={"exposure": s.exposure},
                )
            )
        if s.pint_eff >= 0.6:
            findings.append(
                Finding(
                    code="INTERCEPT_RISK_HIGH",
                    severity=SEVERITY_WARNING,
                    title="有效截获概率偏高",
                    message=(
                        f"Pint_eff={s.pint_eff:.3f}（瞬时 {s.pint_inst:.3f} + 累计暴露 "
                        f"{s.exposure:.3f} 的并集），已处于易被截获状态。"
                    ),
                    evidence={"pint_eff": s.pint_eff, "pint_inst": s.pint_inst,
                              "exposure": s.exposure},
                )
            )
        elif s.pint_eff <= 0.35:
            findings.append(
                Finding(
                    code="LOW_INTERCEPT_STATE",
                    severity=SEVERITY_INFO,
                    title="处于低截获状态",
                    message=f"Pint_eff={s.pint_eff:.3f}，当前辐射不易被截获。",
                    evidence={"pint_eff": s.pint_eff},
                )
            )

        # --- 干扰侧 ---
        for jammer in s.jammers:
            if not jammer.active:
                continue
            if jammer.mode == "adaptive":
                findings.append(
                    Finding(
                        code="JAMMING_ESCALATED",
                        severity=SEVERITY_WARNING if jammer.jam_noise_ratio >= 1.0
                        else SEVERITY_INFO,
                        title="自适应干扰机正在压制",
                        message=(
                            f"干扰机 {jammer.jammer_id} 当前动作「{jammer.action_cn or jammer.action}」，"
                            f"威胁度 {jammer.threat:.3f}，J/N={jammer.jam_noise_ratio:.2f}。"
                            f"它是按 ESM 累计暴露、雷达辐射强度与干扰有效性规则切换动作的**规则型**对手。"
                        ),
                        evidence={"jammer_id": jammer.jammer_id, "action": jammer.action,
                                  "threat": jammer.threat,
                                  "jam_noise_ratio": jammer.jam_noise_ratio},
                    )
                )
            else:
                findings.append(
                    Finding(
                        code="JAMMING_ESCALATED",
                        severity=SEVERITY_INFO,
                        title="固定时间窗干扰进行中",
                        message=(
                            f"干扰机 {jammer.jammer_id} 处于预置工作窗内，"
                            f"J/N={jammer.jam_noise_ratio:.2f}。"
                        ),
                        evidence={"jammer_id": jammer.jammer_id,
                                  "jam_noise_ratio": jammer.jam_noise_ratio},
                    )
                )

        # --- 功率使用效率 ---
        if s.power is not None and s.power.feasible_levels:
            max_feasible = s.power.feasible_levels[-1]
            if s.power.level >= max_feasible:
                findings.append(
                    Finding(
                        code="POWER_AT_MAX_FEASIBLE",
                        severity=SEVERITY_WARNING,
                        title="功率已到可行上限",
                        message=(
                            f"当前档位 {s.power.level}（{s.power.tx_power_w:.1f} W）"
                            f"已是剩余能量允许的最高档，无法再靠提功率换探测性能。"
                        ),
                        evidence={"level": s.power.level, "max_feasible": max_feasible},
                    )
                )

        # ---------------- v4.0：观测链路与可信决策 ----------------
        self._diagnose_observability(s, findings)
        self._diagnose_trust(s, findings)

        # ---------------- v4.5：测量 → 通信 → 融合 → 协同 ----------------
        self._diagnose_measurement(s, findings)
        self._diagnose_communication(s, findings)
        self._diagnose_fusion(s, findings)
        self._diagnose_cooperation(s, findings)
        # ---------------- v4.5 P2：系统级压力（机动/交接/时序/传感器健康）--------
        self._diagnose_system_stress(s, findings)

        severity = SEVERITY_INFO
        for finding in findings:
            if finding.severity == SEVERITY_CRITICAL:
                severity = SEVERITY_CRITICAL
                break
            if finding.severity == SEVERITY_WARNING:
                severity = SEVERITY_WARNING

        summary = self._compose_summary(s, severity, findings)
        recommendations = self._compose_recommendations(s, findings)

        return DiagnosisResult(
            provider=self.name,
            model=self.model,
            status=STATUS_OK,
            severity=severity,
            summary=summary,
            findings=findings,
            recommendations=recommendations,
            confidence=0.85,
            latency_ms=0.0,
        )

    @staticmethod
    def _diagnose_observability(s: StateSnapshot, findings: List[Finding]) -> None:
        """v4.0：解释「智能体为什么会看不准」。

        这些发现完全基于观测模型**上报**的量，不含真值。
        """
        obs = s.observability
        if obs is None or obs.mode != "pomdp":
            return

        if obs.dropped_fields:
            findings.append(
                Finding(
                    code="OBSERVATION_MISSING",
                    severity=SEVERITY_WARNING,
                    title="本步存在丢测",
                    message=(
                        f"有 {len(obs.dropped_fields)} 项状态量本步未测到"
                        f"（{', '.join(obs.dropped_fields[:4])}"
                        f"{'…' if len(obs.dropped_fields) > 4 else ''}），"
                        "相关估计沿用上一次测量，误差会随时间累积。"
                    ),
                    evidence={
                        "dropped_fields": list(obs.dropped_fields),
                        "observation_quality": round(obs.observation_quality, 4),
                    },
                )
            )

        if obs.observation_quality < 0.55:
            findings.append(
                Finding(
                    code="OBSERVATION_DEGRADED",
                    severity=SEVERITY_WARNING,
                    title="观测链路退化",
                    message=(
                        f"当前观测质量仅 {obs.observation_quality:.2f}（1.0 为最优）。"
                        "在此条件下任何策略（含 AI）的状态判断都不可靠，"
                        "应降低对单步决策的信任。"
                    ),
                    evidence={
                        "observation_quality": round(obs.observation_quality, 4),
                        "stale_fields": list(obs.stale_fields),
                    },
                )
            )

        esm_sigma = obs.sigma.get("interceptor_range")
        if esm_sigma is not None and esm_sigma > 0.0:
            findings.append(
                Finding(
                    code="ESM_POSITION_UNKNOWN",
                    severity=SEVERITY_INFO,
                    title="侦察机位置不可直接观测",
                    message=(
                        f"侦察机距离估计标准差约 {esm_sigma:.0f} m，"
                        "真实位置不在观测向量内。基于该估计做的暴露与功率判断"
                        "存在结构性误差，无法通过更聪明的算法消除。"
                    ),
                    evidence={"interceptor_range_sigma_m": esm_sigma},
                )
            )

        exposure_sigma = obs.sigma.get("exposure")
        if exposure_sigma is not None and exposure_sigma > 0.0:
            findings.append(
                Finding(
                    code="EXPOSURE_ESTIMATE_UNCERTAIN",
                    severity=SEVERITY_INFO,
                    title="累计暴露只能估计",
                    message=(
                        f"累计暴露的估计标准差约 {exposure_sigma:.3f}，"
                        "它是低截获性能的核心指标，却无法被精确观测。"
                    ),
                    evidence={"exposure_sigma": exposure_sigma},
                )
            )

    @staticmethod
    def _diagnose_trust(s: StateSnapshot, findings: List[Finding]) -> None:
        """v4.0：解释「AI 为什么不确定、为什么触发回退」。"""
        trust = s.trust
        if trust is None:
            return

        if trust.uncertainty_source == "ensemble" and trust.q_std_max is not None:
            if trust.disagreement is not None and trust.disagreement >= 0.4:
                findings.append(
                    Finding(
                        code="AI_HIGH_UNCERTAINTY",
                        severity=SEVERITY_WARNING,
                        title="集成各成员判断分歧",
                        message=(
                            f"{trust.ensemble_size} 个网络中有 "
                            f"{trust.disagreement * 100:.0f}% 的成员给出的最优动作"
                            "与集成结果不一致，说明该状态下的估值不可靠。"
                        ),
                        evidence={
                            "ensemble_size": trust.ensemble_size,
                            "disagreement": trust.disagreement,
                            "q_std_max": trust.q_std_max,
                        },
                    )
                )
            if trust.ood_score is not None and trust.ood_score >= 3.0:
                findings.append(
                    Finding(
                        code="AI_OOD_INPUT",
                        severity=SEVERITY_WARNING,
                        title="当前观测偏离训练分布",
                        message=(
                            f"观测的标准化偏离度达 {trust.ood_score:.2f}"
                            "（训练分布内通常在 1 附近）。网络在这种输入上没有"
                            "外推依据，其 Q 值排序可能只是插值假象。"
                        ),
                        evidence={"ood_score": trust.ood_score},
                    )
                )
            if trust.q_margin is not None and 0.0 <= trust.q_margin < 0.05:
                findings.append(
                    Finding(
                        code="AI_SMALL_Q_MARGIN",
                        severity=SEVERITY_INFO,
                        title="AI 决策缺乏区分度",
                        message=(
                            f"最优与次优动作的 Q 值仅差 {trust.q_margin:.4f}，"
                            "动作排序接近随机，此时的 argmax 不应被当作强结论。"
                        ),
                        evidence={"q_margin": trust.q_margin},
                    )
                )

        if trust.decision_mode == "fallback_rule":
            findings.append(
                Finding(
                    code="AI_FALLBACK_TRIGGERED",
                    severity=SEVERITY_WARNING,
                    title="已回退到规则策略",
                    message=(
                        f"触发原因：{trust.reason_cn}。本步控制权已交给可解释的"
                        "规则策略。注意：回退**不保证**结果更好——规则策略在部分"
                        "可观测条件下同样会错，回退的价值在于行为可解释、有界。"
                    ),
                    evidence={
                        "triggered": list(trust.triggered),
                        "reason_code": trust.reason_code,
                        "fallback_rate": trust.fallback_rate,
                    },
                )
            )
        elif trust.decision_mode == "shield":
            findings.append(
                Finding(
                    code="AI_SHIELD_APPLIED",
                    severity=SEVERITY_INFO,
                    title="安全护盾生效",
                    message=(
                        f"AI 的动作被抬升到满足探测要求的最低档位。"
                        f"触发原因：{trust.reason_cn}。"
                        "护盾只抬升功率、从不降低功率，因此不会增加任务失败风险，"
                        "但会牺牲一点低截获性能。"
                    ),
                    evidence={
                        "triggered": list(trust.triggered),
                        "shield_rate": trust.shield_rate,
                    },
                )
            )

    # ------------------------------------------------------------------
    # v4.5：测量 / 通信 / 融合 / 协同
    #
    # 设计纪律：
    #  * 只读 `s.*_state`（由 env 的结构化访问器填充，**不含真值**）；
    #  * 每一条发现都附**数值证据**，措辞与证据字段一一对应；
    #  * 「为什么现在没有目标信息」必须逐原因分开，不得笼统写成"观测退化"。
    # ------------------------------------------------------------------

    #: 原因码 -> 发现码（测量层）
    _MEASUREMENT_REASON_CODES: Dict[str, str] = {
        "out_of_fov": "TARGET_OUT_OF_FOV",
        "beyond_range": "TARGET_BEYOND_RANGE",
        "occluded": "TARGET_OCCLUDED",
        "not_updated": "SENSOR_NOT_UPDATED",
        "missed_detection": "MISSED_DETECTION",
        "sensor_unavailable": "SENSOR_UNAVAILABLE",
    }

    #: 通信丢弃原因 -> 发现码
    _COMM_DROP_CODES: Dict[str, str] = {
        "lost": "COMM_PACKET_LOST",
        "expired": "COMM_MESSAGE_EXPIRED",
        "link_down": "COMM_LINK_DOWN",
        "queue_full": "COMM_QUEUE_FULL",
    }

    @staticmethod
    def _diagnose_measurement(s: StateSnapshot, findings: List[Finding]) -> None:
        """逐原因说明「为什么现在没有目标信息」。"""
        ms = s.measurement_state
        if ms is None:
            return

        # 逐原因：占比超过 1% 就单独成条，避免被"整体观测退化"淹没
        for reason, code in RuleProvider._MEASUREMENT_REASON_CODES.items():
            count = int(ms.reason_counts.get(reason, 0))
            if count <= 0:
                continue
            rate = float(ms.reason_rates.get(reason, 0.0))
            if rate < 0.01:
                continue
            severity = SEVERITY_WARNING if rate >= 0.3 else SEVERITY_INFO
            findings.append(
                Finding(
                    code=code,
                    severity=severity,
                    title=FINDING_CODES.get(code, code),
                    message=(
                        f"本步有 {count} 项判定属于「{FINDING_CODES.get(code, code)}」"
                        f"（占全部判定的 {rate * 100:.1f}%）。"
                        f"可用测量 {ms.n_measurements} 条，其中新产生 {ms.n_fresh} 条。"
                    ),
                    evidence={
                        "reason": reason, "count": count, "rate": round(rate, 6),
                        "n_measurements": ms.n_measurements,
                        "n_fresh": ms.n_fresh,
                        "observation_quality": ms.observation_quality,
                    },
                )
            )

        # 传感器不可用（显式列出是哪一台）
        unavailable = [x["sensor_id"] for x in ms.sensors if not x.get("available", True)]
        if unavailable:
            findings.append(
                Finding(
                    code="SENSOR_UNAVAILABLE",
                    severity=SEVERITY_WARNING,
                    title="存在不可用的传感器",
                    message=f"以下传感器当前不可用：{', '.join(unavailable)}。",
                    evidence={"unavailable_sensors": unavailable},
                )
            )

        # 完全没有测量
        if ms.n_measurements == 0:
            findings.append(
                Finding(
                    code="MISSED_DETECTION",
                    severity=SEVERITY_WARNING,
                    title="本步没有任何可用测量",
                    message=(
                        "本步既没有新测量也没有可沿用的旧测量。"
                        "具体原因见上方的逐原因分解——"
                        "不要把它笼统理解为「观测退化」："
                        "指向、距离、遮挡、更新时刻与概率丢测的工程对策完全不同。"
                    ),
                    evidence={"n_measurements": 0,
                              "reason_counts": dict(ms.reason_counts)},
                )
            )

    @staticmethod
    def _diagnose_communication(s: StateSnapshot, findings: List[Finding]) -> None:
        """说明「这条远端测量有没有到」。"""
        cs = s.communication_state
        if cs is None or not cs.n_links:
            return

        for reason, code in RuleProvider._COMM_DROP_CODES.items():
            count = int(cs.drop_reasons.get(reason, 0))
            if count <= 0:
                continue
            findings.append(
                Finding(
                    code=code,
                    severity=SEVERITY_WARNING if code != "COMM_MESSAGE_EXPIRED"
                    else SEVERITY_INFO,
                    title=FINDING_CODES.get(code, code),
                    message=(
                        f"累计 {count} 条消息因「{FINDING_CODES.get(code, code)}」"
                        f"未能送达。当前共发送 {cs.n_messages_sent} 条，"
                        f"送达 {cs.n_delivered} 条（送达率 {cs.delivery_rate:.3f}）。"
                    ),
                    evidence={"drop_reason": reason, "count": count,
                              "n_messages_sent": cs.n_messages_sent,
                              "n_delivered": cs.n_delivered,
                              "delivery_rate": round(cs.delivery_rate, 6)},
                )
            )

        if cs.n_arrived_stale > 0:
            findings.append(
                Finding(
                    code="REMOTE_MEASUREMENT_DELAYED",
                    severity=SEVERITY_WARNING,
                    title=FINDING_CODES["REMOTE_MEASUREMENT_DELAYED"],
                    message=(
                        f"本步有 {cs.n_arrived_stale} 条远端测量**已经到达**，"
                        f"但因超过时效被融合层拒绝。它们不是丢包，"
                        f"而是「到了但太旧」——两者对策不同。"
                    ),
                    evidence={"n_arrived_stale": cs.n_arrived_stale,
                              "latency_mean_s": cs.latency_mean_s,
                              "latency_p95_s": cs.latency_p95_s},
                )
            )

        if cs.n_in_flight > 0:
            findings.append(
                Finding(
                    code="REMOTE_MEASUREMENT_DELAYED",
                    severity=SEVERITY_INFO,
                    title="仍有消息在途",
                    message=(
                        f"当前有 {cs.n_in_flight} 条消息尚未到达"
                        f"（平均延迟 {cs.latency_mean_s:.3f}s，"
                        f"p95 {cs.latency_p95_s:.3f}s）。"
                        "这些消息的内容**不得**被提前使用。"
                    ),
                    evidence={"n_in_flight": cs.n_in_flight,
                              "latency_mean_s": cs.latency_mean_s},
                )
            )

    @staticmethod
    def _diagnose_fusion(s: StateSnapshot, findings: List[Finding]) -> None:
        """说明「为什么航迹进入外推」「为什么航迹不确定」。"""
        fs = s.fusion_state
        if fs is None or not fs.enabled:
            return

        coasting = [t for t in fs.tracks if t.status == "coasting"]
        if coasting:
            findings.append(
                Finding(
                    code="TRACK_COASTING",
                    severity=SEVERITY_WARNING,
                    title=FINDING_CODES["TRACK_COASTING"],
                    message=(
                        f"{len(coasting)} 条航迹当前没有观测支撑，靠匀速模型外推："
                        f"{', '.join(t.track_id for t in coasting[:4])}。"
                        "外推期间位置不确定度会随时间增长。"
                    ),
                    evidence={
                        "coasting_tracks": [t.track_id for t in coasting],
                        "n_tracks": fs.n_tracks,
                        "details": [
                            {"track_id": t.track_id, "misses": t.misses,
                             "measurement_age_s": t.measurement_age_s,
                             "sigma_x": round(t.sigma_x, 3),
                             "sigma_y": round(t.sigma_y, 3)}
                            for t in coasting[:4]
                        ],
                    },
                )
            )

        uncertain = [t for t in fs.tracks if t.position_sigma_norm > 200.0]
        if uncertain:
            findings.append(
                Finding(
                    code="TRACK_UNCERTAIN",
                    severity=SEVERITY_WARNING,
                    title=FINDING_CODES["TRACK_UNCERTAIN"],
                    message=(
                        f"{len(uncertain)} 条航迹的位置不确定度超过 200 m："
                        + ", ".join(
                            f"{t.track_id}(σ={t.position_sigma_norm:.1f} m)"
                            for t in uncertain[:4]
                        )
                    ),
                    evidence={
                        "uncertain_tracks": [
                            {"track_id": t.track_id,
                             "sigma_norm_m": round(t.position_sigma_norm, 3),
                             "freshness": round(t.freshness, 4),
                             "n_sources": t.n_sources}
                            for t in uncertain[:4]
                        ],
                    },
                )
            )

        if fs.gate_rejected_total > 0 and fs.n_tracks >= 2:
            findings.append(
                Finding(
                    code="ASSOCIATION_AMBIGUOUS",
                    severity=SEVERITY_INFO,
                    title=FINDING_CODES["ASSOCIATION_AMBIGUOUS"],
                    message=(
                        f"累计 {fs.gate_rejected_total} 次测量未通过关联门限，"
                        f"而当前存在 {fs.n_tracks} 条航迹。"
                        "多目标且相互接近时，最近邻关联可能把测量串到错误航迹上；"
                        "该风险在本版**尚未用压力场景量化**。"
                    ),
                    evidence={"gate_rejected_total": fs.gate_rejected_total,
                              "n_tracks": fs.n_tracks},
                )
            )

    @staticmethod
    def _diagnose_cooperation(s: StateSnapshot, findings: List[Finding]) -> None:
        """说明「本地看不见但远端补回了什么」。"""
        co = s.cooperation_state
        if co is None:
            return

        contributing = [t for t in (s.fusion_state.tracks if s.fusion_state else [])
                        if t.has_remote_contribution]
        if contributing:
            findings.append(
                Finding(
                    code="REMOTE_SENSOR_CONTRIBUTION",
                    severity=SEVERITY_INFO,
                    title=FINDING_CODES["REMOTE_SENSOR_CONTRIBUTION"],
                    message=(
                        f"{len(contributing)} 条航迹获得过远端共享测量的贡献："
                        + ", ".join(
                            f"{t.track_id}(远端更新 {t.remote_updates} 次)"
                            for t in contributing[:4]
                        )
                    ),
                    evidence={
                        "tracks_with_remote": [t.track_id for t in contributing],
                        "policy": co.policy,
                        "remote_measurements_used": co.remote_measurements_used,
                        "remote_utilization": round(co.remote_utilization, 6),
                    },
                )
            )

        if co.tracks_supported_remotely_only > 0:
            findings.append(
                Finding(
                    code="COOPERATIVE_TRACK_RECOVERED",
                    severity=SEVERITY_INFO,
                    title=FINDING_CODES["COOPERATIVE_TRACK_RECOVERED"],
                    message=(
                        f"{co.tracks_supported_remotely_only} 条航迹**仅靠远端共享测量**维持"
                        "（本地传感器对这些航迹没有任何贡献）。"
                        "这说明本地当前看不见的目标由远端补上了——"
                        "注意这是**结构性事实**（溯源可查），"
                        "而不是精度收益；收益大小须由离线评测用真值计算。"
                    ),
                    evidence={
                        "tracks_supported_remotely_only":
                            co.tracks_supported_remotely_only,
                        "policy": co.policy,
                        "remote_measurements_used": co.remote_measurements_used,
                    },
                )
            )

        if co.sharing_enabled and co.remote_measurements_arrived > 0 and \
                co.remote_measurements_rejected > 0:
            findings.append(
                Finding(
                    code="REMOTE_MEASUREMENT_DELAYED",
                    severity=SEVERITY_INFO,
                    title="部分远端测量到达但未被采用",
                    message=(
                        f"本步到达 {co.remote_measurements_arrived} 条远端测量，"
                        f"实际进入航迹 {co.remote_measurements_used} 条，"
                        f"被拒 {co.remote_measurements_rejected} 条。"
                        "被拒的常见原因是时效不足或观测对象不匹配。"
                    ),
                    evidence={
                        "arrived": co.remote_measurements_arrived,
                        "used": co.remote_measurements_used,
                        "rejected": co.remote_measurements_rejected,
                    },
                )
            )

        if not co.sharing_enabled:
            findings.append(
                Finding(
                    code="REMOTE_SENSOR_CONTRIBUTION",
                    severity=SEVERITY_INFO,
                    title="当前未启用多平台共享",
                    message=(
                        "本平台只使用自己的测量，因此任何「看不见的目标」都无法由"
                        "远端补回。这不是观测退化，而是信息获取方式的选择。"
                    ),
                    evidence={"sharing_enabled": False, "policy": co.policy},
                )
            )

    @staticmethod
    def _compose_summary(
        s: StateSnapshot, severity: str, findings: List[Finding]
    ) -> str:
        energy_txt = (
            f"剩余能量 {s.energy.remaining_j:.1f}/{s.energy.budget_j:.0f} J"
            if s.energy
            else "能量未知"
        )
        power_txt = (
            f"当前功率 {s.power.tx_power_w:.1f} W（档位 {s.power.level}）"
            if s.power
            else "功率未知"
        )
        level_cn = {
            SEVERITY_CRITICAL: "态势紧急",
            SEVERITY_WARNING: "需要关注",
            SEVERITY_INFO: "态势平稳",
        }.get(severity, "态势未知")
        top = [f.title for f in findings if f.severity != SEVERITY_INFO][:3]
        extra = f"；主要问题：{'、'.join(top)}" if top else ""
        return (
            f"t={s.time:.1f}s（第 {s.step_index}/{s.horizon_steps} 步）{level_cn}。"
            f"Pd={s.pd_min:.3f}（要求 {s.required_pd:.3f}），"
            f"Pint_eff={s.pint_eff:.3f}，暴露={s.exposure:.3f}，"
            f"{power_txt}，{energy_txt}{extra}。"
        )

    @staticmethod
    def _compose_recommendations(
        s: StateSnapshot, findings: List[Finding]
    ) -> List[str]:
        codes = {f.code for f in findings}
        recs: List[str] = []
        if "DETECTION_VIOLATED" in codes and "POWER_AT_MAX_FEASIBLE" not in codes:
            recs.append("提高发射功率以满足探测要求，但需同时评估暴露上升的代价。")
        if "ENERGY_LOW" in codes or "ENERGY_OVERRUN_RISK" in codes:
            recs.append("降低平均功率或主动放弃部分最贵的任务步，把能量留给后段。")
        if "HIGH_EXPOSURE" in codes:
            recs.append("在不违反探测约束的前提下压低功率，让累计暴露随时间回落。")
        if "POWER_INEFFICIENT" in codes or "DETECTION_WELL_WITHIN_MARGIN" in codes:
            recs.append("探测余量充足，可尝试降一档功率以同时降低暴露与能耗。")
        if not recs:
            recs.append("维持当前策略，继续观察能量与暴露的演化。")
        return recs

    # ------------------------------------------------------------------
    # 能力二：策略解释（只允许引用反事实证据）
    # ------------------------------------------------------------------

    def explain_decision(self, payload: Dict[str, Any]) -> ExplainResult:
        report = payload.get("counterfactual") or payload
        verdict = report.get("verdict", {}) or {}
        direction = report.get("direction", "hold")
        code = verdict.get("code", "")
        reasons = verdict.get("reason", "")
        evidence_codes = list(verdict.get("evidence_codes", []) or [])
        chosen = report.get("chosen", {}) or {}
        legend = verdict.get("evidence_legend", {}) or {}

        head = {
            "raise": "本步**升功率**",
            "lower": "本步**降功率**",
            "hold": "本步**维持功率**",
        }.get(direction, "本步维持功率")

        lines: List[str] = []
        lines.append(
            f"{head}：执行档位 {chosen.get('level')}"
            f"（{float(chosen.get('tx_power_w', 0.0)):.1f} W），判定代码 `{code}`。"
        )
        if reasons:
            lines.append(f"依据：{reasons}。")

        # 引用反事实对比表（只列数值，不加工）
        rows = report.get("counterfactuals", []) or []
        if rows:
            picked = []
            for row in rows:
                if not row.get("feasible", True):
                    continue
                picked.append(
                    f"{float(row['tx_power_w']):.0f}W → Pd {float(row['pd_min']):.3f}、"
                    f"Pint_eff {float(row['pint_eff']):.3f}、"
                    f"暴露(下一步) {float(row['exposure_next']):.4f}、"
                    f"能耗 {float(row['step_energy_j']):.1f}J、"
                    f"单步收益 {float(row['immediate_reward']):+.4f}"
                )
            if picked:
                lines.append("反事实试算（同一步、只改功率）：" + "；".join(picked) + "。")

        if evidence_codes:
            labels = [legend.get(c, c) for c in evidence_codes]
            lines.append("证据代码：" + "、".join(f"{c}（{lab}）" if lab != c else c
                                                 for c, lab in zip(evidence_codes, labels)) + "。")

        used = {
            "pd_min": (report.get("state", {}) or {}).get("pd_min"),
            "required_pd": (report.get("state", {}) or {}).get("required_pd"),
            "pint_eff": (report.get("state", {}) or {}).get("pint_eff"),
            "exposure": (report.get("state", {}) or {}).get("exposure"),
            "remaining_energy_j": (report.get("state", {}) or {}).get("remaining_energy_j"),
            "candidates": [r.get("tx_power_w") for r in rows],
        }

        return ExplainResult(
            provider=self.name,
            model=self.model,
            status=STATUS_OK,
            step_index=int(report.get("step_index", 0)),
            direction=direction,
            explanation=" ".join(lines),
            verdict_code=code,
            evidence_codes=evidence_codes,
            used_numbers=used,
            confidence=0.9,
        )

    # ------------------------------------------------------------------
    # 能力五/六（v4.5）：证据链只读解释
    # ------------------------------------------------------------------

    def explain_track(self, payload: Dict[str, Any]) -> DiagnosisResult:
        """解释单条航迹：只依据结构化证据，不读真值。"""
        from ai.track_explain import explain_track as _explain_track

        return _explain_track(payload)

    def explain_cooperation(self, payload: Dict[str, Any]) -> DiagnosisResult:
        """解释协同感知：只讲结构性事实，不宣称收益（收益属离线评测）。"""
        from ai.track_explain import explain_cooperation as _explain_cooperation

        return _explain_cooperation(payload)

    # ------------------------------------------------------------------
    # 系统级压力诊断（v4.5 P2 第二阶段）
    # ------------------------------------------------------------------

    @staticmethod
    def _diagnose_system_stress(s: StateSnapshot,
                                findings: List[Finding]) -> None:
        """把「机动失配 / 交接 / 时序 / 传感器健康」四类事实说清楚。

        只依据 `system_stress_state` 里的结构化证据；证据缺失时什么都不说
        （**不猜**）。所有数值都取自证据字段，因此可被
        `ai/evidence_check.py` 逐项溯源。
        """
        state = getattr(s, "system_stress_state", None)
        if not state:
            return

        # --- ① 机动导致运动模型失配 ---
        maneuver = state.get("maneuver") or {}
        for entry in maneuver.get("tracks", []):
            normalised = entry.get("normalised_residual") or 0.0
            inconsistent = bool(entry.get("innovation_inconsistent"))
            if not (inconsistent or normalised > 3.0):
                continue
            findings.append(Finding(
                code="MANEUVER_MODEL_MISMATCH", severity=SEVERITY_WARNING,
                title=FINDING_CODES["MANEUVER_MODEL_MISMATCH"],
                message=(
                    f"航迹 {entry['track_id']} 最新测量的预测残差 "
                    f"{entry['residual_m']:.1f} m，是该传感器上报标准差 "
                    f"{entry['reported_sigma_m']:.1f} m 的 {normalised:.2f} 倍，"
                    f"新息 {entry['innovation_mahalanobis_sq']:.2f} 已超出"
                    f"{7.815:.3f}（3 自由度卡方 95% 分位）。"
                    "这是**运动模型失配的特征**：常速度模型跟不上目标机动。"
                    "本证据只给出现象，**不断言**目标一定在机动。"
                ),
                evidence={
                    "track_id": entry["track_id"],
                    "residual_m": entry["residual_m"],
                    "reported_sigma_m": entry["reported_sigma_m"],
                    "normalised_residual": entry["normalised_residual"],
                    "innovation_mahalanobis_sq":
                        entry["innovation_mahalanobis_sq"],
                    "misses": entry.get("misses", 0),
                },
            ))
        if maneuver.get("n_mismatch_suspected", 0) > 0:
            findings.append(Finding(
                code="INNOVATION_INCONSISTENT", severity=SEVERITY_INFO,
                title=FINDING_CODES["INNOVATION_INCONSISTENT"],
                message=(
                    f"共 {maneuver['n_mismatch_suspected']} 条航迹的残差与上报"
                    "协方差不一致。两种可能必须并列：**运动模型不匹配**，"
                    "或**上报的协方差偏小**（噪声低估）；"
                    "仅凭新息无法在两者之间做取舍。"
                ),
                evidence={
                    "n_mismatch_suspected": maneuver["n_mismatch_suspected"],
                    "mismatch_track_ids": maneuver.get("mismatch_track_ids", []),
                },
            ))

        # --- ② 航迹正在跨传感器交接 ---
        handover = state.get("handover") or {}
        if handover.get("n_handover_transitions", 0) > 0:
            transitions = handover.get("transitions") or []
            first = transitions[0] if transitions else {}
            findings.append(Finding(
                code="TRACK_HANDOVER_COMPLETED", severity=SEVERITY_INFO,
                title=FINDING_CODES["TRACK_HANDOVER_COMPLETED"],
                message=(
                    f"共 {handover['n_handover_transitions']} 次跨传感器来源切换，"
                    f"涉及 {handover['n_sensors_contributing']} 个传感器；"
                    f"例如航迹 {first.get('track_id')} 的来源在 "
                    f"{first.get('at_s')} s 从 {first.get('from_sensor')} "
                    f"切换到 {first.get('to_sensor')}。"
                    "这说明**新传感器的测量确实进入了同一条航迹**"
                    "（航迹接力），而不是两个传感器各建一条航迹。"
                ),
                evidence={
                    "n_handover_transitions": handover["n_handover_transitions"],
                    "n_sensors_contributing":
                        handover["n_sensors_contributing"],
                    "track_id": first.get("track_id"),
                    "from_sensor": first.get("from_sensor"),
                    "to_sensor": first.get("to_sensor"),
                    "at_s": first.get("at_s"),
                },
            ))
        per_sensor = handover.get("per_sensor") or {}
        # 只有"**链路确实存在**、但所有航迹来源仍只来自一个传感器"时，
        # 才能说交接失败。单雷达场景下没有第二条链路，
        # 报"交接失败"就是误报（这一条很容易写错）。
        n_links = int((state.get("timing") or {}).get("n_links", 0) or 0)
        if len(per_sensor) == 1 and n_links > 0:
            findings.append(Finding(
                code="TRACK_HANDOVER_FAILED", severity=SEVERITY_WARNING,
                title=FINDING_CODES["TRACK_HANDOVER_FAILED"],
                message=(
                    f"存在 {n_links} 条通信链路，但所有航迹来源都只来自一个"
                    f"传感器（{list(per_sensor)[0]}），没有任何跨传感器接力。"
                    "这提示**远端信息从未进入航迹**——"
                    "需检查远端测量是否因延迟、过期或时效门限被拦下。"
                ),
                evidence={"sensors_contributing": sorted(per_sensor),
                          "n_links": n_links},
            ))

        # --- ③ 通信时序：中断 / 突发 / 恢复拥塞 / 乱序 ---
        timing = state.get("timing") or {}
        if timing.get("link_outage_observed"):
            findings.append(Finding(
                code="COMM_LINK_OUTAGE", severity=SEVERITY_WARNING,
                title=FINDING_CODES["COMM_LINK_OUTAGE"],
                message=(
                    f"链路中断期间丢弃了 {timing['n_outage_dropped']} 条消息，"
                    f"送达率 {timing['delivery_rate']:.4f}。"
                    "中断期间远端信息**根本不可用**，航迹只能靠本地测量与预测维持。"
                ),
                evidence={"n_outage_dropped": timing["n_outage_dropped"],
                          "delivery_rate": timing["delivery_rate"]},
            ))
        if timing.get("burst_loss_observed"):
            findings.append(Finding(
                code="COMM_BURST_LOSS", severity=SEVERITY_INFO,
                title=FINDING_CODES["COMM_BURST_LOSS"],
                message=(
                    f"突发丢包共丢弃 {timing['n_burst_dropped']} 条消息。"
                    "突发与独立丢包不同：它是**连续**的信息空洞，"
                    "在同等丢包率下造成的航迹空窗更长。"
                ),
                evidence={"n_burst_dropped": timing["n_burst_dropped"],
                          "delivery_rate": timing["delivery_rate"]},
            ))
        if timing.get("congestion_observed"):
            findings.append(Finding(
                code="COMM_RECOVERY_CONGESTION", severity=SEVERITY_INFO,
                title=FINDING_CODES["COMM_RECOVERY_CONGESTION"],
                message=(
                    f"链路恢复后拥塞又丢弃了 {timing['n_congestion_dropped']} 条消息，"
                    f"平均延迟 {timing['latency_mean_s']:.3f} s。"
                    "恢复不等于立刻恢复可用带宽。"
                ),
                evidence={
                    "n_congestion_dropped": timing["n_congestion_dropped"],
                    "latency_mean_s": timing["latency_mean_s"],
                },
            ))
        if timing.get("out_of_order_observed"):
            oosm = timing.get("oosm") or {}
            policy_cn = oosm.get("policy_cn") or oosm.get("policy") or "未知策略"
            findings.append(Finding(
                code="MEASUREMENT_OUT_OF_ORDER", severity=SEVERITY_WARNING,
                title=FINDING_CODES["MEASUREMENT_OUT_OF_ORDER"],
                message=(
                    f"乱序到达率 {timing['out_of_order_rate']:.4f}，"
                    f"最大乱序滞后 {timing['max_reorder_lag_s']:.3f} s；"
                    f"当前时序策略为「{policy_cn}」，"
                    f"迟到应用 {oosm.get('n_released_late', 0)} 条"
                    f"（平均扣留 {oosm.get('mean_hold_s', 0.0):.2f} s）。"
                    "乱序的风险是**先用新信息、再用旧信息**更新同一条航迹，"
                    "旧包会把状态往回拽。"
                ),
                evidence={
                    "out_of_order_rate": timing["out_of_order_rate"],
                    "max_reorder_lag_s": timing["max_reorder_lag_s"],
                    "n_released_late": oosm.get("n_released_late", 0),
                    "mean_hold_s": oosm.get("mean_hold_s", 0.0),
                },
            ))

        # --- ④ 传感器长期残差异常（可能的系统偏差）---
        health_state = state.get("sensor_health") or {}
        for sensor_id, entry in (health_state.get("per_sensor") or {}).items():
            score = entry.get("normalised_residual_mean") or 0.0
            # 判据与评测层保持一致：绝对 3.0 以上，**或**落入相对可疑名单。
            # ⚠️ AI 只看到航迹**保留的**来源窗口（每条航迹最多 16 条），
            # 因此它的分数与离线评测（用全部来源）不会完全相同——
            # 这一点必须说明，否则两个数字对不上会被当成 bug。
            relative_hit = sensor_id in (health_state.get("suspicious_sensor_ids")
                                         or [])
            if score <= 3.0 and not relative_hit:
                continue
            basis = ("绝对值 > 3.0" if score > 3.0
                     else "相对同场景同伴偏大（>1.5 倍且 >2.0）")
            findings.append(Finding(
                code="SENSOR_BIAS_SUSPECTED", severity=SEVERITY_WARNING,
                title=FINDING_CODES["SENSOR_BIAS_SUSPECTED"],
                message=(
                    f"传感器 {sensor_id} 的归一化残差均值 {score:.3f}"
                    f"（残差 {entry['residual_mean_m']:.1f} m 对其自称标准差），"
                    f"判定依据：{basis}；"
                    f"另有 {entry['innovation_inconsistent_count']} 次新息越界。"
                    "这**提示可能存在系统偏差**；但系统偏差与目标机动会给出"
                    "相似特征，需结合其他证据判断。本诊断**不自动剔除**该传感器。"
                ),
                evidence={"sensor_id": sensor_id,
                          "normalised_residual_mean": score,
                          "residual_mean_m": entry["residual_mean_m"],
                          "innovation_inconsistent_count":
                              entry["innovation_inconsistent_count"],
                          "n_sources": entry["n_sources"],
                          "detection_basis": basis},
            ))
        suspicious = health_state.get("suspicious_sensor_ids") or []
        if suspicious:
            scores = health_state.get("health_score") or {}
            values = list(scores.values()) or [0.0]
            findings.append(Finding(
                code="SENSOR_NOISE_UNDERREPORTED", severity=SEVERITY_INFO,
                title=FINDING_CODES["SENSOR_NOISE_UNDERREPORTED"],
                message=(
                    f"存在明显差于同伴的传感器：{suspicious}；"
                    f"最好/最差健康分 {min(values):.3f} / {max(values):.3f}。"
                    "残差远大于自称精度的常见原因是**上报精度优于实际**"
                    "（噪声低估）——这会让融合把更大的权重交给误差更大的传感器。"
                    "⚠️ 该判据是**相对**的：只有两部传感器时，"
                    "它只能指出更差的那部，**不能**断定偏差就在它身上。"
                ),
                evidence={"suspicious_sensor_ids": suspicious,
                          "health_best": min(values),
                          "health_worst": max(values)},
            ))

    # ------------------------------------------------------------------
    # 能力三：多策略对比
    # ------------------------------------------------------------------

    def compare_policies(self, payload: Dict[str, Any]) -> CompareResult:
        rows: List[Dict[str, Any]] = list(payload.get("summaries", []) or [])
        metric_order: List[str] = list(
            payload.get("metric_order")
            or [
                "horizon_satisfaction_rate",
                "violation_rate",
                "avg_tx_power_w",
                "cumulative_energy_j",
                "avg_intercept_prob",
                "avg_exposure",
                "composite_reward",
            ]
        )
        if not rows:
            return CompareResult(
                provider=self.name, model=self.model, status="error",
                error="compare_policies 需要 payload['summaries']（各策略的汇总指标列表）",
            )

        ranking = sorted(
            [str(r.get("label", "?")) for r in rows],
            key=lambda name: -float(
                next((r.get("composite_reward", float("-inf")) for r in rows
                      if str(r.get("label", "?")) == name), float("-inf"))
            ),
        )

        highlights: List[str] = []
        tradeoffs: List[str] = []
        best_by_metric: Dict[str, str] = {}
        for metric in metric_order:
            values = [
                (str(r.get("label", "?")), float(r[metric]))
                for r in rows
                if r.get(metric) is not None
            ]
            if not values:
                continue
            lower = LOWER_IS_BETTER.get(metric, True)
            best = min(values, key=lambda kv: kv[1]) if lower else max(values, key=lambda kv: kv[1])
            best_by_metric[metric] = best[0]
            highlights.append(
                f"{METRIC_CN.get(metric, metric)}最优：**{best[0]}** = {best[1]:.4f}"
            )

        # 典型权衡：满足率最高 vs 综合收益最高 若不是同一个策略，明确指出
        sat_best = best_by_metric.get("horizon_satisfaction_rate")
        reward_best = best_by_metric.get("composite_reward")
        if sat_best and reward_best and sat_best != reward_best:
            tradeoffs.append(
                f"探测满足率最高的是 {sat_best}，而综合收益最高的是 {reward_best}"
                f"—— 两者不一致说明该奖励下"
                f"存在「牺牲部分探测可靠性换取低暴露/低能耗」的权衡。"
            )
        expo_best = best_by_metric.get("avg_exposure")
        if expo_best and sat_best and expo_best != sat_best:
            tradeoffs.append(
                f"平均暴露最低的是 {expo_best}，与满足率最优的 {sat_best} 不同，"
                f"说明低暴露与高可靠探测在本场景下难以兼得。"
            )

        table = [
            {
                "label": str(r.get("label", "?")),
                **{m: r.get(m) for m in metric_order if r.get(m) is not None},
            }
            for r in rows
        ]
        # 按综合收益排序，便于阅读
        table.sort(
            key=lambda row: -float(row.get("composite_reward", float("-inf")) or float("-inf"))
        )

        return CompareResult(
            provider=self.name,
            model=self.model,
            status=STATUS_OK,
            metric_order=metric_order,
            ranking=ranking,
            highlights=highlights,
            tradeoffs=tradeoffs,
            table=table,
            confidence=0.85,
        )

    # ------------------------------------------------------------------
    # 能力四：实验总结报告
    # ------------------------------------------------------------------

    def generate_report(self, payload: Dict[str, Any]) -> ReportResult:
        summaries: List[Dict[str, Any]] = list(payload.get("summaries", []) or [])
        context: Dict[str, Any] = dict(payload.get("context", {}) or {})
        title = str(payload.get("title") or "低截获雷达功率调控仿真实验总结")

        if not summaries:
            return ReportResult(
                provider=self.name, model=self.model, status="error",
                title=title, error="generate_report 需要 payload['summaries']",
            )

        comparison = self.compare_policies({"summaries": summaries})

        sections: List[Dict[str, str]] = []
        sections.append(
            {
                "heading": "一、实验设置",
                "body": "；".join(f"{k}：{v}" for k, v in context.items())
                or "（未提供实验设置上下文）",
            }
        )
        sections.append(
            {
                "heading": "二、策略总体排序（按综合收益）",
                "body": " > ".join(comparison.ranking),
            }
        )
        sections.append(
            {
                "heading": "三、各指标最优策略",
                "body": "\n".join(f"- {h}" for h in comparison.highlights),
            }
        )
        if comparison.tradeoffs:
            sections.append(
                {
                    "heading": "四、关键权衡",
                    "body": "\n".join(f"- {t}" for t in comparison.tradeoffs),
                }
            )

        # 逐策略一句话画像
        portraits: List[str] = []
        for row in comparison.table:
            label = row.get("label", "?")
            sat = row.get("horizon_satisfaction_rate")
            rew = row.get("composite_reward")
            pw = row.get("avg_tx_power_w")
            en = row.get("cumulative_energy_j")
            pieces = []
            if sat is not None:
                pieces.append(f"满足率 {float(sat):.4f}")
            if pw is not None:
                pieces.append(f"平均功率 {float(pw):.2f} W")
            if en is not None:
                pieces.append(f"能耗 {float(en):.1f} J")
            if rew is not None:
                pieces.append(f"综合收益 {float(rew):+.4f}")
            portraits.append(f"- **{label}**：" + "，".join(pieces))
        sections.append({"heading": "五、逐策略画像", "body": "\n".join(portraits)})

        conclusion = (
            f"在 {context.get('场景', '本场景')} 下共比较 {len(summaries)} 个策略；"
            f"综合收益最高的是 {comparison.ranking[0] if comparison.ranking else '—'}。"
            + ("存在明显权衡：" + "；".join(comparison.tradeoffs) if comparison.tradeoffs
               else "各指标最优策略较为一致，权衡不显著。")
        )

        return ReportResult(
            provider=self.name,
            model=self.model,
            status=STATUS_OK,
            title=title,
            sections=sections,
            conclusion=conclusion,
            confidence=0.85,
        )
