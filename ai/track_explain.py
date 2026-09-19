"""航迹与协同解释器（v4.5）。

两个只读解释器，**不做任何控制**，输入输出全部是结构化证据。

`explain_track`：输入单条航迹 + 它的溯源/协方差/新鲜度/支持传感器，
输出"这条航迹是怎么形成的、为什么进入外推、不确定度从哪来"。

`explain_cooperation`：输入"不共享 / 理想共享 / 受限共享"的结构化结果，
输出"远端信息在哪些时刻真正帮到了本地感知、哪些收益被延迟或丢包吃掉"。

⚠️ 两条纪律：
1. **只解释结构性事实**。像"共享把 RMSE 降低了 57.6%"这种**收益大小**
   必须由离线评测用真值计算，属于评测通道，本模块**不产出**这类数字
   （输入里没有，也就编不出来——这也正好让证据校验能拦住幻觉）。
2. 所有结论都附 `evidence` 数值字段，与 `ai/evidence_check.py` 的
   可溯校验配合，保证自然语言里的数字都能在证据里找到。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai.schema import (
    FINDING_CODES,
    DiagnosisResult,
    Finding,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    STATUS_DEGRADED,
    STATUS_ERROR,
    STATUS_OK,
)

#: 航迹状态 -> 中文
_TRACK_STATUS_CN: Dict[str, str] = {
    "confirmed": "已确认",
    "tentative": "暂定（证据不足）",
    "coasting": "外推中（无观测支撑）",
    "dropped": "已删除",
}

#: 位置不确定度阈值（米）：超过即提示"不确定"
_UNCERTAIN_SIGMA_M = 200.0
#: 新鲜度下限：低于即提示"数据陈旧"
_STALE_FRESHNESS = 0.4
#: 外推多少步以内算"短时保持"（可接受），超过算"长时间丢失"
_SHORT_COAST_MISSES = 3

#: 通信丢弃原因 -> 发现码（与 rule_provider 保持一致）
_DROP_CODES: Dict[str, str] = {
    "lost": "COMM_PACKET_LOST",
    "expired": "COMM_MESSAGE_EXPIRED",
    "link_down": "COMM_LINK_DOWN",
    "queue_full": "COMM_QUEUE_FULL",
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def explain_track(payload: Dict[str, Any]) -> DiagnosisResult:
    """解释单条航迹。"""
    track = dict(payload.get("track") or {})
    if not track:
        return DiagnosisResult(
            provider="rule", model="track-explainer",
            status=STATUS_ERROR, severity=SEVERITY_INFO,
            summary="没有提供 track 字段，无法解释。",
            error="missing 'track'",
            confidence=0.0,
        )

    track_id = str(track.get("track_id", ""))
    status = str(track.get("status", "unknown"))
    status_cn = _TRACK_STATUS_CN.get(status, status)
    hits = int(_f(track.get("hits"), 0))
    misses = int(_f(track.get("misses"), 0))
    local_updates = int(_f(track.get("local_updates"), 0))
    remote_updates = int(_f(track.get("remote_updates"), 0))
    n_sources = int(_f(track.get("n_sources"), 0))
    # ⚠️ "没给" 和 "等于 0" 是两件事：协方差/新鲜度缺失时必须说"未提供"，
    # 否则会输出"不确定度 0.0 m"这种等于宣称"完全确定"的假结论。
    sigma_raw = track.get("sigma_position")
    has_sigma = isinstance(sigma_raw, dict) and bool(sigma_raw)
    sigma: Dict[str, Any] = sigma_raw if isinstance(sigma_raw, dict) else {}
    sigma_norm: Optional[float] = None
    if has_sigma:
        sigma_norm = (
            _f(sigma.get("x")) ** 2 + _f(sigma.get("y")) ** 2 + _f(sigma.get("z")) ** 2
        ) ** 0.5
    has_freshness = track.get("freshness") is not None
    freshness = _f(track.get("freshness"), 0.0)
    age = track.get("measurement_age_s")
    source_sensors = list(track.get("source_sensors") or [])
    platforms = list(track.get("platforms") or [])
    recent = list(track.get("recent_sources") or [])
    has_remote = bool(track.get("has_remote_contribution"))

    findings: List[Finding] = []

    # --- 1) 形成与更新历史 ---
    findings.append(Finding(
        code="REMOTE_SENSOR_CONTRIBUTION" if has_remote else "TRACK_MAINTAINED",
        severity=SEVERITY_INFO,
        title="航迹形成与更新情况",
        message=(
            f"航迹 {track_id} 当前状态为「{status_cn}」，"
            f"累计命中 {hits} 次、漏检 {misses} 次；"
            f"本地更新 {local_updates} 次、远端共享更新 {remote_updates} 次；"
            f"共 {n_sources} 条溯源记录，支持它的传感器为 "
            f"{source_sensors if source_sensors else '（无）'}。"
        ),
        evidence={
            "track_id": track_id, "status": status, "hits": hits, "misses": misses,
            "local_updates": local_updates, "remote_updates": remote_updates,
            "n_sources": n_sources, "source_sensors": source_sensors,
            "platforms": platforms,
        },
    ))

    # --- 2) 为什么外推 ---
    if status == "coasting" or misses > 0:
        code = "TRACK_COASTING"
        severity = (SEVERITY_INFO if misses <= _SHORT_COAST_MISSES
                    else SEVERITY_WARNING)
        findings.append(Finding(
            code=code, severity=severity,
            title=FINDING_CODES[code],
            message=(
                f"该航迹连续 {misses} 步没有获得任何测量，正在用匀速模型外推。"
                + (f"最近一次测量距今 {_f(age):.3f} s。" if age is not None else "")
                + ("连续外推超过 " f"{_SHORT_COAST_MISSES} 步，"
                   "位置漂移风险显著上升。" if misses > _SHORT_COAST_MISSES else
                   "属于短时保持，短期内可接受。")
            ),
            evidence={"misses": misses, "measurement_age_s": age,
                      "status": status,
                      "freshness": round(freshness, 6) if has_freshness else None},
        ))

    # --- 3) 不确定性 ---
    if sigma_norm is not None and sigma_norm > _UNCERTAIN_SIGMA_M:
        findings.append(Finding(
            code="TRACK_UNCERTAIN", severity=SEVERITY_WARNING,
            title=FINDING_CODES["TRACK_UNCERTAIN"],
            message=(
                f"位置不确定度范数约 {sigma_norm:.1f} m，超过 "
                f"{_UNCERTAIN_SIGMA_M:.0f} m 阈值。"
                "不确定度的来源是量测噪声、外推时长与观测几何，"
                "不是单一传感器的问题。"
            ),
            evidence={"sigma_norm_m": round(sigma_norm, 3),
                      "sigma_x": _f(sigma.get("x")),
                      "sigma_y": _f(sigma.get("y")),
                      "sigma_z": _f(sigma.get("z")),
                      "hits": hits, "misses": misses},
        ))

    # --- 4) 新鲜度 ---
    if has_freshness and freshness < _STALE_FRESHNESS:
        findings.append(Finding(
            code="TRACK_COASTING", severity=SEVERITY_WARNING,
            title="航迹数据陈旧",
            message=(
                f"新鲜度 {freshness:.3f} 低于阈值 {_STALE_FRESHNESS}。"
                "新鲜度按**测量时刻**计算（而非到达时刻），"
                "因此延迟很大的共享测量即使刚到，也算陈旧。"
            ),
            evidence={"freshness": round(freshness, 6),
                      "measurement_age_s": age},
        ))

    # --- 5) 协同贡献 ---
    if has_remote:
        findings.append(Finding(
            code="REMOTE_SENSOR_CONTRIBUTION", severity=SEVERITY_INFO,
            title=FINDING_CODES["REMOTE_SENSOR_CONTRIBUTION"],
            message=(
                f"该航迹获得过 {remote_updates} 次远端共享测量贡献，"
                f"来源平台 {platforms if platforms else '（未记录）'}。"
            ),
            evidence={"remote_updates": remote_updates, "platforms": platforms,
                      "recent_sources": recent},
        ))
        if local_updates == 0:
            findings.append(Finding(
                code="COOPERATIVE_TRACK_RECOVERED", severity=SEVERITY_INFO,
                title=FINDING_CODES["COOPERATIVE_TRACK_RECOVERED"],
                message=(
                    "该航迹**仅靠远端共享测量**维持：本地传感器对它没有任何贡献。"
                    "这是可溯源的结构性事实；具体精度收益须由离线评测计算。"
                ),
                evidence={"local_updates": 0, "remote_updates": remote_updates,
                          "track_id": track_id},
            ))

    severity = SEVERITY_INFO
    for finding in findings:
        if finding.severity == SEVERITY_CRITICAL:
            severity = SEVERITY_CRITICAL
            break
        if finding.severity == SEVERITY_WARNING:
            severity = SEVERITY_WARNING

    summary = (
        f"航迹 {track_id}：状态「{status_cn}」，命中 {hits}、漏检 {misses}，"
        + (f"位置不确定度 {sigma_norm:.1f} m，"
           if sigma_norm is not None else "位置不确定度未提供（无协方差证据），")
        + (f"新鲜度 {freshness:.3f}，" if has_freshness else "新鲜度未提供，")
        + f"支持传感器 {len(source_sensors)} 个"
        + ("，含远端共享贡献" if has_remote else "，仅本地测量")
        + "。"
    )
    recommendations: List[str] = []
    if status == "coasting":
        recommendations.append("该航迹处于外推：如需维持精度，应考虑调整传感器指向或等待远端共享测量到达")
    if sigma_norm is not None and sigma_norm > _UNCERTAIN_SIGMA_M:
        recommendations.append("不确定度偏高：多视角（多雷达）观测可显著收缩协方差")
    if not has_sigma:
        recommendations.append("缺少协方差证据：建议在航迹快照里带上 sigma_position 再做不确定度判断")
    if not has_remote and (payload.get("communication") or {}).get("n_links", 0) == 0:
        recommendations.append("当前未启用共享：本地看不见的目标无法由远端补回")
    if not recommendations:
        recommendations.append("航迹状态正常，无需特别处置")

    return DiagnosisResult(
        provider="rule", model="track-explainer",
        status=STATUS_OK, severity=severity,
        summary=summary, findings=findings, recommendations=recommendations,
        confidence=0.85,
    )


def explain_cooperation(payload: Dict[str, Any]) -> DiagnosisResult:
    """解释协同感知（只讲结构性事实，不报收益大小）。"""
    cases = list(payload.get("cases") or [])
    fusion = dict(payload.get("fusion") or {})
    communication = dict(payload.get("communication") or {})
    notes = list(payload.get("notes") or [])

    if not cases and not fusion and not communication:
        return DiagnosisResult(
            provider="rule", model="cooperation-explainer",
            status=STATUS_ERROR, severity=SEVERITY_INFO,
            summary="没有提供任何协同证据（cases / fusion / communication 均为空）。",
            error="missing cooperation evidence",
            confidence=0.0,
        )

    findings: List[Finding] = []

    # --- 1) 链路与送达 ---
    if communication:
        policy = str(communication.get("policy", "none"))
        policy_cn = str(communication.get("policy_cn", policy))
        n_links = int(_f(communication.get("n_links"), 0))
        delivery = _f(communication.get("delivery_rate"), 0.0)
        latency = _f(communication.get("latency_mean_s"), 0.0)
        p95 = _f(communication.get("latency_p95_s"), 0.0)
        findings.append(Finding(
            code="REMOTE_MEASUREMENT_DELAYED" if n_links else
                 "REMOTE_SENSOR_CONTRIBUTION",
            severity=SEVERITY_INFO,
            title="通信条件",
            message=(
                f"共享策略「{policy_cn}」，{n_links} 条链路；"
                f"送达率 {delivery:.4f}，平均延迟 {latency:.3f} s，"
                f"p95 延迟 {p95:.3f} s。"
                "**「多平台」不等于完美共享**：下面的协同结论必须与这组条件一起解释。"
            ),
            evidence={"policy": policy, "n_links": n_links,
                      "delivery_rate": round(delivery, 6),
                      "latency_mean_s": round(latency, 6),
                      "latency_p95_s": round(p95, 6)},
        ))
        for reason, code in _DROP_CODES.items():
            count = int(_f((communication.get("drop_reasons") or {}).get(reason), 0))
            if count <= 0:
                continue
            findings.append(Finding(
                code=code,
                severity=SEVERITY_WARNING if code != "COMM_MESSAGE_EXPIRED"
                else SEVERITY_INFO,
                title=FINDING_CODES[code],
                message=(
                    f"{count} 条消息因「{FINDING_CODES[code]}」未能参与协同。"
                    "这类损失是通信条件造成的，不是感知算法的问题。"
                ),
                evidence={"drop_reason": reason, "count": count,
                          "policy": policy},
            ))

    # --- 2) 结构性协同事实 ---
    with_remote = int(_f(fusion.get("n_tracks_with_remote"), 0))
    local_only = int(_f(fusion.get("n_tracks_local_only"), 0))
    used = int(_f(fusion.get("remote_measurements_used"), 0))
    arrived = int(_f(fusion.get("remote_measurements_arrived"), 0))
    rejected = int(_f(fusion.get("remote_measurements_rejected"), 0))
    remotely_only = int(_f(fusion.get("tracks_supported_remotely_only"), 0))
    if fusion:
        findings.append(Finding(
            code="REMOTE_SENSOR_CONTRIBUTION" if with_remote else
                 "TRACK_COASTING",
            severity=SEVERITY_INFO,
            title="远端信息的实际去向",
            message=(
                f"当前 {with_remote + local_only} 条航迹中，{with_remote} 条含远端贡献、"
                f"{local_only} 条仅本地支撑；送达的远端测量里 {used} 条进入航迹、"
                f"{rejected} 条被拒。"
            ),
            evidence={"n_tracks_with_remote": with_remote,
                      "n_tracks_local_only": local_only,
                      "remote_measurements_arrived": arrived,
                      "remote_measurements_used": used,
                      "remote_measurements_rejected": rejected},
        ))
        if remotely_only > 0:
            findings.append(Finding(
                code="COOPERATIVE_TRACK_RECOVERED", severity=SEVERITY_INFO,
                title=FINDING_CODES["COOPERATIVE_TRACK_RECOVERED"],
                message=(
                    f"{remotely_only} 条航迹**仅靠远端共享**维持 —— "
                    "本地在这些时刻对该目标没有贡献。"
                    "这是「协同确实补上了本地缺口」的**结构性证据**，"
                    "但它不等于「精度变好了」：收益大小必须由离线评测用真值计算。"
                ),
                evidence={"tracks_supported_remotely_only": remotely_only,
                          "remote_measurements_used": used},
            ))

    # --- 3) 策略对照（若给了多组结果） ---
    if cases:
        rows: List[str] = []
        for case in cases:
            policy_cn = str(case.get("policy_cn", case.get("policy", "")))
            coverage = case.get("track_coverage")
            utilization = case.get("remote_utilization")
            delivery = case.get("delivery_rate")
            rows.append(
                f"{policy_cn}：航迹覆盖 {_f(coverage):.3f}，"
                f"远端利用率 {_f(utilization):.4f}，送达率 {_f(delivery):.4f}"
            )
        findings.append(Finding(
            code="COOPERATIVE_TRACK_RECOVERED", severity=SEVERITY_INFO,
            title="三种共享策略的结构性对照",
            message="；".join(rows) + "。"
            "注意这些是**结构指标**（覆盖/利用率/送达率），"
            "不含真值层面的精度结论——精度必须在离线评测里用真值计算。",
            evidence={"cases": [dict(c) for c in cases]},
        ))

    if notes:
        findings.append(Finding(
            code="REMOTE_SENSOR_CONTRIBUTION", severity=SEVERITY_INFO,
            title="补充说明", message="；".join(str(n) for n in notes),
            evidence={"notes": [str(n) for n in notes]},
        ))

    severity = SEVERITY_INFO
    if any(f.severity == SEVERITY_WARNING for f in findings):
        severity = SEVERITY_WARNING

    summary = (
        f"共 {len(cases)} 组协同对照"
        + (f"，当前 {with_remote} 条航迹含远端贡献" if fusion else "")
        + "。协同效果必须与通信条件（送达率、延迟、丢弃原因）一起解释；"
        "本解释只给结构性事实，精度收益需离线用真值评测。"
    )
    recommendations = [
        "报协同收益时必须同时给出该收益对应的通信条件（延迟/丢包/过期）",
        "若要声称精度改善，请用离线评测脚本（真值只在那里使用）",
    ]
    if rejected > 0:
        recommendations.append(
            f"有 {rejected} 条远端测量到达但未被采用：先查时效阈值与观测对象过滤"
        )

    return DiagnosisResult(
        provider="rule", model="cooperation-explainer",
        status=STATUS_OK, severity=severity,
        summary=summary, findings=findings, recommendations=recommendations,
        confidence=0.85,
    )
