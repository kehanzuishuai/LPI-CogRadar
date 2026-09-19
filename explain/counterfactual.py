"""可解释决策模块：对可行动作做**反事实试算**，产出结构化证据。

设计原则（很重要）
------------------
这里**不做自然语言解释**，只做两件事：

1. 用 `Simulator.preview()`（纯函数、不改变状态）对若干候选功率档位逐一试算，
   算出各自会带来的 Pd、Pint、暴露、能耗与**单步收益**；
2. 把差异归因成**可核验的结构化证据**（代码 + 数值），例如
   「降一档会让 Pd 0.83 → 0.71，跌破 required_pd 0.80，触发 1.5 的固定惩罚」。

自然语言由 `ai/` 认知诊断层负责生成，而且**只能引用这里给出的证据代码与数值**，
不允许大模型凭空编造理由。这样「解释」永远可追溯到仿真里的具体数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 反事实默认关注的功率档位（用户可读性优先：直接给瓦数）
DEFAULT_FOCUS_POWERS_W: Tuple[float, ...] = (18.0, 25.0, 35.0)

#: 证据代码 -> 中文说明（AI 层只能引用这些代码）
EVIDENCE_CODES: Dict[str, str] = {
    "MINIMAL_SATISFYING": "当前档位已是满足探测要求的最低可行档",
    "AVOID_VIOLATION": "更低档位会跌破 required_pd，触发未达标固定惩罚",
    "VIOLATION_ACCEPTED": "当前档位未达标，是为省电/降暴露而主动放弃该步",
    "ENERGY_LIMITED": "受剩余能量限制，更高档位已经不可行",
    "ENERGY_MARGIN": "剩余能量不足以再支撑同档位若干步",
    "HIGHER_POWER_WORSE": "更高档位的单步收益更低（暴露与能耗代价超过探测收益）",
    "LOWER_POWER_BETTER": "更低档位的单步收益更高（当前档位偏保守）",
    "REDUCE_EXPOSURE": "降功率可显著降低累计暴露，从而压低后续被截获风险",
    "EXPOSURE_COST_DOMINANT": "该步的暴露代价已超过探测收益，宜压低功率",
    "DETECTION_DOMINANT": "该步的探测收益占主导，宜优先保证 Pd",
    "GREEDY_OPTIMAL": "当前档位与逐档贪心（单步最优）一致",
}


@dataclass
class CounterfactualOutcome:
    """对某个候选功率档位的反事实试算结果。"""

    level: int
    tx_power_w: float
    feasible: bool
    is_chosen: bool

    pd_min: float = 0.0
    detection_term: float = 0.0
    task_satisfied: bool = False
    pint_inst: float = 0.0
    pint_eff: float = 0.0
    exposure_before: float = 0.0
    exposure_next: float = 0.0
    step_energy_j: float = 0.0
    remaining_energy_after_j: float = 0.0
    steps_affordable_after: int = 0
    jam_noise_ratio: float = 0.0

    immediate_reward: float = 0.0
    delta_reward_vs_chosen: float = 0.0
    delta_pd_vs_chosen: float = 0.0
    delta_pint_vs_chosen: float = 0.0
    delta_exposure_next_vs_chosen: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "tx_power_w": round(self.tx_power_w, 4),
            "feasible": self.feasible,
            "is_chosen": self.is_chosen,
            "pd_min": round(self.pd_min, 6),
            "detection_term": round(self.detection_term, 6),
            "task_satisfied": self.task_satisfied,
            "pint_inst": round(self.pint_inst, 6),
            "pint_eff": round(self.pint_eff, 6),
            "exposure_before": round(self.exposure_before, 6),
            "exposure_next": round(self.exposure_next, 6),
            "step_energy_j": round(self.step_energy_j, 4),
            "remaining_energy_after_j": round(self.remaining_energy_after_j, 4),
            "steps_affordable_after": self.steps_affordable_after,
            "jam_noise_ratio": round(self.jam_noise_ratio, 6),
            "immediate_reward": round(self.immediate_reward, 6),
            "delta_reward_vs_chosen": round(self.delta_reward_vs_chosen, 6),
            "delta_pd_vs_chosen": round(self.delta_pd_vs_chosen, 6),
            "delta_pint_vs_chosen": round(self.delta_pint_vs_chosen, 6),
            "delta_exposure_next_vs_chosen": round(
                self.delta_exposure_next_vs_chosen, 6
            ),
        }


@dataclass
class CounterfactualReport:
    """一步的完整反事实证据包（JSON 可序列化，直接喂给 AI 层）。"""

    step_index: int
    time: float
    chosen_level: int
    chosen_power_w: float
    direction: str  # "hold" | "raise" | "lower"
    verdict_code: str
    verdict_cn: str
    reason: str
    evidence_codes: List[str] = field(default_factory=list)
    candidates: List[CounterfactualOutcome] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    greedy_level: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "time": round(self.time, 4),
            "chosen": {"level": self.chosen_level,
                       "tx_power_w": round(self.chosen_power_w, 4)},
            "direction": self.direction,
            "verdict": {
                "code": self.verdict_code,
                "text_cn": self.verdict_cn,
                "reason": self.reason,
                "evidence_codes": self.evidence_codes,
                "evidence_legend": {
                    c: EVIDENCE_CODES.get(c, c) for c in self.evidence_codes
                },
            },
            "greedy_level": self.greedy_level,
            "state": self.state,
            "counterfactuals": [c.to_dict() for c in self.candidates],
        }


# ----------------------------------------------------------------------

def _affordable_steps(sim: Any, remaining_j: float) -> int:
    """剩余能量在当前几何下还能支撑多少步（按最低档估算上界）。"""
    min_energy = sim.min_step_energy_j
    if min_energy <= 0:
        return 0
    return int(remaining_j // min_energy)


def build_counterfactual_report(
    sim: Any,
    chosen_level: int,
    focus_powers_w: Sequence[float] = DEFAULT_FOCUS_POWERS_W,
    include_neighbours: bool = True,
) -> CounterfactualReport:
    """对某一步的候选功率做反事实试算，并给出结构化归因。

    参数
    ----
    sim           : Simulator（会用 preview() 试算，**不改变任何状态**）
    chosen_level  : 实际执行的功率档位（DQN / 规则策略的输出）
    focus_powers_w: 额外关注的功率值（瓦），会被吸附到最近的档位；
                    默认 18 / 25 / 35 W，对应「低 / 中 / 高」三档典型选择
    include_neighbours: 是否同时纳入执行档位的相邻档位
    """
    sim._ensure_ready()
    assert sim.scenario is not None and sim.radar is not None

    levels: List[float] = sim.power_levels_w
    chosen_level = int(chosen_level)

    # --- 组装候选档位集合 ---
    candidate_levels: List[int] = []
    if include_neighbours:
        for offset in (-1, 1):
            level = chosen_level + offset
            if 0 <= level < sim.num_levels:
                candidate_levels.append(level)
    for power_w in focus_powers_w:
        candidate_levels.append(sim.scenario.nearest_level_index(float(power_w)))
    candidate_levels.append(chosen_level)
    candidate_levels = sorted(set(candidate_levels))

    # --- 逐档试算（纯函数，不改变状态）---
    outcomes: List[CounterfactualOutcome] = []
    for level in candidate_levels:
        pt_w = levels[level]
        evaluation = sim.preview(pt_w)
        energy = sim.step_energy_j(level)
        remaining_after = max(0.0, sim.remaining_energy_j - energy)
        reward = sim.action_reward(evaluation, pt_w)
        outcomes.append(
            CounterfactualOutcome(
                level=level,
                tx_power_w=pt_w,
                feasible=sim.is_level_feasible(level),
                is_chosen=(level == chosen_level),
                pd_min=evaluation.pd_min,
                detection_term=evaluation.detection_term,
                task_satisfied=evaluation.task_satisfied,
                pint_inst=evaluation.intercept_prob_instant,
                pint_eff=evaluation.intercept_prob,
                exposure_before=evaluation.exposure,
                exposure_next=evaluation.exposure_next,
                step_energy_j=energy,
                remaining_energy_after_j=remaining_after,
                steps_affordable_after=_affordable_steps(sim, remaining_after),
                jam_noise_ratio=evaluation.jam_noise_ratio,
                immediate_reward=reward,
            )
        )

    chosen = next((o for o in outcomes if o.is_chosen), outcomes[-1])
    for outcome in outcomes:
        outcome.delta_reward_vs_chosen = outcome.immediate_reward - chosen.immediate_reward
        outcome.delta_pd_vs_chosen = outcome.pd_min - chosen.pd_min
        outcome.delta_pint_vs_chosen = outcome.pint_eff - chosen.pint_eff
        outcome.delta_exposure_next_vs_chosen = (
            outcome.exposure_next - chosen.exposure_next
        )

    # --- 归因 ---
    feasible_outcomes = [o for o in outcomes if o.feasible]
    best_feasible = max(feasible_outcomes, key=lambda o: o.immediate_reward)
    satisfying = [o for o in feasible_outcomes if o.task_satisfied]
    minimal_satisfying = min(satisfying, key=lambda o: o.tx_power_w) if satisfying else None

    lower = [o for o in feasible_outcomes if o.tx_power_w < chosen.tx_power_w]
    higher = [o for o in feasible_outcomes if o.tx_power_w > chosen.tx_power_w]
    better_lower = [o for o in lower if o.immediate_reward > chosen.immediate_reward]
    better_higher = [o for o in higher if o.immediate_reward > chosen.immediate_reward]

    evidence: List[str] = []
    direction = "hold"
    verdict_code = "GREEDY_OPTIMAL"
    reason_parts: List[str] = []

    if (
        minimal_satisfying is not None
        and minimal_satisfying.level == chosen.level
        and chosen.task_satisfied
    ):
        verdict_code = "MINIMAL_SATISFYING"
        evidence.append("MINIMAL_SATISFYING")
        reason_parts.append(
            f"当前 {chosen.tx_power_w:.0f} W 已是满足探测要求的最低可行档"
            f"（Pd={chosen.pd_min:.3f} ≥ required_pd={sim.radar.required_pd}）"
        )
        if higher:
            worst_higher = max(higher, key=lambda o: o.tx_power_w)
            reason_parts.append(
                f"升到 {worst_higher.tx_power_w:.0f} W 只能把 Pd 提到 "
                f"{worst_higher.pd_min:.3f}（探测项已封顶），"
                f"却让 Pint_eff 从 {chosen.pint_eff:.3f} 升到 "
                f"{worst_higher.pint_eff:.3f}、多耗 {worst_higher.step_energy_j - chosen.step_energy_j:.1f} J"
            )
            evidence.append("HIGHER_POWER_WORSE")
        if lower:
            lowest = min(lower, key=lambda o: o.tx_power_w)
            reason_parts.append(
                f"降到 {lowest.tx_power_w:.0f} W 会让 Pd 掉到 {lowest.pd_min:.3f}"
                f"（< {sim.radar.required_pd}），触发未达标惩罚"
            )
            evidence.append("AVOID_VIOLATION")

    elif not chosen.task_satisfied:
        verdict_code = "VIOLATION_ACCEPTED"
        evidence.append("VIOLATION_ACCEPTED")
        if minimal_satisfying is None:
            reason_parts.append(
                "可行档位里没有一个能达到探测要求，只能放弃该步"
            )
            evidence.append("ENERGY_LIMITED")
        else:
            reason_parts.append(
                f"当前 {chosen.tx_power_w:.0f} W 未达标（Pd={chosen.pd_min:.3f}），"
                f"要达标需 {minimal_satisfying.tx_power_w:.0f} W，"
                f"差额能耗 {minimal_satisfying.step_energy_j - chosen.step_energy_j:.1f} J"
            )

    elif sim.max_feasible_level() == chosen.level and not chosen.task_satisfied:
        verdict_code = "ENERGY_LIMITED"
        evidence.append("ENERGY_LIMITED")
        reason_parts.append(
            f"剩余能量只够到最高可行档 {chosen.tx_power_w:.0f} W，"
            f"仍无法满足 Pd_required"
        )

    elif better_lower and not better_higher:
        direction = "lower"
        best = max(better_lower, key=lambda o: o.immediate_reward)
        verdict_code = "LOWER_POWER_BETTER"
        evidence += ["LOWER_POWER_BETTER", "REDUCE_EXPOSURE"]
        reason_parts.append(
            f"降到 {best.tx_power_w:.0f} W 的单步收益更高"
            f"（{best.immediate_reward:+.4f} vs 当前 {chosen.immediate_reward:+.4f}）："
            f"暴露从 {chosen.exposure_next:.4f} 降到 {best.exposure_next:.4f}"
        )
        if not best.task_satisfied:
            evidence.append("VIOLATION_ACCEPTED")
            reason_parts.append(
                f"代价是该步 Pd 降到 {best.pd_min:.3f}，未达标"
            )

    elif better_higher and not better_lower:
        direction = "raise"
        best = max(better_higher, key=lambda o: o.immediate_reward)
        verdict_code = "DETECTION_DOMINANT"
        evidence.append("DETECTION_DOMINANT")
        reason_parts.append(
            f"升到 {best.tx_power_w:.0f} W 的单步收益更高"
            f"（{best.immediate_reward:+.4f} vs 当前 {chosen.immediate_reward:+.4f}）："
            f"Pd 从 {chosen.pd_min:.3f} 升到 {best.pd_min:.3f}，"
            f"若当前未达标则同时消掉固定惩罚"
        )

    else:
        evidence.append("GREEDY_OPTIMAL")
        reason_parts.append(
            f"当前 {chosen.tx_power_w:.0f} W 与逐档贪心的单步最优一致"
            f"（单步收益 {chosen.immediate_reward:+.4f}）"
        )

    # 能量侧证据
    if sim.remaining_energy_j > 0:
        margin_steps = chosen.steps_affordable_after
        if margin_steps <= 3:
            evidence.append("ENERGY_MARGIN")
            reason_parts.append(
                f"按当前档位只能再撑 {margin_steps} 步，能量是硬约束"
            )

    state = {
        "time": round(sim.current_time, 4),
        "step_index": int(sim.step_index),
        "pd_min": round(chosen.pd_min, 6),
        "required_pd": float(sim.radar.required_pd),
        "pint_eff": round(chosen.pint_eff, 6),
        "pint_inst": round(chosen.pint_inst, 6),
        "exposure": round(sim.exposure.value, 6),
        "jam_noise_ratio": round(chosen.jam_noise_ratio, 6),
        "remaining_energy_j": round(sim.remaining_energy_j, 4),
        "energy_budget_j": float(sim.radar.energy_budget_j),
        "feasible_levels": sim.feasible_levels(),
    }
    if sim.jammers and sim.jammers[0].is_adaptive:
        state["jammer_mode"] = sim.jammers[0].controller.mode

    return CounterfactualReport(
        step_index=int(sim.step_index),
        time=float(sim.current_time),
        chosen_level=chosen_level,
        chosen_power_w=chosen.tx_power_w,
        direction=direction,
        verdict_code=verdict_code,
        verdict_cn=EVIDENCE_CODES.get(verdict_code, verdict_code),
        reason="；".join(reason_parts),
        evidence_codes=sorted(set(evidence)),
        candidates=outcomes,
        state=state,
        greedy_level=best_feasible.level,
    )


def build_key_moment_reports(
    sim: Any,
    results: Sequence[Any],
    max_moments: int = 5,
    focus_powers_w: Sequence[float] = DEFAULT_FOCUS_POWERS_W,
) -> List[Dict[str, Any]]:
    """在关键时刻生成反事实报告。

    现成的 `StepResult` 序列无法重放状态，因此这里的做法是：
    先按「关键时刻」的判据挑出步号（档位发生变化、出现未达标、能量进入警戒），
    再让调用方用**同一条策略重放**到该步。为了简单可靠，这里只在
    `results` 上做筛选并标注，具体重放由 `explain.replay` 风格的调用方完成。

    返回的是「值得解释的时刻」清单（含判据），供上层决定重放哪些步。
    """
    moments: List[Dict[str, Any]] = []
    prev_level: Optional[int] = None
    for result in results:
        reasons: List[str] = []
        if prev_level is not None and result.power_level != prev_level:
            reasons.append("power_switched")
        if result.task_violated:
            reasons.append("detection_violated")
        if result.remaining_energy_j <= 3 * max(
            result.step_energy_j, 1e-9
        ):
            reasons.append("energy_margin_low")
        if result.power_level == max(
            (r.power_level for r in results[:1]), default=0
        ):
            pass
        prev_level = result.power_level
        if reasons:
            moments.append(
                {
                    "step_index": result.step_index,
                    "time": round(float(result.time), 4),
                    "power_level": result.power_level,
                    "tx_power_w": round(float(result.tx_power_w), 4),
                    "pd_min": round(float(result.pd_min), 6),
                    "pint_eff": round(float(result.intercept_prob), 6),
                    "exposure": round(float(result.exposure), 6),
                    "remaining_energy_j": round(float(result.remaining_energy_j), 4),
                    "reasons": reasons,
                }
            )

    # 优先保留"未达标"与"档位切换"的时刻，按时间均匀抽取
    prioritized = sorted(
        moments,
        key=lambda m: (
            "detection_violated" not in m["reasons"],
            "power_switched" not in m["reasons"],
            m["step_index"],
        ),
    )[:max_moments]
    return sorted(prioritized, key=lambda m: m["step_index"])
