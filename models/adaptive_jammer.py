"""自适应智能干扰机（规则型对手，**不是学习型对手**）。

定位与诚实性声明
----------------
本模块实现的是一个**基于规则/状态机的自适应干扰机**：它按预先写死的
态势判据在四种干扰动作之间切换。它**不是**强化学习对手，也没有在训练中
优化自己的策略。工程里一律称其为「规则自适应干扰机」，
不得宣传为「学习型对手」——后者需要独立训练一个干扰方智能体（见 README 后续工作）。

它存在的意义是把原来「固定时间窗 + 随机起伏」的开环干扰，
升级成**闭环对抗**：

    雷达辐射 → 敌方侦察累积暴露 → 干扰机升功率压制 → 雷达 Pd 下降
             → 雷达重新调功率（往往要提功率）→ 暴露进一步上升 → ...

四种干扰动作
------------
    NO_JAM        不干扰
    LOW_POWER     低功率压制（约为额定功率的 0.6 倍）
    HIGH_POWER    高功率压制（约 1.5 倍）
    INTERMITTENT  间歇干扰（高功率按周期开/关，用于省电与规避反辐射）

三类驱动信号（全部来自敌方视角可观测的量）
------------------------------------------
1. **ESM 累计暴露** `exposure`：暴露越高说明侦察机越可能完成识别与测向，
   越值得投入干扰资源；
2. **雷达辐射强度** `radar_power_ratio`：最近若干步雷达平均发射功率 / 最大档功率，
   辐射越强说明目标价值或探测需求越高；
3. **历史探测行为** `recent_satisfaction_rate`：最近若干步雷达"达标"的比例。
   若雷达一直在达标（干扰无效），则升级到高功率；若雷达已经明显被压制，
   则降级为间歇干扰省电。

综合威胁度：

    threat = w_exp · exposure
           + w_rad · radar_power_ratio
           + w_ineff · (1 − recent_satisfaction_rate)

再按阈值映射到四种动作。所有阈值与权重都写在配置里，可逐项调整。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Tuple


class JammerMode(str, Enum):
    """四种干扰动作。"""

    NO_JAM = "no_jam"
    LOW_POWER = "low_power"
    HIGH_POWER = "high_power"
    INTERMITTENT = "intermittent"


#: 各模式对干扰机额定功率的倍数（INTERMITTENT 在"开"的时候用高功率倍数）
MODE_POWER_SCALE: Dict[str, float] = {
    JammerMode.NO_JAM.value: 0.0,
    JammerMode.LOW_POWER.value: 0.6,
    JammerMode.HIGH_POWER.value: 1.5,
    JammerMode.INTERMITTENT.value: 1.5,
}

#: 中文名（报告与自然语言解释用）
MODE_CN: Dict[str, str] = {
    JammerMode.NO_JAM.value: "不干扰",
    JammerMode.LOW_POWER.value: "低功率压制",
    JammerMode.HIGH_POWER.value: "高功率压制",
    JammerMode.INTERMITTENT.value: "间歇干扰",
}


@dataclass
class AdaptiveJammerParams:
    """规则自适应干扰机的判据参数（全部可配置）。"""

    # --- 威胁度权重 ---
    w_exposure: float = 1.0  # ESM 累计暴露的权重
    w_radiation: float = 0.8  # 雷达辐射强度的权重
    w_ineffective: float = 0.6  # 干扰无效程度的权重

    # --- 模式切换阈值 ---
    engage_threshold: float = 0.35  # threat 超过它开始干扰
    escalate_threshold: float = 0.65  # 超过它升到高功率
    stand_down_effective: float = 0.35  # 雷达达标率低于它说明干扰已奏效 -> 转间歇

    # --- 间歇干扰占空比 ---
    intermittent_period: int = 4  # 周期（步），一半开一半关
    high_power_streak_limit: int = 8  # 连续高功率超过该步数 -> 转间歇（省电+规避反辐射）

    # --- 观测窗口 ---
    history_window: int = 5  # 统计雷达平均功率与达标率的窗口

    # --- 模式滞回（避免每步抖动） ---
    mode_hold_steps: int = 2

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "AdaptiveJammerParams":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class AdaptiveJammerController:
    """规则自适应干扰机的决策状态机（每个干扰机一个实例，逐 episode 复位）。"""

    params: AdaptiveJammerParams = field(default_factory=AdaptiveJammerParams)
    jammer_id: str = "JAM"

    # --- 运行状态 ---
    mode: str = JammerMode.NO_JAM.value
    threat: float = 0.0
    steps_in_mode: int = 0
    high_power_streak: int = 0
    phase: int = 0
    power_scale: float = 0.0

    _power_history: Deque[float] = field(default_factory=deque, repr=False)
    _satisfaction_history: Deque[int] = field(default_factory=deque, repr=False)
    trace: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._power_history = deque(maxlen=self.params.history_window)
        self._satisfaction_history = deque(maxlen=self.params.history_window)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.mode = JammerMode.NO_JAM.value
        self.threat = 0.0
        self.steps_in_mode = 0
        self.high_power_streak = 0
        self.phase = 0
        self.power_scale = 0.0
        self._power_history.clear()
        self._satisfaction_history.clear()
        self.trace.clear()

    # ------------------------------------------------------------------
    # 观测与决策
    # ------------------------------------------------------------------

    def _radiation_ratio(self, max_power_w: float) -> float:
        if not self._power_history or max_power_w <= 0:
            return 0.0
        mean_power = sum(self._power_history) / len(self._power_history)
        return min(1.0, mean_power / max_power_w)

    def _ineffectiveness(self) -> float:
        """干扰无效程度 = 1 − 最近达标率。窗口为空时视为"未知"，取 0.5。"""
        if not self._satisfaction_history:
            return 0.5
        rate = sum(self._satisfaction_history) / len(self._satisfaction_history)
        return 1.0 - rate

    def compute_threat(self, exposure: float, max_power_w: float) -> float:
        p = self.params
        threat = (
            p.w_exposure * min(1.0, max(0.0, exposure))
            + p.w_radiation * self._radiation_ratio(max_power_w)
            + p.w_ineffective * self._ineffectiveness()
        )
        return threat

    def _select_mode(self, threat: float) -> str:
        p = self.params
        if threat < p.engage_threshold:
            return JammerMode.NO_JAM.value
        if threat < p.escalate_threshold:
            return JammerMode.LOW_POWER.value

        # 威胁高时在高功率与间歇之间选择，两种触发都会转间歇：
        #   1) 雷达已被明显压制（达标率低）—— 保持压制同时省电、规避反辐射；
        #   2) 已经连续高功率压制太久 —— 占空比管理，避免长时间当信标。
        radar_suppressed = (1.0 - self._ineffectiveness()) < p.stand_down_effective
        streak_exceeded = self.high_power_streak >= p.high_power_streak_limit
        if self.mode == JammerMode.HIGH_POWER.value and (radar_suppressed or streak_exceeded):
            return JammerMode.INTERMITTENT.value
        # 间歇模式下威胁仍高，且在间歇"关"的相位结束后回到高功率
        if self.mode == JammerMode.INTERMITTENT.value:
            return JammerMode.INTERMITTENT.value
        return JammerMode.HIGH_POWER.value

    def observe(
        self,
        exposure: float,
        radar_power_w: float,
        task_satisfied: bool,
        max_power_w: float,
        step_index: int,
        current_time: float,
    ) -> str:
        """在一步结束后观测雷达状态，并决定**下一步**的干扰模式。

        因果顺序：本步的干扰模式在步开始时就已经确定；这里看到的是本步产生的
        新证据（暴露量、雷达功率、是否达标），用来决定下一步，绝不回头改变本步。
        """
        self._power_history.append(float(radar_power_w))
        self._satisfaction_history.append(1 if task_satisfied else 0)

        threat = self.compute_threat(exposure, max_power_w)
        candidate = self._select_mode(threat)

        # 模式滞回：刚切过模式就要求它至少保持 mode_hold_steps 步
        if candidate != self.mode and self.steps_in_mode < self.params.mode_hold_steps:
            candidate = self.mode

        if candidate != self.mode:
            self.mode = candidate
            self.steps_in_mode = 0
        else:
            self.steps_in_mode += 1

        self.high_power_streak = (
            self.high_power_streak + 1 if self.mode == JammerMode.HIGH_POWER.value else 0
        )

        self.threat = threat
        # 本步（step_index）观察到的证据，用来决定**下一步**的干扰模式，
        # 因此相位与功率倍数都按 step_index + 1 计算
        self.phase = step_index + 1
        self.power_scale = self.scale_for_mode(self.mode, self.phase)

        self.trace.append(
            {
                "step_index": step_index,
                "applies_to_step": step_index + 1,
                "time": current_time,
                "jammer_id": self.jammer_id,
                "mode": self.mode,
                "mode_cn": MODE_CN.get(self.mode, self.mode),
                "threat": round(threat, 6),
                "power_scale": round(self.power_scale, 4),
                "exposure": round(float(exposure), 6),
                "radar_power_w": round(float(radar_power_w), 4),
                "radar_satisfied": bool(task_satisfied),
                "radiation_ratio": round(self._radiation_ratio(max_power_w), 6),
                "recent_satisfaction_rate": round(1.0 - self._ineffectiveness(), 6),
            }
        )
        return self.mode

    # ------------------------------------------------------------------

    def scale_for_mode(self, mode: str, step_index: int) -> float:
        """给定模式与步号，返回该步的功率倍数（间歇模式按周期开/关）。"""
        base = MODE_POWER_SCALE.get(mode, 0.0)
        if mode == JammerMode.INTERMITTENT.value:
            period = max(2, int(self.params.intermittent_period))
            on = (step_index % period) < (period // 2 + period % 2)
            return base if on else 0.0
        return base

    @property
    def current_scale(self) -> float:
        """当前步实际生效的功率倍数（用于干扰功率计算）。"""
        return self.power_scale

    def mode_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.trace:
            counts[item["mode"]] = counts.get(item["mode"], 0) + 1
        return counts

    def describe_trace_summary(self) -> str:
        counts = self.mode_counts()
        if not counts:
            return "（无决策记录）"
        return "、".join(
            f"{MODE_CN.get(m, m)}×{c}" for m, c in sorted(counts.items())
        )
