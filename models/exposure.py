"""累计暴露 / 侦察证据模型。

物理动机
--------
敌方侦察接收机（ESM）不是「每一步独立掷骰子」：
它把一段时间内截获到的雷达辐射**累积**起来。连续多步以较大功率辐射，
即使单步截获概率不高，累积证据也会让 ESM 完成信号分选、型号识别与测向，
从而**抬高后续的被截获风险**（并可为干扰机/反辐射武器提供引导）。

模型（第一版，刻意做得简单、可审计）
------------------------------------
暴露量是一个一阶递推状态 e ∈ [0, 1]：

    e(0)   = 0
    e(t+1) = min( 1,  decay · e(t) + gain · Pint_inst(t) )      （在 step 之后更新）

当步的有效截获概率是「本步瞬时截获」与「累积证据已经暴露」的并集：

    Pint_eff(t) = 1 − (1 − Pint_inst(t)) · (1 − e(t))          （用 step 之前的 e）

**因果顺序很关键**：本步的动作影响的是**未来**的 e，从而影响**未来**的 Pint_eff，
而不会回头改变本步的 Pint_eff。这正是把原本的 contextual bandit
改造成真正时序决策问题的关键机制：
    现在提高功率 → 本步 Pint_inst 上升 → 未来 e 上升 → 未来 Pint_eff 上升
    → 未来奖励下降
因此「只看当步收益」的短视策略不再最优。

参数
----
decay ∈ (0, 1)  证据保持率。越接近 1 表示 ESM 的"记忆"越长，时序耦合越强。
gain  > 0       每步瞬时截获对证据的贡献系数。
                稳态暴露量约为 gain · Pint_inst / (1 − decay)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExposureTracker:
    """累计暴露量的递推跟踪器。"""

    decay: float = 0.9
    gain: float = 0.2

    value: float = 0.0
    history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError(f"exposure decay 必须落在 [0, 1) 内，当前 {self.decay}")
        if self.gain <= 0.0:
            raise ValueError(f"exposure gain 必须为正，当前 {self.gain}")

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.value = 0.0
        self.history = []

    def record(self, exposure_before: float) -> None:
        """记录一步的暴露量（用于统计曲线）。"""
        self.history.append(exposure_before)

    def next_value(self, pint_inst: float) -> float:
        """给定本步瞬时截获概率，返回下一步的暴露量（不修改状态）。"""
        return min(1.0, self.decay * self.value + self.gain * max(0.0, pint_inst))

    def update(self, pint_inst: float) -> float:
        """按本步瞬时截获概率推进暴露量，返回推进后的值。"""
        self.value = self.next_value(pint_inst)
        return self.value

    @staticmethod
    def effective_pint(pint_inst: float, exposure_before: float) -> float:
        """有效截获概率 = 瞬时截获 ∪ 累积证据已暴露。"""
        pint_inst = min(1.0, max(0.0, pint_inst))
        exposure_before = min(1.0, max(0.0, exposure_before))
        return 1.0 - (1.0 - pint_inst) * (1.0 - exposure_before)

    def effective_pint_now(self, pint_inst: float) -> float:
        """用当前暴露量计算有效截获概率。"""
        return self.effective_pint(pint_inst, self.value)

    # ------------------------------------------------------------------

    @property
    def steady_state(self) -> float:
        """恒定 Pint_inst = p 时的稳态暴露量（p 需由调用方给出时用 helper）。"""
        return self.gain / (1.0 - self.decay) if self.decay < 1.0 else float("inf")

    def steady_state_for(self, pint_inst: float) -> float:
        """恒定瞬时截获概率下的稳态暴露量，便于标定时核对量级。"""
        if self.decay >= 1.0:
            return 1.0
        return min(1.0, self.gain * pint_inst / (1.0 - self.decay))
