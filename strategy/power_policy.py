"""发射功率控制策略（低截获雷达智能功率调控仿真 第一版）。

取代原 strategy/anti_jam.py 的跳频选择策略（后者已弱化为遗留模块）。

三个策略：
    FixedPowerPolicy     固定功率基线——始终使用同一档功率（默认满功率）
    RuleBasedPowerPolicy 规则功率控制基线——在满足探测要求的前提下取最低功率
    RandomPowerPolicy    随机策略——仅用于演示动作空间，作为 RL 训练前的下限参照

共同接口：
    select_level(sim) -> int   返回 0..num_levels-1 的功率档位索引。
    reset()                    复位内部状态（run() 前调用）。

关键：Simulator.preview() 是纯函数，规则策略可以安全地对 11 档功率逐一「试算」，
这是它能在不修改环境状态的前提下选出最低可用档位的原因。
"""

from __future__ import annotations

import copy
import random
from typing import Any, List, Optional


class PowerPolicy:
    """策略基类。

    能量硬约束下的统一约定
    ----------------------
    所有策略都必须**只从 `sim.feasible_levels()` 里选动作**——
    即本步所需能量 Pt·Δt 不超过剩余能量的档位。
    `Simulator.step()` 会对不可行动作直接抛 `InfeasibleActionError`，
    因此策略层不做可行性检查就会崩。
    基类提供 `_clamp_to_feasible()` 做兜底：策略若因数值原因算出不可行档位，
    自动退到当前可执行的最高档。
    """

    name: str = "policy"
    description: str = ""

    def reset(self) -> None:
        """复位内部状态。run() 会在一段实验开始前调用。"""
        return None

    @staticmethod
    def _clamp_to_feasible(sim: Any, level: int) -> int:
        """把档位裁剪到当前可行集合内（兜底，正常路径不该用到）。"""
        clipped = sim.clip_to_feasible(level)
        if clipped is None:
            raise RuntimeError(
                "剩余能量已买不起任何一档功率，episode 应当已结束（sim.is_done 为真）"
            )
        return int(clipped)

    def select_level(self, sim: Any) -> int:
        """返回本步采用的功率档位索引。子类必须实现。"""
        raise NotImplementedError

    def describe(self) -> str:
        text = f"{self.name}"
        if self.description:
            text += f"：{self.description}"
        return text


class FixedPowerPolicy(PowerPolicy):
    """固定功率基线。

    真实雷达长期沿用「固定发射功率」的设计：不论干扰强弱、目标远近，
    始终以同一档功率辐射。本项目用它作为对照组，展示两个代价：
    1) 距离近、干扰弱时仍满功率发射 —— 白白增加被截获与被测向的风险；
    2) 能量消耗与最大功率成正比 —— 快速耗尽能量预算。
    """

    name = "固定功率"

    def __init__(self, level_index: Optional[int] = None) -> None:
        # None 表示取配置里的 fixed_power_level（默认最高档）
        self.level_index = level_index
        self._resolved: Optional[int] = None

    def reset(self) -> None:
        self._resolved = None

    def _resolve(self, sim: Any) -> int:
        if self.level_index is not None:
            return int(self.level_index)
        if self._resolved is None:
            self._resolved = int(sim.scenario.resolve_fixed_power_level())
        return self._resolved

    def select_level(self, sim: Any) -> int:
        # 固定功率也可能在后期买不起（例如始终 80 W），此时退到当前可行的最高档
        return self._clamp_to_feasible(sim, self._resolve(sim))

    def describe(self) -> str:
        target = "配置指定档" if self.level_index is None else f"档位{self.level_index}"
        return f"{self.name}（{target}；能量不足时裁剪到可行最高档）"


class RuleBasedPowerPolicy(PowerPolicy):
    """规则功率控制基线。

    规则：**在满足探测任务要求的前提下，选最低的那一档功率。**

        required_snr = snr_db_for_prob(required_pd, snr50_db, pd_slope_db)
        从低到高遍历功率档位，取第一个满足 SNR >= required_snr + margin_db 的档位

    这条规则直接对应低截获雷达的工程直觉：「用刚好够用的功率探测」。
    它不使用任何学习，因此在干扰突变、多目标遮挡等复杂情形下不够聪明——
    这正是后续用 DQN 替换它的动机。

    参数
    ----
    margin_db      : 选择档位时的 SNR 安全余量。默认 0.05 dB —— 只为避开
                     浮点边界（策略比较 SNR，任务判定比较 Pd，两者互为反函数），
                     量级可忽略，不改变「刚好够用」的规则本意。
    hysteresis_db  : 降档时额外要求的余量。当前档位仍够用且更低档位在
                     (margin + hysteresis) 下也够用时才降档，可减少档位切换次数。
    """

    name = "规则功率控制"

    def __init__(self, margin_db: float = 0.05, hysteresis_db: float = 0.0) -> None:
        self.margin_db = margin_db
        self.hysteresis_db = hysteresis_db
        self.switch_count = 0

    def reset(self) -> None:
        self.switch_count = 0

    # ------------------------------------------------------------------

    def _satisfies(self, evaluation: Any, extra_margin_db: float) -> bool:
        threshold = evaluation.required_snr_db + self.margin_db + extra_margin_db
        return evaluation.snr_radar_db_min >= threshold

    def select_level(self, sim: Any) -> int:
        levels: List[float] = sim.power_levels_w
        feasible = sim.feasible_levels()
        current = sim.previous_power_level

        # 1) 当前档位仍然可行且够用：尝试降档（需要额外余量，抑制抖动）
        if current in feasible and self._satisfies(sim.preview(levels[current]), 0.0):
            for level in feasible:
                if level >= current:
                    break
                if self._satisfies(sim.preview(levels[level]), self.hysteresis_db):
                    if level != current:
                        self.switch_count += 1
                    return self._clamp_to_feasible(sim, level)
            return self._clamp_to_feasible(sim, current)

        # 2) 当前档位不够用（首步或干扰增强）：从低到高找最低**可行**且够用的档
        for level in feasible:
            if self._satisfies(sim.preview(levels[level]), 0.0):
                if level != current:
                    self.switch_count += 1
                return level

        # 3) 可行档位里没有一个能满足：用当前可执行的最高档，
        #    认账这一步的探测失败（而不是硬撑到超预算）
        best = sim.max_feasible_level()
        if best is None:
            raise RuntimeError("剩余能量已买不起任何一档功率，episode 应当已结束")
        if best != current:
            self.switch_count += 1
        return int(best)

    def describe(self) -> str:
        return (
            f"{self.name}（最低**可行**可用档；余量 {self.margin_db} dB，"
            f"降档滞回 {self.hysteresis_db} dB）"
        )


class RandomPowerPolicy(PowerPolicy):
    """随机功率策略。

    不用于对比性能，只用于验证 Gymnasium 风格动作空间是否可用，
    并作为强化学习训练前的随机下限参照。
    """

    name = "随机功率"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

    def select_level(self, sim: Any) -> int:
        feasible = sim.feasible_levels()
        return feasible[self._rng.randrange(len(feasible))]

    def describe(self) -> str:
        return f"{self.name}（seed={self.seed}，在**可行档位**内随机；仅作动作空间演示与随机下限）"


class GreedyOraclePolicy(PowerPolicy):
    """逐档贪心策略（**短视基线**）。

    每一步用 Simulator.preview() 遍历全部功率档位，取**单步收益**最大的一档：

        a* = argmax_a  composite_reward(det_term(s,a), Pint_eff(s,a), Pt_a/Pt_max, violation)

    ⚠️ 命名与定位说明（第二版重要变更）
    ------------------------------------
    在第一版（contextual bandit 环境）里，因为动作不影响后续状态，逐步贪心即为全局最优，
    这个策略曾是**所有策略的上界**，因此叫「逐档穷举最优(上界)」。

    第二版引入了能量硬约束与累计暴露，动作会改变未来的状态（剩余能量、暴露量），
    **逐步贪心不再是最优**，所以这里改名为「逐档贪心(短视)」，
    它只是「只看当步、不往后看」的基线。真正的非短视参考见 LookaheadPolicy。

    仍然保留它是因为：它是理解「时序结构到底值多少钱」的最佳对照
    —— 短视贪心与前瞻规划之间的差距，就是新环境里时序决策的价值。
    """

    name = "逐档贪心(短视)"

    def __init__(self) -> None:
        self.switch_count = 0

    def reset(self) -> None:
        self.switch_count = 0

    def select_level(self, sim: Any) -> int:
        levels: List[float] = sim.power_levels_w
        best_level = 0
        best_reward = float("-inf")
        # 只在可行档位里穷举（能量硬约束）
        for level in sim.feasible_levels():
            evaluation = sim.preview(levels[level])
            reward = sim.action_reward(evaluation, levels[level])
            if reward > best_reward:
                best_reward = reward
                best_level = level

        if sim.previous_power_level >= 0 and best_level != sim.previous_power_level:
            self.switch_count += 1
        return best_level

    def describe(self) -> str:
        return f"{self.name}（每步在**可行档位**内穷举取单步收益最大者，不考虑未来；需要完整模型知识）"


class LookaheadPolicy(PowerPolicy):
    """滚动时域前瞻策略 —— **非短视参考基线**。

    对每个候选动作做 H 步前瞻推演，取折扣累计收益最大的动作：

        a* = argmax_a  Σ_{h=0}^{H-1} γ^h · r(s_h, a_h)
        s_0 = 当前状态, a_0 = a, 之后各步用 rollout 策略（默认逐档贪心）

    实现方式：`copy.deepcopy(simulator)` 复制出状态完全一致的仿真器副本，
    在副本上真实推进（从而正确演化剩余能量与累计暴露），不影响主仿真器。

    它同样需要完整模型知识，现实中不可部署；用途有两个：
    1. 作为**非短视上界**，量化「时序结构值多少钱」；
    2. 让 DQN 的表现有一个比短视贪心更有意义的比较对象。

    参数
    ----
    horizon        : 前瞻步数（越大越接近真正的最优，但计算量线性增长）
    discount       : 前瞻用的折扣因子
    """

    name = "前瞻规划(H步)"

    def __init__(self, horizon: int = 6, discount: float = 0.95) -> None:
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1")
        self.horizon = horizon
        self.discount = discount
        self.switch_count = 0

    def reset(self) -> None:
        self.switch_count = 0

    def _rollout_level(self, sim: Any) -> int:
        """前瞻推演过程中使用的默认策略：逐档贪心（只看当步，且只在可行档内）。"""
        levels: List[float] = sim.power_levels_w
        best_level, best_reward = 0, float("-inf")
        for level in sim.feasible_levels():
            evaluation = sim.preview(levels[level])
            reward = sim.action_reward(evaluation, levels[level])
            if reward > best_reward:
                best_reward = reward
                best_level = level
        return best_level

    def _value(self, sim_copy: Any, first_level: int) -> float:
        """从副本当前状态出发，先执行 first_level，再按 rollout 策略推演 H 步。"""
        total = 0.0
        level = first_level
        for h in range(self.horizon):
            if sim_copy.is_done:
                break
            result = sim_copy.step(level)
            total += (self.discount ** h) * result.reward
            level = self._rollout_level(sim_copy)
        return total

    def select_level(self, sim: Any) -> int:
        best_level = 0
        best_value = float("-inf")
        # 只在可行档位里前瞻（能量硬约束）
        for level in sim.feasible_levels():
            sim_copy = copy.deepcopy(sim)
            value = self._value(sim_copy, level)
            if value > best_value:
                best_value = value
                best_level = level

        if sim.previous_power_level >= 0 and best_level != sim.previous_power_level:
            self.switch_count += 1
        return best_level

    def describe(self) -> str:
        return (
            f"{self.name}（每步对**可行档位**各做 {self.horizon} 步前瞻，γ={self.discount}；"
            f"需要完整模型知识，不可部署）"
        )

    @classmethod
    def full_horizon(cls, num_steps: int, discount: float = 1.0) -> "LookaheadPolicy":
        """构造「完整任务视野」的前瞻策略。

        所有脚本统一通过本工厂构造前瞻基线，避免各脚本各写一套 horizon 规则
        （horizon = 任务步数 + 1，保证从任何一步都能看到 episode 结束）而产生漂移。
        """
        return cls(horizon=int(num_steps) + 1, discount=discount)
