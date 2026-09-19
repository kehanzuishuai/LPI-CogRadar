"""场景（全局仿真配置）模型。

改造说明
--------
新增低截获雷达所需的全局参数：离散功率档位、固定功率基线档位、综合收益权重。
末尾 `legacy` 段字段来自通信抗干扰阶段，新流程不再读取，
保留下来以免破坏 backup_original 与历史实验脚本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .reward import DEFAULT_REWARD_WEIGHTS


@dataclass
class Scenario:
    # --- 新流程使用 ---
    scenario_name: str
    sim_duration: float
    time_step: float
    random_seed: int = 42
    description: str = ""
    power_levels_w: List[float] = field(default_factory=list)
    fixed_power_level: int = -1  # 固定功率基线使用的档位；-1 表示最高档
    lpi_pint_threshold: float = 0.5  # 低截获达标判据：Pint 不超过该值视为「低截获状态」
    exposure_decay: float = 0.9  # 累计暴露量的证据保持率（第一版默认）
    exposure_gain: float = 0.2  # 每步瞬时截获对暴露证据的贡献系数
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS)
    )

    # --- 通信抗干扰阶段遗留字段（新流程不读取）---
    area_width: float = 0.0
    area_height: float = 0.0
    background_noise: float = 0.0
    max_comm_distance: float = 0.0

    def __post_init__(self) -> None:
        if self.sim_duration <= 0:
            raise ValueError("sim_duration 必须为正")
        if self.time_step <= 0:
            raise ValueError("time_step 必须为正")
        if not 0.0 < self.lpi_pint_threshold < 1.0:
            raise ValueError("lpi_pint_threshold 必须落在 (0, 1) 开区间内")
        if not 0.0 <= self.exposure_decay < 1.0:
            raise ValueError("exposure_decay 必须落在 [0, 1) 内")
        if self.exposure_gain <= 0.0:
            raise ValueError("exposure_gain 必须为正")

        # 功率档位只在提供时校验。留空表示沿用通信抗干扰阶段的遗留用法
        # （backup_original/simulator.py 构造 Scenario 时不带功率档位），
        # 新流程的配置里 power_levels_w 是必填项，由 Simulator.load_config 把关。
        if self.power_levels_w:
            if any(level <= 0 for level in self.power_levels_w):
                raise ValueError("power_levels_w 中每一档功率都必须为正")
            if list(self.power_levels_w) != sorted(self.power_levels_w):
                raise ValueError("power_levels_w 必须按升序排列，策略依赖这一约定")

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------

    @property
    def num_levels(self) -> int:
        if not self.power_levels_w:
            raise ValueError("本场景未定义 power_levels_w（遗留通信场景），无动作空间")
        return len(self.power_levels_w)

    @property
    def num_steps(self) -> int:
        """一个 episode 的步数（含 t=0 的初始步）。"""
        return int(round(self.sim_duration / self.time_step)) + 1

    @property
    def max_power_w(self) -> float:
        if not self.power_levels_w:
            raise ValueError("本场景未定义 power_levels_w（遗留通信场景），无最大功率")
        return self.power_levels_w[-1]

    def time_at(self, step_index: int) -> float:
        return step_index * self.time_step

    def resolve_fixed_power_level(self) -> int:
        """固定功率基线实际使用的档位索引。"""
        if self.fixed_power_level < 0:
            return self.num_levels - 1
        if self.fixed_power_level >= self.num_levels:
            raise ValueError(
                f"fixed_power_level={self.fixed_power_level} 超出档位范围 "
                f"(0..{self.num_levels - 1})"
            )
        return self.fixed_power_level

    def nearest_level_index(self, power_w: float) -> int:
        """把任意功率值吸附到最近的档位（用于解析配置里的标称功率）。"""
        return min(
            range(self.num_levels),
            key=lambda i: abs(self.power_levels_w[i] - power_w),
        )
