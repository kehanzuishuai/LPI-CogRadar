"""干扰源模型（动态干扰）。

干扰机通过抬高雷达接收机的噪声基底起作用：
    SINR = S / (N + J_eff)
因此干扰会拉低 Pd，迫使雷达提高 Pt —— 而提高 Pt 又会让 Pint 上升。
这个「干扰 -> 提功率 -> 更易被截获」的链条是本项目的核心机理。

J_eff 中的抗干扰处理增益（脉压增益、相参积累、旁瓣对消等难以精确建模的因素）
合并为配置项 suppression_db 显式给出。

v4.1 多平台升级
---------------
继承 `SceneEntity`：获得统一标识、三维位置与速度、姿态、时间戳、平台归属，
以及全工程唯一的距离实现（本文件不再自算距离）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from models.entity import KIND_JAMMER, SceneEntity

from .adaptive_jammer import AdaptiveJammerController, AdaptiveJammerParams


@dataclass
class Jammer(SceneEntity):
    jammer_id: str
    x: float
    y: float

    # --- 干扰机发射参数 ---
    peak_power_w: float = 200.0
    gain_db: float = 10.0
    system_loss_db: float = 3.0

    # --- 雷达侧抗干扰等效处理增益（合并建模）---
    suppression_db: float = 60.0

    # --- 工作时间窗 ---
    start_time: float = 0.0
    end_time: float = 1.0e9

    # --- 强度 ---
    intensity: float = 1.0
    duty_cycle: float = 1.0

    # --- 动态特性：开关之外还有强度起伏 ---
    dynamic: bool = False
    fluctuation: float = 0.25  # 相对起伏幅度，例如 0.25 表示 ±25%
    fluctuation_step: float = 0.5  # 随机游走单步幅度

    # --- 干扰模式：fixed = 原固定时间窗模式（默认，保证旧实验可复现）；
    #     adaptive = 规则自适应智能干扰机（见 models/adaptive_jammer.py）---
    jammer_mode: str = "fixed"
    adaptive_params: Dict[str, Any] = field(default_factory=dict)

    # --- 平台机动 ---
    vx: float = 0.0
    vy: float = 0.0

    is_active: bool = True

    # --- v4.1 多平台扩展 ---
    z: float = 0.0
    vz: float = 0.0
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    timestamp_s: float = 0.0
    platform_id: str = ""

    ENTITY_KIND = KIND_JAMMER
    ID_FIELD = "jammer_id"

    # 由 prepare() 预生成，保证同一种子下完全可复现
    _fluctuation_series: List[float] = field(default_factory=list, repr=False, compare=False)
    # 自适应模式下的决策状态机（fixed 模式下不使用）
    _controller: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.peak_power_w < 0:
            raise ValueError(f"[{self.jammer_id}] peak_power_w 不能为负")
        if self.end_time < self.start_time:
            raise ValueError(f"[{self.jammer_id}] end_time 不能早于 start_time")
        if self.fluctuation < 0:
            raise ValueError(f"[{self.jammer_id}] fluctuation 不能为负")
        if self.jammer_mode not in ("fixed", "adaptive"):
            raise ValueError(
                f"[{self.jammer_id}] jammer_mode 只能是 'fixed' 或 'adaptive'，"
                f"当前 {self.jammer_mode!r}"
            )
        if self.jammer_mode == "adaptive":
            self._controller = AdaptiveJammerController(
                params=AdaptiveJammerParams.from_dict(self.adaptive_params),
                jammer_id=self.jammer_id,
            )

    @property
    def is_adaptive(self) -> bool:
        return self.jammer_mode == "adaptive"

    @property
    def controller(self) -> Any:
        return self._controller

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def prepare(self, num_steps: int, rng: random.Random) -> None:
        """预生成整段仿真的强度起伏序列（有界随机游走），并复位自适应决策器。"""
        if self.dynamic and self.fluctuation > 0.0:
            value = 1.0
            series: List[float] = []
            for _ in range(num_steps + 1):
                value += rng.uniform(-self.fluctuation_step, self.fluctuation_step)
                value = max(1.0 - self.fluctuation, min(1.0 + self.fluctuation, value))
                series.append(value)
            self._fluctuation_series = series
        else:
            self._fluctuation_series = [1.0] * (num_steps + 1)

        if self._controller is not None:
            self._controller.reset()

    def update_position(self, dt: float) -> None:
        self.advance(dt)

    # ------------------------------------------------------------------
    # 干扰强度
    # ------------------------------------------------------------------

    def is_jamming(self, current_time: float) -> bool:
        """是否处于「可能辐射」的状态。

        fixed 模式沿用原语义：只在预置时间窗内；
        adaptive 模式不受时间窗约束（何时干扰由决策器按态势决定），
        只要平台在线即可，实际贡献由 factor_at() 是否为 0 决定。
        """
        if not self.is_active:
            return False
        if self.is_adaptive:
            return True
        return self.start_time <= current_time <= self.end_time

    def factor_at(self, step_index: int) -> float:
        """返回该步的等效干扰强度系数（0 表示未工作）。

        fixed 模式：= intensity * duty_cycle * 动态起伏
        adaptive 模式：再乘上规则自适应决策器给出的功率倍数
                       （不干扰 = 0；间歇干扰在"关"的步为 0）
        """
        if not self.is_active:
            return 0.0

        series = self._fluctuation_series
        if not series:
            fluctuation = 1.0
        else:
            fluctuation = series[min(step_index, len(series) - 1)]

        factor = self.intensity * self.duty_cycle * fluctuation

        if self._controller is not None:
            factor *= self._controller.current_scale

        return factor
