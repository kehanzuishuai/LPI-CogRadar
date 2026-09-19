"""雷达平台模型（低截获雷达智能功率调控仿真 第一版）。

雷达是被控对象：其发射功率 Pt 是唯一动作变量。
本模型是**纯数据 + 几何**，不计算信噪比、不依赖 engine.simulator——
所有物理量（回波功率、SNR、Pd）由 engine/equations.py 计算。

v4.1 多平台升级
---------------
继承 `SceneEntity`：获得统一标识（`entity_id`，本类为 `radar_id`）、
三维位置与速度、姿态、时间戳、平台归属。速度字段仍叫
`velocity_x/velocity_y`（历史命名，基类通过 `VELOCITY_FIELDS` 适配），
新增 `velocity_z`。

本类**没有** `range_to` —— 升级前它也没有，距离统一由各实体的
`SceneEntity.range_to()` / `range_to_entity()` 提供，实现只有一份。
"""

from __future__ import annotations

from dataclasses import dataclass

from models.entity import KIND_RADAR, SceneEntity


@dataclass
class Radar(SceneEntity):
    radar_id: str
    x: float
    y: float

    # --- 发射功率（标称值；实际每步取值来自动作档位）---
    tx_power_w: float = 10.0

    # --- 天线 ---
    peak_gain_db: float = 30.0  # 主瓣增益
    sidelobe_gain_db: float = 10.0  # 旁瓣增益（截获链路用；越低越难被侦察）
    main_beam_width_deg: float = 10.0  # 主瓣波束宽度，用于判断侦察机是否被主瓣照射

    # --- 波形与接收机 ---
    wavelength_m: float = 0.1
    bandwidth_hz: float = 1.0e6
    noise_figure_db: float = 3.0
    system_loss_db: float = 3.0
    temperature_k: float = 290.0

    # --- 探测概率 ROC 简化参数 ---
    snr50_db: float = 6.0  # Pd = 0.5 对应的 SNR
    pd_slope_db: float = 2.0  # 曲线陡峭度

    # --- 任务要求与能量约束 ---
    required_pd: float = 0.8
    energy_budget_j: float = 6000.0
    terminate_on_energy_exhausted: bool = False

    # --- 平台机动（第一版默认静止）---
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    freq_hz: float = 3.0e9
    is_active: bool = True

    # --- v4.1 多平台扩展 ---
    z: float = 0.0
    velocity_z: float = 0.0
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    timestamp_s: float = 0.0
    platform_id: str = ""

    ENTITY_KIND = KIND_RADAR
    ID_FIELD = "radar_id"
    #: 雷达的速度字段是历史命名，与其它实体不同
    VELOCITY_FIELDS = ("velocity_x", "velocity_y", "velocity_z")

    def __post_init__(self) -> None:
        if self.tx_power_w <= 0:
            raise ValueError(f"[{self.radar_id}] tx_power_w 必须为正")
        if self.wavelength_m <= 0:
            raise ValueError(f"[{self.radar_id}] wavelength_m 必须为正")
        if not 0.0 < self.required_pd < 1.0:
            raise ValueError(f"[{self.radar_id}] required_pd 必须落在 (0, 1) 开区间内")
        if self.sidelobe_gain_db > self.peak_gain_db:
            raise ValueError(f"[{self.radar_id}] 旁瓣增益不应高于主瓣增益")
        if self.pd_slope_db <= 0:
            raise ValueError(f"[{self.radar_id}] pd_slope_db 必须为正")
