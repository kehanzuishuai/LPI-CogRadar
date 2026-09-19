"""敌方侦察接收机（ESM / 电子支援措施）模型。

侦察链路是**单程**的：雷达只要在辐射，其旁瓣（或主瓣）泄漏就会被侦察机截获。
截获概率 Pint 是低截获性能的直接度量——本项目要把它压下去，
而压低它的手段（降低 Pt）又会拉低探测概率 Pd，这就是核心矛盾。

v4.1 多平台升级
---------------
继承 `SceneEntity`：获得统一标识、三维位置与速度、姿态、时间戳、平台归属，
以及全工程唯一的距离实现（本文件不再自算距离）。
"""

from __future__ import annotations

from dataclasses import dataclass

from models.entity import KIND_INTERCEPTOR, SceneEntity


@dataclass
class EnemyInterceptor(SceneEntity):
    interceptor_id: str
    x: float
    y: float

    # --- 侦察接收机参数 ---
    gain_db: float = 12.0
    bandwidth_hz: float = 2.0e6
    noise_figure_db: float = 6.0
    system_loss_db: float = 2.0
    temperature_k: float = 290.0

    # --- 截获判决 ROC 简化参数 ---
    snr50_db: float = 22.0  # Pint = 0.5 对应的截获 SNR
    pint_slope_db: float = 3.0  # 曲线陡峭度

    # --- 平台机动 ---
    vx: float = 0.0
    vy: float = 0.0

    freq_hz: float = 3.0e9
    is_active: bool = True

    # --- v4.1 多平台扩展 ---
    z: float = 0.0
    vz: float = 0.0
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    timestamp_s: float = 0.0
    platform_id: str = ""

    ENTITY_KIND = KIND_INTERCEPTOR
    ID_FIELD = "interceptor_id"

    def __post_init__(self) -> None:
        if self.bandwidth_hz <= 0:
            raise ValueError(f"[{self.interceptor_id}] bandwidth_hz 必须为正")
        if self.pint_slope_db <= 0:
            raise ValueError(f"[{self.interceptor_id}] pint_slope_db 必须为正")
