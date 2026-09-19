"""侦察截获记录。

本文件在通信抗干扰阶段存放的是 ReceiveRecord（通信收包记录：
期望源 / 实际源、是否解码成功、BER 估计）。改造为低截获雷达仿真后，
它承担对应角色但语义改变——记录的是「敌方侦察接收机是否截获了雷达辐射」：

    ReceiveRecord(通信收包)  ->  InterceptionRecord(侦察截获)

原 ReceiveRecord 在工程中从未被任何模块引用，故直接替换，不留兼容别名。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InterceptionRecord:
    """单个敌方侦察接收机在某一步的截获结果。"""

    interceptor_id: str
    range_m: float
    beam: str  # "main" = 主瓣照射；"sidelobe" = 旁瓣照射
    tx_gain_db: float  # 雷达朝向该侦察机的发射增益（决定主/旁瓣）
    received_power_w: float  # 侦察机收到的雷达信号功率
    noise_power_w: float
    snr_db: float
    pint: float  # 截获概率

    @property
    def illuminated_by_main_beam(self) -> bool:
        return self.beam == "main"
