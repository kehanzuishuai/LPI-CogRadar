"""行为枚举。

通信抗干扰阶段只有 TRANSMIT / RECEIVE / INTERFERE / SILENT。
雷达功率调控阶段补充雷达侧的工作状态，其中 POWER_CONTROL 表示
「按策略调整发射功率」这一核心行为。
"""

from enum import Enum


class Behavior(str, Enum):
    # 遗留值（保持向后兼容）
    TRANSMIT = "transmit"
    RECEIVE = "receive"
    INTERFERE = "interfere"
    SILENT = "silent"

    # 雷达侧新增
    SCAN = "scan"
    TRACK = "track"
    POWER_CONTROL = "power_control"
    RADIATE = "radiate"
